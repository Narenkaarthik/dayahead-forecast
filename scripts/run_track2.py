"""Track 2: do the 80% and 90% intervals keep their promise, and does recalibration help?

    python scripts/run_track2.py --data data/raw/Solar

Run scripts/run_track1.py first: variant V1 comes from its LightGBM predictions.

Four variants of the same LightGBM model's intervals (daylight hours only):
  V1 raw-full   Track 1 LightGBM quantiles, trained on all history before the test month
  V2 raw-split  LightGBM trained up to 60 days before the test month (pvcal.models.split_model)
  V3 CQR        V2 widened or narrowed by one constant per month, measured on those 60 days
  V4 ACI        like V3, but the level is nudged after every forecast day (gamma chosen on tasks 1-3)

Intervals: 80% = [q0.10, q0.90], 90% = [q0.05, q0.95]. Tasks 1-15 are run; tasks 1-3 only
choose gamma; tasks 4-15 are reported. Confidence intervals resample whole forecast days.

Writes results/tables/T2_track2, T2_paired_differences, T3_by_cloud, T4_by_month (.csv/.md),
aci_gamma_selection.csv, aci_alpha_path.csv, track2_diagnostics.csv, and
results/figures/F1_fan_chart.png and F2_monthly_coverage.png.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pvcal.calibrate import (aci_daily, apply_adjustment, conformity_scores, cqr_quantile,
                             day_bootstrap, percentile_ci)
from pvcal.config import (ACI_GAMMAS, ALL_TASK_IDS, DEV_TASK_IDS, INTERVALS, LEAKAGE_CHECK_TASKS,
                          TCC_CUTOFF)
from pvcal.data import load_solar, local_time
from pvcal.features import is_daylight
from pvcal.leakage import CANARY
from pvcal.models import split_model
from pvcal.tables import save_table
from pvcal.tasks import SCORED_TASK_IDS, make_task

VARIANTS = ("V1 raw-full", "V2 raw-split", "V3 CQR", "V4 ACI")
NAMES = dict(zip(("V1", "V2", "V3", "V4"), VARIANTS))  # short code -> full name
TERCILES = ("low", "mid", "high")


def tag(nominal: float) -> str:
    """0.9 -> "90"; used in column names like "V3_lo_90"."""
    return f"{round(nominal * 100)}"


def level_columns(nominal: float) -> tuple[str, str]:
    """The quantile columns of an interval, e.g. 0.9 -> ("q0.05", "q0.95")."""
    lo, hi = INTERVALS[nominal]
    return f"q{lo:.2f}", f"q{hi:.2f}"


def check_split_model_leakage(df: pd.DataFrame, task, params: dict, reference) -> None:
    """Rerun the split model with every test-month outcome replaced by an impossible value.

    Its predictions for the calibration window and the test month must not change; the CQR
    intervals depend only on those predictions and on calibration-window outcomes, so they
    cannot change either.
    """
    in_test = (df["time"] > task.test_start) & (df["time"] <= task.test_end)
    cal, test = split_model(df.assign(power=df["power"].where(~in_test, CANARY)), task, params)
    cols = [c for c in cal.columns if c.startswith("q")]
    for new, old in zip((cal, test), reference):
        if not np.array_equal(new[cols].to_numpy(), old[cols].to_numpy()):
            raise AssertionError(f"task {task.task_id}: split-model forecasts changed when "
                                 "test-month power was altered")


def run_aci(rows: pd.DataFrame, scores: dict, nominal: float, gamma: float):
    """ACI on the given daylight rows (sorted by day) for one interval level."""
    lo_c, hi_c = (f"V2_lo_{tag(nominal)}", f"V2_hi_{tag(nominal)}")
    return aci_daily(rows["day"].to_numpy(), rows["task"].to_numpy(), rows["power"].to_numpy(),
                     rows[lo_c].to_numpy(), rows[hi_c].to_numpy(), scores[nominal],
                     alpha=1 - nominal, gamma=gamma)


def check_aci_is_causal(rows: pd.DataFrame, scores: dict, nominal: float, gamma: float) -> None:
    """Corrupt the outcomes of one mid-sequence day and every later day: earlier intervals must not move."""
    days = rows["day"].unique()
    d = days[len(days) // 2]
    L, U, _, _ = run_aci(rows, scores, nominal, gamma)
    L2, U2, _, _ = run_aci(rows.assign(power=rows["power"].where(rows["day"] < d, CANARY)), scores, nominal, gamma)
    upto = (rows["day"] <= d).to_numpy()
    if not (np.array_equal(L[upto], L2[upto]) and np.array_equal(U[upto], U2[upto])):
        raise AssertionError("ACI intervals changed when outcomes from later days were altered")


def interval_stats(y, L, U, nominal: float) -> dict:
    """Coverage (PICP), mean width (MPIW) and mean Winkler score of intervals [L, U]."""
    alpha = 1 - nominal
    covered = (y >= L) & (y <= U)
    winkler = (U - L) + (2 / alpha) * (np.maximum(L - y, 0) + np.maximum(y - U, 0))
    return {"picp": covered.mean(), "mpiw": (U - L).mean(), "winkler": winkler.mean()}


def boot_ci(rows: pd.DataFrame, days, value: str, mask=None) -> tuple[float, float]:
    """95% CI of the row mean of `value` (within `mask`) by resampling whole forecast days.

    Per-day sums and row counts are taken over the same fixed list of days for every call,
    so with the same seed every call resamples the same days: comparisons are paired.
    """
    sub = rows if mask is None else rows[mask]
    per_day = sub.groupby("day")[value].agg(["sum", "count"]).reindex(days, fill_value=0)
    return percentile_ci(day_bootstrap(per_day["sum"].to_numpy(), per_day["count"].to_numpy()))


def pct(v: float) -> str:
    return f"{v:.1%}"


def pct_ci(v: float, lo: float, hi: float) -> str:
    return f"{v:.1%} [{lo:.1%}, {hi:.1%}]"


def num_ci(v: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def plot_fan_chart(v1: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Zone 1: observed power vs the V1 median and 80%/90% bands, for two fixed 7-day blocks.

    The blocks are chosen by a fixed rule, not by eye: among all runs of 7 consecutive
    forecast days in tasks 4-15, the one with the lowest mean daylight forecast cloud cover
    for zone 1, and the one with the highest.
    """
    z1 = v1[v1["zone"] == 1].sort_values("time")
    daily = z1[z1["daylight"]].groupby("day")["tcc"].agg(["sum", "count"])
    week = daily.rolling(7).sum().dropna()  # each row: totals over that day and the 6 before it
    mean_tcc = week["sum"] / week["count"]
    picks = {"Lowest forecast cloud cover": mean_tcc.idxmin(), "Highest forecast cloud cover": mean_tcc.idxmax()}

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharey=True)
    chosen = []
    for ax, (label, last_day) in zip(axes, picks.items()):
        block_days = daily.index[daily.index.get_loc(last_day) - 6: daily.index.get_loc(last_day) + 1]
        rows = z1[z1["day"].isin(block_days)]
        x = local_time(rows["time"]).dt.tz_localize(None)  # farm clock time, UTC+10
        ax.fill_between(x, rows["q0.05"], rows["q0.95"], color="tab:blue", alpha=0.18, lw=0, label="90% interval")
        ax.fill_between(x, rows["q0.10"], rows["q0.90"], color="tab:blue", alpha=0.32, lw=0, label="80% interval")
        ax.plot(x, rows["q0.50"], color="tab:blue", lw=1.2, label="Forecast median")
        ax.plot(x, rows["power"], color="black", lw=1.0, label="Observed")
        ax.set_title(f"{label}: 7 forecast days from {block_days[0]:%Y-%m-%d}, "
                     f"mean daylight forecast cloud cover {mean_tcc[last_day]:.2f}", fontsize=9)
        ax.set_ylabel("Power (fraction of capacity)")
        chosen.append({"block": label, "first_day": block_days[0], "mean_daylight_tcc": mean_tcc[last_day]})
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="upper right")
    axes[1].set_xlabel("Local time at the farms (UTC+10)")
    fig.suptitle("Zone 1, V1 raw-full LightGBM intervals", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return pd.DataFrame(chosen)


def plot_monthly_coverage(t4: pd.DataFrame, path: Path) -> None:
    """PICP by test month for V1-V4, with bootstrap 95% CIs, one panel per interval level."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    months = sorted(t4["month"].unique())
    x = np.arange(len(months))
    for ax, nominal in zip(axes, (0.9, 0.8)):
        for k, variant in enumerate(VARIANTS):
            s = t4[(t4["variant"] == variant) & (t4["level"] == nominal)].set_index("month").loc[months]
            err = [s["picp"] - s["picp_lo"], s["picp_hi"] - s["picp"]]
            ax.errorbar(x + (k - 1.5) * 0.15, s["picp"], yerr=err, fmt="o", ms=3.5, capsize=2, lw=1, label=variant)
        ax.axhline(nominal, color="grey", ls="--", lw=1, label=f"Nominal {nominal:.0%}")
        ax.set_xticks(x, months, rotation=45, fontsize=8)
        ax.set_title(f"{nominal:.0%} intervals", fontsize=10)
    axes[0].set_ylabel("Coverage (PICP), daylight hours")
    handles, labels = axes[0].get_legend_handles_labels()
    labels = [lab.replace(" 90%", "") for lab in labels]  # one shared legend: "Nominal" means each panel's level
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=5, loc="lower center")  # below, clear of the data
    fig.suptitle("Coverage by test month, with 95% day-bootstrap intervals", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))  # leave room at the bottom for the legend
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data/raw/Solar")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # print tables with dashes on any Windows console
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = load_solar(args.data)
    params = json.loads((out / "lightgbm_params.json").read_text(encoding="utf-8"))

    # --- 1. Split model (V2) and CQR (V3) for every task ------------------------------------
    scores = {nominal: {} for nominal in INTERVALS}  # {level: {task: calibration scores}}
    pieces, cqr_rows, collapsed = [], [], {nominal: 0 for nominal in INTERVALS}
    for tid in ALL_TASK_IDS:
        start = time.perf_counter()
        task = make_task(tid)
        cal, test = split_model(df, task, params)
        if tid in LEAKAGE_CHECK_TASKS:
            check_split_model_leakage(df, task, params, (cal, test))
        cal_day = cal[cal["daylight"]]
        test_day = test[test["daylight"]].copy()
        for nominal in INTERVALS:
            lo_c, hi_c = level_columns(nominal)
            s = conformity_scores(cal_day["power"], cal_day[lo_c], cal_day[hi_c])
            scores[nominal][tid] = s
            Q = cqr_quantile(s, 1 - nominal)
            log = {}
            L, U = apply_adjustment(test_day[lo_c], test_day[hi_c], Q, log)
            collapsed[nominal] += log.get("collapsed_rows", 0)
            t = tag(nominal)
            test_day[f"V2_lo_{t}"], test_day[f"V2_hi_{t}"] = test_day[lo_c], test_day[hi_c]
            test_day[f"V3_lo_{t}"], test_day[f"V3_hi_{t}"] = L, U
            cqr_rows.append({"task": tid, "month": task.label, "level": nominal, "n_cal": s.size, "Q": Q,
                             "cal_picp_V2": float(np.mean(s <= 0))})
        pieces.append(test_day)
        print(f"split model + CQR: task {tid} ({task.label}) [{time.perf_counter() - start:.0f}s]", flush=True)

    rows = pd.concat(pieces, ignore_index=True)
    rows["day"] = rows["issue_time"].dt.tz_localize(None)  # forecast day, as a plain timestamp
    rows = rows.sort_values(["day", "zone", "time"]).reset_index(drop=True)  # ACI walks day by day

    # --- 2. ACI (V4): choose gamma on tasks 1-3, then run tasks 1-15 ------------------------
    dev = rows[rows["task"].isin(DEV_TASK_IDS)].reset_index(drop=True)
    selection = []
    for gamma in ACI_GAMMAS:
        for nominal in INTERVALS:
            L, U, _, _ = run_aci(dev, scores, nominal, gamma)
            stats = interval_stats(dev["power"].to_numpy(), L, U, nominal)
            selection.append({"gamma": gamma, "level": nominal, "picp_dev": stats["picp"],
                              "mpiw_dev": stats["mpiw"], "winkler_dev": stats["winkler"]})
    selection = pd.DataFrame(selection)
    s90 = selection[selection["level"] == 0.9].copy()
    s90["abs_gap"] = (s90["picp_dev"] - 0.9).abs().round(4)  # equal to 4 decimals counts as a tie
    gamma = float(s90.sort_values(["abs_gap", "winkler_dev"]).iloc[0]["gamma"])
    selection["chosen"] = selection["gamma"] == gamma
    selection.to_csv(out / "tables" / "aci_gamma_selection.csv", index=False, float_format="%.6g", encoding="utf-8")
    print(f"ACI gamma chosen on tasks {DEV_TASK_IDS}: {gamma}")

    paths, edge_rows = [], []
    for nominal in INTERVALS:
        L, U, path, counts = run_aci(rows, scores, nominal, gamma)
        rows[f"V4_lo_{tag(nominal)}"], rows[f"V4_hi_{tag(nominal)}"] = L, U
        paths.append(path.assign(level=nominal))
        edge_rows.append({"level": nominal, "gamma": gamma, **counts,
                          "cqr_collapsed_rows": collapsed[nominal]})
        check_aci_is_causal(rows, scores, nominal, gamma)
    pd.concat(paths).to_csv(out / "tables" / "aci_alpha_path.csv", index=False, float_format="%.6g", encoding="utf-8")
    pd.DataFrame(cqr_rows).to_csv(out / "tables" / "cqr_adjustments.csv", index=False, float_format="%.6g",
                                  encoding="utf-8")
    pd.DataFrame(edge_rows).to_csv(out / "tables" / "track2_diagnostics.csv", index=False, encoding="utf-8")

    # --- 3. V1 from Track 1; keep the scored months ------------------------------------------
    v1 = pd.read_csv(out / "predictions" / "track1_lightgbm.csv")
    for col in ("time", "issue_time"):
        v1[col] = pd.to_datetime(v1[col], utc=True)
    v1["day"] = v1["issue_time"].dt.tz_localize(None)
    scored = rows[rows["task"].isin(SCORED_TASK_IDS)].reset_index(drop=True)
    v1_cols = {}
    for nominal in INTERVALS:
        lo_c, hi_c = level_columns(nominal)
        v1_cols.update({lo_c: f"V1_lo_{tag(nominal)}", hi_c: f"V1_hi_{tag(nominal)}"})
    scored = scored.merge(v1[v1["daylight"]][["zone", "time", *v1_cols]].rename(columns=v1_cols),
                          on=["zone", "time"], how="left", validate="one_to_one")
    if scored.filter(like="V1_").isna().any().any():
        raise ValueError("V1 predictions are missing for some daylight test rows; rerun run_track1.py")

    # Forecast cloud-cover terciles, cut on daylight rows before the scored period.
    early = df[is_daylight(df) & (df["time"] < TCC_CUTOFF)]
    cuts = early["tcc"].quantile([1 / 3, 2 / 3]).to_numpy()
    scored["tercile"] = pd.cut(scored["tcc"], [-np.inf, cuts[0], cuts[1], np.inf], labels=TERCILES)
    scored["month"] = scored["task"].map(lambda tid: make_task(tid).label)  # e.g. "2013-07"

    # Per-row coverage, width and Winkler for every variant and level.
    y = scored["power"].to_numpy()
    for nominal in INTERVALS:
        t = tag(nominal)
        for v in NAMES:
            L, U = scored[f"{v}_lo_{t}"].to_numpy(), scored[f"{v}_hi_{t}"].to_numpy()
            scored[f"cov_{v}_{t}"] = ((y >= L) & (y <= U)).astype(float)
            scored[f"wid_{v}_{t}"] = U - L
    days = np.sort(scored["day"].unique())

    # --- 4. Tables -------------------------------------------------------------------------
    t2, t2_disp, diffs, diffs_disp, t3, t3_disp, t4 = [], [], [], [], [], [], []
    for nominal in INTERVALS:
        t = tag(nominal)
        for v, name in NAMES.items():
            L, U = scored[f"{v}_lo_{t}"].to_numpy(), scored[f"{v}_hi_{t}"].to_numpy()
            st = interval_stats(y, L, U, nominal)
            p_ci = boot_ci(scored, days, f"cov_{v}_{t}")
            w_ci = boot_ci(scored, days, f"wid_{v}_{t}")
            t2.append({"variant": name, "level": nominal, "picp": st["picp"], "picp_lo": p_ci[0], "picp_hi": p_ci[1],
                       "mpiw": st["mpiw"], "mpiw_lo": w_ci[0], "mpiw_hi": w_ci[1], "winkler": st["winkler"]})
            t2_disp.append({"Variant": name, "Level": f"{nominal:.0%}", "PICP [95% CI]": pct_ci(st["picp"], *p_ci),
                            "MPIW [95% CI]": num_ci(st["mpiw"], *w_ci), "Winkler": f"{st['winkler']:.3f}"})
            for terc in TERCILES:
                mask = (scored["tercile"] == terc).to_numpy()
                stt = interval_stats(y[mask], L[mask], U[mask], nominal)
                ci = boot_ci(scored, days, f"cov_{v}_{t}", mask)
                t3.append({"variant": name, "tercile": terc, "level": nominal, "n_rows": int(mask.sum()),
                           "picp": stt["picp"], "picp_lo": ci[0], "picp_hi": ci[1], "mpiw": stt["mpiw"]})
                t3_disp.append({"Variant": name, "Forecast cloud": terc, "Level": f"{nominal:.0%}",
                                "PICP [95% CI]": pct_ci(stt["picp"], *ci), "MPIW": f"{stt['mpiw']:.3f}"})
            for month in sorted(scored["month"].unique()):
                mask = (scored["month"] == month).to_numpy()
                stm = interval_stats(y[mask], L[mask], U[mask], nominal)
                p_ci = boot_ci(scored, days, f"cov_{v}_{t}", mask)
                w_ci = boot_ci(scored, days, f"wid_{v}_{t}", mask)
                t4.append({"variant": name, "month": month, "level": nominal, "picp": stm["picp"],
                           "picp_lo": p_ci[0], "picp_hi": p_ci[1], "mpiw": stm["mpiw"],
                           "mpiw_lo": w_ci[0], "mpiw_hi": w_ci[1]})
        for (a, b) in (("V3", "V2"), ("V4", "V3")):
            scored["_dcov"] = scored[f"cov_{a}_{t}"] - scored[f"cov_{b}_{t}"]
            scored["_dwid"] = scored[f"wid_{a}_{t}"] - scored[f"wid_{b}_{t}"]
            c_ci, w_ci = boot_ci(scored, days, "_dcov"), boot_ci(scored, days, "_dwid")
            label = f"{NAMES[a]} minus {NAMES[b]}"
            diffs.append({"comparison": label, "level": nominal,
                          "d_picp": scored["_dcov"].mean(), "d_picp_lo": c_ci[0], "d_picp_hi": c_ci[1],
                          "d_mpiw": scored["_dwid"].mean(), "d_mpiw_lo": w_ci[0], "d_mpiw_hi": w_ci[1]})
            diffs_disp.append({"Comparison": label, "Level": f"{nominal:.0%}",
                               "Coverage change, points [95% CI]": num_ci(100 * scored["_dcov"].mean(),
                                                                          100 * c_ci[0], 100 * c_ci[1], 1),
                               "Width change [95% CI]": num_ci(scored["_dwid"].mean(), *w_ci)})
    scored = scored.drop(columns=["_dcov", "_dwid"])

    n_days, n_rows = len(days), len(scored)
    base_note = (f"Tasks 4-15 (July 2013 - June 2014), daylight hours, all zones pooled ({n_rows:,} hours, "
                 f"{n_days} forecast days). 95% CIs resample whole forecast days (B = 1000, seed 0). "
                 f"ACI step size gamma = {gamma}, chosen on tasks 1-3.")
    save_table(out / "tables" / "T2_track2", pd.DataFrame(t2), pd.DataFrame(t2_disp),
               note=base_note + " PICP = share of outcomes inside the interval (coverage); "
                    "MPIW = mean width as a fraction of capacity; Winkler = width plus a penalty for misses "
                    "(lower is better).")
    save_table(out / "tables" / "T2_paired_differences", pd.DataFrame(diffs), pd.DataFrame(diffs_disp),
               note=base_note + " Differences are paired: both variants are scored on the same resampled days.")
    save_table(out / "tables" / "T3_by_cloud", pd.DataFrame(t3), pd.DataFrame(t3_disp),
               note=base_note + f" Terciles of hourly forecast total cloud cover (tcc), cut at {cuts[0]:.3f} and "
                    f"{cuts[1]:.3f}: the 1/3 and 2/3 quantiles of daylight tcc before {TCC_CUTOFF:%Y-%m-%d}.")
    t4 = pd.DataFrame(t4)
    t4_disp = t4.assign(level=t4["level"].map("{:.0%}".format),
                        picp=[pct_ci(r.picp, r.picp_lo, r.picp_hi) for r in t4.itertuples()],
                        mpiw=[num_ci(r.mpiw, r.mpiw_lo, r.mpiw_hi) for r in t4.itertuples()])
    t4_disp = t4_disp[["variant", "month", "level", "picp", "mpiw"]].rename(columns={
        "variant": "Variant", "month": "Month", "level": "Level", "picp": "PICP [95% CI]", "mpiw": "MPIW [95% CI]"})
    save_table(out / "tables" / "T4_by_month", t4, t4_disp, note=base_note)

    # --- 5. Figures ------------------------------------------------------------------------
    v1_scored = v1[v1["task"].isin(SCORED_TASK_IDS)]
    blocks = plot_fan_chart(v1_scored, out / "figures" / "F1_fan_chart.png")
    blocks.to_csv(out / "tables" / "F1_blocks.csv", index=False, float_format="%.6g", encoding="utf-8")
    plot_monthly_coverage(t4, out / "figures" / "F2_monthly_coverage.png")

    print("\n" + (out / "tables" / "T2_track2.md").read_text(encoding="utf-8"))
    print((out / "tables" / "T2_paired_differences.md").read_text(encoding="utf-8"))
    print(pd.DataFrame(edge_rows).to_string(index=False))

    # Red flag: coverage at nominal (to 0.1 point) in every slice would be suspicious.
    all_slices = pd.concat([pd.DataFrame(t2)[["level", "picp"]], pd.DataFrame(t3)[["level", "picp"]],
                            t4[["level", "picp"]]])
    if ((all_slices["picp"] - all_slices["level"]).abs() < 0.001).all():
        print("\nRED FLAG: coverage sits exactly at nominal in every slice. Check for leakage.")



if __name__ == "__main__":
    main()
