"""Track 1: score five forecasters on the 12 scored months with the competition's metric.

    python scripts/run_track1.py --data data/raw/Solar

Models: the official benchmark (unmodified), the persistence ensemble, linear QR, LightGBM QR
(settings frozen by scripts/tune_lightgbm.py) and LightGBM QR on clear-sky-normalised power
(same frozen settings). Tasks 4-15 (July 2013 - June 2014).

Per task and model it computes
  pinball_all     mean pinball loss over the 99 levels, all hours, all 3 zones (the GEFCom2014 score)
  pinball_day     the same on daylight hours only
  mae_median_day  mean absolute error of the median (the 0.50 quantile) on daylight hours

Writes (results/...):
  tables/T1_track1.csv/.md        averages over tasks 4-15, plus the reported winner
  tables/T1_track1_by_task.csv    one row per model and task
  tables/diagnostics_track1.csv   per model: how often and how much quantiles crossed before
                                  sorting, capped values, non-converged linear fits
  figures/F3_reliability.png      observed share of y <= q_tau against tau
  figures/diagnostics/radiation_vs_power.png
  predictions/track1_lightgbm.csv LightGBM quantiles for Track 2 (variant V1; gitignored)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pvcal.baselines import finalize, persistence_ensemble, yearly_persistence
from pvcal.config import LEAKAGE_CHECK_TASKS, SEED, WINNER_AVG_PINBALL
from pvcal.data import load_solar
from pvcal.features import is_daylight
from pvcal.leakage import assert_no_lookahead
from pvcal.metrics import TAUS, pinball, reliability
from pvcal.models import lightgbm_cs_qr, lightgbm_qr, linear_qr
from pvcal.tables import save_table
from pvcal.tasks import scored_tasks, split

BENCHMARK = "Official benchmark (last year, same hour)"
PERSISTENCE = "Persistence ensemble (last 30 days)"
LINEAR = "Linear QR"
LGBM = "LightGBM QR"
LGBM_CS = "LightGBM QR, clear-sky normalised"
QR_MODELS = (LINEAR, LGBM, LGBM_CS)  # expensive: leakage-checked on LEAKAGE_CHECK_TASKS only
WINNER = "GEFCom2014 winner (reported, approximate)"
SAVED_LEVELS = (0.05, 0.10, 0.50, 0.90, 0.95)  # the LightGBM quantiles Track 2 needs
MEDIAN = int(np.flatnonzero(np.isclose(TAUS, 0.5))[0])  # column of the 0.50 quantile


def official_benchmark(train, test, log=None):
    """Left unmodified (no finalize) so its scores match the competition's own."""
    return yearly_persistence(train, test)


def persistence(train, test, log=None):
    return finalize(persistence_ensemble(train, test), is_daylight(test), log)


def task_scores(y: np.ndarray, q: np.ndarray, day: np.ndarray) -> dict:
    return {
        "pinball_all": pinball(y, q),
        "pinball_day": pinball(y[day], q[day]),
        "mae_median_day": float(np.mean(np.abs(y[day] - q[day, MEDIAN]))),
    }


def plot_reliability(daytime: dict, path: Path) -> None:
    """For a calibrated forecast, a share tau of outcomes falls below the tau-quantile: the diagonal."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Perfect calibration")
    for name, (y, q) in daytime.items():
        ax.plot(TAUS, reliability(y, q), lw=1.6, label=name)
    ax.set_xlabel("Quantile level tau")
    ax.set_ylabel("Observed share of outcomes <= forecast quantile")
    ax.set_title("Reliability, daylight hours, tasks 4-15", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_radiation_vs_power(df: pd.DataFrame, path: Path) -> None:
    """Sanity check: forecast sunshine should line up with observed output."""
    day = df[is_daylight(df)].sample(6000, random_state=SEED)  # a sample keeps the plot readable
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for zone, rows in day.groupby("zone"):
        ax.scatter(rows["ssrd"], rows["power"], s=3, alpha=0.35, label=f"Zone {zone}")
    ax.set_xlabel("Forecast surface solar radiation, SSRD (W/m^2, hourly mean)")
    ax.set_ylabel("Observed power (fraction of capacity)")
    ax.set_title("Forecast radiation vs observed power (6,000 daylight hours)", fontsize=10)
    ax.legend(frameon=False, markerscale=4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data/raw/Solar")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # print tables with dashes on any Windows console
    out = Path(args.out)
    for sub in ("tables", "figures/diagnostics", "predictions"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    params = json.loads((out / "lightgbm_params.json").read_text(encoding="utf-8"))
    print(f"LightGBM settings (frozen on dev tasks): {params}")
    models = {
        BENCHMARK: official_benchmark,
        PERSISTENCE: persistence,
        LINEAR: linear_qr,
        LGBM: partial(lightgbm_qr, params=params),
        LGBM_CS: partial(lightgbm_cs_qr, params=params),
    }

    df = load_solar(args.data)
    plot_radiation_vs_power(df, out / "figures" / "diagnostics" / "radiation_vs_power.png")

    by_task, diagnostics, daytime, saved = [], [], {}, []
    for name, forecast in models.items():
        log = {}  # diagnostics for the scored runs only (leakage reruns are not counted)
        ys, qs = [], []
        for task in scored_tasks():
            start = time.perf_counter()
            if name not in QR_MODELS or task.task_id in LEAKAGE_CHECK_TASKS:
                assert_no_lookahead(forecast, df, task)  # fails loudly before anything is scored
            train, test = split(df, task)
            q = forecast(train, test, log=log)
            y, day = test["power"].to_numpy(), is_daylight(test)
            by_task.append({"model": name, "task": task.task_id, "month": task.label, **task_scores(y, q, day)})
            ys.append(y[day])
            qs.append(q[day])
            if name == LGBM:
                cols = {f"q{lvl:.2f}": q[:, int(np.flatnonzero(np.isclose(TAUS, lvl))[0])] for lvl in SAVED_LEVELS}
                saved.append(test[["zone", "time", "issue_time", "power", "tcc"]]
                             .assign(task=task.task_id, daylight=day, **cols))
            print(f"{name}: task {task.task_id} ({task.label}) "
                  f"pinball {by_task[-1]['pinball_all']:.4f}  [{time.perf_counter() - start:.0f}s]", flush=True)
        crossed = log.get("crossing_rows", 0)
        diagnostics.append({
            "model": name,
            "crossing_rows": crossed,
            # How big the crossings were: the largest drop between neighbouring levels, averaged
            # over the rows that crossed, and its maximum; and how far sorting moved values on average.
            "mean_crossing_size": log.get("crossing_size_sum", 0.0) / crossed if crossed else 0.0,
            "max_crossing_size": log.get("crossing_size_max", 0.0),
            "mean_sort_shift": log.get("sort_shift_sum", 0.0) / log["values"] if log.get("values") else 0.0,
            "capped_values": log.get("capped_values", 0),
            "nonconverged_fits": log.get("nonconverged_fits", 0),
        })
        if name != BENCHMARK:  # a point forecast has no meaningful reliability curve
            daytime[name] = (np.concatenate(ys), np.concatenate(qs))

    by_task = pd.DataFrame(by_task)
    by_task.to_csv(out / "tables" / "T1_track1_by_task.csv", index=False, float_format="%.6g", encoding="utf-8")

    metrics = ["pinball_all", "pinball_day", "mae_median_day"]
    t1 = by_task.groupby("model", sort=False)[metrics].mean().reset_index()
    t1 = pd.concat([t1, pd.DataFrame([{"model": WINNER, "pinball_all": WINNER_AVG_PINBALL}])], ignore_index=True)
    display = pd.DataFrame({
        "Model": t1["model"],
        "Pinball, all hours": t1["pinball_all"].map("{:.4f}".format),
        "Pinball, daylight": t1["pinball_day"].map(lambda v: "–" if pd.isna(v) else f"{v:.4f}"),
        "MAE of median, daylight": t1["mae_median_day"].map(lambda v: "–" if pd.isna(v) else f"{v:.4f}"),
    })
    save_table(out / "tables" / "T1_track1", t1, display,
               note="Averages over the 12 scored tasks (July 2013 - June 2014), equal weight per task. "
                    "Pinball loss is averaged over the 99 quantile levels. Lower is better.")
    pd.DataFrame(diagnostics).to_csv(out / "tables" / "diagnostics_track1.csv", index=False, float_format="%.6g",
                                     encoding="utf-8")
    plot_reliability(daytime, out / "figures" / "F3_reliability.png")
    pd.concat(saved, ignore_index=True).to_csv(out / "predictions" / "track1_lightgbm.csv", index=False,
                                               float_format="%.6g", encoding="utf-8")

    print("\n" + (out / "tables" / "T1_track1.md").read_text(encoding="utf-8"))
    print(pd.DataFrame(diagnostics).to_string(index=False))
    beaten = t1[(t1["model"] != WINNER) & (t1["pinball_all"] < WINNER_AVG_PINBALL)]
    if not beaten.empty:
        print(f"\nRED FLAG: {list(beaten['model'])} beat the reported winner ({WINNER_AVG_PINBALL}). "
              "Treat this as leakage until proven otherwise.")


if __name__ == "__main__":
    main()
