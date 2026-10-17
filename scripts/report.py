"""Fill README.md's results sections from the generated tables, so no number is typed by hand.

    python scripts/report.py

README.md contains pairs of markers like

    <!-- BEGIN T1 -->
    (anything here is replaced)
    <!-- END T1 -->

and this script replaces what is between each pair with a generated block: a table from
results/tables/*.md, or text built from the result CSVs (the short answer and the key
numbers). Run it after run_track1.py and run_track2.py; scripts/run_all.py does that.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from pvcal.config import WINNER_AVG_PINBALL

LINEAR, LGBM, LGBM_CS = "Linear QR", "LightGBM QR", "LightGBM QR, clear-sky normalised"
QR_MODELS = (LINEAR, LGBM, LGBM_CS)
BENCHMARK = "Official benchmark (last year, same hour)"
PERSISTENCE = "Persistence ensemble (last 30 days)"


def fill(text: str, name: str, content: str) -> str:
    """Replace everything between <!-- BEGIN name --> and <!-- END name --> with `content`."""
    # DOTALL lets ".*?" span several lines; "?" makes it stop at the first END marker.
    pattern = re.compile(rf"(<!-- BEGIN {name} -->).*?(<!-- END {name} -->)", re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"README.md has no {name} markers")
    return pattern.sub(lambda m: m.group(1) + "\n" + content.strip() + "\n" + m.group(2), text)


def table(tables: Path, name: str) -> str:
    return (tables / f"{name}.md").read_text(encoding="utf-8")


def numbers(out: Path) -> dict:
    """Every number the short answer and key-numbers block quote, read from generated files."""
    tables = out / "tables"
    t1 = pd.read_csv(tables / "T1_track1.csv").set_index("model")
    t2 = pd.read_csv(tables / "T2_track2.csv")
    diffs = pd.read_csv(tables / "T2_paired_differences.csv")
    t4 = pd.read_csv(tables / "T4_by_month.csv")
    t3 = pd.read_csv(tables / "T3_by_cloud.csv")
    t5 = pd.read_csv(tables / "T5_conditional.csv").set_index(["variant", "level"])
    gamma = pd.read_csv(tables / "aci_gamma_selection.csv").query("chosen")["gamma"].iloc[0]
    half_life = pd.read_csv(tables / "weighted_half_life_selection.csv").query("chosen")["half_life_days"].iloc[0]

    def cov(variant, level):
        return t2[(t2["variant"] == variant) & (t2["level"] == level)].iloc[0]

    def diff(comparison_start, level):
        r = diffs[diffs["comparison"].str.startswith(comparison_start) & (diffs["level"] == level)]
        return r.iloc[0]

    def months(variant):
        return t4[(t4["variant"] == variant) & (t4["level"] == 0.9)]

    def cloud(variant):
        return t3[(t3["variant"] == variant) & (t3["level"] == 0.9)].set_index("tercile")

    v1_month, v3_month = months("V1 raw-full"), months("V3 CQR")
    best = t1.loc[list(QR_MODELS), "pinball_all"].idxmin()
    return {
        "t1": t1, "best": best, "gamma": gamma, "half_life": int(half_life),
        "t5": lambda variant, level=0.9: t5.loc[(variant, level)],
        "params": json.loads((out / "lightgbm_params.json").read_text(encoding="utf-8")),
        "cov": cov, "diff": diff,
        "v1_month_min": v1_month.loc[v1_month["picp"].idxmin()],
        "v1_month_max": v1_month.loc[v1_month["picp"].idxmax()],
        "v3_month_min": v3_month.loc[v3_month["picp"].idxmin()],
        "v3_month_max": v3_month.loc[v3_month["picp"].idxmax()],
        "v1_cloud": cloud("V1 raw-full"), "v3_cloud": cloud("V3 CQR"),
    }


def side(row, nominal: float) -> str:
    """Where a coverage estimate sits relative to its promise, judged by its 95% confidence interval."""
    if row["picp_hi"] < nominal:
        return "below"
    if row["picp_lo"] > nominal:
        return "above"
    return "consistent with"


VERDICT = {"below": "falls short of its promise", "consistent with": "is on target",
           "above": "covers more than it promises"}  # a full verdict, for one interval
TARGET = {"below": "below target", "consistent with": "on target", "above": "above target"}  # a short one


def short_answer(n: dict) -> str:
    """Four bullet points: accuracy, raw coverage, recalibration, follow-up fixes.

    Every number and every verdict word is chosen by the results (a verdict comes from
    whether the 95% confidence interval contains the promised level), so the text cannot
    drift from the tables when they change.
    """
    t1, cov = n["t1"], n["cov"]
    v1_90, v1_80 = cov("V1 raw-full", 0.9), cov("V1 raw-full", 0.8)
    v2_90, v3_90, v4_90 = cov("V2 raw-split", 0.9), cov("V3 CQR", 0.9), cov("V4 ACI", 0.9)
    lo, hi = n["v1_month_min"], n["v1_month_max"]
    c1, c3 = n["v1_cloud"], n["v3_cloud"]
    d32 = n["diff"]("V3", 0.9)

    lgbm, cs = t1.loc[LGBM, "pinball_all"], t1.loc[LGBM_CS, "pinball_all"]
    shown_cs, shown_lgbm = round(cs, 4), round(lgbm, 4)  # compare at the precision shown
    change = ("improves it to" if shown_cs < shown_lgbm else "leaves it at" if shown_cs == shown_lgbm
              else "worsens it to")
    accuracy = (f"**Accuracy.** LightGBM QR averages a pinball loss of {lgbm:.4f} over the 12 test months, against "
                f"{t1.loc[PERSISTENCE, 'pinball_all']:.4f} for the persistence ensemble and "
                f"{t1.loc[BENCHMARK, 'pinball_all']:.4f} for the official benchmark; predicting power relative to an "
                f"empirical clear-sky envelope {change} {cs:.4f}. The competition winner's reported "
                f"{WINNER_AVG_PINBALL:.4f} is not reproduced here, so that comparison is approximate.")

    raw = (f"**Raw coverage.** Over the year, the raw LightGBM 90% interval {VERDICT[side(v1_90, 0.9)]}: it covers "
           f"{v1_90['picp']:.1%} of daylight hours. The 80% interval {VERDICT[side(v1_80, 0.8)]}, at "
           f"{v1_80['picp']:.1%}. Coverage also swings: the 90% interval covers {lo['picp']:.1%} in {lo['month']} "
           f"but {hi['picp']:.1%} in {hi['month']}, and {c1.loc['low', 'picp']:.1%} under the clearest third of "
           f"forecasts but {c1.loc['high', 'picp']:.1%} under the cloudiest.")

    s3, s4 = side(v3_90, 0.9), side(v4_90, 0.9)
    where = f"both {TARGET[s3]}" if s3 == s4 else f"CQR {TARGET[s3]}, ACI {TARGET[s4]}"
    relative = abs(d32["d_mpiw"]) / v2_90["mpiw"]  # the width change as a share of the split model's width
    cost = "a small width cost" if relative < 0.10 else "a width cost"
    w3, w4, w1 = f"{v3_90['winkler']:.3f}", f"{v4_90['winkler']:.3f}", f"{v1_90['winkler']:.3f}"
    calibrated_w = f"{w3} for both CQR and ACI" if w3 == w4 else f"{w3} for CQR and {w4} for ACI"
    beaten = min(v3_90["winkler"], v4_90["winkler"]) < v1_90["winkler"]
    recal = (f"**Recalibration.** Using the 60 days before each month, CQR brings yearly 90% coverage to "
             f"{v3_90['picp']:.1%} and ACI to {v4_90['picp']:.1%} ({where}), at {cost}: CQR's intervals are "
             f"{abs(d32['d_mpiw']):.3f} of capacity {'wider' if d32['d_mpiw'] > 0 else 'narrower'} than those of "
             f"the uncalibrated split model. But the swings stay: CQR's monthly 90% coverage still ranges from "
             f"{n['v3_month_min']['picp']:.1%} to {n['v3_month_max']['picp']:.1%}, and it covers "
             f"{c3.loc['low', 'picp']:.1%} under the clearest forecasts against {c3.loc['high', 'picp']:.1%} "
             f"under the cloudiest. "
             + ("A recalibrated variant improves on" if beaten else "Neither method beats")
             + f" the raw full-data model's 90% Winkler score (lower is better): {w1} for the raw model, "
               f"against {calibrated_w}.")
    v3, v5, v6 = n["t5"]("V3 CQR"), n["t5"]("V5 Mondrian CQR"), n["t5"]("V6 weighted CQR")
    d53 = n["diff"]("V5", 0.9)
    gap3, gap5 = 100 * v3["cloud_gap"], 100 * v5["cloud_gap"]
    shrink = "shrinks" if abs(gap5) < abs(gap3) else "does not shrink"
    zero = (", no longer distinguishable from zero" if v5["cloud_gap_lo"] <= 0 <= v5["cloud_gap_hi"] else "")
    mad3, mad5, mad6 = (f"{100 * v['month_mad']:.1f}" for v in (v3, v5, v6))
    weighted = ("reduces" if float(mad6) < float(mad3) else "does not reduce")
    fixes = (f"**Follow-up fixes.** Calibrating each forecast-cloud tercile separately (Mondrian CQR) {shrink} the "
             f"90% clear-minus-cloudy coverage gap from {gap3:.1f} to {gap5:.1f} points{zero}, for intervals "
             f"{abs(d53['d_mpiw']):.3f} of capacity {'wider' if d53['d_mpiw'] > 0 else 'narrower'} than CQR's. "
             f"Weighting recent calibration days more (half-life {n['half_life']} days, chosen on the development "
             f"months) {weighted} the month-to-month misses: the mean monthly distance from 90% is {mad6} points, "
             f"against {mad3} for CQR and {mad5} for Mondrian CQR. Both were added after the test results showed the "
             f"problem; neither has a setting tuned on the test months.")
    return "\n".join(f"- {bullet}" for bullet in (accuracy, raw, recal, fixes))


def key_numbers(n: dict) -> str:
    t1, best, cov, diff = n["t1"], n["best"], n["cov"], n["diff"]
    gain = 1 - t1.loc[best, "pinball_all"] / t1.loc[BENCHMARK, "pinball_all"]
    lines = [
        f"- Track 1, average pinball loss over tasks 4-15: linear QR {t1.loc[LINEAR, 'pinball_all']:.4f}, "
        f"LightGBM QR {t1.loc[LGBM, 'pinball_all']:.4f}, clear-sky-normalised LightGBM QR "
        f"{t1.loc[LGBM_CS, 'pinball_all']:.4f}, persistence ensemble {t1.loc[PERSISTENCE, 'pinball_all']:.4f}, "
        f"official benchmark {t1.loc[BENCHMARK, 'pinball_all']:.4f}. Reported winner: {WINNER_AVG_PINBALL:.4f} (approximate).",
        f"- {best} improves on the official benchmark by {gain:.1%}.",
        f"- LightGBM settings chosen on tasks 1-3: {', '.join(f'{k} = {v}' for k, v in n['params'].items())}. "
        f"Chosen on tasks 1-3 as well: ACI step size gamma = {n['gamma']:g}, weighted-CQR half-life "
        f"{n['half_life']} days.",
    ]
    for level in (0.9, 0.8):
        parts = ", ".join(f"{v.split()[0]} {cov(v, level)['picp']:.1%} (width {cov(v, level)['mpiw']:.3f})"
                          for v in ("V1 raw-full", "V2 raw-split", "V3 CQR", "V4 ACI", "V5 Mondrian CQR",
                                    "V6 weighted CQR"))
        lines.append(f"- {level:.0%} intervals, coverage over tasks 4-15: {parts}.")
    for level in (0.9, 0.8):
        d32, d43 = diff("V3", level), diff("V4", level)
        lines.append(
            f"- {level:.0%}: CQR minus raw-split changes coverage by {100 * d32['d_picp']:+.1f} points "
            f"[{100 * d32['d_picp_lo']:+.1f}, {100 * d32['d_picp_hi']:+.1f}] and width by {d32['d_mpiw']:+.3f}; "
            f"ACI minus CQR by {100 * d43['d_picp']:+.1f} points [{100 * d43['d_picp_lo']:+.1f}, "
            f"{100 * d43['d_picp_hi']:+.1f}] and {d43['d_mpiw']:+.3f}.")
    c = n["v1_cloud"]
    lines.append(f"- Raw 90% coverage (V1) by forecast cloud tercile: low {c.loc['low', 'picp']:.1%}, "
                 f"mid {c.loc['mid', 'picp']:.1%}, high {c.loc['high', 'picp']:.1%}.")
    gaps = "; ".join(f"{v.split()[0]} {100 * n['t5'](v)['cloud_gap']:+.1f} points "
                     f"[{100 * n['t5'](v)['cloud_gap_lo']:+.1f}, {100 * n['t5'](v)['cloud_gap_hi']:+.1f}], monthly "
                     f"miss {100 * n['t5'](v)['month_mad']:.1f}"
                     for v in ("V1 raw-full", "V3 CQR", "V5 Mondrian CQR", "V6 weighted CQR"))
    lines.append(f"- 90% clear-minus-cloudy coverage gap and mean monthly distance from 90% (points): {gaps}.")
    return "\n".join(lines)


def track1_note(n: dict, out: Path) -> str:
    """How the quantile models compare with the baselines, and the leakage red-flag status."""
    t1 = n["t1"]
    diag = pd.read_csv(out / "tables" / "diagnostics_track1.csv").set_index("model")
    # Daylight hours in tasks 4-15: the same hours Track 2 scores (one variant and level of T3, summed).
    t3 = pd.read_csv(out / "tables" / "T3_by_cloud.csv")
    n_day = int(t3[(t3["variant"] == "V1 raw-full") & (t3["level"] == 0.9)]["n_rows"].sum())
    lines = []
    for model in QR_MODELS:
        vs_bench = 1 - t1.loc[model, "pinball_all"] / t1.loc[BENCHMARK, "pinball_all"]
        vs_pers = 1 - t1.loc[model, "pinball_all"] / t1.loc[PERSISTENCE, "pinball_all"]
        failed = int(diag.loc[model, "nonconverged_fits"])
        # Only the linear model's solver can fail to converge; LightGBM always builds its fixed number of trees.
        converged = ("" if model != LINEAR else
                     " Every fit converged." if failed == 0 else f" {failed:,} fits did not converge.")
        d = diag.loc[model]
        lines.append(f"{model} improves on the official benchmark by {vs_bench:.1%} and on the persistence "
                     f"ensemble by {vs_pers:.1%} (average pinball, all hours). Its separately fitted levels crossed "
                     f"(a lower level's quantile above a higher one's) in {d['crossing_rows']:,} of the {n_day:,} "
                     f"daylight hours, but the crossings were small: the largest drop between neighbouring levels "
                     f"averaged {d['mean_crossing_size']:.4f} of capacity (at most {d['max_crossing_size']:.3f}), and "
                     f"sorting moved quantile values by {d['mean_sort_shift']:.5f} of capacity on average. "
                     f"{d['capped_values']:,} values above 1 were capped.{converged}")
    beaten = [m for m in QR_MODELS if t1.loc[m, "pinball_all"] < WINNER_AVG_PINBALL]
    lines.append("No model beats the reported winner's average, so the leakage red flag (a score below "
                 f"{WINNER_AVG_PINBALL}) did not trigger." if not beaten else
                 f"RED FLAG: {', '.join(beaten)} beat the reported winner; treat as leakage until proven otherwise.")
    return "\n\n".join(lines)


def runtime_note(out: Path) -> str:
    path = out / "runtime.json"
    if not path.exists():
        return "Run `scripts/run_all.py` to record how long each step takes on your machine."
    rt = json.loads(path.read_text(encoding="utf-8"))
    steps = rt["minutes_by_step"]
    detail = ", ".join(f"`{script}` {m} min" for script, m in steps.items())
    return (f"The last full run took {sum(steps.values()):.1f} minutes before the report step "
            f"({detail}) on a machine with {rt['cpu_count']} logical CPU cores.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="results")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out, readme = Path(args.out), Path(args.readme)
    tables = out / "tables"

    n = numbers(out)
    text = readme.read_text(encoding="utf-8")
    text = fill(text, "SHORT", short_answer(n))
    text = fill(text, "KEY", key_numbers(n))
    text = fill(text, "T1_NOTE", track1_note(n, out))
    text = fill(text, "RUNTIME", runtime_note(out))
    for name in ("T1_track1", "tuning_lightgbm", "T2_track2", "T2_paired_differences", "T3_by_cloud", "T4_by_month",
                 "T5_conditional"):
        text = fill(text, name, table(tables, name))
    readme.write_text(text, encoding="utf-8")
    print(f"Filled {readme} from {tables}.")
    print("\nShort answer:\n" + short_answer(n))


if __name__ == "__main__":
    main()
