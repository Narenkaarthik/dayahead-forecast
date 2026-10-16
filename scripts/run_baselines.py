"""Baselines: check the data, then score two reference forecasts on the 12 test months.

    python scripts/run_baselines.py --data data/raw/Solar

Writes results/baseline_scores.csv (one row per model and task), results/baseline_summary.md,
and two figures in results/figures/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pvcal.baselines import finalize, persistence_ensemble, yearly_persistence
from pvcal.data import load_solar
from pvcal.features import is_daylight
from pvcal.leakage import assert_no_lookahead
from pvcal.metrics import summarize
from pvcal.tasks import scored_tasks, split

# Average pinball loss of the competition winner over the 12 scored tasks, as reported in
# "Probabilistic Energy Forecasting using Quantile Regressions based on a new Nearest
# Neighbors Quantile Filter" (arXiv:1903.07390). Reported, not reproduced here.
WINNER_AVG_PINBALL = 0.0121


def official_benchmark(train, test):
    """The competition's own benchmark, left unmodified so its scores match the official ones."""
    return yearly_persistence(train, test)


def persistence_baseline(train, test):
    return finalize(persistence_ensemble(train, test), is_daylight(test))


MODELS = {
    "Official benchmark (last year, same hour)": official_benchmark,
    "Persistence ensemble (last 30 days)": persistence_baseline,
}
POINT_FORECASTS = {"Official benchmark (last year, same hour)"}  # intervals have zero width


def score_all(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, forecast in MODELS.items():
        for task in scored_tasks():
            assert_no_lookahead(forecast, df, task)  # fail loudly before any score is recorded
            train, test = split(df, task)
            q = forecast(train, test)
            scores = summarize(test["power"], q, is_daylight(test))
            rows.append({"model": name, "task": task.task_id, "month": task.label, **scores})
    return pd.DataFrame(rows)


def summary_table(scores: pd.DataFrame) -> str:
    mean = scores.groupby("model", sort=False).mean(numeric_only=True)
    lines = [
        "| Model | Pinball, all hours | Pinball, daylight | 80% coverage | 90% coverage | 90% width | 90% Winkler |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in mean.iterrows():
        if name in POINT_FORECASTS:
            interval_cells = ["n/a (point forecast)"] + ["–"] * 3
        else:
            interval_cells = [
                f"{r.picp80_day:.1%}",
                f"{r.picp90_day:.1%}",
                f"{r.width90_day:.3f}",
                f"{r.winkler90_day:.3f}",
            ]
        lines.append(f"| {name} | {r.pinball_all:.4f} | {r.pinball_day:.4f} | " + " | ".join(interval_cells) + " |")
    lines.append(f"| GEFCom2014 winner (reported, not reproduced) | {WINNER_AVG_PINBALL:.4f} | – | – | – | – | – |")
    return "\n".join(lines)


def plot_hourly_profile(df: pd.DataFrame, path: Path) -> None:
    profile = df.groupby([df["time"].dt.hour, "zone"])["power"].mean().unstack()
    order = [*range(18, 24), *range(0, 18)]  # start in the evening so daylight is one contiguous curve
    profile = profile.loc[order]
    x = range(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for zone in profile.columns:
        ax.plot(x, profile[zone], marker="o", ms=3, label=f"Zone {zone}")
    ax.axvline(order.index(2), color="grey", ls=":", lw=1)
    ax.set_xticks(list(x)[::2], [f"{h:02d}" for h in order[::2]])
    ax.set_xlabel("Hour of timestamp (UTC, hour-ending)")
    ax.set_ylabel("Mean power (fraction of capacity)")
    ax.set_title("Output peaks at 02:00-03:00 UTC, i.e. near local noon at UTC+10: timestamps are UTC",
                 fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coverage_by_month(scores: pd.DataFrame, model: str, path: Path) -> None:
    s = scores[scores["model"] == model]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for col, nominal, color in (("picp90_day", 0.9, "tab:blue"), ("picp80_day", 0.8, "tab:orange")):
        ax.plot(s["month"], s[col], marker="o", color=color, label=f"{nominal:.0%} interval")
        ax.axhline(nominal, color=color, ls="--", lw=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Observed coverage, daylight hours")
    ax.set_title(f"{model}: coverage vs. nominal (dashed), by test month", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data/raw/Solar", help="extracted 'Solar' folder of GEFCom2014-S_V2")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # print tables with dashes on any Windows console

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = load_solar(args.data)
    peak = df.groupby(df["time"].dt.hour)["power"].mean().idxmax()
    print(f"Loaded {len(df):,} rows, {df['time'].min():%Y-%m-%d} to {df['time'].max():%Y-%m-%d}; "
          f"mean output peaks at {peak:02d}:00 UTC")
    plot_hourly_profile(df, out / "figures" / "hourly_profile.png")

    scores = score_all(df)
    scores.to_csv(out / "baseline_scores.csv", index=False, float_format="%.5f")
    table = summary_table(scores)
    (out / "baseline_summary.md").write_text(
        "Averages over the 12 scored tasks (July 2013 - June 2014). "
        "Coverage, width and Winkler use daylight hours only.\n\n" + table + "\n",
        encoding="utf-8",  # without this, Windows writes cp1252 and the dashes turn into garbage
    )
    plot_coverage_by_month(scores, "Persistence ensemble (last 30 days)", out / "figures" / "baseline_coverage_by_month.png")
    print("All models passed the leakage check.\n")
    print(table)


if __name__ == "__main__":
    main()
