"""Choose LightGBM's settings on the development tasks only, then freeze them.

    python scripts/tune_lightgbm.py --data data/raw/Solar

Tries each of the 8 configurations in pvcal.config.LGBM_GRID on dev tasks 1-3 (April-June
2013), at the five levels in TUNE_TAUS, and scores the mean daytime pinball loss. The
lowest wins; ties (equal to 5 decimal places) go to fewer trees. The scored test months
(tasks 4-15) are never touched here, so the choice cannot be tuned toward the test results.

Writes results/tables/tuning_lightgbm.csv and .md, and results/lightgbm_params.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pvcal.config import DEV_TASK_IDS, LEARNING_RATE, LGBM_GRID, TUNE_TAUS
from pvcal.data import load_solar
from pvcal.features import is_daylight
from pvcal.metrics import pinball
from pvcal.models import lightgbm_qr
from pvcal.tables import save_table
from pvcal.tasks import make_task, split


def grid_configs() -> list[dict]:
    """Every combination of the grid values, e.g. {"num_leaves": 15, "min_child_samples": 20, ...}."""
    names = list(LGBM_GRID)
    return [dict(zip(names, values)) for values in itertools.product(*LGBM_GRID.values())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data/raw/Solar")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # print tables with dashes on any Windows console
    out = Path(args.out)

    df = load_solar(args.data)
    splits = {tid: split(df, make_task(tid)) for tid in DEV_TASK_IDS}

    rows = []
    for config in grid_configs():
        start = time.perf_counter()
        row = dict(config)
        for tid, (train, test) in splits.items():
            q = lightgbm_qr(train, test, taus=TUNE_TAUS, params=config)
            day = is_daylight(test)
            row[f"pinball_day_task{tid}"] = pinball(test["power"].to_numpy()[day], q[day], TUNE_TAUS)
        row["mean_pinball_day"] = float(np.mean([row[f"pinball_day_task{t}"] for t in DEV_TASK_IDS]))
        rows.append(row)
        print(f"{config}: mean daytime pinball {row['mean_pinball_day']:.5f} "
              f"({time.perf_counter() - start:.0f}s)", flush=True)

    table = pd.DataFrame(rows)
    # Lowest mean loss wins; equal to 5 decimals counts as a tie, which goes to fewer trees.
    ranked = table.assign(_rank=table["mean_pinball_day"].round(5)).sort_values(["_rank", "n_estimators"])
    best = {k: int(ranked.iloc[0][k]) for k in LGBM_GRID}
    table["chosen"] = [all(r[k] == best[k] for k in LGBM_GRID) for _, r in table.iterrows()]

    display = table.copy()
    for col in [c for c in table.columns if c.startswith(("pinball", "mean"))]:
        display[col] = table[col].map(lambda v: f"{v:.5f}")
    display["chosen"] = table["chosen"].map({True: "**yes**", False: ""})
    save_table(out / "tables" / "tuning_lightgbm", table, display,
               note=f"LightGBM tuning on dev tasks {', '.join(map(str, DEV_TASK_IDS))} (April-June 2013): "
                    f"mean daytime pinball loss at levels {', '.join(map(str, TUNE_TAUS))}; "
                    f"learning rate {LEARNING_RATE}.")
    (out / "lightgbm_params.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    print(f"\nChosen and frozen in {out / 'lightgbm_params.json'}: {best}")


if __name__ == "__main__":
    main()
