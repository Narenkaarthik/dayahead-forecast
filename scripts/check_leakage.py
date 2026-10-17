"""Run the leakage check for every quantile model on every test month.

    python scripts/check_leakage.py --data data/raw/Solar

run_track1.py checks the expensive quantile models on tasks 4 and 15 only, with all 99
levels. This script checks them on all 12 test months, with the five tuning levels
(pvcal.config.TUNE_TAUS) to keep it fast: the check tests the code path (does any forecast
change when outcomes or later weather forecasts are altered?), which does not depend on the
number of levels. It stops at the first failure.

Writes results/tables/leakage_checks.csv (one row per model and task).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import pandas as pd

from pvcal.config import TUNE_TAUS
from pvcal.data import load_solar
from pvcal.leakage import assert_no_lookahead
from pvcal.models import lightgbm_cs_qr, lightgbm_qr, linear_qr
from pvcal.tasks import scored_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data/raw/Solar")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)

    params = json.loads((out / "lightgbm_params.json").read_text(encoding="utf-8"))
    taus = list(TUNE_TAUS)
    models = {
        "Linear QR": partial(linear_qr, taus=taus),
        "LightGBM QR": partial(lightgbm_qr, taus=taus, params=params),
        "LightGBM QR, clear-sky normalised": partial(lightgbm_cs_qr, taus=taus, params=params),
    }
    df = load_solar(args.data)
    rows = []
    for name, model in models.items():
        for task in scored_tasks():
            start = time.perf_counter()
            assert_no_lookahead(model, df, task)  # raises AssertionError on any look-ahead
            rows.append({"model": name, "task": task.task_id, "month": task.label, "levels": len(taus),
                         "passed": True})
            print(f"{name}: task {task.task_id} ({task.label}) passed [{time.perf_counter() - start:.0f}s]",
                  flush=True)
    pd.DataFrame(rows).to_csv(out / "tables" / "leakage_checks.csv", index=False, encoding="utf-8")
    print(f"All {len(rows)} checks passed.")


if __name__ == "__main__":
    main()
