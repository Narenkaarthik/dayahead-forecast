"""Regenerate every result, table, figure and README section from the raw data, in order.

    python scripts/run_all.py --data data/raw/Solar

Steps: baselines -> LightGBM tuning (dev tasks) -> Track 1 -> leakage check on all test months
-> Track 2 -> README report.
Each step is a separate script, so any one of them can also be rerun on its own. The run
stops at the first step that fails (for example, a leakage check). Expect roughly two hours on
a 16-core PC, most of it fitting 99 quantile levels for three models on 12 months in Track 1.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).parent
RUNTIME = Path("results") / "runtime.json"  # step timings, quoted in the README by report.py
STEPS = [
    ("Baselines", "run_baselines.py", True),
    ("LightGBM tuning on dev tasks 1-3", "tune_lightgbm.py", True),
    ("Track 1: quantile models", "run_track1.py", True),
    ("Leakage check of every quantile model on every test month", "check_leakage.py", True),
    ("Track 2: calibration study", "run_track2.py", True),
    ("README report", "report.py", False),  # reads results/, not the raw data
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data/raw/Solar")
    args = parser.parse_args()

    start = time.perf_counter()
    minutes = {}
    for label, script, needs_data in STEPS:
        if script == "report.py":  # record the timings first, so the report can quote them
            RUNTIME.write_text(json.dumps({"minutes_by_step": minutes, "cpu_count": os.cpu_count()},
                                          indent=2) + "\n", encoding="utf-8")
        step_start = time.perf_counter()
        print(f"\n=== {label} ({script}) ===", flush=True)
        command = [sys.executable, str(SCRIPTS / script)] + (["--data", args.data] if needs_data else [])
        subprocess.run(command, check=True)  # check=True: stop here if the step fails
        minutes[script] = round((time.perf_counter() - step_start) / 60, 1)
        print(f"--- {label} finished in {minutes[script]} min", flush=True)
    print(f"\nAll steps finished in {(time.perf_counter() - start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
