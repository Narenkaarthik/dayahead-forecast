"""Forecast one month with a trained model: a Python function and a command-line tool.

    pvcal-forecast --data data/raw/Solar --month 2014-06 --model lightgbm --out june.csv

The model is trained on every hour before 00:00 UTC on day 1 of the month (exactly as in the
evaluation) and forecasts the 99 quantiles of every hour of that month for all three zones,
from the day-ahead weather forecasts in the data. The month must lie inside the data
(May 2012 to June 2014), because the weather forecasts for it are needed.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_solar
from .metrics import TAUS
from .models import lightgbm_cs_qr, lightgbm_qr, linear_qr
from .tasks import Task, split

MODELS = {"linear": linear_qr, "lightgbm": lightgbm_qr, "lightgbm-cs": lightgbm_cs_qr}


def forecast_month(df: pd.DataFrame, month: str, model: str = "lightgbm", params: dict | None = None,
                   taus=TAUS) -> pd.DataFrame:
    """Train on everything before `month` ("YYYY-MM") and forecast its quantiles.

    Returns one row per zone and hour: zone, time (UTC, hour-ending) and one column per
    level, named like "q0.05".
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}")
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    task = Task(0, start, start + pd.DateOffset(months=1))
    train, test = split(df, task)
    forecaster = MODELS[model] if model == "linear" else partial(MODELS[model], params=params)
    q = np.asarray(forecaster(train, test, taus=taus))
    columns = {f"q{tau:.2f}": q[:, j] for j, tau in enumerate(taus)}
    return pd.concat([test[["zone", "time"]].reset_index(drop=True), pd.DataFrame(columns)], axis=1)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Forecast the 99 quantiles of one month of solar power.")
    parser.add_argument("--data", default="data/raw/Solar", help="the extracted GEFCom2014 'Solar' folder")
    parser.add_argument("--month", required=True, help="month to forecast, YYYY-MM (2012-05 to 2014-06)")
    parser.add_argument("--model", default="lightgbm", choices=sorted(MODELS))
    parser.add_argument("--params", default="results/lightgbm_params.json",
                        help="LightGBM settings (JSON); ignored by the linear model")
    parser.add_argument("--out", required=True, help="CSV file to write")
    args = parser.parse_args(argv)

    params = None
    if args.model != "linear" and Path(args.params).exists():
        params = json.loads(Path(args.params).read_text(encoding="utf-8"))
    result = forecast_month(load_solar(args.data), args.month, args.model, params)
    result.to_csv(args.out, index=False, float_format="%.6g")
    print(f"Wrote {len(result):,} rows x {result.shape[1] - 2} quantile levels to {args.out}")


if __name__ == "__main__":
    main()
