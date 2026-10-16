"""A reusable look-ahead check. Run it on every model you add (see tests/test_leakage.py).

It covers the batch protocol, where each model forecasts a whole test month at once.
The online calibration method (ACI, in pvcal.calibrate) needs a day-by-day version of
the same idea: corrupt outcomes from day d onward, and the intervals up to day d must
not change (tests/test_leakage.py and scripts/run_track2.py check it).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .data import NWP_VARS
from .tasks import Task, split

Forecaster = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]
CANARY = 99.0  # an impossible value for normalized power and for every NWP field


def assert_no_lookahead(forecast: Forecaster, df: pd.DataFrame, task: Task) -> None:
    """Raise AssertionError if `forecast` uses information unavailable at issue time.

    1. Outcomes: every observed power value in the test month is replaced by a canary.
       No forecast in the month may change.
    2. Future weather forecasts: NWP rows issued after the first test day are replaced by
       canaries. Forecasts for the first test day may not change. This catches, for
       example, a feature scaler fitted on the test month.
    """
    train, test = split(df, task)
    reference = np.asarray(forecast(train, test), dtype=float)

    outcomes_hidden = test.assign(power=CANARY)
    again = np.asarray(forecast(train, outcomes_hidden), dtype=float)
    if not np.allclose(reference, again, equal_nan=True):
        raise AssertionError(
            f"task {task.task_id}: forecasts changed when test-month power was altered, "
            "so the model reads the outcomes it is predicting"
        )

    later_issue = (test["issue_time"] > test["issue_time"].min()).to_numpy()
    future_hidden = test.copy()
    future_hidden.loc[later_issue, NWP_VARS] = CANARY
    again = np.asarray(forecast(train, future_hidden), dtype=float)
    first_day = ~later_issue
    if not np.allclose(reference[first_day], again[first_day], equal_nan=True):
        raise AssertionError(
            f"task {task.task_id}: first-day forecasts changed when later weather forecasts "
            "were altered, so the model uses information issued after its own issue time"
        )
