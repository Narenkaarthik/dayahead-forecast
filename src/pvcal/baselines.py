"""Reference forecasts, plus the clean-up step every model's output goes through.

Every forecaster in this project has the same signature:

    forecast(train, test) -> array of shape (len(test), len(TAUS))

`train` holds history up to the issue time of the first test day; `test` holds the
rows to forecast (its power column must never be read). That shared signature is what
lets pvcal.leakage.assert_no_lookahead check any model the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import POWER_CAP
from .metrics import TAUS


def yearly_persistence(train: pd.DataFrame, test: pd.DataFrame, taus=TAUS) -> np.ndarray:
    """The official GEFCom2014 benchmark: last year's power at the same hour, in every quantile."""
    history = train.set_index(["zone", "time"])["power"]
    keys = pd.MultiIndex.from_arrays([test["zone"], test["time"] - pd.DateOffset(years=1)])
    point = history.reindex(keys).to_numpy(dtype=float)
    if np.isnan(point).any():
        raise ValueError("training data does not reach back one year before the test month")
    return np.repeat(point[:, None], len(taus), axis=1)


def persistence_ensemble(
    train: pd.DataFrame, test: pd.DataFrame, taus=TAUS, window_days: int = 30
) -> np.ndarray:
    """Empirical quantiles of the last `window_days` of observed power at the same UTC hour, per zone."""
    cutoff = train["time"].max()
    recent = train[train["time"] > cutoff - pd.Timedelta(days=window_days)]
    table = (
        recent.assign(hour=recent["time"].dt.hour)
        .groupby(["zone", "hour"])["power"]
        .quantile(list(taus))
        .unstack()
    )
    keys = pd.MultiIndex.from_arrays([test["zone"], test["time"].dt.hour])
    return table.reindex(keys).to_numpy(dtype=float)


def finalize(q, daylight=None, log: dict | None = None) -> np.ndarray:
    """Make quantile forecasts valid: sorted (no crossing), within [0, 1], and zero at forecast night.

    If `log` is a dict, these totals are added to it (taken before the clean-up, on
    daylight rows when a daylight mask is given):
      crossing_rows:     rows where some quantile was below the one for a lower level;
      crossing_size_sum: summed size of each crossing row's largest drop between neighbouring levels;
      crossing_size_max: the largest such drop seen;
      sort_shift_sum:    summed |sorted - unsorted| over all values (how far sorting moved them);
      values:            number of values checked;
      capped_values:     values above POWER_CAP (1.0) that were lowered to it.
    """
    q = np.asarray(q, dtype=float)
    if log is not None:
        rows = q if daylight is None else q[np.asarray(daylight, dtype=bool)]
        drop = np.maximum(rows[:, :-1] - rows[:, 1:], 0.0).max(axis=1) if rows.size else np.zeros(0)
        log["crossing_rows"] = log.get("crossing_rows", 0) + int((drop > 0).sum())
        log["crossing_size_sum"] = log.get("crossing_size_sum", 0.0) + float(drop.sum())
        log["crossing_size_max"] = max(log.get("crossing_size_max", 0.0), float(drop.max(initial=0.0)))
        log["sort_shift_sum"] = log.get("sort_shift_sum", 0.0) + float(np.abs(np.sort(rows, axis=1) - rows).sum())
        log["values"] = log.get("values", 0) + int(rows.size)
        log["capped_values"] = log.get("capped_values", 0) + int((rows > POWER_CAP).sum())
    q = np.sort(q, axis=1)  # sorting each row removes quantile crossing
    q = np.clip(q, 0.0, POWER_CAP)  # power is a fraction of capacity: between 0 and 1
    if daylight is not None:
        q[~np.asarray(daylight, dtype=bool)] = 0.0
    return q
