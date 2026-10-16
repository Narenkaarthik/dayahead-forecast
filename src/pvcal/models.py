"""Forecasters: linear quantile regression and LightGBM quantile regression.

Both follow the project's forecaster contract (see pvcal.baselines):

    forecast(train, test) -> array of shape (len(test), len(taus))

* They train on daylight rows only. At night power is ~0 and the forecast is set to 0
  anyway, so night rows would only teach the model the obvious.
* They fit one separate model per quantile level tau ("one model per level").
* They never read test["power"]. Inputs are the weather forecasts for the hour, calendar
  terms and the zone, all known when the forecast is issued.
* They end with finalize(): sort each row (fixes quantile crossing), clip to [0, 1], and
  set forecast-night rows to 0.

Glossary:
  quantile regression (QR): a regression that predicts a chosen quantile of the outcome
    instead of its mean, by minimising the pinball loss for that level tau.
  design matrix: the table of model inputs, one row per hour and one column per input.
  one-hot encoding: turning a category (hour 3, zone 2) into 0/1 columns, one per category.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .baselines import finalize
from .config import CAL_WINDOW_DAYS, LEARNING_RATE, QUANTREG_MAX_ITER, SEED, SPLIT_LEVELS
from .data import NWP_VARS, ZONES
from .features import calendar_features, is_daylight
from .metrics import TAUS
from .tasks import Task, split


# --- Linear quantile regression -----------------------------------------------------------


def linear_design(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Design matrices for linear QR, built from the daylight rows of train and of test.

    Columns, identical and in the same order for both tables:
      const          intercept (a column of ones; statsmodels does not add one itself)
      12 NWP vars    standardised with the *training* rows' mean and standard deviation
      doy_sin/cos    position in the year, so the model can follow the seasons
      hour_HH        one-hot UTC hour; categories come from the training rows and the
                     first is dropped (it becomes the reference level inside const)
      zone_2/zone_3  one-hot zone with fixed categories {1, 2, 3}, first dropped

    Why drop one category: with an intercept, the full set of hour columns always sums
    to 1, the same as const, so the model could not tell them apart (the "dummy
    variable trap"). A test hour never seen in training gets all-zero hour columns,
    which means "treat it like the reference hour".

    The returned frames keep the original row labels of the daylight rows.
    """
    tr = train[is_daylight(train)]
    te = test[is_daylight(test)]

    # Standardise with training statistics only. Using test-month statistics would leak
    # information from later in the month (the leakage check's second test catches it).
    mean = tr[NWP_VARS].mean()
    std = tr[NWP_VARS].std().replace(0.0, 1.0)  # a constant column would divide by zero

    hours = sorted(tr["time"].dt.hour.unique())  # hour categories seen in training

    def build(rows: pd.DataFrame) -> pd.DataFrame:
        nwp = (rows[NWP_VARS] - mean) / std
        cal = calendar_features(rows)[["doy_sin", "doy_cos"]]
        # One-hot: one 0/1 column per category, skipping the first (the reference level).
        # A value that matches no column (an hour unseen in training) gets all zeros.
        hour_of_row = rows["time"].dt.hour.to_numpy()
        hour = pd.DataFrame({f"hour_{h:02d}": (hour_of_row == h).astype(float) for h in hours[1:]},
                            index=rows.index)
        zone_of_row = rows["zone"].to_numpy()
        zone = pd.DataFrame({f"zone_{z}": (zone_of_row == z).astype(float) for z in ZONES[1:]},
                            index=rows.index)
        const = pd.DataFrame({"const": 1.0}, index=rows.index)
        return pd.concat([const, nwp, cal, hour, zone], axis=1)

    return build(tr), build(te)


def linear_qr(train: pd.DataFrame, test: pd.DataFrame, taus=TAUS, log: dict | None = None) -> np.ndarray:
    """Linear quantile regression: one statsmodels QuantReg fit per level in `taus`.

    If `log` is a dict, it receives `nonconverged_fits` (fits that stopped with a
    convergence warning) plus finalize()'s crossing and capping counts.
    """
    # Imported here so the rest of the package works without the optional "models" extra.
    from statsmodels.regression.quantile_regression import QuantReg
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, IterationLimitWarning

    X_train, X_test = linear_design(train, test)
    y_train = train.loc[X_train.index, "power"].to_numpy()  # outcomes of the training rows only
    test_day = is_daylight(test)

    q = np.zeros((len(test), len(taus)))  # night rows stay 0
    model = QuantReg(y_train, X_train.to_numpy())
    for j, tau in enumerate(taus):
        # Record warnings instead of printing them, so non-convergence can be counted.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = model.fit(q=tau, max_iter=QUANTREG_MAX_ITER)
        if log is not None:
            failed = any(issubclass(w.category, (IterationLimitWarning, ConvergenceWarning)) for w in caught)
            log["nonconverged_fits"] = log.get("nonconverged_fits", 0) + int(failed)
        q[test_day, j] = X_test.to_numpy() @ fit.params  # prediction = inputs x coefficients
    return finalize(q, test_day, log)


# --- LightGBM quantile regression ---------------------------------------------------------


def lgbm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Inputs for LightGBM: the 12 NWP variables, UTC hour, doy_sin/cos and zone.

    Trees split on thresholds, so they need no scaling or one-hot encoding. The hour is
    an integer (trees can isolate any hour with two splits), and zone is a pandas
    Categorical with fixed categories, which LightGBM treats as a category, not a number.
    """
    cal = calendar_features(df)
    return pd.DataFrame(
        {
            **{v: df[v] for v in NWP_VARS},
            "hour": df["time"].dt.hour,
            "doy_sin": cal["doy_sin"],
            "doy_cos": cal["doy_cos"],
            "zone": pd.Categorical(df["zone"], categories=list(ZONES)),
        },
        index=df.index,
    )


def lightgbm_qr(train: pd.DataFrame, test: pd.DataFrame, taus=TAUS, params: dict | None = None,
                log: dict | None = None) -> np.ndarray:
    """LightGBM quantile regression: one gradient-boosted tree model per level in `taus`.

    `params` holds the tuned settings (num_leaves, min_child_samples, n_estimators),
    frozen in results/lightgbm_params.json; None uses LightGBM's defaults.
    deterministic=True and force_row_wise=True make repeated runs give identical numbers.
    """
    from lightgbm import LGBMRegressor  # optional "models" extra

    train_day, test_day = is_daylight(train), is_daylight(test)
    X_train = lgbm_features(train[train_day])
    y_train = train.loc[train_day, "power"]
    X_test = lgbm_features(test[test_day])

    q = np.zeros((len(test), len(taus)))
    for j, tau in enumerate(taus):
        model = LGBMRegressor(
            objective="quantile",  # minimise the pinball loss ...
            alpha=tau,  # ... for this level
            learning_rate=LEARNING_RATE,
            deterministic=True,
            force_row_wise=True,
            random_state=SEED,  # LightGBM's `seed`
            verbose=-1,  # silence training messages
            **(params or {}),
        )
        model.fit(X_train, y_train)
        q[test_day, j] = model.predict(X_test)
    return finalize(q, test_day, log)


# --- Track 2: the split model (variant V2) ------------------------------------------------


def split_model(df: pd.DataFrame, task: Task, params: dict | None = None,
                levels=SPLIT_LEVELS, cal_days: int = CAL_WINDOW_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LightGBM trained up to `cal_days` before the test month, for conformal calibration.

    The task's history is split in time order:
      fit rows          everything up to 00:00 UTC, `cal_days` days before the test month
      calibration rows  the `cal_days` days just before the test month (the model never
                        trains on them, so its errors there are honest out-of-sample errors)
    The model predicts the calibration rows and the test month at `levels`; rows are
    sorted, clipped to [0, 1] and zero at night (lightgbm_qr's finalize step).

    Returns (cal, test): frames with zone, time, issue_time, power, tcc, task, daylight and
    one column per level, named like "q0.05". Test-month power is copied along for scoring
    later; it is never used to fit or predict.
    """
    history, test = split(df, task)  # chronological: history ends at the test month's start
    cal_start = task.test_start - pd.Timedelta(days=cal_days)  # also a 00:00 UTC issue boundary
    fit = history[history["time"] <= cal_start]
    cal = history[history["time"] > cal_start]
    both = pd.concat([cal, test], ignore_index=True)
    q = lightgbm_qr(fit, both, taus=levels, params=params)

    keep = ["zone", "time", "issue_time", "power", "tcc"]
    out = both[keep].assign(task=task.task_id, daylight=is_daylight(both),
                            **{f"q{lvl:.2f}": q[:, j] for j, lvl in enumerate(levels)})
    return out.iloc[:len(cal)].reset_index(drop=True), out.iloc[len(cal):].reset_index(drop=True)


# --- Fewer fitted levels -> 99 levels -----------------------------------------------------


def interpolate_quantiles(q_fit, fit_levels, taus=TAUS) -> np.ndarray:
    """Fill in quantile levels by straight-line interpolation between fitted levels.

    q_fit has one column per level in `fit_levels` (increasing). Each row is sorted first,
    so the result never crosses. Levels outside the fitted range take the nearest end value.
    At a fitted level the result equals the fitted value exactly.
    """
    q = np.sort(np.asarray(q_fit, dtype=float), axis=1)
    levels = np.asarray(fit_levels, dtype=float)
    taus = np.asarray(taus, dtype=float)
    # For each target tau, find the fitted levels just below (lo) and at-or-above (hi) it.
    hi = np.clip(np.searchsorted(levels, taus), 1, len(levels) - 1)
    lo = hi - 1
    # Weight on the upper neighbour: 0 at levels[lo], 1 at levels[hi]; clipped outside the range.
    w = np.clip((taus - levels[lo]) / (levels[hi] - levels[lo]), 0.0, 1.0)
    return q[:, lo] * (1.0 - w) + q[:, hi] * w
