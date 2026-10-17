"""Unit tests for the quantile regression models, on synthetic data (fast settings: 3 levels, 20 trees)."""

import numpy as np
import pandas as pd
import pytest

from pvcal.baselines import finalize
from pvcal.features import is_daylight
from pvcal.models import (clear_sky_envelope, interpolate_quantiles, lightgbm_cs_qr, lightgbm_qr,
                          linear_design, linear_qr)
from pvcal.tasks import make_task, split

FAST_TAUS = [0.1, 0.5, 0.9]
FAST_LGBM = {"n_estimators": 20}


@pytest.fixture(scope="module")
def train_test(synthetic_df):
    return split(synthetic_df, make_task(4))


def test_linear_design_has_identical_columns_for_train_and_test(train_test):
    X_train, X_test = linear_design(*train_test)
    assert list(X_train.columns) == list(X_test.columns)
    assert X_train.columns[0] == "const" and {"zone_2", "zone_3"} <= set(X_train.columns)
    assert "zone_1" not in X_train.columns  # the first category is dropped


def test_linear_design_uses_daylight_rows_only(train_test):
    train, test = train_test
    X_train, X_test = linear_design(train, test)
    assert len(X_train) == is_daylight(train).sum() and len(X_test) == is_daylight(test).sum()


def test_changing_test_inputs_leaves_the_training_matrix_unchanged(train_test):
    train, test = train_test
    X_before, _ = linear_design(train, test)
    altered = test.assign(t2m=test["t2m"] + 50.0, rh=-test["rh"])
    X_after, _ = linear_design(train, altered)
    pd.testing.assert_frame_equal(X_before, X_after)  # scaling uses training statistics only


def test_an_unseen_test_hour_gets_all_zero_hour_columns(train_test):
    train, test = train_test
    seen = set(train.loc[is_daylight(train), "time"].dt.hour)
    unseen = next(h for h in range(24) if h not in seen)  # a night hour in the synthetic data
    odd = test.iloc[[0]].copy()
    odd["time"] = odd["time"].dt.floor("D") + pd.Timedelta(hours=unseen)
    odd["ssrd"] = 500.0  # pretend the forecast says it is light, so the row is kept
    _, X_test = linear_design(train, odd)
    hour_cols = [c for c in X_test.columns if c.startswith("hour_")]
    assert (X_test[hour_cols].to_numpy() == 0).all()


@pytest.mark.parametrize("model", [
    lambda tr, te: linear_qr(tr, te, taus=FAST_TAUS),
    lambda tr, te: lightgbm_qr(tr, te, taus=FAST_TAUS, params=FAST_LGBM),
    lambda tr, te: lightgbm_cs_qr(tr, te, taus=FAST_TAUS, params=FAST_LGBM),
], ids=["linear_qr", "lightgbm_qr", "lightgbm_cs_qr"])
def test_quantile_models_return_valid_forecasts(train_test, model):
    train, test = train_test
    q = model(train, test)
    assert q.shape == (len(test), len(FAST_TAUS))
    assert (np.diff(q, axis=1) >= 0).all()  # no crossing
    assert q.min() >= 0.0 and q.max() <= 1.0
    assert (q[~is_daylight(test)] == 0.0).all()  # zero at forecast night


def test_finalize_caps_sorts_zeroes_night_and_counts():
    q = np.array([[0.2, 0.1, 1.3],  # crossed, one value above 1
                  [0.9, 0.8, 0.7]])  # crossed, but it is night
    log = {}
    out = finalize(q, daylight=[True, False], log=log)
    np.testing.assert_array_equal(out, [[0.1, 0.2, 1.0], [0.0, 0.0, 0.0]])
    # Counted on daylight rows only: one crossing row, whose largest drop is 0.2 - 0.1 = 0.1;
    # sorting swaps 0.2 and 0.1, moving two values by 0.1 each; 1.3 is capped.
    assert log["crossing_rows"] == 1 and log["capped_values"] == 1 and log["values"] == 3
    assert log["crossing_size_sum"] == pytest.approx(0.1)
    assert log["crossing_size_max"] == pytest.approx(0.1)
    assert log["sort_shift_sum"] == pytest.approx(0.2)


def test_interpolate_quantiles_reproduces_the_fitted_levels_exactly():
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    q_fit = np.array([[0.1, 0.2, 0.4, 0.5, 0.9],
                      [0.0, 0.0, 0.1, 0.3, 0.3]])
    out = interpolate_quantiles(q_fit, levels, taus=levels)
    np.testing.assert_array_equal(out, q_fit)


def test_interpolate_quantiles_is_linear_between_levels_and_flat_outside():
    out = interpolate_quantiles([[0.2, 0.6]], [0.2, 0.6], taus=[0.1, 0.3, 0.4, 0.7])
    np.testing.assert_allclose(out, [[0.2, 0.3, 0.4, 0.6]])


def test_clear_sky_envelope_is_a_high_quantile_within_the_day_of_year_window():
    # One zone, one hour, one value per day of 2013: value = day of year / 365.
    times = pd.date_range("2013-01-01 02:00", "2013-12-31 02:00", freq="D", tz="UTC")
    df = pd.DataFrame({"zone": 1, "time": times, "ssrd": 500.0, "power": times.dayofyear / 365})
    env = clear_sky_envelope(df, "power", level=1.0, half_window=10)
    assert env[(1, 2, 100)] == pytest.approx(110 / 365)  # the largest value within days 90-110
    assert env[(1, 2, 1)] == pytest.approx(365 / 365)  # the window wraps: 31 December is a neighbour


def test_clear_sky_envelope_uses_daylight_training_rows_only(train_test):
    train, test = train_test
    env = clear_sky_envelope(train, "power")
    assert set(env.index.get_level_values("hour")) == set(train.loc[is_daylight(train), "time"].dt.hour)


def test_forecast_month_returns_one_row_per_hour_and_one_column_per_level(synthetic_df):
    from pvcal.forecast import forecast_month

    out = forecast_month(synthetic_df, "2013-07", model="lightgbm", params=FAST_LGBM, taus=FAST_TAUS)
    assert list(out.columns) == ["zone", "time", "q0.10", "q0.50", "q0.90"]
    assert len(out) == 3 * 31 * 24  # three zones, every hour of July
    assert out["time"].min() == pd.Timestamp("2013-07-01 01:00", tz="UTC")
    with pytest.raises(ValueError):
        forecast_month(synthetic_df, "2013-07", model="random-forest")
