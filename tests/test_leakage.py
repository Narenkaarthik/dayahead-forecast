from functools import partial

import numpy as np
import pandas as pd
import pytest

from pvcal.baselines import finalize, persistence_ensemble, yearly_persistence
from pvcal.calibrate import (aci_daily, apply_adjustment, conformity_scores, cqr_quantile, mondrian_quantiles,
                             recency_weights, weighted_cqr_quantile)
from pvcal.features import build_features, is_daylight
from pvcal.leakage import CANARY, assert_no_lookahead
from pvcal.metrics import TAUS
from pvcal.models import lightgbm_cs_qr, lightgbm_qr, linear_qr, split_model
from pvcal.tasks import make_task

FAST_TAUS = [0.1, 0.5, 0.9]


def persistence_baseline(train, test):
    return finalize(persistence_ensemble(train, test), is_daylight(test))


@pytest.mark.parametrize("task_id", [4, 15])
def test_baselines_pass_the_leakage_check(synthetic_df, task_id):
    assert_no_lookahead(persistence_baseline, synthetic_df, make_task(task_id))
    assert_no_lookahead(yearly_persistence, synthetic_df, make_task(task_id))


@pytest.mark.parametrize("task_id", [4, 15])
def test_quantile_models_pass_the_leakage_check(synthetic_df, task_id):
    # Fast settings (3 levels, 20 trees): the check tests the code path, not the accuracy.
    assert_no_lookahead(partial(linear_qr, taus=FAST_TAUS), synthetic_df, make_task(task_id))
    assert_no_lookahead(partial(lightgbm_qr, taus=FAST_TAUS, params={"n_estimators": 20}),
                        synthetic_df, make_task(task_id))
    assert_no_lookahead(partial(lightgbm_cs_qr, taus=FAST_TAUS, params={"n_estimators": 20}),
                        synthetic_df, make_task(task_id))


def test_detector_catches_a_model_that_reads_outcomes(synthetic_df):
    def cheat(train, test):
        return np.repeat(test["power"].to_numpy()[:, None], len(TAUS), axis=1)

    with pytest.raises(AssertionError, match="reads the outcomes"):
        assert_no_lookahead(cheat, synthetic_df, make_task(15))


def test_detector_catches_scaling_with_test_month_statistics(synthetic_df):
    def cheat(train, test):
        x = test["ssrd"].to_numpy()
        z = (x - x.mean()) / x.std()  # uses weather forecasts issued later in the month
        return np.repeat(z[:, None], len(TAUS), axis=1)

    with pytest.raises(AssertionError, match="issued after"):
        assert_no_lookahead(cheat, synthetic_df, make_task(15))


def test_features_never_include_observed_power(synthetic_df):
    assert "power" not in build_features(synthetic_df).columns


# --- Track 2: the calibration variants ------------------------------------------------------


def test_aci_intervals_up_to_day_d_ignore_outcomes_from_day_d_on():
    # ACI may use only outcomes from forecast days that have finished. Corrupt every
    # outcome from day d onward: intervals for days <= d must not move (day d's interval is
    # set before day d happens). Later days must move, or the test would prove nothing.
    rng = np.random.default_rng(0)
    n_days, per_day, d = 30, 12, 10
    day = np.repeat(np.arange(n_days), per_day)
    y = rng.uniform(0, 1, day.size)
    lo, hi = np.full_like(y, 0.3), np.full_like(y, 0.7)
    scores = {0: rng.normal(0, 0.1, 500)}
    L, U, _, _ = aci_daily(day, np.zeros_like(day), y, lo, hi, scores, alpha=0.1, gamma=0.05)
    y_bad = np.where(day >= d, CANARY, y)
    L2, U2, _, _ = aci_daily(day, np.zeros_like(day), y_bad, lo, hi, scores, alpha=0.1, gamma=0.05)
    upto = day <= d
    np.testing.assert_array_equal(L[upto], L2[upto])
    np.testing.assert_array_equal(U[upto], U2[upto])
    assert not np.array_equal(U[~upto], U2[~upto])


def test_split_model_and_cqr_variants_ignore_test_month_outcomes(synthetic_df):
    # V2 (split model) predictions and the V3, V5 and V6 intervals must not change when every
    # test-month outcome is replaced by an impossible value.
    task = make_task(4)
    in_test = (synthetic_df["time"] > task.test_start) & (synthetic_df["time"] <= task.test_end)
    corrupted = synthetic_df.assign(power=synthetic_df["power"].where(~in_test, CANARY))

    def v2_to_v6_intervals(df):
        cal, test = split_model(df, task, params={"n_estimators": 20})
        c = cal[cal["daylight"]]
        s = conformity_scores(c["power"], c["q0.05"], c["q0.95"])
        lo, hi = test["q0.05"], test["q0.95"]
        v3 = apply_adjustment(lo, hi, cqr_quantile(s, alpha=0.1))
        # V5: per-tercile adjustment (synthetic tcc is uniform, so cut at 1/3 and 2/3).
        cuts = [1 / 3, 2 / 3]
        Q_by_group = mondrian_quantiles(s, np.digitize(c["tcc"], cuts), alpha=0.1)
        v5 = apply_adjustment(lo, hi, np.array([Q_by_group[g] for g in np.digitize(test["tcc"], cuts)]))
        # V6: recency-weighted adjustment.
        days_before = (task.test_start - c["issue_time"]) / pd.Timedelta(days=1)
        v6 = apply_adjustment(lo, hi, weighted_cqr_quantile(s, recency_weights(days_before, 15), alpha=0.1))
        return test, (v3, v5, v6)

    test, intervals = v2_to_v6_intervals(synthetic_df)
    test2, intervals2 = v2_to_v6_intervals(corrupted)
    level_cols = [c for c in test.columns if c.startswith("q")]
    np.testing.assert_array_equal(test[level_cols].to_numpy(), test2[level_cols].to_numpy())  # V2
    for (L, U), (L2, U2) in zip(intervals, intervals2):  # V3, V5, V6
        np.testing.assert_array_equal(L, L2)
        np.testing.assert_array_equal(U, U2)
