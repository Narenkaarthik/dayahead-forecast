"""Tests for pvcal.calibrate on synthetic data, where the right answer is known."""

import numpy as np
import pytest

from pvcal.calibrate import (
    aci_daily,
    apply_adjustment,
    conformity_scores,
    cqr_quantile,
    day_bootstrap,
    mondrian_quantiles,
    percentile_ci,
    recency_weights,
    weighted_cqr_quantile,
)


def test_conformity_scores_are_signed_distances_outside_the_interval():
    # Inside with 0.1 room below and 0.3 above -> -0.1; 0.2 below the interval -> 0.2; 0.1 above -> 0.1.
    np.testing.assert_allclose(conformity_scores([0.3, 0.0, 0.7], [0.2, 0.2, 0.2], [0.6, 0.6, 0.6]),
                               [-0.1, 0.2, 0.1])


@pytest.mark.parametrize("alpha, expected", [
    (0.5, 3.0),     # rank ceil(6 * 0.5) = 3 -> third smallest
    (0.2, 5.0),     # rank ceil(6 * 0.8) = ceil(4.8) = 5 -> the largest
    (0.1, np.inf),  # rank ceil(6 * 0.9) = ceil(5.4) = 6 > n = 5 -> full range
    (1.0, 1.0),     # rank 0 -> clamped to 1 -> the smallest score
])
def test_cqr_quantile_hand_computed(alpha, expected):
    assert cqr_quantile([3.0, 1.0, 2.0, 5.0, 4.0], alpha) == expected


def test_cqr_quantile_is_not_fooled_by_floating_point():
    # (19 + 1) * (1 - 0.05) is 19 in exact arithmetic; the rank must be 19, not 20 (= inf).
    assert cqr_quantile(np.arange(1.0, 20.0), 0.05) == 19.0


def test_apply_adjustment_widens_narrows_clips_and_collapses():
    lo, hi = np.array([0.30, 0.05, 0.40]), np.array([0.50, 0.98, 0.42])
    L, U = apply_adjustment(lo, hi, 0.1)  # widen by 0.1 on each side, then clip to [0, 1]
    np.testing.assert_allclose(L, [0.20, 0.0, 0.30])
    np.testing.assert_allclose(U, [0.60, 1.0, 0.52])
    log = {}
    L, U = apply_adjustment(lo, hi, -0.05, log=log)  # narrow; the third interval is only 0.02 wide
    np.testing.assert_allclose(L, [0.35, 0.10, 0.41])
    np.testing.assert_allclose(U, [0.45, 0.93, 0.41])  # collapsed to its midpoint
    assert log == {"collapsed_rows": 1}
    L, U = apply_adjustment(lo, hi, np.inf)
    assert (L == 0).all() and (U == 1).all()


def test_cqr_reaches_nominal_coverage_on_exchangeable_data():
    # Outcomes are uniform on [0, 1]; the base interval [0.3, 0.7] covers only 40%.
    # CQR's guarantee: average coverage is between 1 - alpha and 1 - alpha + 1 / (n + 1).
    rng = np.random.default_rng(0)
    alpha, n_cal, n_test, n_splits = 0.1, 200, 1000, 300
    coverage = []
    for _ in range(n_splits):
        y = rng.uniform(0, 1, n_cal + n_test)
        lo, hi = np.full_like(y, 0.3), np.full_like(y, 0.7)
        Q = cqr_quantile(conformity_scores(y[:n_cal], lo[:n_cal], hi[:n_cal]), alpha)
        L, U = apply_adjustment(lo[n_cal:], hi[n_cal:], Q)
        coverage.append(np.mean((y[n_cal:] >= L) & (y[n_cal:] <= U)))
    assert 1 - alpha - 0.005 <= np.mean(coverage) <= 1 - alpha + 1 / (n_cal + 1) + 0.005


def _drifting_days(n_days=600, rows_per_day=12, seed=0):
    """Outcomes around 0.5 whose noise variance doubles halfway through the sequence."""
    rng = np.random.default_rng(seed)
    sd = np.where(np.arange(n_days) < n_days // 2, 0.08, 0.08 * np.sqrt(2))  # variance x2
    y = np.clip(0.5 + rng.normal(size=(n_days, rows_per_day)) * sd[:, None], 0, 1).ravel()
    day = np.repeat(np.arange(n_days), rows_per_day)
    cal = np.clip(0.5 + rng.normal(size=600) * 0.08, 0, 1)  # calibrated on the calm regime
    base = np.full_like(y, 0.5)  # a point forecast as the base interval
    return day, y, base, conformity_scores(cal, np.full_like(cal, 0.5), np.full_like(cal, 0.5))


def test_aci_tracks_a_variance_shift_better_than_static_cqr():
    day, y, base, scores = _drifting_days()
    alpha = 0.1
    second_half = day >= day.max() // 2 + 1
    L, U = apply_adjustment(base, base, cqr_quantile(scores, alpha))  # static CQR
    static = np.mean(((y >= L) & (y <= U))[second_half])
    L, U, path, _ = aci_daily(day, np.zeros_like(day), y, base, base, {0: scores}, alpha, gamma=0.02)
    adaptive = np.mean(((y >= L) & (y <= U))[second_half])
    assert static < 0.9 - 0.03  # the shift really does break static calibration
    assert abs(adaptive - 0.9) < abs(static - 0.9)
    assert len(path) == day.max() + 1  # one alpha per day


def test_aci_starts_at_nominal_alpha_and_logs_edge_cases():
    day, y, base, scores = _drifting_days(n_days=40)
    _, _, path, counts = aci_daily(day, np.zeros_like(day), y, base, base, {0: scores}, 0.1, gamma=5.0)
    assert path["alpha_d"].iloc[0] == 0.1
    # A huge step size throws alpha outside (0, 1), and every such day is counted.
    outside = (path["alpha_d"] <= 0).sum() + (path["alpha_d"] >= 1).sum()
    assert outside > 0
    assert counts["alpha_nonpositive_days"] + counts["alpha_at_least_one_days"] == outside


def test_aci_rejects_unsorted_days():
    with pytest.raises(ValueError):
        aci_daily([1, 0], [0, 0], [0.5, 0.5], [0.4, 0.4], [0.6, 0.6], {0: [0.0]}, 0.1, 0.01)


def test_day_bootstrap_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    covered = rng.binomial(12, 0.85, size=200).astype(float)  # covered hours per day
    hours = np.full(200, 12.0)
    reps = day_bootstrap(covered, hours, n_boot=500, seed=0)
    np.testing.assert_array_equal(reps, day_bootstrap(covered, hours, n_boot=500, seed=0))
    lo, hi = percentile_ci(reps)
    assert lo < covered.sum() / hours.sum() < hi


@pytest.mark.parametrize("n", [1, 5, 19, 200])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.5])
def test_weighted_cqr_with_equal_weights_is_ordinary_cqr(n, alpha):
    scores = np.random.default_rng(n).normal(size=n)
    assert weighted_cqr_quantile(scores, np.ones(n), alpha) == cqr_quantile(scores, alpha)


def test_weighted_cqr_hand_computed():
    # Scores 1, 2, 3 with weights 1, 1, 0.25; the test point adds weight 1 at +infinity.
    # Total 3.25; cumulative shares: 1 -> 1/3.25 = 0.308, 2 -> 2/3.25 = 0.615, 3 -> 2.25/3.25 = 0.692.
    scores, weights = [1.0, 2.0, 3.0], [1.0, 1.0, 0.25]
    assert weighted_cqr_quantile(scores, weights, alpha=0.5) == 2.0   # needs 0.5: reached at score 2
    assert weighted_cqr_quantile(scores, weights, alpha=0.35) == 3.0  # needs 0.65: reached at score 3
    assert weighted_cqr_quantile(scores, weights, alpha=0.2) == np.inf  # needs 0.8: only the test point's mass


def test_recency_weights_halve_every_half_life():
    np.testing.assert_allclose(recency_weights([0, 15, 30], half_life=15), [1.0, 0.5, 0.25])


def test_mondrian_quantiles_are_cqr_within_each_group():
    scores = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    groups = np.array(["clear"] * 3 + ["cloudy"] * 3)
    Q = mondrian_quantiles(scores, groups, alpha=0.5)
    assert Q == {"clear": cqr_quantile([1, 2, 3], 0.5), "cloudy": cqr_quantile([10, 20, 30], 0.5)}
    assert Q["clear"] == 2.0 and Q["cloudy"] == 20.0


def test_apply_adjustment_accepts_one_adjustment_per_row():
    L, U = apply_adjustment([0.3, 0.3, 0.3], [0.5, 0.5, 0.5], np.array([0.1, -0.05, np.inf]))
    np.testing.assert_allclose(L, [0.2, 0.35, 0.0])
    np.testing.assert_allclose(U, [0.6, 0.45, 1.0])
