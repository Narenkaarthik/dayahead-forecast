import numpy as np
import pytest

from pvcal.metrics import (
    TAUS,
    interval,
    mean_width,
    picp,
    pinball,
    pinball_by_level,
    reliability,
    summarize,
    winkler,
)


def test_taus_are_the_99_competition_levels():
    assert len(TAUS) == 99 and TAUS[0] == 0.01 and TAUS[-1] == 0.99


def test_pinball_is_asymmetric_by_level():
    # Forecasting 1 when the outcome is 2: an under-forecast costs tau per unit.
    np.testing.assert_allclose(pinball_by_level([2.0], [[1.0, 1.0]], taus=[0.25, 0.75]), [0.25, 0.75])
    # Forecasting 1 when the outcome is 0: an over-forecast costs (1 - tau) per unit.
    np.testing.assert_allclose(pinball_by_level([0.0], [[1.0, 1.0]], taus=[0.25, 0.75]), [0.75, 0.25])


def test_perfect_forecast_scores_zero():
    y = np.array([0.2, 0.7])
    assert pinball(y, np.repeat(y[:, None], 99, axis=1)) == 0.0


def test_point_forecast_scores_half_its_mae():
    # Why the official benchmark (one value copied into all 99 quantiles) scores MAE / 2.
    rng = np.random.default_rng(0)
    y, c = rng.uniform(0, 1, 500), rng.uniform(0, 1, 500)
    assert pinball(y, np.repeat(c[:, None], 99, axis=1)) == pytest.approx(0.5 * np.mean(np.abs(y - c)))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        pinball([1.0, 2.0], np.zeros((2, 98)))


def test_interval_picks_the_right_levels():
    q = np.tile(TAUS, (4, 1))
    lo, hi = interval(q, TAUS, 0.9)
    assert np.allclose(lo, 0.05) and np.allclose(hi, 0.95)
    lo, hi = interval(q, TAUS, 0.8)
    assert np.allclose(lo, 0.10) and np.allclose(hi, 0.90)
    with pytest.raises(ValueError):
        interval(q, TAUS, 0.85)  # needs levels 0.075 and 0.925


def test_picp_width_and_winkler_hand_computed():
    y = np.array([2.0, 5.0, 0.5])
    lo, hi = np.ones(3), np.full(3, 3.0)
    assert picp(y, lo, hi) == pytest.approx(1 / 3)
    assert mean_width(lo, hi) == pytest.approx(2.0)
    # widths are 2; miss penalties are (2 / 0.1) * [0, 2, 0.5] = [0, 40, 10]
    assert winkler(y, lo, hi, alpha=0.1) == pytest.approx((2 + 42 + 12) / 3)


def test_reliability_is_the_diagonal_for_calibrated_forecasts():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 1, 20_000)
    q = np.tile(TAUS, (y.size, 1))  # the true quantiles of U(0, 1)
    np.testing.assert_allclose(reliability(y, q), TAUS, atol=0.015)


def test_night_rows_do_not_count_toward_coverage():
    y = np.array([0.0, 0.0, 0.5])
    q = np.zeros((3, 99))
    q[2] = 0.4  # the only daylight row misses
    scores = summarize(y, q, daylight=[False, False, True])
    assert scores["picp90_day"] == 0.0  # the two trivially covered night rows are excluded


def test_pinball_hand_computed_examples():
    # tau = 0.9, forecast 0.8. Outcome 1.0 is 0.2 above: cost 0.9 * 0.2 = 0.18.
    assert pinball([1.0], [[0.8]], taus=[0.9]) == pytest.approx(0.18)
    # Outcome 0.5 is 0.3 below: cost (1 - 0.9) * 0.3 = 0.03.
    assert pinball([0.5], [[0.8]], taus=[0.9]) == pytest.approx(0.03)


@pytest.mark.parametrize("y, expected", [(0.4, 0.4), (0.1, 2.4), (0.7, 2.4)])
def test_winkler_hand_computed_examples(y, expected):
    # Interval [0.2, 0.6] at alpha = 0.1: width 0.4, plus (2 / 0.1) x distance outside.
    # y = 0.1 is 0.1 below: 0.4 + 20 * 0.1 = 2.4. y = 0.7 is 0.1 above: also 2.4.
    assert winkler([y], [0.2], [0.6], alpha=0.1) == pytest.approx(expected)
