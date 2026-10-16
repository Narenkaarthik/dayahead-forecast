"""Scores for quantile and interval forecasts.

Quantile forecasts are arrays of shape (n_rows, n_levels), one column per level in `taus`.
Interval scores are computed on daylight rows only: at night both the forecast and the
outcome are zero, so every interval "covers" and would inflate coverage for free.
"""

from __future__ import annotations

import numpy as np

TAUS = np.round(np.arange(1, 100) / 100, 2)  # the 99 levels scored in GEFCom2014


def _as_2d(y, q, taus):
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    taus = np.asarray(taus, dtype=float)
    if q.shape != (y.shape[0], taus.shape[0]):
        raise ValueError(f"q has shape {q.shape}, expected {(y.shape[0], taus.shape[0])}")
    return y[:, None], q, taus[None, :]


def pinball_by_level(y, q, taus=TAUS) -> np.ndarray:
    """Mean pinball loss for each quantile level."""
    y, q, taus = _as_2d(y, q, taus)
    diff = y - q
    return np.mean(np.maximum(taus * diff, (taus - 1.0) * diff), axis=0)


def pinball(y, q, taus=TAUS) -> float:
    """Mean pinball loss over all rows and levels: the GEFCom2014 score for one task."""
    return float(np.mean(pinball_by_level(y, q, taus)))


def interval(q, taus, coverage: float) -> tuple[np.ndarray, np.ndarray]:
    """Central interval with the given nominal coverage, e.g. 0.9 -> levels 0.05 and 0.95."""
    taus = np.asarray(taus, dtype=float)
    lo_level, hi_level = (1 - coverage) / 2, (1 + coverage) / 2
    lo = np.flatnonzero(np.isclose(taus, lo_level))
    hi = np.flatnonzero(np.isclose(taus, hi_level))
    if lo.size != 1 or hi.size != 1:
        raise ValueError(f"levels {lo_level:.3f} and {hi_level:.3f} are not both in taus")
    q = np.asarray(q, dtype=float)
    return q[:, lo[0]], q[:, hi[0]]


def picp(y, lo, hi) -> float:
    """Prediction interval coverage probability: share of outcomes inside [lo, hi]."""
    y = np.asarray(y, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo, hi) -> float:
    """Average interval width, in the same units as y (here: fraction of capacity)."""
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def winkler(y, lo, hi, alpha: float) -> float:
    """Winkler (interval) score for a (1 - alpha) interval: width plus a miss penalty. Lower is better."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    penalty = (2.0 / alpha) * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))
    return float(np.mean((hi - lo) + penalty))


def reliability(y, q) -> np.ndarray:
    """Observed frequency of y <= q for each level. Perfect calibration gives back the levels."""
    y = np.asarray(y, dtype=float)[:, None]
    return np.mean(y <= np.asarray(q, dtype=float), axis=0)


def summarize(y, q, daylight, taus=TAUS) -> dict[str, float]:
    """Headline scores for one set of forecasts."""
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    day = np.asarray(daylight, dtype=bool)
    scores = {
        "pinball_all": pinball(y, q, taus),
        "pinball_day": pinball(y[day], q[day], taus),
    }
    for cov in (0.8, 0.9):
        lo, hi = interval(q[day], taus, cov)
        tag = f"{round(cov * 100)}"
        scores[f"picp{tag}_day"] = picp(y[day], lo, hi)
        scores[f"width{tag}_day"] = mean_width(lo, hi)
        scores[f"winkler{tag}_day"] = winkler(y[day], lo, hi, alpha=1 - cov)
    return scores
