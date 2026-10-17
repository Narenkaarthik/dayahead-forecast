"""Conformal recalibration of prediction intervals.

Terms used below:
  alpha              the promised miss rate: 0.10 for a 90% interval, 0.20 for an 80% one.
  conformity score   E = max(lo - y, y - hi): how far outcome y fell outside [lo, hi]
                     (negative when it fell inside).
  CQR                conformalized quantile regression (Romano, Patterson & Candes, 2019):
                     widen every interval by one constant Q, taken from the scores on a
                     calibration window held out from training.
  Mondrian CQR       CQR with a separate Q for each group of hours (here, forecast cloud
                     terciles), so the adjustment can differ between clear and cloudy hours.
  weighted CQR       CQR where recent calibration days count more (Barber et al., 2023).
  ACI                adaptive conformal inference (Gibbs & Candes, 2021): the level used to
                     pick Q is nudged after every forecast day.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import BOOTSTRAP_B, POWER_CAP, SEED


def conformity_scores(y, lo, hi) -> np.ndarray:
    """E = max(lo - y, y - hi) for each row: the signed distance outside the interval."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    return np.maximum(lo - y, y - hi)


def cqr_quantile(scores, alpha: float) -> float:
    """The CQR adjustment Q: the ceil((n + 1)(1 - alpha))-th smallest calibration score.

    The test point is treated as an (n + 1)-th score; under exchangeability its rank among
    all n + 1 is uniform, so this rank gives coverage of at least 1 - alpha exactly.
    Edge cases: rank > n returns inf (the caller uses the full range [0, 1]); rank < 1
    (alpha >= 1, which ACI can reach) returns the smallest score.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    # The 1e-9 stops floating-point noise (e.g. 90.00000000000001) from bumping the rank up by one.
    rank = math.ceil((s.size + 1) * (1.0 - alpha) - 1e-9)
    if rank > s.size:
        return np.inf
    return float(s[max(rank, 1) - 1])


def weighted_cqr_quantile(scores, weights, alpha: float) -> float:
    """Weighted conformal quantile (Barber, Candes, Ramdas & Tibshirani, 2023).

    Calibration score i gets weight w_i in [0, 1]; the test point gets weight 1 and sits at
    +infinity. Q is the smallest score at which the normalised cumulative weight reaches
    1 - alpha, or inf if only the test point's mass gets there. With all w_i = 1 this is
    exactly cqr_quantile.
    """
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(s, kind="stable")
    cumulative = np.cumsum(w[order]) / (w.sum() + 1.0)  # the "+ 1" is the test point's weight
    reached = np.flatnonzero(cumulative >= 1.0 - alpha - 1e-12)
    return float(s[order][reached[0]]) if reached.size else np.inf


def recency_weights(days_before: np.ndarray, half_life: float) -> np.ndarray:
    """Weight 0.5 ** (days_before / half_life): a day one half-life older counts half as much."""
    return 0.5 ** (np.asarray(days_before, dtype=float) / half_life)


def apply_adjustment(lo, hi, Q, log: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The recalibrated interval [lo - Q, hi + Q], clipped to [0, 1].

    Q may be a number or one value per row (Mondrian CQR). Q > 0 widens, Q < 0 narrows;
    Q = inf gives the full range [0, 1]. If a negative Q would put the lower end above the
    upper end, the interval collapses to its midpoint (counted in log["collapsed_rows"]).
    """
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    Q = np.broadcast_to(np.asarray(Q, dtype=float), lo.shape)
    full = np.isinf(Q)
    L, U = lo - np.where(full, 0.0, Q), hi + np.where(full, 0.0, Q)
    collapsed = L > U
    if log is not None:
        log["collapsed_rows"] = log.get("collapsed_rows", 0) + int(collapsed.sum())
    mid = (L + U) / 2
    L, U = np.where(collapsed, mid, L), np.where(collapsed, mid, U)
    L, U = np.where(full, 0.0, L), np.where(full, POWER_CAP, U)
    return np.clip(L, 0.0, POWER_CAP), np.clip(U, 0.0, POWER_CAP)


def mondrian_quantiles(scores, groups, alpha: float) -> dict:
    """Mondrian CQR: a separate CQR adjustment for each group of calibration rows."""
    scores, groups = np.asarray(scores, dtype=float), np.asarray(groups)
    return {g: cqr_quantile(scores[groups == g], alpha) for g in np.unique(groups)}


def aci_daily(day, task, y, lo, hi, scores_by_task: dict, alpha: float, gamma: float):
    """Adaptive conformal inference, updated once per forecast day.

    Rows are daylight hours sorted by forecast day; `task` selects which calibration scores
    (`scores_by_task[task]`) a row uses. For each day d in order: pick Q at level
    1 - alpha_d (alpha_d <= 0 gives [0, 1]; alpha_d >= 1 gives the smallest score), form
    the day's intervals, and only after the day is over update
        alpha_{d+1} = alpha_d + gamma * (alpha - err_d),
    where err_d is the day's miss rate. Day d's interval therefore uses outcomes from
    earlier days only.

    Returns (L, U, path, counts): the intervals, a table of alpha_d and err_d per day, and
    how often each edge case happened.
    """
    day, task = np.asarray(day), np.asarray(task)
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    if not pd.Index(day).is_monotonic_increasing:
        raise ValueError("rows must be sorted by forecast day")
    L, U = np.empty_like(lo), np.empty_like(hi)
    counts = {"alpha_nonpositive_days": 0, "rank_above_n_days": 0, "alpha_at_least_one_days": 0}
    days, starts = np.unique(day, return_index=True)
    ends = np.append(starts[1:], len(day))
    a, path = alpha, []
    for d, start, end in zip(days, starts, ends):
        rows = slice(start, end)
        scores = scores_by_task[task[start]]
        if a <= 0:
            Q = np.inf
            counts["alpha_nonpositive_days"] += 1
        elif a >= 1:
            Q = float(np.min(scores))
            counts["alpha_at_least_one_days"] += 1
        else:
            Q = cqr_quantile(scores, a)
            counts["rank_above_n_days"] += int(np.isinf(Q))
        L[rows], U[rows] = apply_adjustment(lo[rows], hi[rows], Q)
        err = float(((y[rows] < L[rows]) | (y[rows] > U[rows])).mean())
        path.append((d, task[start], a, err))
        a = a + gamma * (alpha - err)
    return L, U, pd.DataFrame(path, columns=["day", "task", "alpha_d", "err_d"]), counts


def day_bootstrap(num, den, n_boot: int = BOOTSTRAP_B, seed: int = SEED) -> np.ndarray:
    """Bootstrap a ratio sum(num) / sum(den) by resampling whole forecast days.

    num and den hold one total per day (e.g. covered hours and daylight hours, so the ratio
    is coverage). Hours within a day share one weather forecast and are correlated, so days,
    not hours, are resampled. The same seed and number of days always draw the same days,
    which makes comparisons between variants paired.
    """
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    picks = np.random.default_rng(seed).integers(0, num.size, size=(n_boot, num.size))
    with np.errstate(invalid="ignore", divide="ignore"):  # a replicate with no rows gives NaN
        return num[picks].sum(axis=1) / den[picks].sum(axis=1)


def percentile_ci(replicates, level: float = 0.95) -> tuple[float, float]:
    """The central `level` percentile interval of bootstrap replicates (2.5% and 97.5% for 95%)."""
    tail = (1.0 - level) / 2 * 100
    lo, hi = np.nanpercentile(replicates, [tail, 100 - tail])
    return float(lo), float(hi)
