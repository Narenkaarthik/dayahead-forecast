"""Conformal recalibration of prediction intervals: CQR (static) and ACI (adaptive).

Glossary (the terms used in this file):
  interval [lo, hi]   a prediction interval from two quantiles, e.g. [q0.05, q0.95] for 90%.
  alpha               the promised miss rate: 0.10 for a 90% interval, 0.20 for an 80% one.
  conformity score    E = max(lo - y, y - hi): how far outcome y fell outside its interval.
                      Positive = a miss by that much; negative = inside, with that much room.
  calibration window  recent data, before the test month, used to measure the scores.
  CQR                 conformalized quantile regression (Romano, Patterson & Candes, 2019):
                      widen (or narrow) every interval by one constant Q, chosen so that on
                      the calibration window a share 1 - alpha of the scores is <= Q.
  ACI                 adaptive conformal inference (Gibbs & Candes, 2021): like CQR, but
                      the level used to pick Q is nudged after every day, up after a day
                      with too many misses and down after a day with too few.
  bootstrap           resampling the data with replacement many times to see how much a
                      statistic would vary; here whole forecast days are resampled, because
                      hours within a day share the same weather and are not independent.
"""

from __future__ import annotations  # lets type hints like "dict | None" work on older Pythons

import math  # math.ceil for the rank rule

import numpy as np  # arrays and vectorised arithmetic
import pandas as pd  # the alpha path is returned as a table

from .config import BOOTSTRAP_B, POWER_CAP, SEED  # resample count, the [0, 1] cap, the seed


def conformity_scores(y, lo, hi) -> np.ndarray:
    """E = max(lo - y, y - hi) for each row: the signed distance outside the interval."""
    y = np.asarray(y, dtype=float)  # observed outcomes as a float array
    lo = np.asarray(lo, dtype=float)  # lower interval ends
    hi = np.asarray(hi, dtype=float)  # upper interval ends
    return np.maximum(lo - y, y - hi)  # below the interval: lo - y > 0; above: y - hi > 0


def cqr_quantile(scores, alpha: float) -> float:
    """The CQR adjustment Q: the ceil((n + 1)(1 - alpha))-th smallest calibration score.

    The "+ 1" counts the future point being predicted; it is what makes the coverage
    guarantee hold in finite samples (when calibration and test data are exchangeable,
    i.e. their order does not matter). Edge cases:
      rank > n  -> np.inf: no score is large enough; the caller uses the full range [0, 1].
      rank < 1  -> the smallest score (happens in ACI when alpha >= 1).
    """
    s = np.sort(np.asarray(scores, dtype=float))  # scores from smallest to largest
    n = s.size  # number of calibration points
    # The tiny 1e-9 guards against floating-point error, e.g. 90.00000000000001 -> ceil 91.
    rank = math.ceil((n + 1) * (1.0 - alpha) - 1e-9)  # the 1-based rank of the score to use
    if rank > n:  # asks for a score beyond the largest one we have
        return np.inf  # signal "use the full range"
    rank = max(rank, 1)  # alpha >= 1 gives rank <= 0: use the smallest score
    return float(s[rank - 1])  # rank is 1-based, Python indexing is 0-based


def apply_adjustment(lo, hi, Q: float, log: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The recalibrated interval [lo - Q, hi + Q], clipped to [0, 1].

    Q > 0 widens the interval and Q < 0 narrows it; both are correct CQR behaviour.
    Q = inf gives the full range [0, 1]. If a negative Q would push the lower end above
    the upper end, the interval collapses to its midpoint (zero width); `log` counts
    such rows under "collapsed_rows".
    """
    lo = np.asarray(lo, dtype=float)  # base lower ends
    hi = np.asarray(hi, dtype=float)  # base upper ends
    if np.isinf(Q):  # no finite adjustment is large enough
        return np.zeros_like(lo), np.full_like(hi, POWER_CAP)  # the whole possible range
    L = lo - Q  # move the lower end down by Q (up, if Q is negative)
    U = hi + Q  # move the upper end up by Q (down, if Q is negative)
    collapsed = L > U  # a narrowing larger than half the width turns the interval inside out
    if log is not None:  # count the collapsed rows if asked
        log["collapsed_rows"] = log.get("collapsed_rows", 0) + int(collapsed.sum())
    mid = (L + U) / 2  # for collapsed rows, use a zero-width interval at the midpoint
    L = np.where(collapsed, mid, L)  # lower end: midpoint where collapsed
    U = np.where(collapsed, mid, U)  # upper end: midpoint where collapsed
    return np.clip(L, 0.0, POWER_CAP), np.clip(U, 0.0, POWER_CAP)  # power lies in [0, 1]


def aci_daily(day, task, y, lo, hi, scores_by_task: dict, alpha: float, gamma: float):
    """Adaptive conformal inference, updated once per forecast day.

    Arguments (one entry per daylight row, rows sorted by day):
      day             the forecast day of each row (its issue_time)
      task            the task each row belongs to (selects that task's calibration scores)
      y, lo, hi       outcome and base interval
      scores_by_task  {task id: conformity scores on that task's calibration window}
      alpha           the nominal miss rate (0.10 for 90%); also the starting alpha_d
      gamma           step size: how strongly one day's misses move alpha_d

    For each day d, in order:
      1. pick Q from that task's scores at level 1 - alpha_d (alpha_d <= 0 -> [0, 1];
         alpha_d >= 1 -> the smallest score) and form the day's intervals;
      2. after the day ends, err_d = share of its rows outside the interval;
      3. alpha_{d+1} = alpha_d + gamma * (alpha - err_d): too many misses -> lower alpha
         -> wider intervals tomorrow; too few -> narrower.
    Day d's interval uses outcomes from days before d only, so there is no look-ahead.

    Returns (L, U, path, counts): the intervals, a table with alpha_d and err_d per day,
    and how often each edge case happened.
    """
    day = np.asarray(day)  # forecast-day labels, one per row
    task = np.asarray(task)  # task ids, one per row
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))  # numeric arrays
    if not pd.Index(day).is_monotonic_increasing:  # days must come in time order
        raise ValueError("rows must be sorted by forecast day")
    L = np.empty_like(lo)  # the ACI lower ends, filled day by day
    U = np.empty_like(hi)  # the ACI upper ends
    counts = {"alpha_nonpositive_days": 0, "rank_above_n_days": 0, "alpha_at_least_one_days": 0}
    days, starts = np.unique(day, return_index=True)  # each day and the row where it starts
    ends = np.append(starts[1:], len(day))  # each day ends where the next one starts
    a = alpha  # alpha_d: start at the nominal miss rate
    path = []  # one record per day: (day, task, alpha_d, err_d)
    for d, start, end in zip(days, starts, ends):  # walk through the days in time order
        rows = slice(start, end)  # this day's rows
        scores = scores_by_task[task[start]]  # calibration scores of the task this day belongs to
        if a <= 0:  # promised miss rate 0: only the full range is safe
            Q = np.inf  # apply_adjustment turns inf into [0, 1]
            counts["alpha_nonpositive_days"] += 1  # log the edge case
        elif a >= 1:  # promised miss rate 100%: the narrowest interval allowed
            Q = float(np.min(scores))  # the smallest score
            counts["alpha_at_least_one_days"] += 1  # log the edge case
        else:  # the normal case
            Q = cqr_quantile(scores, a)  # the CQR rule at level 1 - alpha_d
            counts["rank_above_n_days"] += int(np.isinf(Q))  # rank > n also gives [0, 1]
        L[rows], U[rows] = apply_adjustment(lo[rows], hi[rows], Q)  # today's intervals
        miss = (y[rows] < L[rows]) | (y[rows] > U[rows])  # which of today's outcomes fell outside
        err = float(miss.mean())  # today's miss rate, known only once the day is over
        path.append((d, task[start], a, err))  # remember alpha_d and err_d
        a = a + gamma * (alpha - err)  # the ACI update for tomorrow
    path = pd.DataFrame(path, columns=["day", "task", "alpha_d", "err_d"])  # as a table
    return L, U, path, counts


def day_bootstrap(num, den, n_boot: int = BOOTSTRAP_B, seed: int = SEED) -> np.ndarray:
    """Bootstrap a ratio sum(num) / sum(den) by resampling whole forecast days.

    num and den hold one total per day, e.g. num = covered hours that day and den = daylight
    hours that day, so the ratio is coverage (PICP). Each replicate draws as many days as
    there are, with replacement, and recomputes the ratio. The same seed and number of days
    always draw the same days, which is what makes paired comparisons (V3 - V2 on the same
    resampled days) possible. Returns the n_boot replicate values.
    """
    num = np.asarray(num, dtype=float)  # per-day numerators
    den = np.asarray(den, dtype=float)  # per-day denominators
    rng = np.random.default_rng(seed)  # a seeded random generator: reproducible draws
    picks = rng.integers(0, num.size, size=(n_boot, num.size))  # n_boot rows of resampled day indices
    with np.errstate(invalid="ignore", divide="ignore"):  # a replicate with no rows gives NaN, not a warning
        return num[picks].sum(axis=1) / den[picks].sum(axis=1)  # the ratio in each replicate


def percentile_ci(replicates, level: float = 0.95) -> tuple[float, float]:
    """The central `level` percentile interval of bootstrap replicates (2.5% and 97.5% for 95%)."""
    tail = (1.0 - level) / 2 * 100  # 2.5 for a 95% interval
    lo, hi = np.nanpercentile(replicates, [tail, 100 - tail])  # ignore NaN replicates
    return float(lo), float(hi)
