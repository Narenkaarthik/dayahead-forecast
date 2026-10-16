"""Fixed settings for the models, the calibration study and the evaluation, in one place.

Every number a model or script depends on lives here with a short reason, so nothing
is a "magic number" hidden inside a function. Changing a value here changes the study,
so log any change in DECISIONS.md.
"""

from __future__ import annotations

import pandas as pd

# --- Reproducibility -------------------------------------------------------------------
SEED = 0  # one seed for everything random (LightGBM, bootstrap, plot sampling)

# --- Tasks -----------------------------------------------------------------------------
# Tasks 1-3 (April-June 2013) are for development: every choice is made on them and then
# frozen. Tasks 4-15 are the scored test months (pvcal.tasks.SCORED_TASK_IDS).
DEV_TASK_IDS = (1, 2, 3)
ALL_TASK_IDS = tuple(range(1, 16))
# The leakage check reruns a model twice, so for the expensive quantile models it runs on
# the first and last scored task only. It tests the code path, not the data, so two are enough.
LEAKAGE_CHECK_TASKS = (4, 15)

# --- Reference score -------------------------------------------------------------------
# Average pinball loss of the GEFCom2014 winner over the 12 scored tasks, as reported in
# arXiv:1903.07390. Reported, not reproduced; the official ranking used a weighted relative
# score, so any comparison with it is approximate.
WINNER_AVG_PINBALL = 0.0121

# --- Post-processing -------------------------------------------------------------------
POWER_CAP = 1.0  # power is a fraction of capacity, so no quantile should exceed 1

# --- Linear quantile regression --------------------------------------------------------
QUANTREG_MAX_ITER = 5000  # iteration limit for statsmodels' QuantReg solver

# --- LightGBM quantile regression ------------------------------------------------------
LEARNING_RATE = 0.05
# The tuning grid: 2 x 2 x 2 = 8 configurations, compared on the dev tasks only.
LGBM_GRID = {
    "num_leaves": (15, 31),
    "min_child_samples": (20, 100),
    "n_estimators": (300, 800),
}
# Tuning scores only these five levels (fitting all 99 for 8 configs would be slow, and
# these span the lower tail, the middle and the upper tail).
TUNE_TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Runtime fallback: fit only these 21 levels and interpolate to 99. Used only if the full
# Track 1 run is projected to take more than FALLBACK_IF_HOURS.
FALLBACK_LEVELS = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                   0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99)
FALLBACK_IF_HOURS = 3.0

# --- Calibration study (Track 2) -------------------------------------------------------
# Central prediction intervals: nominal coverage -> (lower level, upper level).
INTERVALS = {0.8: (0.10, 0.90), 0.9: (0.05, 0.95)}
# The four quantile levels the split model needs for those two intervals.
SPLIT_LEVELS = (0.05, 0.10, 0.90, 0.95)
CAL_WINDOW_DAYS = 60  # calibration window: the 60 days just before each test month
ACI_GAMMAS = (0.005, 0.01, 0.02, 0.05)  # candidate ACI step sizes, chosen on dev tasks
BOOTSTRAP_B = 1000  # bootstrap resamples of forecast days
# Cloud-cover terciles are cut using daylight rows *before* the first scored month, so the
# cut points never depend on test-period data.
TCC_CUTOFF = pd.Timestamp("2013-07-01", tz="UTC")
