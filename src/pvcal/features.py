"""Model inputs.

Everything here is known at the 00:00 UTC issue time: the ECMWF forecast for the
hour and calendar terms. Observed power is never an input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import NWP_VARS

# Forecast SSRD at or below this is treated as night. On the real data, observed power
# at such hours never exceeds 0.018 of capacity (tests/test_real_data.py).
NIGHT_SSRD_WM2 = 1.0


def is_daylight(df: pd.DataFrame, threshold: float = NIGHT_SSRD_WM2) -> np.ndarray:
    """Daylight according to the forecast, so the mask is available at issue time."""
    return (df["ssrd"] > threshold).to_numpy()


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    hour = df["time"].dt.hour
    doy = df["time"].dt.dayofyear
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=df.index,
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """NWP forecasts plus calendar terms, one row per input row (the models in pvcal.models build their own)."""
    return pd.concat([df[NWP_VARS], calendar_features(df)], axis=1)
