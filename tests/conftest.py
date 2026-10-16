"""A synthetic stand-in for the real data, so unit tests run without downloading anything."""

import numpy as np
import pandas as pd
import pytest

from pvcal.data import NWP_VARS, ZONES, issue_time, lead_hours


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    times = pd.date_range("2012-04-01 01:00", "2014-07-01 00:00", freq="h", tz="UTC")
    local_hour = ((times.hour + 10) % 24).to_numpy()
    sun = np.clip(np.sin(np.pi * (local_hour - 6) / 12), 0, None)  # zero at night
    frames = []
    for zone in ZONES:
        clouds = rng.uniform(0, 1, len(times))
        frame = pd.DataFrame({"zone": zone, "time": times})
        for var in NWP_VARS:
            frame[var] = rng.normal(size=len(times))
        frame["tcc"] = clouds
        frame["ssrd"] = 1000 * sun * (1 - 0.7 * clouds)
        frame["power"] = np.clip(frame["ssrd"] / 1000 + rng.normal(0, 0.03, len(times)) * sun, 0, None)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["issue_time"] = issue_time(df["time"])
    df["lead"] = lead_hours(df["time"], df["issue_time"])
    return df
