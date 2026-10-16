"""Checks every claim this project makes about the raw data. Skipped until the data is downloaded."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pvcal.baselines import yearly_persistence
from pvcal.data import HOURS_PER_ZONE, load_benchmark, load_solar
from pvcal.features import is_daylight
from pvcal.tasks import scored_tasks, split

DATA = Path(os.environ.get("GEFCOM_SOLAR_DIR", Path(__file__).parents[1] / "data" / "raw" / "Solar"))

pytestmark = pytest.mark.skipif(
    not (DATA / "Task 15" / "predictors15.csv").exists(),
    reason="GEFCom2014 solar data not downloaded (see data/README.md)",
)


@pytest.fixture(scope="module")
def df():
    return load_solar(DATA)  # also validates: 3 zones, no gaps, no duplicates, no missing power


def test_complete_hourly_series(df):
    assert len(df) == 3 * HOURS_PER_ZONE


def test_timestamps_are_utc(df):
    # Mean output peaks at 02:00 UTC, which is local noon at UTC+10.
    assert df.groupby(df["time"].dt.hour)["power"].mean().idxmax() in (1, 2, 3)


def test_deaccumulated_fields_are_physical(df):
    assert df["ssrd"].between(0, 1400).all()  # hourly-mean W/m^2
    assert (df[["strd", "tsr", "tp"]] >= 0).all().all()


def test_forecast_night_really_is_dark(df):
    assert df.loc[~is_daylight(df), "power"].max() < 0.02


def test_june_2014_power_matches_the_solution_file(df):
    sol = pd.read_csv(DATA / "Solution to Task 15" / "Solution to Task 15.csv", encoding="latin1")
    june = df[df["time"] > pd.Timestamp("2014-06-01", tz="UTC")]
    np.testing.assert_allclose(june["power"].to_numpy(), sol["POWER"].to_numpy(), atol=1e-6)


def test_official_benchmark_is_last_years_power(df):
    for task in scored_tasks():
        train, test = split(df, task)
        keys, q = load_benchmark(DATA, task.task_id)
        assert (keys["time"].to_numpy() == test["time"].to_numpy()).all()
        assert (keys["zone"].to_numpy() == test["zone"].to_numpy()).all()
        np.testing.assert_allclose(q, yearly_persistence(train, test), atol=1e-6)


def test_power_is_a_fraction_of_capacity(df):
    # Max is 1.0036 (one hour above 1), so forecasts are capped at 1 (pvcal.config.POWER_CAP).
    assert df["power"].between(0, 1.05).all()
