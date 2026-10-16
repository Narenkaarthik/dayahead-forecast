import numpy as np
import pandas as pd
import pytest

from pvcal.data import ACCUMULATED, deaccumulate, issue_time, lead_hours
from pvcal.tasks import make_task, scored_tasks, split


def test_hour_ending_stamps_map_to_the_right_forecast():
    t = pd.Series(pd.to_datetime(
        ["2014-06-01 01:00", "2014-06-01 23:00", "2014-06-02 00:00", "2014-06-02 01:00"], utc=True
    ))
    issued = issue_time(t)
    # The 00:00 stamp is lead 24 of the previous day's forecast, not lead 0 of a new one.
    assert list(issued.dt.strftime("%Y-%m-%d")) == ["2014-06-01"] * 3 + ["2014-06-02"]
    assert list(lead_hours(t, issued)) == [1, 23, 24, 1]


def _accumulated(hourly: np.ndarray) -> pd.DataFrame:
    """Build a frame whose accumulated fields sum `hourly` from each 00:00 UTC issue."""
    times = pd.date_range("2014-06-01 01:00", periods=len(hourly), freq="h", tz="UTC")
    df = pd.DataFrame({"zone": 1, "time": times})
    df["issue_time"] = issue_time(df["time"])
    df["lead"] = lead_hours(df["time"], df["issue_time"])
    running_total = pd.Series(hourly).groupby(df["issue_time"]).cumsum().to_numpy()
    for col in ACCUMULATED:
        df[col] = running_total
    return df


def test_deaccumulate_recovers_hourly_values_across_issue_resets():
    hourly = np.random.default_rng(1).uniform(0, 3.6e6, 48)  # two forecasts of 24 steps
    out = deaccumulate(_accumulated(hourly))
    np.testing.assert_allclose(out["ssrd"], hourly / 3600)  # J/m^2 per hour -> W/m^2
    np.testing.assert_allclose(out["tp"], hourly * 1000)  # m -> mm


def test_deaccumulate_clips_negative_noise():
    hourly = np.full(24, 3600.0)
    hourly[5] = -640.0  # the size of the dips seen at night in the real SSRD
    out = deaccumulate(_accumulated(hourly))
    assert out["ssrd"].iloc[5] == 0.0 and (out["ssrd"] >= 0).all()


def test_task_calendar():
    assert make_task(1).label == "2013-04"
    assert make_task(4).label == "2013-07"
    assert make_task(10).label == "2014-01"
    assert make_task(15).label == "2014-06"
    assert [t.task_id for t in scored_tasks()] == list(range(4, 16))
    with pytest.raises(ValueError):
        make_task(16)


def test_split_is_chronological_and_covers_the_whole_month(synthetic_df):
    for task in scored_tasks():
        train, test = split(synthetic_df, task)
        assert train["time"].max() == task.test_start  # train ends at 00:00 UTC on the 1st
        assert test["time"].min() == task.test_start + pd.Timedelta(hours=1)
        assert test["time"].max() == task.test_end
        assert len(test) == 3 * 24 * task.test_start.days_in_month
