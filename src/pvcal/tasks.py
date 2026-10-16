"""The competition's rolling monthly tasks.

Task n trains on everything up to 00:00 UTC on the first day of its test month and
is scored on that month. Task 1 tests April 2013. Tasks 1-3 were trial rounds;
tasks 4-15 (July 2013 - June 2014) were scored, which gives one out-of-sample month
for every season.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FIRST_TEST_MONTH = pd.Timestamp("2013-04-01", tz="UTC")
SCORED_TASK_IDS = tuple(range(4, 16))


@dataclass(frozen=True)
class Task:
    task_id: int
    test_start: pd.Timestamp  # last training timestamp; test rows are strictly after it
    test_end: pd.Timestamp  # last test timestamp (inclusive)

    @property
    def label(self) -> str:
        return self.test_start.strftime("%Y-%m")


def make_task(task_id: int) -> Task:
    if not 1 <= task_id <= 15:
        raise ValueError("GEFCom2014 has tasks 1-15")
    start = FIRST_TEST_MONTH + pd.DateOffset(months=task_id - 1)
    return Task(task_id, start, start + pd.DateOffset(months=1))


def scored_tasks() -> list[Task]:
    return [make_task(n) for n in SCORED_TASK_IDS]


def split(df: pd.DataFrame, task: Task) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: train ends at the test month's first 00:00 UTC stamp."""
    train = df[df["time"] <= task.test_start]
    test = df[(df["time"] > task.test_start) & (df["time"] <= task.test_end)]
    if train.empty or test.empty:
        raise ValueError(f"task {task.task_id}: empty train or test set")
    if train["time"].max() >= test["time"].min():
        raise AssertionError(f"task {task.task_id}: train and test overlap in time")
    return train.reset_index(drop=True), test.reset_index(drop=True)
