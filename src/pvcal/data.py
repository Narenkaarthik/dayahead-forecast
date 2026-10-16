"""Loading and cleaning the GEFCom2014 solar track data.

Facts about the raw files (each one is checked by tests/test_real_data.py):

* TIMESTAMP is UTC and hour-ending: "20120401 01:00" covers 00:00-01:00 UTC.
  The farms sit at roughly UTC+10, so average output peaks around 02:00 UTC.
* The ECMWF forecasts are issued at 00:00 UTC and cover lead times 1-24 h.
  The row stamped "20120402 00:00" is lead 24 of the forecast issued on 2012-04-01.
* SSRD, STRD, TSR and TP are accumulated from the issue time and reset at lead 1,
  so they are differenced here into hourly values.
* In the V2 release, Task 15/predictors15.csv holds the forecasts *and* the
  observed power for the whole period (April 2012 - June 2014).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VAR_NAMES = {
    "VAR78": "tclw",  # total column liquid water, kg/m^2
    "VAR79": "tciw",  # total column ice water, kg/m^2
    "VAR134": "sp",  # surface pressure, Pa
    "VAR157": "rh",  # relative humidity at 1000 hPa, %
    "VAR164": "tcc",  # total cloud cover, 0-1
    "VAR165": "u10",  # 10 m U wind, m/s
    "VAR166": "v10",  # 10 m V wind, m/s
    "VAR167": "t2m",  # 2 m temperature, K
    "VAR169": "ssrd",  # surface solar radiation down (accumulated J/m^2 -> hourly W/m^2)
    "VAR175": "strd",  # surface thermal radiation down (accumulated J/m^2 -> hourly W/m^2)
    "VAR178": "tsr",  # top net solar radiation (accumulated J/m^2 -> hourly W/m^2)
    "VAR228": "tp",  # total precipitation (accumulated m -> hourly mm)
}
NWP_VARS = list(VAR_NAMES.values())
RADIATION = ["ssrd", "strd", "tsr"]
ACCUMULATED = RADIATION + ["tp"]

ZONES = (1, 2, 3)
HOURS_PER_ZONE = 19704  # 2012-04-01 01:00 to 2014-07-01 00:00 UTC, hourly
LOCAL_TZ = "Etc/GMT-10"  # UTC+10 (the sign in Etc/ names is inverted); for plots only


def issue_time(time: pd.Series) -> pd.Series:
    """00:00 UTC issue time of the forecast that covers each hour-ending timestamp."""
    return (time - pd.Timedelta(hours=1)).dt.floor("D")


def lead_hours(time: pd.Series, issued: pd.Series) -> pd.Series:
    """Hours since the forecast was issued (1-24)."""
    return ((time - issued) / pd.Timedelta(hours=1)).astype(int)


def local_time(time: pd.Series) -> pd.Series:
    """Approximate local clock time at the farms. Use it for plots, not for modelling."""
    return time.dt.tz_convert(LOCAL_TZ)


def deaccumulate(df: pd.DataFrame) -> pd.DataFrame:
    """Turn accumulated fields into hourly values: W/m^2 for radiation, mm for precipitation.

    Expects columns zone, time, issue_time, lead and the ACCUMULATED columns.
    """
    out = df.sort_values(["zone", "time"]).copy()
    by_forecast = out.groupby(["zone", "issue_time"], sort=False)
    first_step = out["lead"] == 1
    for col in ACCUMULATED:
        hourly = by_forecast[col].diff().where(~first_step, out[col])
        # Diffs of about -600 J/m^2 appear at night: numerical noise in the NWP output.
        out[col] = hourly.clip(lower=0.0)
    out[RADIATION] = out[RADIATION] / 3600.0  # J/m^2 accumulated over one hour -> mean W/m^2
    out["tp"] = out["tp"] * 1000.0  # m -> mm
    return out


def _find_predictors(path: Path) -> Path:
    if path.is_file():
        return path
    csv = path / "Task 15" / "predictors15.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"Expected {csv}. Point --data at the extracted 'Solar' folder of "
            "GEFCom2014-S_V2 (see data/README.md)."
        )
    return csv


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1").rename(columns={"ZONEID": "zone", "POWER": "power"})
    df["time"] = pd.to_datetime(df.pop("TIMESTAMP"), format="%Y%m%d %H:%M", utc=True)
    return df


def _fill_power_from_solution(df: pd.DataFrame, solar_dir: Path) -> pd.DataFrame:
    """Older copies of the package lack June 2014 power in predictors15.csv; take it from the solution file."""
    solution = solar_dir / "Solution to Task 15" / "Solution to Task 15.csv"
    if not df["power"].isna().any() or not solution.exists():
        return df
    sol = _read_csv(solution).set_index(["zone", "time"])["power"]
    keys = pd.MultiIndex.from_frame(df[["zone", "time"]])
    df["power"] = df["power"].fillna(pd.Series(sol.reindex(keys).to_numpy(), index=df.index))
    return df


def _validate(df: pd.DataFrame) -> None:
    zones = tuple(sorted(df["zone"].unique()))
    if zones != ZONES:
        raise ValueError(f"expected zones {ZONES}, found {zones}")
    counts = df.groupby("zone").size()
    if (counts != HOURS_PER_ZONE).any():
        raise ValueError(f"expected {HOURS_PER_ZONE} hours per zone, found {counts.to_dict()}")
    if df.duplicated(["zone", "time"]).any():
        raise ValueError("duplicate (zone, time) rows")
    steps = df.groupby("zone")["time"].diff().dropna()
    if (steps != pd.Timedelta(hours=1)).any():
        raise ValueError("the hourly series has gaps")
    if df["power"].isna().any():
        raise ValueError("observed power is missing for some hours")


def load_solar(path: str | Path) -> pd.DataFrame:
    """Load the full dataset, cleaned: one row per zone and hour.

    Columns: zone, time (UTC, hour-ending), issue_time, lead, the 12 NWP variables
    (radiation as hourly W/m^2, precipitation as hourly mm) and power (0-1 of capacity).
    """
    path = Path(path)
    csv = _find_predictors(path)
    df = _read_csv(csv).rename(columns=VAR_NAMES)
    df = _fill_power_from_solution(df, csv.parent.parent)
    df = df.sort_values(["zone", "time"]).reset_index(drop=True)
    _validate(df)
    df["issue_time"] = issue_time(df["time"])
    df["lead"] = lead_hours(df["time"], df["issue_time"])
    df = deaccumulate(df)
    return df[["zone", "time", "issue_time", "lead", *NWP_VARS, "power"]].reset_index(drop=True)


def load_benchmark(solar_dir: str | Path, task_id: int) -> tuple[pd.DataFrame, np.ndarray]:
    """The official benchmark forecast for one task.

    It is last year's power at the same hour, written into all 99 quantile columns
    (tests/test_real_data.py confirms this). Returns (zone/time frame, quantile array).
    """
    csv = Path(solar_dir) / f"Task {task_id}" / f"benchmark{task_id:02d}.csv"
    b = _read_csv(csv).sort_values(["zone", "time"]).reset_index(drop=True)
    qcols = [c for c in b.columns if c not in ("zone", "time")]
    if len(qcols) != 99:
        raise ValueError(f"expected 99 quantile columns in {csv}, found {len(qcols)}")
    return b[["zone", "time"]], b[qcols].to_numpy(dtype=float)
