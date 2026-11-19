"""Model 2: calving alert, ported from ``calving()`` in HealthEventsDetection.ipynb.

A cow's temperature drops shortly before calving. The model takes the median temperature of
each half-day (00:00-12:00 and 12:00-24:00 UTC), ignoring readings outside 37-40 °C (drinking
drops and spikes), and flags a calving-like drop when a half-day is more than 0.5 °C below
the average of the previous four. It then checks whether the temperature rises again
afterwards: a half-day more than 0.4 °C below the average of itself and the next nine.

Two versions are available:

* ``CalvingVersion.FIXED`` (default) compares each half-day with the half-days that really
  came before it, skips half-days with too little data instead of treating them as 0 °C,
  and only looks for the rise after the drop.
* ``CalvingVersion.AS_DELIVERED`` reproduces the notebook exactly, including its indexing
  slip (each half-day is compared with the wrong neighbours), the 0 °C filling of missing
  half-days, and a recovery check that is not tied to the drop.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

HALF_DAY = "12h"


class CalvingVersion(StrEnum):
    FIXED = "fixed"
    AS_DELIVERED = "as_delivered"


@dataclass(frozen=True)
class CalvingParams:
    low_c: float = 37.0
    high_c: float = 40.0
    drop_c: float = 0.5
    rise_c: float = 0.4
    history: int = 4
    recover_start: int = 6
    forward: int = 10
    min_readings: int = 12
    last_complete_share: float = 0.75

    @property
    def min_half_days(self) -> int:
        return self.history + 1


@dataclass(frozen=True)
class HalfDayEvent:
    """A flagged half-day: when it started (UTC), its median, and what it was compared with."""

    start: pd.Timestamp
    temperature: float
    reference: float

    @property
    def change(self) -> float:
        return self.temperature - self.reference


@dataclass(frozen=True)
class CalvingResult:
    version: CalvingVersion
    enough_data: bool
    calving: bool
    recovered: bool
    recovery_checked: bool
    calving_events: tuple[HalfDayEvent, ...]
    recovery_events: tuple[HalfDayEvent, ...]
    half_days: pd.DataFrame

    @property
    def first_calving(self) -> HalfDayEvent | None:
        return self.calving_events[0] if self.calving_events else None


def check_calving(
    readings: pd.DataFrame,
    version: CalvingVersion = CalvingVersion.FIXED,
    drinking_removed: bool = False,
    interval_minutes: float = 5.0,
    params: CalvingParams | None = None,
) -> CalvingResult:
    """Run the calving model on one cow's readings (columns ``ts`` in UTC and ``temperature``)."""
    params = params or CalvingParams()
    if version is CalvingVersion.AS_DELIVERED:
        return _as_delivered(readings, params, drinking_removed)
    return _fixed(readings, params, drinking_removed, interval_minutes)


def half_day_table(
    readings: pd.DataFrame, params: CalvingParams, drinking_removed: bool, interval_minutes: float
) -> pd.DataFrame:
    """Median temperature of each UTC half-day, and whether the model can use the half-day.

    A half-day is skipped when it has fewer than ``min_readings`` readings in range. The last
    half-day of a file is usually cut off by the export, so, as in the notebook, it is left
    out unless it is nearly complete.
    """
    frame = readings.assign(start=readings["ts"].dt.floor(HALF_DAY))
    kept = (
        frame
        if drinking_removed
        else frame[frame["temperature"].between(params.low_c, params.high_c)]
    )
    table = pd.DataFrame({"readings": frame.groupby("start").size()})
    table["readings_used"] = kept.groupby("start").size()
    table["median_c"] = kept.groupby("start")["temperature"].median()
    table["readings_used"] = table["readings_used"].fillna(0).astype(int)
    table["note"] = ""
    too_few = table["readings_used"] < params.min_readings
    table.loc[too_few, "note"] = "too few readings in 37-40 °C"
    expected = 12 * 60 / interval_minutes
    if not table.empty and table["readings"].iloc[-1] < params.last_complete_share * expected:
        table.loc[table.index[-1], "note"] = "last half-day, incomplete"
    table["used"] = table["note"].eq("")
    return table.reset_index()


def _fixed(
    readings: pd.DataFrame, params: CalvingParams, drinking_removed: bool, interval_minutes: float
) -> CalvingResult:
    table = half_day_table(readings, params, drinking_removed, interval_minutes)
    used = table[table["used"]].reset_index(drop=True)
    values = used["median_c"].to_numpy(dtype=float)
    baseline = pd.Series(values).shift(1).rolling(params.history).mean().to_numpy()
    table["baseline_c"] = np.nan
    table.loc[table["used"], "baseline_c"] = baseline
    table["change_c"] = table["median_c"] - table["baseline_c"]

    enough = len(values) >= params.min_half_days
    calving_idx = [
        i for i in range(params.history, len(values)) if values[i] - baseline[i] < -params.drop_c
    ]
    calving_events = tuple(_event(used, i, values[i], baseline[i]) for i in calving_idx)

    recovery_events: tuple[HalfDayEvent, ...] = ()
    recovery_checked = False
    if calving_idx:
        first = calving_idx[0]
        recovery_checked = len(values) - 1 - first >= 2
        recovery_events = tuple(
            _event(used, j, values[j], float(values[j : j + params.forward].mean()))
            for j in range(first, len(values))
            if values[j : j + params.forward].mean() - values[j] > params.rise_c
        )
    return CalvingResult(
        version=CalvingVersion.FIXED,
        enough_data=enough,
        calving=bool(calving_events),
        recovered=bool(recovery_events),
        recovery_checked=recovery_checked,
        calving_events=calving_events,
        recovery_events=recovery_events,
        half_days=table,
    )


def _as_delivered(
    readings: pd.DataFrame, params: CalvingParams, drinking_removed: bool
) -> CalvingResult:
    """Line-for-line port of the notebook's calving(), kept for comparison."""
    data = readings.assign(date=readings["ts"].dt.date, hour=readings["ts"].dt.hour)
    if not drinking_removed:
        data = data[(data["temperature"] >= params.low_c) & (data["temperature"] <= params.high_c)]
    if data.empty:
        return _empty_result(CalvingVersion.AS_DELIVERED)
    data = data.assign(time_of_day=np.where(data["hour"] < 12, "first_half", "second_half"))
    medians = data.groupby(["date", "time_of_day"])["temperature"].median().unstack(fill_value=0)
    for half in ("first_half", "second_half"):
        if half not in medians.columns:
            medians[half] = np.nan
    medians.loc[medians.index[-1], "second_half"] = np.nan  # the notebook drops the last half-day
    flat = medians.reset_index().melt(
        id_vars=["date"],
        value_vars=["first_half", "second_half"],
        var_name="time_of_day",
        value_name="temperature",
    )
    flat = flat.dropna()
    flat["sort_key"] = flat["date"].astype(str) + flat["time_of_day"]
    flat = flat.sort_values(by="sort_key")
    flat["start"] = pd.to_datetime(flat["date"]).dt.tz_localize("UTC") + pd.to_timedelta(
        np.where(flat["time_of_day"] == "first_half", 0, 12), unit="h"
    )

    # The notebook reads the current value by label (flatData['temperature'][i]) but the
    # window by position ([i-4:i]); after sorting, labels and positions differ.
    by_label = flat["temperature"]
    by_position = flat["temperature"].to_numpy(dtype=float)

    def current(i: int) -> float:
        return float(by_label.loc[i]) if i in by_label.index else float("nan")

    calving_events = []
    for i in range(params.history, len(flat)):
        window = float(by_position[i - params.history : i].mean())
        if current(i) - window < -params.drop_c:
            calving_events.append(_event(flat, i, current(i), window))
    recovery_events = []
    if calving_events:
        for i in range(params.recover_start, len(flat)):
            window = float(by_position[i : i + params.forward].mean())
            if window - current(i) > params.rise_c:
                recovery_events.append(_event(flat, i, current(i), window))

    table = flat[["start", "temperature"]].rename(columns={"temperature": "median_c"})
    table = table.assign(used=True, filled_with_zero=table["median_c"].eq(0)).reset_index(drop=True)
    return CalvingResult(
        version=CalvingVersion.AS_DELIVERED,
        enough_data=len(flat) >= params.min_half_days,
        calving=bool(calving_events),
        recovered=bool(recovery_events),
        recovery_checked=bool(calving_events) and len(flat) > params.recover_start,
        calving_events=tuple(calving_events),
        recovery_events=tuple(recovery_events),
        half_days=table,
    )


def _event(
    frame: pd.DataFrame, position: int, temperature: float, reference: float
) -> HalfDayEvent:
    return HalfDayEvent(
        start=pd.Timestamp(frame["start"].iloc[position]),
        temperature=float(temperature),
        reference=float(reference),
    )


def _empty_result(version: CalvingVersion) -> CalvingResult:
    return CalvingResult(
        version=version,
        enough_data=False,
        calving=False,
        recovered=False,
        recovery_checked=False,
        calving_events=(),
        recovery_events=(),
        half_days=pd.DataFrame(
            {
                "start": pd.Series(dtype="datetime64[ns, UTC]"),
                "median_c": pd.Series(dtype=float),
                "used": pd.Series(dtype=bool),
            }
        ),
    )
