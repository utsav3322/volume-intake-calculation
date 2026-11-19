"""Daily drinking summary, ported from ``loadData()`` + ``generateX()`` in WaterIntakeReg.ipynb.

The notebook finds drinking readings by comparing a cow's full temperature file with
CowMonitor's copy of the same data in which the drinking readings were removed: any reading
missing from the copy counts as drinking. When that copy is not available, this module can
estimate the drinking readings from the temperature alone: a reading more than 1.25 °C below
the cow's median of the previous 12 hours. On the project's four cows that estimate catches
about 84% of CowMonitor's drinking readings, and about 83% of what it flags matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from cowmonitor.loaders import UnsupportedFileError
from cowmonitor.timezones import coverage_percent, local_dates

FEATURE_THRESHOLDS = (32.5, 33.0, 33.5)
MIN_MATCHED_SHARE = 0.9
DIP_GAP_MINUTES = 15.0
ESTIMATED_WARNING = (
    "Drinking readings are estimated from the temperature alone. Upload CowMonitor's "
    "drinking-removed file for the exact figures the project used."
)

# Friendly column names and the notebook's feature names they correspond to.
NOTEBOOK_FEATURES = {
    "Drinking readings": "NumDrink",
    "Coldest drinking reading °C": "MinDrop",
    "Drinking readings below 32.5 °C": "NumBelowT1",
    "Drinking readings below 33 °C": "NumBelowT2",
    "Drinking readings below 33.5 °C": "NumBelowT3",
    "Average drinking temperature °C": "AvgDrink",
    "Sum of drinking temperatures": "TotalDrink",
    "Average temperature when not drinking °C": "AvgTemp",
}


class DrinkingSource(StrEnum):
    COWMONITOR_COPY = "copy"
    ESTIMATED = "estimated"


@dataclass(frozen=True)
class DetectorParams:
    drop_c: float = 1.25
    baseline_hours: float = 12.0
    min_baseline_readings: int = 12


@dataclass(frozen=True)
class DrinkingMarks:
    """One cow's readings (``ts``, ``temperature``) with a boolean ``drinking`` column."""

    readings: pd.DataFrame
    source: DrinkingSource
    warnings: tuple[str, ...] = ()


def marks_from_copy(readings: pd.DataFrame, copy: pd.DataFrame) -> DrinkingMarks:
    """Mark as drinking every reading that CowMonitor's drinking-removed copy left out."""
    start, end = copy["ts"].min(), copy["ts"].max()
    inside = readings["ts"].between(start, end)
    matched = copy["ts"].isin(readings["ts"])
    if matched.mean() < MIN_MATCHED_SHARE:
        raise UnsupportedFileError(
            "The drinking-removed file doesn't match this cow's readings: only "
            f"{matched.mean():.0%} of its times appear in the temperature file. "
            "Check that both files are for the same device and period."
        )
    warnings = []
    if (~inside).any():
        warnings.append(
            f"{int((~inside).sum())} readings fall outside the drinking-removed file's time range "
            "and were left out."
        )
    if (~matched).any():
        warnings.append(
            f"{int((~matched).sum())} times in the drinking-removed file are not in the "
            "temperature file."
        )
    within = readings[inside].reset_index(drop=True)
    marked = within.assign(drinking=~within["ts"].isin(copy["ts"]))
    if not marked["drinking"].any():
        warnings.append(
            "The drinking-removed file has every reading of the temperature file, so no drinking "
            "was found. Check that it is CowMonitor's processed copy."
        )
    return DrinkingMarks(marked, DrinkingSource.COWMONITOR_COPY, tuple(warnings))


def estimate_marks(readings: pd.DataFrame, params: DetectorParams | None = None) -> DrinkingMarks:
    """Estimate drinking readings from the temperature alone (no CowMonitor copy needed)."""
    params = params or DetectorParams()
    series = readings.set_index("ts")["temperature"]
    baseline = series.rolling(
        f"{params.baseline_hours:g}h", closed="left", min_periods=params.min_baseline_readings
    ).median()
    baseline = baseline.fillna(series.median())
    drinking = (series < baseline - params.drop_c).to_numpy()
    marked = readings.reset_index(drop=True).assign(drinking=drinking)
    return DrinkingMarks(marked, DrinkingSource.ESTIMATED, (ESTIMATED_WARNING,))


def dip_starts(readings: pd.DataFrame) -> pd.Series:
    """True at the first reading of each dip: a run of drinking readings with no gap over 15 min."""
    previous = readings["drinking"].shift(1, fill_value=False)
    long_gap = readings["ts"].diff().dt.total_seconds().div(60).gt(DIP_GAP_MINUTES)
    return readings["drinking"] & (~previous | long_gap)


def daily_summary(marks: DrinkingMarks, timezone: str, interval_minutes: float) -> pd.DataFrame:
    """Per local calendar day: the notebook's 8 drinking features plus dips and coverage."""
    frame = marks.readings.assign(date=local_dates(marks.readings["ts"], timezone))
    frame["dip_start"] = dip_starts(frame)

    days = pd.Index(sorted(frame["date"].unique()), name="date")
    drink = frame[frame["drinking"]].groupby("date")["temperature"]
    normal = frame[~frame["drinking"]].groupby("date")["temperature"]
    table = pd.DataFrame(index=days)
    table["Readings"] = frame.groupby("date").size()
    table["Coverage %"] = coverage_percent(table["Readings"], timezone, interval_minutes)
    table["Drinking dips"] = frame.groupby("date")["dip_start"].sum()
    table["Drinking readings"] = drink.size()
    table["Coldest drinking reading °C"] = drink.min()
    for threshold in FEATURE_THRESHOLDS:
        table[f"Drinking readings below {threshold:g} °C"] = drink.agg(
            lambda values, limit=threshold: int((values < limit).sum())
        )
    table["Average drinking temperature °C"] = drink.mean()
    table["Sum of drinking temperatures"] = drink.sum()
    table["Average temperature when not drinking °C"] = normal.mean()

    counts = ["Drinking dips", "Drinking readings"] + [
        f"Drinking readings below {threshold:g} °C" for threshold in FEATURE_THRESHOLDS
    ]
    table[counts] = table[counts].fillna(0).astype(int)
    table["Sum of drinking temperatures"] = table["Sum of drinking temperatures"].fillna(0.0)
    table = table.reset_index().rename(columns={"date": "Date"})
    return table.round(
        {
            "Coverage %": 1,
            "Coldest drinking reading °C": 2,
            "Average drinking temperature °C": 2,
            "Sum of drinking temperatures": 2,
            "Average temperature when not drinking °C": 2,
        }
    )
