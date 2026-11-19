"""Daily water use from an H2O Degree water-meter export (``loadWaterMeter()`` in the notebook).

The meter sends cumulative registers every hour (gallons, flow events, minutes of flow), and
the export carries the change since the previous report. Adding the hourly changes per day
gives the gallons the cows on that meter drank that day. A missed report loses no water: the
next report's change covers it, because the registers are cumulative.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cowmonitor.loaders import MeterData

FULL_DAY_REPORTS = 20


@dataclass(frozen=True)
class WaterSummary:
    daily: pd.DataFrame
    by_hour: pd.DataFrame
    total_gallons: float
    total_events: int
    total_minutes: float
    average_gallons_full_day: float | None
    first_read: pd.Timestamp
    last_read: pd.Timestamp
    lost_reports: int
    totals_match: bool
    warnings: tuple[str, ...]


def summarise_water(meter: MeterData) -> WaterSummary:
    """Gallons, flow events and minutes of flow per day, plus an hour-of-day profile."""
    readings = meter.readings
    dates = readings["read_time"].dt.date
    grouped = readings.groupby(dates)
    daily = pd.DataFrame(
        {
            "Gallons": grouped["gallons"].sum().round(2),
            "Flow events": grouped["events"].sum().astype(int),
            "Minutes of flow": grouped["minutes"].sum().round(1),
            "Hourly reports": grouped.size(),
        }
    )
    daily["Full day"] = daily["Hourly reports"] >= FULL_DAY_REPORTS
    daily = daily.rename_axis("Date").reset_index()

    full_dates = set(daily.loc[daily["Full day"], "Date"])
    on_full_days = readings[dates.isin(full_dates)]
    by_hour = (
        on_full_days.groupby(on_full_days["read_time"].dt.hour)["gallons"]
        .mean()
        .round(2)
        .rename("Average gallons")
        .rename_axis("Hour")
        .reset_index()
    )

    report_steps = readings["report_number"].diff().dropna()
    lost_reports = int((report_steps[report_steps > 1] - 1).sum())
    hourly_after_first = float(readings["gallons"].iloc[1:].sum())
    register_change = float(readings["gallons_total"].iloc[-1] - readings["gallons_total"].iloc[0])
    totals_match = abs(hourly_after_first - register_change) <= 0.05 + 0.001 * abs(register_change)

    warnings = list(meter.warnings)
    partial = daily.loc[~daily["Full day"], "Date"].astype(str).tolist()
    if partial:
        warnings.append(
            f"Partial days (fewer than {FULL_DAY_REPORTS} hourly reports): {', '.join(partial)}. "
            "Their totals may be low."
        )
    if not totals_match:
        warnings.append(
            "The hourly values don't add up to the meter's running total "
            f"({hourly_after_first:.1f} vs {register_change:.1f} gallons). Check the export."
        )
    average = float(daily.loc[daily["Full day"], "Gallons"].mean()) if full_dates else None
    return WaterSummary(
        daily=daily,
        by_hour=by_hour,
        total_gallons=round(float(readings["gallons"].sum()), 2),
        total_events=int(readings["events"].sum()),
        total_minutes=round(float(readings["minutes"].sum()), 1),
        average_gallons_full_day=average,
        first_read=readings["read_time"].iloc[0],
        last_read=readings["read_time"].iloc[-1],
        lost_reports=lost_reports,
        totals_match=totals_match,
        warnings=tuple(warnings),
    )
