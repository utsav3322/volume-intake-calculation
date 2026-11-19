"""Time-zone helpers: CowMonitor timestamps are UTC, results are shown in the farm's time."""

from __future__ import annotations

import zoneinfo

import pandas as pd

DEFAULT_TIMEZONE = "America/New_York"
_FALLBACK_ZONES = (
    "UTC",
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "Europe/London",
    "Europe/Berlin",
    "Asia/Kolkata",
    "Australia/Sydney",
)


def available_timezones() -> list[str]:
    zones = sorted(zone for zone in zoneinfo.available_timezones() if "/" in zone or zone == "UTC")
    return zones or list(_FALLBACK_ZONES)


def validate_timezone(name: str) -> str:
    """Return the name unchanged if it is a known IANA time zone, otherwise raise ValueError."""
    try:
        zoneinfo.ZoneInfo(name)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Unknown time zone: {name!r}") from exc
    return name


def local_dates(timestamps: pd.Series, timezone: str) -> pd.Series:
    """Calendar dates of UTC timestamps in the given time zone."""
    return timestamps.dt.tz_convert(timezone).dt.date


def local_day_minutes(dates: pd.Index, timezone: str) -> pd.Series:
    """Length in minutes of each local calendar day: 1440, or 1380/1500 when clocks change."""
    midnights = pd.DatetimeIndex(pd.to_datetime(dates))
    starts = midnights.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
    ends = (midnights + pd.Timedelta(days=1)).tz_localize(
        timezone, ambiguous="NaT", nonexistent="shift_forward"
    )
    minutes = pd.Series((ends - starts).total_seconds() / 60, index=dates)
    return minutes.fillna(1440.0)


def coverage_percent(
    readings_per_day: pd.Series, timezone: str, interval_minutes: float
) -> pd.Series:
    """Share of each local day (indexed by date) that the readings cover, at most 100%."""
    day_minutes = local_day_minutes(readings_per_day.index, timezone)
    return (100 * readings_per_day * interval_minutes / day_minutes).clip(upper=100)


def format_local(moment: pd.Timestamp, timezone: str) -> str:
    return moment.tz_convert(timezone).strftime("%d %b %Y %H:%M")
