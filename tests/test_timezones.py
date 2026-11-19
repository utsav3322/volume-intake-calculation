"""Time zones and clock changes: farm-local days, coverage and chart gaps."""

from __future__ import annotations

import pandas as pd
import pytest

from cowmonitor.charts import temperature_chart
from cowmonitor.drinking import daily_summary, estimate_marks
from cowmonitor.loaders import load_upload
from cowmonitor.timezones import available_timezones, format_local, validate_timezone
from tests.helpers import steady_temperatures, temperature_frame, to_csv_bytes


def _readings(frame: pd.DataFrame) -> pd.DataFrame:
    loaded = load_upload("x.csv", to_csv_bytes(frame))
    assert loaded.temperature is not None
    return loaded.temperature.for_cow(loaded.temperature.cows[0].key)


@pytest.mark.unit
def test_timezone_helpers() -> None:
    assert "America/New_York" in available_timezones()
    assert validate_timezone("UTC") == "UTC"
    with pytest.raises(ValueError, match="Unknown time zone"):
        validate_timezone("Mars/Base")
    moment = pd.Timestamp("2024-04-06 04:00", tz="UTC")
    assert format_local(moment, "America/New_York") == "06 Apr 2024 00:00"


@pytest.mark.unit
def test_days_follow_the_farm_calendar() -> None:
    temperatures = steady_temperatures(1)
    temperatures[:3] = [31.5, 33.0, 34.0]
    readings = _readings(temperature_frame(temperatures))
    table = daily_summary(estimate_marks(readings), "America/New_York", 5.0)
    # 24 hours from midnight UTC span two New York (UTC-4) dates: 00:00-04:00 UTC is the last
    # 4 hours of 5 April there, and the drink at the start belongs to that day.
    assert list(table["Date"].astype(str)) == ["2024-04-05", "2024-04-06"]
    assert table["Readings"].sum() == len(readings)
    assert list(table["Coverage %"]) == [pytest.approx(16.7), pytest.approx(83.3)]
    assert list(table["Drinking dips"]) == [1, 0]


@pytest.mark.unit
def test_complete_daylight_saving_day_is_full_coverage() -> None:
    # 10 March 2024 in New York lasts 23 hours: 05:00 UTC to 04:00 UTC the next day.
    frame = temperature_frame(steady_temperatures(23 / 24), start="2024-03-10T05:00:00Z")
    table = daily_summary(estimate_marks(_readings(frame)), "America/New_York", 5.0)
    assert list(table["Date"].astype(str)) == ["2024-03-10"]
    assert table.loc[0, "Coverage %"] == pytest.approx(100.0)


@pytest.mark.unit
def test_chart_breaks_at_real_gaps_but_not_at_clock_changes() -> None:
    readings = _readings(
        temperature_frame(steady_temperatures(4 / 24), start="2024-03-10T05:00:00Z")
    )
    across_clock_change = temperature_chart(readings, "America/New_York", "t")
    assert None not in list(across_clock_change.data[0].y)
    with_gap = readings.drop(index=range(10, 30)).reset_index(drop=True)
    assert list(temperature_chart(with_gap, "America/New_York", "t").data[0].y).count(None) == 1
