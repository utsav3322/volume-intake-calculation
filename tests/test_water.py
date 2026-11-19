"""Daily water use from water-meter exports."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from cowmonitor.loaders import LoadedFile, load_upload
from cowmonitor.water import WaterSummary, summarise_water
from tests.helpers import meter_csv

METER_2D_DAILY = [45.59, 39.76, 38.90, 41.89, 46.00, 41.85, 40.68, 49.90, 39.95, 38.98,
                  49.00, 52.00, 47.80, 53.90, 47.90]  # fmt: skip


def _summary(load: Callable[[str], LoadedFile], name: str) -> WaterSummary:
    meter = load(f"DrinkingData/{name}").meter
    assert meter is not None
    return summarise_water(meter)


@pytest.mark.integration
def test_meter_2d_daily_gallons_and_totals(load_data_file: Callable[[str], LoadedFile]) -> None:
    summary = _summary(load_data_file, "2D.csv")
    assert summary.daily["Gallons"].tolist() == pytest.approx(METER_2D_DAILY)
    assert summary.total_gallons == pytest.approx(674.1)
    assert summary.total_events == 2687
    assert summary.total_minutes == pytest.approx(363)
    assert summary.lost_reports == 19
    assert summary.totals_match
    assert summary.daily["Full day"].all()
    assert len(summary.by_hour) == 24


@pytest.mark.integration
def test_meter_2f_totals(load_data_file: Callable[[str], LoadedFile]) -> None:
    summary = _summary(load_data_file, "2F.csv")
    assert summary.total_gallons == pytest.approx(788.06)
    assert summary.total_events == 1442
    assert summary.total_minutes == pytest.approx(862)
    assert summary.lost_reports == 24
    assert summary.totals_match


@pytest.mark.integration
def test_week_file_matches_the_same_days_of_the_full_file(
    load_data_file: Callable[[str], LoadedFile],
) -> None:
    week = _summary(load_data_file, "2d_417.csv").daily.set_index("Date")["Gallons"]
    full = _summary(load_data_file, "2D.csv").daily.set_index("Date")["Gallons"]
    assert len(week) == 7
    assert week.tolist() == pytest.approx(full.loc[week.index].tolist())


@pytest.mark.unit
def test_days_are_full_dates_so_months_do_not_merge() -> None:
    meter = load_upload("m.csv", meter_csv(start="2024-03-31 12:10", hours=36)).meter
    assert meter is not None
    summary = summarise_water(meter)
    assert summary.daily["Date"].astype(str).tolist() == ["2024-03-31", "2024-04-01"]
    assert summary.daily["Gallons"].tolist() == [24.0, 48.0]
    assert summary.daily["Full day"].tolist() == [False, True]
    assert summary.average_gallons_full_day == pytest.approx(48.0)
    assert any("Partial days" in warning for warning in summary.warnings)


@pytest.mark.unit
def test_hourly_values_that_do_not_add_up_are_flagged() -> None:
    text = meter_csv(hours=24, newest_first=False).decode("utf-8")
    lines = text.split("\r\n")
    fields = lines[5].split(",")
    fields[14] = "9.99"
    lines[5] = ",".join(fields)
    meter = load_upload("m.csv", "\r\n".join(lines).encode("utf-8")).meter
    assert meter is not None
    summary = summarise_water(meter)
    assert not summary.totals_match
    assert any("don't add up" in warning for warning in summary.warnings)
