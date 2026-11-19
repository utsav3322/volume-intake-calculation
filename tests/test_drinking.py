"""Daily drinking summary: parity with the notebook's features, and the estimated mode."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cowmonitor.drinking import (
    NOTEBOOK_FEATURES,
    DrinkingSource,
    daily_summary,
    estimate_marks,
    marks_from_copy,
)
from cowmonitor.loaders import LoadedFile, UnsupportedFileError, load_upload
from tests.helpers import steady_temperatures, temperature_frame, to_csv_bytes

DIP_STARTS = (100, 160, 220)  # three drinking dips on the first day, 3 readings each


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two local (New York) days of one cow, and its drinking-removed copy."""
    temperatures = steady_temperatures(2)
    dip_rows = [start + k for start in DIP_STARTS for k in range(3)]
    temperatures[dip_rows] = [34.0, 35.0, 36.5] * len(DIP_STARTS)
    full = temperature_frame(temperatures, start="2024-04-06T04:00:00Z")
    copy = full.drop(index=dip_rows)
    return full, copy


def _readings(frame: pd.DataFrame) -> pd.DataFrame:
    loaded = load_upload("x.csv", to_csv_bytes(frame))
    assert loaded.temperature is not None
    return loaded.temperature.for_cow(loaded.temperature.cows[0].key)


@pytest.mark.unit
def test_copy_mode_counts_dips_and_days_without_drinking() -> None:
    full, copy = _frames()
    marks = marks_from_copy(_readings(full), _readings(copy))
    assert marks.source is DrinkingSource.COWMONITOR_COPY
    table = daily_summary(marks, "America/New_York", interval_minutes=5.0)
    assert list(table["Date"].astype(str)) == ["2024-04-06", "2024-04-07"]
    first, second = table.iloc[0], table.iloc[1]
    assert first["Drinking dips"] == 3
    assert first["Drinking readings"] == 9
    assert first["Coldest drinking reading °C"] == pytest.approx(34.0)
    assert first["Drinking readings below 33.5 °C"] == 0
    assert second["Drinking dips"] == 0
    assert second["Drinking readings"] == 0
    assert np.isnan(second["Coldest drinking reading °C"])
    assert first["Coverage %"] == pytest.approx(100.0)


@pytest.mark.unit
def test_estimated_mode_finds_the_same_dips() -> None:
    full, _ = _frames()
    marks = estimate_marks(_readings(full))
    assert marks.source is DrinkingSource.ESTIMATED
    assert marks.warnings
    table = daily_summary(marks, "America/New_York", interval_minutes=5.0)
    assert table.iloc[0]["Drinking dips"] == 3
    assert table.iloc[1]["Drinking dips"] == 0


@pytest.mark.unit
def test_readings_outside_the_copy_are_left_out() -> None:
    full, copy = _frames()
    first_day_copy = copy[pd.to_datetime(copy["timestamp"]) < pd.Timestamp("2024-04-07 04:00Z")]
    marks = marks_from_copy(_readings(full), _readings(first_day_copy))
    assert len(marks.readings) < len(full)
    assert any("outside" in warning for warning in marks.warnings)


@pytest.mark.unit
def test_copy_from_another_period_is_rejected() -> None:
    full, copy = _frames()
    shifted = copy.assign(
        timestamp=(pd.to_datetime(copy["timestamp"]) + pd.Timedelta(minutes=2)).dt.strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        )
    )
    with pytest.raises(UnsupportedFileError, match="doesn't match"):
        marks_from_copy(_readings(full), _readings(shifted))


@pytest.mark.integration
@pytest.mark.parametrize("cow", ["1354", "1297"])
def test_copy_mode_reproduces_the_notebook_features(
    fixtures_dir: Path, load_data_file: Callable[[str], LoadedFile], cow: str
) -> None:
    raw = load_data_file(f"DrinkingData/{cow}_raw.csv").temperature
    copy = load_data_file(f"DrinkingData/{cow}_bouts.csv").temperature
    assert raw is not None and copy is not None
    marks = marks_from_copy(raw.for_cow(raw.cows[0].key), copy.for_cow(copy.cows[0].key))
    table = daily_summary(marks, "America/New_York", raw.interval_minutes)
    expected = pd.read_csv(fixtures_dir / f"notebook_drinking_features_{cow}.csv")
    table["Day"] = pd.to_datetime(table["Date"]).dt.day
    merged = table.merge(expected, left_on="Day", right_on="Date", suffixes=("", "_nb"))
    assert len(merged) == len(expected) == 15
    for friendly, notebook in NOTEBOOK_FEATURES.items():
        assert np.allclose(merged[friendly], merged[notebook], atol=0.01), friendly


@pytest.mark.integration
def test_estimated_mode_agrees_with_cowmonitor_on_the_four_cows(
    load_data_file: Callable[[str], LoadedFile],
) -> None:
    true_positive = flagged = actual = 0
    for cow in ("1297", "1354", "1568", "1709"):
        raw = load_data_file(f"DrinkingData/{cow}_raw.csv").temperature
        copy = load_data_file(f"DrinkingData/{cow}_bouts.csv").temperature
        assert raw is not None and copy is not None
        readings = raw.for_cow(raw.cows[0].key)
        truth = marks_from_copy(readings, copy.for_cow(copy.cows[0].key)).readings
        guess = estimate_marks(readings).readings.set_index("ts").loc[truth["ts"], "drinking"]
        true_positive += int((guess.to_numpy() & truth["drinking"].to_numpy()).sum())
        flagged += int(guess.sum())
        actual += int(truth["drinking"].sum())
    precision, recall = true_positive / flagged, true_positive / actual
    assert precision > 0.8
    assert recall > 0.8


@pytest.mark.integration
def test_repeated_row_in_a_delivered_copy_is_reported(
    load_data_file: Callable[[str], LoadedFile],
) -> None:
    copy = load_data_file("DrinkingData/1568_417_bouts.csv").temperature
    assert copy is not None
    assert any("repeated" in warning for warning in copy.warnings)


@pytest.mark.unit
def test_copy_that_removed_nothing_is_flagged() -> None:
    full, _ = _frames()
    readings = _readings(full)
    marks = marks_from_copy(readings, readings)
    assert not marks.readings["drinking"].any()
    assert any("no drinking was found" in warning for warning in marks.warnings)
