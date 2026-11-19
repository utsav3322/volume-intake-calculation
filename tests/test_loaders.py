"""Reading uploads and recognising temperature and water-meter files."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cowmonitor.loaders import FileKind, LoadedFile, UnsupportedFileError, load_upload
from tests.helpers import meter_csv, steady_temperatures, temperature_frame, to_csv_bytes


def _load(frame: pd.DataFrame, name: str = "upload.csv") -> LoadedFile:
    return load_upload(name, to_csv_bytes(frame))


@pytest.mark.unit
def test_minimal_columns_are_enough() -> None:
    frame = temperature_frame(steady_temperatures(1), animal=None)
    loaded = _load(frame)
    assert loaded.kind is FileKind.TEMPERATURE
    data = loaded.temperature
    assert data is not None
    assert [cow.key for cow in data.cows] == ["281b0400002004e2"]
    assert data.interval_minutes == pytest.approx(5.0)
    assert len(data.readings) == len(frame)
    assert str(data.readings["ts"].dt.tz) == "UTC"


@pytest.mark.unit
def test_column_names_are_matched_loosely() -> None:
    frame = temperature_frame(steady_temperatures(0.5)).rename(
        columns={"deviceId": " Device_ID ", "temperature": "Temp", "timestamp": "TimeStamp"}
    )
    data = _load(frame).temperature
    assert data is not None
    assert data.cows[0].device_id == "281b0400002004e2"
    assert data.cows[0].animal_id == "1354"
    assert data.cows[0].label == "Cow 1354 (device 281b0400002004e2)"


@pytest.mark.unit
def test_animal_id_is_used_when_there_is_no_device_column() -> None:
    frame = temperature_frame(steady_temperatures(0.5)).drop(columns=["deviceId"])
    data = _load(frame).temperature
    assert data is not None
    assert data.cows[0].key == "1354"
    assert data.cows[0].label == "Cow 1354"


@pytest.mark.unit
def test_file_without_any_id_is_one_cow() -> None:
    frame = temperature_frame(steady_temperatures(0.5), animal=None).drop(columns=["deviceId"])
    data = _load(frame, name="barn.csv").temperature
    assert data is not None
    assert len(data.cows) == 1


@pytest.mark.unit
def test_several_devices_become_several_cows() -> None:
    first = temperature_frame(steady_temperatures(1), device="dev-a", animal="1")
    second = temperature_frame(steady_temperatures(1, level=39.2), device="dev-b", animal="2")
    mixed = pd.concat([second, first]).sample(frac=1.0, random_state=1)
    data = _load(mixed).temperature
    assert data is not None
    assert sorted(cow.key for cow in data.cows) == ["dev-a", "dev-b"]
    one = data.for_cow("dev-b")
    assert list(one.columns) == ["ts", "temperature"]
    assert one["ts"].is_monotonic_increasing
    assert one["temperature"].mean() == pytest.approx(39.2, abs=0.05)


@pytest.mark.unit
def test_bad_rows_are_skipped_with_warnings() -> None:
    frame = temperature_frame(steady_temperatures(0.5)).astype(str)
    frame.loc[3, "temperature"] = "n/a"
    frame.loc[4, "timestamp"] = "not a time"
    frame.loc[5, "temperature"] = "99.0"
    duplicated = pd.concat([frame, frame.iloc[[10]]])
    data = _load(duplicated).temperature
    assert data is not None
    assert len(data.readings) == len(frame) - 3
    joined = " ".join(data.warnings)
    assert "unreadable" in joined
    assert "outside 10–45 °C" in joined
    assert "repeated" in joined


@pytest.mark.unit
def test_timestamps_without_time_zone_are_read_as_utc_with_a_warning() -> None:
    frame = temperature_frame(steady_temperatures(0.2))
    frame["timestamp"] = frame["timestamp"].str.replace(".000Z", "", regex=False)
    data = _load(frame).temperature
    assert data is not None
    assert any("no time zone" in warning for warning in data.warnings)
    assert data.readings["ts"].iloc[0] == pd.Timestamp("2024-04-06 00:00", tz="UTC")


@pytest.mark.unit
def test_offset_style_timestamps_are_parsed() -> None:
    frame = temperature_frame(steady_temperatures(0.2))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).astype(str)
    data = _load(frame).temperature
    assert data is not None
    assert not data.warnings
    assert data.readings["ts"].iloc[1] == pd.Timestamp("2024-04-06 00:05", tz="UTC")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "empty"),
        (b"   \r\n  ", "empty"),
        (b"name,score\nalice,3\n", "doesn't look like"),
        (b"temperature,value\n38.5,1\n", "doesn't look like"),
    ],
)
def test_unrecognised_files_are_rejected(content: bytes, message: str) -> None:
    with pytest.raises(UnsupportedFileError, match=message):
        load_upload("x.csv", content)


@pytest.mark.unit
def test_temperature_file_with_no_usable_rows_is_rejected() -> None:
    with pytest.raises(UnsupportedFileError, match="No usable temperature readings"):
        load_upload("x.csv", b"deviceId,temperature,timestamp\nd1,abc,2024-04-06T00:00:00Z\n")


@pytest.mark.unit
def test_unknown_timestamp_format_is_refused() -> None:
    with pytest.raises(UnsupportedFileError, match="aren't in a format the app reads"):
        load_upload("x.csv", b"deviceId,temperature,timestamp\nd1,38.5,next tuesday\n")


@pytest.mark.unit
def test_meter_export_is_recognised_and_sorted_oldest_first() -> None:
    loaded = load_upload("2X.csv", meter_csv(hours=30))
    assert loaded.kind is FileKind.METER
    meter = loaded.meter
    assert meter is not None
    readings = meter.readings
    assert len(readings) == 30
    assert readings["read_time"].is_monotonic_increasing
    assert readings["read_time"].iloc[0] == pd.Timestamp("2024-04-06 00:10")
    assert readings["gallons"].sum() == pytest.approx(60.0)
    assert readings["battery_v"].iloc[0] == pytest.approx(3.603)


@pytest.mark.unit
def test_meter_blank_hours_count_as_zero() -> None:
    content = meter_csv(hours=3).decode("utf-8")
    lines = content.split("\r\n")
    fields = lines[1].split(",")
    fields[14] = ""
    fields[17] = ""
    fields[20] = ""
    fields[18] = "N/A"
    lines[1] = ",".join(fields)
    meter = load_upload("m.csv", "\r\n".join(lines).encode("utf-8")).meter
    assert meter is not None
    assert meter.readings["gallons"].iloc[-1] == 0.0
    assert meter.readings["events"].iloc[-1] == 0.0


@pytest.mark.integration
@pytest.mark.parametrize("folder", ["AbnormalData", "CalvingData", "DrinkingData"])
def test_every_delivered_temperature_file_loads_as_one_cow(
    data_dir: Path, load_data_file: Callable[[str], LoadedFile], folder: str
) -> None:
    files = sorted(path.name for path in (data_dir / folder).glob("*.csv") if path.name[0] != "2")
    assert files
    for name in files:
        loaded = load_data_file(f"{folder}/{name}")
        assert loaded.kind is FileKind.TEMPERATURE, name
        data = loaded.temperature
        assert data is not None
        assert len(data.cows) == 1, name
        assert 4.5 <= data.interval_minutes <= 5.5, name


@pytest.mark.integration
def test_file_with_undefined_id_column_uses_the_device(
    load_data_file: Callable[[str], LoadedFile],
) -> None:
    data = load_data_file("AbnormalData/1585_N.csv").temperature
    assert data is not None
    assert data.cows[0].device_id
    assert data.cows[0].animal_id is None


@pytest.mark.integration
@pytest.mark.parametrize(("name", "rows"), [("2D.csv", 357), ("2F.csv", 353), ("2d_417.csv", 166)])
def test_delivered_meter_files_load(
    load_data_file: Callable[[str], LoadedFile], name: str, rows: int
) -> None:
    loaded = load_data_file(f"DrinkingData/{name}")
    assert loaded.kind is FileKind.METER
    assert loaded.meter is not None
    assert len(loaded.meter.readings) == rows


@pytest.mark.unit
def test_separate_date_and_time_columns_are_combined() -> None:
    frame = temperature_frame(steady_temperatures(0.2))
    stamps = pd.to_datetime(frame.pop("timestamp"))
    frame["Date"] = stamps.dt.strftime("%Y-%m-%d")
    frame["Time"] = stamps.dt.strftime("%H:%M:%S")
    data = _load(frame).temperature
    assert data is not None
    assert data.readings["ts"].iloc[0] == pd.Timestamp("2024-04-06 00:00", tz="UTC")
    assert data.readings["ts"].iloc[-1].date() == pd.Timestamp("2024-04-06").date()
    assert len(data.readings) == len(frame)


@pytest.mark.unit
def test_day_first_dates_are_recognised() -> None:
    frame = temperature_frame(steady_temperatures(3), start="2024-04-12T00:00:00Z")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.strftime("%d/%m/%Y %H:%M")
    data = _load(frame).temperature
    assert data is not None
    assert data.readings["ts"].iloc[0] == pd.Timestamp("2024-04-12 00:00", tz="UTC")
    assert data.readings["ts"].iloc[-1].date() == pd.Timestamp("2024-04-14").date()


@pytest.mark.unit
def test_ambiguous_day_month_dates_are_refused() -> None:
    frame = temperature_frame(steady_temperatures(2), start="2024-04-01T00:00:00Z")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.strftime("%d/%m/%Y %H:%M")
    with pytest.raises(UnsupportedFileError, match="day/month or month/day"):
        _load(frame)


@pytest.mark.unit
def test_cow_left_without_readings_is_named() -> None:
    good = temperature_frame(steady_temperatures(1), device="dev-a", animal="1")
    broken = temperature_frame(np.zeros(288), device="dev-b", animal="2")
    data = _load(pd.concat([good, broken])).temperature
    assert data is not None
    assert [cow.key for cow in data.cows] == ["dev-a"]
    assert any("No usable readings for dev-b" in warning for warning in data.warnings)


@pytest.mark.unit
def test_meter_keeps_both_reports_in_the_repeated_daylight_saving_hour() -> None:
    lines = meter_csv(start="2024-11-03 00:10", hours=3, newest_first=False).decode().split("\r\n")
    # The 02:10 report becomes a second 01:10 report, as when clocks go back an hour.
    lines = [line.replace("11/3/2024 2:10", "11/3/2024 1:10") for line in lines]
    meter = load_upload("m.csv", "\r\n".join(lines).encode()).meter
    assert meter is not None
    assert len(meter.readings) == 3
    assert not meter.warnings


@pytest.mark.integration
@pytest.mark.parametrize(
    ("relative_path", "hinted"),
    [
        ("DrinkingData/1354_417_bouts.csv", True),
        ("DrinkingData/1354_417_raw.csv", False),
        ("AbnormalData/1572_Y.csv", False),  # a full export with no reading below 35 °C
    ],
)
def test_drinking_removed_copy_is_spotted_at_upload(
    load_data_file: Callable[[str], LoadedFile], relative_path: str, hinted: bool
) -> None:
    from cowmonitor.loaders import LIKELY_COPY_WARNING

    data = load_data_file(relative_path).temperature
    assert data is not None
    assert (LIKELY_COPY_WARNING in data.warnings) is hinted
