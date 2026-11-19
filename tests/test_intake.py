"""Water intake estimate: the heat-balance arithmetic, the water file, and the calibration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cowmonitor.drinking import DrinkingMarks, DrinkingSource
from cowmonitor.intake import (
    CALIBRATION,
    LITRES_PER_GALLON,
    REFERENCE_WATER_C,
    REWARM_LITRES_PER_HOUR,
    IntakeEstimate,
    estimate_intake,
)
from cowmonitor.intake_calibration import (
    CowFiles,
    Pen,
    PenDrinks,
    compare,
    evaluate,
    prepare,
    sweep,
)
from cowmonitor.loaders import (
    LoadedFile,
    MeterData,
    UnsupportedFileError,
    load_upload,
    load_water_temperature,
)
from cowmonitor.modules import IntakeOptions, run_module
from tests.helpers import meter_csv, steady_temperatures, temperature_frame, to_csv_bytes
from tools.calibrate_intake import SWEEP_C, load_project_pens

TZ = "America/New_York"
START = "2024-04-06T04:00:00Z"  # local midnight in New York, so each 288 readings is one day
PER_DAY = 288
LEVEL = 38.8
DIP = [34.0, 35.0, 36.5, 37.5, 38.2]  # 4.8 + 3.8 + 2.3 + 1.3 + 0.6 = 12.8 °C below normal
DIP_AREA = 12.8 * 5 / 60  # °C·h, at 5 minutes per reading
SAMPLE_WATER = Path(__file__).resolve().parents[1] / "sample_data"
SAMPLE_NAME = "synthetic_water_temperature_april2024.csv"


def _readings(frame: pd.DataFrame) -> pd.DataFrame:
    loaded = load_upload("cow.csv", to_csv_bytes(frame))
    assert loaded.temperature is not None
    return loaded.temperature.for_cow(loaded.temperature.cows[0].key)


def _marks(days: float = 1, starts: tuple[int, ...] = (100,)) -> DrinkingMarks:
    """A flat cow with one DIP at each start; its first 3 readings are marked as drinking."""
    temperatures = np.full(int(days * PER_DAY), LEVEL)
    drinking = np.zeros(len(temperatures), dtype=bool)
    for start in starts:
        temperatures[start : start + len(DIP)] = DIP
        drinking[start : start + 3] = True
    readings = _readings(temperature_frame(temperatures, start=START))
    return DrinkingMarks(readings.assign(drinking=drinking), DrinkingSource.COWMONITOR_COPY)


def _estimate(marks: DrinkingMarks, **options: object) -> IntakeEstimate:
    return estimate_intake(marks, interval_minutes=5.0, timezone=TZ, **options)  # type: ignore[arg-type]


def _water(times: list[str], temperatures: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"ts": pd.to_datetime(times, utc=True), "temperature": temperatures})


@pytest.mark.unit
def test_litres_follow_the_heat_balance() -> None:
    estimate = _estimate(_marks(), rewarm_litres_per_hour=100.0)
    drink = estimate.drinks.iloc[0]
    assert len(estimate.drinks) == 1
    assert drink["normal_c"] == pytest.approx(LEVEL)
    assert drink["coldest_c"] == pytest.approx(34.0)
    assert drink["minutes"] == pytest.approx(15.0)
    assert drink["area_ch"] == pytest.approx(DIP_AREA)
    assert drink["litres"] == pytest.approx(100.0 * DIP_AREA / (LEVEL - REFERENCE_WATER_C))
    day = estimate.daily.iloc[0]
    assert day["Drinks"] == 1
    assert day["Litres"] == pytest.approx(drink["litres"], abs=0.05)
    assert day["Gallons"] == pytest.approx(day["Litres"] / LITRES_PER_GALLON, abs=0.02)
    assert day["Note"] == ""
    assert estimate.coverage.tolist() == [100.0]
    assert estimate.water_c == REFERENCE_WATER_C
    assert not estimate.water_from_file
    assert not estimate.warnings


@pytest.mark.unit
def test_the_same_dip_means_less_water_when_the_water_is_colder() -> None:
    cold = _estimate(_marks(), default_water_c=5.0).drinks["litres"].iloc[0]
    warm = _estimate(_marks(), default_water_c=25.0).drinks["litres"].iloc[0]
    assert cold / warm == pytest.approx((LEVEL - 25.0) / (LEVEL - 5.0))


@pytest.mark.unit
def test_a_drink_window_ends_at_the_next_drink() -> None:
    together = _estimate(_marks(starts=(100, 112)))  # one hour apart
    apart = _estimate(_marks(starts=(100,)))
    assert len(together.drinks) == 2
    assert together.drinks["area_ch"].tolist() == pytest.approx([DIP_AREA, DIP_AREA])
    assert together.drinks["litres"].sum() == pytest.approx(2 * apart.drinks["litres"].iloc[0])


@pytest.mark.unit
def test_a_drink_just_before_a_gap_counts_only_the_readings_there_are() -> None:
    marks = _marks(starts=(100,))
    kept = DrinkingMarks(marks.readings.drop(index=range(103, 130)), marks.source)
    drink = _estimate(kept).drinks.iloc[0]
    assert drink["area_ch"] == pytest.approx((4.8 + 3.8 + 2.3) * 5 / 60)


@pytest.mark.unit
def test_each_drink_uses_the_nearest_water_reading_within_two_hours() -> None:
    marks = _marks(starts=(100, 200))  # 08:20 and 16:40 local time
    water = _water(["2024-04-06T12:00:00Z", "2024-04-06T13:30:00Z"], [8.0, 30.0])
    estimate = _estimate(marks, water=water, default_water_c=12.0)
    assert estimate.drinks["water_c"].tolist() == [8.0, 12.0]
    assert estimate.drinks["water_from_file"].tolist() == [True, False]
    assert estimate.water_c == 8.0  # the drinks matched in the file, not the setting
    assert estimate.drinks_from_file == 1
    assert any("no reading within 2 hours of 1 of 2 drinks" in w for w in estimate.warnings)


@pytest.mark.unit
def test_water_close_to_body_temperature_is_flagged_or_not_estimated() -> None:
    hot = _estimate(_marks(), default_water_c=37.0)
    assert hot.drinks["litres"].isna().all()
    assert any(w.startswith("No estimate for 1 drink:") for w in hot.warnings)
    assert hot.daily["Litres"].isna().all()
    assert hot.daily["Note"].tolist() == ["no estimate: 1 drink could not be estimated"]
    warm = _estimate(_marks(), default_water_c=30.0)
    assert warm.drinks["litres"].notna().all()
    assert any("less reliable" in w for w in warm.warnings)


@pytest.mark.unit
def test_a_day_with_a_drink_that_has_no_estimate_gets_no_total() -> None:
    marks = _marks(starts=(100, 200))
    water = _water(["2024-04-06T20:40:00Z"], [37.5])  # only the second drink, too warm
    estimate = _estimate(marks, water=water)
    day = estimate.daily.iloc[0]
    assert estimate.drinks["litres"].isna().tolist() == [False, True]
    assert day["Litres in drinks found"] == pytest.approx(
        estimate.drinks["litres"].iloc[0], abs=0.05
    )
    assert np.isnan(day["Litres"])
    assert day["Note"] == "no estimate: 1 drink could not be estimated"


@pytest.mark.unit
def test_days_with_gaps_are_scaled_up_and_sparse_days_get_no_estimate() -> None:
    marks = _marks(days=3, starts=(100, PER_DAY + 150, 2 * PER_DAY + 10))
    position = np.arange(len(marks.readings))
    gap = (position >= PER_DAY) & (position < PER_DAY + 72)  # 6 hours missing on day 2
    sparse = position >= 2 * PER_DAY + 72  # day 3 keeps only its first 6 hours
    readings = marks.readings[~gap & ~sparse].reset_index(drop=True)
    estimate = _estimate(DrinkingMarks(readings, marks.source))
    daily = estimate.daily
    assert daily["Coverage %"].tolist() == pytest.approx([100.0, 75.0, 25.0])
    found, litres = daily["Litres in drinks found"], daily["Litres"]
    assert litres[0] == pytest.approx(found[0], abs=0.05)
    assert litres[1] == pytest.approx(found[1] / 0.75, abs=0.1)
    assert np.isnan(litres[2])
    assert daily["Note"].tolist() == ["", "", "no estimate: readings cover under 50% of the day"]


@pytest.mark.unit
def test_no_drinks_means_zero_litres_and_a_warning() -> None:
    estimate = _estimate(_marks(starts=()))
    assert estimate.drinks.empty
    assert estimate.daily["Litres"].tolist() == [0.0]
    assert estimate.water_c == REFERENCE_WATER_C
    assert estimate.warnings[0].startswith("No drinking dips were found")


@pytest.mark.unit
def test_water_file_keeps_readings_a_cow_file_would_drop() -> None:
    frame = temperature_frame(np.full(96, 4.0), freq="15min", device="trough-1", animal=None)
    water = load_water_temperature("water.csv", to_csv_bytes(frame))
    assert water.readings["temperature"].tolist() == [4.0] * 96
    assert [cow.device_id for cow in water.cows] == ["trough-1"]
    assert not water.warnings


@pytest.mark.unit
def test_warm_water_is_still_water() -> None:
    frame = temperature_frame(np.full(96, 32.0), freq="15min", animal=None)
    assert (
        load_water_temperature("sunny.csv", to_csv_bytes(frame)).readings["temperature"].max() == 32
    )
    with pytest.raises(UnsupportedFileError, match="look like water temperatures"):
        load_upload("sunny.csv", to_csv_bytes(frame))


@pytest.mark.unit
@pytest.mark.parametrize("level", [4.0, 18.0])
def test_water_file_uploaded_as_the_cow_file_is_refused(level: float) -> None:
    frame = temperature_frame(np.full(96, level), freq="15min", animal=None)
    with pytest.raises(UnsupportedFileError, match="look like water temperatures"):
        load_upload("water.csv", to_csv_bytes(frame))


@pytest.mark.unit
def test_bolus_that_logged_before_going_into_the_cow_still_loads() -> None:
    in_a_box = np.full(4 * PER_DAY, 22.0)
    temperatures = np.concatenate([in_a_box, steady_temperatures(3)])
    loaded = load_upload("bolus.csv", to_csv_bytes(temperature_frame(temperatures)))
    assert loaded.temperature is not None
    assert len(loaded.temperature.readings) == len(temperatures)


@pytest.mark.unit
def test_times_before_the_bolus_went_in_get_no_estimate() -> None:
    temperatures = np.concatenate([np.full(2 * PER_DAY, 22.0), np.full(PER_DAY, LEVEL)])
    drinking = np.zeros(len(temperatures), dtype=bool)
    for start in (2 * PER_DAY + 100, 2 * PER_DAY + 200):
        temperatures[start : start + len(DIP)] = DIP
        drinking[start : start + 3] = True
    readings = _readings(temperature_frame(temperatures, start=START))
    marks = DrinkingMarks(readings.assign(drinking=drinking), DrinkingSource.COWMONITOR_COPY)
    estimate = _estimate(marks)
    daily = estimate.daily
    assert len(estimate.drinks) == 2
    assert daily["Litres"].isna().tolist() == [True, True, False]
    assert daily["Coverage %"].iloc[0] == 0.0
    assert daily["Note"].iloc[0] == "no estimate: readings cover under 50% of the day"
    assert any(w.startswith("For 4") and "not in the cow" in w for w in estimate.warnings)


@pytest.mark.unit
def test_herd_file_with_boluses_on_a_shelf_loads_with_a_warning() -> None:
    cow = temperature_frame(steady_temperatures(1), device="in-cow", animal="1")
    shelf = [
        temperature_frame(np.full(PER_DAY, 21.0), device=f"shelf-{i}", animal=None) for i in (1, 2)
    ]
    loaded = load_upload("herd.csv", to_csv_bytes(pd.concat([cow, *shelf])))
    assert loaded.temperature is not None
    assert len(loaded.temperature.cows) == 3
    assert any(
        w.startswith("The readings of shelf-1, shelf-2 look like water or air temperatures")
        for w in loaded.temperature.warnings
    )
    with pytest.raises(UnsupportedFileError, match="look like water temperatures"):
        load_upload("shelf.csv", to_csv_bytes(pd.concat(shelf)))


@pytest.mark.unit
def test_water_timestamps_without_a_time_zone_are_read_as_farm_time() -> None:
    content = b"temperature,timestamp\n10.0,2024-04-06 08:00\n10.5,2024-04-06 08:15\n"
    water = load_water_temperature("logger.csv", content, TZ)
    assert water.readings["ts"].iloc[0] == pd.Timestamp("2024-04-06T12:00:00Z")
    assert water.warnings == (
        "The timestamps have no time zone, so they were read as farm time (America/New_York).",
    )
    as_utc = load_water_temperature("logger.csv", content)
    assert as_utc.readings["ts"].iloc[0] == pd.Timestamp("2024-04-06T08:00:00Z")


@pytest.mark.unit
def test_wrong_files_in_the_water_box_are_refused() -> None:
    cow = temperature_frame(np.full(96, LEVEL), animal=None)
    with pytest.raises(UnsupportedFileError, match="rumen temperature"):
        load_water_temperature("cow.csv", to_csv_bytes(cow))
    with pytest.raises(UnsupportedFileError, match="water-meter export"):
        load_water_temperature("2D.csv", meter_csv())
    with pytest.raises(UnsupportedFileError, match="needs a timestamp column"):
        load_water_temperature("notes.csv", b"name,score\nalice,3\n")
    with pytest.raises(UnsupportedFileError, match="empty"):
        load_water_temperature("empty.csv", b"  \n")
    two = pd.concat(
        [
            temperature_frame(np.full(10, 12.0), device="trough-1", animal=None),
            temperature_frame(np.full(10, 14.0), device="trough-2", animal=None),
        ]
    )
    with pytest.raises(UnsupportedFileError, match="2 sensors"):
        load_water_temperature("two.csv", to_csv_bytes(two))


@pytest.mark.unit
def test_sample_water_file_covers_the_project_weeks_at_the_reference_temperature() -> None:
    water = load_water_temperature(SAMPLE_NAME, (SAMPLE_WATER / SAMPLE_NAME).read_bytes())
    readings = water.readings
    assert readings["temperature"].unique().tolist() == [REFERENCE_WATER_C]
    assert readings["ts"].min() <= pd.Timestamp("2024-04-06T04:00Z")
    assert readings["ts"].max() >= pd.Timestamp("2024-04-21T04:00Z")
    assert water.interval_minutes == pytest.approx(15.0)


@pytest.mark.unit
def test_calibration_copes_with_the_hour_repeated_when_clocks_go_back() -> None:
    temperatures = steady_temperatures(2)
    for start in range(40, len(temperatures) - 10, 36):
        temperatures[start : start + 4] = [34.0, 35.5, 36.8, 37.9]
    cow = load_upload(
        "cow.csv", to_csv_bytes(temperature_frame(temperatures, start="2024-11-02T04:00Z"))
    )
    assert cow.temperature is not None
    reads = list(pd.date_range("2024-11-02 00:20", "2024-11-03 23:20", freq="1h"))
    reads.insert(26, pd.Timestamp("2024-11-03 01:20"))  # 01:20 EDT, then 01:20 EST
    meter = MeterData(pd.DataFrame({"read_time": reads, "gallons": 2.0}))
    prepared = prepare([Pen("test", meter, (CowFiles(cow.temperature),))])
    comparison = compare(prepared, REFERENCE_WATER_C)
    # 3 Nov is the export's last day: no report holds the water of its last hour, so it is
    # left out; 1 Nov has a single report.
    assert comparison.daily["date"].astype(str).tolist() == ["2024-11-02"]
    assert len(comparison.hourly) == len(reads) - 1
    assert comparison.hourly["metered_litres"].max() == pytest.approx(4 * LITRES_PER_GALLON)


# --- Calibration on the project's water meters ---------------------------------------------


@pytest.fixture(scope="module")
def project_pens(data_dir: Path) -> dict[bool, list[PenDrinks]]:
    pens = load_project_pens(data_dir / "DrinkingData")
    return {use_copy: prepare(pens, use_copy=use_copy) for use_copy in (True, False)}


@pytest.mark.integration
@pytest.mark.parametrize(("use_copy", "tolerance"), [(True, 0.002), (False, 0.01)])
def test_stored_rewarm_rate_is_the_calibrated_one(
    project_pens: dict[bool, list[PenDrinks]], use_copy: bool, tolerance: float
) -> None:
    result = evaluate(compare(project_pens[use_copy], REFERENCE_WATER_C))
    assert result.rewarm_litres_per_hour == pytest.approx(REWARM_LITRES_PER_HOUR, rel=tolerance)


@pytest.mark.integration
def test_stored_calibration_figures_are_the_measured_ones(
    project_pens: dict[bool, list[PenDrinks]],
) -> None:
    comparison = compare(project_pens[True], REFERENCE_WATER_C)
    result = evaluate(comparison, REWARM_LITRES_PER_HOUR)
    assert sum(len(pen.cows) for pen in project_pens[True]) == CALIBRATION.cow_count
    assert len(comparison.daily) == CALIBRATION.days
    assert CALIBRATION.sweep_c == (min(SWEEP_C), max(SWEEP_C))
    assert result.daily_error_gal == pytest.approx(CALIBRATION.daily_error_gal, abs=0.01)
    assert result.daily_error_pct == pytest.approx(CALIBRATION.daily_error_pct, abs=0.1)
    cross = sorted(error for _, error in result.cross_error_gal)
    assert cross == pytest.approx(list(CALIBRATION.cross_error_gal), abs=0.01)
    assert result.hourly_r == pytest.approx(CALIBRATION.hourly_r, abs=0.001)
    errors = sweep(project_pens[True], SWEEP_C)["Daily error gal"]
    assert (errors.min(), errors.max()) == pytest.approx(CALIBRATION.sweep_error_gal, abs=0.01)
    assert len(errors) == 31


@pytest.mark.integration
@pytest.mark.parametrize("use_copy", [True, False])
def test_estimates_match_the_meters(
    project_pens: dict[bool, list[PenDrinks]], use_copy: bool
) -> None:
    result = evaluate(compare(project_pens[use_copy], REFERENCE_WATER_C), REWARM_LITRES_PER_HOUR)
    assert result.daily_error_gal <= 4.7
    assert result.daily_error_pct <= 9.6
    assert all(abs(error) <= 2.0 for _, error in result.total_error_pct)
    assert all(error <= 4.8 for _, error in result.cross_error_gal)
    assert result.leave_one_day_out_gal < result.guess_average_gal
    assert result.hourly_r >= 0.83


@pytest.mark.integration
def test_the_meters_cannot_tell_the_water_temperature(
    project_pens: dict[bool, list[PenDrinks]],
) -> None:
    table = sweep(project_pens[True], [0.0, REFERENCE_WATER_C, 30.0])
    errors = table["Daily error gal"]
    assert errors.max() - errors.min() <= 0.02 * errors.mean()
    assert table["Hourly r"].min() >= 0.83
    # Only K changes: colder assumed water needs a faster re-warming rate for the same dips.
    assert table["K litres/hour"].is_monotonic_decreasing


@pytest.mark.integration
def test_calibration_tool_reports_the_rate_and_the_sweep(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import tools.calibrate_intake as tool

    monkeypatch.setattr(tool, "SWEEP_C", (0.0, 30.0))
    monkeypatch.setattr("sys.argv", ["calibrate_intake", str(data_dir / "DrinkingData")])
    with caplog.at_level(logging.INFO, logger="calibrate_intake"):
        tool.main()
    report = caplog.text
    reference = f"Refitted at {REFERENCE_WATER_C:g} °C"
    assert f"{reference}, with the drinking-removed copy: K = 133.30" in report
    assert f"{reference}, estimated dips: K = 132.64" in report
    assert report.count("Assumed water temperature sweep") == 2


@pytest.mark.unit
def test_calibration_needs_one_cow_per_file() -> None:
    herd = pd.concat(
        [
            temperature_frame(steady_temperatures(1), device="dev-a", animal="1"),
            temperature_frame(steady_temperatures(1), device="dev-b", animal="2"),
        ]
    )
    loaded = load_upload("herd.csv", to_csv_bytes(herd))
    assert loaded.temperature is not None
    meter = MeterData(
        pd.DataFrame({"read_time": [pd.Timestamp("2024-04-06 01:00")], "gallons": 1.0})
    )
    with pytest.raises(ValueError, match="must hold one cow; this one holds 2"):
        prepare([Pen("herd", meter, (CowFiles(loaded.temperature),))])


@pytest.mark.unit
def test_module_says_when_only_some_drinks_matched_the_water_file() -> None:
    temperatures = np.full(PER_DAY, LEVEL)
    for start in (100, 200):
        temperatures[start : start + len(DIP)] = DIP
    loaded = load_upload("cow.csv", to_csv_bytes(temperature_frame(temperatures, start=START)))
    assert loaded.temperature is not None
    frame = temperature_frame(np.full(4, 8.0), start="2024-04-06T11:30:00Z", freq="15min")
    water = load_water_temperature("water.csv", to_csv_bytes(frame.drop(columns="Animal Id")))
    options = IntakeOptions(cow=loaded.temperature.cows[0].key, water=water)
    result = run_module(loaded, "intake", options)
    label = next(m.value for m in result.metrics if m.label == "Water temperature")
    assert label == "8.0 °C (file, 1 of 2 drinks)"
    assert next(m.value for m in result.metrics if m.label == "Total").endswith("over 1 day")


def _cow_1354(load: Callable[[str], LoadedFile]) -> tuple[LoadedFile, IntakeOptions]:
    raw = load("DrinkingData/1354_raw.csv")
    copy = load("DrinkingData/1354_bouts.csv").temperature
    assert raw.temperature is not None
    return raw, IntakeOptions(cow=raw.temperature.cows[0].key, copy=copy)


@pytest.mark.integration
def test_module_with_the_sample_water_file_matches_the_setting(
    load_data_file: Callable[[str], LoadedFile],
) -> None:
    raw, options = _cow_1354(load_data_file)
    water = load_water_temperature(SAMPLE_NAME, (SAMPLE_WATER / SAMPLE_NAME).read_bytes())
    with_file = run_module(raw, "intake", IntakeOptions(**{**vars(options), "water": water}))
    setting = run_module(raw, "intake", options)
    assert (
        with_file.tables[0].frame["Litres"].tolist() == setting.tables[0].frame["Litres"].tolist()
    )
    assert len(with_file.charts) == 3
    assert len(setting.charts) == 2


@pytest.mark.integration
def test_a_colder_water_file_rescales_every_drink(
    load_data_file: Callable[[str], LoadedFile],
) -> None:
    raw, options = _cow_1354(load_data_file)
    sample = pd.read_csv(SAMPLE_WATER / SAMPLE_NAME).assign(temperature=8.0)
    water = load_water_temperature("cold.csv", to_csv_bytes(sample, bom=False))
    cold = run_module(raw, "intake", IntakeOptions(**{**vars(options), "water": water}))
    setting = run_module(raw, "intake", options)
    cold_drinks, setting_drinks = cold.tables[1].frame, setting.tables[1].frame
    assert cold_drinks["Water from"].eq("file").all()
    expected = (setting_drinks["Normal °C"] - REFERENCE_WATER_C) / (cold_drinks["Normal °C"] - 8.0)
    ratio = cold_drinks["Litres"] / setting_drinks["Litres"]
    assert ratio.to_numpy() == pytest.approx(expected.to_numpy(), rel=0.01)
    assert next(m.value for m in cold.metrics if m.label == "Water temperature") == (
        "8.0 °C (median, from file)"
    )
