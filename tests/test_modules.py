"""Each module end to end: from a loaded file to the result the page shows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cowmonitor.intake import REFERENCE_WATER_C
from cowmonitor.loaders import FileKind, LoadedFile, UnsupportedFileError, load_upload
from cowmonitor.modules import (
    MODULES,
    AbnormalOptions,
    CalvingOptions,
    DrinkingOptions,
    IntakeOptions,
    WaterOptions,
    get_module,
    modules_for,
    run_module,
)
from cowmonitor.results import ModuleResult, table_to_csv
from tests.helpers import steady_temperatures, temperature_frame, to_csv_bytes


def _metric(result: ModuleResult, label: str) -> str:
    return next(metric.value for metric in result.metrics if metric.label == label)


@pytest.mark.unit
def test_registry_offers_modules_by_file_kind() -> None:
    assert [m.key for m in modules_for(FileKind.TEMPERATURE)] == [
        "abnormal",
        "calving",
        "drinking",
        "intake",
    ]
    assert [m.key for m in modules_for(FileKind.METER)] == ["water"]
    assert get_module("calving").experimental
    assert get_module("intake").experimental
    assert len(MODULES) == 5
    with pytest.raises(KeyError):
        get_module("overview")


@pytest.mark.unit
def test_wrong_file_kind_wrong_options_and_bad_time_zone_are_rejected() -> None:
    loaded = load_upload("t.csv", to_csv_bytes(temperature_frame(steady_temperatures(2))))
    cow = loaded.temperature.cows[0].key if loaded.temperature else ""
    with pytest.raises(UnsupportedFileError, match="needs a water-meter export"):
        run_module(loaded, "water", WaterOptions())
    with pytest.raises(TypeError):
        run_module(loaded, "abnormal", CalvingOptions(cow=cow))
    with pytest.raises(ValueError, match="Unknown time zone"):
        run_module(loaded, "abnormal", AbnormalOptions(cow=cow, timezone="Nowhere/Land"))
    with pytest.raises(ValueError, match="Water temperature must be"):
        run_module(loaded, "intake", IntakeOptions(cow=cow, water_c=60.0))


@pytest.mark.unit
def test_drinking_removed_looking_file_gets_a_warning() -> None:
    loaded = load_upload("t.csv", to_csv_bytes(temperature_frame(steady_temperatures(3))))
    assert loaded.temperature is not None
    result = run_module(loaded, "abnormal", AbnormalOptions(cow=loaded.temperature.cows[0].key))
    assert _metric(result, "Verdict") == "Abnormal"
    assert any("drinking-removed" in warning for warning in result.warnings)


@pytest.mark.unit
def test_csv_download_escapes_formula_like_text() -> None:
    frame = pd.DataFrame({"Cow": ["=cmd()", "+1", "1354"], "Gallons": [-1.5, 2.0, 3.0]})
    text = table_to_csv(frame).decode("utf-8")
    assert "'=cmd()" in text
    assert "'+1" in text
    assert "-1.5" in text  # numbers are left alone


@pytest.mark.integration
@pytest.mark.parametrize(("name", "verdict"), [("1431_Y.csv", "Abnormal"), ("101_N.csv", "Normal")])
def test_abnormal_module(
    load_data_file: Callable[[str], LoadedFile], name: str, verdict: str
) -> None:
    loaded = load_data_file(f"AbnormalData/{name}")
    assert loaded.temperature is not None
    result = run_module(loaded, "abnormal", AbnormalOptions(cow=loaded.temperature.cows[0].key))
    assert _metric(result, "Verdict") == verdict
    assert _metric(result, "Limits").startswith("Per day")
    assert len(result.charts) == 2
    caption = result.charts[1].caption
    assert caption is not None and caption.startswith("Dotted lines")


@pytest.mark.unit
def test_module_options_offer_no_choice_of_method() -> None:
    assert [f.name for f in fields(AbnormalOptions)] == ["cow", "timezone"]
    assert [f.name for f in fields(CalvingOptions)] == ["cow", "drinking_removed", "timezone"]


@pytest.mark.integration
def test_abnormal_module_covers_every_cow_in_a_herd_file(data_dir: Path) -> None:
    first = (data_dir / "AbnormalData/1431_Y.csv").read_bytes().rstrip(b"\r\n")
    second_rows = (data_dir / "AbnormalData/101_N.csv").read_bytes().splitlines()[1:]
    content = first + b"\n" + b"\n".join(second_rows)
    loaded = load_upload("herd.csv", content)
    assert loaded.temperature is not None
    assert len(loaded.temperature.cows) == 2
    result = run_module(loaded, "abnormal", AbnormalOptions(cow=loaded.temperature.cows[0].key))
    by_cow = result.tables[0].frame
    assert sorted(by_cow["Verdict"]) == ["Abnormal", "Normal"]
    assert _metric(result, "Abnormal cows in file") == "1 of 2"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("name", "rises_again"),
    [("1734_Y.csv", "Not checked (too little data after the drop)"), ("1597_Y.csv", "Yes")],
)
def test_calving_module(
    load_data_file: Callable[[str], LoadedFile], name: str, rises_again: str
) -> None:
    loaded = load_data_file(f"CalvingData/{name}")
    assert loaded.temperature is not None
    result = run_module(loaded, "calving", CalvingOptions(cow=loaded.temperature.cows[0].key))
    assert _metric(result, "Calving-like drop") == "Yes"
    assert _metric(result, "First drop") != "–"
    assert _metric(result, "Temperature rises again") == rises_again
    assert any(w.startswith("Experimental. On the project's 16") for w in result.warnings)
    assert not any("As delivered" in w for w in result.warnings)
    assert result.subtitle.startswith("Half-days are")
    flags = result.tables[1].frame["Flag"]
    assert flags.str.contains("calving-like drop").any()
    assert flags.str.contains("rises again after this").any() == (rises_again == "Yes")


@pytest.mark.integration
def test_drinking_module_with_and_without_the_copy(
    load_data_file: Callable[[str], LoadedFile],
) -> None:
    raw = load_data_file("DrinkingData/1354_raw.csv")
    copy = load_data_file("DrinkingData/1354_bouts.csv")
    other_copy = load_data_file("DrinkingData/1568_bouts.csv")
    assert raw.temperature is not None and copy.temperature is not None
    cow = raw.temperature.cows[0].key
    exact = run_module(raw, "drinking", DrinkingOptions(cow=cow, copy=copy.temperature))
    estimated = run_module(raw, "drinking", DrinkingOptions(cow=cow))
    assert _metric(exact, "Source") == "CowMonitor's drinking-removed file"
    assert _metric(estimated, "Source") == "Estimated"
    assert len(exact.tables[0].frame) == 15
    assert 7 <= float(_metric(exact, "Drinking dips per day")) <= 13
    with pytest.raises(UnsupportedFileError, match="no data for"):
        run_module(raw, "drinking", DrinkingOptions(cow=cow, copy=other_copy.temperature))


@pytest.mark.integration
def test_water_module(load_data_file: Callable[[str], LoadedFile]) -> None:
    loaded = load_data_file("DrinkingData/2D.csv")
    result = run_module(loaded, "water", WaterOptions(meter_name="2D", cows_on_meter="1354, 1568"))
    assert result.title == "Daily water use: 2D (cows 1354, 1568)"
    assert _metric(result, "Total gallons") == "674.1"
    assert _metric(result, "Highest day") == "53.9 gal (19 Apr)"
    assert len(result.tables[0].frame) == 15
    assert any("Missed hourly reports: 19" in note for note in result.notes)


def _herd(*cows: tuple[str, np.ndarray]) -> LoadedFile:
    frames = [temperature_frame(temps, device=name, animal=name[-1]) for name, temps in cows]
    return load_upload("herd.csv", to_csv_bytes(pd.concat(frames)))


@pytest.mark.unit
def test_every_temperature_module_handles_a_herd_file() -> None:
    from tests.test_calving import PER_HALF_DAY

    steady = steady_temperatures(6)
    hot = np.full(6 * 2 * PER_HALF_DAY, 41.0)
    loaded = _herd(("dev-1", steady), ("dev-2", hot))
    assert loaded.temperature is not None
    assert len(loaded.temperature.cows) == 2
    calving = run_module(loaded, "calving", CalvingOptions(cow="dev-2"))
    assert _metric(calving, "Calving-like drop") == "Not enough data"
    assert len(calving.tables[0].frame) == 2
    abnormal = run_module(loaded, "abnormal", AbnormalOptions(cow="dev-2"))
    assert len(abnormal.tables[0].frame) == 2
    drinking = run_module(loaded, "drinking", DrinkingOptions(cow="dev-1"))
    assert _metric(drinking, "Coldest drinking reading") == "–"
    intake = run_module(loaded, "intake", IntakeOptions(cow="dev-1"))
    assert _metric(intake, "Average per day") == "0.0 gal (0 L)"
    assert any(w.startswith("No drinking dips were found") for w in intake.warnings)


# Notes the water intake page doesn't show as warning boxes: the same facts are in its
# metrics ("Drinking readings from"), chart caption and "How this is calculated" notes.
_INTAKE_HIDDEN = ("Experimental.", "Drinking readings are estimated", "Days with gaps")


@pytest.mark.unit
def test_intake_shows_no_boxes_for_calibration_estimated_drinking_or_gappy_days() -> None:
    # 4 hours missing on 7 April (local time), so that day is 83% covered and scaled up.
    frame = temperature_frame(steady_temperatures(3)).drop(index=range(408, 456))
    loaded = load_upload("gappy.csv", to_csv_bytes(frame))
    assert loaded.temperature is not None
    cow = loaded.temperature.cows[0].key
    intake = run_module(loaded, "intake", IntakeOptions(cow=cow))
    drinking = run_module(loaded, "drinking", DrinkingOptions(cow=cow))
    assert intake.tables[0].frame["Coverage %"].between(50, 99).any()
    assert not any(w.startswith(_INTAKE_HIDDEN) for w in intake.warnings)
    # Still shown: why a day has no estimate, and the drinking summary's own note.
    assert any(w.startswith("No estimate for days under 50%") for w in intake.warnings)
    assert any(w.startswith("Drinking readings are estimated") for w in drinking.warnings)


@pytest.mark.integration
def test_intake_module_for_one_cow(load_data_file: Callable[[str], LoadedFile]) -> None:
    raw = load_data_file("DrinkingData/1354_raw.csv")
    copy = load_data_file("DrinkingData/1354_bouts.csv")
    assert raw.temperature is not None and copy.temperature is not None
    cow = raw.temperature.cows[0].key
    exact = run_module(raw, "intake", IntakeOptions(cow=cow, copy=copy.temperature))
    estimated = run_module(raw, "intake", IntakeOptions(cow=cow))
    warmer = run_module(
        raw, "intake", IntakeOptions(cow=cow, copy=copy.temperature, water_c=REFERENCE_WATER_C + 6)
    )
    assert exact.title == "Water intake estimate: Cow 1354 (device 281b0400002004e2)"
    assert _metric(exact, "Average per day") == "24.6 gal (93 L)"
    assert _metric(exact, "Water temperature") == f"{REFERENCE_WATER_C:.1f} °C (setting)"
    assert _metric(exact, "Drinking readings from") == "CowMonitor's drinking-removed file"
    assert _metric(estimated, "Drinking readings from") == "Estimated"
    assert 23.0 <= float(_metric(estimated, "Average per day").split()[0]) <= 26.0
    assert _metric(warmer, "Average per day") == "32.8 gal (124 L)"
    assert not any(w.startswith(_INTAKE_HIDDEN) for w in exact.warnings + estimated.warnings)
    daily, drinks = exact.tables[0].frame, exact.tables[1].frame
    assert daily["Coverage %"].iloc[0] == pytest.approx(87, abs=1)  # 6 April: scaled, no box
    assert len(daily) == 15
    assert daily["Drinks"].sum() == len(drinks)
    assert drinks["Water from"].eq("setting").all()
    assert [chart.title for chart in exact.charts] == [
        "Estimated gallons per day",
        "Temperature and the cow's normal",
    ]


@pytest.mark.unit
def test_abnormal_check_warns_about_gaps() -> None:
    temperatures = steady_temperatures(6)
    frame = temperature_frame(temperatures)
    holed = frame.drop(index=range(288, 288 * 3))
    loaded = load_upload("gappy.csv", to_csv_bytes(holed))
    assert loaded.temperature is not None
    result = run_module(loaded, "abnormal", AbnormalOptions(cow=loaded.temperature.cows[0].key))
    assert any("Gaps in the data" in warning for warning in result.warnings)
