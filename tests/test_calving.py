"""Model 2 (calving alert): parity with the 2024 notebook, and the fixed version's behaviour."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cowmonitor.calving import CalvingVersion, check_calving
from cowmonitor.loaders import LoadedFile, load_upload
from tests.helpers import temperature_frame, to_csv_bytes

PER_HALF_DAY = 144  # 12 hours of 5-minute readings


def _readings(levels: list[float], skip_half_days: tuple[int, ...] = ()) -> pd.DataFrame:
    """One cow whose half-day medians follow ``levels``; listed half-days have no readings."""
    rng = np.random.default_rng(7)
    temperatures = np.concatenate([level + rng.normal(0, 0.05, PER_HALF_DAY) for level in levels])
    frame = temperature_frame(temperatures)
    keep = np.ones(len(frame), dtype=bool)
    for half_day in skip_half_days:
        keep[half_day * PER_HALF_DAY : (half_day + 1) * PER_HALF_DAY] = False
    loaded = load_upload("x.csv", to_csv_bytes(frame[keep]))
    assert loaded.temperature is not None
    return loaded.temperature.for_cow(loaded.temperature.cows[0].key)


@pytest.mark.unit
def test_step_drop_is_found_at_the_right_half_day() -> None:
    levels = [38.8] * 10 + [37.9] * 2 + [38.8] * 6
    result = check_calving(_readings(levels))
    assert result.enough_data
    assert result.calving
    first = result.first_calving
    assert first is not None
    assert first.start == pd.Timestamp("2024-04-11 00:00", tz="UTC")
    assert first.change == pytest.approx(-0.9, abs=0.05)
    assert result.recovered
    assert result.recovery_events[0].start >= first.start


@pytest.mark.unit
def test_steady_temperature_gives_no_alert() -> None:
    result = check_calving(_readings([38.8] * 14))
    assert result.enough_data
    assert not result.calving
    assert not result.recovered


@pytest.mark.unit
def test_missing_half_day_is_skipped_not_read_as_zero() -> None:
    levels = [38.8] * 16
    readings = _readings(levels, skip_half_days=(9,))
    fixed = check_calving(readings, CalvingVersion.FIXED)
    as_delivered = check_calving(readings, CalvingVersion.AS_DELIVERED)
    assert not fixed.calving
    assert as_delivered.calving  # the notebook fills the gap with 0 °C and sees a "drop"
    assert as_delivered.half_days["filled_with_zero"].any()


@pytest.mark.unit
def test_recovery_is_only_looked_for_after_the_drop() -> None:
    # A slow dip and rise early on (never a 0.5 °C step), then a drop that does not recover.
    levels = [38.8] * 5 + [38.65, 38.5, 38.35, 38.2] + [38.8] * 5 + [37.9] * 4
    result = check_calving(_readings(levels))
    assert result.calving
    assert result.recovery_checked
    assert not result.recovered


@pytest.mark.unit
def test_too_little_data_is_not_judged() -> None:
    result = check_calving(_readings([38.8] * 4))
    assert not result.enough_data
    assert not result.calving


@pytest.mark.unit
def test_drinking_removed_file_skips_the_temperature_filter() -> None:
    readings = _readings([36.6] * 8)
    assert not check_calving(readings).enough_data
    assert check_calving(readings, drinking_removed=True).enough_data


@pytest.mark.unit
def test_last_half_day_is_left_out_when_incomplete() -> None:
    readings = _readings([38.8] * 8)
    partial = readings[readings["ts"] < "2024-04-09 18:00Z"].reset_index(drop=True)
    table = check_calving(partial).half_days
    assert table["note"].iloc[-1] == "last half-day, incomplete"
    assert not table["used"].iloc[-1]
    assert table["used"].iloc[:-1].all()


def _calving_files(data_dir: Path) -> list[str]:
    return sorted(path.name for path in (data_dir / "CalvingData").glob("*.csv"))


def _run_all(
    data_dir: Path, load_data_file: Callable[[str], LoadedFile], version: CalvingVersion
) -> pd.DataFrame:
    rows = []
    for name in _calving_files(data_dir):
        data = load_data_file(f"CalvingData/{name}").temperature
        assert data is not None
        result = check_calving(
            data.for_cow(data.cows[0].key), version, interval_minutes=data.interval_minutes
        )
        rows.append(
            {
                "id": name.split("_")[0],
                "truth": int("_Y" in name),
                "prediction": int(result.calving),
                "recovered": int(result.recovered),
            }
        )
    return pd.DataFrame(rows).set_index("id")


@pytest.mark.integration
def test_as_delivered_version_reproduces_the_2024_notebook(
    data_dir: Path, fixtures_dir: Path, load_data_file: Callable[[str], LoadedFile]
) -> None:
    expected = pd.read_csv(fixtures_dir / "notebook_calving_2024.csv", dtype={"id": str})
    expected = expected.set_index("id")
    actual = _run_all(data_dir, load_data_file, CalvingVersion.AS_DELIVERED)
    assert len(actual) == 16
    assert (actual["prediction"] == expected.loc[actual.index, "Prediction"]).all()
    assert (actual["recovered"] == expected.loc[actual.index, "Recovered"]).all()
    assert (actual["prediction"] == actual["truth"]).sum() == 13


@pytest.mark.integration
def test_fixed_version_results_on_the_16_test_cows(
    data_dir: Path, load_data_file: Callable[[str], LoadedFile]
) -> None:
    actual = _run_all(data_dir, load_data_file, CalvingVersion.FIXED)
    wrong = actual[actual["prediction"] != actual["truth"]]
    assert (actual["prediction"] == actual["truth"]).sum() == 12
    assert sorted(wrong[wrong["truth"] == 0].index) == ["101"]
    assert sorted(wrong[wrong["truth"] == 1].index) == ["1419", "1572", "1712"]


@pytest.mark.unit
@pytest.mark.parametrize("version", list(CalvingVersion))
def test_cow_with_no_readings_in_range_is_handled(version: CalvingVersion) -> None:
    from cowmonitor.charts import half_day_chart

    result = check_calving(_readings([41.0] * 6), version)
    assert not result.enough_data
    assert not result.calving
    figure = half_day_chart(result, "America/New_York", "hot cow")
    assert figure.layout.annotations
