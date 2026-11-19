"""Model 1 (abnormal drinking check): parity with the 2024 notebook and per-day limits."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cowmonitor.abnormal import LimitsMode, check_abnormal
from cowmonitor.loaders import LoadedFile, load_upload
from tests.helpers import steady_temperatures, temperature_frame, to_csv_bytes


def _readings(temperatures: np.ndarray) -> pd.DataFrame:
    loaded = load_upload("x.csv", to_csv_bytes(temperature_frame(temperatures)))
    assert loaded.temperature is not None
    return loaded.temperature.for_cow(loaded.temperature.cows[0].key)


def _with_drops(days: float, drops_per_day: int, depth: float = 31.0) -> np.ndarray:
    temperatures = steady_temperatures(days)
    per_day = 288
    for day in range(int(days)):
        for k in range(drops_per_day):
            temperatures[day * per_day + 10 + k * 20] = depth
    return temperatures


@pytest.mark.unit
def test_cow_with_regular_deep_drops_is_normal() -> None:
    result = check_abnormal(_readings(_with_drops(6, drops_per_day=8)), LimitsMode.PER_DAY)
    assert result.enough_data
    assert not result.abnormal
    assert result.verdict == "Normal"
    assert result.scale == pytest.approx(result.days / 6.0)


@pytest.mark.unit
def test_cow_without_deep_drops_is_abnormal() -> None:
    result = check_abnormal(_readings(steady_temperatures(6)), LimitsMode.PER_DAY)
    assert result.abnormal
    assert result.few_deep_drops
    assert not result.long_high_temperature
    assert "too few deep drinking drops" in result.reason.lower()


@pytest.mark.unit
def test_long_hot_period_is_abnormal_even_with_drops() -> None:
    temperatures = _with_drops(6, drops_per_day=8, depth=31.0) + 1.5
    temperatures[temperatures < 33] = 31.0
    result = check_abnormal(_readings(temperatures), LimitsMode.PER_DAY)
    assert result.long_high_temperature
    assert result.abnormal
    assert "long high-temperature period" in result.reason.lower()


@pytest.mark.unit
def test_per_day_limits_scale_but_fixed_limits_do_not() -> None:
    readings = _readings(_with_drops(2, drops_per_day=4))
    per_day = check_abnormal(readings, LimitsMode.PER_DAY)
    fixed = check_abnormal(readings, LimitsMode.FIXED)
    assert fixed.scale == 1.0
    assert [check.limit for check in fixed.checks] == [10, 20, 30, 800, 30]
    assert per_day.checks[0].limit == pytest.approx(10 * per_day.days / 6.0)
    # 8 drops in 2 days: under the 6-day limit of 10, but above the 2-day limit of about 3.3.
    assert not per_day.abnormal
    assert fixed.abnormal


@pytest.mark.unit
def test_less_than_a_day_is_not_judged() -> None:
    result = check_abnormal(_readings(steady_temperatures(0.5)), LimitsMode.PER_DAY)
    assert not result.enough_data
    assert not result.abnormal
    assert result.verdict == "Not enough data"


def _abnormal_files(data_dir: Path) -> list[str]:
    return sorted(path.name for path in (data_dir / "AbnormalData").glob("*.csv"))


@pytest.mark.integration
def test_fixed_limits_reproduce_the_2024_notebook(
    data_dir: Path, fixtures_dir: Path, load_data_file: Callable[[str], LoadedFile]
) -> None:
    expected = pd.read_csv(fixtures_dir / "notebook_abnormal_2024.csv", dtype={"id": str})
    expected = expected.set_index("id")
    assert len(expected) == len(_abnormal_files(data_dir)) == 21
    for name in _abnormal_files(data_dir):
        cow_id = name.split("_")[0]
        data = load_data_file(f"AbnormalData/{name}").temperature
        assert data is not None
        result = check_abnormal(data.for_cow(data.cows[0].key), LimitsMode.FIXED)
        assert int(result.abnormal) == expected.loc[cow_id, "Prediction"], name
        assert int(result.long_high_temperature) == expected.loc[cow_id, "TempPeak"], name
    assert (expected["Prediction"] == expected["GroundTruth"]).all()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("window", "minimum_accuracy"),
    [("full", 1.0), ("first 2d", 0.95), ("last 2d", 0.95), ("first 3d", 0.95), ("last 3d", 0.95)],
)
def test_per_day_limits_hold_up_on_shorter_files(
    data_dir: Path,
    load_data_file: Callable[[str], LoadedFile],
    window: str,
    minimum_accuracy: float,
) -> None:
    correct = []
    for name in _abnormal_files(data_dir):
        data = load_data_file(f"AbnormalData/{name}").temperature
        assert data is not None
        readings = data.for_cow(data.cows[0].key)
        if window != "full":
            which, days = window.split()
            span = pd.Timedelta(days=int(days[0]))
            ts = readings["ts"]
            keep = ts < ts.iloc[0] + span if which == "first" else ts > ts.iloc[-1] - span
            readings = readings[keep].reset_index(drop=True)
        result = check_abnormal(readings, LimitsMode.PER_DAY)
        correct.append(int(result.abnormal) == int("_Y" in name))
    assert np.mean(correct) >= minimum_accuracy


@pytest.mark.unit
def test_a_full_day_of_readings_counts_as_one_day() -> None:
    result = check_abnormal(_readings(steady_temperatures(1)), LimitsMode.PER_DAY)
    assert result.days == pytest.approx(1.0)
    assert result.enough_data


@pytest.mark.unit
def test_short_file_reason_is_given_in_hours() -> None:
    result = check_abnormal(_readings(steady_temperatures(20 / 24)), LimitsMode.PER_DAY)
    assert not result.enough_data
    assert result.reason == "Only 20 hours of data; at least 24 hours are needed."
