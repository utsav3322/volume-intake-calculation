"""Hostile or oversized uploads are refused quickly, and file contents can't inject formatting."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

import cowmonitor.loaders as loaders
from cowmonitor.charts import MAX_POINTS, temperature_chart
from cowmonitor.loaders import UnsupportedFileError, load_upload
from cowmonitor.modules import AbnormalOptions, run_module
from cowmonitor.results import table_to_csv
from tests.helpers import meter_csv, steady_temperatures, temperature_frame, to_csv_bytes
from tests.test_app import _page


@pytest.mark.unit
def test_giant_timestamp_cell_is_refused_quickly() -> None:
    rows = ["deviceId,temperature,timestamp"] + ["a,38.5,x"] * 30
    rows.append("a,38.5," + "1." * 100_000)
    started = time.perf_counter()
    with pytest.raises(UnsupportedFileError):
        load_upload("hostile.csv", "\n".join(rows).encode())
    assert time.perf_counter() - started < 2.0


@pytest.mark.unit
def test_very_wide_file_is_refused_before_parsing() -> None:
    content = ("temperature,timestamp" + "," * 200_000 + "\n38.5,2024-04-06T00:00:00Z\n").encode()
    started = time.perf_counter()
    with pytest.raises(UnsupportedFileError, match="columns"):
        load_upload("wide.csv", content)
    assert time.perf_counter() - started < 2.0


@pytest.mark.unit
def test_row_cow_and_meter_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loaders, "MAX_ROWS", 100)
    with pytest.raises(UnsupportedFileError, match="more than 100 rows"):
        load_upload("long.csv", to_csv_bytes(temperature_frame(steady_temperatures(1))))
    with pytest.raises(UnsupportedFileError, match="more than 100 reports"):
        load_upload("meter.csv", meter_csv(hours=150))

    monkeypatch.setattr(loaders, "MAX_ROWS", 500_000)
    monkeypatch.setattr(loaders, "MAX_COWS", 2)
    herd = pd.concat(
        temperature_frame(steady_temperatures(0.1), device=f"d{i}", animal=str(i)) for i in range(3)
    )
    with pytest.raises(UnsupportedFileError, match="3 devices"):
        load_upload("herd.csv", to_csv_bytes(herd))


@pytest.mark.unit
def test_ids_are_reduced_to_plain_characters() -> None:
    frame = temperature_frame(steady_temperatures(0.2), device="![](https://evil.example/x.png)")
    data = load_upload("x.csv", to_csv_bytes(frame)).temperature
    assert data is not None
    key = data.cows[0].key
    assert len(key) <= loaders.MAX_ID_CHARS
    assert not set(key) & set("![]()/<>*`")


@pytest.mark.unit
def test_timestamps_outside_2000_to_2100_are_not_used() -> None:
    content = b"deviceId,temperature,timestamp\nd1,38.5,0001-01-01T00:30:00Z\n"
    with pytest.raises(UnsupportedFileError, match="No usable temperature readings"):
        load_upload("old.csv", content)


@pytest.mark.unit
def test_csv_download_quotes_text_cells() -> None:
    text = table_to_csv(pd.DataFrame({"Cow": ["x;=cmd|' /C calc'!A0;y"], "Days": [1.5]})).decode()
    assert '"x;=cmd' in text
    assert text.strip().endswith(",1.5")


@pytest.mark.unit
def test_long_series_are_simplified_but_keep_drops() -> None:
    temperatures = steady_temperatures(200)  # 57,600 readings
    temperatures[30_000] = 31.0
    frame = temperature_frame(temperatures)
    data = load_upload("x.csv", to_csv_bytes(frame)).temperature
    assert data is not None
    figure = temperature_chart(data.for_cow(data.cows[0].key), "UTC", "<b>cow</b>")
    values = [v for v in figure.data[0].y if v is not None]
    assert len(values) <= MAX_POINTS
    assert min(values) == pytest.approx(31.0)
    assert figure.layout.title.text.startswith("&lt;b&gt;cow&lt;/b&gt; (simplified")


@pytest.mark.unit
def test_two_hundred_cows_run_quickly_with_grouped_warnings() -> None:
    herd = pd.concat(
        temperature_frame(steady_temperatures(0.5), device=f"d{i:03d}", animal=str(i))
        for i in range(200)
    )
    loaded = load_upload("herd.csv", to_csv_bytes(herd))
    assert loaded.temperature is not None
    started = time.perf_counter()
    result = run_module(loaded, "abnormal", AbnormalOptions(cow="d000"))
    assert time.perf_counter() - started < 5.0
    no_verdict = [w for w in result.warnings if "so no verdict" in w]
    assert len(no_verdict) == 1
    assert "and 195 more" in no_verdict[0]


@pytest.mark.unit
def test_page_escapes_text_that_came_from_the_file() -> None:
    good = temperature_frame(steady_temperatures(1), device="dev-a", animal="1")
    broken = temperature_frame(np.zeros(288), device="**x**", animal="2")
    page = _page()
    page = page.file_uploader[0].upload("herd.csv", to_csv_bytes(pd.concat([good, broken]))).run()
    assert not page.exception
    warning = next(w.value for w in page.warning if "No usable readings" in w.value)
    assert "\\_\\_x\\_\\_" in warning
