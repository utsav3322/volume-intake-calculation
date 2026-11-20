"""The Streamlit page, driven headlessly: upload a file, pick a module, press Generate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block

from cowmonitor.access import PASSCODE
from cowmonitor.intake import REFERENCE_WATER_C
from tests.helpers import meter_csv, steady_temperatures, temperature_frame, to_csv_bytes

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _locked_page() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def _page() -> AppTest:
    """The app as it is once the passcode has been entered."""
    page = AppTest.from_file(APP, default_timeout=60)
    page.session_state["unlocked"] = True
    return page.run()


@pytest.fixture
def typed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Codes "typed" into the passcode boxes; like the browser, each is sent on one run."""
    codes: list[str] = []

    def boxes(*, key: str, length: int) -> str | None:
        assert (key, length) == ("passcode", len(PASSCODE))
        return codes.pop(0) if codes else None

    monkeypatch.setattr("passcode_boxes.passcode_boxes", boxes)
    return codes


def _enter_passcode(page: AppTest, typed: list[str], passcode: str) -> AppTest:
    typed.append(passcode)
    return page.run()


def _passcode_boxes(page: AppTest) -> list[object]:
    return list(page.get("bidi_component"))


def _upload(page: AppTest, name: str, content: bytes) -> AppTest:
    return page.file_uploader[0].upload(name, content, "text/csv").run()


def _generate(page: AppTest) -> AppTest:
    return next(button for button in page.button if button.label == "Generate").click().run()


def _metrics(page: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in page.metric}


@pytest.fixture
def all_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offer the modules hidden for now, so their pages stay tested until they come back."""
    monkeypatch.setattr("cowmonitor.modules.HIDDEN_MODULES", frozenset())


def _captions(page: AppTest) -> list[str]:
    return [caption.value for caption in page.caption]


def _help_texts(page: AppTest) -> list[str]:
    """Every tooltip on the page; a non-empty one shows as a "?" icon or a hover text."""
    nodes = [*page.sidebar, *page.main]
    protos = [getattr(node, "proto", None) for node in nodes if not isinstance(node, Block)]
    return [text for text in (getattr(proto, "help", "") for proto in protos) if text]


@pytest.mark.unit
def test_the_passcode_screen_comes_first() -> None:
    page = _locked_page()  # the real passcode boxes, with nothing typed yet
    assert not page.exception
    assert page.title[0].value == "CowMonitor toolkit"
    assert len(_passcode_boxes(page)) == 1
    assert not page.text_input
    assert not page.button, "the boxes send the code themselves; there is no Unlock button"
    assert not page.error
    assert not page.file_uploader
    assert not [node for node in page.sidebar if not isinstance(node, Block)]
    main = [type(node).__name__ for node in page.main.children.values()]
    assert main == ["Title", "UnknownElement"], "only the title and the passcode boxes"


@pytest.mark.unit
@pytest.mark.parametrize("passcode", ["", "1234", "0008", "9000"])
def test_a_wrong_passcode_is_refused(typed: list[str], passcode: str) -> None:
    page = _enter_passcode(_locked_page(), typed, passcode)
    assert not page.exception
    assert [error.value for error in page.error] == ["Wrong passcode. Try again."]
    assert not page.file_uploader
    page = _enter_passcode(page, typed, PASSCODE)
    assert len(page.file_uploader) == 1


@pytest.mark.unit
def test_the_right_passcode_opens_the_app(typed: list[str]) -> None:
    page = _enter_passcode(_locked_page(), typed, PASSCODE)
    assert not page.exception
    assert not page.error
    assert not _passcode_boxes(page)
    assert "passcode" not in page.session_state
    assert len(page.file_uploader) == 1
    assert [info.value for info in page.info] == ["Upload a file to begin."]
    assert page.sidebar.button[0].label == "Start over"


@pytest.mark.unit
def test_page_opens_with_an_upload_prompt() -> None:
    page = _page()
    assert not page.exception
    assert page.title[0].value == "CowMonitor toolkit"
    assert [info.value for info in page.info] == ["Upload a file to begin."]
    assert len(page.file_uploader) == 1
    texts = [markdown.value for markdown in page.main.markdown]
    assert not any("CowMonitor temperature export" in text for text in texts)
    assert not any("Modules:" in text for text in texts)


def _page_css(page: AppTest) -> str:
    return "".join(element.proto.body for element in page.get("html"))


@pytest.mark.unit
def test_cloud_edit_button_is_hidden_before_and_after_the_passcode() -> None:
    for page in (_locked_page(), _page()):
        css = _page_css(page)
        assert '[data-testid="stToolbarActionButton"]:has(button[aria-label*="edit" i])' in css
        assert "display: none" in css
        assert "share" not in css.lower() and "github" not in css.lower()


@pytest.mark.unit
def test_unrecognised_file_shows_an_error_and_no_modules() -> None:
    page = _upload(_page(), "notes.csv", b"name,score\nalice,3\n")
    assert not page.exception
    assert "doesn't look like" in page.error[0].value
    assert not page.radio


@pytest.mark.unit
def test_sidebar_has_only_the_time_zone_and_start_over() -> None:
    sidebar = _page().sidebar
    assert [node.type for node in sidebar if not isinstance(node, Block)] == [
        "header",
        "selectbox",
        "button",
    ]
    assert sidebar.selectbox[0].label == "Farm time zone"
    assert sidebar.button[0].label == "Start over"


@pytest.mark.unit
@pytest.mark.usefixtures("all_modules")
def test_no_help_tooltips_anywhere() -> None:
    page = _page()
    assert _help_texts(page) == []
    content = to_csv_bytes(temperature_frame(steady_temperatures(3)))
    page = _upload(page, "cow.csv", content)
    for module in ("abnormal", "calving", "drinking", "intake"):
        page = _generate(page.radio(key="module-temperature").set_value(module).run())
        assert page.metric, module
        assert _help_texts(page) == [], module
    page = _generate(_upload(_page(), "2D.csv", meter_csv()))
    assert page.metric
    assert _help_texts(page) == []


@pytest.mark.unit
def test_temperature_file_offers_only_the_water_intake_estimate() -> None:
    content = to_csv_bytes(temperature_frame(steady_temperatures(3)))
    page = _upload(_page(), "cow.csv", content)
    module = page.radio(key="module-temperature")
    assert module.options == ["Water intake estimate (experimental)"]
    assert module.value == "intake"
    assert "Not available for this file: Daily water use (needs a water-meter export)." in (
        _captions(page)
    )


@pytest.mark.unit
def test_meter_file_doesnt_name_the_hidden_modules() -> None:
    page = _upload(_page(), "2D.csv", meter_csv())
    assert page.radio(key="module-meter").options == ["Daily water use"]
    assert "Not available for this file: Water intake estimate (needs a temperature export)." in (
        _captions(page)
    )


@pytest.mark.unit
@pytest.mark.usefixtures("all_modules")
def test_temperature_file_offers_four_modules_when_all_are_switched_on() -> None:
    content = to_csv_bytes(temperature_frame(steady_temperatures(3)))
    module = _upload(_page(), "cow.csv", content).radio(key="module-temperature")
    assert module.options == [
        "Abnormal drinking check",
        "Calving alert (experimental)",
        "Daily drinking summary",
        "Water intake estimate (experimental)",
    ]
    assert module.value == "abnormal"


def _cow_with_dips() -> bytes:
    temperatures = steady_temperatures(2)
    for start in (100, 200, 400):
        temperatures[start : start + 4] = [34.0, 35.5, 36.8, 37.9]
    return to_csv_bytes(temperature_frame(temperatures, start="2024-04-06T04:00:00Z"))


@pytest.mark.unit
def test_intake_estimate_with_a_water_file_through_the_page() -> None:
    page = _upload(_page(), "cow.csv", _cow_with_dips())
    page = page.radio(key="module-temperature").set_value("intake").run()
    assert [box.label for box in page.number_input] == [
        "Water temperature when there is no file (°C)"
    ]
    assert page.number_input[0].value == REFERENCE_WATER_C
    water = temperature_frame(np.full(200, 8.0), freq="15min", device="trough", animal=None)
    page.file_uploader[2].upload("water.csv", to_csv_bytes(water), "text/csv")
    page = _generate(page)
    assert not page.exception
    assert not page.error
    assert page.subheader[0].value.startswith("Water intake estimate")
    metrics = _metrics(page)
    assert metrics["Water temperature"] == "8.0 °C (median, from file)"
    assert metrics["Drinks per day"] == "1.5"
    assert len(page.get("plotly_chart")) == 3
    assert len(page.dataframe) == 2


@pytest.mark.unit
def test_intake_estimate_refuses_a_cow_file_in_the_water_box() -> None:
    page = _upload(_page(), "cow.csv", _cow_with_dips())
    page = page.radio(key="module-temperature").set_value("intake").run()
    page.file_uploader[2].upload("other.csv", _cow_with_dips(), "text/csv")
    page = _generate(page)
    assert not page.exception
    assert page.error[0].value.startswith("Water temperature file\\: ")
    assert "rumen temperature" in page.error[0].value
    assert not page.subheader


@pytest.mark.unit
def test_intake_estimate_names_the_box_when_a_water_file_is_in_the_copy_box() -> None:
    page = _upload(_page(), "cow.csv", _cow_with_dips())
    page = page.radio(key="module-temperature").set_value("intake").run()
    water = temperature_frame(np.full(200, 8.0), freq="15min", device="trough", animal=None)
    page.file_uploader[1].upload("water.csv", to_csv_bytes(water), "text/csv")
    page = _generate(page)
    assert not page.exception
    assert page.error[0].value.startswith("Drinking\\-removed file\\: ")
    assert "look like water temperatures" in page.error[0].value
    assert not page.subheader


@pytest.mark.unit
def test_water_file_in_step_one_gets_a_helpful_message() -> None:
    water = temperature_frame(np.full(200, 8.0), freq="15min", device="trough", animal=None)
    page = _upload(_page(), "water.csv", to_csv_bytes(water))
    assert not page.exception
    assert "look like water temperatures" in page.error[0].value
    assert not page.radio


@pytest.mark.unit
@pytest.mark.parametrize(
    ("module", "inputs"),
    [("abnormal", ["selectbox"]), ("calving", ["selectbox", "checkbox"])],
)
@pytest.mark.usefixtures("all_modules")
def test_settings_offer_no_choice_of_method(module: str, inputs: list[str]) -> None:
    content = to_csv_bytes(temperature_frame(steady_temperatures(3)))
    page = _upload(_page(), "cow.csv", content)
    page = page.radio(key="module-temperature").set_value(module).run()
    assert [radio.label for radio in page.radio] == ["**2. Choose a module**"]
    form = next(node for node in page.main if isinstance(node, Block) and node.type == "form")
    in_form = [node.type for node in form if not isinstance(node, Block)]
    assert [kind for kind in in_form if kind not in ("markdown", "button")] == inputs


@pytest.mark.unit
@pytest.mark.usefixtures("all_modules")
def test_abnormal_check_always_uses_per_day_limits() -> None:
    content = to_csv_bytes(temperature_frame(steady_temperatures(3)))
    page = _generate(_upload(_page(), "cow.csv", content))
    assert page.subheader[0].value.startswith("Abnormal drinking check")
    assert _metrics(page)["Limits"].startswith("Per day")


@pytest.mark.unit
@pytest.mark.usefixtures("all_modules")
def test_calving_alert_always_uses_the_fixed_version() -> None:
    content = to_csv_bytes(temperature_frame(steady_temperatures(3)))
    page = _upload(_page(), "cow.csv", content)
    page = _generate(page.radio(key="module-temperature").set_value("calving").run())
    assert not page.exception
    assert page.subheader[0].value.startswith("Calving alert")
    assert any(warning.value.startswith("Experimental") for warning in page.warning)
    assert not any("As delivered" in caption.value for caption in page.caption)
    assert not any("Fixed version" in caption.value for caption in page.caption)


@pytest.mark.unit
@pytest.mark.usefixtures("all_modules")
def test_changing_module_or_time_zone_hides_old_results() -> None:
    content = to_csv_bytes(temperature_frame(steady_temperatures(3)))
    page = _generate(_upload(_page(), "cow.csv", content))
    assert page.subheader[0].value.startswith("Abnormal drinking check")
    page = page.sidebar.selectbox[0].set_value("UTC").run()
    assert not page.subheader
    page = _generate(page)
    assert page.subheader[0].value.startswith("Abnormal drinking check")
    page = page.radio(key="module-temperature").set_value("calving").run()
    assert not page.subheader


@pytest.mark.unit
def test_start_over_clears_the_file() -> None:
    content = to_csv_bytes(temperature_frame(steady_temperatures(3)))
    page = _generate(_upload(_page(), "cow.csv", content))
    page = next(button for button in page.button if button.label == "Start over").click().run()
    assert not page.exception
    assert [info.value for info in page.info] == ["Upload a file to begin."]
    assert not page.subheader
    assert not _passcode_boxes(page), "Start over clears the file, not the passcode"
    assert len(page.file_uploader) == 1


@pytest.mark.integration
@pytest.mark.parametrize(
    ("relative_path", "module", "title", "metric", "value"),
    [
        ("AbnormalData/1431_Y.csv", "abnormal", "Abnormal drinking check", "Verdict", "Abnormal"),
        ("AbnormalData/101_N.csv", "abnormal", "Abnormal drinking check", "Verdict", "Normal"),
        ("CalvingData/1734_Y.csv", "calving", "Calving alert", "Calving-like drop", "Yes"),
        ("DrinkingData/1354_raw.csv", "drinking", "Daily drinking summary", "Source", "Estimated"),
        ("DrinkingData/2D.csv", "water", "Daily water use", "Total gallons", "674.1"),
    ],
)
@pytest.mark.usefixtures("all_modules")
def test_each_module_through_the_page(
    data_dir: Path, relative_path: str, module: str, title: str, metric: str, value: str
) -> None:
    page = _upload(_page(), Path(relative_path).name, (data_dir / relative_path).read_bytes())
    kind = "meter" if module == "water" else "temperature"
    page = page.radio(key=f"module-{kind}").set_value(module).run()
    page = _generate(page)
    assert not page.exception
    assert not page.error
    assert page.subheader[0].value.startswith(title)
    assert _metrics(page)[metric] == value
    assert len(page.dataframe) >= 1
    assert len(page.get("plotly_chart")) >= 1


@pytest.mark.integration
def test_intake_estimate_with_the_project_files_and_the_sample_water_file(data_dir: Path) -> None:
    raw = (data_dir / "DrinkingData/1354_raw.csv").read_bytes()
    copy = (data_dir / "DrinkingData/1354_bouts.csv").read_bytes()
    sample = Path(APP).parent / "sample_data" / "synthetic_water_temperature_april2024.csv"
    page = _upload(_page(), "1354_raw.csv", raw)
    page = page.radio(key="module-temperature").set_value("intake").run()
    page.file_uploader[1].upload("1354_bouts.csv", copy, "text/csv")
    page.file_uploader[2].upload(sample.name, sample.read_bytes(), "text/csv")
    page = _generate(page)
    assert not page.exception
    assert not page.error
    metrics = _metrics(page)
    assert metrics["Average per day"] == "24.6 gal (93 L)"
    assert metrics["Water temperature"] == f"{REFERENCE_WATER_C:.1f} °C (median, from file)"
    assert metrics["Drinking readings from"] == "CowMonitor's drinking-removed file"


@pytest.mark.integration
@pytest.mark.usefixtures("all_modules")
def test_drinking_summary_with_the_cowmonitor_copy(data_dir: Path) -> None:
    raw = (data_dir / "DrinkingData/1354_raw.csv").read_bytes()
    copy = (data_dir / "DrinkingData/1354_bouts.csv").read_bytes()
    page = _upload(_page(), "1354_raw.csv", raw)
    page = page.radio(key="module-temperature").set_value("drinking").run()
    # Form widgets are sent together with the Generate click, as in a browser.
    page.file_uploader[1].upload("1354_bouts.csv", copy, "text/csv")
    page = _generate(page)
    assert not page.exception
    assert _metrics(page)["Source"] == "CowMonitor's drinking-removed file"


@pytest.mark.integration
@pytest.mark.usefixtures("all_modules")
def test_wrong_copy_shows_an_error(data_dir: Path) -> None:
    raw = (data_dir / "DrinkingData/1354_raw.csv").read_bytes()
    other = (data_dir / "DrinkingData/1568_bouts.csv").read_bytes()
    page = _upload(_page(), "1354_raw.csv", raw)
    page = page.radio(key="module-temperature").set_value("drinking").run()
    page.file_uploader[1].upload("1568_bouts.csv", other, "text/csv")
    page = _generate(page)
    assert not page.exception
    assert "no data for" in page.error[0].value
    assert not page.subheader


@pytest.mark.unit
@pytest.mark.usefixtures("all_modules")
def test_switching_module_and_back_clears_the_old_result() -> None:
    content = to_csv_bytes(temperature_frame(steady_temperatures(3)))
    page = _upload(_page(), "cow.csv", content)
    page = _generate(page.radio(key="module-temperature").set_value("abnormal").run())
    assert page.subheader[0].value.startswith("Abnormal drinking check")
    page = page.radio(key="module-temperature").set_value("calving").run()
    page = page.radio(key="module-temperature").set_value("abnormal").run()
    assert not page.subheader


@pytest.mark.unit
def test_unexpected_failure_shows_a_friendly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import cowmonitor.modules

    def broken(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(cowmonitor.modules, "run_module", broken)
    content = to_csv_bytes(temperature_frame(steady_temperatures(3)))
    page = _generate(_upload(_page(), "cow.csv", content))
    assert not page.exception
    assert page.error[0].value.startswith("Something went wrong")
    assert not page.subheader


@pytest.mark.integration
@pytest.mark.usefixtures("all_modules")
def test_meter_file_in_the_copy_box_gets_a_helpful_message(data_dir: Path) -> None:
    raw = (data_dir / "DrinkingData/1354_417_raw.csv").read_bytes()
    meter = (data_dir / "DrinkingData/2d_417.csv").read_bytes()
    page = _upload(_page(), "1354_417_raw.csv", raw)
    page = page.radio(key="module-temperature").set_value("drinking").run()
    page.file_uploader[1].upload("2d_417.csv", meter, "text/csv")
    page = _generate(page)
    assert not page.exception
    assert "water\\-meter export" in page.error[0].value
    assert "\\_bouts" in page.error[0].value


@pytest.mark.integration
@pytest.mark.usefixtures("all_modules")
def test_week_files_pair_up_in_the_drinking_summary(data_dir: Path) -> None:
    raw = (data_dir / "DrinkingData/1354_417_raw.csv").read_bytes()
    copy = (data_dir / "DrinkingData/1354_417_bouts.csv").read_bytes()
    page = _upload(_page(), "1354_417_raw.csv", raw)
    page = page.radio(key="module-temperature").set_value("drinking").run()
    page.file_uploader[1].upload("1354_417_bouts.csv", copy, "text/csv")
    page = _generate(page)
    assert not page.exception
    assert not page.error
    assert _metrics(page)["Source"] == "CowMonitor's drinking-removed file"
    assert _metrics(page)["Days"] == "7"
    assert not any("looks like CowMonitor" in w.value for w in page.warning)
