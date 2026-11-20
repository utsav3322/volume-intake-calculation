"""CowMonitor toolkit: upload one CSV file, choose a module, and press Generate.

Run with ``streamlit run app.py``. Uploaded files are kept in memory for the current browser
session only; refreshing the page clears them and every result.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from pathlib import PurePath

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from cowmonitor.access import PASSCODE, is_passcode
from cowmonitor.intake import REFERENCE_WATER_C
from cowmonitor.loaders import (
    WATER_MAX_C,
    WATER_MIN_C,
    FileKind,
    LoadedFile,
    TemperatureData,
    UnsupportedFileError,
    load_upload,
    load_water_temperature,
)
from cowmonitor.modules import (
    AbnormalOptions,
    CalvingOptions,
    DrinkingOptions,
    IntakeOptions,
    ModuleOptions,
    ModuleSpec,
    WaterOptions,
    get_module,
    modules_for,
    offered_modules,
    run_module,
)
from cowmonitor.results import Metric, ModuleResult, table_to_csv
from cowmonitor.timezones import DEFAULT_TIMEZONE, available_timezones, format_local
from passcode_boxes import passcode_boxes

logger = logging.getLogger("cowmonitor.app")

METRICS_PER_ROW = 3
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()<>#+\-.!|~$:])")
_UNLOCKED = "unlocked"
# Streamlit Community Cloud adds buttons (Share, star, edit, GitHub) to the page's toolbar. An
# icon-only one is labelled with Cloud's name for it, so this hides the edit (pencil) one and
# leaves the others. It only matches buttons whose label mentions editing.
_HIDE_CLOUD_EDIT_BUTTON = """<style>
[data-testid="stToolbarActionButton"]:has(button[aria-label*="edit" i]),
[data-testid="stToolbarActionButton"]:has(button[aria-label*="codespace" i]),
[data-testid="stToolbarActionButton"]:has(button[aria-label*="pencil" i]) {
  display: none !important;
}
</style>"""


def _md(text: str) -> str:
    """Escape Markdown so text that came from a file shows literally: no links, images or styles."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def main() -> None:
    st.set_page_config(page_title="CowMonitor toolkit", layout="wide")
    st.html(_HIDE_CLOUD_EDIT_BUTTON)
    if not _passcode_gate():
        return
    st.session_state.setdefault("nonce", 0)
    timezone = _sidebar()
    st.title("CowMonitor toolkit")
    st.caption(
        "Upload one CSV file, choose a module, and press **Generate**. Files stay in memory "
        "only; refreshing the page clears everything."
    )
    loaded = _upload_step(timezone)
    if loaded is None:
        st.info("Upload a file to begin.")
        return
    spec = _module_step(loaded)
    _settings_step(loaded, spec, timezone)
    _results_step(loaded, spec, timezone)


def _passcode_gate() -> bool:
    """Show only the passcode boxes until the passcode is entered in this browser session."""
    if st.session_state.get(_UNLOCKED):
        return True
    st.title("CowMonitor toolkit")
    entered = passcode_boxes(key="passcode", length=len(PASSCODE))
    if entered is not None:
        if is_passcode(entered):
            st.session_state[_UNLOCKED] = True
            st.rerun()
        st.error("Wrong passcode. Try again.")
    return False


def _sidebar() -> str:
    with st.sidebar:
        st.header("Settings")
        zones = available_timezones()
        timezone = st.selectbox(
            "Farm time zone",
            zones,
            index=zones.index(DEFAULT_TIMEZONE) if DEFAULT_TIMEZONE in zones else 0,
        )
        if st.button("Start over"):
            _reset()
            st.rerun()
    return str(timezone)


def _reset() -> None:
    nonce = int(st.session_state.get("nonce", 0)) + 1
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state["nonce"] = nonce
    st.session_state[_UNLOCKED] = True  # Start over clears the file, not the passcode.


def _clear_result() -> None:
    st.session_state.pop("result", None)
    st.session_state.pop("result_for", None)


def _upload_step(timezone: str) -> LoadedFile | None:
    uploaded = st.file_uploader(
        "**1. Upload a CSV file**", type=["csv"], key=f"upload-{st.session_state['nonce']}"
    )
    if uploaded is None:
        _clear_result()
        st.session_state.pop("loaded", None)
        return None
    content = uploaded.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    cached = st.session_state.get("loaded")
    if isinstance(cached, LoadedFile) and cached.sha256 == digest:
        loaded = cached
    else:
        try:
            loaded = load_upload(uploaded.name, content)
        except UnsupportedFileError as exc:
            _clear_result()
            st.error(_md(str(exc)))
            return None
        st.session_state["loaded"] = loaded
    _describe(loaded, timezone)
    return loaded


def _describe(loaded: LoadedFile, timezone: str) -> None:
    if loaded.temperature is not None:
        data = loaded.temperature
        first = data.readings["ts"].min()
        last = data.readings["ts"].max()
        cows = len(data.cows)
        st.success(
            f"**Temperature export**: {cows} cow{'s' if cows != 1 else ''}, "
            f"{len(data.readings):,} readings, {format_local(first, timezone)} to "
            f"{format_local(last, timezone)} ({timezone})."
        )
        for warning in data.warnings:
            st.warning(_md(warning))
    elif loaded.meter is not None:
        readings = loaded.meter.readings
        st.success(
            f"**Water-meter export**: {len(readings):,} hourly reports, "
            f"{readings['read_time'].min():%d %b %Y %H:%M} to "
            f"{readings['read_time'].max():%d %b %Y %H:%M} (meter's local time)."
        )
        for warning in loaded.meter.warnings:
            st.warning(_md(warning))


def _module_step(loaded: LoadedFile) -> ModuleSpec:
    specs = modules_for(loaded.kind)
    key = st.radio(
        "**2. Choose a module**",
        [spec.key for spec in specs],
        format_func=lambda k: get_module(k).title
        + (" (experimental)" if get_module(k).experimental else ""),
        captions=[spec.summary for spec in specs],
        key=f"module-{loaded.kind.value}",
        on_change=_clear_result,
    )
    unavailable = [spec.title for spec in offered_modules() if spec.accepts is not loaded.kind]
    needs = (
        "a water-meter export" if loaded.kind is FileKind.TEMPERATURE else "a temperature export"
    )
    if unavailable:
        st.caption(f"Not available for this file: {', '.join(unavailable)} (needs {needs}).")
    return get_module(str(key))


def _settings_step(loaded: LoadedFile, spec: ModuleSpec, timezone: str) -> None:
    with st.form(f"settings-{spec.key}"):
        st.markdown("**3. Settings**")
        builder = _settings_widgets(loaded, spec, timezone)
        submitted = st.form_submit_button("Generate", type="primary")
    if not submitted:
        return
    try:
        options = builder()
        with st.spinner("Calculating..."):
            result = run_module(loaded, spec.key, options)
    except UnsupportedFileError as exc:
        _clear_result()
        st.error(_md(str(exc)))
        return
    except Exception:
        logger.exception("Module %s failed on %r", spec.key, loaded.name)
        _clear_result()
        st.error(
            "Something went wrong while calculating. Check that the file is a complete export "
            "and try again."
        )
        return
    st.session_state["result"] = result
    st.session_state["result_for"] = (loaded.sha256, spec.key, timezone)


def _settings_widgets(
    loaded: LoadedFile, spec: ModuleSpec, timezone: str
) -> Callable[[], ModuleOptions]:
    """Draw the module's settings and return a function that builds its options on submit."""
    if spec.key == "water":
        name = st.text_input("Meter name", value=PurePath(loaded.name).stem)
        cows = st.text_input("Cows on this meter (optional)", placeholder="e.g. 1354, 1568")
        return lambda: WaterOptions(meter_name=name, cows_on_meter=cows)

    data = loaded.temperature
    if data is None:
        raise UnsupportedFileError("This module needs a temperature export.")
    many = spec.key in ("abnormal", "calving") and len(data.cows) > 1
    cow = st.selectbox(
        "Cow to show in detail (the results table covers every cow)" if many else "Cow",
        [c.key for c in data.cows],
        format_func=lambda key: data.cow(key).label,
    )
    # The abnormal check and the calving alert each run one method, the recommended one (see
    # cowmonitor.modules), so there is no method to choose here.
    if spec.key == "abnormal":
        return lambda: AbnormalOptions(cow=cow, timezone=timezone)
    if spec.key == "calving":
        removed = st.checkbox(
            "This file already has drinking readings removed (CowMonitor's processed copy)"
        )
        return lambda: CalvingOptions(cow=cow, drinking_removed=removed, timezone=timezone)
    nonce = st.session_state["nonce"]
    if spec.key == "intake":
        copy_upload = st.file_uploader(
            "Optional: CowMonitor's drinking-removed file for this cow",
            type=["csv"],
            key=f"intake-copy-{nonce}",
        )
        water_upload = st.file_uploader(
            "Optional: water temperature file (timestamp and temperature, optionally deviceId)",
            type=["csv"],
            key=f"intake-water-{nonce}",
        )
        water_c = st.number_input(
            "Water temperature when there is no file (°C)",
            min_value=WATER_MIN_C,
            max_value=WATER_MAX_C,
            value=REFERENCE_WATER_C,
            step=0.5,
        )

        def build_intake() -> ModuleOptions:
            return IntakeOptions(
                cow=cow,
                timezone=timezone,
                water_c=float(water_c),
                copy=_load_copy(copy_upload),
                water=_load_water(water_upload, timezone),
                water_name=water_upload.name if water_upload is not None else "",
            )

        return build_intake
    copy_upload = st.file_uploader(
        "Optional: CowMonitor's drinking-removed file for this cow",
        type=["csv"],
        key=f"copy-{nonce}",
    )
    return lambda: DrinkingOptions(cow=cow, timezone=timezone, copy=_load_copy(copy_upload))


def _load_copy(upload: UploadedFile | None) -> TemperatureData | None:
    """CowMonitor's drinking-removed copy from the settings, if one was uploaded."""
    if upload is None:
        return None
    try:
        extra = load_upload(upload.name, upload.getvalue())
    except UnsupportedFileError as exc:
        raise UnsupportedFileError(f"Drinking-removed file: {exc}") from exc
    if extra.temperature is None:
        raise UnsupportedFileError(
            "That is a water-meter export. This box takes CowMonitor's drinking-removed "
            "file for the same cow (in the project's data, the *_bouts.csv file). To see "
            "water use, upload the meter file in step 1 and choose Daily water use."
        )
    return extra.temperature


def _load_water(upload: UploadedFile | None, timezone: str) -> TemperatureData | None:
    """The water temperature file from the settings, if one was uploaded."""
    if upload is None:
        return None
    try:
        return load_water_temperature(upload.name, upload.getvalue(), timezone)
    except UnsupportedFileError as exc:
        raise UnsupportedFileError(f"Water temperature file: {exc}") from exc


def _results_step(loaded: LoadedFile, spec: ModuleSpec, timezone: str) -> None:
    result = st.session_state.get("result")
    if not isinstance(result, ModuleResult):
        return
    if st.session_state.get("result_for") != (loaded.sha256, spec.key, timezone):
        return
    _render(result)


def _render(result: ModuleResult) -> None:
    st.divider()
    st.subheader(_md(result.title))
    if result.subtitle:
        st.caption(_md(result.subtitle))
    for warning in result.warnings:
        st.warning(_md(warning))
    _metrics(result.metrics)
    for index, chart in enumerate(result.charts):
        st.plotly_chart(chart.figure, width="stretch", key=f"chart-{index}")
        if chart.caption:
            st.caption(_md(chart.caption))
    for index, table in enumerate(result.tables):
        st.markdown(f"**{_md(table.title)}**")
        st.dataframe(table.frame, hide_index=True, width="stretch")
        st.download_button(
            "Download CSV",
            data=table_to_csv(table.frame),
            file_name=table.file_name,
            mime="text/csv",
            key=f"download-{index}",
        )
    if result.notes:
        with st.expander("How this is calculated"):
            for note in result.notes:
                st.markdown(_md(note))


def _metrics(metrics: tuple[Metric, ...]) -> None:
    for start in range(0, len(metrics), METRICS_PER_ROW):
        columns = st.columns(METRICS_PER_ROW)
        for column, metric in zip(columns, metrics[start : start + METRICS_PER_ROW], strict=False):
            column.metric(metric.label, metric.value)


main()
