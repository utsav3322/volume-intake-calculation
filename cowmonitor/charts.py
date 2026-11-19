"""Plotly charts for the module results (no Streamlit here)."""

from __future__ import annotations

import html
from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cowmonitor.abnormal import AbnormalResult
from cowmonitor.calving import CalvingResult, HalfDayEvent

TEMPERATURE_COLOUR = "#4C78A8"
DRINKING_COLOUR = "#E45756"
MEDIAN_COLOUR = "#F58518"
BASELINE_COLOUR = "#9D755D"
RECOVERY_COLOUR = "#54A24B"
LOW_LINES = (32.0, 33.5, 34.5)
HIGH_LINES = (39.0, 40.0)
MAX_POINTS = 20_000
GAP_MINUTES = 60.0


def _local(timestamps: pd.Series, timezone: str) -> pd.Series:
    return timestamps.dt.tz_convert(timezone).dt.tz_localize(None)


def _simplify(readings: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """At most MAX_POINTS points: the lowest and highest reading of each equal-sized slot."""
    if len(readings) <= MAX_POINTS:
        return readings, False
    slots = np.arange(len(readings)) * (MAX_POINTS // 2) // len(readings)
    grouped = readings.groupby(slots)["temperature"]
    keep = np.union1d(grouped.idxmin().to_numpy(), grouped.idxmax().to_numpy())
    return readings.loc[keep].reset_index(drop=True), True


def _gap_minutes(shown: pd.DataFrame, simplified: bool) -> float:
    if not simplified or len(shown) < 2:
        return GAP_MINUTES
    span = (shown["ts"].iloc[-1] - shown["ts"].iloc[0]).total_seconds() / 60
    return max(GAP_MINUTES, 3 * span / (MAX_POINTS // 2))


def _break_at_gaps(
    utc: pd.Series, local: pd.Series, values: pd.Series, gap_minutes: float = GAP_MINUTES
) -> tuple[list[object], list[object]]:
    """Insert a blank point wherever readings stop for over an hour, so lines don't bridge gaps.

    Gaps are measured on the UTC timestamps, so a daylight-saving change is not a gap.
    """
    gap_after = utc.diff().dt.total_seconds().div(60).gt(gap_minutes).shift(-1, fill_value=False)
    xs: list[object] = []
    ys: list[object] = []
    for moment, value, gap in zip(local, values, gap_after, strict=True):
        xs.append(moment)
        ys.append(value)
        if gap:
            xs.append(moment)
            ys.append(None)
    return xs, ys


def _layout(figure: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    figure.update_layout(
        title=html.escape(title, quote=False),
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        height=420,
    )
    return figure


def temperature_chart(
    readings: pd.DataFrame,
    timezone: str,
    title: str,
    drinking: pd.Series | None = None,
    thresholds: Sequence[float] = LOW_LINES + HIGH_LINES,
    normal: pd.Series | None = None,
) -> go.Figure:
    """Every reading as a line (broken at gaps), optional drinking marks and threshold lines.

    ``normal``, if given, is drawn as a dashed line: the cow's normal temperature. Very long
    series are simplified to the lowest and highest reading in each time slot, so the chart
    stays light while drinking drops and peaks remain visible.
    """
    readings = readings.reset_index(drop=True)
    if normal is not None:
        readings = readings.assign(normal=normal.to_numpy())
    if drinking is not None:
        drinking = drinking.reset_index(drop=True)
    shown, simplified = _simplify(readings)
    if simplified:
        title = f"{title} (simplified: lowest and highest reading per time slot)"
    local = _local(shown["ts"], timezone)
    gap = _gap_minutes(shown, simplified)
    xs, ys = _break_at_gaps(shown["ts"], local, shown["temperature"], gap)
    figure = go.Figure(
        go.Scattergl(
            x=xs,
            y=ys,
            mode="lines",
            name="Temperature",
            line={"width": 1, "color": TEMPERATURE_COLOUR},
            connectgaps=False,
        )
    )
    if normal is not None:
        xs, ys = _break_at_gaps(shown["ts"], local, shown["normal"], gap)
        figure.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines",
                name="Cow's normal",
                line={"width": 1, "dash": "dash", "color": BASELINE_COLOUR},
                connectgaps=False,
            )
        )
    if drinking is not None and bool(drinking.any()):
        marked = readings[drinking.to_numpy()]
        step = -(-len(marked) // MAX_POINTS)
        marked = marked.iloc[::step]
        figure.add_trace(
            go.Scattergl(
                x=_local(marked["ts"], timezone),
                y=marked["temperature"],
                mode="markers",
                name="Drinking readings",
                marker={"size": 5, "color": DRINKING_COLOUR},
            )
        )
    for level in thresholds:
        figure.add_hline(
            y=level,
            line={"dash": "dot", "width": 1, "color": "#888"},
            annotation_text=f"{level:g} °C",
            annotation_position="right",
        )
    return _layout(figure, title, f"Time ({timezone})", "Temperature (°C)")


def abnormal_counts_chart(result: AbnormalResult, title: str) -> go.Figure:
    labels = [check.description for check in result.checks]
    figure = go.Figure(
        [
            go.Bar(
                x=labels,
                y=[check.count for check in result.checks],
                name="Count in this file",
                marker_color=TEMPERATURE_COLOUR,
            ),
            go.Bar(
                x=labels,
                y=[round(check.limit, 1) for check in result.checks],
                name="Limit",
                marker_color="#BAB0AC",
            ),
        ]
    )
    figure.update_layout(barmode="group")
    return _layout(figure, title, "", "Readings")


def half_day_chart(result: CalvingResult, timezone: str, title: str) -> go.Figure:
    """Half-day medians, the average of the previous four, and the flagged half-days."""
    table = result.half_days
    used = table[table["used"] & table["median_c"].gt(0)]
    if used.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No half-days with enough readings between 37 and 40 °C",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        return _layout(figure, title, "", "Median temperature (°C)")
    middle = _local(used["start"] + pd.Timedelta(hours=6), timezone)
    figure = go.Figure(
        go.Scatter(
            x=middle,
            y=used["median_c"],
            mode="lines+markers",
            name="Half-day median",
            line={"color": MEDIAN_COLOUR},
        )
    )
    if "baseline_c" in used.columns:
        figure.add_trace(
            go.Scatter(
                x=middle,
                y=used["baseline_c"],
                mode="lines",
                name="Average of previous 4",
                line={"dash": "dash", "color": BASELINE_COLOUR},
            )
        )
    _add_events(
        figure,
        result.calving_events,
        timezone,
        "Calving-like drop",
        "triangle-down",
        DRINKING_COLOUR,
    )
    _add_events(
        figure, result.recovery_events, timezone, "Rises again", "triangle-up", RECOVERY_COLOUR
    )
    return _layout(figure, title, f"Middle of half-day ({timezone})", "Median temperature (°C)")


def _add_events(
    figure: go.Figure,
    events: Sequence[HalfDayEvent],
    timezone: str,
    name: str,
    symbol: str,
    colour: str,
) -> None:
    if not events:
        return
    starts = pd.Series([event.start for event in events])
    figure.add_trace(
        go.Scatter(
            x=_local(starts + pd.Timedelta(hours=6), timezone),
            y=[event.temperature for event in events],
            mode="markers",
            name=name,
            marker={"symbol": symbol, "size": 14, "color": colour},
        )
    )


def daily_bar_chart(
    frame: pd.DataFrame, value: str, title: str, y_title: str, average: float | None = None
) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=frame["Date"].astype(str), y=frame[value], name=value, marker_color=TEMPERATURE_COLOUR
        )
    )
    if average is not None and not np.isnan(average):
        figure.add_hline(
            y=average,
            line={"dash": "dash", "color": DRINKING_COLOUR},
            annotation_text=f"Average {average:.1f}",
            annotation_position="top left",
        )
    return _layout(figure, title, "Date", y_title)


def hourly_chart(by_hour: pd.DataFrame, title: str) -> go.Figure:
    figure = go.Figure(
        go.Bar(x=by_hour["Hour"], y=by_hour["Average gallons"], marker_color=TEMPERATURE_COLOUR)
    )
    figure.update_xaxes(dtick=1)
    return _layout(
        figure, title, "Hour of day (meter's local time)", "Average gallons in that hour"
    )
