"""Builders for small synthetic CSV files used by the unit tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def temperature_frame(
    temperatures: Sequence[float] | np.ndarray,
    start: str = "2024-04-06T00:00:00Z",
    freq: str = "5min",
    device: str = "281b0400002004e2",
    animal: str | None = "1354",
) -> pd.DataFrame:
    """A CowMonitor-style export for one cow, with ISO timestamps ending in Z."""
    times = pd.date_range(start=start, periods=len(temperatures), freq=freq)
    frame = pd.DataFrame(
        {
            "deviceId": device,
            "temperature": np.round(np.asarray(temperatures, dtype=float), 2),
            "timestamp": times.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        }
    )
    if animal is not None:
        frame.insert(0, "Animal Id", animal)
    return frame


def to_csv_bytes(frame: pd.DataFrame, bom: bool = True) -> bytes:
    text = frame.to_csv(index=False)
    return (("﻿" if bom else "") + text).encode("utf-8")


def steady_temperatures(days: float, level: float = 38.8, freq_minutes: int = 5) -> np.ndarray:
    """A flat temperature series of the given length with a small daily wobble."""
    count = int(days * 24 * 60 / freq_minutes)
    phase = np.arange(count) * freq_minutes / (24 * 60) * 2 * np.pi
    return level + 0.1 * np.sin(phase)


def meter_csv(
    start: str = "2024-04-06 00:10",
    hours: int = 48,
    gallons_per_hour: float = 2.0,
    newest_first: bool = True,
) -> bytes:
    """An H2O Degree-style hourly export: no header, 23 fields, a date line before each day."""
    first = pd.Timestamp(start)
    gallons_total, events_total, minutes_total = 1000.0, 500, 300
    records: list[tuple[pd.Timestamp, list[str]]] = []
    for hour in range(hours):
        moment = first + pd.Timedelta(hours=hour)
        gallons_total += gallons_per_hour
        events_total += 3
        minutes_total += 1
        stamp = f"{moment.month}/{moment.day}/{moment.year} {moment.hour}:{moment.minute:02d}"
        fields = [""] * 23
        fields[0], fields[1], fields[2], fields[3], fields[4] = (
            stamp,
            moment.strftime("%a"),
            "S",
            ".. 00230C",
            stamp,
        )
        fields[5], fields[7], fields[8] = str(1000 + hour), "1", "-60"
        fields[9], fields[10], fields[11] = "9", "7", "7"
        fields[13], fields[14] = f"{gallons_total:.2f}", f"{gallons_per_hour:.2f}"
        fields[16], fields[17] = str(events_total), "3"
        fields[19], fields[20], fields[21] = str(minutes_total), "1", "3.603"
        records.append((moment, fields))

    lines: list[str] = []
    current_day = ""
    for moment, fields in sorted(records, key=lambda item: item[0], reverse=newest_first):
        day = f"{moment.month}/{moment.day}/{moment.year}"
        if day != current_day:
            lines.append(",".join([day] + [""] * 22))
            current_day = day
        lines.append(",".join(fields))
    return ("﻿" + "\r\n".join(lines) + "\r\n").encode("utf-8")
