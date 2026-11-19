"""Read an uploaded CSV file and recognise what kind of data it holds.

Two kinds of file are recognised from their contents, not their names:

* A CowMonitor temperature export: a header row with a timestamp column and a temperature
  column, usually also ``deviceId`` and ``Animal Id``. One file may hold several cows.
* An H2O Degree water-meter export: no header, one row per hourly meter report, and a
  date-only line before each day (newest first, as the H2O Degree website lists them).

Uploads are untrusted, so reading is bounded: a limited number of columns, rows and cows,
timestamps parsed only in known formats, and IDs reduced to plain characters.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

import pandas as pd

logger = logging.getLogger(__name__)

MIN_TEMPERATURE_C = 10.0
MAX_TEMPERATURE_C = 45.0
WATER_MIN_C = 0.0
WATER_MAX_C = 40.0
# A rumen reads 38-40 °C apart from brief drinking dips, and drinking water stays below 35 °C.
# A cow's file needs some rumen-like readings; a water file must not be mostly rumen-like.
RUMEN_C = 35.0
MIN_RUMEN_SHARE = 0.1
DEAD_SENSOR_C = 1.0
DEFAULT_INTERVAL_MINUTES = 5.0
MAX_COLUMNS = 60
MAX_ROWS = 500_000
MAX_COWS = 200
MAX_TIMESTAMP_CHARS = 40
MAX_ID_CHARS = 40
LIKELY_COPY_WARNING = (
    "This looks like CowMonitor's drinking-removed file: it has no alert column and no "
    "readings below 35 °C. The modules need the full temperature export (in the project's "
    "data, the *_raw.csv file). Use this file as the optional second file in the Daily "
    "drinking summary."
)
LOOKS_LIKE_WATER = (
    "These readings look like water temperatures (median {median:.1f} °C), not a cow's rumen "
    "temperature (about 38–40 °C). The cow's temperature export goes in step 1; a water "
    "temperature file goes in the settings of the Water intake estimate. (If this is a bolus "
    "file, the sensor may not have been inside the cow.)"
)
EARLIEST = pd.Timestamp("2000-01-01", tz="UTC")
LATEST = pd.Timestamp("2100-01-01", tz="UTC")

_TIMESTAMP_NAMES = ("timestamp", "datetime", "timestamputc", "datetimeutc", "readingtime", "ts")
_DATE_NAMES = ("date", "readingdate")
_TIME_NAMES = ("time",)
_TEMPERATURE_NAMES = ("temperature", "temp", "temperaturec", "tempc", "rumentemperature")
_DEVICE_NAMES = ("deviceid", "device", "bolusid", "sensorid", "deviceeui")
_ANIMAL_NAMES = ("animalid", "animal", "cowid", "cow", "animalnumber")

_TIMEZONE_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
_METER_DATETIME = re.compile(r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}$")
_NUMERIC_DATE = re.compile(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-]\d{2,4}")
_UNSAFE_ID_CHARACTERS = r"[^A-Za-z0-9 ._:-]"
_METER_MIN_FIELDS = 22

# Formats tried, in order, when timestamps are not ISO 8601.
_MONTH_FIRST_FORMATS = (
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m-%d-%Y %H:%M:%S",
    "%m-%d-%Y %H:%M",
)
_DAY_FIRST_FORMATS = (
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
)
_YEAR_FIRST_FORMATS = ("%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M")

# Positions (from 0) of the fields used from an H2O Degree hourly report. The meter keeps
# cumulative registers for gallons, flow events and minutes of flow; the website adds the
# change since the previous report and the battery voltage.
_METER_FIELDS = {
    0: "read_time",
    5: "report_number",
    13: "gallons_total",
    14: "gallons",
    16: "events_total",
    17: "events",
    19: "minutes_total",
    20: "minutes",
    21: "battery_v",
}


class FileKind(StrEnum):
    """What an uploaded file contains."""

    TEMPERATURE = "temperature"
    METER = "meter"


class Sensor(StrEnum):
    """Where a temperature sensor sits: in a cow's rumen (a bolus) or in the water trough."""

    RUMEN = "rumen"
    WATER = "water"


class UnsupportedFileError(ValueError):
    """The upload can't be read; the message is written to be shown to the user."""


@dataclass(frozen=True)
class Cow:
    """One animal in a temperature file, identified by its bolus (device) when possible."""

    key: str
    animal_id: str | None = None
    device_id: str | None = None

    @property
    def label(self) -> str:
        if self.animal_id and self.device_id:
            return f"Cow {self.animal_id} (device {self.device_id})"
        if self.animal_id:
            return f"Cow {self.animal_id}"
        if self.device_id:
            return f"Device {self.device_id}"
        return self.key


@dataclass(frozen=True)
class TemperatureData:
    """Readings with columns ``cow``, ``ts`` (UTC) and ``temperature`` (°C).

    Each "cow" is a rumen bolus, or for a water temperature file the one water sensor.
    """

    readings: pd.DataFrame
    cows: tuple[Cow, ...]
    interval_minutes: float
    warnings: tuple[str, ...] = ()

    @cached_property
    def _cows_by_key(self) -> dict[str, Cow]:
        return {cow.key: cow for cow in self.cows}

    @cached_property
    def _readings_by_cow(self) -> dict[str, pd.DataFrame]:
        return {
            str(key): group[["ts", "temperature"]].reset_index(drop=True)
            for key, group in self.readings.groupby("cow", sort=False)
        }

    def cow(self, key: str) -> Cow:
        return self._cows_by_key[key]

    def for_cow(self, key: str) -> pd.DataFrame:
        """One cow's readings, oldest first, with columns ``ts`` and ``temperature``."""
        return self._readings_by_cow[key]


@dataclass(frozen=True)
class MeterData:
    """Hourly water-meter reports, oldest first; ``read_time`` is the meter's local time."""

    readings: pd.DataFrame
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadedFile:
    """An uploaded file after it has been read and recognised."""

    name: str
    kind: FileKind
    sha256: str
    temperature: TemperatureData | None = None
    meter: MeterData | None = None


def load_upload(name: str, content: bytes) -> LoadedFile:
    """Read an uploaded CSV and recognise whether it is a temperature or a water-meter export."""
    text, rows = _read_rows(content, "The file")
    digest = hashlib.sha256(content).hexdigest()
    if _is_temperature_header(rows[0]):
        temperature = read_temperature(text, rows[0])
        logger.info("Read %r as temperature data: %d rows", name, len(temperature.readings))
        return LoadedFile(name, FileKind.TEMPERATURE, digest, temperature=temperature)
    if _is_meter_export(rows):
        meter = read_meter(text)
        logger.info("Read %r as water-meter data: %d rows", name, len(meter.readings))
        return LoadedFile(name, FileKind.METER, digest, meter=meter)
    raise UnsupportedFileError(
        "This doesn't look like a CowMonitor temperature export (columns such as deviceId, "
        "temperature and timestamp) or an H2O Degree water-meter export."
    )


def load_water_temperature(
    name: str, content: bytes, timezone: str | None = None
) -> TemperatureData:
    """Read a water temperature file: timestamp and temperature columns, optionally deviceId.

    The file must hold one water sensor (one trough); its readings come back as one "cow".
    Timestamps without a time zone are read in ``timezone`` (the farm's), if given: trough
    loggers usually record local time, unlike CowMonitor.
    """
    text, rows = _read_rows(content, "The file")
    if not _is_temperature_header(rows[0]):
        kind = "That is a water-meter export. " if _is_meter_export(rows) else ""
        raise UnsupportedFileError(
            f"{kind}A water temperature file needs a timestamp column and a temperature "
            "column (deviceId optional), like a CowMonitor export."
        )
    water = read_temperature(text, rows[0], sensor=Sensor.WATER, local_timezone=timezone)
    if len(water.cows) > 1:
        raise UnsupportedFileError(
            f"The file has {len(water.cows)} sensors "
            f"({summarise_names([cow.label for cow in water.cows])}). Upload the readings of "
            "the one trough this cow drinks from."
        )
    logger.info("Read %r as water temperature data: %d rows", name, len(water.readings))
    return water


def read_temperature(
    text: str,
    header: list[str],
    *,
    sensor: Sensor = Sensor.RUMEN,
    local_timezone: str | None = None,
) -> TemperatureData:
    """Parse a temperature export (whose first row is ``header``) into clean readings per cow.

    For a water ``sensor``, 0-40 °C is plausible instead of 10-45 °C and a file that reads like
    a rumen is refused; for a rumen sensor, a file that reads like water is refused. Timestamps
    without a time zone are read in ``local_timezone`` if given, otherwise as UTC.
    """
    positions = _positions_by_normalised_name(header)
    ts_at = _pick(positions, _TIMESTAMP_NAMES)
    date_at = _pick(positions, _DATE_NAMES)
    time_at = _pick(positions, _TIME_NAMES)
    temperature_at = _pick(positions, _TEMPERATURE_NAMES)
    device_at = _pick(positions, _DEVICE_NAMES)
    animal_at = _pick(positions, _ANIMAL_NAMES)
    if temperature_at is None:
        raise UnsupportedFileError("A temperature file needs a temperature column.")
    if ts_at is None and (date_at is None or time_at is None):
        raise UnsupportedFileError(
            "A temperature file needs a timestamp column (or separate date and time columns)."
        )

    wanted = sorted(
        {
            i
            for i in (ts_at, date_at, time_at, temperature_at, device_at, animal_at)
            if i is not None
        }
    )
    try:
        frame = pd.read_csv(
            io.StringIO(text),
            dtype=str,
            keep_default_na=False,
            usecols=wanted,
            nrows=MAX_ROWS + 1,
        )
    except (pd.errors.ParserError, ValueError) as exc:
        raise UnsupportedFileError(f"The file couldn't be read as a CSV table: {exc}") from exc
    if len(frame) > MAX_ROWS:
        raise UnsupportedFileError(
            f"The file has more than {MAX_ROWS:,} rows. Split it into smaller files "
            "(for example one per month) and upload them one at a time."
        )
    frame.columns = [f"c{i}" for i in wanted]
    frame = frame.fillna("")

    def column(position: int | None) -> pd.Series | None:
        return frame[f"c{position}"].astype(str).str.strip() if position is not None else None

    ts_text = column(ts_at)
    if ts_text is None:
        ts_text = column(date_at) + " " + column(time_at)  # type: ignore[operator]

    warnings: list[str] = []
    timestamps, had_timezone = _parse_timestamps(ts_text)
    if not had_timezone and local_timezone is not None:
        timestamps = _as_local_time(timestamps, local_timezone)
        warnings.append(
            f"The timestamps have no time zone, so they were read as farm time ({local_timezone})."
        )
    elif not had_timezone:
        warnings.append(
            "The timestamps have no time zone, so they were read as UTC "
            "(CowMonitor exports use UTC)."
        )
    device = _clean_ids(column(device_at))
    animal = _clean_ids(column(animal_at))
    readings = pd.DataFrame(
        {
            "cow": _cow_keys(device, animal, len(frame)),
            "ts": timestamps,
            "temperature": pd.to_numeric(column(temperature_at), errors="coerce"),
            "animal": animal if animal is not None else "",
            "device": device if device is not None else "",
        }
    )
    cow_count = readings["cow"].nunique()
    if cow_count > MAX_COWS:
        raise UnsupportedFileError(
            f"The file has {cow_count:,} devices; the app handles up to {MAX_COWS} at a time. "
            "Split it into smaller files."
        )

    all_cows = set(readings["cow"])
    unreadable = readings["ts"].isna() | readings["temperature"].isna()
    if unreadable.any():
        warnings.append(
            f"{int(unreadable.sum())} rows had an unreadable time or temperature and were skipped."
        )
    readings = readings[~unreadable]
    water = sensor is Sensor.WATER
    low, high = (WATER_MIN_C, WATER_MAX_C) if water else (MIN_TEMPERATURE_C, MAX_TEMPERATURE_C)
    implausible = ~readings["temperature"].between(low, high)
    if implausible.any():
        warnings.append(
            f"{int(implausible.sum())} readings outside {low:.0f}–{high:.0f} °C were skipped "
            "as sensor errors."
        )
    before = readings
    readings = readings[~implausible].sort_values(["cow", "ts"], kind="stable")
    repeated = readings.duplicated(["cow", "ts"], keep="first")
    if repeated.any():
        warnings.append(
            f"{int(repeated.sum())} repeated readings (same cow and time) were skipped."
        )
    readings = readings[~repeated].reset_index(drop=True)
    warnings.extend(_check_sensor(readings if not readings.empty else before, sensor))
    if readings.empty:
        raise UnsupportedFileError("No usable temperature readings were found in the file.")

    cows = tuple(_describe_cow(str(key), group) for key, group in readings.groupby("cow"))
    lost = sorted(all_cows - {cow.key for cow in cows})
    if lost:
        warnings.append(f"No usable readings for {summarise_names(lost)}, so they were left out.")
    span = readings["ts"].max() - readings["ts"].min()
    if (
        not water
        and "alert" not in positions
        and readings["temperature"].min() >= 35.0
        and span >= pd.Timedelta(days=1)
    ):
        warnings.append(LIKELY_COPY_WARNING)
    return TemperatureData(
        readings=readings[["cow", "ts", "temperature"]],
        cows=cows,
        interval_minutes=_typical_interval(readings),
        warnings=tuple(warnings),
    )


def read_meter(text: str) -> MeterData:
    """Parse an H2O Degree hourly export, keeping the gallons, flow-event and minute registers."""
    records: list[list[str]] = []
    try:
        for row in csv.reader(io.StringIO(text)):
            if _METER_MIN_FIELDS <= len(row) <= MAX_COLUMNS and _METER_DATETIME.match(
                row[0].strip()
            ):
                records.append(row)
                if len(records) > MAX_ROWS:
                    raise UnsupportedFileError(
                        f"The meter file has more than {MAX_ROWS:,} reports. Export a shorter "
                        "period."
                    )
    except csv.Error as exc:
        raise UnsupportedFileError(f"The meter file couldn't be read: {exc}") from exc
    if not records:
        raise UnsupportedFileError("No hourly meter reports were found in the file.")

    frame = pd.DataFrame(
        {name: [row[index].strip() for row in records] for index, name in _METER_FIELDS.items()}
    )
    frame["read_time"] = pd.to_datetime(
        frame["read_time"], format="%m/%d/%Y %H:%M", errors="coerce"
    )
    for name in _METER_FIELDS.values():
        if name != "read_time":
            frame[name] = pd.to_numeric(frame[name], errors="coerce")

    warnings: list[str] = []
    unreadable = frame["read_time"].isna() | frame["gallons_total"].isna()
    if unreadable.any():
        warnings.append(
            f"{int(unreadable.sum())} meter rows had no readable time or running total "
            "and were skipped."
        )
    frame = frame[~unreadable].copy()
    # A blank "this hour" value means nothing flowed in that hour.
    frame[["gallons", "events", "minutes"]] = frame[["gallons", "events", "minutes"]].fillna(0.0)
    # Sort and de-duplicate on the report number too: when clocks go back, the meter's local
    # time repeats an hour, and both reports in that hour are real.
    frame = frame.sort_values(["read_time", "report_number"], kind="stable")
    repeated = frame.duplicated(["read_time", "report_number"], keep="first")
    if repeated.any():
        warnings.append(f"{int(repeated.sum())} repeated meter reports were skipped.")
    frame = frame[~repeated].reset_index(drop=True)
    if frame.empty:
        raise UnsupportedFileError("No usable hourly meter reports were found in the file.")
    return MeterData(readings=frame, warnings=tuple(warnings))


def _check_sensor(readings: pd.DataFrame, sensor: Sensor) -> list[str]:
    """Refuse a water file where a cow's file belongs, and a cow's file where water belongs.

    A device whose readings are hardly ever rumen-like is water-like. A cow's file is refused
    only if every device in it is; otherwise those devices get a warning and still load, as
    they did before water files existed (a bolus may log for a while before it goes in).
    """
    if readings.empty:
        return []
    median = float(readings["temperature"].median())
    if sensor is Sensor.WATER:
        if RUMEN_C <= median <= MAX_TEMPERATURE_C:
            raise UnsupportedFileError(
                f"These readings look like a cow's rumen temperature (median {median:.1f} °C), "
                "not water. Upload the cow's file in step 1, and here the readings of the water "
                "trough."
            )
        return []
    by_device = readings.groupby("cow")["temperature"]
    rumen_share = by_device.agg(lambda values: float(values.ge(RUMEN_C).mean()))
    water_like = (rumen_share < MIN_RUMEN_SHARE) & by_device.median().ge(DEAD_SENSOR_C)
    if water_like.all():
        raise UnsupportedFileError(LOOKS_LIKE_WATER.format(median=median))
    if not water_like.any():
        return []
    names = summarise_names(sorted(str(key) for key in water_like.index[water_like]))
    return [
        f"The readings of {names} look like water or air temperatures, not a rumen (about "
        "38–40 °C): the bolus may not have been inside the cow."
    ]


def _as_local_time(timestamps: pd.Series, timezone: str) -> pd.Series:
    """Timestamps parsed as UTC but meant as local time, moved to UTC from that time zone.

    The repeated hour when clocks go back can't be placed, so it becomes unreadable.
    """
    wall_clock = timestamps.dt.tz_localize(None)
    local = wall_clock.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
    return local.dt.tz_convert("UTC")


def summarise_names(names: list[str], limit: int = 5) -> str:
    """'a, b, c' or 'a, b, c, d, e and 12 more', so long lists stay readable."""
    if len(names) <= limit:
        return ", ".join(names)
    return f"{', '.join(names[:limit])} and {len(names) - limit} more"


def _read_rows(content: bytes, what: str) -> tuple[str, list[list[str]]]:
    """Decode an upload and read its first rows, refusing empty and over-wide files."""
    if not content.strip():
        raise UnsupportedFileError(f"{what} is empty.")
    text = _decode(content)
    rows = _first_rows(text, limit=25)
    if not rows:
        raise UnsupportedFileError(f"{what} is empty.")
    widest = max(len(row) for row in rows)
    if widest > MAX_COLUMNS:
        raise UnsupportedFileError(
            f"{what} has rows with {widest:,} columns; the app reads files with up to "
            f"{MAX_COLUMNS}."
        )
    return text, rows


def _decode(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("cp1252", errors="replace")


def _first_rows(text: str, limit: int) -> list[list[str]]:
    rows: list[list[str]] = []
    try:
        for row in csv.reader(io.StringIO(text)):
            if any(cell.strip() for cell in row):
                rows.append(row)
                if len(rows) >= limit:
                    break
    except csv.Error as exc:
        raise UnsupportedFileError(f"The file couldn't be read as CSV: {exc}") from exc
    return rows


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())


def _positions_by_normalised_name(header: list[str]) -> dict[str, int]:
    found: dict[str, int] = {}
    for position, name in enumerate(header):
        found.setdefault(_normalise(name), position)
    return found


def _pick(positions: dict[str, int], candidates: tuple[str, ...]) -> int | None:
    for candidate in candidates:
        if candidate in positions:
            return positions[candidate]
    return None


def _is_temperature_header(row: list[str]) -> bool:
    names = {_normalise(cell) for cell in row}
    has_time = bool(names & set(_TIMESTAMP_NAMES)) or (
        bool(names & set(_DATE_NAMES)) and bool(names & set(_TIME_NAMES))
    )
    return bool(names & set(_TEMPERATURE_NAMES)) and has_time


def _is_meter_export(rows: list[list[str]]) -> bool:
    return any(
        len(row) >= _METER_MIN_FIELDS and _METER_DATETIME.match(row[0].strip()) for row in rows
    )


def _parse_timestamps(text: pd.Series) -> tuple[pd.Series, bool]:
    """Parse timestamps to UTC in known formats only; also report whether they had a time zone.

    ISO 8601 (what CowMonitor exports) is tried first, then a short list of numeric date
    formats. Free-form guessing is never used: it is slow on hostile input and silently
    misreads ambiguous dates.
    """
    text = text.where(text.str.len() <= MAX_TIMESTAMP_CHARS, "")
    present = text[text.ne("")]
    had_timezone = present.empty or bool(present.head(500).str.contains(_TIMEZONE_SUFFIX).any())
    parsed = pd.to_datetime(text, utc=True, format="ISO8601", errors="coerce")
    if not present.empty and parsed.notna().mean() < 0.5:
        parsed = _parse_known_format(text, present)
    return parsed.where(parsed.between(EARLIEST, LATEST)), had_timezone


def _parse_known_format(text: pd.Series, present: pd.Series) -> pd.Series:
    formats = _DAY_FIRST_FORMATS if _is_day_first(present) else _MONTH_FIRST_FORMATS
    sample = present.head(200)
    for candidate in formats + _YEAR_FIRST_FORMATS:
        if pd.to_datetime(sample, format=candidate, errors="coerce").notna().mean() >= 0.9:
            return pd.to_datetime(text, utc=True, format=candidate, errors="coerce")
    raise UnsupportedFileError(
        "The timestamps aren't in a format the app reads. Use ISO format, such as "
        "2024-04-06T04:01:00Z (as CowMonitor exports), or month/day/year such as "
        "04/06/2024 04:01."
    )


def _is_day_first(values: pd.Series) -> bool:
    """Decide whether numeric dates such as 06/04/2024 are day-first; refuse to guess if unclear."""
    parts = values.str.extract(_NUMERIC_DATE).dropna()
    if parts.empty:
        return False
    first, second = parts[0].astype(int), parts[1].astype(int)
    if (first > 12).any() and not (second > 12).any():
        return True
    if (second > 12).any() and not (first > 12).any():
        return False
    raise UnsupportedFileError(
        "The dates could be read as day/month or month/day (for example 06/04/2024). Export "
        "the timestamps in ISO format, such as 2024-04-06T04:01:00Z, and upload again."
    )


def _clean_ids(values: pd.Series | None) -> pd.Series | None:
    """IDs reduced to letters, digits and . _ : - (others become _) and at most 40 characters."""
    if values is None:
        return None
    ids = values.str.strip('"').str.replace(_UNSAFE_ID_CHARACTERS, "_", regex=True)
    ids = ids.str.slice(0, MAX_ID_CHARS).str.strip()
    return ids if ids.ne("").any() else None


def _cow_keys(device: pd.Series | None, animal: pd.Series | None, rows: int) -> pd.Series:
    if device is not None:
        return device.where(device.ne(""), "unknown device")
    if animal is not None:
        return animal.where(animal.ne(""), "unknown cow")
    return pd.Series(["all readings"] * rows)


def _most_common(values: pd.Series) -> str | None:
    present = values[values.ne("")]
    return str(present.mode().iloc[0]) if not present.empty else None


def _describe_cow(key: str, group: pd.DataFrame) -> Cow:
    return Cow(
        key=key, animal_id=_most_common(group["animal"]), device_id=_most_common(group["device"])
    )


def _typical_interval(readings: pd.DataFrame) -> float:
    gaps = readings.groupby("cow")["ts"].diff().dt.total_seconds().div(60).dropna()
    gaps = gaps[gaps > 0]
    return float(gaps.median()) if not gaps.empty else DEFAULT_INTERVAL_MINUTES
