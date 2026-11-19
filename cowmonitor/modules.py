"""The modules the app offers, and how each one turns an uploaded file into results."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from cowmonitor import charts
from cowmonitor.abnormal import (
    HIGH_RULE,
    LOW_RULE,
    MIN_DAYS,
    REFERENCE_DAYS,
    WARN_DAYS,
    AbnormalResult,
    LimitsMode,
    check_abnormal,
)
from cowmonitor.calving import CalvingResult, CalvingVersion, check_calving
from cowmonitor.drinking import (
    ESTIMATED_WARNING,
    NOTEBOOK_FEATURES,
    DrinkingMarks,
    DrinkingSource,
    daily_summary,
    estimate_marks,
    marks_from_copy,
)
from cowmonitor.intake import (
    LITRES_PER_GALLON,
    MIN_DAY_COVERAGE,
    RECOVERY_HOURS,
    REFERENCE_WATER_C,
    IntakeEstimate,
    estimate_intake,
)
from cowmonitor.loaders import (
    LIKELY_COPY_WARNING,
    WATER_MAX_C,
    WATER_MIN_C,
    Cow,
    FileKind,
    LoadedFile,
    MeterData,
    TemperatureData,
    UnsupportedFileError,
    summarise_names,
)
from cowmonitor.results import Chart, Metric, ModuleResult, Table
from cowmonitor.timezones import DEFAULT_TIMEZONE, validate_timezone
from cowmonitor.water import summarise_water

DRINKING_REMOVED_MIN_C = 35.0
LOW_COVERAGE = 0.8
CALVING_WARN_DAYS = 6.0
FULL_DAY_COVERAGE = 90.0


@dataclass(frozen=True)
class ModuleSpec:
    key: str
    title: str
    summary: str
    accepts: FileKind
    experimental: bool = False


MODULES: tuple[ModuleSpec, ...] = (
    ModuleSpec(
        "abnormal",
        "Abnormal drinking check",
        "Model 1: too few deep drinking drops, or a long high-temperature period.",
        FileKind.TEMPERATURE,
    ),
    ModuleSpec(
        "calving",
        "Calving alert",
        "Model 2: the temperature drop before calving, and whether it rises again.",
        FileKind.TEMPERATURE,
        experimental=True,
    ),
    ModuleSpec(
        "drinking",
        "Daily drinking summary",
        "Drinking dips per day and the project's 8 daily drinking numbers.",
        FileKind.TEMPERATURE,
    ),
    ModuleSpec(
        "intake",
        "Water intake estimate",
        "Litres and gallons drunk per day, from the drinking dips and the water temperature.",
        FileKind.TEMPERATURE,
        experimental=True,
    ),
    ModuleSpec(
        "water",
        "Daily water use",
        "Gallons per day from a water-meter export, for the cows on that meter.",
        FileKind.METER,
    ),
)


# Switched off for now at the owner's request (19 November 2026): the page doesn't offer
# these, but their code and tests stay. Take a key out of this set to offer it again.
HIDDEN_MODULES: frozenset[str] = frozenset({"abnormal", "calving", "drinking"})


def offered_modules() -> tuple[ModuleSpec, ...]:
    """The modules the page offers, in menu order."""
    return tuple(spec for spec in MODULES if spec.key not in HIDDEN_MODULES)


def modules_for(kind: FileKind) -> tuple[ModuleSpec, ...]:
    """The offered modules that can run on this kind of file."""
    return tuple(spec for spec in offered_modules() if spec.accepts is kind)


def get_module(key: str) -> ModuleSpec:
    for spec in MODULES:
        if spec.key == key:
            return spec
    raise KeyError(key)


# The app runs one method per model, the recommended one: per-day limits for the abnormal
# check and the fixed calving version. The notebook's originals stay in abnormal.py and
# calving.py only for the tests that compare them with the 2024 results.
ABNORMAL_LIMITS = LimitsMode.PER_DAY
CALVING_VERSION = CalvingVersion.FIXED


@dataclass(frozen=True)
class AbnormalOptions:
    cow: str
    timezone: str = DEFAULT_TIMEZONE


@dataclass(frozen=True)
class CalvingOptions:
    cow: str
    drinking_removed: bool = False
    timezone: str = DEFAULT_TIMEZONE


@dataclass(frozen=True)
class DrinkingOptions:
    cow: str
    timezone: str = DEFAULT_TIMEZONE
    copy: TemperatureData | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class IntakeOptions:
    cow: str
    timezone: str = DEFAULT_TIMEZONE
    water_c: float = REFERENCE_WATER_C
    copy: TemperatureData | None = field(default=None, compare=False, repr=False)
    water: TemperatureData | None = field(default=None, compare=False, repr=False)
    water_name: str = ""


@dataclass(frozen=True)
class WaterOptions:
    meter_name: str = ""
    cows_on_meter: str = ""


ModuleOptions = AbnormalOptions | CalvingOptions | DrinkingOptions | IntakeOptions | WaterOptions


def run_module(file: LoadedFile, key: str, options: ModuleOptions) -> ModuleResult:
    """Run one module on a loaded file; raises UnsupportedFileError with a user-facing message."""
    spec = get_module(key)
    if file.kind is not spec.accepts:
        raise UnsupportedFileError(
            f"{spec.title} needs {_kind_phrase(spec.accepts)}, but this file is "
            f"{_kind_phrase(file.kind)}."
        )
    if not isinstance(options, WaterOptions):
        validate_timezone(options.timezone)
    if key == "abnormal" and isinstance(options, AbnormalOptions):
        return _run_abnormal(_temperature(file), options)
    if key == "calving" and isinstance(options, CalvingOptions):
        return _run_calving(_temperature(file), options)
    if key == "drinking" and isinstance(options, DrinkingOptions):
        return _run_drinking(_temperature(file), options)
    if key == "intake" and isinstance(options, IntakeOptions):
        return _run_intake(_temperature(file), options)
    if key == "water" and isinstance(options, WaterOptions):
        return _run_water(_meter(file), options)
    raise TypeError(f"{type(options).__name__} doesn't fit module {key!r}.")


# --- Model 1: abnormal drinking -------------------------------------------------------------


def _run_abnormal(data: TemperatureData, options: AbnormalOptions) -> ModuleResult:
    cow = data.cow(options.cow)
    results = {
        c.key: check_abnormal(data.for_cow(c.key), ABNORMAL_LIMITS, data.interval_minutes)
        for c in data.cows
    }
    result = results[cow.key]
    readings = data.for_cow(cow.key)

    by_cow = pd.DataFrame([_abnormal_row(c, results[c.key]) for c in data.cows])
    rule = pd.DataFrame(
        [
            {
                "Check": check.description,
                "Count": check.count,
                "Alert needs": f"{check.comparison} {check.limit:.1f}",
                "Met": "Yes" if check.met else "No",
            }
            for check in result.checks
        ]
    )
    flagged = sum(r.abnormal for r in results.values())
    judged = sum(r.enough_data for r in results.values())
    return ModuleResult(
        title=f"Abnormal drinking check: {cow.label}",
        subtitle=result.reason,
        metrics=(
            Metric("Verdict", result.verdict),
            Metric("Days of data", f"{result.days:.1f}"),
            Metric("Limits", f"Per day (×{result.scale:.2f})"),
            Metric("Abnormal cows in file", f"{flagged} of {judged}"),
        ),
        charts=(
            Chart(
                "Counts against limits",
                charts.abnormal_counts_chart(result, f"Counts against limits: {cow.label}"),
                "Abnormal if all three low counts are below their limits (too few deep drinking "
                "drops), or both high counts are above theirs (a long high-temperature period).",
            ),
            Chart(
                "Temperature",
                charts.temperature_chart(readings, options.timezone, f"Temperature: {cow.label}"),
                "Dotted lines: 32, 33.5 and 34.5 °C (deep drinking drops) and 39 and 40 °C "
                "(high temperature), the levels this check counts. Breaks in the line are gaps "
                "over 1 hour.",
            ),
        ),
        tables=(
            Table("Results for every cow in the file", by_cow, "abnormal_results.csv"),
            Table(f"Rule check: {cow.label}", rule, f"abnormal_rule_{_slug(cow)}.csv"),
        ),
        warnings=tuple(_abnormal_warnings(data, results)),
        notes=(
            "Rule from the project (model 1): abnormal if a cow has fewer than "
            + ", ".join(f"{count} readings below {temp:g} °C" for temp, count in LOW_RULE)
            + ", or more than "
            + " and ".join(f"{count} above {temp:g} °C" for temp, count in HIGH_RULE)
            + f", per {REFERENCE_DAYS:g} days of data.",
            "On the project's 21 test cows the original rule matched all 21 labels. With per-day "
            "limits it matched 20 or 21 of 21 on 2- and 3-day slices of the same files, where the "
            "original counts managed 14 to 19. These are the cows the rule was built on, so "
            "accuracy on new cows is unknown.",
        ),
    )


def _abnormal_row(cow: Cow, result: AbnormalResult) -> dict[str, object]:
    row: dict[str, object] = {
        "Cow": cow.label,
        "Verdict": result.verdict,
        "Reason": result.reason,
        "Days": round(result.days, 1),
    }
    for check in result.checks:
        row[check.description] = check.count
    return row


def _abnormal_warnings(data: TemperatureData, results: dict[str, AbnormalResult]) -> list[str]:
    no_verdict, gappy, short, removed = [], [], [], []
    for cow in data.cows:
        result = results[cow.key]
        readings = data.for_cow(cow.key)
        coverage = len(readings) * data.interval_minutes / (result.days * 1440)
        if not result.enough_data:
            no_verdict.append(cow.label)
            continue
        if coverage < LOW_COVERAGE:
            gappy.append(cow.label)
        elif result.days < WARN_DAYS:
            short.append(cow.label)
        if _looks_drinking_removed(readings):
            removed.append(cow.label)
    messages = [
        (no_verdict, f"Less than {MIN_DAYS:g} day of data, so no verdict: {{}}."),
        (
            gappy,
            "Gaps in the data (under 80% of the expected readings) for {}. Per-day limits "
            "assume steady data, so these verdicts may lean to 'Abnormal'.",
        ),
        (
            short,
            "Less than 2 days of data for {}. On 1-day slices of the test files the check was "
            "right 90–95% of the time.",
        ),
        (removed, _DRINKING_REMOVED_MESSAGE),
    ]
    return list(data.warnings) + [
        template.format(summarise_names(names)) for names, template in messages if names
    ]


# --- Model 2: calving ----------------------------------------------------------------------


def _run_calving(data: TemperatureData, options: CalvingOptions) -> ModuleResult:
    cow = data.cow(options.cow)
    results = {
        c.key: check_calving(
            data.for_cow(c.key), CALVING_VERSION, options.drinking_removed, data.interval_minutes
        )
        for c in data.cows
    }
    result = results[cow.key]
    timezone = options.timezone
    first = result.first_calving
    by_cow = pd.DataFrame([_calving_row(c, results[c.key], timezone) for c in data.cows])
    metrics = [
        Metric("Calving-like drop", _calving_verdict(result)),
        Metric("First drop", _half_day_label(first.start, timezone) if first else "–"),
        Metric("Drop size", f"{first.change:+.2f} °C" if first else "–"),
        Metric("Temperature rises again", _recovery_verdict(result)),
        Metric("Half-days used", str(int(result.half_days["used"].sum()))),
    ]
    return ModuleResult(
        title=f"Calving alert: {cow.label}",
        subtitle=f"Half-days are 00:00–12:00 and 12:00–24:00 UTC, shown in {timezone}.",
        metrics=tuple(metrics),
        charts=(
            Chart(
                "Half-day temperatures",
                charts.half_day_chart(
                    result, timezone, f"Half-day median temperature: {cow.label}"
                ),
                "Each point is the median of a half-day's readings between 37 and 40 °C. A drop of "
                "more than 0.5 °C below the average of the previous four half-days is flagged.",
            ),
        ),
        tables=(
            Table("Results for every cow in the file", by_cow, "calving_results.csv"),
            Table(
                f"Half-days: {cow.label}",
                _half_day_frame(result, timezone),
                f"calving_half_days_{_slug(cow)}.csv",
            ),
        ),
        warnings=tuple(_calving_warnings(data, results)),
        notes=(
            "The project's model 2: half-day medians of readings between 37 and 40 °C; a calving-"
            "like drop is a half-day more than 0.5 °C below the average of the previous four; it "
            "rises again if a later half-day is more than 0.4 °C below the average of itself and "
            "the next nine.",
        ),
    )


def _calving_verdict(result: CalvingResult) -> str:
    if not result.enough_data:
        return "Not enough data"
    return "Yes" if result.calving else "No"


def _recovery_verdict(result: CalvingResult) -> str:
    if not result.calving:
        return "–"
    if not result.recovery_checked:
        return "Not checked (too little data after the drop)"
    return "Yes" if result.recovered else "No"


def _calving_row(cow: Cow, result: CalvingResult, timezone: str) -> dict[str, object]:
    first = result.first_calving
    return {
        "Cow": cow.label,
        "Calving-like drop": _calving_verdict(result),
        "First drop": _half_day_label(first.start, timezone) if first else "",
        "Drop °C": round(first.change, 2) if first else None,
        "Rises again": _recovery_verdict(result),
        "Half-days used": int(result.half_days["used"].sum()),
    }


def _half_day_label(start: pd.Timestamp, timezone: str) -> str:
    begin = start.tz_convert(timezone)
    end = (start + pd.Timedelta(hours=12)).tz_convert(timezone)
    return f"{begin:%d %b %H:%M} – {end:%d %b %H:%M}"


def _half_day_frame(result: CalvingResult, timezone: str) -> pd.DataFrame:
    table = result.half_days
    if table.empty:
        return pd.DataFrame(columns=["Half-day", "Median °C"])
    flagged = {event.start for event in result.calving_events}
    rising = {event.start for event in result.recovery_events}
    frame = pd.DataFrame({"Half-day": [_half_day_label(s, timezone) for s in table["start"]]})
    for source, target in (
        ("readings", "Readings"),
        ("readings_used", "Readings in 37–40 °C"),
        ("median_c", "Median °C"),
        ("baseline_c", "Average of previous 4 °C"),
        ("change_c", "Change °C"),
    ):
        if source in table.columns:
            frame[target] = table[source].round(2).to_numpy()
    frame["Flag"] = [
        _flag(start, flagged, rising, table, i) for i, start in enumerate(table["start"])
    ]
    return frame


def _flag(
    start: pd.Timestamp,
    flagged: set[pd.Timestamp],
    rising: set[pd.Timestamp],
    table: pd.DataFrame,
    position: int,
) -> str:
    labels = []
    if start in flagged:
        labels.append("calving-like drop")
    if start in rising:
        labels.append("rises again after this")
    if "note" in table.columns and table["note"].iloc[position]:
        labels.append(f"skipped: {table['note'].iloc[position]}")
    return "; ".join(labels)


def _calving_warnings(data: TemperatureData, results: dict[str, CalvingResult]) -> list[str]:
    warnings = list(data.warnings)
    warnings.append(
        "Experimental. On the project's 16 test cows this alert was right for 12 (75%): it "
        "missed 3 of 7 calvings and raised 1 false alarm."
    )
    no_verdict, short = [], []
    for cow in data.cows:
        readings = data.for_cow(cow.key)
        days = (readings["ts"].iloc[-1] - readings["ts"].iloc[0]).total_seconds() / 86400
        if not results[cow.key].enough_data:
            no_verdict.append(cow.label)
        elif days < CALVING_WARN_DAYS:
            short.append(cow.label)
    if no_verdict:
        warnings.append(
            "Fewer than 5 usable half-days (about 2½ days), so no verdict: "
            f"{summarise_names(no_verdict)}."
        )
    if short:
        warnings.append(
            f"Less than 6 days of data for {summarise_names(short)}; the test files covered "
            "6 to 30 days."
        )
    return warnings


# --- Daily drinking summary -----------------------------------------------------------------


def _run_drinking(data: TemperatureData, options: DrinkingOptions) -> ModuleResult:
    cow = data.cow(options.cow)
    marks, warnings = _drinking_marks(data, cow, options.copy)
    table = daily_summary(marks, options.timezone, data.interval_minutes)
    full = table[table["Coverage %"] >= FULL_DAY_COVERAGE]
    reference = full if not full.empty else table
    partial = table.loc[table["Coverage %"] < FULL_DAY_COVERAGE, "Date"].astype(str).tolist()
    if partial:
        warnings.append(
            f"Partial days (under {FULL_DAY_COVERAGE:.0f}% of readings): {', '.join(partial)}. "
            "Averages use full days only."
        )
    dips_per_day = float(reference["Drinking dips"].mean())
    source = _source_label(marks)
    return ModuleResult(
        title=f"Daily drinking summary: {cow.label}",
        subtitle=f"Drinking readings from: {source}. Days are in {options.timezone}.",
        metrics=(
            Metric("Days", str(len(table))),
            Metric("Drinking dips per day", f"{dips_per_day:.1f}"),
            Metric("Drinking readings per day", f"{reference['Drinking readings'].mean():.1f}"),
            Metric(
                "Coldest drinking reading", _degrees(table["Coldest drinking reading °C"].min())
            ),
            Metric("Source", source),
        ),
        charts=(
            Chart(
                "Drinking dips per day",
                charts.daily_bar_chart(
                    table,
                    "Drinking dips",
                    f"Drinking dips per day: {cow.label}",
                    "Dips",
                    dips_per_day,
                ),
                "A dip is a run of drinking readings with no gap over 15 minutes.",
            ),
            Chart(
                "Temperature with drinking readings",
                charts.temperature_chart(
                    marks.readings,
                    options.timezone,
                    f"Temperature with drinking readings: {cow.label}",
                    drinking=marks.readings["drinking"],
                    thresholds=(32.5, 33.0, 33.5),
                ),
            ),
        ),
        tables=(
            Table(f"Daily drinking summary: {cow.label}", table, f"drinking_{_slug(cow)}.csv"),
        ),
        warnings=tuple(warnings),
        notes=(
            "Columns match the project's water-intake features: "
            + "; ".join(f"{friendly} = {name}" for friendly, name in NOTEBOOK_FEATURES.items())
            + ". Drinking dips and coverage are added by this app.",
        ),
    )


def _drinking_marks(
    data: TemperatureData, cow: Cow, copy: TemperatureData | None
) -> tuple[DrinkingMarks, list[str]]:
    """Drinking readings from CowMonitor's copy if given, else estimated; plus the warnings."""
    readings = data.for_cow(cow.key)
    warnings = list(data.warnings)
    if copy is not None:
        marks = marks_from_copy(readings, copy.for_cow(_matching_copy_cow(cow, copy)))
        warnings.extend(
            f"Drinking-removed file: {w}" for w in copy.warnings if w != LIKELY_COPY_WARNING
        )
    else:
        marks = estimate_marks(readings)
    warnings.extend(marks.warnings)
    return marks, warnings


def _matching_copy_cow(cow: Cow, copy: TemperatureData) -> str:
    for candidate in copy.cows:
        if candidate.key == cow.key:
            return candidate.key
        if cow.device_id and candidate.device_id == cow.device_id:
            return candidate.key
        if cow.animal_id and candidate.animal_id == cow.animal_id and not candidate.device_id:
            return candidate.key
    if len(copy.cows) == 1 and not (copy.cows[0].device_id and cow.device_id):
        return copy.cows[0].key
    raise UnsupportedFileError(
        f"The drinking-removed file has no data for {cow.label}; it holds "
        f"{summarise_names([c.label for c in copy.cows])}."
    )


# --- Water intake estimate -----------------------------------------------------------------


def _run_intake(data: TemperatureData, options: IntakeOptions) -> ModuleResult:
    if not WATER_MIN_C <= options.water_c <= WATER_MAX_C:
        raise ValueError(f"Water temperature must be {WATER_MIN_C:g}–{WATER_MAX_C:g} °C.")
    cow = data.cow(options.cow)
    marks, warnings = _drinking_marks(data, cow, options.copy)
    # The "Drinking readings from" metric already says when they are estimated.
    warnings = [warning for warning in warnings if warning != ESTIMATED_WARNING]
    water = None
    if options.water is not None:
        water = options.water.for_cow(options.water.cows[0].key)
        warnings.extend(f"Water temperature file: {w}" for w in options.water.warnings)
    estimate = estimate_intake(
        marks,
        interval_minutes=data.interval_minutes,
        timezone=options.timezone,
        water=water,
        default_water_c=options.water_c,
    )
    warnings.extend(estimate.warnings)
    warnings.extend(_sparse_day_warnings(estimate.coverage))
    return ModuleResult(
        title=f"Water intake estimate: {cow.label}",
        subtitle="Heat balance: litres = K × dip area ÷ (normal − water temperature). Days are "
        f"in {options.timezone}.",
        metrics=_intake_metrics(estimate, marks, options.water_c),
        charts=_intake_charts(estimate, water, cow, options),
        tables=(
            Table(
                f"Estimated water intake per day: {cow.label}",
                estimate.daily,
                f"water_intake_{_slug(cow)}.csv",
            ),
            Table(
                f"Drinks: {cow.label}",
                _drinks_frame(estimate, options.timezone),
                f"drinks_{_slug(cow)}.csv",
            ),
        ),
        warnings=tuple(warnings),
    )


def _intake_metrics(
    estimate: IntakeEstimate, marks: DrinkingMarks, setting_c: float
) -> tuple[Metric, ...]:
    daily = estimate.daily
    average = _average_gallons(daily)
    days = int(daily["Litres"].notna().sum())
    full = daily[daily["Coverage %"] >= FULL_DAY_COVERAGE]
    drinks_per_day = float((full if not full.empty else daily)["Drinks"].mean())
    return (
        Metric(
            "Average per day",
            (
                f"{average:.1f} gal ({average * LITRES_PER_GALLON:.0f} L)"
                if average is not None
                else "–"
            ),
        ),
        Metric(
            "Total", f"{daily['Gallons'].sum():,.1f} gal over {days} day{'' if days == 1 else 's'}"
        ),
        Metric("Drinks per day", f"{drinks_per_day:.1f}"),
        Metric("Water temperature", _water_label(estimate, setting_c)),
        Metric("Drinking readings from", _source_label(marks)),
    )


def _intake_charts(
    estimate: IntakeEstimate, water: pd.DataFrame | None, cow: Cow, options: IntakeOptions
) -> tuple[Chart, ...]:
    figures = [
        Chart(
            "Estimated gallons per day",
            charts.daily_bar_chart(
                estimate.daily,
                "Gallons",
                f"Estimated gallons per day: {cow.label}",
                "Gallons",
                _average_gallons(estimate.daily),
            ),
            "Each day adds up its drinks. Days with gaps in the readings are scaled up for the "
            f"missing time; days under {MIN_DAY_COVERAGE:.0f}% covered, or with a drink that "
            "couldn't be estimated, have no bar.",
        ),
        Chart(
            "Temperature and the cow's normal",
            charts.temperature_chart(
                estimate.readings,
                options.timezone,
                f"Temperature and drinks: {cow.label}",
                drinking=estimate.readings["drinking"],
                thresholds=(),
                normal=estimate.readings["normal"],
            ),
            "Each drink's dip area lies between the dashed normal line and the readings, from the "
            f"start of the drink until {RECOVERY_HOURS:g} hours later or the next drink.",
        ),
    ]
    if water is not None:
        name = options.water_name.strip() or "the water file"
        figures.append(
            Chart(
                "Water temperature",
                charts.temperature_chart(
                    water, options.timezone, f"Water temperature: {name}", thresholds=()
                ),
            )
        )
    return tuple(figures)


def _average_gallons(daily: pd.DataFrame) -> float | None:
    estimated = daily["Gallons"].dropna()
    return float(estimated.mean()) if not estimated.empty else None


def _water_label(estimate: IntakeEstimate, setting_c: float) -> str:
    drinks = len(estimate.drinks)
    if not estimate.water_from_file:
        return f"{setting_c:.1f} °C (setting)"
    if estimate.drinks_from_file == drinks:
        return f"{estimate.water_c:.1f} °C (median, from file)"
    return f"{estimate.water_c:.1f} °C (file, {estimate.drinks_from_file} of {drinks} drinks)"


def _source_label(marks: DrinkingMarks) -> str:
    if marks.source is DrinkingSource.COWMONITOR_COPY:
        return "CowMonitor's drinking-removed file"
    return "Estimated"


def _drinks_frame(estimate: IntakeEstimate, timezone: str) -> pd.DataFrame:
    drinks = estimate.drinks
    return pd.DataFrame(
        {
            "Start": drinks["start"].dt.tz_convert(timezone).dt.strftime("%Y-%m-%d %H:%M"),
            "Drinking minutes": drinks["minutes"].round(1),
            "Coldest °C": drinks["coldest_c"].round(2),
            "Normal °C": drinks["normal_c"].round(2),
            "Drop °C": (drinks["normal_c"] - drinks["coldest_c"]).round(2),
            "Dip area °C·h": drinks["area_ch"].round(3),
            "Water °C": drinks["water_c"].round(1),
            "Water from": drinks["water_from_file"].map({True: "file", False: "setting"}),
            "Litres": drinks["litres"].round(2),
            "Gallons": (drinks["litres"] / LITRES_PER_GALLON).round(3),
        }
    )


def _sparse_day_warnings(coverage: pd.Series) -> list[str]:
    """Days with too few readings for an estimate (days with gaps are scaled up silently)."""
    sparse = coverage[coverage < MIN_DAY_COVERAGE]
    if sparse.empty:
        return []
    return [
        f"No estimate for days under {MIN_DAY_COVERAGE:.0f}% covered by readings: "
        f"{_day_list(sparse)}."
    ]


def _day_list(coverage: pd.Series) -> str:
    # Whole percent, rounded down, so a day just under a limit never shows as on it.
    return summarise_names([f"{day} ({int(cover)}%)" for day, cover in coverage.items()])


# --- Daily water use -----------------------------------------------------------------------


def _run_water(meter: MeterData, options: WaterOptions) -> ModuleResult:
    summary = summarise_water(meter)
    daily = summary.daily
    full = daily[daily["Full day"]]
    reference = full if not full.empty else daily
    highest = reference.loc[reference["Gallons"].idxmax()]
    lowest = reference.loc[reference["Gallons"].idxmin()]
    name = options.meter_name.strip() or "this meter"
    cows = options.cows_on_meter.strip()
    average = summary.average_gallons_full_day
    return ModuleResult(
        title=f"Daily water use: {name}" + (f" (cows {cows})" if cows else ""),
        subtitle=f"{summary.first_read:%d %b %Y %H:%M} to {summary.last_read:%d %b %Y %H:%M}, "
        "meter's local time",
        metrics=(
            Metric("Total gallons", f"{summary.total_gallons:,.1f}"),
            Metric("Average per full day", f"{average:.1f} gal" if average is not None else "–"),
            Metric("Highest day", f"{highest['Gallons']:.1f} gal ({highest['Date']:%d %b})"),
            Metric("Lowest day", f"{lowest['Gallons']:.1f} gal ({lowest['Date']:%d %b})"),
            Metric("Flow events", f"{summary.total_events:,}"),
            Metric("Minutes of flow", f"{summary.total_minutes:,.0f}"),
        ),
        charts=(
            Chart(
                "Gallons per day",
                charts.daily_bar_chart(
                    daily, "Gallons", f"Gallons per day: {name}", "Gallons", average
                ),
            ),
            Chart(
                "Gallons by hour of day",
                charts.hourly_chart(summary.by_hour, f"Average gallons by hour of day: {name}"),
                "When the cows on this meter drink, averaged over full days.",
            ),
        ),
        tables=(Table("Daily water use", daily, f"water_{_slug_text(name)}.csv"),),
        warnings=summary.warnings,
        notes=(
            "Gallons are for all cows drinking from this meter together; a shared meter can't "
            "split them between cows.",
            f"Missed hourly reports: {summary.lost_reports}. No water is lost: the meter's totals "
            "are cumulative, so the next report includes it.",
        ),
    )


# --- Shared helpers ------------------------------------------------------------------------


def _temperature(file: LoadedFile) -> TemperatureData:
    if file.temperature is None:
        raise UnsupportedFileError("This file has no temperature readings.")
    return file.temperature


def _meter(file: LoadedFile) -> MeterData:
    if file.meter is None:
        raise UnsupportedFileError("This file has no water-meter readings.")
    return file.meter


def _kind_phrase(kind: FileKind) -> str:
    return "a temperature export" if kind is FileKind.TEMPERATURE else "a water-meter export"


_DRINKING_REMOVED_MESSAGE = (
    f"No readings below {DRINKING_REMOVED_MIN_C:g} °C for {{}}. If this is CowMonitor's "
    "drinking-removed file, upload the full temperature export instead: without drinking "
    "drops every cow looks abnormal."
)


def _looks_drinking_removed(readings: pd.DataFrame) -> bool:
    days = (readings["ts"].iloc[-1] - readings["ts"].iloc[0]).total_seconds() / 86400
    return days >= MIN_DAYS and readings["temperature"].min() >= DRINKING_REMOVED_MIN_C


def _degrees(value: float) -> str:
    return "–" if pd.isna(value) else f"{value:.1f} °C"


def _slug(cow: Cow) -> str:
    return _slug_text(cow.animal_id or cow.device_id or cow.key)


def _slug_text(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text.strip())
    return cleaned.strip("_") or "meter"


__all__ = [
    "MODULES",
    "AbnormalOptions",
    "CalvingOptions",
    "DrinkingOptions",
    "IntakeOptions",
    "ModuleOptions",
    "ModuleSpec",
    "WaterOptions",
    "get_module",
    "modules_for",
    "run_module",
]
