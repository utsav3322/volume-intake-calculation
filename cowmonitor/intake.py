"""Water intake estimate: litres drunk, from the rumen temperature dips and the water temperature.

Every litre a cow drinks has to be warmed from the water's temperature to the cow's normal
temperature, which takes 4.19 kJ for each °C of difference. The cow's body supplies that heat
while the rumen reading climbs back, faster the further the rumen is below normal (Newton's
law of cooling). So the area of a dip, degrees below normal × hours, is proportional to the
heat, and

    litres = K × area ÷ (normal temperature − water temperature)

K, the cow's re-warming rate in litres per hour, depends on the rumen's size, where the bolus
sits and how much water moves on before it is warmed, so it can't be worked out from first
principles. It is calibrated on the project's water meters (``cowmonitor.intake_calibration``,
run by ``tools/calibrate_intake.py``). The meters can't tell what temperature the water was in
those weeks: for any assumed temperature from 0 to 30 °C, refitting K fits them equally well.
K is therefore fitted at the measured temperature of the water the cows drank,
``REFERENCE_WATER_C``.

Drinks taken while the bolus sends nothing are missed, so a day's litres are scaled up by the
share of the day that has readings (days under half covered get no estimate).

Times when the cow's normal is below 35 °C (a bolus not yet in the cow) are left out.

A known limit: the normal is a high percentile, so ordinary readings sit a little below it and
each drink's window also collects that small gap, about a quarter of a drink's area on the
project's cows. K absorbs it on average, so the estimate partly follows how often a cow drinks:
roughly 15% high for a cow that sips far more often than those cows, and as much low for one
that drinks far less often. Lower baselines and noise margins fitted the meters worse.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cowmonitor.drinking import DrinkingMarks, dip_starts
from cowmonitor.loaders import RUMEN_C
from cowmonitor.timezones import coverage_percent, local_dates

LITRES_PER_GALLON = 3.785411784
NORMAL_WINDOW_HOURS = 6.0
NORMAL_QUANTILE = 0.9
RECOVERY_HOURS = 2.0
WATER_MATCH_HOURS = 2.0
MIN_DIFFERENCE_C = 3.0
LOW_DIFFERENCE_C = 10.0
MIN_DAY_COVERAGE = 50.0


@dataclass(frozen=True)
class CalibrationSummary:
    """Where the stored K comes from and how close it came to the meters (gallons per day).

    ``days`` is the number of complete meter days compared; ``sweep_c`` the range of assumed
    water temperatures tried, with the daily error range they gave in ``sweep_error_gal``.
    """

    meters: str
    cows: str
    cow_count: int
    period: str
    days: int
    daily_error_gal: float
    daily_error_pct: float
    cross_error_gal: tuple[float, float]
    hourly_r: float
    sweep_c: tuple[float, float]
    sweep_error_gal: tuple[float, float]


# Calibrated by tools/calibrate_intake.py with CowMonitor's drinking-removed files; dips
# estimated from the temperature alone give nearly the same K. 15 °C (59 °F) is the measured
# temperature of the test pens' water bowls, which is what the cows drank (the storage tank
# was 10 °C, 50 °F); the meters fit any assumed temperature equally well. Recalibrate if the
# method in this file or REFERENCE_WATER_C changes (tests/test_intake.py checks these numbers
# against the meters).
REFERENCE_WATER_C = 15.0
REWARM_LITRES_PER_HOUR = 133.3
CALIBRATION = CalibrationSummary(
    meters="2D and 2F",
    cows="1354 + 1568 and 1297 + 1709",
    cow_count=4,
    period="6–20 April 2024",
    days=28,
    daily_error_gal=4.58,
    daily_error_pct=9.4,
    cross_error_gal=(4.58, 4.59),
    hourly_r=0.833,
    sweep_c=(0.0, 30.0),
    sweep_error_gal=(4.56, 4.59),
)


@dataclass(frozen=True)
class IntakeEstimate:
    """One cow's estimated water intake.

    ``drinks`` has one row per drink (``start`` in UTC, ``minutes``, ``coldest_c``,
    ``normal_c``, ``area_ch``, ``water_c``, ``water_from_file``, ``litres``); ``daily`` has one
    row per local day, for display; ``coverage`` is the unrounded % of each day with readings;
    ``readings`` adds the cow's ``normal`` temperature to the readings. ``water_c`` is the
    median water temperature of the drinks matched in the water file, else the setting.
    """

    drinks: pd.DataFrame
    daily: pd.DataFrame
    readings: pd.DataFrame
    coverage: pd.Series
    water_c: float
    drinks_from_file: int
    warnings: tuple[str, ...]

    @property
    def water_from_file(self) -> bool:
        return self.drinks_from_file > 0


def normal_temperature(readings: pd.DataFrame) -> pd.Series:
    """The cow's normal temperature: the 90th percentile of the readings in a centred 6 hours.

    A high percentile, so the slow climb back after each drink doesn't pull it down.
    """
    series = readings.set_index("ts")["temperature"]
    window = f"{NORMAL_WINDOW_HOURS:g}h"
    normal = series.rolling(window, center=True, min_periods=1).quantile(NORMAL_QUANTILE)
    return pd.Series(normal.to_numpy(), index=readings.index)


def drink_litres(drinks: pd.DataFrame, rewarm_litres_per_hour: float) -> pd.Series:
    """The heat balance for each drink, or no estimate if the water is too close to the cow."""
    difference = drinks["normal_c"] - drinks["water_c"]
    litres = rewarm_litres_per_hour * drinks["area_ch"] / difference
    return litres.where(difference >= MIN_DIFFERENCE_C)


def daily_litres(litres: pd.Series, dates: pd.Series, coverage: pd.Series) -> pd.DataFrame:
    """Per local day (``coverage``'s index): litres found, the day's estimate, drinks missing.

    ``litres`` and ``dates`` are per drink; ``coverage`` is the % of each day with readings.
    The estimate scales the drinks found up for missing readings. A day gets none if it is
    under half covered or one of its drinks has no estimate.
    """
    by_day = litres.groupby(dates)
    found = by_day.sum().reindex(coverage.index, fill_value=0.0)
    missing = litres.isna().groupby(dates).sum().reindex(coverage.index, fill_value=0)
    covered = (coverage >= MIN_DAY_COVERAGE) & missing.eq(0)
    return pd.DataFrame(
        {"found": found, "litres": (found * 100 / coverage).where(covered), "missing": missing}
    )


def estimate_intake(
    marks: DrinkingMarks,
    *,
    interval_minutes: float,
    timezone: str,
    water: pd.DataFrame | None = None,
    default_water_c: float = REFERENCE_WATER_C,
    rewarm_litres_per_hour: float = REWARM_LITRES_PER_HOUR,
) -> IntakeEstimate:
    """Litres per drink and per local day for one cow.

    ``water`` holds water temperature readings (``ts`` in UTC, ``temperature``); each drink
    uses the reading nearest its start within 2 hours, otherwise ``default_water_c``.
    """
    frame = marks.readings.reset_index(drop=True)
    frame["normal"] = normal_temperature(frame)
    in_cow = frame["normal"] >= RUMEN_C
    drinks = _find_drinks(frame, interval_minutes)
    drinks = drinks[drinks["normal_c"] >= RUMEN_C].reset_index(drop=True)
    drinks["water_c"], drinks["water_from_file"] = _water_at(
        drinks["start"], water, default_water_c
    )
    drinks["litres"] = drink_litres(drinks, rewarm_litres_per_hour)
    dates = local_dates(frame["ts"], timezone)
    days = pd.Index(sorted(dates.unique()))
    counts = frame[in_cow].groupby(dates[in_cow]).size().reindex(days, fill_value=0)
    coverage = coverage_percent(counts, timezone, interval_minutes)
    matched = drinks.loc[drinks["water_from_file"], "water_c"]
    outside_hours = float((~in_cow).sum()) * interval_minutes / 60
    return IntakeEstimate(
        drinks=drinks,
        daily=_daily(drinks, counts, coverage, timezone),
        readings=frame,
        coverage=coverage,
        water_c=float(matched.median()) if not matched.empty else default_water_c,
        drinks_from_file=len(matched),
        warnings=tuple(_warnings(drinks, water, default_water_c, outside_hours)),
    )


def _find_drinks(frame: pd.DataFrame, interval_minutes: float) -> pd.DataFrame:
    """One row per dip, with its area below normal from its start until RECOVERY_HOURS later
    or the next drink, whichever comes first."""
    starts = dip_starts(frame)
    number = starts.cumsum()
    started_at = frame["ts"].where(starts).ffill()
    owned = number.gt(0) & (frame["ts"] - started_at).le(pd.Timedelta(hours=RECOVERY_HOURS))
    below = (frame["normal"] - frame["temperature"]).clip(lower=0) * interval_minutes / 60
    area = below[owned].groupby(number[owned]).sum()
    drinking = frame["drinking"]
    runs = frame.loc[drinking, "temperature"].groupby(number[drinking])
    ids = number[starts].to_numpy()
    return pd.DataFrame(
        {
            "start": frame.loc[starts, "ts"].reset_index(drop=True),
            "minutes": runs.size().reindex(ids).to_numpy() * interval_minutes,
            "coldest_c": runs.min().reindex(ids).to_numpy(),
            "normal_c": frame.loc[starts, "normal"].to_numpy(),
            "area_ch": area.reindex(ids, fill_value=0.0).to_numpy(),
        }
    )


def _water_at(
    starts: pd.Series, water: pd.DataFrame | None, default_c: float
) -> tuple[pd.Series, pd.Series]:
    """The water temperature for each drink, and whether it came from the water file."""
    if water is None or water.empty or starts.empty:
        return (
            pd.Series(default_c, index=starts.index, dtype=float),
            pd.Series(False, index=starts.index, dtype=bool),
        )
    left = pd.DataFrame({"start": starts.dt.as_unit("ns"), "row": starts.index})
    right = pd.DataFrame(
        {"ts": water["ts"].dt.as_unit("ns"), "water": water["temperature"].astype(float)}
    )
    matched = pd.merge_asof(
        left.sort_values("start"),
        right.sort_values("ts"),
        left_on="start",
        right_on="ts",
        direction="nearest",
        tolerance=pd.Timedelta(hours=WATER_MATCH_HOURS),
    ).set_index("row")
    values = matched["water"].reindex(starts.index)
    return values.fillna(default_c), values.notna()


def _daily(
    drinks: pd.DataFrame, counts: pd.Series, coverage: pd.Series, timezone: str
) -> pd.DataFrame:
    dates = local_dates(drinks["start"], timezone)
    by_day = drinks.groupby(dates)
    litres = daily_litres(drinks["litres"], dates, coverage)
    table = pd.DataFrame(index=coverage.index)
    table["Readings"] = counts
    table["Coverage %"] = coverage
    table["Drinks"] = by_day.size().reindex(table.index, fill_value=0).astype(int)
    table["Drinking minutes"] = by_day["minutes"].sum().reindex(table.index, fill_value=0.0)
    table["Dip area °C·h"] = by_day["area_ch"].sum().reindex(table.index, fill_value=0.0)
    table["Water °C"] = by_day["water_c"].mean()
    table["Litres in drinks found"] = litres["found"]
    table["Litres"] = litres["litres"]
    table["Gallons"] = table["Litres"] / LITRES_PER_GALLON
    table["Note"] = _day_notes(coverage, litres["missing"])
    table = table.rename_axis("Date").reset_index()
    return table.round(
        {
            "Coverage %": 1,
            "Drinking minutes": 1,
            "Dip area °C·h": 2,
            "Water °C": 1,
            "Litres in drinks found": 1,
            "Litres": 1,
            "Gallons": 2,
        }
    )


def _day_notes(coverage: pd.Series, missing: pd.Series) -> list[str]:
    notes = []
    for cover, count in zip(coverage, missing, strict=True):
        if cover < MIN_DAY_COVERAGE:
            notes.append(f"no estimate: readings cover under {MIN_DAY_COVERAGE:.0f}% of the day")
        elif count:
            notes.append(f"no estimate: {_count(int(count))} could not be estimated")
        else:
            notes.append("")
    return notes


def _warnings(
    drinks: pd.DataFrame, water: pd.DataFrame | None, default_c: float, outside_hours: float
) -> list[str]:
    warnings = []
    if outside_hours:
        warnings.append(
            f"For {outside_hours:.0f} hours the readings stayed under {RUMEN_C:g} °C, as if the "
            "bolus was not in the cow; those times were left out of the estimate."
        )
    if drinks.empty:
        warnings.append("No drinking dips were found, so no water intake could be estimated.")
        return warnings
    if water is not None:
        unmatched = int((~drinks["water_from_file"]).sum())
        if unmatched:
            warnings.append(
                f"The water temperature file has no reading within {WATER_MATCH_HOURS:g} hours "
                f"of {unmatched} of {len(drinks)} drinks; {default_c:g} °C (the setting) was used "
                "for them."
            )
    difference = drinks["normal_c"] - drinks["water_c"]
    too_close = int((difference < MIN_DIFFERENCE_C).sum())
    close = int(((difference >= MIN_DIFFERENCE_C) & (difference < LOW_DIFFERENCE_C)).sum())
    if too_close:
        warnings.append(
            f"No estimate for {_count(too_close)}: the water was within {MIN_DIFFERENCE_C:g} °C "
            "of the cow's temperature, so the dips are too small to measure. The days with "
            "those drinks get no total."
        )
    if close:
        warnings.append(
            f"For {_count(close)} the water was within {LOW_DIFFERENCE_C:g} °C of the cow's "
            "temperature. The dips are then small, so those estimates are less reliable."
        )
    return warnings


def _count(number: int) -> str:
    return f"{number} drink" if number == 1 else f"{number} drinks"
