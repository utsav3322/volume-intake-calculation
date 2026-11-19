"""Calibrate the water intake estimate on water meters shared by known cows.

A meter only sees the cows on it together, so each meter day (and meter hour) is compared
with the sum of those cows' estimates, worked out with the app's own functions: each cow's day
is scaled up for missing readings. Each meter report counts towards the day of the hour it
measured, and incomplete meter days (fewer than 20 reports, or the export's last day) are left
out. The re-warming rate K is set so that the estimates add up to the metered water over all
meter days. ``sweep`` repeats this for a range of assumed water temperatures; that is how the
reference water temperature was examined.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from cowmonitor.drinking import estimate_marks, marks_from_copy
from cowmonitor.intake import LITRES_PER_GALLON, daily_litres, drink_litres, estimate_intake
from cowmonitor.loaders import MeterData, TemperatureData
from cowmonitor.timezones import DEFAULT_TIMEZONE, local_dates
from cowmonitor.water import FULL_DAY_REPORTS


@dataclass(frozen=True)
class CowFiles:
    """One cow's temperature export and, optionally, CowMonitor's drinking-removed copy."""

    raw: TemperatureData
    copy: TemperatureData | None = None


@dataclass(frozen=True)
class Pen:
    """A water meter and the cows that drink from it."""

    name: str
    meter: MeterData
    cows: tuple[CowFiles, ...]


@dataclass(frozen=True)
class CowDrinks:
    """One cow's drinks (``start``, ``area_ch``, ``normal_c``) and the unrounded % of each
    local day that has readings."""

    drinks: pd.DataFrame
    coverage: pd.Series


@dataclass(frozen=True)
class PenDrinks:
    """A pen's drinks, found once, with its meter's hourly gallons and the farm time zone."""

    name: str
    cows: tuple[CowDrinks, ...]
    meter: pd.DataFrame
    timezone: str


@dataclass(frozen=True)
class Comparison:
    """Metered litres and the estimate at K = 1 litre per hour, per meter day and meter hour."""

    daily: pd.DataFrame
    hourly: pd.DataFrame


@dataclass(frozen=True)
class Accuracy:
    """How close the estimates come to the meters at re-warming rate K (errors in gallons).

    ``cross_error_gal`` holds, for each pair of meters, the daily error on the second meter
    when K is fitted on the first alone.
    """

    rewarm_litres_per_hour: float
    daily_error_gal: float
    daily_error_pct: float
    total_error_pct: tuple[tuple[str, float], ...]
    cross_error_gal: tuple[tuple[str, float], ...]
    leave_one_day_out_gal: float
    guess_average_gal: float
    hourly_r: float


def prepare(
    pens: Sequence[Pen], use_copy: bool = True, timezone: str = DEFAULT_TIMEZONE
) -> list[PenDrinks]:
    """Find every drink once; an assumed water temperature only rescales them afterwards."""
    prepared = []
    for pen in pens:
        cows = []
        for cow in pen.cows:
            readings = cow.raw.for_cow(_only_cow(cow.raw, "temperature file"))
            if use_copy and cow.copy is not None:
                copy = cow.copy.for_cow(_only_cow(cow.copy, "drinking-removed file"))
                marks = marks_from_copy(readings, copy)
            else:
                marks = estimate_marks(readings)
            estimate = estimate_intake(
                marks, interval_minutes=cow.raw.interval_minutes, timezone=timezone
            )
            cows.append(
                CowDrinks(estimate.drinks[["start", "area_ch", "normal_c"]], estimate.coverage)
            )
        # When clocks go back the meter reports the same local hour twice; count it once.
        meter = pen.meter.readings.groupby("read_time", as_index=False)["gallons"].sum()
        prepared.append(PenDrinks(pen.name, tuple(cows), meter, timezone))
    return prepared


def compare(pens: Sequence[PenDrinks], water_c: float) -> Comparison:
    """Meter days and hours against the pen's drinks at K = 1, assuming water at ``water_c``."""
    daily, hourly = [], []
    for pen in pens:
        meter = pen.meter
        measured = (meter["read_time"] - pd.Timedelta(minutes=30)).dt.date
        reports = meter.groupby(measured).size()
        # A meter day is complete with 20+ reports and a report counted towards the next day,
        # which holds the water of the day's last hour.
        next_day = [day + timedelta(days=1) in reports.index for day in reports.index]
        complete = (reports >= FULL_DAY_REPORTS) & pd.Series(next_day, index=reports.index)
        metered = meter.groupby(measured)["gallons"].sum()[complete]
        per_cow = [_cow_days(cow, water_c, pen.timezone) for cow in pen.cows]
        estimated = pd.concat(per_cow, axis=1, join="inner").sum(axis=1)
        days = [day for day in estimated.index if day in metered.index]
        daily.append(
            pd.DataFrame(
                {
                    "pen": pen.name,
                    "date": days,
                    "metered_litres": metered.reindex(days).to_numpy() * LITRES_PER_GALLON,
                    "unit_litres": estimated.reindex(days).to_numpy(),
                }
            )
        )
        drinks = pd.concat([cow.drinks for cow in pen.cows], ignore_index=True)
        unit = drink_litres(drinks.assign(water_c=water_c), 1.0).fillna(0.0)
        local = drinks["start"].dt.tz_convert(pen.timezone).dt.tz_localize(None)
        reads = meter["read_time"]
        edges = pd.concat([pd.Series([reads.iloc[0] - pd.Timedelta(hours=1)]), reads])
        slot = pd.cut(local, bins=pd.DatetimeIndex(edges), labels=False, right=True)
        per_slot = unit.groupby(slot).sum()
        hourly.append(
            pd.DataFrame(
                {
                    "pen": pen.name,
                    "read_time": reads.to_numpy(),
                    "metered_litres": meter["gallons"].to_numpy() * LITRES_PER_GALLON,
                    "unit_litres": per_slot.reindex(range(len(meter)), fill_value=0.0).to_numpy(),
                }
            )
        )
    return Comparison(pd.concat(daily, ignore_index=True), pd.concat(hourly, ignore_index=True))


def calibrate(daily: pd.DataFrame) -> float:
    """K that makes the estimates add up to the metered water over these days."""
    return float(daily["metered_litres"].sum() / daily["unit_litres"].sum())


def evaluate(comparison: Comparison, rewarm_litres_per_hour: float | None = None) -> Accuracy:
    """How close the estimates come to the meters, with K refitted unless one is given."""
    daily = comparison.daily
    k = calibrate(daily) if rewarm_litres_per_hour is None else rewarm_litres_per_hour
    metered = daily["metered_litres"].to_numpy() / LITRES_PER_GALLON
    unit = daily["unit_litres"].to_numpy() / LITRES_PER_GALLON
    errors = np.abs(k * unit - metered)
    pens = daily["pen"].to_numpy()
    names = list(dict.fromkeys(pens))
    totals = tuple(
        (name, float(100 * (k * unit[pens == name].sum() / metered[pens == name].sum() - 1)))
        for name in names
    )
    cross = tuple(
        (
            f"{train}->{test}",
            float(
                np.abs(
                    calibrate(daily[pens == train]) * unit[pens == test] - metered[pens == test]
                ).mean()
            ),
        )
        for train in names
        for test in names
        if train != test
    )
    leave_out, guess = [], []
    for i in range(len(daily)):
        rest = np.arange(len(daily)) != i
        leave_out.append(abs(metered[rest].sum() / unit[rest].sum() * unit[i] - metered[i]))
        guess.append(abs(metered[rest].mean() - metered[i]))
    hourly = comparison.hourly
    return Accuracy(
        rewarm_litres_per_hour=k,
        daily_error_gal=float(errors.mean()),
        daily_error_pct=float(100 * errors.mean() / metered.mean()),
        total_error_pct=totals,
        cross_error_gal=cross,
        leave_one_day_out_gal=float(np.mean(leave_out)),
        guess_average_gal=float(np.mean(guess)),
        hourly_r=float(np.corrcoef(hourly["unit_litres"], hourly["metered_litres"])[0, 1]),
    )


def sweep(pens: Sequence[PenDrinks], water_temperatures: Sequence[float]) -> pd.DataFrame:
    """Refit K for each assumed water temperature and report the accuracy of each."""
    rows = []
    for water_c in water_temperatures:
        result = evaluate(compare(pens, water_c))
        row: dict[str, object] = {
            "Water °C": water_c,
            "K litres/hour": result.rewarm_litres_per_hour,
            "Daily error gal": result.daily_error_gal,
            "Daily error %": result.daily_error_pct,
            "Leave-one-day-out gal": result.leave_one_day_out_gal,
            "Hourly r": result.hourly_r,
        }
        row.update({f"Cross {name} gal": value for name, value in result.cross_error_gal})
        row.update({f"Total {name} %": value for name, value in result.total_error_pct})
        rows.append(row)
    return pd.DataFrame(rows)


def _only_cow(data: TemperatureData, what: str) -> str:
    if len(data.cows) != 1:
        raise ValueError(f"Each {what} must hold one cow; this one holds {len(data.cows)}.")
    return data.cows[0].key


def _cow_days(cow: CowDrinks, water_c: float, timezone: str) -> pd.Series:
    """Litres per local day at K = 1, as the app's daily estimate; days without one dropped."""
    unit = drink_litres(cow.drinks.assign(water_c=water_c), 1.0)
    days = daily_litres(unit, local_dates(cow.drinks["start"], timezone), cow.coverage)
    return days["litres"].dropna()
