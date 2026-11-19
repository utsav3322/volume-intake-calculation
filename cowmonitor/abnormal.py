"""Model 1: abnormal drinking check, ported from ``abnormal()`` in HealthEventsDetection.ipynb.

A healthy cow's rumen temperature drops sharply each time she drinks cold water. The model
counts readings below 32, 33.5 and 34.5 °C (deep drops) and above 39 and 40 °C (heat):

* too few deep drops: fewer than 10, 20 and 30 readings below those three temperatures, or
* a long high-temperature period: more than 800 readings above 39 °C and more than 30
  above 40 °C,

makes the cow "abnormal". The original limits are raw counts tuned on files about 6 days
long, so a shorter file looks abnormal just for being short. ``LimitsMode.PER_DAY`` scales
the limits to the file's length; ``LimitsMode.FIXED`` keeps the original counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

REFERENCE_DAYS = 6.0
MIN_DAYS = 1.0
WARN_DAYS = 2.0

# (temperature °C, count limit): an alert needs FEWER readings below each low temperature...
LOW_RULE: tuple[tuple[float, int], ...] = ((32.0, 10), (33.5, 20), (34.5, 30))
# ...or MORE readings above each high temperature.
HIGH_RULE: tuple[tuple[float, int], ...] = ((39.0, 800), (40.0, 30))


class LimitsMode(StrEnum):
    PER_DAY = "per_day"
    FIXED = "fixed"


@dataclass(frozen=True)
class CountCheck:
    """One count in the rule, its limit, and whether that part of the alert is met."""

    description: str
    count: int
    limit: float
    comparison: str
    met: bool


@dataclass(frozen=True)
class AbnormalResult:
    abnormal: bool
    few_deep_drops: bool
    long_high_temperature: bool
    enough_data: bool
    days: float
    scale: float
    checks: tuple[CountCheck, ...]

    @property
    def verdict(self) -> str:
        if not self.enough_data:
            return "Not enough data"
        return "Abnormal" if self.abnormal else "Normal"

    @property
    def reason(self) -> str:
        if not self.enough_data:
            return (
                f"Only {self.days * 24:.0f} hours of data; at least {MIN_DAYS * 24:.0f} hours "
                "are needed."
            )
        causes = []
        if self.few_deep_drops:
            causes.append("too few deep drinking drops")
        if self.long_high_temperature:
            causes.append("a long high-temperature period")
        if causes:
            return " and ".join(causes).capitalize()
        return "Enough deep drinking drops and no long high-temperature period"


def check_abnormal(
    readings: pd.DataFrame, mode: LimitsMode, interval_minutes: float = 5.0
) -> AbnormalResult:
    """Apply the model to one cow's readings (columns ``ts`` and ``temperature``).

    The data length runs from the first to the last reading plus one reading interval, so
    288 five-minute readings make exactly one day.
    """
    temperature = readings["temperature"]
    timestamps = readings["ts"]
    span_minutes = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 60
    days = (span_minutes + interval_minutes) / 1440
    scale = days / REFERENCE_DAYS if mode is LimitsMode.PER_DAY else 1.0

    low_checks = [
        _check(
            f"Readings below {limit_c:g} °C",
            int((temperature < limit_c).sum()),
            count * scale,
            "fewer than",
        )
        for limit_c, count in LOW_RULE
    ]
    high_checks = [
        _check(
            f"Readings above {limit_c:g} °C",
            int((temperature > limit_c).sum()),
            count * scale,
            "more than",
        )
        for limit_c, count in HIGH_RULE
    ]
    few_deep_drops = all(check.met for check in low_checks)
    long_high_temperature = all(check.met for check in high_checks)
    enough_data = days >= MIN_DAYS
    return AbnormalResult(
        abnormal=enough_data and (few_deep_drops or long_high_temperature),
        few_deep_drops=few_deep_drops,
        long_high_temperature=long_high_temperature,
        enough_data=enough_data,
        days=days,
        scale=scale,
        checks=tuple(low_checks + high_checks),
    )


def _check(description: str, count: int, limit: float, comparison: str) -> CountCheck:
    met = count < limit if comparison == "fewer than" else count > limit
    return CountCheck(
        description=description, count=count, limit=limit, comparison=comparison, met=met
    )
