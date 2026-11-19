"""What a module hands back to the page: headline numbers, charts, tables, warnings and notes."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


@dataclass(frozen=True)
class Metric:
    label: str
    value: str


@dataclass(frozen=True)
class Chart:
    title: str
    figure: go.Figure
    caption: str | None = None


@dataclass(frozen=True)
class Table:
    title: str
    frame: pd.DataFrame
    file_name: str


@dataclass(frozen=True)
class ModuleResult:
    title: str
    subtitle: str = ""
    metrics: tuple[Metric, ...] = ()
    charts: tuple[Chart, ...] = ()
    tables: tuple[Table, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def table_to_csv(frame: pd.DataFrame) -> bytes:
    """CSV bytes for a download; text cells a spreadsheet would run as formulas are escaped."""
    safe = frame.copy()
    for column in safe.columns:
        if not pd.api.types.is_numeric_dtype(safe[column]):
            safe[column] = safe[column].map(_escape_cell)
    return safe.to_csv(index=False, quoting=csv.QUOTE_NONNUMERIC).encode("utf-8")


def _escape_cell(value: object) -> object:
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value
