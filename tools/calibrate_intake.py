"""Calibrate the water intake estimate on the project's water meters 2D and 2F.

Run from the streamlit_app folder:

    python -m tools.calibrate_intake ["../Data/Data Files/DrinkingData"]

It reports the re-warming rate K for the reference water temperature in cowmonitor/intake.py,
how close the estimates come to the meters, and the same for every assumed water temperature
from 0 to 30 °C.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cowmonitor.intake import REFERENCE_WATER_C, REWARM_LITRES_PER_HOUR
from cowmonitor.intake_calibration import Accuracy, CowFiles, Pen, compare, evaluate, prepare, sweep
from cowmonitor.loaders import LoadedFile, load_upload

logger = logging.getLogger("calibrate_intake")

DESCRIPTION = "Calibrate the water intake estimate on the project's water meters 2D and 2F."
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "Data" / "Data Files" / "DrinkingData"
PROJECT_PENS = (("2D", ("1354", "1568")), ("2F", ("1297", "1709")))
SWEEP_C = tuple(float(t) for t in range(0, 31))


def _load(folder: Path, name: str) -> LoadedFile:
    return load_upload(name, (folder / name).read_bytes())


def load_project_pens(folder: Path) -> list[Pen]:
    """Meters 2D and 2F with their cows' *_raw.csv and *_bouts.csv files."""
    pens = []
    for meter_name, cows in PROJECT_PENS:
        meter = _load(folder, f"{meter_name}.csv").meter
        if meter is None:
            raise ValueError(f"{meter_name}.csv is not a water-meter export")
        files = []
        for cow in cows:
            raw = _load(folder, f"{cow}_raw.csv").temperature
            copy = _load(folder, f"{cow}_bouts.csv").temperature
            if raw is None or copy is None:
                raise ValueError(f"The files for cow {cow} are not temperature exports")
            files.append(CowFiles(raw, copy))
        pens.append(Pen(meter_name, meter, tuple(files)))
    return pens


def _describe(label: str, result: Accuracy) -> str:
    totals = ", ".join(f"{name} {value:+.1f}%" for name, value in result.total_error_pct)
    cross = ", ".join(f"{name} {value:.2f}" for name, value in result.cross_error_gal)
    return (
        f"{label}: K = {result.rewarm_litres_per_hour:.2f} litres/hour | daily error "
        f"{result.daily_error_gal:.2f} gal ({result.daily_error_pct:.1f}%) | totals {totals} | "
        f"calibrated on one meter, other meter: {cross} gal | leave one day out "
        f"{result.leave_one_day_out_gal:.2f} gal (guessing the average "
        f"{result.guess_average_gal:.2f}) | hourly r {result.hourly_r:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("data", nargs="?", type=Path, default=DEFAULT_DATA)
    folder = parser.parse_args().data
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pens = load_project_pens(folder)
    for use_copy, label in ((True, "with the drinking-removed copy"), (False, "estimated dips")):
        prepared = prepare(pens, use_copy=use_copy)
        reference = compare(prepared, REFERENCE_WATER_C)
        refitted = f"Refitted at {REFERENCE_WATER_C:g} °C, {label}"
        logger.info(_describe(refitted, evaluate(reference)))
        stored = f"With the stored K = {REWARM_LITRES_PER_HOUR:g}, {label}"
        logger.info(_describe(stored, evaluate(reference, REWARM_LITRES_PER_HOUR)))
        table = sweep(prepared, SWEEP_C)
        logger.info("Assumed water temperature sweep, %s:\n%s\n", label, table.round(3).to_string())


if __name__ == "__main__":
    main()
