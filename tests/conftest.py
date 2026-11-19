"""Shared fixtures: where the project's delivered CSV files and the reference results live."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from cowmonitor.loaders import LoadedFile, load_upload

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR.parent / "Data" / "Data Files"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """The project's Data/Data Files folder; tests that need it are skipped without it."""
    if not DATA_DIR.is_dir():
        pytest.skip("The project's Data/Data Files folder is not available")
    return DATA_DIR


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def load_data_file(data_dir: Path) -> Callable[[str], LoadedFile]:
    """Load a delivered CSV the same way the app loads an upload."""

    def _load(relative_path: str) -> LoadedFile:
        path = data_dir / relative_path
        return load_upload(path.name, path.read_bytes())

    return _load
