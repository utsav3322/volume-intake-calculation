"""The Streamlit settings in ``.streamlit/config.toml`` that hosting relies on."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

CONFIG = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"


def _config() -> dict[str, Any]:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_the_host_chooses_the_address_and_port() -> None:
    # Streamlit Community Cloud has to reach the app; a fixed address or port can stop it.
    server = _config()["server"]
    assert "address" not in server
    assert "port" not in server


@pytest.mark.unit
def test_uploads_are_limited_to_25_mb() -> None:
    # The only place the README's 25 MB limit is set.
    assert _config()["server"]["maxUploadSize"] == 25
