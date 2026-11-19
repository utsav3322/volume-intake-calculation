"""The passcode boxes: what reaches Python from the browser."""

from __future__ import annotations

import pytest

from passcode_boxes import submitted_code


@pytest.mark.unit
def test_a_typed_code_is_passed_on_as_typed() -> None:
    assert submitted_code("0009") == "0009"
    assert submitted_code("12") == "12"  # app.py's check refuses it, not this


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, 9, 9.0, ["0009"], {"code": "0009"}, b"0009"])
def test_anything_but_text_counts_as_nothing_typed(value: object) -> None:
    # The browser can be made to send any JSON, so only text is treated as a code.
    assert submitted_code(value) is None
