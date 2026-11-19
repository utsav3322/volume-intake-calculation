"""The passcode that unlocks the app."""

from __future__ import annotations

import re

import pytest

from cowmonitor.access import PASSCODE, is_passcode


@pytest.mark.unit
def test_the_passcode_is_0009() -> None:
    assert PASSCODE == "0009"


@pytest.mark.unit
@pytest.mark.parametrize("entered", ["0009", " 0009 ", "0009\n"])
def test_the_passcode_is_accepted_with_spaces_around_it(entered: str) -> None:
    assert is_passcode(entered)


# "0009" typed with full-width digits, which look the same on screen.
FULL_WIDTH_0009 = "".join(chr(0xFF10 + int(digit)) for digit in "0009")


@pytest.mark.unit
@pytest.mark.parametrize("entered", ["", "9", "009", "00090", "0008", "OOO9", FULL_WIDTH_0009])
def test_anything_else_is_refused(entered: str) -> None:
    assert not is_passcode(entered)


@pytest.mark.unit
def test_the_passcode_uses_only_characters_the_boxes_accept() -> None:
    # One letter (A-Z, a-z) or digit per box on the passcode screen.
    assert re.fullmatch(r"[A-Za-z0-9]+", PASSCODE)


@pytest.mark.unit
def test_a_passcode_with_letters_is_case_sensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cowmonitor.access.PASSCODE", "a1B2")
    assert is_passcode("a1B2")
    assert not is_passcode("A1B2")
    assert not is_passcode("a1b2")
    assert not is_passcode("0009")
