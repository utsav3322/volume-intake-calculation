"""The passcode that unlocks the app."""

from __future__ import annotations

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
