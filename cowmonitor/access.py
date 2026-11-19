"""The passcode that unlocks the app.

It is fixed here, as the app's owner asked. It is a light lock, not real security: four digits
are quick to guess, and anyone who can read this file can see it. Keep the repository and any
online copy of the app private (README, "Put it online").
"""

from __future__ import annotations

import hmac

PASSCODE = "0009"


def is_passcode(entered: str) -> bool:
    """Whether ``entered`` is the passcode, ignoring spaces around it."""
    return hmac.compare_digest(entered.strip().encode(), PASSCODE.encode())
