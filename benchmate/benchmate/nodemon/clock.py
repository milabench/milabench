from __future__ import annotations

import ctypes
from typing import NamedTuple


class _Timex(ctypes.Structure):
    """The head of Linux's struct timex, padded past its full size."""

    _fields_ = [
        ("modes", ctypes.c_uint),
        ("offset", ctypes.c_long),
        ("freq", ctypes.c_long),
        ("maxerror", ctypes.c_long),
        ("esterror", ctypes.c_long),
        ("_rest", ctypes.c_byte * 256),
    ]

    #: adjtimex's return value when the clock is not synchronized.
    time_error = 5


class ClockState(NamedTuple):
    """Whether the kernel clock is NTP-synchronized, and its maximum error; None when unknown."""

    synced: bool | None
    maxerror_us: int | None # an upper bound, not the actual offset.


def clock_state() -> ClockState:
    """Whether the kernel clock is NTP-synchronized, and its maximum error, via adjtimex(2).

    With modes 0 the call only reads, so it needs no privileges.
    """
    timex = _Timex()
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        state = libc.adjtimex(ctypes.byref(timex))
    except (OSError, AttributeError):
        state = -1
    if state < 0:
        return ClockState(None, None)
    return ClockState(state != _Timex.time_error, timex.maxerror)
