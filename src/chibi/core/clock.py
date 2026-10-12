"""Injectable time source so time-based logic (needs decay, cooldowns,
budgets) is deterministic in tests."""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """Wall-clock epoch seconds (persisted timestamps, time of day)."""
        ...

    def monotonic(self) -> float:
        """Monotonic seconds (durations, cooldowns)."""
        ...


class SystemClock:
    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()


class FakeClock:
    """Manually advanced clock for tests."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self._now = start
        self._mono = 1000.0

    def now(self) -> float:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time only moves forward")
        self._now += seconds
        self._mono += seconds

    def set_now(self, epoch_seconds: float) -> None:
        """Jump wall-clock time (e.g. to a given time of day) without touching monotonic time."""
        self._now = epoch_seconds
