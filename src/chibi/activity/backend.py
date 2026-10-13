"""Where activity information comes from.

The monitor only talks to an :class:`ActivityBackend`; the operating-system
specifics live behind it.  Windows gets :class:`WindowsActivityBackend`
(ctypes + psutil); every other platform gets :class:`NullActivityBackend`,
which honestly reports "not supported" instead of pretending to observe
anything.

Privacy: a :class:`ForegroundInfo` may carry a window title and an exe path.
Both are excluded from ``repr`` so that an accidental ``log.debug("%s", info)``
can never write them to a log file.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForegroundInfo:
    """The window the user is looking at right now.

    ``process_name`` is the lower-cased executable name (``code.exe``).
    ``title`` is only filled when the caller asked for it (the user granted
    ``WINDOW_TITLES``); it is used transiently and never leaves the monitor.
    """

    pid: int
    process_name: str
    exe_path: str | None = field(default=None, repr=False)
    title: str | None = field(default=None, repr=False)
    #: The foreground window handle (0 if unknown); screen vision captures it.
    hwnd: int = field(default=0, repr=False, compare=False)


@runtime_checkable
class ActivityBackend(Protocol):
    def is_supported(self) -> bool:
        """Whether this backend can observe anything on this machine."""
        ...

    def foreground(self, read_title: bool) -> ForegroundInfo | None:
        """The foreground app, or None if there is none (lock screen, desktop switch...).

        The window title is read only when *read_title* is True.
        """
        ...

    def idle_seconds(self) -> float | None:
        """Seconds since the last keyboard/mouse input (None if unknown).

        Only *when* input happened - never which keys.
        """
        ...

    def running_processes(self) -> set[str]:
        """Lower-case executable names of the running processes."""
        ...


class NullActivityBackend:
    """Backend for platforms without activity support: observes nothing."""

    def is_supported(self) -> bool:
        return False

    def foreground(self, read_title: bool) -> ForegroundInfo | None:
        return None

    def idle_seconds(self) -> float | None:
        return None

    def running_processes(self) -> set[str]:
        return set()


def default_backend() -> ActivityBackend:
    """The best backend for this OS: Windows gets the real one, others the null one.

    A Windows backend that fails to bind (exotic environment, missing DLL
    export) degrades to the null backend instead of crashing the app.
    """
    if sys.platform == "win32":
        try:
            from chibi.activity.windows_backend import WindowsActivityBackend

            return WindowsActivityBackend()
        except Exception:
            log.exception("Windows activity backend unavailable; activity monitoring is disabled")
    return NullActivityBackend()
