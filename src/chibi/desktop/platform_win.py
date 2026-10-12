"""Win32 helpers via ctypes, with safe no-ops everywhere else.

* :func:`apply_no_activate` makes a window that never takes focus: clicking
  her must not steal the keyboard from the game or editor the user is in.
* :func:`foreground_fullscreen` answers "does the foreground window cover
  its whole monitor?" (a game, a video, a presentation) from geometry only -
  no title, no process name, so it needs no permission.

Signatures are declared explicitly (``argtypes``/``restype``) on a private
``WinDLL`` instance: with ctypes' default ``int`` return type a 64-bit HWND
or LONG_PTR is silently truncated, and setting prototypes on the shared
``ctypes.windll.user32`` would leak into other modules.

Nothing here raises into callers: a failing call is logged once and
reported as "no" (``False``), which is always the harmless answer.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
WS_EX_NOACTIVATE = 0x08000000
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
MONITOR_DEFAULTTONEAREST = 0x00000002

#: Desktop and taskbar windows: "fullscreen-sized" but never a fullscreen app.
SHELL_CLASSES: frozenset[str] = frozenset({"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"})

_warned: set[str] = set()


def _warn_once(key: str, message: str, *args: object) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(message, *args)


@dataclass(frozen=True)
class Box:
    """A rectangle in physical pixels (``right``/``bottom`` exclusive, like RECT)."""

    left: int
    top: int
    right: int
    bottom: int


def covers(window: Box, monitor: Box) -> bool:
    """True if *window* covers all of *monitor* (borderless and exclusive
    fullscreen windows are monitor-sized or larger)."""
    if monitor.right <= monitor.left or monitor.bottom <= monitor.top:
        return False
    return (window.left <= monitor.left and window.top <= monitor.top
            and window.right >= monitor.right and window.bottom >= monitor.bottom)


class WinApi(Protocol):
    """The handful of user32 calls :func:`foreground_fullscreen` needs; tests
    substitute a fake to check the decision logic on any OS."""

    def foreground_window(self) -> int: ...

    def is_visible(self, hwnd: int) -> bool: ...

    def is_minimized(self, hwnd: int) -> bool: ...

    def class_name(self, hwnd: int) -> str: ...

    def window_rect(self, hwnd: int) -> Box | None: ...

    def monitor_rect(self, hwnd: int) -> Box | None: ...


if IS_WINDOWS:  # pragma: no cover - exercised by the Windows CI job
    from ctypes import wintypes

    _HWND = ctypes.c_void_p
    _LONG_PTR = ctypes.c_ssize_t
    _IS_64BIT = ctypes.sizeof(ctypes.c_void_p) == 8

    class _MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
        ]

    def _load_user32() -> Any:
        user32 = ctypes.WinDLL("user32", use_last_error=True)

        def proto(name: str, restype: Any, *argtypes: Any) -> Any:
            fn = getattr(user32, name)
            fn.restype = restype
            fn.argtypes = list(argtypes)
            return fn

        api = {
            "GetForegroundWindow": proto("GetForegroundWindow", _HWND),
            "IsWindowVisible": proto("IsWindowVisible", wintypes.BOOL, _HWND),
            "IsIconic": proto("IsIconic", wintypes.BOOL, _HWND),
            "GetClassNameW": proto("GetClassNameW", ctypes.c_int, _HWND, wintypes.LPWSTR, ctypes.c_int),
            "GetWindowRect": proto("GetWindowRect", wintypes.BOOL, _HWND, ctypes.POINTER(wintypes.RECT)),
            "MonitorFromWindow": proto("MonitorFromWindow", wintypes.HMONITOR, _HWND, wintypes.DWORD),
            "GetMonitorInfoW": proto("GetMonitorInfoW", wintypes.BOOL, wintypes.HMONITOR,
                                     ctypes.POINTER(_MONITORINFO)),
            "SetWindowPos": proto("SetWindowPos", wintypes.BOOL, _HWND, _HWND, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int, wintypes.UINT),
        }
        # GetWindowLongPtrW only exists in 64-bit user32; 32-bit uses the LONG versions.
        get_name, set_name = (("GetWindowLongPtrW", "SetWindowLongPtrW") if _IS_64BIT
                              else ("GetWindowLongW", "SetWindowLongW"))
        long_type = _LONG_PTR if _IS_64BIT else wintypes.LONG
        api["GetWindowLongPtr"] = proto(get_name, long_type, _HWND, ctypes.c_int)
        api["SetWindowLongPtr"] = proto(set_name, long_type, _HWND, ctypes.c_int, long_type)
        return api

    try:
        _USER32: dict[str, Any] | None = _load_user32()
    except (OSError, AttributeError) as exc:
        _USER32 = None
        log.warning("user32 unavailable: %s", exc)

    class _RealWinApi:
        def __init__(self, user32: dict[str, Any]) -> None:
            self._u = user32

        def foreground_window(self) -> int:
            return int(self._u["GetForegroundWindow"]() or 0)

        def is_visible(self, hwnd: int) -> bool:
            return bool(self._u["IsWindowVisible"](hwnd))

        def is_minimized(self, hwnd: int) -> bool:
            return bool(self._u["IsIconic"](hwnd))

        def class_name(self, hwnd: int) -> str:
            buf = ctypes.create_unicode_buffer(256)
            n = self._u["GetClassNameW"](hwnd, buf, len(buf))
            return buf.value[:n] if n > 0 else ""

        def window_rect(self, hwnd: int) -> Box | None:
            rect = wintypes.RECT()
            if not self._u["GetWindowRect"](hwnd, ctypes.byref(rect)):
                return None
            return Box(rect.left, rect.top, rect.right, rect.bottom)

        def monitor_rect(self, hwnd: int) -> Box | None:
            monitor = self._u["MonitorFromWindow"](hwnd, MONITOR_DEFAULTTONEAREST)
            if not monitor:
                return None
            info = _MONITORINFO()
            info.cbSize = ctypes.sizeof(_MONITORINFO)
            if not self._u["GetMonitorInfoW"](monitor, ctypes.byref(info)):
                return None
            r = info.rcMonitor
            return Box(r.left, r.top, r.right, r.bottom)

    _API: WinApi | None = _RealWinApi(_USER32) if _USER32 is not None else None
else:
    _USER32 = None
    _API = None


def supported() -> bool:
    """True when the Win32 calls are available (Windows with a loadable user32)."""
    return _API is not None


def default_api() -> WinApi | None:
    return _API


def apply_no_activate(hwnd: int) -> bool:
    """Add WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW to *hwnd* (and drop WS_EX_APPWINDOW).

    Qt's ``WindowDoesNotAcceptFocus`` alone still lets a click activate the
    window on some Windows builds; with WS_EX_NOACTIVATE a click is delivered
    without taking focus from the foreground app.  Returns True if applied.
    """
    if _USER32 is None or not hwnd:
        return False
    try:  # pragma: no cover - Windows only
        get_long, set_long = _USER32["GetWindowLongPtr"], _USER32["SetWindowLongPtr"]
        # Get/SetWindowLongPtr return 0 both for "style is 0" and for failure;
        # only GetLastError tells them apart.  Verify by reading the style back
        # rather than trusting the call (an invalid handle - e.g. the fake ids
        # of Qt's offscreen platform - would otherwise "succeed").
        ctypes.set_last_error(0)
        style = int(get_long(hwnd, GWL_EXSTYLE))
        if style == 0 and ctypes.get_last_error() != 0:
            _warn_once("no_activate", "could not read window style (error %d)", ctypes.get_last_error())
            return False
        wanted = (style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
        if wanted != style:
            set_long(hwnd, GWL_EXSTYLE, wanted)
            # Extended styles are cached by the window manager until the frame is refreshed.
            flags = SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED
            _USER32["SetWindowPos"](hwnd, None, 0, 0, 0, 0, flags)
        applied = int(get_long(hwnd, GWL_EXSTYLE))
        needed = WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
        if applied & needed != needed:
            _warn_once("no_activate", "window style not applied (error %d)", ctypes.get_last_error())
            return False
        return True
    except Exception as exc:  # pragma: no cover - never break the window over a style bit
        _warn_once("no_activate", "could not make window non-activating: %s", exc)
        return False


def foreground_fullscreen(own_hwnds: Iterable[int] = (), api: WinApi | None = None) -> bool:
    """Whether the foreground window covers its whole monitor.

    Ignored (False): no foreground window, our own windows, invisible or
    minimized windows and the desktop / taskbar shell windows (they are
    monitor-sized but nobody is "watching" them).  *api* is for tests.
    """
    api = api if api is not None else _API
    if api is None:
        return False
    try:
        hwnd = int(api.foreground_window() or 0)
        if not hwnd or hwnd in {int(h) for h in own_hwnds}:
            return False
        if not api.is_visible(hwnd) or api.is_minimized(hwnd):
            return False
        if api.class_name(hwnd) in SHELL_CLASSES:
            return False
        window = api.window_rect(hwnd)
        monitor = api.monitor_rect(hwnd)
        if window is None or monitor is None:
            return False
        return covers(window, monitor)
    except Exception as exc:
        _warn_once("fullscreen", "fullscreen check failed: %s", exc)
        return False
