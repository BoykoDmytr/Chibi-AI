"""Windows activity backend: ctypes (user32/kernel32) + psutil, no pywin32.

What it reads, and nothing more:

* the foreground window's owning process (name, exe path);
* the window title - only when the caller passes ``read_title=True``, which
  the monitor does only while the user granted ``WINDOW_TITLES``;
* *when* the last keyboard/mouse input happened (``GetLastInputInfo``).  This
  is a single timestamp kept by Windows; no hook is installed and no key is
  ever seen, so this cannot work as a keylogger;
* the names of running processes.

Every ctypes function gets explicit ``argtypes``/``restype``.  Without them
ctypes assumes ``int`` and silently truncates 64-bit window handles.  The DLLs
are loaded through private ``WinDLL`` instances (not the shared
``ctypes.windll``), so the signatures declared here cannot clash with ones
other modules declare for the same functions.

The module imports cleanly everywhere (CI runs on Linux too); binding happens
when :class:`Win32Api` is constructed, which raises ``OSError`` off Windows.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Mapping
from typing import Any

import psutil

from chibi.activity.backend import ForegroundInfo

log = logging.getLogger(__name__)

#: UWP (Store) apps run inside a frame window owned by this host process; the
#: real app owns a child window of the frame.
UWP_FRAME_HOST = "applicationframehost.exe"
#: Titles longer than this are cut: classification only needs the beginning.
MAX_TITLE_CHARS = 2048
#: Safety bound for child-window enumeration of a single frame window.
MAX_CHILD_WINDOWS = 512
_TICK_MASK = 0xFFFFFFFF

#: pid -> (lower-case process name, exe path or None); None if the process is gone/unreadable.
ProcessLookup = Callable[[int], tuple[str, str | None] | None]


def elapsed_ms(now_tick: int, last_tick: int) -> int:
    """Milliseconds from *last_tick* to *now_tick* (both ``GetTickCount`` values).

    The tick counter is 32-bit and wraps every ~49.7 days; masking the
    difference keeps the result right across the wrap.
    """
    return (now_tick - last_tick) & _TICK_MASK


class Win32Api:
    """The handful of Win32 calls the backend needs, with 64-bit-safe signatures."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("the Win32 activity API is only available on Windows")
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._wintypes = wintypes
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        hwnd_t = ctypes.c_void_p  # HWND is a pointer: 64-bit on 64-bit Windows

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        self._LASTINPUTINFO = LASTINPUTINFO
        self._WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, hwnd_t, wintypes.LPARAM)

        user32.GetForegroundWindow.argtypes = []
        user32.GetForegroundWindow.restype = hwnd_t
        user32.GetWindowThreadProcessId.argtypes = [hwnd_t, ctypes.POINTER(wintypes.DWORD)]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        user32.GetWindowTextLengthW.argtypes = [hwnd_t]
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = [hwnd_t, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.EnumChildWindows.argtypes = [hwnd_t, self._WNDENUMPROC, wintypes.LPARAM]
        user32.EnumChildWindows.restype = wintypes.BOOL
        user32.GetLastInputInfo.argtypes = [ctypes.POINTER(LASTINPUTINFO)]
        user32.GetLastInputInfo.restype = wintypes.BOOL
        kernel32.GetTickCount.argtypes = []
        kernel32.GetTickCount.restype = wintypes.DWORD

        self._user32 = user32
        self._kernel32 = kernel32

    def functions(self) -> Mapping[str, Any]:
        """The bound function pointers (for signature checks in tests)."""
        u, k = self._user32, self._kernel32
        return {
            "GetForegroundWindow": u.GetForegroundWindow,
            "GetWindowThreadProcessId": u.GetWindowThreadProcessId,
            "GetWindowTextLengthW": u.GetWindowTextLengthW,
            "GetWindowTextW": u.GetWindowTextW,
            "EnumChildWindows": u.EnumChildWindows,
            "GetLastInputInfo": u.GetLastInputInfo,
            "GetTickCount": k.GetTickCount,
        }

    def foreground_window(self) -> int | None:
        hwnd = self._user32.GetForegroundWindow()
        return int(hwnd) if hwnd else None

    def window_pid(self, hwnd: int) -> int:
        """Owning process id of *hwnd*, 0 if the window is gone."""
        pid = self._wintypes.DWORD(0)
        if not self._user32.GetWindowThreadProcessId(hwnd, self._ctypes.byref(pid)):
            return 0
        return int(pid.value)

    def window_title(self, hwnd: int) -> str:
        """Caption text.  For windows of other processes Windows returns the
        cached caption without messaging the owner, so a hung app cannot block us."""
        length = self._user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        length = min(length, MAX_TITLE_CHARS)
        buffer = self._ctypes.create_unicode_buffer(length + 1)
        copied = self._user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value[:copied] if copied > 0 else ""

    def child_windows(self, hwnd: int, limit: int = MAX_CHILD_WINDOWS) -> list[int]:
        children: list[int] = []

        def collect(child: int | None, _lparam: int) -> bool:
            # An exception escaping a ctypes callback is only printed; keep it inside.
            try:
                if child:
                    children.append(int(child))
                return len(children) < limit
            except Exception:
                return False

        # The callback object must stay referenced while Windows may call it:
        # it is a local that lives until EnumChildWindows (synchronous) returns.
        callback = self._WNDENUMPROC(collect)
        self._user32.EnumChildWindows(hwnd, callback, 0)
        return children

    def last_input_tick(self) -> int | None:
        """``GetTickCount`` value of the last input event, None if the call failed."""
        info = self._LASTINPUTINFO()
        info.cbSize = self._ctypes.sizeof(info)
        if not self._user32.GetLastInputInfo(self._ctypes.byref(info)):
            return None
        return int(info.dwTime)

    def tick_count(self) -> int:
        return int(self._kernel32.GetTickCount())


def psutil_identity(pid: int) -> tuple[str, str | None] | None:
    """(lower-case name, exe path) of *pid*.

    The exe path of elevated or protected processes is unreadable without
    admin rights; that is normal and gives ``None`` for the path, not an error.
    """
    try:
        proc = psutil.Process(pid)
        with proc.oneshot():
            name = proc.name()
            try:
                exe: str | None = proc.exe() or None
            except (psutil.Error, OSError):
                exe = None
    except (psutil.Error, OSError):
        return None
    if not name:
        return None
    return name.lower(), exe


class WindowsActivityBackend:
    """Real activity backend for Windows 10/11.

    *api* and *process_lookup* are injectable so the decision logic (UWP
    resolution, title gating) can be exercised without Windows; production
    code passes neither.
    """

    def __init__(self, api: Win32Api | None = None, process_lookup: ProcessLookup | None = None) -> None:
        self._api = api if api is not None else Win32Api()
        self._lookup: ProcessLookup = process_lookup if process_lookup is not None else psutil_identity

    def is_supported(self) -> bool:
        return True

    def foreground(self, read_title: bool) -> ForegroundInfo | None:
        hwnd = self._api.foreground_window()
        if hwnd is None:
            return None
        pid = self._api.window_pid(hwnd)
        if pid <= 0:
            return None
        identity = self._lookup(pid)
        if identity is None:
            return None
        name, exe = identity
        if name == UWP_FRAME_HOST:
            resolved = self._resolve_uwp(hwnd, pid)
            if resolved is not None:
                pid, name, exe = resolved
        title: str | None = None
        if read_title:
            title = self._api.window_title(hwnd) or None
        return ForegroundInfo(pid=pid, process_name=name, exe_path=exe, title=title)

    def _resolve_uwp(self, frame_hwnd: int, host_pid: int) -> tuple[int, str, str | None] | None:
        """The Store app hosted by an ApplicationFrameHost frame: the owner of
        the first child window that belongs to a different process."""
        for child in self._api.child_windows(frame_hwnd):
            child_pid = self._api.window_pid(child)
            if child_pid <= 0 or child_pid == host_pid:
                continue
            identity = self._lookup(child_pid)
            if identity is not None:
                return child_pid, identity[0], identity[1]
        return None

    def idle_seconds(self) -> float | None:
        last = self._api.last_input_tick()  # read before "now" so the difference is never negative
        if last is None:
            return None
        return elapsed_ms(self._api.tick_count(), last) / 1000.0

    def running_processes(self) -> set[str]:
        names: set[str] = set()
        try:
            for proc in psutil.process_iter(["name"]):
                name = proc.info.get("name")
                if name:
                    names.add(name.lower())
        except (psutil.Error, OSError) as exc:
            # The exception text may name a process; the type is enough to debug.
            log.warning("process scan failed (%s)", type(exc).__name__)
        return names
