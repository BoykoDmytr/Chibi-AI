"""Win32 helpers.  The decision logic is tested everywhere with a fake API;
the real ctypes calls only on Windows (and that is the only proof they work)."""

from __future__ import annotations

import ctypes
import logging
import os
import sys

import pytest

from chibi.desktop import platform_win
from chibi.desktop.platform_win import SHELL_CLASSES, Box, covers, foreground_fullscreen

MONITOR = Box(0, 0, 1920, 1080)
#: Notepad, Explorer, VS Code, Chrome: caption + thick frame + min/max boxes.
WS_OVERLAPPEDWINDOW = 0x00CF0000
#: Borderless games, F11 browsers, video players: no caption, no frame.
WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000
#: A maximized captioned window: its 8 px resize border is pushed off-screen.
MAXIMIZED_OVERHANG = Box(-8, -8, 1928, 1088)


class FakeApi:
    def __init__(self, hwnd: int = 42, rect: Box | None = MONITOR, monitor: Box | None = MONITOR,
                 visible: bool = True, minimized: bool = False, cls: str = "UnrealWindow",
                 maximized: bool = False, style: int = WS_POPUP | WS_VISIBLE) -> None:
        self.hwnd, self.rect, self.monitor = hwnd, rect, monitor
        self.visible, self.minimized, self.cls = visible, minimized, cls
        self.maximized, self.window_style = maximized, style

    def foreground_window(self) -> int:
        return self.hwnd

    def is_visible(self, hwnd: int) -> bool:
        return self.visible

    def is_minimized(self, hwnd: int) -> bool:
        return self.minimized

    def class_name(self, hwnd: int) -> str:
        return self.cls

    def window_rect(self, hwnd: int) -> Box | None:
        return self.rect

    def monitor_rect(self, hwnd: int) -> Box | None:
        return self.monitor

    def is_maximized(self, hwnd: int) -> bool:
        return self.maximized

    def style(self, hwnd: int) -> int:
        return self.window_style


class BrokenApi(FakeApi):
    def window_rect(self, hwnd: int) -> Box | None:
        raise OSError("access denied")


def test_covers():
    assert covers(MONITOR, MONITOR)
    # geometry alone: a rect beyond the monitor covers it (whether it is
    # *fullscreen* also depends on the style, see foreground_fullscreen)
    assert covers(MAXIMIZED_OVERHANG, MONITOR)
    assert not covers(Box(0, 0, 1920, 1040), MONITOR)  # maximized above the taskbar
    assert not covers(Box(10, 0, 1920, 1080), MONITOR)
    assert covers(Box(1920, 0, 3840, 1080), Box(1920, 0, 3840, 1080))  # second monitor
    assert not covers(MONITOR, Box(0, 0, 0, 0))  # degenerate monitor


def test_fullscreen_logic_with_fake_api():
    assert foreground_fullscreen(set(), api=FakeApi())
    assert not foreground_fullscreen({42}, api=FakeApi())  # our own window
    assert not foreground_fullscreen(set(), api=FakeApi(hwnd=0))
    assert not foreground_fullscreen(set(), api=FakeApi(visible=False))
    assert not foreground_fullscreen(set(), api=FakeApi(minimized=True))
    assert not foreground_fullscreen(set(), api=FakeApi(rect=Box(100, 100, 900, 700)))
    assert not foreground_fullscreen(set(), api=FakeApi(rect=None))
    assert not foreground_fullscreen(set(), api=FakeApi(monitor=None))
    for cls in SHELL_CLASSES:
        assert not foreground_fullscreen(set(), api=FakeApi(cls=cls))


def test_maximized_captioned_window_is_not_fullscreen():
    # Regression (win-1): on a monitor without a taskbar (auto-hide, or a
    # second display) a maximized Notepad's rect overhangs the whole monitor.
    # She used to hide whenever such a window had focus.
    notepad = FakeApi(rect=MAXIMIZED_OVERHANG, cls="Notepad", maximized=True, style=WS_OVERLAPPEDWINDOW)
    assert not foreground_fullscreen(set(), api=notepad)
    second = Box(1920, 0, 3840, 1080)
    browser = FakeApi(rect=Box(1912, -8, 3848, 1088), monitor=second, cls="Chrome_WidgetWin_1",
                      maximized=True, style=WS_OVERLAPPEDWINDOW | WS_VISIBLE)
    assert not foreground_fullscreen(set(), api=browser)


def test_real_fullscreen_windows_still_count():
    # a zoomed borderless window at the same rect (a game that maximizes a popup)
    assert foreground_fullscreen(set(), api=FakeApi(rect=MAXIMIZED_OVERHANG, maximized=True))
    # borderless / exclusive game, F11 browser, video player: monitor-sized, no caption
    assert foreground_fullscreen(set(), api=FakeApi(maximized=False))
    assert foreground_fullscreen(set(), api=FakeApi(cls="Chrome_WidgetWin_1", maximized=True,
                                                    style=WS_POPUP | WS_VISIBLE))
    # an old "windowed fullscreen" game: a captioned, NOT maximized window
    # moved so its frame is off-screen
    old_game = FakeApi(rect=Box(-8, -31, 1928, 1088), maximized=False, style=WS_OVERLAPPEDWINDOW)
    assert foreground_fullscreen(set(), api=old_game)
    # an unreadable style (0) is not a caption: geometry decides, as before
    assert foreground_fullscreen(set(), api=FakeApi(rect=MAXIMIZED_OVERHANG, maximized=True, style=0))


def test_fullscreen_never_raises(caplog):
    caplog.set_level(logging.WARNING, logger="chibi.desktop.platform_win")
    platform_win._warned.discard("fullscreen")
    assert foreground_fullscreen(set(), api=BrokenApi()) is False
    assert foreground_fullscreen(set(), api=BrokenApi()) is False
    assert sum("fullscreen check failed" in r.getMessage() for r in caplog.records) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows fallback")
def test_safe_noops_off_windows():
    assert not platform_win.supported()
    assert platform_win.default_api() is None
    assert platform_win.apply_no_activate(12345) is False
    assert platform_win.foreground_fullscreen({1, 2}) is False


def test_apply_no_activate_ignores_null_handle():
    assert platform_win.apply_no_activate(0) is False


# -- real Win32 (only meaningful on Windows) ------------------------------------------------------

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Win32 API")
#: Needs a real HWND: Qt's offscreen platform hands out fake window ids.  CI
#: runs these in a separate step with QT_QPA_PLATFORM=windows.
native_windows_only = pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("QT_QPA_PLATFORM", "") == "offscreen",
    reason="needs native Windows windows (QT_QPA_PLATFORM=windows)",
)


@windows_only
def test_ctypes_prototypes_are_pointer_sized():
    from ctypes import wintypes

    u = platform_win._USER32
    assert u is not None
    assert u["GetForegroundWindow"].restype is ctypes.c_void_p
    assert u["GetWindowRect"].argtypes[0] is ctypes.c_void_p
    assert u["IsZoomed"].argtypes == [ctypes.c_void_p]
    assert u["IsZoomed"].restype is wintypes.BOOL
    assert u["MonitorFromWindow"].argtypes[0] is ctypes.c_void_p
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert u["GetWindowLongPtr"].__name__ == "GetWindowLongPtrW"
        assert u["GetWindowLongPtr"].restype is ctypes.c_ssize_t
        assert u["SetWindowLongPtr"].argtypes[2] is ctypes.c_ssize_t


@native_windows_only
def test_apply_no_activate_sets_the_extended_style(qapp):
    from PySide6.QtWidgets import QWidget

    widget = QWidget()
    widget.resize(50, 50)
    widget.show()
    try:
        hwnd = int(widget.winId())
        assert platform_win.apply_no_activate(hwnd)
        style = int(platform_win._USER32["GetWindowLongPtr"](hwnd, platform_win.GWL_EXSTYLE))
        assert style & platform_win.WS_EX_NOACTIVATE
        assert style & platform_win.WS_EX_TOOLWINDOW
    finally:
        widget.close()


@windows_only
def test_foreground_fullscreen_runs_on_windows():
    assert isinstance(platform_win.foreground_fullscreen(set()), bool)
    api = platform_win.default_api()
    assert api is not None
    hwnd = api.foreground_window()
    if hwnd:
        assert api.monitor_rect(hwnd) is not None
        assert isinstance(api.is_maximized(hwnd), bool)
        assert 0 <= api.style(hwnd) <= 0xFFFFFFFF


@native_windows_only
def test_real_style_and_zoom_of_a_maximized_window(qapp):
    from PySide6.QtWidgets import QWidget

    widget = QWidget()  # an ordinary top-level window: caption + thick frame
    widget.showMaximized()
    qapp.processEvents()
    try:
        api = platform_win.default_api()
        assert api is not None
        hwnd = int(widget.winId())
        assert api.is_maximized(hwnd)
        assert api.style(hwnd) & platform_win.WS_CAPTION == platform_win.WS_CAPTION
    finally:
        widget.close()


@windows_only
def test_apply_no_activate_reports_failure_for_an_invalid_window():
    # A handle that is not a window must not be reported as "applied".
    assert platform_win.apply_no_activate(0x7FFF_FFF0) is False
