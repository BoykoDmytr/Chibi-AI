"""Real window capture - only provable on Windows with real HWNDs
(the CI "Native-window tests" step runs this with QT_QPA_PLATFORM=windows)."""

import os
import sys

import pytest

from chibi.vision.capture_win import CaptureError, Win32WindowCapture

native = pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("QT_QPA_PLATFORM") != "windows",
    reason="needs real Win32 windows",
)


def _show_solid_window(qapp, color):
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QWidget

    w = QWidget()
    w.setAutoFillBackground(True)
    pal = w.palette()
    pal.setColor(QPalette.ColorRole.Window, QColor(*color))
    w.setPalette(pal)
    w.resize(320, 240)
    w.show()
    w.raise_()
    for _ in range(20):
        qapp.processEvents()
    return w


@native
def test_printwindow_captures_the_window_content(qapp):
    window = _show_solid_window(qapp, (200, 40, 90))
    try:
        frame = Win32WindowCapture().capture(int(window.winId()))
        assert frame.width >= 300 and frame.height >= 200
        b, g, r = frame.data[(frame.height // 2 * frame.width + frame.width // 2) * 4:][:3]
        assert (abs(r - 200), abs(g - 40), abs(b - 90)) < (8, 8, 8)
        assert frame.method in ("printwindow", "screen")
    finally:
        window.close()


@native
def test_occluding_window_is_not_in_the_capture(qapp):
    """PrintWindow returns the target's own pixels even if another window
    (a messenger, a notification) covers it."""
    target = _show_solid_window(qapp, (10, 180, 10))
    cover = _show_solid_window(qapp, (250, 250, 0))
    cover.move(target.pos())
    for _ in range(20):
        qapp.processEvents()
    try:
        frame = Win32WindowCapture(allow_screen_fallback=False).capture(int(target.winId()))
        _b, g, r = frame.data[(frame.height // 2 * frame.width + frame.width // 2) * 4:][:3]
        assert g > 150 and r < 60, "the covering window leaked into the capture"
    finally:
        cover.close()
        target.close()


@native
def test_closed_window_is_reported_as_gone(qapp):
    window = _show_solid_window(qapp, (1, 2, 3))
    hwnd = int(window.winId())
    window.close()
    window.deleteLater()
    for _ in range(10):
        qapp.processEvents()
    with pytest.raises(CaptureError):
        Win32WindowCapture().capture(hwnd)
    with pytest.raises(CaptureError) as info:
        Win32WindowCapture().capture(0)
    assert info.value.reason == "gone"
