"""Capture one window's pixels on Windows (Phase 3: screen vision).

Method: ``PrintWindow(hwnd, PW_CLIENTONLY | PW_RENDERFULLCONTENT)``, a
documented Win32 API that asks DWM for *that window's* content - other
windows on top of it (a messenger, a notification, Mia herself) are not in
the picture.  It works for normal and borderless-fullscreen DirectX games on
Windows 10/11.  Exclusive-fullscreen games may return a black frame; then,
if the game is the foreground window, the visible screen area under it is
copied instead (``BitBlt`` without ``CAPTUREBLT``, so layered windows such as
Mia are excluded).  Still black -> :class:`CaptureError` ("black").

Windows Graphics Capture would also handle exclusive fullscreen but needs a
Direct3D 11 interop layer; :class:`CaptureBackend` is the seam for adding it.

No hooks, no injection, no reading of game memory: this is what any
screenshot tool does, and only for windows the user allowed.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger(__name__)

PW_CLIENTONLY = 0x1
PW_RENDERFULLCONTENT = 0x2
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0
#: Larger client areas are refused (8K and up: memory, not a game window).
MAX_PIXELS = 7680 * 4320


@dataclass(frozen=True, slots=True)
class Frame:
    """BGRA (Qt's Format_RGB32 / ARGB32 byte order), top-down rows."""

    width: int
    height: int
    data: bytes
    hwnd: int
    captured_at: float
    method: str  # "printwindow" | "screen" | "fake"


class CaptureError(RuntimeError):
    """``reason``: "unsupported" | "gone" | "minimized" | "too_small" | "too_big"
    | "black" | "failed"."""

    def __init__(self, reason: str, message: str = "", *, user_message: str = "") -> None:
        super().__init__(message or reason)
        self.reason = reason
        self.user_message = user_message or _USER_MESSAGES.get(reason, "Не вдалося побачити екран.")


_USER_MESSAGES = {
    "unsupported": "Бачення екрана працює лише у Windows.",
    "gone": "Вікно гри вже закрилося.",
    "minimized": "Вікно гри згорнуте - мені нічого не видно.",
    "too_small": "Вікно надто маленьке, щоб щось роздивитися.",
    "too_big": "Вікно завелике для аналізу.",
    "black": "Гра не віддає зображення (схоже на ексклюзивний повноекранний режим). "
             "Спробуй у грі режим «Вікно без рамки».",
    "failed": "Не вдалося захопити вікно.",
}


class CaptureBackend(Protocol):
    def capture(self, hwnd: int) -> Frame: ...


def is_mostly_black(data: bytes, width: int, height: int, threshold: int = 10,
                    samples: int = 4096) -> bool:
    """True if (nearly) every sampled pixel is black - a failed capture.

    Dark game scenes still have some bright pixels (HUD, light sources);
    a failed PrintWindow/BitBlt is uniformly 0.
    """
    total = width * height
    if total <= 0 or len(data) < total * 4:
        return True
    step = max(1, total // samples)
    bright = 0
    for checked, index in enumerate(range(0, total, step), start=1):
        offset = index * 4
        if max(data[offset], data[offset + 1], data[offset + 2]) > threshold:
            bright += 1
            if bright > checked // 200 + 2:  # > 0.5% bright pixels: real content
                return False
    return True


class Win32WindowCapture:
    """PrintWindow with a foreground-only screen fallback."""

    def __init__(self, *, allow_screen_fallback: bool = True, min_size: int = 64) -> None:
        if sys.platform != "win32":
            raise CaptureError("unsupported")
        self.allow_screen_fallback = allow_screen_fallback
        self.min_size = min_size
        self._api = _Win32()

    def capture(self, hwnd: int) -> Frame:
        api = self._api
        if not hwnd or not api.user32.IsWindow(hwnd):
            raise CaptureError("gone")
        if api.user32.IsIconic(hwnd):
            raise CaptureError("minimized")
        width, height, left, top = api.client_area(hwnd)
        if width < self.min_size or height < self.min_size:
            raise CaptureError("too_small")
        if width * height > MAX_PIXELS:
            raise CaptureError("too_big")
        data = api.grab(hwnd, width, height, left, top, use_screen=False)
        method = "printwindow"
        if data is None or is_mostly_black(data, width, height):
            if self.allow_screen_fallback and api.user32.GetForegroundWindow() == hwnd:
                data = api.grab(hwnd, width, height, left, top, use_screen=True)
                method = "screen"
            if data is None:
                raise CaptureError("failed")
            if is_mostly_black(data, width, height):
                raise CaptureError("black")
        return Frame(width, height, data, hwnd, time.time(), method)


class _Win32:  # pragma: no cover - exercised by the Windows CI job
    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.wintypes = wintypes
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        self.user32, self.gdi32 = user32, gdi32

        HWND, HDC, HBITMAP, HGDIOBJ = wintypes.HWND, wintypes.HDC, wintypes.HBITMAP, wintypes.HGDIOBJ
        user32.IsWindow.argtypes = [HWND]
        user32.IsIconic.argtypes = [HWND]
        user32.GetForegroundWindow.restype = HWND
        user32.GetClientRect.argtypes = [HWND, ctypes.POINTER(wintypes.RECT)]
        user32.ClientToScreen.argtypes = [HWND, ctypes.POINTER(wintypes.POINT)]
        user32.GetDC.argtypes = [HWND]
        user32.GetDC.restype = HDC
        user32.ReleaseDC.argtypes = [HWND, HDC]
        user32.PrintWindow.argtypes = [HWND, HDC, wintypes.UINT]
        user32.PrintWindow.restype = wintypes.BOOL
        gdi32.CreateCompatibleDC.argtypes = [HDC]
        gdi32.CreateCompatibleDC.restype = HDC
        gdi32.CreateDIBSection.argtypes = [HDC, ctypes.c_void_p, wintypes.UINT,
                                           ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD]
        gdi32.CreateDIBSection.restype = HBITMAP
        gdi32.SelectObject.argtypes = [HDC, HGDIOBJ]
        gdi32.SelectObject.restype = HGDIOBJ
        gdi32.DeleteObject.argtypes = [HGDIOBJ]
        gdi32.DeleteDC.argtypes = [HDC]
        gdi32.BitBlt.argtypes = [HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
        gdi32.BitBlt.restype = wintypes.BOOL

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                        ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                        ("biClrImportant", wintypes.DWORD)]

        self.BITMAPINFOHEADER = BITMAPINFOHEADER

    def client_area(self, hwnd: int) -> tuple[int, int, int, int]:
        rect = self.wintypes.RECT()
        if not self.user32.GetClientRect(hwnd, self.ctypes.byref(rect)):
            raise CaptureError("gone")
        origin = self.wintypes.POINT(0, 0)
        self.user32.ClientToScreen(hwnd, self.ctypes.byref(origin))
        return rect.right - rect.left, rect.bottom - rect.top, origin.x, origin.y

    def grab(self, hwnd: int, width: int, height: int, left: int, top: int,
             use_screen: bool) -> bytes | None:
        ctypes = self.ctypes
        user32, gdi32 = self.user32, self.gdi32
        screen_dc = user32.GetDC(None)
        if not screen_dc:
            return None
        mem_dc = gdi32.CreateCompatibleDC(screen_dc)
        header = self.BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(header)
        header.biWidth = width
        header.biHeight = -height  # top-down rows
        header.biPlanes = 1
        header.biBitCount = 32
        header.biCompression = BI_RGB
        bits = ctypes.c_void_p()
        bitmap = gdi32.CreateDIBSection(mem_dc, ctypes.byref(header), DIB_RGB_COLORS,
                                        ctypes.byref(bits), None, 0)
        if not bitmap or not bits.value:
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(None, screen_dc)
            return None
        previous = gdi32.SelectObject(mem_dc, bitmap)
        try:
            if use_screen:
                ok = gdi32.BitBlt(mem_dc, 0, 0, width, height, screen_dc, left, top, SRCCOPY)
            else:
                ok = user32.PrintWindow(hwnd, mem_dc, PW_CLIENTONLY | PW_RENDERFULLCONTENT)
            gdi32.GdiFlush()
            if not ok:
                log.debug("capture call failed (error %s)", ctypes.get_last_error())
                return None
            return ctypes.string_at(bits.value, width * height * 4)
        finally:
            gdi32.SelectObject(mem_dc, previous)
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(None, screen_dc)


def exclude_from_capture(hwnd: int) -> bool:  # pragma: no cover - Windows only
    """Hide one of *our* windows from every screen capture (Windows 10 2004+),
    so Mia never analyses herself or her own speech bubbles."""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    WDA_EXCLUDEFROMCAPTURE = 0x11
    try:
        user32 = ctypes.WinDLL("user32")
        user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
        user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
        return bool(user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))
    except (OSError, AttributeError, ctypes.ArgumentError):
        return False
