"""Global push-to-talk hotkey on Windows via ``RegisterHotKey``.

Deliberately *not* a low-level keyboard hook (``SetWindowsHookEx``): a hook
sees every key the user types, which is exactly what keyloggers do -
antivirus and anti-cheat software flag it, and it would contradict the
privacy promise.  ``RegisterHotKey`` only tells us that our one combination
was pressed.  Releasing the key (for hold-to-talk) is detected by polling
``GetAsyncKeyState`` for that key only, and only while it is held.

The hotkey lives on its own thread with a Win32 message loop; callbacks run
on that thread, so they must only hand work to the main thread.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
WM_USER = 0x0400
PM_NOREMOVE = 0x0000
ERROR_HOTKEY_ALREADY_REGISTERED = 1409

_MODIFIERS = {"ctrl": MOD_CONTROL, "control": MOD_CONTROL, "alt": MOD_ALT, "shift": MOD_SHIFT,
              "win": MOD_WIN}
_KEYS: dict[str, int] = {
    "space": 0x20, "pause": 0x13, "scrolllock": 0x91, "insert": 0x2D, "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pagedown": 0x22, "`": 0xC0, "backquote": 0xC0, "capslock": 0x14,
    **{f"f{i}": 0x6F + i for i in range(1, 25)},
    **{chr(c): ord(chr(c).upper()) for c in range(ord("a"), ord("z") + 1)},
    **{str(d): 0x30 + d for d in range(10)},
    **{f"num{d}": 0x60 + d for d in range(10)},
}
#: Keys that may be used without a modifier (they don't type text).
_BARE_OK = {"pause", "scrolllock", *(f"f{i}" for i in range(13, 25))}
_KEY_TITLES = {"space": "Пробіл", "`": "`", "backquote": "`", "scrolllock": "Scroll Lock",
               "pageup": "Page Up", "pagedown": "Page Down", "capslock": "Caps Lock"}

#: Choices offered in the settings (all tested to not type text).
PRESETS = ("ctrl+alt+space", "ctrl+shift+space", "ctrl+alt+m", "alt+`", "f13", "pause",
           "scrolllock", "ctrl+f12")


class HotkeyError(RuntimeError):
    def __init__(self, message: str, *, user_message: str = "") -> None:
        super().__init__(message)
        self.user_message = user_message or message


@dataclass(frozen=True, slots=True)
class Hotkey:
    modifiers: int
    vk: int
    text: str  # canonical "ctrl+alt+space"

    @property
    def title(self) -> str:
        """Human-readable: "Ctrl + Alt + Пробіл"."""
        parts = []
        for name in self.text.split("+"):
            if name in _KEY_TITLES:
                parts.append(_KEY_TITLES[name])
            elif name in _MODIFIERS:
                parts.append(name.capitalize())
            else:
                parts.append(name.upper())
        return " + ".join(parts)


def parse_hotkey(text: str) -> Hotkey:
    """"Ctrl+Alt+Space" -> Hotkey.  Raises ValueError for unknown keys or
    combinations that would steal ordinary typing (a bare "m", "shift+m")."""
    parts = [p.strip().lower() for p in text.replace(" ", "").split("+") if p.strip()]
    if not parts:
        raise ValueError("empty hotkey")
    *mods, key = parts
    modifiers = 0
    for m in mods:
        if m not in _MODIFIERS:
            raise ValueError(f"unknown modifier {m!r}")
        modifiers |= _MODIFIERS[m]
    if key not in _KEYS:
        raise ValueError(f"unknown key {key!r}")
    if key not in _BARE_OK and not modifiers & (MOD_CONTROL | MOD_ALT | MOD_WIN):
        raise ValueError(f"{text!r} would block normal typing; add Ctrl, Alt or Win")
    order = [name for name in ("ctrl", "alt", "shift", "win") if modifiers & _MODIFIERS[name]]
    return Hotkey(modifiers, _KEYS[key], "+".join([*order, key]))


def supported() -> bool:
    return sys.platform == "win32"


class GlobalHotkey:
    """Calls *on_press* when the combination is pressed and, if given,
    *on_release* when its main key is let go (hold-to-talk)."""

    _next_id = 0xC0DE

    def __init__(self, hotkey: str | Hotkey, on_press: Callable[[], None],
                 on_release: Callable[[], None] | None = None, poll_interval: float = 0.03) -> None:
        self.hotkey = hotkey if isinstance(hotkey, Hotkey) else parse_hotkey(hotkey)
        self._on_press = on_press
        self._on_release = on_release
        self._poll = poll_interval
        GlobalHotkey._next_id += 1
        self._id = GlobalHotkey._next_id
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ready = threading.Event()
        self._error: int | None = None
        self._stopping = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, timeout: float = 2.0) -> None:
        """Register the hotkey; raises HotkeyError if Windows refuses it
        (another program owns the combination) or this is not Windows."""
        if not supported():
            raise HotkeyError("global hotkeys need Windows",
                              user_message="Гаряча клавіша працює лише у Windows.")
        if self.running:
            return
        self._ready.clear()
        self._stopping.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="chibi-hotkey", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise HotkeyError("hotkey thread did not start")
        if self._error is not None:
            self._thread.join(1.0)
            self._thread = None
            if self._error == ERROR_HOTKEY_ALREADY_REGISTERED:
                raise HotkeyError(
                    f"{self.hotkey.text} is already registered",
                    user_message=f"Комбінацію {self.hotkey.title} вже зайняла інша програма. "
                                 "Обери іншу в налаштуваннях голосу.",
                )
            raise HotkeyError(f"RegisterHotKey failed: error {self._error}",
                              user_message="Не вдалося зареєструвати гарячу клавішу.")

    def stop(self, timeout: float = 2.0) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stopping.set()
        if thread.is_alive() and self._thread_id:
            import ctypes

            ctypes.WinDLL("user32").PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        thread.join(timeout)
        self._thread = None

    def _run(self) -> None:  # pragma: no cover - exercised by the Windows CI job
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32")
        user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                       wintypes.UINT, wintypes.UINT]
        user32.GetMessageW.restype = wintypes.BOOL
        user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                        wintypes.UINT, wintypes.UINT, wintypes.UINT]
        user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
        user32.RegisterHotKey.restype = wintypes.BOOL
        user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        user32.GetAsyncKeyState.restype = ctypes.c_short

        msg = wintypes.MSG()
        self._thread_id = kernel32.GetCurrentThreadId()
        # Make sure this thread has a message queue before anyone posts to it.
        user32.PeekMessageW(ctypes.byref(msg), None, WM_USER, WM_USER, PM_NOREMOVE)
        if not user32.RegisterHotKey(None, self._id, self.hotkey.modifiers | MOD_NOREPEAT,
                                     self.hotkey.vk):
            self._error = ctypes.get_last_error()
            self._ready.set()
            return
        self._ready.set()
        log.info("push-to-talk hotkey %s registered", self.hotkey.text)
        try:
            while not self._stopping.is_set():
                result = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if result in (0, -1):
                    break
                if msg.message != WM_HOTKEY or msg.wParam != self._id:
                    continue
                self._safe(self._on_press)
                if self._on_release is not None:
                    while (user32.GetAsyncKeyState(self.hotkey.vk) & 0x8000
                           and not self._stopping.is_set()):
                        time.sleep(self._poll)
                    self._safe(self._on_release)
        finally:
            user32.UnregisterHotKey(None, self._id)
            log.info("push-to-talk hotkey released")

    @staticmethod
    def _safe(callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:  # never let a callback kill the message loop
            log.exception("hotkey callback failed")
