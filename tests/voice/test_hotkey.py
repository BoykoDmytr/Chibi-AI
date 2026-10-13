import sys

import pytest

from chibi.voice import hotkey_win
from chibi.voice.hotkey_win import (
    MOD_ALT,
    MOD_CONTROL,
    MOD_SHIFT,
    PRESETS,
    GlobalHotkey,
    HotkeyError,
    parse_hotkey,
)


def test_parse_canonicalises():
    hk = parse_hotkey(" Alt + Ctrl + SPACE ")
    assert hk.modifiers == MOD_CONTROL | MOD_ALT
    assert hk.vk == 0x20
    assert hk.text == "ctrl+alt+space"
    assert hk.title == "Ctrl + Alt + Пробіл"


@pytest.mark.parametrize(("text", "vk"), [("f13", 0x7C), ("pause", 0x13), ("ctrl+f12", 0x7B),
                                           ("ctrl+alt+m", ord("M")), ("alt+`", 0xC0),
                                           ("ctrl+shift+1", 0x31), ("scrolllock", 0x91)])
def test_parse_keys(text, vk):
    assert parse_hotkey(text).vk == vk


@pytest.mark.parametrize("text", ["m", "shift+m", "space", "f5", "shift+space", "", "ctrl+",
                                  "ctrl+alt+nosuchkey", "hyper+m"])
def test_rejects_combinations_that_steal_typing_or_are_unknown(text):
    with pytest.raises(ValueError):
        parse_hotkey(text)


def test_shift_is_kept_as_a_modifier():
    assert parse_hotkey("ctrl+shift+space").modifiers == MOD_CONTROL | MOD_SHIFT


def test_all_presets_parse():
    for preset in PRESETS:
        assert parse_hotkey(preset).text == preset


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows behaviour")
def test_start_outside_windows_raises_a_friendly_error():
    hk = GlobalHotkey("ctrl+alt+space", on_press=lambda: None)
    with pytest.raises(HotkeyError) as info:
        hk.start()
    assert "Windows" in info.value.user_message
    hk.stop()  # no-op
    assert not hotkey_win.supported()


@pytest.mark.skipif(sys.platform != "win32", reason="needs the Win32 API")
def test_register_and_release_on_windows():
    """Real RegisterHotKey / message loop / UnregisterHotKey round trip.

    Pressing the key is not simulated (CI has no interactive input); this
    proves registration, conflict detection and clean shutdown.
    """
    first = GlobalHotkey("ctrl+alt+shift+f11", on_press=lambda: None)
    first.start()
    assert first.running
    second = GlobalHotkey("ctrl+alt+shift+f11", on_press=lambda: None)
    with pytest.raises(HotkeyError) as info:
        second.start()
    assert "інша програма" in info.value.user_message
    first.stop()
    assert not first.running
    # Released: the combination can be registered again.
    third = GlobalHotkey("ctrl+alt+shift+f11", on_press=lambda: None)
    third.start()
    third.stop()
