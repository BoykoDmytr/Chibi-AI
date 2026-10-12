import sys

import pytest

from chibi.activity.backend import ActivityBackend, ForegroundInfo, NullActivityBackend, default_backend


def test_null_backend_observes_nothing():
    backend = NullActivityBackend()
    assert isinstance(backend, ActivityBackend)
    assert backend.is_supported() is False
    assert backend.foreground(read_title=True) is None
    assert backend.foreground(read_title=False) is None
    assert backend.idle_seconds() is None
    assert backend.running_processes() == set()


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows fallback")
def test_default_backend_off_windows_is_null():
    backend = default_backend()
    assert isinstance(backend, NullActivityBackend)
    assert not backend.is_supported()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_default_backend_on_windows_is_real():
    from chibi.activity.windows_backend import WindowsActivityBackend

    backend = default_backend()
    assert isinstance(backend, WindowsActivityBackend)
    assert backend.is_supported()


def test_foreground_info_repr_hides_title_and_path():
    info = ForegroundInfo(pid=7, process_name="chrome.exe", exe_path=r"C:\Users\Olena\chrome.exe",
                          title="Мій банк - секрет")
    text = repr(info)
    assert "chrome.exe" in text
    assert "Olena" not in text
    assert "секрет" not in text
    assert info.title == "Мій банк - секрет"  # still available to the monitor


def test_foreground_info_defaults():
    info = ForegroundInfo(pid=1, process_name="code.exe")
    assert info.exe_path is None and info.title is None
