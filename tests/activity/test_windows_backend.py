"""Windows backend.

Two kinds of tests live here:

* decision logic (title gating, UWP resolution, tick wraparound) driven by a
  fake Win32 API - runs everywhere, proves the *logic*, not the API;
* real Win32 calls, ``skipif`` off Windows - the only proof that the ctypes
  signatures and the API calls work (CI runs them on windows-latest).
"""

import os
import sys

import psutil
import pytest

from chibi.activity import windows_backend as wb
from chibi.activity.backend import ForegroundInfo
from chibi.activity.windows_backend import WindowsActivityBackend, elapsed_ms, psutil_identity


class FakeApi:
    def __init__(self):
        self.foreground = 100
        self.pids = {100: 10}
        self.titles = {100: "Secret title"}
        self.children: dict[int, list[int]] = {}
        self.last_tick: int | None = 1000
        self.now_tick = 4000
        self.title_calls = 0

    def foreground_window(self):
        return self.foreground

    def window_pid(self, hwnd):
        return self.pids.get(hwnd, 0)

    def window_title(self, hwnd):
        self.title_calls += 1
        return self.titles.get(hwnd, "")

    def child_windows(self, hwnd, limit=512):
        return list(self.children.get(hwnd, []))

    def last_input_tick(self):
        return self.last_tick

    def tick_count(self):
        return self.now_tick


PROCESSES = {
    10: ("code.exe", r"C:\Program Files\VS Code\Code.exe"),
    20: ("applicationframehost.exe", r"C:\Windows\System32\ApplicationFrameHost.exe"),
    30: ("calculatorapp.exe", None),
}


@pytest.fixture
def api():
    return FakeApi()


@pytest.fixture
def backend(api):
    return WindowsActivityBackend(api=api, process_lookup=PROCESSES.get)  # type: ignore[arg-type]


# -- logic (all platforms) ------------------------------------------------------------


def test_module_imports_everywhere():
    assert wb.WindowsActivityBackend is WindowsActivityBackend


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows behaviour")
def test_unusable_off_windows():
    with pytest.raises(OSError):
        wb.Win32Api()
    with pytest.raises(OSError):
        WindowsActivityBackend()


def test_tick_wraparound():
    assert elapsed_ms(5000, 1000) == 4000
    assert elapsed_ms(0x10, 0xFFFFFFF0) == 0x20  # GetTickCount wrapped after 49.7 days
    assert elapsed_ms(1000, 1000) == 0


def test_foreground_without_title(backend, api):
    info = backend.foreground(read_title=False)
    expected = ForegroundInfo(pid=10, process_name="code.exe", exe_path=r"C:\Program Files\VS Code\Code.exe")
    assert info == expected
    assert info.title is None
    assert api.title_calls == 0  # the title is not even read


def test_foreground_with_title(backend, api):
    info = backend.foreground(read_title=True)
    assert info.title == "Secret title"
    assert api.title_calls == 1


def test_empty_title_is_none(backend, api):
    api.titles[100] = ""
    assert backend.foreground(read_title=True).title is None


def test_no_foreground_window(backend, api):
    api.foreground = None
    assert backend.foreground(read_title=True) is None


def test_window_without_process(backend, api):
    api.pids = {}
    assert backend.foreground(read_title=False) is None
    api.pids = {100: 99}  # process vanished between the calls
    assert backend.foreground(read_title=False) is None


def test_uwp_app_is_resolved_to_the_hosted_process(backend, api):
    api.pids = {100: 20, 201: 20, 202: 30}
    api.children = {100: [201, 202]}
    api.titles = {100: "Calculator"}
    info = backend.foreground(read_title=True)
    assert info.pid == 30
    assert info.process_name == "calculatorapp.exe"
    assert info.exe_path is None
    assert info.title == "Calculator"  # the frame's caption


def test_uwp_frame_without_foreign_child_reports_the_host(backend, api):
    api.pids = {100: 20, 201: 20}
    api.children = {100: [201]}
    info = backend.foreground(read_title=False)
    assert info.process_name == "applicationframehost.exe"


def test_idle_seconds(backend, api):
    api.last_tick, api.now_tick = 1000, 4500
    assert backend.idle_seconds() == 3.5
    api.last_tick, api.now_tick = 0xFFFFFF00, 0x100
    assert backend.idle_seconds() == 0.512
    api.last_tick = None
    assert backend.idle_seconds() is None


def test_running_processes_are_lower_case_names(backend):
    names = backend.running_processes()
    assert names
    assert all(name == name.lower() for name in names)
    assert psutil.Process().name().lower() in names


def test_psutil_identity_of_this_process():
    name, exe = psutil_identity(os.getpid())
    assert name == psutil.Process().name().lower()
    assert exe is None or isinstance(exe, str)


def test_psutil_identity_of_a_missing_process():
    pid = max(psutil.pids()) + 100_000
    assert psutil_identity(pid) is None


# -- real Win32 API (Windows only) ------------------------------------------------------


windows_only = pytest.mark.skipif(sys.platform != "win32", reason="real Win32 API")


@windows_only
def test_every_function_has_a_signature():
    import ctypes

    api = wb.Win32Api()
    functions = api.functions()
    assert set(functions) == {
        "GetForegroundWindow", "GetWindowThreadProcessId", "GetWindowTextLengthW", "GetWindowTextW",
        "EnumChildWindows", "GetLastInputInfo", "GetTickCount",
    }
    for name, function in functions.items():
        assert function.argtypes is not None, name
        assert function.restype is not None, name
    assert functions["GetForegroundWindow"].restype is ctypes.c_void_p  # 64-bit handle, not int


@windows_only
def test_real_foreground_returns_sane_types():
    backend = WindowsActivityBackend()
    assert backend.is_supported()
    info = backend.foreground(read_title=False)
    # CI may run without an interactive desktop: no foreground window is a valid answer.
    if info is not None:
        assert isinstance(info.pid, int) and info.pid > 0
        assert info.process_name and info.process_name == info.process_name.lower()
        assert info.exe_path is None or isinstance(info.exe_path, str)
        assert info.title is None
    titled = backend.foreground(read_title=True)
    if titled is not None:
        assert titled.title is None or isinstance(titled.title, str)


@windows_only
def test_real_idle_and_processes():
    backend = WindowsActivityBackend()
    idle = backend.idle_seconds()
    assert idle is None or (isinstance(idle, float) and idle >= 0)
    names = backend.running_processes()
    assert psutil.Process().name().lower() in names


@windows_only
def test_real_child_window_enumeration():
    api = wb.Win32Api()
    hwnd = api.foreground_window()
    if hwnd is None:
        pytest.skip("no foreground window in this session")
    children = api.child_windows(hwnd, limit=5)
    assert isinstance(children, list) and len(children) <= 5
    assert all(isinstance(c, int) for c in children)
    assert api.window_pid(hwnd) > 0
    assert isinstance(api.window_title(hwnd), str)
