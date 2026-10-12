import logging
import threading
import time
from collections.abc import Callable
from dataclasses import replace

import pytest

from chibi.activity.backend import ForegroundInfo
from chibi.activity.classifier import ActivityClassifier
from chibi.activity.monitor import ActivityMonitor
from chibi.core.config import ActivitySettings
from chibi.core.event_bus import EventBus
from chibi.core.events import (
    ActivityCategory as C,
)
from chibi.core.events import (
    ActivityMilestone,
    AppClosed,
    AppInfo,
    AppLaunched,
    Event,
    ForegroundAppChanged,
    GameEnded,
    GameStarted,
    Permission,
    UserIdle,
    UserReturned,
)
from chibi.gaming.games_db import GameDatabase
from chibi.memory.repositories import PermissionRepository
from chibi.privacy.filters import PrivacyFilter
from chibi.privacy.permissions import PermissionManager

STEAM_GAME = r"D:\SteamLibrary\steamapps\common\Cool Game\Binaries\CoolGame.exe"


class FakeBackend:
    """Scripted backend: a test sets what the next cycle observes."""

    def __init__(self) -> None:
        self.fg: ForegroundInfo | None = None
        self.idle: float | None = 0.0
        self.processes: set[str] = {"explorer.exe", "svchost.exe"}
        self.calls: list[str] = []
        self.title_requests: list[bool] = []
        self.always_return_title = False  # a misbehaving backend
        self.hooks: dict[str, Callable[[], None]] = {}
        self.fail: set[str] = set()

    def is_supported(self) -> bool:
        return True

    def _call(self, name: str) -> None:
        self.calls.append(name)
        hook = self.hooks.get(name)
        if hook is not None:
            hook()
        if name in self.fail:
            raise RuntimeError(rf"{name} failed for secret-app.exe in C:\Users\olena")

    def foreground(self, read_title: bool) -> ForegroundInfo | None:
        self.title_requests.append(read_title)
        self._call("foreground")
        if self.fg is None:
            return None
        if read_title or self.always_return_title:
            return self.fg
        return replace(self.fg, title=None)

    def idle_seconds(self) -> float | None:
        self._call("idle")
        return self.idle

    def running_processes(self) -> set[str]:
        self._call("scan")
        return set(self.processes)


def fg(process: str, title: str | None = None, exe: str | None = None) -> ForegroundInfo:
    return ForegroundInfo(pid=1234, process_name=process, exe_path=exe, title=title)


class Harness:
    def __init__(self, db, clock, bus: EventBus | None = None) -> None:
        self.clock = clock
        self.bus = bus or EventBus(max_queue=1000)
        self.events: list[Event] = []
        self.bus.subscribe(Event, self.events.append)
        self.backend = FakeBackend()
        self.permissions = PermissionManager(PermissionRepository(db), None, clock)
        self.settings = ActivitySettings(
            poll_interval_seconds=3.0, process_scan_interval_seconds=15.0, idle_threshold_minutes=5
        )
        self.monitor = ActivityMonitor(
            self.backend, self.permissions, self.bus, ActivityClassifier(GameDatabase.load()),
            PrivacyFilter(), lambda: self.settings, clock,
        )

    def grant(self, titles: bool = False) -> None:
        self.permissions.grant(Permission.APP_METADATA)
        if titles:
            self.permissions.grant(Permission.WINDOW_TITLES)

    def poll(self, advance: float = 3.0) -> list[Event]:
        self.clock.advance(advance)
        start = len(self.events)
        self.monitor.poll_once()
        self.bus.dispatch_pending()
        return self.events[start:]

    def run_for(self, seconds: float, step: float = 60.0) -> list[Event]:
        out: list[Event] = []
        for _ in range(round(seconds / step)):
            out += self.poll(step)
        return out


@pytest.fixture
def h(db, clock):
    return Harness(db, clock)


def of(events: list[Event], kind: type) -> list:
    return [e for e in events if isinstance(e, kind)]


def wait_for(condition: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


# -- permission gate ------------------------------------------------------------------------


def test_nothing_at_all_without_permission(h):
    h.backend.fg = fg("code.exe")
    h.backend.processes |= {"cs2.exe", "code.exe"}
    h.backend.idle = 10_000
    for _ in range(5):
        assert h.poll() == []
    assert h.backend.calls == []  # the backend is not even asked


def test_first_cycle_reports_foreground_and_running_games(h):
    h.grant()
    h.backend.fg = fg("code.exe")
    h.backend.processes |= {"code.exe", "cs2.exe", "steam.exe"}
    events = h.poll(0)
    assert [type(e) for e in events] == [ForegroundAppChanged, GameStarted]
    changed, started = events
    assert changed.app == AppInfo("code.exe", "VS Code")
    assert (changed.category, changed.confidence) == (C.PROGRAMMING, 0.9)
    assert (started.game_id, started.title, started.process_name) == ("cs2", "Counter-Strike 2", "cs2.exe")
    assert of(events, AppLaunched) == []  # apps already running were not "launched" now
    assert all(e.timestamp == h.clock.now() for e in events)


@pytest.mark.parametrize("stage", ["foreground", "idle", "scan"])
def test_revocation_during_a_cycle_discards_its_results(h, stage):
    h.grant()
    h.backend.fg = fg("code.exe")
    h.backend.processes |= {"cs2.exe"}
    h.backend.idle = 400
    h.backend.hooks[stage] = lambda: h.permissions.revoke(Permission.APP_METADATA)
    assert h.poll(0) == []
    assert h.backend.calls[-1] == stage  # nothing more was read after the revocation
    # nothing gathered is remembered: after a new grant everything is reported afresh
    h.backend.hooks.clear()
    h.grant()
    events = h.poll()
    assert len(of(events, ForegroundAppChanged)) == 1
    assert len(of(events, GameStarted)) == 1


def test_revocation_between_publishes_drops_the_rest(db, clock):
    holder: list[Harness] = []

    class RevokingBus(EventBus):
        def publish(self, event):
            result = super().publish(event)
            holder[0].permissions.revoke(Permission.APP_METADATA)
            return result

    h = Harness(db, clock, bus=RevokingBus(max_queue=1000))
    holder.append(h)
    h.grant()
    h.backend.fg = fg("code.exe")
    h.backend.processes |= {"cs2.exe"}
    events = h.poll(0)
    assert [type(e) for e in events] == [ForegroundAppChanged]  # GameStarted was discarded
    h.grant()
    assert [type(e) for e in h.poll()] == [ForegroundAppChanged]  # state was dropped, so it is new again


def test_revoking_forgets_everything_silently(h):
    h.grant()
    h.backend.fg = fg("code.exe")
    h.backend.processes |= {"cs2.exe"}
    h.poll(0)
    h.permissions.revoke(Permission.APP_METADATA)
    h.backend.processes -= {"cs2.exe"}
    calls = len(h.backend.calls)
    assert h.poll(15) == []  # no GameEnded: the tracker saves sessions on the PermissionChanged itself
    assert len(h.backend.calls) == calls
    h.grant()
    events = h.poll(15)
    assert [type(e) for e in events] == [ForegroundAppChanged]


# -- window titles ------------------------------------------------------------------------


def test_titles_are_requested_only_with_permission(h):
    h.grant()
    h.backend.fg = fg("chrome.exe", title="Lo-fi beats - YouTube")
    events = h.poll(0)
    assert h.backend.title_requests == [False]
    assert of(events, ForegroundAppChanged)[0].category is C.BROWSING
    h.permissions.grant(Permission.WINDOW_TITLES)
    events = h.poll()
    assert h.backend.title_requests[-1] is True
    assert of(events, ForegroundAppChanged)[0].category is C.WATCHING_VIDEO
    h.permissions.revoke(Permission.WINDOW_TITLES)
    events = h.poll()
    assert h.backend.title_requests[-1] is False
    assert of(events, ForegroundAppChanged)[0].category is C.BROWSING


def test_a_title_returned_without_permission_is_ignored(h):
    h.grant()
    h.backend.always_return_title = True
    h.backend.fg = fg("chrome.exe", title="Lo-fi beats - YouTube")
    events = h.poll(0)
    assert of(events, ForegroundAppChanged)[0].category is C.BROWSING


def test_titles_never_leave_the_monitor(h):
    h.grant(titles=True)
    secret = "Секретний проєкт Олени"
    h.backend.processes |= {"chrome.exe", "code.exe"}
    collected = []
    for title in (f"{secret} - YouTube", f"{secret} - GitHub", f"{secret} - weather"):
        h.backend.fg = fg("chrome.exe", title=title, exe=r"C:\Users\olena\chrome.exe")
        collected += h.poll(20)
    h.backend.idle = 1000
    collected += h.poll(20)
    h.backend.idle = 0
    collected += h.run_for(3 * 3600, step=600)
    assert collected
    text = repr(collected)
    assert "Секретний" not in text
    assert "olena" not in text


# -- privacy --------------------------------------------------------------------------------


def test_private_apps_are_redacted(h):
    h.grant(titles=True)
    h.backend.fg = fg("telegram.exe", title="Мама")
    h.backend.processes |= {"telegram.exe"}
    events = h.poll(0)
    assert len(events) == 1
    (changed,) = events
    assert changed.app == AppInfo.private_app()
    assert (changed.category, changed.confidence) == (C.PRIVATE, 1.0)
    # switching to another private app reveals nothing
    h.backend.fg = fg("signal.exe")
    assert h.poll() == []
    # private apps launching and closing are never reported
    h.backend.processes |= {"viber.exe"}
    assert h.poll(15) == []
    h.backend.processes -= {"viber.exe", "telegram.exe"}
    assert h.poll(15) == []
    # hours in a private app: no milestone
    assert of(h.run_for(4 * 3600, step=600), ActivityMilestone) == []
    assert "telegram" not in repr(h.events).lower()


def test_private_title(h):
    h.grant(titles=True)
    h.backend.fg = fg("chrome.exe", title="Приват24 — Google Chrome")
    (changed,) = h.poll(0)
    assert changed.category is C.PRIVATE and changed.app.private


def test_excluded_game_is_invisible(h):
    h.monitor.reconfigure(ActivityClassifier(GameDatabase.load()), PrivacyFilter(["cs2.exe"]))
    h.grant()
    h.backend.fg = fg("cs2.exe")
    h.backend.processes |= {"cs2.exe"}
    events = h.poll(0)
    assert [type(e) for e in events] == [ForegroundAppChanged]
    assert events[0].category is C.PRIVATE
    h.backend.processes -= {"cs2.exe"}
    assert h.poll(15) == []


def test_failing_privacy_filter_fails_closed(h):
    class BrokenPrivacy(PrivacyFilter):
        def is_private(self, info):
            raise RuntimeError("boom")

        def is_private_process(self, process_name):
            raise RuntimeError("boom")

    h.monitor.reconfigure(ActivityClassifier(GameDatabase.load()), BrokenPrivacy())
    h.grant()
    h.backend.fg = fg("cs2.exe")
    h.backend.processes |= {"cs2.exe"}
    events = h.poll(0)
    assert [(type(e), e.category) for e in events] == [(ForegroundAppChanged, C.PRIVATE)]


# -- foreground ---------------------------------------------------------------------------


def test_foreground_change_is_reported_once(h):
    h.grant()
    h.backend.fg = fg("code.exe")
    assert len(of(h.poll(0), ForegroundAppChanged)) == 1
    assert h.poll() == []
    assert h.poll() == []
    h.backend.fg = fg("chrome.exe")
    (changed,) = of(h.poll(), ForegroundAppChanged)
    assert changed.app.display_name == "Google Chrome"
    assert (changed.category, changed.confidence) == (C.BROWSING, 0.6)
    h.backend.fg = None  # lock screen: keep what we had
    assert h.poll() == []
    h.backend.fg = fg("chrome.exe")
    assert h.poll() == []


def test_category_change_of_the_same_app_is_reported(h):
    h.grant(titles=True)
    h.backend.fg = fg("firefox.exe", title="Weather")
    h.poll(0)
    h.backend.fg = fg("firefox.exe", title="Cats - YouTube")
    (changed,) = of(h.poll(), ForegroundAppChanged)
    assert changed.category is C.WATCHING_VIDEO
    h.backend.fg = fg("firefox.exe", title="Dogs - YouTube")
    assert h.poll() == []


# -- idle -----------------------------------------------------------------------------------


def test_idle_and_return(h):
    h.grant()
    h.backend.fg = fg("code.exe")
    h.poll(0)
    h.backend.idle = 299
    assert h.poll() == []
    h.backend.idle = 300
    assert [(type(e), e.idle_seconds) for e in h.poll()] == [(UserIdle, 300)]
    for idle in (450, 1000):
        h.backend.idle = idle
        assert h.poll() == []
    h.backend.idle = 1.5
    assert [(type(e), e.away_seconds) for e in h.poll()] == [(UserReturned, 1000)]
    h.backend.idle = 2
    assert h.poll() == []


def test_return_detected_when_idle_drops(h):
    h.grant()
    h.backend.fg = fg("code.exe")
    h.poll(0)
    h.backend.idle = 300
    h.poll()
    h.backend.idle = 600
    h.poll()
    h.backend.idle = 8  # input 8 s ago: the user came back between two polls
    assert [type(e) for e in h.poll()] == [UserReturned]


def test_short_idle_is_not_an_absence(h):
    h.grant()
    h.backend.fg = fg("code.exe")
    h.poll(0)
    h.backend.idle = 120
    h.poll()
    h.backend.idle = 0
    assert h.poll() == []


@pytest.mark.parametrize("process", ["cs2.exe", "vlc.exe"])
def test_no_idle_while_gaming_or_watching(h, process):
    h.grant()
    h.backend.fg = fg(process)
    h.backend.processes |= {process}
    h.poll(0)
    h.backend.idle = 1800  # gamepad / film: no keyboard or mouse input
    assert of(h.poll(), UserIdle) == []
    h.backend.fg = fg("code.exe")
    assert of(h.poll(), UserIdle) != []


def test_idle_threshold_is_read_every_cycle(h):
    h.grant()
    h.backend.fg = fg("code.exe")
    h.poll(0)
    h.backend.idle = 61
    assert h.poll() == []
    h.settings.idle_threshold_minutes = 1
    assert [type(e) for e in h.poll()] == [UserIdle]


def test_unknown_idle_is_ignored(h):
    h.grant()
    h.backend.idle = None
    h.backend.fg = fg("code.exe")
    h.poll(0)
    assert h.poll() == []


# -- milestones -----------------------------------------------------------------------------


def test_milestones_once_per_streak(h):
    h.grant()
    h.backend.fg = fg("code.exe")
    h.poll(0)
    milestones = of(h.run_for(4 * 3600), ActivityMilestone)
    assert [m.seconds for m in milestones] == [3600, 7200, 10800]
    assert all(m.category is C.PROGRAMMING and m.app == AppInfo("code.exe", "VS Code") for m in milestones)


def test_same_category_in_another_app_continues_the_streak(h):
    h.grant()
    h.backend.fg = fg("code.exe")
    h.poll(0)
    h.run_for(1800)
    h.backend.fg = fg("pycharm64.exe")
    (milestone,) = of(h.run_for(1800), ActivityMilestone)
    assert milestone.app.display_name == "PyCharm"


def test_an_idle_period_ends_the_streak(h):
    h.grant()
    h.backend.fg = fg("code.exe")
    h.poll(0)
    h.run_for(3000)
    h.backend.idle = 300
    h.poll(60)
    h.backend.idle = 900
    h.poll(600)
    h.backend.idle = 0
    h.poll(60)
    assert of(h.run_for(3000), ActivityMilestone) == []  # 50 + 50 minutes, but not continuous
    assert len(of(h.run_for(600), ActivityMilestone)) == 1


def test_a_category_change_ends_the_streak(h):
    h.grant()
    h.backend.fg = fg("code.exe")
    h.poll(0)
    h.run_for(3000)
    h.backend.fg = fg("chrome.exe")
    h.run_for(300)
    h.backend.fg = fg("code.exe")  # first seen again one poll (60 s) later: the new streak starts there
    assert of(h.run_for(3000), ActivityMilestone) == []
    assert of(h.run_for(600), ActivityMilestone) == []
    assert len(of(h.poll(60), ActivityMilestone)) == 1


def test_a_brief_switch_keeps_the_streak(h):
    h.grant()
    h.backend.fg = fg("code.exe")
    h.poll(0)
    h.run_for(3540)
    h.backend.fg = fg("telegram.exe")  # half a minute in a messenger
    assert of(h.poll(30), ActivityMilestone) == []
    h.backend.fg = fg("code.exe")
    (milestone,) = of(h.poll(30), ActivityMilestone)
    assert milestone.seconds == 3600


def test_a_freeze_ends_the_streak(h):
    # The PC slept for two hours with the editor in front: that is not two hours of coding.
    h.grant()
    h.backend.fg = fg("code.exe")
    h.poll(0)
    h.run_for(1800)
    assert of(h.poll(7300), ActivityMilestone) == []
    assert of(h.run_for(3540), ActivityMilestone) == []  # the new streak began at the resume
    assert [m.seconds for m in of(h.poll(60), ActivityMilestone)] == [3600]


def test_sleeping_with_a_game_open_is_not_play(h):
    """perf-2: 30 minutes of play, 8 hours of sleep with the game open, 10 more minutes."""
    h.grant()
    h.backend.fg = fg("cs2.exe")
    h.backend.processes |= {"cs2.exe"}
    h.poll(0)
    h.run_for(1800)
    h.backend.idle = 1.0  # a key press woke the PC
    assert of(h.poll(8 * 3600), ActivityMilestone) == []
    assert of(h.run_for(600), ActivityMilestone) == []
    h.backend.processes -= {"cs2.exe"}
    h.backend.fg = fg("explorer.exe")
    (ended,) = of(h.poll(15), GameEnded)
    # 1800 + 600 + 15 s observed, plus the normal 3 s poll wait before the freeze
    assert ended.duration_seconds == 1800 + 600 + 15 + 3
    assert not ended.forgotten


def test_a_freeze_does_not_count_for_running_apps(h):
    h.grant()
    h.backend.processes |= {"code.exe"}
    h.poll(0)
    h.poll(8 * 3600)
    h.backend.processes -= {"code.exe"}
    (closed,) = of(h.poll(15), AppClosed)
    assert closed.ran_seconds == 15 + 3


def test_a_wall_clock_jump_alone_ends_the_streak_but_keeps_durations(h):
    # Where the monotonic clock stops during sleep only the wall clock shows the gap;
    # durations are monotonic, so they already leave the sleep out.
    h.grant()
    h.backend.fg = fg("cs2.exe")
    h.backend.processes |= {"cs2.exe"}
    h.poll(0)
    h.run_for(3000)
    h.clock.set_now(h.clock.now() + 8 * 3600)
    assert of(h.run_for(600), ActivityMilestone) == []  # 3600 s of play, but not continuous
    h.backend.processes -= {"cs2.exe"}
    h.backend.fg = fg("explorer.exe")
    (ended,) = of(h.poll(15), GameEnded)
    assert ended.duration_seconds == 3000 + 600 + 15


@pytest.mark.parametrize("interval", [3.0, 30.0])
def test_regular_cycles_are_not_a_freeze(h, interval):
    # Up to max(60 s, 5 poll intervals) between cycles is just a slow cycle.
    h.settings.poll_interval_seconds = interval
    h.grant()
    h.backend.fg = fg("cs2.exe")
    h.backend.processes |= {"cs2.exe"}
    h.poll(0)
    step = max(60.0, 5 * interval)
    milestones = of(h.run_for(3600, step=step), ActivityMilestone)
    assert [m.seconds for m in milestones] == [3600]
    h.backend.processes -= {"cs2.exe"}
    h.backend.fg = fg("explorer.exe")
    (ended,) = of(h.poll(15), GameEnded)
    assert ended.duration_seconds == 3600 + 15


def test_no_milestones_for_unknown_apps(h):
    h.grant()
    h.backend.fg = fg("something.exe")
    h.poll(0)
    assert of(h.run_for(4 * 3600, step=600), ActivityMilestone) == []


# -- process scan ---------------------------------------------------------------------------


def test_scan_runs_on_its_own_interval(h):
    h.grant()
    h.poll(0)
    assert h.backend.calls.count("scan") == 1
    for _ in range(4):
        h.poll(3)
    assert h.backend.calls.count("scan") == 1
    h.poll(3)  # 15 s since the first scan
    assert h.backend.calls.count("scan") == 2


def test_launch_and_close_of_known_apps_only(h):
    h.grant()
    h.backend.processes |= {"code.exe"}
    assert h.poll(0) == []
    h.backend.processes |= {"vlc.exe", "random-tool.exe", "telegram.exe"}
    events = h.poll(15)
    assert [(type(e), e.app, e.category) for e in events] == [
        (AppLaunched, AppInfo("vlc.exe", "VLC"), C.WATCHING_VIDEO)
    ]
    h.backend.processes -= {"code.exe", "random-tool.exe", "telegram.exe"}
    events = h.poll(15)
    assert [(type(e), e.app.display_name, e.ran_seconds) for e in events] == [(AppClosed, "VS Code", 30)]


def test_one_app_with_several_processes_counts_once(h):
    h.grant()
    h.poll(0)
    h.backend.processes |= {"steam.exe", "steamwebhelper.exe"}
    assert [(type(e), e.app.display_name) for e in h.poll(15)] == [(AppLaunched, "Steam")]
    h.backend.processes -= {"steamwebhelper.exe"}
    assert h.poll(15) == []
    h.backend.processes -= {"steam.exe"}
    assert [type(e) for e in h.poll(15)] == [AppClosed]


def test_game_start_and_end_with_duration(h):
    h.grant()
    h.poll(0)
    h.backend.processes |= {"eldenring.exe"}
    events = h.poll(15)
    assert [type(e) for e in events] == [GameStarted, AppLaunched]
    assert events[0].game_id == "eldenring" and events[1].category is C.GAMING
    assert h.run_for(45 * 60) == []
    h.backend.processes -= {"eldenring.exe"}
    events = h.poll(15)
    ended = of(events, GameEnded)
    assert [(e.game_id, e.title, e.duration_seconds) for e in ended] == [
        ("eldenring", "Elden Ring", 45 * 60 + 15)
    ]
    assert len(of(events, AppClosed)) == 1


def test_game_with_several_processes_is_one_session(h):
    h.grant()
    h.poll(0)
    h.backend.processes |= {"stalker2.exe", "stalker2-win64-shipping.exe"}
    assert len(of(h.poll(15), GameStarted)) == 1
    h.backend.processes -= {"stalker2.exe"}
    assert of(h.poll(15), GameEnded) == []
    h.backend.processes -= {"stalker2-win64-shipping.exe"}
    assert [e.game_id for e in of(h.poll(15), GameEnded)] == ["stalker2"]


def test_heuristic_foreground_game_has_a_session(h):
    h.grant()
    h.backend.fg = fg("coolgame.exe", exe=STEAM_GAME)
    h.backend.processes |= {"coolgame.exe"}
    events = h.poll(0)
    assert [type(e) for e in events] == [ForegroundAppChanged, GameStarted]
    changed, started = events
    assert (changed.category, changed.confidence, changed.app.display_name) == (C.GAMING, 0.75, "Cool Game")
    assert (started.game_id, started.title) == ("exe:coolgame.exe", "Cool Game")
    h.backend.fg = fg("code.exe")
    assert of(h.poll(15), GameEnded) == []  # still running in the background
    h.backend.processes -= {"coolgame.exe"}
    (ended,) = of(h.poll(15), GameEnded)
    assert ended.game_id == "exe:coolgame.exe" and ended.duration_seconds == 30


def test_foreground_game_missed_by_the_scan_does_not_flap(h):
    h.grant()
    h.backend.fg = fg("coolgame.exe", exe=STEAM_GAME)  # the scan never lists it (name mismatch, say)
    events = h.poll(0)
    assert len(of(events, GameStarted)) == 1
    for _ in range(4):
        assert h.poll(15) == []
    h.backend.fg = fg("code.exe")
    assert [e.game_id for e in of(h.poll(15), GameEnded)] == ["exe:coolgame.exe"]


def test_title_detected_game_has_a_session(h):
    h.grant(titles=True)
    h.backend.fg = fg("javaw.exe", title="Minecraft* 1.21.1 - Singleplayer")
    h.backend.processes |= {"javaw.exe"}
    assert [e.game_id for e in of(h.poll(0), GameStarted)] == ["minecraft"]
    h.backend.processes -= {"javaw.exe"}
    h.backend.fg = fg("explorer.exe")
    assert [e.game_id for e in of(h.poll(15), GameEnded)] == ["minecraft"]


def test_an_empty_scan_is_not_everything_closing(h):
    h.grant()
    h.backend.processes |= {"code.exe", "cs2.exe"}
    h.poll(0)
    h.backend.processes = set()
    assert h.poll(15) == []
    h.backend.processes = {"explorer.exe", "code.exe", "cs2.exe"}
    assert h.poll(15) == []


# -- robustness ---------------------------------------------------------------------------


def test_errors_are_logged_rate_limited_without_details(h, caplog):
    h.grant()
    h.backend.fg = fg("code.exe")
    h.backend.fail = {"foreground"}
    with caplog.at_level(logging.WARNING, logger="chibi.activity.monitor"):
        for _ in range(5):
            assert h.poll(3) == []
        failures = [r for r in caplog.records if "cycle failed" in r.getMessage()]
        assert len(failures) == 1
        assert "RuntimeError" in failures[0].getMessage()
        assert "secret-app" not in caplog.text and "olena" not in caplog.text
        h.backend.fail = set()
        assert len(of(h.poll(3), ForegroundAppChanged)) == 1  # the next cycle works
        h.backend.fail = {"idle"}
        h.poll(60)
        failures = [r for r in caplog.records if "cycle failed" in r.getMessage()]
        assert len(failures) == 2
        assert "4 similar errors suppressed" in failures[1].getMessage()


def test_settings_errors_fall_back_to_defaults(db, clock):
    h = Harness(db, clock)

    def broken():
        raise RuntimeError("no settings")

    classifier = ActivityClassifier(GameDatabase.load())
    h.monitor = ActivityMonitor(h.backend, h.permissions, h.bus, classifier, PrivacyFilter(), broken, clock)
    h.grant()
    h.backend.fg = fg("code.exe")
    assert len(of(h.poll(0), ForegroundAppChanged)) == 1


# -- thread -----------------------------------------------------------------------------


def monitor_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "activity-monitor" and t.is_alive()]


def test_thread_starts_and_stops_promptly(h):
    h.settings.poll_interval_seconds = 30.0  # stop() must not wait for the next cycle
    h.grant()
    h.backend.fg = fg("code.exe")
    h.monitor.start()
    try:
        assert wait_for(lambda: h.bus.pending > 0)
        assert h.monitor.is_running()
        (thread,) = monitor_threads()
        assert thread.daemon
    finally:
        began = time.monotonic()
        h.monitor.stop(timeout=2.0)
        took = time.monotonic() - began
    assert took < 1.0
    assert not h.monitor.is_running()
    assert monitor_threads() == []
    h.bus.dispatch_pending()
    assert len(of(h.events, ForegroundAppChanged)) == 1


def test_start_after_stop_and_double_start(h):
    h.settings.poll_interval_seconds = 30.0
    h.grant()
    h.backend.fg = fg("code.exe")
    h.monitor.stop()  # stopping a monitor that never ran is fine
    h.monitor.start()
    h.monitor.start()  # no second thread
    try:
        assert wait_for(lambda: h.bus.pending > 0)
        assert len(monitor_threads()) == 1
    finally:
        h.monitor.stop()
    h.bus.dispatch_pending()
    h.monitor.start()  # a restart observes afresh
    try:
        assert wait_for(lambda: h.bus.pending > 0)
    finally:
        h.monitor.stop()
    h.bus.dispatch_pending()
    assert len(of(h.events, ForegroundAppChanged)) == 2
    assert monitor_threads() == []


def test_thread_survives_failing_cycles(h):
    h.settings.poll_interval_seconds = 0.05
    h.grant()
    h.backend.fail = {"foreground"}
    h.monitor.start()
    try:
        assert wait_for(lambda: h.backend.calls.count("foreground") >= 3)
        assert h.monitor.is_running()
    finally:
        h.monitor.stop()
    assert monitor_threads() == []


def test_thread_does_nothing_without_permission(h):
    h.settings.poll_interval_seconds = 0.05
    h.backend.fg = fg("code.exe")
    h.monitor.start()
    try:
        time.sleep(0.15)
        assert h.monitor.is_running()
    finally:
        h.monitor.stop()
    assert h.backend.calls == []
    assert h.bus.pending == 0


# -- reconfigure ----------------------------------------------------------------------------


def test_reconfigure_applies_new_exclusions_and_games(h):
    h.grant()
    h.backend.fg = fg("notepad.exe")
    (changed,) = h.poll(0)
    assert (changed.category, changed.app.display_name) == (C.WORKING, "Блокнот")
    h.monitor.reconfigure(ActivityClassifier(GameDatabase.load()), PrivacyFilter(["notepad.exe"]))
    (changed,) = h.poll()
    assert changed.category is C.PRIVATE
    games = GameDatabase.load([{"process": "tool.exe", "title": "Tool Quest"}])
    h.monitor.reconfigure(ActivityClassifier(games), PrivacyFilter())
    h.backend.fg = fg("tool.exe")
    h.backend.processes |= {"tool.exe"}
    events = h.poll(15)
    assert [e.category for e in of(events, ForegroundAppChanged)] == [C.GAMING]
    assert [e.game_id for e in of(events, GameStarted)] == ["custom:tool.exe"]


def test_app_that_becomes_private_is_forgotten_silently(h):
    h.grant()
    h.backend.processes |= {"code.exe"}
    h.poll(0)
    h.monitor.reconfigure(ActivityClassifier(GameDatabase.load()), PrivacyFilter(["code.exe"]))
    assert h.poll(15) == []  # no "VS Code closed"


def ends(events: list[Event]) -> list[tuple[str, str, float, bool]]:
    return [(e.game_id, e.title, e.duration_seconds, e.forgotten) for e in of(events, GameEnded)]


def test_game_that_becomes_private_while_running_is_forgotten(h):
    """privsec-1: excluding a running game must not end it by name (or with a duration)."""
    h.grant()
    h.backend.fg = fg("cs2.exe")
    h.backend.processes |= {"cs2.exe"}
    h.poll(0)
    h.run_for(40 * 60)
    h.monitor.reconfigure(ActivityClassifier(GameDatabase.load()), PrivacyFilter(["cs2.exe"]))
    events = h.poll()  # not even a scan is needed
    assert ends(events) == [("cs2", "", 0.0, True)]
    assert [e.category for e in of(events, ForegroundAppChanged)] == [C.PRIVATE]
    assert "Counter-Strike" not in repr(events)
    # It stays forgotten: no new start, and no "closed" when the scan sees it or when it exits.
    assert h.run_for(600) == []
    h.backend.processes -= {"cs2.exe"}
    h.backend.fg = fg("explorer.exe")
    assert [type(e) for e in h.poll(15)] == [ForegroundAppChanged]
    assert "Counter-Strike" not in repr(h.events[-1])


def test_game_excluded_just_as_it_closes_is_forgotten(h):
    h.grant()
    h.backend.processes |= {"eldenring.exe"}
    h.poll(0)
    h.run_for(600)
    h.monitor.reconfigure(ActivityClassifier(GameDatabase.load()), PrivacyFilter(["eldenring.exe"]))
    h.backend.processes -= {"eldenring.exe"}
    events = h.poll(15)
    assert ends(events) == [("eldenring", "", 0.0, True)]
    assert of(events, AppClosed) == []


def test_foreground_only_game_that_becomes_private_is_forgotten(h):
    h.grant()
    h.backend.fg = fg("coolgame.exe", exe=STEAM_GAME)  # the scan never lists it
    h.poll(0)
    h.monitor.reconfigure(ActivityClassifier(GameDatabase.load()), PrivacyFilter(["coolgame.exe"]))
    events = h.poll(15)
    assert ends(events) == [("exe:coolgame.exe", "", 0.0, True)]
    assert "Cool Game" not in repr(events)
    assert h.poll(15) == []


def test_excluding_one_process_of_a_game_keeps_its_session(h):
    h.grant()
    h.poll(0)
    h.backend.processes |= {"stalker2.exe", "stalker2-win64-shipping.exe"}
    h.poll(15)
    h.monitor.reconfigure(ActivityClassifier(GameDatabase.load()), PrivacyFilter(["stalker2.exe"]))
    assert of(h.poll(15), GameEnded) == []
    # The game stays visible through its other process; once that exits, it ends as usual.
    h.backend.processes -= {"stalker2-win64-shipping.exe"}
    assert ends(h.poll(15)) == [("stalker2", "S.T.A.L.K.E.R. 2: Серце Чорнобиля", 30.0, False)]


def test_a_game_made_visible_again_starts_a_new_session(h):
    h.grant()
    h.backend.processes |= {"cs2.exe"}
    h.poll(0)
    h.monitor.reconfigure(ActivityClassifier(GameDatabase.load()), PrivacyFilter(["cs2.exe"]))
    assert ends(h.poll(15)) == [("cs2", "", 0.0, True)]
    h.monitor.reconfigure(ActivityClassifier(GameDatabase.load()), PrivacyFilter())
    events = h.poll(15)
    assert [e.game_id for e in of(events, GameStarted)] == ["cs2"]


def test_reconfigure_while_running(h):
    h.settings.poll_interval_seconds = 0.05
    h.grant()
    h.backend.fg = fg("code.exe")
    h.backend.processes |= {"cs2.exe"}
    classifiers = [ActivityClassifier(GameDatabase.load()) for _ in range(2)]
    filters = [PrivacyFilter(), PrivacyFilter(["code.exe"])]
    h.monitor.start()
    try:
        for i in range(50):
            h.monitor.reconfigure(classifiers[i % 2], filters[i % 2])
            time.sleep(0.002)
        assert wait_for(lambda: h.backend.calls.count("foreground") >= 3)
        assert h.monitor.is_running()
    finally:
        h.monitor.stop()
    h.bus.dispatch_pending()
    assert of(h.events, ForegroundAppChanged)
