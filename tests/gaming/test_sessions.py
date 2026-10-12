import pytest

from chibi.core.events import (
    AppShuttingDown,
    Event,
    GameEnded,
    GameStarted,
    ObservationStopped,
    Permission,
    PermissionChanged,
)
from chibi.gaming.sessions import GameSessionTracker
from chibi.memory.repositories import GameSessionRepository, PermissionRepository
from chibi.privacy.permissions import PermissionManager


@pytest.fixture
def repo(db):
    return GameSessionRepository(db)


@pytest.fixture
def seen(bus):
    events: list[Event] = []
    bus.subscribe(Event, events.append)
    return events


def started(game_id="cs2", title="Counter-Strike 2"):
    return GameStarted(game_id=game_id, title=title, process_name=f"{game_id}.exe")


def ended(game_id="cs2", title="Counter-Strike 2", duration=0.0):
    return GameEnded(game_id=game_id, title=title, duration_seconds=duration)


def rows(db):
    return db.conn.execute(
        "SELECT game_id, title, started_at, ended_at, duration FROM game_sessions ORDER BY id"
    ).fetchall()


def test_finished_session_is_stored(bus, repo, clock, db):
    GameSessionTracker(bus, repo, clock)
    start = clock.now()
    bus.publish(started())
    bus.dispatch_pending()
    clock.advance(1800)
    bus.publish(ended(duration=1800))
    bus.dispatch_pending()
    (row,) = rows(db)
    assert (row["game_id"], row["title"]) == ("cs2", "Counter-Strike 2")
    assert row["started_at"] == start
    assert row["ended_at"] == start + 1800
    assert row["duration"] == 1800
    assert repo.total_for("cs2") == (1800.0, 1)


def test_short_sessions_are_not_stored(bus, repo, clock, db):
    GameSessionTracker(bus, repo, clock)
    bus.publish(started())
    bus.dispatch_pending()
    clock.advance(59)
    bus.publish(ended(duration=59))
    bus.dispatch_pending()
    assert rows(db) == []


def test_exactly_one_minute_is_stored(bus, repo, clock, db):
    GameSessionTracker(bus, repo, clock)
    bus.publish(started())
    bus.dispatch_pending()
    clock.advance(60)
    bus.publish(ended(duration=60))
    bus.dispatch_pending()
    assert len(rows(db)) == 1


def test_duplicate_start_keeps_the_first(bus, repo, clock, db):
    tracker = GameSessionTracker(bus, repo, clock)
    first = clock.now()
    bus.publish(started())
    bus.dispatch_pending()
    clock.advance(300)
    bus.publish(started())  # e.g. the monitor restarted and saw the game again
    bus.dispatch_pending()
    assert [s.started_at for s in tracker.open_sessions()] == [first]
    clock.advance(300)
    bus.publish(ended(duration=300))
    bus.dispatch_pending()
    assert rows(db)[0]["duration"] == 600


def test_end_without_start_is_ignored(bus, repo, clock, db):
    GameSessionTracker(bus, repo, clock)
    bus.publish(ended(duration=5000))
    bus.dispatch_pending()
    assert rows(db) == []


def test_parallel_games_are_tracked_separately(bus, repo, clock, db):
    GameSessionTracker(bus, repo, clock)
    bus.publish(started("cs2"))
    bus.dispatch_pending()
    clock.advance(120)
    bus.publish(started("dota2", "Dota 2"))
    bus.dispatch_pending()
    clock.advance(120)
    bus.publish(ended("cs2", duration=240))
    bus.dispatch_pending()
    clock.advance(120)
    bus.publish(ended("dota2", "Dota 2", duration=240))
    bus.dispatch_pending()
    assert [(r["game_id"], r["duration"]) for r in rows(db)] == [("cs2", 240), ("dota2", 240)]


@pytest.mark.parametrize(
    "stop_event",
    [
        PermissionChanged(permission=Permission.APP_METADATA, granted=False),
        ObservationStopped(),
        AppShuttingDown(),
    ],
)
def test_stopping_observation_saves_open_sessions_silently(bus, repo, clock, db, seen, stop_event):
    tracker = GameSessionTracker(bus, repo, clock)
    bus.publish(started("cs2"))
    bus.publish(started("dota2", "Dota 2"))
    bus.dispatch_pending()
    clock.advance(30)
    bus.publish(started("osu", "osu!"))
    bus.dispatch_pending()
    clock.advance(600)
    seen.clear()
    bus.publish(stop_event)
    bus.dispatch_pending()
    stored = {(r["game_id"], r["duration"]) for r in rows(db)}
    assert stored == {("cs2", 630), ("dota2", 630), ("osu", 600)}
    assert tracker.open_sessions() == []
    assert seen == [stop_event]  # the tracker itself publishes nothing
    assert bus.pending == 0
    # a stale GameEnded queued before the revocation does not store the session twice
    bus.publish(ended("cs2", duration=630))
    bus.dispatch_pending()
    assert len(rows(db)) == 3


def test_other_permission_changes_do_nothing(bus, repo, clock, db):
    tracker = GameSessionTracker(bus, repo, clock)
    bus.publish(started())
    bus.dispatch_pending()
    clock.advance(600)
    bus.publish(PermissionChanged(permission=Permission.WINDOW_TITLES, granted=False))
    bus.publish(PermissionChanged(permission=Permission.APP_METADATA, granted=True))
    bus.dispatch_pending()
    assert rows(db) == []
    assert len(tracker.open_sessions()) == 1


def test_flush_returns_count_and_skips_short_sessions(bus, repo, clock, db):
    tracker = GameSessionTracker(bus, repo, clock)
    bus.publish(started("cs2"))
    bus.dispatch_pending()
    clock.advance(100)
    bus.publish(started("dota2", "Dota 2"))
    bus.dispatch_pending()
    clock.advance(10)
    assert tracker.flush() == 1
    assert [r["game_id"] for r in rows(db)] == ["cs2"]
    assert tracker.open_sessions() == []


def test_start_while_not_permitted_is_ignored(bus, repo, clock, db):
    permissions = PermissionManager(PermissionRepository(db), None, clock)
    tracker = GameSessionTracker(bus, repo, clock, permissions=permissions)
    bus.publish(started())  # queued just before a revocation, dispatched after it
    bus.dispatch_pending()
    assert tracker.open_sessions() == []
    permissions.grant(Permission.APP_METADATA)
    bus.publish(started())
    bus.dispatch_pending()
    assert len(tracker.open_sessions()) == 1
    clock.advance(120)
    permissions.revoke(Permission.APP_METADATA)
    # the end was published while still permitted but dispatched after the revocation: still stored
    bus.publish(ended(duration=120))
    bus.dispatch_pending()
    assert len(rows(db)) == 1


def test_storage_errors_do_not_escape(bus, clock):
    class BrokenRepo:
        def add(self, *args):
            raise RuntimeError("disk full")

    tracker = GameSessionTracker(bus, BrokenRepo(), clock)  # type: ignore[arg-type]
    bus.publish(started())
    bus.dispatch_pending()
    clock.advance(600)
    assert tracker.flush() == 0


def test_close_unsubscribes(bus, repo, clock):
    tracker = GameSessionTracker(bus, repo, clock)
    tracker.close()
    bus.publish(started())
    bus.dispatch_pending()
    assert tracker.open_sessions() == []


# -- privsec-1: a game made private while it ran -------------------------------------------


def test_forgotten_game_is_dropped_without_a_trace(bus, repo, clock, db, seen):
    tracker = GameSessionTracker(bus, repo, clock)
    bus.publish(started())
    bus.publish(started("dota2", "Dota 2"))
    bus.dispatch_pending()
    clock.advance(2400)
    bus.publish(GameEnded(game_id="cs2", title="", duration_seconds=0.0, forgotten=True))
    bus.dispatch_pending()
    assert rows(db) == []
    assert [s.game_id for s in tracker.open_sessions()] == ["dota2"]
    # neither a flush (revocation, shutdown) nor a stale normal end brings it back
    assert tracker.flush() == 1
    bus.publish(ended(duration=2400))
    bus.dispatch_pending()
    assert [r["game_id"] for r in rows(db)] == ["dota2"]
    assert not [e for e in seen if not isinstance(e, (GameStarted, GameEnded))]  # publishes nothing


def test_game_excluded_while_running_is_not_stored_end_to_end(bus, repo, clock, db):
    """privsec-1 through the real monitor: exclusion -> forgotten end -> no row, now or later."""
    from chibi.activity.classifier import ActivityClassifier
    from chibi.activity.monitor import ActivityMonitor
    from chibi.core.config import ActivitySettings
    from chibi.gaming.games_db import GameDatabase
    from chibi.privacy.filters import PrivacyFilter
    from tests.activity.test_monitor import FakeBackend, fg

    permissions = PermissionManager(PermissionRepository(db), None, clock)
    permissions.grant(Permission.APP_METADATA)
    tracker = GameSessionTracker(bus, repo, clock, permissions)
    backend = FakeBackend()
    games = GameDatabase.load()
    monitor = ActivityMonitor(backend, permissions, bus, ActivityClassifier(games), PrivacyFilter(),
                              lambda: ActivitySettings(), clock)
    backend.fg = fg("cs2.exe")
    backend.processes |= {"cs2.exe"}
    monitor.poll_once()
    bus.dispatch_pending()
    for _ in range(40):
        clock.advance(60)
        monitor.poll_once()
        bus.dispatch_pending()
    assert [s.game_id for s in tracker.open_sessions()] == ["cs2"]
    monitor.reconfigure(ActivityClassifier(games), PrivacyFilter(["cs2.exe"]))
    clock.advance(3)
    monitor.poll_once()
    bus.dispatch_pending()
    assert tracker.open_sessions() == []
    tracker.flush()
    assert rows(db) == []


# -- perf-2: time the monitor did not count as played -------------------------------------


def test_sleep_reported_by_the_monitor_is_cut_from_the_start(bus, repo, clock, db):
    GameSessionTracker(bus, repo, clock)
    start = clock.now()
    bus.publish(started())
    bus.dispatch_pending()
    clock.advance(1800 + 8 * 3600 + 600)  # 30 min of play, 8 h asleep, 10 more minutes
    bus.publish(ended(duration=2400))
    bus.dispatch_pending()
    (row,) = rows(db)
    assert row["duration"] == 2400
    assert row["ended_at"] == clock.now()
    assert row["started_at"] == start + 8 * 3600


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -5.0])
def test_a_nonsense_duration_falls_back_to_the_wall_clock(bus, repo, clock, db, duration):
    GameSessionTracker(bus, repo, clock)
    bus.publish(started())
    bus.dispatch_pending()
    clock.advance(600)
    bus.publish(ended(duration=duration))
    bus.dispatch_pending()
    assert [r["duration"] for r in rows(db)] == [600]


def test_a_longer_duration_never_moves_the_start_back(bus, repo, clock, db):
    # e.g. the wall clock was set back during the game: the tracker's own start is a lower bound
    GameSessionTracker(bus, repo, clock)
    start = clock.now()
    bus.publish(started())
    bus.dispatch_pending()
    clock.advance(600)
    bus.publish(ended(duration=5000))
    bus.dispatch_pending()
    assert [(r["started_at"], r["duration"]) for r in rows(db)] == [(start, 600)]


def test_after_a_monitor_restart_the_duration_counts_from_the_new_start(bus, repo, clock, db):
    GameSessionTracker(bus, repo, clock)
    first = clock.now()
    bus.publish(started())
    bus.dispatch_pending()
    clock.advance(600)
    bus.publish(started())  # the monitor restarted: its duration counts from here
    bus.dispatch_pending()
    clock.advance(1800 + 8 * 3600)  # including a sleep it noticed
    bus.publish(ended(duration=1800))
    bus.dispatch_pending()
    assert [(r["started_at"], r["duration"]) for r in rows(db)] == [(first + 8 * 3600, 2400)]


def test_sleeping_with_a_game_open_end_to_end(bus, repo, clock, db):
    """perf-2 through the real monitor: the stored session leaves the sleep out."""
    from chibi.activity.classifier import ActivityClassifier
    from chibi.activity.monitor import ActivityMonitor
    from chibi.core.config import ActivitySettings
    from chibi.gaming.games_db import GameDatabase
    from chibi.privacy.filters import PrivacyFilter
    from tests.activity.test_monitor import FakeBackend, fg

    GameSessionTracker(bus, repo, clock)
    backend = FakeBackend()

    class Granted:
        def is_granted(self, permission):
            return True

    monitor = ActivityMonitor(backend, Granted(), bus, ActivityClassifier(GameDatabase.load()),
                              PrivacyFilter(), lambda: ActivitySettings(), clock)

    def play(seconds, step=3.0):
        for _ in range(round(seconds / step)):
            clock.advance(step)
            monitor.poll_once()
            bus.dispatch_pending()

    backend.fg = fg("cs2.exe")
    backend.processes |= {"cs2.exe"}
    monitor.poll_once()
    bus.dispatch_pending()
    play(1800)
    clock.advance(8 * 3600)  # asleep: both clocks move (Windows' monotonic clock includes sleep)
    play(600)
    backend.processes -= {"cs2.exe"}
    backend.fg = fg("explorer.exe")
    play(15)
    (row,) = rows(db)
    assert row["duration"] == 1800 + 600 + 3  # the first scan after the close came 3 s later


# -- restart_open (history wipe) ------------------------------------------------------------


def test_restart_open_keeps_running_games_from_now_on(bus, repo, clock, db):
    tracker = GameSessionTracker(bus, repo, clock)
    bus.publish(started())
    bus.dispatch_pending()
    clock.advance(1200)
    wiped_at = clock.now()
    tracker.restart_open()
    assert rows(db) == []  # nothing is stored
    assert [(s.game_id, s.started_at) for s in tracker.open_sessions()] == [("cs2", wiped_at)]
    clock.advance(900)
    bus.publish(ended(duration=2100))  # the monitor still counts from the real start
    bus.dispatch_pending()
    assert [(r["started_at"], r["duration"]) for r in rows(db)] == [(wiped_at, 900)]


def test_restart_open_with_an_explicit_time_and_flush(bus, repo, clock, db):
    tracker = GameSessionTracker(bus, repo, clock)
    bus.publish(started())
    bus.publish(started("dota2", "Dota 2"))
    bus.dispatch_pending()
    clock.advance(600)
    tracker.restart_open(now=clock.now() - 100)
    clock.advance(30)
    assert tracker.flush() == 2  # 130 s each, all after the wipe
    assert [r["duration"] for r in rows(db)] == [130, 130]


def test_restart_open_never_brings_back_time_from_before_the_wipe(bus, repo, clock, db):
    # A sleep after the wipe is still cut; the part before the wipe never counts.
    tracker = GameSessionTracker(bus, repo, clock)
    bus.publish(started())
    bus.dispatch_pending()
    clock.advance(1200)
    wiped_at = clock.now()
    tracker.restart_open()
    clock.advance(300 + 8 * 3600 + 300)
    bus.publish(ended(duration=1200 + 600))
    bus.dispatch_pending()
    assert [(r["started_at"], r["duration"]) for r in rows(db)] == [(wiped_at + 8 * 3600, 600)]


def test_restart_open_without_sessions_does_nothing(bus, repo, clock, db):
    tracker = GameSessionTracker(bus, repo, clock)
    tracker.restart_open()
    assert tracker.open_sessions() == []
    assert rows(db) == []
