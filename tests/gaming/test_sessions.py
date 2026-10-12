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
    bus.publish(ended("cs2"))
    bus.dispatch_pending()
    clock.advance(120)
    bus.publish(ended("dota2", "Dota 2"))
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
