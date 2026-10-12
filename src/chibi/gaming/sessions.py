"""Game sessions: ``GameStarted``/``GameEnded`` from the monitor -> rows in SQLite.

Runs on the main thread (it owns the SQLite connection).  Game sessions are
the only thing about other apps the companion ever writes to disk: game id,
title, start and end - see ARCHITECTURE.md.

* Sessions shorter than a minute are not stored (a launcher flash, a crash
  on start, a misdetected helper).
* When observation stops (``APP_METADATA`` revoked, the kill switch, app
  shutdown) open sessions are saved up to that moment and forgotten.  The
  monitor drops its own state silently at the same time, so a later
  ``GameEnded`` cannot arrive for them - and if a stale one does (it was
  already queued), it is ignored instead of stored twice.
* Start/end times come from this tracker's own clock, so a session is
  measured the same way whether it ends normally or by revocation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from chibi.core.clock import Clock
from chibi.core.event_bus import EventBus
from chibi.core.events import (
    AppShuttingDown,
    GameEnded,
    GameStarted,
    ObservationStopped,
    Permission,
    PermissionChanged,
)
from chibi.memory.repositories import GameSessionRepository

log = logging.getLogger(__name__)

MIN_SESSION_SECONDS = 60.0


class PermissionChecker(Protocol):
    def is_granted(self, permission: Permission) -> bool: ...


@dataclass(frozen=True)
class OpenSession:
    game_id: str
    title: str
    started_at: float  # epoch seconds


class GameSessionTracker:
    def __init__(
        self,
        bus: EventBus,
        repo: GameSessionRepository,
        clock: Clock,
        permissions: PermissionChecker | None = None,
    ) -> None:
        """*permissions* (optional) lets the tracker ignore game events that were
        queued just before ``APP_METADATA`` was revoked."""
        self._repo = repo
        self._clock = clock
        self._permissions = permissions
        self._open: dict[str, OpenSession] = {}
        self._unsubscribers: list[Callable[[], None]] = [
            bus.subscribe(GameStarted, self._on_started),
            bus.subscribe(GameEnded, self._on_ended),
            bus.subscribe(PermissionChanged, self._on_permission_changed),
            bus.subscribe(ObservationStopped, self._on_observation_stopped),
            bus.subscribe(AppShuttingDown, self._on_shutdown),
        ]

    def open_sessions(self) -> list[OpenSession]:
        return list(self._open.values())

    def flush(self) -> int:
        """Save every open session up to now and forget it.  Returns how many were stored."""
        now = self._clock.now()
        sessions, self._open = list(self._open.values()), {}
        return sum(self._store(s.game_id, s.title, s.started_at, now) for s in sessions)

    def close(self) -> None:
        """Stop listening to the bus (open sessions are kept; call flush() to save them)."""
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()

    # -- handlers -----------------------------------------------------------------

    def _observing(self) -> bool:
        if self._permissions is None:
            return True
        try:
            return bool(self._permissions.is_granted(Permission.APP_METADATA))
        except Exception:
            return False

    def _on_started(self, event: GameStarted) -> None:
        if not self._observing():
            return
        if event.game_id in self._open:
            return  # e.g. the monitor restarted: keep the original start
        self._open[event.game_id] = OpenSession(event.game_id, event.title, self._clock.now())

    def _on_ended(self, event: GameEnded) -> None:
        session = self._open.pop(event.game_id, None)
        if session is None:
            # Never saw it start (or already saved it when observation stopped).
            return
        # No permission check here: the session began while observation was
        # allowed, and its end may be dispatched just after a revocation.
        self._store(session.game_id, session.title, session.started_at, self._clock.now())

    def _on_permission_changed(self, event: PermissionChanged) -> None:
        if event.permission is Permission.APP_METADATA and not event.granted:
            self.flush()

    def _on_observation_stopped(self, _event: ObservationStopped) -> None:
        self.flush()

    def _on_shutdown(self, _event: AppShuttingDown) -> None:
        self.flush()

    def _store(self, game_id: str, title: str, started_at: float, ended_at: float) -> bool:
        if ended_at - started_at < MIN_SESSION_SECONDS:
            return False
        try:
            self._repo.add(game_id, title, started_at, ended_at)
        except Exception:
            log.exception("could not save a game session")
            return False
        return True
