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
* Start/end times come from this tracker's own (wall) clock, so a session is
  measured the same way whether it ends normally or by revocation.  One
  correction: the monitor's ``GameEnded.duration_seconds`` leaves out time it
  could not observe (the PC slept with the game open), and whatever part of
  the wall-clock span it says was not played is cut from the start - eight
  hours of sleep are not eight hours of play.
* A game the user made private while it ran ends with
  ``GameEnded(forgotten=True)``: the session is dropped, nothing is stored.
* After a history wipe, :meth:`GameSessionTracker.restart_open` keeps games
  that still run as sessions starting at the wipe, so their remaining part is
  recorded while nothing from before the wipe comes back.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
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
    started_at: float  # epoch seconds: start of what will be stored
    #: Epoch seconds since which the monitor measures this game, i.e. where its
    #: ``GameEnded.duration_seconds`` counts from.  Equal to ``started_at``
    #: unless the monitor restarted (later) or the history was wiped (earlier).
    measured_from: float | None = None


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

    def restart_open(self, now: float | None = None) -> None:
        """After a history wipe: keep following the games that still run, from *now* on.

        Nothing is stored.  Each open session now starts at *now* (default:
        this tracker's clock), so the part played before the wipe never comes
        back, while the part after it is recorded when the game ends (use this
        instead of :meth:`flush`, whose rows the wipe would only delete).
        """
        at = self._clock.now() if now is None else now
        self._open = {
            game_id: replace(
                session,
                started_at=max(session.started_at, at),
                measured_from=_measured_from(session),  # the monitor still counts from there
            )
            for game_id, session in self._open.items()
        }

    def discard(self, game_id: str) -> bool:
        """Forget an open session without storing it (the user just made the game
        private).  Done synchronously by the app when exclusions change, so a
        revoke/kill-switch flush in the same instant can't persist it before the
        monitor's ``GameEnded(forgotten=True)`` arrives.  Returns whether one was open."""
        return self._open.pop(game_id, None) is not None

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
        now = self._clock.now()
        session = self._open.get(event.game_id)
        if session is not None:
            # E.g. the monitor restarted: keep the original start, but its
            # duration for this game now counts from here.
            self._open[event.game_id] = replace(session, measured_from=now)
            return
        self._open[event.game_id] = OpenSession(event.game_id, event.title, now, now)

    def _on_ended(self, event: GameEnded) -> None:
        session = self._open.pop(event.game_id, None)
        if session is None:
            # Never saw it start (or already saved it when observation stopped).
            return
        if event.forgotten:
            return  # the user made the game private while it ran: keep no trace of it
        # No permission check here: the session began while observation was
        # allowed, and its end may be dispatched just after a revocation.
        ended_at = self._clock.now()
        started_at = session.started_at
        duration = event.duration_seconds
        if isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= 0:
            # The part of the monitor's span it did not count as played (the PC
            # slept) is cut from the start.  If the sleep came before a history
            # wipe this undercounts; that is the safe side - never wiped time.
            not_played = (ended_at - _measured_from(session)) - duration
            if not_played > 0:
                started_at = min(ended_at, started_at + not_played)
        self._store(session.game_id, session.title, started_at, ended_at)

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


def _measured_from(session: OpenSession) -> float:
    return session.started_at if session.measured_from is None else session.measured_from
