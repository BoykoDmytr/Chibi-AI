"""The living tamagotchi: owns her current :class:`~chibi.tamagotchi.needs.Needs`
and decides when time passes for them.

* :meth:`TamagotchiService.load` restores the saved needs and lets her "sleep
  through" the time the app was closed (``NeedsModel.offline_catch_up``), so
  absence is never punished.
* :meth:`TamagotchiService.tick` (the app calls it every ~30 s) advances the
  needs by the monotonic time since the previous tick.  A gap of more than
  10 minutes means the process was frozen - laptop sleep or hibernation - and
  is treated like the app being closed (offline catch-up) instead of hours of
  awake decay.
* Interactions arrive as :class:`~chibi.core.events.CharacterInteraction` (or
  directly via :meth:`TamagotchiService.interact` from the chat and menus) and
  are answered with :class:`~chibi.core.events.InteractionOutcome`, which the
  reaction engine turns into speech.
* :class:`~chibi.core.events.NeedsChanged` is published only when something
  moved by at least one point, so the bus is not flooded every tick.

Persistence uses :class:`~chibi.memory.repositories.StateRepository` keys
``needs`` (the snapshot), ``last_seen`` (epoch seconds of the last save, so a
crash loses at most a minute of catch-up accuracy) and ``friendship_bonus_day``
(local date of the last daily-start XP bonus, so restarting the app does not
farm XP).

Main (Qt) thread only: the repository's SQLite connection and the bus
handlers live there, so no locking is needed.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import Any

from chibi.core.clock import Clock
from chibi.core.event_bus import EventBus
from chibi.core.events import (
    AppStarted,
    CharacterInteraction,
    ForegroundAppChanged,
    InteractionKind,
    InteractionOutcome,
    NeedsChanged,
)
from chibi.memory.repositories import StateRepository
from chibi.tamagotchi.needs import STAT_NAMES, Needs, NeedsModel, finite_number

log = logging.getLogger(__name__)

KEY_NEEDS = "needs"
KEY_LAST_SEEN = "last_seen"
KEY_BONUS_DAY = "friendship_bonus_day"

#: Periodic saves keep ``last_seen`` fresh without writing to disk every tick.
SAVE_INTERVAL_SECONDS = 60.0
#: Ticks come every ~30 s; a longer gap than this means the process was frozen.
CLOCK_JUMP_SECONDS = 600.0
#: An interaction or app switch this recent counts as "the user is around".
USER_ACTIVE_SECONDS = 600.0
PET_WINDOW_SECONDS = 600.0
CLICK_WINDOW_SECONDS = 5.0
#: Only the counts up to the "too much" thresholds matter; bounding the
#: windows keeps a click-spamming user from growing them without limit.
_MAX_TRACKED = 32
#: NeedsChanged is published once a need moved at least this far.
PUBLISH_THRESHOLD = 1.0


def _count_recent(times: deque[float], now: float, window: float) -> int:
    """Drop timestamps older than ``window`` and count the rest."""
    while times and now - times[0] >= window:
        times.popleft()
    return len(times)


def _local_day(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds).date().isoformat()


class TamagotchiService:
    def __init__(self, bus: EventBus, repo: StateRepository, clock: Clock,
                 model: NeedsModel | None = None) -> None:
        self._bus = bus
        self._repo = repo
        self._clock = clock
        self._model = model if model is not None else NeedsModel()
        self.needs = Needs()
        self._sleeping = False
        self._last_mono = clock.monotonic()
        self._last_wall = clock.now()
        self._last_save_mono = self._last_mono
        self._last_published = dataclasses.replace(self.needs)
        self._last_activity: float | None = None
        self._pets: deque[float] = deque(maxlen=_MAX_TRACKED)
        self._clicks: deque[float] = deque(maxlen=_MAX_TRACKED)
        self._loaded = False
        self._bonus_pending = False
        self._unsubscribers: list[Callable[[], None]] = [
            bus.subscribe(CharacterInteraction, self._on_interaction),
            bus.subscribe(AppStarted, self._on_app_started),
            bus.subscribe(ForegroundAppChanged, self._on_foreground_changed),
        ]

    # -- state ----------------------------------------------------------------------------

    @property
    def model(self) -> NeedsModel:
        return self._model

    @property
    def sleeping(self) -> bool:
        return self._sleeping

    @sleeping.setter
    def sleeping(self, value: bool) -> None:
        self.set_sleeping(value)

    @property
    def mood(self) -> str:
        """Ukrainian mood word for the current needs (``"Весела"``)."""
        return self._model.mood(self.needs)

    @property
    def user_active(self) -> bool:
        """The user interacted with her or switched apps in the last 10 minutes."""
        if self._last_activity is None:
            return False
        return self._clock.monotonic() - self._last_activity < USER_ACTIVE_SECONDS

    def set_sleeping(self, sleeping: bool) -> None:
        """Sync with what the character is doing (the controller calls this when
        her posture becomes / stops being SLEEP).

        Time since the last tick is settled first, so it counts under the
        state she was actually in.
        """
        sleeping = bool(sleeping)
        if sleeping == self._sleeping:
            return
        self.tick()
        self._sleeping = sleeping
        log.debug("tamagotchi %s", "asleep" if sleeping else "awake")

    # -- persistence ----------------------------------------------------------------------

    def load(self) -> None:
        """Restore saved needs and catch up on the time the app was closed."""
        needs = Needs.from_dict(self._repo.get(KEY_NEEDS))
        now = self._clock.now()
        last_seen = finite_number(self._repo.get(KEY_LAST_SEEN))
        if last_seen is not None:
            away = now - last_seen
            if away > 0:
                needs = self._model.offline_catch_up(needs, away)
                log.info("tamagotchi caught up on %.0f s away", away)
            elif away < 0:
                # The wall clock went backwards (changed by hand / NTP): nothing to catch up.
                log.info("last save is %.0f s in the future; no catch-up", -away)
        self.needs = needs
        self._last_mono = self._clock.monotonic()
        self._last_wall = now
        self._last_save_mono = self._last_mono
        self._loaded = True
        self._publish_needs(force=True)
        if self._bonus_pending:
            self._bonus_pending = False
            self._grant_daily_bonus()

    def save(self) -> None:
        """Persist the needs and the time they were current.  Never raises:
        it runs from a timer and on shutdown, where a full disk must not crash
        the app (the next periodic save tries again)."""
        self._last_save_mono = self._clock.monotonic()
        self._store(KEY_NEEDS, self.needs.to_dict())
        self._store(KEY_LAST_SEEN, self._clock.now())

    def _store(self, key: str, value: Any) -> bool:
        try:
            self._repo.set(key, value)
        except Exception:
            log.exception("could not save tamagotchi state %r", key)
            return False
        return True

    def close(self) -> None:
        """Stop listening to the bus (the app saves separately on shutdown)."""
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()

    # -- time -----------------------------------------------------------------------------

    def tick(self) -> None:
        """Advance the needs to now.  Called every ~30 s by the app."""
        jumped = self._advance()
        self._publish_needs()
        if jumped or self._clock.monotonic() - self._last_save_mono >= SAVE_INTERVAL_SECONDS:
            self.save()

    def _advance(self) -> bool:
        """Apply the time since the previous advance.  Returns True if it was a
        freeze (handled as offline catch-up), which deserves an immediate save."""
        mono = self._clock.monotonic()
        wall = self._clock.now()
        dt = mono - self._last_mono
        wall_dt = wall - self._last_wall
        self._last_mono = mono
        self._last_wall = wall
        # Either clock may reveal a freeze: on Linux / macOS the monotonic clock
        # stops during suspend while the wall clock keeps going; on Windows both
        # may advance.  Either way she slept through it - never hours of decay.
        gap = max(dt, wall_dt)
        if gap > CLOCK_JUMP_SECONDS:
            log.info("%.0f s passed since the last tick (system sleep?); catching up", gap)
            self.needs = self._model.offline_catch_up(self.needs, gap)
            return True
        if dt > 0:
            self.needs = self._model.tick(self.needs, dt, sleeping=self._sleeping,
                                          user_active=self.user_active)
        return False

    # -- interactions -----------------------------------------------------------------------

    def interact(self, kind: InteractionKind | str, *, announce: bool = True) -> str:
        """Apply an interaction and return its outcome (see :class:`InteractionOutcome`).

        ``announce=False`` is for the chat, which composes its own reply: no
        ``InteractionOutcome`` is published, so she doesn't say two lines.
        ``NeedsChanged`` is published either way so the UI stays current.
        """
        kind = InteractionKind(kind)
        # Settle elapsed time first: rules like "full" / "not tired" must see
        # the needs as they are now, under the activity state before this one.
        # (Published together with the interaction's effect below, not separately.)
        if self._advance():
            self.save()
        now = self._clock.monotonic()
        self._last_activity = now
        recent_pets = recent_clicks = 0
        # Every attempt counts, including refused ones: someone who keeps
        # spam-clicking keeps annoying her until they actually pause.
        if kind is InteractionKind.PET:
            recent_pets = _count_recent(self._pets, now, PET_WINDOW_SECONDS)
            self._pets.append(now)
        elif kind is InteractionKind.CLICK:
            recent_clicks = _count_recent(self._clicks, now, CLICK_WINDOW_SECONDS)
            self._clicks.append(now)
        self.needs, outcome = self._model.interact(
            self.needs, kind, recent_pets=recent_pets, recent_clicks=recent_clicks
        )
        if kind is InteractionKind.REST and outcome == "sleep":
            self._sleeping = True
        elif kind is InteractionKind.WAKE:
            self._sleeping = False
        log.debug("interaction %s -> %s", kind.value, outcome)
        if announce:
            self._bus.publish(InteractionOutcome(kind=kind, outcome=outcome))
        self._publish_needs(force=True)
        return outcome

    # -- bus handlers ---------------------------------------------------------------------

    def _on_interaction(self, event: CharacterInteraction) -> None:
        self.interact(event.kind)

    def _on_foreground_changed(self, _event: ForegroundAppChanged) -> None:
        self._last_activity = self._clock.monotonic()

    def _on_app_started(self, _event: AppStarted) -> None:
        if not self._loaded:
            # The bonus would be overwritten by load(); apply it right after.
            self._bonus_pending = True
            return
        self._grant_daily_bonus()

    def _grant_daily_bonus(self) -> None:
        """First start of the app on a calendar day: a little friendship XP."""
        today = _local_day(self._clock.now())
        if self._repo.get(KEY_BONUS_DAY) == today:
            return
        self.needs = self._model.daily_bonus(self.needs)
        log.info("first start today: friendship bonus")
        # Saved at once, so a crash before the next periodic save can neither
        # lose today's bonus nor grant it twice.
        self._store(KEY_BONUS_DAY, today)
        self.save()
        self._publish_needs()

    # -- publishing -------------------------------------------------------------------------

    def _publish_needs(self, *, force: bool = False) -> None:
        needs = self.needs
        if not force and not self._moved(self._last_published, needs):
            return
        self._bus.publish(
            NeedsChanged(
                hunger=needs.hunger,
                energy=needs.energy,
                happiness=needs.happiness,
                boredom=needs.boredom,
                affection=needs.affection,
                friendship_level=needs.friendship_level,
            )
        )
        self._last_published = dataclasses.replace(needs)

    @staticmethod
    def _moved(before: Needs, after: Needs) -> bool:
        if before.friendship_level != after.friendship_level:
            return True
        return any(abs(getattr(after, name) - getattr(before, name)) >= PUBLISH_THRESHOLD
                   for name in STAT_NAMES)
