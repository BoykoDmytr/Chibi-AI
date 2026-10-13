"""SpeechComposer - the one entry point for every line she says on her own or
in reaction to something.

Rules (the reaction engine, the decision engine) decide *that* she speaks and
which phrase keys fit; the composer decides *how*:

* **With AI** (a brain is wired, ``settings.ai.use_for_reactions`` is on and
  the brain reports the purpose available): the brain phrases a fresh line
  and a timeout is armed through the injected ``schedule`` callable
  (default ``QTimer.singleShot``).  Whichever comes first wins and publishes
  **exactly one** :class:`~chibi.core.events.SpeechRequested`: the AI line,
  or - on failure, refusal or timeout - the phrase-bank line from
  ``selector.say``.  An AI result that arrives after the fallback was
  published is dropped.
* **Without AI**: ``selector.say`` and publish synchronously - exactly the
  Phase 1 behaviour.

Guards:

* at most :data:`MAX_PENDING_AI` lines wait for the model at once; anything
  beyond that is said from the phrase bank right away (a burst of events
  must not queue up a burst of late lines);
* a line requested by the user (``solicited``) waits at most
  :data:`SOLICITED_TIMEOUT_SECONDS` - they are watching for her reaction;
* when quiet mode or do-not-disturb is switched on, or the app shuts down,
  every pending line is abandoned (neither the AI line nor its fallback is
  said): a line chosen before "помовч" must not be said after it.

The return value of :meth:`SpeechComposer.compose` tells the caller whether
a line is on its way, so budgets and cooldowns count it at once.  On the AI
path that is always True (the fallback may still find no phrase - logged).

Quiet mode's state-only gestures are not composed: the reaction engine
publishes them itself and never asks the AI.

Main (Qt) thread only.  No Qt import at module level: the default scheduler
imports ``QTimer`` lazily, so headless tests inject their own.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from chibi.ai.prompts import BANK_ONLY_PREFIXES, purpose_for
from chibi.ai.types import Purpose
from chibi.core.clock import Clock
from chibi.core.config import Settings
from chibi.core.event_bus import EventBus
from chibi.core.events import AppShuttingDown, DoNotDisturbChanged, QuietModeChanged, SpeechRequested

if TYPE_CHECKING:  # pragma: no cover - typing only
    from chibi.personality.phrases import PhraseSelector

log = logging.getLogger(__name__)

DEFAULT_AI_TIMEOUT_SECONDS = 8.0
#: The user clicked / petted / fed her and is watching: don't keep them waiting.
SOLICITED_TIMEOUT_SECONDS = 5.0
#: Lines waiting for the model at the same time.
MAX_PENDING_AI = 2

Schedule = Callable[[int, Callable[[], None]], None]


class PhraseBrain(Protocol):
    """What the composer needs from :class:`~chibi.ai.brain.AIBrain`."""

    def available(self, purpose: Purpose) -> bool: ...

    def phrase(self, keys: Sequence[str], variables: Mapping[str, str], solicited: bool,
               on_done: Callable[[SpeechRequested | None], None]) -> None: ...


def qt_schedule(msec: int, callback: Callable[[], None]) -> None:
    """Run *callback* once after *msec* on the Qt main thread's event loop."""
    from PySide6.QtCore import QTimer  # lazy: keeps the personality package Qt-free

    QTimer.singleShot(msec, callback)


@dataclass(eq=False)
class _Pending:
    id: int
    keys: tuple[str, ...]
    variables: dict[str, Any]
    solicited: bool
    done: bool = False
    #: Why it ended: "ai", "fallback", "abandoned".
    outcome: str = field(default="")


class SpeechComposer:
    def __init__(self, bus: EventBus, selector: PhraseSelector, brain: PhraseBrain | None,
                 settings: Callable[[], Settings], clock: Clock, *,
                 ai_timeout: float = DEFAULT_AI_TIMEOUT_SECONDS,
                 schedule: Schedule | None = None) -> None:
        """``schedule(msec, callback)`` arms the AI timeout (default: ``QTimer.singleShot``)."""
        self._bus = bus
        self._selector = selector
        self._brain = brain
        self._settings = settings
        self._clock = clock
        self._timeout = max(0.1, float(ai_timeout))
        self._schedule = schedule or qt_schedule
        self._pending: dict[int, _Pending] = {}
        self._ids = itertools.count(1)
        self._unsubscribers: list[Callable[[], None]] = [
            bus.subscribe(QuietModeChanged, self._on_quiet),
            bus.subscribe(DoNotDisturbChanged, self._on_quiet),
            bus.subscribe(AppShuttingDown, lambda _event: self.abandon_pending()),
        ]

    @property
    def brain(self) -> PhraseBrain | None:
        return self._brain

    def set_brain(self, brain: PhraseBrain | None) -> None:
        """Wire (or remove) the AI brain after construction."""
        self._brain = brain

    @property
    def pending(self) -> int:
        """Lines currently waiting for the model."""
        return len(self._pending)

    def close(self) -> None:
        """Unsubscribe from the bus and abandon pending lines (shutdown, tests)."""
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        self.abandon_pending()

    # -- composing --------------------------------------------------------------------------

    def compose(self, *keys: str, solicited: bool = False, **variables: Any) -> bool:
        """Say one line for the first usable key.  True if a line will be published
        (now from the phrase bank, or later from the AI with a phrase-bank fallback)."""
        keys = tuple(key for key in keys if key)
        if not keys:
            return False
        if self._use_ai(keys):
            return self._compose_ai(keys, solicited, variables)
        return self._say_offline(keys, solicited, variables)

    def abandon_pending(self) -> None:
        """Forget every line still waiting for the model: nothing is said for them."""
        if not self._pending:
            return
        log.debug("abandoning %d pending AI line(s)", len(self._pending))
        for pending in self._pending.values():
            pending.done = True
            pending.outcome = "abandoned"
        self._pending.clear()

    # -- internals --------------------------------------------------------------------------

    def _use_ai(self, keys: tuple[str, ...]) -> bool:
        brain = self._brain
        if brain is None or keys[0].startswith(BANK_ONLY_PREFIXES):
            return False
        try:
            if not self._settings().ai.use_for_reactions:
                return False
        except Exception:
            log.exception("settings unavailable; using the phrase bank")
            return False
        if len(self._pending) >= MAX_PENDING_AI:
            log.debug("%d AI lines pending: %s from the phrase bank", len(self._pending), keys[0])
            return False
        try:
            return bool(brain.available(purpose_for(keys)))
        except Exception:
            log.exception("brain availability check failed; using the phrase bank")
            return False

    def _compose_ai(self, keys: tuple[str, ...], solicited: bool, variables: dict[str, Any]) -> bool:
        assert self._brain is not None
        pending = _Pending(next(self._ids), keys, dict(variables), solicited)
        self._pending[pending.id] = pending
        timeout = min(self._timeout, SOLICITED_TIMEOUT_SECONDS) if solicited else self._timeout
        try:
            self._schedule(int(timeout * 1000), lambda: self._on_timeout(pending))
        except Exception:
            log.exception("could not arm the AI timeout; using the phrase bank")
            self._finish(pending, "fallback")
            return self._fallback(pending)
        texts = {name: str(value) for name, value in variables.items()
                 if value is not None and str(value).strip()}
        try:
            self._brain.phrase(keys, texts, solicited, lambda speech: self._on_ai(pending, speech))
        except Exception:
            log.exception("brain.phrase failed; using the phrase bank")
            self._on_ai(pending, None)
        return True

    def _on_ai(self, pending: _Pending, speech: SpeechRequested | None) -> None:
        if pending.done:
            if speech is not None:
                log.debug("late AI line for %s dropped (%s)", pending.keys[0], pending.outcome)
            return
        if speech is None or not speech.text.strip():
            self._finish(pending, "fallback")
            self._fallback(pending)
            return
        self._finish(pending, "ai")
        if speech.solicited != pending.solicited:
            speech = SpeechRequested(text=speech.text, state=speech.state, portrait=speech.portrait,
                                     solicited=pending.solicited, source=speech.source)
        self._bus.publish(speech)
        log.debug("AI line from %s", speech.source)

    def _on_timeout(self, pending: _Pending) -> None:
        if pending.done:
            return
        log.info("AI line for %s timed out; using the phrase bank", pending.keys[0])
        self._finish(pending, "fallback")
        self._fallback(pending)

    def _finish(self, pending: _Pending, outcome: str) -> None:
        pending.done = True
        pending.outcome = outcome
        self._pending.pop(pending.id, None)

    def _fallback(self, pending: _Pending) -> bool:
        if not self._say_offline(pending.keys, pending.solicited, pending.variables):
            log.debug("no phrase-bank line for %s either", pending.keys[0])
            return False
        return True

    def _say_offline(self, keys: tuple[str, ...], solicited: bool, variables: Mapping[str, Any]) -> bool:
        speech = self._selector.say(*keys, solicited=solicited, **variables)
        if speech is None:
            return False
        self._bus.publish(speech)
        return True

    def _on_quiet(self, event: QuietModeChanged | DoNotDisturbChanged) -> None:
        if event.enabled:
            self.abandon_pending()
