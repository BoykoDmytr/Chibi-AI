"""Observer - she watches the screen like a friend sitting next to the user and
decides herself when something she sees is worth a word.

Active only while ``settings.vision.commentary == "observer"``: then screen
vision publishes every analysed frame as
:class:`~chibi.core.events.VisionObserved` (after its privacy re-checks) and
never comments on its own; this class keeps a short rolling log of what she
saw and, now and then, asks the model (:meth:`AIBrain.consider_comment`)
whether anything is worth saying.  Most of those decisions end in silence,
and silence is kept cheap.

**The log**: at most :data:`LOG_LIMIT` moments no older than
:data:`LOG_MAX_AGE_SECONDS`, in memory only.  A frame that shows (nearly) the
same as the newest moment updates it instead of adding a copy, so the log
says "the same error for 7 minutes" rather than repeating it.  Someone's
correspondence (``chat`` / ``email`` scenes) is never logged.  An event that
needs a second frame (a "death", a "victory") counts only once confirmed, and
an event below ``vision.min_confidence`` is logged as ``other``.  The log is
forgotten on the kill switch, when ``SCREEN_CAPTURE`` / ``CLOUD_VISION`` /
``CLOUD_AI`` is revoked, when the commentary leaves ``"observer"``, when the
vision, activity or AI settings change (allowed apps, exclusions, another
provider), when a game ends, and when what she saw may no longer go to the
configured AI provider.

**When she asks the model** - every condition, cheapest first:

* the moment has something to go on (a description or an event), the user
  did not ask for this frame themselves ("подивись на екран" is answered by
  vision), a spoiler is not hidden from her, and it is fresh
  (:data:`FRESH_SECONDS`);
* no decision is in flight (single-flight);
* not a game while ``gaming.reaction_frequency`` is 0 ("silent in games");
* the user is present (not between ``UserIdle`` and ``UserReturned``), she
  is not asleep, no do-not-disturb, no quiet mode, nobody chatted or talked
  to her in the last :data:`CHAT_QUIET_SECONDS`, and a line would be seen or
  heard (not hidden over a fullscreen app without a voice);
* it is not a near-repeat of what she last commented on (within the log's
  age) or last asked about (within :data:`RECONSIDER_SECONDS`), and it is
  interesting: a notable event, the vision model's "interesting", a new
  scene or app, a new event, or a glance after :data:`GLANCE_SECONDS`;
* her cooldown after speaking has passed (:func:`comment_cooldown`, 2-10
  minutes by ``reaction_frequency`` in games / ``commentary_frequency``
  elsewhere, and talkativeness); after a silent decision the next one waits
  :data:`SILENCE_GAP_SECONDS`, growing with every silence in a row up to the
  cooldown (a notable event skips this gap, never the cooldown); after a
  failure :data:`FAILURE_BACKOFF_SECONDS`; two requests are never closer than
  :data:`MIN_DECISION_SPACING_SECONDS` (unless the second is a notable event);
* fewer than :func:`decision_cap` decisions in the last hour (at most
  :data:`MAX_DECISIONS_PER_HOUR`), whatever they came to;
* the shared unsolicited gate allows a line (``can_speak(ignore_gap=notable,
  ignore_fullscreen=True)``: budget, minimum gap, DND, quiet);
* the AI takes ``Purpose.INITIATIVE`` requests, and what she saw may go to
  the configured provider (``may_share``, checked right before sending).

**Stale answers are dropped**: a newer, different observation, a foreground
app change, the user going away or talking to her, quiet mode / DND, a
revoked permission or a settings change cancel the decision in flight
(generation number + cancel token).  When an answer arrives the presence,
quiet, chat and gate checks run again, and an answer older than
:data:`RESULT_MAX_AGE_SECONDS` is not said.

**When she speaks** the line is published as
``SpeechRequested(solicited=False, source="observer:ai")`` with the state
from the model's control line, counted through ``on_spoken`` (the shared
hourly budget and gap) and remembered for "don't repeat yourself".

Privacy: nothing here is persisted; nothing logs a description, an app
title, a window title or her line - only reasons, outcomes and counts.
Main (Qt) thread only; deterministic for a given clock and ``rng``.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import random
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from chibi.ai.brain import (
    OBSERVER_MAX_MOMENTS,
    OBSERVER_SOURCE,
    OUTCOME_CANCELLED,
    OUTCOME_FAILED,
    OUTCOME_SPOKE,
    OUTCOME_UNAVAILABLE,
    BrainContext,
    CommentDecision,
    SeenMoment,
    WatchContext,
    similarity,
)
from chibi.ai.types import CancelToken, Purpose
from chibi.core.clock import Clock
from chibi.core.config import Settings
from chibi.core.event_bus import EventBus
from chibi.core.events import (
    DoNotDisturbChanged,
    ForegroundAppChanged,
    FullscreenChanged,
    GameEnded,
    ObservationStopped,
    Permission,
    PermissionChanged,
    QuietModeChanged,
    SettingsChanged,
    UserIdle,
    UserReturned,
    VisionObserved,
    VoiceTranscript,
)

log = logging.getLogger(__name__)

MINUTE = 60.0
HOUR = 3600.0

COMMENTARY_OBSERVER = "observer"

#: The rolling log of what she saw.
LOG_LIMIT = OBSERVER_MAX_MOMENTS
LOG_MAX_AGE_SECONDS = 15 * MINUTE
#: An observation older than this when it arrives is logged but never asked about.
FRESH_SECONDS = 60.0
#: After the user wrote or talked to her she does not start commenting for this long.
CHAT_QUIET_SECONDS = 60.0
#: Her pause after saying something, by frequency (see comment_cooldown).
MIN_COOLDOWN_SECONDS = 2 * MINUTE
MAX_COOLDOWN_SECONDS = 10 * MINUTE
#: Cooldowns and gaps vary by up to this fraction (seeded), so she is not a metronome.
COOLDOWN_JITTER = 0.15
#: After a silent decision; grows by SILENCE_GAP_GROWTH with each silence in a row.
SILENCE_GAP_SECONDS = 60.0
SILENCE_GAP_GROWTH = 1.5
#: After a failed request (offline, budget, error).
FAILURE_BACKOFF_SECONDS = 3 * MINUTE
#: No two decision requests closer than this (also when the first one went stale).
MIN_DECISION_SPACING_SECONDS = 30.0
#: Decision requests per rolling hour, whatever they came to (see decision_cap).
MAX_DECISIONS_PER_HOUR = 20
MIN_DECISIONS_PER_HOUR = 6
#: An answer that took longer than this is not said (the moment has passed).
RESULT_MAX_AGE_SECONDS = 60.0
#: The same thing she already asked about is not asked about again before this.
RECONSIDER_SECONDS = 10 * MINUTE
#: Nothing new for this long: she may look up and consider the screen again.
GLANCE_SECONDS = 8 * MINUTE
#: Her own lines remembered for "don't repeat yourself".
SAID_KEPT = 4
SAID_MAX_AGE_SECONDS = 30 * MINUTE
#: Two descriptions at least this similar (token Jaccard) are the same moment.
DUPLICATE_SIMILARITY = 0.6

#: Vision vocabularies (chibi.vision.analysis) this module reasons about.
GAME_SCENES = frozenset({"gameplay", "menu", "inventory", "map", "dialogue", "death_screen",
                         "victory_screen", "cutscene"})
GAME_EVENTS = frozenset({"player_death", "victory", "level_up", "boss_fight", "combat", "low_health",
                         "loot", "achievement", "new_area", "puzzle"})
#: Someone's correspondence: never logged, never commented on.
PRIVATE_SCENES = frozenset({"chat", "email"})
#: Moments worth a word on their own (they skip the gap after a silence).
NOTABLE_EVENTS = frozenset({"player_death", "victory", "level_up", "boss_fight", "achievement",
                            "new_area", "low_health", "loot", "error_on_screen", "build_failed",
                            "tests_passed"})
#: Events vision only trusts after a second frame (VisionObserved.confirmed).
CONFIRM_EVENTS = frozenset({"player_death", "victory", "boss_fight", "achievement", "level_up"})
EMPTY_EVENTS = frozenset({"none", "other"})

#: Settings sections whose change may make what she saw off limits (allowed apps,
#: exclusions, another AI provider): forget it.
_FORGET_SECTIONS = frozenset({"", "vision", "activity", "ai"})
#: Settings sections whose change makes a decision in flight stale.
_STALE_SECTIONS = frozenset({"gaming", "personality", "character", "behavior"})
#: Revoking any of these forgets what she saw.
_SCREEN_PERMISSIONS = frozenset({Permission.SCREEN_CAPTURE, Permission.CLOUD_VISION, Permission.CLOUD_AI})


class CommentBrain(Protocol):
    """What the observer needs from :class:`~chibi.ai.brain.AIBrain`."""

    def available(self, purpose: Purpose) -> bool: ...

    def consider_comment(self, observations: Sequence[SeenMoment], context: WatchContext,
                         on_done: Callable[[CommentDecision], None]) -> CancelToken | None: ...


class PermissionChecker(Protocol):
    def is_granted(self, permission: Permission) -> bool: ...


def _clamp_percent(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return 50.0
    return min(max(float(value), 0.0), 100.0)


def frequency(settings: Settings, is_game: bool) -> float:
    """0..100: ``gaming.reaction_frequency`` in games, ``commentary_frequency`` elsewhere."""
    if is_game:
        return _clamp_percent(settings.gaming.reaction_frequency)
    return _clamp_percent(settings.personality.commentary_frequency)


def comment_cooldown(settings: Settings, is_game: bool) -> float:
    """Seconds she waits after a comment: 10 minutes at frequency 0 down to 2 at
    100, a fifth longer for a quiet character and a fifth shorter for a
    talkative one, always within 2..10 minutes."""
    span = MAX_COOLDOWN_SECONDS - MIN_COOLDOWN_SECONDS
    base = MAX_COOLDOWN_SECONDS - span * frequency(settings, is_game) / 100
    talk = 1.2 - 0.4 * _clamp_percent(settings.personality.talkativeness) / 100
    return min(max(base * talk, MIN_COOLDOWN_SECONDS), MAX_COOLDOWN_SECONDS)


def decision_cap(settings: Settings, is_game: bool) -> int:
    """Decision requests allowed per rolling hour: 6 at frequency 0 up to
    :data:`MAX_DECISIONS_PER_HOUR` at 100."""
    scaled = round(MAX_DECISIONS_PER_HOUR * (0.3 + 0.7 * frequency(settings, is_game) / 100))
    return max(MIN_DECISIONS_PER_HOUR, min(MAX_DECISIONS_PER_HOUR, scaled))


@dataclass(frozen=True)
class _Moment:
    """One entry of the rolling log (in memory only)."""

    scene: str
    event: str
    description: str
    app_title: str
    is_game: bool
    notable: bool
    first_seen: float  # monotonic
    last_seen: float  # monotonic


def _same(a: _Moment, b: _Moment) -> bool:
    """(Nearly) the same thing on the screen."""
    return (a.scene == b.scene and a.event == b.event and a.app_title == b.app_title
            and similarity(a.description, b.description) >= DUPLICATE_SIMILARITY)


@dataclass
class _Pending:
    generation: int
    moment: _Moment
    asked_at: float  # monotonic
    notable: bool
    token: CancelToken | None = None


class Observer:
    """See the module docstring.  Main (Qt) thread only."""

    def __init__(
        self,
        bus: EventBus,
        brain: CommentBrain,
        settings: Callable[[], Settings],
        clock: Clock,
        rng: random.Random,
        can_speak: Callable[..., bool],
        on_spoken: Callable[[], None],
        *,
        context: Callable[[], BrainContext] | None = None,
        last_chat: Callable[[], float | None] | None = None,
        details: Callable[[VisionObserved], Any] | None = None,
        may_share: Callable[[], bool] | None = None,
        permissions: PermissionChecker | None = None,
        sleeping: Callable[[], bool] | None = None,
    ) -> None:
        """``can_speak`` is the shared unsolicited gate
        (``ReactionEngine.can_speak_unsolicited``, called with ``ignore_gap`` and
        ``ignore_fullscreen=True``); ``on_spoken`` counts her line against the
        shared budget (``ReactionEngine.note_unsolicited``).

        Optional: ``context`` is the app's :class:`BrainContext` (activity, game,
        ``user_away``); ``last_chat`` the monotonic time of the user's last chat
        message; ``details`` maps a ``VisionObserved`` to what vision knows about
        it besides (``interesting``, ``spoiler``, ``is_game``, ``solicited``;
        ``VisionIntegration.details_for``); ``may_share`` says whether what she
        saw may go to the configured AI provider now
        (``VisionIntegration.screen_data_may_leave``); ``permissions`` is checked
        for ``SCREEN_CAPTURE`` when an observation arrives and for
        ``APP_METADATA`` when ``UserIdle`` does; ``sleeping`` - she does not
        watch while asleep."""
        self._bus = bus
        self._brain = brain
        self._settings = settings
        self._clock = clock
        self._rng = rng
        self._can_speak = can_speak
        self._on_spoken = on_spoken
        self._context = context
        self._last_chat = last_chat
        self._details = details
        self._may_share = may_share
        self._permissions = permissions
        self._sleeping = sleeping

        self._log: deque[_Moment] = deque()
        self._said: deque[tuple[float, str]] = deque(maxlen=SAID_KEPT)
        self._decisions: deque[float] = deque()  # monotonic times of decision requests
        self._pending: _Pending | None = None
        self._generation = 0
        self._speech_ready_at: float | None = None
        self._decision_ready_at: float | None = None
        self._silences = 0
        self._last_decision_at: float | None = None
        self._last_consulted: tuple[_Moment, float] | None = None
        self._last_commented: tuple[_Moment, float] | None = None
        self._last_comment_at: float | None = None
        self._last_voice: float | None = None

        self._away = False
        self._fullscreen = False
        self._quiet = False
        self._quiet_until: float | None = None  # epoch seconds; None = until lifted
        self._dnd_override: bool | None = None
        self._foreground: tuple[str, str] | None = None

        self._unsubscribers: list[Callable[[], None]] = [
            bus.subscribe(VisionObserved, self._on_observed),
            bus.subscribe(ForegroundAppChanged, self._on_foreground),
            bus.subscribe(UserIdle, self._on_user_idle),
            bus.subscribe(UserReturned, self._on_user_returned),
            bus.subscribe(FullscreenChanged, self._on_fullscreen),
            bus.subscribe(QuietModeChanged, self._on_quiet),
            bus.subscribe(DoNotDisturbChanged, self._on_dnd),
            bus.subscribe(ObservationStopped, self._on_observation_stopped),
            bus.subscribe(PermissionChanged, self._on_permission),
            bus.subscribe(SettingsChanged, self._on_settings_changed),
            bus.subscribe(GameEnded, self._on_game_ended),
            bus.subscribe(VoiceTranscript, self._on_voice),
        ]

    # -- public API ------------------------------------------------------------------------

    @property
    def pending(self) -> bool:
        """A decision is in flight."""
        return self._pending is not None

    @property
    def moments(self) -> int:
        """How many moments the rolling log holds (diagnostics, tests)."""
        return len(self._log)

    def decisions_last_hour(self) -> int:
        self._prune_decisions(self._clock.monotonic())
        return len(self._decisions)

    def clear(self) -> None:
        """Forget what she saw and said about it, and drop a decision in flight.
        Cooldowns and the hourly cap stay: forgetting must not make her chattier."""
        had = bool(self._log or self._said or self._pending)
        self._invalidate("forgotten")
        self._log.clear()
        self._said.clear()
        self._last_consulted = None
        self._last_commented = None
        if had:
            log.info("observer: forgot what she saw")

    def close(self) -> None:
        """Unsubscribe and forget (app shutdown, tests).  Idempotent."""
        self.clear()
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()

    # -- observations ----------------------------------------------------------------------

    def _on_observed(self, event: VisionObserved) -> None:
        settings = self._read_settings()
        if settings is None or settings.vision.commentary != COMMENTARY_OBSERVER:
            if self._log:
                self.clear()
            return
        if not self._granted(Permission.SCREEN_CAPTURE):
            self.clear()  # queued before a revoke, delivered after it
            return
        info = self._details_for(event)
        now = self._clock.monotonic()
        moment = self._moment(event, info, settings, now)
        pending = self._pending
        if pending is not None and (moment is None or not _same(pending.moment, moment)):
            self._invalidate("newer observation")
        if moment is None:
            return
        self._remember(moment, now)
        reason = self._blocked(moment, event, info, settings, now)
        if reason is not None:
            log.debug("observer: not asking (%s)", reason)
            return
        self._ask(moment, settings, now)

    def _moment(self, event: VisionObserved, info: Any, settings: Settings, now: float) -> _Moment | None:
        """The log entry for *event*, or None when it must not be kept (correspondence)."""
        scene = event.scene if isinstance(event.scene, str) else ""
        if not scene or scene in PRIVATE_SCENES:
            return None
        kind = event.event if isinstance(event.event, str) and event.event else "none"
        confidence = event.confidence if isinstance(event.confidence, (int, float)) else 0.0
        if kind not in EMPTY_EVENTS and (confidence < settings.vision.min_confidence
                                         or (kind in CONFIRM_EVENTS and not event.confirmed)):
            # Not sure enough (or still waiting for its second frame): never claimed as seen.
            kind = "other"
        description = event.description.strip() if isinstance(event.description, str) else ""
        app_title = event.app_title.strip() if isinstance(event.app_title, str) else ""
        is_game = getattr(info, "is_game", None)
        if not isinstance(is_game, bool):
            is_game = scene in GAME_SCENES or kind in GAME_EVENTS
        notable = kind in NOTABLE_EVENTS or (bool(event.react) and kind not in EMPTY_EVENTS)
        return _Moment(scene=scene, event=kind, description=description, app_title=app_title,
                       is_game=is_game, notable=notable, first_seen=now, last_seen=now)

    def _remember(self, moment: _Moment, now: float) -> None:
        self._prune_log(now)
        if self._log and _same(self._log[-1], moment):
            self._log[-1] = dataclasses.replace(moment, first_seen=self._log[-1].first_seen)
            return
        self._log.append(moment)
        while len(self._log) > LOG_LIMIT:
            self._log.popleft()

    def _prune_log(self, now: float) -> None:
        while self._log and now - self._log[0].last_seen > LOG_MAX_AGE_SECONDS:
            self._log.popleft()

    def _prune_decisions(self, now: float) -> None:
        while self._decisions and now - self._decisions[0] >= HOUR:
            self._decisions.popleft()

    # -- deciding whether to ask -----------------------------------------------------------

    def _blocked(self, moment: _Moment, event: VisionObserved, info: Any, settings: Settings,
                 now: float) -> str | None:
        """Why the model is not asked about *moment* now, or None (see the module docstring)."""
        if not moment.description and moment.event in EMPTY_EVENTS:
            return "nothing to go on"
        if getattr(info, "solicited", False) is True:
            return "the user asked"
        if getattr(info, "spoiler", False) is True and not settings.gaming.hints_and_spoilers:
            return "spoiler"
        if self._clock.now() - event.timestamp > FRESH_SECONDS:
            return "old"
        if self._pending is not None:
            return "busy"
        if moment.is_game and frequency(settings, True) <= 0:
            return "silent in games"
        reason = self._presence_blocked(settings, now)
        if reason is not None:
            return reason
        if self._last_commented is not None:
            said_about, at = self._last_commented
            if now - at < LOG_MAX_AGE_SECONDS and _same(said_about, moment):
                return "said that already"
        if self._last_consulted is not None:
            asked_about, at = self._last_consulted
            if now - at < RECONSIDER_SECONDS and _same(asked_about, moment):
                return "considered already"
        if self._interest(moment, info, now) is None:
            return "routine"
        if self._speech_ready_at is not None and now < self._speech_ready_at:
            return "cooldown"
        if not moment.notable and self._decision_ready_at is not None and now < self._decision_ready_at:
            return "gap"
        self._prune_decisions(now)
        if len(self._decisions) >= decision_cap(settings, moment.is_game):
            return "hourly cap"
        if not self._gate(moment.notable):
            return "gate"
        if not self._ai_available():
            return "no ai"
        if not self._shareable():
            self.clear()
            return "not shareable"
        return None

    def _presence_blocked(self, settings: Settings, now: float) -> str | None:
        """Conditions checked both before asking and before saying the answer."""
        if self._user_away():
            return "away"
        if self._is_sleeping():
            return "asleep"
        if settings.behavior.do_not_disturb or self._dnd_override:
            return "do not disturb"
        if self._quiet_active():
            return "quiet"
        last = self._last_talk()
        if last is not None and now - last < CHAT_QUIET_SECONDS:
            return "chatting"
        if self._unseen(settings):
            return "unseen"
        return None

    def _interest(self, moment: _Moment, info: Any, now: float) -> str | None:
        if moment.notable:
            return "event"
        if getattr(info, "interesting", False) is True:
            return "interesting"
        last = self._last_consulted
        if last is None:
            return "first look"
        earlier = last[0]
        if (earlier.scene, earlier.app_title) != (moment.scene, moment.app_title):
            return "new scene"
        if moment.event != earlier.event and moment.event not in EMPTY_EVENTS:
            return "new event"
        if self._last_decision_at is None or now - self._last_decision_at >= GLANCE_SECONDS:
            return "glance"
        return None

    def _ask(self, moment: _Moment, settings: Settings, now: float) -> None:
        self._generation += 1
        generation = self._generation
        pending = _Pending(generation=generation, moment=moment, asked_at=now, notable=moment.notable)
        self._pending = pending
        self._decisions.append(now)
        self._last_decision_at = now
        self._last_consulted = (moment, now)
        spacing = now + MIN_DECISION_SPACING_SECONDS
        self._decision_ready_at = max(self._decision_ready_at or spacing, spacing)
        moments = self._seen_moments(now)
        context = self._watch_context(moment, settings, now)
        log.info("observer: asking the model (%d moment(s), %d decision(s) this hour)",
                 len(moments), len(self._decisions))
        try:
            token = self._brain.consider_comment(moments, context,
                                                 lambda decision: self._on_decision(generation, decision))
        except Exception as exc:
            log.error("observer decision could not be requested (%s)", type(exc).__name__)
            if self._pending is pending:
                self._pending = None
                self._decision_ready_at = now + FAILURE_BACKOFF_SECONDS
            return
        if self._pending is pending:
            pending.token = token

    def _seen_moments(self, now: float) -> list[SeenMoment]:
        return [SeenMoment(scene=m.scene, event=m.event, description=m.description, app_title=m.app_title,
                           seconds_ago=max(0.0, now - m.last_seen),
                           lasted_seconds=max(0.0, m.last_seen - m.first_seen))
                for m in self._log]

    def _watch_context(self, moment: _Moment, settings: Settings, now: float) -> WatchContext:
        brain = self._brain_context()
        said = tuple(text for at, text in self._said if now - at < SAID_MAX_AGE_SECONDS)
        since = None if self._last_comment_at is None else (now - self._last_comment_at) / MINUTE
        return WatchContext(
            activity_category=brain.activity_category if brain else None,
            activity_app=brain.activity_app if brain else None,
            game_title=brain.game_title if brain else None,
            game_minutes=brain.game_minutes if brain else None,
            is_game=moment.is_game,
            hints_allowed=bool(settings.gaming.hints_and_spoilers),
            said_recently=said,
            minutes_since_comment=since,
        )

    # -- the answer ------------------------------------------------------------------------

    def _on_decision(self, generation: int, decision: CommentDecision) -> None:
        pending = self._pending
        if pending is None or pending.generation != generation or generation != self._generation:
            log.debug("observer: stale decision dropped")
            return
        self._pending = None
        now = self._clock.monotonic()
        settings = self._read_settings()
        outcome = getattr(decision, "outcome", OUTCOME_FAILED)
        if outcome == OUTCOME_CANCELLED:
            return
        if outcome in (OUTCOME_FAILED, OUTCOME_UNAVAILABLE) or settings is None:
            log.info("observer decision: %s", outcome)
            self._decision_ready_at = now + FAILURE_BACKOFF_SECONDS
            return
        speech = getattr(decision, "speech", None)
        if outcome != OUTCOME_SPOKE or speech is None or not getattr(speech, "text", ""):
            log.info("observer decision: %s", outcome)
            self._note_silence(pending.moment, settings, now)
            return
        reason = self._delivery_blocked(pending, settings, now)
        if reason is not None:
            log.info("observer: her line is not said (%s)", reason)
            self._note_silence(pending.moment, settings, now)
            return
        self._bus.publish(dataclasses.replace(speech, solicited=False, source=OBSERVER_SOURCE))
        try:
            self._on_spoken()
        except Exception:
            log.exception("counting an observer line failed")
        self._said.append((now, speech.text))
        self._last_commented = (pending.moment, now)
        self._last_comment_at = now
        self._silences = 0
        cooldown = comment_cooldown(settings, pending.moment.is_game) * self._jitter()
        self._speech_ready_at = self._decision_ready_at = now + cooldown
        log.info("observer decision: spoke (next in %.0f s)", cooldown)

    def _delivery_blocked(self, pending: _Pending, settings: Settings, now: float) -> str | None:
        if settings.vision.commentary != COMMENTARY_OBSERVER:
            return "observer off"
        if now - pending.asked_at > RESULT_MAX_AGE_SECONDS:
            return "too late"
        if not self._granted(Permission.SCREEN_CAPTURE):
            return "capture revoked"
        last = self._last_talk()
        if last is not None and last >= pending.asked_at:
            return "chatting"
        reason = self._presence_blocked(settings, now)
        if reason is not None:
            return reason
        if not self._gate(pending.notable):
            return "gate"
        return None

    def _note_silence(self, moment: _Moment, settings: Settings, now: float) -> None:
        self._silences += 1
        full = comment_cooldown(settings, moment.is_game)
        gap = min(full, SILENCE_GAP_SECONDS * SILENCE_GAP_GROWTH ** min(self._silences - 1, 16))
        self._decision_ready_at = now + gap * self._jitter()

    def _invalidate(self, why: str) -> None:
        """Make the decision in flight stale (its answer will be dropped) and cancel it."""
        self._generation += 1
        pending, self._pending = self._pending, None
        if pending is None:
            return
        if pending.token is not None:
            pending.token.cancel()
        log.debug("observer: decision in flight dropped (%s)", why)

    # -- bus handlers ----------------------------------------------------------------------

    def _on_foreground(self, event: ForegroundAppChanged) -> None:
        private = event.app.private or event.category.value == "private"
        key = ("private", "") if private else (event.category.value, event.app.process_name)
        if self._foreground is not None and key != self._foreground:
            self._invalidate("foreground changed")
        self._foreground = key

    def _on_user_idle(self, _event: UserIdle) -> None:
        if not self._granted(Permission.APP_METADATA):
            return  # queued before the monitor stopped: no UserReturned would follow
        self._away = True
        self._invalidate("user away")

    def _on_user_returned(self, _event: UserReturned) -> None:
        self._away = False

    def _on_fullscreen(self, event: FullscreenChanged) -> None:
        self._fullscreen = bool(event.active)

    def _on_quiet(self, event: QuietModeChanged) -> None:
        self._quiet = bool(event.enabled)
        self._quiet_until = event.until if event.enabled else None
        if event.enabled:
            self._invalidate("quiet mode")

    def _on_dnd(self, event: DoNotDisturbChanged) -> None:
        self._dnd_override = bool(event.enabled)
        if event.enabled:
            self._invalidate("do not disturb")

    def _on_observation_stopped(self, _event: ObservationStopped) -> None:
        self._away = False
        self._foreground = None
        self.clear()

    def _on_permission(self, event: PermissionChanged) -> None:
        if event.granted:
            return
        if event.permission in _SCREEN_PERMISSIONS:
            self.clear()
        elif event.permission is Permission.APP_METADATA:
            self._away = False  # the monitor stops: no UserReturned will come
            self._foreground = None

    def _on_settings_changed(self, event: SettingsChanged) -> None:
        if event.section in ("", "behavior"):
            self._dnd_override = None  # the persisted setting is authoritative again
        settings = self._read_settings()
        if settings is None or settings.vision.commentary != COMMENTARY_OBSERVER or \
                event.section in _FORGET_SECTIONS:
            self.clear()
        elif event.section in _STALE_SECTIONS:
            self._invalidate("settings changed")

    def _on_game_ended(self, _event: GameEnded) -> None:
        # Also a game made private while running (forgotten): nothing of it stays.
        self.clear()

    def _on_voice(self, _event: VoiceTranscript) -> None:
        # The user is talking to her: no comment of her own now.  The text is not kept.
        self._last_voice = self._clock.monotonic()
        self._invalidate("user talking")

    # -- helpers ---------------------------------------------------------------------------

    def _read_settings(self) -> Settings | None:
        try:
            return self._settings()
        except Exception:
            log.exception("reading the settings failed; the observer stays quiet")
            return None

    def _granted(self, permission: Permission) -> bool:
        if self._permissions is None:
            return True
        try:
            return bool(self._permissions.is_granted(permission))
        except Exception:
            log.exception("permission check failed")
            return False

    def _details_for(self, event: VisionObserved) -> Any:
        if self._details is None:
            return None
        try:
            return self._details(event)
        except Exception:
            log.exception("observation details unavailable")
            return None

    def _brain_context(self) -> BrainContext | None:
        if self._context is None:
            return None
        try:
            return self._context()
        except Exception:
            log.exception("brain context unavailable")
            return None

    def _user_away(self) -> bool:
        if self._away:
            return True
        context = self._brain_context()
        return bool(context is not None and context.user_away)

    def _is_sleeping(self) -> bool:
        if self._sleeping is None:
            return False
        try:
            return bool(self._sleeping())
        except Exception:
            log.exception("sleeping() failed")
            return False

    def _quiet_active(self) -> bool:
        if not self._quiet:
            return False
        return self._quiet_until is None or self._clock.now() < self._quiet_until

    def _last_talk(self) -> float | None:
        """Monotonic time the user last wrote or talked to her."""
        times = [self._last_voice]
        if self._last_chat is not None:
            try:
                times.append(self._last_chat())
            except Exception:
                log.exception("last_chat() failed")
        numbers = [t for t in times if isinstance(t, (int, float)) and not isinstance(t, bool)
                   and math.isfinite(t)]
        return max(numbers) if numbers else None

    def _unseen(self, settings: Settings) -> bool:
        """Hidden over a fullscreen app with no voice to carry the line."""
        if not (self._fullscreen and settings.behavior.hide_on_fullscreen):
            return False
        voice = settings.voice
        return not (voice.tts_enabled and voice.speak == "all" and voice.speak_when_hidden)

    def _gate(self, notable: bool) -> bool:
        try:
            return bool(self._can_speak(ignore_gap=notable, ignore_fullscreen=True))
        except Exception:
            log.exception("the unsolicited-speech gate failed")
            return False

    def _ai_available(self) -> bool:
        try:
            return bool(self._brain.available(Purpose.INITIATIVE))
        except Exception:
            log.exception("AI availability check failed")
            return False

    def _shareable(self) -> bool:
        if self._may_share is None:
            return True
        try:
            return bool(self._may_share())
        except Exception:
            log.exception("may_share() failed")
            return False

    def _jitter(self) -> float:
        return 1.0 + self._rng.uniform(-COOLDOWN_JITTER, COOLDOWN_JITTER)
