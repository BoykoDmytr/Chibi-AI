"""Decision engine: what she does on her own initiative, and when.

Replaces the reaction engine's random idle chatter when wired.  Every ~10 s
:meth:`DecisionEngine.tick` looks at the situation (:class:`InitiativeContext`,
supplied by the app) and her emotions, scores a fixed set of
:class:`Candidate` initiatives and performs the best one above
:data:`THRESHOLD` - or, most of the time, nothing:

    score = priority * emotion_fit * relevance - interruption_cost
            - novelty penalty (decays over an hour) + small seeded jitter

* ``priority`` comes from the personality sliders (humor favours jokes,
  curiosity questions...);
* ``emotion_fit`` from the emotion engine's 0..100 values (a joke fits a
  happy mood, a nap fits boredom with low energy);
* ``relevance`` from the situation (asking about a game fits 15-90 minutes
  into it, a break fits a long session);
* ``interruption_cost`` from what the user is doing: gaming 0.8, video 0.7,
  programming / working 0.5, browsing 0.3, idle 0.1 (unknown 0.3).  A
  non-speech action (a jump, a wave) costs half as much.  Fullscreen means
  she never speaks (nor moves: she is hidden).

Gates, in order - nothing at all happens:

* before ``AppStarted`` (the greeting comes first) and for
  :data:`GLOBAL_GAP_SECONDS` after it and after every initiative;
* while the user is away, while she sleeps (posture ``sleep``), in quiet
  mode or do-not-disturb, in fullscreen.

Speech candidates additionally need ``can_speak()`` (the reaction engine's
unsolicited-speech gate: budget, minimum gap, fullscreen, DND, quiet) and
their lines are reported through ``on_spoken`` so both engines share one
hourly budget.  "Spontaneous" candidates (jokes, questions, jumps...) only
get a chance on a seeded random roll whose rate follows talkativeness and
commentary frequency (about 2.4 opportunities an hour by default);
situational ones (a wave when the user comes back, a nap from boredom) are
considered on every tick.  Each candidate also has its own cooldown.

Speech goes through :meth:`SpeechComposer.compose` (AI-phrased when
available, phrase bank otherwise); actions are published as
:class:`~chibi.core.events.InitiativeAction` for the desktop controller.

Main (Qt) thread only; deterministic for a given clock and ``rng``.
Nothing here logs app names or game titles - only candidate names.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from chibi.core.clock import Clock
from chibi.core.config import PersonalitySettings, Settings
from chibi.core.event_bus import EventBus
from chibi.core.events import (
    AppStarted,
    DoNotDisturbChanged,
    InitiativeAction,
    QuietModeChanged,
    SettingsChanged,
)
from chibi.personality.reactions import speakable_title
from chibi.personality.text_uk import format_duration

log = logging.getLogger(__name__)

HOUR = 3600.0
MINUTE = 60.0

#: Minimum score for anything to happen.
THRESHOLD = 0.2
#: Seeded noise added to every score, so equal situations do not always pick the same.
JITTER = 0.08
#: Something she did lately scores lower for a while (decays with this time
#: constant), so the best candidate is not simply repeated once its cooldown ends.
NOVELTY_PENALTY = 0.3
NOVELTY_DECAY_SECONDS = HOUR
#: No two initiatives (speech or action) closer than this.
GLOBAL_GAP_SECONDS = 4 * MINUTE
DEFAULT_TICK_SECONDS = 10.0
MAX_TICK_SECONDS = 60.0
#: "The user just came back" lasts this long (for the wave).
RETURN_WINDOW_SECONDS = 90.0

#: Speech interruption cost per activity category.
SPEECH_INTERRUPTION: dict[str, float] = {
    "gaming": 0.8,
    "watching_video": 0.7,
    "programming": 0.5,
    "working": 0.5,
    "browsing": 0.3,
    "idle": 0.1,
}
#: Unknown, private or unobserved activity.
DEFAULT_INTERRUPTION = 0.3
#: An animation interrupts less than a sentence.
ACTION_INTERRUPTION_FACTOR = 0.5
#: Activities during which idle chatter waits (she does not chat over a game or a film).
_ABSORBING = frozenset({"gaming", "watching_video"})
#: Long sessions worth a break suggestion (a film is not interrupted for one).
_BREAK_ACTIVITIES = frozenset({"programming", "working", "browsing"})

BREAK_GAME_MINUTES = 120.0
BREAK_ACTIVITY_MINUTES = 90.0
ASK_GAME_MIN_MINUTES = 10.0
ASK_GAME_MAX_MINUTES = 180.0
PROPOSE_TALK_AFTER_MINUTES = 120.0

#: Emotion values used when the engine cannot be read.
NEUTRAL_EMOTIONS: dict[str, float] = {
    "happiness": 60.0, "energy": 70.0, "boredom": 30.0, "affection": 50.0,
    "curiosity": 60.0, "excitement": 20.0, "hunger": 30.0, "trust": 40.0,
}


@dataclass(frozen=True)
class InitiativeContext:
    """The situation as the app sees it (all observation-derived fields are
    empty while activity monitoring is not permitted)."""

    now: float  # monotonic seconds
    posture: str = "idle"  # CharacterState value of her posture
    activity_category: str | None = None  # ActivityCategory value, None = unknown
    game_running: bool = False
    game_minutes: float | None = None
    fullscreen: bool = False
    user_away: bool = False
    #: None = no chat yet this session (the engine then counts from the session start).
    minutes_since_last_chat: float | None = None
    time_of_day: str = "day"  # "morning" | "day" | "evening" | "night"
    ai_available: bool = False
    #: Title of the running game (for "initiative.ask_game"); speakable, never logged.
    game_title: str | None = None
    #: How long the current activity category has lasted, in minutes.
    activity_minutes: float | None = None
    #: Seconds since the user came back from being away; None = not recently
    #: (the engine fills it from ``user_away`` transitions when the app does not).
    seconds_since_return: float | None = None


def _one(_ctx: InitiativeContext) -> float:
    return 1.0


def _no_variables(_ctx: InitiativeContext) -> Mapping[str, Any]:
    return {}


@dataclass(frozen=True)
class Candidate:
    name: str
    keys: tuple[str, ...]
    priority: float
    cooldown: float  # seconds
    requires: Callable[[InitiativeContext], bool]
    emotion_fit: Callable[[Mapping[str, float]], float]
    interruption_cost: Callable[[InitiativeContext], float]
    action: str | None = None  # InitiativeAction.action; None = speech
    relevance: Callable[[InitiativeContext], float] = _one
    #: Only considered on the random "talk now?" roll (False: every tick).
    spontaneous: bool = True
    #: Phrase variables for the composer.
    variables: Callable[[InitiativeContext], Mapping[str, Any]] = _no_variables


class Composer(Protocol):
    def compose(self, *keys: str, solicited: bool = False, **variables: Any) -> bool: ...


class Emotions(Protocol):
    def values(self) -> dict[str, float]: ...


# =========================================================================================
# Costs, fits and the default candidates
# =========================================================================================


def _activity(ctx: InitiativeContext) -> str:
    if ctx.game_running:
        return "gaming"
    return ctx.activity_category or "unknown"


def speech_interruption(ctx: InitiativeContext) -> float:
    """How much a sentence would interrupt the user now (inf in fullscreen)."""
    if ctx.fullscreen:
        return math.inf
    return SPEECH_INTERRUPTION.get(_activity(ctx), DEFAULT_INTERRUPTION)


def action_interruption(ctx: InitiativeContext) -> float:
    """An animation interrupts half as much (still never in fullscreen)."""
    return speech_interruption(ctx) * ACTION_INTERRUPTION_FACTOR


def _v(values: Mapping[str, float], name: str) -> float:
    value = values.get(name, NEUTRAL_EMOTIONS[name])
    try:
        number = float(value)
    except (TypeError, ValueError):
        return NEUTRAL_EMOTIONS[name] / 100.0
    if not math.isfinite(number):
        number = NEUTRAL_EMOTIONS[name]
    return min(max(number, 0.0), 100.0) / 100.0


def _chatter_allowed(ctx: InitiativeContext) -> bool:
    return _activity(ctx) not in _ABSORBING


def _night_relevance(ctx: InitiativeContext) -> float:
    return 0.7 if ctx.time_of_day == "night" else 1.0


def _joke_fit(v: Mapping[str, float]) -> float:
    fit = 0.6 + 0.4 * _v(v, "happiness") + 0.1 * _v(v, "excitement")
    return fit * (0.6 if _v(v, "energy") < 0.3 else 1.0)


def _thought_fit(v: Mapping[str, float]) -> float:
    return 0.7 + 0.3 * _v(v, "boredom")


def _question_fit(v: Mapping[str, float]) -> float:
    return 0.55 + 0.45 * _v(v, "curiosity")


def _game_fit(v: Mapping[str, float]) -> float:
    return 0.6 + 0.3 * _v(v, "curiosity") + 0.2 * _v(v, "excitement")


def _break_fit(v: Mapping[str, float]) -> float:
    return 0.7 + 0.4 * _v(v, "affection")


def _talk_fit(v: Mapping[str, float]) -> float:
    """Only a high affection makes her ask for a chat."""
    return max(0.0, (_v(v, "affection") - 0.55) / 0.45) * 1.2 + 0.1 * _v(v, "boredom")


def _memory_fit(v: Mapping[str, float]) -> float:
    return 0.4 + 0.4 * _v(v, "affection") + 0.3 * _v(v, "trust")


def _wave_fit(v: Mapping[str, float]) -> float:
    return 0.5 + 0.5 * _v(v, "affection")


def _nap_fit(v: Mapping[str, float]) -> float:
    return 1.0 if _v(v, "boredom") > 0.85 and _v(v, "energy") < 0.40 else 0.0


def _jump_fit(v: Mapping[str, float]) -> float:
    return min(max((_v(v, "excitement") - 0.7) / 0.3, 0.0), 1.0) * 1.2


def _game_ok(ctx: InitiativeContext) -> bool:
    minutes = ctx.game_minutes
    return (ctx.game_running and not ctx.fullscreen and bool(ctx.game_title)
            and minutes is not None and ASK_GAME_MIN_MINUTES <= minutes <= ASK_GAME_MAX_MINUTES)


def _game_relevance(ctx: InitiativeContext) -> float:
    minutes = ctx.game_minutes or 0.0
    return 1.0 if 15.0 <= minutes <= 90.0 else 0.8


def _session_minutes(ctx: InitiativeContext) -> float:
    """Length of the long session a break would interrupt (0 when there is none)."""
    if ctx.game_running:
        return ctx.game_minutes or 0.0
    if ctx.activity_category in _BREAK_ACTIVITIES:
        return ctx.activity_minutes or 0.0
    return 0.0


def _break_ok(ctx: InitiativeContext) -> bool:
    minutes = _session_minutes(ctx)
    limit = BREAK_GAME_MINUTES if ctx.game_running else BREAK_ACTIVITY_MINUTES
    return minutes >= limit


def _break_relevance(ctx: InitiativeContext) -> float:
    return min(1.25, _session_minutes(ctx) / BREAK_GAME_MINUTES)


def _break_variables(ctx: InitiativeContext) -> Mapping[str, Any]:
    minutes = _session_minutes(ctx)
    return {"duration": format_duration(minutes * MINUTE)} if minutes > 0 else {}


def _game_variables(ctx: InitiativeContext) -> Mapping[str, Any]:
    return {"game": speakable_title(ctx.game_title)} if ctx.game_title else {}


def _talk_ok(ctx: InitiativeContext) -> bool:
    minutes = ctx.minutes_since_last_chat
    return _chatter_allowed(ctx) and minutes is not None and minutes >= PROPOSE_TALK_AFTER_MINUTES


def _returned(ctx: InitiativeContext) -> bool:
    return ctx.seconds_since_return is not None and 0 <= ctx.seconds_since_return <= RETURN_WINDOW_SECONDS


def default_candidates(p: PersonalitySettings) -> tuple[Candidate, ...]:
    """The initiatives she knows, with priorities from the personality sliders."""
    humor, curiosity = p.humor / 100.0, p.curiosity / 100.0
    affection, independence = p.affection / 100.0, p.independence / 100.0
    speech = speech_interruption
    action = action_interruption
    return (
        Candidate("tell_joke", ("idle.joke",), 0.6 + 0.5 * humor, 20 * MINUTE,
                  _chatter_allowed, _joke_fit, speech, relevance=_night_relevance),
        Candidate("share_thought", ("idle.thought",), 0.95, 20 * MINUTE,
                  _chatter_allowed, _thought_fit, speech, relevance=_night_relevance),
        Candidate("ask_question", ("idle.question",), 0.55 + 0.6 * curiosity, 25 * MINUTE,
                  _chatter_allowed, _question_fit, speech, relevance=_night_relevance),
        Candidate("ask_about_game", ("initiative.ask_game",), 1.2 + 0.3 * curiosity, 60 * MINUTE,
                  _game_ok, _game_fit, speech, relevance=_game_relevance, variables=_game_variables),
        Candidate("suggest_break", ("initiative.suggest_break",), 1.2 + 0.3 * affection, 90 * MINUTE,
                  _break_ok, _break_fit, speech, relevance=_break_relevance, variables=_break_variables),
        Candidate("propose_talk", ("initiative.propose_talk",), 0.8 + 0.4 * (1.0 - independence),
                  3 * HOUR, _talk_ok, _talk_fit, speech),
        # Offline the AI cannot pick a memory: skipped.  "idle.thought" is the
        # phrase-bank fallback if the model fails (or nothing is remembered yet).
        Candidate("share_memory", ("initiative.share_memory", "idle.thought"), 0.9, 2 * HOUR,
                  lambda ctx: ctx.ai_available and _chatter_allowed(ctx), _memory_fit, speech),
        Candidate("wave", (), 0.6, 30 * MINUTE, _returned, _wave_fit, action,
                  action="wave", spontaneous=False),
        Candidate("sleep_from_boredom", (), 0.8, HOUR, lambda ctx: True, _nap_fit, action,
                  action="sleep", spontaneous=False),
        # A burst of joy, but not over a game or a film: hopping around there distracts.
        Candidate("jump", (), 0.8, 15 * MINUTE, _chatter_allowed, _jump_fit, action, action="jump"),
    )


def initiative_rate(p: PersonalitySettings) -> float:
    """Spontaneous opportunities per hour (before gates, cooldowns and the budget)."""
    return 0.6 + 3.0 * p.talkativeness / 100.0 + 0.8 * p.commentary_frequency / 100.0


def score(candidate: Candidate, ctx: InitiativeContext, values: Mapping[str, float]) -> float | None:
    """The candidate's score without jitter; None when its requirements are not met."""
    try:
        if not candidate.requires(ctx):
            return None
        cost = candidate.interruption_cost(ctx)
        if math.isinf(cost):
            return None
        value = candidate.priority * candidate.emotion_fit(values) * candidate.relevance(ctx) - cost
    except Exception:
        log.exception("scoring %s failed", candidate.name)
        return None
    return value if math.isfinite(value) else None


# =========================================================================================
# Engine
# =========================================================================================


class DecisionEngine:
    def __init__(
        self,
        bus: EventBus,
        composer: Composer,
        settings: Callable[[], Settings],
        clock: Clock,
        rng: random.Random,
        emotion: Emotions,
        context: Callable[[], InitiativeContext],
        can_speak: Callable[[], bool],
        *,
        on_spoken: Callable[[], None] | None = None,
        candidates: tuple[Candidate, ...] | None = None,
    ) -> None:
        """``on_spoken`` is told about every line she starts (wire it to
        ``ReactionEngine.note_unsolicited`` so both engines share the budget);
        ``candidates`` replaces the default set (tests)."""
        self._bus = bus
        self._composer = composer
        self._settings = settings
        self._clock = clock
        self._rng = rng
        self._emotion = emotion
        self._context = context
        self._can_speak = can_speak
        self._on_spoken = on_spoken
        self._fixed_candidates = candidates
        self._candidate_cache: tuple[tuple[int, ...], tuple[Candidate, ...]] | None = None

        self._started_at: float | None = None
        self._last_initiative: float | None = None
        self._cooldowns: dict[str, float] = {}
        self._last_tick: float | None = None
        self._was_away = False
        self._returned_at: float | None = None
        self._quiet = False
        self._quiet_deadline: float | None = None
        self._dnd_override: bool | None = None
        #: Name -> score of the last evaluation (diagnostics).
        self.last_scores: dict[str, float] = {}

        self._unsubscribers: list[Callable[[], None]] = [
            bus.subscribe(AppStarted, self._on_app_started),
            bus.subscribe(QuietModeChanged, self._on_quiet),
            bus.subscribe(DoNotDisturbChanged, self._on_dnd),
            bus.subscribe(SettingsChanged, self._on_settings_changed),
        ]

    def close(self) -> None:
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()

    # -- public -----------------------------------------------------------------------------

    def candidates(self) -> tuple[Candidate, ...]:
        if self._fixed_candidates is not None:
            return self._fixed_candidates
        p = self._settings().personality
        key = dataclasses.astuple(p)
        if self._candidate_cache is None or self._candidate_cache[0] != key:
            self._candidate_cache = (key, default_candidates(p))
        return self._candidate_cache[1]

    def tick(self) -> None:
        """Called every ~10 s on the main thread: maybe do one thing."""
        now = self._clock.monotonic()
        dt = DEFAULT_TICK_SECONDS if self._last_tick is None else now - self._last_tick
        self._last_tick = now
        dt = min(max(dt, 0.0), MAX_TICK_SECONDS)
        try:
            ctx = self._context()
        except Exception:
            log.exception("initiative context unavailable")
            return
        self._track_return(ctx, now)
        reason = self._blocked_reason(ctx, now)
        if reason is not None:
            log.debug("no initiative: %s", reason)
            return
        settings = self._settings()
        rate = initiative_rate(settings.personality)
        roll = self._rng.random() < 1.0 - math.exp(-rate * dt / HOUR)
        ctx = self._effective(ctx, now)
        values = self._emotion_values()
        speech_ok: bool | None = None
        best: tuple[float, Candidate] | None = None
        self.last_scores = {}
        for candidate in self.candidates():
            if candidate.spontaneous and not roll:
                continue
            if not self._ready(candidate, now):
                continue
            value = score(candidate, ctx, values)
            if value is None:
                continue
            if candidate.action is None:
                if speech_ok is None:
                    speech_ok = self._speech_allowed()
                if not speech_ok:
                    continue
            value -= self._novelty_penalty(candidate, now)
            value += self._rng.uniform(-JITTER, JITTER)
            self.last_scores[candidate.name] = value
            if value >= THRESHOLD and (best is None or value > best[0]):
                best = (value, candidate)
        if best is not None:
            self._perform(best[1], ctx, now)

    # -- gates ------------------------------------------------------------------------------

    def _blocked_reason(self, ctx: InitiativeContext, now: float) -> str | None:
        if self._started_at is None:
            return "not started"
        if ctx.user_away:
            return "user away"
        if ctx.posture == "sleep":
            return "asleep"
        if ctx.fullscreen:
            return "fullscreen"
        if self._quiet_active(now):
            return "quiet mode"
        if self._dnd_active():
            return "do not disturb"
        if self._last_initiative is not None and now - self._last_initiative < GLOBAL_GAP_SECONDS:
            return "global gap"
        return None

    def _quiet_active(self, now: float) -> bool:
        if not self._quiet:
            return False
        if self._quiet_deadline is not None and now >= self._quiet_deadline:
            self._quiet, self._quiet_deadline = False, None
            return False
        return True

    def _dnd_active(self) -> bool:
        """Like the reaction engine: the setting, or a DoNotDisturbChanged(True) since
        the behaviour settings were last applied."""
        if self._dnd_override:
            return True
        try:
            return bool(self._settings().behavior.do_not_disturb)
        except Exception:
            log.exception("settings unavailable; assuming do-not-disturb")
            return True

    def _speech_allowed(self) -> bool:
        try:
            return bool(self._can_speak())
        except Exception:
            log.exception("can_speak() failed; staying quiet")
            return False

    def _ready(self, candidate: Candidate, now: float) -> bool:
        last = self._cooldowns.get(candidate.name)
        return last is None or now - last >= candidate.cooldown

    def _novelty_penalty(self, candidate: Candidate, now: float) -> float:
        last = self._cooldowns.get(candidate.name)
        if last is None:
            return 0.0
        return NOVELTY_PENALTY * math.exp(-max(0.0, now - last) / NOVELTY_DECAY_SECONDS)

    # -- context ----------------------------------------------------------------------------

    def _track_return(self, ctx: InitiativeContext, now: float) -> None:
        if self._was_away and not ctx.user_away:
            self._returned_at = now
        self._was_away = ctx.user_away

    def _effective(self, ctx: InitiativeContext, now: float) -> InitiativeContext:
        """Fill what the engine knows better than the app: time since the user came
        back, and "no chat yet" counted from the session start."""
        changes: dict[str, Any] = {}
        if ctx.seconds_since_return is None and self._returned_at is not None:
            changes["seconds_since_return"] = now - self._returned_at
        if ctx.minutes_since_last_chat is None and self._started_at is not None:
            changes["minutes_since_last_chat"] = (now - self._started_at) / MINUTE
        return dataclasses.replace(ctx, **changes) if changes else ctx

    def _emotion_values(self) -> dict[str, float]:
        try:
            values = dict(self._emotion.values())
        except Exception:
            log.exception("emotion values unavailable; assuming a neutral mood")
            return dict(NEUTRAL_EMOTIONS)
        return {**NEUTRAL_EMOTIONS, **values}

    # -- acting -----------------------------------------------------------------------------

    def _perform(self, candidate: Candidate, ctx: InitiativeContext, now: float) -> None:
        self._cooldowns[candidate.name] = now
        if candidate.action is not None:
            self._bus.publish(InitiativeAction(action=candidate.action))
            self._last_initiative = now
            log.debug("initiative action %s", candidate.name)
            return
        try:
            variables = dict(candidate.variables(ctx))
        except Exception:
            log.exception("variables for %s failed", candidate.name)
            return
        if not self._composer.compose(*candidate.keys, solicited=False, **variables):
            log.debug("initiative %s had nothing to say", candidate.name)
            return
        self._last_initiative = now
        if self._on_spoken is not None:
            try:
                self._on_spoken()
            except Exception:
                log.exception("on_spoken failed")
        log.debug("initiative line %s", candidate.name)

    # -- bus ----------------------------------------------------------------------------------

    def _on_app_started(self, _event: AppStarted) -> None:
        now = self._clock.monotonic()
        if self._started_at is None:
            self._started_at = now
        # The greeting just played: the first initiative waits a full gap.
        self._last_initiative = now

    def _on_quiet(self, event: QuietModeChanged) -> None:
        if not event.enabled:
            self._quiet, self._quiet_deadline = False, None
            return
        self._quiet = True
        if event.until is None:
            self._quiet_deadline = None
        else:
            remaining = max(0.0, event.until - self._clock.now())
            self._quiet_deadline = self._clock.monotonic() + remaining

    def _on_dnd(self, event: DoNotDisturbChanged) -> None:
        self._dnd_override = event.enabled

    def _on_settings_changed(self, event: SettingsChanged) -> None:
        if event.section in ("", "behavior"):
            self._dnd_override = None
