"""Emotion engine: eight 0..100 values that colour how she speaks and moves.

Two kinds of values, one owner each - never copies:

* ``hunger``, ``energy``, ``happiness``, ``boredom`` and ``affection`` *are*
  the tamagotchi's needs.  They are read from the ``needs()`` snapshot every
  time; the tamagotchi stays the single source of truth.
* ``curiosity``, ``excitement`` and ``trust`` belong to this engine and are
  persisted in :class:`~chibi.memory.repositories.StateRepository` under the
  key ``"emotion"`` (``{curiosity, excitement, trust, last_tick}`` plus the
  daily petting-trust tally).

How the owned values move - deterministic rules, no randomness, so behaviour
is explainable and testable to the decimal:

* **Time** (:meth:`EmotionEngine.tick`): excitement relaxes toward 20 with a
  5-minute half-life, curiosity toward the personality "curiosity" slider with
  a 20-minute half-life (exponential relaxation, so any long gap simply lands
  on the baseline).  Trust never decays.  The elapsed time is the larger of
  the monotonic and the wall-clock delta (the monotonic clock stops during
  system sleep on some OSes) and is capped; after the app was closed, the
  absence is measured with the wall clock from ``last_tick``.  Absence never
  punishes: relaxation only pulls values to their resting point and trust is
  untouched, so a week away finds her exactly as calm and as trusting as she
  was.
* **Events** - see the ``*_DELTA`` constants below.  Each change is clamped to
  0..100.  Trust from petting is capped per local calendar day (and the tally
  is persisted, so restarting the app cannot farm it).

:meth:`EmotionEngine.dominant` maps the values to a
:class:`~chibi.character.states.CharacterState` value with ordered rules, and
:meth:`EmotionEngine.describe` gives a short Ukrainian mood line for prompts
(operator-side context: built only from numbers, never from user text).

:class:`~chibi.core.events.EmotionChanged` is published after :meth:`load`
and whenever a value moved at least :data:`PUBLISH_THRESHOLD` points since
the previous publish (or the dominant state changed).  State is written at
most every :data:`SAVE_INTERVAL_SECONDS` and on :meth:`save` (shutdown).

Privacy: foreground-app events are reduced to their *category*, kept in
memory only (30-minute "seen" times) and forgotten by the kill switch; private
apps are ignored entirely (as are the meaningless idle / unknown categories)
and nothing about apps is persisted.

Main (Qt) thread only, like the repository and the bus handlers.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from chibi.core.clock import Clock
from chibi.core.config import Settings
from chibi.core.event_bus import EventBus
from chibi.core.events import (
    ActivityCategory,
    ActivityMilestone,
    EmotionChanged,
    ForegroundAppChanged,
    GameEnded,
    GameStarted,
    InteractionKind,
    InteractionOutcome,
    NeedsChanged,
    ObservationStopped,
    UserReturned,
)
from chibi.memory.repositories import StateRepository
from chibi.tamagotchi.needs import Needs

log = logging.getLogger(__name__)

EMOTIONS: tuple[str, ...] = (
    "happiness", "energy", "boredom", "affection", "curiosity", "excitement", "hunger", "trust",
)
#: Values read from the tamagotchi's needs.
NEED_EMOTIONS: tuple[str, ...] = ("happiness", "energy", "boredom", "affection", "hunger")
#: Values owned (and persisted) by the engine.
OWN_EMOTIONS: tuple[str, ...] = ("curiosity", "excitement", "trust")

STATE_KEY = "emotion"

VALUE_MIN = 0.0
VALUE_MAX = 100.0

EXCITEMENT_BASELINE = 20.0
EXCITEMENT_HALF_LIFE_SECONDS = 5 * 60.0
CURIOSITY_HALF_LIFE_SECONDS = 20 * 60.0
#: Used when the settings cannot be read.
DEFAULT_CURIOSITY = 60.0
DEFAULT_EXCITEMENT = EXCITEMENT_BASELINE
DEFAULT_TRUST = 40.0

#: Relaxation has fully converged long before this; longer gaps are cut so a
#: bogus clock value can never produce inf / NaN maths.
MAX_DECAY_SECONDS = 7 * 24 * 3600.0
PUBLISH_THRESHOLD = 3.0
SAVE_INTERVAL_SECONDS = 60.0
#: Only owned-value moves larger than this make a periodic save worthwhile.
SAVE_EPSILON = 0.01

# -- event rules -----------------------------------------------------------------------------
GAME_START_EXCITEMENT_DELTA = 25.0
GAME_START_CURIOSITY_DELTA = 10.0
#: A game session at least this long ends with a little extra excitement.
LONG_GAME_SECONDS = 30 * 60.0
GAME_END_LONG_EXCITEMENT_DELTA = 5.0
PET_HAPPY_TRUST_DELTA = 0.5
#: Most trust petting can add per local calendar day.
PET_TRUST_DAILY_CAP = 5.0
FEED_FED_TRUST_DELTA = 0.3
CLICK_ANNOYED_EXCITEMENT_DELTA = -5.0
TALK_OK_CURIOSITY_DELTA = 3.0
USER_RETURNED_EXCITEMENT_DELTA = 15.0
USER_RETURNED_TRUST_DELTA = 0.2
#: Switching to a kind of activity not seen for this long makes her curious.
NEW_CATEGORY_WINDOW_SECONDS = 30 * 60.0
NEW_CATEGORY_CURIOSITY_DELTA = 4.0
#: The same activity for a long time is routine.
MILESTONE_CURIOSITY_DELTA = -5.0
#: Categories that say nothing about what the user does (or must not be looked at).
_IGNORED_CATEGORIES = frozenset({ActivityCategory.PRIVATE, ActivityCategory.IDLE, ActivityCategory.UNKNOWN})

# -- dominant state (checked in order, the first match wins) --------------------------------
DOMINANT_SLEEPY_ENERGY_BELOW = 20.0
DOMINANT_STARVING_HUNGER_ABOVE = 80.0
DOMINANT_EXCITED_ABOVE = 75.0
DOMINANT_SAD_HAPPINESS_BELOW = 30.0
DOMINANT_CURIOUS_ABOVE = 80.0
DOMINANT_BORED_ABOVE = 80.0
DOMINANT_SHY_HAPPINESS_ABOVE = 80.0
DOMINANT_SHY_AFFECTION_ABOVE = 70.0

#: describe(): at most this many mood words.
DESCRIBE_MAX_PARTS = 4
DESCRIBE_CALM = "спокійна"

#: Per emotion: the "neutral" point and bands (comparison, threshold, Ukrainian
#: words, feminine).  The first band that matches wins; salience is the distance
#: from neutral, so the strongest feelings are named first.
_DESCRIPTORS: dict[str, tuple[float, tuple[tuple[str, float, str], ...]]] = {
    "happiness": (55.0, ((">=", 85.0, "дуже щаслива"), (">=", 70.0, "весела"),
                         ("<=", 15.0, "зовсім сумна"), ("<=", 30.0, "сумна"),
                         ("<=", 42.0, "трохи засмучена"))),
    "excitement": (20.0, ((">=", 75.0, "у захваті"), (">=", 50.0, "піднесена"),
                          (">=", 35.0, "жвава"))),
    "energy": (60.0, (("<=", 20.0, "дуже сонна"), ("<=", 35.0, "трохи втомлена"),
                      (">=", 90.0, "сповнена сил"), (">=", 78.0, "бадьора"))),
    "curiosity": (50.0, ((">=", 80.0, "дуже цікаво"), (">=", 65.0, "цікаво"),
                         ("<=", 15.0, "байдужа"))),
    "boredom": (30.0, ((">=", 80.0, "дуже нудно"), (">=", 60.0, "нудно"), (">=", 45.0, "трохи нудно"))),
    "hunger": (30.0, ((">=", 80.0, "дуже голодна"), (">=", 60.0, "голодна"), (">=", 45.0, "трохи голодна"),
                      ("<=", 8.0, "сита"))),
    "affection": (50.0, ((">=", 85.0, "дуже ніжна"), (">=", 70.0, "лагідна"),
                         ("<=", 20.0, "трохи відсторонена"))),
    "trust": (40.0, ((">=", 85.0, "повністю довіряє"), (">=", 65.0, "довіряє"), ("<=", 15.0, "насторожена"))),
}
#: Reading order of the description (mood first, needs later).
_DESCRIBE_ORDER: tuple[str, ...] = (
    "happiness", "excitement", "energy", "curiosity", "boredom", "hunger", "affection", "trust",
)


def _clamp(value: float) -> float:
    return min(max(value, VALUE_MIN), VALUE_MAX)


def _finite(value: Any) -> float | None:
    """A finite float from stored JSON, or None (booleans count as corrupt)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def relax(value: float, target: float, half_life: float, seconds: float) -> float:
    """Exponential relaxation of *value* toward *target*: after one *half_life*
    half of the distance is left.  Composes exactly (relaxing by a then b ==
    relaxing by a + b), so tick frequency does not change the result."""
    if seconds <= 0.0 or half_life <= 0.0:
        return value
    return target + (value - target) * 0.5 ** (seconds / half_life)


def _local_day(epoch_seconds: float) -> str:
    try:
        return datetime.fromtimestamp(epoch_seconds).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


class EmotionEngine:
    """See the module docstring.  ``needs`` returns the tamagotchi's current
    :class:`~chibi.tamagotchi.needs.Needs` (any object with the five need
    attributes, or a mapping of them, works)."""

    def __init__(self, bus: EventBus, repo: StateRepository, clock: Clock,
                 needs: Callable[[], Needs], settings: Callable[[], Settings]) -> None:
        self._bus = bus
        self._repo = repo
        self._clock = clock
        self._needs = needs
        self._settings = settings
        self._curiosity = self._curiosity_baseline()
        self._excitement = DEFAULT_EXCITEMENT
        self._trust = DEFAULT_TRUST
        self._pet_trust_day = ""
        self._pet_trust_today = 0.0
        self._last_mono = clock.monotonic()
        self._last_wall = clock.now()
        self._last_save_mono = self._last_mono
        self._saved: dict[str, Any] | None = None
        self._loaded = False
        self._last_published: dict[str, float] | None = None
        self._last_dominant: str | None = None
        self._last_needs: dict[str, float] = Needs().to_dict()
        #: activity category -> monotonic time it was last in the foreground (memory only).
        self._category_seen: dict[ActivityCategory, float] = {}
        self._current_category: ActivityCategory | None = None
        self._unsubscribers: list[Callable[[], None]] = [
            bus.subscribe(GameStarted, self._on_game_started),
            bus.subscribe(GameEnded, self._on_game_ended),
            bus.subscribe(InteractionOutcome, self._on_interaction_outcome),
            bus.subscribe(UserReturned, self._on_user_returned),
            bus.subscribe(ForegroundAppChanged, self._on_foreground_changed),
            bus.subscribe(ActivityMilestone, self._on_activity_milestone),
            bus.subscribe(NeedsChanged, self._on_needs_changed),
            bus.subscribe(ObservationStopped, self._on_observation_stopped),
        ]

    # -- public API -----------------------------------------------------------------------

    def values(self) -> dict[str, float]:
        """The eight values (0..100) as of now, in :data:`EMOTIONS` order."""
        self._ensure_loaded()
        self._advance()
        return self._snapshot()

    def dominant(self) -> str:
        """The :class:`~chibi.character.states.CharacterState` value that best
        expresses the current mood."""
        return dominant_state(self.values())

    def describe(self) -> str:
        """Short natural Ukrainian mood line, e.g. ``"весела, дуже цікаво,
        трохи голодна"`` - for the prompt context."""
        return describe_values(self.values())

    def tick(self) -> None:
        """Advance time (decay), publish on a meaningful change, persist at most
        every :data:`SAVE_INTERVAL_SECONDS`.  Called periodically by the app."""
        self._ensure_loaded()
        self._advance()
        self._maybe_publish()
        if self._clock.monotonic() - self._last_save_mono >= SAVE_INTERVAL_SECONDS:
            self._persist(force=False)

    def load(self) -> None:
        """(Re)load the owned values and catch up on the time the app was
        closed.  Missing or corrupt state means defaults (also used after
        "delete all data").  Publishes one EmotionChanged."""
        stored = self._repo_get()
        data: Mapping[str, Any] = stored if isinstance(stored, Mapping) else {}
        if stored is not None and not isinstance(stored, Mapping):
            log.warning("stored emotion state is not an object; using defaults")
        baseline = self._curiosity_baseline()
        self._curiosity = self._stored_value(data, "curiosity", baseline)
        self._excitement = self._stored_value(data, "excitement", DEFAULT_EXCITEMENT)
        self._trust = self._stored_value(data, "trust", DEFAULT_TRUST)
        day = data.get("pet_trust_day")
        tally = _finite(data.get("pet_trust_today"))
        self._pet_trust_day = day if isinstance(day, str) else ""
        self._pet_trust_today = min(max(tally or 0.0, 0.0), PET_TRUST_DAILY_CAP)
        now = self._clock.now()
        last_tick = _finite(data.get("last_tick"))
        if last_tick is not None:
            away = now - last_tick
            if away > 0:
                self._relax(min(away, MAX_DECAY_SECONDS))
                log.info("emotions caught up on %.0f s away", away)
        self._category_seen.clear()
        self._current_category = None
        self._last_mono = self._clock.monotonic()
        self._last_wall = now
        self._last_save_mono = self._last_mono
        self._saved = None
        self._loaded = True
        self._maybe_publish(force=True)

    def save(self) -> None:
        """Persist now (shutdown).  Never raises."""
        self._ensure_loaded()
        self._advance()
        self._persist(force=True)

    def close(self) -> None:
        """Stop listening to the bus (the app saves separately on shutdown)."""
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()

    # -- time -------------------------------------------------------------------------------

    def _advance(self) -> None:
        mono = self._clock.monotonic()
        wall = self._clock.now()
        elapsed = max(mono - self._last_mono, wall - self._last_wall, 0.0)
        self._last_mono = mono
        self._last_wall = wall
        if elapsed > 0.0:
            self._relax(min(elapsed, MAX_DECAY_SECONDS))

    def _relax(self, seconds: float) -> None:
        self._excitement = _clamp(relax(self._excitement, EXCITEMENT_BASELINE,
                                        EXCITEMENT_HALF_LIFE_SECONDS, seconds))
        self._curiosity = _clamp(relax(self._curiosity, self._curiosity_baseline(),
                                       CURIOSITY_HALF_LIFE_SECONDS, seconds))

    # -- events -----------------------------------------------------------------------------

    def _apply(self, *, excitement: float = 0.0, curiosity: float = 0.0, trust: float = 0.0) -> None:
        """Settle elapsed time first (a fresh bump must not decay for time that
        passed before it), then apply the deltas and publish if it shows."""
        self._ensure_loaded()
        self._advance()
        self._excitement = _clamp(self._excitement + excitement)
        self._curiosity = _clamp(self._curiosity + curiosity)
        self._trust = _clamp(self._trust + trust)
        self._maybe_publish()

    def _on_game_started(self, _event: GameStarted) -> None:
        self._apply(excitement=GAME_START_EXCITEMENT_DELTA, curiosity=GAME_START_CURIOSITY_DELTA)

    def _on_game_ended(self, event: GameEnded) -> None:
        if event.forgotten:
            return  # the user excluded the game: forget the session entirely
        duration = _finite(event.duration_seconds) or 0.0
        if duration >= LONG_GAME_SECONDS:
            self._apply(excitement=GAME_END_LONG_EXCITEMENT_DELTA)

    def _on_interaction_outcome(self, event: InteractionOutcome) -> None:
        self._ensure_loaded()  # before the pet tally is touched: load() replaces it
        # StrEnum members equal their values, so a plain-string kind works too.
        rule = (str(event.kind), event.outcome)
        if rule == (InteractionKind.PET.value, "happy"):
            self._apply(trust=self._pet_trust_allowance())
        elif rule == (InteractionKind.FEED.value, "fed"):
            self._apply(trust=FEED_FED_TRUST_DELTA)
        elif rule == (InteractionKind.CLICK.value, "annoyed"):
            self._apply(excitement=CLICK_ANNOYED_EXCITEMENT_DELTA)
        elif rule == (InteractionKind.TALK.value, "ok"):
            self._apply(curiosity=TALK_OK_CURIOSITY_DELTA)

    def _pet_trust_allowance(self) -> float:
        """Trust one happy pet may add today (0 once the daily cap is reached)."""
        today = _local_day(self._clock.now())
        if today != self._pet_trust_day:
            self._pet_trust_day = today
            self._pet_trust_today = 0.0
        allowance = max(0.0, min(PET_HAPPY_TRUST_DELTA, PET_TRUST_DAILY_CAP - self._pet_trust_today))
        self._pet_trust_today += allowance
        return allowance

    def _on_user_returned(self, _event: UserReturned) -> None:
        self._apply(excitement=USER_RETURNED_EXCITEMENT_DELTA, trust=USER_RETURNED_TRUST_DELTA)

    def _on_foreground_changed(self, event: ForegroundAppChanged) -> None:
        self._ensure_loaded()  # before the category history is touched: load() resets it
        now = self._clock.monotonic()
        previous = self._current_category
        if previous is not None:
            # It was on screen until this very moment.
            self._category_seen[previous] = now
        category = event.category
        if event.app.private or category in _IGNORED_CATEGORIES:
            self._current_category = None
            return
        last_seen = self._category_seen.get(category)
        self._current_category = category
        self._category_seen[category] = now
        if category == previous:
            return
        if last_seen is None or now - last_seen >= NEW_CATEGORY_WINDOW_SECONDS:
            self._apply(curiosity=NEW_CATEGORY_CURIOSITY_DELTA)

    def _on_activity_milestone(self, event: ActivityMilestone) -> None:
        if event.app.private or event.category in _IGNORED_CATEGORIES:
            return
        self._apply(curiosity=MILESTONE_CURIOSITY_DELTA)

    def _on_needs_changed(self, _event: NeedsChanged) -> None:
        # The needs callable is the source of truth; the event is only a nudge.
        self._ensure_loaded()
        self._advance()
        self._maybe_publish()

    def _on_observation_stopped(self, _event: ObservationStopped) -> None:
        # Kill switch: forget even the in-memory activity history.
        self._category_seen.clear()
        self._current_category = None

    # -- snapshot / publishing ----------------------------------------------------------------

    def _snapshot(self) -> dict[str, float]:
        needs = self._read_needs()
        own = {"curiosity": self._curiosity, "excitement": self._excitement, "trust": self._trust}
        return {name: needs[name] if name in needs else own[name] for name in EMOTIONS}

    def _read_needs(self) -> dict[str, float]:
        try:
            source = self._needs()
        except Exception:
            log.exception("needs snapshot unavailable; using the last one")
            return dict(self._last_needs)
        values: dict[str, float] = {}
        for name in NEED_EMOTIONS:
            raw = source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
            number = _finite(raw)
            values[name] = _clamp(number) if number is not None else self._last_needs[name]
        self._last_needs = values
        return dict(values)

    def _maybe_publish(self, *, force: bool = False) -> None:
        values = self._snapshot()
        dominant = dominant_state(values)
        if not force and self._last_published is not None:
            moved = max(abs(values[name] - self._last_published[name]) for name in EMOTIONS)
            # The epsilon keeps ten +0.3 steps (2.9999999999999996 in floats) counting as 3.
            if moved < PUBLISH_THRESHOLD - 1e-9 and dominant == self._last_dominant:
                return
        self._bus.publish(EmotionChanged(
            **{name: round(values[name], 1) for name in EMOTIONS},
            dominant=dominant,
        ))
        self._last_published = values
        self._last_dominant = dominant

    # -- persistence ------------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _state(self) -> dict[str, Any]:
        return {
            "curiosity": round(self._curiosity, 3),
            "excitement": round(self._excitement, 3),
            "trust": round(self._trust, 3),
            "pet_trust_day": self._pet_trust_day,
            "pet_trust_today": round(self._pet_trust_today, 3),
        }

    def _changed_since_save(self, state: dict[str, Any]) -> bool:
        saved = self._saved
        if saved is None:
            return True
        if (saved["pet_trust_day"], saved["pet_trust_today"]) != (
                state["pet_trust_day"], state["pet_trust_today"]):
            return True
        return any(abs(state[name] - saved[name]) > SAVE_EPSILON for name in OWN_EMOTIONS)

    def _persist(self, *, force: bool) -> None:
        self._last_save_mono = self._clock.monotonic()
        state = self._state()
        if not force and not self._changed_since_save(state):
            return
        try:
            self._repo.set(STATE_KEY, {**state, "last_tick": self._clock.now()})
        except Exception:
            log.exception("could not save emotion state")
            return
        self._saved = state

    def _repo_get(self) -> Any:
        try:
            return self._repo.get(STATE_KEY)
        except Exception:
            log.exception("could not read emotion state; using defaults")
            return None

    def _curiosity_baseline(self) -> float:
        try:
            value = _finite(self._settings().personality.curiosity)
        except Exception:
            log.exception("settings unavailable; default curiosity baseline")
            return DEFAULT_CURIOSITY
        return _clamp(value) if value is not None else DEFAULT_CURIOSITY

    @staticmethod
    def _stored_value(data: Mapping[str, Any], name: str, default: float) -> float:
        number = _finite(data.get(name))
        if number is None:
            if name in data:
                log.warning("ignoring invalid stored emotion %s", name)
            return default
        return _clamp(number)


# -- pure helpers (also used by tests and the decision engine) ------------------------------


def dominant_state(values: Mapping[str, float]) -> str:
    """Ordered rules -> CharacterState value; the first match wins."""
    if values["energy"] < DOMINANT_SLEEPY_ENERGY_BELOW:
        return "yawn"
    if values["hunger"] > DOMINANT_STARVING_HUNGER_ABOVE:
        return "sad"
    if values["excitement"] > DOMINANT_EXCITED_ABOVE:
        return "happy"
    if values["happiness"] < DOMINANT_SAD_HAPPINESS_BELOW:
        return "sad"
    if values["curiosity"] > DOMINANT_CURIOUS_ABOVE:
        return "curious"
    if values["boredom"] > DOMINANT_BORED_ABOVE:
        return "thinking"
    if (values["happiness"] > DOMINANT_SHY_HAPPINESS_ABOVE
            and values["affection"] > DOMINANT_SHY_AFFECTION_ABOVE):
        return "shy"
    return "idle"


def describe_values(values: Mapping[str, float]) -> str:
    """Up to :data:`DESCRIBE_MAX_PARTS` Ukrainian mood words for the strongest
    feelings, in a natural reading order; ``"спокійна"`` when nothing stands out."""
    found: list[tuple[float, str, str]] = []
    for name, (neutral, bands) in _DESCRIPTORS.items():
        value = values[name]
        for comparison, threshold, words in bands:
            if (value >= threshold) if comparison == ">=" else (value <= threshold):
                found.append((abs(value - neutral), name, words))
                break
    if not found:
        return DESCRIBE_CALM
    strongest = sorted(found, key=lambda item: -item[0])[:DESCRIBE_MAX_PARTS]
    chosen = {name: words for _salience, name, words in strongest}
    return ", ".join(chosen[name] for name in _DESCRIBE_ORDER if name in chosen)
