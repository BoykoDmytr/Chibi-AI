"""Tamagotchi needs: the numbers behind her mood, and the rules that move them.

Everything here is pure.  :class:`NeedsModel` takes a :class:`Needs` snapshot
plus elapsed time or interaction context and returns a *new* snapshot; there is
no clock inside, so every rule is testable to the decimal with plain numbers
and :class:`~chibi.tamagotchi.service.TamagotchiService` alone decides when
time passes.

Design rule: she never punishes the user for being away.  While the app is
closed (or the laptop sleeps) she "sleeps" too:
:meth:`NeedsModel.offline_catch_up` recovers her energy and only lets hunger
and boredom creep up to mild caps, so coming back after a week finds her a bit
peckish and glad to see you - never starving, bored stiff or miserable.
Friendship XP only ever grows; no rule here subtracts from it.

Scales: every need is 0..100.  ``hunger`` 0 = full, 100 = starving; the other
needs read the natural way round (``energy`` 100 = rested, ``boredom``
100 = bored stiff).
"""

from __future__ import annotations

import bisect
import dataclasses
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from chibi.core.events import InteractionKind

log = logging.getLogger(__name__)

STAT_MIN = 0.0
STAT_MAX = 100.0
#: XP has no meaningful upper bound, but it must stay finite so a corrupt or
#: absurd stored value can never turn into ``inf`` (which JSON cannot store).
MAX_FRIENDSHIP_XP = 1e9

DEFAULT_HUNGER = 30.0
DEFAULT_ENERGY = 80.0
DEFAULT_HAPPINESS = 70.0
DEFAULT_BOREDOM = 20.0
DEFAULT_AFFECTION = 50.0
DEFAULT_FRIENDSHIP_XP = 0.0

#: XP needed for friendship levels 1..10.
FRIENDSHIP_THRESHOLDS: tuple[float, ...] = (0, 10, 30, 60, 100, 160, 240, 340, 460, 600)
MAX_FRIENDSHIP_LEVEL = len(FRIENDSHIP_THRESHOLDS)

# --- while the app runs (per hour) ---------------------------------------------------
HUNGER_PER_HOUR_AWAKE = 7.0
HUNGER_PER_HOUR_ASLEEP = 3.0
ENERGY_PER_HOUR_AWAKE = -6.0
ENERGY_PER_HOUR_ASLEEP = 25.0
#: Alone on the desktop she gets bored fast; with the user around (an
#: interaction or an app switch recently) only slowly.
BOREDOM_PER_HOUR_IDLE = 12.0
BOREDOM_PER_HOUR_USER_ACTIVE = 4.0
BOREDOM_PER_HOUR_ASLEEP = -10.0
HAPPINESS_DRIFT_PER_HOUR = 10.0
AFFECTION_DRIFT_PER_HOUR = 1.0
AFFECTION_BASELINE = 50.0
#: Drift alone never pushes affection below this: neglect while the app runs
#: may cool her down, but forgetting to pet her must not make her dislike you.
AFFECTION_DRIFT_FLOOR = 30.0

# happiness target: base, penalties for unmet needs, bonus/penalty for affection
HAPPINESS_BASE = 70.0
HUNGER_PENALTY_ABOVE = 50.0
HUNGER_PENALTY_PER_POINT = 0.4
BOREDOM_PENALTY_ABOVE = 50.0
BOREDOM_PENALTY_PER_POINT = 0.3
ENERGY_PENALTY_BELOW = 30.0
ENERGY_PENALTY_PER_POINT = 0.3
AFFECTION_WEIGHT = 0.1

#: ``tick`` integrates in steps no longer than this, so a long ``dt`` gives the
#: same result as many short ticks (happiness drifts toward a target that
#: itself moves as hunger and energy change).
TICK_STEP_SECONDS = 300.0
#: Every need has converged long before a week of continuous ticking (the
#: slowest rule, affection drift, needs at most 50 h), so longer ``dt`` values
#: are cut here instead of looping for ages over a bogus duration.
MAX_TICK_SECONDS = 7 * 24 * 3600.0

# --- while the app was closed (per hour) -----------------------------------------------
OFFLINE_ENERGY_PER_HOUR = 25.0
OFFLINE_HUNGER_PER_HOUR = 2.0
#: Absence alone never makes her hungrier than this (or than she already was).
OFFLINE_HUNGER_CAP = 55.0
OFFLINE_BOREDOM_PER_HOUR = 3.0
OFFLINE_BOREDOM_CAP = 50.0
#: Absence alone never makes her sadder than this (or than she already was).
OFFLINE_HAPPINESS_FLOOR = 50.0
#: All offline caps are reached within ~28 h; a month is "forever".
MAX_AWAY_SECONDS = 30 * 24 * 3600.0

# --- interactions ----------------------------------------------------------------------
FEED_FULL_BELOW = 15.0
FEED_STARVING_ABOVE = 75.0
FEED_HUNGER_RELIEF = 35.0
FEED_STARVING_HUNGER_RELIEF = 45.0
FEED_HAPPINESS = 6.0
FEED_ENERGY = 3.0
FEED_XP = 2.0

#: Pets in the last 10 minutes (before this one) that make another one "too much".
PET_TOO_MUCH_AT = 6
PET_AFFECTION = 4.0
PET_HAPPINESS = 4.0
PET_BOREDOM = -8.0
PET_XP = 3.0

REST_NOT_TIRED_ABOVE = 80.0
WAKE_GRUMPY_BELOW = 30.0

#: Clicks in the last 5 seconds (before this one) that make another one annoying.
CLICK_ANNOYED_AT = 5
CLICK_ANNOYED_HAPPINESS = -1.0
CLICK_BOREDOM = -3.0
CLICK_XP = 0.5

TALK_BOREDOM = -5.0
TALK_XP = 1.0
CALL_BOREDOM = -2.0

#: First start of the app on a calendar day.
DAILY_START_XP = 5.0

# --- mood (checked in this order; the first match wins) --------------------------------
MOOD_SLEEPY_ENERGY_BELOW = 20.0
MOOD_HUNGRY_ABOVE = 75.0
MOOD_BORED_ABOVE = 75.0
MOOD_SAD_BELOW = 30.0
MOOD_VERY_HAPPY_ABOVE = 80.0
MOOD_CHEERFUL_ABOVE = 60.0

#: field name -> (default, lower bound, upper bound)
_LIMITS: dict[str, tuple[float, float, float]] = {
    "hunger": (DEFAULT_HUNGER, STAT_MIN, STAT_MAX),
    "energy": (DEFAULT_ENERGY, STAT_MIN, STAT_MAX),
    "happiness": (DEFAULT_HAPPINESS, STAT_MIN, STAT_MAX),
    "boredom": (DEFAULT_BOREDOM, STAT_MIN, STAT_MAX),
    "affection": (DEFAULT_AFFECTION, STAT_MIN, STAT_MAX),
    "friendship_xp": (DEFAULT_FRIENDSHIP_XP, 0.0, MAX_FRIENDSHIP_XP),
}

#: The 0..100 needs (everything except XP), e.g. for change detection.
STAT_NAMES: tuple[str, ...] = ("hunger", "energy", "happiness", "boredom", "affection")


def _clamp_field(name: str, value: Any) -> float:
    default, lo, hi = _LIMITS[name]
    number = float(value)
    if math.isnan(number):
        return default
    return min(max(number, lo), hi)


def finite_number(value: Any) -> float | None:
    """``value`` (typically read back from JSON) as a finite float, or None if it is not a usable number.

    Booleans are rejected on purpose: ``True`` is an ``int`` in Python, but a
    stored ``true`` where a number belongs means the data is corrupt.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:  # an int too large for a float
        return None
    return number if math.isfinite(number) else None


def _duration(seconds: Any, cap: float) -> float:
    """A usable elapsed time: NaN, negative or non-numeric -> 0; ``inf`` and huge values -> ``cap``."""
    try:
        value = float(seconds)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if math.isnan(value) or value <= 0.0:
        return 0.0
    return min(value, cap)


def _drift(value: float, target: float, max_step: float) -> float:
    """Move ``value`` toward ``target`` by at most ``max_step`` without overshooting."""
    if value < target:
        return min(value + max_step, target)
    return max(value - max_step, target)


@dataclass
class Needs:
    """One snapshot of her needs.

    Every assignment is clamped (needs to 0..100, XP to >= 0) and NaN falls
    back to the field's default, so no rule, corrupt file or caller can ever
    produce an out-of-range value.  :class:`NeedsModel` never mutates a
    ``Needs`` it is given; it returns a new one.
    """

    hunger: float = DEFAULT_HUNGER
    energy: float = DEFAULT_ENERGY
    happiness: float = DEFAULT_HAPPINESS
    boredom: float = DEFAULT_BOREDOM
    affection: float = DEFAULT_AFFECTION
    friendship_xp: float = DEFAULT_FRIENDSHIP_XP

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _LIMITS:
            value = _clamp_field(name, value)
        object.__setattr__(self, name, value)

    @property
    def friendship_level(self) -> int:
        """1..10: how many XP thresholds she has passed."""
        return bisect.bisect_right(FRIENDSHIP_THRESHOLDS, self.friendship_xp)

    @property
    def friendship_progress(self) -> float:
        """0..1 progress toward the next friendship level (1.0 at the maximum level)."""
        level = self.friendship_level
        if level >= MAX_FRIENDSHIP_LEVEL:
            return 1.0
        lo = FRIENDSHIP_THRESHOLDS[level - 1]
        hi = FRIENDSHIP_THRESHOLDS[level]
        return min(max((self.friendship_xp - lo) / (hi - lo), 0.0), 1.0)

    def to_dict(self) -> dict[str, float]:
        """JSON-friendly snapshot (also usable as a ``Mapping[str, float]`` view of the needs)."""
        return {name: float(getattr(self, name)) for name in _LIMITS}

    @classmethod
    def from_dict(cls, data: Any) -> Needs:
        """Tolerant inverse of :meth:`to_dict`.

        Stored state may be old, hand-edited or corrupt; none of that may
        crash the app or produce an impossible value.  Missing fields and
        fields that are not finite numbers take their default, out-of-range
        numbers are clamped, unknown keys are ignored.
        """
        if not isinstance(data, Mapping):
            if data is not None:
                log.warning("stored needs are not an object; using defaults")
            return cls()
        values: dict[str, float] = {}
        invalid: list[str] = []
        for name in _LIMITS:
            if name not in data:
                continue
            number = finite_number(data[name])
            if number is None:
                invalid.append(name)
            else:
                values[name] = number
        if invalid:
            log.warning("ignoring invalid stored needs: %s", ", ".join(invalid))
        return cls(**values)


class NeedsModel:
    """The rules.  Pure functions over :class:`Needs`; no clock, no state."""

    # -- time ---------------------------------------------------------------------------

    def tick(self, needs: Needs, dt: float, *, sleeping: bool, user_active: bool) -> Needs:
        """Advance ``needs`` by ``dt`` seconds of the app running.

        ``user_active``: the user interacted or switched apps recently, which
        keeps boredom low.  It is ignored while ``sleeping``.  Non-positive or
        NaN ``dt`` changes nothing.
        """
        seconds = _duration(dt, MAX_TICK_SECONDS)
        result = dataclasses.replace(needs)
        if seconds <= 0.0:
            return result
        steps = max(1, math.ceil(seconds / TICK_STEP_SECONDS))
        hours = seconds / steps / 3600.0
        for _ in range(steps):
            result = self._step(result, hours, sleeping=sleeping, user_active=user_active)
        return result

    def _step(self, needs: Needs, hours: float, *, sleeping: bool, user_active: bool) -> Needs:
        target = self.happiness_target(needs)
        if sleeping:
            hunger_rate, energy_rate, boredom_rate = (
                HUNGER_PER_HOUR_ASLEEP, ENERGY_PER_HOUR_ASLEEP, BOREDOM_PER_HOUR_ASLEEP)
        else:
            hunger_rate, energy_rate = HUNGER_PER_HOUR_AWAKE, ENERGY_PER_HOUR_AWAKE
            boredom_rate = BOREDOM_PER_HOUR_USER_ACTIVE if user_active else BOREDOM_PER_HOUR_IDLE
        affection = _drift(needs.affection, AFFECTION_BASELINE, AFFECTION_DRIFT_PER_HOUR * hours)
        # Drift may cool affection down, never below the floor (unless it already was).
        affection = max(affection, min(needs.affection, AFFECTION_DRIFT_FLOOR))
        return Needs(
            hunger=needs.hunger + hunger_rate * hours,
            energy=needs.energy + energy_rate * hours,
            happiness=_drift(needs.happiness, target, HAPPINESS_DRIFT_PER_HOUR * hours),
            boredom=needs.boredom + boredom_rate * hours,
            affection=affection,
            friendship_xp=needs.friendship_xp,
        )

    def happiness_target(self, needs: Needs) -> float:
        """Where happiness settles if nothing else happens.

        70 when every need is met; hunger above 50, boredom above 50 and
        energy below 30 pull it down, affection above / below 50 nudges it.
        """
        target = (
            HAPPINESS_BASE
            - HUNGER_PENALTY_PER_POINT * max(0.0, needs.hunger - HUNGER_PENALTY_ABOVE)
            - BOREDOM_PENALTY_PER_POINT * max(0.0, needs.boredom - BOREDOM_PENALTY_ABOVE)
            - ENERGY_PENALTY_PER_POINT * max(0.0, ENERGY_PENALTY_BELOW - needs.energy)
            + AFFECTION_WEIGHT * (needs.affection - AFFECTION_BASELINE)
        )
        return min(max(target, STAT_MIN), STAT_MAX)

    def offline_catch_up(self, needs: Needs, away_seconds: float) -> Needs:
        """She slept while the app was closed (or the computer was asleep).

        Energy recovers; hunger and boredom rise slowly and only up to mild
        caps (never above what they already were, if that was higher);
        happiness drifts toward its target but absence alone never pushes it
        below ``min(current, 50)``; affection and XP are untouched.
        Negative or NaN ``away_seconds`` changes nothing.
        """
        away = _duration(away_seconds, MAX_AWAY_SECONDS)
        if away <= 0.0:
            return dataclasses.replace(needs)
        hours = away / 3600.0
        rested = dataclasses.replace(
            needs,
            energy=needs.energy + OFFLINE_ENERGY_PER_HOUR * hours,
            hunger=min(needs.hunger + OFFLINE_HUNGER_PER_HOUR * hours,
                       max(needs.hunger, OFFLINE_HUNGER_CAP)),
            boredom=min(needs.boredom + OFFLINE_BOREDOM_PER_HOUR * hours,
                        max(needs.boredom, OFFLINE_BOREDOM_CAP)),
        )
        happiness = _drift(needs.happiness, self.happiness_target(rested), HAPPINESS_DRIFT_PER_HOUR * hours)
        rested.happiness = max(happiness, min(needs.happiness, OFFLINE_HAPPINESS_FLOOR))
        return rested

    def daily_bonus(self, needs: Needs) -> Needs:
        """The user came back on a new calendar day: a little friendship XP."""
        return dataclasses.replace(needs, friendship_xp=needs.friendship_xp + DAILY_START_XP)

    # -- interactions (outcome strings: see chibi.core.events.InteractionOutcome) ----------

    def feed(self, needs: Needs) -> tuple[Needs, str]:
        if needs.hunger < FEED_FULL_BELOW:
            return dataclasses.replace(needs), "full"
        if needs.hunger > FEED_STARVING_ABOVE:
            relief, outcome = FEED_STARVING_HUNGER_RELIEF, "starving_fed"
        else:
            relief, outcome = FEED_HUNGER_RELIEF, "fed"
        fed = dataclasses.replace(
            needs,
            hunger=needs.hunger - relief,
            happiness=needs.happiness + FEED_HAPPINESS,
            energy=needs.energy + FEED_ENERGY,
            friendship_xp=needs.friendship_xp + FEED_XP,
        )
        return fed, outcome

    def pet(self, needs: Needs, recent_pets: int = 0) -> tuple[Needs, str]:
        """``recent_pets``: pets in the last 10 minutes, not counting this one."""
        if recent_pets >= PET_TOO_MUCH_AT:
            return dataclasses.replace(needs), "too_much"
        petted = dataclasses.replace(
            needs,
            affection=needs.affection + PET_AFFECTION,
            happiness=needs.happiness + PET_HAPPINESS,
            boredom=needs.boredom + PET_BOREDOM,
            friendship_xp=needs.friendship_xp + PET_XP,
        )
        return petted, "happy"

    def rest(self, needs: Needs) -> tuple[Needs, str]:
        """Asked to go to sleep.  Changes no need; the service starts her sleep on "sleep"."""
        outcome = "not_tired" if needs.energy > REST_NOT_TIRED_ABOVE else "sleep"
        return dataclasses.replace(needs), outcome

    def click(self, needs: Needs, recent_clicks: int = 0) -> tuple[Needs, str]:
        """``recent_clicks``: clicks in the last 5 seconds, not counting this one."""
        if recent_clicks >= CLICK_ANNOYED_AT:
            return dataclasses.replace(needs, happiness=needs.happiness + CLICK_ANNOYED_HAPPINESS), "annoyed"
        poked = dataclasses.replace(
            needs,
            boredom=needs.boredom + CLICK_BOREDOM,
            friendship_xp=needs.friendship_xp + CLICK_XP,
        )
        return poked, "poke"

    def wake(self, needs: Needs) -> tuple[Needs, str]:
        outcome = "grumpy" if needs.energy < WAKE_GRUMPY_BELOW else "ok"
        return dataclasses.replace(needs), outcome

    def talk(self, needs: Needs) -> tuple[Needs, str]:
        talked = dataclasses.replace(
            needs, boredom=needs.boredom + TALK_BOREDOM, friendship_xp=needs.friendship_xp + TALK_XP
        )
        return talked, "ok"

    def call(self, needs: Needs) -> tuple[Needs, str]:
        return dataclasses.replace(needs, boredom=needs.boredom + CALL_BOREDOM), "ok"

    def interact(
        self, needs: Needs, kind: InteractionKind, *, recent_pets: int = 0, recent_clicks: int = 0
    ) -> tuple[Needs, str]:
        """Dispatch ``kind`` to its rule.  Kinds without a rule (drags, and any
        kind added later) change nothing and answer "ok"."""
        kind = InteractionKind(kind)
        if kind is InteractionKind.FEED:
            return self.feed(needs)
        if kind is InteractionKind.PET:
            return self.pet(needs, recent_pets)
        if kind is InteractionKind.REST:
            return self.rest(needs)
        if kind is InteractionKind.CLICK:
            return self.click(needs, recent_clicks)
        if kind is InteractionKind.WAKE:
            return self.wake(needs)
        if kind is InteractionKind.TALK:
            return self.talk(needs)
        if kind is InteractionKind.CALL:
            return self.call(needs)
        return dataclasses.replace(needs), "ok"

    # -- presentation ----------------------------------------------------------------------

    def mood(self, needs: Needs) -> str:
        """One Ukrainian adjective (feminine) for the status window.

        Unmet physical needs win over feelings: a starving girl is "Голодна"
        even if she is otherwise happy.
        """
        if needs.energy < MOOD_SLEEPY_ENERGY_BELOW:
            return "Сонна"
        if needs.hunger > MOOD_HUNGRY_ABOVE:
            return "Голодна"
        if needs.boredom > MOOD_BORED_ABOVE:
            return "Нудьгує"
        if needs.happiness < MOOD_SAD_BELOW:
            return "Сумна"
        if needs.happiness > MOOD_VERY_HAPPY_ABOVE:
            return "Щаслива"
        if needs.happiness > MOOD_CHEERFUL_ABOVE:
            return "Весела"
        return "Спокійна"
