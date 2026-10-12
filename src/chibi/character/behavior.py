"""Autonomous behavior: what she does on her own when nobody asks anything.

:meth:`AutonomousBehavior.decide` is called about once a second by the
controller with a :class:`BehaviorContext` snapshot and returns a
:class:`Decision` - or, most of the time, None.  It never touches the state
machine or the movement itself: the controller applies the decision through
``CharacterStateMachine.request`` (so priorities still arbitrate) and
``Movement.walk_to``.

Design points:

* All timing comes from ``ctx.now`` and all randomness from the injected
  ``random.Random``, so a seeded run is exactly reproducible in tests.
* No decisions while an action, a drag or speech is showing
  (``ctx.current != ctx.posture``) or while she walks - a walk is finished by
  the controller.  Getting sleepy is the exception for walking: a yawn stops
  the walk (state machine rule) and she lies down afterwards.
* Every rule consumes its timer or cooldown when it fires, so a decision the
  state machine rejects (e.g. the user made her sit at USER priority) is not
  re-issued every second.
* She never falls asleep with the cursor right next to her unless she is
  really exhausted - nodding off while the user is reaching for her feels
  broken.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from chibi.character.movement import Rect
from chibi.character.states import CharacterState, Facing, StatePriority

log = logging.getLogger(__name__)

S = CharacterState

# -- sleep ---------------------------------------------------------------------------
#: Below this energy she gets sleepy regardless of user activity.
TIRED_ENERGY = 18.0
#: Below this she sleeps even with the cursor next to her.
EXHAUSTED_ENERGY = 10.0
#: At most one yawn per this many seconds; the sleep follows the yawn.
YAWN_COOLDOWN = 600.0
#: Natural wake-up needs this much energy...
WAKE_ENERGY = 95.0
#: ...and at least this long asleep (s).
MIN_SLEEP_SECONDS = 20 * 60.0
#: After waking up she stays awake at least this long (s) unless exhausted,
#: so a user who wakes her does not see her drop straight back to sleep.
WAKE_GRACE_SECONDS = 90.0
#: Retry interval for a wake-up the state machine rejected (s).
WAKE_RETRY_SECONDS = 60.0
#: How long the wake-up's REACTION priority is held (s).  REACTION is needed to
#: get through the state machine's sleep gate, but a posture keeps its priority,
#: so without a limit every later AMBIENT decision (fidgets, walks) would be
#: rejected and she would stand frozen after waking up.
WAKE_PRIORITY_HOLD_SECONDS = 3.0
#: Retry interval for lying down when the state machine rejected it (s).
SLEEP_RETRY_SECONDS = 30.0

# -- wandering -----------------------------------------------------------------------
#: Mean pause between walks at independence 50 (s); halves at 100, doubles at 0.
WANDER_MEAN_SECONDS = 25.0
WANDER_MIN_SECONDS = 10.0
WANDER_MAX_SECONDS = 60.0
#: A walk shorter than this looks like a twitch, not a stroll (px).
WANDER_MIN_DISTANCE = 120.0
#: Fraction of the area width at each side she does not wander into
#: (keeps her off the screen corners where icons and the start button live).
WANDER_EDGE_FRACTION = 0.08

# -- sitting -------------------------------------------------------------------------
SIT_MEAN_INTERVAL = 70.0
SIT_MIN_SECONDS = 30.0
SIT_MAX_SECONDS = 90.0
#: Sitting is for a content character; a bored one paces instead.
SIT_MAX_BOREDOM = 75.0
#: Retry interval for a stand-up the state machine rejected (s).
STAND_RETRY_SECONDS = 15.0

# -- fidgets, curiosity, boredom ------------------------------------------------------
FIDGET_MIN_SECONDS = 9.0
FIDGET_MAX_SECONDS = 22.0
FIDGET_VARIANTS = ("idle_look", "idle_smile")
CURIOUS_COOLDOWN = 30.0
#: Per-call chance to notice the cursor at curiosity 0 and the extra at 100.
CURIOUS_BASE_CHANCE = 0.05
CURIOUS_CHANCE_SPAN = 0.40
BORED_THRESHOLD = 80.0
BORED_COOLDOWN = 180.0
BORED_CHANCE = 0.15

_CALM_POSTURES = frozenset({S.IDLE, S.SIT})
_MOVING = frozenset({S.WALK, S.RUN})


@dataclass(frozen=True)
class BehaviorContext:
    now: float  # monotonic seconds
    posture: CharacterState
    current: CharacterState
    energy: float  # 0..100 (tamagotchi needs)
    boredom: float
    happiness: float
    hunger: float
    walking_enabled: bool
    locked: bool  # position locked by the user
    on_ground: bool
    eco: bool  # eco mode: high CPU load / battery saver - no walking
    independence: int  # personality sliders, 0..100
    curiosity: int
    idle_seconds: float  # since last interaction or (if permitted) user input
    sleep_after_seconds: float
    cursor_near: bool
    cursor_side: Facing | None  # which side of her the cursor is on
    area: Rect  # work area of her monitor
    x: float  # feet x


@dataclass(frozen=True)
class Decision:
    state: CharacterState | None
    priority: StatePriority
    duration: float | None = None
    walk_to_x: float | None = None
    variant: str | None = None  # clip variant to play (fidgets)
    facing: Facing | None = None
    reason: str = ""  # which rule fired, for diagnostics


def wander_interval(independence: float, rng: random.Random) -> float:
    """Seconds until the next stroll: ~25 s on average at independence 50,
    more often for an independent character, always within 10..60 s."""
    ind = min(max(float(independence), 0.0), 100.0)
    mean = WANDER_MEAN_SECONDS * 2.0 ** ((50.0 - ind) / 50.0)
    return min(max(mean * rng.uniform(0.6, 1.4), WANDER_MIN_SECONDS), WANDER_MAX_SECONDS)


def pick_wander_target(area: Rect, x: float, rng: random.Random) -> float | None:
    """A target x at least :data:`WANDER_MIN_DISTANCE` away from *x*, avoiding
    the outer :data:`WANDER_EDGE_FRACTION` of *area* on both sides.  None if
    the area is too small for a stroll."""
    margin = area.width * WANDER_EDGE_FRACTION
    lo, hi = area.left + margin, area.right - margin
    segments = [
        (lo, min(hi, x - WANDER_MIN_DISTANCE)),
        (max(lo, x + WANDER_MIN_DISTANCE), hi),
    ]
    lengths = [max(0.0, b - a) for a, b in segments]
    total = sum(lengths)
    if total <= 0.0:
        return None
    u = rng.uniform(0.0, total)
    if u <= lengths[0] and lengths[0] > 0.0:
        return segments[0][0] + u
    return segments[1][0] + min(u - lengths[0], lengths[1])


class AutonomousBehavior:
    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._last_posture: CharacterState | None = None
        # sleep
        self._last_yawn: float | None = None
        self._sleep_started: float | None = None
        self._woke_at: float | None = None
        self._next_wake_attempt = 0.0
        self._next_sleep_attempt = 0.0
        # wandering / sitting
        self._next_wander: float | None = None
        self._sit_since: float | None = None
        self._sit_until: float | None = None
        self._next_sit: float | None = None
        self._next_stand_attempt = 0.0
        # fidgets / curiosity / boredom
        self._next_fidget: float | None = None
        self._last_curious: float | None = None
        self._last_bored: float | None = None

    def decide(self, ctx: BehaviorContext) -> Decision | None:
        self._observe(ctx)
        if ctx.posture == S.SLEEP:
            return self._natural_wake(ctx)
        if ctx.current != ctx.posture:
            return None  # an action, speech or drag is showing
        decision = self._get_sleepy(ctx)
        if decision is not None or ctx.posture in _MOVING:
            return decision
        if self._woke_at is not None and ctx.now - self._woke_at < WAKE_PRIORITY_HOLD_SECONDS:
            return None  # just woke up: a beat of stillness (and the wake priority still holds)
        for rule in (self._notice_cursor, self._end_sit, self._wander, self._bored,
                     self._start_sit, self._fidget):
            decision = rule(ctx)
            if decision is not None:
                log.debug("behavior: %s -> %s", decision.reason,
                          decision.state.value if decision.state else None)
                return decision
        return None

    # -- bookkeeping ----------------------------------------------------------------

    def _observe(self, ctx: BehaviorContext) -> None:
        """Track posture transitions and (re)arm timers.  Runs on every call, so
        timers stay correct even on calls where no rule gets to run."""
        now = ctx.now
        posture = ctx.posture
        if posture != self._last_posture:
            previous = self._last_posture
            self._last_posture = posture
            if posture == S.SLEEP:
                self._sleep_started = now
                self._next_wake_attempt = 0.0
            elif previous == S.SLEEP:
                self._sleep_started = None
                self._woke_at = now
            if posture == S.SIT:
                self._sit_since = now
                self._sit_until = now + self._rng.uniform(SIT_MIN_SECONDS, SIT_MAX_SECONDS)
                self._next_stand_attempt = 0.0
            else:
                if previous == S.SIT:
                    self._next_sit = now + self._sit_interval()
                self._sit_since = self._sit_until = None

        # Walks are spaced from the end of the previous one, and a fresh pause
        # starts whenever wandering becomes possible again (unlocked, eco off).
        if self._can_wander(ctx):
            if self._next_wander is None:
                self._next_wander = now + wander_interval(ctx.independence, self._rng)
        else:
            self._next_wander = None
        if self._next_sit is None:
            self._next_sit = now + self._sit_interval()
        if self._next_fidget is None:
            self._next_fidget = now + self._fidget_interval()

    @staticmethod
    def _can_wander(ctx: BehaviorContext) -> bool:
        return (ctx.walking_enabled and not ctx.locked and not ctx.eco and ctx.on_ground
                and ctx.posture in _CALM_POSTURES)

    def _sit_interval(self) -> float:
        return SIT_MEAN_INTERVAL * self._rng.uniform(0.5, 1.5)

    def _fidget_interval(self) -> float:
        return self._rng.uniform(FIDGET_MIN_SECONDS, FIDGET_MAX_SECONDS)

    # -- sleep ----------------------------------------------------------------------

    def _natural_wake(self, ctx: BehaviorContext) -> Decision | None:
        started = self._sleep_started if self._sleep_started is not None else ctx.now
        if (ctx.energy >= WAKE_ENERGY
                and ctx.now - started >= MIN_SLEEP_SECONDS
                and ctx.idle_seconds < ctx.sleep_after_seconds
                and ctx.now >= self._next_wake_attempt):
            self._next_wake_attempt = ctx.now + WAKE_RETRY_SECONDS
            # REACTION: a sleeping character ignores anything below it.
            return Decision(S.IDLE, StatePriority.REACTION, duration=WAKE_PRIORITY_HOLD_SECONDS,
                            reason="wake")
        return None

    def _get_sleepy(self, ctx: BehaviorContext) -> Decision | None:
        """Yawn first, lie down on a later call."""
        exhausted = ctx.energy < EXHAUSTED_ENERGY
        if not (ctx.energy < TIRED_ENERGY or ctx.idle_seconds >= ctx.sleep_after_seconds):
            return None
        if ctx.cursor_near and not exhausted:
            return None
        if self._woke_at is not None and ctx.now - self._woke_at < WAKE_GRACE_SECONDS and not exhausted:
            return None
        if self._last_yawn is None or ctx.now - self._last_yawn >= YAWN_COOLDOWN:
            self._last_yawn = ctx.now
            return Decision(S.YAWN, StatePriority.BEHAVIOR, reason="yawn")
        if ctx.now > self._last_yawn and ctx.now >= self._next_sleep_attempt:
            self._next_sleep_attempt = ctx.now + SLEEP_RETRY_SECONDS
            return Decision(S.SLEEP, StatePriority.BEHAVIOR, reason="sleep")
        return None

    # -- awake rules ----------------------------------------------------------------

    def _notice_cursor(self, ctx: BehaviorContext) -> Decision | None:
        if not ctx.cursor_near or ctx.posture not in _CALM_POSTURES:
            return None
        if self._last_curious is not None and ctx.now - self._last_curious < CURIOUS_COOLDOWN:
            return None
        curiosity = min(max(ctx.curiosity, 0), 100) / 100.0
        if self._rng.random() >= CURIOUS_BASE_CHANCE + CURIOUS_CHANCE_SPAN * curiosity:
            return None
        self._last_curious = ctx.now
        return Decision(S.CURIOUS, StatePriority.AMBIENT, facing=ctx.cursor_side, reason="cursor")

    def _end_sit(self, ctx: BehaviorContext) -> Decision | None:
        if ctx.posture != S.SIT or self._sit_until is None or ctx.now < self._sit_until:
            return None
        return self._stand_up(ctx, "sit_over")

    def _stand_up(self, ctx: BehaviorContext, reason: str) -> Decision | None:
        if ctx.now < self._next_stand_attempt:
            return None
        self._next_stand_attempt = ctx.now + STAND_RETRY_SECONDS
        return Decision(S.IDLE, StatePriority.AMBIENT, reason=reason)

    def _wander(self, ctx: BehaviorContext) -> Decision | None:
        if self._next_wander is None or ctx.now < self._next_wander:
            return None
        if ctx.posture == S.SIT:
            # Stand up first; the walk follows on a later call (timer stays due).
            # A sit lasts at least SIT_MIN_SECONDS so she does not pop up at once.
            if self._sit_since is not None and ctx.now - self._sit_since < SIT_MIN_SECONDS:
                return None
            return self._stand_up(ctx, "stand_to_wander")
        self._next_wander = ctx.now + wander_interval(ctx.independence, self._rng)
        target = pick_wander_target(ctx.area, ctx.x, self._rng)
        if target is None:
            return None
        facing = Facing.RIGHT if target > ctx.x else Facing.LEFT
        return Decision(S.WALK, StatePriority.AMBIENT, walk_to_x=target, facing=facing, reason="wander")

    def _bored(self, ctx: BehaviorContext) -> Decision | None:
        if ctx.boredom <= BORED_THRESHOLD or ctx.posture not in _CALM_POSTURES:
            return None
        if self._last_bored is not None and ctx.now - self._last_bored < BORED_COOLDOWN:
            return None
        if self._rng.random() >= BORED_CHANCE:
            return None
        self._last_bored = ctx.now
        return Decision(self._rng.choice((S.SAD, S.THINKING)), StatePriority.AMBIENT, reason="bored")

    def _start_sit(self, ctx: BehaviorContext) -> Decision | None:
        if ctx.posture != S.IDLE or ctx.boredom >= SIT_MAX_BOREDOM:
            return None
        if self._next_sit is None or ctx.now < self._next_sit:
            return None
        self._next_sit = ctx.now + self._sit_interval()
        return Decision(S.SIT, StatePriority.AMBIENT, reason="sit")

    def _fidget(self, ctx: BehaviorContext) -> Decision | None:
        if ctx.posture != S.IDLE or self._next_fidget is None or ctx.now < self._next_fidget:
            return None
        self._next_fidget = ctx.now + self._fidget_interval()
        variant = self._rng.choice(FIDGET_VARIANTS)
        # "Looking around" means looking somewhere: pick a side.
        facing = self._rng.choice((Facing.LEFT, Facing.RIGHT)) if variant == "idle_look" else None
        return Decision(S.IDLE, StatePriority.AMBIENT, variant=variant, facing=facing, reason="fidget")
