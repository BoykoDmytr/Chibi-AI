from __future__ import annotations

import random
from dataclasses import replace
from itertools import pairwise
from typing import Any

import pytest

from chibi.character.behavior import (
    BORED_COOLDOWN,
    CURIOUS_COOLDOWN,
    FIDGET_MAX_SECONDS,
    FIDGET_MIN_SECONDS,
    MIN_SLEEP_SECONDS,
    SIT_MAX_SECONDS,
    SIT_MIN_SECONDS,
    WAKE_GRACE_SECONDS,
    WAKE_PRIORITY_HOLD_SECONDS,
    WANDER_EDGE_FRACTION,
    WANDER_MIN_DISTANCE,
    YAWN_COOLDOWN,
    AutonomousBehavior,
    BehaviorContext,
    Decision,
    pick_wander_target,
    wander_interval,
)
from chibi.character.movement import Movement, Rect
from chibi.character.state_machine import CharacterStateMachine, StateChange
from chibi.character.states import CharacterState, Facing, StatePriority
from chibi.core.clock import FakeClock

S = CharacterState
P = StatePriority
AREA = Rect(0, 0, 1920, 1040)
LO = AREA.left + AREA.width * WANDER_EDGE_FRACTION
HI = AREA.right - AREA.width * WANDER_EDGE_FRACTION

DEFAULTS: dict[str, Any] = dict(
    posture=S.IDLE, current=S.IDLE, energy=80.0, boredom=30.0, happiness=70.0, hunger=20.0,
    walking_enabled=True, locked=False, on_ground=True, eco=False, independence=50, curiosity=60,
    idle_seconds=5.0, sleep_after_seconds=600.0, cursor_near=False, cursor_side=None, area=AREA, x=960.0,
)


def ctx(now: float, **overrides: Any) -> BehaviorContext:
    return BehaviorContext(now=now, **{**DEFAULTS, **overrides})


def behavior(seed: int = 1) -> AutonomousBehavior:
    return AutonomousBehavior(random.Random(seed))


def drive(b: AutonomousBehavior, seconds: int, start: float = 0.0,
          **overrides: Any) -> list[tuple[float, Decision]]:
    """Call decide once a second with a fixed context (the decisions are not applied)."""
    out = []
    for i in range(seconds):
        now = start + i
        d = b.decide(ctx(now, **overrides))
        if d is not None:
            out.append((now, d))
    return out


class Sim:
    """Behavior wired to the real state machine and movement, like the controller does."""

    WALK_SPEED = 80.0

    def __init__(self, seed: int = 1, **params: Any) -> None:
        self.clock = FakeClock()
        self.sm = CharacterStateMachine(self.clock)
        self.mv = Movement(960, AREA.bottom)
        self.mv.set_bounds(AREA, 60, 200)
        self.behavior = AutonomousBehavior(random.Random(seed))
        self.params = {k: v for k, v in DEFAULTS.items() if k not in ("posture", "current", "x")}
        self.params.update(params)
        self.log: list[tuple[float, Decision, bool, float]] = []
        self.sm.add_listener(self._on_change)

    def _on_change(self, change: StateChange) -> None:
        if change.posture is not S.WALK and self.mv.walking:
            self.mv.stop()

    @property
    def now(self) -> float:
        return self.clock.monotonic()

    def context(self) -> BehaviorContext:
        return BehaviorContext(now=self.now, posture=self.sm.posture, current=self.sm.current,
                               x=self.mv.x, **self.params)

    def step(self) -> None:
        self.clock.advance(1.0)
        self.sm.tick()
        self.mv.tick(1.0)
        if self.sm.posture is S.WALK and not self.mv.walking:
            self.sm.request(S.IDLE, P.AMBIENT)  # arrived
        x_before = self.mv.x
        d = self.behavior.decide(self.context())
        if d is None or d.state is None:
            return
        accepted = self.sm.request(d.state, d.priority, d.duration)
        if accepted and d.walk_to_x is not None:
            self.mv.walk_to(d.walk_to_x, self.WALK_SPEED)
        self.log.append((self.now, d, accepted, x_before))

    def run(self, seconds: int) -> Sim:
        for _ in range(seconds):
            self.step()
        return self

    def reasons(self, reason: str) -> list[tuple[float, Decision, bool, float]]:
        return [entry for entry in self.log if entry[1].reason == reason]


# -- helpers ---------------------------------------------------------------------------------------


def test_wander_interval_range_and_independence() -> None:
    rng = random.Random(3)
    means = {}
    for ind in (0, 50, 100):
        samples = [wander_interval(ind, rng) for _ in range(2000)]
        assert all(10.0 <= s <= 60.0 for s in samples)
        means[ind] = sum(samples) / len(samples)
    assert means[100] < means[50] < means[0]
    assert means[50] == pytest.approx(25.0, abs=1.5)
    assert all(10.0 <= wander_interval(v, rng) <= 60.0 for v in (-50, 500))  # out-of-range sliders


def test_pick_wander_target_properties() -> None:
    rng = random.Random(5)
    for x in (100.0, 400.0, 960.0, 1700.0, 1900.0):
        for _ in range(300):
            target = pick_wander_target(AREA, x, rng)
            assert target is not None
            assert LO <= target <= HI
            assert abs(target - x) >= WANDER_MIN_DISTANCE


def test_pick_wander_target_near_edge_goes_inwards() -> None:
    rng = random.Random(9)
    targets = [pick_wander_target(AREA, 200.0, rng) for _ in range(200)]
    assert all(t is not None and t >= 320.0 for t in targets)


def test_pick_wander_target_on_tiny_area() -> None:
    assert pick_wander_target(Rect(0, 0, 200, 1040), 100.0, random.Random(1)) is None


def test_pick_wander_target_on_negative_monitor() -> None:
    left = Rect(-1920, 0, 0, 1080)
    rng = random.Random(2)
    for _ in range(100):
        t = pick_wander_target(left, -960.0, rng)
        assert t is not None and -1920 * (1 - WANDER_EDGE_FRACTION) <= t <= -1920 * WANDER_EDGE_FRACTION
        assert abs(t + 960) >= WANDER_MIN_DISTANCE


# -- wandering -------------------------------------------------------------------------------------


def test_wanders_over_simulated_minutes() -> None:
    sim = Sim(seed=7, walking_enabled=True).run(15 * 60)
    walks = sim.reasons("wander")
    assert len(walks) >= 12
    for _, d, accepted, x_before in walks:
        assert d.state is S.WALK and d.priority is P.AMBIENT and accepted
        assert d.walk_to_x is not None
        assert LO <= d.walk_to_x <= HI
        assert abs(d.walk_to_x - x_before) >= WANDER_MIN_DISTANCE
        assert d.facing is (Facing.RIGHT if d.walk_to_x > x_before else Facing.LEFT)
    # she really walked there
    lo, hi = sim.mv.walk_range or (0, 0)
    assert lo <= sim.mv.x <= hi
    gaps = [b[0] - a[0] for a, b in pairwise(walks)]
    assert min(gaps) >= 10.0


@pytest.mark.parametrize("blocker", [dict(locked=True), dict(eco=True), dict(walking_enabled=False),
                                     dict(on_ground=False)])
def test_never_wanders_when_blocked(blocker: dict[str, Any]) -> None:
    sim = Sim(seed=11, **blocker).run(20 * 60)
    assert not [e for e in sim.log if e[1].state is S.WALK]
    assert sim.mv.x == 960
    assert sim.reasons("fidget")  # still alive: fidgets go on


def test_more_independent_wanders_more() -> None:
    counts = {ind: len(Sim(seed=4, independence=ind).run(30 * 60).reasons("wander")) for ind in (0, 100)}
    assert counts[100] > counts[0] * 1.5


def test_wander_timer_restarts_after_lock() -> None:
    b = behavior(2)
    locked = drive(b, 120, locked=True)
    assert all(d.state is not S.WALK for _, d in locked)
    # unlocking starts a fresh pause instead of firing an overdue walk at once
    first_after_unlock = [t for t, d in drive(b, 120, start=200) if d.reason == "wander"]
    assert first_after_unlock and first_after_unlock[0] >= 200 + 10


def test_wander_from_sit_stands_up_first() -> None:
    b = behavior(3)
    sitting = dict(posture=S.SIT, current=S.SIT, independence=100)  # wander due within 17.5 s
    decisions = drive(b, int(SIT_MIN_SECONDS) + 1, **sitting)
    assert len(decisions) == 1  # she sits at least SIT_MIN_SECONDS before getting up to walk
    t_stand, stand = decisions[0]
    assert t_stand == SIT_MIN_SECONDS
    assert stand == Decision(S.IDLE, P.AMBIENT, reason="stand_to_wander")
    follow = b.decide(ctx(t_stand + 1.0, independence=100))  # standing now: the walk follows
    assert follow is not None and follow.state is S.WALK and follow.walk_to_x is not None


def test_rejected_stand_up_is_not_repeated_every_second() -> None:
    b = behavior(3)
    decisions = drive(b, 600, posture=S.SIT, current=S.SIT, walking_enabled=True, cursor_near=False)
    stands = [t for t, d in decisions if d.state is S.IDLE]
    assert len(stands) >= 2
    assert all(b2 - a >= 15 for a, b2 in pairwise(stands))


# -- sleep -----------------------------------------------------------------------------------------


def test_sleep_sequence_yawn_then_sleep() -> None:
    b = behavior()
    sleepy = dict(idle_seconds=700.0)
    first = b.decide(ctx(100.0, **sleepy))
    assert first == Decision(S.YAWN, P.BEHAVIOR, reason="yawn")
    # while the yawn plays nothing else is decided
    assert b.decide(ctx(101.0, current=S.YAWN, **sleepy)) is None
    second = b.decide(ctx(102.0, **sleepy))
    assert second == Decision(S.SLEEP, P.BEHAVIOR, reason="sleep")


def test_sleep_sequence_in_simulation() -> None:
    sim = Sim(seed=1, walking_enabled=False, idle_seconds=0.0)
    sim.run(30)
    assert sim.sm.posture is S.IDLE
    sim.params["idle_seconds"] = 900.0
    sim.run(10)
    reasons = [d.reason for _, d, _, _ in sim.log if d.reason in ("yawn", "sleep")]
    assert reasons == ["yawn", "sleep"]
    yawn_t = sim.reasons("yawn")[0][0]
    sleep_t = sim.reasons("sleep")[0][0]
    assert sleep_t - yawn_t >= 1.8  # the yawn played completely first
    assert sim.sm.posture is S.SLEEP and sim.sm.current is S.SLEEP
    sim.run(600)
    assert len(sim.log) == len([e for e in sim.log if e[0] <= sleep_t])  # asleep: nothing else happens


def test_low_energy_makes_her_sleep_even_while_user_is_active() -> None:
    decisions = drive(behavior(), 5, energy=15.0, idle_seconds=0.0)
    assert [d.state for _, d in decisions[:2]] == [S.YAWN, S.SLEEP]


def test_yawn_at_most_once_per_ten_minutes() -> None:
    b = behavior()
    decisions = drive(b, 1800, idle_seconds=900.0)  # sleep never accepted (context stays IDLE)
    yawns = [t for t, d in decisions if d.state is S.YAWN]
    sleeps = [t for t, d in decisions if d.state is S.SLEEP]
    assert len(yawns) == 3
    assert all(b2 - a >= YAWN_COOLDOWN for a, b2 in pairwise(yawns))
    assert all(b2 - a >= 30 for a, b2 in pairwise(sleeps))  # rejected: retried slowly


def test_cursor_near_prevents_sleep() -> None:
    decisions = drive(behavior(), 600, idle_seconds=900.0, energy=12.0, cursor_near=True,
                      cursor_side=Facing.LEFT)
    assert decisions  # she is awake and doing things...
    assert all(d.state not in (S.YAWN, S.SLEEP) for _, d in decisions)  # ...but never nods off


def test_exhausted_sleeps_despite_cursor() -> None:
    decisions = drive(behavior(), 3, energy=8.0, cursor_near=True, cursor_side=Facing.RIGHT)
    assert [d.state for _, d in decisions[:2]] == [S.YAWN, S.SLEEP]


def test_sleepy_while_walking_yawns() -> None:
    b = behavior()
    d = b.decide(ctx(0.0, posture=S.WALK, current=S.WALK, energy=12.0))
    assert d is not None and d.state is S.YAWN
    assert b.decide(ctx(5.0, posture=S.WALK, current=S.WALK)) is None  # rested: walking is left alone


# -- waking ----------------------------------------------------------------------------------------


def asleep(now: float, **overrides: Any) -> BehaviorContext:
    base: dict[str, Any] = dict(posture=S.SLEEP, current=S.SLEEP, energy=100.0, idle_seconds=10.0)
    base.update(overrides)
    return ctx(now, **base)


def test_natural_wake_rule() -> None:
    b = behavior()
    assert b.decide(asleep(0.0)) is None  # falls asleep at t=0
    assert b.decide(asleep(MIN_SLEEP_SECONDS - 1)) is None  # not long enough
    assert b.decide(asleep(MIN_SLEEP_SECONDS, energy=94.0)) is None  # not rested
    assert b.decide(asleep(MIN_SLEEP_SECONDS, idle_seconds=600.0)) is None  # user away: keep sleeping
    wake = b.decide(asleep(MIN_SLEEP_SECONDS))
    assert wake == Decision(S.IDLE, P.REACTION, duration=WAKE_PRIORITY_HOLD_SECONDS, reason="wake")


def test_after_natural_wake_she_is_not_frozen() -> None:
    """The REACTION wake-up must not block her AMBIENT behavior afterwards."""
    sim = Sim(seed=5, energy=100.0, idle_seconds=10.0)
    sim.sm.request(S.SLEEP, P.BEHAVIOR)
    sim.run(int(MIN_SLEEP_SECONDS) + 5)
    wake = sim.reasons("wake")
    assert len(wake) == 1 and wake[0][2]  # accepted
    assert sim.sm.posture is not S.SLEEP
    woke_at = wake[0][0]
    sim.run(5 * 60)
    later = [(t, d, ok) for t, d, ok, _ in sim.log if t > woke_at]
    assert any(d.reason in ("wander", "fidget", "sit") for _, d, _ in later)
    assert all(ok for _, d, ok in later if d.priority is P.AMBIENT)


def test_rejected_wake_is_retried_slowly() -> None:
    b = behavior()
    b.decide(asleep(0.0))
    wakes = [t for t in range(int(MIN_SLEEP_SECONDS), int(MIN_SLEEP_SECONDS) + 300)
             if b.decide(asleep(float(t))) is not None]
    assert len(wakes) == 5 and wakes[1] - wakes[0] == 60


def test_no_other_decisions_while_asleep() -> None:
    b = behavior()
    decisions = drive(b, 1200, posture=S.SLEEP, current=S.SLEEP, energy=50.0, boredom=95.0,
                      cursor_near=True, cursor_side=Facing.LEFT, curiosity=100)
    assert decisions == []


def test_wake_grace_period() -> None:
    b = behavior()
    b.decide(asleep(0.0, energy=40))
    woke = 100.0
    # just woken up (and tired): she stays awake for the grace period
    early = drive(b, int(WAKE_GRACE_SECONDS), start=woke, energy=15.0)
    assert all(d.state not in (S.YAWN, S.SLEEP) for _, d in early)
    late = drive(b, 30, start=woke + WAKE_GRACE_SECONDS, energy=15.0)
    assert late and late[0][1].state is S.YAWN


def test_exhausted_ignores_wake_grace() -> None:
    b = behavior()
    b.decide(asleep(0.0, energy=40))
    d = b.decide(ctx(100.0, energy=5.0))
    assert d is not None and d.state is S.YAWN


# -- gating ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("current", [S.TALK, S.DRAGGED, S.WAVE, S.CURIOUS])
def test_no_decisions_during_actions(current: CharacterState) -> None:
    b = behavior()
    b.decide(ctx(0.0))  # arm timers
    decisions = drive(b, 600, start=1.0, current=current, energy=5.0, boredom=95.0, cursor_near=True,
                      cursor_side=Facing.LEFT, curiosity=100, idle_seconds=10_000.0)
    assert decisions == []


def test_no_decisions_while_walking_unless_sleepy() -> None:
    decisions = drive(behavior(), 600, posture=S.WALK, current=S.WALK, boredom=95.0,
                      cursor_near=True, cursor_side=Facing.LEFT, curiosity=100)
    assert decisions == []


def test_gaming_posture_only_watches() -> None:
    decisions = drive(behavior(), 1200, posture=S.GAMING, current=S.GAMING, boredom=95.0,
                      cursor_near=True, cursor_side=Facing.RIGHT, curiosity=100)
    assert decisions == []


# -- cursor curiosity ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", [Facing.LEFT, Facing.RIGHT])
def test_cursor_curiosity_faces_the_cursor(side: Facing) -> None:
    decisions = drive(behavior(), 300, cursor_near=True, cursor_side=side, walking_enabled=False)
    curious = [(t, d) for t, d in decisions if d.state is S.CURIOUS]
    assert curious
    for _, d in curious:
        assert d.facing is side and d.priority is P.AMBIENT and d.reason == "cursor"
    times = [t for t, _ in curious]
    assert all(b - a >= CURIOUS_COOLDOWN for a, b in pairwise(times))


def test_curiosity_scales_the_chance() -> None:
    def count(curiosity: int) -> int:
        total = 0
        for seed in range(20):
            decisions = drive(behavior(seed), 200, cursor_near=True, cursor_side=Facing.LEFT,
                              curiosity=curiosity, walking_enabled=False)
            total += sum(1 for _, d in decisions if d.state is S.CURIOUS)
        return total

    assert count(100) > count(0)


def test_cursor_noticed_while_sitting() -> None:
    decisions = drive(behavior(4), 120, posture=S.SIT, current=S.SIT, cursor_near=True,
                      cursor_side=Facing.RIGHT, curiosity=100)
    assert any(d.state is S.CURIOUS and d.facing is Facing.RIGHT for _, d in decisions)


# -- sitting, fidgets, boredom ---------------------------------------------------------------------


def test_fidgets_while_idle() -> None:
    decisions = drive(behavior(6), 600, walking_enabled=False, boredom=90.0)  # bored: no sitting
    fidgets = [(t, d) for t, d in decisions if d.reason == "fidget"]
    assert len(fidgets) >= 600 / FIDGET_MAX_SECONDS - 2
    for _, d in fidgets:
        assert d.state is S.IDLE and d.priority is P.AMBIENT
        assert d.variant in ("idle_look", "idle_smile")
        if d.variant == "idle_look":
            assert d.facing in (Facing.LEFT, Facing.RIGHT)
        else:
            assert d.facing is None
    assert {d.variant for _, d in fidgets} == {"idle_look", "idle_smile"}
    times = [t for t, _ in fidgets]
    assert all(b - a >= FIDGET_MIN_SECONDS for a, b in pairwise(times))


def test_sits_for_a_while_then_stands() -> None:
    sim = Sim(seed=8, walking_enabled=False).run(30 * 60)
    sits = sim.reasons("sit")
    ends = sim.reasons("sit_over")
    assert len(sits) >= 5 and len(ends) >= 5
    for (t_sit, *_), (t_end, *_) in zip(sits, ends, strict=False):
        assert SIT_MIN_SECONDS <= t_end - t_sit <= SIT_MAX_SECONDS + 1
    assert all(d.state is S.SIT and d.priority is P.AMBIENT for _, d, _, _ in sits)


def test_no_sitting_when_bored() -> None:
    sim = Sim(seed=8, walking_enabled=False, boredom=80.0).run(20 * 60)
    assert not sim.reasons("sit")


def test_bored_sad_or_thinking_with_cooldown() -> None:
    sim = Sim(seed=12, walking_enabled=False, boredom=95.0).run(30 * 60)
    bored = sim.reasons("bored")
    assert len(bored) >= 5
    assert {d.state for _, d, _, _ in bored} <= {S.SAD, S.THINKING}
    assert len({d.state for _, d, _, _ in bored}) == 2
    times = [t for t, *_ in bored]
    assert all(b - a >= BORED_COOLDOWN for a, b in pairwise(times))


def test_not_bored_below_threshold() -> None:
    sim = Sim(seed=12, walking_enabled=False, boredom=80.0).run(20 * 60)
    assert not sim.reasons("bored")


# -- determinism -----------------------------------------------------------------------------------


def _trace(seed: int) -> list[tuple[float, str, Any, Any]]:
    sim = Sim(seed=seed).run(20 * 60)
    return [(t, d.reason, d.state, d.walk_to_x) for t, d, _, _ in sim.log]


def test_deterministic_given_seed() -> None:
    assert _trace(21) == _trace(21)
    assert _trace(21) != _trace(22)


def test_context_is_frozen() -> None:
    c = ctx(0.0)
    with pytest.raises(AttributeError):
        c.energy = 1.0  # type: ignore[misc]
    assert replace(c, energy=1.0).energy == 1.0
