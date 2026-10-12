from __future__ import annotations

import logging
import math

import pytest

from chibi.character.state_machine import (
    DEFAULT_ACTION_SECONDS,
    MIN_ACTION_SECONDS,
    CharacterStateMachine,
    StateChange,
)
from chibi.character.states import POSTURES, CharacterState, StatePriority
from chibi.core.clock import FakeClock

S = CharacterState
P = StatePriority


@pytest.fixture
def sm(clock: FakeClock) -> CharacterStateMachine:
    return CharacterStateMachine(clock)


@pytest.fixture
def changes(sm: CharacterStateMachine) -> list[StateChange]:
    seen: list[StateChange] = []
    sm.add_listener(seen.append)
    return seen


def snapshot(sm: CharacterStateMachine) -> tuple[CharacterState, CharacterState, StatePriority]:
    return sm.current, sm.posture, sm.priority


# -- defaults ---------------------------------------------------------------------------------


def test_default_durations_cover_every_action() -> None:
    assert set(DEFAULT_ACTION_SECONDS) == set(CharacterState) - POSTURES - {S.DRAGGED}
    assert dict(DEFAULT_ACTION_SECONDS) == {
        S.TALK: 3.0, S.LAUGH: 2.5, S.HAPPY: 3.0, S.ANGRY: 3.0, S.SAD: 4.0, S.SURPRISED: 1.8,
        S.CURIOUS: 3.0, S.THINKING: 3.0, S.WAVE: 2.4, S.EAT: 3.0, S.SHY: 3.0, S.YAWN: 1.8,
        S.BYE: 2.2, S.POINT: 1.6,
    }


def test_initial_state(sm: CharacterStateMachine) -> None:
    assert snapshot(sm) == (S.IDLE, S.IDLE, P.AMBIENT)
    assert sm.tick() is None
    assert not sm.dragged and not sm.asleep and not sm.action_active
    assert sm.action_remaining() is None


def test_initial_posture_can_be_chosen(clock: FakeClock) -> None:
    assert CharacterStateMachine(clock, S.SLEEP).posture is S.SLEEP
    with pytest.raises(ValueError):
        CharacterStateMachine(clock, S.TALK)


# -- postures and actions ---------------------------------------------------------------------


def test_postures_persist(sm: CharacterStateMachine, clock: FakeClock, changes: list[StateChange]) -> None:
    assert sm.request(S.SIT)
    assert snapshot(sm) == (S.SIT, S.SIT, P.AMBIENT)
    clock.advance(3600)
    assert sm.tick() is None
    assert sm.current is S.SIT
    assert changes == [StateChange(S.IDLE, S.SIT, S.SIT)]


@pytest.mark.parametrize("action", sorted(DEFAULT_ACTION_SECONDS, key=str))
def test_actions_expire_after_default_duration(clock: FakeClock, action: CharacterState) -> None:
    sm = CharacterStateMachine(clock)
    sm.request(S.SIT)
    assert sm.request(action, P.USER)
    assert snapshot(sm) == (action, S.SIT, P.USER)
    clock.advance(DEFAULT_ACTION_SECONDS[action] - 0.01)
    assert sm.tick() is None and sm.action_active
    clock.advance(0.011)
    assert sm.tick() == StateChange(action, S.SIT, S.SIT)
    assert snapshot(sm) == (S.SIT, S.SIT, P.AMBIENT)
    assert sm.tick() is None


def test_explicit_duration(sm: CharacterStateMachine, clock: FakeClock) -> None:
    sm.request(S.TALK, P.USER, duration=7.5)
    assert sm.action_remaining() == pytest.approx(7.5)
    clock.advance(7.4)
    assert sm.tick() is None
    clock.advance(0.1)
    assert sm.tick() is not None


@pytest.mark.parametrize(("duration", "expected"),
                         [(0.0, MIN_ACTION_SECONDS), (-3.0, MIN_ACTION_SECONDS),
                          (math.nan, 3.0), (math.inf, 3.0), (None, 3.0)])
def test_odd_durations(sm: CharacterStateMachine, duration: float | None, expected: float) -> None:
    sm.request(S.TALK, P.USER, duration=duration)
    assert sm.action_remaining() == pytest.approx(expected)


def test_posture_persists_despite_duration(sm: CharacterStateMachine, clock: FakeClock) -> None:
    sm.request(S.SIT, duration=1.0)
    clock.advance(10)
    assert sm.tick() is None and sm.current is S.SIT


def test_posture_duration_limits_priority_hold(sm: CharacterStateMachine, clock: FakeClock,
                                               changes: list[StateChange]) -> None:
    sm.request(S.SLEEP, P.BEHAVIOR)
    assert sm.request(S.IDLE, P.REACTION, duration=3.0)  # e.g. a natural wake-up
    clock.advance(2.9)
    assert sm.tick() is None
    assert sm.priority is P.REACTION
    assert not sm.request(S.SIT, P.AMBIENT)
    clock.advance(0.1)
    assert sm.request(S.WAVE, P.AMBIENT)  # hold over: accepted even before tick()
    clock.advance(2.4)
    sm.tick()
    assert snapshot(sm) == (S.IDLE, S.IDLE, P.AMBIENT)


def test_priority_hold_expires_in_tick_without_state_change(sm: CharacterStateMachine, clock: FakeClock,
                                                            changes: list[StateChange]) -> None:
    sm.request(S.SIT, P.REACTION, duration=5.0)
    count = len(changes)
    clock.advance(5.0)
    assert sm.tick() is None  # priority dropped, posture unchanged: nothing to report
    assert snapshot(sm) == (S.SIT, S.SIT, P.AMBIENT)
    assert len(changes) == count


@pytest.mark.parametrize("duration", [None, math.nan, math.inf])
def test_posture_without_finite_duration_holds_priority(sm: CharacterStateMachine, clock: FakeClock,
                                                        duration: float | None) -> None:
    sm.request(S.SIT, P.REACTION, duration=duration)
    clock.advance(3600)
    sm.tick()
    assert sm.priority is P.REACTION
    assert not sm.request(S.IDLE, P.AMBIENT)


def test_new_request_clears_priority_hold(sm: CharacterStateMachine, clock: FakeClock) -> None:
    sm.request(S.SIT, P.REACTION, duration=2.0)
    sm.request(S.GAMING, P.REACTION)  # no duration: held until replaced
    clock.advance(10)
    sm.tick()
    assert sm.priority is P.REACTION


def test_posture_request_cancels_running_action(sm: CharacterStateMachine, clock: FakeClock,
                                                changes: list[StateChange]) -> None:
    sm.request(S.TALK)
    assert sm.request(S.SIT)
    assert snapshot(sm) == (S.SIT, S.SIT, P.AMBIENT)
    clock.advance(10)
    assert sm.tick() is None
    assert changes[-1] == StateChange(S.TALK, S.SIT, S.SIT)


def test_action_replaces_action(sm: CharacterStateMachine, clock: FakeClock) -> None:
    sm.request(S.TALK, P.REACTION)
    clock.advance(1)
    assert sm.request(S.LAUGH, P.USER)
    assert snapshot(sm) == (S.LAUGH, S.IDLE, P.USER)
    assert sm.action_remaining() == pytest.approx(2.5)


# -- priorities ---------------------------------------------------------------------------------


def test_lower_priority_rejected_while_action_runs(sm: CharacterStateMachine) -> None:
    sm.request(S.TALK, P.USER)
    for priority in (P.AMBIENT, P.BEHAVIOR, P.REACTION):
        assert not sm.request(S.WAVE, priority)
        assert not sm.request(S.SIT, priority)
    assert snapshot(sm) == (S.TALK, S.IDLE, P.USER)


def test_equal_and_higher_priority_win(sm: CharacterStateMachine) -> None:
    sm.request(S.TALK, P.REACTION)
    assert sm.request(S.WAVE, P.REACTION)
    assert sm.request(S.LAUGH, P.USER)
    assert sm.current is S.LAUGH


def test_expired_action_accepts_anything_before_tick(sm: CharacterStateMachine, clock: FakeClock,
                                                     changes: list[StateChange]) -> None:
    sm.request(S.TALK, P.USER)
    clock.advance(3.0)
    assert sm.request(S.CURIOUS, P.AMBIENT)
    assert snapshot(sm) == (S.CURIOUS, S.IDLE, P.AMBIENT)
    assert changes[-1] == StateChange(S.TALK, S.CURIOUS, S.IDLE)


def test_expired_action_then_posture_request(sm: CharacterStateMachine, clock: FakeClock,
                                             changes: list[StateChange]) -> None:
    sm.request(S.TALK, P.USER)
    clock.advance(5)
    assert sm.request(S.IDLE, P.AMBIENT)
    assert snapshot(sm) == (S.IDLE, S.IDLE, P.AMBIENT)
    assert changes[-1] == StateChange(S.TALK, S.IDLE, S.IDLE)
    assert sm.tick() is None


def test_posture_keeps_its_priority(sm: CharacterStateMachine) -> None:
    assert sm.request(S.GAMING, P.REACTION)
    assert sm.priority is P.REACTION
    assert not sm.request(S.IDLE, P.AMBIENT)
    assert not sm.request(S.WALK, P.BEHAVIOR)
    assert not sm.request(S.WAVE, P.BEHAVIOR)
    assert sm.request(S.IDLE, P.REACTION)
    assert snapshot(sm) == (S.IDLE, S.IDLE, P.REACTION)


def test_priority_drops_to_ambient_after_an_action(sm: CharacterStateMachine, clock: FakeClock) -> None:
    sm.request(S.SIT, P.REACTION)
    sm.request(S.WAVE, P.USER)
    clock.advance(2.4)
    assert sm.tick() == StateChange(S.WAVE, S.SIT, S.SIT)
    assert sm.priority is P.AMBIENT
    assert sm.request(S.IDLE, P.AMBIENT)  # autonomous behavior may move her again


def test_drag_priority_is_reserved_for_dragging(sm: CharacterStateMachine) -> None:
    assert sm.request(S.SIT, P.DRAG)
    assert sm.priority is P.USER  # capped: a posture held at DRAG could never be replaced
    assert sm.request(S.TALK, P.USER)


def test_string_states_are_accepted(sm: CharacterStateMachine) -> None:
    assert sm.request("laugh", P.USER)
    assert sm.current is S.LAUGH
    assert not sm.request("moonwalk", P.USER)
    assert sm.current is S.LAUGH


# -- dragging ---------------------------------------------------------------------------------


def test_dragged_needs_drag_priority(sm: CharacterStateMachine) -> None:
    for priority in (P.AMBIENT, P.BEHAVIOR, P.REACTION, P.USER):
        assert not sm.request(S.DRAGGED, priority)
    assert sm.request(S.DRAGGED, P.DRAG)
    assert sm.dragged and sm.priority is P.DRAG


def test_drag_is_held_until_release(sm: CharacterStateMachine, clock: FakeClock,
                                    changes: list[StateChange]) -> None:
    sm.request(S.DRAGGED, P.DRAG)
    clock.advance(3600)
    assert sm.tick() is None
    assert sm.action_remaining() is None
    for state in (S.TALK, S.IDLE, S.SLEEP):
        assert not sm.request(state, P.USER)
        assert not sm.request(state, P.DRAG)
    assert sm.request(S.DRAGGED, P.DRAG)  # re-request: accepted, nothing changes
    assert sm.current is S.DRAGGED
    assert changes == [StateChange(S.IDLE, S.DRAGGED, S.IDLE)]
    sm.release_drag()
    assert snapshot(sm) == (S.IDLE, S.IDLE, P.AMBIENT)
    assert changes[-1] == StateChange(S.DRAGGED, S.IDLE, S.IDLE)


def test_release_without_drag_is_a_no_op(sm: CharacterStateMachine, changes: list[StateChange]) -> None:
    sm.request(S.SIT)
    sm.release_drag()
    assert sm.current is S.SIT
    assert len(changes) == 1


def test_drag_from_sit_lands_idle(sm: CharacterStateMachine) -> None:
    sm.request(S.SIT, P.USER)
    assert sm.request(S.DRAGGED, P.DRAG)
    assert sm.posture is S.SIT  # posture underneath is kept during the drag...
    sm.release_drag()
    assert snapshot(sm) == (S.IDLE, S.IDLE, P.AMBIENT)  # ...and she lands standing


def test_drag_wakes_and_stops_walk(sm: CharacterStateMachine) -> None:
    sm.request(S.SLEEP, P.USER)
    assert sm.request(S.DRAGGED, P.DRAG)
    assert sm.posture is S.IDLE and not sm.asleep
    sm.release_drag()
    sm.request(S.WALK)
    sm.request(S.DRAGGED, P.DRAG)
    assert sm.posture is S.IDLE


# -- sleeping ---------------------------------------------------------------------------------


def test_asleep_rejects_requests_below_reaction(sm: CharacterStateMachine) -> None:
    assert sm.request(S.SLEEP, P.BEHAVIOR)
    assert sm.asleep
    for priority in (P.AMBIENT, P.BEHAVIOR):
        for state in (S.WAVE, S.YAWN, S.IDLE, S.WALK, S.SIT):
            assert not sm.request(state, priority)
    assert snapshot(sm) == (S.SLEEP, S.SLEEP, P.BEHAVIOR)


def test_asleep_rejects_low_priority_even_if_posture_priority_is_ambient(clock: FakeClock) -> None:
    sm = CharacterStateMachine(clock, S.SLEEP)
    assert sm.priority is P.AMBIENT
    assert not sm.request(S.TALK, P.BEHAVIOR)


def test_action_wakes_her(sm: CharacterStateMachine, clock: FakeClock, changes: list[StateChange]) -> None:
    sm.request(S.SLEEP, P.BEHAVIOR)
    assert sm.request(S.SURPRISED, P.REACTION)
    assert snapshot(sm) == (S.SURPRISED, S.IDLE, P.REACTION)
    assert changes[-1] == StateChange(S.SLEEP, S.SURPRISED, S.IDLE)
    clock.advance(1.8)
    assert sm.tick() == StateChange(S.SURPRISED, S.IDLE, S.IDLE)


def test_natural_wake_with_reaction_posture(sm: CharacterStateMachine) -> None:
    sm.request(S.SLEEP, P.BEHAVIOR)
    assert sm.request(S.IDLE, P.REACTION)
    assert snapshot(sm) == (S.IDLE, S.IDLE, P.REACTION)


def test_posture_change_while_asleep(sm: CharacterStateMachine) -> None:
    sm.request(S.SLEEP, P.BEHAVIOR)
    assert sm.request(S.SIT, P.USER)
    assert snapshot(sm) == (S.SIT, S.SIT, P.USER)


def test_user_sleep_blocks_reaction_wake(sm: CharacterStateMachine) -> None:
    """Postures keep their priority: a USER-requested sleep outranks REACTION."""
    sm.request(S.SLEEP, P.USER)
    assert not sm.request(S.IDLE, P.REACTION)
    assert sm.request(S.IDLE, P.USER)


# -- walking ----------------------------------------------------------------------------------


@pytest.mark.parametrize("moving", [S.WALK, S.RUN])
def test_action_stops_walk(sm: CharacterStateMachine, clock: FakeClock, changes: list[StateChange],
                           moving: CharacterState) -> None:
    sm.request(moving)
    assert sm.request(S.WAVE)
    assert snapshot(sm) == (S.WAVE, S.IDLE, P.AMBIENT)
    assert changes[-1] == StateChange(moving, S.WAVE, S.IDLE)
    clock.advance(2.4)
    assert sm.tick() == StateChange(S.WAVE, S.IDLE, S.IDLE)


def test_posture_replaces_walk(sm: CharacterStateMachine) -> None:
    sm.request(S.WALK)
    assert sm.request(S.SIT)
    assert snapshot(sm) == (S.SIT, S.SIT, P.AMBIENT)


def test_action_on_sit_or_gaming_keeps_posture(sm: CharacterStateMachine, clock: FakeClock) -> None:
    sm.request(S.GAMING, P.REACTION)
    assert sm.request(S.LAUGH, P.USER)
    assert sm.posture is S.GAMING
    clock.advance(2.5)
    assert sm.tick() == StateChange(S.LAUGH, S.GAMING, S.GAMING)


# -- refresh ----------------------------------------------------------------------------------


def test_same_action_refreshes_duration_without_event(sm: CharacterStateMachine, clock: FakeClock,
                                                      changes: list[StateChange]) -> None:
    sm.request(S.TALK, P.REACTION)
    clock.advance(2.0)
    assert sm.request(S.TALK, P.USER)
    assert sm.priority is P.USER
    assert len(changes) == 1
    clock.advance(2.0)
    assert sm.tick() is None  # refreshed: 3 s from the second request
    clock.advance(1.0)
    assert sm.tick() == StateChange(S.TALK, S.IDLE, S.IDLE)
    assert len(changes) == 2


def test_same_action_lower_priority_does_not_refresh(sm: CharacterStateMachine, clock: FakeClock) -> None:
    sm.request(S.TALK, P.USER)
    clock.advance(2.0)
    assert not sm.request(S.TALK, P.AMBIENT)
    clock.advance(1.0)
    assert sm.tick() is not None


def test_same_posture_refreshes_priority_without_event(sm: CharacterStateMachine,
                                                       changes: list[StateChange]) -> None:
    assert sm.request(S.IDLE, P.REACTION)
    assert sm.priority is P.REACTION
    assert changes == []
    assert sm.request(S.IDLE, P.REACTION)
    assert changes == []


# -- listeners --------------------------------------------------------------------------------


def test_listener_failure_does_not_break_the_machine(sm: CharacterStateMachine, clock: FakeClock,
                                                     caplog: pytest.LogCaptureFixture) -> None:
    def boom(change: StateChange) -> None:
        raise RuntimeError("listener bug")

    seen: list[StateChange] = []
    sm.add_listener(boom)
    sm.add_listener(seen.append)
    with caplog.at_level(logging.ERROR, logger="chibi.character.state_machine"):
        assert sm.request(S.WAVE)
        clock.advance(2.4)
        change = sm.tick()
    assert change == StateChange(S.WAVE, S.IDLE, S.IDLE)
    assert seen == [StateChange(S.IDLE, S.WAVE, S.IDLE), change]
    assert "listener bug" in caplog.text
    assert sm.request(S.SIT)  # still fully working


def test_reentrant_listener_changes_arrive_in_order(sm: CharacterStateMachine) -> None:
    order: list[tuple[str, CharacterState]] = []

    def first(change: StateChange) -> None:
        order.append(("first", change.current))
        if change.current is S.SURPRISED:
            sm.request(S.LAUGH, P.USER)

    def second(change: StateChange) -> None:
        order.append(("second", change.current))

    sm.add_listener(first)
    sm.add_listener(second)
    sm.request(S.SURPRISED, P.REACTION)
    assert order == [("first", S.SURPRISED), ("second", S.SURPRISED),
                     ("first", S.LAUGH), ("second", S.LAUGH)]
    assert sm.current is S.LAUGH


def test_remove_listener(sm: CharacterStateMachine) -> None:
    seen: list[StateChange] = []
    sm.add_listener(seen.append)
    sm.remove_listener(seen.append)
    sm.remove_listener(seen.append)  # unknown: ignored
    sm.request(S.SIT)
    assert seen == []


def test_rejected_requests_do_not_notify(sm: CharacterStateMachine, changes: list[StateChange]) -> None:
    sm.request(S.TALK, P.USER)
    sm.request(S.WAVE, P.AMBIENT)
    sm.request(S.DRAGGED, P.USER)
    assert changes == [StateChange(S.IDLE, S.TALK, S.IDLE)]
