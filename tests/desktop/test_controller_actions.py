"""Phase 5 in the controller: initiative actions from the decision engine under
every guard, the landing squash, idle variety and frame-accurate blinks."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pytest
from PySide6.QtCore import QPointF, Qt

from chibi.character.animation import CharacterPack
from chibi.character.behavior import Decision
from chibi.character.movement import Rect
from chibi.character.state_machine import DEFAULT_ACTION_SECONDS
from chibi.character.states import CharacterState, Facing, StatePriority
from chibi.core.events import (
    FullscreenChanged,
    GameStarted,
    InitiativeAction,
    SystemLoadChanged,
)
from chibi.desktop import renderers
from chibi.desktop.controller import (
    APPROACH_GAP,
    APPROACH_MAX_DISTANCE,
    ECO_JUMP_SECONDS,
    FPS_CALM,
    FPS_FAST,
    IDLE_STRETCH_CHANCE,
    LANDING_SQUASH_SECONDS,
    NAP_MIN_IDLE_SECONDS,
    NAP_WAKE_GRACE_SECONDS,
    CharacterController,
    ScreenArea,
    idle_variety,
    landing_squash,
)
from chibi.desktop.renderers import IMPULSE_SQUASH_MAX
from tests.desktop.helpers import MIA_DIR, FakeCursor, FakeScreens, alpha_bbox, run

S = CharacterState
P = StatePriority

ACTIONS = ("wave", "jump", "dance", "sit", "sleep", "approach_cursor")
TIRED = (20.0, 30.0, 70.0, 20.0)  # hunger, energy, happiness, boredom: tired enough for a nap
#: Where each action shows when it happens.
SHOWS = {"wave": S.WAVE, "jump": S.JUMP, "dance": S.DANCE, "sit": S.SIT, "sleep": S.YAWN,
         "approach_cursor": S.WALK}


def act(bus, action) -> None:
    bus.publish(InitiativeAction(action=action))
    bus.dispatch_pending()


def still(ctrl: CharacterController) -> CharacterController:
    """No autonomous behavior: only what the test does happens."""
    ctrl._behavior.decide = lambda _ctx: None  # type: ignore[method-assign]
    return ctrl


@pytest.fixture
def ready(make_controller, clock):
    """A tired character, alone for a while, cursor far to her left: every action is possible."""

    def build(**kwargs):
        kwargs.setdefault("needs", TIRED)
        kwargs.setdefault("cursor", FakeCursor(200.0, 300.0))
        ctrl = still(make_controller(**kwargs))
        run(ctrl, clock, 1.0)
        clock.advance(NAP_MIN_IDLE_SECONDS + 20.0)  # nobody around for a while
        run(ctrl, clock, 0.5)
        assert ctrl.state is S.IDLE and ctrl.posture is S.IDLE
        return ctrl

    return build


def performed(ctrl: CharacterController, action: str) -> bool:
    if action == "sit":
        return ctrl.posture is S.SIT
    if action == "approach_cursor":
        return ctrl.posture is S.WALK and ctrl.movement.walking
    return ctrl.state is SHOWS[action]


# -- every action, and every guard -----------------------------------------------------------------


@pytest.mark.parametrize("action", ACTIONS)
def test_each_action_happens_when_nothing_stops_it(ready, bus, action):
    ctrl = ready()
    act(bus, action)
    assert performed(ctrl, action)
    assert ctrl.state_machine.priority is P.BEHAVIOR


def _guard(guard: str, ctrl: CharacterController, bus, settings) -> None:
    if guard == "hidden":
        bus.publish(FullscreenChanged(active=True))
        bus.dispatch_pending()
        assert not ctrl.is_visible()
    elif guard == "hidden_by_user":
        ctrl.set_hidden_by_user(True)
    elif guard == "dragged":
        ctrl.window.drag_started.emit()
        assert ctrl.state is S.DRAGGED
    elif guard == "locked":
        settings.character.lock_position = True
        ctrl.apply_settings()
    elif guard == "walking_disabled":
        settings.character.walking_enabled = False
        ctrl.apply_settings()
    elif guard == "eco":
        bus.publish(SystemLoadChanged(high=True, cpu_percent=97.0))
        bus.dispatch_pending()
        assert ctrl.eco
    elif guard == "quiet":
        settings.behavior.do_not_disturb = True  # silent gestures are fine
        ctrl.apply_settings()
    else:  # pragma: no cover - test table typo
        raise AssertionError(guard)


#: guard -> the actions that still happen under it
ALLOWED = {
    "hidden": set(),
    "hidden_by_user": set(),
    "dragged": set(),
    "locked": set(ACTIONS) - {"approach_cursor"},
    "walking_disabled": set(ACTIONS) - {"approach_cursor"},
    "eco": set(ACTIONS) - {"approach_cursor", "dance"},
    "quiet": set(ACTIONS),
}


@pytest.mark.parametrize("guard", list(ALLOWED))
@pytest.mark.parametrize("action", ACTIONS)
def test_actions_respect_every_guard(ready, bus, settings, clock, guard, action):
    ctrl = ready()
    _guard(guard, ctrl, bus, settings)
    before = (ctrl.state, ctrl.posture, ctrl.movement.x, ctrl.movement.y)
    act(bus, action)
    if action in ALLOWED[guard]:
        assert performed(ctrl, action), (guard, action)
        return
    assert (ctrl.state, ctrl.posture, ctrl.movement.x, ctrl.movement.y) == before, (guard, action)
    assert not ctrl.movement.walking
    run(ctrl, clock, 5.0)
    assert ctrl.movement.x == before[2]  # nothing was set in motion either
    assert ctrl.posture is not S.SLEEP


def test_unknown_or_malformed_actions_are_ignored(ready, bus):
    ctrl = ready()
    for action in ("fly", "", "DANCE", "approach-cursor", "wave " * 40, 123, None):
        act(bus, action)
    assert (ctrl.state, ctrl.posture) == (S.IDLE, S.IDLE)
    assert not ctrl.movement.walking


@pytest.mark.parametrize("action", ACTIONS)
def test_actions_never_wake_her(ready, bus, clock, action):
    ctrl = ready()
    ctrl.request_state(S.SLEEP, P.BEHAVIOR)
    act(bus, action)
    assert ctrl.posture is S.SLEEP and ctrl.state is S.SLEEP
    assert not ctrl.movement.walking


@pytest.mark.parametrize("action", ["jump", "dance", "wave", "sit"])
def test_actions_do_not_interrupt_what_the_user_asked_for(ready, bus, action):
    ctrl = ready()
    ctrl.say("Слухаю тебе!", state="talk", solicited=True)
    act(bus, action)
    assert ctrl.state is S.TALK and ctrl.posture is S.IDLE


def test_no_hopping_or_sitting_while_she_watches_a_game(ready, bus, clock):
    ctrl = ready()
    bus.publish(GameStarted(game_id="minecraft", title="Minecraft", process_name="javaw.exe"))
    bus.dispatch_pending()
    run(ctrl, clock, 3.5)  # past the gaming posture's priority hold: only the guard is left
    assert ctrl.posture is S.GAMING and ctrl.state_machine.priority is P.AMBIENT
    for action in ("jump", "dance", "sit", "sleep", "approach_cursor"):
        act(bus, action)
        assert ctrl.state is S.GAMING and ctrl.posture is S.GAMING, action
    act(bus, "wave")  # a wave is fine
    assert ctrl.state is S.WAVE and ctrl.posture is S.GAMING


# -- each action in detail ---------------------------------------------------------------------------


def test_jump_and_dance_play_their_length_at_full_frame_rate(ready, bus, clock):
    ctrl = ready()
    act(bus, "jump")
    run(ctrl, clock, 0.2)
    assert ctrl.state is S.JUMP and ctrl.fps == FPS_FAST
    run(ctrl, clock, DEFAULT_ACTION_SECONDS[S.JUMP] - 0.4)
    assert ctrl.state is S.JUMP
    run(ctrl, clock, 0.4)
    assert ctrl.state is S.IDLE and ctrl.state_machine.priority is P.AMBIENT
    run(ctrl, clock, 0.2)
    assert ctrl.fps == FPS_CALM
    act(bus, "dance")
    run(ctrl, clock, DEFAULT_ACTION_SECONDS[S.DANCE] - 0.3)
    assert ctrl.state is S.DANCE and ctrl.fps == FPS_FAST
    assert ctrl.visual_state().overlay == "hearts"
    run(ctrl, clock, 0.5)
    assert ctrl.state is S.IDLE


def test_eco_mode_jump_is_a_single_hop(ready, bus, clock):
    ctrl = ready()
    bus.publish(SystemLoadChanged(high=False, cpu_percent=10.0, on_battery_saver=True))
    bus.dispatch_pending()
    act(bus, "jump")
    assert ctrl.state is S.JUMP
    run(ctrl, clock, ECO_JUMP_SECONDS + 0.1)
    assert ctrl.state is S.IDLE


def test_jump_restarts_on_the_ground(ready, bus, clock):
    """The jump effect runs on the clip's own time: a jump never starts mid-air."""
    ctrl = ready()
    run(ctrl, clock, 0.73)  # some arbitrary global animation time
    act(bus, "jump")
    run(ctrl, clock, 0.1)
    visual = ctrl.visual_state()
    assert visual.clip_t == pytest.approx(0.1)
    tr = renderers.fx_transform(visual.clip.fx, visual.t, 100.0, -100.0, visual.clip_t)
    assert tr.map(QPointF(0.0, 0.0)).y() == pytest.approx(0.0)  # still crouching on the floor


def test_sit_lasts_until_her_behavior_stands_her_up(make_controller, bus, clock, settings):
    settings.character.walking_enabled = False
    ctrl = make_controller(seed=4)
    run(ctrl, clock, 1.0)
    states: list[str] = []
    ctrl.state_changed.connect(states.append)
    assert ctrl.posture is S.IDLE  # seeded: nothing else going on yet
    act(bus, "sit")
    assert ctrl.posture is S.SIT and ctrl.state_machine.priority is P.BEHAVIOR
    run(ctrl, clock, 4.0, dt=0.5)
    assert ctrl.state_machine.priority is P.AMBIENT  # only a short hold...
    assert ctrl.posture is S.SIT or ctrl.state is not S.IDLE  # ...the sit itself goes on
    run(ctrl, clock, 100.0, dt=0.5)
    assert "idle" in states[states.index("sit") + 1:]  # stood up again by herself


def test_nap_yawns_then_sleeps(ready, bus, clock):
    ctrl = ready()
    states: list[str] = []
    ctrl.state_changed.connect(states.append)
    act(bus, "sleep")
    assert ctrl.state is S.YAWN
    run(ctrl, clock, DEFAULT_ACTION_SECONDS[S.YAWN] + 1.2)
    assert ctrl.posture is S.SLEEP
    assert states == ["yawn", "idle", "sleep"]
    run(ctrl, clock, 60.0, dt=1.0)
    assert ctrl.posture is S.SLEEP  # her own logic wakes her once rested


@pytest.mark.parametrize("why", ["rested", "user_active", "cursor_near", "just_woke"])
def test_no_nap_when_she_should_not(ready, bus, clock, why):
    if why == "rested":
        ctrl = ready(needs=(20.0, 75.0, 70.0, 20.0))
    elif why == "cursor_near":
        ctrl = ready(cursor=FakeCursor(-5000.0, -5000.0))
        cursor = ctrl._cursor_fn
        cursor.pos = QPointF(ctrl.movement.x + 20, ctrl.movement.y - 100)
        run(ctrl, clock, 0.6)
    else:
        ctrl = ready()
    if why == "user_active":
        ctrl.window.clicked.emit()
    if why == "just_woke":
        ctrl.request_state(S.SLEEP, P.USER, 1.0)
        run(ctrl, clock, 1.5)
        ctrl.window.clicked.emit()  # the user woke her
        bus.dispatch_pending()
        clock.advance(NAP_MIN_IDLE_SECONDS + 10.0)  # alone again, but she only just woke up
        run(ctrl, clock, 0.5)
    act(bus, "sleep")
    run(ctrl, clock, 5.0)
    assert ctrl.posture is not S.SLEEP
    if why == "just_woke":
        clock.advance(NAP_WAKE_GRACE_SECONDS)
        run(ctrl, clock, 0.5)
        act(bus, "sleep")
        run(ctrl, clock, 4.0)
        assert ctrl.posture is S.SLEEP  # later on, a nap is fine


def test_user_click_during_the_yawn_cancels_the_nap(ready, bus, clock):
    ctrl = ready()
    act(bus, "sleep")
    assert ctrl.state is S.YAWN
    ctrl.window.clicked.emit()
    bus.dispatch_pending()
    run(ctrl, clock, 8.0)
    assert ctrl.posture is not S.SLEEP


def test_approach_walks_a_short_way_towards_the_cursor(ready, bus, clock):
    ctrl = ready()  # cursor at x=200, far to her left
    x0 = ctrl.movement.x
    act(bus, "approach_cursor")
    assert ctrl.posture is S.WALK
    assert ctrl.movement.target_x == pytest.approx(x0 - APPROACH_MAX_DISTANCE)
    run(ctrl, clock, 10.0, until=lambda: ctrl.posture is not S.WALK)
    assert ctrl.posture is S.IDLE
    assert ctrl.movement.x == pytest.approx(x0 - APPROACH_MAX_DISTANCE, abs=1.0)
    assert ctrl.visual_state().facing is Facing.LEFT  # still looking at the cursor's side
    run(ctrl, clock, 0.5)
    assert ctrl.state_machine.priority is P.AMBIENT


def test_approach_stops_short_and_skips_tiny_steps(ready, bus, clock):
    ctrl = ready()
    cursor = ctrl._cursor_fn
    x0 = ctrl.movement.x
    cursor.pos = QPointF(x0 - APPROACH_GAP - 100.0, 400.0)
    act(bus, "approach_cursor")
    run(ctrl, clock, 10.0, until=lambda: ctrl.posture is not S.WALK)
    assert ctrl.movement.x == pytest.approx(x0 - 100.0, abs=1.0)  # stops short of the cursor
    act(bus, "approach_cursor")  # now she is close enough
    assert ctrl.posture is S.IDLE and not ctrl.movement.walking


def test_approach_stays_on_her_screen(ready, bus, clock):
    screens = FakeScreens([ScreenArea("A", Rect(0, 0, 1920, 1040), primary=True),
                           ScreenArea("B", Rect(1920, 0, 3840, 1040))])
    ctrl = ready(screens=screens, cursor=FakeCursor(3500.0, 500.0))
    act(bus, "approach_cursor")
    run(ctrl, clock, 10.0, until=lambda: ctrl.posture is not S.WALK)
    assert ctrl.movement.x == pytest.approx(ctrl.movement.walk_range[1], abs=1.0)
    assert ctrl.screen.name == "A"
    act(bus, "approach_cursor")  # pressed against the edge: nothing more to do
    assert not ctrl.movement.walking


def test_approach_from_sitting(ready, bus, clock):
    ctrl = ready()
    ctrl.request_state(S.SIT, P.AMBIENT)
    act(bus, "approach_cursor")
    assert ctrl.posture is S.WALK and ctrl.movement.walking


def test_locking_mid_approach_stops_her(ready, bus, clock, settings):
    ctrl = ready()
    act(bus, "approach_cursor")
    run(ctrl, clock, 1.0)
    settings.character.lock_position = True
    ctrl.apply_settings()
    assert ctrl.posture is S.IDLE and not ctrl.movement.walking


# -- landing squash ------------------------------------------------------------------------------------


def test_landing_squash_curve():
    assert landing_squash(0.0, 1.0) == pytest.approx(IMPULSE_SQUASH_MAX)
    assert landing_squash(LANDING_SQUASH_SECONDS / 2, 1.0) < 0.0  # a little rebound stretch
    assert landing_squash(LANDING_SQUASH_SECONDS, 1.0) == 0.0
    assert landing_squash(-0.01, 1.0) == 0.0
    assert landing_squash(0.0, 0.5) == pytest.approx(IMPULSE_SQUASH_MAX / 2)
    assert landing_squash(0.0, 7.0) == pytest.approx(IMPULSE_SQUASH_MAX)
    assert landing_squash(0.0, float("nan")) == 0.0
    assert landing_squash(float("nan"), 1.0) == 0.0


def test_a_drop_lands_with_a_squash(make_controller, clock):
    ctrl = still(make_controller())
    run(ctrl, clock, 0.5)
    ctrl.window.drag_started.emit()
    ctrl.window.dragged.emit(QPointF(800.0, 300.0))
    ctrl.window.drag_finished.emit()
    squashes: list[float] = []
    for _ in range(120):  # 4 s at 30 fps
        run(ctrl, clock, 1 / 30, dt=1 / 30)
        squashes.append(ctrl.visual_state().squash)
    assert ctrl.movement.on_ground
    assert max(squashes) == pytest.approx(IMPULSE_SQUASH_MAX, rel=0.05)  # a hard landing: full squash
    assert min(squashes) < 0.0  # and the rebound
    assert ctrl.visual_state().squash == 0.0 and ctrl._impact_t is None
    assert ctrl.fps == FPS_CALM


def test_squash_frames_run_at_full_rate_and_repaint(make_controller, clock):
    ctrl = still(make_controller())
    run(ctrl, clock, 0.5)
    ctrl.window.drag_started.emit()
    ctrl.window.dragged.emit(QPointF(800.0, 1000.0))  # 40 px above the floor: a gentle drop
    ctrl.window.drag_finished.emit()
    run(ctrl, clock, 2.0, dt=1 / 30, until=lambda: ctrl.movement.on_ground)
    gentle = ctrl.visual_state().squash
    assert 0.0 < gentle < 0.5 * IMPULSE_SQUASH_MAX
    assert ctrl.fps == FPS_FAST
    calls = []
    ctrl.window.update = lambda *a: calls.append(1)  # type: ignore[method-assign]
    run(ctrl, clock, 0.1, dt=1 / 30)
    assert len(calls) >= 3  # repainted every frame of the squash
    run(ctrl, clock, LANDING_SQUASH_SECONDS)
    assert ctrl.fps == FPS_CALM


def test_picking_her_up_mid_squash_clears_it(make_controller, clock):
    ctrl = still(make_controller())
    ctrl.window.drag_started.emit()
    ctrl.window.dragged.emit(QPointF(800.0, 500.0))
    ctrl.window.drag_finished.emit()
    run(ctrl, clock, 3.0, dt=1 / 30, until=lambda: ctrl.visual_state().squash > 0)
    ctrl.window.drag_started.emit()
    assert ctrl.visual_state().squash == 0.0


# -- idle variety ----------------------------------------------------------------------------------------


def test_idle_variety_is_rare_and_only_replaces_fidgets():
    rng = random.Random(11)
    fidget = Decision(S.IDLE, P.AMBIENT, variant="idle_look", facing=Facing.LEFT, reason="fidget")
    results = [idle_variety(fidget, rng) for _ in range(20_000)]
    stretches = sum(1 for d in results if d.state is S.STRETCH)
    assert stretches / len(results) == pytest.approx(IDLE_STRETCH_CHANCE, abs=0.01)
    assert all(d is fidget for d in results if d.state is not S.STRETCH)
    assert all(d.priority is P.AMBIENT for d in results)
    # anything else passes through untouched, without drawing from the rng
    state = rng.getstate()
    for other in (Decision(S.WALK, P.AMBIENT, walk_to_x=10.0), Decision(S.IDLE, P.AMBIENT, reason="stand"),
                  Decision(S.SIT, P.AMBIENT), Decision(None, P.AMBIENT)):
        assert idle_variety(other, rng) is other
    assert rng.getstate() == state


def test_the_variety_rng_leaves_the_behavior_sequence_alone(make_controller):
    ctrl = make_controller(seed=9, start=False)
    assert ctrl._rng.getstate() == random.Random(9).getstate()


def test_she_stretches_now_and_then_when_nothing_else_happens(make_controller, clock, settings):
    settings.character.walking_enabled = False
    settings.behavior.sleep_after_idle_minutes = 240
    ctrl = make_controller(seed=4, needs=(20.0, 80.0, 70.0, 20.0))
    states: list[str] = []
    ctrl.state_changed.connect(states.append)
    run(ctrl, clock, 30 * 60.0, dt=1.0)
    stretches = states.count("stretch")
    assert 1 <= stretches <= 8, states  # rare: a few times in half an hour (seeded: deterministic)
    assert stretches < states.count("sit")  # she sits far more often than she stretches


def test_she_stretches_after_waking_up_by_herself(make_controller, clock, settings):
    settings.behavior.sleep_after_idle_minutes = 240
    ctrl = make_controller(needs=(20.0, 100.0, 70.0, 20.0))
    ctrl.request_state(S.SLEEP, P.BEHAVIOR)
    run(ctrl, clock, 2.0)
    states: list[str] = []
    ctrl.state_changed.connect(states.append)
    clock.advance(21 * 60.0)  # slept long enough, fully rested
    run(ctrl, clock, 3.0)
    assert states[:2] == ["idle", "stretch"]
    run(ctrl, clock, DEFAULT_ACTION_SECONDS[S.STRETCH])
    # the stretch is over and her own behavior is free again (it may well sit her down)
    assert ctrl.state is not S.STRETCH and ctrl.state_machine.priority is P.AMBIENT


# -- blinks land on the frame, at any frame rate -----------------------------------------------------


@pytest.fixture
def blink_pack(tmp_path: Path) -> CharacterPack:
    root = tmp_path / "blink"
    (root / "frames").mkdir(parents=True)
    for frame in ("front_0", "front_1"):
        shutil.copy(MIA_DIR / "frames" / f"{frame}.png", root / "frames" / f"{frame}.png")
    source = json.loads((MIA_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "format": 1, "name": "Blink", "reference_height": source["reference_height"],
        "frames": {k: source["frames"][k] for k in ("front_0", "front_1")},
        "clips": {"idle": {"frames": [{"id": "front_0", "ms": 900}, {"id": "front_1", "ms": 140}],
                           "loop": True, "facing": "front", "fx": {"breathe": 0.012}}},
        "states": {"idle": {"clip": "idle"}},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return CharacterPack.load(root)


def tick_like_the_timer(ctrl: CharacterController, clock, seconds: float, closed) -> list[float]:
    """Step at the frame timer's own intervals; returns how long each blink was seen (ms)."""
    elapsed = 0.0
    shown_at: float | None = None
    blinks: list[float] = []
    types = set()
    while elapsed < seconds * 1000.0:
        interval = ctrl._frame_timer.interval()
        types.add(ctrl._frame_timer.timerType())
        clock.advance(interval / 1000.0)
        ctrl.step(interval / 1000.0)
        elapsed += interval
        if closed(ctrl) and shown_at is None:
            shown_at = elapsed
        elif not closed(ctrl) and shown_at is not None:
            blinks.append(elapsed - shown_at)
            shown_at = None
    assert Qt.TimerType.PreciseTimer in types and Qt.TimerType.CoarseTimer in types
    return blinks


@pytest.mark.parametrize("eco", [False, True])
def test_sprite_blinks_last_exactly_their_frame(make_controller, clock, bus, blink_pack, eco):
    ctrl = still(make_controller(pack=blink_pack))
    if eco:
        bus.publish(SystemLoadChanged(high=True, cpu_percent=95.0))
        bus.dispatch_pending()
        assert ctrl.fps == FPS_CALM // 2  # 200 ms ticks: a 140 ms blink used to be skipped or doubled
    blinks = tick_like_the_timer(ctrl, clock, 10.5, lambda c: c.player.sample().frame_id == "front_1")
    assert len(blinks) == 10  # one per 1.04 s cycle, none skipped
    assert all(abs(ms - 140.0) <= 1.5 for ms in blinks), blinks


@pytest.mark.parametrize("eco", [False, True])
def test_vector_blinks_last_exactly_130_ms(make_controller, clock, bus, eco):
    ctrl = still(make_controller(pack=None))
    if eco:
        bus.publish(SystemLoadChanged(high=True, cpu_percent=95.0))
        bus.dispatch_pending()
    blinks = tick_like_the_timer(ctrl, clock, 16.0, lambda c: renderers._blinking(c._anim_t))
    assert len(blinks) == 4
    assert all(abs(ms - 130.0) <= 1.5 for ms in blinks), blinks


def test_fast_states_do_not_need_boundary_ticks(make_controller, clock):
    ctrl = still(make_controller())
    ctrl.request_state(S.DANCE, P.USER)
    run(ctrl, clock, 0.1)
    assert ctrl.fps == FPS_FAST
    assert ctrl._frame_timer.interval() == round(1000 / FPS_FAST)


# -- the new states through the controller, both renderers -------------------------------------------------


@pytest.mark.parametrize("pack", ["mia", None])
@pytest.mark.parametrize("state", [S.JUMP, S.DANCE, S.EXCITED, S.PROUD, S.STRETCH])
def test_new_states_render_in_her_window(make_controller, clock, pack, state):
    ctrl = still(make_controller(pack=pack))
    assert ctrl.request_state(state, P.USER)
    run(ctrl, clock, 0.4)
    assert ctrl.state is state
    assert ctrl.player.clip.name == state.value
    assert alpha_bbox(ctrl.window.grab().toImage()) is not None
    fast = state in (S.JUMP, S.DANCE, S.EXCITED)
    assert ctrl.fps == (FPS_FAST if fast else FPS_CALM)
    if pack == "mia":
        assert ctrl._default_portrait(state)  # her speech bubble has a face for it
