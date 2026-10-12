"""CharacterController with the real mia pack, a fake clock and fake screens."""

from __future__ import annotations

import itertools
import math

import pytest
from PySide6.QtCore import QPoint, QPointF

from chibi.character.animation import CharacterPack
from chibi.character.movement import Rect
from chibi.character.states import CharacterState, Facing, StatePriority
from chibi.core.config import Settings
from chibi.core.events import (
    CharacterInteraction,
    FullscreenChanged,
    GameEnded,
    GameStarted,
    InteractionKind,
    InteractionOutcome,
    ObservationStopped,
    SettingsChanged,
    SpeechRequested,
    SystemLoadChanged,
    UserIdle,
    UserReturned,
)
from chibi.desktop import overlays
from chibi.desktop.controller import (
    FPS_CALM,
    FPS_FAST,
    FPS_SLEEP,
    OVERLAY_SIZE,
    POSITION_KEY,
    SPEECH_QUEUE_MAX,
    CharacterController,
    ScreenArea,
    decode_position,
    default_position,
    encode_position,
    screen_for_point,
    speech_seconds,
)
from chibi.desktop.renderers import SpriteRenderer, VectorRenderer
from chibi.memory.repositories import StateRepository
from tests.desktop.helpers import FakeCursor, FakeScreens, alpha_bbox, run

S = CharacterState
P = StatePriority
K = InteractionKind

SCREEN_A = ScreenArea("A", Rect(0, 0, 1920, 1040), primary=True)
SCREEN_B = ScreenArea("B", Rect(1920, -200, 3200, 824), dpr=1.5)


@pytest.fixture
def interactions(bus) -> list[InteractionKind]:
    seen: list[InteractionKind] = []
    bus.subscribe(CharacterInteraction, lambda e: seen.append(e.kind))
    return seen


def settle(ctrl: CharacterController, clock) -> None:
    """Let a just-started character finish anything in progress."""
    run(ctrl, clock, 2.0)


# -- pure helpers ---------------------------------------------------------------------------------


def test_speech_seconds_bounds():
    assert speech_seconds("") == 3.0
    assert speech_seconds("x" * 40) == pytest.approx(2.2 + 0.055 * 40)
    assert speech_seconds("x" * 1000) == 12.0


def test_screen_for_point_contains_or_nearest():
    screens = [SCREEN_A, SCREEN_B]
    assert screen_for_point(screens, 100, 100) is SCREEN_A
    assert screen_for_point(screens, 2500, 0) is SCREEN_B
    assert screen_for_point(screens, 2500, 900) is SCREEN_B  # below B's work area: nearest
    assert screen_for_point(screens, -400, 500) is SCREEN_A
    assert screen_for_point([], 0, 0) is None


def test_position_codec_round_trip():
    data = encode_position(500.0, 1040.0, SCREEN_A)
    assert data == {"screen": "A", "rx": pytest.approx(500 / 1920, abs=1e-6), "y_from_bottom": 0.0,
                    "x": 500.0, "y": 1040.0}
    x, y, screen = decode_position(data, [SCREEN_A, SCREEN_B])
    assert x == pytest.approx(500.0, abs=0.01)
    assert (y, screen) == (1040.0, SCREEN_A)


def test_position_follows_resolution_change():
    data = encode_position(2560.0, 824.0, SCREEN_B)  # middle of B, on its floor
    bigger_b = ScreenArea("B", Rect(1920, 0, 4480, 1400))
    x, y, screen = decode_position(data, [SCREEN_A, bigger_b])
    assert screen is bigger_b
    assert x == pytest.approx(1920 + 0.5 * 2560, abs=0.1) and y == 1400


@pytest.mark.parametrize("data", [
    None, "garbage", [], {"screen": "gone", "x": 99999, "y": 99999},
    {"screen": "A", "rx": math.nan, "y_from_bottom": 0},
    {"screen": "A", "rx": 7, "y_from_bottom": 0, "x": "a", "y": None},
    {"rx": True, "y_from_bottom": 0},
])
def test_invalid_positions_are_rejected(data):
    assert decode_position(data, [SCREEN_A]) is None


def test_absolute_position_used_when_screen_was_renamed():
    data = {"screen": "old-name", "rx": 0.5, "y_from_bottom": 0, "x": 2000.0, "y": 824.0}
    x, y, screen = decode_position(data, [SCREEN_A, SCREEN_B])
    assert (x, y, screen) == (2000.0, 824.0, SCREEN_B)


def test_default_position_is_bottom_right_of_primary():
    x, y, screen = default_position([SCREEN_B, SCREEN_A], 66.0)
    assert screen is SCREEN_A
    assert y == SCREEN_A.area.bottom
    assert SCREEN_A.area.right - 300 < x < SCREEN_A.area.right - 66
    assert default_position([], 66.0)[2].primary


# -- startup, layout ---------------------------------------------------------------------------


def test_starts_visible_on_the_floor_with_feet_at_the_window_anchor(make_controller, clock):
    ctrl = make_controller()
    assert ctrl.is_visible()
    mv = ctrl.movement
    assert mv.y == SCREEN_A.area.bottom and mv.on_ground
    geo = ctrl.window.geometry()
    anchor = ctrl.window.anchor()
    assert geo.x() + anchor.x() == round(mv.x)
    assert geo.y() + anchor.y() == round(mv.y)
    assert anchor.x() == geo.width() / 2  # horizontally centred
    assert ctrl.state is S.IDLE
    assert isinstance(ctrl.renderer, SpriteRenderer)
    assert ctrl.fps == FPS_CALM and ctrl.frame_timer_active


def test_window_has_room_for_every_state_and_overlay(make_controller):
    ctrl = make_controller()
    h = ctrl.height_px
    e = ctrl.renderer.full_extents(h)
    head = ctrl.renderer.head_bounds(h)
    anchor = ctrl.window.anchor()
    w, wh = ctrl.window.width(), ctrl.window.height()
    reach = [r * OVERLAY_SIZE * h for r in overlays.OVERLAY_REACH]
    assert anchor.x() >= max(e.left, head.left + reach[0])
    assert w - anchor.x() >= max(e.right, head.right + reach[1])
    assert anchor.y() >= max(e.up, head.up + reach[2])
    assert wh - anchor.y() >= max(e.down, 0)


def test_renders_into_the_window(make_controller, clock):
    ctrl = make_controller()
    ctrl.request_state(S.ANGRY, P.USER)  # overlay too
    run(ctrl, clock, 0.3)
    image = ctrl.window.grab().toImage()
    assert alpha_bbox(image) is not None


def test_vector_fallback_without_pack(make_controller, clock):
    ctrl = make_controller(pack=None)
    assert isinstance(ctrl.renderer, VectorRenderer)
    assert ctrl.portrait_pixmap("smile") is None
    ctrl.say("Я тут!")
    assert ctrl.bubble.is_showing()  # still talks, just without a portrait
    run(ctrl, clock, 0.5)
    assert alpha_bbox(ctrl.window.grab().toImage()) is not None


def test_renderer_setting_switches_live(make_controller, settings, bus, clock):
    ctrl = make_controller()
    settings.character.renderer = "vector"
    bus.publish(SettingsChanged(section="character"))
    bus.dispatch_pending()
    assert isinstance(ctrl.renderer, VectorRenderer)
    run(ctrl, clock, 0.3)
    settings.character.renderer = "sprite"
    ctrl.apply_settings()
    assert isinstance(ctrl.renderer, SpriteRenderer)
    assert ctrl.player.clip is ctrl._pack.clip_for(ctrl.state)


def test_scale_setting_resizes_window_and_speed(make_controller, settings):
    ctrl = make_controller()
    width = ctrl.window.width()
    settings.character.scale = 2.0
    ctrl.apply_settings()
    assert ctrl.height_px == 440
    assert ctrl.window.width() > 1.8 * width
    geo = ctrl.window.geometry()
    assert geo.y() + ctrl.window.anchor().y() == round(ctrl.movement.y)  # feet stay put


# -- walking --------------------------------------------------------------------------------------


def test_autonomous_walk_reaches_its_target_and_stops(make_controller, clock):
    ctrl = make_controller(seed=3)
    states: list[str] = []
    ctrl.state_changed.connect(states.append)
    run(ctrl, clock, 120, until=lambda: ctrl.posture is S.WALK)
    assert ctrl.posture is S.WALK, "no stroll within two minutes"
    assert ctrl.fps == FPS_FAST
    target = ctrl.movement.target_x
    start = ctrl.movement.x
    xs = []
    run(ctrl, clock, 60, until=lambda: (xs.append(ctrl.movement.x) or False) or ctrl.posture is not S.WALK)
    assert ctrl.posture is S.IDLE
    assert abs(ctrl.movement.x - target) <= 1.0
    direction = math.copysign(1, target - start)
    assert all((b - a) * direction >= -1e-9 for a, b in itertools.pairwise(xs))  # never backwards
    assert states[:2] == ["walk", "idle"]
    assert ctrl.window.geometry().x() + ctrl.window.anchor().x() == round(ctrl.movement.x)
    run(ctrl, clock, 0.2)
    assert ctrl.fps == FPS_CALM


def test_no_walking_when_disabled_or_locked(make_controller, settings, clock):
    settings.character.walking_enabled = False
    ctrl = make_controller(seed=3)
    x = ctrl.movement.x
    run(ctrl, clock, 180)
    assert ctrl.movement.x == x


def test_call_over_walks_to_the_cursor_then_waves(make_controller, clock, bus, interactions):
    cursor = FakeCursor(700.0, 300.0)
    ctrl = make_controller(cursor=cursor)
    settle(ctrl, clock)
    ctrl.call_over()
    bus.dispatch_pending()
    assert interactions == [K.CALL]
    assert ctrl.posture in (S.WALK, S.RUN)
    # The reply's gesture arrives while she walks: it must not stop the walk.
    bus.publish(SpeechRequested(text="Біжу-біжу!", state="happy", solicited=True))
    bus.dispatch_pending()
    assert ctrl.posture in (S.WALK, S.RUN) and ctrl.bubble.is_showing()
    run(ctrl, clock, 30, until=lambda: ctrl.posture not in (S.WALK, S.RUN))
    assert abs(ctrl.movement.x - 700.0) <= 1.0
    assert ctrl.state is S.WAVE


def test_call_over_runs_when_far_and_waves_when_close(make_controller, clock):
    cursor = FakeCursor(100.0, 300.0)
    ctrl = make_controller(cursor=cursor)
    settle(ctrl, clock)
    ctrl.call_over()
    assert ctrl.posture is S.RUN
    run(ctrl, clock, 30, until=lambda: ctrl.posture is not S.RUN)
    cursor.pos = QPointF(ctrl.movement.x + 10, 300)
    ctrl.call_over()
    assert ctrl.state is S.WAVE  # already there


def test_call_over_when_locked_only_waves(make_controller, settings, clock):
    settings.character.lock_position = True
    ctrl = make_controller(cursor=FakeCursor(100.0, 300.0))
    x = ctrl.movement.x
    ctrl.call_over()
    run(ctrl, clock, 5)
    assert ctrl.movement.x == x


# -- speech ---------------------------------------------------------------------------------------


def test_speech_shows_bubble_and_talks_then_returns(make_controller, clock, bus):
    ctrl = make_controller()
    settle(ctrl, clock)
    bus.publish(SpeechRequested(text="Привіт! Як справи?", solicited=True))
    bus.dispatch_pending()
    assert ctrl.bubble.is_showing()
    assert ctrl.bubble.text == "Привіт! Як справи?"
    assert ctrl.state is S.TALK
    assert ctrl.bubble._portrait is not None  # the talk state's default portrait
    seconds = speech_seconds("Привіт! Як справи?")
    run(ctrl, clock, seconds - 0.3)
    assert ctrl.state is S.TALK  # talk lasts as long as the bubble
    run(ctrl, clock, 0.5)
    assert ctrl.state is S.IDLE
    assert not ctrl.bubble.is_showing()


def test_bubble_follows_her_head(make_controller, clock):
    ctrl = make_controller()
    ctrl.say("Дивись!")
    p = ctrl.bubble.placement
    tip = QPointF(p.window.x() + p.tip.x(), p.window.y() + p.tip.y())
    head = ctrl.renderer.head_point(ctrl.window.anchor(), ctrl.height_px, ctrl.visual_state())
    head_global = QPointF(ctrl.window.x() + head.x(), ctrl.window.y() + head.y())
    assert abs(tip.x() - head_global.x()) <= 1.0
    assert head_global.y() - ctrl.height_px * 0.3 < tip.y() < head_global.y()  # above the head centre


def test_other_states_are_capped(make_controller, clock):
    ctrl = make_controller()
    ctrl.say("Ха-ха, це було дуже-дуже смішно, розкажи ще раз, будь ласка!", state="laugh")
    assert ctrl.state is S.LAUGH
    run(ctrl, clock, 2.5 + 1.0 + 0.2)  # DEFAULT_ACTION_SECONDS[LAUGH] + 1
    assert ctrl.state is S.IDLE
    assert ctrl.bubble.is_showing()  # the line itself stays longer


def test_empty_text_plays_the_state_only(make_controller):
    ctrl = make_controller()
    ctrl.say("", state="happy")
    assert ctrl.state is S.HAPPY
    assert not ctrl.bubble.isVisible()


def test_unsolicited_lines_queue_and_solicited_replace(make_controller, clock):
    ctrl = make_controller()
    ctrl.say("Перша", solicited=False)
    assert ctrl.bubble.text == "Перша"
    ctrl.say("Друга", solicited=False)
    assert ctrl.bubble.text == "Перша" and ctrl.queued_lines == 1
    ctrl.say("Відповідь", solicited=True)
    assert ctrl.bubble.text == "Відповідь"  # the user asked: shown at once
    run(ctrl, clock, speech_seconds("Відповідь") + 0.2)
    assert ctrl.bubble.text == "Друга" and ctrl.bubble.is_showing()
    assert ctrl.queued_lines == 0
    run(ctrl, clock, speech_seconds("Друга") + 0.2)
    assert not ctrl.bubble.is_showing()


def test_queue_drops_the_oldest(make_controller):
    ctrl = make_controller()
    ctrl.say("Зараз", solicited=False)
    for text in ("a1", "a2", "a3", "a4"):
        ctrl.say(text, solicited=False)
    assert ctrl.queued_lines == SPEECH_QUEUE_MAX
    assert [line.text for line in ctrl._queue] == ["a3", "a4"]


def test_bubbles_disabled_keeps_the_gesture(make_controller, settings):
    settings.character.show_bubbles = False
    ctrl = make_controller()
    ctrl.say("Ой!", state="surprised")
    assert ctrl.state is S.SURPRISED
    assert not ctrl.bubble.isVisible()


def test_unknown_state_in_speech_still_shows_text(make_controller):
    ctrl = make_controller()
    ctrl.say("Хм.", state="dancing")
    assert ctrl.bubble.text == "Хм."
    assert ctrl.state is S.IDLE


# -- hiding, fullscreen, eco ---------------------------------------------------------------------


def test_fullscreen_hides_and_stops_the_frame_timer(make_controller, clock, bus):
    ctrl = make_controller()
    ctrl.say("Привіт", solicited=False)
    bus.publish(FullscreenChanged(active=True))
    bus.dispatch_pending()
    assert not ctrl.is_visible()
    assert not ctrl.frame_timer_active and ctrl.fps == 0
    assert ctrl._hidden_timer.isActive()  # state machine keeps ticking, slowly
    assert not ctrl.bubble.isVisible()
    ctrl.say("Ніхто не почує", solicited=False)
    assert ctrl.queued_lines == 0 and not ctrl.bubble.isVisible()  # dropped while hidden
    ctrl.say("", state="happy", solicited=True)
    assert ctrl.state is S.HAPPY  # solicited gesture still applies
    run(ctrl, clock, 5, dt=1.0)
    assert ctrl.state is S.IDLE
    bus.publish(FullscreenChanged(active=False))
    bus.dispatch_pending()
    assert ctrl.is_visible() and ctrl.frame_timer_active
    assert not ctrl._hidden_timer.isActive()


def test_fullscreen_ignored_when_setting_off(make_controller, settings, bus):
    settings.behavior.hide_on_fullscreen = False
    ctrl = make_controller()
    bus.publish(FullscreenChanged(active=True))
    bus.dispatch_pending()
    assert ctrl.is_visible()


def test_hidden_by_user(make_controller):
    ctrl = make_controller()
    ctrl.set_hidden_by_user(True)
    assert not ctrl.is_visible() and ctrl.fps == 0
    ctrl.set_hidden_by_user(False)
    assert ctrl.is_visible() and ctrl.fps > 0


def test_eco_mode_halves_the_frame_rate(make_controller, settings, bus, clock):
    ctrl = make_controller()
    assert ctrl.fps == FPS_CALM
    bus.publish(SystemLoadChanged(high=True, cpu_percent=95.0))
    bus.dispatch_pending()
    assert ctrl.eco
    assert ctrl.fps == FPS_CALM // 2
    assert ctrl._frame_timer.interval() == round(1000 / (FPS_CALM // 2))
    bus.publish(SystemLoadChanged(high=False, cpu_percent=20.0, on_battery_saver=True))
    bus.dispatch_pending()
    assert ctrl.eco  # battery saver counts too
    settings.behavior.auto_eco_mode = False
    ctrl.apply_settings()
    assert not ctrl.eco and ctrl.fps == FPS_CALM


def test_frame_rate_follows_the_state(make_controller, clock):
    ctrl = make_controller()
    ctrl.request_state(S.SLEEP, P.USER)
    run(ctrl, clock, 0.2)
    assert ctrl.fps == FPS_SLEEP
    ctrl.request_state(S.HAPPY, P.USER)  # hop: fast
    run(ctrl, clock, 0.2)
    assert ctrl.fps == FPS_FAST
    assert ctrl._frame_timer.timerType().name == "PreciseTimer"


def test_repaints_only_when_something_changed(make_controller, tiny_pack_dir, clock):
    ctrl = make_controller(pack=CharacterPack.load(tiny_pack_dir))
    calls = []
    ctrl.window.update = lambda *a: calls.append(1)  # type: ignore[method-assign]
    run(ctrl, clock, 0.3)  # settle the first paint
    calls.clear()
    for _ in range(5):  # same frame, no effects, nothing moving
        ctrl.step(0.0)
    run(ctrl, clock, 0.3)
    assert calls == []
    run(ctrl, clock, 1.0)  # crosses into the clip's second frame
    assert calls


# -- dragging ---------------------------------------------------------------------------------------


def test_drag_sequence(make_controller, clock, bus, interactions, db):
    repo = StateRepository(db, clock)
    ctrl = make_controller(repo=repo)
    settle(ctrl, clock)
    ctrl.window.drag_started.emit()
    assert ctrl.state is S.DRAGGED
    ctrl.window.dragged.emit(QPointF(800.0, 500.0))
    assert (ctrl.movement.x, ctrl.movement.y) == (800.0, 500.0)
    assert ctrl.window.geometry().x() + ctrl.window.anchor().x() == 800
    run(ctrl, clock, 1.0)
    assert ctrl.state is S.DRAGGED and ctrl.movement.y == 500.0  # held: no falling
    assert ctrl.fps == FPS_FAST
    ctrl.window.drag_finished.emit()
    assert ctrl.state is S.IDLE
    assert ctrl.movement.falling
    run(ctrl, clock, 5.0, until=lambda: ctrl.movement.on_ground)
    assert ctrl.movement.y == SCREEN_A.area.bottom
    bus.dispatch_pending()
    assert interactions[:2] == [K.DRAG_START, K.DRAG_END]
    saved = repo.get(POSITION_KEY)
    assert saved["screen"] == "A" and saved["y_from_bottom"] == 0.0
    assert saved["x"] == pytest.approx(800.0)


def test_drop_without_gravity_stays_mid_air(make_controller, settings, clock):
    settings.character.gravity = False
    ctrl = make_controller()
    ctrl.window.drag_started.emit()
    ctrl.window.dragged.emit(QPointF(600.0, 400.0))
    ctrl.window.drag_finished.emit()
    run(ctrl, clock, 3.0)
    assert ctrl.movement.y == 400.0 and not ctrl.movement.falling


def test_drag_to_another_monitor(make_controller, clock):
    ctrl = make_controller(screens=FakeScreens([SCREEN_A, SCREEN_B]))
    ctrl.window.drag_started.emit()
    ctrl.window.dragged.emit(QPointF(2500.0, 100.0))
    assert ctrl.screen is SCREEN_B
    ctrl.window.drag_finished.emit()
    run(ctrl, clock, 5.0, until=lambda: ctrl.movement.on_ground)
    assert ctrl.movement.y == SCREEN_B.area.bottom
    assert ctrl.movement.area == SCREEN_B.area


def test_locked_ignores_drags(make_controller, settings):
    settings.character.lock_position = True
    ctrl = make_controller()
    assert ctrl.window.locked
    ctrl.window.drag_started.emit()
    assert ctrl.state is not S.DRAGGED


# -- position persistence -------------------------------------------------------------------------


def test_position_round_trip(make_controller, clock, db):
    repo = StateRepository(db, clock)
    first = make_controller(repo=repo, screens=FakeScreens([SCREEN_A, SCREEN_B]))
    first.window.drag_started.emit()
    first.window.dragged.emit(QPointF(2300.0, 824.0))
    first.window.drag_finished.emit()
    run(first, clock, 1.0)
    first.shutdown()
    second = make_controller(repo=repo, screens=FakeScreens([SCREEN_A, SCREEN_B]))
    assert second.screen.name == "B"
    assert second.movement.x == pytest.approx(2300.0, abs=0.1)
    assert second.movement.y == 824.0


def test_off_screen_position_falls_back(make_controller, clock, db):
    repo = StateRepository(db, clock)
    repo.set(POSITION_KEY, {"screen": "unplugged", "x": 99999, "y": 99999})
    ctrl = make_controller(repo=repo)
    x, y, _ = default_position([SCREEN_A], ctrl._half_width)
    assert (ctrl.movement.x, ctrl.movement.y) == (x, y)
    assert ctrl.screen == SCREEN_A


# -- interactions and events ---------------------------------------------------------------------


def test_click_publishes_click_or_wake(make_controller, clock, bus, interactions):
    ctrl = make_controller()
    ctrl.window.clicked.emit()
    ctrl.request_state(S.SLEEP, P.USER, 1.0)
    run(ctrl, clock, 2.0)
    assert ctrl.posture is S.SLEEP
    ctrl.window.clicked.emit()
    bus.dispatch_pending()
    assert interactions == [K.CLICK, K.WAKE]
    assert ctrl.posture is S.IDLE


def test_rest_and_wake_outcomes(make_controller, clock, bus):
    ctrl = make_controller()
    states: list[str] = []
    ctrl.state_changed.connect(states.append)
    bus.publish(InteractionOutcome(kind=K.REST, outcome="sleep"))
    bus.dispatch_pending()
    assert ctrl.posture is S.SLEEP
    run(ctrl, clock, 10.0)
    assert ctrl.posture is S.SLEEP  # nothing ambient wakes her
    bus.publish(InteractionOutcome(kind=K.WAKE, outcome="grumpy"))
    bus.dispatch_pending()
    assert ctrl.posture is S.IDLE
    assert states[:2] == ["sleep", "idle"]
    bus.publish(InteractionOutcome(kind=K.REST, outcome="not_tired"))
    bus.dispatch_pending()
    assert ctrl.posture is S.IDLE


def test_user_returned_wakes_her_with_a_wave(make_controller, clock, bus):
    ctrl = make_controller()
    ctrl.request_state(S.SLEEP, P.USER, 1.0)
    run(ctrl, clock, 2.0)
    bus.publish(UserReturned(away_seconds=1200.0))
    bus.dispatch_pending()
    assert ctrl.state is S.WAVE and ctrl.posture is S.IDLE
    assert ctrl.player.clip.name == "greet"  # the pack's greeting variant


def test_games_set_the_gaming_posture(make_controller, clock, bus):
    ctrl = make_controller()
    bus.publish(GameStarted(game_id="minecraft", title="Minecraft", process_name="javaw.exe"))
    bus.publish(GameStarted(game_id="cs2", title="CS2", process_name="cs2.exe"))
    bus.dispatch_pending()
    assert ctrl.posture is S.GAMING
    ctrl.say("Ого!", state="surprised", solicited=False)
    run(ctrl, clock, 8.0)
    assert ctrl.posture is S.GAMING and ctrl.state is S.GAMING  # back to watching after the gesture
    bus.publish(GameEnded(game_id="cs2", title="CS2", duration_seconds=60))
    bus.dispatch_pending()
    assert ctrl.posture is S.GAMING  # another game is still running
    bus.publish(GameEnded(game_id="minecraft", title="Minecraft", duration_seconds=600))
    bus.dispatch_pending()
    assert ctrl.posture is S.IDLE


def test_idle_time_tracks_interactions_and_user_input(make_controller, clock, bus):
    ctrl = make_controller()
    run(ctrl, clock, 30.0, dt=1.0)
    assert ctrl.idle_seconds() == pytest.approx(30.0, abs=1.01)
    ctrl.window.clicked.emit()
    assert ctrl.idle_seconds() == 0.0
    bus.publish(UserReturned(away_seconds=600))
    bus.dispatch_pending()
    run(ctrl, clock, 100.0, dt=1.0)
    assert ctrl.idle_seconds() == 0.0  # the user is active at the computer
    bus.publish(UserIdle(idle_seconds=300.0))
    bus.dispatch_pending()
    # idle since the last interaction (the click 100 s ago), not the 300 s of no input
    assert ctrl.idle_seconds() == pytest.approx(100.0, abs=1.0)
    run(ctrl, clock, 50.0, dt=1.0)
    bus.publish(UserReturned(away_seconds=50))
    bus.dispatch_pending()
    assert ctrl.idle_seconds() == 0.0
    bus.publish(ObservationStopped())  # no input info any more: back to interactions only
    bus.dispatch_pending()
    assert ctrl.idle_seconds() == pytest.approx(150.0, abs=1.0)


def test_falls_asleep_when_idle_and_tired(make_controller, clock, settings):
    settings.behavior.sleep_after_idle_minutes = 2
    ctrl = make_controller(needs=(20, 15, 60, 20))  # tired
    run(ctrl, clock, 300.0, dt=0.5, until=lambda: ctrl.posture is S.SLEEP)
    assert ctrl.posture is S.SLEEP
    run(ctrl, clock, 0.3)
    assert ctrl.fps == FPS_SLEEP
    assert ctrl.visual_state().overlay == "zzz"


def test_fidget_variant_plays_once(make_controller, clock):
    ctrl = make_controller()
    assert ctrl.request_state(S.IDLE, P.AMBIENT, variant="idle_look")
    assert ctrl.player.clip.name == "idle_look"
    run(ctrl, clock, 3.0)
    assert ctrl.player.clip.name == "idle"


def test_portraits_are_cached(make_controller):
    ctrl = make_controller()
    smile = ctrl.portrait_pixmap("smile")
    assert smile is not None and not smile.isNull()
    assert ctrl.portrait_pixmap("smile") is smile
    assert ctrl.portrait_pixmap("no-such-portrait") is None
    assert ctrl.portrait_pixmap("") is None


def test_context_menu_is_relayed(make_controller):
    ctrl = make_controller()
    got = []
    ctrl.context_menu_requested.connect(got.append)
    ctrl.window.context_menu_requested.emit(QPoint(10, 20))
    assert got == [QPoint(10, 20)]


def test_facing_from_decisions(make_controller, clock):
    ctrl = make_controller()
    ctrl.request_state(S.CURIOUS, P.AMBIENT)
    ctrl._facing = Facing.RIGHT
    assert ctrl.visual_state().facing is Facing.RIGHT


def test_shutdown_stops_everything(make_controller, clock, bus, db):
    repo = StateRepository(db, clock)
    ctrl = make_controller(repo=repo)
    ctrl.shutdown()
    assert not ctrl.frame_timer_active and not ctrl._hidden_timer.isActive()
    assert not ctrl.is_visible()
    assert repo.get(POSITION_KEY) is not None
    bus.publish(SpeechRequested(text="Ще тут?", solicited=True))
    bus.dispatch_pending()
    assert not ctrl.bubble.isVisible()
    ctrl.step(1.0)  # harmless after shutdown
    ctrl.shutdown()  # idempotent


def test_own_window_ids_do_not_create_windows(make_controller):
    ctrl = make_controller(start=False)
    assert ctrl.own_window_ids() == set()
    ctrl.start()
    assert ctrl.own_window_ids()  # the character window exists now


def test_bad_needs_and_settings_callbacks_are_survived(qapp, bus, clock, db, mia_pack):
    calls = {"n": 0}

    def flaky_settings() -> Settings:
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("settings store locked")
        return Settings()

    ctrl = CharacterController(bus, flaky_settings, clock, StateRepository(db, clock), mia_pack,
                               lambda: (math.nan, 1, 2, 3), cursor_pos=FakeCursor(),
                               screens=FakeScreens([SCREEN_A]))
    try:
        ctrl.start()
        run(ctrl, clock, 5.0)
        assert ctrl.is_visible()
    finally:
        ctrl.shutdown()


# -- priority holds never stick (regressions) -----------------------------------------------------


def test_interrupted_call_walk_does_not_leave_user_priority(make_controller, settings, clock):
    ctrl = make_controller(cursor=FakeCursor(300.0, 300.0))
    settle(ctrl, clock)
    ctrl.call_over()
    run(ctrl, clock, 1.0)
    assert ctrl.posture in (S.WALK, S.RUN)
    settings.character.lock_position = True
    ctrl.apply_settings()  # locking stops the walk
    assert ctrl.posture is S.IDLE
    run(ctrl, clock, 0.3)
    assert ctrl.state_machine.priority is P.AMBIENT  # her own behavior can move her again
    assert ctrl.visual_state().facing is Facing.LEFT  # still looking where she was going


def test_completed_call_walk_returns_to_ambient(make_controller, clock):
    ctrl = make_controller(cursor=FakeCursor(1200.0, 300.0))
    settle(ctrl, clock)
    ctrl.call_over()
    run(ctrl, clock, 40.0, until=lambda: ctrl.state is S.WAVE)
    run(ctrl, clock, 4.0)
    assert ctrl.state is S.IDLE
    assert ctrl.state_machine.priority is P.AMBIENT


def test_posture_gesture_from_speech_holds_priority_briefly(make_controller, clock, bus):
    ctrl = make_controller()
    bus.publish(SpeechRequested(text="", state="sit", solicited=True))
    bus.dispatch_pending()
    assert ctrl.posture is S.SIT
    assert ctrl.state_machine.priority is P.USER
    run(ctrl, clock, 3.5)
    assert ctrl.posture is S.SIT  # the posture persists...
    assert ctrl.state_machine.priority is P.AMBIENT  # ...but only briefly at USER priority


def test_display_change_drops_cached_pixmaps(make_controller, clock):
    ctrl = make_controller()
    run(ctrl, clock, 0.2)
    ctrl.window.grab()  # paints: fills the cache
    assert ctrl.renderer.cached_pixmaps > 0
    ctrl.window.display_changed.emit()
    assert ctrl.renderer.cached_pixmaps == 0
