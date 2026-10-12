"""CharacterController and activity observation: games, user input, the kill
switch, quiet mode.  Regressions from the review (integ-1/2/4/5, QT-1,
perf-1/3, privsec-2)."""

from __future__ import annotations

import logging

import pytest

from chibi.character.states import CharacterState, StatePriority
from chibi.core.events import (
    ActivityCategory,
    AppInfo,
    DoNotDisturbChanged,
    ForegroundAppChanged,
    GameEnded,
    GameStarted,
    ObservationStopped,
    Permission,
    PermissionChanged,
    QuietModeChanged,
    UserIdle,
    UserReturned,
)
from chibi.desktop.controller import CharacterController
from chibi.memory.repositories import StateRepository
from tests.desktop.helpers import FakeCursor, FakeScreens, run

S = CharacterState
P = StatePriority


class FakePermissions:
    def __init__(self, granted: bool = True) -> None:
        self.granted = granted

    def is_granted(self, permission: Permission) -> bool:
        return self.granted and permission is Permission.APP_METADATA


@pytest.fixture
def make_guarded(qapp, bus, clock, db, settings, mia_pack):
    """A started controller that checks APP_METADATA through *permissions*."""
    from chibi.character.movement import Rect
    from chibi.desktop.controller import ScreenArea

    created: list[CharacterController] = []

    def build(permissions: FakePermissions) -> CharacterController:
        ctrl = CharacterController(
            bus, lambda: settings, clock, StateRepository(db, clock), mia_pack, lambda: (20, 80, 70, 20),
            cursor_pos=FakeCursor(),
            screens=FakeScreens([ScreenArea("A", Rect(0, 0, 1920, 1040), primary=True)]),
            permissions=permissions,
        )
        created.append(ctrl)
        ctrl.start()
        return ctrl

    yield build
    for ctrl in created:
        ctrl.shutdown()


def publish(bus, *events) -> None:
    for event in events:
        bus.publish(event)
    bus.dispatch_pending()


def started(game_id: str) -> GameStarted:
    return GameStarted(game_id=game_id, title=game_id.upper(), process_name=f"{game_id}.exe")


def ended(game_id: str, forgotten: bool = False) -> GameEnded:
    return GameEnded(game_id=game_id, title="" if forgotten else game_id.upper(), duration_seconds=90.0,
                     forgotten=forgotten)


def foreground() -> ForegroundAppChanged:
    return ForegroundAppChanged(app=AppInfo("code.exe", "VS Code"), category=ActivityCategory.PROGRAMMING,
                                confidence=0.9)


def chat_reply(ctrl: CharacterController, seconds: float = 6.0) -> None:
    """The user talks to her: a reply gesture at USER priority."""
    ctrl.say("Привіт! Я тут.", state="wave", seconds=seconds, solicited=True)
    assert ctrl.state is S.WAVE and ctrl.state_machine.priority is P.USER


# -- games and the kill switch (integ-1, QT-1, perf-1) ----------------------------------------------


@pytest.mark.parametrize("stop", [ObservationStopped(), PermissionChanged(Permission.APP_METADATA, False)])
def test_kill_switch_mid_game_leaves_gaming_and_forgets_the_game(make_controller, clock, bus, stop):
    ctrl = make_controller()
    publish(bus, started("stalker2"))
    assert ctrl.posture is S.GAMING
    publish(bus, stop)  # the monitor drops its state: no GameEnded will come
    assert ctrl.posture is S.IDLE
    run(ctrl, clock, 60.0, dt=0.5)
    assert ctrl.posture is not S.GAMING
    # observation on again: a later game must end normally (no stale "stalker2")
    publish(bus, started("cs2"))
    assert ctrl.posture is S.GAMING
    publish(bus, ended("cs2"))
    assert ctrl.posture is S.IDLE


def test_other_permissions_do_not_forget_games(make_controller, bus):
    ctrl = make_controller()
    publish(bus, started("cs2"), PermissionChanged(Permission.WINDOW_TITLES, False),
            PermissionChanged(Permission.APP_METADATA, True))
    assert ctrl.posture is S.GAMING


def test_kill_switch_drops_a_pending_gaming_posture(make_controller, clock, bus):
    ctrl = make_controller()
    chat_reply(ctrl)
    publish(bus, started("cs2"))  # rejected by the reply: retried later...
    assert ctrl.posture is not S.GAMING
    publish(bus, ObservationStopped())  # ...unless observation stopped meanwhile
    run(ctrl, clock, 10.0)
    assert ctrl.posture is not S.GAMING


def test_kill_switch_during_a_user_action_still_leaves_gaming(make_controller, clock, bus):
    ctrl = make_controller()
    publish(bus, started("cs2"))
    chat_reply(ctrl)
    publish(bus, ObservationStopped())
    run(ctrl, clock, 8.0)
    assert ctrl.posture is S.IDLE and ctrl.state is S.IDLE


def test_kill_switch_drops_queued_lines(make_controller, clock, bus):
    # A game-start line waiting behind another one used to be spoken after the
    # kill switch - and its "gaming" state put her back into GAMING for good.
    ctrl = make_controller()
    publish(bus, started("cs2"))
    ctrl.say("Добрий вечір!", solicited=False)
    ctrl.say("Counter-Strike! Тримаю кулаки за клатч.", state="gaming", solicited=False)
    assert ctrl.queued_lines == 1
    publish(bus, ObservationStopped())
    assert ctrl.queued_lines == 0
    run(ctrl, clock, 20.0)
    assert ctrl.bubble.text == "Добрий вечір!"
    assert ctrl.posture is not S.GAMING


def test_gaming_line_without_a_game_does_not_stick(make_controller, clock):
    ctrl = make_controller()
    ctrl.say("Рошан сам себе не переможе!", state="gaming", solicited=False)
    assert ctrl.state is S.TALK and ctrl.posture is S.IDLE
    run(ctrl, clock, 20.0)
    assert ctrl.posture is not S.GAMING


def test_gaming_line_during_a_game_keeps_the_gamepad(make_controller, clock, bus):
    ctrl = make_controller()
    publish(bus, started("dota2"))
    ctrl.say("Рошан сам себе не переможе!", state="gaming", solicited=False)
    assert ctrl.posture is S.GAMING
    run(ctrl, clock, 20.0)
    assert ctrl.posture is S.GAMING


# -- leaving GAMING while a USER action shows (integ-2) ---------------------------------------------


def test_game_end_during_a_user_action_is_retried(make_controller, clock, bus):
    ctrl = make_controller()
    publish(bus, started("cs2"))
    chat_reply(ctrl)
    publish(bus, ended("cs2"))  # REACTION < USER: rejected for now
    assert ctrl.state is S.WAVE
    run(ctrl, clock, 8.0)
    assert ctrl.posture is S.IDLE and ctrl.state is S.IDLE
    run(ctrl, clock, 60.0, dt=0.5)
    assert ctrl.posture is not S.GAMING


def test_new_game_cancels_the_pending_leave(make_controller, clock, bus):
    ctrl = make_controller()
    publish(bus, started("cs2"))
    chat_reply(ctrl)
    publish(bus, ended("cs2"), started("dota2"))
    run(ctrl, clock, 8.0)
    assert ctrl.posture is S.GAMING and ctrl.state is S.GAMING


def test_game_end_during_a_drag_settles_after_the_drop(make_controller, clock, bus):
    ctrl = make_controller()
    publish(bus, started("cs2"))
    ctrl.window.drag_started.emit()
    publish(bus, ended("cs2"))
    ctrl.window.drag_finished.emit()
    run(ctrl, clock, 5.0)
    assert ctrl.posture is S.IDLE


def test_forgotten_game_end_leaves_gaming_like_a_normal_end(make_controller, clock, bus):
    ctrl = make_controller()
    publish(bus, started("secret"))
    assert ctrl.posture is S.GAMING
    publish(bus, ended("secret", forgotten=True))
    assert ctrl.posture is S.IDLE
    publish(bus, started("cs2"), ended("cs2"))  # nothing left behind
    assert ctrl.posture is S.IDLE


# -- quiet mode and Do Not Disturb drop queued lines (integ-4) ---------------------------------------


def queue_two(ctrl: CharacterController) -> None:
    ctrl.say("Ооо, нова гра!", solicited=False)
    ctrl.say("Добрий вечір! Я вже засвітила зірки.", solicited=False)
    assert ctrl.queued_lines == 1


@pytest.mark.parametrize("event", [QuietModeChanged(enabled=True), DoNotDisturbChanged(enabled=True)])
def test_quiet_or_dnd_drops_queued_lines(make_controller, clock, bus, event):
    ctrl = make_controller()
    queue_two(ctrl)
    publish(bus, event)
    ctrl.say("Мовчу як риба...", solicited=True)  # her acknowledgement
    assert ctrl.queued_lines == 0
    run(ctrl, clock, 15.0)
    assert ctrl.bubble.text == "Мовчу як риба..."  # nothing unsolicited after the promise
    assert not ctrl.bubble.is_showing()


@pytest.mark.parametrize("event", [QuietModeChanged(enabled=False), DoNotDisturbChanged(enabled=False)])
def test_lifting_quiet_keeps_the_queue(make_controller, bus, event):
    ctrl = make_controller()
    queue_two(ctrl)
    publish(bus, event)
    assert ctrl.queued_lines == 1


def test_dnd_setting_drops_queued_lines(make_controller, settings):
    ctrl = make_controller()
    queue_two(ctrl)
    settings.behavior.do_not_disturb = True
    ctrl.apply_settings()
    assert ctrl.queued_lines == 0
    assert ctrl.bubble.is_showing()  # the line already showing stays


# -- user input known while the monitor runs (integ-5, perf-3) ---------------------------------------


def test_input_monitoring_keeps_an_active_user_awake(make_controller, clock, settings):
    settings.behavior.sleep_after_idle_minutes = 2
    control = make_controller()
    run(control, clock, 300.0, dt=0.5, until=lambda: control.posture is S.SLEEP)
    assert control.posture is S.SLEEP  # no input info: only clicks on her count
    control.shutdown()

    ctrl = make_controller()
    ctrl.set_input_monitoring(True)  # the monitor runs and has sent no UserIdle
    run(ctrl, clock, 300.0, dt=0.5)
    assert ctrl.idle_seconds() == 0.0
    assert ctrl.posture is not S.SLEEP


def test_input_monitoring_follows_idle_events_and_stops(make_controller, clock, bus):
    ctrl = make_controller()
    run(ctrl, clock, 30.0, dt=1.0)
    ctrl.set_input_monitoring(True)
    assert ctrl.idle_seconds() == 0.0
    publish(bus, UserIdle(idle_seconds=300.0))
    assert ctrl.idle_seconds() == pytest.approx(30.0, abs=1.01)  # capped by the last interaction
    ctrl.set_input_monitoring(True)  # redundant: an idle user is not marked present
    assert ctrl.idle_seconds() == pytest.approx(30.0, abs=1.01)
    ctrl.set_input_monitoring(False)
    run(ctrl, clock, 10.0, dt=1.0)
    assert ctrl.idle_seconds() == pytest.approx(40.0, abs=1.01)  # back to interactions only


def test_foreground_change_proves_monitoring(make_controller, clock, bus):
    ctrl = make_controller()
    run(ctrl, clock, 30.0, dt=1.0)
    publish(bus, foreground())
    assert ctrl.idle_seconds() == 0.0
    publish(bus, UserIdle(idle_seconds=120.0), foreground())  # an idle user stays idle
    assert ctrl.idle_seconds() > 0.0
    publish(bus, ObservationStopped())
    assert ctrl.idle_seconds() == pytest.approx(30.0, abs=1.01)


# -- activity events after the kill switch (privsec-2) ----------------------------------------------


def test_activity_events_without_permission_do_not_change_her(make_guarded, clock, bus):
    permissions = FakePermissions(granted=False)
    ctrl = make_guarded(permissions)
    run(ctrl, clock, 30.0, dt=1.0)
    ctrl.request_state(S.SLEEP, P.USER, 1.0)
    run(ctrl, clock, 2.0)
    publish(bus, started("cs2"), UserReturned(away_seconds=600.0), foreground())
    assert ctrl.posture is S.SLEEP and ctrl.state is S.SLEEP  # no gamepad, no greeting
    publish(bus, UserIdle(idle_seconds=10.0))
    assert ctrl.idle_seconds() == pytest.approx(32.0, abs=1.01)  # input still unknown
    permissions.granted = True
    publish(bus, started("dota2"))
    assert ctrl.posture is S.GAMING
    publish(bus, ended("dota2"))  # only "dota2" ran: the ignored "cs2" left nothing behind
    assert ctrl.posture is S.IDLE


def test_events_queued_before_the_kill_switch_are_dropped(make_guarded, clock, bus):
    permissions = FakePermissions(granted=True)
    ctrl = make_guarded(permissions)
    bus.publish(started("cs2"))  # gathered while permitted, not dispatched yet
    bus.publish(UserReturned(away_seconds=600.0))
    permissions.granted = False  # the kill switch flips the permission in memory first
    publish(bus, ObservationStopped(), started("cs2"))  # ...and a racy one after it
    assert ctrl.posture is S.IDLE and ctrl.state is not S.WAVE
    run(ctrl, clock, 10.0)
    assert ctrl.posture is not S.GAMING


def test_game_end_is_handled_even_after_a_revoke(make_guarded, bus):
    permissions = FakePermissions(granted=True)
    ctrl = make_guarded(permissions)
    publish(bus, started("cs2"))
    permissions.granted = False
    publish(bus, ended("cs2"))  # an end only removes state
    assert ctrl.posture is S.IDLE


def test_failing_permission_check_counts_as_not_granted(make_guarded, bus, caplog):
    class Broken:
        def is_granted(self, permission: Permission) -> bool:
            raise RuntimeError("lock poisoned")

    caplog.set_level(logging.ERROR, logger="chibi.desktop.controller")
    ctrl = make_guarded(Broken())  # type: ignore[arg-type]
    publish(bus, started("cs2"), foreground(), foreground())
    assert ctrl.posture is S.IDLE
    assert sum("permission check failed" in r.getMessage() for r in caplog.records) == 1
