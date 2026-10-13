"""The controller's speech-router hook: the voice integration decides when a
``SpeechRequested`` reaches the bubble (at once, or when its audio starts)."""

from __future__ import annotations

from chibi.character.states import CharacterState as S
from chibi.core.events import SpeechRequested
from chibi.desktop.controller import speech_seconds
from tests.desktop.helpers import run


class Router:
    def __init__(self):
        self.calls = []

    def __call__(self, event, show):
        self.calls.append((event, show))


def publish(bus, text="Привіт!", solicited=True, **kwargs):
    event = SpeechRequested(text=text, solicited=solicited, **kwargs)
    bus.publish(event)
    bus.dispatch_pending()
    return event


def test_without_a_router_lines_show_directly(make_controller, bus):
    ctrl = make_controller()
    publish(bus)
    assert ctrl.bubble.is_showing() and ctrl.bubble.text == "Привіт!"


def test_router_gets_the_event_and_decides_when_it_shows(make_controller, bus, clock):
    ctrl = make_controller()
    router = Router()
    ctrl.set_speech_router(router)
    event = publish(bus, state="laugh", portrait="laugh")
    (routed, show), = router.calls
    assert routed is event
    assert not ctrl.bubble.isVisible()  # waits for the audio
    show(routed, 1.5)  # the audio started; it lasts 1.5 s
    assert ctrl.bubble.text == "Привіт!"
    assert ctrl.state is S.LAUGH
    run(ctrl, clock, 1.3)
    assert ctrl.bubble.is_showing()
    run(ctrl, clock, 0.4)
    assert not ctrl.bubble.is_showing()  # the bubble lasted as long as the audio


def test_show_without_seconds_uses_the_reading_time(make_controller, bus, clock):
    ctrl = make_controller()
    ctrl.set_speech_router(lambda event, show: show(event, None))
    publish(bus, text="Як справи?")
    run(ctrl, clock, speech_seconds("Як справи?") - 0.3)
    assert ctrl.bubble.is_showing()


def test_unsolicited_routed_lines_keep_the_queue_rules(make_controller, bus):
    ctrl = make_controller()
    ctrl.set_speech_router(lambda event, show: show(event, None))
    publish(bus, text="Перша", solicited=False)
    publish(bus, text="Друга", solicited=False)
    assert ctrl.bubble.text == "Перша" and ctrl.queued_lines == 1


def test_a_failing_router_never_costs_the_line(make_controller, bus, caplog):
    ctrl = make_controller()

    def broken(event, show):
        raise RuntimeError(event.text)

    ctrl.set_speech_router(broken)
    publish(bus, text="Секретна репліка")
    assert ctrl.bubble.text == "Секретна репліка"
    assert "speech router failed (RuntimeError)" in caplog.text
    assert "Секретна репліка" not in caplog.text  # lines are never logged


def test_a_router_failing_after_showing_does_not_show_twice(make_controller, bus, monkeypatch):
    ctrl = make_controller()
    said = []
    original = ctrl.say

    def spy(*args, **kwargs):
        said.append(args[0])
        original(*args, **kwargs)

    monkeypatch.setattr(ctrl, "say", spy)

    def shows_then_breaks(event, show):
        show(event, None)
        raise RuntimeError("late")

    ctrl.set_speech_router(shows_then_breaks)
    publish(bus)
    assert said == ["Привіт!"]


def test_router_can_be_removed(make_controller, bus):
    ctrl = make_controller()
    router = Router()
    ctrl.set_speech_router(router)
    ctrl.set_speech_router(None)
    publish(bus)
    assert router.calls == []
    assert ctrl.bubble.text == "Привіт!"


def test_late_show_after_shutdown_is_harmless(make_controller, bus):
    ctrl = make_controller()
    router = Router()
    ctrl.set_speech_router(router)
    publish(bus)
    event, show = router.calls[0]
    ctrl.shutdown()
    show(event, 2.0)  # the audio started after the app began shutting down
    assert not ctrl.bubble.isVisible()


def test_direct_say_is_not_routed(make_controller):
    ctrl = make_controller()
    router = Router()
    ctrl.set_speech_router(router)
    ctrl.say("Напряму")
    assert router.calls == []
    assert ctrl.bubble.text == "Напряму"
