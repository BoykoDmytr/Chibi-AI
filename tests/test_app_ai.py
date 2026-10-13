"""Phase 2 wiring in ChibiApp: AI chat with the offline fallback, model-requested
actions, memory and key facade methods, wipe-all, activity context."""

import time

import pytest
from PySide6.QtCore import QCoreApplication

import chibi.app as app_module
from chibi.ai.providers.fake import FakeProvider
from chibi.ai.secrets import MemorySecretStore
from chibi.ai.service import AIService
from chibi.app import ChibiApp
from chibi.core.events import (
    ActivityCategory,
    AppInfo,
    ForegroundAppChanged,
    InitiativeAction,
    ObservationStopped,
    SpeechRequested,
    UserIdle,
)
from chibi.settings.theme import apply_dark_theme
from tests.test_app import UnsupportedBackend


def pump_until(predicate, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def secrets(monkeypatch):
    store = MemorySecretStore()
    monkeypatch.setattr(app_module, "default_secret_store", lambda _dir: store)
    return store


@pytest.fixture
def provider(monkeypatch):
    fake = FakeProvider(local=True)

    class Service(AIService):
        def __init__(self, *args, **kwargs):
            def factory(ai, _secrets):
                return fake if ai.provider != "none" else None

            super().__init__(*args, provider_factory=factory, **kwargs)

    monkeypatch.setattr(app_module, "AIService", Service)
    return fake


@pytest.fixture
def make_app(qapp, secrets):
    apply_dark_theme(qapp)
    created = []

    def factory():
        app = ChibiApp(qapp, backend=UnsupportedBackend())
        created.append(app)
        return app

    yield factory
    for app in created:
        app.close()


def enable_local_ai(app):
    s = app.settings()
    s.ai.provider = "ollama"
    s.ai.model = "fake-model"
    app.apply_settings(s)


def chat(app, text):
    deltas, replies = [], []
    app.send_chat_stream(text, deltas.append, replies.append)
    assert pump_until(lambda: replies)
    assert len(replies) == 1
    return "".join(deltas), replies[0]


def test_without_ai_the_offline_responder_answers(make_app):
    app = make_app()
    app.start()
    status = app.ai_status()
    assert not status.online and status.reason == "AI вимкнено" and status.provider == "none"
    _streamed, reply = chat(app, "привіт")
    assert reply.text  # an offline phrase-bank answer
    assert not app.status().ai_connected


def test_ai_chat_streams_and_applies_the_requested_action(make_app, provider):
    app = make_app()
    app.start()
    enable_local_ai(app)
    assert app.ai_status().online
    provider.replies.append("<<state=happy;action=feed>>\nНям-ням, дякую!")
    hunger = app.needs().hunger
    spoken = []
    app.bus.subscribe(SpeechRequested, spoken.append)
    streamed, reply = chat(app, "Поїж, будь ласка")
    assert reply.text == "Ням-ням, дякую!"
    assert "<<" not in streamed
    assert reply.state == "happy"
    assert reply.action in ("fed", "")  # "" if she was already full
    if reply.action == "fed":
        assert app.needs().hunger < hunger
    assert app.status().ai_connected
    assert any(r.purpose.value == "chat" for r in provider.requests)


def test_ai_failure_falls_back_to_the_offline_answer(make_app, provider):
    app = make_app()
    app.start()
    enable_local_ai(app)
    provider.replies.append(RuntimeError("model crashed"))
    _streamed, reply = chat(app, "привіт")
    assert reply.text  # offline answer, the user is never left without one


def test_quiet_action_from_the_model(make_app, provider):
    app = make_app()
    app.start()
    enable_local_ai(app)
    provider.replies.append("<<state=talk;action=quiet_on>>\nДобре, мовчу.")
    _streamed, reply = chat(app, "помовч трохи")
    assert reply.action == "quiet_on"
    assert app.status().quiet_until is not None


def test_memory_facade_round_trip(make_app):
    app = make_app()
    app.add_memory("preference", "Любить каву без цукру")
    app.add_memory("fact", "Номер картки 4111 1111 1111 1111")  # refused: sensitive
    memories = app.list_memories()
    assert [m.text for m in memories] == ["Любить каву без цукру"]
    memory_id = memories[0].id
    app.set_memory_pinned(memory_id, True)
    assert app.list_memories("preference")[0].pinned
    with pytest.raises(ValueError):
        app.update_memory(memory_id, "мій пароль: qwerty12345")
    app.update_memory(memory_id, "Любить каву з молоком")
    assert app.list_memories(query="молоком")
    app.delete_memory(memory_id)
    assert app.list_memories() == []
    app.add_memory("joke", "Жарт про кота")
    app.clear_memories()
    assert app.list_memories() == []


def test_api_keys_go_to_the_secret_store(make_app, secrets):
    app = make_app()
    assert not app.ai_has_key("anthropic")
    app.ai_set_key("anthropic", "  sk-ant-test-key-123456  ")
    assert app.ai_has_key("anthropic")
    assert secrets.get("anthropic_api_key") == "sk-ant-test-key-123456"
    app.ai_set_key("anthropic", "")
    assert not app.ai_has_key("anthropic")
    with pytest.raises(ValueError):
        app.ai_set_key("ollama", "x")


def test_wipe_all_forgets_memories_and_keys(make_app, secrets, monkeypatch):
    monkeypatch.setattr(app_module.QTimer, "singleShot", lambda *a, **k: None)  # no first-run dialog
    app = make_app()
    app.add_memory("fact", "Має кота Мурчика")
    app.ai_set_key("openai", "sk-test-0123456789abcdef")
    app.wipe_all()
    assert app.list_memories() == []
    assert not app.ai_has_key("openai")
    assert app.emotion()  # fresh defaults, still usable


def test_activity_context_respects_privacy_and_the_kill_switch(make_app):
    app = make_app()
    app.start()
    app.bus.publish(ForegroundAppChanged(AppInfo("code.exe", "VS Code"), ActivityCategory.PROGRAMMING, 0.9))
    assert pump_until(lambda: app._brain_context().activity_app == "VS Code")
    ctx = app._initiative_context()
    assert ctx.activity_category == "programming" and ctx.activity_minutes is not None
    app.bus.publish(ForegroundAppChanged(AppInfo.private_app(), ActivityCategory.PRIVATE, 1.0))
    assert pump_until(lambda: app._brain_context().activity_category == "private")
    assert app._brain_context().activity_app is None
    app.bus.publish(UserIdle(idle_seconds=600))
    assert pump_until(lambda: app._brain_context().user_away)
    app.bus.publish(ObservationStopped())
    assert pump_until(lambda: app._brain_context().activity_category is None)
    assert not app._brain_context().user_away


def test_emotion_and_ai_status_views(make_app):
    app = make_app()
    values = app.emotion()
    assert set(values) >= {"happiness", "curiosity", "trust"}
    status = app.ai_status()
    assert status.requests_limit_day == app.settings().ai.max_requests_per_day
    assert not status.cloud


def test_shutdown_stops_ai_and_saves_emotion(make_app, provider):
    app = make_app()
    app.start()
    enable_local_ai(app)
    app.shutdown()
    app.close()  # database closed after the AI worker: no errors


def test_initiative_actions_do_not_crash_without_a_handler(make_app):
    app = make_app()
    app.start()
    app.bus.publish(InitiativeAction(action="wave"))
    pump_until(lambda: False, timeout=0.1)
