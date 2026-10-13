"""Phases 3-6 wiring in ChibiApp: voice routing and the microphone, screen
vision from the menu and the chat, gaming companion mode, gifts and mini-games,
the new settings pages, wipe-all and shutdown; the voice companion (streamed
answers, one answer at a time for voice requests, the microphone pause, the
observer)."""

import time
from dataclasses import replace

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QMenu

import chibi.app as app_module
from chibi.ai.providers.fake import FakeProvider
from chibi.ai.secrets import MemorySecretStore
from chibi.ai.service import AIService
from chibi.ai.types import Purpose
from chibi.app import (
    BUSY_REPLY,
    CHAT_IN_FLIGHT_SECONDS,
    LOOKING_REPLY,
    MIC_PAUSE_PUSH_HINT,
    MIC_PAUSE_TEXT,
    PENDING_TRANSCRIPT_SECONDS,
    ChibiApp,
)
from chibi.core.events import Permission, SpeechRequested, VisionObserved, VoiceTranscript
from chibi.desktop.tray import TrayIcon
from chibi.games import GAME_IDS, GAME_TITLES
from chibi.settings.facade import ChatReply
from chibi.settings.settings_dialog import PAGES, SettingsDialog
from chibi.settings.theme import apply_dark_theme
from chibi.vision.integration import REASON_OFF
from chibi.voice.integration import (
    AZURE_KEY_NAME,
    LISTENING_NAME,
    PAUSED_BY_USER,
    PAUSED_DND,
)
from tests.test_app import UnsupportedBackend, pump


@pytest.fixture
def secrets(monkeypatch):
    store = MemorySecretStore()
    monkeypatch.setattr(app_module, "default_secret_store", lambda _dir: store)
    return store


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


def menu_texts(menu):
    return [a.text() for a in menu.actions() if not a.isSeparator()]


def chat(app, text):
    replies = []
    app.send_chat_stream(text, lambda _delta: None, replies.append)
    assert len(replies) == 1
    return replies[0]


# -- menus ------------------------------------------------------------------------------------------


def test_context_menu_offers_voice_vision_games_and_gifts(make_app):
    app = make_app()
    app.start()
    texts = menu_texts(app._context_menu)
    assert "Сказати голосом" in texts and "Подивитися на екран" in texts and "Подарунки" in texts
    assert not any("фазі" in t for t in texts)  # no "coming in phase N" leftovers
    games = next(a.menu() for a in app._context_menu.actions() if a.text() == "Пограти")
    assert menu_texts(games) == [GAME_TITLES[g] for g in GAME_IDS]
    app._refresh_menus()
    # Recognition is off by default: the item says why instead of failing on click.
    assert not app._act_voice.isEnabled() and app._act_voice.toolTip()


def test_games_and_the_shop_open_and_close_on_shutdown(make_app):
    app = make_app()
    app.start()
    app.open_game("rps")
    app.open_shop()
    window = app._games_hub.window("rps")
    assert window is not None and window.isVisible() and app._games_hub.shop_window is not None
    app.shutdown()
    assert app._games_hub.window("rps") is None and app._games_hub.shop_window is None


def test_daily_stars_are_granted_once_at_start(make_app):
    app = make_app()
    app.start()
    pump(50)
    stars = app._gifts.wallet.stars
    assert stars > 0
    app._gifts.wallet.daily_start()
    assert app._gifts.wallet.stars == stars  # same day: no second bonus


# -- settings pages ---------------------------------------------------------------------------------


def test_settings_dialog_has_the_new_pages_backed_by_the_real_app(make_app):
    app = make_app()
    app.start()
    keys = [key for key, _title in PAGES]
    assert keys.index("ai") < keys.index("voice") < keys.index("vision") < keys.index("gaming")
    dialog = SettingsDialog(app, confirm=lambda _t, _x: True)
    try:
        for key in ("voice", "vision", "gaming"):
            dialog.show_page(key)
            assert dialog.current_page() == key
        dialog.gaming_page.collect(app.settings())  # nothing edited: nothing changes
        assert dialog.apply()
        assert app.settings().voice == app._settings.voice
    finally:
        dialog.close()
        dialog.deleteLater()


def test_microphone_consent_text_is_the_voice_pages(make_app):
    from chibi.settings.settings_dialog import PERMISSION_TEXTS
    from chibi.settings.voice_page import MICROPHONE_CONSENT_TEXT

    assert PERMISSION_TEXTS[Permission.MICROPHONE].consent_text == MICROPHONE_CONSENT_TEXT


def test_voice_and_vision_facade_methods(make_app):
    app = make_app()
    app.start()
    status = app.voice_status()
    assert not status.tts_ready and not status.stt_ready and status.state == "off"
    assert app.voice_models() and app.voice_download_state() is None
    assert app.voice_has_azure_key() is False
    view = app.vision_status()
    assert view.mode == "off" and view.state == "off"
    assert app.vision_recent() == [] and app.vision_games()


# -- chat: "подивись на екран" ----------------------------------------------------------------------


def test_look_request_with_vision_off_answers_with_the_reason(make_app):
    app = make_app()
    app.start()
    reply = chat(app, "Подивись на екран, що там?")
    assert reply.text == REASON_OFF  # the reason, never an invented description


def test_look_request_that_vision_accepts(make_app, monkeypatch):
    app = make_app()
    app.start()
    monkeypatch.setattr(app._vision, "request", lambda: True)
    assert chat(app, "що ти бачиш на екрані?").text == LOOKING_REPLY
    assert chat(app, "Привіт!").text != LOOKING_REPLY


def test_vision_notices_outside_the_chat_appear_in_her_bubble(make_app, monkeypatch):
    app = make_app()
    app.start()
    said = []
    monkeypatch.setattr(app._controller, "say", lambda text, *a, **k: said.append(text))
    app._vision_notice("Не вдалося роздивитися екран.")
    assert said == ["Не вдалося роздивитися екран."]


def test_brain_context_carries_the_latest_observation(make_app, monkeypatch):
    app = make_app()
    monkeypatch.setattr(app._vision, "latest_observation", lambda: "combat/boss_fight: бос")
    assert app._brain_context().observation == "combat/boss_fight: бос"


# -- voice ------------------------------------------------------------------------------------------


def test_transcripts_need_the_microphone_permission_and_go_to_the_chat(make_app, monkeypatch):
    app = make_app()
    app.start()
    sent = []
    monkeypatch.setattr(app, "send_chat_stream", lambda text, *_a: sent.append(text))
    app.bus.publish(VoiceTranscript(text="Привіт, Міє"))
    pump(50)
    assert sent == []  # no permission (revoked meanwhile): dropped
    app.permissions.grant(Permission.MICROPHONE)
    app.bus.publish(VoiceTranscript(text="  Привіт, Міє "))
    app.bus.publish(VoiceTranscript(text="   "))
    pump(50)
    assert sent == ["Привіт, Міє"]


def test_transcript_with_the_chat_open_appears_there_and_keeps_the_draft(make_app):
    app = make_app()
    app.start()
    app.permissions.grant(Permission.MICROPHONE)
    app.show_chat()
    chat_window = app._chat
    chat_window.input.setText("недописане")
    before = chat_window.message_count()
    app.bus.publish(VoiceTranscript(text="як справи?"))
    pump(100)
    assert chat_window.message_count() > before
    assert chat_window.input.text() == "недописане"


def test_transcript_while_the_chat_is_still_answering(make_app, monkeypatch):
    app = make_app()
    app.start()
    app.permissions.grant(Permission.MICROPHONE)
    app.show_chat()
    lines = []
    monkeypatch.setattr(app._chat, "is_waiting", lambda: True)
    monkeypatch.setattr(app._chat, "add_system_line", lines.append)
    app.bus.publish(VoiceTranscript(text="ще одне"))
    pump(50)
    assert lines == [BUSY_REPLY]


def test_chat_window_gets_the_microphone_button_state(make_app):
    app = make_app()
    app.start()
    app.show_chat()
    assert not app._chat.is_listening()
    app._toggle_voice()  # recognition off: the reason goes to the chat, nothing breaks
    pump(20)


def test_chat_replies_go_through_the_voice_router(make_app, monkeypatch):
    app = make_app()
    app.start()
    routed = []

    def route(event, show):
        routed.append(event)
        show(event, None)

    monkeypatch.setattr(app._voice, "route", route)
    reply = chat(app, "Привіт!")
    assert routed and routed[-1].source == "chat" and routed[-1].solicited
    assert routed[-1].text == reply.text


def test_companion_mode_hides_bubbles_of_voiced_comments_over_a_game(make_app, monkeypatch):
    app = make_app()
    shown = []
    monkeypatch.setattr(app._voice, "route", lambda event, show: show(event, 2.5))
    comment = SpeechRequested(text="Ого, бос!", state="excited", solicited=False, source="vision.boss_fight")

    def route(event):
        app._route_speech(event, lambda e, seconds: shown.append((e.text, seconds)))

    route(comment)  # no game running: the bubble stays
    monkeypatch.setattr(app, "_current_game", lambda: ("S.T.A.L.K.E.R. 2", 12.0))
    route(comment)  # voiced over a game: heard, not read
    route(replace(comment, solicited=True))  # answers keep their bubble
    s = app.settings()
    s.gaming.companion_mode = False
    app.apply_settings(s)
    route(comment)
    assert shown == [("Ого, бос!", 2.5), ("", 2.5), ("Ого, бос!", 2.5), ("Ого, бос!", 2.5)]
    monkeypatch.setattr(app._voice, "route", lambda event, show: show(event, None))
    s.gaming.companion_mode = True
    app.apply_settings(s)
    shown.clear()
    route(comment)  # not voiced (voice off): the bubble is the only way she can say it
    assert shown == [("Ого, бос!", None)]


def test_voice_settings_reconfigure_synchronously(make_app, monkeypatch):
    app = make_app()
    app.start()
    calls = []
    monkeypatch.setattr(app._voice, "reconfigure", lambda: calls.append(1))
    s = app.settings()
    s.voice.speed = 1.2
    app.apply_settings(s)
    s.character.name = "Ліна"
    app.apply_settings(s)
    s.behavior.quiet_minutes = 45
    app.apply_settings(s)
    assert calls == [1, 1]


# -- data and shutdown ----------------------------------------------------------------------------


def test_wipe_all_deletes_the_azure_key_gifts_and_what_she_saw(make_app, secrets, monkeypatch):
    app = make_app()
    app.start()
    secrets.set(AZURE_KEY_NAME, "azure-secret")
    app._gifts.wallet.earn(10)
    cleared = []
    monkeypatch.setattr(app._vision, "clear_history", lambda: cleared.append(1))
    monkeypatch.setattr(app_module.QTimer, "singleShot", lambda *_a: None)  # no first-run dialog
    app.wipe_all()
    assert secrets.get(AZURE_KEY_NAME) is None
    assert app._gifts.wallet.stars == 0 and app._gifts.inventory() == {}
    assert cleared


def test_wipe_history_forgets_what_she_saw(make_app, monkeypatch):
    app = make_app()
    cleared = []
    monkeypatch.setattr(app._vision, "clear_history", lambda: cleared.append(1))
    app.wipe_history()
    assert cleared == [1]


def test_vision_and_voice_stop_before_the_ai_service(make_app, monkeypatch):
    app = make_app()
    app.start()
    order = []
    for name, obj in (("vision", app._vision), ("voice", app._voice), ("ai", app._ai)):
        original = obj.shutdown
        monkeypatch.setattr(obj, "shutdown", lambda n=name, f=original: (order.append(n), f()))
    app.shutdown()
    assert order == ["vision", "voice", "ai"]


def test_voice_menu_item_says_what_a_click_does(make_app, monkeypatch):
    app = make_app()
    app.start()
    status = replace(app._voice.status(), stt_ready=True, state="standby")
    monkeypatch.setattr(app._voice, "status", lambda: status)
    app._refresh_menus()
    assert app._act_voice.text() == "Сказати голосом"  # waiting for her name: a click starts talking
    status = replace(status, state="listening")
    app._refresh_menus()
    assert app._act_voice.text() == "Перестати слухати"


def test_wipe_all_also_withdraws_the_cloud_ai_consent(make_app, monkeypatch):
    app = make_app()
    app.start()
    app.permissions.grant(Permission.CLOUD_AI)
    monkeypatch.setattr(app_module.QTimer, "singleShot", lambda *_a: None)
    app.wipe_all()
    assert not any(app.permissions.snapshot().values())


# -- Phase 6: the voice companion -------------------------------------------------------------------


def pump_until(predicate, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def fake_ai(monkeypatch):
    fake = FakeProvider(local=True)

    class Service(AIService):
        def __init__(self, *args, **kwargs):
            def factory(ai, _secrets):
                return fake if ai.provider != "none" else None

            super().__init__(*args, provider_factory=factory, **kwargs)

    monkeypatch.setattr(app_module, "AIService", Service)
    return fake


def enable_local_ai(app):
    s = app.settings()
    s.ai.provider = "ollama"
    s.ai.model = "fake-model"
    app.apply_settings(s)


class RecordingStream:
    """Stands in for VoiceIntegration's ReplyStream."""

    def __init__(self):
        self.calls = []

    def feed(self, delta):
        self.calls.append(("feed", delta))

    def finish(self, text, state="talk", portrait=""):
        self.calls.append(("finish", text, state))

    def abort(self):
        self.calls.append(("abort",))


class FakeVoiceService:
    """Just enough of VoiceService for the integration to queue her sentences."""

    can_speak = True
    listen_mode = "push"

    def __init__(self, log):
        self.log = log

    def say(self, text, **_kwargs):
        self.log.append(("say", text))
        return True

    def hold_for_reply(self, _seconds):
        pass

    def release_reply_hold(self):
        pass

    def check_hold(self):
        pass

    def note_reply_shown(self, _seconds):
        pass

    def shutdown(self):
        pass


def record_streams(app, monkeypatch):
    streams = []

    def begin_reply(_source="chat", *, show):
        streams.append(RecordingStream())
        return streams[-1]

    monkeypatch.setattr(app._voice, "begin_reply", begin_reply)
    return streams


def test_ai_answer_is_voiced_sentence_by_sentence_while_it_streams(make_app, fake_ai, monkeypatch):
    app = make_app()
    app.start()
    enable_local_ai(app)
    pump(50)
    log = []
    monkeypatch.setattr(app._voice, "_service", FakeVoiceService(log))
    monkeypatch.setattr(app._voice, "_should_voice", lambda _event: True)
    routed = []
    monkeypatch.setattr(app._voice, "route", lambda event, _show: routed.append(event))
    fake_ai.replies.append("<<state=happy;action=none>>\nПривіт, рада тебе чути! "
                           "Сьогодні чудовий день для прогулянки. Що будемо робити?")
    replies = []

    def done(reply):
        log.append(("done", reply.text))
        replies.append(reply)

    app.send_chat_stream("Привіт", lambda piece: log.append(("delta", piece)), done)
    assert pump_until(lambda: replies)
    said = [text for kind, text in log if kind == "say"]
    assert len(said) == 3 and " ".join(said) == replies[0].text  # every sentence exactly once
    first_said = next(i for i, (kind, _text) in enumerate(log) if kind == "say")
    last_delta = max(i for i, (kind, _text) in enumerate(log) if kind == "delta")
    assert first_said < last_delta  # she started talking while the model was still writing
    assert [e for e in routed if e.source == "chat"] == []  # the finished answer is not routed again


def test_streamed_answer_finishes_with_the_applied_action(make_app, fake_ai, monkeypatch):
    app = make_app()
    app.start()
    enable_local_ai(app)
    streams = record_streams(app, monkeypatch)
    routed = []
    monkeypatch.setattr(app._voice, "route", lambda event, _show: routed.append(event))
    fake_ai.replies.append("<<state=happy;action=quiet_on>>\nДобре, помовчу трохи.")
    replies = []
    app.send_chat_stream("помовч", lambda _d: None, replies.append)
    assert pump_until(lambda: replies)
    reply = replies[0]
    assert reply.action == "quiet_on" and app.status().quiet_until is not None
    calls = streams[0].calls
    assert "".join(c[1] for c in calls if c[0] == "feed") == reply.text
    assert [c for c in calls if c[0] != "feed"] == [("finish", reply.text, reply.state)]
    assert [e for e in routed if e.source == "chat"] == []


def test_ai_failure_aborts_the_stream_and_says_the_offline_answer_once(make_app, fake_ai, monkeypatch):
    app = make_app()
    app.start()
    enable_local_ai(app)
    streams = record_streams(app, monkeypatch)
    routed = []
    monkeypatch.setattr(app._voice, "route", lambda event, _show: routed.append(event))
    fake_ai.replies.append(RuntimeError("model crashed"))
    replies = []
    app.send_chat_stream("привіт", lambda _d: None, replies.append)
    assert pump_until(lambda: replies)
    assert streams[0].calls == [("abort",)]
    answers = [(e.text, e.solicited) for e in routed if e.source == "chat"]
    assert answers == [(replies[0].text, True)]


def test_offline_and_look_answers_do_not_open_a_stream(make_app, monkeypatch):
    app = make_app()
    app.start()
    streams = record_streams(app, monkeypatch)
    routed = []
    monkeypatch.setattr(app._voice, "route", lambda event, _show: routed.append(event))
    offline = chat(app, "Привіт!")  # AI off: the phrase bank answers
    monkeypatch.setattr(app._vision, "request", lambda: True)
    look = chat(app, "подивись на екран")
    assert streams == []
    answers = [e.text for e in routed if e.source == "chat"]
    assert answers == [offline.text, look.text] and look.text == LOOKING_REPLY


@pytest.fixture
def held_chat(make_app, monkeypatch):
    """An app whose AI answers only when the test says so: (app, chats) with chats a
    list of (text, on_done)."""
    app = make_app()
    app.start()
    pump(50)  # the greeting is said before the brain is replaced
    app.permissions.grant(Permission.MICROPHONE)
    chats = []
    monkeypatch.setattr(app._brain, "available", lambda _purpose: True)
    monkeypatch.setattr(app._brain, "chat", lambda text, _delta, on_done: chats.append((text, on_done)))
    record_streams(app, monkeypatch)
    return app, chats


def hear(app, *texts):
    for text in texts:
        app.bus.publish(VoiceTranscript(text=text))
    pump(50)


def test_voice_requests_wait_for_the_answer_in_flight_newest_only(held_chat):
    app, chats = held_chat
    hear(app, "котра година?", "а погода?", "і ще одне")
    assert [text for text, _done in chats] == ["котра година?"]  # never two answers at once
    chats[0][1](ChatReply(text="Пів на третю."))
    pump(50)
    assert [text for text, _done in chats] == ["котра година?", "і ще одне"]
    chats[1][1](ChatReply(text="Гаразд."))
    pump(50)
    assert len(chats) == 2


def test_waiting_voice_request_is_dropped_when_the_microphone_is_revoked(held_chat):
    app, chats = held_chat
    hear(app, "перше", "друге")
    app.permissions.revoke(Permission.MICROPHONE)
    pump(50)
    chats[0][1](ChatReply(text="Ок."))
    pump(50)
    assert len(chats) == 1


def test_waiting_voice_request_that_waited_too_long_is_dropped(held_chat):
    app, chats = held_chat
    hear(app, "перше", "друге")
    text, heard_at = app._pending_transcript
    app._pending_transcript = (text, heard_at - PENDING_TRANSCRIPT_SECONDS - 1)
    chats[0][1](ChatReply(text="Ок."))
    pump(50)
    assert len(chats) == 1


def test_an_answer_that_never_arrives_does_not_hold_the_voice_back_for_good(held_chat):
    app, chats = held_chat
    hear(app, "перше")
    app._chats_in_flight = {key: started - CHAT_IN_FLIGHT_SECONDS - 1
                            for key, started in app._chats_in_flight.items()}
    hear(app, "друге")
    assert [text for text, _done in chats] == ["перше", "друге"]


def test_voice_request_waiting_for_a_hidden_answer_goes_to_the_chat_once_open(held_chat):
    app, chats = held_chat
    hear(app, "перше")  # the chat window is hidden: answered straight away
    app.show_chat()
    before = app._chat.message_count()
    hear(app, "друге")
    assert len(chats) == 1 and app._chat.message_count() == before
    chats[0][1](ChatReply(text="Ок."))
    pump(50)
    assert [text for text, _done in chats] == ["перше", "друге"]
    assert app._chat.message_count() > before and app._chat.is_waiting()


def test_microphone_pause_item_in_her_menu(make_app, monkeypatch):
    app = make_app()
    app.start()
    texts = menu_texts(app._context_menu)
    assert texts.index(MIC_PAUSE_TEXT) == texts.index("Не турбувати") + 1
    app._refresh_menus()
    item = app._act_mic_pause
    real_status = app._voice.status
    assert not item.isEnabled() and item.toolTip() == real_status().stt_reason  # recognition is off
    mode = "push"

    def status():
        reason = PAUSED_BY_USER if app._voice.open_mic_paused else ""
        return replace(real_status(), stt_ready=True, stt_reason="", listen_mode=mode, pause_reason=reason)

    monkeypatch.setattr(app._voice, "status", status)
    app._refresh_menus()
    assert not item.isEnabled() and item.toolTip() == MIC_PAUSE_PUSH_HINT  # push-to-talk: nothing to pause
    mode = "name"
    app._refresh_menus()
    assert item.isEnabled() and not item.isChecked() and item.toolTip() == LISTENING_NAME
    item.trigger()
    assert app._voice.open_mic_paused and item.isChecked() and item.toolTip() == PAUSED_BY_USER
    item.trigger()
    assert not app._voice.open_mic_paused and not item.isChecked()


def test_tray_tooltip_says_how_she_listens_or_why_the_microphone_is_paused(make_app, monkeypatch):
    app = make_app()
    tray = TrayIcon(QMenu(), name="Мія")
    monkeypatch.setattr(app, "_tray", tray)
    status = replace(app._voice.status(), stt_ready=True, listen_mode="name", state="idle",
                     pause_reason=PAUSED_DND)
    monkeypatch.setattr(app._voice, "status", lambda: status)
    app._sync_voice_ui()
    assert PAUSED_DND in tray.tooltip() and not tray.listening
    status = replace(status, pause_reason="", state="standby")
    app._sync_voice_ui()
    assert tray.listening and LISTENING_NAME in tray.tooltip() and PAUSED_DND not in tray.tooltip()
    status = replace(status, listen_mode="push", state="idle")
    app._sync_voice_ui()
    assert LISTENING_NAME not in tray.tooltip() and not tray.listening


def test_observer_comments_on_what_she_sees_through_the_ai(make_app, fake_ai):
    app = make_app()  # not started: no greeting to count against the shared gap
    enable_local_ai(app)  # a local model: what she saw may go to it
    s = app.settings()
    s.vision.commentary = "observer"
    app.apply_settings(s)
    app.permissions.grant(Permission.SCREEN_CAPTURE)
    pump(50)
    spoken = []
    app.bus.subscribe(SpeechRequested, lambda e: spoken.append(e) if e.source == "observer:ai" else None)
    fake_ai.replies.append("<<state=thinking;action=none>>\nОго, червоне в терміналі. Допомогти розібратися?")
    app.bus.publish(VisionObserved(app_title="Термінал", scene="terminal", event="error_on_screen",
                                   confidence=0.9, description="у терміналі червона помилка збірки"))
    assert pump_until(lambda: spoken)
    assert spoken[0].text == "Ого, червоне в терміналі. Допомогти розібратися?" and not spoken[0].solicited
    assert fake_ai.requests[-1].purpose is Purpose.INITIATIVE


def test_observer_stays_quiet_with_commentary_on_events(make_app, fake_ai):
    app = make_app()
    enable_local_ai(app)
    app.permissions.grant(Permission.SCREEN_CAPTURE)
    pump(50)
    requests = len(fake_ai.requests)
    app.bus.publish(VisionObserved(app_title="Термінал", scene="terminal", event="error_on_screen",
                                   confidence=0.9, description="червоне повідомлення про помилку"))
    pump(100)
    assert len(fake_ai.requests) == requests and app._observer.moments == 0


def test_observer_stops_before_the_ai_service(make_app, monkeypatch):
    app = make_app()
    app.start()
    order = []
    for name, obj, method in (("observer", app._observer, "close"), ("ai", app._ai, "shutdown")):
        original = getattr(obj, method)
        monkeypatch.setattr(obj, method, lambda n=name, f=original: (order.append(n), f()))
    app.shutdown()
    assert order == ["observer", "ai"]


def test_forgetting_what_she_saw_includes_the_observer(make_app, monkeypatch):
    app = make_app()
    calls = []
    monkeypatch.setattr(app._observer, "clear", lambda: calls.append("clear"))
    monkeypatch.setattr(app_module.QTimer, "singleShot", lambda *_a: None)  # no first-run dialog
    app.vision_clear_history()
    assert calls == ["clear"]
    app.wipe_history()
    assert calls == ["clear"] * 2
    app.wipe_all()
    assert len(calls) >= 3
