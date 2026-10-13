"""Phases 3-5 wiring in ChibiApp: voice routing and the microphone, screen
vision from the menu and the chat, gaming companion mode, gifts and mini-games,
the new settings pages, wipe-all and shutdown."""

from dataclasses import replace

import pytest

import chibi.app as app_module
from chibi.ai.secrets import MemorySecretStore
from chibi.app import BUSY_REPLY, LOOKING_REPLY, ChibiApp
from chibi.core.events import Permission, SpeechRequested, VoiceTranscript
from chibi.games import GAME_IDS, GAME_TITLES
from chibi.settings.settings_dialog import PAGES, SettingsDialog
from chibi.settings.theme import apply_dark_theme
from chibi.vision.integration import REASON_OFF
from chibi.voice.integration import AZURE_KEY_NAME
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
