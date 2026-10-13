from __future__ import annotations

import logging
from dataclasses import replace

import pytest
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel

from chibi.settings import chat_window
from chibi.settings.chat_window import (
    MAX_INPUT_CHARS,
    OFFLINE_BANNER,
    PLACEHOLDER_OFFLINE,
    PLACEHOLDER_ONLINE,
    TIMEOUT_LINE,
    TYPING_FRAMES,
    ChatWindow,
    visible_stream_text,
)
from chibi.settings.facade import ChatReply
from tests.settings.fakes import StreamScript


@pytest.fixture
def make_window(facade, track):
    def make() -> ChatWindow:
        window = track(ChatWindow(facade))
        window.show()
        return window

    return make


def bubbles(window: ChatWindow, kind: str) -> list[str]:
    return [label.text().replace("​", "") for label in window.messages.findChildren(QLabel)
            if label.property("bubble") == kind]


def test_shows_honest_offline_banner_with_the_reason(make_window, facade):
    facade.ai_status_view = replace(facade.ai_status_view, provider="anthropic", reason="немає API-ключа")
    window = make_window()
    assert window.isVisible()
    assert window.banner.text() == "Офлайн-режим: немає API-ключа — я розумію лише прості фрази"
    assert window.input.placeholderText() == PLACEHOLDER_OFFLINE


def test_banner_names_provider_and_model_when_online(make_window, facade):
    facade.ai_status_view = replace(facade.ai_status_view, provider="anthropic", model="claude-opus-5-5",
                                    online=True, reason="", cloud=True)
    window = make_window()
    assert window.banner.text() == "AI: Anthropic Claude · claude-opus-5-5"
    assert window.input.placeholderText() == PLACEHOLDER_ONLINE


def test_banner_falls_back_when_status_cannot_be_read(make_window, facade, caplog):
    facade.ai_status_error = RuntimeError("boom")
    with caplog.at_level(logging.ERROR):
        window = make_window()
        window.refresh_status()
    assert window.banner.text() == OFFLINE_BANNER
    assert len([r for r in caplog.records if "AI status" in r.getMessage()]) == 1  # logged once


def test_banner_is_refreshed_on_show_and_after_each_reply(make_window, facade):
    window = make_window()
    assert window.banner.text().startswith("Офлайн-режим: AI вимкнено")
    online = replace(facade.ai_status_view, provider="ollama", model="qwen3:8b", online=True, reason="")
    facade.ai_status_view = online
    window.hide()
    window.show()
    assert window.banner.text() == "AI: Ollama · qwen3:8b"
    facade.ai_status_view = replace(online, online=False, reason="сервіс недоступний")
    window.input.setText("привіт")
    window.send()
    assert window.banner.text() == "Офлайн-режим: сервіс недоступний — я розумію лише прості фрази"


def test_send_button_sends_and_shows_reply(make_window, facade):
    facade.replies = [ChatReply(text="Привіт-привіт!", state="happy")]
    window = make_window()
    window.input.setText("  привіт  ")
    window.send_button.click()
    assert facade.chat_sent == ["привіт"]
    assert bubbles(window, "user") == ["привіт"]
    assert bubbles(window, "her") == ["Привіт-привіт!"]
    assert window.input.text() == ""


def test_enter_sends(make_window, facade):
    window = make_window()
    window.input.setText("як справи?")
    QTest.keyClick(window.input, Qt.Key.Key_Return)
    assert facade.chat_sent == ["як справи?"]
    window.input.setText("ще")
    QTest.keyClick(window.input, Qt.Key.Key_Enter)
    assert facade.chat_sent == ["як справи?", "ще"]


@pytest.mark.parametrize("blank", ["", "   ", "\t \t"])
def test_blank_input_is_ignored(make_window, facade, blank):
    window = make_window()
    window.input.setText(blank)
    window.send_button.click()
    QTest.keyClick(window.input, Qt.Key.Key_Return)
    assert facade.chat_sent == []
    assert window.message_count() == 0


def test_input_is_limited_to_500_chars(make_window, facade):
    window = make_window()
    assert window.input.maxLength() == MAX_INPUT_CHARS == 500
    window.input.setText("я" * 800)
    window.send()
    assert len(facade.chat_sent[0]) == 500


def test_quiet_action_shows_a_system_line(make_window, facade):
    facade.replies = [ChatReply(text="Добре, мовчу.", action="quiet_on")]
    window = make_window()
    window.input.setText("помовч")
    window.send()
    assert bubbles(window, "system") == ["🔇 Мія помовчить"]


def test_system_line_uses_current_name(make_window, facade):
    s = facade.settings()
    s.character.name = "Ліна"
    facade.apply_settings(s)
    facade.replies = [ChatReply(text="Ням!", action="fed")]
    window = make_window()
    window.input.setText("погодувати")
    window.send()
    assert bubbles(window, "system") == ["🍙 Ліна поїла"]


def test_no_action_or_unknown_action_adds_no_system_line(make_window, facade):
    facade.replies = [ChatReply(text="Привіт!"), ChatReply(text="Що?", action="something_new")]
    window = make_window()
    for text in ("привіт", "абв"):
        window.input.setText(text)
        window.send()
    assert bubbles(window, "system") == []
    assert window.message_count() == 4


def test_reply_without_text_shows_only_the_action(make_window, facade):
    facade.replies = [ChatReply(text="", action="rest")]
    window = make_window()
    window.input.setText("іди спати")
    window.send()
    assert bubbles(window, "her") == []
    assert bubbles(window, "system") == ["💤 Мія пішла відпочити"]


def test_user_text_is_never_rendered_as_html(make_window):
    window = make_window()
    window.input.setText("<b>жирний</b> <img src=x>")
    window.send()
    user = [label for label in window.messages.findChildren(QLabel) if label.property("bubble") == "user"]
    assert user[0].textFormat() == Qt.TextFormat.PlainText
    assert user[0].text() == "<b>жирний</b> <img src=x>"


def test_long_words_get_break_opportunities(make_window):
    window = make_window()
    word = "а" * 120
    window.input.setText(word)
    window.send()
    label = next(lb for lb in window.messages.findChildren(QLabel) if lb.property("bubble") == "user")
    assert "​" in label.text()
    assert label.text().replace("​", "") == word
    assert label.width() <= window.scroll.viewport().width()


def test_failing_reply_shows_error_line_and_never_logs_the_message(make_window, facade, caplog):
    secret = "мій пароль від банку 12345"
    # even an exception that quotes the input must not leak it into the log
    facade.chat_error = ValueError(f"cannot parse {secret!r}")
    window = make_window()
    window.input.setText(secret)
    with caplog.at_level(logging.DEBUG):
        window.send()
    assert bubbles(window, "system") == ["Щось пішло не так — спробуй ще раз."]
    assert any("chat reply failed" in record.getMessage() for record in caplog.records)
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(record.exc_info is None for record in caplog.records)


def test_history_is_memory_only_and_capped(make_window, facade, monkeypatch):
    monkeypatch.setattr(chat_window, "MAX_MESSAGES", 6)
    window = make_window()
    assert "не зберігається" in window.scroll.toolTip()
    for i in range(10):
        window.input.setText(f"повідомлення {i}")
        window.send()
    assert window.message_count() == 6
    QCoreApplication.processEvents()
    assert bubbles(window, "user")[-1] == "повідомлення 9"


def test_auto_scrolls_to_the_newest_message(make_window, facade):
    window = make_window()
    for i in range(30):
        window.input.setText(f"рядок {i}")
        window.send()
        QCoreApplication.processEvents()
    QCoreApplication.processEvents()
    bar = window.scroll.verticalScrollBar()
    assert bar.maximum() > 0
    assert bar.value() == bar.maximum()


def test_her_messages_have_a_small_portrait_with_fallback(make_window, facade):
    from PySide6.QtGui import QColor, QPixmap

    window = make_window()
    window.input.setText("привіт")
    window.send()
    facade.portraits = {"joy": QPixmap(16, 16)}
    facade.portraits["joy"].fill(QColor("red"))
    facade.replies = [ChatReply(text="Ура!", state="happy")]
    window.input.setText("ти молодець")
    window.send()
    from chibi.settings.widgets import PortraitLabel

    avatars = window.messages.findChildren(PortraitLabel)
    assert [a.current for a in avatars] == ["", "joy"]
    assert all(a.width() == chat_window.AVATAR_SIZE for a in avatars)


# --- streaming (Phase 2) -------------------------------------------------------------------------


def her_bubbles(window: ChatWindow) -> list[QLabel]:
    return [label for label in window.messages.findChildren(QLabel) if label.property("bubble") == "her"]


def test_her_bubble_appears_at_once_with_a_typing_indicator(make_window, facade):
    facade.deferred = True
    window = make_window()
    window.input.setText("розкажи щось")
    window.send()
    assert facade.chat_sent == ["розкажи щось"]
    assert bubbles(window, "user") == ["розкажи щось"]
    assert bubbles(window, "her") == [TYPING_FRAMES[-1]]
    assert window.is_waiting()
    assert not window.send_button.isEnabled()
    assert window.typing_timer.isActive()
    window.typing_timer.timeout.emit()
    assert bubbles(window, "her")[0] in TYPING_FRAMES


def test_deltas_are_appended_and_the_final_reply_replaces_them(make_window, facade):
    facade.deferred = True
    facade.streams = [StreamScript(deltas=["При", "віт", ", як ти?"],
                                   reply=ChatReply(text="Привіт, як ти? 😊", state="happy"))]
    facade.portraits = {"joy": QPixmap(16, 16)}
    window = make_window()
    window.input.setText("привіт")
    window.send()
    facade.resolve("delta")
    assert bubbles(window, "her") == ["При"]
    assert not window.typing_timer.isActive()  # real text replaced the indicator
    facade.resolve("delta")
    facade.resolve("delta")
    assert bubbles(window, "her") == ["Привіт, як ти?"]
    assert not window.send_button.isEnabled()
    facade.resolve("done")
    assert bubbles(window, "her") == ["Привіт, як ти? 😊"]  # replaced, not appended
    assert window.send_button.isEnabled() and not window.is_waiting()
    from chibi.settings.widgets import PortraitLabel

    assert [a.current for a in window.messages.findChildren(PortraitLabel)] == ["joy"]
    assert window.message_count() == 2


def test_bubble_grows_with_the_stream(make_window, facade):
    facade.deferred = True
    facade.streams = [StreamScript(deltas=["Ну", " це дуже-дуже довга відповідь " * 6],
                                   reply=ChatReply(text="ок"))]
    window = make_window()
    window.input.setText("ну?")
    window.send()
    facade.resolve("delta")
    short = her_bubbles(window)[0].width()
    facade.resolve("delta")
    bubble = her_bubbles(window)[0]
    assert bubble.width() > short
    assert bubble.width() <= window.scroll.viewport().width()


def test_send_is_ignored_while_a_reply_is_in_flight(make_window, facade):
    facade.deferred = True
    window = make_window()
    window.input.setText("перше")
    window.send()
    window.input.setText("друге")
    QTest.keyClick(window.input, Qt.Key.Key_Return)
    window.send_button.click()  # disabled: nothing happens
    assert facade.chat_sent == ["перше"]
    assert window.input.text() == "друге"  # kept for later, not lost
    facade.resolve("done")
    window.send()
    assert facade.chat_sent == ["перше", "друге"]


def test_offline_reply_arrives_through_on_done_only(make_window, facade):
    facade.replies = [ChatReply(text="Я поки розумію лише прості фрази.", state="thinking")]
    window = make_window()
    window.input.setText("що таке квантова механіка?")
    window.send()
    assert bubbles(window, "her") == ["Я поки розумію лише прості фрази."]
    assert window.send_button.isEnabled() and not window.typing_timer.isActive()


def test_streamed_reply_with_action_adds_the_system_line(make_window, facade):
    facade.deferred = True
    facade.streams = [StreamScript(deltas=["Добре"],
                                   reply=ChatReply(text="Добре, мовчу.", action="quiet_on"))]
    window = make_window()
    window.input.setText("помовч")
    window.send()
    facade.resolve("delta")
    facade.resolve("done")
    assert bubbles(window, "her") == ["Добре, мовчу."]
    assert bubbles(window, "system") == ["🔇 Мія помовчить"]


def test_reply_without_text_removes_the_typing_bubble(make_window, facade):
    facade.deferred = True
    facade.streams = [StreamScript(reply=ChatReply(text="", action="fed"))]
    window = make_window()
    window.input.setText("погодуй себе")
    window.send()
    assert len(her_bubbles(window)) == 1
    facade.resolve("done")
    assert bubbles(window, "her") == []
    assert bubbles(window, "system") == ["🍙 Мія поїла"]
    assert window.message_count() == 2


def test_control_line_is_never_shown_while_streaming(make_window, facade):
    facade.deferred = True
    facade.streams = [StreamScript(deltas=["<<state=", "happy;action=none>>\n", "Ура!"],
                                   reply=ChatReply(text="Ура!"))]
    window = make_window()
    window.input.setText("я виграв")
    window.send()
    facade.resolve("delta")
    facade.resolve("delta")
    assert bubbles(window, "her") == [TYPING_FRAMES[-1]]  # still "typing"
    facade.resolve("delta")
    assert bubbles(window, "her") == ["Ура!"]


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("Привіт", "Привіт"),
        ("<", ""),
        ("<<state=ha", ""),
        ("<<state=happy;action=none>>", ""),
        ("<<state=happy;action=none>>\nПривіт", "Привіт"),
        ("  <<state=happy>> Привіт", "Привіт"),
        ("<<не контрольний рядок\nа текст", "<<не контрольний рядок\nа текст"),
        ("2 < 3", "2 < 3"),
    ],
)
def test_visible_stream_text(raw, shown):
    assert visible_stream_text(raw) == shown


def test_failing_stream_shows_error_line_and_removes_typing_bubble(make_window, facade):
    facade.chat_error = RuntimeError("network down")
    window = make_window()
    window.input.setText("привіт")
    window.send()
    assert bubbles(window, "her") == []
    assert bubbles(window, "system") == ["Щось пішло не так — спробуй ще раз."]
    assert window.send_button.isEnabled() and not window.is_waiting()


def test_watchdog_ends_a_reply_that_never_comes_and_drops_the_late_answer(make_window, facade):
    facade.deferred = True
    window = make_window()
    window.input.setText("агов")
    window.send()
    assert window.watchdog.isActive()
    window.watchdog.timeout.emit()
    assert window.send_button.isEnabled() and not window.is_waiting()
    assert bubbles(window, "her") == []
    assert bubbles(window, "system") == [TIMEOUT_LINE]
    facade.resolve("done")  # far too late: ignored
    assert bubbles(window, "her") == []


def test_watchdog_keeps_partial_text(make_window, facade):
    facade.deferred = True
    facade.streams = [StreamScript(deltas=["Зараз розкажу"], reply=ChatReply(text="x"))]
    window = make_window()
    window.input.setText("розкажи")
    window.send()
    facade.resolve("delta")
    window.watchdog.timeout.emit()
    assert bubbles(window, "her") == ["Зараз розкажу"]


def test_late_callbacks_after_the_window_is_deleted_do_not_crash(facade, qapp):
    from PySide6.QtCore import QEvent

    facade.deferred = True
    facade.streams = [StreamScript(deltas=["а"], reply=ChatReply(text="а"))]
    window = ChatWindow(facade)
    window.show()
    window.input.setText("привіт")
    window.send()
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    facade.resolve("delta")
    facade.resolve("done")  # no RuntimeError from a deleted widget


def test_stream_never_logs_the_text(make_window, facade, caplog):
    secret = "мій пароль qwerty123"
    facade.deferred = True
    facade.streams = [StreamScript(deltas=[secret], reply=ChatReply(text=secret))]
    window = make_window()
    with caplog.at_level(logging.DEBUG):
        window.input.setText(secret)
        window.send()
        facade.resolve("delta")
        facade.resolve("done")
    assert all(secret not in record.getMessage() for record in caplog.records)


# --- microphone button (Phase 4) -----------------------------------------------------------------


@pytest.fixture
def make_voice_window(facade, track):
    def make(on_mic=None) -> ChatWindow:
        window = track(ChatWindow(facade, on_mic=on_mic))
        window.show()
        return window

    return make


def test_mic_button_is_disabled_until_voice_is_set_up(make_window):
    window = make_window()
    button = window.mic_button
    assert button.isVisible() and not button.isEnabled()
    assert button.toolTip() == "Голосове введення недоступне: спершу налаштуй голос (Налаштування → Голос)."
    assert button.accessibleName() == "Мікрофон"


def test_mic_button_tooltip_names_the_hotkey_and_click_toggles(make_voice_window):
    clicks = []
    window = make_voice_window(on_mic=lambda: clicks.append(1))
    window.set_voice_input(True, hotkey="Ctrl + Alt + Пробіл")
    button = window.mic_button
    assert button.isEnabled()
    assert button.toolTip() == "Сказати голосом (Ctrl + Alt + Пробіл)"
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert clicks == [1]
    assert not button.isChecked()  # pressed only once the voice really listens


def test_mic_button_shows_listening_state(make_voice_window):
    window = make_voice_window(on_mic=lambda: None)
    window.set_voice_input(True)
    window.set_voice_state("listening")
    button = window.mic_button
    assert button.isChecked() and window.is_listening()
    assert button.text() == chat_window.MIC_LISTENING_TEXT
    assert button.toolTip() == chat_window.MIC_LISTENING_TOOLTIP
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)  # stop: still pressed until the state changes
    assert button.isChecked()
    window.set_voice_state("transcribing")
    assert not button.isChecked() and not button.isEnabled()
    assert button.text() == chat_window.MIC_BUSY_TEXT
    window.set_voice_state("idle")
    assert button.isEnabled() and button.text() == chat_window.MIC_IDLE_TEXT
    assert button.toolTip() == chat_window.MIC_TOOLTIP


def test_unavailable_voice_input_says_why(make_voice_window):
    clicks = []
    window = make_voice_window(on_mic=lambda: clicks.append(1))
    window.set_voice_input(False, reason="Мікрофон вимкнено в «Приватності».")
    button = window.mic_button
    assert not button.isEnabled()
    assert button.toolTip() == "Голосове введення недоступне: мікрофон вимкнено в «Приватності»."
    button.click()
    assert clicks == []


def test_mic_button_without_a_callback_stays_disabled(make_window):
    window = make_window()
    window.set_voice_input(True)
    assert not window.mic_button.isEnabled()


def test_mic_button_is_keyboard_reachable_between_input_and_send(make_voice_window):
    clicks = []
    window = make_voice_window(on_mic=lambda: clicks.append(1))
    window.set_voice_input(True)
    assert window.mic_button.focusPolicy() & Qt.FocusPolicy.TabFocus
    assert window.input.nextInFocusChain() is window.mic_button
    assert window.mic_button.nextInFocusChain() is window.send_button
    window.mic_button.setFocus()
    QTest.keyClick(window.mic_button, Qt.Key.Key_Space)
    assert clicks == [1]


def test_failing_mic_callback_is_contained(make_voice_window, caplog):
    def broken():
        raise RuntimeError("секретний текст")

    window = make_voice_window(on_mic=broken)
    window.set_voice_input(True)
    window.mic_button.click()
    assert "voice input toggle failed: RuntimeError" in caplog.text
    assert "секретний" not in caplog.text


def test_chat_still_sends_with_the_mic_button(make_voice_window, facade):
    window = make_voice_window(on_mic=lambda: None)
    window.input.setText("привіт")
    window.send_button.click()
    assert facade.chat_sent == ["привіт"]
