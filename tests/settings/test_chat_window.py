from __future__ import annotations

import logging

import pytest
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel

from chibi.settings import chat_window
from chibi.settings.chat_window import MAX_INPUT_CHARS, OFFLINE_BANNER, ChatWindow
from chibi.settings.facade import ChatReply


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


def test_shows_honest_offline_banner(make_window):
    window = make_window()
    assert window.isVisible()
    assert window.banner.text() == OFFLINE_BANNER
    assert "AI ще не підключено" in OFFLINE_BANNER


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
