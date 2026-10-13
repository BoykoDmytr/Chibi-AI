from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QPushButton, QVBoxLayout

from chibi.settings.widgets import (
    NeedsBar,
    PortraitLabel,
    SliderRow,
    SubmitLineEdit,
    ai_provider_name,
    ai_title,
    ask_yes_no,
    count_uk,
    format_duration,
    format_local_date,
    format_local_datetime,
    format_usd,
    is_alive,
    percent,
    plural_uk,
    state_name_uk,
    text_avatar,
)

HOURS = ("година", "години", "годин")


@pytest.mark.parametrize(
    ("n", "expected"),
    [(0, "годин"), (1, "година"), (2, "години"), (4, "години"), (5, "годин"), (11, "годин"),
     (12, "годин"), (14, "годин"), (21, "година"), (22, "години"), (25, "годин"), (101, "година"),
     (111, "годин"), (112, "годин"), (1004, "години")],
)
def test_plural_uk(n, expected):
    assert plural_uk(n, *HOURS) == expected


def test_count_uk():
    assert count_uk(152, "гру", "гри", "ігор") == "152 гри"
    assert count_uk(1, "гру", "гри", "ігор") == "1 гру"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "менше хвилини"),
        (59.9, "менше хвилини"),
        (-5, "менше хвилини"),
        (float("nan"), "менше хвилини"),
        (float("inf"), "менше хвилини"),
        (60, "1 хвилина"),
        (3 * 60, "3 хвилини"),
        (45 * 60, "45 хвилин"),
        (3600, "1 година"),
        (2 * 3600 + 5 * 60, "2 години 5 хвилин"),
        (11 * 3600 + 21 * 60, "11 годин 21 хвилина"),
        (24 * 3600 + 59, "24 години"),
    ],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


def test_local_time_formatting_tolerates_garbage():
    assert format_local_datetime(float("nan")) == "—"
    assert format_local_date(1e30) == "—"
    assert format_local_datetime(1_700_000_000.0).count(":") == 1


def test_percent_uses_no_break_space():
    assert percent(42.4) == "42 %"


def test_state_name_fallback():
    assert state_name_uk("walk") == "Гуляє"
    assert state_name_uk("???") == "Чимось зайнята"


def test_slider_row_shows_live_value(qapp, track):
    row = track(SliderRow("Розмір", 50, 250, 100, formatter=lambda v: f"{v} %", description="опис"))
    seen = []
    row.valueChanged.connect(seen.append)
    assert row.value_label.text() == "100 %"
    row.setValue(175)
    assert row.value() == 175 and row.value_label.text() == "175 %" and seen == [175]
    row.setValue(999)
    assert row.value() == 250


@pytest.mark.parametrize(
    ("value", "inverted", "tone"),
    [(80, False, "pink"), (39, False, "warning"), (19, False, "danger"), (0, False, "danger"),
     (10, True, "pink"), (65, True, "warning"), (85, True, "danger")],
)
def test_needs_bar_tones(qapp, track, value, inverted, tone):
    bar = track(NeedsBar("Тест", "pink", inverted=inverted))
    bar.set_value(value)
    assert bar.tone() == tone
    assert bar.value() == value


def test_needs_bar_clamps_and_rejects_unknown_tone(qapp, track):
    bar = track(NeedsBar("Тест"))
    bar.set_value(150)
    assert bar.value() == 100
    bar.set_value(float("nan"))
    assert bar.value() == 0
    with pytest.raises(ValueError):
        NeedsBar("Тест", "rainbow")


def test_submit_line_edit_consumes_enter_inside_a_dialog(qapp, track):
    dialog = track(QDialog())
    layout = QVBoxLayout(dialog)
    edit = SubmitLineEdit()
    ok = QPushButton("OK")
    ok.setDefault(True)
    layout.addWidget(edit)
    layout.addWidget(ok)
    clicked, submitted = [], []
    ok.clicked.connect(lambda: clicked.append(True))
    edit.submitted.connect(lambda: submitted.append(edit.text()))
    dialog.show()
    edit.setText("abc")
    QTest.keyClick(edit, Qt.Key.Key_Return)
    QTest.keyClick(edit, Qt.Key.Key_Enter)
    assert submitted == ["abc", "abc"]
    assert clicked == []
    assert dialog.isVisible()


def test_portrait_label_falls_back_and_caches(qapp, track):
    calls = []
    red = QPixmap(40, 40)
    red.fill(QColor("red"))

    def loader(name):
        calls.append(name)
        if name == "boom":
            raise OSError("broken file")
        return red if name == "calm" else None

    label = track(PortraitLabel(loader, 64, fallback_text="Мія"))
    assert label.show_portrait("missing", "boom", "calm") == "calm"
    assert label.show_portrait("missing", "boom", "calm") == "calm"
    assert calls == ["missing", "boom", "calm"]  # each name loaded once
    assert label.pixmap().deviceIndependentSize().width() == 64
    assert label.show_portrait("missing") == ""
    assert label.current == ""
    assert not label.pixmap().isNull()


def test_text_avatar_draws_first_letter_in_a_circle(qapp):
    avatar = text_avatar("мія", 48, 2.0)
    assert avatar.devicePixelRatio() == 2.0
    assert avatar.width() == 96
    image = avatar.toImage()
    assert image.pixelColor(0, 0).alpha() == 0  # transparent corner
    assert image.pixelColor(10, 48).alpha() == 255  # inside the circle


# --- Phase 2 helpers ------------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "text"), [(0.034, "$0.03"), (1, "$1.00"), (12.5, "$12.50"),
                                             (-1, "$0.00"), (float("nan"), "$0.00")])
def test_format_usd(value, text):
    assert format_usd(value) == text


def test_ai_titles():
    assert ai_title("anthropic", "claude-opus-5-5") == "Anthropic Claude · claude-opus-5-5"
    assert ai_title("ollama", "  ") == "Ollama"
    assert ai_provider_name("openai") == "OpenAI-сумісний сервер"
    assert ai_provider_name("something-new") == "something-new"


@pytest.mark.parametrize("value", [0, 10, 50, 100])
def test_needs_bar_without_alerts_keeps_its_colour(qapp, track, value):
    bar = track(NeedsBar("Цікавість", "lilac", alerts=False))
    bar.set_value(value)
    assert bar.tone() == "lilac"


def test_is_alive_tracks_deleted_widgets(qapp):
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QWidget

    widget = QWidget()
    assert is_alive(widget)
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not is_alive(widget)
    assert not is_alive(None)


def test_ask_yes_no_is_plain_text_and_defaults_to_cancel(qapp):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox

    seen = {}

    def inspect_and_press_enter():
        box = QApplication.activeModalWidget()
        assert isinstance(box, QMessageBox)
        seen["format"] = box.textFormat()
        seen["text"] = box.text()
        seen["default"] = box.defaultButton().text()
        QTest.keyClick(box, Qt.Key.Key_Return)  # a stray Enter must not mean "yes"

    QTimer.singleShot(0, inspect_and_press_enter)
    answer = ask_yes_no(None, "Забути?", "<b>спогад</b>")
    assert answer is False
    assert seen == {"format": Qt.TextFormat.PlainText, "text": "<b>спогад</b>", "default": "Скасувати"}
