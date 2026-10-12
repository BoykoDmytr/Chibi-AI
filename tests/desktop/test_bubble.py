from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import QPointF, QRect, QSize, Qt
from PySide6.QtGui import QPixmap

from chibi.desktop.bubble import (
    CHARS_PER_SECOND,
    MAX_CHARS,
    PADDING_X,
    PORTRAIT_GAP,
    PORTRAIT_SIZE,
    WRAP_WIDTH,
    SpeechBubble,
    clamp_text,
    place_bubble,
)
from tests.desktop.helpers import MIA_DIR, alpha_bbox


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def bubble(qapp, clock) -> Iterator[SpeechBubble]:
    b = SpeechBubble(time_source=clock)
    yield b
    b.hide_now()
    b.close()
    b.deleteLater()


LONG = ("Знаєш, я тут подумала: якщо ти ще трохи посидиш за комп'ютером, то я встигну "
        "порахувати всі пікселі на екрані. А потім ще раз, для певності. ") * 3


def test_flags_never_take_focus_or_input(bubble):
    flags = bubble.windowFlags()
    for flag in (Qt.WindowType.FramelessWindowHint, Qt.WindowType.Tool, Qt.WindowType.WindowStaysOnTopHint,
                 Qt.WindowType.WindowDoesNotAcceptFocus, Qt.WindowType.WindowTransparentForInput):
        assert flags & flag, flag
    assert bubble.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert bubble.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_text_does_nothing(bubble, text):
    assert not bubble.show_text(text, None, 4)
    assert not bubble.isVisible()
    assert not bubble.is_showing()


def test_show_follow_and_hide(bubble):
    assert bubble.show_text("Привіт!", None, 4)
    bubble.follow(QPointF(400, 500), QRect(0, 0, 1920, 1040))
    assert bubble.is_showing() and bubble.text == "Привіт!"
    p = bubble.placement
    assert p.side == "above"
    assert bubble.geometry() == p.window
    tip = QPointF(p.window.x() + p.tip.x(), p.window.y() + p.tip.y())
    assert tip == QPointF(400, 500)  # the tail points at the head
    bubble.hide_now()
    assert not bubble.isVisible() and not bubble.is_showing()


def test_typewriter_reveal(bubble, clock):
    text = "Ой, а що це ти там робиш? Покажи!"
    bubble.show_text(text, None, 6)
    assert bubble.revealing
    assert len(bubble.revealed_text) <= 1
    clock.now += 0.5
    bubble._on_reveal_tick()
    assert len(bubble.revealed_text) == int(0.5 * CHARS_PER_SECOND) + 1
    assert text.startswith(bubble.revealed_text)
    clock.now += 5.0
    bubble._on_reveal_tick()
    assert bubble.revealed_text == text
    assert not bubble.revealing  # the timer runs only while revealing


def test_auto_hide_timer_and_fade(bubble):
    bubble.show_text("Бувай!", None, 3.5)
    assert bubble._hide_timer.isActive()
    assert bubble._hide_timer.interval() == 3500
    bubble.fade_out()
    assert not bubble.is_showing()  # on its way out
    bubble.show_text("Ні, стривай!", None, 3)  # a new line cancels the fade
    assert bubble.is_showing()
    assert bubble.windowOpacity() == 1.0


def test_long_text_is_capped(bubble):
    assert len(LONG) > MAX_CHARS
    bubble.show_text(LONG, None, 12)
    assert len(bubble.text) == MAX_CHARS
    assert bubble.text.endswith("…")


def test_clamp_text_strips_control_characters():
    assert clamp_text("  a\x00b\x07c\nd  ") == "abc\nd"
    assert clamp_text("x" * 10, limit=5) == "xxxx…"


def test_wraps_at_wrap_width(bubble):
    bubble.show_text("Коротко.", None, 4)
    short = bubble.body_size()
    bubble.show_text(LONG, None, 12)
    long = bubble.body_size()
    assert long.width() <= WRAP_WIDTH + 2 * PADDING_X + 1
    assert long.width() > short.width()
    assert long.height() > 3 * short.height()  # several lines


def test_portrait_makes_room(bubble):
    bubble.show_text("Привіт!", None, 4)
    without = bubble.body_size()
    portrait = QPixmap(str(MIA_DIR / "portraits" / "smile.png"))
    bubble.show_text("Привіт!", portrait, 4)
    with_portrait = bubble.body_size()
    assert with_portrait.width() == without.width() + PORTRAIT_SIZE + PORTRAIT_GAP
    assert with_portrait.height() >= PORTRAIT_SIZE


@pytest.mark.parametrize("tip", [(5, 5), (315, 5), (5, 195), (315, 195), (160, 100), (160, 10), (-50, 400)])
def test_stays_inside_a_small_screen_area(bubble, tip):
    area = QRect(0, 0, 320, 200)
    bubble.show_text(LONG[:150], None, 6)
    bubble.follow(QPointF(*tip), area)
    p = bubble.placement
    body = p.body.translated(p.window.topLeft().toPointF())
    assert body.left() >= area.left() - 0.01 and body.right() <= area.right() + 1.01
    assert body.top() >= area.top() - 0.01 and body.bottom() <= area.bottom() + 1.01


def test_paints_with_partial_reveal_and_emoji(bubble, clock):
    bubble.show_text("Ура! 🎉 Ми виграли! 😊", QPixmap(str(MIA_DIR / "portraits" / "joy.png")), 5)
    bubble.follow(QPointF(400, 400), QRect(0, 0, 1000, 800))
    clock.now += 0.15
    bubble._on_reveal_tick()
    partial = alpha_bbox(bubble.grab().toImage())
    clock.now += 10
    bubble._on_reveal_tick()
    assert bubble.revealed_text == bubble.text
    assert alpha_bbox(bubble.grab().toImage()) is not None and partial is not None


def test_place_bubble_prefers_above_then_side_then_below():
    area = QRect(0, 0, 1000, 800)
    size = QSize(200, 80)
    assert place_bubble(size, QPointF(500, 400), area).side == "above"
    near_top = place_bubble(size, QPointF(300, 30), area)
    assert near_top.side == "right"  # more room on the right
    assert place_bubble(size, QPointF(900, 30), area).side == "left"
    narrow = QRect(0, 0, 240, 800)
    assert place_bubble(size, QPointF(120, 30), narrow).side == "below"


def test_place_bubble_tail_follows_head_when_clamped():
    area = QRect(0, 0, 1000, 800)
    p = place_bubble((200, 80), QPointF(990, 400), area)
    assert p.side == "above"
    assert p.window.x() + p.body.right() <= 1000.01
    assert p.window.x() + p.tip.x() == pytest.approx(990)


def test_absurd_durations_are_capped(bubble):
    bubble.show_text("Довго-довго", None, 1e12)
    assert bubble._hide_timer.isActive()
    assert bubble._hide_timer.interval() == 600_000
