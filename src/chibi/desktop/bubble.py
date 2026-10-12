"""Speech bubble: a separate click-through window next to her head.

A soft cream bubble with a subtle shadow, dark text wrapped at ~260 logical
px, an optional portrait on the left and a tail pointing at her head.

* It is its own top-level window (not part of the character window) so the
  character window stays small and the bubble can sit wherever there is
  room: above her head, beside it near the top of the screen, or below as a
  last resort - always inside the screen's work area.
* It never takes focus and is transparent for input: it is decoration, and
  a bubble that swallowed clicks would block whatever is under it.
* Text appears typewriter-style (~45 characters/s).  The reveal timer runs
  only while revealing; the layout is computed for the full text up front
  so lines never re-wrap as words appear.
* After *seconds* it fades out (window opacity animation).
"""

from __future__ import annotations

import bisect
import logging
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QEasingCurve, QPointF, QPropertyAnimation, QRect, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
    QShowEvent,
    QTextLayout,
    QTextOption,
)
from PySide6.QtWidgets import QWidget

from chibi.desktop import platform_win

log = logging.getLogger(__name__)

#: Text wraps at this width (logical px).
WRAP_WIDTH = 260.0
#: Longer lines are cut with an ellipsis - a bubble is not a document.
MAX_CHARS = 400
CHARS_PER_SECOND = 45.0
PORTRAIT_SIZE = 48
PADDING_X = 14.0
PADDING_Y = 10.0
PORTRAIT_GAP = 10.0
RADIUS = 14.0
TAIL = 12.0
TAIL_HALF_WIDTH = 9.0
#: Room around the bubble for the drawn shadow.
SHADOW = 6.0
#: Gap between the tail tip and her head.
HEAD_GAP = 2.0
FADE_MS = 280
#: Longest a single bubble stays up.
MAX_SECONDS = 600.0
REVEAL_INTERVAL_MS = 30

_FILL = QColor(255, 249, 238)
_BORDER = QColor(222, 205, 186)
_TEXT = QColor(52, 40, 36)
_SHADOW = QColor(60, 40, 30)


def clamp_text(text: str, limit: int = MAX_CHARS) -> str:
    """Trim, cap at *limit* characters with an ellipsis and drop control characters."""
    text = "".join(ch for ch in text.strip() if ch == "\n" or ch >= " ")
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


@dataclass(frozen=True)
class Placement:
    """Where the bubble goes: its window rect (global) and tail geometry
    in window coordinates.  ``side`` is the side of the head the bubble is on."""

    window: QRect
    body: QRectF
    tip: QPointF
    side: str  # "above" | "below" | "left" | "right"


def place_bubble(body_size: QSize | tuple[float, float], tip: QPointF, area: QRect) -> Placement:
    """Pure placement logic (tested without windows).

    Above the head if it fits, otherwise beside it (towards the roomier
    side), otherwise below; always clamped into *area*.  The tail tip stays
    on the head, the tail base slides along the bubble edge.
    """
    if isinstance(body_size, QSize):
        bw, bh = float(body_size.width()), float(body_size.height())
    else:
        bw, bh = float(body_size[0]), float(body_size[1])
    left, top = float(area.left()), float(area.top())
    right, bottom = left + area.width(), top + area.height()
    tx, ty = tip.x(), tip.y()

    def clamp(v: float, lo: float, hi: float) -> float:
        return lo if hi < lo else min(max(v, lo), hi)

    room_above = ty - HEAD_GAP - TAIL - top
    room_below = bottom - (ty + HEAD_GAP + TAIL)
    room_left = tx - HEAD_GAP - TAIL - left
    room_right = right - (tx + HEAD_GAP + TAIL)
    if room_above >= bh + SHADOW:
        side = "above"
    elif max(room_left, room_right) >= bw + SHADOW:
        side = "right" if room_right >= room_left else "left"
    elif room_below >= bh + SHADOW:
        side = "below"
    else:
        side = "above" if room_above >= room_below else "below"

    if side in ("above", "below"):
        bx = clamp(tx - bw / 2, left + SHADOW, right - SHADOW - bw)
        by = ty - HEAD_GAP - TAIL - bh if side == "above" else ty + HEAD_GAP + TAIL
        by = clamp(by, top + SHADOW, bottom - SHADOW - bh)
    else:
        bx = tx + HEAD_GAP + TAIL if side == "right" else tx - HEAD_GAP - TAIL - bw
        bx = clamp(bx, left + SHADOW, right - SHADOW - bw)
        by = clamp(ty - bh / 2, top + SHADOW, bottom - SHADOW - bh)

    # The window covers the body (plus shadow) and reaches the tail tip.
    wl = math.floor(min(bx - SHADOW, tx))
    wt = math.floor(min(by - SHADOW, ty))
    wr = math.ceil(max(bx + bw + SHADOW, tx))
    wb = math.ceil(max(by + bh + SHADOW, ty))
    window = QRect(wl, wt, max(1, wr - wl), max(1, wb - wt))
    body = QRectF(bx - wl, by - wt, bw, bh)
    return Placement(window, body, QPointF(tx - wl, ty - wt), side)


class SpeechBubble(QWidget):
    def __init__(self, parent: QWidget | None = None, *,
                 time_source: Callable[[], float] = time.monotonic) -> None:
        flags = (Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint
                 | Qt.WindowType.WindowDoesNotAcceptFocus | Qt.WindowType.WindowTransparentForInput
                 | Qt.WindowType.NoDropShadowWindowHint)
        super().__init__(parent, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setWindowTitle("Chibi bubble")
        self._now = time_source
        self._text = ""
        self._portrait: QPixmap | None = None
        self._layout: QTextLayout | None = None
        self._text_size = QSize(0, 0)
        self._body_size = QSize(0, 0)
        self._shown_at = 0.0
        self._revealed = 0
        self._placement: Placement | None = None
        self._follow_key: tuple | None = None
        self._last_anchor: tuple[QPointF, QRect] | None = None
        self._shape_cache: tuple[Placement, QPainterPath] | None = None
        #: UTF-16 offset of each character (QTextLayout counts UTF-16 units).
        self._utf16: list[int] = [0]
        self._fading = False
        self._no_activate_hwnd = 0

        font = QFont(self.font())
        font.setPointSizeF(max(font.pointSizeF(), 10.5))
        self._font = font

        self._reveal_timer = QTimer(self)
        self._reveal_timer.setInterval(REVEAL_INTERVAL_MS)
        self._reveal_timer.timeout.connect(self._on_reveal_tick)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.fade_out)
        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(FADE_MS)
        self._fade.setEasingCurve(QEasingCurve.Type.InQuad)
        self._fade.finished.connect(self._on_fade_finished)

    # -- public API -----------------------------------------------------------------

    @property
    def text(self) -> str:
        return self._text

    @property
    def revealed_text(self) -> str:
        return self._text[: self._revealed]

    @property
    def placement(self) -> Placement | None:
        return self._placement

    @property
    def revealing(self) -> bool:
        return self._reveal_timer.isActive()

    def body_size(self) -> QSize:
        return QSize(self._body_size)

    def is_showing(self) -> bool:
        """Visible and not on its way out."""
        return self.isVisible() and not self._fading

    def show_text(self, text: str, portrait: QPixmap | None = None, seconds: float = 4.0) -> bool:
        """Show *text* for *seconds* (empty text does nothing).  Returns True if shown.

        Call :meth:`follow` right after so it appears in the right place.
        """
        text = clamp_text(text or "")
        if not text:
            return False
        self._fade.stop()
        self._fading = False
        self.setWindowOpacity(1.0)
        self._text = text
        self._portrait = self._prepare_portrait(portrait)
        self._build_layout()
        self._shown_at = self._now()
        self._revealed = 0
        self._update_reveal()
        if self._revealed < len(self._text):
            self._reveal_timer.start()
        if math.isfinite(seconds) and seconds > 0:
            # QTimer intervals are int32 milliseconds: cap absurd durations
            self._hide_timer.start(max(1, round(min(seconds, MAX_SECONDS) * 1000)))
        else:
            self._hide_timer.stop()
        self._follow_key = None
        if self._last_anchor is not None:
            # Re-place for the new size at once; follow() refines it on the next frame.
            self.follow(*self._last_anchor)
        self.show()
        self.update()
        return True

    def follow(self, head_global: QPointF, screen_area: QRect) -> None:
        """Point the tail at *head_global* (top of her head) and stay inside *screen_area*."""
        if not self._text:
            return
        key = (round(head_global.x()), round(head_global.y()), screen_area.getRect(),
               self._body_size.width(), self._body_size.height())
        if key == self._follow_key:
            return
        self._last_anchor = (QPointF(head_global), QRect(screen_area))
        self._follow_key = key
        self._placement = place_bubble(self._body_size, QPointF(key[0], key[1]), screen_area)
        if self.geometry() != self._placement.window:
            self.setGeometry(self._placement.window)
        self.update()

    def hide_now(self) -> None:
        self._reveal_timer.stop()
        self._hide_timer.stop()
        self._fade.stop()
        self._fading = False
        self.hide()
        self.setWindowOpacity(1.0)

    def fade_out(self) -> None:
        """Fade away (no-op if already hidden or fading)."""
        if not self.isVisible() or self._fading:
            return
        self._hide_timer.stop()
        self._fading = True
        self._fade.stop()
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(0.0)
        self._fade.start()

    # -- layout ---------------------------------------------------------------------

    def _prepare_portrait(self, portrait: QPixmap | None) -> QPixmap | None:
        """Scale once (smoothly) to the portrait slot at the screen's pixel ratio."""
        if portrait is None or portrait.isNull():
            return None
        dpr = max(1.0, self.devicePixelRatioF())
        side = round(PORTRAIT_SIZE * dpr)
        scaled = portrait.scaled(side, side, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                 Qt.TransformationMode.SmoothTransformation)
        # Waist-up portraits: keep the top centre, where the face is.
        x = max(0, (scaled.width() - side) // 2)
        cropped = scaled.copy(x, 0, side, side)
        cropped.setDevicePixelRatio(dpr)
        return cropped

    def _build_layout(self) -> None:
        text = self._text.replace("\n", "\u2028")
        offsets = [0]
        for ch in text:
            offsets.append(offsets[-1] + (2 if ord(ch) > 0xFFFF else 1))
        self._utf16 = offsets
        layout = QTextLayout(text, self._font)
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout.setTextOption(option)
        metrics = QFontMetricsF(self._font)
        layout.beginLayout()
        y = 0.0
        natural = 0.0
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(WRAP_WIDTH)
            line.setPosition(QPointF(0.0, y))
            y += line.height()
            natural = max(natural, line.naturalTextWidth())
        layout.endLayout()
        self._layout = layout
        text_w = math.ceil(min(WRAP_WIDTH, max(natural, metrics.averageCharWidth())))
        text_h = math.ceil(max(y, metrics.height()))
        self._text_size = QSize(text_w, text_h)
        portrait_w = PORTRAIT_SIZE + PORTRAIT_GAP if self._portrait is not None else 0.0
        width = math.ceil(PADDING_X * 2 + portrait_w + text_w)
        content_h = max(text_h, PORTRAIT_SIZE if self._portrait is not None else 0)
        height = math.ceil(PADDING_Y * 2 + content_h)
        self._body_size = QSize(width, height)

    # -- reveal / fade --------------------------------------------------------------

    def _update_reveal(self) -> None:
        elapsed = max(0.0, self._now() - self._shown_at)
        self._revealed = min(len(self._text), int(elapsed * CHARS_PER_SECOND) + 1)

    def _on_reveal_tick(self) -> None:
        before = self._revealed
        self._update_reveal()
        if self._revealed >= len(self._text):
            self._reveal_timer.stop()
        if self._revealed != before:
            self.update()

    def _on_fade_finished(self) -> None:
        if self._fading:
            self._fading = False
            self._reveal_timer.stop()
            self.hide()
            self.setWindowOpacity(1.0)

    # -- events ---------------------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if sys.platform == "win32":
            hwnd = int(self.winId())
            if hwnd and hwnd != self._no_activate_hwnd and platform_win.apply_no_activate(hwnd):
                self._no_activate_hwnd = hwnd

    def paintEvent(self, event: QPaintEvent) -> None:
        placement = self._placement
        if placement is None or self._layout is None:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            self._paint_body(painter, placement)
            self._paint_content(painter, placement.body)
        finally:
            painter.end()

    def _shape(self, placement: Placement) -> QPainterPath:
        """Body + tail outline; cached per placement (the reveal repaints ~30x/s)."""
        cached = self._shape_cache
        if cached is not None and cached[0] is placement:
            return cached[1]
        shape = self._build_shape(placement)
        self._shape_cache = (placement, shape)
        return shape

    def _build_shape(self, placement: Placement) -> QPainterPath:
        body = placement.body
        path = QPainterPath()
        path.addRoundedRect(body, RADIUS, RADIUS)
        tip = placement.tip
        tail = QPainterPath()
        if placement.side in ("above", "below"):
            edge_y = body.bottom() if placement.side == "above" else body.top()
            inward = -1.0 if placement.side == "above" else 1.0
            lo, hi = body.left() + RADIUS + TAIL_HALF_WIDTH, body.right() - RADIUS - TAIL_HALF_WIDTH
            base_x = min(max(tip.x(), lo), hi) if lo <= hi else body.center().x()
            tail.moveTo(base_x - TAIL_HALF_WIDTH, edge_y + inward * 2.0)
            tail.quadTo(base_x - 1.0, edge_y - inward * TAIL * 0.4, tip.x(), tip.y())
            tail.quadTo(base_x + 2.0, edge_y - inward * TAIL * 0.3, base_x + TAIL_HALF_WIDTH,
                        edge_y + inward * 2.0)
        else:
            edge_x = body.left() if placement.side == "right" else body.right()
            inward = 1.0 if placement.side == "right" else -1.0
            lo, hi = body.top() + RADIUS + TAIL_HALF_WIDTH, body.bottom() - RADIUS - TAIL_HALF_WIDTH
            base_y = min(max(tip.y(), lo), hi) if lo <= hi else body.center().y()
            tail.moveTo(edge_x + inward * 2.0, base_y - TAIL_HALF_WIDTH)
            tail.quadTo(edge_x - inward * TAIL * 0.4, base_y - 1.0, tip.x(), tip.y())
            tail.quadTo(edge_x - inward * TAIL * 0.3, base_y + 2.0, edge_x + inward * 2.0,
                        base_y + TAIL_HALF_WIDTH)
        tail.closeSubpath()
        return path.united(tail)

    def _paint_body(self, painter: QPainter, placement: Placement) -> None:
        shape = self._shape(placement)
        painter.setPen(Qt.PenStyle.NoPen)
        # soft shadow: a few expanding translucent copies offset downwards
        for i, alpha in enumerate((26, 16, 9)):
            grow = 1.0 + i * 1.6
            painter.save()
            painter.translate(0.0, 1.5 + i)
            shadow = QColor(_SHADOW)
            shadow.setAlpha(alpha)
            painter.setPen(QPen(shadow, grow * 2.0))
            painter.setBrush(shadow)
            painter.drawPath(shape)
            painter.restore()
        painter.setPen(QPen(_BORDER, 1.2))
        painter.setBrush(QBrush(_FILL))
        painter.drawPath(shape)

    def _paint_content(self, painter: QPainter, body: QRectF) -> None:
        x = body.left() + PADDING_X
        if self._portrait is not None:
            slot = QRectF(x, body.top() + (body.height() - PORTRAIT_SIZE) / 2, PORTRAIT_SIZE, PORTRAIT_SIZE)
            clip = QPainterPath()
            clip.addRoundedRect(slot, PORTRAIT_SIZE * 0.32, PORTRAIT_SIZE * 0.32)
            painter.save()
            painter.setClipPath(clip)
            painter.drawPixmap(slot.topLeft(), self._portrait)
            painter.restore()
            painter.setPen(QPen(_BORDER, 1.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(clip)
            x += PORTRAIT_SIZE + PORTRAIT_GAP
        layout = self._layout
        if layout is None:
            return
        y = body.top() + (body.height() - self._text_size.height()) / 2
        painter.setPen(_TEXT)
        painter.setFont(self._font)
        text = layout.text()
        revealed = self._revealed
        for i in range(layout.lineCount()):
            line = layout.lineAt(i)
            # QTextLayout positions are UTF-16 units; map them to str indices.
            start = bisect.bisect_left(self._utf16, line.textStart())
            if start >= revealed:
                break
            end = min(bisect.bisect_left(self._utf16, line.textStart() + line.textLength()), revealed)
            chunk = text[start:end].replace("\u2028", "")
            pos = line.position()
            painter.drawText(QPointF(x + pos.x(), y + pos.y() + line.ascent()), chunk)
