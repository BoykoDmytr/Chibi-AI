"""Window for «Лови печиво».

:class:`CookieField` paints a :class:`~chibi.games.cookies.CookieCatch` with
QPainter (round cookies with chocolate chips, dark smoking burnt ones, a
basket, "+1" pop-ups) letterboxed into whatever size the window has, and
maps clicks back to field coordinates.

The frame timer (:data:`FRAME_MS`, just under 60 fps) runs only while a game
is in progress and the window is visible and not minimised.  Hiding or
minimising the window *pauses* the game - the user comes back to a paused
game, not to one that ran out while they were away - and the time spent
hidden is never fed to the physics.

Keyboard: ``Space`` starts / pauses / resumes, ``←`` / ``→`` move the basket
(it appears with the first arrow key), ``Enter`` starts a new game once one
is over, ``Esc`` closes the window.

:attr:`CookieWindow.finished` carries the :class:`~chibi.games.GameResult`
(once per game); :meth:`CookieWindow.show_reward` shows what it was worth.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Callable

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFocusEvent,
    QFont,
    QHideEvent,
    QKeyEvent,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QShowEvent,
)
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from chibi.games import GAME_TITLES
from chibi.games.cookies import (
    BASKET_HEIGHT,
    BASKET_WIDTH,
    COOKIE_RADIUS,
    FIELD_HEIGHT,
    FIELD_WIDTH,
    POPUP_SECONDS,
    Catch,
    Cookie,
    CookieCatch,
)
from chibi.settings import theme
from chibi.settings.widgets import make_button, make_label

log = logging.getLogger(__name__)

#: ~58.8 fps: never above 60.
FRAME_MS = 17

COOKIE_BODY = QColor("#e2ab6c")
COOKIE_EDGE = QColor("#b5793f")
COOKIE_CHIP = QColor("#5b3a1e")
BURNT_BODY = QColor("#4a3428")
BURNT_EDGE = QColor("#2a1c14")
BURNT_CHIP = QColor("#17100b")
SMOKE = QColor(210, 210, 220, 150)
#: Chocolate chips: (x, y) in cookie radii, radius in field units.
CHIPS: tuple[tuple[float, float, float], ...] = (
    (-0.42, -0.30, 3.0), (0.36, -0.42, 2.6), (0.05, 0.08, 3.2), (-0.30, 0.46, 2.4), (0.48, 0.30, 2.8),
)

TEXT_READY = "Пробіл або «Почати» — і поїхали!"
TEXT_PAUSED = "Пауза. Пробіл — продовжити."
_ENTER_KEYS = (int(Qt.Key.Key_Return), int(Qt.Key.Key_Enter))
_ARROWS = {int(Qt.Key.Key_Left): -1, int(Qt.Key.Key_Right): 1}


def format_clock(seconds: float) -> str:
    """``"0:07"`` - whole seconds left, rounded up (0:01 until it really is over)."""
    whole = max(0, math.ceil(seconds - 1e-9))
    return f"{whole // 60}:{whole % 60:02d}"


class CookieField(QWidget):
    """Draws the game; emits ``pressed(x, y)`` in field coordinates for clicks inside it."""

    pressed = Signal(float, float)

    def __init__(self, game: CookieCatch, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._game = game
        self._overlay = ""
        self.setMinimumSize(round(FIELD_WIDTH), round(FIELD_HEIGHT))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("Поле гри «Лови печиво»")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_game(self, game: CookieCatch) -> None:
        self._game = game
        self.update()

    def set_overlay(self, text: str) -> None:
        if text != self._overlay:
            self._overlay = text
            self.update()

    @property
    def overlay(self) -> str:
        return self._overlay

    def focusInEvent(self, event: QFocusEvent) -> None:
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.update()

    # -- geometry -------------------------------------------------------------------------

    def _layout(self) -> tuple[float, float, float]:
        """(scale, x offset, y offset) of the letterboxed field."""
        scale = max(1e-6, min(self.width() / FIELD_WIDTH, self.height() / FIELD_HEIGHT))
        return scale, (self.width() - FIELD_WIDTH * scale) / 2, (self.height() - FIELD_HEIGHT * scale) / 2

    def to_field(self, point: QPointF) -> tuple[float, float] | None:
        scale, ox, oy = self._layout()
        x, y = (point.x() - ox) / scale, (point.y() - oy) / scale
        if 0 <= x <= FIELD_WIDTH and 0 <= y <= FIELD_HEIGHT:
            return x, y
        return None

    def from_field(self, x: float, y: float) -> QPointF:
        scale, ox, oy = self._layout()
        return QPointF(ox + x * scale, oy + y * scale)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            point = self.to_field(event.position())
            if point is not None:
                self.pressed.emit(*point)
            event.accept()
            return
        super().mousePressEvent(event)

    # -- painting -------------------------------------------------------------------------

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(self.rect(), QColor(theme.BACKGROUND))
            scale, ox, oy = self._layout()
            painter.save()
            painter.translate(ox, oy)
            painter.scale(scale, scale)
            self._paint_field(painter)
            painter.restore()
            if self._overlay:
                self._paint_overlay(painter)
        finally:
            painter.end()

    def _paint_field(self, painter: QPainter) -> None:
        field = QRectF(0, 0, FIELD_WIDTH, FIELD_HEIGHT)
        gradient = QLinearGradient(0, 0, 0, FIELD_HEIGHT)
        gradient.setColorAt(0.0, QColor(theme.SURFACE_ALT))
        gradient.setColorAt(1.0, QColor(theme.SURFACE))
        path = QPainterPath()
        path.addRoundedRect(field, theme.RADIUS_LARGE, theme.RADIUS_LARGE)
        painter.fillPath(path, QBrush(gradient))
        painter.setClipPath(path)
        game = self._game
        for cookie in game.cookies:
            self._paint_cookie(painter, cookie)
        if game.basket_active:
            basket = QRectF(game.basket_x - BASKET_WIDTH / 2, FIELD_HEIGHT - BASKET_HEIGHT,
                            BASKET_WIDTH, BASKET_HEIGHT)
            painter.setPen(QPen(QColor(theme.LILAC_PRESSED), 2))
            painter.setBrush(QColor(theme.LILAC))
            painter.drawRoundedRect(basket, 5, 5)
        font = QFont()
        font.setBold(True)
        font.setPixelSize(16)
        painter.setFont(font)
        for popup in game.popups:
            fade = max(0.0, 1.0 - popup.age / POPUP_SECONDS)
            color = QColor(theme.SUCCESS if popup.good else theme.DANGER)
            color.setAlphaF(fade)
            painter.setPen(color)
            painter.drawText(QPointF(popup.x - 10, popup.y - 8 - popup.age * 40), popup.text)
        painter.setClipping(False)
        # A visible focus ring: the arrow keys work while the field has focus.
        painter.setPen(QPen(QColor(theme.LILAC if self.hasFocus() else theme.BORDER), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(field.adjusted(0.5, 0.5, -0.5, -0.5), theme.RADIUS_LARGE, theme.RADIUS_LARGE)

    @staticmethod
    def _paint_cookie(painter: QPainter, cookie: Cookie) -> None:
        r = COOKIE_RADIUS
        body, edge, chip = ((BURNT_BODY, BURNT_EDGE, BURNT_CHIP) if cookie.burnt
                            else (COOKIE_BODY, COOKIE_EDGE, COOKIE_CHIP))
        painter.save()
        painter.translate(cookie.x, cookie.y)
        painter.rotate(cookie.spin)
        painter.setPen(QPen(edge, 2))
        painter.setBrush(body)
        painter.drawEllipse(QPointF(0, 0), r, r)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(chip)
        for dx, dy, size in CHIPS:
            painter.drawEllipse(QPointF(dx * r, dy * r), size, size)
        painter.restore()
        if cookie.burnt:  # two wisps of smoke, upright whatever the spin
            painter.setPen(QPen(SMOKE, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for offset in (-5.0, 5.0):
                wisp = QPainterPath(QPointF(cookie.x + offset, cookie.y - r - 2))
                wisp.cubicTo(QPointF(cookie.x + offset - 5, cookie.y - r - 7),
                             QPointF(cookie.x + offset + 5, cookie.y - r - 11),
                             QPointF(cookie.x + offset, cookie.y - r - 16))
                painter.drawPath(wisp)

    def _paint_overlay(self, painter: QPainter) -> None:
        font = QFont()
        font.setBold(True)
        font.setPointSizeF(13)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = min(self.width() - 24, metrics.horizontalAdvance(self._overlay) + 36)
        box = QRectF((self.width() - width) / 2, (self.height() - 52) / 2, width, 52)
        background = QColor(theme.BACKGROUND)
        background.setAlpha(215)
        painter.setPen(QPen(QColor(theme.LILAC), 1))
        painter.setBrush(background)
        painter.drawRoundedRect(box, theme.RADIUS, theme.RADIUS)
        painter.setPen(QColor(theme.TEXT))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._overlay)


class CookieWindow(QWidget):
    #: GameResult, once per finished game.
    finished = Signal(object)

    def __init__(self, rng: random.Random, parent: QWidget | None = None, *, name: str = "Мія",
                 now: Callable[[], float] = time.monotonic) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self._rng = rng
        self._name = name or "Мія"
        self._now = now
        self._running = False
        self._last: float | None = None
        self._held: list[int] = []  # arrow keys held, most recent last
        self.game = CookieCatch(rng)
        self.setWindowTitle(GAME_TITLES["cookies"])
        self.setMinimumWidth(460)

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._frame)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        top = QHBoxLayout()
        top.addWidget(make_label(GAME_TITLES["cookies"], "title", wrap=False))
        top.addStretch(1)
        self.score_label = make_label("", "value", wrap=False)
        self.time_label = make_label("", "pill", wrap=False)
        top.addWidget(self.score_label)
        top.addSpacing(12)
        top.addWidget(self.time_label)
        layout.addLayout(top)
        layout.addWidget(make_label(
            "Клацай по печиву, поки воно падає, а підгоріле не чіпай! "
            "Можна й кошиком: стрілки ← →. Пробіл — пауза.", "hint"))

        self.field = CookieField(self.game)
        self.field.pressed.connect(self._on_pressed)
        layout.addWidget(self.field, 1)

        bottom = QHBoxLayout()
        self.reward_label = make_label("", "success")
        bottom.addWidget(self.reward_label, 1)
        bottom.addStretch(0)  # keeps the buttons compact while the reward line is hidden
        self.start_button = make_button("Почати", "primary")
        self.start_button.clicked.connect(self.toggle)
        bottom.addWidget(self.start_button)
        self.again_button = make_button("Ще раз", "primary")
        self.again_button.setToolTip("Enter — нова гра")
        self.again_button.clicked.connect(self.new_game)
        bottom.addWidget(self.again_button)
        layout.addLayout(bottom)
        self._reset_view()

    # -- public ---------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        """A game is in progress and not paused (the timer may still be stopped while hidden)."""
        return self._running

    @property
    def timer_active(self) -> bool:
        return self._timer.isActive()

    def set_name(self, name: str) -> None:
        self._name = name or "Мія"

    def start(self) -> None:
        if self.game.finished or self._running:
            return
        self._running = True
        self._sync()
        # A focused button would eat the arrow keys (they move focus between
        # buttons); the field passes them on to this window.
        self.field.setFocus()

    def pause(self) -> None:
        if not self._running:
            return
        self._running = False
        self._held.clear()
        self.game.set_basket_direction(0)
        self._sync()

    def toggle(self) -> None:
        if self._running:
            self.pause()
        else:
            self.start()

    def new_game(self) -> None:
        self._running = False
        self._held.clear()
        self.game = CookieCatch(self._rng)
        self.field.set_game(self.game)
        self._reset_view()
        self.start_button.setFocus()

    def show_reward(self, text: str) -> None:
        self.reward_label.setText(text)
        self.reward_label.setVisible(bool(text))

    def advance(self, dt: float) -> None:
        """Advance the game by ``dt`` seconds and redraw (the frame timer; tests)."""
        if not self._running:
            return
        self.game.step(dt)
        self._update_labels()
        self.field.update()
        if self.game.finished:
            self._finish()

    # -- timer ----------------------------------------------------------------------------

    def _sync(self) -> None:
        """Run the frame timer exactly while a game runs in a visible, non-minimised window."""
        want = self._running and self.isVisible() and not self.isMinimized() and not self.game.finished
        if want and not self._timer.isActive():
            self._last = self._now()
            self._timer.start()
        elif not want and self._timer.isActive():
            self._timer.stop()
            self._last = None
        self._update_labels()

    def _frame(self) -> None:
        now = self._now()
        dt = 0.0 if self._last is None else now - self._last
        self._last = now
        self.advance(dt)

    # -- input ----------------------------------------------------------------------------

    def _on_pressed(self, x: float, y: float) -> None:
        if self.game.finished:
            return
        if not self._running:
            self.start()  # a click on the field is the most natural "go"
            return
        if self.game.click(x, y) is not Catch.MISS:
            self._update_labels()
            self.field.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Space:
            if not event.isAutoRepeat() and not self.game.finished:
                self.toggle()
            event.accept()
            return
        if key in _ARROWS:
            if self._running and not event.isAutoRepeat():
                if key in self._held:
                    self._held.remove(key)
                self._held.append(key)
                self.game.set_basket_direction(_ARROWS[key])
                self.field.update()
            event.accept()
            return
        if key in _ENTER_KEYS and self.game.finished:
            self.new_game()
            event.accept()
            return
        if key == Qt.Key.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in _ARROWS and not event.isAutoRepeat():
            if key in self._held:
                self._held.remove(key)
            self.game.set_basket_direction(_ARROWS[self._held[-1]] if self._held else 0)
            event.accept()
            return
        super().keyReleaseEvent(event)

    # -- visibility -----------------------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._sync()

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        self.pause()  # coming back finds the game paused, not over
        self._sync()

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange and self.isMinimized():
            self.pause()

    # -- view -----------------------------------------------------------------------------

    def _reset_view(self) -> None:
        self.show_reward("")
        self._sync()

    def _update_labels(self) -> None:
        game = self.game
        self.score_label.setText(f"Рахунок: {game.score}")
        self.time_label.setText(f"⏱ {format_clock(game.time_left)}")
        if game.finished:
            self.field.set_overlay(f"Час вийшов! Рахунок: {game.score}.")
        elif self._running:
            self.field.set_overlay("")
        elif game.elapsed > 0:
            self.field.set_overlay(TEXT_PAUSED)
        else:
            self.field.set_overlay(TEXT_READY)
        self.start_button.setVisible(not game.finished)
        idle_text = "Продовжити" if game.elapsed > 0 else "Почати"
        self.start_button.setText("Пауза" if self._running else idle_text)
        self.again_button.setVisible(game.finished)

    def _finish(self) -> None:
        self._running = False
        self._held.clear()
        self._sync()
        self.again_button.setFocus()
        self.finished.emit(self.game.result())
