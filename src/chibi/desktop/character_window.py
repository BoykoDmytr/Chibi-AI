"""The transparent, frameless window she lives in.

It is deliberately dumb: it knows nothing about states or animation.  It
turns mouse input into signals and delegates painting to a callback set by
the controller, which also decides its size and position.

Window rules:

* frameless, translucent, a tool window (no taskbar button) that never takes
  focus - clicking her must not pull the keyboard out of the user's game or
  editor.  On Windows ``WS_EX_NOACTIVATE`` is added to the native window
  (``platform_win.apply_no_activate``) because Qt's hint alone is not
  honoured everywhere;
* left press + move beyond the start-drag distance is a drag, a left click
  without moving is a click, right click asks for the context menu and
  Ctrl+wheel asks for a different scale - one step per wheel notch, asked
  for once the gesture pauses (each request is a settings save);
* lock mode disables dragging only; click-through makes the window
  transparent for input altogether (re-enabled from the tray menu).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QHideEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QShowEvent,
    QWheelEvent,
    QWindow,
)
from PySide6.QtWidgets import QApplication, QWidget

from chibi.desktop import platform_win

log = logging.getLogger(__name__)

MIN_SCALE = 0.5
MAX_SCALE = 2.5
#: Ctrl+wheel changes the scale by this factor per notch.
SCALE_STEP = 1.1
#: ``angleDelta`` of one wheel notch (1/8 degree units).  Touchpads, pinch
#: gestures (Ctrl+wheel on Windows) and high-resolution wheels send many
#: smaller deltas: they are summed, so a whole notch is still one step.
WHEEL_NOTCH = 120
#: The scale is requested this long after the last Ctrl+wheel event: one
#: gesture is one settings save and one sprite-cache rebuild, not ten.
SCALE_REQUEST_DELAY_MS = 250
#: Floor for the drag threshold.  Qt takes QApplication.startDragDistance() from
#: the system: 4 px on Windows (SM_CXDRAG), often 10 elsewhere.
MIN_DRAG_DISTANCE = 4

PaintCallback = Callable[[QPainter], None]


def _base_flags(always_on_top: bool) -> Qt.WindowType:
    flags = (Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
             | Qt.WindowType.WindowDoesNotAcceptFocus | Qt.WindowType.NoDropShadowWindowHint)
    if always_on_top:
        flags |= Qt.WindowType.WindowStaysOnTopHint
    return flags


class CharacterWindow(QWidget):
    clicked = Signal()
    drag_started = Signal()
    #: Global position (logical px) where her feet anchor should go: cursor minus the grab offset.
    dragged = Signal(QPointF)
    drag_finished = Signal()
    context_menu_requested = Signal(QPoint)
    scale_requested = Signal(float)
    #: The window moved to another screen or its device pixel ratio changed.
    display_changed = Signal()

    def __init__(self, always_on_top: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent, _base_flags(always_on_top))
        self._always_on_top = always_on_top
        self._click_through = False
        self._locked = False
        self._scale = 1.0
        self._anchor = QPointF(0.0, 0.0)
        self._paint_callback: PaintCallback | None = None
        self._press_global: QPointF | None = None
        self._grab_offset = QPointF(0.0, 0.0)
        self._dragging = False
        self._no_activate_hwnd = 0
        self._hooked_handle: QWindow | None = None
        self._wheel_acc = 0
        #: Scale chosen by a Ctrl+wheel gesture still in progress (None: none).
        self._pending_scale: float | None = None
        self._scale_timer = QTimer(self)
        self._scale_timer.setSingleShot(True)
        self._scale_timer.setInterval(SCALE_REQUEST_DELAY_MS)
        self._scale_timer.timeout.connect(self.commit_pending_scale)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setWindowTitle("Chibi")

    # -- configuration --------------------------------------------------------------

    def set_paint_callback(self, callback: PaintCallback | None) -> None:
        self._paint_callback = callback
        self.update()

    def set_anchor(self, anchor: QPointF) -> None:
        """Her feet anchor in window coordinates (used for the drag grab offset)."""
        self._anchor = QPointF(anchor)

    def anchor(self) -> QPointF:
        return QPointF(self._anchor)

    def set_scale(self, scale: float) -> None:
        """Current display scale, the base for Ctrl+wheel steps.

        A Ctrl+wheel gesture in progress keeps its own scale: settings applied
        meanwhile (any section) re-send the old scale and must not undo it.
        """
        self._scale = min(max(float(scale), MIN_SCALE), MAX_SCALE)

    @property
    def scale_request_pending(self) -> bool:
        return self._pending_scale is not None

    def commit_pending_scale(self) -> None:
        """End the Ctrl+wheel gesture: request its scale (if it differs from the
        current one).  The debounce timer calls this; tests may too."""
        self._scale_timer.stop()
        self._wheel_acc = 0
        scale, self._pending_scale = self._pending_scale, None
        if scale is not None and scale != self._scale:
            self._scale = scale
            self.scale_requested.emit(scale)

    @property
    def locked(self) -> bool:
        return self._locked

    def set_locked(self, locked: bool) -> None:
        """Lock mode: no dragging (clicks and the menu still work)."""
        self._locked = bool(locked)
        if self._locked and self._dragging:
            self._finish_drag()

    @property
    def click_through(self) -> bool:
        return self._click_through

    def set_click_through(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._click_through:
            return
        self._click_through = enabled
        if enabled and self._dragging:
            self._finish_drag()
        self._reapply_flags()

    @property
    def always_on_top(self) -> bool:
        return self._always_on_top

    def set_always_on_top(self, on_top: bool) -> None:
        on_top = bool(on_top)
        if on_top == self._always_on_top:
            return
        self._always_on_top = on_top
        self._reapply_flags()

    def is_dragging(self) -> bool:
        return self._dragging

    def _reapply_flags(self) -> None:
        """Changing window flags recreates the native window and hides it:
        restore geometry and visibility afterwards."""
        flags = _base_flags(self._always_on_top)
        if self._click_through:
            flags |= Qt.WindowType.WindowTransparentForInput
        geometry = self.geometry()
        visible = self.isVisible()
        self.setWindowFlags(flags)
        self.setGeometry(geometry)
        if visible:
            self.show()

    # -- events ---------------------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if sys.platform == "win32":
            hwnd = int(self.winId())
            if hwnd and hwnd != self._no_activate_hwnd and platform_win.apply_no_activate(hwnd):
                self._no_activate_hwnd = hwnd
        # Changing window flags can recreate the QWindow: hook whichever one exists now.
        handle = self.windowHandle()
        if handle is not None and handle is not self._hooked_handle:
            handle.screenChanged.connect(self._on_screen_changed)
            self._hooked_handle = handle

    def _on_screen_changed(self, _screen: object) -> None:
        self.display_changed.emit()

    def closeEvent(self, event: QCloseEvent) -> None:
        # Closed at shutdown: a gesture still waiting must not save settings
        # into an app that is going away.
        self._scale_timer.stop()
        self._pending_scale = None
        self._wheel_acc = 0
        super().closeEvent(event)

    def hideEvent(self, event: QHideEvent) -> None:
        # Hidden mid-drag (fullscreen app appeared, flags changed): the release
        # will never arrive, so end the drag now instead of leaving her stuck.
        if self._dragging:
            self._finish_drag()
        self._press_global = None
        super().hideEvent(event)

    def changeEvent(self, event: QEvent) -> None:
        dpr_change = getattr(QEvent.Type, "DevicePixelRatioChange", None)
        if dpr_change is not None and event.type() == dpr_change:
            self.display_changed.emit()
        super().changeEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        callback = self._paint_callback
        if callback is None:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            callback(painter)
        except Exception:
            log.exception("paint callback failed")
        finally:
            painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.globalPosition()
            self._press_global = QPointF(pos)
            top_left = QPointF(self.mapToGlobal(QPoint(0, 0)))
            self._grab_offset = pos - (top_left + self._anchor)
            self._dragging = False
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            event.accept()
            self.context_menu_requested.emit(event.globalPosition().toPoint())
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._press_global is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            super().mouseMoveEvent(event)
            return
        pos = event.globalPosition()
        if not self._dragging:
            if self._locked:
                return
            distance = (pos - self._press_global).manhattanLength()
            if distance < max(MIN_DRAG_DISTANCE, QApplication.startDragDistance()):
                return
            self._dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.drag_started.emit()
        self.dragged.emit(QPointF(pos - self._grab_offset))
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._press_global is None:
            super().mouseReleaseEvent(event)
            return
        event.accept()
        was_dragging = self._dragging
        self._press_global = None
        if was_dragging:
            self._finish_drag()
        else:
            self.clicked.emit()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier) or delta == 0:
            event.ignore()
            return
        event.accept()
        if (delta > 0) != (self._wheel_acc > 0):
            self._wheel_acc = 0  # turned the other way: a leftover part-notch does not count
        self._wheel_acc += delta
        steps = int(self._wheel_acc / WHEEL_NOTCH)  # toward zero: whole notches only
        self._wheel_acc -= steps * WHEEL_NOTCH
        self._scale_timer.start()  # (re)start: the gesture ends when the events stop
        if not steps:
            return
        base = self._pending_scale if self._pending_scale is not None else self._scale
        self._pending_scale = round(min(max(base * SCALE_STEP ** steps, MIN_SCALE), MAX_SCALE), 3)

    def _finish_drag(self) -> None:
        self._dragging = False
        self._press_global = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.drag_finished.emit()
