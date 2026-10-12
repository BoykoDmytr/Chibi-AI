from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QPainter, QWheelEvent
from PySide6.QtWidgets import QApplication

from chibi.desktop.character_window import MAX_SCALE, CharacterWindow

L = Qt.MouseButton.LeftButton
R = Qt.MouseButton.RightButton
NO = Qt.MouseButton.NoButton


@pytest.fixture
def window(qapp) -> Iterator[CharacterWindow]:
    w = CharacterWindow(always_on_top=True)
    w.setFixedSize(200, 260)
    w.set_anchor(QPointF(100, 240))
    w.move(300, 400)
    w.show()
    yield w
    w.close()
    w.deleteLater()


class Recorder:
    def __init__(self, w: CharacterWindow) -> None:
        self.events: list[tuple] = []
        w.clicked.connect(lambda: self.events.append(("clicked",)))
        w.drag_started.connect(lambda: self.events.append(("drag_started",)))
        w.dragged.connect(lambda p: self.events.append(("dragged", p.x(), p.y())))
        w.drag_finished.connect(lambda: self.events.append(("drag_finished",)))
        w.context_menu_requested.connect(lambda p: self.events.append(("menu", p.x(), p.y())))
        w.scale_requested.connect(lambda s: self.events.append(("scale", s)))

    def names(self) -> list[str]:
        return [e[0] for e in self.events]


def mouse(w: CharacterWindow, kind: QEvent.Type, local: tuple[float, float], button=L, buttons=L) -> None:
    local_p = QPointF(*local)
    global_p = QPointF(w.mapToGlobal(QPoint(0, 0))) + local_p
    event = QMouseEvent(kind, local_p, global_p, button, buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(w, event)


def press(w, x, y, button=L):
    mouse(w, QEvent.Type.MouseButtonPress, (x, y), button, button)


def move(w, x, y):
    mouse(w, QEvent.Type.MouseMove, (x, y), NO, L)


def release(w, x, y, button=L):
    mouse(w, QEvent.Type.MouseButtonRelease, (x, y), button, NO)


def wheel(w, delta: int, ctrl: bool) -> None:
    mods = Qt.KeyboardModifier.ControlModifier if ctrl else Qt.KeyboardModifier.NoModifier
    local = QPointF(100, 100)
    event = QWheelEvent(local, QPointF(w.mapToGlobal(QPoint(100, 100))), QPoint(0, 0), QPoint(0, delta),
                        NO, mods, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(w, event)


def test_window_flags_and_attributes(window):
    flags = window.windowFlags()
    for flag in (Qt.WindowType.FramelessWindowHint, Qt.WindowType.Tool,
                 Qt.WindowType.WindowDoesNotAcceptFocus, Qt.WindowType.NoDropShadowWindowHint,
                 Qt.WindowType.WindowStaysOnTopHint):
        assert flags & flag, flag
    for attr in (Qt.WidgetAttribute.WA_TranslucentBackground, Qt.WidgetAttribute.WA_ShowWithoutActivating,
                 Qt.WidgetAttribute.WA_NoSystemBackground):
        assert window.testAttribute(attr), attr
    assert window.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert window.cursor().shape() == Qt.CursorShape.PointingHandCursor


def test_not_on_top_when_disabled(qapp):
    w = CharacterWindow(always_on_top=False)
    assert not (w.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
    w.set_always_on_top(True)
    assert w.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    w.deleteLater()


def test_click_without_moving(window):
    rec = Recorder(window)
    press(window, 100, 150)
    move(window, 101, 151)  # tiny jitter is still a click
    release(window, 101, 151)
    assert rec.names() == ["clicked"]


def test_drag_sequence_reports_feet_position(window):
    rec = Recorder(window)
    press(window, 100, 140)  # grabbed 100 px above the feet anchor (100, 240)
    move(window, 160, 140)
    move(window, 220, 100)
    release(window, 220, 100)
    assert rec.names() == ["drag_started", "dragged", "dragged", "drag_finished"]
    origin = window.mapToGlobal(QPoint(0, 0))
    # feet = cursor - grab offset: the anchor follows the cursor rigidly
    assert rec.events[1][1:] == pytest.approx((origin.x() + 160, origin.y() + 240))
    assert rec.events[2][1:] == pytest.approx((origin.x() + 220, origin.y() + 200))
    assert not window.is_dragging()


def test_lock_prevents_drag_but_not_clicks(window):
    rec = Recorder(window)
    window.set_locked(True)
    press(window, 100, 140)
    move(window, 200, 40)
    release(window, 200, 40)
    assert rec.names() == ["clicked"]


def test_right_click_requests_menu(window):
    rec = Recorder(window)
    press(window, 50, 60, R)
    release(window, 50, 60, R)
    origin = window.mapToGlobal(QPoint(0, 0))
    assert rec.events == [("menu", origin.x() + 50, origin.y() + 60)]


def test_ctrl_wheel_scales(window):
    rec = Recorder(window)
    window.set_scale(1.0)
    wheel(window, 120, ctrl=False)
    assert rec.events == []
    wheel(window, 120, ctrl=True)
    wheel(window, -120, ctrl=True)
    assert [e[1] for e in rec.events] == pytest.approx([1.1, 1.0])
    window.set_scale(MAX_SCALE)
    wheel(window, 120, ctrl=True)
    assert len(rec.events) == 2  # already at the maximum: nothing to request


def test_click_through_toggles_input_transparency(window):
    geometry = window.geometry()
    window.set_click_through(True)
    assert window.windowFlags() & Qt.WindowType.WindowTransparentForInput
    assert window.isVisible()
    assert window.geometry() == geometry
    assert window.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    window.set_click_through(False)
    assert not (window.windowFlags() & Qt.WindowType.WindowTransparentForInput)
    assert window.isVisible() and window.geometry() == geometry


def test_hiding_mid_drag_finishes_it(window):
    rec = Recorder(window)
    press(window, 100, 140)
    move(window, 180, 140)
    window.hide()
    assert rec.names()[-1] == "drag_finished"
    assert not window.is_dragging()


def test_paint_delegates_and_survives_errors(window):
    calls = []

    def ok(painter: QPainter) -> None:
        calls.append(painter.device())

    window.set_paint_callback(ok)
    window.grab()
    assert calls

    def broken(painter: QPainter) -> None:
        raise RuntimeError("bug in a renderer")

    window.set_paint_callback(broken)
    window.grab()  # logged, not raised into Qt's event loop


def test_screen_change_is_reported_even_after_flag_changes(window, qapp):
    hits = []
    window.display_changed.connect(lambda: hits.append(1))
    screen = qapp.primaryScreen()
    window.windowHandle().screenChanged.emit(screen)
    assert hits == [1]
    window.set_click_through(True)  # may recreate the native window
    window.windowHandle().screenChanged.emit(screen)
    assert hits == [1, 1]
