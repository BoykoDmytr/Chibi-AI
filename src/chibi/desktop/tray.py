"""Tray icon with an "observing" indicator.

The tray is where the user always finds her (and the kill switch), even
when she is hidden or click-through.  Privacy is visible here: while any
observation permission is active the icon carries a small green "eye"
badge and the tooltip says so - the user should never have to wonder
whether the app is watching.

Icons are built from a portrait (rounded face crop) or, without a pack,
from the vector chibi's face.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from chibi.character.states import CharacterState
from chibi.desktop.renderers import VectorRenderer, VisualState

log = logging.getLogger(__name__)

OBSERVING_ON = "Спостереження активне"
OBSERVING_OFF = "Спостереження вимкнене"

#: Sizes rendered into the icons (Windows picks 16/20/24/32 depending on DPI).
ICON_SIZES: tuple[int, ...] = (16, 20, 24, 32, 40, 48, 64)
_BADGE_GREEN = QColor(46, 196, 110)


def _rounded(image: QImage, size: int) -> QPixmap:
    """*image* (square) scaled smoothly into a rounded-square *size* px icon."""
    out = QPixmap(size, size)
    out.fill(Qt.GlobalColor.transparent)
    scaled = image.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                          Qt.TransformationMode.SmoothTransformation)
    painter = QPainter(out)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0.0, 0.0, size, size), size * 0.24, size * 0.24)
        painter.setClipPath(clip)
        painter.fillRect(QRectF(0.0, 0.0, size, size), QColor(255, 244, 236))
        painter.drawImage(QPointF((size - scaled.width()) / 2, 0.0), scaled)
    finally:
        painter.end()
    return out


def make_icon_from_pixmap(portrait: QPixmap) -> QIcon:
    """A tray icon from a waist-up portrait: the face (top centre) in a rounded square."""
    if portrait is None or portrait.isNull():
        return vector_icon()
    image = portrait.toImage()
    side = min(image.width(), image.height())
    # The face is in the upper part of a waist-up portrait: crop a square from the top centre.
    crop = max(1, round(side * 0.82))
    x = max(0, (image.width() - crop) // 2)
    face = image.copy(x, 0, crop, crop)
    icon = QIcon()
    for size in ICON_SIZES:
        icon.addPixmap(_rounded(face, size))
    return icon


def vector_icon() -> QIcon:
    """Fallback icon: the vector chibi's head, no pack needed."""
    renderer = VectorRenderer()
    base = 256
    image = QImage(base, base, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # Draw her large enough that the head (~45 % of her height) fills the icon.
        height = base * 2.0
        anchor = QPointF(base / 2, height * 0.69 + base * 0.47)
        renderer.paint(painter, anchor, height, VisualState(CharacterState.IDLE, t=0.0))
    finally:
        painter.end()
    icon = QIcon()
    for size in ICON_SIZES:
        icon.addPixmap(_rounded(image, size))
    return icon


def with_badge(icon: QIcon, sizes: tuple[int, ...] = ICON_SIZES) -> QIcon:
    """*icon* with a green "eye" dot in the bottom-right corner."""
    badged = QIcon()
    for size in sizes:
        pixmap = icon.pixmap(size, size)
        if pixmap.isNull():
            continue
        pixmap = QPixmap(pixmap)
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            w, h = pixmap.width() / pixmap.devicePixelRatio(), pixmap.height() / pixmap.devicePixelRatio()
            r = max(2.6, min(w, h) * 0.2)
            c = QPointF(w - r - 0.5, h - r - 0.5)
            painter.setPen(QPen(QColor(255, 255, 255), max(1.0, r * 0.3)))
            painter.setBrush(_BADGE_GREEN)
            painter.drawEllipse(c, r, r)
            if r >= 4.5:  # big enough for a little eye glyph
                eye = QPainterPath(QPointF(c.x() - r * 0.62, c.y()))
                eye.quadTo(c.x(), c.y() - r * 0.62, c.x() + r * 0.62, c.y())
                eye.quadTo(c.x(), c.y() + r * 0.62, c.x() - r * 0.62, c.y())
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(255, 255, 255))
                painter.drawPath(eye)
                painter.setBrush(QColor(20, 90, 50))
                painter.drawEllipse(c, r * 0.24, r * 0.24)
        finally:
            painter.end()
        badged.addPixmap(pixmap)
    return badged


class TrayIcon(QObject):
    """Wraps :class:`QSystemTrayIcon`.  The app owns the menu."""

    #: Left click / double click on the icon (Windows "Trigger").
    activated = Signal()

    def __init__(self, menu: QMenu | None, icon: QIcon | None = None, name: str = "Мія",
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._menu = menu
        self._base_icon = icon if icon is not None and not icon.isNull() else vector_icon()
        self._name = name
        self._observing = False
        self._tray = QSystemTrayIcon(self._base_icon, self)
        if menu is not None:
            self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_activated)
        self._refresh()

    @staticmethod
    def available() -> bool:
        return QSystemTrayIcon.isSystemTrayAvailable()

    @property
    def tray(self) -> QSystemTrayIcon:
        return self._tray

    @property
    def observing(self) -> bool:
        return self._observing

    def icon(self) -> QIcon:
        return self._tray.icon()

    def tooltip(self) -> str:
        return self._tray.toolTip()

    def show(self) -> None:
        if not self.available():
            log.info("system tray not available; the tray icon is not shown")
        self._tray.show()

    def hide(self) -> None:
        self._tray.hide()

    def set_icon(self, icon: QIcon) -> None:
        if icon is None or icon.isNull():
            return
        self._base_icon = icon
        self._refresh()

    def set_name(self, name: str) -> None:
        self._name = name.strip() or self._name
        self._refresh()

    def set_observing(self, observing: bool) -> None:
        observing = bool(observing)
        if observing == self._observing:
            return
        self._observing = observing
        self._refresh()

    def show_message(self, title: str, text: str, seconds: float = 5.0) -> None:
        self._tray.showMessage(title, text, QSystemTrayIcon.MessageIcon.Information,
                               max(1000, round(seconds * 1000)))

    def _refresh(self) -> None:
        self._tray.setIcon(with_badge(self._base_icon) if self._observing else self._base_icon)
        status = OBSERVING_ON if self._observing else OBSERVING_OFF
        self._tray.setToolTip(f"{self._name}\n{status}")

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.activated.emit()
