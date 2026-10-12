from __future__ import annotations

import pytest
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QMenu

from chibi.desktop.tray import (
    ICON_SIZES,
    OBSERVING_OFF,
    OBSERVING_ON,
    TrayIcon,
    make_icon_from_pixmap,
    vector_icon,
    with_badge,
)
from tests.desktop.helpers import MIA_DIR, alpha_bbox


def _is_green(color: QColor) -> bool:
    return color.green() > 150 and color.red() < 120 and color.blue() < 160 and color.alpha() > 200


def _has_green_corner(icon, size: int = 32) -> bool:
    image = icon.pixmap(size, size).toImage()
    w, h = image.width(), image.height()
    return any(_is_green(image.pixelColor(x, y))
               for x in range(w // 2, w) for y in range(h // 2, h))


@pytest.fixture
def menu(qapp):
    m = QMenu()
    m.addAction("Вийти")
    yield m
    m.deleteLater()


def test_vector_icon_has_a_face(qapp):
    icon = vector_icon()
    assert not icon.isNull()
    for size in (16, 32, 64):
        assert alpha_bbox(icon.pixmap(size, size).toImage()) is not None
    assert len(icon.availableSizes()) == len(ICON_SIZES)


def test_icon_from_portrait(qapp):
    portrait = QPixmap(str(MIA_DIR / "portraits" / "smile.png"))
    icon = make_icon_from_pixmap(portrait)
    image = icon.pixmap(64, 64).toImage()
    assert image.width() == 64
    assert image.pixelColor(0, 0).alpha() == 0  # rounded corners
    assert image.pixelColor(32, 32).alpha() == 255


def test_icon_from_null_pixmap_falls_back(qapp):
    assert not make_icon_from_pixmap(QPixmap()).isNull()


def test_badge_draws_a_green_dot(qapp):
    base = vector_icon()
    assert not _has_green_corner(base)
    assert _has_green_corner(with_badge(base))


def test_observing_indicator(qapp, menu):
    tray = TrayIcon(menu, name="Мія")
    assert isinstance(TrayIcon.available(), bool)
    assert OBSERVING_OFF in tray.tooltip() and "Мія" in tray.tooltip()
    assert not _has_green_corner(tray.icon())
    tray.set_observing(True)
    assert tray.observing
    assert OBSERVING_ON in tray.tooltip()
    assert _has_green_corner(tray.icon())
    tray.set_observing(False)
    assert OBSERVING_OFF in tray.tooltip()
    assert not _has_green_corner(tray.icon())
    assert tray.tray.contextMenu() is menu


def test_set_icon_keeps_badge(qapp, menu):
    tray = TrayIcon(menu)
    tray.set_observing(True)
    tray.set_icon(make_icon_from_pixmap(QPixmap(str(MIA_DIR / "portraits" / "calm.png"))))
    assert _has_green_corner(tray.icon())
    tray.set_name("  Ліна ")
    assert tray.tooltip().startswith("Ліна")


def test_activation_signal(qapp, menu):
    from PySide6.QtWidgets import QSystemTrayIcon

    tray = TrayIcon(menu)
    hits = []
    tray.activated.connect(lambda: hits.append(1))
    tray.tray.activated.emit(QSystemTrayIcon.ActivationReason.Trigger)
    tray.tray.activated.emit(QSystemTrayIcon.ActivationReason.Context)
    assert hits == [1]
