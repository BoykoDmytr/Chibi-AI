from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QtMsgType, qInstallMessageHandler
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from chibi.settings import theme


@pytest.fixture
def themed_app(qapp):
    """Apply the theme, and restore the application's look afterwards so other
    test modules (desktop rendering) never depend on test order."""
    old_sheet = qapp.styleSheet()
    old_palette = QPalette(qapp.palette())
    old_style = qapp.style().name() or "fusion"
    old_flag = qapp.property("chibi_dark_theme")
    yield qapp
    # setStyle(name) installs a plain style, never the ChibiStyle proxy: keeping
    # the "already themed" flag after it would stop every later apply_dark_theme
    # from installing the proxy again (test_app themes the app before this module
    # when run first).
    qapp.setProperty("chibi_dark_theme", None)
    if old_flag:
        theme.apply_dark_theme(qapp)
    else:
        qapp.setStyle(old_style)
    qapp.setStyleSheet(old_sheet)
    qapp.setPalette(old_palette)


def _showcase() -> QWidget:
    window = QWidget()
    layout = QVBoxLayout(window)
    for variant in ("", "primary", "accent", "danger"):
        button = QPushButton(variant or "default")
        if variant:
            button.setProperty("variant", variant)
        layout.addWidget(button)
    big = QPushButton("big")
    big.setProperty("variant", "danger")
    big.setProperty("big", True)
    layout.addWidget(big)
    check = QCheckBox("check")
    check.setChecked(True)
    layout.addWidget(check)
    slider = QSlider()
    layout.addWidget(slider)
    for tone in theme.TONES:
        bar = QProgressBar()
        bar.setProperty("tone", tone)
        bar.setValue(50)
        layout.addWidget(bar)
    nav = QListWidget()
    nav.setObjectName("navList")
    nav.addItems(["a", "b"])
    layout.addWidget(nav)
    for role in ("title", "subtitle", "section", "hint", "error", "success", "warning", "value", "banner",
                 "pill"):
        label = QLabel(role)
        label.setProperty("role", role)
        layout.addWidget(label)
    for kind in ("her", "user", "system"):
        label = QLabel(kind)
        label.setProperty("bubble", kind)
        layout.addWidget(label)
    layout.addWidget(QLineEdit("text"))
    spin = QSpinBox()
    spin.setSuffix(" хв")
    layout.addWidget(spin)
    combo = QComboBox()
    combo.addItems(["Спрайти", "Векторний"])
    layout.addWidget(combo)
    tabs = QTabWidget()
    tabs.addTab(QWidget(), "one")
    layout.addWidget(tabs)
    group = QGroupBox("group")
    layout.addWidget(group)
    return window


def test_theme_applies_without_warnings(themed_app, track):
    messages: list[str] = []

    def handler(mode, _context, message):
        if mode != QtMsgType.QtDebugMsg:
            messages.append(message)

    previous = qInstallMessageHandler(handler)
    try:
        theme.apply_dark_theme(themed_app)
        window = track(_showcase())
        window.show()
        window.grab()
        menu = track(QMenu())
        action = menu.addAction("Не турбувати")
        action.setCheckable(True)
        action.setChecked(True)
        menu.addSeparator()
        menu.addAction("Вийти")
        menu.grab()
    finally:
        qInstallMessageHandler(previous)
    # the offscreen platform's own notice is not ours
    ours = [m for m in messages if "propagateSizeHints" not in m]
    assert ours == []


def test_palette_and_colors(themed_app):
    theme.apply_dark_theme(themed_app)
    pal = themed_app.palette()
    assert pal.color(QPalette.ColorRole.Window) == QColor(theme.BACKGROUND)
    assert pal.color(QPalette.ColorRole.WindowText) == QColor(theme.TEXT)
    assert pal.color(QPalette.ColorRole.Base) == QColor(theme.SURFACE)
    assert pal.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text) == QColor(theme.TEXT_DISABLED)
    assert (theme.BACKGROUND, theme.SURFACE, theme.SURFACE_ALT, theme.BORDER) == (
        "#1b1c23", "#24252f", "#2c2e3a", "#3a3c4a")
    assert (theme.TEXT, theme.TEXT_SECONDARY, theme.PINK, theme.LILAC, theme.DANGER, theme.SUCCESS) == (
        "#ecebf3", "#a6a4b8", "#ff9ec7", "#b99cff", "#ff6b81", "#7ee0b5")
    sheet = themed_app.styleSheet()
    for color in (theme.PINK, theme.LILAC, theme.DANGER, theme.SUCCESS, theme.BACKGROUND):
        assert color in sheet
    assert "QMenu" in sheet and "QToolTip" in sheet


def test_stylesheet_never_styles_every_widget():
    """The translucent character window shares the app: no bare QWidget/QLabel rule."""
    sheet = theme.stylesheet()
    for line in sheet.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("QWidget {")
        assert not stripped.startswith("QLabel {")
        assert not stripped.startswith("* {")


def test_reapplying_is_safe_and_keeps_one_style(themed_app):
    theme.apply_dark_theme(themed_app)
    first = theme._installed_style
    theme.apply_dark_theme(themed_app)
    assert theme._installed_style is first
    assert themed_app.property("chibi_dark_theme") is True


def test_arrow_images_exist_and_feed_the_stylesheet():
    images = theme.ensure_arrow_images()
    assert images is not None
    for name, path in images.items():
        assert Path(path).is_file()
        assert Path(path).with_name(f"{name}@2x.png").is_file()
    sheet = theme.stylesheet(images)
    assert "QSpinBox::up-arrow" in sheet and images["up"] in sheet
    assert "QSpinBox::up-arrow" not in theme.stylesheet()  # fallback leaves buttons to the base style


def test_checked_checkbox_is_drawn_in_accent_pink(themed_app, track):
    theme.apply_dark_theme(themed_app)
    box = track(QCheckBox("x"))
    box.setChecked(True)
    box.resize(60, 30)
    box.show()
    image = box.grab().toImage()
    pink = QColor(theme.PINK)

    def close(c: QColor) -> bool:
        return abs(c.red() - pink.red()) < 12 and abs(c.green() - pink.green()) < 12 and \
            abs(c.blue() - pink.blue()) < 12

    hits = sum(close(image.pixelColor(x, y)) for x in range(min(image.width(), 24))
               for y in range(image.height()))
    assert hits > 40
