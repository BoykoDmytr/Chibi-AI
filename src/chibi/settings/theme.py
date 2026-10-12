"""Dark theme for every window of the app (settings, status, chat, menus).

One palette + one application stylesheet keep the look consistent and let the
windows stay free of colour code: widgets opt into variants through dynamic
properties instead of per-widget stylesheets, e.g.

* ``QPushButton[variant="primary" | "accent" | "danger"]``, ``[big="true"]``;
* ``QProgressBar[tone="pink" | "lilac" | "success" | "warning" | "danger"]``;
* ``QLabel[role="title" | "subtitle" | "section" | "hint" | "error" | "success"
  | "value" | "banner" | "pill" | "warning"]`` and ``QLabel[bubble="her" | "user" | "system"]``;
* ``QFrame[card="true"]`` for rounded surface panels;
* ``QListWidget#navList`` for the settings navigation.

The stylesheet deliberately has **no** bare ``QWidget``/``QLabel`` rule: the
character window and speech bubble (``desktop``) are translucent custom-painted
widgets in the same application and must not get a background from here.
``QMenu`` and ``QToolTip`` are styled globally on purpose - the character's
context menu and the tray menu share this look.

Check boxes are drawn by a small :class:`QProxyStyle` (rounded box, pink fill,
real check mark) because a stylesheet alone cannot draw a check mark without
image files.  Spin-box and combo-box arrows *can* only come from image files
once their buttons are styled, so :func:`apply_dark_theme` renders two tiny
chevron PNGs into a private temporary directory (removed at exit).  If that is
impossible the buttons are left unstyled and Fusion draws them, with the
proxy's chevrons.
"""

from __future__ import annotations

import atexit
import logging
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from string import Template

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QLineEdit,
    QProxyStyle,
    QStyle,
    QStyleOption,
    QWidget,
)

log = logging.getLogger(__name__)

# --- colours -----------------------------------------------------------------------

BACKGROUND = "#1b1c23"
SURFACE = "#24252f"
SURFACE_ALT = "#2c2e3a"
SURFACE_HOVER = "#353747"
BORDER = "#3a3c4a"
BORDER_HOVER = "#4b4d5f"
TEXT = "#ecebf3"
TEXT_SECONDARY = "#a6a4b8"
TEXT_DISABLED = "#6b697e"
#: Text drawn on top of the light accent colours.
ON_ACCENT = "#1b1c23"

PINK = "#ff9ec7"
PINK_HOVER = "#ffb6d5"
PINK_PRESSED = "#e886b1"
LILAC = "#b99cff"
LILAC_HOVER = "#cab3ff"
LILAC_PRESSED = "#a184ea"
DANGER = "#ff6b81"
DANGER_HOVER = "#ff8798"
DANGER_PRESSED = "#e5566d"
SUCCESS = "#7ee0b5"
WARNING = "#ffd27a"

#: ``QProgressBar[tone=...]`` values and their chunk colours.
TONES: dict[str, str] = {
    "pink": PINK,
    "lilac": LILAC,
    "success": SUCCESS,
    "warning": WARNING,
    "danger": DANGER,
}

RADIUS_SMALL = 8
RADIUS = 10
RADIUS_LARGE = 12

_COLORS: dict[str, str] = {
    "bg": BACKGROUND,
    "surface": SURFACE,
    "surface_alt": SURFACE_ALT,
    "surface_hover": SURFACE_HOVER,
    "border": BORDER,
    "border_hover": BORDER_HOVER,
    "text": TEXT,
    "text_secondary": TEXT_SECONDARY,
    "text_disabled": TEXT_DISABLED,
    "on_accent": ON_ACCENT,
    "pink": PINK,
    "pink_hover": PINK_HOVER,
    "pink_pressed": PINK_PRESSED,
    "lilac": LILAC,
    "lilac_hover": LILAC_HOVER,
    "lilac_pressed": LILAC_PRESSED,
    "danger": DANGER,
    "danger_hover": DANGER_HOVER,
    "danger_pressed": DANGER_PRESSED,
    "success": SUCCESS,
    "warning": WARNING,
}

_QSS = Template(
    """
/* Tooltips and menus are top-level popup windows: a border-radius there leaves the
   corners unpainted (black on Windows) unless the window is translucent from
   creation, which other modules' menus are not.  So the popups stay square and
   only the items inside are rounded. */
QToolTip {
    background-color: $surface_alt; color: $text;
    border: 1px solid $border; padding: 6px 8px;
}

/* --- menus (context menu, tray menu) --- */
QMenu {
    background-color: $surface; color: $text;
    border: 1px solid $border; padding: 6px;
}
QMenu::item { padding: 7px 24px 7px 14px; border-radius: 6px; background-color: transparent; }
QMenu::item:selected { background-color: $surface_hover; color: $pink; }
QMenu::item:disabled { color: $text_disabled; }
QMenu::separator { height: 1px; background-color: $border; margin: 5px 8px; }

/* --- buttons --- */
QPushButton {
    background-color: $surface_alt; color: $text;
    border: 1px solid $border; border-radius: 8px;
    padding: 7px 16px; min-height: 20px;
}
QPushButton:hover { background-color: $surface_hover; border-color: $border_hover; }
QPushButton:pressed { background-color: $border; }
QPushButton:focus { border-color: $lilac; }
QPushButton:disabled { background-color: $surface; color: $text_disabled; border-color: $surface_alt; }

QPushButton[variant="primary"] {
    background-color: $pink; color: $on_accent; border-color: $pink; font-weight: 600;
}
QPushButton[variant="primary"]:hover { background-color: $pink_hover; border-color: $pink_hover; }
QPushButton[variant="primary"]:pressed { background-color: $pink_pressed; border-color: $pink_pressed; }
QPushButton[variant="primary"]:focus { border-color: $text; }
QPushButton[variant="primary"]:disabled {
    background-color: $surface_alt; color: $text_disabled; border-color: $surface_alt;
}

QPushButton[variant="accent"] {
    background-color: $lilac; color: $on_accent; border-color: $lilac; font-weight: 600;
}
QPushButton[variant="accent"]:hover { background-color: $lilac_hover; border-color: $lilac_hover; }
QPushButton[variant="accent"]:pressed { background-color: $lilac_pressed; border-color: $lilac_pressed; }
QPushButton[variant="accent"]:focus { border-color: $text; }
QPushButton[variant="accent"]:disabled {
    background-color: $surface_alt; color: $text_disabled; border-color: $surface_alt;
}

QPushButton[variant="danger"] { background-color: transparent; color: $danger; border-color: $danger; }
QPushButton[variant="danger"]:hover { background-color: $danger; color: $on_accent; border-color: $danger; }
QPushButton[variant="danger"]:pressed { background-color: $danger_pressed; color: $on_accent; }
QPushButton[variant="danger"]:disabled {
    background-color: transparent; color: $text_disabled; border-color: $surface_alt;
}

QPushButton[big="true"] { padding: 12px 22px; border-radius: 12px; font-size: 11pt; font-weight: 700; }
QPushButton[variant="danger"][big="true"] { background-color: $danger; color: $on_accent; }
QPushButton[variant="danger"][big="true"]:hover {
    background-color: $danger_hover; border-color: $danger_hover;
}
QPushButton[variant="danger"][big="true"]:pressed { background-color: $danger_pressed; }

/* --- check boxes (the indicator is drawn by the proxy style) --- */
QCheckBox { spacing: 10px; color: $text; }
QCheckBox:disabled { color: $text_disabled; }

/* --- text inputs --- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {
    background-color: $surface; color: $text;
    border: 1px solid $border; border-radius: 8px; padding: 6px 10px;
    selection-background-color: $lilac; selection-color: $on_accent;
}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover,
QPlainTextEdit:hover, QTextEdit:hover { border-color: $border_hover; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus { border-color: $pink; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    background-color: $bg; color: $text_disabled; border-color: $surface_alt;
}
QComboBox QAbstractItemView {
    background-color: $surface; color: $text; border: 1px solid $border;
    selection-background-color: $surface_hover; selection-color: $pink; outline: 0; padding: 4px;
}

/* --- sliders: accent groove --- */
QSlider { min-height: 24px; }
QSlider::groove:horizontal { height: 6px; background-color: $border; border-radius: 3px; }
QSlider::sub-page:horizontal {
    background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 $lilac, stop: 1 $pink);
    border-radius: 3px;
}
QSlider::add-page:horizontal { background-color: $border; border-radius: 3px; }
QSlider::handle:horizontal {
    background-color: $text; border: 3px solid $pink;
    width: 12px; height: 12px; margin: -6px 0; border-radius: 9px;
}
QSlider::handle:horizontal:hover { background-color: $pink_hover; }
QSlider::handle:horizontal:focus { border-color: $lilac; }
QSlider::sub-page:horizontal:disabled { background-color: $text_disabled; }
QSlider::handle:horizontal:disabled { background-color: $text_disabled; border-color: $border; }

/* --- progress bars: rounded, colour per bar via the "tone" property --- */
QProgressBar {
    background-color: $surface_alt; border: none; border-radius: 6px;
    min-height: 12px; max-height: 12px; text-align: center; color: $text;
}
QProgressBar::chunk { background-color: $pink; border-radius: 6px; }
QProgressBar[tone="pink"]::chunk { background-color: $pink; }
QProgressBar[tone="lilac"]::chunk { background-color: $lilac; }
QProgressBar[tone="success"]::chunk { background-color: $success; }
QProgressBar[tone="warning"]::chunk { background-color: $warning; }
QProgressBar[tone="danger"]::chunk { background-color: $danger; }

/* --- lists, tables --- */
QListWidget, QListView, QTableWidget, QTableView, QTreeView {
    background-color: $surface; color: $text; border: 1px solid $border; border-radius: 8px;
    alternate-background-color: $surface_alt; outline: 0;
    selection-background-color: $surface_hover; selection-color: $pink;
}
QListWidget::item, QListView::item { padding: 5px 8px; }
QListWidget::item:selected, QListView::item:selected,
QTableWidget::item:selected, QTableView::item:selected { background-color: $surface_hover; color: $pink; }
QHeaderView { background-color: transparent; }
QHeaderView::section {
    background-color: $surface_alt; color: $text_secondary; border: none;
    border-bottom: 1px solid $border; padding: 6px 8px; font-weight: 600;
}
QTableCornerButton::section { background-color: $surface_alt; border: none; }

QListWidget#navList {
    background-color: $surface; border: none; border-radius: 12px; padding: 8px;
}
QListWidget#navList::item {
    padding: 10px 12px; margin: 2px 0; border-radius: 8px; color: $text_secondary;
}
QListWidget#navList::item:hover { background-color: $surface_alt; color: $text; }
QListWidget#navList::item:selected { background-color: $surface_hover; color: $pink; font-weight: 600; }

/* --- tabs / stacked pages --- */
QTabWidget::pane {
    background-color: $surface; border: 1px solid $border; border-radius: 10px; top: -1px;
}
QTabBar::tab {
    background-color: transparent; color: $text_secondary; padding: 8px 16px; margin-right: 2px;
    border-top-left-radius: 8px; border-top-right-radius: 8px;
}
QTabBar::tab:hover { color: $text; background-color: $surface_alt; }
QTabBar::tab:selected { color: $pink; background-color: $surface; }
QStackedWidget { background-color: transparent; }

/* --- scrolling --- */
QScrollArea { background-color: transparent; border: none; }
QWidget#scrollContent { background-color: transparent; }
QScrollBar:vertical { background-color: transparent; width: 10px; margin: 2px; }
QScrollBar:horizontal { background-color: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:vertical { background-color: $border; border-radius: 3px; min-height: 28px; }
QScrollBar::handle:horizontal { background-color: $border; border-radius: 3px; min-width: 28px; }
QScrollBar::handle:hover { background-color: $lilac; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; border: none; background: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

/* --- group boxes and cards --- */
QGroupBox {
    background-color: $surface; border: 1px solid $border; border-radius: 12px;
    margin-top: 22px; padding: 14px 12px 12px 12px;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 10px; padding: 0 4px; color: $lilac; font-weight: 600;
}
QFrame[card="true"] { background-color: $surface; border: 1px solid $border; border-radius: 12px; }

/* --- labels by role --- */
QLabel[role="title"] { color: $text; font-size: 17pt; font-weight: 700; }
QLabel[role="subtitle"] { color: $text_secondary; font-size: 11pt; }
QLabel[role="section"] { color: $pink; font-size: 12pt; font-weight: 700; padding-top: 6px; }
QLabel[role="hint"] { color: $text_secondary; }
QLabel[role="error"] { color: $danger; font-weight: 600; }
QLabel[role="success"] { color: $success; font-weight: 600; }
QLabel[role="warning"] { color: $warning; }
QLabel[role="value"] { color: $pink; font-weight: 600; }
QLabel[role="banner"] {
    background-color: rgba(185, 156, 255, 30); color: $text;
    border: 1px solid $lilac; border-radius: 10px; padding: 10px 12px;
}
QLabel[role="pill"] {
    background-color: $surface_alt; color: $text_secondary; border-radius: 10px; padding: 4px 10px;
}
QLabel[role="pill"][state="on"] { color: $success; }
QLabel[role="pill"][state="alert"] { color: $warning; }

/* --- chat bubbles --- */
QLabel[bubble="her"] {
    background-color: $surface_alt; color: $text;
    border-radius: 12px; border-top-left-radius: 4px; padding: 8px 12px;
}
QLabel[bubble="user"] {
    background-color: $pink; color: $on_accent;
    border-radius: 12px; border-top-right-radius: 4px; padding: 8px 12px;
}
QLabel[bubble="system"] { color: $text_secondary; padding: 2px 4px; }
"""
)


#: Flat spin buttons / combo drop-down; only used when the arrow images exist,
#: because a styled sub-control draws its arrow from ``image`` alone.
_ARROW_QSS = Template(
    """
QSpinBox, QDoubleSpinBox { padding-right: 26px; }
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right; width: 24px;
    border: none; border-top-right-radius: 8px; background-color: transparent;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right; width: 24px;
    border: none; border-bottom-right-radius: 8px; background-color: transparent;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background-color: $surface_hover; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url("$up"); width: 10px; height: 6px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url("$down"); width: 10px; height: 6px; }
QSpinBox::up-arrow:disabled, QSpinBox::up-arrow:off,
QDoubleSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:off { image: url("$up_off"); }
QSpinBox::down-arrow:disabled, QSpinBox::down-arrow:off,
QDoubleSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:off { image: url("$down_off"); }
QComboBox { padding-right: 30px; }
QComboBox::drop-down {
    subcontrol-origin: border; subcontrol-position: center right; width: 30px;
    border: none; background-color: transparent;
}
QComboBox::down-arrow { image: url("$down"); width: 10px; height: 6px; }
QComboBox::down-arrow:disabled { image: url("$down_off"); }
"""
)

#: name -> (points up, colour)
_ARROWS: dict[str, tuple[bool, str]] = {
    "up": (True, TEXT_SECONDARY),
    "down": (False, TEXT_SECONDARY),
    "up_off": (True, TEXT_DISABLED),
    "down_off": (False, TEXT_DISABLED),
}
_ARROW_W, _ARROW_H = 10, 6


def stylesheet(arrow_images: Mapping[str, str] | None = None) -> str:
    """The application stylesheet.

    *arrow_images* maps ``up``, ``down``, ``up_off``, ``down_off`` to image file
    paths (see :func:`ensure_arrow_images`); without it spin/combo buttons are
    left to the base style.
    """
    qss = _QSS.substitute(_COLORS)
    if arrow_images and all(k in arrow_images for k in _ARROWS):
        qss += _ARROW_QSS.substitute(_COLORS, **arrow_images)
    return qss


def _chevron_image(up: bool, color: str, scale: int) -> QImage:
    image = QImage(_ARROW_W * scale, _ARROW_H * scale, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        _draw_chevron(painter, QRectF(0, 0, _ARROW_W * scale, _ARROW_H * scale), QColor(color),
                      1.6 * scale, up=up)
    finally:
        painter.end()
    return image


_arrow_dir: Path | None = None


def ensure_arrow_images() -> dict[str, str] | None:
    """Render the chevron PNGs (1x and @2x for high-DPI) once per process.

    Returns stylesheet-ready paths (forward slashes, also on Windows), or None
    if they could not be written.  The directory comes from ``mkdtemp`` (private
    to the user, unpredictable name) and is removed when the process exits.
    """
    global _arrow_dir
    try:
        if _arrow_dir is None or not _arrow_dir.is_dir():
            directory = Path(tempfile.mkdtemp(prefix="chibi-theme-"))
            atexit.register(shutil.rmtree, directory, True)
            _arrow_dir = directory
        paths: dict[str, str] = {}
        for name, (up, color) in _ARROWS.items():
            base = _arrow_dir / f"{name}.png"
            for scale, path in ((1, base), (2, _arrow_dir / f"{name}@2x.png")):
                if not path.is_file() and not _chevron_image(up, color, scale).save(str(path), "PNG"):
                    raise OSError(f"could not write {path.name}")
            paths[name] = base.as_posix()
        return paths
    except OSError:
        log.warning("theme arrow images unavailable; spin/combo buttons use the base style", exc_info=True)
        return None


def build_palette() -> QPalette:
    """Palette for native-drawn parts (window backgrounds, Fusion arrows, text cursors)."""
    pal = QPalette()
    role = QPalette.ColorRole
    colors: dict[QPalette.ColorRole, str] = {
        role.Window: BACKGROUND,
        role.WindowText: TEXT,
        role.Base: SURFACE,
        role.AlternateBase: SURFACE_ALT,
        role.ToolTipBase: SURFACE_ALT,
        role.ToolTipText: TEXT,
        role.PlaceholderText: TEXT_DISABLED,
        role.Text: TEXT,
        role.Button: SURFACE_ALT,
        role.ButtonText: TEXT,
        role.BrightText: PINK,
        role.Light: SURFACE_HOVER,
        role.Midlight: BORDER_HOVER,
        role.Mid: BORDER,
        role.Dark: "#15161c",
        role.Shadow: "#0d0e12",
        role.Highlight: LILAC,
        role.HighlightedText: ON_ACCENT,
        role.Link: PINK,
        role.LinkVisited: LILAC,
    }
    if hasattr(role, "Accent"):  # Qt >= 6.6
        colors[role.Accent] = PINK
    for r, value in colors.items():
        pal.setColor(r, QColor(value))
    disabled = QPalette.ColorGroup.Disabled
    for r in (role.WindowText, role.Text, role.ButtonText):
        pal.setColor(disabled, r, QColor(TEXT_DISABLED))
    pal.setColor(disabled, role.Base, QColor(BACKGROUND))
    pal.setColor(disabled, role.Button, QColor(SURFACE))
    pal.setColor(disabled, role.Highlight, QColor(BORDER))
    return pal


class ChibiStyle(QProxyStyle):
    """Fusion with rounded, accent-coloured check boxes and flat chevron buttons."""

    INDICATOR_SIZE = 18
    #: Left text inset of the editor embedded in spin boxes (the stylesheet engine
    #: places it flush against the border and ignores the box's padding).
    SPIN_TEXT_MARGIN = 8

    def __init__(self) -> None:
        super().__init__("Fusion")

    def polish(self, arg: object) -> None:  # type: ignore[override]  # overloaded: widget/palette/app
        super().polish(arg)
        if isinstance(arg, QAbstractSpinBox):
            editor = arg.findChild(QLineEdit)
            if editor is not None and editor.textMargins().left() < self.SPIN_TEXT_MARGIN:
                margins = editor.textMargins()
                margins.setLeft(self.SPIN_TEXT_MARGIN)
                editor.setTextMargins(margins)

    def pixelMetric(self, metric: QStyle.PixelMetric, option: QStyleOption | None = None,
                    widget: QWidget | None = None) -> int:
        if metric in (QStyle.PixelMetric.PM_IndicatorWidth, QStyle.PixelMetric.PM_IndicatorHeight):
            return self.INDICATOR_SIZE
        return super().pixelMetric(metric, option, widget)

    def drawPrimitive(self, element: QStyle.PrimitiveElement, option: QStyleOption, painter: QPainter,
                      widget: QWidget | None = None) -> None:
        pe = QStyle.PrimitiveElement
        if element == pe.PE_IndicatorCheckBox:
            paint_check_indicator(painter, QRectF(option.rect), option.state)
            return
        if element in (pe.PE_IndicatorArrowDown, pe.PE_IndicatorSpinDown, pe.PE_IndicatorSpinMinus):
            paint_chevron(painter, QRectF(option.rect), option.state, up=False)
            return
        if element in (pe.PE_IndicatorArrowUp, pe.PE_IndicatorSpinUp, pe.PE_IndicatorSpinPlus):
            paint_chevron(painter, QRectF(option.rect), option.state, up=True)
            return
        super().drawPrimitive(element, option, painter, widget)


def paint_chevron(painter: QPainter, rect: QRectF, state: QStyle.StateFlag, *, up: bool) -> None:
    """Small rounded chevron for spin boxes, combo boxes and scroll arrows."""
    enabled = bool(state & QStyle.StateFlag.State_Enabled)
    w = min(rect.width(), float(_ARROW_W))
    h = min(rect.height(), w * _ARROW_H / _ARROW_W)
    if w < 3 or h < 2:
        return
    box = QRectF(0, 0, w, h)
    box.moveCenter(rect.center())
    _draw_chevron(painter, box, QColor(TEXT_SECONDARY if enabled else TEXT_DISABLED), 1.6, up=up)


def _draw_chevron(painter: QPainter, box: QRectF, color: QColor, width: float, *, up: bool) -> None:
    # Inset by half the pen so round caps stay inside the box (matters for the PNGs).
    inner = box.adjusted(width / 2, width / 2, -width / 2, -width / 2)
    top, bottom = inner.top(), inner.bottom()
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath(QPointF(inner.left(), bottom if up else top))
    path.lineTo(QPointF(inner.center().x(), top if up else bottom))
    path.lineTo(QPointF(inner.right(), bottom if up else top))
    painter.drawPath(path)
    painter.restore()


def paint_check_indicator(painter: QPainter, rect: QRectF, state: QStyle.StateFlag) -> None:
    """Rounded box; pink with a dark check mark when on, a dash when partially checked."""
    flags = QStyle.StateFlag
    enabled = bool(state & flags.State_Enabled)
    on = bool(state & flags.State_On)
    partial = bool(state & flags.State_NoChange)
    hover = bool(state & flags.State_MouseOver)
    focus = bool(state & flags.State_HasFocus)

    side = min(rect.width(), rect.height()) - 2
    box = QRectF(0, 0, side, side)
    box.moveCenter(rect.center())

    if on or partial:
        fill = QColor(PINK if enabled else TEXT_DISABLED)
        if enabled and hover:
            fill = QColor(PINK_HOVER)
        border = fill
    else:
        fill = QColor(SURFACE if enabled else BACKGROUND)
        border = QColor(PINK if enabled and (hover or focus) else (BORDER_HOVER if enabled else SURFACE_ALT))

    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(border, 1.5))
    painter.setBrush(fill)
    painter.drawRoundedRect(box.adjusted(0.75, 0.75, -0.75, -0.75), 5, 5)
    mark = QPen(QColor(ON_ACCENT), max(1.8, side / 8))
    mark.setCapStyle(Qt.PenCapStyle.RoundCap)
    mark.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(mark)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    x, y, s = box.left(), box.top(), side
    if on:
        path = QPainterPath(QPointF(x + s * 0.26, y + s * 0.53))
        path.lineTo(QPointF(x + s * 0.43, y + s * 0.70))
        path.lineTo(QPointF(x + s * 0.75, y + s * 0.33))
        painter.drawPath(path)
    elif partial:
        painter.drawLine(QPointF(x + s * 0.28, y + s * 0.5), QPointF(x + s * 0.72, y + s * 0.5))
    painter.restore()


#: Keeps the Python wrapper of the installed style alive for the app's lifetime.
_installed_style: ChibiStyle | None = None


def apply_dark_theme(app: QApplication) -> None:
    """Install the dark palette, the check-box style and the stylesheet on *app*.

    Safe to call more than once: the proxy style is installed only the first
    time (replacing the style re-polishes every widget and deletes the old
    style object), palette and stylesheet are simply set again.
    """
    global _installed_style
    if not app.property("chibi_dark_theme"):
        style = ChibiStyle()
        app.setStyle(style)
        _installed_style = style
        app.setProperty("chibi_dark_theme", True)
    app.setPalette(build_palette())
    app.setStyleSheet(stylesheet(ensure_arrow_images()))
    log.debug("dark theme applied")
