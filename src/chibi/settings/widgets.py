"""Small widgets and helpers shared by the settings, status, chat and first-run windows.

They carry no app logic: look (via the dynamic properties understood by
:mod:`chibi.settings.theme`), Ukrainian number/time formatting and a couple of
behaviours Qt doesn't give for free (an Enter key that doesn't trigger a
dialog's default button, a portrait with a drawn fallback avatar).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from datetime import datetime

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QKeyEvent, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from chibi.settings import theme

log = logging.getLogger(__name__)

# --- Ukrainian formatting ----------------------------------------------------------------

#: Character state value -> what she is doing, as shown in the status window.
STATE_NAMES_UK: dict[str, str] = {
    "idle": "Байдикує",
    "walk": "Гуляє",
    "run": "Бігає",
    "sit": "Сидить",
    "sleep": "Спить",
    "gaming": "Грає",
    "talk": "Розмовляє",
    "laugh": "Сміється",
    "happy": "Радіє",
    "angry": "Сердиться",
    "sad": "Сумує",
    "surprised": "Дивується",
    "curious": "Роздивляється",
    "thinking": "Думає",
    "wave": "Махає рукою",
    "eat": "Їсть",
    "shy": "Соромиться",
    "yawn": "Позіхає",
    "bye": "Прощається",
    "point": "Щось показує",
    "dragged": "Її переносять",
}

#: Character state value -> default portrait of the "mia" pack (mirrors manifest.json
#: ``states.*.portrait``; the facade only exposes portraits by name).
STATE_PORTRAITS: dict[str, str] = {
    "idle": "calm",
    "walk": "calm",
    "run": "excited",
    "sit": "smile",
    "sleep": "sleep_cat",
    "gaming": "typing",
    "talk": "smile",
    "laugh": "laugh",
    "happy": "joy",
    "angry": "angry",
    "sad": "offended",
    "surprised": "wow",
    "curious": "question",
    "thinking": "thinking",
    "wave": "wink",
    "eat": "eat",
    "shy": "shy",
    "yawn": "tired",
    "bye": "bye",
    "point": "point",
    "dragged": "wow",
}


def state_name_uk(state: str) -> str:
    return STATE_NAMES_UK.get(state, "Чимось зайнята")


def plural_uk(n: int, one: str, few: str, many: str) -> str:
    """Ukrainian plural form: 1 година, 2 години, 5 годин, 11 годин, 21 година."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def count_uk(n: int, one: str, few: str, many: str) -> str:
    return f"{n} {plural_uk(n, one, few, many)}"


def format_duration(seconds: float) -> str:
    """``"2 години 5 хвилин"``, ``"1 година"``, ``"45 хвилин"``, ``"менше хвилини"``."""
    if not math.isfinite(seconds) or seconds < 60:
        return "менше хвилини"
    hours, minutes = divmod(int(seconds // 60), 60)
    parts = []
    if hours:
        parts.append(count_uk(hours, "година", "години", "годин"))
    if minutes:
        parts.append(count_uk(minutes, "хвилина", "хвилини", "хвилин"))
    return " ".join(parts)


def _local(ts: float) -> datetime | None:
    try:
        return datetime.fromtimestamp(ts)
    except (OverflowError, OSError, ValueError, TypeError):
        return None


def format_local_datetime(ts: float) -> str:
    """Epoch seconds -> ``"12.10.2026 14:05"`` in local time (``"—"`` if invalid)."""
    dt = _local(ts)
    return dt.strftime("%d.%m.%Y %H:%M") if dt else "—"


def format_local_date(ts: float) -> str:
    dt = _local(ts)
    return dt.strftime("%d.%m.%Y") if dt else "—"


def format_local_time(ts: float) -> str:
    dt = _local(ts)
    return dt.strftime("%H:%M") if dt else "—"


def percent(value: float) -> str:
    """``"72 %"`` with a no-break space, as Ukrainian typography wants."""
    return f"{round(value)} %"


# --- styling helpers -----------------------------------------------------------------------


def set_style_property(widget: QWidget, name: str, value: object) -> None:
    """Set a dynamic property the stylesheet selects on, and re-polish so it takes effect."""
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def make_button(text: str, variant: str = "", *, big: bool = False,
                parent: QWidget | None = None) -> QPushButton:
    """``variant``: ``""`` (neutral), ``"primary"`` (pink), ``"accent"`` (lilac) or ``"danger"``."""
    button = QPushButton(text, parent)
    if variant:
        button.setProperty("variant", variant)
    if big:
        button.setProperty("big", True)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


def make_label(text: str = "", role: str = "", *, wrap: bool = True,
               parent: QWidget | None = None) -> QLabel:
    """Plain-text label (never interprets HTML) with an optional theme role."""
    label = QLabel(parent)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setText(text)
    label.setWordWrap(wrap)
    if role:
        label.setProperty("role", role)
    return label


class SectionHeader(QLabel):
    """Pink bold heading that separates groups of options on a page."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setText(text)
        self.setProperty("role", "section")


class InfoLabel(QLabel):
    """Secondary-colour, word-wrapped explanation text."""

    def __init__(self, text: str = "", parent: QWidget | None = None, *, role: str = "hint") -> None:
        super().__init__(parent)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setText(text)
        self.setWordWrap(True)
        self.setProperty("role", role)


class Card(QFrame):
    """Rounded surface panel with its own vertical layout (``card.body``)."""

    def __init__(self, parent: QWidget | None = None, *, spacing: int = 10) -> None:
        super().__init__(parent)
        self.setProperty("card", True)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(16, 14, 16, 14)
        self.body.setSpacing(spacing)


class CheckRow(QWidget):
    """A check box with an explanation aligned under its text."""

    def __init__(self, text: str, hint: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)
        self.checkbox = QCheckBox(text, self)
        layout.addWidget(self.checkbox)
        self.hint = InfoLabel(hint, self)
        # indicator width + spacing (see theme), so the hint lines up with the text
        self.hint.setContentsMargins(28, 0, 0, 0)
        self.hint.setVisible(bool(hint))
        layout.addWidget(self.hint)

    def isChecked(self) -> bool:
        return self.checkbox.isChecked()

    def setChecked(self, checked: bool) -> None:
        self.checkbox.setChecked(checked)


class SliderRow(QWidget):
    """Title, live value and a horizontal slider, with an optional explanation below."""

    valueChanged = Signal(int)

    def __init__(self, title: str, minimum: int, maximum: int, value: int, *, step: int = 1,
                 description: str = "", formatter: Callable[[int], str] | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._formatter = formatter or str
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(2)
        self.title_label = make_label(title, wrap=False, parent=self)
        self.value_label = make_label("", "value", wrap=False, parent=self)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setMinimumWidth(56)
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(minimum, maximum)
        self.slider.setSingleStep(step)
        self.slider.setPageStep(max(step, (maximum - minimum) // 10))
        self.slider.setAccessibleName(title)
        grid.addWidget(self.title_label, 0, 0)
        grid.addWidget(self.value_label, 0, 1)
        grid.addWidget(self.slider, 1, 0, 1, 2)
        self.description_label: InfoLabel | None = None
        if description:
            self.description_label = InfoLabel(description, self)
            grid.addWidget(self.description_label, 2, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        self.slider.valueChanged.connect(self._on_value)
        self.slider.setValue(value)
        self._on_value(self.slider.value())

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int) -> None:
        self.slider.setValue(int(value))

    def _on_value(self, value: int) -> None:
        self.value_label.setText(self._formatter(value))
        self.valueChanged.emit(value)


class NeedsBar(QWidget):
    """One tamagotchi need: title, percentage and a rounded bar whose colour warns when low.

    ``inverted`` is for needs where *high* is bad (boredom): the bar shows the
    raw value but turns warning/danger as it fills up.
    """

    WARN_BELOW = 40
    DANGER_BELOW = 20

    def __init__(self, title: str, tone: str = "pink", *, inverted: bool = False, tooltip: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if tone not in theme.TONES:
            raise ValueError(f"unknown tone {tone!r}")
        self._base_tone = tone
        self._inverted = inverted
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self.title_label = make_label(title, wrap=False, parent=self)
        self.value_label = make_label("", "hint", wrap=False, parent=self)
        top.addWidget(self.title_label)
        top.addStretch(1)
        top.addWidget(self.value_label)
        layout.addLayout(top)
        self.bar = QProgressBar(self)
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setAccessibleName(title)
        self.bar.setProperty("tone", tone)
        layout.addWidget(self.bar)
        if tooltip:
            self.setToolTip(tooltip)
        self.set_value(0)

    def set_value(self, value: float) -> None:
        v = 0 if not math.isfinite(value) else min(max(round(value), 0), 100)
        self.bar.setValue(v)
        self.value_label.setText(percent(v))
        goodness = 100 - v if self._inverted else v
        if goodness < self.DANGER_BELOW:
            tone = "danger"
        elif goodness < self.WARN_BELOW:
            tone = "warning"
        else:
            tone = self._base_tone
        set_style_property(self.bar, "tone", tone)

    def value(self) -> int:
        return self.bar.value()

    def tone(self) -> str:
        return str(self.bar.property("tone"))


class SubmitLineEdit(QLineEdit):
    """Line edit that emits ``submitted`` on Enter and *consumes* the key.

    A plain QLineEdit lets Enter propagate, so inside a dialog it would also
    press the dialog's default button (e.g. "Гаразд" while adding an exclusion).
    """

    submitted = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            self.submitted.emit()
            return
        super().keyPressEvent(event)


def scroll_page(content: QWidget) -> QScrollArea:
    """Wrap a page so it stays usable in a small window (vertical scrolling only)."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    content.setObjectName("scrollContent")
    area.setWidget(content)
    area.viewport().setAutoFillBackground(False)
    return area


# --- portraits ---------------------------------------------------------------------------


def text_avatar(text: str, size: int, device_pixel_ratio: float = 1.0) -> QPixmap:
    """Round lilac-to-pink avatar with the first letter of *text* (fallback for a missing portrait)."""
    dpr = max(1.0, device_pixel_ratio)
    px = max(1, round(size * dpr))
    pixmap = QPixmap(px, px)
    pixmap.fill(Qt.GlobalColor.transparent)
    letter = next((ch for ch in text if ch.isalnum()), "?").upper()
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        gradient = QLinearGradient(0, 0, px, px)
        gradient.setColorAt(0.0, QColor(theme.LILAC))
        gradient.setColorAt(1.0, QColor(theme.PINK))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(QRectF(0.5, 0.5, px - 1, px - 1))
        font = QFont()
        font.setBold(True)
        font.setPixelSize(max(1, round(px * 0.46)))
        painter.setFont(font)
        painter.setPen(QColor(theme.ON_ACCENT))
        painter.drawText(QRectF(0, 0, px, px), Qt.AlignmentFlag.AlignCenter, letter)
    finally:
        painter.end()
    pixmap.setDevicePixelRatio(dpr)
    return pixmap


class PortraitLabel(QLabel):
    """Shows a named portrait from the character pack, or a drawn letter avatar.

    Scaled pixmaps are cached per name and missing names are remembered, so a
    window refreshing every couple of seconds doesn't hit the disk each time.
    """

    def __init__(self, loader: Callable[[str], QPixmap | None], size: int, *, fallback_text: str = "?",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loader = loader
        self._size = size
        self._fallback_text = fallback_text
        self._cache: dict[str, QPixmap] = {}
        self._missing: set[str] = set()
        self._current: str | None = None  # "" = text avatar
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    @property
    def current(self) -> str | None:
        """Name of the portrait shown, ``""`` for the letter avatar, None before the first call."""
        return self._current

    def set_fallback_text(self, text: str) -> None:
        if text != self._fallback_text:
            self._fallback_text = text
            if self._current == "":
                self._current = None  # redraw the avatar with the new letter

    def show_portrait(self, *names: str) -> str:
        """Show the first of *names* that exists; returns its name, or ``""`` for the avatar."""
        for name in names:
            if not name:
                continue
            pixmap = self._get(name)
            if pixmap is None:
                continue
            if name != self._current:
                self.setPixmap(pixmap)
                self._current = name
            return name
        if self._current != "":
            self.setPixmap(text_avatar(self._fallback_text, self._size, self.devicePixelRatioF()))
            self._current = ""
        return ""

    def _get(self, name: str) -> QPixmap | None:
        if name in self._cache:
            return self._cache[name]
        if name in self._missing:
            return None
        try:
            raw = self._loader(name)
        except Exception:
            log.exception("loading portrait %r failed", name)
            raw = None
        if raw is None or raw.isNull():
            self._missing.add(name)
            return None
        dpr = max(1.0, self.devicePixelRatioF())
        side = max(1, round(self._size * dpr))
        scaled = raw.scaled(side, side, Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        scaled.setDevicePixelRatio(dpr)
        self._cache[name] = scaled
        return scaled
