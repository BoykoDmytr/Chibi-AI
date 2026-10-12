"""Chat window.  Phase 1 is honest about being offline: a banner says AI is not
connected and she only understands simple phrases.

* Text is always shown as plain text (never interpreted as HTML) and is never
  logged.
* History lives only in this window's memory (capped), nothing is written to
  disk - the tooltip on the message area says so.
* ``facade.send_chat`` is synchronous in Phase 1 (offline responder); Phase 2
  will need an async variant so an AI call never blocks the UI thread.
"""

from __future__ import annotations

import logging
import re

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QPixmap, QShowEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from chibi.settings.facade import AppFacade, ChatReply
from chibi.settings.widgets import (
    STATE_PORTRAITS,
    PortraitLabel,
    SubmitLineEdit,
    make_button,
    make_label,
)

log = logging.getLogger(__name__)

MAX_INPUT_CHARS = 500
#: Oldest messages are dropped beyond this many rows (memory/widget count bound).
MAX_MESSAGES = 200
#: Bubbles take at most this share of the message area's width.
BUBBLE_WIDTH_RATIO = 0.72
AVATAR_SIZE = 32

OFFLINE_BANNER = "Офлайн-режим: AI ще не підключено — я розумію лише прості фрази"
HISTORY_TOOLTIP = "Розмова не зберігається: після перезапуску програми історія зникне."

#: ChatReply.action -> system line (``{name}`` = her name)
ACTION_LINES: dict[str, str] = {
    "quiet_on": "🔇 {name} помовчить",
    "quiet_off": "🔊 {name} знову може говорити",
    "fed": "🍙 {name} поїла",
    "petted": "💗 {name} задоволено мружиться",
    "rest": "💤 {name} пішла відпочити",
}

# Runs of non-space characters longer than this get invisible break opportunities,
# otherwise one long word (a URL, "аааааа...") would stretch the bubble past the window.
_LONG_RUN = re.compile(r"\S{25,}")


def _soft_wrap(text: str, every: int = 20) -> str:
    def split(match: re.Match[str]) -> str:
        run = match.group(0)
        return "​".join(run[i:i + every] for i in range(0, len(run), every))

    return _LONG_RUN.sub(split, text)


class ChatWindow(QWidget):
    def __init__(self, facade: AppFacade, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self._facade = facade
        self._rows: list[QWidget] = []
        self._bubbles: list[QLabel] = []
        self._avatars: dict[str, QPixmap | None] = {}
        self._at_bottom = True
        self._force_scroll = False
        self.setMinimumSize(380, 420)
        self.resize(440, 560)

        self.banner = make_label(OFFLINE_BANNER, "banner")

        self.messages = QWidget()
        self.messages.setObjectName("scrollContent")
        self.messages_layout = QVBoxLayout(self.messages)
        self.messages_layout.setContentsMargins(4, 4, 4, 4)
        self.messages_layout.setSpacing(8)
        self.messages_layout.addStretch(1)  # keeps messages at the bottom while few

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidget(self.messages)
        self.scroll.viewport().setAutoFillBackground(False)
        self.scroll.setToolTip(HISTORY_TOOLTIP)
        self.scroll.viewport().installEventFilter(self)
        bar = self.scroll.verticalScrollBar()
        bar.rangeChanged.connect(self._on_range_changed)
        bar.valueChanged.connect(self._on_scrolled)

        self.input = SubmitLineEdit()
        self.input.setMaxLength(MAX_INPUT_CHARS)
        self.input.setPlaceholderText("Напиши щось… (наприклад, «привіт»)")
        self.input.setToolTip(HISTORY_TOOLTIP)
        self.input.submitted.connect(self.send)
        self.send_button = make_button("Надіслати", "primary")
        self.send_button.clicked.connect(self.send)
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addWidget(self.input, 1)
        bottom.addWidget(self.send_button)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        root.addWidget(self.banner)
        root.addWidget(self.scroll, 1)
        root.addLayout(bottom)

        self._update_title()

    # --- public API ---------------------------------------------------------------------

    def send(self) -> None:
        """Send the input line (ignored if blank) and show her reply."""
        text = self.input.text().strip()[:MAX_INPUT_CHARS]
        if not text:
            return
        self.input.clear()
        self._update_title()
        self.add_user_message(text)
        try:
            reply = self._facade.send_chat(text)
        except Exception as exc:
            # Only the type: an exception message may quote the chat text, which is never logged.
            log.error("chat reply failed: %s", type(exc).__name__)
            self.add_system_line("Щось пішло не так — спробуй ще раз.")
            return
        self.show_reply(reply)

    def show_reply(self, reply: ChatReply) -> None:
        if reply.text:
            portrait = reply.portrait or STATE_PORTRAITS.get(reply.state, "smile")
            self.add_her_message(reply.text, portrait)
        line = ACTION_LINES.get(reply.action)
        if line:
            self.add_system_line(line.format(name=self._name()))
        elif reply.action:
            log.debug("no chat line for action %r", reply.action)

    def add_user_message(self, text: str) -> None:
        bubble = self._bubble(text, "user")
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(40, 0, 0, 0)
        layout.addStretch(1)
        layout.addWidget(bubble)
        self._append(row, bubble)

    def add_her_message(self, text: str, portrait: str = "smile") -> None:
        bubble = self._bubble(text, "her")
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 40, 0)
        layout.setSpacing(8)
        avatar = PortraitLabel(self._avatar, AVATAR_SIZE, fallback_text=self._name())
        avatar.show_portrait(portrait, "smile", "calm")
        layout.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(bubble)
        layout.addStretch(1)
        self._append(row, bubble)

    def add_system_line(self, text: str) -> None:
        label = make_label(text, wrap=True)
        label.setProperty("bubble", "system")
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._append(label, None)

    def message_count(self) -> int:
        return len(self._rows)

    # --- internals ---------------------------------------------------------------------------

    def _name(self) -> str:
        try:
            return self._facade.settings().character.name
        except Exception:
            log.exception("reading the character name failed")
            return "Мія"

    def _update_title(self) -> None:
        self.setWindowTitle(f"Чат — {self._name()}")

    def _avatar(self, name: str) -> QPixmap | None:
        # One facade lookup per portrait name; each row still scales its own copy.
        if name not in self._avatars:
            self._avatars[name] = self._facade.portrait(name)
        return self._avatars[name]

    def _bubble(self, text: str, kind: str) -> QLabel:
        bubble = QLabel()
        bubble.setTextFormat(Qt.TextFormat.PlainText)  # user text must never become HTML
        bubble.setText(_soft_wrap(text))
        bubble.setWordWrap(True)
        bubble.setProperty("bubble", kind)
        bubble.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bubble.setProperty("natural_width", self._natural_width(bubble))
        self._fit(bubble, self._bubble_width())
        return bubble

    @staticmethod
    def _natural_width(bubble: QLabel) -> int:
        """Width of the bubble with its text unwrapped (padding included).

        A word-wrapped QLabel asks for an arbitrary narrow width by itself;
        without wrapping its size hint is exactly "longest line + padding".
        """
        bubble.ensurePolished()
        bubble.setWordWrap(False)
        width = bubble.sizeHint().width()
        bubble.setWordWrap(True)
        return width + 1

    @staticmethod
    def _fit(bubble: QLabel, max_width: int) -> None:
        # Fixed (not min/max) width: the row layout then computes the wrapped
        # height for the width the bubble really gets.
        bubble.setFixedWidth(max(1, min(int(bubble.property("natural_width") or 0), max_width)))

    def _bubble_width(self) -> int:
        return max(160, int(self.scroll.viewport().width() * BUBBLE_WIDTH_RATIO))

    def _append(self, row: QWidget, bubble: QLabel | None) -> None:
        self._force_scroll = True
        self.messages_layout.addWidget(row)  # after the leading stretch: newest at the bottom
        self._rows.append(row)
        if bubble is not None:
            self._bubbles.append(bubble)
        while len(self._rows) > MAX_MESSAGES:
            old = self._rows.pop(0)
            self._bubbles = [b for b in self._bubbles if not old.isAncestorOf(b)]
            self.messages_layout.removeWidget(old)
            old.hide()  # removeWidget alone leaves it painted until the deferred delete runs
            old.deleteLater()

    def _on_range_changed(self, _minimum: int, maximum: int) -> None:
        if self._force_scroll or self._at_bottom:
            self.scroll.verticalScrollBar().setValue(maximum)
            self._force_scroll = False

    def _on_scrolled(self, value: int) -> None:
        self._at_bottom = value >= self.scroll.verticalScrollBar().maximum() - 4

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._update_title()
        self.input.setFocus()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.scroll.viewport() and event.type() == QEvent.Type.Resize:
            width = self._bubble_width()
            for bubble in self._bubbles:
                self._fit(bubble, width)
        return super().eventFilter(watched, event)
