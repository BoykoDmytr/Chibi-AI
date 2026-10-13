"""Chat window.  The banner says honestly whether she answers through an AI
model (and which one) or is in offline mode with only simple phrases.

* Replies arrive through ``facade.send_chat_stream``: her bubble appears at
  once with a typing indicator, AI text pieces are appended as they stream in,
  and the final reply replaces the bubble text (the app may have cleaned it
  up).  Offline replies come through ``on_done`` alone.  The UI never waits:
  the callbacks run later on the Qt main thread (or immediately, offline).
* One reply at a time: the send button is disabled while a reply is in
  flight.  A watchdog re-enables it should a reply never arrive, and a late
  answer for a finished or abandoned exchange - or for a window that was
  closed and deleted meanwhile - is dropped.
* Text is always shown as plain text (never interpreted as HTML) and is never
  logged.
* History lives only in this window's memory (capped), nothing is written to
  disk - the tooltip on the message area says so.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QPixmap, QShowEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from chibi.settings.facade import AIStatusView, AppFacade, ChatReply
from chibi.settings.widgets import (
    STATE_PORTRAITS,
    PortraitLabel,
    SubmitLineEdit,
    ai_title,
    is_alive,
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
#: Typing indicator frames and their pace.
TYPING_FRAMES = ("•", "• •", "• • •")
TYPING_MS = 420
#: Give up on a reply after this long.  The app always answers (an AI failure
#: falls back to an offline line); this only keeps a lost answer from
#: disabling the send button forever.
REPLY_WATCHDOG_MS = 300_000

OFFLINE_SUFFIX = "я розумію лише прості фрази"
#: Shown when the AI status can't be read at all.
OFFLINE_BANNER = f"Офлайн-режим — {OFFLINE_SUFFIX}"
HISTORY_TOOLTIP = "Розмова не зберігається: після перезапуску програми історія зникне."
PLACEHOLDER_OFFLINE = "Напиши щось… (наприклад, «привіт»)"
PLACEHOLDER_ONLINE = "Напиши їй що завгодно…"
ERROR_LINE = "Щось пішло не так — спробуй ще раз."
TIMEOUT_LINE = "Вона так і не відповіла — спробуй ще раз."

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
#: The reply control line (``<<state=happy;action=none>>``) is the app's business;
#: should one ever reach the stream it is hidden here too.
_CONTROL_LINE = re.compile(r"\A\s*<<[^\n]*?>>[ \t]*(\r?\n)?")


def _soft_wrap(text: str, every: int = 20) -> str:
    def split(match: re.Match[str]) -> str:
        run = match.group(0)
        return "​".join(run[i:i + every] for i in range(0, len(run), every))

    return _LONG_RUN.sub(split, text)


def visible_stream_text(raw: str) -> str:
    """What to show of the streamed text so far: without a leading control
    line, and nothing while such a line may still be arriving."""
    stripped = raw.lstrip()
    if stripped.startswith("<<"):
        match = _CONTROL_LINE.match(raw)
        if match is not None:
            raw = raw[match.end():]
        elif "\n" not in stripped:
            return ""  # the control line isn't complete yet
    elif stripped == "<":
        return ""  # could be the start of one
    return raw.lstrip("\n")


def banner_text(view: AIStatusView) -> str:
    if view.online:
        return f"AI: {ai_title(view.provider, view.model)}"
    reason = view.reason.strip()
    return f"Офлайн-режим: {reason} — {OFFLINE_SUFFIX}" if reason else OFFLINE_BANNER


@dataclass
class _Pending:
    """The reply being received."""

    seq: int
    row: QWidget
    bubble: QLabel
    avatar: PortraitLabel
    raw: str = ""
    shown: str = ""


class ChatWindow(QWidget):
    def __init__(self, facade: AppFacade, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self._facade = facade
        self._rows: list[QWidget] = []
        self._bubbles: list[QLabel] = []
        self._avatars: dict[str, QPixmap | None] = {}
        self._at_bottom = True
        self._force_scroll = False
        self._seq = 0
        self._pending: _Pending | None = None
        self._typing_frame = 0
        self._status_failed = False
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
        self.input.setPlaceholderText(PLACEHOLDER_OFFLINE)
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

        self.typing_timer = QTimer(self)
        self.typing_timer.setInterval(TYPING_MS)
        self.typing_timer.timeout.connect(self._animate_typing)
        self.watchdog = QTimer(self)
        self.watchdog.setSingleShot(True)
        self.watchdog.setInterval(REPLY_WATCHDOG_MS)
        self.watchdog.timeout.connect(self._on_watchdog)

        self._update_title()
        self.refresh_status()

    # --- public API ---------------------------------------------------------------------

    def is_waiting(self) -> bool:
        """True while a reply is in flight."""
        return self._pending is not None

    def send(self) -> None:
        """Send the input line (ignored if blank or while she is still answering)."""
        if self._pending is not None:
            return
        text = self.input.text().strip()[:MAX_INPUT_CHARS]
        if not text:
            return
        self.input.clear()
        self._update_title()
        self.add_user_message(text)
        self._seq += 1
        seq = self._seq
        self._pending = self._start_her_bubble(seq)
        self.send_button.setEnabled(False)
        self.watchdog.start()

        def on_delta(piece: str) -> None:
            if is_alive(self):
                self._on_delta(seq, piece)

        def on_done(reply: ChatReply) -> None:
            if is_alive(self):
                self._on_done(seq, reply)

        try:
            self._facade.send_chat_stream(text, on_delta, on_done)
        except Exception as exc:
            # Only the type: an exception message may quote the chat text, which is never logged.
            log.error("chat reply failed: %s", type(exc).__name__)
            if self._pending is not None and self._pending.seq == seq:
                self._finish(keep=False)
                self.add_system_line(ERROR_LINE)

    def show_reply(self, reply: ChatReply) -> None:
        """Show a complete reply as a new bubble (plus its action line)."""
        if reply.text:
            self.add_her_message(reply.text, self._portrait_for(reply))
        self._show_action(reply)

    def add_user_message(self, text: str) -> None:
        bubble = self._bubble(text, "user")
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(40, 0, 0, 0)
        layout.addStretch(1)
        layout.addWidget(bubble)
        self._append(row, bubble)

    def add_her_message(self, text: str, portrait: str = "smile") -> None:
        self._her_row(text, portrait)

    def add_system_line(self, text: str) -> None:
        label = make_label(text, wrap=True)
        label.setProperty("bubble", "system")
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._append(label, None)

    def message_count(self) -> int:
        return len(self._rows)

    def refresh_status(self) -> None:
        """Show in the banner whether she talks through AI right now."""
        try:
            view = self._facade.ai_status()
        except Exception:
            if not self._status_failed:
                log.exception("reading the AI status failed")
            self._status_failed = True
            self.banner.setText(OFFLINE_BANNER)
            self.input.setPlaceholderText(PLACEHOLDER_OFFLINE)
            return
        self._status_failed = False
        self.banner.setText(banner_text(view))
        self.input.setPlaceholderText(PLACEHOLDER_ONLINE if view.online else PLACEHOLDER_OFFLINE)

    # --- the reply in flight ------------------------------------------------------------------

    def _start_her_bubble(self, seq: int) -> _Pending:
        row, bubble, avatar = self._her_row(TYPING_FRAMES[-1], "thinking", "smile")
        bubble.setAccessibleName("Вона друкує")
        self._typing_frame = len(TYPING_FRAMES) - 1
        self.typing_timer.start()
        return _Pending(seq=seq, row=row, bubble=bubble, avatar=avatar)

    def _animate_typing(self) -> None:
        pending = self._pending
        if pending is None or pending.shown or not is_alive(pending.bubble):
            self.typing_timer.stop()
            return
        self._typing_frame = (self._typing_frame + 1) % len(TYPING_FRAMES)
        self._set_bubble_text(pending.bubble, TYPING_FRAMES[self._typing_frame])

    def _on_delta(self, seq: int, piece: str) -> None:
        pending = self._pending
        if pending is None or pending.seq != seq or not piece:
            return
        pending.raw += piece
        shown = visible_stream_text(pending.raw)
        if not shown.strip() or shown == pending.shown:
            return
        pending.shown = shown
        self.typing_timer.stop()
        if is_alive(pending.bubble):
            self._set_bubble_text(pending.bubble, shown)

    def _on_done(self, seq: int, reply: ChatReply) -> None:
        pending = self._pending
        if pending is None or pending.seq != seq:
            return  # a late answer for an exchange the watchdog already ended
        if not isinstance(reply, ChatReply):
            log.error("chat reply of unexpected type %s", type(reply).__name__)
            self._finish(keep=bool(pending.shown))
            self.add_system_line(ERROR_LINE)
            return
        # The final text replaces whatever streamed in (the app may have cleaned it up).
        if reply.text and is_alive(pending.bubble):
            self._set_bubble_text(pending.bubble, reply.text)
            pending.bubble.setAccessibleName("")
            if is_alive(pending.avatar):
                pending.avatar.show_portrait(self._portrait_for(reply), "smile", "calm")
        self._finish(keep=bool(reply.text))
        self._show_action(reply)
        self.refresh_status()

    def _on_watchdog(self) -> None:
        if self._pending is None:
            return
        log.warning("chat reply did not arrive in time")
        self._finish(keep=bool(self._pending.shown))
        self.add_system_line(TIMEOUT_LINE)

    def _finish(self, *, keep: bool) -> None:
        """End the exchange; *keep* her bubble or drop it (it shows no real text)."""
        pending = self._pending
        self._pending = None
        self.typing_timer.stop()
        self.watchdog.stop()
        self.send_button.setEnabled(True)
        if pending is not None and not keep:
            self._remove_row(pending.row)

    # --- internals ---------------------------------------------------------------------------

    def _show_action(self, reply: ChatReply) -> None:
        line = ACTION_LINES.get(reply.action)
        if line:
            self.add_system_line(line.format(name=self._name()))
        elif reply.action:
            log.debug("no chat line for action %r", reply.action)

    @staticmethod
    def _portrait_for(reply: ChatReply) -> str:
        return reply.portrait or STATE_PORTRAITS.get(reply.state, "smile")

    def _her_row(self, text: str, *portraits: str) -> tuple[QWidget, QLabel, PortraitLabel]:
        bubble = self._bubble(text, "her")
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 40, 0)
        layout.setSpacing(8)
        avatar = PortraitLabel(self._avatar, AVATAR_SIZE, fallback_text=self._name())
        avatar.show_portrait(*portraits, "smile", "calm")
        layout.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(bubble)
        layout.addStretch(1)
        self._append(row, bubble)
        return row, bubble, avatar

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
        bubble.setWordWrap(True)
        bubble.setProperty("bubble", kind)
        bubble.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._set_bubble_text(bubble, text)
        return bubble

    def _set_bubble_text(self, bubble: QLabel, text: str) -> None:
        bubble.setText(_soft_wrap(text))
        bubble.setProperty("natural_width", self._natural_width(bubble))
        self._fit(bubble, self._bubble_width())
        if self._at_bottom:
            self._force_scroll = True  # a growing bubble keeps the newest text in view

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
            self._remove_row(self._rows[0])

    def _remove_row(self, row: QWidget) -> None:
        if row in self._rows:
            self._rows.remove(row)
        if not is_alive(row):
            return
        self._bubbles = [b for b in self._bubbles if is_alive(b) and not row.isAncestorOf(b)]
        self.messages_layout.removeWidget(row)
        row.hide()  # removeWidget alone leaves it painted until the deferred delete runs
        row.setParent(None)  # out of the message list at once, not only after the deferred delete
        row.deleteLater()

    def _on_range_changed(self, _minimum: int, maximum: int) -> None:
        if self._force_scroll or self._at_bottom:
            self.scroll.verticalScrollBar().setValue(maximum)
            self._force_scroll = False

    def _on_scrolled(self, value: int) -> None:
        self._at_bottom = value >= self.scroll.verticalScrollBar().maximum() - 4

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._update_title()
        self.refresh_status()
        self.input.setFocus()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.scroll.viewport() and event.type() == QEvent.Type.Resize:
            width = self._bubble_width()
            for bubble in self._bubbles:
                if is_alive(bubble):
                    self._fit(bubble, width)
        return super().eventFilter(watched, event)
