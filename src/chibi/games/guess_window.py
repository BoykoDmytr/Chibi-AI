"""Window for «Вгадай число».

Type a number and press ``Enter`` (or "Перевірити"); her comment, the tries
left and the history of guesses with ↑ / ↓ hints update at once.  Once the
game is over ``Enter`` starts a new one, ``Esc`` closes the window.  The
input takes digits only; a number outside 1..20 gets a friendly reminder
and costs no try.

:attr:`GuessWindow.finished` carries the :class:`~chibi.games.GameResult`
(once per game); :meth:`GuessWindow.show_reward` shows what it was worth.
"""

from __future__ import annotations

import logging
import random

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator, QKeyEvent
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from chibi.games import GAME_TITLES
from chibi.games.guess import HIGH, LOW, Feedback, GuessNumber, Hint
from chibi.settings.widgets import Card, SubmitLineEdit, count_uk, make_button, make_label

log = logging.getLogger(__name__)

_ENTER_KEYS = (int(Qt.Key.Key_Return), int(Qt.Key.Key_Enter))
_ARROWS: dict[Hint, str] = {Hint.HIGHER: "↑", Hint.LOWER: "↓", Hint.CORRECT: "✓"}


class GuessWindow(QWidget):
    #: GameResult, once per finished game.
    finished = Signal(object)

    def __init__(self, rng: random.Random, parent: QWidget | None = None, *, name: str = "Мія") -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self._rng = rng
        self._name = name or "Мія"
        self.game = GuessNumber(rng)
        self._history: list[str] = []
        self.setWindowTitle(GAME_TITLES["guess"])
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        layout.addWidget(make_label(GAME_TITLES["guess"], "title", wrap=False))
        self.intro_label = make_label("", "hint")
        layout.addWidget(self.intro_label)

        card = Card()
        self.tries_label = make_label("", "value", wrap=False)
        card.body.addWidget(self.tries_label)
        self.comment_label = make_label("", "subtitle")
        # Room for a two-line comment from the start: a top-level window does not
        # grow by itself when a label wraps later.  Polished first, so the metrics
        # are those of the theme's subtitle font, not the default one.
        self.comment_label.ensurePolished()
        self.comment_label.setMinimumHeight(self.comment_label.fontMetrics().lineSpacing() * 2 + 8)
        card.body.addWidget(self.comment_label)
        self.history_label = make_label("", "hint", wrap=False)
        card.body.addWidget(self.history_label)
        layout.addWidget(card)

        row = QHBoxLayout()
        self.input = SubmitLineEdit()
        self.input.setValidator(QIntValidator(LOW, HIGH, self.input))
        self.input.setMaxLength(2)
        self.input.setPlaceholderText(f"Число від {LOW} до {HIGH}")
        self.input.setAccessibleName("Твоє число")
        self.input.submitted.connect(self.submit)
        row.addWidget(self.input, 1)
        self.check_button = make_button("Перевірити", "primary")
        self.check_button.clicked.connect(self.submit)
        row.addWidget(self.check_button)
        layout.addLayout(row)

        bottom = QHBoxLayout()
        self.reward_label = make_label("", "success")
        bottom.addWidget(self.reward_label, 1)
        bottom.addStretch(0)  # keeps the button compact while the reward line is hidden
        self.again_button = make_button("Ще раз", "primary")
        self.again_button.setToolTip("Enter — нова гра")
        self.again_button.clicked.connect(self.new_game)
        bottom.addWidget(self.again_button, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(bottom)
        self._reset_view()

    # -- public ---------------------------------------------------------------------------

    def set_name(self, name: str) -> None:
        self._name = name or "Мія"
        self._refresh()

    def submit(self, text: str | None = None) -> Feedback | None:
        """Check the typed number (Enter / "Перевірити")."""
        if self.game.finished:
            return None
        value = self.input.text() if not isinstance(text, str) else text
        if not value.strip():
            self.comment_label.setText(f"Введи число від {LOW} до {HIGH} 🙂")
            return None
        feedback = self.game.guess(value)
        self.comment_label.setText(feedback.comment)
        if feedback.hint in _ARROWS and feedback.guess is not None:
            self._history.append(f"{feedback.guess} {_ARROWS[feedback.hint]}")
        self.input.clear()
        if feedback.finished:
            self._finish()
        self._refresh()
        return feedback

    def new_game(self) -> None:
        self.game = GuessNumber(self._rng)
        self._history = []
        self._reset_view()
        self.input.setFocus()

    def show_reward(self, text: str) -> None:
        self.reward_label.setText(text)
        self.reward_label.setVisible(bool(text))

    # -- internals ------------------------------------------------------------------------

    def _reset_view(self) -> None:
        self.comment_label.setText("Ну що, вгадаєш?")
        self.input.clear()
        self.show_reward("")
        self._refresh()

    def _refresh(self) -> None:
        game = self.game
        self.intro_label.setText(
            f"{self._name} загадала число від {LOW} до {HIGH}. "
            f"У тебе {count_uk(game.max_tries, 'спроба', 'спроби', 'спроб')}. Введи число й натисни Enter. "
            "↑ — моє число більше, ↓ — менше.")
        dots = "●" * game.tries_left + "○" * (game.max_tries - game.tries_left)
        left = count_uk(game.tries_left, "спроба", "спроби", "спроб")
        self.tries_label.setText(f"{dots}   залишилося: {left}")
        self.tries_label.setAccessibleName(f"Залишилося: {left}")
        if self._history:
            self.history_label.setText("Твої спроби: " + " · ".join(self._history))
        else:
            self.history_label.setText("")
        playing = not game.finished
        self.input.setEnabled(playing)
        self.check_button.setEnabled(playing)
        self.again_button.setVisible(not playing)

    def _finish(self) -> None:
        result = self.game.result()
        self._refresh()
        self.again_button.setFocus()
        self.finished.emit(result)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in _ENTER_KEYS and self.game.finished:
            self.new_game()
            event.accept()
            return
        if key == Qt.Key.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)
