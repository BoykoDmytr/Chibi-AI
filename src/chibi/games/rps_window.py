"""Window for Камінь-ножиці-папір.

Fully keyboard-playable: ``1`` / ``2`` / ``3`` pick rock / scissors / paper,
``Enter`` (or the focused "Ще раз" button) starts a new game once one is
over, ``Esc`` closes the window.  Before every round the status line says
that she has already chosen - which is literally true, see
:mod:`chibi.games.rps`.

The window only draws the game; :attr:`RpsWindow.finished` carries the
:class:`~chibi.games.GameResult` (once per game) to whoever pays the reward,
and :meth:`RpsWindow.show_reward` shows what it was worth.
"""

from __future__ import annotations

import logging
import random
from functools import partial

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from chibi.games import GAME_TITLES
from chibi.games.rps import MOVE_ICONS, MOVE_NAMES, MOVES, Move, Outcome, RockPaperScissors, Round
from chibi.settings.widgets import Card, make_button, make_label

log = logging.getLogger(__name__)

ROUND_TEXT: dict[Outcome, str] = {
    Outcome.WIN: "раунд твій!",
    Outcome.LOSE: "цей раунд мій!",
    Outcome.DRAW: "нічия, переграємо!",
}
END_TEXT: dict[Outcome, str] = {
    Outcome.WIN: "Перемога твоя! 🎉",
    Outcome.LOSE: "Я перемогла! Але гра була класна.",
    Outcome.DRAW: "Нічия! Сили рівні 😄",
}
_DIGIT_KEYS: dict[int, Move] = {
    int(Qt.Key.Key_1): MOVES[0],
    int(Qt.Key.Key_2): MOVES[1],
    int(Qt.Key.Key_3): MOVES[2],
}
_ENTER_KEYS = (int(Qt.Key.Key_Return), int(Qt.Key.Key_Enter))


def move_label(move: Move) -> str:
    return f"{MOVE_ICONS[move]} {MOVE_NAMES[move]}"


class RpsWindow(QWidget):
    #: GameResult, once per finished game.
    finished = Signal(object)

    def __init__(self, rng: random.Random, parent: QWidget | None = None, *, name: str = "Мія") -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self._rng = rng
        self._name = name or "Мія"
        self.game = RockPaperScissors(rng)
        self.setWindowTitle(GAME_TITLES["rps"])
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        layout.addWidget(make_label(GAME_TITLES["rps"], "title", wrap=False))
        layout.addWidget(make_label("Гра до двох перемог. Обирай кнопкою або клавішами 1, 2, 3.", "hint"))

        card = Card()
        top = QHBoxLayout()
        self.score_label = make_label("", "value", wrap=False)
        self.round_label = make_label("", "hint", wrap=False)
        top.addWidget(self.score_label)
        top.addStretch(1)
        top.addWidget(self.round_label)
        card.body.addLayout(top)
        self.status_label = make_label("", "subtitle")
        card.body.addWidget(self.status_label)
        self.last_label = make_label("", "")
        card.body.addWidget(self.last_label)
        layout.addWidget(card)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.move_buttons: dict[Move, QPushButton] = {}
        for number, move in enumerate(MOVES, 1):
            button = make_button(f"{move_label(move)}  ·  {number}", "accent", big=True)
            button.setAccessibleName(f"{MOVE_NAMES[move]}, клавіша {number}")
            button.clicked.connect(partial(self.choose, move))
            row.addWidget(button)
            self.move_buttons[move] = button
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

    def choose(self, move: Move) -> Round | None:
        """Play one round with the user's move (button, key)."""
        if self.game.finished:
            return None
        played = self.game.play(move)
        self.last_label.setText(
            f"Ти: {move_label(played.user)} · {self._name}: {move_label(played.her)} — "
            f"{ROUND_TEXT[played.outcome]}")
        if self.game.finished:
            self._finish()
        self._refresh()
        return played

    def new_game(self) -> None:
        self.game = RockPaperScissors(self._rng)
        self._reset_view()
        self.move_buttons[MOVES[0]].setFocus()

    def show_reward(self, text: str) -> None:
        self.reward_label.setText(text)
        self.reward_label.setVisible(bool(text))

    # -- internals ------------------------------------------------------------------------

    def _reset_view(self) -> None:
        self.last_label.setText("")
        self.show_reward("")
        self._refresh()

    def _refresh(self) -> None:
        game = self.game
        self.score_label.setText(f"Ти {game.user_wins} : {game.her_wins} {self._name}")
        self.round_label.setText(f"Раунд {game.round_number}")
        if game.finished:
            self.status_label.setText(END_TEXT[game.outcome()])
        elif not game.rounds:
            self.status_label.setText(f"{self._name} уже загадала свій хід 🤫 Обирай!")
        else:
            self.status_label.setText(f"{self._name} знову загадала хід — твоя черга!")
        for button in self.move_buttons.values():
            button.setEnabled(not game.finished)
        self.again_button.setVisible(game.finished)

    def _finish(self) -> None:
        result = self.game.result()
        self._refresh()
        self.again_button.setFocus()
        self.finished.emit(result)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in _DIGIT_KEYS and not event.isAutoRepeat():
            self.choose(_DIGIT_KEYS[key])
            event.accept()
            return
        if key in _ENTER_KEYS and self.game.finished:
            self.new_game()
            event.accept()
            return
        if key == Qt.Key.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)
