"""Камінь-ножиці-папір: the first to two round wins takes the game.

Fairness is structural, not a promise: her move for a round is drawn from
the injected ``random.Random`` when the round *begins* - at construction for
the first round, right after the previous round for the others - and stored
before :meth:`RockPaperScissors.play` ever sees the user's move.  ``play``
only reads the stored move, so nothing about the user's choice can
influence hers.  The move is not exposed until the round is over.

Draws don't count towards the two wins; after :data:`MAX_ROUNDS` rounds the
game ends anyway (whoever leads wins, an even score is a draw), so a run of
draws cannot go on forever.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum

from chibi.games import GameResult, frozen_effects

WINS_NEEDED = 2
MAX_ROUNDS = 7


class Move(StrEnum):
    ROCK = "rock"
    SCISSORS = "scissors"
    PAPER = "paper"


#: Display order; the keyboard shortcuts 1 / 2 / 3 follow it.
MOVES: tuple[Move, ...] = (Move.ROCK, Move.SCISSORS, Move.PAPER)
MOVE_NAMES: dict[Move, str] = {Move.ROCK: "Камінь", Move.SCISSORS: "Ножиці", Move.PAPER: "Папір"}
MOVE_ICONS: dict[Move, str] = {Move.ROCK: "✊", Move.SCISSORS: "✌️", Move.PAPER: "✋"}
#: move -> the move it beats.
BEATS: dict[Move, Move] = {Move.ROCK: Move.SCISSORS, Move.SCISSORS: Move.PAPER, Move.PAPER: Move.ROCK}


class Outcome(StrEnum):
    """From the user's side."""

    WIN = "win"
    LOSE = "lose"
    DRAW = "draw"


def judge(user: Move, her: Move) -> Outcome:
    if user is her:
        return Outcome.DRAW
    return Outcome.WIN if BEATS[user] is her else Outcome.LOSE


#: Stars per outcome: a lost game still pays - playing together is the point.
STARS: dict[Outcome, int] = {Outcome.WIN: 8, Outcome.DRAW: 5, Outcome.LOSE: 3}
#: Needs effects per outcome.  When she wins she is especially pleased.
EFFECTS = {
    Outcome.WIN: frozen_effects(boredom=-25, happiness=8, friendship_xp=4),
    Outcome.DRAW: frozen_effects(boredom=-20, happiness=7, friendship_xp=3),
    Outcome.LOSE: frozen_effects(boredom=-20, happiness=10, friendship_xp=3),
}


@dataclass(frozen=True)
class Round:
    number: int
    user: Move
    her: Move
    outcome: Outcome


class RockPaperScissors:
    def __init__(self, rng: random.Random, *, wins_needed: int = WINS_NEEDED,
                 max_rounds: int = MAX_ROUNDS) -> None:
        if wins_needed < 1 or max_rounds < 1:
            raise ValueError("wins_needed and max_rounds must be positive")
        self._rng = rng
        self._wins_needed = wins_needed
        self._max_rounds = max_rounds
        self.rounds: list[Round] = []
        self.user_wins = 0
        self.her_wins = 0
        self._her_next: Move | None = None
        self._commit()

    @property
    def finished(self) -> bool:
        return (self.user_wins >= self._wins_needed or self.her_wins >= self._wins_needed
                or len(self.rounds) >= self._max_rounds)

    @property
    def round_number(self) -> int:
        """The round being played now (1-based); the last one once finished."""
        return len(self.rounds) if self.finished else len(self.rounds) + 1

    @property
    def committed(self) -> bool:
        """Her move for the current round is already chosen (always, until the game ends)."""
        return self._her_next is not None

    def play(self, user_move: Move | str) -> Round:
        """Play one round with the user's move.  Raises RuntimeError once the game is over."""
        if self.finished or self._her_next is None:
            raise RuntimeError("the game is over")
        user = Move(user_move)
        her = self._her_next
        self._her_next = None
        result = Round(len(self.rounds) + 1, user, her, judge(user, her))
        self.rounds.append(result)
        if result.outcome is Outcome.WIN:
            self.user_wins += 1
        elif result.outcome is Outcome.LOSE:
            self.her_wins += 1
        self._commit()
        return result

    def outcome(self) -> Outcome:
        """The game's outcome so far (final once :attr:`finished`)."""
        if self.user_wins > self.her_wins:
            return Outcome.WIN
        if self.user_wins < self.her_wins:
            return Outcome.LOSE
        return Outcome.DRAW

    def result(self) -> GameResult:
        if not self.finished:
            raise RuntimeError("the game is not over yet")
        outcome = self.outcome()
        return GameResult(game="rps", score=self.user_wins, won=outcome is Outcome.WIN,
                          stars=STARS[outcome], verdict=outcome.value, effects=EFFECTS[outcome])

    def _commit(self) -> None:
        """Choose her move for the next round now, before the user's move exists."""
        self._her_next = None if self.finished else self._rng.choice(MOVES)
