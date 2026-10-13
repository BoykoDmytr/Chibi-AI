"""«Вгадай число»: she thinks of a number from 1 to 20, the user has five
tries, and she answers "більше" / "менше" with a playful comment.

Five tries are always enough with halving (2⁵ = 32 > 20), so the game is
winnable every time.  A guess outside the range, something that is not a
number, or a number already tried does not cost a try.  Her comments get
warmer the closer the guess is; which line she picks comes from the injected
``random.Random``, so a seeded game is fully reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum

from chibi.games import GameResult, frozen_effects

LOW = 1
HIGH = 20
MAX_TRIES = 5
#: |guess - secret| up to this is "hot", up to WARM is "warm", beyond is "cold".
HOT = 2
WARM = 6


class Hint(StrEnum):
    HIGHER = "higher"  # her number is bigger
    LOWER = "lower"
    CORRECT = "correct"
    INVALID = "invalid"  # not a number / out of range: no try used
    REPEAT = "repeat"  # tried already: no try used
    OVER = "over"  # the game has ended


@dataclass(frozen=True)
class Feedback:
    guess: int | None
    hint: Hint
    tries_left: int
    comment: str
    finished: bool
    won: bool


_COMMENTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("higher", "cold"): (
        "Більше! Набагато більше.",
        "Холодно-холодно! Бери вище.",
        "Ой, ні. Моє число значно більше.",
    ),
    ("higher", "warm"): (
        "Більше, але вже тепліше!",
        "Трохи більше. Ти на правильному шляху!",
        "Тепло! Треба ще вище.",
    ),
    ("higher", "hot"): (
        "Гаряче! Але треба трішки більше.",
        "Майже! Ще крапельку вище.",
        "Ух, зовсім поруч! Трошки більше.",
    ),
    ("lower", "cold"): (
        "Менше! Набагато менше.",
        "Холодно! Бери нижче.",
        "Ой, ні. Моє число значно менше.",
    ),
    ("lower", "warm"): (
        "Менше, але вже тепліше!",
        "Трохи менше. Уже близько!",
        "Тепло! Треба ще нижче.",
    ),
    ("lower", "hot"): (
        "Гаряче! Але треба трішки менше.",
        "Майже! Ще крапельку нижче.",
        "Ух, зовсім поруч! Трошки менше.",
    ),
}
_CORRECT = (
    "Вгадано! Як у тебе це виходить?",
    "Так! Саме це число я й загадала!",
    "Бінго! Ти читаєш мої думки?",
)
_FIRST_TRY = "З першої спроби?! Це вже магія!"
_INVALID = f"Я загадала число від {LOW} до {HIGH} 🙂"
_REPEAT = "Це число вже було — спробуй інше 🙂"
_OVER = "Гру вже завершено. Зіграймо ще раз?"

STARS_WIN_BASE = 4
STARS_PER_TRY_LEFT = 2
STARS_LOSE = 2
EFFECTS_WIN = frozen_effects(boredom=-25, happiness=10, friendship_xp=4)
EFFECTS_LOSE = frozen_effects(boredom=-18, happiness=6, friendship_xp=2)


def parse_guess(value: int | str) -> int | None:
    """An int in range from user input (``" 7 "``), or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.strip().isdecimal() and len(value.strip()) <= 3:
        number = int(value.strip())
    else:
        return None
    return number if LOW <= number <= HIGH else None


class GuessNumber:
    def __init__(self, rng: random.Random, *, max_tries: int = MAX_TRIES) -> None:
        if max_tries < 1:
            raise ValueError("max_tries must be positive")
        self._rng = rng
        self._secret = rng.randint(LOW, HIGH)
        self._max_tries = max_tries
        self.guesses: list[int] = []
        self.won = False

    @property
    def tries_left(self) -> int:
        return self._max_tries - len(self.guesses)

    @property
    def max_tries(self) -> int:
        return self._max_tries

    @property
    def finished(self) -> bool:
        return self.won or self.tries_left <= 0

    @property
    def revealed(self) -> int | None:
        """Her number, once the game is over (the window shows it after a loss)."""
        return self._secret if self.finished else None

    def guess(self, value: int | str) -> Feedback:
        if self.finished:
            return self._feedback(None, Hint.OVER, _OVER)
        number = parse_guess(value)
        if number is None:
            return self._feedback(None, Hint.INVALID, _INVALID)
        if number in self.guesses:
            return self._feedback(number, Hint.REPEAT, _REPEAT)
        self.guesses.append(number)
        if number == self._secret:
            self.won = True
            comment = _FIRST_TRY if len(self.guesses) == 1 else self._rng.choice(_CORRECT)
            return self._feedback(number, Hint.CORRECT, comment)
        hint = Hint.HIGHER if self._secret > number else Hint.LOWER
        distance = abs(self._secret - number)
        warmth = "hot" if distance <= HOT else "warm" if distance <= WARM else "cold"
        comment = self._rng.choice(_COMMENTS[(hint.value, warmth)])
        if self.finished:
            comment = f"{comment} А спроби скінчилися: я загадала {self._secret}."
        return self._feedback(number, hint, comment)

    def _feedback(self, number: int | None, hint: Hint, comment: str) -> Feedback:
        return Feedback(number, hint, self.tries_left, comment, self.finished, self.won)

    def result(self) -> GameResult:
        if not self.finished:
            raise RuntimeError("the game is not over yet")
        if self.won:
            left = self.tries_left
            return GameResult(game="guess", score=left + 1, won=True,
                              stars=STARS_WIN_BASE + STARS_PER_TRY_LEFT * left, verdict="win",
                              effects=EFFECTS_WIN)
        return GameResult(game="guess", score=0, won=False, stars=STARS_LOSE, verdict="lose",
                          effects=EFFECTS_LOSE)
