"""Mini-games she plays with the user (Phase 5).

Each game is split in two: a pure-Python logic module (``rps``, ``cookies``,
``guess``) that is deterministic for a given ``random.Random`` and fully
unit-tested, and a small Qt window (``*_window``) that only draws it and
forwards input.  A finished game produces a :class:`GameResult`;
:class:`chibi.games.rewards.GameRewards` turns it into stars, needs effects
and her reaction, and :class:`chibi.games.hub.GamesHub` connects the windows
to it for the app.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

#: Game ids, in menu order.
GAME_IDS: tuple[str, ...] = ("rps", "cookies", "guess")

#: Game id -> title shown in menus and window titles.
GAME_TITLES: dict[str, str] = {
    "rps": "Камінь-ножиці-папір",
    "cookies": "Лови печиво",
    "guess": "Вгадай число",
}


@dataclass(frozen=True)
class GameResult:
    """How a finished game went, from the user's side.

    ``won`` is the user's win; ``stars`` what the game is worth *before* the
    wallet's daily cap; ``verdict`` the phrase-key suffix for her reaction
    (``game.<game>.<verdict>``: rps ``win``/``lose``/``draw``, cookies
    ``great``/``ok``/``poor``, guess ``win``/``lose``); ``effects`` the needs
    deltas for :meth:`~chibi.tamagotchi.service.TamagotchiService.apply_effect`.
    """

    game: str
    score: int
    won: bool
    stars: int
    verdict: str
    effects: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def phrase_key(self) -> str:
        return f"game.{self.game}.{self.verdict}"


def frozen_effects(**effects: float) -> Mapping[str, float]:
    """An immutable effects mapping (a frozen GameResult must stay frozen)."""
    return MappingProxyType({name: float(value) for name, value in effects.items()})
