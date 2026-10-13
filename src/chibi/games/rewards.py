"""What a finished mini-game is worth: stars into the wallet (within the
daily cap), needs effects through the tamagotchi, and her reaction through
the speech composer.

Beyond the daily cap a game still cheers her up and she still reacts - only
the stars stop, and the message says so kindly.  A failing part (a full
disk, a broken phrase bank) never stops the others.

Main (Qt) thread only.  Pure Python: no Qt here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from chibi.games import GAME_TITLES, GameResult
from chibi.tamagotchi.gifts import LineComposer, NeedsEffects, Wallet

log = logging.getLogger(__name__)

MSG_CAPPED_NONE = "Зірочки з ігор на сьогодні вже зібрано 🌙 А гратися можна й далі — їй однаково весело!"


@dataclass(frozen=True)
class Reward:
    stars: int  # actually credited
    offered: int  # what the game was worth
    message: str  # Ukrainian, for the game window

    @property
    def capped(self) -> bool:
        return self.stars < self.offered


def reward_message(granted: int, offered: int) -> str:
    if offered <= 0:
        return ""
    if granted >= offered:
        return f"Нагорода: +{granted} ⭐"
    if granted > 0:
        return (f"Нагорода: +{granted} ⭐ — це всі зірочки з ігор на сьогодні. "
                "А гратися можна й далі — їй однаково весело!")
    return MSG_CAPPED_NONE


class GameRewards:
    def __init__(self, wallet: Wallet, tamagotchi: NeedsEffects, composer: LineComposer) -> None:
        self._wallet = wallet
        self._tamagotchi = tamagotchi
        self._composer = composer

    @property
    def wallet(self) -> Wallet:
        return self._wallet

    def reward(self, result: GameResult) -> Reward:
        offered = max(0, int(result.stars))
        try:
            granted = self._wallet.earn(offered)
        except Exception:
            log.exception("game %s: crediting stars failed", result.game)
            granted = 0
        try:
            self._tamagotchi.apply_effect(result.effects, reason=f"game.{result.game}")
        except Exception:
            log.exception("game %s: applying its effect failed", result.game)
        try:
            # ``minigame`` is context for the AI brain; the phrase bank ignores it.
            self._composer.compose(result.phrase_key, solicited=True,
                                   minigame=GAME_TITLES.get(result.game, result.game))
        except Exception:
            log.exception("game %s: composing her reaction failed", result.game)
        log.info("game %s finished (%s): %d of %d stars", result.game, result.verdict, granted, offered)
        return Reward(stars=granted, offered=offered, message=reward_message(granted, offered))
