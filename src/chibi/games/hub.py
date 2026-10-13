"""The one object ``app.py`` builds for mini-games and the gift shop.

:class:`GamesHub` opens the three game windows and the shop (one of each,
created on first use and reused afterwards), pays every finished game
through :class:`~chibi.games.rewards.GameRewards` and shows the reward in
the game's window.  Opening a game she was not already playing gets a
cheerful line from her ("game.start").  :meth:`GamesHub.close_all` closes
and deletes every window (shutdown, "delete all data").

Main (Qt) thread only; no work here is slow enough for a worker.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget

from chibi.games import GAME_IDS, GameResult
from chibi.games.cookie_window import CookieWindow
from chibi.games.guess_window import GuessWindow
from chibi.games.rewards import GameRewards, Reward
from chibi.games.rps_window import RpsWindow
from chibi.settings.shop_window import ShopBackend, ShopWindow
from chibi.settings.widgets import is_alive
from chibi.tamagotchi.gifts import LineComposer

log = logging.getLogger(__name__)

START_KEY = "game.start"
INVITE_KEY = "game.invite"

GameWindow = RpsWindow | CookieWindow | GuessWindow

_FACTORIES: dict[str, type[RpsWindow] | type[CookieWindow] | type[GuessWindow]] = {
    "rps": RpsWindow,
    "cookies": CookieWindow,
    "guess": GuessWindow,
}


class GamesHub(QObject):
    def __init__(self, rewards: GameRewards, shop: ShopBackend, composer: LineComposer, *,
                 rng: random.Random, name: Callable[[], str] = lambda: "Мія",
                 portrait: Callable[[str], QPixmap | None] | None = None,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rewards = rewards
        self._shop_backend = shop
        self._composer = composer
        self._rng = rng
        self._name = name
        self._portrait = portrait
        self._games: dict[str, GameWindow] = {}
        self._shop: ShopWindow | None = None
        self.last_reward: Reward | None = None

    # -- windows ----------------------------------------------------------------------------

    def open_game(self, game_id: str) -> GameWindow:
        """Show (creating if needed) the window of ``game_id`` ("rps", "cookies", "guess")."""
        if game_id not in GAME_IDS:
            raise ValueError(f"unknown game {game_id!r}")
        window = self._games.get(game_id)
        if window is None or not is_alive(window):
            window = _FACTORIES[game_id](self._rng, name=self._her_name())
            window.finished.connect(lambda result, w=window: self._on_finished(w, result))
            self._games[game_id] = window
        else:
            window.set_name(self._her_name())
        if not window.isVisible():
            self._say(START_KEY)
        _raise(window)
        return window

    def open_shop(self) -> ShopWindow:
        if self._shop is None or not is_alive(self._shop):
            self._shop = ShopWindow(self._shop_backend, portrait=self._portrait, name=self._her_name())
        _raise(self._shop)
        return self._shop

    def window(self, game_id: str) -> GameWindow | None:
        window = self._games.get(game_id)
        return window if window is not None and is_alive(window) else None

    @property
    def shop_window(self) -> ShopWindow | None:
        return self._shop if self._shop is not None and is_alive(self._shop) else None

    def invite(self) -> bool:
        """Her "Пограймо?" (e.g. from her menu or an initiative); True if a line will be said."""
        return self._say(INVITE_KEY, solicited=False)

    def close_all(self) -> None:
        """Close and delete every window (shutdown, "delete all data")."""
        windows: list[QWidget] = [w for w in self._games.values() if is_alive(w)]
        if self._shop is not None and is_alive(self._shop):
            windows.append(self._shop)
        for window in windows:
            window.close()
            window.deleteLater()
        self._games.clear()
        self._shop = None

    # -- internals --------------------------------------------------------------------------

    def _on_finished(self, window: GameWindow, result: object) -> None:
        if not isinstance(result, GameResult):
            log.warning("a game window finished without a result")
            return
        reward = self._rewards.reward(result)
        self.last_reward = reward
        if is_alive(window):
            window.show_reward(reward.message)

    def _say(self, key: str, *, solicited: bool = True) -> bool:
        try:
            return bool(self._composer.compose(key, solicited=solicited))
        except Exception:
            log.exception("composing %s failed", key)
            return False

    def _her_name(self) -> str:
        try:
            return self._name() or "Мія"
        except Exception:
            return "Мія"


def _raise(widget: QWidget) -> None:
    if widget.isMinimized():
        # show() keeps a minimised window minimised: restore it explicitly.
        state = widget.windowState() & ~Qt.WindowState.WindowMinimized
        widget.setWindowState(state | Qt.WindowState.WindowActive)
    widget.show()
    widget.raise_()
    widget.activateWindow()
