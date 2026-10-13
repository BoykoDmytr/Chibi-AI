"""GamesHub: lazily created windows, rewards shown after a game, her lines."""

import random

import pytest
from PySide6.QtCore import QCoreApplication, QEvent

from chibi.games.cookie_window import CookieWindow
from chibi.games.guess_window import GuessWindow
from chibi.games.hub import GamesHub
from chibi.games.rewards import GameRewards
from chibi.games.rps import Move
from chibi.games.rps_window import RpsWindow
from chibi.settings.shop_window import ShopWindow
from chibi.settings.widgets import is_alive
from tests.games.fakes import World


@pytest.fixture
def world(db, bus, clock):
    return World(db, bus, clock)


@pytest.fixture
def hub(world, qapp):
    gifts = world.gifts()
    rewards = GameRewards(gifts.wallet, world.tamagotchi, world.composer)
    hub = GamesHub(rewards, gifts, world.composer, rng=random.Random(0),
                   name=lambda: world.settings.character.name)
    hub.gifts = gifts
    yield hub
    hub.close_all()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_opens_each_game_once_and_greets(hub, world):
    rps = hub.open_game("rps")
    assert isinstance(rps, RpsWindow) and rps.isVisible()
    assert world.composer.calls == [(("game.start",), True)]
    assert hub.open_game("rps") is rps  # already open: raised, no second line
    assert len(world.composer.calls) == 1
    assert isinstance(hub.open_game("cookies"), CookieWindow)
    assert isinstance(hub.open_game("guess"), GuessWindow)
    assert hub.window("guess") is not None


def test_reopening_a_closed_game_greets_again(hub, world):
    window = hub.open_game("guess")
    window.close()
    assert hub.open_game("guess") is window
    assert world.composer.keys == [("game.start",), ("game.start",)]


def test_unknown_game(hub):
    with pytest.raises(ValueError):
        hub.open_game("chess")


def test_finished_game_pays_and_shows_the_reward(hub, world):
    window = hub.open_game("rps")
    while not window.game.finished:
        window.choose(Move.ROCK)
    reward = hub.last_reward
    assert reward is not None and reward.stars == hub.gifts.wallet.stars > 0
    assert window.reward_label.text() == reward.message
    assert world.composer.keys[-1][0].startswith("game.rps.")


def test_her_name_follows_the_settings(hub, world):
    world.settings.character.name = "Ліна"
    window = hub.open_game("rps")
    assert window.status_label.text().startswith("Ліна")
    window.close()
    world.settings.character.name = "Мія"
    hub.open_game("rps")
    assert window.status_label.text().startswith("Мія")


def test_shop(hub):
    shop = hub.open_shop()
    assert isinstance(shop, ShopWindow) and shop.isVisible()
    assert hub.open_shop() is shop
    assert hub.shop_window is shop


def test_invite_is_unsolicited(hub, world):
    assert hub.invite()
    assert world.composer.calls == [(("game.invite",), False)]


def test_composer_failure_is_not_fatal(hub, world):
    world.composer.error = RuntimeError("no")
    assert not hub.invite()
    assert hub.open_game("guess").isVisible()


def test_close_all(hub):
    windows = [hub.open_game(g) for g in ("rps", "cookies", "guess")] + [hub.open_shop()]
    hub.close_all()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not any(is_alive(w) for w in windows)
    assert hub.window("rps") is None and hub.shop_window is None
    assert hub.open_game("rps").isVisible()  # a fresh one after that
