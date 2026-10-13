"""GameRewards: stars within the daily cap, needs effects, her reaction - and
kindness once the cap is reached."""

import logging

import pytest

from chibi.core.events import NeedsChanged
from chibi.games import GameResult, frozen_effects
from chibi.games.rewards import MSG_CAPPED_NONE, GameRewards, reward_message
from chibi.tamagotchi.gifts import DAILY_EARN_CAP, Wallet
from tests.games.fakes import World

DAY = 24 * 3600.0
EFFECTS = frozen_effects(boredom=-20, happiness=8, friendship_xp=3)


def result(stars=8, verdict="win", game="rps"):
    return GameResult(game=game, score=2, won=verdict == "win", stars=stars, verdict=verdict, effects=EFFECTS)


@pytest.fixture
def world(db, bus, clock):
    return World(db, bus, clock)


@pytest.fixture
def wallet(world):
    w = Wallet(world.repo, world.clock)
    w.load()
    return w


@pytest.fixture
def rewards(world, wallet):
    return GameRewards(wallet, world.tamagotchi, world.composer)


def test_pays_stars_applies_effects_and_she_reacts(rewards, world, wallet):
    world.tamagotchi.needs.boredom = 60
    world.tamagotchi.needs.happiness = 50
    reward = rewards.reward(result())
    assert (reward.stars, reward.offered, reward.capped) == (8, 8, False)
    assert reward.message == "Нагорода: +8 ⭐"
    assert wallet.stars == 8
    assert world.tamagotchi.needs.boredom == pytest.approx(40)
    assert world.tamagotchi.needs.happiness == pytest.approx(58)
    assert world.composer.calls == [(("game.rps.win",), True)]
    assert world.composer.variables == [{"minigame": "Камінь-ножиці-папір"}]
    assert world.recorder.of(NeedsChanged)


def test_phrase_key_per_game_and_verdict(rewards, world):
    rewards.reward(result(verdict="great", game="cookies"))
    rewards.reward(result(verdict="lose", game="guess"))
    assert world.composer.keys == [("game.cookies.great",), ("game.guess.lose",)]


def test_daily_cap_is_kind(rewards, world, wallet):
    for _ in range(7):
        rewards.reward(result(stars=8))
    assert wallet.stars == 56
    partial = rewards.reward(result(stars=8))
    assert (partial.stars, partial.capped) == (4, True)
    assert "це всі зірочки з ігор на сьогодні" in partial.message
    assert "гратися можна й далі" in partial.message
    world.tamagotchi.needs.boredom = 80
    none = rewards.reward(result(stars=8))
    assert (none.stars, none.capped, none.message) == (0, True, MSG_CAPPED_NONE)
    # Still fun: the needs effect and her reaction happen anyway.
    assert world.tamagotchi.needs.boredom == pytest.approx(60)
    assert len(world.composer.calls) == 9
    assert wallet.stars == DAILY_EARN_CAP


def test_cap_resets_the_next_day(rewards, wallet, clock):
    wallet.earn(DAILY_EARN_CAP)
    clock.advance(DAY)
    assert rewards.reward(result(stars=8)).stars == 8


def test_failures_are_isolated(rewards, world, wallet, monkeypatch, caplog):
    def boom(*_a, **_k):
        raise RuntimeError("no")

    monkeypatch.setattr(wallet, "earn", boom)
    monkeypatch.setattr(world.tamagotchi, "apply_effect", boom)
    world.composer.error = RuntimeError("no")
    with caplog.at_level(logging.ERROR):
        reward = rewards.reward(result(stars=8))
    assert reward.stars == 0
    assert "crediting stars failed" in caplog.text
    assert "applying its effect failed" in caplog.text
    assert "composing her reaction failed" in caplog.text


def test_reward_messages():
    assert reward_message(5, 5) == "Нагорода: +5 ⭐"
    assert reward_message(0, 0) == ""
    assert reward_message(0, 5) == MSG_CAPPED_NONE
    assert reward_message(2, 5).startswith("Нагорода: +2 ⭐")


def test_negative_stars_are_not_debts(rewards, wallet):
    wallet.earn(10)
    reward = rewards.reward(result(stars=-5))
    assert reward.stars == 0 and reward.offered == 0
    assert wallet.stars == 10
