"""TamagotchiService.apply_effect: gifts and mini-games change her needs."""

import logging
import math

import pytest

from chibi.core.events import NeedsChanged
from chibi.memory.repositories import StateRepository
from chibi.tamagotchi.needs import FRIENDSHIP_THRESHOLDS, Needs, NeedsModel
from chibi.tamagotchi.service import (
    CLOCK_JUMP_SECONDS,
    EFFECT_FIELDS,
    KEY_LAST_SEEN,
    KEY_NEEDS,
    TamagotchiService,
)
from tests.games.fakes import FlakyRepo, Recorder

approx = pytest.approx


@pytest.fixture
def repo(db, clock):
    return FlakyRepo(db, clock)


@pytest.fixture
def rec(bus):
    return Recorder(bus)


@pytest.fixture
def service(bus, repo, clock, rec):
    svc = TamagotchiService(bus, repo, clock)
    svc.load()
    svc.needs = Needs(hunger=40, energy=60, happiness=50, boredom=50, affection=50, friendship_xp=5)
    rec.drain()
    repo.writes.clear()
    return svc


def test_effect_fields_are_the_needs_and_xp():
    assert {"hunger", "energy", "happiness", "boredom", "affection", "friendship_xp"} == EFFECT_FIELDS


def test_deltas_are_added(service):
    result = service.apply_effect({"hunger": -10, "happiness": 4, "boredom": -15, "energy": 3,
                                   "affection": 2, "friendship_xp": 1.5}, reason="gift.cookie")
    assert result == Needs(hunger=30, energy=63, happiness=54, boredom=35, affection=52, friendship_xp=6.5)
    assert service.needs == result


def test_results_are_clamped(service):
    service.apply_effect({"hunger": -500, "happiness": 500, "boredom": -80, "energy": 1e9}, reason="t")
    n = service.needs
    assert (n.hunger, n.happiness, n.boredom, n.energy) == (0.0, 100.0, 0.0, 100.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "5", None, True, [1]])
def test_invalid_values_are_ignored(service, value, caplog):
    before = Needs(**service.needs.to_dict())
    with caplog.at_level(logging.WARNING):
        service.apply_effect({"happiness": value, "energy": 2}, reason="gift.x")
    assert service.needs.happiness == before.happiness
    assert service.needs.energy == before.energy + 2
    assert "ignoring invalid delta" in caplog.text


def test_unknown_needs_are_ignored(service, caplog):
    with caplog.at_level(logging.WARNING):
        service.apply_effect({"mood": 10, "friendship_level": 3, "hunger": -1}, reason="t")
    assert service.needs.hunger == 39
    assert caplog.text.count("ignoring invalid delta") == 2


def test_friendship_never_decreases(service, caplog):
    with caplog.at_level(logging.WARNING):
        service.apply_effect({"friendship_xp": -5}, reason="t")
    assert service.needs.friendship_xp == 5
    assert "friendship never decreases" in caplog.text


def test_publishes_needs_changed_even_for_a_tiny_or_empty_effect(service, rec):
    service.apply_effect({"happiness": 0.1}, reason="t")
    events = rec.of(NeedsChanged)
    assert len(events) == 1 and events[0].happiness == approx(50.1)
    service.apply_effect({}, reason="t")
    assert len(rec.of(NeedsChanged)) == 1


def test_level_up_is_published(service, rec):
    service.needs.friendship_xp = FRIENDSHIP_THRESHOLDS[2] - 1  # level 2, one XP short of 3
    service.apply_effect({"friendship_xp": 3}, reason="game.rps")
    (event,) = rec.of(NeedsChanged)
    assert event.friendship_level == 3


def test_saved_at_once_and_restored(service, bus, db, clock):
    service.apply_effect({"hunger": -20}, reason="gift.cake")
    repo = StateRepository(db, clock)
    assert repo.get(KEY_NEEDS)["hunger"] == 20
    assert repo.get(KEY_LAST_SEEN) == clock.now()
    again = TamagotchiService(bus, repo, clock)
    again.load()
    assert again.needs.hunger == 20


def test_save_failure_keeps_the_effect_in_memory(service, repo, rec):
    repo.broken = {KEY_NEEDS, KEY_LAST_SEEN}
    service.apply_effect({"happiness": 10}, reason="t")
    assert service.needs.happiness == 60
    assert rec.of(NeedsChanged)


def test_elapsed_time_is_settled_first(service, clock):
    start = Needs(**service.needs.to_dict())
    clock.advance(CLOCK_JUMP_SECONDS / 2)  # a normal gap between ticks, not a freeze
    service.apply_effect({"hunger": -10}, reason="t")
    ticked = NeedsModel().tick(start, CLOCK_JUMP_SECONDS / 2, sleeping=False, user_active=False)
    assert service.needs.hunger == approx(ticked.hunger - 10)
    assert service.needs.energy == approx(ticked.energy)


def test_a_freeze_is_caught_up_offline_and_saved(service, clock, repo):
    start = Needs(**service.needs.to_dict())
    clock.advance(CLOCK_JUMP_SECONDS * 6)
    service.apply_effect({"boredom": -5}, reason="t")
    caught_up = NeedsModel().offline_catch_up(start, CLOCK_JUMP_SECONDS * 6)
    assert service.needs.boredom == approx(caught_up.boredom - 5)
    assert service.needs.energy == approx(caught_up.energy)
    assert repo.writes.count(KEY_NEEDS) >= 2  # the catch-up, then the effect


def test_counts_as_the_user_being_around(service):
    assert not service.user_active
    service.apply_effect({"happiness": 1}, reason="t")
    assert service.user_active


def test_returns_a_copy(service):
    result = service.apply_effect({"happiness": 1}, reason="t")
    result.happiness = 0
    assert service.needs.happiness == 51
