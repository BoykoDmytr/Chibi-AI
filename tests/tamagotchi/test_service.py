import logging
import random
import sqlite3
from datetime import datetime

import pytest

from chibi.core.events import (
    ActivityCategory,
    AppInfo,
    AppStarted,
    CharacterInteraction,
    Event,
    ForegroundAppChanged,
    InteractionKind,
    InteractionOutcome,
    NeedsChanged,
)
from chibi.memory.repositories import StateRepository
from chibi.tamagotchi.needs import Needs, NeedsModel
from chibi.tamagotchi.service import TamagotchiService

HOUR = 3600.0
DAY = 24 * HOUR

approx = pytest.approx


class CountingRepo(StateRepository):
    """Real repository that also records which keys were written."""

    def __init__(self, db, clock=None):
        super().__init__(db, clock)
        self.writes: list[str] = []

    def set(self, key, value):
        self.writes.append(key)
        super().set(key, value)


class BrokenRepo(StateRepository):
    def set(self, key, value):
        raise sqlite3.OperationalError("disk I/O error")


class Recorder:
    def __init__(self, bus):
        self.bus = bus
        self.events: list[Event] = []
        bus.subscribe(Event, self.events.append)

    def drain(self) -> list[Event]:
        self.bus.dispatch_pending()
        out, self.events[:] = list(self.events), []
        return out


@pytest.fixture
def repo(db, clock):
    return CountingRepo(db, clock)


@pytest.fixture
def rec(bus):
    return Recorder(bus)


@pytest.fixture
def service(bus, repo, clock):
    svc = TamagotchiService(bus, repo, clock)
    svc.load()
    bus.dispatch_pending()
    repo.writes.clear()
    return svc


def kinds(events):
    return [type(e) for e in events]


def tick_for(service, clock, seconds, step=30.0):
    elapsed = 0.0
    while elapsed < seconds:
        clock.advance(step)
        service.tick()
        elapsed += step


def local_noon(year=2025, month=3, day=10):
    return datetime(year, month, day, 12, 0).timestamp()


def foreground():
    return ForegroundAppChanged(app=AppInfo("code.exe", "VS Code"), category=ActivityCategory.PROGRAMMING,
                                confidence=0.9)


# --- load / save ----------------------------------------------------------------------------


def test_fresh_load_uses_defaults_and_publishes_snapshot(bus, repo, clock, rec):
    svc = TamagotchiService(bus, repo, clock)
    svc.load()
    assert svc.needs == Needs()
    assert svc.sleeping is False
    events = rec.drain()
    assert kinds(events) == [NeedsChanged]
    assert events[0].hunger == 30.0
    assert events[0].friendship_level == 1


def test_persistence_round_trip(bus, db, clock):
    first = TamagotchiService(bus, StateRepository(db, clock), clock)
    first.load()
    first.needs = Needs(hunger=12, energy=44, happiness=91, boredom=7, affection=66, friendship_xp=123)
    first.save()
    second = TamagotchiService(bus, StateRepository(db, clock), clock)
    second.load()  # no time passed: no catch-up
    assert second.needs == first.needs


def test_save_writes_needs_and_last_seen(service, repo, clock):
    clock.advance(5)
    service.save()
    assert repo.get("needs") == service.needs.to_dict()
    assert repo.get("last_seen") == clock.now()


def test_load_catches_up_on_time_away(bus, db, clock):
    model = NeedsModel()
    first = TamagotchiService(bus, StateRepository(db, clock), clock)
    first.load()
    first.needs = Needs(energy=10, hunger=40, boredom=30, happiness=40)
    saved = Needs(energy=10, hunger=40, boredom=30, happiness=40)
    first.save()
    clock.advance(8 * HOUR)
    second = TamagotchiService(bus, StateRepository(db, clock), clock)
    second.load()
    assert second.needs == model.offline_catch_up(saved, 8 * HOUR)
    assert second.needs.energy == 100.0
    assert second.needs.hunger == approx(55.0)


def test_week_away_never_makes_her_starving_sad_or_bored(bus, db, clock):
    first = TamagotchiService(bus, StateRepository(db, clock), clock)
    first.load()
    first.needs = Needs(hunger=50, energy=5, happiness=55, boredom=45)
    first.save()
    clock.advance(7 * DAY)
    second = TamagotchiService(bus, StateRepository(db, clock), clock)
    second.load()
    n = second.needs
    assert n.hunger <= 55.0
    assert n.boredom <= 50.0
    assert n.happiness >= 50.0
    assert n.energy == 100.0
    assert second.mood not in ("Сонна", "Голодна", "Нудьгує", "Сумна")


def test_load_with_last_seen_in_the_future_does_not_catch_up(bus, repo, clock):
    repo.set("needs", Needs(energy=10).to_dict())
    repo.set("last_seen", clock.now() + DAY)
    svc = TamagotchiService(bus, repo, clock)
    svc.load()
    assert svc.needs == Needs(energy=10)


@pytest.mark.parametrize("last_seen", ["yesterday", True, None, [1], {"t": 1}, float("nan"), float("inf")])
def test_load_ignores_corrupt_last_seen(bus, repo, clock, last_seen):
    repo.set("needs", Needs(energy=10).to_dict())
    repo.set("last_seen", last_seen)
    svc = TamagotchiService(bus, repo, clock)
    svc.load()
    assert svc.needs == Needs(energy=10)


@pytest.mark.parametrize(
    "stored",
    ["garbage", 7, [1, 2], {"hunger": "lots", "energy": 900, "happiness": None, "friendship_xp": -5}],
)
def test_load_tolerates_corrupt_needs(bus, repo, clock, stored):
    repo.set("needs", stored)
    svc = TamagotchiService(bus, repo, clock)
    svc.load()
    n = svc.needs
    for name in ("hunger", "energy", "happiness", "boredom", "affection"):
        assert 0.0 <= getattr(n, name) <= 100.0
    assert n.friendship_xp >= 0.0
    if isinstance(stored, dict):
        assert (n.hunger, n.energy, n.happiness, n.friendship_xp) == (30.0, 100.0, 70.0, 0.0)
    else:
        assert n == Needs()


def test_load_tolerates_unparseable_json(bus, db, clock):
    with db.transaction() as conn:
        conn.execute("INSERT INTO kv_state(key, value, updated_at) VALUES('needs', '{not json', 0)")
        conn.execute("INSERT INTO kv_state(key, value, updated_at) VALUES('last_seen', 'NaNa', 0)")
    svc = TamagotchiService(bus, StateRepository(db, clock), clock)
    svc.load()
    assert svc.needs == Needs()


def test_save_failure_is_logged_not_raised(bus, db, clock, caplog):
    svc = TamagotchiService(bus, BrokenRepo(db, clock), clock)
    svc.load()
    with caplog.at_level(logging.ERROR, logger="chibi.tamagotchi.service"):
        svc.save()
        clock.advance(61)
        svc.tick()  # periodic save fails too, without breaking the tick
    assert "could not save" in caplog.text


# --- tick -----------------------------------------------------------------------------------


def test_tick_advances_by_monotonic_time(service, clock):
    clock.advance(30)
    service.tick()
    assert service.needs.hunger == approx(30 + 7 * 30 / 3600)
    assert service.needs.energy == approx(80 - 6 * 30 / 3600)
    assert service.needs.boredom == approx(20 + 12 * 30 / 3600)


def test_tick_without_time_passing_changes_nothing(service):
    before = Needs(**service.needs.to_dict())
    service.tick()
    service.tick()
    assert service.needs == before


def test_one_hour_of_ticks_matches_the_model(service, clock):
    tick_for(service, clock, HOUR)
    expected = NeedsModel().tick(Needs(), HOUR, sleeping=False, user_active=False)
    for name in ("hunger", "energy", "happiness", "boredom", "affection"):
        assert getattr(service.needs, name) == approx(getattr(expected, name), abs=1e-6)


def test_needs_changed_only_after_a_one_point_move(service, clock, rec):
    clock.advance(30)
    service.tick()
    assert rec.drain() == []
    # boredom rises 12/h while she is alone: one point takes 300 s
    tick_for(service, clock, 240)
    assert rec.drain() == []
    tick_for(service, clock, 30)
    events = rec.drain()
    assert kinds(events) == [NeedsChanged]
    assert events[0].boredom == approx(21.0)
    tick_for(service, clock, 30)
    assert rec.drain() == []


def test_saves_at_most_every_sixty_seconds(service, repo, clock):
    clock.advance(30)
    service.tick()
    assert repo.writes == []
    clock.advance(30)
    service.tick()
    assert repo.writes == ["needs", "last_seen"]
    clock.advance(30)
    service.tick()
    assert repo.writes == ["needs", "last_seen"]
    clock.advance(30)
    service.tick()
    assert repo.writes == ["needs", "last_seen"] * 2
    assert repo.get("last_seen") == clock.now()


def test_laptop_sleep_wall_clock_jump_uses_offline_catch_up(service, clock, repo):
    service.needs = Needs(energy=10, hunger=40, boredom=30)
    start = Needs(energy=10, hunger=40, boredom=30)
    # Linux / macOS: the monotonic clock stops during suspend, the wall clock doesn't.
    clock.set_now(clock.now() + 8 * HOUR)
    service.tick()
    assert service.needs == NeedsModel().offline_catch_up(start, 8 * HOUR)
    assert service.needs.energy == 100.0
    assert service.needs.hunger == approx(55.0)
    assert repo.writes == ["needs", "last_seen"]  # saved at once


def test_laptop_sleep_with_both_clocks_advancing_is_not_awake_decay(service, clock):
    # Windows: the monotonic clock may include the time spent asleep.
    clock.advance(8 * HOUR)
    service.tick()
    assert service.needs == NeedsModel().offline_catch_up(Needs(), 8 * HOUR)
    assert service.needs.hunger == approx(46.0)  # 30 + 2/h, not 30 + 7/h
    assert service.needs.energy == 100.0


def test_gap_just_under_the_jump_threshold_is_a_normal_tick(service, clock):
    clock.advance(600)
    service.tick()
    assert service.needs.energy == approx(80 - 6 * 600 / 3600)


def test_wall_clock_set_backwards_still_ticks_normally(service, clock):
    clock.set_now(clock.now() - DAY)
    clock.advance(30)
    service.tick()
    assert service.needs.hunger == approx(30 + 7 * 30 / 3600)


def test_sleeping_recovers_energy(service, clock):
    service.needs = Needs(energy=20)
    service.set_sleeping(True)
    assert service.sleeping is True
    tick_for(service, clock, HOUR)
    assert service.needs.energy == approx(45.0)
    assert service.needs.boredom == approx(10.0)


def test_set_sleeping_settles_elapsed_time_under_the_old_state(service, clock):
    clock.advance(300)
    service.set_sleeping(True)  # the last 5 min were awake
    assert service.needs.energy == approx(80 - 6 * 300 / 3600)
    clock.advance(300)
    service.set_sleeping(False)  # the next 5 min were asleep
    assert service.needs.energy == approx(80 - 6 * 300 / 3600 + 25 * 300 / 3600)


def test_sleeping_property_setter(service):
    service.sleeping = True
    assert service.sleeping is True
    service.sleeping = False
    assert service.sleeping is False


# --- user activity -------------------------------------------------------------------------


def test_foreground_change_marks_user_active(service, bus, clock):
    assert service.user_active is False
    bus.publish(foreground())
    bus.dispatch_pending()
    assert service.user_active is True
    tick_for(service, clock, 300)
    assert service.needs.boredom == approx(20 + 4 * 300 / 3600)


def test_user_activity_expires_after_ten_minutes(service, bus, clock):
    bus.publish(foreground())
    bus.dispatch_pending()
    tick_for(service, clock, 600)
    assert service.user_active is False
    boredom = service.needs.boredom
    tick_for(service, clock, 300)
    assert service.needs.boredom == approx(boredom + 12 * 300 / 3600)


def test_interaction_marks_user_active(service):
    service.interact(InteractionKind.DRAG_START)
    assert service.user_active is True


# --- interactions -----------------------------------------------------------------------------


def test_interact_publishes_outcome_then_needs(service, rec):
    service.needs = Needs(hunger=50)
    assert service.interact(InteractionKind.FEED) == "fed"
    events = rec.drain()
    assert kinds(events) == [InteractionOutcome, NeedsChanged]
    assert (events[0].kind, events[0].outcome) == (InteractionKind.FEED, "fed")
    assert events[1].hunger == approx(15.0)
    assert service.needs.hunger == approx(15.0)


def test_interact_without_announce_publishes_no_outcome(service, rec):
    service.needs = Needs(hunger=50)
    assert service.interact(InteractionKind.FEED, announce=False) == "fed"
    assert kinds(rec.drain()) == [NeedsChanged]
    assert service.needs.hunger == approx(15.0)


def test_interact_accepts_plain_strings(service):
    assert service.interact("pet", announce=False) == "happy"
    with pytest.raises(ValueError):
        service.interact("tickle")


def test_character_interaction_event_is_handled(service, bus, rec):
    service.needs = Needs(hunger=90)
    bus.publish(CharacterInteraction(kind=InteractionKind.FEED))
    events = rec.drain()
    outcomes = [e for e in events if isinstance(e, InteractionOutcome)]
    assert [(o.kind, o.outcome) for o in outcomes] == [(InteractionKind.FEED, "starving_fed")]
    assert NeedsChanged in kinds(events)
    assert service.needs.hunger == approx(45.0)


def test_pet_window_limits_to_six_per_ten_minutes(service, clock):
    outcomes = []
    for _ in range(8):
        outcomes.append(service.interact(InteractionKind.PET, announce=False))
        clock.advance(10)
    assert outcomes == ["happy"] * 6 + ["too_much"] * 2
    affection, xp = service.needs.affection, service.needs.friendship_xp
    assert service.interact(InteractionKind.PET) == "too_much"
    # too much changes nothing (beyond 10 s of normal drift)
    assert service.needs.affection == approx(affection, abs=0.01)
    assert service.needs.friendship_xp == xp
    clock.advance(600)
    assert service.interact(InteractionKind.PET) == "happy"


def test_click_window_limits_to_five_per_five_seconds(service, clock):
    outcomes = []
    for _ in range(7):
        outcomes.append(service.interact(InteractionKind.CLICK, announce=False))
        clock.advance(0.2)
    assert outcomes == ["poke"] * 5 + ["annoyed"] * 2
    clock.advance(5)
    assert service.interact(InteractionKind.CLICK) == "poke"


def test_click_spam_window_stays_bounded(service, clock):
    for _ in range(1000):
        service.interact(InteractionKind.CLICK, announce=False)
    assert len(service._clicks) <= 32
    assert service.interact(InteractionKind.CLICK) == "annoyed"


def test_rest_sleep_and_wake(service, rec):
    service.needs = Needs(energy=50)
    assert service.interact(InteractionKind.REST) == "sleep"
    assert service.sleeping is True
    assert service.interact(InteractionKind.WAKE) == "ok"
    assert service.sleeping is False
    service.needs = Needs(energy=20)
    service.set_sleeping(True)
    assert service.interact(InteractionKind.WAKE) == "grumpy"
    assert service.sleeping is False
    outcomes = [(e.kind, e.outcome) for e in rec.drain() if isinstance(e, InteractionOutcome)]
    assert outcomes == [(InteractionKind.REST, "sleep"), (InteractionKind.WAKE, "ok"),
                        (InteractionKind.WAKE, "grumpy")]


def test_rest_when_not_tired_keeps_her_awake(service):
    service.needs = Needs(energy=95)
    assert service.interact(InteractionKind.REST) == "not_tired"
    assert service.sleeping is False


def test_interaction_settles_elapsed_time_first(service, clock, rec):
    clock.advance(300)
    service.interact(InteractionKind.CALL)
    assert service.needs.hunger == approx(30 + 7 * 300 / 3600)
    assert service.needs.boredom == approx(20 + 12 * 300 / 3600 - 2)
    assert kinds(rec.drain()) == [InteractionOutcome, NeedsChanged]


def test_interaction_right_after_laptop_sleep_sees_caught_up_needs(service, clock, repo, rec):
    service.needs = Needs(energy=10)
    clock.set_now(clock.now() + 8 * HOUR)
    assert service.interact(InteractionKind.REST) == "not_tired"  # she slept through the suspend
    assert service.needs.energy == 100.0
    assert repo.writes == ["needs", "last_seen"]
    assert kinds(rec.drain()) == [InteractionOutcome, NeedsChanged]


def test_interactions_are_not_saved_more_often_than_ticks(service, repo, clock):
    for _ in range(20):
        service.interact(InteractionKind.CLICK, announce=False)
        clock.advance(1)
    assert repo.writes == []


# --- friendship ---------------------------------------------------------------------------------


def test_daily_start_bonus_once_per_calendar_day(bus, db, clock, rec):
    clock.set_now(local_noon())
    repo = StateRepository(db, clock)
    svc = TamagotchiService(bus, repo, clock)
    svc.load()
    bus.publish(AppStarted())
    bus.dispatch_pending()
    assert svc.needs.friendship_xp == 5.0
    assert repo.get("needs")["friendship_xp"] == 5.0  # saved at once
    # restart later the same day: no second bonus
    svc.close()
    clock.advance(3 * HOUR)
    again = TamagotchiService(bus, repo, clock)
    again.load()
    bus.publish(AppStarted())
    bus.dispatch_pending()
    assert again.needs.friendship_xp == 5.0
    # next day: bonus again
    again.close()
    clock.advance(DAY)
    third = TamagotchiService(bus, repo, clock)
    third.load()
    bus.publish(AppStarted(away_seconds=DAY))
    bus.dispatch_pending()
    assert third.needs.friendship_xp == 10.0


def test_daily_bonus_before_load_is_applied_after_load(bus, repo, clock):
    repo.set("needs", Needs(friendship_xp=8).to_dict())
    svc = TamagotchiService(bus, repo, clock)
    bus.publish(AppStarted(first_run=False))
    bus.dispatch_pending()
    svc.load()
    assert svc.needs.friendship_xp == 13.0


def test_level_up_is_published(service, rec):
    service.needs = Needs(friendship_xp=9)
    service._last_published = Needs(friendship_xp=9)
    service.interact(InteractionKind.TALK, announce=False)
    events = rec.drain()
    assert events[-1].friendship_level == 2


def test_friendship_never_decreases_in_the_service(service, clock, bus):
    rng = random.Random(42)
    xp = service.needs.friendship_xp
    for _ in range(600):
        roll = rng.random()
        if roll < 0.5:
            service.interact(rng.choice(list(InteractionKind)), announce=rng.random() < 0.5)
        elif roll < 0.9:
            clock.advance(rng.uniform(0, 120))
            service.tick()
        elif roll < 0.95:
            clock.set_now(clock.now() + rng.uniform(0, 2 * DAY))  # suspend
            service.tick()
        else:
            service.set_sleeping(rng.random() < 0.5)
        bus.dispatch_pending()
        assert service.needs.friendship_xp >= xp
        xp = service.needs.friendship_xp
    assert service.needs.friendship_level >= 2


# --- lifecycle ---------------------------------------------------------------------------------


def test_close_unsubscribes(service, bus, rec):
    service.close()
    service.needs = Needs(hunger=50)
    bus.publish(CharacterInteraction(kind=InteractionKind.FEED))
    events = rec.drain()
    assert not any(isinstance(e, InteractionOutcome) for e in events)
    assert service.needs.hunger == 50.0


def test_custom_model_is_used(bus, repo, clock):
    class Generous(NeedsModel):
        def feed(self, needs):
            return Needs(hunger=0, friendship_xp=needs.friendship_xp + 100), "fed"

    svc = TamagotchiService(bus, repo, clock, model=Generous())
    svc.load()
    assert svc.interact(InteractionKind.FEED) == "fed"
    assert svc.needs.friendship_xp == 100.0
    assert svc.model.__class__ is Generous


def test_mood_reflects_needs(service):
    service.needs = Needs(hunger=90)
    assert service.mood == "Голодна"
