"""EmotionEngine: every rule, decay maths, persistence, publishing, no punishment."""

import json
import logging
import math
import sqlite3

import pytest

from chibi.core.config import Settings
from chibi.core.events import (
    ActivityCategory,
    ActivityMilestone,
    AppInfo,
    EmotionChanged,
    Event,
    ForegroundAppChanged,
    GameEnded,
    GameStarted,
    InteractionKind,
    InteractionOutcome,
    NeedsChanged,
    ObservationStopped,
    UserReturned,
)
from chibi.emotion import engine as engine_module
from chibi.emotion.engine import (
    _DESCRIPTORS,
    EMOTIONS,
    STATE_KEY,
    EmotionEngine,
    describe_values,
    dominant_state,
    relax,
)
from chibi.memory.repositories import StateRepository
from chibi.tamagotchi.needs import Needs

MINUTE = 60.0
HOUR = 3600.0
DAY = 24 * HOUR

approx = pytest.approx


class CountingRepo(StateRepository):
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

    def drain(self) -> list[EmotionChanged]:
        self.bus.dispatch_pending()
        out = [e for e in self.events if isinstance(e, EmotionChanged)]
        self.events.clear()
        return out


class NeedsBox:
    def __init__(self):
        self.needs = Needs()


@pytest.fixture
def box():
    return NeedsBox()


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def repo(db, clock):
    return CountingRepo(db, clock)


@pytest.fixture
def rec(bus):
    return Recorder(bus)


def make_engine(bus, repo, clock, box, settings) -> EmotionEngine:
    return EmotionEngine(bus, repo, clock, lambda: box.needs, lambda: settings)


@pytest.fixture
def engine(bus, repo, clock, box, settings, rec):
    emotion = make_engine(bus, repo, clock, box, settings)
    emotion.load()
    rec.drain()
    return emotion


def send(bus, event):
    bus.publish(event)
    bus.dispatch_pending()


def app_event(category, private=False):
    app = AppInfo.private_app() if private else AppInfo("app.exe", "App")
    return ForegroundAppChanged(app=app, category=category, confidence=0.9)


def outcome(kind, result):
    return InteractionOutcome(kind=kind, outcome=result)


def needs_event(needs: Needs) -> NeedsChanged:
    return NeedsChanged(hunger=needs.hunger, energy=needs.energy, happiness=needs.happiness,
                        boredom=needs.boredom, affection=needs.affection,
                        friendship_level=needs.friendship_level)


NEUTRAL = {
    "happiness": 60.0, "energy": 60.0, "boredom": 30.0, "affection": 50.0,
    "curiosity": 50.0, "excitement": 20.0, "hunger": 30.0, "trust": 40.0,
}


# -- values ------------------------------------------------------------------------------------


def test_defaults(bus, repo, clock, box, settings):
    settings.personality.curiosity = 75
    emotion = make_engine(bus, repo, clock, box, settings)
    emotion.load()
    values = emotion.values()
    assert list(values) == list(EMOTIONS)
    assert values["curiosity"] == 75.0
    assert values["excitement"] == 20.0
    assert values["trust"] == 40.0


def test_needs_values_come_from_the_tamagotchi(engine, box):
    box.needs = Needs(hunger=90, energy=15, happiness=25, boredom=85, affection=77)
    values = engine.values()
    assert (values["hunger"], values["energy"], values["happiness"], values["boredom"],
            values["affection"]) == (90, 15, 25, 85, 77)
    box.needs.hunger = 10  # no copy is kept: the live snapshot wins
    assert engine.values()["hunger"] == 10


def test_needs_may_be_a_mapping(bus, repo, clock, settings):
    snapshot = {"hunger": 50.0, "energy": 40.0, "happiness": 55.0, "boredom": 10.0, "affection": 65.0}
    emotion = EmotionEngine(bus, repo, clock, lambda: snapshot, lambda: settings)
    assert emotion.values()["affection"] == 65.0


def test_failing_needs_callable_uses_the_last_snapshot(bus, repo, clock, settings, caplog):
    state = {"fail": False, "needs": Needs(hunger=42)}

    def needs():
        if state["fail"]:
            raise RuntimeError("boom")
        return state["needs"]

    emotion = EmotionEngine(bus, repo, clock, needs, lambda: settings)
    assert emotion.values()["hunger"] == 42
    state["fail"] = True
    with caplog.at_level(logging.ERROR):
        assert emotion.values()["hunger"] == 42


def test_failing_settings_use_the_default_curiosity(bus, repo, clock, box):
    def settings():
        raise RuntimeError("no settings")

    emotion = EmotionEngine(bus, repo, clock, lambda: box.needs, settings)
    assert emotion.values()["curiosity"] == engine_module.DEFAULT_CURIOSITY


def test_values_load_lazily(bus, repo, clock, box, settings, rec):
    emotion = make_engine(bus, repo, clock, box, settings)
    assert emotion.values()["trust"] == 40.0
    assert len(rec.drain()) == 1  # the implicit load announced itself once


# -- event rules -------------------------------------------------------------------------------------


def test_game_started(engine, bus):
    send(bus, GameStarted(game_id="stalker2", title="S.T.A.L.K.E.R. 2", process_name="stalker2.exe"))
    values = engine.values()
    assert values["excitement"] == 45.0
    assert values["curiosity"] == 70.0


@pytest.mark.parametrize("duration, forgotten, delta", [
    (30 * MINUTE, False, 5.0),
    (3 * HOUR, False, 5.0),
    (29 * MINUTE, False, 0.0),
    (3 * HOUR, True, 0.0),
    (math.nan, False, 0.0),
])
def test_game_ended(engine, bus, duration, forgotten, delta):
    send(bus, GameEnded(game_id="g", title="" if forgotten else "G", duration_seconds=duration,
                        forgotten=forgotten))
    assert engine.values()["excitement"] == 20.0 + delta


@pytest.mark.parametrize("kind, result, name, delta", [
    (InteractionKind.PET, "happy", "trust", 0.5),
    (InteractionKind.PET, "too_much", "trust", 0.0),
    (InteractionKind.FEED, "fed", "trust", 0.3),
    (InteractionKind.FEED, "full", "trust", 0.0),
    (InteractionKind.CLICK, "annoyed", "excitement", -5.0),
    (InteractionKind.CLICK, "poke", "excitement", 0.0),
    (InteractionKind.TALK, "ok", "curiosity", 3.0),
    (InteractionKind.REST, "sleep", "trust", 0.0),
])
def test_interaction_outcomes(engine, bus, kind, result, name, delta):
    before = engine.values()
    send(bus, outcome(kind, result))
    after = engine.values()
    assert after[name] == approx(before[name] + delta)
    for other in ("trust", "excitement", "curiosity"):
        if other != name:
            assert after[other] == before[other]


def test_plain_string_kinds_work(engine, bus):
    send(bus, outcome("talk", "ok"))  # type: ignore[arg-type]
    assert engine.values()["curiosity"] == 63.0


def test_petting_trust_is_capped_per_day(engine, bus, clock):
    for _ in range(15):
        send(bus, outcome(InteractionKind.PET, "happy"))
    assert engine.values()["trust"] == approx(45.0)
    send(bus, outcome(InteractionKind.FEED, "fed"))  # feeding is not part of the petting cap
    assert engine.values()["trust"] == approx(45.3)
    clock.advance(DAY)
    send(bus, outcome(InteractionKind.PET, "happy"))
    assert engine.values()["trust"] == approx(45.8)


def test_petting_cap_survives_a_restart(engine, bus, repo, clock, box, settings):
    for _ in range(10):
        send(bus, outcome(InteractionKind.PET, "happy"))
    engine.save()
    engine.close()
    again = make_engine(bus, repo, clock, box, settings)
    again.load()
    send(bus, outcome(InteractionKind.PET, "happy"))
    assert again.values()["trust"] == approx(45.0)


def test_user_returned(engine, bus):
    send(bus, UserReturned(away_seconds=2 * HOUR))
    values = engine.values()
    assert values["excitement"] == 35.0
    assert values["trust"] == approx(40.2)


def test_new_activity_category_makes_her_curious(engine, bus, clock):
    send(bus, app_event(ActivityCategory.PROGRAMMING))
    assert engine.values()["curiosity"] == 64.0
    # another app of the same kind: nothing new
    send(bus, app_event(ActivityCategory.PROGRAMMING))
    assert engine.values()["curiosity"] == 64.0
    before = engine.values()["curiosity"]
    send(bus, app_event(ActivityCategory.BROWSING))
    assert engine.values()["curiosity"] == approx(before + 4.0)
    # back to programming 10 minutes later: seen recently
    clock.advance(10 * MINUTE)
    before = engine.values()["curiosity"]
    send(bus, app_event(ActivityCategory.PROGRAMMING))
    assert engine.values()["curiosity"] == approx(before)
    # 31 minutes in programming, then browsing again (last seen 31 min ago)
    clock.advance(31 * MINUTE)
    before = engine.values()["curiosity"]
    send(bus, app_event(ActivityCategory.BROWSING))
    assert engine.values()["curiosity"] == approx(before + 4.0)


def test_time_in_a_category_counts_as_seen_until_it_is_left(engine, bus, clock):
    send(bus, app_event(ActivityCategory.GAMING))
    clock.advance(2 * HOUR)  # gaming the whole time
    send(bus, app_event(ActivityCategory.WORKING))
    clock.advance(5 * MINUTE)
    before = engine.values()["curiosity"]
    send(bus, app_event(ActivityCategory.GAMING))  # left gaming only 5 minutes ago
    assert engine.values()["curiosity"] == approx(before)


@pytest.mark.parametrize("event", [
    app_event(ActivityCategory.PRIVATE, private=True),
    app_event(ActivityCategory.PROGRAMMING, private=True),
    app_event(ActivityCategory.IDLE),
    app_event(ActivityCategory.UNKNOWN),
])
def test_private_and_meaningless_apps_are_ignored(engine, bus, event):
    send(bus, event)
    assert engine.values()["curiosity"] == 60.0


def test_a_private_app_still_ends_the_previous_activity(engine, bus, clock):
    send(bus, app_event(ActivityCategory.PROGRAMMING))
    clock.advance(HOUR)
    send(bus, app_event(ActivityCategory.PRIVATE, private=True))
    clock.advance(31 * MINUTE)
    before = engine.values()["curiosity"]
    send(bus, app_event(ActivityCategory.PROGRAMMING))
    assert engine.values()["curiosity"] == approx(before + 4.0)


def test_kill_switch_forgets_activity_history(engine, bus):
    send(bus, app_event(ActivityCategory.PROGRAMMING))
    send(bus, ObservationStopped())
    before = engine.values()["curiosity"]
    send(bus, app_event(ActivityCategory.PROGRAMMING))
    assert engine.values()["curiosity"] == approx(before + 4.0)


def test_activity_milestone_is_routine(engine, bus):
    send(bus, ActivityMilestone(category=ActivityCategory.PROGRAMMING, app=AppInfo("code.exe", "VS Code"),
                                seconds=2 * HOUR))
    assert engine.values()["curiosity"] == 55.0
    send(bus, ActivityMilestone(category=ActivityCategory.PRIVATE, app=AppInfo.private_app(), seconds=HOUR))
    assert engine.values()["curiosity"] == 55.0


def test_values_are_clamped(engine, bus):
    for _ in range(10):
        send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))
    values = engine.values()
    assert values["excitement"] == 100.0
    assert values["curiosity"] == 100.0
    for _ in range(40):
        send(bus, outcome(InteractionKind.CLICK, "annoyed"))
    assert engine.values()["excitement"] == 0.0
    for _ in range(40):
        send(bus, ActivityMilestone(category=ActivityCategory.WORKING, app=AppInfo("x.exe", "X"), seconds=1))
    assert engine.values()["curiosity"] == 0.0


def test_close_stops_listening(engine, bus):
    engine.close()
    send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))
    assert engine.values()["excitement"] == 20.0


# -- decay -------------------------------------------------------------------------------------------


def test_relax_maths():
    assert relax(45.0, 20.0, 300.0, 300.0) == 32.5
    assert relax(45.0, 20.0, 300.0, 600.0) == 26.25
    assert relax(10.0, 20.0, 300.0, 300.0) == 15.0
    assert relax(45.0, 20.0, 300.0, 0.0) == 45.0
    assert relax(45.0, 20.0, 300.0, -5.0) == 45.0
    # composes exactly: tick frequency does not matter
    two_steps = relax(relax(80.0, 20.0, 300.0, 120.0), 20.0, 300.0, 480.0)
    assert two_steps == approx(relax(80.0, 20.0, 300.0, 600.0))


def test_excitement_half_life_is_five_minutes(engine, bus, clock):
    send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))
    clock.advance(5 * MINUTE)
    engine.tick()
    assert engine.values()["excitement"] == approx(32.5)
    clock.advance(5 * MINUTE)
    assert engine.values()["excitement"] == approx(26.25)


def test_curiosity_relaxes_toward_the_slider_with_a_20_minute_half_life(engine, bus, clock, settings):
    send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))  # curiosity 70
    clock.advance(20 * MINUTE)
    assert engine.values()["curiosity"] == approx(65.0)
    send(bus, ActivityMilestone(category=ActivityCategory.WORKING, app=AppInfo("x.exe", "X"), seconds=1))
    send(bus, ActivityMilestone(category=ActivityCategory.WORKING, app=AppInfo("x.exe", "X"), seconds=1))
    assert engine.values()["curiosity"] == approx(55.0)
    clock.advance(20 * MINUTE)
    assert engine.values()["curiosity"] == approx(57.5)
    # moving the slider moves the resting point
    settings.personality.curiosity = 90
    clock.advance(20 * MINUTE)
    assert engine.values()["curiosity"] == approx(90 + (57.5 - 90) / 2)


def test_trust_never_decays(engine, bus, clock):
    send(bus, outcome(InteractionKind.PET, "happy"))
    clock.advance(3 * DAY)
    engine.tick()
    assert engine.values()["trust"] == approx(40.5)


def test_many_short_ticks_equal_one_long_tick(bus, repo, clock, box, settings):
    a = make_engine(bus, repo, clock, box, settings)
    a.load()
    send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))
    expected = relax(45.0, 20.0, 300.0, 600.0)
    for _ in range(20):
        clock.advance(30)
        a.tick()
    assert a.values()["excitement"] == approx(expected)


def test_a_bump_does_not_decay_for_time_before_it(engine, bus, clock):
    send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))  # 45
    clock.advance(290)
    send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))
    after_second = relax(45.0, 20.0, 300.0, 290.0) + 25.0
    assert engine.values()["excitement"] == approx(after_second)
    clock.advance(10)
    engine.tick()
    assert engine.values()["excitement"] == approx(relax(after_second, 20.0, 300.0, 10.0))


def test_system_sleep_is_measured_with_the_wall_clock(engine, bus, clock):
    send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))
    # the monotonic clock stood still while the laptop slept for two hours
    clock.set_now(clock.now() + 2 * HOUR)
    engine.tick()
    assert engine.values()["excitement"] == approx(20.0)


def test_wall_clock_going_backwards_is_harmless(engine, bus, clock):
    send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))
    clock.set_now(clock.now() - 5 * HOUR)
    engine.tick()
    assert engine.values()["excitement"] == approx(45.0)
    clock.advance(5 * MINUTE)
    assert engine.values()["excitement"] == approx(32.5)


# -- absence: never punished -------------------------------------------------------------------------


def _unhappy_state(engine, bus):
    for _ in range(4):
        send(bus, outcome(InteractionKind.CLICK, "annoyed"))  # excitement 0
    for _ in range(8):
        send(bus, ActivityMilestone(category=ActivityCategory.WORKING, app=AppInfo("x.exe", "X"), seconds=1))
    for _ in range(4):
        send(bus, outcome(InteractionKind.PET, "happy"))  # trust 42


def test_a_week_away_is_not_punished(engine, bus, repo, clock, box, settings):
    _unhappy_state(engine, bus)
    before = engine.values()
    assert before["excitement"] == 0.0 and before["curiosity"] == 20.0
    engine.save()
    engine.close()
    clock.advance(7 * DAY)
    again = make_engine(bus, repo, clock, box, settings)
    again.load()
    after = again.values()
    assert after["trust"] == approx(before["trust"])  # trust never fades
    assert after["excitement"] == approx(20.0)  # back at rest, not lower
    assert after["curiosity"] == approx(60.0)  # back at her slider
    for name in ("excitement", "curiosity", "trust"):
        assert after[name] >= before[name]


def test_a_very_long_absence_is_capped_and_safe(engine, bus, repo, clock, box, settings):
    send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))
    engine.save()
    clock.advance(400 * DAY)
    again = make_engine(bus, repo, clock, box, settings)
    again.load()
    values = again.values()
    assert all(math.isfinite(v) and 0.0 <= v <= 100.0 for v in values.values())
    assert values["excitement"] == approx(20.0)


def test_last_tick_in_the_future_means_no_catch_up(engine, bus, repo, clock, box, settings):
    send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))
    engine.save()
    clock.set_now(clock.now() - DAY)  # the user set the clock back
    again = make_engine(bus, repo, clock, box, settings)
    again.load()
    assert again.values()["excitement"] == approx(45.0)


# -- persistence -------------------------------------------------------------------------------------


def test_round_trip(engine, bus, repo, clock, box, settings):
    send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))
    send(bus, outcome(InteractionKind.PET, "happy"))
    send(bus, outcome(InteractionKind.TALK, "ok"))
    before = engine.values()
    engine.save()
    stored = repo.get(STATE_KEY)
    assert set(stored) >= {"curiosity", "excitement", "trust", "last_tick"}
    assert stored["last_tick"] == clock.now()
    again = make_engine(bus, repo, clock, box, settings)
    again.load()
    assert again.values() == approx(before)


def test_persists_at_most_every_minute(engine, bus, repo, clock):
    repo.writes.clear()
    for _ in range(12):  # two minutes of activity, ticking every 10 s
        send(bus, outcome(InteractionKind.TALK, "ok"))
        clock.advance(10)
        engine.tick()
    assert 1 <= repo.writes.count(STATE_KEY) <= 2


def test_no_writes_when_nothing_changes(engine, bus, repo, clock):
    send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))
    clock.advance(DAY)
    engine.tick()  # converged; saved once
    repo.writes.clear()
    for _ in range(20):
        clock.advance(MINUTE)
        engine.tick()
    assert repo.writes == []


def test_save_always_writes(engine, repo):
    repo.writes.clear()
    engine.save()
    engine.save()
    assert repo.writes == [STATE_KEY, STATE_KEY]


def test_saving_never_raises(bus, db, clock, box, settings, caplog):
    emotion = EmotionEngine(bus, BrokenRepo(db, clock), clock, lambda: box.needs, lambda: settings)
    emotion.load()
    send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))
    with caplog.at_level(logging.ERROR):
        clock.advance(2 * MINUTE)
        emotion.tick()
        emotion.save()
    assert "could not save emotion state" in caplog.text


@pytest.mark.parametrize("stored, expected", [
    ("garbage", {"curiosity": 60.0, "excitement": 20.0, "trust": 40.0}),
    ([1, 2, 3], {"curiosity": 60.0, "excitement": 20.0, "trust": 40.0}),
    ({"curiosity": True, "excitement": "x", "trust": None},
     {"curiosity": 60.0, "excitement": 20.0, "trust": 40.0}),
    ({"curiosity": 500, "excitement": -5, "trust": 77.5},
     {"curiosity": 100.0, "excitement": 0.0, "trust": 77.5}),
    ({"trust": 90, "last_tick": "yesterday", "pet_trust_today": "lots"},
     {"curiosity": 60.0, "excitement": 20.0, "trust": 90.0}),
])
def test_corrupt_state_is_tolerated(bus, repo, clock, box, settings, stored, expected):
    repo.set(STATE_KEY, stored)
    emotion = make_engine(bus, repo, clock, box, settings)
    emotion.load()
    values = emotion.values()
    assert {name: values[name] for name in expected} == expected


def test_nan_in_stored_state_is_tolerated(db, bus, repo, clock, box, settings):
    with db.transaction() as conn:
        conn.execute("INSERT INTO kv_state(key, value, updated_at) VALUES(?, ?, 0)",
                     (STATE_KEY, json.dumps({"trust": math.nan, "excitement": math.inf})))
    emotion = make_engine(bus, repo, clock, box, settings)
    emotion.load()
    assert emotion.values()["trust"] == 40.0
    assert emotion.values()["excitement"] == 20.0


def test_reload_after_wipe_resets_to_defaults(engine, bus, repo):
    send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))
    for _ in range(4):
        send(bus, outcome(InteractionKind.PET, "happy"))
    engine.save()
    repo.delete(STATE_KEY)
    engine.load()
    values = engine.values()
    assert (values["curiosity"], values["excitement"], values["trust"]) == (60.0, 20.0, 40.0)


# -- publishing -------------------------------------------------------------------------------------


def test_publishes_once_after_load(bus, repo, clock, box, settings, rec):
    emotion = make_engine(bus, repo, clock, box, settings)
    emotion.load()
    events = rec.drain()
    assert len(events) == 1
    event = events[0]
    assert event.trust == 40.0 and event.excitement == 20.0 and event.curiosity == 60.0
    assert event.happiness == box.needs.happiness
    assert event.dominant == "idle"


def test_small_changes_are_not_published(engine, bus, rec):
    send(bus, outcome(InteractionKind.PET, "happy"))
    assert rec.drain() == []
    engine.tick()
    assert rec.drain() == []


def test_changes_of_three_points_are_published(engine, bus, rec):
    send(bus, outcome(InteractionKind.TALK, "ok"))
    events = rec.drain()
    assert len(events) == 1
    assert events[0].curiosity == 63.0


def test_small_changes_accumulate(engine, bus, rec):
    for _ in range(5):
        send(bus, outcome(InteractionKind.FEED, "fed"))
    assert rec.drain() == []
    for _ in range(5):
        send(bus, outcome(InteractionKind.FEED, "fed"))
    events = rec.drain()
    assert len(events) == 1
    assert events[0].trust == 43.0


def test_needs_changes_are_republished(engine, bus, box, rec):
    box.needs.hunger = box.needs.hunger + 1
    send(bus, needs_event(box.needs))
    assert rec.drain() == []
    box.needs.hunger = 40
    send(bus, needs_event(box.needs))
    events = rec.drain()
    assert len(events) == 1 and events[0].hunger == 40.0


def test_a_new_dominant_state_is_published_even_for_a_small_move(engine, bus, box, rec):
    box.needs = Needs(energy=21)
    send(bus, needs_event(box.needs))
    rec.drain()
    box.needs = Needs(energy=19.5)
    send(bus, needs_event(box.needs))
    events = rec.drain()
    assert len(events) == 1
    assert events[0].dominant == "yawn"


def test_tick_publishes_decay(engine, bus, clock, rec):
    send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))
    rec.drain()
    clock.advance(10 * MINUTE)
    engine.tick()
    events = rec.drain()
    assert len(events) == 1
    assert events[0].excitement == 26.2  # rounded to one decimal


# -- dominant ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("changes, expected", [
    ({}, "idle"),
    ({"energy": 19.9}, "yawn"),
    ({"energy": 20.0}, "idle"),
    ({"hunger": 80.1}, "sad"),
    ({"hunger": 80.0}, "idle"),
    ({"excitement": 76}, "happy"),
    ({"excitement": 75}, "idle"),
    ({"happiness": 29}, "sad"),
    ({"happiness": 30}, "idle"),
    ({"curiosity": 81}, "curious"),
    ({"boredom": 81}, "thinking"),
    ({"happiness": 81, "affection": 71}, "shy"),
    ({"happiness": 81, "affection": 70}, "idle"),
    ({"happiness": 80, "affection": 90}, "idle"),
    # order of the rules
    ({"energy": 10, "hunger": 95}, "yawn"),
    ({"hunger": 95, "excitement": 95}, "sad"),
    ({"excitement": 95, "happiness": 10}, "happy"),
    ({"happiness": 10, "curiosity": 95}, "sad"),
    ({"curiosity": 95, "boredom": 95}, "curious"),
    ({"boredom": 95, "happiness": 90, "affection": 90}, "thinking"),
])
def test_dominant_rules(changes, expected):
    assert dominant_state({**NEUTRAL, **changes}) == expected


def test_engine_dominant_follows_its_values(engine, bus, box):
    assert engine.dominant() == "idle"
    for _ in range(3):
        send(bus, GameStarted(game_id="g", title="G", process_name="g.exe"))
    assert engine.dominant() == "happy"
    box.needs = Needs(energy=5)
    assert engine.dominant() == "yawn"


# -- describe ---------------------------------------------------------------------------------------


def test_describe_example():
    values = {**NEUTRAL, "happiness": 75, "curiosity": 85, "hunger": 50}
    assert describe_values(values) == "весела, дуже цікаво, трохи голодна"


def test_describe_calm():
    assert describe_values(NEUTRAL) == "спокійна"


def test_describe_names_the_strongest_feelings_first_but_reads_naturally():
    values = {
        "happiness": 10, "energy": 10, "boredom": 95, "affection": 95,
        "curiosity": 95, "excitement": 95, "hunger": 95, "trust": 41,
    }
    parts = describe_values(values).split(", ")
    assert len(parts) == 4
    assert parts == ["у захваті", "дуже сонна", "дуже нудно", "дуже голодна"]


def test_describe_uses_feminine_forms():
    for _neutral, bands in _DESCRIPTORS.values():
        for _cmp, _threshold, words in bands:
            for word in words.split():
                assert not word.endswith(("ий", "ій")), words


def test_engine_describe(engine, bus):
    assert engine.describe() == "весела, бадьора"  # default needs
    send(bus, outcome(InteractionKind.TALK, "ok"))
    send(bus, outcome(InteractionKind.TALK, "ok"))
    assert "цікаво" in engine.describe()


def test_events_before_load_are_not_lost(bus, repo, clock, box, settings):
    emotion = make_engine(bus, repo, clock, box, settings)  # load() not called yet
    send(bus, app_event(ActivityCategory.PROGRAMMING))
    curiosity = emotion.values()["curiosity"]
    assert curiosity == 64.0
    send(bus, app_event(ActivityCategory.PROGRAMMING))  # same category: remembered across the lazy load
    assert emotion.values()["curiosity"] == curiosity
    for _ in range(12):
        send(bus, outcome(InteractionKind.PET, "happy"))
    assert emotion.values()["trust"] == approx(45.0)
