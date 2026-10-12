import dataclasses
import math
import random
import time

import pytest

from chibi.core.events import InteractionKind
from chibi.tamagotchi.needs import Needs, NeedsModel

HOUR = 3600.0
DAY = 24 * HOUR
WEEK = 7 * DAY

approx = pytest.approx

STATS = ("hunger", "energy", "happiness", "boredom", "affection")


@pytest.fixture
def model():
    return NeedsModel()


def random_needs(rng: random.Random) -> Needs:
    return Needs(*(rng.uniform(0, 100) for _ in STATS), friendship_xp=rng.uniform(0, 700))


def assert_in_range(needs: Needs) -> None:
    for name in STATS:
        assert 0.0 <= getattr(needs, name) <= 100.0, name
    assert needs.friendship_xp >= 0.0


# --- Needs ------------------------------------------------------------------------------


def test_defaults():
    n = Needs()
    assert (n.hunger, n.energy, n.happiness, n.boredom, n.affection, n.friendship_xp) == (
        30.0, 80.0, 70.0, 20.0, 50.0, 0.0)


def test_constructor_and_assignment_clamp():
    n = Needs(hunger=150, energy=-5, happiness=1e9, boredom=-1e9, affection=101, friendship_xp=-3)
    assert (n.hunger, n.energy, n.happiness, n.boredom, n.affection, n.friendship_xp) == (
        100.0, 0.0, 100.0, 0.0, 100.0, 0.0)
    n.hunger = -20
    n.energy = 300
    n.friendship_xp = -1
    assert (n.hunger, n.energy, n.friendship_xp) == (0.0, 100.0, 0.0)


def test_nan_falls_back_to_default_and_inf_is_clamped():
    n = Needs(hunger=math.nan, energy=math.inf, boredom=-math.inf, friendship_xp=math.inf)
    assert n.hunger == 30.0
    assert n.energy == 100.0
    assert n.boredom == 0.0
    assert math.isfinite(n.friendship_xp)


def test_replace_also_clamps():
    assert dataclasses.replace(Needs(), happiness=250).happiness == 100.0


@pytest.mark.parametrize(
    ("xp", "level"),
    [(0, 1), (9.99, 1), (10, 2), (29.5, 2), (30, 3), (60, 4), (100, 5), (160, 6), (240, 7),
     (340, 8), (459, 8), (460, 9), (599.9, 9), (600, 10), (10_000, 10)],
)
def test_friendship_level_thresholds(xp, level):
    assert Needs(friendship_xp=xp).friendship_level == level


@pytest.mark.parametrize(
    ("xp", "progress"),
    [(0, 0.0), (5, 0.5), (10, 0.0), (20, 0.5), (45, 0.5), (530, 0.5), (600, 1.0), (5000, 1.0)],
)
def test_friendship_progress(xp, progress):
    assert Needs(friendship_xp=xp).friendship_progress == approx(progress)


def test_friendship_progress_stays_in_unit_range():
    for xp in [x * 0.25 for x in range(0, 2600)]:
        assert 0.0 <= Needs(friendship_xp=xp).friendship_progress <= 1.0


def test_to_dict_from_dict_round_trip():
    n = Needs(hunger=12.5, energy=33.0, happiness=91.0, boredom=4.0, affection=77.0, friendship_xp=123.5)
    data = n.to_dict()
    assert data == {"hunger": 12.5, "energy": 33.0, "happiness": 91.0, "boredom": 4.0,
                    "affection": 77.0, "friendship_xp": 123.5}
    assert Needs.from_dict(data) == n


@pytest.mark.parametrize("data", [None, "garbage", 42, [1, 2, 3], True])
def test_from_dict_non_object_gives_defaults(data):
    assert Needs.from_dict(data) == Needs()


def test_from_dict_bad_values_take_defaults_and_numbers_are_clamped():
    data = {
        "hunger": "very",          # wrong type -> default
        "energy": 500,              # out of range -> clamped
        "happiness": True,          # bool is not a number here -> default
        "boredom": None,            # -> default
        "affection": float("nan"),  # -> default
        "friendship_xp": -40,       # -> clamped to 0
        "unknown": 5,               # ignored
    }
    n = Needs.from_dict(data)
    assert (n.hunger, n.energy, n.happiness, n.boredom, n.affection, n.friendship_xp) == (
        30.0, 100.0, 70.0, 20.0, 50.0, 0.0)


def test_from_dict_missing_fields_take_defaults():
    assert Needs.from_dict({"hunger": 90}) == Needs(hunger=90)
    assert Needs.from_dict({}) == Needs()


def test_from_dict_rejects_infinite_and_huge_numbers():
    n = Needs.from_dict({"hunger": float("inf"), "friendship_xp": 10**400})
    assert n.hunger == 30.0
    assert n.friendship_xp == 0.0


# --- tick -----------------------------------------------------------------------------------


def test_tick_awake_idle_one_hour(model):
    n = model.tick(Needs(), HOUR, sleeping=False, user_active=False)
    assert n.hunger == approx(37.0)
    assert n.energy == approx(74.0)
    assert n.boredom == approx(32.0)
    assert n.happiness == approx(70.0)  # already at its target
    assert n.affection == approx(50.0)
    assert n.friendship_xp == 0.0


def test_tick_awake_with_user_active_bores_slower(model):
    n = model.tick(Needs(), HOUR, sleeping=False, user_active=True)
    assert n.boredom == approx(24.0)
    assert n.hunger == approx(37.0)
    assert n.energy == approx(74.0)


def test_tick_sleeping_one_hour(model):
    start = Needs(energy=40, boredom=30)
    for active in (False, True):  # user activity doesn't matter while she sleeps
        n = model.tick(start, HOUR, sleeping=True, user_active=active)
        assert n.hunger == approx(33.0)
        assert n.energy == approx(65.0)
        assert n.boredom == approx(20.0)


def test_tick_clamps_at_bounds(model):
    n = model.tick(Needs(hunger=98, energy=3, boredom=99), HOUR, sleeping=False, user_active=False)
    assert (n.hunger, n.energy, n.boredom) == (100.0, 0.0, 100.0)
    n = model.tick(Needs(energy=95, boredom=4), HOUR, sleeping=True, user_active=False)
    assert (n.energy, n.boredom) == (100.0, 0.0)


def test_tick_is_rate_based_regardless_of_tick_size(model):
    many = Needs(happiness=40, affection=80)
    for _ in range(120):
        many = model.tick(many, 30.0, sleeping=False, user_active=False)
    one = model.tick(Needs(happiness=40, affection=80), HOUR, sleeping=False, user_active=False)
    for name in STATS:
        assert getattr(many, name) == approx(getattr(one, name), abs=1e-6), name


@pytest.mark.parametrize("dt", [0.0, -30.0, math.nan, -math.inf])
def test_tick_with_no_time_changes_nothing(model, dt):
    start = Needs(hunger=40, happiness=10)
    assert model.tick(start, dt, sleeping=False, user_active=False) == start


def test_tick_does_not_mutate_its_input(model):
    start = Needs(hunger=40)
    before = dataclasses.replace(start)
    result = model.tick(start, HOUR, sleeping=False, user_active=False)
    assert start == before
    assert result is not start


def test_tick_absurd_duration_finishes_quickly_and_converges(model):
    began = time.perf_counter()
    n = model.tick(Needs(), 1e12, sleeping=False, user_active=False)
    assert time.perf_counter() - began < 2.0
    assert (n.hunger, n.energy, n.boredom) == (100.0, 0.0, 100.0)
    assert n.happiness == approx(model.happiness_target(n), abs=1e-6)
    n = model.tick(Needs(), math.inf, sleeping=True, user_active=False)
    assert n.energy == 100.0


def test_happiness_drifts_up_toward_target(model):
    n = model.tick(Needs(happiness=50), HOUR, sleeping=False, user_active=False)
    assert n.happiness == approx(60.0)  # 10 points per hour toward 70
    n = model.tick(Needs(happiness=65), HOUR, sleeping=False, user_active=False)
    assert n.happiness == approx(70.0)  # no overshoot


def test_happiness_drifts_down_toward_target(model):
    n = model.tick(Needs(happiness=100), HOUR, sleeping=False, user_active=False)
    assert n.happiness == approx(90.0)


def test_happiness_target_formula(model):
    assert model.happiness_target(Needs()) == approx(70.0)
    assert model.happiness_target(Needs(hunger=100)) == approx(50.0)
    assert model.happiness_target(Needs(boredom=100)) == approx(55.0)
    assert model.happiness_target(Needs(energy=0)) == approx(61.0)
    assert model.happiness_target(Needs(affection=100)) == approx(75.0)
    assert model.happiness_target(Needs(affection=0)) == approx(65.0)
    worst = Needs(hunger=100, boredom=100, energy=0, affection=0)
    assert model.happiness_target(worst) == approx(70 - 20 - 15 - 9 - 5)


def test_unmet_needs_make_her_sadder_over_time(model):
    n = model.tick(Needs(hunger=100, boredom=100, energy=0, happiness=70), HOUR,
                   sleeping=False, user_active=False)
    assert n.happiness == approx(60.0)


def test_affection_drifts_toward_fifty(model):
    assert model.tick(Needs(affection=80), HOUR, sleeping=False, user_active=False).affection == approx(79)
    assert model.tick(Needs(affection=10), HOUR, sleeping=False, user_active=False).affection == approx(11)
    assert model.tick(Needs(affection=50), HOUR, sleeping=False, user_active=False).affection == 50.0
    assert model.tick(Needs(affection=50.4), HOUR, sleeping=False, user_active=False).affection == 50.0


def test_affection_never_drifts_below_thirty(model):
    n = Needs(affection=95)
    for _ in range(200):
        n = model.tick(n, HOUR, sleeping=False, user_active=False)
        assert n.affection >= 30.0
    assert n.affection == approx(50.0)


def test_tick_never_changes_friendship_xp(model):
    rng = random.Random(7)
    for _ in range(200):
        start = random_needs(rng)
        n = model.tick(start, rng.uniform(0, 3 * DAY), sleeping=rng.random() < 0.5,
                       user_active=rng.random() < 0.5)
        assert n.friendship_xp == start.friendship_xp
        assert_in_range(n)


# --- offline catch-up ---------------------------------------------------------------------


def test_offline_one_hour(model):
    n = model.offline_catch_up(Needs(energy=40, hunger=30, boredom=20), HOUR)
    assert n.energy == approx(65.0)
    assert n.hunger == approx(32.0)
    assert n.boredom == approx(23.0)


def test_offline_week_away_never_punishes(model):
    start = Needs(hunger=40, energy=10, happiness=60, boredom=30, affection=70, friendship_xp=55)
    n = model.offline_catch_up(start, WEEK)
    assert n.energy == 100.0
    assert n.hunger == approx(55.0)
    assert n.boredom == approx(50.0)
    assert n.happiness >= 50.0
    assert n.affection == 70.0
    assert n.friendship_xp == 55.0
    assert model.mood(n) not in ("Сонна", "Голодна", "Нудьгує", "Сумна")


def test_offline_week_away_from_defaults_is_a_good_mood(model):
    n = model.offline_catch_up(Needs(), WEEK)
    assert (n.hunger, n.energy, n.boredom) == (approx(55.0), 100.0, approx(50.0))
    assert n.happiness == approx(68.0)  # target with hunger 55: 70 - 0.4 * 5
    assert model.mood(n) == "Весела"


def test_offline_hunger_and_boredom_never_rise_above_what_they_were_when_already_high(model):
    n = model.offline_catch_up(Needs(hunger=80, boredom=90), WEEK)
    assert n.hunger == 80.0
    assert n.boredom == 90.0


def test_offline_hunger_rises_at_most_two_per_hour(model):
    assert model.offline_catch_up(Needs(hunger=10), 5 * HOUR).hunger == approx(20.0)


def test_offline_happiness_floor(model):
    # already sad: absence doesn't make it worse, rest lifts her toward the target
    sad = model.offline_catch_up(Needs(happiness=20, hunger=100, boredom=100, energy=0), WEEK)
    assert sad.happiness >= 20.0
    # happy and very hungry: target is 50 - 0.3 * 50 = 35 but she never drops below 50
    hungry = model.offline_catch_up(Needs(happiness=90, hunger=100, boredom=100), WEEK)
    assert hungry.happiness == approx(50.0)


def test_offline_happiness_can_still_settle_from_euphoria(model):
    n = model.offline_catch_up(Needs(happiness=100), WEEK)
    assert 50.0 <= n.happiness < 100.0


@pytest.mark.parametrize("away", [0.0, -HOUR, math.nan, -math.inf])
def test_offline_invalid_or_no_absence_changes_nothing(model, away):
    start = Needs(hunger=60, energy=5, happiness=10, boredom=70)
    assert model.offline_catch_up(start, away) == start


def test_offline_infinite_absence_is_safe(model):
    n = model.offline_catch_up(Needs(), math.inf)
    assert n.energy == 100.0
    assert n.hunger == approx(55.0)
    assert_in_range(n)


def test_offline_invariants_hold_for_any_state_and_absence(model):
    rng = random.Random(1234)
    for _ in range(500):
        start = random_needs(rng)
        away = rng.choice([rng.uniform(0, HOUR), rng.uniform(0, 3 * DAY), rng.uniform(0, 365 * DAY)])
        n = model.offline_catch_up(start, away)
        assert_in_range(n)
        assert n.energy >= start.energy
        assert n.hunger <= max(start.hunger, 55.0) + 1e-9
        assert n.hunger >= start.hunger
        assert n.boredom <= max(start.boredom, 50.0) + 1e-9
        assert n.happiness >= min(start.happiness, 50.0) - 1e-9
        assert n.affection == start.affection
        assert n.friendship_xp == start.friendship_xp


# --- interactions ---------------------------------------------------------------------------


def test_feed_when_full(model):
    start = Needs(hunger=14.9)
    assert model.feed(start) == (start, "full")


def test_feed_normal(model):
    n, outcome = model.feed(Needs(hunger=50, happiness=60, energy=40, friendship_xp=1))
    assert outcome == "fed"
    assert (n.hunger, n.happiness, n.energy, n.friendship_xp) == (15.0, 66.0, 43.0, 3.0)


def test_feed_boundaries(model):
    assert model.feed(Needs(hunger=15))[1] == "fed"
    assert model.feed(Needs(hunger=15))[0].hunger == 0.0  # clamped
    assert model.feed(Needs(hunger=75))[1] == "fed"
    assert model.feed(Needs(hunger=75.1))[1] == "starving_fed"


def test_feed_starving(model):
    n, outcome = model.feed(Needs(hunger=90, happiness=98, energy=99))
    assert outcome == "starving_fed"
    assert n.hunger == approx(45.0)
    assert (n.happiness, n.energy) == (100.0, 100.0)  # clamped
    assert n.friendship_xp == 2.0


def test_pet_happy(model):
    n, outcome = model.pet(Needs(affection=50, happiness=60, boredom=5), recent_pets=5)
    assert outcome == "happy"
    assert (n.affection, n.happiness, n.boredom, n.friendship_xp) == (54.0, 64.0, 0.0, 3.0)


def test_pet_too_much(model):
    start = Needs()
    assert model.pet(start, recent_pets=6) == (start, "too_much")
    assert model.pet(start, recent_pets=40)[1] == "too_much"


def test_rest(model):
    assert model.rest(Needs(energy=80)) == (Needs(energy=80), "sleep")
    assert model.rest(Needs(energy=80.5)) == (Needs(energy=80.5), "not_tired")


def test_click(model):
    n, outcome = model.click(Needs(boredom=20, friendship_xp=1), recent_clicks=4)
    assert outcome == "poke"
    assert (n.boredom, n.friendship_xp) == (17.0, 1.5)
    n, outcome = model.click(Needs(happiness=50, boredom=20), recent_clicks=5)
    assert outcome == "annoyed"
    assert (n.happiness, n.boredom, n.friendship_xp) == (49.0, 20.0, 0.0)


def test_wake(model):
    assert model.wake(Needs(energy=29.9)) == (Needs(energy=29.9), "grumpy")
    assert model.wake(Needs(energy=30)) == (Needs(energy=30), "ok")


def test_talk_and_call(model):
    n, outcome = model.talk(Needs(boredom=20))
    assert (outcome, n.boredom, n.friendship_xp) == ("ok", 15.0, 1.0)
    n, outcome = model.call(Needs(boredom=20))
    assert (outcome, n.boredom, n.friendship_xp) == ("ok", 18.0, 0.0)


@pytest.mark.parametrize("kind", [InteractionKind.DRAG_START, InteractionKind.DRAG_END])
def test_drags_change_nothing(model, kind):
    start = Needs(hunger=44)
    assert model.interact(start, kind) == (start, "ok")


DOCUMENTED_OUTCOMES = {
    InteractionKind.FEED: {"fed", "full", "starving_fed"},
    InteractionKind.PET: {"happy", "too_much"},
    InteractionKind.REST: {"sleep", "not_tired"},
    InteractionKind.CLICK: {"poke", "annoyed"},
    InteractionKind.WAKE: {"ok", "grumpy"},
}


@pytest.mark.parametrize("kind", list(InteractionKind))
def test_interact_dispatches_every_kind_to_a_documented_outcome(model, kind):
    rng = random.Random(3)
    for _ in range(50):
        start = random_needs(rng)
        n, outcome = model.interact(start, kind, recent_pets=rng.randint(0, 9),
                                    recent_clicks=rng.randint(0, 9))
        assert outcome in DOCUMENTED_OUTCOMES.get(kind, {"ok"})
        assert_in_range(n)
        assert n.friendship_xp >= start.friendship_xp


def test_interact_matches_the_specific_rules(model):
    start = Needs(hunger=60)
    assert model.interact(start, InteractionKind.FEED) == model.feed(start)
    assert model.interact(start, "pet", recent_pets=6) == model.pet(start, 6)
    assert model.interact(start, InteractionKind.CLICK, recent_clicks=5) == model.click(start, 5)


def test_interactions_do_not_mutate_input(model):
    start = Needs(hunger=60)
    before = dataclasses.replace(start)
    for kind in InteractionKind:
        model.interact(start, kind)
    assert start == before


def test_daily_bonus(model):
    assert model.daily_bonus(Needs(friendship_xp=8)).friendship_xp == 13.0


def test_friendship_xp_never_decreases_through_any_rule(model):
    rng = random.Random(99)
    n = Needs()
    for _ in range(2000):
        before = n.friendship_xp
        roll = rng.random()
        if roll < 0.4:
            n, _ = model.interact(n, rng.choice(list(InteractionKind)), recent_pets=rng.randint(0, 8),
                                  recent_clicks=rng.randint(0, 8))
        elif roll < 0.8:
            n = model.tick(n, rng.uniform(0, 2 * HOUR), sleeping=rng.random() < 0.3,
                           user_active=rng.random() < 0.5)
        else:
            n = model.offline_catch_up(n, rng.uniform(-HOUR, 3 * DAY))
        assert n.friendship_xp >= before
        assert_in_range(n)
    assert n.friendship_level > 1


# --- mood -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("needs", "mood"),
    [
        (Needs(energy=19, hunger=90, boredom=90, happiness=10), "Сонна"),
        (Needs(energy=20, hunger=76, boredom=90, happiness=10), "Голодна"),
        (Needs(hunger=75, boredom=76, happiness=10), "Нудьгує"),
        (Needs(boredom=75, happiness=29), "Сумна"),
        (Needs(happiness=30), "Спокійна"),
        (Needs(happiness=81), "Щаслива"),
        (Needs(happiness=80), "Весела"),
        (Needs(happiness=61), "Весела"),
        (Needs(happiness=60), "Спокійна"),
    ],
)
def test_mood_priority(model, needs, mood):
    assert model.mood(needs) == mood


def test_default_mood_is_cheerful(model):
    assert model.mood(Needs()) == "Весела"
