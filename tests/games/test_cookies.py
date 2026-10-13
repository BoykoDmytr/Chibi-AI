"""«Лови печиво»: deterministic physics, catching, burnt cookies, the basket,
timing and the result."""

import math
import random

import pytest

from chibi.games.cookies import (
    BASKET_HEIGHT,
    BASKET_WIDTH,
    BURNT_PENALTY,
    COOKIE_RADIUS,
    FIELD_HEIGHT,
    FIELD_WIDTH,
    GAME_SECONDS,
    GREAT_SCORE,
    MAX_FRAME_SECONDS,
    MAX_STARS,
    OK_SCORE,
    POPUP_SECONDS,
    SPEED_END,
    SPEED_START,
    STEP_SECONDS,
    Catch,
    Cookie,
    CookieCatch,
)

FPS = 60


def snapshot(game):
    return (round(game.elapsed, 9), game.score, game.caught, game.burnt_hits, game.missed, game.spawned,
            [(c.id, round(c.x, 6), round(c.y, 6), round(c.speed, 6), c.burnt) for c in game.cookies])


def run(game, seconds, fps=FPS, clicker=None):
    for _ in range(round(seconds * fps)):
        game.step(1 / fps)
        if clicker is not None:
            clicker(game)


def greedy(game):
    """Clicks every good cookie once it is a third of the way down."""
    for cookie in list(game.cookies):
        if not cookie.burnt and cookie.y > FIELD_HEIGHT / 3:
            game.click(cookie.x, cookie.y)


def test_same_seed_same_game():
    a, b = CookieCatch(random.Random(5)), CookieCatch(random.Random(5))
    run(a, 12, clicker=greedy)
    run(b, 12, clicker=greedy)
    assert snapshot(a) == snapshot(b)


def test_different_seeds_differ():
    a, b = CookieCatch(random.Random(1)), CookieCatch(random.Random(2))
    run(a, 5)
    run(b, 5)
    assert snapshot(a) != snapshot(b)


@pytest.mark.parametrize("fps", [30, 60, 120, 144])
def test_frame_rate_does_not_change_the_game(fps):
    reference = CookieCatch(random.Random(9))
    run(reference, 10, fps=60)
    other = CookieCatch(random.Random(9))
    run(other, 10, fps=fps)
    assert other.spawned == reference.spawned
    assert other.elapsed == pytest.approx(reference.elapsed, abs=STEP_SECONDS)
    for c1, c2 in zip(reference.cookies, other.cookies, strict=True):
        assert c1.id == c2.id and c1.burnt == c2.burnt
        assert c1.x == pytest.approx(c2.x)
        assert c1.y == pytest.approx(c2.y, abs=c1.speed * STEP_SECONDS + 1e-6)


def test_one_long_frame_equals_many_short_ones_up_to_the_cap():
    a, b = CookieCatch(random.Random(4)), CookieCatch(random.Random(4))
    for _ in range(40):
        a.step(MAX_FRAME_SECONDS)
    run(b, 40 * MAX_FRAME_SECONDS)
    assert snapshot(a) == snapshot(b)


def test_a_huge_frame_is_cut():
    game = CookieCatch(random.Random(0))
    game.step(10.0)
    assert game.elapsed == pytest.approx(MAX_FRAME_SECONDS, abs=STEP_SECONDS)


@pytest.mark.parametrize("dt", [0.0, -1.0, math.nan])
def test_no_time_no_change(dt):
    game = CookieCatch(random.Random(0))
    before = snapshot(game)
    game.step(dt)
    assert snapshot(game) == before


def test_lasts_thirty_seconds():
    game = CookieCatch(random.Random(0))
    assert game.duration == GAME_SECONDS == 30
    run(game, 29.9)
    assert not game.finished
    assert game.time_left == pytest.approx(0.1, abs=STEP_SECONDS)
    run(game, 0.2)
    assert game.finished
    assert game.time_left == 0
    assert game.cookies == []  # nothing left falling after the end
    assert game.popups == []  # nor floating
    before = snapshot(game)
    game.step(1)
    assert snapshot(game) == before
    assert game.click(100, 100) is Catch.IGNORED


def test_cookies_fall_faster_and_more_often():
    game = CookieCatch(random.Random(3))
    assert game.fall_speed() == SPEED_START
    early_interval = game.spawn_interval()
    early = []
    late = []
    while not game.finished:
        known = {c.id for c in game.cookies}
        game.step(1 / FPS)
        for cookie in game.cookies:
            if cookie.id not in known:
                (early if game.elapsed < 10 else late if game.elapsed > 20 else []).append(cookie)
    assert game.fall_speed() == SPEED_END
    assert game.spawn_interval() < early_interval
    assert sum(c.speed for c in late) / len(late) > 1.8 * sum(c.speed for c in early) / len(early)
    assert len(late) > len(early)


def test_a_full_game_drops_dozens_of_cookies_with_some_burnt():
    game = CookieCatch(random.Random(11))
    run(game, GAME_SECONDS + 1)
    assert 40 <= game.spawned <= 55
    assert game.missed > 30  # nobody clicked: the good ones fell through
    assert game.score == 0 and game.result().verdict == "poor"


def place(game, *cookies):
    game.cookies[:] = list(cookies)


def test_click_catches_and_scores():
    game = CookieCatch(random.Random(0))
    place(game, Cookie(1, 100, 100, 50, False, 0))
    assert game.click(100 + COOKIE_RADIUS + 5, 100) is Catch.CAUGHT  # a little outside still counts
    assert (game.score, game.caught, game.cookies) == (1, 1, [])
    assert game.popups[-1].text == "+1" and game.popups[-1].good


def test_click_on_empty_space_is_free():
    game = CookieCatch(random.Random(0))
    place(game, Cookie(1, 100, 100, 50, False, 0))
    assert game.click(200, 200) is Catch.MISS
    assert game.click(100 + COOKIE_RADIUS + 7, 100) is Catch.MISS
    assert game.score == 0 and len(game.cookies) == 1


def test_burnt_cookie_costs_points_but_never_below_zero():
    game = CookieCatch(random.Random(0))
    game.score = 5
    place(game, Cookie(1, 50, 50, 50, True, 0), Cookie(2, 150, 50, 50, True, 0))
    assert game.click(50, 50) is Catch.BURNT
    assert game.score == 5 - BURNT_PENALTY
    assert game.popups[-1].text == f"−{BURNT_PENALTY}" and not game.popups[-1].good
    game.score = 1
    assert game.click(150, 50) is Catch.BURNT
    assert game.score == 0
    assert game.burnt_hits == 2


def test_overlapping_cookies_newest_first():
    game = CookieCatch(random.Random(0))
    place(game, Cookie(1, 100, 100, 50, False, 0), Cookie(2, 105, 100, 50, True, 0))
    assert game.click(102, 100) is Catch.BURNT  # the burnt one is drawn on top
    assert [c.id for c in game.cookies] == [1]


def test_missed_counts_only_good_cookies():
    game = CookieCatch(random.Random(0))
    game._spawn_timer = 99  # no new cookies
    place(game, Cookie(1, 100, FIELD_HEIGHT, 100, False, 0), Cookie(2, 200, FIELD_HEIGHT, 100, True, 0))
    run(game, 1)
    assert game.cookies == []
    assert game.missed == 1


def test_popups_fade_away():
    game = CookieCatch(random.Random(0))
    game._spawn_timer = 99
    place(game, Cookie(1, 100, 100, 0, False, 0))
    game.click(100, 100)
    run(game, POPUP_SECONDS / 2)
    assert len(game.popups) == 1
    run(game, POPUP_SECONDS)
    assert game.popups == []


def test_basket_appears_only_with_the_arrow_keys():
    game = CookieCatch(random.Random(0))
    game._spawn_timer = 99
    x = game.basket_x
    place(game, Cookie(1, x, FIELD_HEIGHT - 60, 120, False, 0))
    run(game, 1)
    assert not game.basket_active
    assert (game.caught, game.missed) == (0, 1)  # fell straight through the hidden basket


def test_basket_catches_good_and_burnt_cookies():
    game = CookieCatch(random.Random(0))
    game._spawn_timer = 99
    game.set_basket_direction(1)
    game.set_basket_direction(0)
    assert game.basket_active
    x = game.basket_x
    place(game, Cookie(1, x, FIELD_HEIGHT - 60, 120, False, 0),
          Cookie(2, x + 10, FIELD_HEIGHT - 80, 120, True, 0))
    run(game, 1)
    assert (game.caught, game.burnt_hits, game.score) == (1, 1, 0)
    place(game, Cookie(3, x + BASKET_WIDTH, FIELD_HEIGHT - 60, 120, False, 0))  # beside the basket
    run(game, 1)
    assert game.missed == 1


def test_basket_moves_and_stays_inside():
    game = CookieCatch(random.Random(0))
    game.set_basket_direction(-5)
    run(game, 5)
    assert game.basket_x == BASKET_WIDTH / 2
    game.set_basket_direction(1)
    run(game, 0.5)
    assert game.basket_x == pytest.approx(BASKET_WIDTH / 2 + 140, abs=1)
    run(game, 5)
    assert game.basket_x == FIELD_WIDTH - BASKET_WIDTH / 2
    assert FIELD_HEIGHT - BASKET_HEIGHT > 0


def test_basket_input_after_the_end_does_not_show_it():
    game = CookieCatch(random.Random(0))
    run(game, GAME_SECONDS + 1)
    game.set_basket_direction(1)
    assert not game.basket_active


def test_a_good_player_scores_great():
    game = CookieCatch(random.Random(2))
    run(game, GAME_SECONDS + 1, clicker=greedy)
    result = game.result()
    assert result.verdict == "great" and result.won
    assert result.score == game.caught >= GREAT_SCORE
    assert result.phrase_key == "game.cookies.great"


@pytest.mark.parametrize(("score", "verdict"), [
    (0, "poor"), (OK_SCORE - 1, "poor"), (OK_SCORE, "ok"), (GREAT_SCORE - 1, "ok"), (GREAT_SCORE, "great"),
    (60, "great"),
])
def test_verdicts(score, verdict):
    game = CookieCatch(random.Random(0), duration=0.1)
    run(game, 1)
    game.score = game.caught = score
    assert game.result().verdict == verdict
    assert game.result().won is (verdict != "poor")


def test_rewards_stay_in_range_for_every_score():
    previous_stars = 0
    for score in range(0, 80):
        game = CookieCatch(random.Random(0), duration=0.1)
        run(game, 1)
        game.score = game.caught = score
        result = game.result()
        assert 2 <= result.stars <= MAX_STARS
        assert result.stars >= previous_stars
        previous_stars = result.stars
        e = result.effects
        assert -30 <= e["boredom"] <= -15
        assert 5 <= e["happiness"] <= 12
        assert 2 <= e["friendship_xp"] <= 5
        assert -12 <= e["hunger"] < 0  # she eats what was caught


def test_result_before_the_end_raises():
    with pytest.raises(RuntimeError):
        CookieCatch(random.Random(0)).result()


def test_invalid_duration():
    with pytest.raises(ValueError):
        CookieCatch(random.Random(0), duration=0)


def test_cookies_spawn_inside_the_field():
    game = CookieCatch(random.Random(8))
    xs = []
    while not game.finished:
        game.step(1 / FPS)
        xs.extend(c.x for c in game.cookies)
    assert min(xs) >= COOKIE_RADIUS and max(xs) <= FIELD_WIDTH - COOKIE_RADIUS
