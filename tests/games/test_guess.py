"""«Вгадай число»: hints for every secret and guess, tries, invalid input,
rewards, and that halving always wins within five tries."""

import random

import pytest

from chibi.games.guess import (
    EFFECTS_LOSE,
    EFFECTS_WIN,
    HIGH,
    HOT,
    LOW,
    MAX_TRIES,
    WARM,
    GuessNumber,
    Hint,
    parse_guess,
)
from tests.games.fakes import ScriptedRandom


def game_with(secret, seed=0):
    return GuessNumber(ScriptedRandom(ints=[secret], seed=seed))


def test_secret_comes_from_the_rng():
    rng, twin = random.Random(42), random.Random(42)
    game = GuessNumber(rng)
    game.guess(HIGH)
    for _ in range(MAX_TRIES):
        game.guess(game.guesses[-1] - 1 if game.guesses[-1] > LOW else HIGH)
    assert game.revealed == twin.randint(LOW, HIGH)


def test_secrets_cover_the_range():
    rng = random.Random(0)
    secrets = {GuessNumber(rng)._secret for _ in range(400)}
    assert secrets == set(range(LOW, HIGH + 1))


@pytest.mark.parametrize("secret", range(LOW, HIGH + 1))
def test_hints_for_every_guess(secret):
    for guess in range(LOW, HIGH + 1):
        game = game_with(secret)
        feedback = game.guess(guess)
        if guess == secret:
            assert feedback.hint is Hint.CORRECT and feedback.won and feedback.finished
        else:
            assert feedback.hint is (Hint.HIGHER if secret > guess else Hint.LOWER)
            assert not feedback.finished
            assert feedback.tries_left == MAX_TRIES - 1
            word = "Більше" if secret > guess else "Менше"
            distance = abs(secret - guess)
            if distance > WARM:
                assert word.lower() in feedback.comment.lower() or "холодно" in feedback.comment.lower()
            elif distance <= HOT:
                assert any(w in feedback.comment for w in ("Гаряче", "Майже", "поруч"))
            else:
                assert any(w in feedback.comment for w in ("тепліше", "Тепло", "Трохи"))
            direction = ("більше", "вище") if secret > guess else ("менше", "нижче")
            assert any(d in feedback.comment.lower() for d in direction), feedback.comment


@pytest.mark.parametrize("secret", range(LOW, HIGH + 1))
def test_halving_always_wins_within_five_tries(secret):
    game = game_with(secret)
    low, high = LOW, HIGH
    while not game.finished:
        middle = (low + high) // 2
        feedback = game.guess(middle)
        if feedback.hint is Hint.HIGHER:
            low = middle + 1
        elif feedback.hint is Hint.LOWER:
            high = middle - 1
    assert game.won
    assert len(game.guesses) <= MAX_TRIES


def test_five_wrong_tries_lose_and_reveal():
    game = game_with(13)
    assert game.revealed is None
    for guess in (1, 2, 3, 4):
        assert not game.guess(guess).finished
    last = game.guess(5)
    assert last.finished and not last.won
    assert "я загадала 13" in last.comment
    assert game.revealed == 13
    result = game.result()
    assert (result.won, result.stars, result.score, result.verdict) == (False, 2, 0, "lose")
    assert result.phrase_key == "game.guess.lose"
    assert result.effects == EFFECTS_LOSE
    assert game.guess(13).hint is Hint.OVER


@pytest.mark.parametrize(("tries", "stars"), [(1, 12), (2, 10), (3, 8), (4, 6), (5, 4)])
def test_stars_reward_fewer_tries(tries, stars):
    game = game_with(20)
    for guess in range(1, tries):
        game.guess(guess)
    feedback = game.guess(20)
    assert feedback.won
    result = game.result()
    assert (result.stars, result.score, result.verdict) == (stars, MAX_TRIES - tries + 1, "win")
    assert result.effects == EFFECTS_WIN


def test_first_try_is_magic():
    assert "З першої спроби" in game_with(7).guess(7).comment


@pytest.mark.parametrize("value", ["", "abc", "0", "21", "-3", "1.5", " ", "1e1", "७७७७", True, 0, 21, None])
def test_invalid_guesses_cost_nothing(value):
    game = game_with(10)
    feedback = game.guess(value)
    assert feedback.hint is Hint.INVALID
    assert feedback.tries_left == MAX_TRIES
    assert "від 1 до 20" in feedback.comment


def test_parse_guess():
    assert parse_guess(" 7 ") == 7
    assert parse_guess("07") == 7
    assert parse_guess(20) == 20
    assert parse_guess("20") == 20
    assert parse_guess(False) is None


def test_repeats_cost_nothing():
    game = game_with(10)
    game.guess(5)
    feedback = game.guess("5")
    assert feedback.hint is Hint.REPEAT
    assert feedback.tries_left == MAX_TRIES - 1
    assert game.guesses == [5]


def test_comments_are_reproducible_with_a_seed():
    def comments(seed):
        game = GuessNumber(random.Random(seed))
        return [game.guess(n).comment for n in (1, 20, 10, 5, 15)]

    assert comments(3) == comments(3)


def test_result_before_the_end_raises():
    with pytest.raises(RuntimeError):
        game_with(3).result()


def test_rewards_in_range():
    for effects in (EFFECTS_WIN, EFFECTS_LOSE):
        assert -30 <= effects["boredom"] <= -15
        assert 5 <= effects["happiness"] <= 12
        assert 2 <= effects["friendship_xp"] <= 5


def test_invalid_max_tries():
    with pytest.raises(ValueError):
        GuessNumber(random.Random(), max_tries=0)
