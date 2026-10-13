"""Камінь-ножиці-папір: rules, fairness (her move is fixed before the user's),
best of three, and every way a game can go."""

import itertools
import random

import pytest

from chibi.games import GameResult
from chibi.games.rps import (
    BEATS,
    EFFECTS,
    MAX_ROUNDS,
    MOVES,
    STARS,
    Move,
    Outcome,
    RockPaperScissors,
    judge,
)
from tests.games.fakes import ScriptedRandom

R, S, P = Move.ROCK, Move.SCISSORS, Move.PAPER


def test_judge_every_pair():
    expected = {
        (R, R): Outcome.DRAW, (R, S): Outcome.WIN, (R, P): Outcome.LOSE,
        (S, R): Outcome.LOSE, (S, S): Outcome.DRAW, (S, P): Outcome.WIN,
        (P, R): Outcome.WIN, (P, S): Outcome.LOSE, (P, P): Outcome.DRAW,
    }
    for (user, her), outcome in expected.items():
        assert judge(user, her) is outcome, (user, her)


def test_beats_is_a_cycle():
    assert set(BEATS) == set(BEATS.values()) == set(MOVES)
    for move in MOVES:
        assert BEATS[BEATS[BEATS[move]]] is move
        assert BEATS[move] is not move


def test_moves_order_matches_keys():
    assert MOVES == (R, S, P)  # keys 1, 2, 3


def test_her_first_move_is_drawn_before_the_user_moves():
    rng = random.Random(7)
    twin = random.Random(7)
    game = RockPaperScissors(rng)
    assert game.committed
    twin.choice(MOVES)
    assert rng.getstate() == twin.getstate()  # exactly one draw so far, before any user move


class SpyRandom(random.Random):
    """Logs every draw and every user move in one timeline."""

    def __init__(self, seed, log):
        super().__init__(seed)
        self.log = log

    def choice(self, seq):
        move = super().choice(seq)
        self.log.append(("her", move))
        return move


def test_each_round_her_move_is_chosen_before_the_users():
    timeline = []
    game = RockPaperScissors(SpyRandom(3, timeline))
    while not game.finished:
        timeline.append(("user", R))
        game.play(R)
    # her draw - user move - her draw - user move ...: she always commits first.
    kinds = [kind for kind, _ in timeline]
    assert kinds == ["her", "user"] * len(game.rounds)
    for index, played in enumerate(game.rounds):
        assert timeline[2 * index] == ("her", played.her)


@pytest.mark.parametrize("seed", range(20))
def test_her_moves_do_not_depend_on_the_users(seed):
    """Same seed, different user moves: identical moves from her, round for round."""
    reference = random.Random(seed)
    expected = [reference.choice(MOVES) for _ in range(MAX_ROUNDS)]
    for user_moves in (itertools.repeat(R), itertools.repeat(P), itertools.cycle(MOVES)):
        game = RockPaperScissors(random.Random(seed))
        hers = []
        for move in user_moves:
            if game.finished:
                break
            hers.append(game.play(move).her)
        assert hers == expected[:len(hers)]


def test_her_moves_are_roughly_uniform():
    rng = random.Random(1)
    counts = {move: 0 for move in MOVES}
    for _ in range(600):
        game = RockPaperScissors(rng)
        counts[game.play(R).her] += 1
    assert all(150 < n < 250 for n in counts.values()), counts


def play_out(outcomes):
    """Play a scripted game where the user gets ``outcomes`` (W/L/D) in order."""
    hers = []
    for outcome in outcomes:
        # The user always shows rock; her move decides the outcome.
        hers.append({"W": S, "L": P, "D": R}[outcome])
    game = RockPaperScissors(ScriptedRandom(choices=hers))
    played = []
    for _ in outcomes:
        if game.finished:
            break
        played.append(game.play(R).outcome)
    return game, played


def expected_end(outcomes):
    """(rounds played, user wins, her wins) by the rules: first to two, at most MAX_ROUNDS."""
    wins = losses = 0
    for count, outcome in enumerate(outcomes, 1):
        wins += outcome == "W"
        losses += outcome == "L"
        if wins == 2 or losses == 2 or count == MAX_ROUNDS:
            return count, wins, losses
    return None


def test_every_possible_game():
    """All 3^7 outcome sequences: the game stops exactly when the rules say."""
    seen_outcomes = set()
    for outcomes in itertools.product("WLD", repeat=MAX_ROUNDS):
        game, _played = play_out(outcomes)
        rounds, wins, losses = expected_end(outcomes)
        assert game.finished
        assert len(game.rounds) == rounds
        assert (game.user_wins, game.her_wins) == (wins, losses)
        result = game.result()
        if wins > losses:
            outcome = Outcome.WIN
        elif losses > wins:
            outcome = Outcome.LOSE
        else:
            outcome = Outcome.DRAW
        assert result == GameResult("rps", wins, outcome is Outcome.WIN, STARS[outcome], outcome.value,
                                    EFFECTS[outcome])
        seen_outcomes.add(outcome)
    assert seen_outcomes == set(Outcome)


def test_best_of_three():
    game, played = play_out("WDLW")
    assert played == [Outcome.WIN, Outcome.DRAW, Outcome.LOSE, Outcome.WIN]
    assert game.finished and game.result().won
    assert game.result().phrase_key == "game.rps.win"
    assert game.round_number == 4


def test_play_after_the_end_raises():
    game, _ = play_out("LL")
    assert game.result().verdict == "lose"
    assert not game.committed  # nothing is drawn for a round that will never be played
    with pytest.raises(RuntimeError):
        game.play(R)


def test_result_before_the_end_raises():
    game = RockPaperScissors(random.Random(0))
    with pytest.raises(RuntimeError):
        game.result()


def test_accepts_move_names():
    game = RockPaperScissors(ScriptedRandom(choices=[S]))
    assert game.play("rock").outcome is Outcome.WIN
    with pytest.raises(ValueError):
        game.play("lizard")


def test_round_number():
    game = RockPaperScissors(ScriptedRandom(choices=[R, R]))
    assert game.round_number == 1
    game.play(R)
    assert game.round_number == 2


def test_rewards_are_in_the_documented_ranges():
    assert STARS[Outcome.WIN] > STARS[Outcome.DRAW] > STARS[Outcome.LOSE] > 0
    for effects in EFFECTS.values():
        assert -30 <= effects["boredom"] <= -15
        assert 5 <= effects["happiness"] <= 12
        assert 2 <= effects["friendship_xp"] <= 5


def test_invalid_parameters():
    with pytest.raises(ValueError):
        RockPaperScissors(random.Random(), wins_needed=0)
    with pytest.raises(ValueError):
        RockPaperScissors(random.Random(), max_rounds=0)
