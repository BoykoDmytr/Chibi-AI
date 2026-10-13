"""The mini-game windows (headless Qt): keyboard play, results, rewards,
and the cookie window's frame timer that runs only while it should."""

import random

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from chibi.games import GameResult
from chibi.games.cookie_window import FRAME_MS, TEXT_PAUSED, TEXT_READY, CookieWindow, format_clock
from chibi.games.cookies import FIELD_HEIGHT, GAME_SECONDS, Cookie, Popup
from chibi.games.guess import MAX_TRIES
from chibi.games.guess_window import GuessWindow
from chibi.games.rps import Move
from chibi.games.rps_window import RpsWindow
from tests.games.fakes import ScriptedRandom

R, S, P = Move.ROCK, Move.SCISSORS, Move.PAPER


def collect(window):
    results = []
    window.finished.connect(results.append)
    return results


# -- Камінь-ножиці-папір ----------------------------------------------------------------------


@pytest.fixture
def rps(track):
    def make(*her_moves, name="Мія"):
        window = track(RpsWindow(ScriptedRandom(choices=list(her_moves)), name=name))
        window.show()
        return window

    return make


def test_rps_keys_play_a_whole_game(rps):
    window = rps(S, S)
    results = collect(window)
    assert window.status_label.text() == "Мія уже загадала свій хід 🤫 Обирай!"
    assert window.score_label.text() == "Ти 0 : 0 Мія"
    QTest.keyClick(window, Qt.Key.Key_1)
    assert window.score_label.text() == "Ти 1 : 0 Мія"
    assert window.last_label.text() == "Ти: ✊ Камінь · Мія: ✌️ Ножиці — раунд твій!"
    assert window.round_label.text() == "Раунд 2"
    assert results == []
    QTest.keyClick(window, Qt.Key.Key_1)
    assert len(results) == 1
    (result,) = results
    assert isinstance(result, GameResult) and result.won and result.verdict == "win"
    assert window.status_label.text() == "Перемога твоя! 🎉"
    assert not any(b.isEnabled() for b in window.move_buttons.values())
    assert window.again_button.isVisible()
    QTest.keyClick(window, Qt.Key.Key_1)  # the game is over: ignored
    assert len(results) == 1


def test_rps_keys_two_and_three(rps):
    window = rps(R, R)
    QTest.keyClick(window, Qt.Key.Key_2)  # scissors vs rock
    assert window.game.her_wins == 1
    QTest.keyClick(window, Qt.Key.Key_3)  # paper vs rock
    assert window.game.user_wins == 1


def test_rps_buttons_and_key_propagation_from_a_focused_button(rps):
    window = rps(R, S)
    QTest.mouseClick(window.move_buttons[P], Qt.MouseButton.LeftButton)
    assert window.game.rounds[-1].user is P
    window.move_buttons[R].setFocus()
    QTest.keyClick(window.move_buttons[R], Qt.Key.Key_1)  # a focused button passes digits on
    assert len(window.game.rounds) == 2


def test_rps_reward_and_new_game_with_enter(rps):
    window = rps(P, P, name="Ліна")
    results = collect(window)
    QTest.keyClick(window, Qt.Key.Key_1)
    QTest.keyClick(window, Qt.Key.Key_1)
    assert results[0].verdict == "lose"
    assert window.status_label.text() == "Я перемогла! Але гра була класна."
    assert window.score_label.text() == "Ти 0 : 2 Ліна"
    window.show_reward("Нагорода: +3 ⭐")
    assert window.reward_label.text() == "Нагорода: +3 ⭐" and window.reward_label.isVisible()
    QTest.keyClick(window, Qt.Key.Key_Return)
    assert not window.game.rounds
    assert all(b.isEnabled() for b in window.move_buttons.values())
    assert not window.reward_label.isVisible()
    assert not window.again_button.isVisible()


def test_rps_enter_does_nothing_mid_game(rps):
    window = rps(R, R)
    QTest.keyClick(window, Qt.Key.Key_1)
    QTest.keyClick(window, Qt.Key.Key_Return)
    assert len(window.game.rounds) == 1


def test_rps_escape_closes(rps):
    window = rps()
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert not window.isVisible()


def test_rps_set_name(rps):
    window = rps()
    window.set_name("Ліна")
    assert window.status_label.text().startswith("Ліна")


# -- Вгадай число -----------------------------------------------------------------------------


@pytest.fixture
def guess(track):
    def make(secret):
        window = track(GuessWindow(ScriptedRandom(ints=[secret]), name="Мія"))
        window.show()
        return window

    return make


def type_guess(window, text):
    QTest.keyClicks(window.input, text)
    QTest.keyClick(window.input, Qt.Key.Key_Return)


def test_guess_with_enter(guess):
    window = guess(13)
    results = collect(window)
    assert "Мія загадала число від 1 до 20. У тебе 5 спроб." in window.intro_label.text()
    type_guess(window, "10")
    assert window.history_label.text().startswith("Твої спроби: 10 ↑")
    assert window.tries_label.text().endswith("залишилося: 4 спроби")
    assert window.input.text() == ""
    type_guess(window, "17")
    assert "10 ↑ · 17 ↓" in window.history_label.text()
    type_guess(window, "13")
    assert len(results) == 1 and results[0].won
    assert results[0].stars == 4 + 2 * (MAX_TRIES - 3)
    assert not window.input.isEnabled()
    assert window.again_button.isVisible()


def test_guess_input_takes_only_numbers_and_out_of_range_costs_nothing(guess):
    window = guess(13)
    QTest.keyClicks(window.input, "x7-")
    assert window.input.text() == "7"
    window.input.clear()
    type_guess(window, "25")
    assert window.game.tries_left == MAX_TRIES
    assert "від 1 до 20" in window.comment_label.text()


def test_guess_empty_enter_costs_nothing(guess):
    window = guess(13)
    QTest.keyClick(window.input, Qt.Key.Key_Return)
    assert window.game.tries_left == MAX_TRIES
    assert "Введи число" in window.comment_label.text()


def test_guess_loss_then_new_game_with_enter(guess):
    window = guess(20)
    results = collect(window)
    for n in ("1", "2", "3", "4", "5"):
        type_guess(window, n)
    assert len(results) == 1 and not results[0].won
    assert "я загадала 20" in window.comment_label.text()
    assert window.tries_label.text().endswith("залишилося: 0 спроб")
    window.show_reward("Нагорода: +2 ⭐")
    QTest.keyClick(window, Qt.Key.Key_Enter)
    assert window.game.tries_left == MAX_TRIES
    assert window.input.isEnabled()
    assert window.reward_label.text() == ""


def test_guess_check_button(guess):
    window = guess(4)
    QTest.keyClicks(window.input, "4")
    QTest.mouseClick(window.check_button, Qt.MouseButton.LeftButton)
    assert window.game.won


# -- Лови печиво ------------------------------------------------------------------------------


class FakeTime:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def cookies(track):
    def make(seed=1, show=True):
        clock = FakeTime()
        window = track(CookieWindow(random.Random(seed), now=clock))
        window.clock = clock
        if show:
            window.show()
        return window

    return make


def frames(window, seconds, fps=60):
    for _ in range(round(seconds * fps)):
        window.clock.t += 1 / fps
        window._frame()


def test_frame_rate_is_at_most_sixty():
    assert FRAME_MS >= 1000 / 60


def test_cookie_timer_runs_only_while_visible_and_playing(cookies):
    window = cookies(show=False)
    window.start()
    assert window.running and not window.timer_active  # not visible: no frames
    window.show()
    assert window.timer_active
    window.hide()
    assert not window.timer_active
    assert not window.running  # hiding pauses the game
    window.show()
    assert not window.timer_active  # back to a paused game, not a running one
    assert window.field.overlay == TEXT_READY  # nothing has happened yet


def test_cookie_space_starts_and_pauses(cookies):
    window = cookies()
    assert window.field.overlay == TEXT_READY
    assert window.start_button.text() == "Почати"
    QTest.keyClick(window, Qt.Key.Key_Space)
    assert window.running and window.timer_active
    assert window.field.overlay == ""
    assert window.start_button.text() == "Пауза"
    frames(window, 1)
    QTest.keyClick(window, Qt.Key.Key_Space)
    assert not window.running and not window.timer_active
    assert window.field.overlay == TEXT_PAUSED
    assert window.start_button.text() == "Продовжити"


def test_hidden_time_is_not_played(cookies):
    window = cookies()
    window.start()
    frames(window, 1)
    elapsed = window.game.elapsed
    window.hide()
    window.clock.t += 60  # a minute away
    window.show()
    window.start()
    window.clock.t += 1 / 60
    window._frame()
    assert window.game.elapsed == pytest.approx(elapsed + 1 / 60, abs=1e-6)


def test_minimising_pauses(cookies, qapp):
    window = cookies()
    window.start()
    window.setWindowState(Qt.WindowState.WindowMinimized)
    qapp.processEvents()
    if not window.isMinimized():
        pytest.skip("this platform plugin cannot minimise")
    assert not window.running


def test_cookie_labels_follow_the_game(cookies):
    window = cookies()
    assert window.time_label.text() == "⏱ 0:30"
    window.start()
    frames(window, 5.5)
    assert window.time_label.text() == "⏱ 0:25"
    assert window.score_label.text() == "Рахунок: 0"


def click_field(window, x, y):
    point = window.field.from_field(x, y)
    QTest.mouseClick(window.field, Qt.MouseButton.LeftButton, pos=QPoint(round(point.x()), round(point.y())))


def test_clicking_a_cookie_catches_it(cookies):
    window = cookies()
    window.start()
    frames(window, 0.1)
    window.game.cookies[:] = [Cookie(99, 200, 150, 0, False, 0)]
    click_field(window, 200, 150)
    assert window.game.score == 1
    assert window.score_label.text() == "Рахунок: 1"


def test_clicking_the_field_starts_the_game(cookies):
    window = cookies()
    click_field(window, 200, 150)
    assert window.running


def test_clicks_while_paused_do_not_count(cookies):
    window = cookies()
    window.start()
    window.pause()
    window.game.cookies[:] = [Cookie(99, 200, 150, 0, False, 0)]
    window._on_pressed(200, 150)  # resumes instead of catching
    assert window.running and window.game.score == 0


def test_arrow_keys_drive_the_basket(cookies):
    window = cookies()
    window.start()
    assert window.field.hasFocus() or not window.isActiveWindow()
    x = window.game.basket_x
    QTest.keyPress(window, Qt.Key.Key_Right)
    assert window.game.basket_active
    frames(window, 0.5)
    QTest.keyRelease(window, Qt.Key.Key_Right)
    moved = window.game.basket_x
    assert moved > x
    frames(window, 0.5)
    assert window.game.basket_x == moved  # released: stays
    QTest.keyPress(window, Qt.Key.Key_Left)
    QTest.keyPress(window, Qt.Key.Key_Right)
    QTest.keyRelease(window, Qt.Key.Key_Right)  # left is still held
    frames(window, 0.2)
    assert window.game.basket_x < moved


def test_arrow_keys_are_ignored_before_the_start(cookies):
    window = cookies()
    QTest.keyPress(window, Qt.Key.Key_Right)
    assert not window.game.basket_active


def test_finishing_emits_once_and_enter_restarts(cookies):
    window = cookies()
    results = collect(window)
    window.start()
    frames(window, GAME_SECONDS + 1)
    assert len(results) == 1
    assert results[0].game == "cookies"
    assert not window.running and not window.timer_active
    assert window.field.overlay.startswith("Час вийшов!")
    assert window.again_button.isVisible() and not window.start_button.isVisible()
    QTest.keyClick(window, Qt.Key.Key_Space)  # no restart by accident
    assert not window.running
    window.show_reward("Нагорода: +2 ⭐")
    QTest.keyClick(window, Qt.Key.Key_Return)
    assert window.game.elapsed == 0
    assert window.field.overlay == TEXT_READY
    assert window.reward_label.text() == ""
    assert len(results) == 1


def test_real_timer_moves_the_game(track, qapp):
    window = track(CookieWindow(random.Random(0)))
    window.show()
    window.start()
    QTest.qWait(120)
    assert 0 < window.game.elapsed < 1


def test_painting_everything(cookies):
    window = cookies()
    window.start()
    game = window.game
    game.cookies[:] = [Cookie(1, 100, 100, 50, False, 30), Cookie(2, 200, 120, 50, True, 200)]
    game.popups.append(Popup(150, 150, "+1", True, 0.2))
    game.popups.append(Popup(250, 150, "−2", False, 0.1))
    game.set_basket_direction(1)
    image = window.field.grab().toImage()
    assert not image.isNull()
    window.pause()
    assert not window.field.grab().toImage().isNull()  # with the overlay
    window.field.resize(900, 200)  # letterboxed
    point = window.field.from_field(0, 0)
    assert window.field.to_field(point) == pytest.approx((0, 0))
    corner = window.field.from_field(400, FIELD_HEIGHT)
    assert window.field.to_field(corner) == pytest.approx((400, FIELD_HEIGHT))


def test_click_outside_the_letterboxed_field_is_ignored(cookies):
    window = cookies()
    window.field.resize(900, 300)
    assert window.field.to_field(window.field.from_field(-10, 10)) is None


@pytest.mark.parametrize(("seconds", "text"), [(30, "0:30"), (29.2, "0:30"), (0.4, "0:01"), (0, "0:00"),
                                               (75, "1:15")])
def test_format_clock(seconds, text):
    assert format_clock(seconds) == text


def test_escape_closes(cookies):
    window = cookies()
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert not window.isVisible()
