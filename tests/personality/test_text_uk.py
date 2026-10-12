from datetime import datetime

import pytest

from chibi.personality.text_uk import (
    HOUR_FORMS,
    MINUTE_FORMS,
    format_clock,
    format_duration,
    normalize,
    plural,
    render,
    time_of_day,
)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, "годин"), (1, "годину"), (2, "години"), (4, "години"), (5, "годин"), (10, "годин"),
        (11, "годин"), (12, "годин"), (14, "годин"), (15, "годин"), (20, "годин"), (21, "годину"),
        (22, "години"), (24, "години"), (25, "годин"), (101, "годину"), (111, "годин"), (112, "годин"),
        (122, "години"), (1001, "годину"), (-1, "годину"), (-3, "години"),
        (2.5, "години"), (3.0, "години"), (float("nan"), "годин"),
    ],
)
def test_plural(n, expected):
    assert plural(n, HOUR_FORMS) == expected


def test_plural_needs_three_forms():
    with pytest.raises(ValueError):
        plural(1, ("годину", "години"))


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "менше хвилини"),
        (59.9, "менше хвилини"),
        (60, "1 хвилину"),
        (119, "1 хвилину"),
        (120, "2 хвилини"),
        (11 * 60, "11 хвилин"),
        (21 * 60, "21 хвилину"),
        (22 * 60, "22 хвилини"),
        (45 * 60, "45 хвилин"),
        (3600, "1 годину"),
        (3600 + 59, "1 годину"),
        (5400, "1 годину 30 хвилин"),
        (7200, "2 години"),
        (8100, "2 години 15 хвилин"),
        (5 * 3600 + 60, "5 годин 1 хвилину"),
        (21 * 3600, "21 годину"),
        (86400, "1 день"),
        (86400 + 3 * 3600 + 59 * 60, "1 день 3 години"),
        (2 * 86400, "2 дні"),
        (5 * 86400 + 3600, "5 днів 1 годину"),
    ],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


@pytest.mark.parametrize("bad", [-5, float("nan"), float("inf"), float("-inf"), "abc", None])
def test_format_duration_never_raises(bad):
    assert format_duration(bad) == "менше хвилини"


def test_duration_forms_read_after_vzhe():
    # Accusative forms: "вже 1 годину", not "вже 1 година".
    assert MINUTE_FORMS[0] == "хвилину"
    assert f"вже {format_duration(3600)}" == "вже 1 годину"


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [
        (0, 0, "night"), (4, 59, "night"), (5, 0, "morning"), (10, 59, "morning"), (11, 0, "day"),
        (16, 59, "day"), (17, 0, "evening"), (21, 59, "evening"), (22, 0, "night"), (23, 59, "night"),
    ],
)
def test_time_of_day(hour, minute, expected):
    assert time_of_day(datetime(2026, 3, 10, hour, minute)) == expected


def test_format_clock():
    assert format_clock(datetime(2026, 3, 10, 7, 5)) == "07:05"
    assert format_clock(datetime(2026, 3, 10, 23, 45)) == "23:45"


def test_render_fills_variables():
    assert render("Привіт, {user}! Я {name}.", {"user": "Котик", "name": "Мія"}) == "Привіт, Котик! Я Мія."


def test_render_missing_variable_raises_key_error():
    with pytest.raises(KeyError):
        render("О, {game}!", {"name": "Мія"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Привіт!!!", "привіт"),
        ("  Як   справи ?  ", "як справи"),
        ("Пам’ять", "пам'ять"),
        ("памʼять", "пам'ять"),
        ("пам`ять", "пам'ять"),
        ("пам‘ять", "пам'ять"),
        ("'привіт'", "привіт"),
        ("па-па", "па па"),
        ("Котра година?..", "котра година"),
        ("привiт", "привіт"),  # Latin "i" inside a Cyrillic word
        ("Hi there", "hi there"),  # Latin words stay Latin
        ("тримай,смаколик", "тримай смаколик"),
        ("", ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected
