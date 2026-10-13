"""looks_sensitive(): personal data that must never be stored as a memory."""

import time

import pytest

from chibi.memory.longterm import looks_sensitive

CARDS = [
    "моя картка 4111 1111 1111 1111",
    "4111-1111-1111-1111",
    "картка 5555555555554444, термін 12/27",
    "amex 378282246310005",
    "Discover 6011 1111 1111 1117",
    "номер карти: 4000 0566 5566 5556 - запам'ятай",
    # a card hidden among other numbers
    "переказ 200 грн на 4111 1111 1111 1111 22 числа",
    # smuggling attempts: fullwidth digits, zero-width spaces, NBSP
    "４１１１ １１１１ １１１１ １１１１",
    "4111​1111​1111​1111",
    "4111 1111 1111 1111",
]

IBANS = [
    "UA213223130000026007233566001",
    "мій рахунок UA21 3223 1300 0002 6007 2335 6600 1",
    "ua213223130000026007233566001",
    # Ukrainian format is flagged even with a typo in the checksum
    "UA213223130000026007233566002",
    "DE89 3704 0044 0532 0130 00",
    "IBAN: GB82 WEST 1234 5698 7654 32 (UK)",
    "PL61109010140000071219812874",
]

SECRETS = [
    "мій пароль: qwerty123",
    "пароль від wifi — kotik2020",
    "Password: hunter2",
    "my password is swordfish",
    "пароль kotik2020",
    "мій новий пароль — Kotyk_2020!",
    "пароль: котик",
    "пароль=Pa$$w0rd",
    "пін-код 4321",
    "мій пін-код від картки 0000",
    "PIN 0000",
    "pin code: 7788",
    "пін 1234",
    "cvv 123",
    "CVV2: 456",
    "cvc 999",
    "код з смс 482913",
    "код підтвердження: 5521",
    "OTP 993311",
]

EMAILS = [
    "пиши на roman.test@gmail.com",
    "мій імейл ivan_petrenko+chibi@ukr.net",
    "ПОШТА: Olena.K@Example.COM.UA",
    "адреса олена@приклад.укр",
]

PHONES = [
    "+380 67 123 45 67",
    "+380671234567",
    "380671234567",
    "+38 (067) 123 45 67",
    "067 123 45 67",
    "(067) 123-45-67",
    "0671234567",
    "мій номер 050-123-45-67, дзвони",
    "+1 (555) 123-4567",
    "+44 20 7946 0958",
]

API_KEYS = [
    "sk-ant-api03-AbCdEf1234567890xyzXYZ",
    "мій ключ sk-proj-abcdefghijklmnop1234",
    "ghp_aBcD1234eFgH5678iJkL9012mNoP3456qRsT",
    "github_pat_11ABCDEFG0123456789_abcdefghijklmnop",
    "AKIAIOSFODNN7EXAMPLE",
    "AIzaSyA1234567890abcdefghijklmnopqrstu",
    "xoxb-1234567890-abcdefghij",
    "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "dGhpcyBpcyBhIHNlY3JldCBrZXkgZm9yIHRlc3Rpbmc",
    "token a8f3k2j4l5m6n7b8v9c0x1z2q3w4e5r6t7y8",
    "550e8400-e29b-41d4-a716-446655440000",
]


@pytest.mark.parametrize("text", CARDS + IBANS + SECRETS + EMAILS + PHONES + API_KEYS)
def test_sensitive_text_is_detected(text):
    assert looks_sensitive(text), text


NOT_SENSITIVE = [
    "Я люблю S.T.A.L.K.E.R. 2",
    "мені 25 років",
    "Мого кота звати Мурчик",
    "Я народилася 12.05.1999",
    "Граю в Cyberpunk 2077 вже 300 годин",
    "Я забула пароль від пошти",
    "треба змінити пароль",
    "Пароль надто простий",
    "менеджер паролів KeePass",
    "паролі від усіх акаунтів зберігає в KeePass",
    "Я пишу код на Python 3.12",
    "Користувач пише код на Python",
    "коли код працює, я щаслива",
    "Код да Вінчі — класна книга",
    "my favourite code editor is VS Code",
    "PIN-ап стиль мені подобається",
    "ping 25 ms",
    "Мій улюблений номер — 7",
    "Я працюю з 9 до 18",
    "Рахунок матчу 3:2",
    "Купила RTX 4090 за 80000 гривень",
    "Чекаю на GTA 6 у 2026 році",
    "I love pizza with pineapple",
    "My cat's name is Whiskers",
    "у мене 2 коти і 1 собака",
    "Level 100 у Diablo 4",
    "Читаю книгу 1984",
    "Ціна 1 299 грн",
    "Мій зріст 170 см, вага 60 кг",
    "Скоро 2025-12-31",
    "Steam level 42, 1500 hours in Dota 2",
    "Мій день народження 14 лютого",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "@mia_chibi у телеграмі",
    "у мене 1 000 000 гривень на мрію",
    "Граю 4-5 годин на день",
    "Збираю Lego 75192 Millennium Falcon",
    "state-of-the-art-machine-learning-research",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    # Luhn-invalid 16 digits and a broken generic IBAN checksum are not secrets
    "4111 1111 1111 1112",
    "DE88 3704 0044 0532 0130 00",
    "",
    "   ",
]


@pytest.mark.parametrize("text", NOT_SENSITIVE)
def test_everyday_text_is_not_sensitive(text):
    assert not looks_sensitive(text), text


def test_non_string_input_is_not_sensitive():
    assert not looks_sensitive(None)  # type: ignore[arg-type]
    assert not looks_sensitive(4111111111111111)  # type: ignore[arg-type]


def test_short_number_groups_are_not_a_card():
    # 16 digits that pass Luhn but are printed as small separate numbers.
    assert not looks_sensitive("1 2 3 4 5 6 7 8 9 1 2 3 4 5 6 3")


@pytest.mark.parametrize("text", [
    "a" * 50_000,
    "a." * 25_000,
    "1" * 50_000,
    "1" * 50_000 + "@",
    "a" * 50_000 + "@x",
    "1 " * 25_000,
    "пароль від " * 5_000,
    "+1 " * 15_000,
], ids=["letters", "dots", "digits", "digits-at", "letters-at", "spaced-digits", "password-words",
        "phone-prefixes"])  # short ids: pytest puts the id into an env var (32 KB max on Windows)
def test_detection_is_linear_on_pathological_input(text):
    # A backtracking pattern turns these into seconds (quadratic); linear checks take milliseconds.
    started = time.perf_counter()
    looks_sensitive(text)
    assert time.perf_counter() - started < 1.5
