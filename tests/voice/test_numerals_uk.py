import pytest

from chibi.voice.numerals_uk import cardinal, digits, ordinal, plural_index


@pytest.mark.parametrize(
    ("n", "gender", "case", "expected"),
    [
        (0, "m", "nom", "нуль"),
        (1, "m", "nom", "один"),
        (1, "f", "nom", "одна"),
        (1, "f", "acc", "одну"),
        (1, "n", "nom", "одне"),
        (2, "m", "nom", "два"),
        (2, "f", "nom", "дві"),
        (2, "n", "nom", "два"),
        (5, "f", "nom", "п'ять"),
        (11, "f", "nom", "одинадцять"),
        (12, "f", "nom", "дванадцять"),
        (21, "f", "nom", "двадцять одна"),
        (22, "f", "acc", "двадцять дві"),
        (40, "m", "nom", "сорок"),
        (99, "m", "nom", "дев'яносто дев'ять"),
        (101, "f", "nom", "сто одна"),
        (111, "m", "nom", "сто одинадцять"),
        (200, "m", "nom", "двісті"),
        (1000, "m", "nom", "тисяча"),
        (1001, "m", "nom", "тисяча один"),
        (2000, "m", "nom", "дві тисячі"),
        (5000, "m", "nom", "п'ять тисяч"),
        (21000, "m", "nom", "двадцять одна тисяча"),
        (2077, "m", "nom", "дві тисячі сімдесят сім"),
        (1_000_000, "m", "nom", "один мільйон"),
        (2_500_000, "m", "nom", "два мільйони п'ятсот тисяч"),
        (3_000_000_000, "m", "nom", "три мільярди"),
        (-5, "m", "nom", "мінус п'ять"),
    ],
)
def test_cardinal(n, gender, case, expected):
    assert cardinal(n, gender, case) == expected


def test_huge_numbers_are_read_digit_by_digit():
    assert cardinal(10**13) == "один " + " ".join(["нуль"] * 13)


@pytest.mark.parametrize(
    ("n", "gender", "case", "expected"),
    [
        (1, "m", "nom", "перший"),
        (2, "f", "nom", "друга"),
        (3, "m", "nom", "третій"),
        (3, "f", "nom", "третя"),
        (3, "n", "nom", "третє"),
        (3, "m", "gen", "третього"),
        (3, "f", "gen", "третьої"),
        (3, "f", "loc", "третій"),
        (7, "f", "acc", "сьому"),
        (14, "f", "nom", "чотирнадцята"),
        (14, "f", "loc", "чотирнадцятій"),
        (9, "f", "gen", "дев'ятої"),
        (20, "m", "nom", "двадцятий"),
        (21, "m", "nom", "двадцять перший"),
        (40, "m", "loc", "сороковому"),
        (100, "m", "nom", "сотий"),
        (125, "f", "gen", "сто двадцять п'ятої"),
        (300, "m", "nom", "трьохсотий"),
        (1000, "m", "nom", "тисячний"),
    ],
)
def test_ordinal(n, gender, case, expected):
    assert ordinal(n, gender, case) == expected


@pytest.mark.parametrize(
    ("n", "index"),
    [(1, 0), (21, 0), (101, 0), (11, 2), (2, 1), (4, 1), (22, 1), (12, 2), (14, 2), (5, 2),
     (0, 2), (111, 2), (112, 2), (-1, 0)],
)
def test_plural_index(n, index):
    assert plural_index(n) == index


def test_digits():
    assert digits("007") == "нуль нуль сім"
    assert digits("1-2") == "один два"
