"""Ukrainian numerals in words, for speech synthesis.

The TTS voice reads letters only, so every digit she is about to say becomes
words first.  Ukrainian numerals agree with the noun: "одна година", "одну
годину", "дві хвилини", "одне повідомлення", "двадцять перший раз",
"о чотирнадцятій".  Only what her lines need is covered: cardinals in the
nominative / accusative of all three genders, ordinals in the common cases.
"""

from __future__ import annotations

from typing import Literal

Gender = Literal["m", "f", "n"]
Case = Literal["nom", "acc", "gen", "dat", "loc"]

#: Above this numbers are read digit by digit (phone numbers, ids).
MAX_CARDINAL = 999_999_999_999

_UNITS = ("", "один", "два", "три", "чотири", "п'ять", "шість", "сім", "вісім", "дев'ять")
_TEENS = (
    "десять", "одинадцять", "дванадцять", "тринадцять", "чотирнадцять",
    "п'ятнадцять", "шістнадцять", "сімнадцять", "вісімнадцять", "дев'ятнадцять",
)
_TENS = ("", "", "двадцять", "тридцять", "сорок", "п'ятдесят", "шістдесят", "сімдесят",
         "вісімдесят", "дев'яносто")
_HUNDREDS = ("", "сто", "двісті", "триста", "чотириста", "п'ятсот", "шістсот", "сімсот",
             "вісімсот", "дев'ятсот")

_ONE = {("m", "nom"): "один", ("m", "acc"): "один",
        ("f", "nom"): "одна", ("f", "acc"): "одну",
        ("n", "nom"): "одне", ("n", "acc"): "одне"}
_TWO = {"m": "два", "f": "дві", "n": "два"}

#: (power, (one, few, many), gender of the scale word)
_SCALES = (
    (1_000_000_000, ("мільярд", "мільярди", "мільярдів"), "m"),
    (1_000_000, ("мільйон", "мільйони", "мільйонів"), "m"),
    (1_000, ("тисяча", "тисячі", "тисяч"), "f"),
)

# Masculine nominative ordinals.  Hard stems end in -ий, the soft one in -ій.
_ORD_UNITS = ("", "перший", "другий", "третій", "четвертий", "п'ятий", "шостий", "сьомий",
              "восьмий", "дев'ятий")
_ORD_TEENS = (
    "десятий", "одинадцятий", "дванадцятий", "тринадцятий", "чотирнадцятий",
    "п'ятнадцятий", "шістнадцятий", "сімнадцятий", "вісімнадцятий", "дев'ятнадцятий",
)
_ORD_TENS = ("", "", "двадцятий", "тридцятий", "сороковий", "п'ятдесятий", "шістдесятий",
             "сімдесятий", "вісімдесятий", "дев'яностий")
_ORD_HUNDREDS = ("", "сотий", "двохсотий", "трьохсотий", "чотирьохсотий", "п'ятисотий",
                 "шестисотий", "семисотий", "восьмисотий", "дев'ятисотий")

# Ending of a masculine nominative ordinal -> ending for (gender, case).
_HARD = {
    ("m", "nom"): "ий", ("m", "acc"): "ий", ("m", "gen"): "ого", ("m", "dat"): "ому",
    ("m", "loc"): "ому",
    ("n", "nom"): "е", ("n", "acc"): "е", ("n", "gen"): "ого", ("n", "dat"): "ому",
    ("n", "loc"): "ому",
    ("f", "nom"): "а", ("f", "acc"): "у", ("f", "gen"): "ої", ("f", "dat"): "ій",
    ("f", "loc"): "ій",
}
_SOFT = {
    ("m", "nom"): "ій", ("m", "acc"): "ій", ("m", "gen"): "ього", ("m", "dat"): "ьому",
    ("m", "loc"): "ьому",
    ("n", "nom"): "є", ("n", "acc"): "є", ("n", "gen"): "ього", ("n", "dat"): "ьому",
    ("n", "loc"): "ьому",
    ("f", "nom"): "я", ("f", "acc"): "ю", ("f", "gen"): "ьої", ("f", "dat"): "ій",
    ("f", "loc"): "ій",
}

DIGIT_NAMES = ("нуль", "один", "два", "три", "чотири", "п'ять", "шість", "сім", "вісім",
               "дев'ять")


def plural_index(n: int) -> int:
    """0 = one ("година"), 1 = few ("години"), 2 = many ("годин")."""
    n = abs(n)
    if n % 10 == 1 and n % 100 != 11:
        return 0
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return 1
    return 2


def _below_thousand(n: int, gender: Gender, case: Case) -> list[str]:
    words: list[str] = []
    if n >= 100:
        words.append(_HUNDREDS[n // 100])
        n %= 100
    if 10 <= n <= 19:
        words.append(_TEENS[n - 10])
        return words
    if n >= 20:
        words.append(_TENS[n // 10])
        n %= 10
    if n == 1:
        words.append(_ONE[(gender, "acc" if case == "acc" else "nom")])
    elif n == 2:
        words.append(_TWO[gender])
    elif n:
        words.append(_UNITS[n])
    return words


def cardinal(n: int, gender: Gender = "m", case: Case = "nom") -> str:
    """``21, "f"`` -> "двадцять одна"; ``1, "f", "acc"`` -> "одну"; ``2000`` -> "дві тисячі".

    Only the nominative and accusative are inflected (enough for "N
    хвилин/годин/разів" lines); other cases fall back to the nominative.
    """
    if n < 0:
        return "мінус " + cardinal(-n, gender, case)
    if n == 0:
        return "нуль"
    if n > MAX_CARDINAL:
        return digits(str(n))
    words: list[str] = []
    for power, forms, scale_gender in _SCALES:
        count, n = divmod(n, power)
        if not count:
            continue
        if count == 1 and power == 1_000:
            words.append(forms[0])  # "тисяча", not "одна тисяча"
        else:
            words.extend(_below_thousand(count, scale_gender, "nom"))  # type: ignore[arg-type]
            words.append(forms[plural_index(count)])
    if n:
        words.extend(_below_thousand(n, gender, case))
    return " ".join(words)


def ordinal(n: int, gender: Gender = "m", case: Case = "nom") -> str:
    """``3, "f", "loc"`` -> "третій" (о третій); ``21`` -> "двадцять перший";
    ``14, "f"`` -> "чотирнадцята".  Thousands and above: "тисячний", otherwise
    the cardinal is used (years in titles are read as cardinals anyway)."""
    if n <= 0:
        return cardinal(n, gender, case)
    if n >= 1000:
        if n == 1000:
            return _inflect("тисячний", gender, case)
        return cardinal(n, gender, case)
    prefix: list[str] = []
    rest = n
    if rest >= 100 and rest % 100:
        prefix.append(_HUNDREDS[rest // 100])
        rest %= 100
    if rest >= 100:
        last = _ORD_HUNDREDS[rest // 100]
    elif 10 <= rest <= 19:
        last = _ORD_TEENS[rest - 10]
    elif rest % 10 == 0:
        last = _ORD_TENS[rest // 10]
    else:
        if rest >= 20:
            prefix.append(_TENS[rest // 10])
        last = _ORD_UNITS[rest % 10]
    return " ".join([*prefix, _inflect(last, gender, case)])


def _inflect(masculine_nominative: str, gender: Gender, case: Case) -> str:
    if masculine_nominative.endswith("ій"):
        return masculine_nominative[:-2] + _SOFT[(gender, case)]
    return masculine_nominative[:-2] + _HARD[(gender, case)]


def digits(text: str) -> str:
    """Digit by digit: "007" -> "нуль нуль сім"."""
    return " ".join(DIGIT_NAMES[int(ch)] for ch in text if ch.isdigit())
