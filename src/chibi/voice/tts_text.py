"""Text normalisation for the Ukrainian TTS voice.

The Piper voice (``uk_UA-ukrainian_tts``) reads lower-case Ukrainian letters
and a few punctuation marks *only*: an upper-case letter, a digit or a Latin
word is silently dropped ("Тільки" came out as "ільки" in the round-trip
test).  Everything she says therefore goes through :func:`normalize_for_tts`:

* typography: every apostrophe variant -> ``'``, quotes and emoji dropped,
  dashes unified, ``…`` -> ``...``, role-play ``*зітхає*`` removed;
* names: game titles, apps and acronyms from ``data/pronunciations_uk.json``
  ("VS Code" -> "ві ес код", "GTA" -> "джі ті ей"), Roman numerals in titles
  ("Diablo IV" -> "діабло чотири");
* numbers in words, agreeing with the noun that follows ("1 годину" ->
  "одну годину", "2 хвилини" -> "дві хвилини", "о 14:30" -> "о
  чотирнадцятій тридцять", "50%" -> "п'ятдесят відсотків", "3-й" ->
  "третій");
* any other Latin word is transliterated by rule, short upper-case words
  are spelled letter by letter;
* finally lower case, and every character outside the voice's alphabet
  removed (``^ _ $`` are the voice's control symbols: never pass them).

:class:`SentenceChunker` cuts a streamed reply into sentences so the first
one can be synthesised while the model is still writing the rest.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

from chibi.voice.numerals_uk import Case, Gender, cardinal, digits, ordinal, plural_index

#: Characters the voice can pronounce (besides lower-case Ukrainian letters).
VOICE_PUNCTUATION = " !',-.:;?—"
#: Combining acute accent: an optional stress mark the voice understands.
STRESS_MARK = "́"
UK_LETTERS = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
ALLOWED = frozenset(UK_LETTERS + VOICE_PUNCTUATION + STRESS_MARK)

_APOSTROPHES = "’ʼ‘`´′ʹ"
_RUSSIAN = str.maketrans({"ы": "и", "э": "е", "ъ": "'", "ё": "йо"})

# -- lexicon ------------------------------------------------------------------------------


@dataclass
class PronunciationLexicon:
    """How Latin names, words and acronyms are said.  Keys are matched
    case-insensitively (phrases, words) or exactly (acronyms)."""

    phrases: dict[str, str] = field(default_factory=dict)
    words: dict[str, str] = field(default_factory=dict)
    acronyms: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.phrases = {k.lower(): v for k, v in self.phrases.items()}
        self.words = {k.lower(): v for k, v in self.words.items()}
        self._phrase_re: re.Pattern[str] | None = None

    def add_phrase(self, phrase: str, say: str) -> None:
        phrase = phrase.strip().lower()
        if phrase and say.strip():
            self.phrases[phrase] = say.strip()
            self._phrase_re = None

    def phrase_pattern(self) -> re.Pattern[str] | None:
        if self._phrase_re is None and self.phrases:
            alternatives = sorted(self.phrases, key=len, reverse=True)
            body = "|".join(re.escape(p) for p in alternatives)
            # Not glued to other letters/digits on either side.
            self._phrase_re = re.compile(rf"(?<![\w])(?:{body})(?![\w])", re.IGNORECASE)
        return self._phrase_re


@lru_cache(maxsize=1)
def _builtin_data() -> dict:
    text = resources.files("chibi.voice").joinpath("data/pronunciations_uk.json").read_text("utf-8")
    return json.loads(text)


def default_lexicon() -> PronunciationLexicon:
    """A fresh copy of the built-in lexicon (callers may add game titles to it)."""
    data = _builtin_data()
    return PronunciationLexicon(
        phrases=dict(data.get("phrases", {})),
        words=dict(data.get("words", {})),
        acronyms=dict(data.get("acronyms", {})),
    )


# -- units after numbers -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Unit:
    gender: Gender
    forms: tuple[str, str, str]  # one, few, many (nominative)
    gen_sg: str  # after fractions: "2,5 відсотка"


_UNITS: dict[str, _Unit] = {
    "хв": _Unit("f", ("хвилина", "хвилини", "хвилин"), "хвилини"),
    "год": _Unit("f", ("година", "години", "годин"), "години"),
    "сек": _Unit("f", ("секунда", "секунди", "секунд"), "секунди"),
    "с": _Unit("f", ("секунда", "секунди", "секунд"), "секунди"),
    "грн": _Unit("f", ("гривня", "гривні", "гривень"), "гривні"),
    "км": _Unit("m", ("кілометр", "кілометри", "кілометрів"), "кілометра"),
    "кг": _Unit("m", ("кілограм", "кілограми", "кілограмів"), "кілограма"),
    "гб": _Unit("m", ("гігабайт", "гігабайти", "гігабайтів"), "гігабайта"),
    "gb": _Unit("m", ("гігабайт", "гігабайти", "гігабайтів"), "гігабайта"),
    "мб": _Unit("m", ("мегабайт", "мегабайти", "мегабайтів"), "мегабайта"),
    "mb": _Unit("m", ("мегабайт", "мегабайти", "мегабайтів"), "мегабайта"),
    "fps": _Unit("m", ("кадр на секунду", "кадри на секунду", "кадрів на секунду"),
                 "кадру на секунду"),
    "фпс": _Unit("m", ("кадр на секунду", "кадри на секунду", "кадрів на секунду"),
                 "кадру на секунду"),
    "тис": _Unit("f", ("тисяча", "тисячі", "тисяч"), "тисячі"),
    "млн": _Unit("m", ("мільйон", "мільйони", "мільйонів"), "мільйона"),
    "млрд": _Unit("m", ("мільярд", "мільярди", "мільярдів"), "мільярда"),
    "%": _Unit("m", ("відсоток", "відсотки", "відсотків"), "відсотка"),
    "°": _Unit("m", ("градус", "градуси", "градусів"), "градуса"),
    "$": _Unit("m", ("долар", "долари", "доларів"), "долара"),
    "€": _Unit("n", ("євро", "євро", "євро"), "євро"),
}

# Nouns that make a preceding "1"/"2" feminine ("одна година", "дві гри") or
# neuter ("одне повідомлення").  Stems are matched as word prefixes.
_FEMININE_STEMS = (
    "годин", "хвилин", "секунд", "гривн", "гривен", "доб", "тисяч", "катк", "каток",
    "перемог", "поразк", "поразок", "смерт", "спроб", "сесі", "ноч", "ніч", "книг",
    "книжк", "книжок", "сері", "місі", "задач", "зірк", "зірок", "монет", "карт",
    "частин", "сторінк", "сторінок", "причин", "цукерк", "цукерок", "печеньк", "кав",
    "хвилинк", "хвилинок", "годинк", "годинок", "чашк", "чашок", "парті", "битв",
    "людин", "сотн", "пісн", "пісень", "серій",
)
_FEMININE_WORDS = frozenset({"гра", "гри", "гру", "ігор", "ігри", "ігрові"})
_NEUTER_WORDS = frozenset({
    "завдання", "завдань", "повідомлення", "повідомлень", "життя", "життів", "вікно",
    "вікна", "вікон", "місце", "місця", "місць", "очко", "очка", "очок", "яблуко",
    "яблука", "яблук", "печиво", "печива", "слово", "слова", "слів", "відео", "коло",
    "кола", "речення", "запитання", "питання", "бажання", "досягнення", "досягнень",
    "сердечко", "сердечка", "сердечок", "печенько", "печенька", "печеньок",
})


def _gender_of(word: str) -> Gender:
    word = word.lower()
    if word in _NEUTER_WORDS:
        return "n"
    if word in _FEMININE_WORDS or word.startswith(_FEMININE_STEMS):
        return "f"
    return "m"


def _case_of_noun(word: str, gender: Gender) -> Case:
    # Feminine accusative singular ("годину", "хвилинку", "гру") needs "одну".
    if gender == "f" and word.lower().endswith(("у", "ю")):
        return "acc"
    return "nom"


# -- regexes ----------------------------------------------------------------------------

_ACTION = re.compile(r"\*[^*\n]{1,60}\*")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_DOTTED_ACRONYM = re.compile(r"(?<![\w.])((?:[A-Za-z]\.){2,}[A-Za-z]?)\.?(?![\w])")
_LETTER_DIGIT = re.compile(r"(?<=[A-Za-zА-ЯІЇЄҐа-яіїєґ])(?=\d)|(?<=\d)(?=[A-Za-z])")
_ROMAN = re.compile(r"(?<=[A-Za-z'’:]) +(X{0,3}(?:IX|IV|V?I{0,3}))(?![\w])")
_TIME = re.compile(r"(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])")
_PREPOSITION_BEFORE = re.compile(r"(?:^|\s)([А-Яа-яІіЇїЄєҐґ]+)\s*$")
_CURRENCY_PREFIX = re.compile(r"([$€])\s?(\d[\d  ]*(?:[.,]\d+)?)")
_NUMBER = re.compile(
    r"(?<![\w.,])(?P<neg>[-−])?(?P<int>\d{1,3}(?:[  ]\d{3})+|\d+)"
    r"(?:(?P<sep>[.,])(?P<frac>\d+))?"
    r"(?:-(?P<ord>[а-яіїєґ]{1,4})\b)?"
)
_NEXT_WORD = re.compile(r"\s*([%°$€]|[^\W\d_]+\.?)", re.UNICODE)
_LATIN_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_CYR_UPPER_WORD = re.compile(r"(?<![\w])[А-ЯІЇЄҐ]{2,5}(?![\w])")
_SPACES = re.compile(r"\s+")

# Prepositions that put a following clock time in a given case of the
# feminine ordinal ("о чотирнадцятій", "до чотирнадцятої").
_TIME_CASE: dict[str, Case] = {
    "о": "loc", "об": "loc", "по": "loc",
    "до": "gen", "після": "gen", "з": "gen", "із": "gen", "від": "gen", "близько": "gen",
    "біля": "gen", "на": "acc", "в": "acc", "у": "acc", "за": "acc",
}

_ORDINAL_SUFFIX: dict[str, tuple[Gender, Case]] = {}
for _suffix, _form in {
    "й": ("m", "nom"), "ий": ("m", "nom"), "ій": ("m", "nom"),
    "го": ("m", "gen"), "ого": ("m", "gen"), "ього": ("m", "gen"),
    "му": ("m", "loc"), "ому": ("m", "loc"), "ьому": ("m", "loc"),
    "а": ("f", "nom"), "я": ("f", "nom"), "та": ("f", "nom"), "ша": ("f", "nom"),
    "га": ("f", "nom"), "тя": ("f", "nom"), "ма": ("f", "nom"),
    "ї": ("f", "gen"), "ої": ("f", "gen"), "ьої": ("f", "gen"),
    "у": ("f", "acc"), "ю": ("f", "acc"),
    "е": ("n", "nom"), "є": ("n", "nom"),
}.items():
    _ORDINAL_SUFFIX[_suffix] = _form  # type: ignore[assignment]

# -- Latin ------------------------------------------------------------------------------

_LATIN_LETTERS = {
    "a": "ей", "b": "бі", "c": "сі", "d": "ді", "e": "і", "f": "еф", "g": "джі", "h": "ейч",
    "i": "ай", "j": "джей", "k": "кей", "l": "ел", "m": "ем", "n": "ен", "o": "о", "p": "пі",
    "q": "к'ю", "r": "ар", "s": "ес", "t": "ті", "u": "ю", "v": "ві", "w": "дабл ю",
    "x": "екс", "y": "вай", "z": "зет",
}
_CYRILLIC_LETTERS = {
    "а": "а", "б": "бе", "в": "ве", "г": "ге", "ґ": "ґе", "д": "де", "е": "е", "є": "є",
    "ж": "же", "з": "зе", "и": "и", "і": "і", "ї": "ї", "й": "йот", "к": "ка", "л": "ел",
    "м": "ем", "н": "ен", "о": "о", "п": "пе", "р": "ер", "с": "ес", "т": "те", "у": "у",
    "ф": "еф", "х": "ха", "ц": "це", "ч": "че", "ш": "ша", "щ": "ща", "ь": "м'який знак",
    "ю": "ю", "я": "я",
}
_VOWELS = set("aeiouy")
# Multi-letter English spellings, longest first (rule-based fallback only).
_GRAPHEMES = (
    ("tion", "шн"), ("sion", "жн"), ("ture", "чер"), ("ight", "айт"), ("tch", "ч"),
    ("igh", "ай"), ("sch", "ск"), ("sh", "ш"), ("ch", "ч"), ("ph", "ф"), ("th", "т"),
    ("ck", "к"), ("gh", "г"), ("qu", "кв"), ("kh", "х"), ("zh", "ж"), ("ts", "ц"), ("wh", "в"),
    ("ee", "і"), ("ea", "і"), ("oo", "у"), ("ou", "ау"), ("ow", "оу"), ("ai", "ей"),
    ("ay", "ей"), ("ey", "ей"), ("oa", "оу"), ("oi", "ой"), ("oy", "ой"), ("au", "о"),
    ("aw", "о"), ("ew", "ю"), ("ue", "ю"), ("ya", "я"), ("yo", "йо"), ("yu", "ю"),
    ("ye", "є"), ("ng", "нг"),
)
_CONSONANTS = {
    "b": "б", "d": "д", "f": "ф", "g": "г", "h": "г", "j": "дж", "k": "к", "l": "л",
    "m": "м", "n": "н", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "v": "в",
    "w": "в", "x": "кс", "z": "з",
}


def spell_latin(word: str) -> str:
    return " ".join(_LATIN_LETTERS.get(ch, "") for ch in word.lower() if ch in _LATIN_LETTERS)


def spell_cyrillic(word: str) -> str:
    return " ".join(_CYRILLIC_LETTERS.get(ch, ch) for ch in word.lower())


def transliterate_latin(word: str) -> str:
    """Rough English-style reading of an unknown Latin word in Cyrillic.

    Good enough to be understood ("Raft" -> "рафт", "Ghostrunner" ->
    "гострунер"); names that matter belong in the lexicon instead.
    """
    w = word.lower().replace("'", "")
    if not w:
        return ""
    # Silent final "e" lengthens the vowel before the last consonant ("game", "line").
    magic = {"a": "ей", "i": "ай", "o": "оу", "u": "ю", "e": "і"}
    out: list[str] = []
    i = 0
    n = len(w)
    while i < n:
        if (i == n - 3 and w[-1] == "e" and w[i] in magic and w[i + 1] not in _VOWELS
                and n > 3):
            out.append(magic[w[i]] + _CONSONANTS.get(w[i + 1], ""))
            break
        if i == 0 and w.startswith(("kn", "wr")):
            out.append("н" if w[0] == "k" else "р")
            i += 2
            continue
        for spelling, sound in _GRAPHEMES:
            if w.startswith(spelling, i):
                out.append(sound)
                i += len(spelling)
                break
        else:
            ch = w[i]
            nxt = w[i + 1] if i + 1 < n else ""
            if i + 1 < n and nxt == ch and ch not in _VOWELS:
                i += 1  # doubled consonant reads as one
                continue
            if ch == "c":
                out.append("с" if nxt in ("e", "i", "y") else "к")
            elif ch == "y":
                out.append("й" if i == 0 else "і")
            elif ch == "a":
                out.append("а")
            elif ch == "e":
                out.append("" if i == n - 1 and n > 2 else "е")
            elif ch == "i":
                out.append("і")
            elif ch == "o":
                out.append("о")
            elif ch == "u":
                rest = w[i + 1:]
                closed = (len(rest) >= 2 and rest[0] not in _VOWELS and rest[1] not in _VOWELS) \
                    or (len(rest) == 1 and rest not in _VOWELS)
                out.append("а" if closed else "у")
            else:
                out.append(_CONSONANTS.get(ch, ""))
            i += 1
    return "".join(out)


def _roman_value(token: str) -> int | None:
    values = {"I": 1, "V": 5, "X": 10}
    total = 0
    for idx, ch in enumerate(token):
        v = values[ch]
        if idx + 1 < len(token) and values[token[idx + 1]] > v:
            total -= v
        else:
            total += v
    return total if 1 <= total <= 39 else None


# -- the pipeline -------------------------------------------------------------------------


def normalize_for_tts(text: str, lexicon: PronunciationLexicon | None = None) -> str:
    """Return *text* as the Ukrainian voice can read it (see module docs).

    The result contains only lower-case Ukrainian letters, ``'``, spaces and
    ``!,-.:;?—``; it ends with a sentence mark when it has any letters.
    """
    if not text:
        return ""
    lex = lexicon if lexicon is not None else _shared_lexicon()
    s = unicodedata.normalize("NFC", text)
    s = _clean_typography(s)
    s = _apply_phrases(s, lex)
    s = _DOTTED_ACRONYM.sub(lambda m: m.group(1).replace(".", ""), s)
    s = _apply_roman(s)
    s = _apply_abbreviations(s)
    s = _apply_time(s)
    s = _CURRENCY_PREFIX.sub(lambda m: f"{m.group(2)} {m.group(1)}", s)
    s = _LETTER_DIGIT.sub(" ", s)
    s = _apply_numbers(s)
    s = _apply_latin(s, lex)
    s = _apply_cyrillic_acronyms(s, lex)
    return _finish(s)


@lru_cache(maxsize=1)
def _shared_lexicon() -> PronunciationLexicon:
    return default_lexicon()


def _clean_typography(s: str) -> str:
    s = _ACTION.sub(" ", s)
    s = _URL.sub(" посилання ", s)
    s = _EMAIL.sub(" ", s)
    for ch in _APOSTROPHES:
        s = s.replace(ch, "'")
    s = s.replace("…", "...")
    for ch in "«»„“”\"":
        s = s.replace(ch, " ")
    for ch in "–−―":
        s = s.replace(ch, " — ")
    s = re.sub(r"(?<=\s)-(?=\s)", " — ", s)
    s = re.sub(r"[()\[\]{}]", ", ", s)
    s = s.replace("&", " і ").replace("+", " плюс ").replace("№", " номер ").replace("=", " дорівнює ")
    return s


def _apply_phrases(s: str, lex: PronunciationLexicon) -> str:
    pattern = lex.phrase_pattern()
    if pattern is None:
        return s
    return pattern.sub(lambda m: f" {lex.phrases[m.group(0).lower()]} ", s)


def _apply_roman(s: str) -> str:
    def repl(m: re.Match[str]) -> str:
        token = m.group(1)
        value = _roman_value(token) if token else None
        if value is None or token == "I":  # a lone "I" is more likely English "I"
            return m.group(0)
        return " " + cardinal(value)

    return _ROMAN.sub(repl, s)


_ABBREVIATIONS = (
    (re.compile(r"\bі т\.\s?д\.", re.IGNORECASE), "і так далі"),
    (re.compile(r"\bі т\.\s?п\.", re.IGNORECASE), "і тому подібне"),
    (re.compile(r"\bт\.\s?зв\.", re.IGNORECASE), "так званий"),
    (re.compile(r"\bнапр\.", re.IGNORECASE), "наприклад"),
    (re.compile(r"\bвул\.", re.IGNORECASE), "вулиця"),
    (re.compile(r"\bр\.\s?р\.", re.IGNORECASE), "роки"),
)


def _apply_abbreviations(s: str) -> str:
    for pattern, replacement in _ABBREVIATIONS:
        s = pattern.sub(replacement, s)
    return s


def _apply_time(s: str) -> str:
    def repl(m: re.Match[str]) -> str:
        hour, minute = int(m.group(1)), m.group(2)
        before = _PREPOSITION_BEFORE.search(s[: m.start()])
        case: Case = _TIME_CASE.get(before.group(1).lower(), "nom") if before else "nom"
        spoken_hour = ordinal(hour, "f", case) if hour else "нуль"
        if minute == "00":
            spoken_minute = "рівно" if case == "nom" else ""
        elif minute.startswith("0"):
            spoken_minute = "нуль " + cardinal(int(minute), "f")
        else:
            spoken_minute = cardinal(int(minute), "f")
        return f"{spoken_hour} {spoken_minute}".strip()

    s = _TIME.sub(repl, s)
    # A score or a ratio ("3:2") is read with a dash.
    return re.sub(r"(?<=\d):(?=\d)", " — ", s)


def _apply_numbers(s: str) -> str:
    out: list[str] = []
    pos = 0
    for m in _NUMBER.finditer(s):
        out.append(s[pos:m.start()])
        spoken, consumed = _speak_number(m, s)
        out.append(f" {spoken} ")
        pos = max(m.end(), consumed)
    out.append(s[pos:])
    return "".join(out)


def _speak_number(m: re.Match[str], s: str) -> tuple[str, int]:
    """Words for one number match, and where the consumed text ends (a unit
    abbreviation after the number is replaced together with it)."""
    raw_int = re.sub(r"[  ]", "", m.group("int"))
    negative = bool(m.group("neg"))
    frac = m.group("frac")
    if m.group("sep") == "." and frac and len(frac) == 3 and not negative and len(raw_int) <= 3:
        # "1.500" is a thousands separator in Ukrainian texts, not a fraction.
        raw_int, frac = raw_int + frac, None
    prefix = "мінус " if negative else ""
    # Leading zeros ("007") or huge numbers: read digit by digit.
    if (len(raw_int) > 1 and raw_int.startswith("0")) or len(raw_int) > 12:
        return prefix + digits(raw_int), m.end()
    value = int(raw_int)

    ord_suffix = m.group("ord")
    if ord_suffix:
        gender_case = _ORDINAL_SUFFIX.get(ord_suffix)
        if gender_case is not None:
            return prefix + ordinal(value, *gender_case), m.end()

    nxt = _NEXT_WORD.match(s, m.end())
    unit = None
    unit_end = m.end()
    next_word = ""
    if nxt:
        token = nxt.group(1)
        key = token.lower().rstrip(".")
        if key in _UNITS:
            unit = _UNITS[key]
            unit_end = nxt.end()
        else:
            next_word = token.rstrip(".")

    if frac:
        gender: Gender = unit.gender if unit else _gender_of(next_word)
        words = f"{cardinal(value, gender)} кома {_fraction_digits(frac)}"
        if unit:
            words += " " + unit.gen_sg
        return prefix + words, unit_end

    if unit:
        return prefix + f"{cardinal(value, unit.gender)} {unit.forms[plural_index(value)]}", unit_end
    gender = _gender_of(next_word) if next_word else "m"
    case = _case_of_noun(next_word, gender) if next_word else "nom"
    return prefix + cardinal(value, gender, case), m.end()


def _fraction_digits(frac: str) -> str:
    if frac.startswith("0"):
        return digits(frac)
    return cardinal(int(frac))


def _apply_latin(s: str, lex: PronunciationLexicon) -> str:
    def repl(m: re.Match[str]) -> str:
        word = m.group(0)
        lower = word.lower()
        if word in lex.acronyms:
            return lex.acronyms[word]
        if lower in lex.words:
            return lex.words[lower]
        if lower.endswith("'s") and lower[:-2] in lex.words:
            return lex.words[lower[:-2]] + "с"
        if word.isupper() and (len(word) <= 4 or not any(c in "AEIOUY" for c in word)):
            return spell_latin(word)
        return transliterate_latin(word)

    return _LATIN_WORD.sub(lambda m: f" {repl(m)} ", s)


def _apply_cyrillic_acronyms(s: str, lex: PronunciationLexicon) -> str:
    vowels = set("АЕЄИІЇОУЮЯ")

    def repl(m: re.Match[str]) -> str:
        word = m.group(0)
        if word in lex.acronyms:
            return lex.acronyms[word]
        if not any(c in vowels for c in word):
            return spell_cyrillic(word)
        return word  # shouted word ("ТАК!"), read normally

    return _CYR_UPPER_WORD.sub(repl, s)


def _finish(s: str) -> str:
    s = s.lower().translate(_RUSSIAN)
    s = unicodedata.normalize("NFC", s)
    s = "".join(ch if ch in ALLOWED else " " for ch in s)
    # Apostrophe only between letters; stray ones are quote marks.
    s = re.sub(r"(?<![а-яіїєґ])'|'(?![а-яіїєґ])", " ", s)
    # Hyphen only inside words ("будь-ласка"); elsewhere it was a dash.
    s = re.sub(r"(?<![а-яіїєґ])-|-(?![а-яіїєґ])", " — ", s)
    s = _SPACES.sub(" ", s)
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)
    s = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", s)  # ", ," -> ","
    s = re.sub(r",\s*([.!?])", r"\1", s)
    s = re.sub(r"(—\s*)+—", "—", s)
    s = re.sub(r"!{2,}", "!", s)
    s = re.sub(r"\?{2,}", "?", s)
    s = re.sub(r"\.{4,}", "...", s)
    s = re.sub(r"(?<=[,.!?;:])(?=[а-яіїєґ])", " ", s)
    s = s.strip(" ,;:—")
    if not any(ch in UK_LETTERS for ch in s):
        return ""
    if s[-1] not in ".!?":
        s += "."
    return s


# -- splitting -----------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_for_tts(text: str, max_chars: int = 220, min_chars: int = 12) -> list[str]:
    """Split *normalised* text into pieces synthesised one by one.

    Sentences shorter than *min_chars* are joined with the next ("Ой!
    Привіт." stays one piece: very short clips sound choppy); sentences
    longer than *max_chars* are cut at a comma or dash.
    """
    pieces: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_END.split(text.strip()):
        if not sentence:
            continue
        buffer = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(buffer) >= min_chars:
            pieces.extend(_cut_long(buffer, max_chars))
            buffer = ""
    if buffer:
        if pieces and len(pieces[-1]) + len(buffer) + 1 <= max_chars:
            pieces[-1] = f"{pieces[-1]} {buffer}"
        else:
            pieces.append(buffer)
    return pieces


def _cut_long(sentence: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    while len(sentence) > max_chars:
        window = sentence[:max_chars]
        cut = max(window.rfind(", "), window.rfind(" — "), window.rfind("; "))
        if cut < max_chars // 3:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = max_chars
        head, sentence = sentence[: cut + 1].strip(), sentence[cut + 1:].strip(" ,—;")
        parts.append(head.rstrip(",;—") + ("," if not head.endswith((".", "!", "?")) else ""))
    if sentence:
        parts.append(sentence)
    return parts


class SentenceChunker:
    """Collects streamed text and hands out complete sentences.

    ``feed("Привіт! Як ти")`` -> ``["Привіт!"]``; ``flush()`` -> ``["Як ти"]``.
    Works on raw text (normalise each sentence afterwards): splitting before
    normalisation keeps "о 14:30." intact.
    """

    _END = re.compile(r"[.!?…]+[»\"”)]*(?=\s)")

    def __init__(self, min_chars: int = 12) -> None:
        self._buffer = ""
        self._min_chars = min_chars

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        out: list[str] = []
        start = 0
        for m in self._END.finditer(self._buffer):
            candidate = self._buffer[start:m.end()].strip()
            if len(candidate) >= self._min_chars:
                out.append(candidate)
                start = m.end()
        self._buffer = self._buffer[start:].lstrip() if start else self._buffer
        return out

    def flush(self) -> list[str]:
        rest = self._buffer.strip()
        self._buffer = ""
        return [rest] if rest else []
