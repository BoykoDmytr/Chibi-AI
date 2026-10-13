import json
from importlib import resources

import pytest

from chibi.voice.tts_text import (
    ALLOWED,
    PronunciationLexicon,
    SentenceChunker,
    default_lexicon,
    normalize_for_tts,
    split_for_tts,
    transliterate_latin,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Upper case is dropped by the voice: everything is lower-cased.
        ("Тільки не кажи.", "тільки не кажи."),
        # Numbers agree with the noun that follows (gender, accusative).
        ("Ти вже 2 години 15 хвилин тут", "ти вже дві години п'ятнадцять хвилин тут."),
        ("Граєш уже 1 годину", "граєш уже одну годину."),
        ("21 перемога поспіль", "двадцять одна перемога поспіль."),
        ("1 повідомлення, 2 гри, 1 гра, 101 спроба",
         "одне повідомлення, дві гри, одна гра, сто одна спроба."),
        ("Мія, 2 хвилинки!", "мія, дві хвилинки!"),
        ("Це вже 3-й раз", "це вже третій раз."),
        ("на 2-му рівні", "на другому рівні."),
        # Units and symbols.
        ("Батарея на 50%", "батарея на п'ятдесят відсотків."),
        ("1 % і 3 %", "один відсоток і три відсотки."),
        ("2,5 години", "дві кома п'ять години."),
        ("1.500 грн", "тисяча п'ятсот гривень."),
        ("144 fps", "сто сорок чотири кадри на секунду."),
        ("$5 за це", "п'ять доларів за це."),
        ("10€", "десять євро."),
        ("-5 градусів", "мінус п'ять градусів."),
        ("10 000 очок", "десять тисяч очок."),
        ("Код 007", "код нуль нуль сім."),
        # Clock times take the case the preposition asks for.
        ("о 14:30", "о чотирнадцятій тридцять."),
        ("до 9:05", "до дев'ятої нуль п'ять."),
        ("Зараз 7:00", "зараз сьома рівно."),
        ("Рахунок 3:2", "рахунок три — два."),
        # Names, acronyms, Roman numerals.
        ("Сидиш у VS Code", "сидиш у ві ес код."),
        ("S.T.A.L.K.E.R. 2: Серце Чорнобиля", "сталкер два: серце чорнобиля."),
        ("Counter-Strike 2", "каунтер страйк два."),
        ("Diablo IV і Dark Souls III", "діабло чотири і дарк соулс три."),
        ("GTA V чи RPG?", "джі ті ей п'ять чи ар пі джі?"),
        ("ПК чи Xbox? ШІ знає.", "пе ка чи іксбокс? ші знає."),
        ("Baldur's Gate 3", "балдурс гейт три."),
        ("Sid Meier's Civilization VI", "сід маєрс цивілізейшн шість."),
        # Typography.
        ("«Привіт» — сказала вона", "привіт — сказала вона."),
        ("пам’ять, пам`ять, памʼять", "пам'ять, пам'ять, пам'ять."),
        ("Ну… ок", "ну... ок."),
        ("Я в захваті!!! Серйозно??", "я в захваті! серйозно?"),
        ("Ну-ну. По-моєму, так", "ну-ну. по-моєму, так."),
        ("Вона *зітхає* пішла", "вона пішла."),
        ("Подивись https://example.com/x?y=1 там", "подивись посилання там."),
        ("Ой 🙂 ^_^ $ лол", "ой лол."),
        ("Сьогодні (це жарт) так", "сьогодні, це жарт, так."),
        ("Ёлка ы", "йолка и."),
    ],
)
def test_normalize(text, expected):
    assert normalize_for_tts(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Hello World! ABC DEF 123 456 7890",
        "Ой!!! 🙂🙂 ^^^ ___ $$$ <<>> [[]] {{}} @@@ ### ~~~ |||",
        "S.T.A.L.K.E.R. 2: Heart of Chornobyl — 3:2 о 23:59, 99,9% і 1 000 000 грн",
        "Ⅳ ½ ² ³ ٣ ১ ⁵ 𝟙",  # unicode digits and fractions
        "ÀÉÎÕÜ ñ ç ß ø å",
    ],
)
def test_output_only_contains_voice_alphabet(text):
    result = normalize_for_tts(text)
    assert set(result) <= ALLOWED
    assert "^" not in result and "_" not in result and "$" not in result


@pytest.mark.parametrize("text", ["", "   ", "🙂🙂", "^_^", "...", "123abc"[:0]])
def test_nothing_speakable_gives_empty_string(text):
    assert normalize_for_tts(text) == ""


def test_always_ends_with_sentence_mark():
    assert normalize_for_tts("привіт").endswith(".")
    assert normalize_for_tts("привіт?").endswith("?")
    assert normalize_for_tts("привіт,").endswith(".")


def test_unknown_latin_words_are_transliterated():
    assert normalize_for_tts("Raft") == "рафт."
    assert transliterate_latin("Ghostrunner") == "гостранер"
    assert transliterate_latin("game") == "гейм"
    assert transliterate_latin("knight") == "найт"
    assert transliterate_latin("") == ""


def test_short_unknown_uppercase_words_are_spelled():
    assert normalize_for_tts("XYZ") == "екс вай зет."


def test_cyrillic_shouting_is_read_not_spelled():
    assert normalize_for_tts("ТАК!") == "так!"


def test_custom_lexicon_phrases_win():
    lex = default_lexicon()
    lex.add_phrase("My Indie Game", "май інді гейм")
    assert normalize_for_tts("Граєш у My Indie Game?", lex) == "граєш у май інді гейм?"
    # The shared default lexicon is not modified.
    assert normalize_for_tts("My Indie Game") != "май інді гейм."


def test_empty_lexicon_still_works():
    assert normalize_for_tts("Steam 2", PronunciationLexicon()) == "стім два."


def test_builtin_lexicon_is_well_formed():
    data = json.loads(
        resources.files("chibi.voice").joinpath("data/pronunciations_uk.json").read_text("utf-8")
    )
    for section in ("phrases", "words", "acronyms"):
        for key, say in data[section].items():
            assert key and say, (section, key)
            assert set(say) <= ALLOWED, (section, key, say)
    for key in data["words"]:
        assert key == key.lower(), key


def test_every_known_game_title_becomes_speakable():
    data = json.loads(
        resources.files("chibi.gaming").joinpath("data/games.json").read_text("utf-8")
    )
    for game in data["games"]:
        result = normalize_for_tts(game["title"])
        assert result, game["title"]
        assert set(result) <= ALLOWED


def test_split_for_tts_merges_short_and_cuts_long():
    text = normalize_for_tts("Ой! Привіт. " + "Дуже довге речення, " * 20 + "кінець.")
    pieces = split_for_tts(text, max_chars=120)
    assert pieces[0].startswith("ой! привіт.")
    assert all(len(p) <= 121 for p in pieces)
    assert " ".join(pieces).replace(",", "").split()[-1] == "кінець."


def test_split_for_tts_keeps_short_text_whole():
    assert split_for_tts("ой!") == ["ой!"]
    assert split_for_tts("") == []


def test_sentence_chunker_streams_sentences():
    chunker = SentenceChunker()
    out = []
    for delta in ["Привіт! Як ", "ти? Я тут о 14", ":30. Чекаю", " на тебе"]:
        out += chunker.feed(delta)
    assert out == ["Привіт! Як ти?", "Я тут о 14:30."]
    assert chunker.flush() == ["Чекаю на тебе"]
    assert chunker.flush() == []


def test_sentence_chunker_waits_for_the_space_after_a_dot():
    chunker = SentenceChunker(min_chars=1)
    assert chunker.feed("Версія 2.") == []  # could be "2.5"
    assert chunker.feed("5 вийшла. Ура") == ["Версія 2.5 вийшла."]
    assert chunker.flush() == ["Ура"]
