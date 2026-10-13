import json
from importlib import resources

import pytest

from chibi.voice.tts_text import (
    ALLOWED,
    FIRST_CLAUSE_CHARS,
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


# --- the first piece of a streamed answer may end at a clause -----------------------------------


def chunks(text, step=0, first_clause=FIRST_CLAUSE_CHARS):
    """All pieces of *text*, fed whole (step 0) or *step* characters at a time."""
    chunker = SentenceChunker(first_clause=first_clause)
    out = []
    if step:
        for i in range(0, len(text), step):
            out += chunker.feed(text[i:i + step])
    else:
        out += chunker.feed(text)
    return out + chunker.flush(), chunker.clause_first


LONG_FIRST = ("Ну що ж, давай подумаємо разом, що можна зробити з цим, а потім вирішимо. "
              "Друге речення, теж із комою, тут. Кінець.")


def test_a_long_first_sentence_starts_at_its_first_clause_and_later_sentences_stay_whole():
    pieces, clause_first = chunks(LONG_FIRST)
    assert clause_first
    assert pieces == ["Ну що ж, давай подумаємо разом,", "що можна зробити з цим, а потім вирішимо.",
                      "Друге речення, теж із комою, тут.", "Кінець."]
    assert len(pieces[0]) >= FIRST_CLAUSE_CHARS
    assert " ".join(pieces) == LONG_FIRST  # nothing lost, nothing repeated


@pytest.mark.parametrize("step", [1, 2, 3, 7, 50])
def test_streamed_in_any_pieces_the_clause_split_is_the_same(step):
    assert chunks(LONG_FIRST, step) == chunks(LONG_FIRST)


def test_a_short_first_sentence_is_not_split():
    assert chunks("Привіт! Я тут, поруч із тобою. Зараз розповім, що бачу, і ще дещо цікаве.") == (
        ["Привіт! Я тут, поруч із тобою.", "Зараз розповім, що бачу, і ще дещо цікаве."], False)


def test_the_rest_of_the_clause_sentence_is_said_however_short():
    pieces, _ = chunks("Знаєш, я довго думала над цим питанням разом, так? Ось.")
    assert pieces == ["Знаєш, я довго думала над цим питанням разом,", "так?", "Ось."]


@pytest.mark.parametrize(("text", "first"), [
    ("Знаєш, я довго думала над цим питанням — і ось що вирішила.",
     "Знаєш, я довго думала над цим питанням —"),
    ("Слухай, ось що я тобі зараз скажу про гру: це було круто.",
     "Слухай, ось що я тобі зараз скажу про гру:"),
    ("Я зробила все, що могла сьогодні ввечері; решта завтра.", "Я зробила все, що могла сьогодні ввечері;"),
])
def test_clause_marks(text, first):
    assert chunks(text)[0][0] == first


@pytest.mark.parametrize("text", [
    "Я бачу на мапі ворогів у кількості 1, 2, 3 і ще кількох, будь обережний.",  # a number list
    "Температура процесора зараз аж 2,5 рази вища за норму, тож обережно.",  # a decimal comma
    "Зустрінемось сьогодні ввечері о 14:30 біля входу в парк, добре?",  # a clock time
    "Це дуже-дуже-дуже довга думка без жодної паузи всередині неї, так?",  # hyphenated words
])
def test_never_split_inside_a_number_or_a_word(text):
    pieces, _ = chunks(text)
    assert " ".join(pieces) == text
    # Said piece by piece, the words are exactly those of the whole answer.
    assert words(" ".join(normalize_for_tts(piece) for piece in pieces)) == words(normalize_for_tts(text))
    for piece in pieces[:-1]:
        assert not piece[-2:-1].isdigit()  # no piece ends on a number


def words(text):
    return [w.strip(".,!?;:—") for w in text.split() if w.strip(".,!?;:—")]


def test_a_number_list_is_kept_together():
    pieces, _ = chunks("Я бачу на мапі ворогів у кількості 1, 2, 3 і ще кількох, будь обережний.")
    assert pieces[0] == "Я бачу на мапі ворогів у кількості 1, 2, 3 і ще кількох,"


@pytest.mark.parametrize(("text", "spoken"), [
    ("Він грає вже багато годин поспіль і т. д. Але все одно класно.", "і так далі"),
    ("Він довго грав у шутери, ігри т. зв. «королівської битви» й інші.", "так званий"),
    ("У 1990-2000 р. р. таких ігор було мало. Тепер їх багато.", "роки"),
])
def test_never_split_inside_an_abbreviation(text, spoken):
    for first_clause in (0, FIRST_CLAUSE_CHARS):
        pieces, _ = chunks(text, first_clause=first_clause)
        assert any(spoken in normalize_for_tts(piece) for piece in pieces), pieces


def test_never_split_inside_a_role_play_action():
    pieces, _ = chunks("*зітхає, сміється і довго думає, що ж відповісти* Ну добре, давай спробуємо ще раз.")
    assert pieces[0].count("*") in (0, 2)
    assert all("зітхає" not in normalize_for_tts(piece) for piece in pieces)


def test_without_first_clause_every_piece_is_a_sentence():
    pieces, clause_first = chunks(LONG_FIRST, first_clause=0)
    assert not clause_first
    assert pieces[0] == "Ну що ж, давай подумаємо разом, що можна зробити з цим, а потім вирішимо."
