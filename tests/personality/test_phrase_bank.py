"""The packaged Ukrainian phrase bank: structure, coverage and language guards,
plus how the loader treats broken input."""

import json
import re
import string
from collections import Counter

import pytest

from chibi.character.states import CharacterState
from chibi.core.paths import character_pack_dir
from chibi.personality.phrases import DEFAULT_PATH, REQUIRED_STATES, PhraseBank, template_fields

TONES = {"warm", "playful", "sarcastic", "caring"}
PORTRAITS = {
    "angry", "bye", "calm", "drink", "eat", "excited", "facepalm", "greet", "idea", "joy", "joyful",
    "laugh", "music", "offended", "phone", "point", "question", "shy", "sleep_cat", "smile", "surprised",
    "thinking", "tired", "typing", "wink", "wink_heart", "wow",
}
PLACEHOLDERS = {"name", "user", "game", "app", "duration", "time"}
REQUIREMENTS = PLACEHOLDERS - {"name"}

GAMES = (
    "stalker2", "stalker_classic", "cs2", "dota2", "minecraft", "witcher3", "cyberpunk2077", "bg3",
    "eldenring", "gta5", "valorant", "lol", "terraria", "stardew",
)
GENRES = (
    "shooter", "souls", "rpg", "strategy", "sandbox", "moba", "cozy", "survival", "horror", "racing", "mmo",
)
MOODS = ("sleepy", "hungry", "bored", "sad", "happy", "cheerful", "calm")

MIN_COUNTS: dict[str, int] = {
    "greet.first_run": 4, "greet.morning": 8, "greet.day": 8, "greet.evening": 8, "greet.night": 8,
    "greet.return_long": 6,
    "user_returned.short": 10, "user_returned.long": 10, "user_returned.very_long": 8,
    "game_start.generic": 12,
    **{f"game_start.{game}": 4 for game in GAMES},
    **{f"game_start.genre.{genre}": 4 for genre in GENRES},
    "game_end.short": 6, "game_end.generic": 8, "game_end.long": 8,
    "programming.start": 10, "programming.long": 8, "browsing.start": 6, "browsing.long": 6,
    "video.start": 8, "video.long": 5, "working.long": 6, "gaming.long": 8, "late_night": 8,
    "interaction.click.poke": 10, "interaction.click.annoyed": 6, "interaction.pet.happy": 10,
    "interaction.pet.too_much": 6, "interaction.feed.fed": 8, "interaction.feed.full": 6,
    "interaction.feed.starving_fed": 5, "interaction.rest.sleep": 5, "interaction.rest.not_tired": 5,
    "interaction.wake.ok": 5, "interaction.wake.grumpy": 5, "interaction.call.ok": 6,
    "interaction.drag_start.ok": 8, "interaction.drag_end.ok": 6, "interaction.talk.ok": 5,
    "needs.hungry": 6, "needs.sleepy": 6, "needs.bored": 8,
    "idle.joke": 25, "idle.thought": 15, "idle.question": 12,
    "quiet.ack": 6, "quiet.end": 4,
    "chat.greeting": 8, "chat.how_are_you": 8, "chat.name": 5, "chat.capabilities": 4, "chat.thanks": 5,
    "chat.compliment": 6, "chat.rude": 5, "chat.bye": 5, "chat.time": 4, "chat.unknown": 10,
    "chat.user_sad": 5, "chat.user_good": 5,
    **{f"chat.mood.{mood}": 2 for mood in MOODS},
}
#: Keys whose every phrase must use a placeholder (the rule always passes it).
MUST_USE = {"game_start.generic": "game", "chat.name": "name", "chat.time": "time"}

# Gendered past tense / adjectives about the USER (her own feminine forms are fine).
# The user's gender is unknown, so "ти повернувся/повернулася" etc. are banned.
GENDERED_USER = re.compile(
    r"\bти\s+(?:вже\s+|знову\s+|що\s+|не\s+|там\s+|так\s+|сьогодні\s+|теж\s+)?"
    r"(повернувся|повернулася|втомився|втомилася|прийшов|прийшла|зробив|зробила|був|була|грав|грала"
    r"|пішов|пішла|забув|забула|сидів|сиділа|виграв|виграла|програв|програла|написав|написала"
    r"|зголоднів|зголодніла|готовий|готова|голодний|голодна|радий|рада|впевнений|впевнена|сам|сама"
    r"|втомлений|втомлена|живий|жива)\b",
    re.IGNORECASE,
)
# Bare questions to the user in a gendered form ("Втомився?", "Готовий?").
GENDERED_BARE = re.compile(
    r"(?:^|[.!?…]\s+)(втомився|готовий|голодний|повернувся|прийшов|зголоднів|впевнений)[?!.]", re.IGNORECASE
)
# Letters that exist in Russian but not in Ukrainian: a cheap guard against russisms / wrong layout.
RUSSIAN_ONLY = re.compile(r"[ыэъёЫЭЪЁ]")
FANCY_APOSTROPHES = re.compile("[’‘ʼ`´]")
# Russisms / calques found in review: "панель задач" (Windows says "панель завдань"),
# "рахується" for "counts" (Russian "считается"), "нічогонероблення".
RUSSISMS = re.compile(r"панел\w* задач|рахуєт\w*|рахуюся|нічогонероблен\w*", re.IGNORECASE)
# {duration} is an accusative ("1 годину"): never a subject or bare exclamation,
# never after "цілих".
DURATION_AS_SUBJECT = re.compile(r"(?:^|[.!?…—]\s*)\{duration\}|цілих\s+\{duration\}", re.IGNORECASE)
# A quoted title after a bare preposition needs declension ("у «Відьмак 3»"): say "у грі «...»".
PREPOSITION_BEFORE_TITLE = re.compile(r"\b(?:у|в|для|до|на|про|за|від|з|із)\s+«\{game\}»", re.IGNORECASE)
MIXED_SCRIPT_WORD = re.compile(r"\b(?=\w*[a-zA-Z])(?=\w*[Ѐ-ӿ])\w+\b")


@pytest.fixture(scope="module")
def raw():
    return json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def entries(raw):
    """Every entry of the main bank and of the feature banks (phrases_*_uk.json),
    so the style rules below apply to all of them and duplicate ids across files
    make test_every_entry_loads fail."""
    found = [(key, entry) for key, items in raw["phrases"].items() for entry in items]
    for extra in sorted(DEFAULT_PATH.parent.glob("phrases_*_uk.json")):
        data = json.loads(extra.read_text(encoding="utf-8"))
        found += [(key, entry) for key, items in data["phrases"].items() for entry in items]
    return found


@pytest.fixture(scope="module")
def bank():
    return PhraseBank.load()


def fields_of(text):
    return {f for _l, f, _s, _c in string.Formatter().parse(text) if f}


def test_file_is_valid_json_with_version(raw):
    assert raw["version"] == 1
    assert isinstance(raw["phrases"], dict)


def test_every_entry_loads(entries, bank):
    # The loader skips invalid entries; the packaged bank must have none.
    assert len(bank) == len(entries)


def test_total_count(bank):
    assert len(bank) >= 380


@pytest.mark.parametrize(("key", "minimum"), sorted(MIN_COUNTS.items()))
def test_minimum_counts(bank, key, minimum):
    assert len(bank.phrases(key)) >= minimum, key


def test_ids_unique(entries):
    counts = Counter(entry["id"] for _key, entry in entries)
    assert [i for i, n in counts.items() if n > 1] == []


def test_entry_fields_are_known(entries):
    states = {s.value for s in CharacterState}
    for key, entry in entries:
        assert set(entry) <= {"id", "text", "tone", "state", "portrait", "requires"}, entry["id"]
        assert entry["tone"] in TONES, entry["id"]
        assert entry["state"] in states, entry["id"]
        assert entry["portrait"] == "" or entry["portrait"] in PORTRAITS, entry["id"]
        assert set(entry.get("requires", [])) <= REQUIREMENTS, entry["id"]
        assert fields_of(entry["text"]) <= PLACEHOLDERS, entry["id"]
        template_fields(entry["text"])  # no format specs / attribute access
        assert key.strip() == key


def test_portraits_exist_in_the_default_pack(entries):
    portraits = character_pack_dir("mia") / "portraits"
    if not portraits.is_dir():
        pytest.skip("character pack not present")
    for _key, entry in entries:
        if entry["portrait"]:
            assert (portraits / f"{entry['portrait']}.png").is_file(), entry["id"]


def test_requires_matches_placeholders(entries):
    for _key, entry in entries:
        used = fields_of(entry["text"]) & REQUIREMENTS
        declared = set(entry.get("requires", []))
        assert used == declared, (entry["id"], used, declared)


@pytest.mark.parametrize(("key", "placeholder"), sorted(MUST_USE.items()))
def test_keys_that_always_use_a_placeholder(bank, key, placeholder):
    for phrase in bank.phrases(key):
        assert placeholder in fields_of(phrase.text), phrase.id


def test_texts_are_clean(entries):
    for _key, entry in entries:
        text = entry["text"]
        assert text == text.strip(), entry["id"]
        assert "  " not in text, entry["id"]
        assert len(text) <= 140, (entry["id"], len(text))
        assert not RUSSIAN_ONLY.search(text), entry["id"]
        assert not FANCY_APOSTROPHES.search(text), entry["id"]  # one apostrophe style: '
        assert not MIXED_SCRIPT_WORD.search(text), entry["id"]  # e.g. Latin "i" in "привiт"
        assert text.count("«") == text.count("»"), entry["id"]
        assert text.count("{") == text.count("}"), entry["id"]
        assert text[-1] in ".!?…»)", entry["id"]
        assert not RUSSISMS.search(text), entry["id"]


# Reproaches for being away: the companion never punishes absence.
GUILT_TRIP = re.compile(r"нарешті згадали|покинули|проміняли|без жодної звісточки|не ображаюся",
                        re.IGNORECASE)


def test_welcome_back_lines_do_not_guilt_trip(bank):
    keys = [k for k in sorted(bank.keys()) if k.startswith(("greet.", "user_returned."))]
    offenders = [p.id for k in keys for p in bank.phrases(k) if GUILT_TRIP.search(p.text)]
    assert offenders == []


def test_unknown_replies_are_neutral(bank):
    # The unrecognised message may be sad news ("у мене помер кіт").
    assert not any("цікаво" in p.text.lower() for p in bank.phrases("chat.unknown"))


def test_duration_is_never_a_subject(entries):
    offenders = [entry["id"] for _key, entry in entries if DURATION_AS_SUBJECT.search(entry["text"])]
    assert offenders == []
    assert DURATION_AS_SUBJECT.search("{duration} коду!")
    assert DURATION_AS_SUBJECT.search("Ой! {duration} без тебе.")
    assert DURATION_AS_SUBJECT.search("Цілих {duration}!")
    assert not DURATION_AS_SUBJECT.search("Ти кодиш уже {duration}!")


def test_quoted_titles_follow_a_generic_noun(entries):
    offenders = [entry["id"] for _key, entry in entries if PREPOSITION_BEFORE_TITLE.search(entry["text"])]
    assert offenders == []
    assert PREPOSITION_BEFORE_TITLE.search("Час для «{game}»!")
    assert not PREPOSITION_BEFORE_TITLE.search("Час для гри «{game}»!")


def test_lines_that_accompany_a_posture_carry_it(raw):
    # "Йди спати" puts her to sleep; a "yawn" line would wake her straight away.
    assert REQUIRED_STATES == {"interaction.rest.sleep": "sleep"}
    for key, state in REQUIRED_STATES.items():
        assert raw["phrases"][key], key
        for item in raw["phrases"][key]:
            assert item["state"] == state, item["id"]


def test_loader_enforces_the_posture_state(caplog):
    data = {"version": 1, "phrases": {"interaction.rest.sleep": [entry("a", state="yawn"), entry("b")]}}
    bank = PhraseBank.from_dict(data)
    assert [p.state for p in bank.phrases("interaction.rest.sleep")] == ["sleep", "sleep"]
    assert "required for interaction.rest.sleep" in caplog.text


def test_no_gendered_forms_about_the_user(entries):
    offenders = [
        entry["id"] for _key, entry in entries
        if GENDERED_USER.search(entry["text"]) or GENDERED_BARE.search(entry["text"])
    ]
    assert offenders == []


def test_gender_regex_catches_examples():
    assert GENDERED_USER.search("Ти повернувся!")
    assert GENDERED_USER.search("Ти вже втомилася?")
    assert GENDERED_BARE.search("Привіт! Втомився?")
    assert not GENDERED_USER.search("З поверненням! Ти знову тут.")
    assert not GENDERED_USER.search("Я втомилася.")


def test_every_key_works_without_sarcasm_and_without_a_nickname(bank):
    for key in sorted(bank.keys()):
        phrases = bank.phrases(key)
        assert any(p.tone != "sarcastic" for p in phrases), key
        assert any("user" not in p.requires for p in phrases), key


def test_user_placeholder_only_where_declared(bank):
    with_user = [p for p in bank if "{user}" in p.text]
    assert with_user, "a few phrases should address the user by nickname"
    assert all("user" in p.requires for p in with_user)


def test_tones_are_varied(bank):
    tones = Counter(p.tone for p in bank)
    assert set(tones) == TONES
    assert tones["sarcastic"] < len(bank) / 4  # a bit sarcastic, not mean


def test_jokes_are_distinct(bank):
    texts = [p.text for p in bank.phrases("idle.joke")]
    assert len(set(texts)) == len(texts)


# -- loader robustness ---------------------------------------------------------------------


def entry(id_, text="Привіт!", tone="warm", state="talk", portrait="", **extra):
    return {"id": id_, "text": text, "tone": tone, "state": state, "portrait": portrait, **extra}


def test_invalid_entries_are_skipped_not_fatal(caplog):
    data = {
        "version": 1,
        "phrases": {
            "k": [
                entry("ok.1"),
                entry("ok.1", text="Дубль"),  # duplicate id
                entry("bad.tone", tone="angry"),
                entry("bad.state", state="flying"),
                entry("bad.state_type", state=["talk"]),
                entry("bad.tone_type", tone={"warm": 1}),
                entry("bad.portrait", portrait=3),
                entry("bad.text_type", text=["Привіт"]),
                entry("bad.text", text="   "),
                entry("bad.placeholder", text="Привіт, {friend}!"),
                entry("bad.format", text="Привіт, {name!r}!"),
                entry("bad.attr", text="{name.__class__}"),
                entry("bad.brace", text="Привіт, {name"),
                entry("bad.requires", requires=["weather"]),
                entry("bad.requires_type", requires="game"),
                "not an object",
                {"text": "без id", "tone": "warm", "state": "talk"},
                entry("ok.2", text="О, {game}!", tone="playful"),
            ],
            "broken": "not a list",
        },
    }
    bank = PhraseBank.from_dict(data)
    assert [p.id for p in bank.phrases("k")] == ["ok.1", "ok.2"]
    assert not bank.has("broken")
    assert "duplicate id" in caplog.text


def test_placeholders_imply_requirements():
    bank = PhraseBank.from_dict({"version": 1, "phrases": {"k": [entry("a", text="{user}, {game}!")]}})
    assert bank.phrases("k")[0].requires == ("user", "game")


def test_defaults_for_optional_fields():
    minimal = {"id": "a", "text": "Ой!", "tone": "warm"}
    bank = PhraseBank.from_dict({"version": 1, "phrases": {"k": [minimal]}})
    phrase = bank.phrases("k")[0]
    assert (phrase.state, phrase.portrait, phrase.requires) == ("talk", "", ())


def test_missing_file_gives_empty_bank(tmp_path):
    bank = PhraseBank.load(tmp_path / "nope.json")
    assert len(bank) == 0
    assert not bank.has("greet.morning")


def test_malformed_file_gives_empty_bank(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert len(PhraseBank.load(path)) == 0
    path.write_text('{"version": 1}', encoding="utf-8")
    assert len(PhraseBank.load(path)) == 0


def test_bank_accessors(bank):
    phrase = bank.phrases("greet.morning")[0]
    assert bank.get(phrase.id) == phrase
    assert bank.get("no.such.id") is None
    assert bank.phrases("no.such.key") == []
    assert "greet.morning" in sorted(bank.keys())
    # phrases() returns a copy: callers cannot mutate the bank
    bank.phrases("greet.morning").clear()
    assert bank.phrases("greet.morning")
