import random
from collections import Counter
from itertools import pairwise

import pytest

from chibi.core.config import PersonalitySettings, Settings
from chibi.memory.repositories import PhraseHistoryRepository
from chibi.personality.phrases import PhraseBank, PhraseSelector, tone_weights


def entry(id_, text, tone="warm", state="talk", portrait=""):
    return {"id": id_, "text": text, "tone": tone, "state": state, "portrait": portrait}


def make_bank(phrases):
    return PhraseBank.from_dict({"version": 1, "phrases": phrases})


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def history(db):
    return PhraseHistoryRepository(db)


def make_selector(bank, history, settings, clock, seed=1):
    return PhraseSelector(bank, history, lambda: settings, clock, random.Random(seed))


def said_id(selector, history, key):
    """Say a line for ``key`` and return the id that was recorded."""
    assert selector.say(key) is not None
    return history.recent(1)[0][0]


def test_tone_weights_from_sliders():
    weights = tone_weights(PersonalitySettings(humor=70, sarcasm=40, affection=60))
    assert weights["warm"] == pytest.approx(1.2)
    assert weights["playful"] == pytest.approx(1.3)
    assert weights["sarcastic"] == pytest.approx(0.64)
    assert weights["caring"] == pytest.approx(1.1)
    assert tone_weights(PersonalitySettings(sarcasm=9))["sarcastic"] == 0.0
    assert tone_weights(PersonalitySettings(sarcasm=10))["sarcastic"] == pytest.approx(0.16)


@pytest.mark.parametrize("sarcasm", [0, 5, 9])
def test_no_sarcasm_never_picks_sarcastic(history, settings, clock, sarcasm):
    settings.personality.sarcasm = sarcasm
    bank = make_bank({"k": [entry(f"s{i}", f"Сарказм {i}.", tone="sarcastic") for i in range(5)]
                      + [entry("w", "Тепло.")]})
    selector = make_selector(bank, history, settings, clock)
    said = {selector.say("k").text for _ in range(50)}
    assert said == {"Тепло."}


def test_only_sarcastic_key_is_silent_without_sarcasm(history, settings, clock):
    settings.personality.sarcasm = 0
    bank = make_bank({"k": [entry("s", "Ну-ну.", tone="sarcastic")], "fallback": [entry("f", "Ой.")]})
    selector = make_selector(bank, history, settings, clock)
    assert selector.say("k") is None
    assert selector.say("k", "fallback").text == "Ой."


def test_high_sarcasm_does_pick_sarcastic(history, settings, clock):
    settings.personality.sarcasm = 100
    bank = make_bank({"k": [entry("s", "Ну-ну.", tone="sarcastic"), entry("w", "Тепло.")]})
    selector = make_selector(bank, history, settings, clock)
    assert "Ну-ну." in {selector.say("k").text for _ in range(10)}


def test_tone_weighting_shapes_the_choice(history, settings, clock):
    # pick() does not record, so anti-repeat does not interfere with the distribution.
    settings.personality.humor = 100  # playful 1.6
    settings.personality.affection = 0  # warm 0.6
    bank = make_bank({"k": [entry("p", "Хі-хі.", tone="playful"), entry("w", "Тепло.")]})
    selector = make_selector(bank, history, settings, clock, seed=7)
    counts = Counter(selector.pick("k").phrase.id for _ in range(4000))
    assert counts["p"] / 4000 == pytest.approx(1.6 / 2.2, abs=0.03)


def test_say_returns_speech_and_records_usage(history, settings, clock):
    bank = make_bank({"k": [entry("a", "Привіт, я {name}!", state="wave", portrait="greet")]})
    selector = make_selector(bank, history, settings, clock)
    speech = selector.say("k", solicited=True)
    assert speech.text == "Привіт, я Мія!"
    assert (speech.state, speech.portrait, speech.solicited, speech.source) == ("wave", "greet", True, "k")
    assert history.recent() == [("a", "k", clock.now())]
    assert selector.last_used_at("k") == clock.now()


def test_name_follows_settings(history, settings, clock):
    settings.character.name = "Ліна"
    bank = make_bank({"k": [entry("a", "Я {name}.")]})
    assert make_selector(bank, history, settings, clock).say("k").text == "Я Ліна."


def test_first_key_with_an_eligible_phrase_wins(history, settings, clock):
    bank = make_bank({"game": [entry("g", "О, {game}!")], "generic": [entry("x", "О, гра!")]})
    selector = make_selector(bank, history, settings, clock)
    assert selector.say("missing", "game", "generic").source == "generic"  # no game variable
    assert selector.say("missing", "game", "generic", game="Minecraft").text == "О, Minecraft!"
    assert selector.say("missing") is None


def test_empty_or_none_variables_count_as_missing(history, settings, clock):
    bank = make_bank({"k": [entry("g", "О, {game}!")]})
    selector = make_selector(bank, history, settings, clock)
    assert selector.say("k", game="") is None
    assert selector.say("k", game="   ") is None
    assert selector.say("k", game=None) is None


def test_user_phrases_need_a_nickname(history, settings, clock):
    bank = make_bank({"k": [entry("u", "{user}, привіт!"), entry("p", "Привіт!")]})
    selector = make_selector(bank, history, settings, clock)
    assert {selector.pick("k").text for _ in range(30)} == {"Привіт!"}
    settings.character.user_nickname = "Зайчик"
    assert "Зайчик, привіт!" in {selector.pick("k").text for _ in range(30)}


def test_variables_are_inserted_literally(history, settings, clock):
    settings.character.user_nickname = "{game}"
    settings.character.name = "{name}"
    bank = make_bank({"k": [entry("u", "{user}, я {name}.")]})
    assert make_selector(bank, history, settings, clock).say("k").text == "{game}, я {name}."


def test_anti_repeat_cycles_through_every_phrase(history, settings, clock):
    bank = make_bank({"k": [entry(f"p{i}", f"Фраза {i}.") for i in range(5)]})
    selector = make_selector(bank, history, settings, clock)
    said = [said_id(selector, history, "k") for _ in range(25)]
    for start in range(0, 25, 5):
        assert len(set(said[start:start + 5])) == 5  # window = 4: every phrase once per 5 uses
    assert all(a != b for a, b in pairwise(said))


def test_anti_repeat_window_is_capped_at_30(history, settings, clock):
    bank = make_bank({"k": [entry(f"p{i}", f"Фраза {i}.") for i in range(40)]})
    selector = make_selector(bank, history, settings, clock)
    ids = [said_id(selector, history, "k") for _ in range(120)]
    for start in range(len(ids) - 31):
        assert len(set(ids[start:start + 31])) == 31


def test_two_phrases_alternate_and_one_phrase_repeats(history, settings, clock):
    bank = make_bank({"two": [entry("a", "А."), entry("b", "Б.")], "one": [entry("x", "Ікс.")]})
    selector = make_selector(bank, history, settings, clock)
    texts = [selector.say("two").text for _ in range(6)]
    assert all(a != b for a, b in pairwise(texts))
    assert {selector.say("one").text for _ in range(3)} == {"Ікс."}


def test_anti_repeat_survives_a_restart(history, settings, clock, db):
    bank = make_bank({"k": [entry(f"p{i}", f"Фраза {i}.") for i in range(5)]})
    first = make_selector(bank, history, settings, clock, seed=1)
    before = {first.say("k").text for _ in range(3)}
    # "Restart": a new selector (other seed) over the same database.
    second = make_selector(bank, PhraseHistoryRepository(db), settings, clock, seed=99)
    after = {second.say("k").text for _ in range(2)}
    assert before.isdisjoint(after)
    assert len(before | after) == 5
    assert second.last_used_at("k") == clock.now()


def test_least_recently_used_fallback_when_everything_is_excluded(history, settings, clock):
    # 4 phrases -> window 3; without a nickname only "a" and "b" are eligible, so after
    # both were used everything eligible is excluded and the least recently used wins.
    bank = make_bank({"k": [entry("a", "А."), entry("b", "Б."),
                            entry("u1", "{user} 1."), entry("u2", "{user} 2.")]})
    selector = make_selector(bank, history, settings, clock)
    texts = [selector.say("k").text for _ in range(8)]
    assert set(texts) == {"А.", "Б."}
    assert all(a != b for a, b in pairwise(texts))


def test_history_write_failure_does_not_silence_her(settings, clock, caplog):
    class BrokenHistory:
        def recent(self, limit=200):
            raise RuntimeError("disk on fire")

        def add(self, *args):
            raise RuntimeError("disk on fire")

    bank = make_bank({"k": [entry("a", "А."), entry("b", "Б.")]})
    selector = PhraseSelector(bank, BrokenHistory(), lambda: settings, clock, random.Random(1))
    first, second = selector.say("k").text, selector.say("k").text
    assert first != second  # anti-repeat still works from memory
    assert "could not persist phrase history" in caplog.text


def test_pick_does_not_record(history, settings, clock):
    bank = make_bank({"k": [entry("a", "А.", state="happy")]})
    selector = make_selector(bank, history, settings, clock)
    picked = selector.pick("k")
    assert picked.phrase.state == "happy" and picked.key == "k"
    assert history.recent() == []
    assert selector.last_used_at("k") is None


def test_reload_history_after_wipe(history, settings, clock, db):
    bank = make_bank({"k": [entry("a", "А."), entry("b", "Б.")]})
    selector = make_selector(bank, history, settings, clock)
    selector.say("k")
    db.wipe_history()
    selector.reload_history()
    assert selector.last_used_at("k") is None


def test_packaged_bank_renders_every_key(history, settings, clock):
    settings.character.user_nickname = "Друже"
    bank = PhraseBank.load()
    selector = make_selector(bank, history, settings, clock)
    variables = {"game": "Minecraft", "app": "VS Code", "duration": "2 години", "time": "14:05"}
    for key in sorted(bank.keys()):
        for _ in range(len(bank.phrases(key))):
            speech = selector.say(key, **variables)
            assert speech is not None, key
            assert "{" not in speech.text and "}" not in speech.text, key
