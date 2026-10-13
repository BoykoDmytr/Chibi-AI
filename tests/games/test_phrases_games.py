"""phrases_games_uk.json: every key the gifts and games ask for exists with
enough lines, loads cleanly into the merged bank and keeps her manners.

(The style rules shared by every phrase file - tones, states, portraits,
apostrophes, gendered forms about the user - are checked for all files in
tests/personality/test_phrase_bank.py.)
"""

import json
import re

import pytest

from chibi.personality.phrases import DEFAULT_PATH, PhraseBank
from chibi.tamagotchi.gifts import AGAIN_KEY, GENERIC_KEY, GiftCatalog
from tests.personality.test_phrase_bank import GENDERED_BARE, GENDERED_USER

PATH = DEFAULT_PATH.parent / "phrases_games_uk.json"
GAME_KEYS = (
    "game.rps.win", "game.rps.lose", "game.rps.draw",
    "game.cookies.great", "game.cookies.ok", "game.cookies.poor",
    "game.guess.win", "game.guess.lose",
    "game.invite", "game.start",
)
#: Never punish absence, never guilt-trip about playing or giving.
GUILT = re.compile(
    r"покину|забув|забул|кинув|кинул|не заходи|давно не|чекала на тебе|нарешті згада|без мене"
    r"|як ти міг|як ти могла|ображаюся|ображена|скупий|скупа|жадібн",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def raw():
    return json.loads(PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def own_bank(raw):
    return PhraseBank.from_dict(raw, source=PATH.name)


@pytest.fixture(scope="module")
def merged():
    return PhraseBank.load()


def required_keys():
    gifts = [key for item in GiftCatalog.load() for key in item.reaction_keys]
    return [*gifts, GENERIC_KEY, AGAIN_KEY, *GAME_KEYS]


def test_every_entry_loads(raw, own_bank):
    assert raw["version"] == 1
    assert len(own_bank) == sum(len(items) for items in raw["phrases"].values())


@pytest.mark.parametrize("key", required_keys())
def test_required_keys_have_enough_lines(merged, key):
    phrases = merged.phrases(key)
    assert len(phrases) >= 4, key
    assert any(p.tone != "sarcastic" for p in phrases), key
    assert any("user" not in p.requires for p in phrases), key


def test_merged_into_the_default_bank(merged, own_bank):
    for key in list(own_bank.keys()):
        assert [p.id for p in merged.phrases(key)] == [p.id for p in own_bank.phrases(key)]


def test_only_gift_and_game_keys(own_bank):
    assert all(key.startswith(("gift.", "game.")) for key in list(own_bank.keys()))


def test_no_guilt_and_no_gendered_user(own_bank):
    for phrase in own_bank:
        assert not GUILT.search(phrase.text), phrase.id
        assert not GENDERED_USER.search(phrase.text), phrase.id
        assert not GENDERED_BARE.search(phrase.text), phrase.id


def test_eating_lines_for_treats_she_eats(own_bank):
    for key in ("gift.cookie", "gift.apple", "gift.cake"):
        assert any(p.state == "eat" for p in own_bank.phrases(key)), key


def test_invitation_asks_to_play(own_bank):
    assert any("Пограймо?" in p.text for p in own_bank.phrases("game.invite"))
