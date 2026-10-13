"""Gifts: the star wallet, the catalog and the gift service (buy / give)."""

import json
import logging
import math

import pytest

from chibi.character.states import CharacterState
from chibi.core.events import AppStarted, NeedsChanged, SpeechRequested
from chibi.core.paths import character_pack_dir
from chibi.memory.repositories import StateRepository
from chibi.personality.phrases import PhraseBank
from chibi.tamagotchi.gifts import (
    CATALOG_PATH,
    DAILY_EARN_CAP,
    DAILY_START_BONUS,
    FALLBACK_THANKS,
    KEY_COLLECTION,
    KEY_INVENTORY,
    KEY_WALLET,
    MAX_STACK,
    MAX_STARS,
    MSG_SAVE_FAILED,
    MSG_STACK_FULL,
    MSG_UNKNOWN,
    REPEAT_WINDOW_SECONDS,
    GiftCatalog,
    GiftKind,
    Wallet,
    describe_effects,
    local_day,
    stars_uk,
)
from tests.games.fakes import FlakyRepo, World

DAY = 24 * 3600.0
approx = pytest.approx


# =========================================================================================
# Wallet
# =========================================================================================


@pytest.fixture
def repo(db, clock):
    return FlakyRepo(db, clock)


@pytest.fixture
def wallet(repo, clock):
    w = Wallet(repo, clock)
    w.load()
    return w


def stored(repo):
    return repo.get(KEY_WALLET)


def test_new_wallet_is_empty(wallet):
    assert wallet.stars == 0
    assert wallet.earned_today == 0
    assert wallet.remaining_today() == DAILY_EARN_CAP == 60


def test_earn_credits_and_persists(wallet, repo, clock):
    assert wallet.earn(8) == 8
    assert wallet.stars == 8
    assert stored(repo) == {"stars": 8, "earned_day": local_day(clock.now()), "earned_today": 8,
                            "bonus_day": ""}


def test_daily_cap_and_kind_partial_credit(wallet):
    for _ in range(7):
        assert wallet.earn(8) == 8
    assert wallet.earned_today == 56
    assert wallet.earn(8) == 4  # only up to 60
    assert wallet.earn(8) == 0
    assert wallet.stars == 60
    assert wallet.remaining_today() == 0


def test_cap_resets_on_a_new_day(wallet, clock):
    wallet.earn(60)
    clock.advance(DAY)
    assert wallet.earned_today == 0
    assert wallet.earn(10) == 10
    assert wallet.stars == 70


@pytest.mark.parametrize("amount", [-5, 0, 2.5, math.nan, math.inf, True, "5", None])
def test_earn_ignores_nonsense(wallet, amount):
    assert wallet.earn(amount) == 0
    assert wallet.stars == 0


def test_spend(wallet, repo):
    wallet.earn(10)
    assert not wallet.spend(11)
    assert wallet.stars == 10
    assert wallet.spend(4)
    assert wallet.stars == 6
    assert stored(repo)["stars"] == 6
    assert not wallet.spend(-1)
    assert not wallet.spend(1.5)
    assert wallet.spend(6)
    assert wallet.stars == 0
    assert not wallet.spend(1)  # never negative


def test_spend_is_undone_if_it_cannot_be_saved(wallet, repo):
    wallet.earn(10)
    repo.broken.add(KEY_WALLET)
    assert not wallet.spend(5)
    assert wallet.stars == 10


def test_earned_stars_stay_even_if_saving_fails(wallet, repo, caplog):
    repo.broken.add(KEY_WALLET)
    with caplog.at_level(logging.ERROR):
        assert wallet.earn(5) == 5
    assert wallet.stars == 5
    assert "could not save the wallet" in caplog.text
    repo.broken.clear()
    wallet.earn(1)
    assert stored(repo)["stars"] == 6


def test_daily_start_bonus_once_per_day_without_streaks(wallet, clock):
    assert wallet.daily_start() == DAILY_START_BONUS == 5
    assert wallet.daily_start() == 0
    assert wallet.stars == 5
    clock.advance(6 * DAY)  # away for days: nothing lost, the same bonus as always
    assert wallet.daily_start() == 5
    assert wallet.stars == 10


def test_a_clock_turned_back_does_not_repeat_the_bonus(wallet, repo, clock):
    today = clock.now()
    assert wallet.daily_start() == 5
    clock.set_now(today - DAY)  # the clock goes back a day (or the time zone changes)...
    restarted = Wallet(repo, clock)
    restarted.load()
    assert restarted.daily_start() == 0
    clock.set_now(today)  # ...and forward again: today's bonus was already paid
    assert restarted.daily_start() == 0
    assert restarted.stars == 5
    clock.set_now(today + DAY)  # a really new day
    assert restarted.daily_start() == 5


def test_a_clock_turned_back_does_not_reopen_the_daily_cap(wallet, clock):
    today = clock.now()
    assert wallet.earn(DAILY_EARN_CAP) == DAILY_EARN_CAP
    clock.set_now(today - DAY)
    assert wallet.earned_today == DAILY_EARN_CAP and wallet.remaining_today() == 0
    assert wallet.earn(10) == 0
    clock.set_now(today)
    assert wallet.earn(10) == 0
    assert wallet.stars == DAILY_EARN_CAP
    clock.set_now(today + DAY)
    assert wallet.earned_today == 0
    assert wallet.earn(10) == 10


def test_daily_bonus_does_not_count_towards_the_game_cap(wallet):
    wallet.daily_start()
    assert wallet.earned_today == 0
    assert wallet.earn(60) == 60
    assert wallet.stars == 65


def test_survives_a_restart(wallet, repo, clock):
    wallet.daily_start()
    wallet.earn(30)
    again = Wallet(repo, clock)
    again.load()
    assert (again.stars, again.earned_today) == (35, 30)
    assert again.daily_start() == 0  # the same day: no second bonus after a restart
    assert again.earn(40) == 30


def test_max_stars(wallet, repo, clock):
    repo.set(KEY_WALLET, {"stars": MAX_STARS - 3})
    wallet.load()
    assert wallet.earn(10) == 3
    assert wallet.stars == MAX_STARS
    assert wallet.daily_start() == 5
    assert wallet.stars == MAX_STARS


@pytest.mark.parametrize("data", [
    "garbage", [1, 2], 42, {"stars": -5}, {"stars": "many"}, {"stars": 2.5}, {"stars": True},
    {"stars": math.inf}, {"stars": None},
])
def test_corrupt_wallet_falls_back_to_defaults(repo, clock, data, caplog):
    repo.set(KEY_WALLET, data)
    wallet = Wallet(repo, clock)
    with caplog.at_level(logging.WARNING):
        wallet.load()
    assert wallet.stars == 0
    assert "wallet" in caplog.text


def test_partly_corrupt_wallet_keeps_what_is_valid(repo, clock, caplog):
    today = local_day(clock.now())
    repo.set(KEY_WALLET, {"stars": 12, "earned_day": "вчора", "earned_today": -3, "bonus_day": 5})
    wallet = Wallet(repo, clock)
    with caplog.at_level(logging.WARNING):
        wallet.load()
    assert wallet.stars == 12
    assert wallet.earned_today == 0
    assert wallet.daily_start() == 5  # bonus_day was unusable
    repo.set(KEY_WALLET, {"stars": 1, "earned_day": today, "earned_today": 999})
    wallet.load()
    assert wallet.earned_today == DAILY_EARN_CAP  # clamped: no more game stars today


def test_unreadable_json_in_the_database(db, repo, clock):
    db.conn.execute("INSERT INTO kv_state(key, value, updated_at) VALUES('wallet', '{nope', 0)")
    wallet = Wallet(repo, clock)
    wallet.load()
    assert wallet.stars == 0


def test_listeners(wallet):
    calls = []
    remove = wallet.add_listener(lambda: calls.append("a"))

    def broken():
        raise RuntimeError("boom")

    wallet.add_listener(broken)
    wallet.earn(1)
    wallet.spend(1)
    assert calls == ["a", "a"]
    remove()
    wallet.earn(1)
    assert calls == ["a", "a"]


# =========================================================================================
# Catalog
# =========================================================================================


@pytest.fixture(scope="module")
def catalog():
    return GiftCatalog.load()


def test_packaged_catalog(catalog):
    assert len(catalog) == 10
    prices = {item.id: (item.price, item.kind) for item in catalog}
    assert prices["cookie"] == (5, GiftKind.TREAT)
    assert prices["cocoa"] == (8, GiftKind.TREAT)
    assert prices["flower"] == (12, GiftKind.TREAT)
    assert prices["book"] == (15, GiftKind.TREAT)
    assert prices["cake"] == (25, GiftKind.TREAT)
    assert prices["plush_cat"] == (30, GiftKind.KEEPSAKE)
    assert prices["headphones"] == (40, GiftKind.KEEPSAKE)
    assert prices["star_keychain"] == (20, GiftKind.KEEPSAKE)
    assert [i.price for i in catalog] == sorted(i.price for i in catalog)  # cheapest first


def test_catalog_file_has_no_invalid_entries(catalog):
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert len(raw["gifts"]) == len(catalog)


def test_every_gift_is_complete(catalog):
    bank = PhraseBank.load()
    states = {s.value for s in CharacterState}
    portraits = character_pack_dir("mia") / "portraits"
    for item in catalog:
        assert item.name and item.icon and item.description, item.id
        assert item.effects and item.effects_text, item.id
        assert all(-50 <= v <= 50 for v in item.effects.values()), item.id
        assert item.effects.get("friendship_xp", 0) > 0, item.id  # every gift is a bit of friendship
        assert item.reaction_keys == (f"gift.{item.id}",)
        assert all(len(bank.phrases(key)) >= 4 for key in item.reaction_keys), item.id
        assert item.state in states, item.id
        if portraits.is_dir():
            assert (portraits / f"{item.portrait}.png").is_file(), item.id
        if item.keepsake:
            assert item.memory.startswith("Мені подарували "), item.id
        else:
            assert item.memory == ""
    assert {"gift.generic", "gift.again"} <= set(bank.keys())


def entry(id_="x", **extra):
    base = {"id": id_, "name": "Ікс", "icon": "x", "price": 5, "kind": "treat", "description": "d",
            "effects": {"happiness": 3}, "state": "happy", "portrait": "joy"}
    base.update(extra)
    return base


def test_invalid_entries_are_skipped(caplog):
    data = {"version": 1, "gifts": [
        entry("ok"),
        entry("ok", name="Дубль"),
        entry("Bad Id"),
        entry("noname", name=""),
        entry("longname", name="я" * 41),
        entry("free", price=0),
        entry("pricey", price=1001),
        entry("boolprice", price=True),
        entry("strprice", price="5"),
        entry("kind", kind="pet"),
        entry("noeffects", effects={}),
        entry("badeffect", effects={"mood": 3}),
        entry("naneffect", effects={"happiness": "NaN"}),
        entry("bigeffect", effects={"happiness": 51}),
        entry("lessfriend", effects={"friendship_xp": -1}),
        entry("nomemory", kind="keepsake"),
        entry("badstate", state="flying"),
        entry("liststate", state=["happy"]),
        entry("listkind", kind=["treat"]),
        entry("nokeys", reaction_keys=[]),
        entry("badkeys", reaction_keys=["", 3]),
        entry("badportrait", portrait=7),
        "not an object",
        entry("keep", kind="keepsake", memory="Мені подарували ікс",
              reaction_keys=["gift.keep", "gift.keep"]),
    ]}
    with caplog.at_level(logging.WARNING):
        catalog = GiftCatalog.from_dict(data)
    assert [i.id for i in catalog] == ["ok", "keep"]
    assert catalog.get("keep").reaction_keys == ("gift.keep",)
    assert catalog.get("keep").keepsake
    assert catalog.get("ok").reaction_keys == ("gift.ok",)
    assert "duplicate id" in caplog.text


def test_treat_memory_is_dropped():
    catalog = GiftCatalog.from_dict({"version": 1, "gifts": [entry("t", memory="Мені подарували т")]})
    assert catalog.get("t").memory == ""


def test_effects_are_read_only():
    item = GiftCatalog.from_dict({"version": 1, "gifts": [entry("t")]}).get("t")
    with pytest.raises(TypeError):
        item.effects["happiness"] = 100


@pytest.mark.parametrize("content", ["{nope", json.dumps({"gifts": "x"}), json.dumps([1])])
def test_unreadable_catalog_gives_an_empty_shop(tmp_path, content, caplog):
    path = tmp_path / "gifts.json"
    path.write_text(content, encoding="utf-8")
    with caplog.at_level(logging.ERROR):
        assert len(GiftCatalog.load(path)) == 0
    assert "the shop is empty" in caplog.text


def test_missing_catalog(tmp_path):
    assert len(GiftCatalog.load(tmp_path / "missing.json")) == 0


def test_describe_effects():
    assert describe_effects({"hunger": -10, "happiness": 4, "friendship_xp": 1}) == \
        "Радість +4 · Ситість +10 · Дружба +1"
    assert describe_effects({"boredom": -15, "energy": 2.5, "affection": 0}) == "Енергія +2.5 · Нудьга −15"
    assert describe_effects({}) == ""


@pytest.mark.parametrize(("n", "text"), [(1, "1 зірочка"), (3, "3 зірочки"), (5, "5 зірочок"),
                                         (11, "11 зірочок"), (21, "21 зірочка"), (0, "0 зірочок")])
def test_stars_uk(n, text):
    assert stars_uk(n) == text


# =========================================================================================
# GiftService
# =========================================================================================


@pytest.fixture
def world(db, bus, clock):
    return World(db, bus, clock)


@pytest.fixture
def gifts(world):
    return world.gifts()


def rich(gifts, stars=200):
    gifts.wallet.earn(min(stars, DAILY_EARN_CAP))
    if stars > DAILY_EARN_CAP:
        gifts._repo.set(KEY_WALLET, {**gifts.wallet.to_dict(), "stars": stars})
        gifts.wallet.load()
    return gifts


def test_buy_needs_stars(gifts, world):
    ok, message = gifts.buy("cookie")
    assert not ok
    assert message == "Поки що бракує 5 ⭐. Зірочки приносять міні-ігри — пограймо?"
    assert gifts.inventory() == {}
    assert world.repo.get(KEY_INVENTORY) is None


def test_buy_puts_it_in_the_inventory(gifts, world):
    rich(gifts, 12)
    world.repo.writes.clear()
    ok, message = gifts.buy("cookie")
    assert ok and message == "Куплено: 🍪 Печиво. Подарунок чекає в інвентарі."
    assert world.repo.writes == [KEY_INVENTORY, KEY_WALLET]  # the item first: stars never go alone
    assert gifts.wallet.stars == 7
    assert gifts.inventory() == {"cookie": 1}
    assert world.repo.get(KEY_INVENTORY) == {"cookie": 1}
    gifts.buy("cookie")
    assert gifts.owned("cookie") == 2
    assert gifts.wallet.stars == 2


def test_unknown_gift(gifts):
    assert gifts.buy("unicorn") == (False, MSG_UNKNOWN)
    assert gifts.give("unicorn") == (False, MSG_UNKNOWN)
    assert gifts.buy_and_give("unicorn") == (False, MSG_UNKNOWN)


def test_give_needs_the_gift(gifts):
    ok, message = gifts.give("cookie")
    assert not ok and "Спершу купи" in message


def test_give_applies_effects_and_she_reacts(gifts, world):
    rich(gifts, 10)
    gifts.buy("cookie")
    world.recorder.drain()
    before = world.tamagotchi.needs.to_dict()
    ok, message = gifts.give("cookie")
    assert ok and message == "Мія отримала подарунок: 🍪 Печиво."
    after = world.tamagotchi.needs
    assert after.hunger == approx(before["hunger"] - 10)
    assert after.happiness == approx(before["happiness"] + 4)
    assert after.friendship_xp == approx(before["friendship_xp"] + 1)
    assert gifts.inventory() == {}
    assert world.repo.get(KEY_INVENTORY) == {}
    assert world.composer.calls == [(("gift.cookie", "gift.generic"), True)]
    assert world.composer.variables == [{"gift": "Печиво"}]  # context for the AI brain
    assert world.recorder.of(NeedsChanged)


def test_her_name_comes_from_the_settings(gifts, world):
    world.settings.character.name = "Ліна"
    rich(gifts, 10)
    ok, message = gifts.buy_and_give("apple")
    assert ok and message.startswith("Ліна отримала подарунок")


def test_same_treat_again_soon_has_half_the_effect(gifts, world, clock):
    rich(gifts, 30)
    for _ in range(3):
        gifts.buy("cookie")
    world.tamagotchi.needs.hunger = 80
    gifts.give("cookie")
    assert world.tamagotchi.needs.hunger == approx(70)
    clock.advance(60)
    ok, message = gifts.give("cookie")
    assert ok and "ефект наполовину" in message
    assert world.tamagotchi.needs.hunger == approx(65, abs=0.2)  # half of -10 (+ a minute of hunger)
    assert world.composer.keys[-1] == ("gift.again", "gift.cookie", "gift.generic")
    clock.advance(REPEAT_WINDOW_SECONDS)  # a normal pause later: full effect again
    world.tamagotchi.needs.hunger = 80
    world.tamagotchi.tick()
    hunger = world.tamagotchi.needs.hunger
    gifts.give("cookie")
    assert world.tamagotchi.needs.hunger == approx(hunger - 10)
    assert world.composer.keys[-1] == ("gift.cookie", "gift.generic")


def test_a_different_treat_is_not_a_repeat(gifts, world):
    rich(gifts, 30)
    gifts.buy_and_give("cookie")
    gifts.buy_and_give("apple")
    assert world.composer.keys[-1] == ("gift.apple", "gift.generic")


def test_keepsake_is_remembered_once_and_kept_forever(gifts, world, bus, clock):
    rich(gifts, 100)
    ok, message = gifts.buy_and_give("plush_cat")
    assert ok and "скарб назавжди" in message
    assert [i.id for i in gifts.collection()] == ["plush_cat"]
    assert gifts.has_keepsake("plush_cat")
    assert world.repo.get(KEY_COLLECTION) == {"plush_cat": clock.now()}
    memories = world.memory.list("relationship")
    assert [(m.text, m.source) for m in memories] == [("Мені подарували плюшевого котика", "system")]
    # Owned forever: it cannot be bought (or remembered) again.
    assert gifts.can_buy("plush_cat") == (False, "🐱 Плюшевий котик уже в її колекції — назавжди.")
    assert not gifts.buy("plush_cat")[0]
    assert len(world.memory.list("relationship")) == 1
    assert world.composer.keys[-1] == ("gift.plush_cat", "gift.generic")


def test_keepsake_waiting_in_the_inventory_cannot_be_bought_twice(gifts):
    rich(gifts, 100)
    assert gifts.buy("headphones")[0]
    ok, message = gifts.buy("headphones")
    assert not ok and message == "🎧 Навушники вже є в інвентарі — подаруй звідти!"
    assert gifts.wallet.stars == 60
    assert gifts.collection() == []
    assert gifts.give("headphones")[0]
    assert [i.id for i in gifts.collection()] == ["headphones"]


def test_keepsakes_collect_in_order(gifts, clock):
    rich(gifts, 200)
    gifts.buy_and_give("headphones")
    clock.advance(5)
    gifts.buy_and_give("star_keychain")
    assert [i.id for i in gifts.collection()] == ["headphones", "star_keychain"]


def test_no_memory_without_remember_facts(gifts, world):
    world.settings.ai.remember_facts = False
    rich(gifts, 100)
    gifts.buy_and_give("plush_cat")
    assert world.memory.list() == []
    assert gifts.has_keepsake("plush_cat")


def test_works_without_a_memory_store(db, bus, clock):
    world = World(db, bus, clock, memory=False)
    gifts = rich(world.gifts(), 100)
    assert gifts.buy_and_give("plush_cat")[0]


def test_failing_memory_does_not_spoil_the_gift(gifts, world, monkeypatch, caplog):
    def boom(*_a, **_k):
        raise RuntimeError("db gone")

    monkeypatch.setattr(world.memory, "add", boom)
    rich(gifts, 100)
    with caplog.at_level(logging.ERROR):
        assert gifts.buy_and_give("plush_cat")[0]
    assert "could not remember the keepsake" in caplog.text


def test_buy_and_give(gifts, world):
    rich(gifts, 30)
    ok, message = gifts.buy_and_give("cake")
    assert ok and message == "Мія отримала подарунок: 🎂 Торт."
    assert gifts.wallet.stars == 5
    assert gifts.inventory() == {}
    assert world.composer.keys == [("gift.cake", "gift.generic")]


def test_buy_and_give_without_stars_changes_nothing(gifts, world):
    rich(gifts, 3)
    ok, message = gifts.buy_and_give("cake")
    assert not ok and "бракує 22 ⭐" in message
    assert world.composer.calls == []
    assert gifts.wallet.stars == 3


def test_stack_limit(gifts, world):
    world.repo.set(KEY_INVENTORY, {"apple": MAX_STACK})
    gifts.load()
    rich(gifts, 10)
    assert gifts.buy("apple") == (False, MSG_STACK_FULL)


def test_state_survives_a_restart(gifts, world):
    rich(gifts, 200)
    gifts.buy("cookie")
    gifts.buy("cookie")
    gifts.buy_and_give("plush_cat")
    again = world.gifts()
    assert again.wallet.stars == gifts.wallet.stars
    assert again.inventory() == {"cookie": 2}
    assert [i.id for i in again.collection()] == ["plush_cat"]


def test_corrupt_inventory_and_collection(world, caplog):
    world.repo.set(KEY_INVENTORY, {"cookie": "two", "apple": 2, "Bad Id": 1, "cake": -1, "book": 1.5})
    world.repo.set(KEY_COLLECTION, {"plush_cat": "yesterday", "../x": 1})
    with caplog.at_level(logging.WARNING):
        gifts = world.gifts()
    assert gifts.inventory() == {"apple": 2}
    assert [i.id for i in gifts.collection()] == ["plush_cat"]  # a broken date never costs a keepsake
    world.repo.set(KEY_INVENTORY, ["nope"])
    world.repo.set(KEY_COLLECTION, "nope")
    gifts.load()
    assert gifts.inventory() == {}
    assert gifts.collection() == []


def test_items_unknown_to_this_catalog_are_kept_on_disk(world):
    world.repo.set(KEY_INVENTORY, {"retired_gift": 3, "apple": 1})
    gifts = rich(world.gifts(), 10)
    assert gifts.inventory() == {"apple": 1}  # not shown...
    gifts.buy("apple")
    assert world.repo.get(KEY_INVENTORY) == {"retired_gift": 3, "apple": 2}  # ...but never lost


def test_purchase_rolls_back_if_the_wallet_cannot_be_saved(gifts, world):
    rich(gifts, 10)
    world.repo.broken.add(KEY_WALLET)
    assert gifts.buy("cookie") == (False, MSG_SAVE_FAILED)
    assert gifts.wallet.stars == 10
    assert gifts.inventory() == {}
    assert world.repo.get(KEY_INVENTORY) == {}


def test_purchase_fails_cleanly_if_the_inventory_cannot_be_saved(gifts, world):
    rich(gifts, 10)
    world.repo.broken.add(KEY_INVENTORY)
    assert gifts.buy("cookie") == (False, MSG_SAVE_FAILED)
    assert gifts.wallet.stars == 10
    assert gifts.inventory() == {}


def test_give_fails_cleanly_if_the_inventory_cannot_be_saved(gifts, world):
    rich(gifts, 10)
    gifts.buy("cookie")
    world.repo.broken.add(KEY_INVENTORY)
    assert gifts.give("cookie") == (False, MSG_SAVE_FAILED)
    assert gifts.inventory() == {"cookie": 1}
    assert world.composer.calls == []


def test_keepsake_is_never_lost_when_the_collection_cannot_be_saved(gifts, world):
    rich(gifts, 100)
    gifts.buy("plush_cat")
    world.repo.broken.add(KEY_COLLECTION)
    assert gifts.give("plush_cat") == (False, MSG_SAVE_FAILED)
    assert gifts.owned("plush_cat") == 1  # still in the user's hands
    assert world.repo.get(KEY_INVENTORY) == {"plush_cat": 1}
    assert gifts.collection() == []
    assert world.memory.list() == [] and world.composer.calls == []


def test_keepsake_rolls_back_when_the_inventory_cannot_be_saved(gifts, world):
    rich(gifts, 100)
    gifts.buy("plush_cat")
    world.repo.broken.add(KEY_INVENTORY)
    assert gifts.give("plush_cat") == (False, MSG_SAVE_FAILED)
    assert gifts.owned("plush_cat") == 1
    assert gifts.collection() == []
    assert world.repo.get(KEY_COLLECTION) == {}
    world.repo.broken.clear()
    assert gifts.give("plush_cat")[0]  # works once the disk is fine again
    assert [i.id for i in world.gifts().collection()] == ["plush_cat"]


def test_fallback_line_if_the_phrase_bank_has_nothing(gifts, world):
    world.composer.result = False
    rich(gifts, 10)
    gifts.buy_and_give("cocoa")
    (speech,) = world.recorder.of(SpeechRequested)
    assert (speech.text, speech.state, speech.portrait, speech.solicited) == \
        (FALLBACK_THANKS, "happy", "drink", True)


def test_fallback_line_if_the_composer_fails(gifts, world):
    world.composer.error = RuntimeError("no")
    rich(gifts, 10)
    assert gifts.buy_and_give("cocoa")[0]
    assert [s.text for s in world.recorder.of(SpeechRequested)] == [FALLBACK_THANKS]


def test_failing_needs_update_does_not_stop_the_reaction(gifts, world, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("no")

    monkeypatch.setattr(world.tamagotchi, "apply_effect", boom)
    rich(gifts, 10)
    assert gifts.buy_and_give("cookie")[0]
    assert world.composer.calls


def test_daily_start_bonus_on_app_start(gifts, world, bus, clock):
    bus.publish(AppStarted())
    bus.dispatch_pending()
    assert gifts.wallet.stars == 5
    bus.publish(AppStarted())  # a restart the same day
    bus.dispatch_pending()
    assert gifts.wallet.stars == 5
    clock.advance(DAY)
    bus.publish(AppStarted())
    bus.dispatch_pending()
    assert gifts.wallet.stars == 10


def test_listeners_hear_every_change(gifts):
    calls = []
    remove = gifts.add_listener(lambda: calls.append(1))
    rich(gifts, 10)
    n = len(calls)
    assert n >= 1
    gifts.buy("cookie")
    assert len(calls) > n
    n = len(calls)
    gifts.give("cookie")
    assert len(calls) > n
    remove()
    n = len(calls)
    gifts.wallet.earn(1)
    assert len(calls) == n


def test_close_unsubscribes(gifts, bus):
    gifts.close()
    bus.publish(AppStarted())
    bus.dispatch_pending()
    assert gifts.wallet.stars == 0


def test_reload_after_delete_all(gifts, world, db):
    rich(gifts, 100)
    gifts.buy_and_give("plush_cat")
    gifts.buy("cookie")
    db.wipe_all()
    gifts.load()
    assert (gifts.wallet.stars, gifts.inventory(), gifts.collection()) == (0, {}, [])


def test_plain_repository_works_too(db, bus, clock):
    world = World(db, bus, clock, repo=StateRepository(db, clock))
    gifts = rich(world.gifts(), 10)
    assert gifts.buy_and_give("cookie")[0]


def test_with_the_real_composer_and_phrase_bank(db, bus, clock):
    """End to end without fakes for speech: the extra ``gift`` variable is
    harmless for the phrase bank, and a line from the gift's own key is said."""
    import random

    from chibi.memory.repositories import PhraseHistoryRepository
    from chibi.personality.composer import SpeechComposer
    from chibi.personality.phrases import PhraseSelector

    world = World(db, bus, clock)
    selector = PhraseSelector(PhraseBank.load(), PhraseHistoryRepository(db), world.settings_getter, clock,
                              random.Random(0))
    world.composer = SpeechComposer(bus, selector, None, world.settings_getter, clock)
    gifts = rich(world.gifts(), 20)
    gifts.buy("cookie")
    gifts.buy("cookie")
    gifts.give("cookie")
    gifts.give("cookie")
    spoken = world.recorder.of(SpeechRequested)
    assert [s.source for s in spoken] == ["gift.cookie", "gift.again"]
    assert all(s.solicited and s.text for s in spoken)
