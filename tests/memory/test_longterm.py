"""MemoryStore: sanitation, privacy, near-duplicates, capacity, BM25 search."""

import logging
import math

import pytest

from chibi.core.events import Event, MemoryChanged
from chibi.memory import longterm
from chibi.memory.database import Database
from chibi.memory.longterm import (
    MAX_TEXT_CHARS,
    MEMORY_KINDS,
    Memory,
    MemoryRejectedError,
    MemoryStore,
    _bm25,
    sanitize_text,
    tokenize,
)

DAY = 24 * 3600.0


class Recorder:
    def __init__(self, bus):
        self.bus = bus
        self.events: list[Event] = []
        bus.subscribe(Event, self.events.append)

    def drain(self) -> list[Event]:
        self.bus.dispatch_pending()
        out, self.events[:] = list(self.events), []
        return out

    def memory_kinds(self) -> list[str]:
        return [e.kind for e in self.drain() if isinstance(e, MemoryChanged)]


@pytest.fixture
def store(db, bus, clock):
    return MemoryStore(db, bus, clock)


@pytest.fixture
def rec(bus):
    return Recorder(bus)


def rows(db) -> list[dict]:
    return [dict(r) for r in db.conn.execute("SELECT * FROM memories ORDER BY id")]


# -- kinds / sources ------------------------------------------------------------------------


def test_memory_kinds_match_the_facade():
    from chibi.settings.facade import MEMORY_KINDS as FACADE_KINDS

    assert MEMORY_KINDS == FACADE_KINDS


def test_unknown_kind_or_source_is_rejected(store):
    with pytest.raises(ValueError):
        store.add("secret", "Щось")
    with pytest.raises(ValueError):
        store.add("fact", "Щось", source="screen")
    with pytest.raises(ValueError):
        store.clear("nope")
    with pytest.raises(ValueError):
        store.list(kind="nope")
    with pytest.raises(ValueError):
        store.search("кіт", kinds=["nope"])
    with pytest.raises(ValueError):
        store.count("nope")


@pytest.mark.parametrize("source", ["chat", "system", "game", "user"])
def test_every_source_is_accepted(store, source):
    memory = store.add("fact", f"Факт із джерела {source}", source=source)
    assert memory is not None and memory.source == source


# -- add ----------------------------------------------------------------------------------


def test_add_stores_a_memory(store, db, clock, rec):
    memory = store.add("fact", "Кота звати Мурчик", importance=0.8)
    assert isinstance(memory, Memory)
    assert memory.kind == "fact"
    assert memory.text == "Кота звати Мурчик"
    assert memory.source == "chat"
    assert memory.importance == 0.8
    assert memory.pinned is False
    assert memory.created_at == memory.updated_at == clock.now()
    assert memory.last_used_at is None
    assert rows(db)[0]["text"] == "Кота звати Мурчик"
    assert store.get(memory.id) == memory
    assert rec.memory_kinds() == ["fact"]


def test_memories_survive_a_new_store(db, bus, clock):
    MemoryStore(db, bus, clock).add("preference", "Любить зелений чай")
    again = MemoryStore(db, None, clock)
    assert [m.text for m in again.list()] == ["Любить зелений чай"]


def test_works_without_a_bus(db, clock):
    store = MemoryStore(db, None, clock)
    memory = store.add("joke", "Жарт про кота в коробці")
    assert memory is not None
    store.update(memory.id, "Жарт про кота в мішку")
    store.set_pinned(memory.id, True)
    store.delete(memory.id)
    store.clear()


@pytest.mark.parametrize("text", ["", "   ", "\n\t", "\x00\x01", "​​", None, 42])
def test_empty_text_is_not_stored(store, db, rec, text):
    assert store.add("fact", text) is None  # type: ignore[arg-type]
    assert rows(db) == []
    assert rec.memory_kinds() == []


@pytest.mark.parametrize("text", [
    "Номер картки 4111 1111 1111 1111",
    "Мій пароль: kotik2020",
    "Телефон +380 67 123 45 67",
    "Пошта roman@example.com",
    "IBAN UA213223130000026007233566001",
    "Ключ sk-ant-api03-AbCdEf1234567890xyzXYZ",
])
def test_sensitive_text_is_never_stored(store, db, rec, text):
    assert store.add("fact", text, source="user") is None
    assert rows(db) == []
    assert rec.memory_kinds() == []


def test_sensitive_part_beyond_the_cap_still_blocks(store, db):
    text = "слово " * 60 + "картка 4111 1111 1111 1111"
    assert len(text) > MAX_TEXT_CHARS
    assert store.add("fact", text) is None
    assert rows(db) == []


def test_huge_input_is_bounded(store):
    memory = store.add("fact", "Мурчик " * 200_000)
    assert memory is not None
    assert len(memory.text) <= MAX_TEXT_CHARS


@pytest.mark.parametrize("importance, stored", [(-1, 0.0), (2, 1.0), (0.3, 0.3),
                                                 (math.nan, 0.5), ("x", 0.5)])
def test_importance_is_clamped(store, importance, stored):
    memory = store.add("fact", f"Факт номер {importance!r}", importance=importance)
    assert memory is not None and memory.importance == stored


# -- sanitation -------------------------------------------------------------------------------


def test_sanitize_collapses_whitespace_and_removes_controls():
    assert sanitize_text("  Кіт\tМурчик\n\nлюбить\x00 рибу‮  ") == "Кіт Мурчик любить рибу"


def test_sanitize_keeps_emoji_sequences():
    assert sanitize_text("Програмістка 👩‍💻") == "Програмістка 👩‍💻"


def test_sanitize_short_text_is_unchanged():
    text = "а" * MAX_TEXT_CHARS
    assert sanitize_text(text) == text


def test_long_text_is_cut_at_a_word_boundary_with_an_ellipsis(store):
    text = " ".join(f"слово{i}" for i in range(100))
    memory = store.add("fact", text)
    assert memory is not None
    assert len(memory.text) <= MAX_TEXT_CHARS
    assert memory.text.endswith("…")
    body = memory.text[:-1]
    assert text.startswith(body)
    # the cut is right before a space: no half word
    assert text[len(body)] == " "


def test_one_giant_word_is_cut_hard():
    text = "а" * 1000
    capped = sanitize_text(text)
    assert len(capped) == MAX_TEXT_CHARS
    assert capped.endswith("…")


def test_trailing_punctuation_before_the_ellipsis_is_dropped():
    text = "а" * 250 + ", " + "б" * 100
    assert sanitize_text(text) == "а" * 250 + "…"


# -- near-duplicates --------------------------------------------------------------------------


def test_near_duplicate_updates_the_existing_row(store, db, clock, rec):
    first = store.add("fact", "Мурчик рудий кіт", importance=0.4)
    rec.drain()
    clock.advance(60)
    second = store.add("fact", "Мурчик рудий пухнастий кіт", importance=0.7)
    assert second is not None and first is not None
    assert second.id == first.id
    assert second.text == "Мурчик рудий пухнастий кіт"  # the longer one
    assert second.importance == 0.7
    assert second.created_at == first.created_at
    assert second.updated_at == clock.now()
    assert len(rows(db)) == 1
    assert rec.memory_kinds() == ["fact"]


def test_shorter_duplicate_keeps_the_longer_text_and_max_importance(store, db):
    store.add("fact", "Мурчик рудий пухнастий кіт", importance=0.9)
    memory = store.add("fact", "Мурчик рудий кіт", importance=0.2)
    assert memory is not None
    assert memory.text == "Мурчик рудий пухнастий кіт"
    assert memory.importance == 0.9
    assert len(rows(db)) == 1


def test_stopword_only_difference_is_a_duplicate(store, db):
    store.add("preference", "Користувач любить каву з молоком")
    store.add("preference", "Користувач дуже любить каву з молоком")
    assert [r["text"] for r in rows(db)] == ["Користувач дуже любить каву з молоком"]


def test_exact_duplicate_ignoring_case_and_spaces(store, db):
    store.add("fact", "Кота звати Мурчик")
    store.add("fact", "  кота   ЗВАТИ мурчик ")
    assert len(rows(db)) == 1


def test_negated_restatement_replaces_the_old_fact(store, db):
    store.add("preference", "Користувач не любить каву з молоком")
    store.add("preference", "Користувач любить каву з молоком")
    assert [r["text"] for r in rows(db)] == ["Користувач любить каву з молоком"]


def test_different_facts_are_separate(store, db):
    store.add("fact", "Мурчик рудий кіт")
    store.add("fact", "Мурчик сірий кіт")  # Jaccard 2/4
    assert len(rows(db)) == 2


def test_automatic_restatement_never_rewrites_a_user_memory(store, db):
    mine = store.add("fact", "Мурчик рудий кіт", source="user", importance=0.3)
    merged = store.add("fact", "Мурчик рудий пухнастий кіт", source="chat", importance=0.8)
    assert mine and merged and merged.id == mine.id
    assert merged.text == "Мурчик рудий кіт"
    assert merged.importance == 0.8
    assert len(rows(db)) == 1
    # the user restating it does update the text
    again = store.add("fact", "Мурчик рудий пухнастий кіт", source="user")
    assert again is not None and again.text == "Мурчик рудий пухнастий кіт"


def test_automatic_restatement_never_rewrites_a_pinned_memory(store):
    pinned = store.add("fact", "Мурчик рудий кіт", source="chat")
    assert pinned is not None
    store.set_pinned(pinned.id, True)
    merged = store.add("fact", "Мурчик рудий пухнастий кіт", source="chat")
    assert merged is not None and merged.text == "Мурчик рудий кіт" and merged.pinned


def test_duplicates_are_only_merged_within_a_kind(store, db):
    store.add("fact", "Любить каву з корицею")
    store.add("preference", "Любить каву з корицею")
    assert sorted(r["kind"] for r in rows(db)) == ["fact", "preference"]


# -- capacity ----------------------------------------------------------------------------------


@pytest.fixture
def small_capacity(monkeypatch):
    monkeypatch.setattr(longterm, "MAX_MEMORIES", 3)


def test_capacity_evicts_the_least_important(store, db, clock, small_capacity):
    store.add("fact", "Перший факт про кота", importance=0.9)
    store.add("fact", "Другий факт про собаку", importance=0.2)
    store.add("fact", "Третій факт про папугу", importance=0.5)
    store.add("fact", "Четвертий факт про рибку", importance=0.5)
    texts = [r["text"] for r in rows(db)]
    assert len(texts) == 3
    assert "Другий факт про собаку" not in texts


def test_capacity_never_evicts_pinned(store, db, small_capacity):
    low = store.add("fact", "Перший факт про кота", importance=0.1)
    assert low is not None
    store.set_pinned(low.id, True)
    store.add("fact", "Другий факт про собаку", importance=0.5)
    store.add("fact", "Третій факт про папугу", importance=0.6)
    store.add("fact", "Четвертий факт про рибку", importance=0.7)
    texts = [r["text"] for r in rows(db)]
    assert "Перший факт про кота" in texts
    assert "Другий факт про собаку" not in texts


def test_capacity_prefers_evicting_the_least_recently_used(store, db, clock, small_capacity):
    store.add("fact", "Мурчик любить рибу", importance=0.5)
    clock.advance(10)
    store.add("fact", "Сестру звати Оля", importance=0.5)
    clock.advance(10)
    store.add("fact", "Грає в Dota щовечора", importance=0.5)
    clock.advance(10)
    assert store.search("Мурчик")  # the oldest one was just used
    clock.advance(10)
    store.add("fact", "Живе у Львові", importance=0.5)
    texts = [r["text"] for r in rows(db)]
    assert "Мурчик любить рибу" in texts
    assert "Сестру звати Оля" not in texts


def test_new_memory_is_kept_even_if_least_important(store, db, small_capacity):
    store.add("fact", "Перший факт про кота", importance=0.9)
    store.add("fact", "Другий факт про собаку", importance=0.8)
    store.add("fact", "Третій факт про папугу", importance=0.7)
    new = store.add("fact", "Четвертий факт про рибку", importance=0.0)
    assert new is not None
    assert store.get(new.id) is not None
    assert len(rows(db)) == 3


def test_eviction_of_another_kind_announces_all_kinds(store, rec, small_capacity):
    store.add("joke", "Жарт про кота", importance=0.0)
    store.add("fact", "Другий факт про собаку", importance=0.5)
    store.add("fact", "Третій факт про папугу", importance=0.5)
    rec.drain()
    store.add("fact", "Четвертий факт про рибку", importance=0.5)
    assert rec.memory_kinds() == [""]
    assert store.count("joke") == 0


def test_default_capacity_is_2000():
    assert longterm.MAX_MEMORIES == 2000


# -- update / pin / delete / clear ------------------------------------------------------------


def test_update_changes_text_and_announces(store, clock, rec):
    memory = store.add("relationship", "Сестру звати Оля")
    assert memory is not None
    rec.drain()
    clock.advance(5)
    store.update(memory.id, "  Сестру звати   Олена ")
    updated = store.get(memory.id)
    assert updated is not None
    assert updated.text == "Сестру звати Олена"
    assert updated.updated_at == clock.now()
    assert rec.memory_kinds() == ["relationship"]
    # the new text is what search finds
    assert [m.id for m in store.search("Олена")] == [memory.id]
    assert store.search("Оля") == []


def test_update_with_same_text_is_silent(store, rec):
    memory = store.add("fact", "Кота звати Мурчик")
    assert memory is not None
    rec.drain()
    store.update(memory.id, "Кота звати Мурчик")
    assert rec.memory_kinds() == []


def test_update_refuses_sensitive_text(store, rec):
    memory = store.add("fact", "Телефон сестри є в контактах")
    assert memory is not None
    rec.drain()
    with pytest.raises(MemoryRejectedError) as info:
        store.update(memory.id, "Телефон сестри 067 123 45 67")
    assert isinstance(info.value, ValueError)
    assert "особисті дані" in info.value.user_message
    assert store.get(memory.id).text == "Телефон сестри є в контактах"
    assert rec.memory_kinds() == []


def test_update_refuses_empty_text(store):
    memory = store.add("fact", "Кота звати Мурчик")
    assert memory is not None
    with pytest.raises(MemoryRejectedError):
        store.update(memory.id, "   ")


def test_update_of_unknown_id_is_ignored(store, rec):
    store.update(999, "Щось нове")
    assert rec.memory_kinds() == []


def test_set_pinned(store, rec):
    memory = store.add("fact", "Кота звати Мурчик")
    assert memory is not None
    rec.drain()
    store.set_pinned(memory.id, True)
    assert store.get(memory.id).pinned is True
    assert rec.memory_kinds() == ["fact"]
    store.set_pinned(memory.id, True)  # no change, no event
    assert rec.memory_kinds() == []
    store.set_pinned(memory.id, False)
    assert store.get(memory.id).pinned is False
    store.set_pinned(12345, True)
    assert rec.memory_kinds() == ["fact"]


def test_delete(store, db, rec):
    memory = store.add("joke", "Жарт про кота в коробці")
    assert memory is not None
    rec.drain()
    store.delete(memory.id)
    assert rows(db) == []
    assert rec.memory_kinds() == ["joke"]
    store.delete(memory.id)
    assert rec.memory_kinds() == []


def test_deleted_text_does_not_linger_in_the_token_cache(store):
    memory = store.add("fact", "Таємний факт про Мурчика")
    assert memory is not None
    store.search("Мурчик")
    store.delete(memory.id)
    assert memory.id not in store._token_cache
    other = store.add("fact", "Сестру звати Оля")
    store.clear()
    assert other is not None and other.id not in store._token_cache


def test_clear_one_kind(store, db, rec):
    store.add("fact", "Кота звати Мурчик")
    store.add("joke", "Жарт про кота")
    rec.drain()
    store.clear("joke")
    assert [r["kind"] for r in rows(db)] == ["fact"]
    assert rec.memory_kinds() == ["joke"]


def test_clear_all(store, db, rec):
    store.add("fact", "Кота звати Мурчик")
    store.add("joke", "Жарт про кота")
    rec.drain()
    store.clear()
    assert rows(db) == []
    assert rec.memory_kinds() == [""]


# -- list ------------------------------------------------------------------------------------------


def test_list_orders_pinned_first_then_recent(store, clock):
    a = store.add("fact", "Кота звати Мурчик")
    clock.advance(1)
    b = store.add("fact", "Сестру звати Оля")
    clock.advance(1)
    c = store.add("preference", "Любить зелений чай")
    assert a and b and c
    store.set_pinned(a.id, True)
    assert [m.id for m in store.list()] == [a.id, c.id, b.id]
    assert [m.id for m in store.list(kind="fact")] == [a.id, b.id]
    assert [m.id for m in store.list(limit=1)] == [a.id]
    assert store.list(limit=0) == []


def test_list_query_is_a_case_and_apostrophe_insensitive_substring(store):
    store.add("preference", "Обожнює М’ЯСО по-французьки")
    store.add("fact", "Сестру звати Оля")
    assert [m.text for m in store.list(query="м'ясо")] == ["Обожнює М’ЯСО по-французьки"]
    assert [m.text for m in store.list(query="ОЛЯ")] == ["Сестру звати Оля"]
    assert store.list(query="хтось") == []


def test_count(store):
    store.add("fact", "Кота звати Мурчик")
    store.add("joke", "Жарт про кота")
    assert store.count() == 2
    assert store.count("fact") == 1


# -- tokens ----------------------------------------------------------------------------------------


def test_tokenize_rules():
    assert tokenize("кіт Мурчик") == ["кіт", "мурч"]
    assert tokenize("мій кіт") == ["кіт"]  # stopword dropped
    assert tokenize("м’ясо мʼясо м'ясо м`ясо") == ["мяс"] * 4
    assert tokenize("кава каву кавою каві") == ["кав"] * 4
    assert tokenize("Мурчика") == tokenize("Мурчик")
    assert tokenize("Я люблю S.T.A.L.K.E.R. 2") == ["любл", "stalk", "2"]
    assert tokenize("Привiт кiт") == tokenize("Привіт кіт")  # Latin i typed in Cyrillic words
    assert tokenize("не люблю") == ["не", "любл"]  # negations are kept
    assert tokenize("games game") == ["game", "game"]
    assert tokenize("") == []
    assert tokenize("!!! ...") == []


def test_bm25_matches_the_textbook_formula():
    docs = [["кіт", "мурч"], ["кав", "молок", "кав"], ["кіт"]]
    scores = _bm25(["кіт"], docs)
    n, df, avgdl, k1, b = 3, 2, (2 + 3 + 1) / 3, 1.2, 0.75
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))

    def expected(tf, dl):
        return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

    assert scores[0] == pytest.approx(expected(1, 2))
    assert scores[1] == 0.0
    assert scores[2] == pytest.approx(expected(1, 1))
    assert scores[2] > scores[0]  # shorter document, same term
    assert _bm25(["кіт"], []) == []


# -- search -------------------------------------------------------------------------------------------


def add_corpus(store, clock):
    texts = [
        ("fact", "кіт Мурчик"),
        ("preference", "Любить каву з корицею"),
        ("fact", "Грає в Dota 2 щовечора"),
        ("relationship", "Сестру звати Оля"),
        ("game_event", "Пройшов S.T.A.L.K.E.R. 2 за 80 годин"),
        ("joke", "Жарт про м'ясо по-французьки"),
    ]
    out = {}
    for kind, text in texts:
        memory = store.add(kind, text)
        assert memory is not None
        out[text] = memory
        clock.advance(1)
    return out


def test_search_finds_inflected_ukrainian(store, clock):
    corpus = add_corpus(store, clock)
    assert [m.id for m in store.search("мій кіт")] == [corpus["кіт Мурчик"].id]
    assert [m.id for m in store.search("Як там Мурчика справи?")] == [corpus["кіт Мурчик"].id]
    assert [m.text for m in store.search("хочу кави")] == ["Любить каву з корицею"]
    assert [m.text for m in store.search("мʼясо")] == ["Жарт про м'ясо по-французьки"]
    assert [m.text for m in store.search("stalker")] == ["Пройшов S.T.A.L.K.E.R. 2 за 80 годин"]


def test_cross_script_matching_is_out_of_scope(store, clock):
    add_corpus(store, clock)
    # documented limit: Cyrillic "сталкер" does not find Latin "S.T.A.L.K.E.R."
    assert store.search("сталкер") == []


def test_search_without_meaningful_terms_returns_nothing(store, clock):
    add_corpus(store, clock)
    assert store.search("") == []
    assert store.search("мій і твій") == []
    assert store.search("абракадабра") == []
    assert store.search("кіт", k=0) == []
    assert store.search(None) == []  # type: ignore[arg-type]


def test_search_respects_k_and_kinds(store, clock):
    for i in range(10):
        store.add("fact", f"Кіт номер {i} спить {'дуже ' * i}довго" + " мяу" * i)
    store.add("joke", "Жарт про кота: кіт у коробці")
    assert len(store.search("кіт", k=3)) == 3
    assert [m.kind for m in store.search("кіт", kinds=["joke"])] == ["joke"]
    assert store.search("кіт", kinds=[]) == []
    assert {m.kind for m in store.search("кіт", k=20)} == {"fact", "joke"}


def test_search_marks_results_as_used_and_does_not_announce(store, db, clock, rec):
    corpus = add_corpus(store, clock)
    rec.drain()
    clock.advance(100)
    found = store.search("Мурчик")
    assert found[0].last_used_at == clock.now()
    assert store.get(corpus["кіт Мурчик"].id).last_used_at == clock.now()
    untouched = store.get(corpus["Сестру звати Оля"].id)
    assert untouched.last_used_at is None
    assert rec.memory_kinds() == []


def test_pinned_memories_rank_higher(store):
    a = store.add("fact", "Любить каву з корицею")
    b = store.add("preference", "Любить каву з корицею")
    assert a and b
    store.set_pinned(a.id, True)
    assert [m.id for m in store.search("кава")] == [a.id, b.id]
    store.set_pinned(a.id, False)
    store.set_pinned(b.id, True)
    assert [m.id for m in store.search("кава")] == [b.id, a.id]


def test_recent_memories_rank_higher(store, clock):
    old = store.add("fact", "Любить каву з корицею")
    clock.advance(90 * DAY)
    new = store.add("preference", "Любить каву з корицею")
    assert old and new
    assert [m.id for m in store.search("кава")] == [new.id, old.id]


def test_important_memories_rank_higher(store):
    low = store.add("fact", "Любить каву з корицею", importance=0.1)
    high = store.add("preference", "Любить каву з корицею", importance=0.9)
    assert low and high
    assert [m.id for m in store.search("кава")] == [high.id, low.id]


def test_more_relevant_beats_pinned_noise(store):
    noise = store.add("fact", "Кіт спить на дивані, а надворі дощ і холодно цілий день")
    exact = store.add("fact", "Мурчик кіт")
    assert noise and exact
    store.set_pinned(noise.id, True)
    assert store.search("Мурчик")[0].id == exact.id


def test_search_survives_a_database_error(clock):
    database = Database(":memory:")
    store = MemoryStore(database, None, clock)
    store.add("fact", "кіт Мурчик")
    database.close()
    assert store.search("кіт") == []


# -- privacy in logs ---------------------------------------------------------------------------------


def test_memory_text_never_reaches_the_log(store, caplog):
    caplog.set_level(logging.DEBUG)
    secret_word = "Ксенофонтівна"
    memory = store.add("fact", f"Бабусю звати {secret_word}")
    assert memory is not None
    store.add("fact", f"Бабусю звати {secret_word} Петрівна")
    store.search(secret_word)
    store.update(memory.id, f"Бабусю звали {secret_word}")
    store.add("fact", "Картка 4111 1111 1111 1111")
    with pytest.raises(MemoryRejectedError):
        store.update(memory.id, "Пароль: hunter2")
    store.delete(memory.id)
    assert secret_word not in caplog.text
    assert "4111" not in caplog.text
    assert "hunter2" not in caplog.text
