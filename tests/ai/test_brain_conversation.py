"""AIBrain with the AI service's conversation rule: solicited lines go ahead, memory
extraction waits (one request at a time, a small backlog, discardable), and the chat
history keeps a stable prefix for local servers."""

from __future__ import annotations

import logging
import random
import threading
from collections.abc import Callable
from itertools import pairwise

import pytest

from chibi.ai.brain import MEMORY_BACKLOG, MEMORY_BATCH, history_slack
from chibi.ai.providers.fake import FakeProvider
from chibi.ai.types import ChatTurn, GenerationRequest, Purpose, Role
from chibi.core.events import ConversationActive, Permission, SpeechRequested
from chibi.memory.repositories import PhraseHistoryRepository
from chibi.personality.composer import SpeechComposer
from chibi.personality.phrases import PhraseSelector
from tests.ai.test_brain import BANK, Harness, memories_json, pump

CHAT_REPLY = "<<state=happy;action=none>>\nЗрозуміла!"
NO_MEMORIES = memories_json()


@pytest.fixture
def make(qapp, db, clock, bus):
    harnesses: list[Harness] = []

    def build(replies=(), *, local: bool = True, **kwargs) -> Harness:
        h = Harness(qapp, db, clock, bus, FakeProvider(replies, default_reply=None, local=local), **kwargs)
        harnesses.append(h)
        return h

    yield build
    for h in harnesses:
        h.service.shutdown()


def talk(bus, active: bool) -> None:
    bus.publish(ConversationActive(active=active))
    bus.dispatch_pending()


def purposes(h: Harness) -> list[Purpose]:
    return [request.purpose for request in h.provider.requests]


# -- lines the user asked for ---------------------------------------------------------------------


def test_only_solicited_lines_are_submitted_as_user_waiting(make):
    h = make(["<<state=happy;action=none>>\nО, гра!"] * 2)
    marks: list[object] = []
    inner = h.service.submit

    def submit(request, **kwargs):
        marks.append(kwargs.get("user_waiting", "absent"))
        return inner(request, **kwargs)

    h.service.submit = submit  # type: ignore[method-assign]
    assert h.phrase(("game_start.cs2",), {"game": "CS2"}, solicited=True) is not None
    assert h.phrase(("game_start.cs2",), {"game": "Dota 2"}, solicited=False) is not None
    assert marks == [True, "absent"]  # unsolicited: no new keyword for older stand-ins


def test_a_solicited_line_is_answered_during_a_conversation_an_unsolicited_one_waits(make, bus):
    h = make(["<<state=happy;action=none>>\nО, CS2!", "<<state=happy;action=none>>\nО, Dota!"])
    talk(bus, True)
    assert h.phrase(("game_start.cs2",), {"game": "CS2"}, solicited=True).text == "О, CS2!"
    later: list[SpeechRequested | None] = []
    h.brain.phrase(("game_start.cs2",), {"game": "Dota 2"}, False, later.append)
    pump(h.qapp, lambda: False, 0.1)
    assert later == []
    talk(bus, False)
    assert pump(h.qapp, lambda: bool(later))
    assert later[0] is not None and later[0].text == "О, Dota!"


# -- memory extraction ----------------------------------------------------------------------------


def test_memory_extraction_waits_for_the_conversation_one_request_at_a_time(make, bus):
    said = ["Мого кота звати Мурчик", "Я люблю каву", "Я граю на гітарі"]
    h = make([CHAT_REPLY] * 3 + [
        memories_json(("fact", "Кота користувача звати Мурчик")),
        memories_json(("preference", "Користувач любить каву"), ("fact", "Користувач грає на гітарі")),
    ])
    talk(bus, True)
    for text in said:
        _deltas, reply = h.chat(text)
        assert reply is not None and reply.text == "Зрозуміла!"
    pump(h.qapp, lambda: False, 0.1)
    assert purposes(h) == [Purpose.CHAT] * 3  # no extraction between her answers
    assert [r.purpose for r in h.submitted].count(Purpose.MEMORY) == 1  # the rest wait in the brain
    assert h.service.pending() == 1
    talk(bus, False)
    assert pump(h.qapp, lambda: h.memory.count() == 3)
    assert purposes(h) == [Purpose.CHAT] * 3 + [Purpose.MEMORY] * 2  # the waiting two in one batch
    extracted = [r.turns[0].text for r in h.provider.requests if r.purpose is Purpose.MEMORY]
    assert said[0] in extracted[0] and said[1] in extracted[1] and said[2] in extracted[1]
    assert {m.text for m in h.memory.list()} == {
        "Кота користувача звати Мурчик", "Користувач любить каву", "Користувач грає на гітарі"}


def test_no_message_of_a_long_conversation_is_left_out_of_memory_extraction(make, bus, caplog):
    """A long hands-free talk: every statement is mined once the conversation ends - in a
    few batched requests (the hourly limits see a handful, not one per message)."""
    count = 14
    h = make([CHAT_REPLY] * count + [NO_MEMORIES] * count)
    talk(bus, True)
    for i in range(count):
        h.chat(f"Я люблю число {i:02d}")
    talk(bus, False)

    def settled() -> bool:
        return h.service.pending() == 0 and h.provider.calls >= len(h.submitted)

    assert pump(h.qapp, settled)
    pump(h.qapp, lambda: False, 0.2)
    assert pump(h.qapp, settled)
    extracted = [r.turns[0].text for r in h.provider.requests if r.purpose is Purpose.MEMORY]
    assert all(sum(f"число {i:02d}" in text for text in extracted) == 1 for i in range(count))
    assert len(extracted) <= 1 + -(-(count - 1) // MEMORY_BATCH)
    assert "dropped" not in caplog.text


def test_only_a_huge_backlog_drops_its_oldest_message_and_says_so(make, bus, caplog):
    h = make([NO_MEMORIES] * 40)
    talk(bus, True)
    with caplog.at_level(logging.INFO, logger="chibi.ai.brain"):
        for i in range(MEMORY_BACKLOG + 3):  # 1 with the AI service, the rest waiting
            h.brain._extract_memories(f"Я люблю число {i:02d}")
    assert caplog.text.count("backlog full") == 2
    talk(bus, False)
    assert pump(h.qapp, lambda: h.service.pending() == 0 and h.brain._memory_request is None, 5.0)
    extracted = " ".join(r.turns[0].text for r in h.provider.requests if r.purpose is Purpose.MEMORY)
    assert "число 00" in extracted and "число 01" not in extracted and "число 02" not in extracted
    assert f"число {MEMORY_BACKLOG + 2:02d}" in extracted


def test_batched_statements_each_get_their_own_tags_and_room_for_memories(make, bus):
    h = make([CHAT_REPLY] * 3 + [NO_MEMORIES, memories_json(
        ("preference", "Користувач любить каву"), ("preference", "Користувач любить чай"),
        ("fact", "Користувач грає на гітарі"))])
    talk(bus, True)
    for text in ("Я люблю число 7", "Я люблю каву. А ще я люблю чай", "Я граю на гітарі"):
        h.chat(text)
    talk(bus, False)
    assert pump(h.qapp, lambda: h.memory.count() == 3)
    batch = [r for r in h.provider.requests if r.purpose is Purpose.MEMORY][-1].turns[0].text
    assert batch.count("<user>") == 2 and "кав" in batch and "гітар" in batch


def test_discarded_memories_are_never_extracted(make, bus):
    h = make([CHAT_REPLY] * 2 + [memories_json(("preference", "Користувач любить каву"))] * 2)
    talk(bus, True)
    h.chat("Я люблю каву")
    h.chat("Я люблю чай")
    h.brain.discard_pending_memories()
    talk(bus, False)
    pump(h.qapp, lambda: h.service.pending() == 0)
    pump(h.qapp, lambda: False, 0.1)
    assert Purpose.MEMORY not in purposes(h)
    assert h.memory.count() == 0


def test_memory_extraction_off_while_waiting_drops_the_backlog(make, bus):
    h = make([CHAT_REPLY] * 2 + [memories_json(("preference", "Користувач любить каву"))] * 2)
    talk(bus, True)
    h.chat("Я люблю каву")
    h.chat("Я люблю чай")
    h.settings.ai.remember_facts = False
    talk(bus, False)
    pump(h.qapp, lambda: h.service.pending() == 0)
    pump(h.qapp, lambda: False, 0.1)
    assert purposes(h).count(Purpose.MEMORY) <= 1  # the one already queued may still run
    assert h.memory.count() == 0  # ...but nothing is stored


# -- a stable history prefix -----------------------------------------------------------------------


def test_history_window_keeps_its_first_turn_for_a_while(make):
    h = make([CHAT_REPLY] * 30)
    h.settings.ai.remember_facts = False
    h.settings.ai.context_turns = 12
    limit = h.settings.ai.context_turns - 1
    histories: list[tuple[ChatTurn, ...]] = []
    for i in range(20):
        h.chat(f"Повідомлення номер {i}")
        histories.append(h.submitted[-1].turns[:-1])
    plain = [min(2 * i, limit - limit % 2) for i in range(20)]  # what a sliding window sends
    assert all(len(history) >= want for history, want in zip(histories, plain, strict=True))
    assert all(len(history) <= limit + history_slack(limit) for history in histories)
    assert all(history[0].role is Role.USER for history in histories if history)
    kept = 0
    for before, after in pairwise(histories[6:]):
        if before and after and before[0].text == after[0].text:
            kept += 1
            assert after[:len(before)] == before  # an unchanged prefix: nothing to read again
    assert kept >= 8  # of 13 steps; a plain sliding window keeps none once full


def test_a_paid_cloud_api_gets_the_plain_window_never_more_turns_than_set(make):
    """Only a server of the user's own reuses the prompt it already read; a cloud API would
    bill the extra turns on every request (Claude caches only the persona) - and the AI
    page promises "Реплік розмови в запиті" turns, not more."""
    h = make([CHAT_REPLY] * 30, local=False)
    h.settings.ai.ollama_url = "http://ai.example.com:11434"  # someone else's server
    h.permissions.grant(Permission.CLOUD_AI)
    h.settings.ai.remember_facts = False
    h.settings.ai.context_turns = 12
    limit = h.settings.ai.context_turns - 1
    for i in range(20):
        h.chat(f"Повідомлення номер {i}")
        history = h.submitted[-1].turns[:-1]
        assert len(history) <= limit
        assert len(history) >= min(2 * i, limit - limit % 2)


def test_history_window_restarts_after_the_conversation_is_cleared(make):
    h = make([CHAT_REPLY] * 4)
    h.settings.ai.remember_facts = False
    h.chat("Перше")
    h.chat("Друге")
    h.short.clear()
    h.chat("Третє")
    assert h.submitted[-1].turns[:-1] == ()
    h.chat("Четверте")
    first = h.submitted[-1].turns[0]
    assert first.role is Role.USER and "Третє" in first.text


def test_history_window_never_shrinks_below_the_plain_window_when_the_setting_grows(make):
    h = make([CHAT_REPLY] * 12)
    h.settings.ai.remember_facts = False
    h.settings.ai.context_turns = 4
    for i in range(8):
        h.chat(f"Повідомлення {i}")
    h.settings.ai.context_turns = 12
    h.chat("Ще одне")
    assert len(h.submitted[-1].turns) - 1 == 10  # the plain window of 11 (a user turn first)


# -- her own lines given up by the composer ------------------------------------------------------


class _Schedule:
    def __init__(self) -> None:
        self.timers: list[tuple[int, Callable[[], None]]] = []

    def __call__(self, msec: int, callback: Callable[[], None]) -> None:
        self.timers.append((msec, callback))

    def fire_all(self) -> None:
        timers, self.timers = self.timers, []
        for _msec, callback in timers:
            callback()


def composer_for(h: Harness, bus, db, clock) -> tuple[SpeechComposer, _Schedule]:
    selector = PhraseSelector(BANK, PhraseHistoryRepository(db), lambda: h.settings, clock, random.Random(1))
    schedule = _Schedule()
    return SpeechComposer(bus, selector, h.brain, lambda: h.settings, clock, schedule=schedule), schedule


def test_a_line_the_composer_gave_up_is_withdrawn_from_the_ai_service(make, bus, db, clock):
    """Her own line waiting for the model when the user starts talking is abandoned - and
    so is its request: it must not run (and occupy the model) after the conversation."""
    h = make(["<<state=happy;action=none>>\nЖарт від моделі."] * 4)
    composer, _schedule = composer_for(h, bus, db, clock)
    try:
        talk(bus, True)
        assert h.phrase(("game_start.cs2",), {"game": "CS2"}, solicited=True) is not None  # warm
        talk(bus, False)
        pump(h.qapp, lambda: False, 0.05)
        calls = h.provider.calls
        busy = threading.Event()
        h.provider.gate = busy
        blocker: list[object] = []
        h.service.submit(GenerationRequest(system="s", turns=(ChatTurn(Role.USER, "x"),),
                                           purpose=Purpose.CHAT),
                         on_done=blocker.append, on_error=blocker.append)
        assert pump(h.qapp, lambda: h.provider.calls == calls + 1)
        assert composer.compose("idle.joke")  # queued behind the busy worker
        talk(bus, True)
        clock.advance(15.0)
        busy.set()
        assert pump(h.qapp, lambda: bool(blocker))
        talk(bus, False)
        pump(h.qapp, lambda: h.service.pending() == 0)
        pump(h.qapp, lambda: False, 0.1)
        assert h.provider.calls == calls + 1  # the joke's request never reached the model
        assert [r.purpose for r in h.provider.requests].count(Purpose.INITIATIVE) == 0
    finally:
        composer.close()


def test_a_line_that_timed_out_is_withdrawn_too(make, bus, db, clock):
    h = make(["<<state=happy;action=none>>\nЖарт від моделі."] * 4)
    composer, schedule = composer_for(h, bus, db, clock)
    try:
        busy = threading.Event()
        h.provider.gate = busy
        blocker: list[object] = []
        h.service.submit(GenerationRequest(system="s", turns=(ChatTurn(Role.USER, "x"),),
                                           purpose=Purpose.CHAT),
                         on_done=blocker.append, on_error=blocker.append)
        assert pump(h.qapp, lambda: h.provider.calls == 1)
        spoken: list[SpeechRequested] = []
        bus.subscribe(SpeechRequested, spoken.append)
        assert composer.compose("idle.joke")
        schedule.fire_all()  # the composer's timeout: the phrase bank answers
        bus.dispatch_pending()
        assert [s.source for s in spoken] == ["idle.joke"]
        busy.set()
        assert pump(h.qapp, lambda: bool(blocker))
        pump(h.qapp, lambda: h.service.pending() == 0)
        pump(h.qapp, lambda: False, 0.1)
        assert h.provider.calls == 1  # the late request was not sent after all
    finally:
        composer.close()
