"""AIService while the user talks with her (ConversationActive): background requests
wait, a running one is interrupted and resumed or given up, held ones expire; per-request
timing; keeping an Ollama model loaded (keep_alive, warm-up)."""

from __future__ import annotations

import logging
import threading

import pytest

from chibi.ai.providers.fake import FakeProvider
from chibi.ai.providers.ollama_provider import OllamaProvider
from chibi.ai.service import (
    CUT_SHORT_CHARS_PER_TOKEN,
    HOLD_LIMIT_SECONDS,
    MAX_QUEUE,
    REACTION_DEADLINE_SECONDS,
    ConversationBusyError,
    QueueFullError,
    RequestTiming,
)
from chibi.ai.types import (
    BudgetExceededError,
    CancelledError,
    CancelToken,
    ChatTurn,
    GenerationRequest,
    ProviderUnavailableError,
    Purpose,
    Role,
)
from chibi.core.events import ConversationActive, Permission
from tests.ai.test_service import Harness, Outcome, pump, wait_started

SECRET = "мій пароль від банку 12345"


@pytest.fixture
def make(qapp, db, clock, bus, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    harnesses: list[Harness] = []

    def factory(provider: FakeProvider | None = None, **kwargs) -> Harness:
        h = Harness(qapp, db, clock, bus, provider if provider is not None else FakeProvider(), **kwargs)
        harnesses.append(h)
        return h

    yield factory
    for h in harnesses:
        h.service.shutdown()


def talk(h, active: bool) -> None:
    h.bus.publish(ConversationActive(active=active))
    h.dispatch()


def idle(h, seconds: float = 0.05) -> None:
    pump(h.qapp, lambda: False, seconds)


def submit_waiting(h, purpose: Purpose, text: str = "Привіт") -> Outcome:
    """Submit a request the user asked for (``user_waiting=True``)."""
    outcome = Outcome()
    h.service.submit(h.request(purpose, text), on_done=outcome.on_done, on_error=outcome.on_error,
                     user_waiting=True)
    return outcome


# -- background requests wait ---------------------------------------------------------------------


def test_background_requests_wait_while_the_user_talks_chat_runs(make):
    gate = threading.Event()
    h = make(FakeProvider(default_reply="ok", gate=gate))
    talk(h, True)
    assert h.service.conversation_active
    chat, _ = h.submit(Purpose.CHAT, "chat")
    wait_started(h.provider)
    held = {p: h.submit(p, p.value)[0] for p in (Purpose.MEMORY, Purpose.VISION, Purpose.INITIATIVE,
                                                  Purpose.REACTION)}
    gate.set()
    h.wait(chat)
    assert chat.result.text == "ok"
    second = h.run(Purpose.CHAT, "chat 2")  # a chat request still goes ahead of them all
    assert second.result.text == "ok"
    idle(h)
    assert isinstance(held.pop(Purpose.MEMORY).error, QueueFullError)  # pushed out of the full queue
    assert not any(out.finished for out in held.values())
    assert [r.purpose for r in h.provider.requests] == [Purpose.CHAT, Purpose.CHAT]
    assert h.service.pending() == 3
    talk(h, False)
    h.wait(*held.values())
    assert [r.purpose for r in h.provider.requests] == [
        Purpose.CHAT, Purpose.CHAT, Purpose.REACTION, Purpose.INITIATIVE, Purpose.VISION]
    assert all(out.result.text == "ok" for out in held.values())


def test_user_waiting_tests_and_model_listing_run_during_a_conversation(make):
    h = make(FakeProvider(default_reply="ok", models=["a", "b"]))
    talk(h, True)
    asked = submit_waiting(h, Purpose.REACTION, "solicited")
    h.wait(asked)
    assert asked.result.text == "ok"
    tested: list[tuple[bool, str]] = []
    listed: list[tuple[list[str], str]] = []
    h.service.test(lambda ok, message: tested.append((ok, message)))
    h.service.list_models(lambda models, error: listed.append((models, error)))
    assert pump(h.qapp, lambda: bool(tested and listed))
    assert tested[0][0] is True and listed == [(["a", "b"], "")]


def test_conversation_events_are_idempotent(make):
    h = make(FakeProvider(default_reply="ok"))
    talk(h, False)  # not active yet: nothing happens
    talk(h, True)
    talk(h, True)
    out, _ = h.submit(Purpose.MEMORY)
    idle(h)
    assert not out.finished
    talk(h, False)
    h.wait(out)
    assert not h.service.conversation_active


# -- a running background request is interrupted ------------------------------------------------


def test_running_memory_request_is_interrupted_resumed_and_recorded_once(make):
    gate = threading.Event()
    h = make(FakeProvider(["lost", "chat", "memory"], gate=gate))
    memory, _ = h.submit(Purpose.MEMORY, "mem")
    wait_started(h.provider)
    talk(h, True)
    assert pump(h.qapp, lambda: h.service.pending() == 1)  # back in the queue
    assert not memory.finished
    gate.set()
    chat = h.run(Purpose.CHAT, "chat")
    assert chat.result.text == "chat"
    idle(h)
    assert not memory.finished and h.provider.calls == 2
    talk(h, False)
    h.wait(memory)
    assert memory.result.text == "memory"
    assert [r.turns[0].text for r in h.provider.requests] == ["mem", "chat", "mem"]
    assert h.service.usage_today().requests == 2  # the interrupted attempt is not counted


@pytest.mark.parametrize("purpose", [Purpose.VISION, Purpose.INITIATIVE])
def test_interrupted_vision_and_initiative_fail_with_conversation_busy(make, purpose):
    gate = threading.Event()
    h = make(FakeProvider(gate=gate))
    out, _ = h.submit(purpose)
    wait_started(h.provider)
    talk(h, True)
    h.wait(out)
    assert isinstance(out.error, ConversationBusyError)
    assert isinstance(out.error, CancelledError)  # callers already treat it as "skip, try later"
    assert h.service.pending() == 0 and h.provider.calls == 1
    assert h.service.usage_today().requests == 0
    talk(h, False)
    idle(h)
    assert h.provider.calls == 1  # not retried


def test_interrupted_reaction_resumes_while_still_fresh(make, clock):
    gate = threading.Event()
    h = make(FakeProvider(["lost", "line"], gate=gate))
    out, _ = h.submit(Purpose.REACTION)
    wait_started(h.provider)
    talk(h, True)
    assert pump(h.qapp, lambda: h.service.pending() == 1)
    gate.set()
    clock.advance(REACTION_DEADLINE_SECONDS - 1)
    talk(h, False)
    h.wait(out)
    assert out.result.text == "line"


def test_interrupted_reaction_that_went_stale_fails_cancelled(make, clock):
    gate = threading.Event()
    h = make(FakeProvider(["lost", "line"], gate=gate))
    out, _ = h.submit(Purpose.REACTION)
    wait_started(h.provider)
    talk(h, True)
    assert pump(h.qapp, lambda: h.service.pending() == 1)
    gate.set()
    clock.advance(REACTION_DEADLINE_SECONDS + 1)
    talk(h, False)
    h.wait(out)
    assert type(out.error) is CancelledError  # the composer falls back to the phrase bank
    assert h.provider.calls == 1


def test_a_reaction_that_waited_past_its_deadline_is_not_sent(make, clock):
    gate = threading.Event()
    h = make(FakeProvider(default_reply="ok", gate=gate))
    busy, _ = h.submit(Purpose.CHAT, "busy")
    wait_started(h.provider)
    line, _ = h.submit(Purpose.REACTION, "late")
    clock.advance(REACTION_DEADLINE_SECONDS + 0.5)
    gate.set()
    h.wait(busy, line)
    assert isinstance(line.error, CancelledError)
    assert [r.turns[0].text for r in h.provider.requests] == ["busy"]


def test_a_caller_cancel_wins_over_resuming(make):
    gate = threading.Event()
    h = make(FakeProvider(gate=gate))
    token = CancelToken()
    out, _ = h.submit(Purpose.MEMORY, cancel=token)
    wait_started(h.provider)
    token.cancel()
    talk(h, True)
    h.wait(out)
    assert type(out.error) is CancelledError
    talk(h, False)
    idle(h)
    assert h.provider.calls == 1 and h.service.pending() == 0


class _StubbornProvider(FakeProvider):
    """Finishes its answer even when cancelled (a provider that checks too late)."""

    def generate(self, request: GenerationRequest, cancel: CancelToken | None = None):
        text = self._next(request, None)
        assert cancel is not None and cancel.cancelled  # the conversation preempted it
        return self._result(request, text)


def test_an_answer_that_arrives_despite_the_interruption_is_delivered_once(make):
    gate = threading.Event()
    h = make(_StubbornProvider(["done anyway"], gate=gate))
    out, _ = h.submit(Purpose.MEMORY)
    wait_started(h.provider)
    talk(h, True)
    idle(h)
    gate.set()
    h.wait(out)
    assert out.result.text == "done anyway"
    talk(h, False)
    idle(h)
    assert h.provider.calls == 1 and h.service.usage_today().requests == 1


@pytest.mark.parametrize("purpose", [Purpose.MEMORY, Purpose.VISION])
def test_a_request_to_a_paid_cloud_api_is_let_finish_not_interrupted(make, purpose):
    """A cloud API runs requests side by side and bills each one: interrupting it would
    pay for it twice (a resumed MEMORY) or for nothing (a VISION given up), and its
    keep-alive connection often cannot be aborted anyway."""
    gate = threading.Event()
    h = make(FakeProvider(["answer"], gate=gate, local=False), ai_provider="anthropic", cloud=True,
             key="sk-ant")
    out, _ = h.submit(purpose)
    wait_started(h.provider)
    talk(h, True)
    idle(h)
    assert not out.finished
    gate.set()
    h.wait(out)
    assert out.result.text == "answer"
    talk(h, False)
    idle(h)
    assert h.provider.calls == 1
    assert h.service.usage_today().requests == 1


class _CutShortCloud(FakeProvider):
    """A paid API whose request was already sent when the caller withdrew it."""

    def generate(self, request: GenerationRequest, cancel: CancelToken | None = None):
        self._next(request, None)
        raise CancelledError("cancelled")  # stopped mid-answer, after the API received it


def test_a_cloud_request_cut_short_after_sending_still_counts(make):
    local = make(_CutShortCloud())  # the user's own server: free, and stopped at once
    out = local.run(Purpose.VISION)
    assert type(out.error) is CancelledError and local.service.usage_today().requests == 0
    h = make(_CutShortCloud(local=False), ai_provider="anthropic", cloud=True, key="sk-ant")
    request = GenerationRequest(system="с" * 300, turns=(ChatTurn(Role.USER, "к" * 600),),
                                purpose=Purpose.VISION, context="о" * 300)
    out = Outcome()
    h.service.submit(request, on_done=out.on_done, on_error=out.on_error)
    h.wait(out)
    assert type(out.error) is CancelledError
    usage = h.service.usage_today()
    assert usage.requests == 1  # the hourly limits (and vision's share) see it
    assert usage.input_tokens == 1200 // CUT_SHORT_CHARS_PER_TOKEN


def test_user_waiting_requests_are_not_interrupted(make):
    gate = threading.Event()
    h = make(FakeProvider(["seen"], gate=gate))
    out = submit_waiting(h, Purpose.VISION)
    wait_started(h.provider)
    talk(h, True)
    idle(h)
    gate.set()
    h.wait(out)
    assert out.result.text == "seen"


# -- held requests do not starve --------------------------------------------------------------


def test_held_vision_and_initiative_give_up_after_the_hold_limit_memory_stays(make, clock):
    h = make(FakeProvider(default_reply="ok"))
    talk(h, True)
    vision, _ = h.submit(Purpose.VISION)
    initiative, _ = h.submit(Purpose.INITIATIVE)
    memory, _ = h.submit(Purpose.MEMORY)
    clock.advance(HOLD_LIMIT_SECONDS - 1)
    h.service._expire_held()  # the hold timer's slot
    idle(h)
    assert not (vision.finished or initiative.finished or memory.finished)
    clock.advance(2)
    h.service._expire_held()
    h.wait(vision, initiative)
    assert isinstance(vision.error, ConversationBusyError)
    assert isinstance(initiative.error, ConversationBusyError)
    assert not memory.finished and h.service.pending() == 1
    talk(h, False)
    h.wait(memory)
    assert memory.result.text == "ok"
    assert [r.purpose for r in h.provider.requests] == [Purpose.MEMORY]


def test_held_reaction_past_its_deadline_and_cancelled_requests_are_released(make, clock):
    h = make(FakeProvider(default_reply="ok"))
    talk(h, True)
    line, _ = h.submit(Purpose.REACTION)
    token = CancelToken()
    vision, _ = h.submit(Purpose.VISION, cancel=token)
    token.cancel()
    clock.advance(REACTION_DEADLINE_SECONDS + 1)
    h.service._expire_held()
    h.wait(line, vision)
    assert type(line.error) is CancelledError and type(vision.error) is CancelledError
    assert h.service.pending() == 0


def test_the_hold_timer_runs_only_during_a_conversation(make):
    h = make(FakeProvider())
    assert not h.service._hold_timer.isActive()
    talk(h, True)
    assert h.service._hold_timer.isActive()
    talk(h, False)
    assert not h.service._hold_timer.isActive()


def test_queue_cap_still_holds_with_held_requests(make):
    h = make(FakeProvider(default_reply="ok"))
    talk(h, True)
    held = [h.submit(p, p.value)[0] for p in (Purpose.REACTION, Purpose.MEMORY, Purpose.VISION,
                                                Purpose.INITIATIVE)]
    assert h.service.pending() == MAX_QUEUE
    chat = h.run(Purpose.CHAT, "chat")
    assert chat.result.text == "ok"
    dropped = held[1]  # the lowest priority (MEMORY) goes first
    h.wait(dropped)
    assert isinstance(dropped.error, QueueFullError)
    assert not any(out.finished for out in held if out is not dropped)


def test_held_requests_are_checked_against_the_budget_again(make):
    h = make(FakeProvider(default_reply="ok"))
    h.settings.ai.max_requests_per_hour = 2
    talk(h, True)
    vision, _ = h.submit(Purpose.VISION)
    h.run(Purpose.CHAT, "1")
    h.run(Purpose.CHAT, "2")  # the conversation used up the hour (CHAT has an overdraft)
    talk(h, False)
    h.wait(vision)
    assert isinstance(vision.error, BudgetExceededError)
    assert [r.purpose for r in h.provider.requests] == [Purpose.CHAT, Purpose.CHAT]


def test_shutdown_during_a_conversation_is_prompt_and_silent(make):
    gate = threading.Event()
    h = make(FakeProvider(gate=gate))
    running, _ = h.submit(Purpose.CHAT)
    wait_started(h.provider)
    talk(h, True)
    held, _ = h.submit(Purpose.MEMORY)
    thread = h.service._thread
    h.service.shutdown()
    assert thread is not None and not thread.is_alive()
    assert not h.service._hold_timer.isActive()
    talk(h, False)  # unsubscribed: no effect, no error
    idle(h)
    assert not running.finished and not held.finished


def test_nothing_from_requests_is_logged_around_conversations(make, caplog):
    gate = threading.Event()
    h = make(FakeProvider([f"про {SECRET}"] * 4, gate=gate))
    with caplog.at_level(logging.DEBUG):
        out, _ = h.submit(Purpose.MEMORY, SECRET)
        wait_started(h.provider)
        talk(h, True)
        gate.set()
        h.run(Purpose.CHAT, SECRET, stream=True)
        talk(h, False)
        h.wait(out)
    assert SECRET not in caplog.text
    assert "put aside" in caplog.text


# -- timing -------------------------------------------------------------------------------------


class _ClockedProvider(FakeProvider):
    """Streams in two pieces, moving the fake clock: 2 s to the first text, 3 s more to the end."""

    def __init__(self, clock, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clock = clock
        self.last_load_seconds: float | None = 1.25
        self.last_prompt_seconds: float | None = 0.5

    def stream(self, request, on_delta, cancel=None):
        text = self._next(request, cancel)
        self.clock.advance(2.0)
        on_delta(text[:3])
        self.clock.advance(3.0)
        on_delta(text[3:])
        return self._result(request, text)

    def generate(self, request, cancel=None):
        text = self._next(request, cancel)
        self.clock.advance(4.0)
        return self._result(request, text)


def test_chat_timing_is_measured_logged_and_kept(make, clock, caplog):
    h = make(_ClockedProvider(clock, [f"Привіт! {SECRET}"]))
    assert h.service.last_chat_timing is None
    with caplog.at_level(logging.INFO, logger="chibi.ai.service"):
        out = h.run(Purpose.CHAT, SECRET, stream=True)
    assert out.result.text.startswith("Привіт")
    timing = h.service.last_chat_timing
    assert isinstance(timing, RequestTiming)
    assert (timing.queued, timing.first_text, timing.total) == (0.0, 2.0, 5.0)
    assert (timing.model_load, timing.prompt_read) == (1.25, 0.5)
    assert timing.input_tokens > 0 and timing.output_tokens > 0
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("AI chat:")]
    assert len(lines) == 1
    assert lines[0].startswith("AI chat: queued 0.0 s, first text 2.0 s, done 5.0 s, model load 1.2 s")
    assert SECRET not in caplog.text


def test_queue_wait_counts_and_other_purposes_keep_the_chat_timing(make, clock):
    gate = threading.Event()
    h = make(_ClockedProvider(clock, default_reply="ok", gate=gate))
    busy, _ = h.submit(Purpose.VISION, "busy")
    wait_started(h.provider)
    chat, _ = h.submit(Purpose.CHAT, "chat")
    clock.advance(7.0)
    gate.set()
    h.wait(busy, chat)  # not streamed: the first text is the whole answer
    timing = h.service.last_chat_timing
    assert timing is not None and timing.first_text is None
    # 7 s of the analysis while it waited, 4 s for the analysis to finish, 4 s of its own.
    assert (timing.queued, timing.total) == (11.0, 15.0)
    h.run(Purpose.MEMORY)
    assert h.service.last_chat_timing is timing


def test_failed_and_test_requests_leave_the_chat_timing_alone(make, caplog):
    h = make(FakeProvider([ProviderUnavailableError("x")] * 3, default_reply="ok"))
    with caplog.at_level(logging.INFO, logger="chibi.ai.service"):
        out = h.run(Purpose.CHAT)
    assert isinstance(out.error, ProviderUnavailableError)
    assert h.service.last_chat_timing is None
    assert "AI chat failed (ProviderUnavailableError): queued 0.0 s" in caplog.text
    done: list[bool] = []
    h.service.test(lambda ok, _message: done.append(ok))
    assert pump(h.qapp, lambda: bool(done)) and done == [True]
    assert h.service.last_chat_timing is None


def test_request_timing_describe_has_numbers_only():
    timing = RequestTiming(queued=0.04, first_text=2.26, total=5.1, model_load=0.01, prompt_read=None,
                           input_tokens=812, output_tokens=40)
    assert timing.describe() == "queued 0.0 s, first text 2.3 s, done 5.1 s, 812 in / 40 out tokens"


# -- keeping the model loaded ---------------------------------------------------------------------


class _WarmProvider(FakeProvider):
    """A provider that can load its model ahead (like Ollama)."""

    def __init__(self, *args, warm_error: Exception | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.warmed: list[CancelToken | None] = []
        self.warm_error = warm_error
        self.warm_gate: threading.Event | None = None
        self.last_load_seconds: float | None = None
        self.warm_load = 3.5

    def warm_up(self, cancel: CancelToken | None = None) -> None:
        self.warmed.append(cancel)
        if self.warm_gate is not None:
            while not self.warm_gate.wait(0.01):
                if cancel is not None and cancel.cancelled:
                    raise CancelledError("cancelled")
        if self.warm_error is not None:
            raise self.warm_error
        self.last_load_seconds = self.warm_load


def warmed(h, count: int) -> bool:
    return pump(h.qapp, lambda: len(h.provider.warmed) >= count and not h.service._warming)


def test_warm_up_loads_once_and_not_again_while_the_model_is_likely_loaded(make, clock, caplog):
    h = make(_WarmProvider())
    h.provider.warm_gate = threading.Event()
    with caplog.at_level(logging.INFO, logger="chibi.ai.service"):
        assert h.service.warm_up() is True
        assert h.service.warm_up() is False  # one in flight at most
        h.provider.warm_gate.set()
        assert warmed(h, 1)
    assert "model load 3.5 s" in caplog.text
    assert h.service.warm_up() is False  # just loaded
    minutes = h.settings.ai.keep_model_loaded_minutes
    clock.advance((minutes - 1) * 60 - 1)
    assert h.service.warm_up() is False
    clock.advance(2)
    assert h.service.warm_up() is True
    assert warmed(h, 2)
    assert h.service.usage_today().requests == 0  # nothing generated, nothing billed


class _LoadedProvider(_WarmProvider):
    """Like Ollama: each request reports its own load time (~0 once the model is loaded)."""

    def generate(self, request, cancel=None):
        result = super().generate(request, cancel)
        self.last_load_seconds = 0.02
        return result


def test_a_chat_that_waited_for_the_model_to_load_reports_it_as_model_load(make, clock):
    """The warm-up started a moment before the chat ("name" mode: the conversation starts
    as her name is recognised): the chat's wait is the model loading, not a queue - and
    the speed card must say so, with its advice."""
    from chibi.settings.voice_page import MODEL_LOAD_ADVICE, model_timing_text

    h = make(_LoadedProvider(default_reply="ok"))
    h.provider.warm_gate = threading.Event()
    assert h.service.warm_up() is True
    assert pump(h.qapp, lambda: len(h.provider.warmed) == 1)
    chat, _ = h.submit(Purpose.CHAT)
    clock.advance(4.0)  # Ollama loads the model (it reports 3.5 s of loading)
    h.provider.warm_gate.set()
    h.wait(chat)
    timing = h.service.last_chat_timing
    assert timing is not None
    assert timing.model_load == pytest.approx(3.5, abs=0.05)  # and its own ~0.02 s
    assert timing.queued == pytest.approx(0.5, abs=0.05) and timing.total == pytest.approx(4.0)
    text = model_timing_text(timing)
    assert "завантаження моделі — 3,5\u00a0с" in text and MODEL_LOAD_ADVICE in text
    assert "черга" not in text


def test_a_warm_up_that_found_the_model_loaded_is_not_a_model_load(make, clock):
    h = make(_LoadedProvider(default_reply="ok"))
    h.provider.warm_gate = threading.Event()
    assert h.service.warm_up() is True
    assert pump(h.qapp, lambda: len(h.provider.warmed) == 1)
    chat, _ = h.submit(Purpose.CHAT)
    h.provider.warm_load = 0.0
    clock.advance(0.3)
    h.provider.warm_gate.set()
    h.wait(chat)
    timing = h.service.last_chat_timing
    assert timing is not None and (timing.model_load or 0.0) < 0.05
    assert timing.queued == pytest.approx(0.3)


def test_a_chat_request_counts_as_using_the_model(make, clock):
    h = make(_WarmProvider(default_reply="ok"))
    h.run(Purpose.CHAT)
    assert h.service.warm_up() is False
    h.run(Purpose.VISION, model="moondream")  # another model may have pushed it out
    assert h.service.warm_up() is True
    assert warmed(h, 1)


def test_warm_up_needs_a_provider_that_can_and_a_usable_ai(make):
    plain = make(FakeProvider())
    assert plain.service.warm_up() is False
    off = make(_WarmProvider())
    off.settings.ai.use_for_chat = False
    assert off.service.warm_up() is False
    limited = make(_WarmProvider(default_reply="ok"))
    limited.settings.ai.max_requests_per_day = 1
    limited.run(Purpose.MEMORY)
    assert limited.service.warm_up() is False  # the budget would refuse the chat anyway
    for h in (off, limited):
        assert h.provider.warmed == []


def test_warm_up_is_skipped_when_a_chat_request_is_already_waiting(make):
    gate = threading.Event()
    h = make(_WarmProvider(default_reply="ok", gate=gate))
    out, _ = h.submit(Purpose.CHAT)
    wait_started(h.provider)
    assert h.service.warm_up() is False
    gate.set()
    h.wait(out)


def test_warm_up_runs_before_background_work_and_during_a_conversation(make):
    gate = threading.Event()
    h = make(_WarmProvider(default_reply="ok", gate=gate))
    busy, _ = h.submit(Purpose.VISION, "busy")
    wait_started(h.provider)
    later, _ = h.submit(Purpose.MEMORY, "later")
    talk(h, True)  # the analysis is put aside, the warm-up goes ahead
    h.wait(busy)
    assert h.service.warm_up() is True
    assert warmed(h, 1)
    assert not later.finished
    talk(h, False)
    gate.set()
    h.wait(later)


def test_a_failed_warm_up_is_quiet_and_does_not_trip_the_breaker(make, caplog):
    h = make(_WarmProvider(warm_error=ProviderUnavailableError("down")))
    with caplog.at_level(logging.INFO, logger="chibi.ai.service"):
        for _ in range(4):
            assert h.service.warm_up() is True
            assert warmed(h, len(h.provider.warmed))
    assert len(h.provider.warmed) == 4
    assert h.service.status() == (True, "")
    assert "warm-up failed (ProviderUnavailableError)" in caplog.text


def test_a_warm_up_for_a_remote_server_is_dropped_when_cloud_ai_is_revoked(make):
    """Ollama on another machine needs the cloud permission: once it is revoked nothing
    more goes there - not even a warm-up (it would tell the server the user is active)."""
    gate = threading.Event()
    h = make(_WarmProvider(default_reply="ok", gate=gate, local=False), cloud=True)
    h.settings.ai.ollama_url = "http://192.168.1.20:11434"
    busy, _ = h.submit(Purpose.VISION, "busy")
    wait_started(h.provider)
    assert h.service.warm_up() is True  # queued behind the analysis
    h.permissions.revoke(Permission.CLOUD_AI)
    h.dispatch()
    gate.set()
    h.wait(busy)
    assert pump(h.qapp, lambda: not h.service._warming)
    idle(h)
    assert h.provider.warmed == []
    assert h.service.pending() == 0


def test_the_worker_rechecks_the_cloud_permission_before_a_warm_up(make):
    gate = threading.Event()
    h = make(_WarmProvider(default_reply="ok", gate=gate, local=False), cloud=True)
    h.settings.ai.ollama_url = "http://192.168.1.20:11434"
    busy, _ = h.submit(Purpose.VISION, "busy")
    wait_started(h.provider)
    assert h.service.warm_up() is True
    h.permissions.revoke(Permission.CLOUD_AI)  # the bus has not delivered it yet
    gate.set()
    h.wait(busy)
    assert pump(h.qapp, lambda: not h.service._warming)
    assert h.provider.warmed == []


def test_reconfigure_sets_ollama_keep_alive_from_the_settings(qapp, db, clock, bus):
    from chibi.ai.budget import UsageLedger
    from chibi.ai.secrets import MemorySecretStore
    from chibi.ai.service import AIService
    from chibi.core.config import Settings
    from chibi.memory.repositories import PermissionRepository
    from chibi.privacy.permissions import PermissionManager

    settings = Settings()
    settings.ai.provider = "ollama"
    service = AIService(bus, lambda: settings, PermissionManager(PermissionRepository(db), bus, clock),
                        MemorySecretStore(), UsageLedger(db, clock), clock)
    try:
        provider = service._provider
        assert isinstance(provider, OllamaProvider) and provider.keep_alive == "30m"
        settings.ai.keep_model_loaded_minutes = 90
        service.reconfigure()
        assert isinstance(service._provider, OllamaProvider) and service._provider.keep_alive == "90m"
    finally:
        service.shutdown()


def test_which_servers_are_the_users_own():
    """Interrupted for a conversation and given the stable history prefix: the user's own
    model servers; paid cloud APIs are let finish and get the plain history window."""
    from chibi.ai.providers.anthropic_provider import AnthropicProvider
    from chibi.ai.providers.openai_provider import OpenAIProvider
    from chibi.ai.service import _own_server

    providers = [
        (OllamaProvider("http://localhost:11434", "m"), True),
        (OllamaProvider("http://ai.example.com:11434", "m"), True),  # free, one request at a time
        (OpenAIProvider(None, "http://127.0.0.1:8080/v1", "local"), True),  # llama.cpp
        (OpenAIProvider(None, "http://192.168.1.20:1234/v1", "local"), True),  # LM Studio on the LAN
        (OpenAIProvider("sk-test", "https://api.openai.com/v1", "gpt-5-mini"), False),
        (OpenAIProvider("sk-test", "https://openrouter.ai/api/v1", "x"), False),
        (AnthropicProvider("sk-ant-test"), False),
        (FakeProvider(local=True), True),
        (FakeProvider(local=False), False),
        (None, False),
    ]
    try:
        assert [_own_server(provider) for provider, _own in providers] == [own for _p, own in providers]
    finally:
        for provider, _own in providers:
            if provider is not None:
                provider.close()
