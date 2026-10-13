"""AIService: worker thread, priority queue, main-thread callbacks, retries,
circuit breaker, permission and budget gates, cache, shutdown."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import pytest

from chibi.ai.budget import UsageLedger
from chibi.ai.providers.fake import FakeProvider
from chibi.ai.secrets import MemorySecretStore
from chibi.ai.service import (
    BACKOFF_SECONDS,
    BREAKER_COOLDOWN,
    CACHE_SIZE,
    CACHE_TTL,
    MAX_QUEUE,
    REASON_DOWN,
    REASON_LIMIT,
    REASON_NO_KEY,
    REASON_OFF,
    REASON_PERMISSION,
    TEST_PROMPT,
    AIOfflineError,
    AIService,
)
from chibi.ai.service import QueueFullError as ServiceQueueFull
from chibi.ai.types import (
    BudgetExceededError,
    CancelledError,
    CancelToken,
    ChatTurn,
    GenerationRequest,
    GenerationResult,
    ImageInput,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderRequestError,
    ProviderUnavailableError,
    Purpose,
    Role,
    Usage,
)
from chibi.core.config import Settings
from chibi.core.events import AIStatusChanged, Permission
from chibi.memory.repositories import PermissionRepository
from chibi.privacy.permissions import PermissionManager

SECRET = "мій пароль від банку 12345"


def pump(qapp, condition: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.002)
    qapp.processEvents()
    return condition()


class Outcome:
    """Collects one submit's callbacks (and which thread ran them)."""

    def __init__(self) -> None:
        self.done: list[GenerationResult] = []
        self.errors: list[Exception] = []
        self.deltas: list[str] = []
        self.threads: set[str] = set()

    def on_done(self, result: GenerationResult) -> None:
        self.threads.add(threading.current_thread().name)
        self.done.append(result)

    def on_error(self, exc: Exception) -> None:
        self.threads.add(threading.current_thread().name)
        self.errors.append(exc)

    def on_delta(self, piece: str) -> None:
        self.threads.add(threading.current_thread().name)
        self.deltas.append(piece)

    @property
    def finished(self) -> bool:
        return bool(self.done or self.errors)

    @property
    def error(self) -> Exception:
        assert len(self.errors) == 1 and not self.done, (self.done, self.errors)
        return self.errors[0]

    @property
    def result(self) -> GenerationResult:
        assert len(self.done) == 1 and not self.errors, (self.done, self.errors)
        return self.done[0]


class Harness:
    def __init__(self, qapp, db, clock, bus, provider: FakeProvider | None, *, ai_provider: str = "ollama",
                 cloud: bool = False, key: str | None = None) -> None:
        self.qapp = qapp
        self.bus = bus
        self.clock = clock
        self.settings = Settings()
        self.settings.ai.provider = ai_provider
        self.permissions = PermissionManager(PermissionRepository(db), bus, clock)
        if cloud:
            self.permissions.grant(Permission.CLOUD_AI)
        self.secrets = MemorySecretStore()
        if key:
            self.secrets.set("anthropic_api_key", key)
        self.ledger = UsageLedger(db, clock)
        self.provider = provider
        self.built: list[str] = []
        self.sleeps: list[float] = []
        self.statuses: list[AIStatusChanged] = []
        bus.subscribe(AIStatusChanged, self.statuses.append)
        self.service = AIService(bus, lambda: self.settings, self.permissions, self.secrets, self.ledger,
                                 clock, provider_factory=self._factory, sleep=self._sleep)
        self.on_sleep: Callable[[float], None] | None = None

    def _factory(self, ai, secrets):
        self.built.append(ai.provider)
        if ai.provider == "openai" and not ai.model:
            raise ProviderRequestError("Вибери модель", user_message="Вибери модель для сервера.")
        return self.provider

    def _sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if self.on_sleep is not None:
            self.on_sleep(seconds)

    def request(self, purpose: Purpose = Purpose.REACTION, text: str = "Привіт",
                **kwargs) -> GenerationRequest:
        return GenerationRequest(system="Ти — Мія.", turns=(ChatTurn(Role.USER, text),), purpose=purpose,
                                 **kwargs)

    def submit(self, purpose: Purpose = Purpose.REACTION, text: str = "Привіт", *, stream: bool = False,
               cancel: CancelToken | None = None, **kwargs) -> tuple[Outcome, CancelToken]:
        outcome = Outcome()
        token = self.service.submit(self.request(purpose, text, **kwargs), on_done=outcome.on_done,
                                    on_error=outcome.on_error, on_delta=outcome.on_delta if stream else None,
                                    cancel=cancel)
        return outcome, token

    def wait(self, *outcomes: Outcome, timeout: float = 3.0) -> None:
        assert pump(self.qapp, lambda: all(o.finished for o in outcomes), timeout), "callbacks did not arrive"

    def run(self, purpose: Purpose = Purpose.REACTION, text: str = "Привіт", **kwargs) -> Outcome:
        outcome, _ = self.submit(purpose, text, **kwargs)
        self.wait(outcome)
        return outcome

    def dispatch(self) -> None:
        self.bus.dispatch_pending()


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


def wait_started(provider: FakeProvider, timeout: float = 3.0) -> None:
    assert provider.started.wait(timeout), "the worker never called the provider"


# -- basics -----------------------------------------------------------------------------------------


def test_generate_calls_back_on_main_thread_and_records_usage(make):
    h = make(FakeProvider(["Привіт! Я тут, поруч."]))
    out = h.run(Purpose.CHAT, stream=True)
    assert out.result.text == "Привіт! Я тут, поруч."
    assert "".join(out.deltas) == "Привіт! Я тут, поруч." and len(out.deltas) == 4
    assert out.threads == {threading.main_thread().name}
    assert h.provider.requests[0].purpose == Purpose.CHAT
    today = h.service.usage_today()
    assert today.requests == 1 and today.cost_usd == 0.0 and today.cost_known


def test_submit_never_calls_back_synchronously(make):
    h = make(FakeProvider(["ok"]))
    out, _ = h.submit()
    assert not out.finished  # even an immediate answer arrives through the event loop
    h.wait(out)


def test_settings_stream_off_delivers_one_delta(make):
    h = make(FakeProvider(["Одна ціла відповідь"]))
    h.settings.ai.stream = False
    out = h.run(Purpose.CHAT, stream=True)
    assert out.deltas == ["Одна ціла відповідь"]
    assert out.result.text == "Одна ціла відповідь"


def test_callback_exceptions_are_contained(make, caplog):
    h = make(FakeProvider(["a", "b"]))

    def explode(result):
        raise ValueError(SECRET)

    with caplog.at_level(logging.DEBUG):
        h.service.submit(h.request(text="1"), on_done=explode, on_error=explode)
        out = h.run(text="2")
    assert out.result.text == "b"
    assert SECRET not in caplog.text and "ValueError" in caplog.text


def test_nothing_from_requests_or_replies_is_logged(make, caplog):
    h = make(FakeProvider([f"відповідь про {SECRET}", ProviderUnavailableError("x")], default_reply=None))
    with caplog.at_level(logging.DEBUG):
        h.run(Purpose.CHAT, SECRET, stream=True)
        h.run(Purpose.CHAT, SECRET)
    assert SECRET not in caplog.text


# -- priority queue ---------------------------------------------------------------------------------


def test_priority_order_then_fifo(make):
    gate = threading.Event()
    h = make(FakeProvider(gate=gate))
    first, _ = h.submit(Purpose.MEMORY, "перший")
    wait_started(h.provider)
    order = [(Purpose.MEMORY, "m"), (Purpose.INITIATIVE, "i1"), (Purpose.CHAT, "c"),
             (Purpose.INITIATIVE, "i2")]
    outs = [h.submit(p, t)[0] for p, t in order]
    assert h.service.pending() == 4
    gate.set()
    h.wait(first, *outs)
    assert [r.turns[0].text for r in h.provider.requests] == ["перший", "c", "i1", "i2", "m"]


def test_queue_overflow_drops_lowest_priority_oldest(make):
    gate = threading.Event()
    h = make(FakeProvider(gate=gate))
    running, _ = h.submit(Purpose.CHAT, "running")
    wait_started(h.provider)
    r1, _ = h.submit(Purpose.REACTION, "r1")
    mem, _ = h.submit(Purpose.MEMORY, "mem")
    r2, _ = h.submit(Purpose.REACTION, "r2")
    ini, _ = h.submit(Purpose.INITIATIVE, "ini")
    assert h.service.pending() == MAX_QUEUE
    chat, _ = h.submit(Purpose.CHAT, "chat")  # full: MEMORY goes
    assert pump(h.qapp, lambda: mem.finished)
    assert isinstance(mem.error, ServiceQueueFull) and isinstance(mem.error, ProviderUnavailableError)
    assert "черга переповнена" in mem.error.user_message
    r3, _ = h.submit(Purpose.REACTION, "r3")  # full: INITIATIVE goes
    assert pump(h.qapp, lambda: ini.finished)
    assert isinstance(ini.error, ServiceQueueFull)
    late, _ = h.submit(Purpose.VISION, "late")  # lower than everything queued: itself dropped
    assert pump(h.qapp, lambda: late.finished)
    assert isinstance(late.error, ServiceQueueFull)
    r4, _ = h.submit(Purpose.REACTION, "r4")  # ties: the oldest REACTION (r1) goes
    assert pump(h.qapp, lambda: r1.finished)
    assert isinstance(r1.error, ServiceQueueFull)
    gate.set()
    h.wait(running, chat, r2, r3, r4)
    assert [r.turns[0].text for r in h.provider.requests] == ["running", "chat", "r2", "r3", "r4"]


def test_cancelled_queued_requests_free_their_slot(make):
    gate = threading.Event()
    h = make(FakeProvider(gate=gate))
    h.submit(Purpose.CHAT, "running")
    wait_started(h.provider)
    queued = [h.submit(Purpose.REACTION, f"q{i}") for i in range(MAX_QUEUE)]
    for _out, token in queued:
        token.cancel()
    fresh, _ = h.submit(Purpose.MEMORY, "fresh")
    assert pump(h.qapp, lambda: all(o.finished for o, _ in queued))
    for out, _ in queued:
        assert isinstance(out.error, CancelledError)
    gate.set()
    h.wait(fresh)
    assert fresh.result.text
    assert [r.turns[0].text for r in h.provider.requests] == ["running", "fresh"]


def test_cancel_running_request(make):
    gate = threading.Event()
    h = make(FakeProvider(gate=gate))
    running, token = h.submit(Purpose.CHAT, "long")
    after, _ = h.submit(Purpose.REACTION, "after")
    wait_started(h.provider)
    token.cancel()
    assert pump(h.qapp, lambda: running.finished)
    assert isinstance(running.error, CancelledError)
    gate.set()
    h.wait(after)
    assert after.result.text


def test_caller_token_is_returned_and_respected(make):
    h = make(FakeProvider(["never"]))
    token = CancelToken()
    token.cancel()
    out, returned = h.submit(cancel=token)
    assert returned is token
    h.wait(out)
    assert isinstance(out.error, CancelledError)
    assert h.provider.requests == []


def test_cancel_mid_stream_drops_remaining_deltas(make):
    release = threading.Event()

    class Streamer(FakeProvider):
        def stream(self, request, on_delta, cancel=None):
            self.requests.append(request)
            for i in range(50):
                if cancel is not None:
                    cancel.raise_if_cancelled()
                on_delta(f"w{i} ")
                if i == 1:  # the 3rd delta is already on its way when the user cancels
                    on_delta("late ")
                    release.wait(3.0)
            return GenerationResult("done", "fake", "fake-model")

    h = make(Streamer())
    out, token = h.submit(Purpose.CHAT, stream=True)
    assert pump(h.qapp, lambda: len(out.deltas) >= 2)
    token.cancel()
    release.set()
    h.wait(out)
    assert isinstance(out.error, CancelledError)
    pump(h.qapp, lambda: False, timeout=0.05)
    # deltas queued before the cancel but not yet delivered are dropped
    assert out.deltas in (["w0 ", "w1 "], ["w0 ", "w1 ", "late "])
    assert "w2 " not in out.deltas


# -- retries and the circuit breaker ----------------------------------------------------------------


def test_retries_with_backoff_then_succeeds(make):
    h = make(FakeProvider([ProviderUnavailableError("a"), ProviderUnavailableError("b"), "нарешті"]))
    out = h.run(Purpose.INITIATIVE)
    assert out.result.text == "нарешті"
    assert h.sleeps == list(BACKOFF_SECONDS) == [1.0, 3.0]
    assert h.provider.calls == 3


def test_gives_up_after_two_retries(make):
    h = make(FakeProvider([ProviderUnavailableError("down")] * 3, default_reply=None))
    out = h.run(Purpose.CHAT)
    assert isinstance(out.error, ProviderUnavailableError)
    assert h.provider.calls == 3 and h.sleeps == [1.0, 3.0]


def test_rate_limit_retry_after_is_respected_up_to_ten_seconds(make):
    h = make(FakeProvider([ProviderRateLimitError("slow", retry_after=5.0), "ok",
                           ProviderRateLimitError("slow", retry_after=0.2), "ok",
                           ProviderRateLimitError("slower", retry_after=45.0)], default_reply=None))
    assert h.run(Purpose.CHAT, "1").result.text == "ok"
    assert h.run(Purpose.CHAT, "2").result.text == "ok"
    assert h.sleeps == [5.0, 1.0]
    out = h.run(Purpose.CHAT, "3")
    assert isinstance(out.error, ProviderRateLimitError)  # 45 s is not worth holding the queue
    assert h.sleeps == [5.0, 1.0]


@pytest.mark.parametrize("error", [ProviderAuthError("401"), ProviderRequestError("404"),
                                   ProviderRefusalError("no")])
def test_non_retryable_errors_fail_at_once(make, error):
    h = make(FakeProvider([error], default_reply=None))
    out = h.run(Purpose.CHAT)
    assert out.error is error
    assert h.provider.calls == 1 and h.sleeps == []


def test_refusal_counts_as_a_request(make):
    h = make(FakeProvider([ProviderRefusalError("no")]))
    h.run(Purpose.CHAT)
    assert h.service.usage_today().requests == 1


def test_no_retry_after_partial_stream(make):
    class HalfStream(FakeProvider):
        def stream(self, request, on_delta, cancel=None):
            self.calls += 1
            on_delta("Почала казати")
            raise ProviderUnavailableError("connection reset")

    h = make(HalfStream())
    out = h.run(Purpose.CHAT, stream=True)
    assert isinstance(out.error, ProviderUnavailableError)
    assert h.provider.calls == 1 and out.deltas == ["Почала казати"]


def test_cancel_during_backoff(make):
    h = make(FakeProvider([ProviderUnavailableError("down"), "never"]))
    token = CancelToken()
    h.on_sleep = lambda seconds: token.cancel()
    out, _ = h.submit(Purpose.CHAT, cancel=token)
    h.wait(out)
    assert isinstance(out.error, CancelledError)
    assert h.provider.calls == 1


def test_real_backoff_wait_is_interruptible(qapp, db, clock, bus):
    h = Harness(qapp, db, clock, bus, FakeProvider([ProviderUnavailableError("down"), "never"]))
    h.service._sleep = None  # use the real interruptible wait
    try:
        out, token = h.submit(Purpose.CHAT)
        assert pump(h.qapp, lambda: h.provider.calls == 1)
        started = time.monotonic()
        token.cancel()
        h.wait(out)
        assert time.monotonic() - started < 0.5
        assert isinstance(out.error, CancelledError)
    finally:
        h.service.shutdown()


def test_unexpected_provider_crash_becomes_unavailable(make, caplog):
    h = make(FakeProvider([RuntimeError(SECRET)], default_reply=None))
    with caplog.at_level(logging.DEBUG):
        out = h.run(Purpose.CHAT)
    assert isinstance(out.error, ProviderUnavailableError) and h.provider.calls == 1
    assert SECRET not in caplog.text


def test_circuit_breaker_opens_after_three_unavailable_failures(make):
    down = [ProviderUnavailableError("down")] * 9
    h = make(FakeProvider([*down, "знову тут", *down[:3]], default_reply=None))
    for i in range(3):
        assert isinstance(h.run(Purpose.REACTION, f"r{i}").error, ProviderUnavailableError)
    assert h.service.status() == (False, REASON_DOWN)
    h.dispatch()
    assert h.statuses[-1].online is False and h.statuses[-1].reason == REASON_DOWN
    calls = h.provider.calls
    out = h.run(Purpose.CHAT)
    assert isinstance(out.error, AIOfflineError) and out.error.user_message == REASON_DOWN
    assert h.provider.calls == calls  # nothing was sent while open
    assert not h.service.is_available(Purpose.CHAT)

    h.clock.advance(BREAKER_COOLDOWN + 1)
    assert h.service.status() == (True, "")
    assert h.run(Purpose.CHAT, "after").result.text == "знову тут"
    assert h.service.status() == (True, "")
    # after a success the counter starts over: one more failed request doesn't re-open it
    assert isinstance(h.run(Purpose.CHAT, "x").error, ProviderUnavailableError)
    assert h.service.status() == (True, "")


def test_breaker_half_open_reopens_on_first_failure(make):
    h = make(FakeProvider([ProviderUnavailableError("down")] * 12, default_reply=None))
    for i in range(3):
        h.run(Purpose.REACTION, f"r{i}")
    assert h.service.status()[1] == REASON_DOWN
    h.clock.advance(BREAKER_COOLDOWN + 1)
    assert h.service.status()[0]
    h.run(Purpose.REACTION, "probe")
    assert h.service.status()[1] == REASON_DOWN


def test_rate_limits_and_auth_errors_do_not_trip_the_breaker(make):
    errors = [ProviderRateLimitError("429", retry_after=60)] * 4 + [ProviderAuthError("401")] * 3
    h = make(FakeProvider(errors, default_reply=None))
    for i in range(7):
        h.run(Purpose.CHAT, str(i))
    assert h.service.status() == (True, "")


# -- status, permission and keys --------------------------------------------------------------------


def test_status_reasons(make, monkeypatch):
    h = make(FakeProvider(local=False), ai_provider="none")
    assert h.service.status() == (False, REASON_OFF)
    assert h.built == []  # nothing is built for "none"

    h.settings.ai.provider = "anthropic"
    h.service.reconfigure()
    assert h.service.status() == (False, REASON_PERMISSION)
    h.permissions.grant(Permission.CLOUD_AI)
    h.dispatch()
    assert h.service.status() == (False, REASON_NO_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    assert h.service.status() == (True, "")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    h.secrets.set("anthropic_api_key", "sk-stored")
    assert h.service.status() == (False, REASON_NO_KEY)  # keys are re-read on reconfigure()
    h.service.reconfigure()
    assert h.service.status() == (True, "")

    h.settings.ai.provider = "openai"
    h.settings.ai.model = ""
    h.service.reconfigure()
    online, reason = h.service.status()
    assert not online and "Вибери модель" in reason

    h.settings.ai.model = "local-model"
    h.settings.ai.openai_base_url = "http://localhost:1234/v1"
    h.service.reconfigure()
    assert h.service.status() == (True, "")  # local server: no key, no cloud permission needed


def test_status_changes_are_published_once(make):
    h = make(FakeProvider(model="gemma3:4b"))
    h.dispatch()
    published = [(s.online, s.provider, s.model, s.reason) for s in h.statuses]
    assert published == [(True, "ollama", "gemma3:4b", "")]
    h.service.status()
    h.service.is_available(Purpose.CHAT)
    h.dispatch()
    assert len(h.statuses) == 1
    h.settings.ai.provider = "none"
    h.service.reconfigure()
    h.dispatch()
    assert (h.statuses[-1].online, h.statuses[-1].reason) == (False, REASON_OFF)


def test_is_available_respects_settings_and_budget(make):
    h = make(FakeProvider())
    assert all(h.service.is_available(p) for p in Purpose)
    h.settings.ai.use_for_chat = False
    assert not h.service.is_available(Purpose.CHAT)
    assert h.service.is_available(Purpose.REACTION)
    h.settings.ai.use_for_reactions = False
    assert not h.service.is_available(Purpose.REACTION)
    assert not h.service.is_available(Purpose.INITIATIVE)
    assert h.service.is_available(Purpose.MEMORY) and h.service.is_available(Purpose.VISION)
    h.settings.ai.max_requests_per_hour = 1
    h.ledger.record("fake", "m", Purpose.MEMORY, Usage(), 0.0)
    assert not h.service.is_available(Purpose.MEMORY)


def test_offline_submit_fails_fast_without_calling_provider(make):
    h = make(FakeProvider(local=False), ai_provider="anthropic")
    out = h.run(Purpose.CHAT)
    assert isinstance(out.error, AIOfflineError) and out.error.user_message == REASON_PERMISSION
    assert h.provider.calls == 0


def test_permission_revoked_mid_queue_drops_cloud_requests(make):
    gate = threading.Event()
    h = make(FakeProvider(gate=gate, local=False), ai_provider="anthropic", cloud=True, key="sk-ant")
    running, _ = h.submit(Purpose.CHAT, "running")
    wait_started(h.provider)
    queued = [h.submit(Purpose.REACTION, f"q{i}")[0] for i in range(3)]
    h.permissions.revoke(Permission.CLOUD_AI)
    h.dispatch()
    assert pump(h.qapp, lambda: running.finished and all(o.finished for o in queued))
    for out in [running, *queued]:
        assert isinstance(out.error, AIOfflineError) and out.error.user_message == REASON_PERMISSION
    gate.set()
    pump(h.qapp, lambda: False, timeout=0.05)
    assert [r.turns[0].text for r in h.provider.requests] == ["running"]
    assert h.service.status() == (False, REASON_PERMISSION)


def test_worker_rechecks_permission_before_every_request(make):
    """Even before the bus delivers PermissionChanged, the worker sends nothing."""
    gate = threading.Event()
    h = make(FakeProvider(gate=gate, local=False), ai_provider="anthropic", cloud=True, key="sk-ant")
    running, _ = h.submit(Purpose.CHAT, "running")
    wait_started(h.provider)
    queued, _ = h.submit(Purpose.REACTION, "queued")
    h.permissions.revoke(Permission.CLOUD_AI)  # no dispatch: the event is still queued on the bus
    gate.set()
    h.wait(running, queued)
    assert isinstance(queued.error, AIOfflineError)
    assert [r.turns[0].text for r in h.provider.requests] == ["running"]


def test_permission_recheck_before_retry(make):
    h = make(FakeProvider([ProviderUnavailableError("down"), "never"], local=False),
             ai_provider="anthropic", cloud=True, key="sk-ant")
    h.on_sleep = lambda seconds: h.permissions.revoke(Permission.CLOUD_AI)
    out = h.run(Purpose.CHAT)
    assert isinstance(out.error, AIOfflineError) and h.provider.calls == 1


def test_local_provider_needs_no_cloud_permission(make):
    h = make(FakeProvider(["локально"], local=True), ai_provider="ollama")
    assert not h.permissions.is_granted(Permission.CLOUD_AI)
    assert h.run(Purpose.CHAT).result.text == "локально"


# -- budget ---------------------------------------------------------------------------------------------


def test_daily_request_budget(make):
    h = make(FakeProvider())
    h.settings.ai.max_requests_per_day = 2
    h.run(Purpose.CHAT, "1")
    h.run(Purpose.CHAT, "2")
    assert h.service.status() == (False, REASON_LIMIT)
    out = h.run(Purpose.CHAT, "3")
    assert isinstance(out.error, AIOfflineError) and out.error.user_message == REASON_LIMIT
    assert h.provider.calls == 2
    h.dispatch()
    assert h.statuses[-1].reason == REASON_LIMIT


def test_hourly_budget_error_reaches_on_error(make):
    h = make(FakeProvider())
    h.settings.ai.max_requests_per_hour = 1
    h.run(Purpose.REACTION, "1")
    out = h.run(Purpose.REACTION, "2")
    assert isinstance(out.error, BudgetExceededError) and "годину" in out.error.user_message
    assert h.service.status() == (True, "")  # hourly caps don't mark her offline
    assert h.run(Purpose.CHAT, "chat").result  # the user is waiting: chat gets the overdraft


def test_cost_cap(make):
    h = make(FakeProvider())
    h.settings.ai.max_cost_per_day_usd = 0.5
    h.ledger.record("anthropic", "claude-opus-5-5", Purpose.CHAT, Usage(10, 10), 0.6)
    assert h.service.status() == (False, REASON_LIMIT)
    out = h.run(Purpose.CHAT)
    assert isinstance(out.error, AIOfflineError)


# -- cache ------------------------------------------------------------------------------------------------


def test_identical_non_chat_requests_hit_the_cache(make):
    h = make(FakeProvider(["перша", "друга", "третя"]))
    a = h.run(Purpose.REACTION, "гра почалась")
    b = h.run(Purpose.REACTION, "гра почалась")
    assert a.result.text == b.result.text == "перша"
    assert b.result.from_cache and not a.result.from_cache
    assert h.provider.calls == 1
    assert h.service.usage_today().requests == 1  # cache hits cost nothing
    c = h.run(Purpose.REACTION, "гра почалась", context="інший настрій")
    assert c.result.text == "друга"


def test_chat_and_images_are_never_cached(make):
    h = make(FakeProvider(["1", "2", "3", "4"]))
    h.run(Purpose.CHAT, "привіт")
    assert h.run(Purpose.CHAT, "привіт").result.text == "2"
    request = GenerationRequest(system="s", purpose=Purpose.VISION, turns=(
        ChatTurn(Role.USER, "що там?", images=(ImageInput(b"png"),)),))
    outs = []
    for _ in range(2):
        out = Outcome()
        h.service.submit(request, on_done=out.on_done, on_error=out.on_error)
        h.wait(out)
        outs.append(out)
    assert [o.result.text for o in outs] == ["3", "4"]


def test_cache_expires_after_ten_minutes(make):
    h = make(FakeProvider(["old", "new"]))
    h.run(Purpose.INITIATIVE, "x")
    h.clock.advance(CACHE_TTL - 1)
    assert h.run(Purpose.INITIATIVE, "x").result.text == "old"
    h.clock.advance(2)
    assert h.run(Purpose.INITIATIVE, "x").result.text == "new"


def test_cache_is_lru_bounded(make):
    h = make(FakeProvider(default_reply="r"))
    h.settings.ai.max_requests_per_hour = 2000
    for i in range(CACHE_SIZE + 1):
        assert h.run(Purpose.MEMORY, f"m{i}").result
    calls = h.provider.calls
    assert calls == CACHE_SIZE + 1
    h.run(Purpose.MEMORY, f"m{CACHE_SIZE}")  # newest: cached
    assert h.provider.calls == calls
    h.run(Purpose.MEMORY, "m0")  # oldest: evicted
    assert h.provider.calls == calls + 1


def test_cached_reply_streams_once_to_delta_consumers(make):
    h = make(FakeProvider(["Ура, гра!"]))
    h.run(Purpose.REACTION, "x")
    out = h.run(Purpose.REACTION, "x", stream=True)
    assert out.deltas == ["Ура, гра!"] and out.result.from_cache


# -- test(), list_models(), reconfigure(), shutdown() ------------------------------------------------


def test_test_request_success_and_failure(make):
    h = make(FakeProvider(["Привіт!", ProviderAuthError("401", user_message="Ключ не підходить.")]))
    results: list[tuple[bool, str]] = []
    h.service.test(lambda ok, msg: results.append((ok, msg)))
    assert pump(h.qapp, lambda: len(results) == 1)
    assert results[0][0] is True and "Працює" in results[0][1] and "Привіт!" in results[0][1]
    request = h.provider.requests[0]
    assert request.purpose == Purpose.CHAT and request.turns[0].text == TEST_PROMPT
    h.service.test(lambda ok, msg: results.append((ok, msg)))
    assert pump(h.qapp, lambda: len(results) == 2)
    assert results[1] == (False, "Ключ не підходить.")


def test_test_reports_offline_reason_and_bypasses_breaker(make):
    h = make(FakeProvider([ProviderUnavailableError("down")] * 9 + ["Привіт"], local=False),
             ai_provider="anthropic", cloud=False)
    results: list[tuple[bool, str]] = []
    h.service.test(lambda ok, msg: results.append((ok, msg)))
    assert pump(h.qapp, lambda: results)
    assert results == [(False, REASON_PERMISSION)] and h.provider.calls == 0

    h.permissions.grant(Permission.CLOUD_AI)
    h.secrets.set("anthropic_api_key", "sk")
    h.service.reconfigure()
    for i in range(3):
        h.run(Purpose.REACTION, str(i))
    assert h.service.status()[1] == REASON_DOWN
    h.service.test(lambda ok, msg: results.append((ok, msg)))
    assert pump(h.qapp, lambda: len(results) == 2)
    assert results[1][0] is True
    assert h.service.status() == (True, "")  # a success closes the breaker


def test_list_models(make):
    h = make(FakeProvider(models=["gemma3:4b", "llama3.2:3b"]))
    got: list[tuple[list[str], str]] = []
    h.service.list_models(lambda models, err: got.append((models, err)))
    assert pump(h.qapp, lambda: got)
    assert got == [(["gemma3:4b", "llama3.2:3b"], "")]


def test_list_models_error_and_off(make):
    h = make(FakeProvider(models_error=ProviderUnavailableError("x", user_message="Ollama не запущена")))
    got: list[tuple[list[str], str]] = []
    h.service.list_models(lambda models, err: got.append((models, err)))
    assert pump(h.qapp, lambda: got)
    assert got == [([], "Ollama не запущена")]
    h.settings.ai.provider = "none"
    h.service.reconfigure()
    h.service.list_models(lambda models, err: got.append((models, err)))
    assert pump(h.qapp, lambda: len(got) == 2)
    assert got[1] == ([], REASON_OFF)


def test_list_models_works_before_a_model_is_chosen(make):
    h = make(FakeProvider(models=["gpt-5-mini"], local=False), ai_provider="openai")
    assert not h.service.status()[0]  # "Вибери модель"
    got: list[tuple[list[str], str]] = []
    h.service.list_models(lambda models, err: got.append((models, err)))
    assert pump(h.qapp, lambda: got)
    assert got == [(["gpt-5-mini"], "")]


def test_reconfigure_swaps_and_closes_the_old_provider(make):
    old = FakeProvider(["from old"])
    h = make(old)
    assert h.run(Purpose.CHAT).result.text == "from old"
    new = FakeProvider(["from new"], model="new-model")
    h.provider = new
    h.service.reconfigure()
    assert old.closed
    assert h.run(Purpose.CHAT).result.text == "from new"
    h.dispatch()
    assert h.statuses[-1].model == "new-model"


def test_reconfigure_while_busy_closes_old_provider_afterwards(make):
    gate = threading.Event()
    old = FakeProvider(["old answer"], gate=gate)
    h = make(old)
    out, _ = h.submit(Purpose.CHAT)
    wait_started(old)
    h.provider = FakeProvider(["new"])
    h.service.reconfigure()
    assert not old.closed  # still in use by the worker
    gate.set()
    h.wait(out)
    assert out.result.text == "old answer"
    assert pump(h.qapp, lambda: old.closed)


def test_shutdown_is_prompt_and_silences_callbacks(make):
    gate = threading.Event()
    h = make(FakeProvider(gate=gate))
    running, _ = h.submit(Purpose.CHAT, "busy")
    queued, _ = h.submit(Purpose.REACTION, "waiting")
    wait_started(h.provider)
    thread = h.service._thread
    started = time.monotonic()
    h.service.shutdown()
    assert time.monotonic() - started < 1.0
    assert thread is not None and not thread.is_alive()
    assert h.provider.closed
    pump(h.qapp, lambda: False, timeout=0.05)
    assert not running.finished and not queued.finished
    _out, token = h.submit(Purpose.CHAT)  # after shutdown: refused silently
    assert token.cancelled
    h.service.shutdown()  # idempotent


def test_submit_from_all_purposes_and_main_thread_delivery_of_errors(make):
    h = make(FakeProvider([ProviderUnavailableError("x")] * 3 + ["ok"] * 5))
    outs = [h.run(p, p.value) for p in Purpose]
    for out in outs:
        assert out.threads == {threading.main_thread().name}
    assert isinstance(outs[0].error, ProviderError)


def test_a_queued_screen_frame_never_runs_on_a_provider_configured_after_it(make):
    # Vision checked the provider that was current when it sent the frame; the
    # settings can switch to another (a cloud one) before the worker gets to it.
    gate = threading.Event()
    local = FakeProvider(["chat"], gate=gate)
    h = make(local)
    h.submit(Purpose.CHAT, "running")
    wait_started(local)
    frame = Outcome()
    h.service.submit(GenerationRequest(system="s", purpose=Purpose.VISION, json_mode=True, turns=(
        ChatTurn(Role.USER, "Що тут?", images=(ImageInput(b"\x89PNG-bytes"),)),)),
        on_done=frame.on_done, on_error=frame.on_error)
    text_line, _ = h.submit(Purpose.REACTION, "без картинок")
    other = FakeProvider(['{"scene": "menu"}', "ok"])
    h.provider = other
    h.settings.ai.model = "another-model"
    h.service.reconfigure()
    gate.set()
    h.wait(frame, text_line)
    assert isinstance(frame.error, CancelledError)
    assert [r.turns[0].text for r in other.requests] == ["без картинок"]  # text requests carry on


# -- what she saw on the screen (review fix: screen data bound to the provider that was checked) --------


def test_a_queued_screen_description_never_reaches_a_provider_configured_after_it(make):
    # The observer checked a local provider; the user switches to a cloud one while
    # the decision waits behind a chat answer: it must not go there.
    gate = threading.Event()
    local = FakeProvider(["chat"], gate=gate)
    h = make(local)
    h.permissions.grant(Permission.CLOUD_AI)
    h.submit(Purpose.CHAT, "running")
    wait_started(local)
    decision, _ = h.submit(Purpose.INITIATIVE, "<observation>лист до лікаря про діагноз</observation>",
                           screen_data=True)
    text_line, _ = h.submit(Purpose.REACTION, "без екрана")
    cloud = FakeProvider(["ok"] * 3, local=False)
    h.provider = cloud
    h.settings.ai.provider = "anthropic"
    h.secrets.set("anthropic_api_key", "sk-ant")
    h.service.reconfigure()
    gate.set()
    h.wait(decision, text_line)
    assert isinstance(decision.error, CancelledError)
    assert [r.turns[0].text for r in cloud.requests] == ["без екрана"]


def test_screen_data_goes_to_a_cloud_provider_only_with_cloud_vision(make):
    cloud = FakeProvider(["ok"] * 4, local=False)
    h = make(cloud, ai_provider="anthropic", cloud=True, key="sk-ant")
    out = h.run(Purpose.INITIATIVE, "<observation>помилка</observation>", screen_data=True)
    assert isinstance(out.error, CancelledError) and cloud.requests == []
    h.permissions.grant(Permission.SCREEN_CAPTURE)
    h.permissions.grant(Permission.CLOUD_VISION)
    h.settings.vision.local_only = True
    out = h.run(Purpose.INITIATIVE, "<observation>помилка</observation>", screen_data=True)
    assert isinstance(out.error, CancelledError) and cloud.requests == []
    h.settings.vision.local_only = False
    seen = "<observation>помилка</observation>"
    assert h.run(Purpose.INITIATIVE, seen, screen_data=True).result.text == "ok"
    assert h.run(Purpose.REACTION, "без екрана").result.text == "ok"  # other requests: CLOUD_AI only


def test_revoking_cloud_vision_drops_queued_screen_data(make):
    gate = threading.Event()
    cloud = FakeProvider(["chat", "ok", "ok"], gate=gate, local=False)
    h = make(cloud, ai_provider="anthropic", cloud=True, key="sk-ant")
    h.permissions.grant(Permission.SCREEN_CAPTURE)
    h.permissions.grant(Permission.CLOUD_VISION)
    h.submit(Purpose.CHAT, "running")
    wait_started(cloud)
    decision, _ = h.submit(Purpose.INITIATIVE, "<observation>помилка</observation>", screen_data=True)
    text_line, _ = h.submit(Purpose.REACTION, "без екрана")
    h.permissions.revoke(Permission.CLOUD_VISION)
    h.dispatch()
    gate.set()
    h.wait(decision, text_line)
    assert isinstance(decision.error, CancelledError)
    assert [r.turns[0].text for r in cloud.requests] == ["running", "без екрана"]


def test_screen_data_runs_on_a_local_provider_without_cloud_vision(make):
    h = make(FakeProvider(["ok"]))
    seen = "<observation>помилка</observation>"
    assert h.run(Purpose.INITIATIVE, seen, screen_data=True).result.text == "ok"
