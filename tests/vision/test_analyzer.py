"""ProviderFrameAnalyzer: the vision request handed to the AI service (shape,
thread), error mapping to back-offs, timeout, cancellation, the last-moment
permission gate and the "can this model see images?" check."""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Callable

import pytest

import chibi.ai.providers.ollama_provider as ollama_module
from chibi.ai.budget import UsageLedger
from chibi.ai.providers.fake import FakeProvider
from chibi.ai.secrets import MemorySecretStore
from chibi.ai.service import REASON_LIMIT, REASON_PERMISSION, AIOfflineError, AIService, QueueFullError
from chibi.ai.types import (
    BudgetExceededError,
    CancelledError,
    CancelToken,
    GenerationRequest,
    GenerationResult,
    ImageInput,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderUnavailableError,
    Purpose,
    Role,
)
from chibi.core.config import AISettings, Settings, VisionSettings
from chibi.core.events import ConversationActive
from chibi.memory.repositories import PermissionRepository
from chibi.privacy.permissions import PermissionManager
from chibi.vision import analyzer as analyzer_module
from chibi.vision.analysis import SYSTEM_PROMPT, AnalysisError, Observation, user_prompt
from chibi.vision.analyzer import (
    BUDGET_RETRY,
    DEFAULT_RETRY,
    HOURLY_BUDGET_RETRY,
    HOURLY_BUDGET_USER_MESSAGE,
    MAX_TOKENS,
    NO_VISION_MESSAGE,
    OFFLINE_RETRY,
    ProviderFrameAnalyzer,
    default_vision_probe,
    to_analyzer_error,
)
from chibi.vision.integration import QtInvoker
from chibi.vision.service import AnalyzerError

ANSWER = ('{"scene": "gameplay", "event": "boss_fight", "confidence": 0.8, "interesting": true, '
          '"suggested_emotion": "surprised", "description": "Гравець б\'ється з великим босом", '
          '"spoiler": false}')
IMAGE = b"\xff\xd8jpeg-bytes"
PREVIOUS = Observation(scene="menu", event="none", confidence=0.9, interesting=False,
                       suggested_emotion="idle", description="меню", spoiler=False)


class FakeAI:
    """Records submits (and on which thread); answers at once or parks the callbacks."""

    def __init__(self, replies: list[str | Exception] | None = None, *, defer: bool = False) -> None:
        self.replies = list(replies or [])
        self.defer = defer
        self.requests: list[GenerationRequest] = []
        self.tokens: list[CancelToken | None] = []
        self.threads: list[str] = []
        self.parked: list[tuple[Callable, Callable]] = []
        #: Keywords beyond the basic ones (``user_waiting``), per submit.
        self.extra: list[dict[str, object]] = []

    def submit(self, request, *, on_done, on_error, on_delta=None, cancel=None, **extra):
        self.threads.append(threading.current_thread().name)
        self.requests.append(request)
        self.tokens.append(cancel)
        self.extra.append(extra)
        if self.defer:
            self.parked.append((on_done, on_error))
            return cancel
        reply = self.replies.pop(0) if self.replies else ANSWER
        if isinstance(reply, Exception):
            on_error(reply)
        else:
            on_done(GenerationResult(text=reply, provider="fake", model="fake-model"))
        return cancel


class MainThread:
    """The test thread plays the Qt main thread: posted callables run here."""

    def __init__(self) -> None:
        self.posted: queue.Queue[Callable[[], None]] = queue.Queue()

    def post(self, fn: Callable[[], None]) -> None:
        self.posted.put(fn)

    def run(self, fn: Callable[[], object], *, while_waiting: Callable[[], None] | None = None,
            timeout: float = 3.0) -> object:
        """Run *fn* on a "chibi-vision" thread; return its result or raise its exception."""
        box: dict[str, object] = {}

        def target() -> None:
            try:
                box["result"] = fn()
            except BaseException as exc:
                box["error"] = exc

        thread = threading.Thread(target=target, name="chibi-vision", daemon=True)
        thread.start()
        deadline = time.monotonic() + timeout
        while thread.is_alive() and time.monotonic() < deadline:
            with contextlib.suppress(queue.Empty):
                self.posted.get(timeout=0.005)()
            if while_waiting is not None:
                while_waiting()
        thread.join(0.5)
        assert not thread.is_alive(), "analyze() did not return"
        if "error" in box:
            raise box["error"]  # type: ignore[misc]
        return box["result"]


def make(ai: FakeAI, main: MainThread, provider: str = "ollama", **kwargs) -> ProviderFrameAnalyzer:
    kwargs.setdefault("probe", None)
    return ProviderFrameAnalyzer(ai, main.post, AISettings(provider=provider), VisionSettings(), **kwargs)


def analyze(main: MainThread, analyzer: ProviderFrameAnalyzer, *, urgent: bool = False,
            wanted: Callable[[], bool] | None = None, **kwargs) -> Observation:
    result = main.run(lambda: analyzer.analyze(IMAGE, "image/jpeg", "S.T.A.L.K.E.R. 2", PREVIOUS,
                                               urgent=urgent, wanted=wanted), **kwargs)
    assert isinstance(result, Observation)
    return result


# --- the request ----------------------------------------------------------------------------------


def test_request_shape_image_json_mode_and_vision_purpose():
    ai, main = FakeAI(), MainThread()
    observation = analyze(main, make(ai, main))
    (request,) = ai.requests
    assert request.system == SYSTEM_PROMPT
    assert request.purpose == Purpose.VISION
    assert request.json_mode is True
    assert request.max_tokens == MAX_TOKENS == 300
    (turn,) = request.turns
    assert turn.role == Role.USER
    assert turn.images == (ImageInput(data=IMAGE, media_type="image/jpeg"),)
    assert turn.text == user_prompt("S.T.A.L.K.E.R. 2", PREVIOUS)
    assert observation.event == "boss_fight" and observation.scene == "gameplay"
    assert observation.description == "Гравець б'ється з великим босом"


def test_the_ai_service_is_called_on_the_main_thread_with_a_cancel_token():
    ai, main = FakeAI(), MainThread()
    analyze(main, make(ai, main))
    assert ai.threads == [threading.current_thread().name] != ["chibi-vision"]
    assert isinstance(ai.tokens[0], CancelToken)


def test_the_analysis_is_local_exactly_when_the_ai_provider_is():
    ai, main = FakeAI(), MainThread()
    assert make(ai, main, "ollama").is_local
    assert not make(ai, main, "anthropic").is_local
    local_openai = ProviderFrameAnalyzer(ai, main.post, AISettings(provider="openai",
                                                                   openai_base_url="http://localhost:1234/v1"),
                                         VisionSettings(), probe=None)
    assert local_openai.is_local
    assert not make(ai, main, "openai").is_local  # api.openai.com


def test_vision_model_override_goes_on_the_request():
    ai, main = FakeAI(), MainThread()
    analyzer = ProviderFrameAnalyzer(ai, main.post, AISettings(provider="ollama"),
                                     VisionSettings(model=" llava "), probe=None)
    assert analyzer.model_override == "llava"
    analyze(main, analyzer)
    assert ai.requests[0].model == "llava"
    plain = make(ai, main)
    analyze(main, plain)
    assert ai.requests[1].model is None


def test_unusable_answer_raises_analysis_error():
    ai, main = FakeAI(["я не знаю, що тут"]), MainThread()
    with pytest.raises(AnalysisError):
        analyze(main, make(ai, main))


# --- errors ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("error", "retry", "budget"), [
    (BudgetExceededError("day limit", user_message="Ліміт"), BUDGET_RETRY, True),
    (BudgetExceededError("hour limit", user_message="Ліміт", hourly=True), HOURLY_BUDGET_RETRY, True),
    (AIOfflineError("offline", user_message=REASON_LIMIT), BUDGET_RETRY, True),
    (AIOfflineError("offline", user_message=REASON_PERMISSION), OFFLINE_RETRY, False),
    (AIOfflineError("offline", user_message="AI вимкнено"), OFFLINE_RETRY, False),
    (ProviderAuthError("401", user_message="Ключ не підходить"), OFFLINE_RETRY, False),
    (ProviderUnavailableError("down", user_message="Ollama не запущена"), OFFLINE_RETRY, False),
    (QueueFullError(), DEFAULT_RETRY, False),
    (ProviderRequestError("400", user_message="Погана модель"), DEFAULT_RETRY, False),
    (ProviderRateLimitError("429", retry_after=120.0), 120.0, False),
    (ProviderRateLimitError("429"), DEFAULT_RETRY, False),
    (CancelledError("cancelled"), DEFAULT_RETRY, False),
    (RuntimeError("boom"), DEFAULT_RETRY, False),
], ids=lambda value: type(value).__name__ if isinstance(value, Exception) else str(value))
def test_ai_errors_map_to_back_offs(error, retry, budget):
    ai, main = FakeAI([error]), MainThread()
    with pytest.raises(AnalyzerError) as caught:
        analyze(main, make(ai, main))
    assert caught.value.retry_after == retry
    assert caught.value.budget is budget
    assert caught.value.user_message  # always something to show


def test_offline_reason_is_explained_in_ukrainian():
    mapped = to_analyzer_error(AIOfflineError("x", user_message=REASON_PERMISSION))
    assert mapped.user_message == f"Не можу роздивитися екран: {REASON_PERMISSION}."


def test_a_failing_submit_is_an_analyzer_error_not_a_hang():
    class Broken(FakeAI):
        def submit(self, request, **kwargs):
            raise RuntimeError("no database")

    main = MainThread()
    with pytest.raises(AnalyzerError) as caught:
        analyze(main, make(Broken(), main))
    assert caught.value.retry_after == DEFAULT_RETRY


# --- timeout and cancellation -----------------------------------------------------------------------


def test_default_timeout_is_the_ai_request_timeout_plus_five_seconds():
    ai, main = FakeAI(), MainThread()
    analyzer = ProviderFrameAnalyzer(ai, main.post, AISettings(provider="ollama", request_timeout_seconds=40),
                                     VisionSettings(), probe=None)
    assert analyzer.timeout == 45.0


def test_timeout_cancels_the_request_and_backs_off():
    ai, main = FakeAI(defer=True), MainThread()
    started = time.monotonic()
    with pytest.raises(AnalyzerError) as caught:
        analyze(main, make(ai, main, timeout=0.1))
    assert time.monotonic() - started < 1.5
    assert caught.value.retry_after == DEFAULT_RETRY
    assert ai.tokens[0] is not None and ai.tokens[0].cancelled
    # A late answer after the timeout is harmless.
    on_done, _on_error = ai.parked[0]
    on_done(GenerationResult(text=ANSWER, provider="fake", model="m"))


def test_cancel_current_interrupts_the_wait_at_once():
    ai, main = FakeAI(defer=True), MainThread()
    analyzer = make(ai, main, timeout=30.0)

    def cancel_when_submitted() -> None:
        if ai.requests:
            analyzer.cancel_current()

    started = time.monotonic()
    with pytest.raises(AnalyzerError) as caught:
        analyze(main, analyzer, while_waiting=cancel_when_submitted)
    assert time.monotonic() - started < 2.0
    assert "cancelled" in str(caught.value)
    assert ai.tokens[0].cancelled
    # Still usable afterwards: only that request was abandoned.
    ai.defer = False
    assert analyze(main, analyzer).event == "boss_fight"


def test_closed_analyzer_never_submits():
    ai, main = FakeAI(), MainThread()
    analyzer = make(ai, main)
    analyzer.close()
    with pytest.raises(AnalyzerError):
        analyze(main, analyzer)
    assert ai.requests == [] and analyzer.closed


def test_close_while_waiting_returns_promptly():
    ai, main = FakeAI(defer=True), MainThread()
    analyzer = make(ai, main, timeout=30.0)
    started = time.monotonic()
    with pytest.raises(AnalyzerError):
        analyze(main, analyzer, while_waiting=lambda: analyzer.close() if ai.requests else None)
    assert time.monotonic() - started < 2.0


def test_permission_lost_before_submit_means_the_frame_never_leaves():
    allowed = {"value": True}
    ai, main = FakeAI(), MainThread()
    analyzer = make(ai, main, allowed=lambda: allowed["value"])
    allowed["value"] = False
    with pytest.raises(AnalyzerError):
        analyze(main, analyzer)
    assert ai.requests == []


# --- can the model see images? ----------------------------------------------------------------------


def test_a_model_without_vision_is_reported_once_and_never_asked_again():
    calls = []

    def probe() -> bool:
        calls.append(1)
        return False

    ai, main = FakeAI(), MainThread()
    analyzer = make(ai, main, probe=probe)
    for _ in range(2):
        with pytest.raises(AnalyzerError) as caught:
            analyze(main, analyzer)
        assert str(caught.value) == NO_VISION_MESSAGE == "Ця модель не бачить зображень"
        assert caught.value.retry_after >= 1800
    assert calls == [1]
    assert ai.requests == []


def test_an_unreachable_server_is_not_cached_as_no_vision():
    down = ProviderUnavailableError("down", user_message="Ollama не запущена")
    answers: list[bool | Exception] = [down, True]
    calls = []

    def probe() -> bool:
        calls.append(1)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    ai, main = FakeAI(), MainThread()
    analyzer = make(ai, main, probe=probe)
    with pytest.raises(AnalyzerError) as caught:
        analyze(main, analyzer)
    assert caught.value.retry_after == OFFLINE_RETRY
    assert analyze(main, analyzer).event == "boss_fight"
    assert calls == [1, 1]


def test_the_probe_runs_on_the_worker_thread():
    threads = []
    ai, main = FakeAI(), MainThread()
    analyze(main, make(ai, main, probe=lambda: threads.append(threading.current_thread().name) or True))
    assert threads == ["chibi-vision"]


def test_only_ollama_needs_a_probe():
    assert default_vision_probe(AISettings(provider="anthropic"), "") is None
    assert default_vision_probe(AISettings(provider="openai"), "gpt") is None
    assert default_vision_probe(AISettings(provider="ollama"), "gemma3") is not None


def test_the_ollama_probe_asks_the_configured_server_and_closes_its_client(monkeypatch):
    built = []

    class FakeOllama:
        def __init__(self, base_url, model, *, timeout):
            self.closed = False
            built.append((base_url, model, self))

        def list_models(self):
            return ["llama3.2"]

        @property
        def supports_vision(self):
            return False

        def close(self):
            self.closed = True

    monkeypatch.setattr(ollama_module, "OllamaProvider", FakeOllama)
    settings = AISettings(provider="ollama", ollama_url="http://localhost:11434")
    probe = default_vision_probe(settings, "llama3.2")
    assert probe is not None and probe() is False
    (url, model, provider), = built
    assert (url, model, provider.closed) == ("http://localhost:11434", "llama3.2", True)


def test_default_probe_is_used_for_ollama(monkeypatch):
    seen = []
    monkeypatch.setattr(analyzer_module, "default_vision_probe",
                        lambda ai, model: seen.append((ai.provider, model)) or (lambda: True))
    ProviderFrameAnalyzer(FakeAI(), MainThread().post, AISettings(provider="ollama", model="gemma3"),
                          VisionSettings())
    assert seen == [("ollama", "gemma3")]


# --- end to end with the real AI service and the Qt invoker ------------------------------------------


def test_end_to_end_through_ai_service_and_qt_invoker(qapp, db, clock, bus, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings()
    settings.ai.provider = "ollama"
    provider = FakeProvider([ANSWER], vision=True)
    ledger = UsageLedger(db, clock)
    service = AIService(bus, lambda: settings, PermissionManager(PermissionRepository(db), bus, clock),
                        MemorySecretStore(), ledger, clock, provider_factory=lambda ai, secrets: provider)
    invoker = QtInvoker()
    analyzer = ProviderFrameAnalyzer(service, invoker.post, settings.ai, settings.vision, probe=None)
    box: dict[str, object] = {}

    def work() -> None:
        try:
            box["obs"] = analyzer.analyze(IMAGE, "image/jpeg", "Counter-Strike 2", None)
        except Exception as exc:  # pragma: no cover - reported below
            box["error"] = exc

    thread = threading.Thread(target=work, name="chibi-vision")
    thread.start()
    deadline = time.monotonic() + 3.0
    while thread.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.002)
    thread.join(1.0)
    try:
        assert "error" not in box, box.get("error")
        assert isinstance(box["obs"], Observation) and box["obs"].event == "boss_fight"
        (request,) = provider.requests
        assert request.purpose == Purpose.VISION and request.turns[0].images[0].data == IMAGE
        assert ledger.today().requests == 1  # the budget sees vision requests too
    finally:
        service.shutdown()
        invoker.close()


def test_a_full_hour_is_not_called_a_spent_day():
    mapped = to_analyzer_error(BudgetExceededError("hour", user_message="x", hourly=True))
    assert mapped.user_message == HOURLY_BUDGET_USER_MESSAGE and "сьогодні" not in mapped.user_message
    assert mapped.retry_after == HOURLY_BUDGET_RETRY < BUDGET_RETRY


# --- the user's conversation comes first --------------------------------------------------------------


def test_a_frame_the_user_asked_for_is_marked_as_one_they_wait_for():
    ai, main = FakeAI(), MainThread()
    analyzer = make(ai, main)
    analyze(main, analyzer, urgent=True)
    analyze(main, analyzer)
    # Only the user's frame carries the keyword (an automatic one is plain background work).
    assert ai.extra == [{"user_waiting": True}, {}]


@pytest.mark.parametrize("wanted", [lambda: False, lambda: 1 / 0])
def test_an_abandoned_frame_never_reaches_the_ai(wanted):
    ai, main = FakeAI(), MainThread()
    analyzer = make(ai, main)
    with pytest.raises(AnalyzerError) as caught:
        analyze(main, analyzer, wanted=wanted)
    assert "cancelled" in str(caught.value)
    assert ai.requests == []  # checked on the main thread right before it would leave
    assert analyze(main, analyzer, wanted=lambda: True).event == "boss_fight"


def test_during_a_conversation_the_users_frame_is_analysed_at_once(qapp, db, clock, bus, monkeypatch):
    """Against the real AI service: while the user talks with her, background vision waits
    (the service holds it), but the frame the user asked for runs - so it is marked."""
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings()
    settings.ai.provider = "ollama"
    provider = FakeProvider([ANSWER, ANSWER], vision=True)
    service = AIService(bus, lambda: settings, PermissionManager(PermissionRepository(db), bus, clock),
                        MemorySecretStore(), UsageLedger(db, clock), clock,
                        provider_factory=lambda ai, secrets: provider)
    invoker = QtInvoker()
    analyzer = ProviderFrameAnalyzer(service, invoker.post, settings.ai, settings.vision, probe=None,
                                     timeout=1.0)
    bus.publish(ConversationActive(active=True))
    bus.dispatch_pending()

    def run(urgent: bool) -> object:
        box: dict[str, object] = {}

        def work() -> None:
            try:
                box["obs"] = analyzer.analyze(IMAGE, "image/jpeg", "Counter-Strike 2", None, urgent=urgent)
            except Exception as exc:
                box["error"] = exc

        thread = threading.Thread(target=work, name="chibi-vision")
        thread.start()
        deadline = time.monotonic() + 3.0
        while thread.is_alive() and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.002)
        thread.join(1.0)
        return box.get("obs", box.get("error"))

    try:
        seen = run(urgent=True)
        assert isinstance(seen, Observation) and seen.event == "boss_fight"
        assert len(provider.requests) == 1
        held = run(urgent=False)  # background work: held until the conversation ends
        assert isinstance(held, AnalyzerError) and "timed out" in str(held)
        assert len(provider.requests) == 1
    finally:
        service.shutdown()
        invoker.close()
