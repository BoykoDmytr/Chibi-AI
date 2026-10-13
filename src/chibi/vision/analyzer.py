"""The vision service's :class:`~chibi.vision.service.FrameAnalyzer` on top of
:class:`~chibi.ai.service.AIService`.

Why it is built this way:

* **One place runs model calls.**  The AI service owns the provider, the
  budget, the usage ledger, the cloud-permission gate and the circuit breaker;
  a second, private provider for vision would bypass all of them.  So a frame
  is analysed by submitting an ordinary :class:`GenerationRequest`
  (``Purpose.VISION``, JSON mode, the image in the user turn) to the service.
* **Threads.**  :meth:`ProviderFrameAnalyzer.analyze` runs on the vision
  worker thread, but ``AIService.submit`` must be called on the Qt main thread
  (it reads SQLite through the ledger).  The request is therefore *posted* to
  the main thread with the injected ``post`` (a queued Qt signal in the app)
  and the worker waits on a :class:`threading.Event` - bounded by
  ``request_timeout + 5 s`` and interruptible by :meth:`cancel_current` /
  :meth:`close`, so a kill switch or shutdown never waits for a model.
* **Last gate before the frame leaves.**  ``allowed()`` (screen capture still
  granted, cloud vision still granted for a cloud model) is re-checked on the
  main thread right before the request is handed to the AI service.
* **Can this model see?**  Claude always can and OpenAI-compatible servers are
  assumed to; for Ollama the model's capabilities are asked once (network I/O,
  so on the worker thread, never on the main thread).  A model without vision
  is reported once as "Ця модель не бачить зображень" with a long back-off;
  a server that is not running is *not* cached as "no vision".
* **Errors become** :class:`AnalyzerError` with a back-off the service honours:
  spent budget -> 1 h (``budget=True``), offline / missing permission or key ->
  5 min, anything else -> 1 min.  Messages never contain request content.

Nothing here logs the image, the prompt or the model's answer.
"""

from __future__ import annotations

import copy
import logging
import threading
from collections.abc import Callable
from types import EllipsisType
from typing import Protocol

from chibi.ai.providers.factory import is_cloud
from chibi.ai.service import REASON_LIMIT, AIOfflineError, QueueFullError
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
    ProviderUnavailableError,
    Purpose,
    Role,
)
from chibi.core.config import AISettings, VisionSettings
from chibi.vision import analysis
from chibi.vision.analysis import Observation
from chibi.vision.service import AnalyzerError

log = logging.getLogger(__name__)

MAX_TOKENS = 300
#: Waiting for the AI service: its own request timeout plus this (queueing, retries).
EXTRA_WAIT_SECONDS = 5.0
#: Back-offs handed to the vision service (seconds).
BUDGET_RETRY = 3600.0
OFFLINE_RETRY = 300.0
DEFAULT_RETRY = 60.0
NO_VISION_RETRY = 3600.0
#: The Ollama capability check is a quick local call; never hang on it.
PROBE_TIMEOUT = 10.0

NO_VISION_MESSAGE = "Ця модель не бачить зображень"
NO_VISION_USER_MESSAGE = (
    "Ця модель не бачить зображень. Для Ollama вибери модель із зором, наприклад gemma3 або llava "
    "(сторінка «AI»)."
)
TIMEOUT_USER_MESSAGE = "Модель не встигла роздивитися кадр — спробую пізніше."
CANCELLED_USER_MESSAGE = "Аналіз кадру скасовано."
FAILED_USER_MESSAGE = "Аналіз кадру не вдався."
BUDGET_USER_MESSAGE = "Ліміт AI на сьогодні вичерпано — на екран подивлюся пізніше."

Post = Callable[[Callable[[], None]], None]


class VisionAI(Protocol):
    """What the analyzer needs from :class:`~chibi.ai.service.AIService` (main thread)."""

    def submit(self, request: GenerationRequest, *, on_done: Callable[[GenerationResult], None],
               on_error: Callable[[Exception], None], on_delta: Callable[[str], None] | None = None,
               cancel: CancelToken | None = None) -> CancelToken: ...


def default_vision_probe(ai: AISettings, model: str) -> Callable[[], bool] | None:
    """How to find out whether the configured model accepts images.

    None means "no need to ask" (Claude always does; OpenAI-compatible servers
    are assumed to).  For Ollama the returned callable asks the server - call
    it on a worker thread.  It raises :class:`ProviderError` when the server
    cannot be reached, so "not running yet" is never mistaken for "no vision".
    """
    if ai.provider != "ollama":
        return None
    url = ai.ollama_url

    def probe() -> bool:
        from chibi.ai.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider(url, model, timeout=PROBE_TIMEOUT)
        try:
            provider.list_models()  # ProviderUnavailableError if Ollama is not running
            return bool(provider.supports_vision)
        finally:
            provider.close()

    return probe


def to_analyzer_error(exc: BaseException) -> AnalyzerError:
    """Map an AI-layer failure to the back-off the vision service should take."""
    if isinstance(exc, AnalyzerError):
        return exc
    reason = getattr(exc, "user_message", "")
    reason = reason.strip().rstrip(".") if isinstance(reason, str) else ""
    if isinstance(exc, BudgetExceededError) or (isinstance(exc, AIOfflineError) and reason == REASON_LIMIT):
        return AnalyzerError("AI budget exhausted", user_message=BUDGET_USER_MESSAGE,
                             retry_after=BUDGET_RETRY, budget=True)
    if isinstance(exc, CancelledError):
        return AnalyzerError("analysis cancelled", user_message=CANCELLED_USER_MESSAGE,
                             retry_after=DEFAULT_RETRY)
    user = f"Не можу роздивитися екран: {reason}." if reason else FAILED_USER_MESSAGE
    if isinstance(exc, QueueFullError):  # a busy queue clears quickly
        return AnalyzerError("AI queue full", user_message=user, retry_after=DEFAULT_RETRY)
    if isinstance(exc, (AIOfflineError, ProviderAuthError, ProviderUnavailableError)):
        return AnalyzerError(f"AI unavailable ({type(exc).__name__})", user_message=user,
                             retry_after=OFFLINE_RETRY)
    if isinstance(exc, ProviderRateLimitError):
        wait = exc.retry_after if exc.retry_after is not None else 0.0
        return AnalyzerError("AI rate limited", user_message=user, retry_after=max(DEFAULT_RETRY, wait))
    return AnalyzerError(f"analysis failed ({type(exc).__name__})", user_message=user,
                         retry_after=DEFAULT_RETRY)


class _Call:
    """One request in flight: finished exactly once (answer, error or cancel)."""

    def __init__(self) -> None:
        self.token = CancelToken()
        self.done = threading.Event()
        self.result: GenerationResult | None = None
        self.error: BaseException | None = None
        self._lock = threading.Lock()

    def finish(self, result: GenerationResult | None = None, error: BaseException | None = None) -> None:
        with self._lock:
            if self.done.is_set():
                return
            self.result, self.error = result, error
            self.done.set()

    def cancel(self) -> None:
        self.token.cancel()
        self.finish(error=CancelledError("cancelled"))


class ProviderFrameAnalyzer:
    """A :class:`~chibi.vision.service.FrameAnalyzer` for one AI configuration.

    The settings are copied at construction: a settings change builds a new
    analyzer (and :meth:`close` retires the old one), so a request never mixes
    two configurations.
    """

    def __init__(self, ai: VisionAI, post: Post, ai_settings: AISettings, vision_settings: VisionSettings, *,
                 allowed: Callable[[], bool] = lambda: True,
                 probe: Callable[[], bool] | EllipsisType | None = ...,
                 timeout: float | None = None) -> None:
        self._ai = ai
        self._post = post
        self._allowed = allowed
        self._ai_settings = copy.deepcopy(ai_settings)
        self._model = vision_settings.model.strip()
        self._local = not is_cloud(self._ai_settings)
        self._timeout = float(timeout) if timeout is not None else \
            self._ai_settings.request_timeout_seconds + EXTRA_WAIT_SECONDS
        self._probe: Callable[[], bool] | None = (
            default_vision_probe(self._ai_settings, self._model or self._ai_settings.model)
            if isinstance(probe, EllipsisType) else probe)
        self._sees_images: bool | None = None
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._current: _Call | None = None

    # -- FrameAnalyzer ---------------------------------------------------------------------

    @property
    def is_local(self) -> bool:
        """True when the model runs on this PC (no ``CLOUD_VISION`` needed)."""
        return self._local

    @property
    def timeout(self) -> float:
        """Seconds a request may take, queueing included, before it is abandoned."""
        return self._timeout

    @property
    def model_override(self) -> str:
        """The vision model actually requested ("" = the AI page's model)."""
        return self._model

    def analyze(self, image: bytes, media_type: str, app_title: str,
                previous: Observation | None) -> Observation:
        """Worker thread.  Raises :class:`AnalyzerError` or :class:`~chibi.vision.analysis.AnalysisError`."""
        self._check_open()
        self._ensure_vision()
        request = self.build_request(image, media_type, app_title, previous)
        result = self._run(request)
        return analysis.parse_observation(result.text)

    def build_request(self, image: bytes, media_type: str, app_title: str,
                      previous: Observation | None) -> GenerationRequest:
        turn = ChatTurn(Role.USER, analysis.user_prompt(app_title, previous),
                        images=(ImageInput(data=image, media_type=media_type),))
        return GenerationRequest(system=analysis.SYSTEM_PROMPT, turns=(turn,), purpose=Purpose.VISION,
                                 max_tokens=MAX_TOKENS, json_mode=True, model=self._model or None)

    # -- lifecycle (any thread) -----------------------------------------------------------------

    def cancel_current(self) -> None:
        """Abandon the request in flight (kill switch, permission revoked, vision stopped)."""
        with self._lock:
            call = self._current
        if call is not None:
            call.cancel()

    def close(self) -> None:
        """Retire this analyzer: the request in flight and every later call fail at once."""
        self._closed.set()
        self.cancel_current()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    # -- internals ----------------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed.is_set():
            raise AnalyzerError("analyzer closed", user_message=CANCELLED_USER_MESSAGE,
                                retry_after=DEFAULT_RETRY)

    def _ensure_vision(self) -> None:
        """Ask once whether the model accepts images (worker thread)."""
        if self._probe is None:
            return
        with self._lock:
            sees = self._sees_images
        if sees is None:
            try:
                sees = bool(self._probe())
            except ProviderError as exc:  # not reachable now: ask again next time
                raise to_analyzer_error(exc) from None
            except Exception as exc:
                log.warning("vision capability check failed (%s)", type(exc).__name__)
                raise AnalyzerError("capability check failed", user_message=FAILED_USER_MESSAGE,
                                    retry_after=DEFAULT_RETRY) from None
            with self._lock:
                self._sees_images = sees
            if not sees:
                log.info("the configured model cannot see images")
        self._check_open()
        if not sees:
            raise AnalyzerError(NO_VISION_MESSAGE, user_message=NO_VISION_USER_MESSAGE,
                                retry_after=NO_VISION_RETRY)

    def _run(self, request: GenerationRequest) -> GenerationResult:
        call = _Call()
        with self._lock:
            if self._closed.is_set():
                raise AnalyzerError("analyzer closed", user_message=CANCELLED_USER_MESSAGE,
                                    retry_after=DEFAULT_RETRY)
            self._current = call
        try:
            try:
                self._post(lambda: self._submit(call, request))
            except Exception as exc:  # the main-thread bridge is gone (shutdown)
                call.finish(error=exc)
            if not call.done.wait(self._timeout):
                call.token.cancel()  # the AI service drops it (queued) or aborts it (running)
                log.info("vision analysis timed out after %.0f s", self._timeout)
                raise AnalyzerError("vision analysis timed out", user_message=TIMEOUT_USER_MESSAGE,
                                    retry_after=DEFAULT_RETRY)
        finally:
            with self._lock:
                if self._current is call:
                    self._current = None
        if call.error is not None:
            raise to_analyzer_error(call.error) from None
        if call.result is None:  # pragma: no cover - finish() always sets one of the two
            raise AnalyzerError("no answer", user_message=FAILED_USER_MESSAGE, retry_after=DEFAULT_RETRY)
        return call.result

    def _submit(self, call: _Call, request: GenerationRequest) -> None:
        """Main thread: hand the request to the AI service, unless it is too late."""
        if call.done.is_set() or call.token.cancelled:
            return
        try:
            still_allowed = bool(self._allowed())
        except Exception:
            log.exception("vision permission check failed")
            still_allowed = False
        if self._closed.is_set() or not still_allowed:
            call.cancel()
            return
        try:
            self._ai.submit(request, on_done=lambda result: call.finish(result=result),
                            on_error=lambda exc: call.finish(error=exc), cancel=call.token)
        except Exception as exc:
            log.warning("submitting a vision request failed (%s)", type(exc).__name__)
            call.finish(error=exc)
