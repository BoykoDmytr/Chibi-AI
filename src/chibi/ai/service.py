"""AIService - the only place that runs model calls.

Threading model:

* Every public method is called on the Qt main thread (it reads SQLite through
  the ledger and the settings of the app).
* Model calls run on **one** daemon worker thread that consumes a small
  priority queue: lowest :attr:`Purpose.rank` first, FIFO within a rank.  The
  queue holds at most :data:`MAX_QUEUE` waiting requests; when it overflows the
  lowest-priority, oldest request is dropped with :class:`QueueFullError` (a
  stale reaction is worth less than a fresh one).
* Every callback (``on_delta``, ``on_done``, ``on_error``) is marshalled back to
  the main thread through a queued ``Signal(object)`` carrying a callable, so
  callers never need locks - and are never called re-entrantly from
  :meth:`submit`.  Each submit gets exactly one of ``on_done`` / ``on_error``
  (cancellation -> ``on_error(CancelledError)``); after :meth:`shutdown` no
  callback runs at all.

Privacy: ``Permission.CLOUD_AI`` is re-checked on the worker right before
every attempt (a model warm-up too); revoking it drops queued cloud requests
and warm-ups at once and aborts the running one.  A request that carries what
she saw on the screen (a frame, or ``GenerationRequest.screen_data``: a
description in an observer decision, a chat turn or a line) runs only on the
provider that was current when it was submitted - its sender checked that one
- and goes to a cloud provider only while ``Permission.CLOUD_VISION`` is
granted and "Лише локальний аналіз" is off (re-checked on the worker;
revoking cloud vision drops such requests at once).  Nothing from a request or
a reply is logged.

Resilience: retryable errors are retried twice (1 s, then 3 s; a server's
``Retry-After`` is honoured up to 10 s) unless cancelled or already streamed
to the user; three consecutive "unavailable" failures open a circuit breaker
that keeps her offline for 120 s (then one failure re-opens it, one success
closes it).  Identical non-chat requests without images are answered from a
10-minute LRU cache.  Spending is checked with :class:`BudgetGuard` before a
request is queued and recorded through :class:`UsageLedger` (main thread)
after it completes.

Conversation first.  One worker and (for a local model) one server mean a
chat answer would otherwise wait behind a screen analysis, an observer
decision, a reaction line or the memory extraction of the previous turn - on a
local model that is many seconds.  While the app reports
``ConversationActive(True)`` (the user is speaking, waiting for her answer or
hearing it, or a typed reply is in flight):

* requests of the background purposes (:data:`BACKGROUND_PURPOSES`: REACTION,
  INITIATIVE, VISION, MEMORY) are not started; they wait in the queue.  CHAT,
  the connection test, model listing and the warm-up run as usual, and so does
  a background request submitted with ``user_waiting=True`` (a line or a
  screen analysis the user asked for);
* a background request already running on a model server of the user's own
  (local or LAN, Ollama anywhere: one request at a time, free) when the
  conversation starts is interrupted through its token - the provider aborts
  the HTTP request, so the server is free at once.  REACTION and MEMORY go
  back into the queue and run after the conversation; VISION and INITIATIVE
  fail with :class:`ConversationBusyError` (a :class:`CancelledError`: their
  callers skip and try later).  An answer that arrived complete anyway is
  delivered, not sent again.  A request to a paid cloud API is let finish: the
  API answers the chat side by side, and an interrupted request would be
  billed and then sent - and billed - again;
* held requests do not starve the caller: a VISION or INITIATIVE request held
  longer than :data:`HOLD_LIMIT_SECONDS` fails the same way; MEMORY waits for
  the end of the conversation.  A REACTION that has not started within
  :data:`REACTION_DEADLINE_SECONDS` of its submission is stale whenever that
  happens (the composer has fallen back to the phrase bank by then) and fails
  with :class:`CancelledError`;
* when the conversation ends, held requests are checked against the budget
  again and then run in priority order.  The queue cap applies to held
  requests too.  An interrupted request is not recorded in the ledger; a
  resumed one is recorded once, when it completes.

Usage of requests that end early: a complete answer that arrives after its
caller withdrew it is recorded (and the caller gets :class:`CancelledError`); a
request to a paid API cut short after it was sent is recorded with an estimate
of its input (:data:`CUT_SHORT_CHARS_PER_TOKEN`), since the API bills it.

Timing: every generation request records its queue wait, the time to its first
streamed text and its total (from submission; numbers only, never text).  A
chat request logs them in one INFO line and is kept as
:attr:`AIService.last_chat_timing`.

Keeping the model ready: an Ollama provider sends ``keep_alive``
(``AISettings.keep_model_loaded_minutes``) with every request, and
:meth:`AIService.warm_up` asks it to load the model ahead of a conversation
when it is likely unloaded.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import threading
import time
import traceback
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

from chibi.ai.budget import BudgetGuard, UsageLedger, UsageSummary, estimate_cost
from chibi.ai.providers.factory import build_provider, env_key_available, is_cloud, needs_key, stored_key
from chibi.ai.providers.ollama_provider import OllamaProvider
from chibi.ai.providers.openai_provider import OpenAIProvider, is_private_network_url
from chibi.ai.secrets import SecretStore
from chibi.ai.types import (
    BudgetExceededError,
    CancelledError,
    CancelToken,
    ChatTurn,
    GenerationRequest,
    GenerationResult,
    ProviderError,
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderRequestError,
    ProviderUnavailableError,
    Purpose,
    Role,
    TextProvider,
    Usage,
)
from chibi.core.clock import Clock
from chibi.core.config import Settings
from chibi.core.event_bus import EventBus
from chibi.core.events import AIStatusChanged, ConversationActive, Permission, PermissionChanged
from chibi.privacy.permissions import PermissionManager

log = logging.getLogger(__name__)

MAX_QUEUE = 4
MAX_RETRIES = 2
BACKOFF_SECONDS = (1.0, 3.0)
MAX_RETRY_AFTER = 10.0
BREAKER_THRESHOLD = 3
BREAKER_COOLDOWN = 120.0
CACHE_TTL = 600.0
CACHE_SIZE = 64
SHUTDOWN_JOIN_SECONDS = 2.0

#: Purposes that wait while the user talks with her (see "Conversation first").
BACKGROUND_PURPOSES = frozenset({Purpose.REACTION, Purpose.INITIATIVE, Purpose.VISION, Purpose.MEMORY})
#: Background purposes that go back into the queue when a conversation interrupts them.
RESUMABLE_PURPOSES = frozenset({Purpose.REACTION, Purpose.MEMORY})
#: A screen analysis or observer decision held this long by a conversation is given up.
HOLD_LIMIT_SECONDS = 60.0
#: A reaction line not started this long after its submission is stale: the speech
#: composer has said a phrase-bank line instead (its AI timeout is 8 s, 5 s when solicited).
REACTION_DEADLINE_SECONDS = 8.0
#: How often held requests are looked at while a conversation lasts.
HOLD_CHECK_MS = 5000
#: A request to a paid API cut short after it was sent never reports its usage: it is
#: recorded with its input estimated at this many characters per token (Ukrainian text
#: takes ~2-4) and this many tokens per image (a downscaled 1024 px frame: ~1.5-2 k).
CUT_SHORT_CHARS_PER_TOKEN = 3
CUT_SHORT_IMAGE_TOKENS = 1500

REASON_OFF = "AI вимкнено"
REASON_PERMISSION = "потрібен дозвіл на хмарний AI"
REASON_NO_KEY = "немає API-ключа"
REASON_LIMIT = "ліміт на сьогодні вичерпано"
REASON_DOWN = "сервіс недоступний"

TEST_SYSTEM = "Це перевірка з'єднання. Відповідай одним словом."
TEST_PROMPT = "Скажи одне слово: привіт"
TEST_MAX_TOKENS = 200


class QueueFullError(ProviderUnavailableError):
    """Dropped because the queue overflowed (the provider was never asked)."""

    retryable = False

    def __init__(self, message: str = "AI queue full", *,
                 user_message: str = "AI зайнятий: черга переповнена.") -> None:
        super().__init__(message, user_message=user_message)


class AIOfflineError(ProviderError):
    """The service can't run requests now; ``user_message`` is the status reason."""


class ConversationBusyError(CancelledError):
    """A background request given up because the user is talking with her (interrupted,
    or held too long): nothing is wrong - skip it and try again later."""

    def __init__(self, message: str = "put aside for a conversation") -> None:
        super().__init__(message)
        self.user_message = "Відкладено: зараз триває розмова."


@dataclass(frozen=True)
class RequestTiming:
    """Where the time of one answer went, in seconds since it was asked for."""

    #: Waiting for the worker: other requests ahead of it (a model warm-up it waited
    #: for counts as ``model_load`` instead).
    queued: float
    #: Until the first streamed text (None: the answer was not streamed).
    first_text: float | None
    #: Until the whole answer.
    total: float
    #: The server loaded the model first - for this request, or for a warm-up it waited
    #: behind (Ollama reports it; None: unknown).
    model_load: float | None = None
    #: The server read the prompt (Ollama reports it; None: unknown).
    prompt_read: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def describe(self) -> str:
        """Numbers only, for the log: "queued 0.0 s, first text 2.3 s, done 5.1 s"."""
        parts = [f"queued {self.queued:.1f} s"]
        if self.first_text is not None:
            parts.append(f"first text {self.first_text:.1f} s")
        parts.append(f"done {self.total:.1f} s")
        if self.model_load is not None and self.model_load >= 0.05:
            parts.append(f"model load {self.model_load:.1f} s")
        if self.prompt_read is not None:
            parts.append(f"prompt read {self.prompt_read:.1f} s")
        parts.append(f"{self.input_tokens} in / {self.output_tokens} out tokens")
        return ", ".join(parts)


def user_message(exc: BaseException) -> str:
    """Short Ukrainian explanation of *exc* for the UI."""
    text = getattr(exc, "user_message", "")
    if isinstance(text, str) and text:
        return text
    if isinstance(exc, CancelledError):
        return "Запит скасовано."
    return "Помилка AI."


def _log_failure(what: str, exc: BaseException) -> None:
    """Log a failure without its message (messages can quote chat text); frames only."""
    frames = "".join(traceback.format_tb(exc.__traceback__)) if exc.__traceback__ else ""
    log.error("%s failed: %s\n%s", what, type(exc).__name__, frames.rstrip())


class _JobToken(CancelToken):
    """A job's own token: cancelled by the caller's token *or* by the service
    (shutdown, permission revoked) without mutating the caller's token - or merely
    *preempted*: put aside for a conversation, which stops the provider call too."""

    def __init__(self, parent: CancelToken) -> None:
        super().__init__()
        self._parent = parent
        self._preempted = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self.withdrawn or self._preempted.is_set()

    @property
    def withdrawn(self) -> bool:
        """Cancelled by the caller or the service (for good, not just put aside)."""
        return CancelToken.cancelled.fget(self) or self._parent.cancelled  # type: ignore[attr-defined]

    @property
    def preempted(self) -> bool:
        return self._preempted.is_set()

    def preempt(self) -> None:
        self._preempted.set()

    def renewed(self) -> _JobToken:
        """A fresh token for another run of the same job (same caller token)."""
        return _JobToken(self._parent)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise CancelledError("cancelled")


@dataclass(eq=False)
class _Job:
    seq: int
    kind: str  # "generate" | "models" | "warmup"
    rank: int
    request: GenerationRequest | None
    on_done: Callable[[Any], None]
    on_error: Callable[[Exception], None]
    on_delta: Callable[[str], None] | None
    token: _JobToken
    stream: bool = False
    cache_key: str | None = None
    bypass_breaker: bool = False
    #: Provider for this job only (model listing with an incomplete config).
    provider: TextProvider | None = None
    #: Delivered instead of CancelledError when the service aborted the job.
    abort_error: Exception | None = None
    #: A request with screen data (a frame, a description) runs only on the provider that
    #: was current when it was submitted: its sender checked that one, not one configured later.
    bound_provider: TextProvider | None = None
    #: The user asked for this (a solicited line, a manual screen analysis): it runs
    #: during a conversation although its purpose is a background one.
    user_waiting: bool = False
    #: Clock (monotonic) times: submitted, first started, first streamed text.
    submitted_at: float = 0.0
    started_at: float | None = None
    first_text_at: float | None = None
    #: Not started by then -> stale (reaction lines).
    deadline: float | None = None
    #: A chat answer: its timing becomes ``last_chat_timing``.
    chat_timing: bool = False
    #: The provider running it (set by the worker when it starts the job).
    ran_on: TextProvider | None = None

    @property
    def purpose(self) -> Purpose | None:
        return Purpose(self.request.purpose) if self.request is not None else None

    @property
    def background(self) -> bool:
        """Waits while the user talks with her (see "Conversation first")."""
        return (self.kind == "generate" and not self.user_waiting
                and self.purpose in BACKGROUND_PURPOSES)


class AIService(QObject):
    _invoke = Signal(object)

    def __init__(self, bus: EventBus, settings: Callable[[], Settings], permissions: PermissionManager,
                 secrets: SecretStore, ledger: UsageLedger, clock: Clock, *,
                 provider_factory: Callable[..., TextProvider | None] = build_provider,
                 parent: QObject | None = None,
                 sleep: Callable[[float], None] | None = None) -> None:
        super().__init__(parent)
        self._bus = bus
        self._settings = settings
        self._permissions = permissions
        self._secrets = secrets
        self._ledger = ledger
        self._clock = clock
        self._factory = provider_factory
        #: Test hook replacing the interruptible backoff wait.
        self._sleep = sleep
        self._budget = BudgetGuard(ledger, lambda: self._settings().ai)

        self._cond = threading.Condition()
        self._queue: list[_Job] = []
        self._running: _Job | None = None
        self._provider: TextProvider | None = None
        self._build_error: ProviderError | None = None
        self._has_key = False
        self._retired: list[TextProvider] = []
        self._failures = 0
        self._breaker_until = 0.0
        self._stopping = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = count()

        self._closed = False
        self._cache: OrderedDict[str, tuple[float, GenerationResult]] = OrderedDict()
        self._last_status: tuple[bool, str, str, str] | None = None
        #: The user is talking with her: background requests wait (guarded by _cond).
        self._conversation = False
        self._last_chat_timing: RequestTiming | None = None
        #: When a request to the provider's own model last completed (None: unknown or
        #: another model ran since, so it may have been unloaded).  Guarded by _cond.
        self._model_used_at: float | None = None
        self._warming = False
        #: Recent finished warm-ups: (started, ended, the load Ollama reported or None),
        #: clock times.  A chat that waited behind one waited for the model to load.
        self._warm_runs: deque[tuple[float, float, float | None]] = deque(maxlen=4)

        self._invoke.connect(self._run_on_main, Qt.ConnectionType.QueuedConnection)
        self._breaker_timer = QTimer(self)
        self._breaker_timer.setSingleShot(True)
        self._breaker_timer.timeout.connect(self._refresh_status)
        self._hold_timer = QTimer(self)
        self._hold_timer.setInterval(HOLD_CHECK_MS)
        self._hold_timer.timeout.connect(self._expire_held)
        unsubscribe_permissions = bus.subscribe(PermissionChanged, self._on_permission_changed)
        unsubscribe_conversation = bus.subscribe(ConversationActive, self._on_conversation)

        def unsubscribe() -> None:
            unsubscribe_permissions()
            unsubscribe_conversation()

        self._unsubscribe = unsubscribe
        self.reconfigure()

    # -- public API (main thread) ---------------------------------------------------------

    def status(self) -> tuple[bool, str]:
        """(online, Ukrainian reason when offline).  Publishes AIStatusChanged on change."""
        online, reason = self._compute_status()
        self._publish_status(online, reason)
        return online, reason

    def is_available(self, purpose: Purpose) -> bool:
        """Whether a request for *purpose* would be accepted right now (online, settings, budget)."""
        online, _ = self.status()
        if not online:
            return False
        ai = self._settings().ai
        if purpose == Purpose.CHAT and not ai.use_for_chat:
            return False
        if purpose in (Purpose.REACTION, Purpose.INITIATIVE) and not ai.use_for_reactions:
            return False
        try:
            self._budget.check(purpose)
        except BudgetExceededError:
            return False
        return True

    def submit(self, request: GenerationRequest, *, on_done: Callable[[GenerationResult], None],
               on_error: Callable[[Exception], None], on_delta: Callable[[str], None] | None = None,
               cancel: CancelToken | None = None, user_waiting: bool = False) -> CancelToken:
        """Queue *request*; exactly one of on_done/on_error follows on the main thread.

        *user_waiting*: the user asked for this one (a solicited line, a manual screen
        analysis), so it runs during a conversation like a chat answer."""
        caller_token = cancel if cancel is not None else CancelToken()
        if self._closed:
            caller_token.cancel()
            return caller_token
        purpose = Purpose(request.purpose)
        online, reason = self.status()
        if not online:
            error = AIOfflineError(f"AI offline ({purpose.value})", user_message=reason)
            self._post(lambda: _call(on_error, error))
            return caller_token
        with self._cond:
            provider = self._provider
        cache_key = self._cache_key(request, provider)
        if cache_key is not None:
            hit = self._cache_get(cache_key)
            if hit is not None:
                log.debug("AI %s request answered from the cache", purpose.value)
                self._post(lambda: self._deliver_cached(hit, caller_token, on_done, on_error, on_delta))
                return caller_token
        try:
            self._budget.check(purpose)
        except BudgetExceededError as exc:
            budget_error = exc
            log.info("AI %s request refused by the budget: %s", purpose.value, exc)
            self._post(lambda: _call(on_error, budget_error))
            self._refresh_status()
            return caller_token
        now = self._clock.monotonic()
        job = _Job(
            seq=next(self._seq), kind="generate", rank=purpose.rank, request=request,
            on_done=on_done, on_error=on_error, on_delta=on_delta, token=_JobToken(caller_token),
            stream=on_delta is not None and self._settings().ai.stream, cache_key=cache_key,
            bound_provider=provider if _screen_data(request) else None,
            user_waiting=bool(user_waiting), submitted_at=now,
            deadline=now + REACTION_DEADLINE_SECONDS if purpose == Purpose.REACTION else None,
            chat_timing=purpose == Purpose.CHAT,
        )
        self._enqueue(job)
        return caller_token

    def reconfigure(self) -> None:
        """Rebuild the provider after settings or an API key changed."""
        if self._closed:
            return
        ai = self._settings().ai
        provider: TextProvider | None = None
        error: ProviderError | None = None
        if ai.provider != "none":
            try:
                provider = self._factory(ai, self._secrets)
            except ProviderError as exc:
                error = exc
                log.info("AI provider %s not usable: %s", ai.provider, type(exc).__name__)
            except Exception as exc:
                _log_failure(f"building the {ai.provider} provider", exc)
                error = ProviderRequestError("provider build failed",
                                             user_message="Не вдалося налаштувати AI.")
        if isinstance(provider, OllamaProvider):
            provider.keep_alive = f"{ai.keep_model_loaded_minutes}m"
        has_key = stored_key(self._secrets, ai.provider) is not None
        close_now: TextProvider | None = None
        with self._cond:
            old = self._provider
            self._provider = provider
            self._build_error = error
            self._has_key = has_key
            # A new configuration deserves a fresh chance.
            self._failures = 0
            self._breaker_until = 0.0
            if old is not provider:
                self._model_used_at = None
            if old is not None and old is not provider:
                if self._running is not None:
                    self._retired.append(old)
                else:
                    close_now = old
        if close_now is not None:
            _close(close_now)
        log.info("AI configured: provider %s", ai.provider)
        self._refresh_status()

    def test(self, on_done: Callable[[bool, str], None]) -> None:
        """Send a tiny request; on_done(ok, Ukrainian message) on the main thread."""
        if self._closed:
            return
        online, reason = self.status()
        # The breaker is no reason not to try when the user asks explicitly.
        if not online and reason != REASON_DOWN:
            self._post(lambda: _call(on_done, False, reason))
            return
        try:
            self._budget.check(Purpose.CHAT)
        except BudgetExceededError as exc:
            message = user_message(exc)
            self._post(lambda: _call(on_done, False, message))
            return
        request = GenerationRequest(system=TEST_SYSTEM, turns=(ChatTurn(Role.USER, TEST_PROMPT),),
                                    purpose=Purpose.CHAT, max_tokens=TEST_MAX_TOKENS)

        def done(result: GenerationResult) -> None:
            text = " ".join(result.text.split())
            if not text:
                on_done(False, "Модель надіслала порожню відповідь.")
                return
            snippet = text if len(text) <= 40 else text[:39] + "…"
            on_done(True, f"Працює! {result.model} відповідає: «{snippet}»")

        job = _Job(seq=next(self._seq), kind="generate", rank=Purpose.CHAT.rank, request=request,
                   on_done=done, on_error=lambda exc: on_done(False, user_message(exc)), on_delta=None,
                   token=_JobToken(CancelToken()), bypass_breaker=True,
                   submitted_at=self._clock.monotonic())
        self._enqueue(job)

    def warm_up(self) -> bool:
        """Ask a local model server to load the chat model now, so the answer to what
        the user is saying does not wait for it (call it when the user starts talking to
        her or opens the chat - not at start-up: a model nobody uses would sit in video
        memory for the keep-alive time).  Queued at chat priority, and only when the provider can
        (Ollama), a chat request would be accepted (online, chat AI on, budget, cloud
        permission), the model is likely unloaded (nothing ran on it for
        ``keep_model_loaded_minutes`` - 1 minutes, or never, or another model ran since)
        and no chat request or warm-up is already queued or running.  True when queued."""
        if self._closed or self._warming:
            return False
        with self._cond:
            provider = self._provider
            used_at = self._model_used_at
            busy = any(job.kind == "warmup" or job.purpose == Purpose.CHAT
                       for job in (*self._queue, self._running) if job is not None)
        if provider is None or busy or not callable(getattr(provider, "warm_up", None)):
            return False
        minutes = self._settings().ai.keep_model_loaded_minutes
        if used_at is not None and self._clock.monotonic() - used_at < max(0.0, (minutes - 1) * 60.0):
            return False
        if not self.is_available(Purpose.CHAT):
            return False

        def finished(_result: Any = None) -> None:
            self._warming = False

        def failed(exc: Exception) -> None:
            self._warming = False
            if not isinstance(exc, CancelledError):
                log.info("AI model warm-up failed (%s)", type(exc).__name__)

        job = _Job(seq=next(self._seq), kind="warmup", rank=Purpose.CHAT.rank, request=None,
                   on_done=finished, on_error=failed, on_delta=None, token=_JobToken(CancelToken()),
                   submitted_at=self._clock.monotonic(), bound_provider=provider)
        self._warming = True
        self._enqueue(job)
        return True

    def list_models(self, on_done: Callable[[list[str], str], None]) -> None:
        """Ask the provider which models exist; on_done(models, error message or "").

        Listing sends no user data (only the API key), so it works before the
        cloud permission is granted - the settings dialog needs it to pick one.
        """
        if self._closed:
            return
        ai = self._settings().ai
        with self._cond:
            provider = self._provider
            error = self._build_error
        own: TextProvider | None = None
        if provider is None and error is not None and ai.provider != "none":
            # E.g. an OpenAI-compatible server without a model yet: listing
            # doesn't need one, so build a throwaway provider just for that.
            try:
                own = self._factory(dataclasses.replace(ai, model=ai.model or "list-models"), self._secrets)
            except Exception as exc:
                log.info("cannot build a provider to list models (%s)", type(exc).__name__)
        if provider is None and own is None:
            message = user_message(error) if error is not None else REASON_OFF
            self._post(lambda: _call(on_done, [], message))
            return
        job = _Job(seq=next(self._seq), kind="models", rank=Purpose.CHAT.rank, request=None,
                   on_done=lambda models: on_done(models, ""),
                   on_error=lambda exc: on_done([], user_message(exc)), on_delta=None,
                   token=_JobToken(CancelToken()), provider=own)
        self._enqueue(job)

    def usage_today(self) -> UsageSummary:
        return self._ledger.today()

    def pending(self) -> int:
        """Requests waiting in the queue (not counting the running one), held ones too."""
        with self._cond:
            return len(self._queue)

    @property
    def conversation_active(self) -> bool:
        """The user is talking with her (the last ``ConversationActive``)."""
        with self._cond:
            return self._conversation

    @property
    def own_model_server(self) -> bool:
        """The current provider is a model server of the user's own (local, LAN, Ollama):
        free, one request at a time, and it reuses the prompt prefix it has already read."""
        with self._cond:
            return _own_server(self._provider)

    @property
    def last_chat_timing(self) -> RequestTiming | None:
        """Where the time of the last chat answer went (None before the first one)."""
        return self._last_chat_timing

    def shutdown(self) -> None:
        """Cancel everything and join the worker (at most ~2 s).  No callback runs afterwards."""
        if self._closed:
            return
        self._closed = True
        self._unsubscribe()
        self._breaker_timer.stop()
        self._hold_timer.stop()
        with self._cond:
            self._stopping = True
            for job in self._queue:
                job.token.cancel()
            own = [job.provider for job in self._queue if job.provider is not None]
            self._queue.clear()
            if self._running is not None:
                self._running.token.cancel()
            self._stop_event.set()
            self._cond.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(SHUTDOWN_JOIN_SECONDS)
            if thread.is_alive():
                log.warning("AI worker still busy after %.0f s; leaving it (daemon thread)",
                            SHUTDOWN_JOIN_SECONDS)
        with self._cond:
            providers = [p for p in (self._provider, *self._retired, *own) if p is not None]
            self._retired = []
        for provider in providers:
            _close(provider)
        log.info("AI service stopped")

    # -- status ---------------------------------------------------------------------------

    def _compute_status(self) -> tuple[bool, str]:
        ai = self._settings().ai
        if ai.provider == "none":
            return False, REASON_OFF
        with self._cond:
            provider = self._provider
            error = self._build_error
            has_key = self._has_key
        if error is not None:
            return False, user_message(error)
        if provider is None:
            return False, REASON_OFF
        if is_cloud(ai) and not self._permissions.is_granted(Permission.CLOUD_AI):
            return False, REASON_PERMISSION
        if needs_key(ai) and not has_key and not env_key_available(ai):
            return False, REASON_NO_KEY
        if self._budget.daily_exhausted():
            return False, REASON_LIMIT
        if self._breaker_open():
            return False, REASON_DOWN
        return True, ""

    def _publish_status(self, online: bool, reason: str) -> None:
        if self._closed:
            return
        ai = self._settings().ai
        with self._cond:
            provider = self._provider
        model = (provider.model if provider is not None else ai.model) or ai.model
        state = (online, reason, ai.provider, model)
        if state == self._last_status:
            return
        self._last_status = state
        log.info("AI status: %s%s", "online" if online else "offline", f" ({reason})" if reason else "")
        self._bus.publish(AIStatusChanged(online=online, provider=ai.provider, model=model, reason=reason))

    @Slot()
    def _refresh_status(self) -> None:
        if not self._closed:
            self.status()

    def _breaker_open(self) -> bool:
        with self._cond:
            return self._clock.monotonic() < self._breaker_until

    def _on_permission_changed(self, event: PermissionChanged) -> None:
        if self._closed or event.granted:
            if not self._closed and event.permission == Permission.CLOUD_AI:
                self._refresh_status()
            return
        if event.permission == Permission.CLOUD_AI:
            self._abort_cloud_jobs()
            self._refresh_status()
        elif event.permission == Permission.CLOUD_VISION:
            self._abort_cloud_jobs(screen_only=True)

    def _abort_cloud_jobs(self, *, screen_only: bool = False) -> None:
        """CLOUD_AI was revoked: nothing more may leave the PC (*screen_only*: CLOUD_VISION
        was revoked - nothing taken from the screen may)."""
        dropped: list[_Job] = []

        def affected(job: _Job) -> bool:
            if job.kind == "warmup":  # tells the server the user is active right now
                return not screen_only
            return (job.kind == "generate" and job.provider is None
                    and (not screen_only or (job.request is not None and _screen_data(job.request))))

        with self._cond:
            provider = self._provider
            if provider is None or provider.is_local:
                return
            keep: list[_Job] = []
            for job in self._queue:
                (dropped if affected(job) else keep).append(job)
            self._queue[:] = keep
            running = self._running
            if running is not None and affected(running):
                running.abort_error = (CancelledError("cloud vision revoked") if screen_only else
                                       AIOfflineError("cloud permission revoked",
                                                      user_message=REASON_PERMISSION))
                running.token.cancel()
        if dropped:
            log.info("cloud %s permission revoked: dropped %d queued request(s)",
                     "vision" if screen_only else "AI", len(dropped))
        for job in dropped:
            error: Exception = (CancelledError("cloud vision revoked") if screen_only else
                                AIOfflineError("cloud permission revoked", user_message=REASON_PERMISSION))
            self._post(lambda job=job, error=error: _call(job.on_error, error))

    # -- conversation ---------------------------------------------------------------------

    def _on_conversation(self, event: ConversationActive) -> None:
        """Main thread: the user started or stopped talking with her."""
        if self._closed:
            return
        active = bool(event.active)
        if active:
            preempted: _Job | None = None
            with self._cond:
                if self._conversation:
                    return
                self._conversation = True
                running = self._running
                if running is not None and running.background and not running.token.withdrawn \
                        and _own_server(running.ran_on):
                    running.token.preempt()  # the provider aborts; the worker puts it aside
                    preempted = running
            self._hold_timer.start()
            if preempted is not None:
                log.info("conversation: interrupted a running %s request", _purpose_name(preempted))
            self._expire_held()
            return
        with self._cond:
            if not self._conversation:
                return
            held = [job for job in self._queue if job.background]
        self._hold_timer.stop()
        # Held requests waited past the budget check they had at submission: check again
        # (the conversation itself may have used up the hour) before they can start.
        refused: list[tuple[_Job, BudgetExceededError]] = []
        for job in held:
            purpose = job.purpose
            if purpose is None:
                continue
            try:
                self._budget.check(purpose)
            except BudgetExceededError as exc:
                refused.append((job, exc))
        with self._cond:
            failed = [(job, exc) for job, exc in refused if job in self._queue]
            for job, _exc in failed:
                self._queue.remove(job)
            self._conversation = False
            self._cond.notify_all()
        for job, exc in failed:
            log.info("AI %s request held for a conversation refused by the budget", _purpose_name(job))
            self._post(lambda job=job, exc=exc: _call(job.on_error, exc))

    @Slot()
    def _expire_held(self) -> None:
        """Main thread, while a conversation lasts: give up held requests that would only
        go stale (see "Conversation first"); drop the ones their callers cancelled."""
        if self._closed:
            return
        now = self._clock.monotonic()
        expired: list[tuple[_Job, Exception]] = []
        with self._cond:
            if not self._conversation:
                return
            for job in list(self._queue):
                if not job.background:
                    continue
                error: Exception | None = None
                if job.token.withdrawn:
                    error = CancelledError("cancelled")
                elif job.deadline is not None and now > job.deadline:
                    error = CancelledError("stale")
                elif job.purpose not in RESUMABLE_PURPOSES and now - job.submitted_at > HOLD_LIMIT_SECONDS:
                    error = ConversationBusyError("held too long for a conversation")
                if error is not None:
                    self._queue.remove(job)
                    expired.append((job, error))
        for job, error in expired:
            if isinstance(error, ConversationBusyError):
                log.info("AI %s request held too long for a conversation: given up", _purpose_name(job))
            self._post(lambda job=job, error=error: _call(job.on_error, error))

    # -- queue ----------------------------------------------------------------------------

    def _enqueue(self, job: _Job) -> None:
        cancelled: list[_Job] = []
        with self._cond:
            # Requests cancelled while waiting must not hold a slot.
            for queued in list(self._queue):
                if queued.token.withdrawn:
                    self._queue.remove(queued)
                    cancelled.append(queued)
            victim = self._admit(job)
            self._cond.notify()
            if self._thread is None:
                self._thread = threading.Thread(target=self._worker_main, name="chibi-ai", daemon=True)
                self._thread.start()
        for job_ in cancelled:
            _close_own(job_)
            self._post(lambda j=job_: _call(j.on_error, CancelledError("cancelled")))
        self._drop(victim)

    def _admit(self, job: _Job) -> _Job | None:
        """Under ``_cond``: put *job* into the queue; returns the request the cap pushed out
        (perhaps *job* itself), if any."""
        if len(self._queue) < MAX_QUEUE:
            self._queue.append(job)
            return None
        # Lowest priority (highest rank) loses; among equals the oldest.
        victim = max([*self._queue, job], key=lambda j: (j.rank, -j.seq))
        if victim is not job:
            self._queue.remove(victim)
            self._queue.append(job)
        return victim

    def _drop(self, victim: _Job | None) -> None:
        if victim is None:
            return
        _close_own(victim)
        log.info("AI queue full: dropped a %s request", _purpose_name(victim))
        self._post(lambda: _call(victim.on_error, QueueFullError()))

    def _next_job(self) -> _Job | None:
        """Under ``_cond``: the job to run next - lowest rank, then FIFO - skipping
        background requests while a conversation lasts."""
        best: _Job | None = None
        for job in self._queue:
            if self._conversation and job.background:
                continue
            if best is None or (job.rank, job.seq) < (best.rank, best.seq):
                best = job
        return best

    # -- worker thread --------------------------------------------------------------------

    def _worker_main(self) -> None:
        while True:
            with self._cond:
                job = None
                while not self._stopping:
                    job = self._next_job()
                    if job is not None:
                        break
                    self._cond.wait()
                if self._stopping or job is None:
                    return
                self._queue.remove(job)
                self._running = job
                provider = job.provider if job.provider is not None else self._provider
                job.ran_on = provider
            try:
                if job.deadline is not None and self._clock.monotonic() > job.deadline:
                    self._fail(job, CancelledError("stale"))  # the caller has moved on
                elif job.kind == "models":
                    self._run_models(job, provider)
                elif job.kind == "warmup":
                    self._run_warmup(job, provider)
                else:
                    self._run_generate(job, provider)
            except Exception as exc:  # a bug here must not kill the worker or lose a callback
                _log_failure("AI worker", exc)
                self._fail(job, ProviderUnavailableError("internal error", user_message="Помилка AI."))
            finally:
                with self._cond:
                    self._running = None
                    retired, self._retired = self._retired, []
                for old in retired:
                    _close(old)
                if job.provider is not None:
                    _close(job.provider)

    def _run_models(self, job: _Job, provider: TextProvider | None) -> None:
        if provider is None:
            self._fail(job, AIOfflineError("AI off", user_message=REASON_OFF))
            return
        if job.token.cancelled or self._stopping:
            self._fail(job, CancelledError("cancelled"))
            return
        try:
            models = provider.list_models()
        except ProviderError as exc:
            self._fail(job, exc)
            return
        except Exception as exc:
            _log_failure("listing models", exc)
            self._fail(job, ProviderUnavailableError("list_models crashed", user_message="Помилка AI."))
            return
        self._post(lambda: None if self._job_cancelled(job) else _call(job.on_done, list(models)))

    def _run_generate(self, job: _Job, provider: TextProvider | None) -> None:
        assert job.request is not None
        if provider is None:
            with self._cond:
                error = self._build_error
            reason = user_message(error) if error is not None else REASON_OFF
            self._fail(job, AIOfflineError("AI off", user_message=reason))
            return
        if job.bound_provider is not None and provider is not job.bound_provider:
            log.info("AI provider changed while a request with screen data waited; it was dropped")
            self._fail(job, CancelledError("provider changed"))
            return
        attempt = 0
        while True:
            if job.token.cancelled or self._stopping:
                self._fail(job, CancelledError("cancelled"))
                return
            if not provider.is_local and not self._permissions.is_granted(Permission.CLOUD_AI):
                log.info("cloud AI permission missing: request dropped before sending")
                self._fail(job, AIOfflineError("cloud permission missing", user_message=REASON_PERMISSION))
                return
            if not provider.is_local and _screen_data(job.request) and not self._screen_may_leave():
                log.info("screen data may not go to the cloud: request dropped before sending")
                self._fail(job, CancelledError("cloud vision not allowed"))
                return
            if not job.bypass_breaker and self._breaker_open():
                self._fail(job, AIOfflineError("circuit open", user_message=REASON_DOWN))
                return
            streamed = False

            def on_piece(piece: str) -> None:
                nonlocal streamed
                streamed = True
                if job.first_text_at is None:
                    job.first_text_at = self._clock.monotonic()
                self._post(lambda: None if self._job_cancelled(job) or job.on_delta is None
                           else _call(job.on_delta, piece))

            if job.started_at is None:
                job.started_at = self._clock.monotonic()
            try:
                if job.stream:
                    result = provider.stream(job.request, on_piece, job.token)
                else:
                    result = provider.generate(job.request, job.token)
            except CancelledError as exc:
                self._record_cut_short(job, provider)
                self._fail(job, exc)
                return
            except ProviderError as exc:
                if job.token.cancelled or self._stopping:
                    self._record_cut_short(job, provider)
                    self._fail(job, CancelledError("cancelled"))
                    return
                delay = self._retry_delay(exc, attempt)
                # A half-streamed answer can't be retried without repeating itself.
                if delay is not None and not streamed:
                    attempt += 1
                    log.info("AI %s request failed (%s); retry %d/%d in %.1f s", job.request.purpose.value,
                             type(exc).__name__, attempt, MAX_RETRIES, delay)
                    if not self._wait(delay, job.token):
                        self._fail(job, CancelledError("cancelled"))
                        return
                    continue
                self._fail(job, exc, provider=provider)
                return
            except Exception as exc:
                if job.token.cancelled or self._stopping:
                    self._record_cut_short(job, provider)
                    self._fail(job, CancelledError("cancelled"))  # the abort's own doing
                    return
                _log_failure(f"{provider.name} provider", exc)
                self._fail(job, ProviderUnavailableError("provider crashed", user_message="Помилка AI."),
                           provider=provider)
                return
            timing = self._timing(job, result, provider)
            log.info("AI %s: %s", job.request.purpose.value, timing.describe())
            with self._cond:
                if provider is self._provider:
                    # Another model (e.g. one for screen vision) may have pushed this one out.
                    self._model_used_at = self._clock.monotonic() if result.model == provider.model else None
            self._succeed(job, result, timing)
            return

    def _record_cut_short(self, job: _Job, provider: TextProvider) -> None:
        """Worker thread: *job* was cut short after it was sent.  A paid API has received
        (and bills) it although its usage never arrived: it is recorded with an estimate
        of its input, so the daily and hourly limits (vision's share too) see it.  A
        server of the user's own is free and stops at once: nothing to record."""
        request = job.request
        if request is None or _own_server(provider):
            return
        usage = Usage(input_tokens=_estimated_input_tokens(request))
        model = (request.model or "").strip() or provider.model
        name = provider.name
        log.info("AI %s request cut short after sending: counted with its estimated input",
                 request.purpose.value)
        self._post(lambda: self._record(name, model, request.purpose, usage))

    def _run_warmup(self, job: _Job, provider: TextProvider | None) -> None:
        warm_up = getattr(provider, "warm_up", None)
        if provider is None or provider is not job.bound_provider or not callable(warm_up):
            self._fail(job, CancelledError("provider changed"))
            return
        if job.token.cancelled or self._stopping:
            self._fail(job, CancelledError("cancelled"))
            return
        if not provider.is_local and not self._permissions.is_granted(Permission.CLOUD_AI):
            # Ollama on another machine: nothing goes there without the cloud permission.
            self._fail(job, AIOfflineError("cloud permission missing", user_message=REASON_PERMISSION))
            return
        started = job.started_at = self._clock.monotonic()
        try:
            warm_up(job.token)
        except (CancelledError, ProviderError) as exc:
            # Not counted against the circuit breaker: nobody asked for this one.
            self._fail(job, exc)
            return
        except Exception as exc:
            if job.token.cancelled or self._stopping:
                self._fail(job, CancelledError("cancelled"))
                return
            _log_failure("model warm-up", exc)
            self._fail(job, ProviderUnavailableError("warm-up crashed", user_message="Помилка AI."))
            return
        now = self._clock.monotonic()
        load = _seconds(getattr(provider, "last_load_seconds", None))
        with self._cond:
            if provider is self._provider:
                self._model_used_at = now
            self._warm_runs.append((started, now, load))
        log.info("AI model warm-up: done %.1f s%s", now - job.submitted_at,
                 f", model load {load:.1f} s" if load is not None else "")
        self._post(lambda: _call(job.on_done, None))

    def _timing(self, job: _Job, result: GenerationResult, provider: TextProvider) -> RequestTiming:
        """Worker thread, right after the provider call (its last_* numbers are this call's)."""
        now = self._clock.monotonic()
        start = job.started_at if job.started_at is not None else now

        def since(moment: float) -> float:
            return max(0.0, moment - job.submitted_at)

        # A warm-up that ran while this request waited was the model loading for it (a
        # conversation starting queues one just before the chat): not a queue.
        with self._cond:
            warm_runs = list(self._warm_runs)
        loading = 0.0
        for warm_start, warm_end, warm_load in warm_runs:
            overlap = max(0.0, min(warm_end, start) - max(warm_start, job.submitted_at))
            loading += min(overlap, warm_load) if warm_load is not None else overlap
        own_load = _seconds(getattr(provider, "last_load_seconds", None))
        model_load = own_load if loading <= 0.0 else (own_load or 0.0) + loading
        usage = result.usage
        return RequestTiming(
            queued=max(0.0, since(start) - loading),
            first_text=since(job.first_text_at) if job.first_text_at is not None else None,
            total=since(now),
            model_load=model_load,
            prompt_read=_seconds(getattr(provider, "last_prompt_seconds", None)),
            input_tokens=usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens,
            output_tokens=usage.output_tokens,
        )

    def _screen_may_leave(self) -> bool:
        """Worker thread: what she saw may go to a cloud provider (``CLOUD_VISION`` and not
        "Лише локальний аналіз").  Fails closed."""
        if not self._permissions.is_granted(Permission.CLOUD_VISION):
            return False
        try:
            return not self._settings().vision.local_only
        except Exception:
            return False

    @staticmethod
    def _retry_delay(exc: ProviderError, attempt: int) -> float | None:
        if not exc.retryable or attempt >= MAX_RETRIES:
            return None
        delay = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
        if isinstance(exc, ProviderRateLimitError) and exc.retry_after is not None:
            if exc.retry_after > MAX_RETRY_AFTER:
                return None  # not worth holding the queue; the caller falls back offline
            delay = max(delay, exc.retry_after)
        return delay

    def _wait(self, seconds: float, token: CancelToken) -> bool:
        """Interruptible backoff; False if cancelled or shutting down meanwhile."""
        if self._sleep is not None:
            self._sleep(seconds)
            return not (token.cancelled or self._stopping)
        deadline = time.monotonic() + seconds
        while not (token.cancelled or self._stopping):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            self._stop_event.wait(min(remaining, 0.05))
        return False

    # -- completion (posted to the main thread) ---------------------------------------------

    def _fail(self, job: _Job, exc: Exception, *, provider: TextProvider | None = None) -> None:
        if isinstance(exc, CancelledError) and self._put_aside(job):
            return
        if isinstance(exc, CancelledError) and job.abort_error is not None:
            exc = job.abort_error
        if job.chat_timing:
            now = self._clock.monotonic()
            start = job.started_at if job.started_at is not None else now
            log.info("AI chat failed (%s): queued %.1f s, gave up after %.1f s", type(exc).__name__,
                     max(0.0, start - job.submitted_at), max(0.0, now - job.submitted_at))

        def deliver() -> None:
            if provider is not None:
                self._note_failure(job, provider, exc)
            _call(job.on_error, exc)

        self._post(deliver)

    def _put_aside(self, job: _Job) -> bool:
        """Worker thread: *job* stopped because a conversation preempted it (and nothing
        else cancelled it).  REACTION / MEMORY go back into the queue, the rest fail with
        :class:`ConversationBusyError`.  False when *job* was not preempted."""
        token = job.token
        if job.kind != "generate" or not token.preempted or token.withdrawn:
            return False
        victim: _Job | None = None
        with self._cond:
            if self._stopping:
                return False
            if self._running is job:
                self._running = None
            # A half-streamed answer can't run again without repeating itself.
            resume = job.purpose in RESUMABLE_PURPOSES and job.first_text_at is None
            if resume:
                job.token = token.renewed()
                job.started_at = None
                victim = self._admit(job)
                self._cond.notify()
        if resume:
            log.info("AI %s request put aside until the conversation ends", _purpose_name(job))
            self._drop(victim)
            return True
        error = ConversationBusyError()
        self._post(lambda: _call(job.on_error, error))
        return True

    def _succeed(self, job: _Job, result: GenerationResult, timing: RequestTiming | None = None) -> None:
        def deliver() -> None:
            assert job.request is not None
            self._record(result.provider, result.model, job.request.purpose, result.usage)
            with self._cond:
                self._failures = 0
                self._breaker_until = 0.0
            if job.cache_key is not None and result.text:
                self._cache_put(job.cache_key, result)
            self._refresh_status()
            if self._job_cancelled(job):
                _call(job.on_error, job.abort_error or CancelledError("cancelled"))
                return
            if job.chat_timing and timing is not None:
                self._last_chat_timing = timing
            if job.on_delta is not None and not job.stream and result.text:
                _call(job.on_delta, result.text)
            _call(job.on_done, result)

        self._post(deliver)

    def _note_failure(self, job: _Job, provider: TextProvider, exc: Exception) -> None:
        """Main thread: breaker accounting and usage of billed failures."""
        tripped = False
        with self._cond:
            if isinstance(exc, ProviderUnavailableError):
                self._failures += 1
                if self._failures >= BREAKER_THRESHOLD:
                    self._breaker_until = self._clock.monotonic() + BREAKER_COOLDOWN
                    # Half-open afterwards: the next failure re-opens it at once.
                    self._failures = BREAKER_THRESHOLD - 1
                    tripped = True
            elif not isinstance(exc, ProviderRateLimitError):
                self._failures = 0  # the service answered; it is reachable
        if tripped:
            log.warning("AI provider unavailable %d times in a row: offline for %.0f s",
                        BREAKER_THRESHOLD, BREAKER_COOLDOWN)
            self._breaker_timer.start(int(BREAKER_COOLDOWN * 1000) + 100)
        if isinstance(exc, ProviderRefusalError) and job.request is not None:
            # A refusal is still a request against the limits.
            self._record(provider.name, provider.model, job.request.purpose, Usage())
        self._refresh_status()

    def _record(self, provider: str, model: str, purpose: Purpose, usage: Usage) -> None:
        try:
            cost = estimate_cost(provider, model, usage, self._settings().ai)
            self._ledger.record(provider, model, purpose, usage, cost)
        except Exception as exc:
            log.warning("recording AI usage failed (%s)", type(exc).__name__)

    def _deliver_cached(self, result: GenerationResult, token: CancelToken,
                        on_done: Callable[[GenerationResult], None], on_error: Callable[[Exception], None],
                        on_delta: Callable[[str], None] | None) -> None:
        if token.cancelled:
            _call(on_error, CancelledError("cancelled"))
            return
        if on_delta is not None and result.text:
            _call(on_delta, result.text)
        _call(on_done, dataclasses.replace(result, from_cache=True))

    def _job_cancelled(self, job: _Job) -> bool:
        # Withdrawn, not merely preempted: an answer that arrived anyway is delivered.
        return job.token.withdrawn

    # -- cache (main thread) ----------------------------------------------------------------

    @staticmethod
    def _cache_key(request: GenerationRequest, provider: TextProvider | None) -> str | None:
        if provider is None or request.purpose == Purpose.CHAT or any(t.images for t in request.turns):
            return None
        payload = json.dumps(
            [provider.name, provider.model, request.system, request.context,
             [[t.role.value, t.text] for t in request.turns], request.max_tokens, request.json_mode,
             request.effort, request.model],
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> GenerationResult | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires, result = entry
        if self._clock.monotonic() >= expires:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return result

    def _cache_put(self, key: str, result: GenerationResult) -> None:
        self._cache[key] = (self._clock.monotonic() + CACHE_TTL, result)
        self._cache.move_to_end(key)
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)

    # -- main-thread marshalling -----------------------------------------------------------------

    def _post(self, fn: Callable[[], None]) -> None:
        """Run *fn* on the main thread (from any thread, never synchronously)."""
        with contextlib.suppress(RuntimeError):  # the QObject is gone (app teardown)
            self._invoke.emit(fn)

    @Slot(object)
    def _run_on_main(self, fn: Callable[[], None]) -> None:
        if self._closed:
            return
        try:
            fn()
        except Exception as exc:
            _log_failure("AI completion", exc)


def _call(callback: Callable[..., Any], *args: Any) -> None:
    """Run a caller's callback; its failure is logged (without message) and contained."""
    try:
        callback(*args)
    except Exception as exc:
        _log_failure("AI callback", exc)


def _close(provider: TextProvider) -> None:
    close = getattr(provider, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception as exc:
        log.warning("closing the %s provider failed (%s)", getattr(provider, "name", "?"), type(exc).__name__)


def _close_own(job: _Job) -> None:
    """Close a job's throwaway provider (model listing) when the job never runs."""
    if job.provider is not None:
        _close(job.provider)


def _screen_data(request: GenerationRequest) -> bool:
    """The request carries something taken from the screen (a frame or a description)."""
    return request.screen_data or any(t.images for t in request.turns)


def _own_server(provider: TextProvider | None) -> bool:
    """*provider* is a model server of the user's own (on this PC or the LAN, and Ollama
    anywhere): it works on one request at a time, bills nothing, keeps the prompt it has
    read for the next request, and its client opens a fresh connection per request, so
    an abort reaches it - interrupting a request there frees what the chat answer needs
    and costs nothing.  A paid cloud API runs requests side by side and bills each one:
    an interrupted request would be paid for and then sent - and paid for - again, and
    its reused keep-alive connection often cannot be aborted at all.  Its request is let
    finish."""
    if provider is None:
        return False
    if isinstance(provider, OllamaProvider):
        return True
    if isinstance(provider, OpenAIProvider):
        return is_private_network_url(provider.base_url)
    return bool(provider.is_local)


def _estimated_input_tokens(request: GenerationRequest) -> int:
    """Rough input size of *request* (see :data:`CUT_SHORT_CHARS_PER_TOKEN`)."""
    chars = len(request.system) + len(request.context) + sum(len(turn.text) for turn in request.turns)
    images = sum(len(turn.images) for turn in request.turns)
    return chars // CUT_SHORT_CHARS_PER_TOKEN + images * CUT_SHORT_IMAGE_TOKENS


def _purpose_name(job: _Job) -> str:
    return job.request.purpose.value if job.request is not None else job.kind


def _seconds(value: Any) -> float | None:
    """A provider's reported duration (seconds), or None when it is not a sane number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value < 86400.0:
        return None
    return float(value)
