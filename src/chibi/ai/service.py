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
every attempt; revoking it drops queued cloud requests at once and aborts the
running one.  Nothing from a request or a reply is logged.

Resilience: retryable errors are retried twice (1 s, then 3 s; a server's
``Retry-After`` is honoured up to 10 s) unless cancelled or already streamed
to the user; three consecutive "unavailable" failures open a circuit breaker
that keeps her offline for 120 s (then one failure re-opens it, one success
closes it).  Identical non-chat requests without images are answered from a
10-minute LRU cache.  Spending is checked with :class:`BudgetGuard` before a
request is queued and recorded through :class:`UsageLedger` (main thread)
after it completes.
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
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

from chibi.ai.budget import BudgetGuard, UsageLedger, UsageSummary, estimate_cost
from chibi.ai.providers.factory import build_provider, env_key_available, is_cloud, needs_key, stored_key
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
from chibi.core.events import AIStatusChanged, Permission, PermissionChanged
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
    (shutdown, permission revoked) without mutating the caller's token."""

    def __init__(self, parent: CancelToken) -> None:
        super().__init__()
        self._parent = parent

    @property
    def cancelled(self) -> bool:
        return CancelToken.cancelled.fget(self) or self._parent.cancelled  # type: ignore[attr-defined]

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise CancelledError("cancelled")


@dataclass(eq=False)
class _Job:
    seq: int
    kind: str  # "generate" | "models"
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

        self._invoke.connect(self._run_on_main, Qt.ConnectionType.QueuedConnection)
        self._breaker_timer = QTimer(self)
        self._breaker_timer.setSingleShot(True)
        self._breaker_timer.timeout.connect(self._refresh_status)
        self._unsubscribe = bus.subscribe(PermissionChanged, self._on_permission_changed)
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
               cancel: CancelToken | None = None) -> CancelToken:
        """Queue *request*; exactly one of on_done/on_error follows on the main thread."""
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
        job = _Job(
            seq=next(self._seq), kind="generate", rank=purpose.rank, request=request,
            on_done=on_done, on_error=on_error, on_delta=on_delta, token=_JobToken(caller_token),
            stream=on_delta is not None and self._settings().ai.stream, cache_key=cache_key,
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
                   token=_JobToken(CancelToken()), bypass_breaker=True)
        self._enqueue(job)

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
        """Requests waiting in the queue (not counting the running one)."""
        with self._cond:
            return len(self._queue)

    def shutdown(self) -> None:
        """Cancel everything and join the worker (at most ~2 s).  No callback runs afterwards."""
        if self._closed:
            return
        self._closed = True
        self._unsubscribe()
        self._breaker_timer.stop()
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
        if self._closed or event.permission != Permission.CLOUD_AI:
            return
        if not event.granted:
            self._abort_cloud_jobs()
        self._refresh_status()

    def _abort_cloud_jobs(self) -> None:
        """CLOUD_AI was revoked: nothing more may leave the PC."""
        dropped: list[_Job] = []
        with self._cond:
            provider = self._provider
            if provider is None or provider.is_local:
                return
            keep: list[_Job] = []
            for job in self._queue:
                (dropped if job.kind == "generate" and job.provider is None else keep).append(job)
            self._queue[:] = keep
            running = self._running
            if running is not None and running.kind == "generate" and running.provider is None:
                running.abort_error = AIOfflineError("cloud permission revoked",
                                                     user_message=REASON_PERMISSION)
                running.token.cancel()
        if dropped:
            log.info("cloud AI permission revoked: dropped %d queued request(s)", len(dropped))
        for job in dropped:
            error = AIOfflineError("cloud permission revoked", user_message=REASON_PERMISSION)
            self._post(lambda job=job, error=error: _call(job.on_error, error))

    # -- queue ----------------------------------------------------------------------------

    def _enqueue(self, job: _Job) -> None:
        victim: _Job | None = None
        cancelled: list[_Job] = []
        with self._cond:
            # Requests cancelled while waiting must not hold a slot.
            for queued in list(self._queue):
                if queued.token.cancelled:
                    self._queue.remove(queued)
                    cancelled.append(queued)
            if len(self._queue) >= MAX_QUEUE:
                # Lowest priority (highest rank) loses; among equals the oldest.
                victim = max([*self._queue, job], key=lambda j: (j.rank, -j.seq))
                if victim is not job:
                    self._queue.remove(victim)
                    self._queue.append(job)
            else:
                self._queue.append(job)
            self._cond.notify()
            if self._thread is None:
                self._thread = threading.Thread(target=self._worker_main, name="chibi-ai", daemon=True)
                self._thread.start()
        for job_ in cancelled:
            _close_own(job_)
            self._post(lambda j=job_: _call(j.on_error, CancelledError("cancelled")))
        if victim is not None:
            _close_own(victim)
            log.info("AI queue full: dropped a %s request", _purpose_name(victim))
            self._post(lambda v=victim: _call(v.on_error, QueueFullError()))

    # -- worker thread --------------------------------------------------------------------

    def _worker_main(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stopping:
                    self._cond.wait()
                if self._stopping:
                    return
                job = min(self._queue, key=lambda j: (j.rank, j.seq))
                self._queue.remove(job)
                self._running = job
                provider = job.provider if job.provider is not None else self._provider
            try:
                if job.kind == "models":
                    self._run_models(job, provider)
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
        attempt = 0
        while True:
            if job.token.cancelled or self._stopping:
                self._fail(job, CancelledError("cancelled"))
                return
            if not provider.is_local and not self._permissions.is_granted(Permission.CLOUD_AI):
                log.info("cloud AI permission missing: request dropped before sending")
                self._fail(job, AIOfflineError("cloud permission missing", user_message=REASON_PERMISSION))
                return
            if not job.bypass_breaker and self._breaker_open():
                self._fail(job, AIOfflineError("circuit open", user_message=REASON_DOWN))
                return
            streamed = False

            def on_piece(piece: str) -> None:
                nonlocal streamed
                streamed = True
                self._post(lambda: None if self._job_cancelled(job) or job.on_delta is None
                           else _call(job.on_delta, piece))

            started = time.monotonic()
            try:
                if job.stream:
                    result = provider.stream(job.request, on_piece, job.token)
                else:
                    result = provider.generate(job.request, job.token)
            except CancelledError as exc:
                self._fail(job, exc)
                return
            except ProviderError as exc:
                if job.token.cancelled or self._stopping:
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
                _log_failure(f"{provider.name} provider", exc)
                self._fail(job, ProviderUnavailableError("provider crashed", user_message="Помилка AI."),
                           provider=provider)
                return
            log.info("AI %s request done in %.1f s (%d in / %d out tokens)", job.request.purpose.value,
                     time.monotonic() - started, result.usage.input_tokens, result.usage.output_tokens)
            self._succeed(job, result)
            return

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
        if isinstance(exc, CancelledError) and job.abort_error is not None:
            exc = job.abort_error

        def deliver() -> None:
            if provider is not None:
                self._note_failure(job, provider, exc)
            _call(job.on_error, exc)

        self._post(deliver)

    def _succeed(self, job: _Job, result: GenerationResult) -> None:
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
        return job.token.cancelled

    # -- cache (main thread) ----------------------------------------------------------------

    @staticmethod
    def _cache_key(request: GenerationRequest, provider: TextProvider | None) -> str | None:
        if provider is None or request.purpose == Purpose.CHAT or any(t.images for t in request.turns):
            return None
        payload = json.dumps(
            [provider.name, provider.model, request.system, request.context,
             [[t.role.value, t.text] for t in request.turns], request.max_tokens, request.json_mode,
             request.effort],
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


def _purpose_name(job: _Job) -> str:
    return job.request.purpose.value if job.request is not None else job.kind
