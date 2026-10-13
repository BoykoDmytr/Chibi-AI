"""Ollama provider: a local model server, nothing leaves the PC (when it runs on localhost).

``POST /api/chat`` (NDJSON stream: one JSON object per line, the last one has
``"done": true`` and the token counts), ``GET /api/tags`` (installed models),
``POST /api/show`` (capabilities, for vision), ``POST /api/generate`` with only
the model (:meth:`OllamaProvider.warm_up`: load it, generate nothing).  Plain
``httpx2``.

Ollama unloads a model 5 minutes after its last request, and loading it again
takes 5-20 s; every request therefore carries ``keep_alive`` (the "keep the
model loaded" setting), and :meth:`~OllamaProvider.warm_up` lets the app load
it before the user's words are even recognised.  The prompt layout keeps the
system prompt and the history an unchanged prefix that Ollama does not read
again (see the OpenAI-compatible provider's module docstring), and a cancelled
request is aborted at once (:class:`RequestAbort`): Ollama works on one request
at a time.  :attr:`~OllamaProvider.last_load_seconds` /
:attr:`~OllamaProvider.last_prompt_seconds` tell where the last request's time
went (numbers only).

An empty model name means "the first installed model", resolved lazily on the
worker thread the first time it is needed.  Errors carry instructions the user
can act on: Ollama not running, model not pulled.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from typing import Any

import httpx2

from chibi.ai.providers.openai_provider import (
    RequestAbort,
    build_client,
    clean_json_text,
    image_b64,
    ipv4_loopback,
    is_loopback_url,
    parse_retry_after,
    require_user_turn,
    status_error,
    turn_texts,
)
from chibi.ai.providers.reasoning import ReasoningFilter, strip_reasoning
from chibi.ai.types import (
    CancelToken,
    GenerationRequest,
    GenerationResult,
    ProviderRequestError,
    ProviderUnavailableError,
    Role,
    Usage,
)

log = logging.getLogger(__name__)

SERVICE = "Ollama"
#: A model to suggest when nothing is installed (good Ukrainian, runs on a laptop GPU).
SUGGESTED_MODEL = "gemma3"


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str, *, timeout: float = 60.0,
                 keep_alive: str | None = None, http_client: httpx2.Client | None = None) -> None:
        """*keep_alive*: how long Ollama keeps the model loaded after a request (its
        duration syntax, e.g. ``"30m"``); None leaves Ollama's own default (5 minutes)."""
        self.base_url = base_url.strip().rstrip("/")
        #: Where requests go: "localhost" as 127.0.0.1, like Ollama's own clients - Windows
        #: resolves "localhost" to ::1 first, Ollama listens on 127.0.0.1 only, and the
        #: refused ::1 attempt costs 1-2 s there (Windows retries the SYN) on every request.
        self._address = ipv4_loopback(self.base_url)
        self.model = model.strip()
        self.keep_alive = keep_alive
        # Connecting to a local server either works at once or not at all; the
        # read timeout must cover loading a model into memory on first use.
        self._timeout = httpx2.Timeout(timeout, connect=min(timeout, 5.0))
        self._owns_client = http_client is None
        # A fresh connection per request: free on this machine / the LAN, and a
        # cancelled request can then always be aborted (Ollama answers one at a time).
        self._client = http_client if http_client is not None else \
            build_client(self._timeout, fresh_connections=True)
        self._lock = threading.Lock()
        self._vision: dict[str, bool] = {}
        #: Seconds Ollama spent loading the model / reading the prompt for the last
        #: request on the AI worker (None: not reported).
        self.last_load_seconds: float | None = None
        self.last_prompt_seconds: float | None = None

    def __repr__(self) -> str:
        return f"OllamaProvider({self.base_url!r}, {self.model!r})"

    @property
    def is_local(self) -> bool:
        return is_loopback_url(self.base_url)

    # -- errors --------------------------------------------------------------------------

    def _transport_error(self, exc: Exception) -> ProviderUnavailableError:
        if isinstance(exc, httpx2.TimeoutException) and not isinstance(exc, httpx2.ConnectTimeout):
            return ProviderUnavailableError(
                "Ollama timeout", user_message="Ollama не відповіла вчасно (модель ще завантажується?).")
        return ProviderUnavailableError("Ollama не запущена",
                                        user_message="Ollama не запущена — запусти її і спробуй ще раз.")

    def _raise_for_status(self, response: httpx2.Response, model: str) -> None:
        if response.status_code < 400:
            return
        log.warning("ollama request failed: HTTP %s", response.status_code)
        if response.status_code == 404 and model:
            raise ProviderRequestError(
                "Ollama model not found",
                user_message=f"Модель «{model}» не встановлена в Ollama. "
                             f"Виконай у терміналі: ollama pull {model}",
            )
        raise status_error(response.status_code, service=SERVICE, model=model,
                           retry_after=parse_retry_after(response.headers.get("retry-after")))

    # -- model resolution ------------------------------------------------------------------

    def _ensure_model(self) -> str:
        model = self.model
        if model:
            return model
        installed = [m for m in self.list_models() if "embed" not in m.lower()]
        if not installed:
            raise ProviderRequestError(
                "no Ollama models installed",
                user_message=f"В Ollama ще немає жодної моделі. Встанови, наприклад: "
                             f"ollama pull {SUGGESTED_MODEL}",
            )
        with self._lock:
            if not self.model:
                self.model = installed[0]
                log.info("ollama: using the first installed model %s", self.model)
            return self.model

    # -- requests ------------------------------------------------------------------------

    def _body(self, request: GenerationRequest, model: str, *, stream: bool) -> dict[str, Any]:
        require_user_turn(request)
        messages: list[dict[str, Any]] = []
        if request.system.strip():
            messages.append({"role": "system", "content": request.system})
        for turn, text in zip(request.turns, turn_texts(request), strict=True):
            if not text and not (turn.role == Role.USER and turn.images):
                continue
            message: dict[str, Any] = {"role": turn.role.value, "content": text}
            if turn.role == Role.USER and turn.images:
                message["images"] = [image_b64(img.data) for img in turn.images]
            messages.append(message)
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {"num_predict": max(1, request.max_tokens)},
        }
        if self.keep_alive:
            body["keep_alive"] = self.keep_alive
        if request.json_mode:
            body["format"] = "json"
        return body

    def _note_timing(self, data: dict[str, Any]) -> None:
        """Where the request's time went, from the final object's nanosecond counters."""
        self.last_load_seconds = _seconds(data.get("load_duration"))
        self.last_prompt_seconds = _seconds(data.get("prompt_eval_duration"))

    def generate(self, request: GenerationRequest, cancel: CancelToken | None = None) -> GenerationResult:
        if cancel is not None:
            cancel.raise_if_cancelled()
        self.last_load_seconds = self.last_prompt_seconds = None
        model = (request.model or "").strip() or self._ensure_model()
        body = self._body(request, model, stream=False)
        with RequestAbort(cancel) as abort:
            try:
                response = self._client.post(f"{self._address}/api/chat", json=body, timeout=self._timeout,
                                             extensions=abort.extensions)
            except httpx2.HTTPError as exc:
                raise self._transport_error(exc) from None
        self._raise_for_status(response, model)
        try:
            data = response.json()
            text = data["message"]["content"]
        except (ValueError, KeyError, TypeError):
            raise self._bad_payload() from None
        if not isinstance(text, str):
            raise self._bad_payload()
        # A complete answer is returned even if the token was cancelled meanwhile (see
        # RequestAbort): the AI service decides whether anyone still wants it.
        self._note_timing(data)
        return self._result(strip_reasoning(text), data, model, request)

    def stream(self, request: GenerationRequest, on_delta: Callable[[str], None],
               cancel: CancelToken | None = None) -> GenerationResult:
        if cancel is not None:
            cancel.raise_if_cancelled()
        self.last_load_seconds = self.last_prompt_seconds = None
        model = (request.model or "").strip() or self._ensure_model()
        body = self._body(request, model, stream=True)
        parts: list[str] = []
        reasoning = ReasoningFilter()
        final: dict[str, Any] = {}
        try:
            with RequestAbort(cancel) as abort, \
                    self._client.stream("POST", f"{self._address}/api/chat", json=body,
                                        timeout=self._timeout, extensions=abort.extensions) as response:
                if response.status_code >= 400:
                    response.read()
                    self._raise_for_status(response, model)
                for line in response.iter_lines():
                    if cancel is not None:
                        cancel.raise_if_cancelled()
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError:
                        raise self._bad_payload() from None
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("error"):
                        log.warning("ollama stream reported an error")
                        raise ProviderUnavailableError("Ollama stream error",
                                                       user_message="Ollama перервала відповідь.")
                    message = chunk.get("message")
                    piece = message.get("content") if isinstance(message, dict) else None
                    if isinstance(piece, str) and piece:
                        parts.append(piece)
                        visible = reasoning.feed(piece)
                        if visible:
                            on_delta(visible)
                    if chunk.get("done"):
                        final = chunk
                        break
        except httpx2.HTTPError as exc:
            raise self._transport_error(exc) from None
        self._note_timing(final)
        tail = reasoning.finish()
        if tail:
            on_delta(tail)
        return self._result(strip_reasoning("".join(parts)), final, model, request)

    def warm_up(self, cancel: CancelToken | None = None) -> None:
        """Load the model into memory without generating anything (``POST /api/generate``
        with only the model and ``keep_alive``), so the next answer does not wait for it.
        Returns once it is loaded; raises like :meth:`generate`."""
        if cancel is not None:
            cancel.raise_if_cancelled()
        self.last_load_seconds = self.last_prompt_seconds = None
        model = self._ensure_model()
        body: dict[str, Any] = {"model": model}
        if self.keep_alive:
            body["keep_alive"] = self.keep_alive
        with RequestAbort(cancel) as abort:
            try:
                response = self._client.post(f"{self._address}/api/generate", json=body,
                                             timeout=self._timeout, extensions=abort.extensions)
            except httpx2.HTTPError as exc:
                raise self._transport_error(exc) from None
        self._raise_for_status(response, model)
        try:
            data = response.json()
        except ValueError:
            raise self._bad_payload() from None
        if isinstance(data, dict):
            self._note_timing(data)

    def list_models(self) -> list[str]:
        try:
            response = self._client.get(f"{self._address}/api/tags", timeout=self._timeout)
        except httpx2.HTTPError as exc:
            raise self._transport_error(exc) from None
        self._raise_for_status(response, self.model)
        try:
            models = response.json()["models"]
            return [str(m["name"]) for m in models if isinstance(m, dict) and m.get("name")]
        except (ValueError, KeyError, TypeError):
            raise self._bad_payload() from None

    @property
    def supports_vision(self) -> bool:
        """Whether the model accepts images (``/api/show`` capabilities); False when unknown.

        Network I/O on first use per model (then cached): call it on the worker
        thread, never on the Qt main thread.  Failures are not cached, so a
        server that wasn't running yet is asked again next time.
        """
        try:
            model = self._ensure_model()
        except (ProviderRequestError, ProviderUnavailableError):
            return False
        with self._lock:
            if model in self._vision:
                return self._vision[model]
        try:
            response = self._client.post(f"{self._address}/api/show", json={"model": model},
                                         timeout=self._timeout)
            if response.status_code != 200:
                return False
            data = response.json()
        except (httpx2.HTTPError, ValueError):
            return False
        if not isinstance(data, dict):
            return False
        capabilities = data.get("capabilities")
        if isinstance(capabilities, list):
            result = "vision" in capabilities
        else:
            # Ollama before 0.6 has no capabilities list: a vision projector gives it away.
            details = data.get("details") if isinstance(data.get("details"), dict) else {}
            families = details.get("families") or []
            result = bool(data.get("projector_info")) or any(f in ("clip", "mllama") for f in families)
        with self._lock:
            self._vision[model] = result
        return result

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # -- results -------------------------------------------------------------------------

    @staticmethod
    def _bad_payload() -> ProviderRequestError:
        return ProviderRequestError("ollama: unexpected response",
                                    user_message="Ollama повернула незрозумілу відповідь — перевір адресу.")

    def _result(self, text: str, data: dict[str, Any], model: str,
                request: GenerationRequest) -> GenerationResult:
        if request.json_mode:
            text = clean_json_text(text)
        reason = str(data.get("done_reason") or "")
        stop = {"stop": "end_turn", "length": "max_tokens"}.get(reason, reason)
        usage = Usage(input_tokens=_count(data.get("prompt_eval_count")),
                      output_tokens=_count(data.get("eval_count")))
        return GenerationResult(text=text, provider=self.name, model=model, usage=usage, stop_reason=stop)


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _seconds(nanoseconds: Any) -> float | None:
    """A duration Ollama reports in nanoseconds, in seconds (None when absent)."""
    if isinstance(nanoseconds, bool) or not isinstance(nanoseconds, (int, float)) or not nanoseconds >= 0:
        return None
    return float(nanoseconds) / 1e9
