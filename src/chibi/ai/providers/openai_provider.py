"""OpenAI-compatible chat provider (OpenAI, LM Studio, vLLM, llama.cpp server...).

Plain ``httpx2`` against ``POST {base_url}/chat/completions`` (+ SSE streaming)
and ``GET {base_url}/models``.  Differences between OpenAI itself and the
compatible servers are handled by host: ``max_completion_tokens`` and
``stream_options.include_usage`` only for ``api.openai.com`` (older local
servers reject unknown fields), ``max_tokens`` elsewhere.

This module also holds the small helpers the other providers share (it is the
generic HTTP provider): loopback detection, ``Retry-After`` parsing, lenient
JSON cleanup and HTTP status -> :class:`ProviderError` mapping.

Privacy: the API key is sent only as a Bearer header and never logged; the
``OPENAI_API_KEY`` environment variable is used only for ``api.openai.com`` so
a real OpenAI key can't leak to a third-party server.  Nothing from the
request or the response body is logged or put into exception messages.
"""

from __future__ import annotations

import base64
import email.utils
import ipaddress
import json
import logging
import os
import re
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlsplit

import httpx2

from chibi.ai.providers.reasoning import ReasoningFilter, strip_reasoning
from chibi.ai.types import (
    CancelledError,
    CancelToken,
    GenerationRequest,
    GenerationResult,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderRequestError,
    ProviderUnavailableError,
    Role,
    Usage,
)

log = logging.getLogger(__name__)

OPENAI_HOST = "api.openai.com"
#: Appended to the operator part of the prompt when a request wants JSON back.
JSON_INSTRUCTION = (
    "Reply with exactly one JSON object and nothing else: no explanations, no Markdown code fences."
)
#: Retry-After values above this are not honoured (we would rather fail and fall back).
MAX_RETRY_AFTER = 3600.0
#: Library loggers that log request bodies (chat text, images) or headers at DEBUG level.
HTTP_LOGGERS = ("anthropic", "httpx2", "httpcore2")


def quiet_http_loggers() -> None:
    """Keep HTTP/SDK debug logging out of our log file even in ``--debug`` runs.

    The anthropic SDK logs the full request JSON at DEBUG - chat text included -
    and inherits the root level otherwise.  A developer can still opt in with
    ``ANTHROPIC_LOG`` or by configuring these loggers explicitly.
    """
    if os.environ.get("ANTHROPIC_LOG"):
        return
    for name in HTTP_LOGGERS:
        logger = logging.getLogger(name)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.WARNING)


quiet_http_loggers()


# -- shared helpers ---------------------------------------------------------------------------


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def is_loopback_url(url: str) -> bool:
    """True if *url* points at this machine (``localhost``, 127.0.0.0/8, ``::1``)."""
    host = _host(url)
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return ip.is_loopback or bool(mapped is not None and mapped.is_loopback)


def is_private_network_url(url: str) -> bool:
    """True for loopback or a private-network address (LAN servers often need no API key)."""
    if is_loopback_url(url):
        return True
    host = _host(url)
    if host.endswith((".local", ".lan", ".home.arpa", ".internal")):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private and not ip.is_global


def parse_retry_after(value: str | None) -> float | None:
    """Seconds from a ``Retry-After`` header (delta-seconds or an HTTP date)."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when is None:
            return None
        seconds = when.timestamp() - time.time()
    if seconds != seconds:  # NaN
        return None
    return min(max(seconds, 0.0), MAX_RETRY_AFTER)


_FENCE_RE = re.compile(r"^```[A-Za-z0-9_-]*[ \t]*\n?(.*?)\n?```\s*$", re.S)


def clean_json_text(text: str) -> str:
    """Best effort: the JSON object a model meant, without fences or chatter around it.

    Returns the stripped input unchanged when no JSON value can be found - the
    caller's parser decides what to do then.
    """
    s = text.strip()
    fenced = _FENCE_RE.match(s)
    if fenced:
        s = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    try:
        decoder.decode(s)
        return s
    except ValueError:
        pass
    attempts = 0
    for i, ch in enumerate(s):
        if ch not in "{[":
            continue
        attempts += 1
        if attempts > 20:
            break
        try:
            _, end = decoder.raw_decode(s, i)
        except ValueError:
            continue
        return s[i:end]
    return s


def status_error(status: int, *, service: str, model: str, retry_after: float | None = None,
                 code: str = "") -> ProviderError:
    """Map an HTTP error status to a typed error with a Ukrainian ``user_message``.

    *service* is a name for messages ("Ollama", "OpenAI-сумісний сервер"); messages are
    phrased as "<service>: ..." so they need no case or gender agreement.
    The server's own error text is deliberately not included: it can echo input.
    """
    detail = f"{service} HTTP {status}" + (f" ({code})" if code else "")
    if status in (401, 403):
        return ProviderAuthError(detail, user_message=f"{service}: ключ API не підходить або немає доступу.")
    if status == 402:
        return ProviderAuthError(detail, user_message=f"{service}: проблема з оплатою акаунта.")
    if status == 429:
        return ProviderRateLimitError(detail, retry_after=retry_after,
                                      user_message=f"{service}: забагато запитів, спробую трохи згодом.")
    if status == 404:
        return ProviderRequestError(detail, user_message=f"{service}: модель «{model}» не знайдена "
                                                         "(або неправильна адреса сервера).")
    if status in (408, 409) or status >= 500:
        return ProviderUnavailableError(detail, user_message=f"{service}: сервіс зараз недоступний, "
                                                             "спробую пізніше.")
    if status == 413:
        return ProviderRequestError(detail, user_message=f"{service}: запит завеликий.")
    return ProviderRequestError(detail, user_message=f"{service}: запит відхилено (код {status}).")


def transport_error(exc: Exception, *, service: str) -> ProviderUnavailableError:
    if isinstance(exc, httpx2.TimeoutException):
        return ProviderUnavailableError(f"{service} timeout",
                                        user_message=f"{service}: відповідь не прийшла вчасно.")
    return ProviderUnavailableError(f"{service} connection failed ({type(exc).__name__})",
                                    user_message=f"{service}: немає з'єднання. Перевір адресу та інтернет.")


def image_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def require_user_turn(request: GenerationRequest) -> None:
    if not any(t.role == Role.USER and (t.text.strip() or t.images) for t in request.turns):
        raise ProviderRequestError("request has no user turn", user_message="Порожній запит до AI.")


def _error_code(payload: Any) -> str:
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            code = err.get("code") or err.get("type") or ""
            return str(code)[:60] if isinstance(code, (str, int)) else ""
    return ""


# -- provider ---------------------------------------------------------------------------------


class OpenAIProvider:
    """``/chat/completions`` on OpenAI or any OpenAI-compatible server."""

    name = "openai"
    SERVICE = "OpenAI-сумісний сервер"

    def __init__(self, api_key: str | None, base_url: str, model: str, *, timeout: float = 30.0,
                 http_client: httpx2.Client | None = None) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.model = model.strip()
        self._is_openai = _host(self.base_url) == OPENAI_HOST
        key = (api_key or "").strip() or None
        if key is None and self._is_openai:
            key = (os.environ.get("OPENAI_API_KEY") or "").strip() or None
        self._api_key = key
        self._timeout = httpx2.Timeout(timeout, connect=min(timeout, 10.0))
        self._owns_client = http_client is None
        self._client = http_client if http_client is not None else httpx2.Client(timeout=self._timeout)

    def __repr__(self) -> str:  # never show the key
        return f"OpenAIProvider({self.base_url!r}, {self.model!r})"

    @property
    def is_local(self) -> bool:
        return is_loopback_url(self.base_url)

    @property
    def has_key(self) -> bool:
        return self._api_key is not None

    # -- requests ------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _messages(self, request: GenerationRequest) -> list[dict[str, Any]]:
        require_user_turn(request)
        messages: list[dict[str, Any]] = []
        if request.system.strip():
            messages.append({"role": "system", "content": request.system})
        operator = "\n\n".join(p for p in (request.context.strip(),
                                           JSON_INSTRUCTION if request.json_mode else "") if p)
        if operator:
            messages.append({"role": "system", "content": operator})
        for turn in request.turns:
            if turn.role == Role.USER and turn.images:
                parts: list[dict[str, Any]] = [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{img.media_type};base64,{image_b64(img.data)}"}}
                    for img in turn.images
                ]
                if turn.text:
                    parts.append({"type": "text", "text": turn.text})
                messages.append({"role": "user", "content": parts})
            elif turn.text:
                messages.append({"role": turn.role.value, "content": turn.text})
        return messages

    def model_for(self, request: GenerationRequest) -> str:
        return (request.model or "").strip() or self.model

    def _body(self, request: GenerationRequest, *, stream: bool) -> dict[str, Any]:
        model = self.model_for(request)
        if not model:
            raise ProviderRequestError("no model", user_message="Вибери модель в налаштуваннях AI.")
        body: dict[str, Any] = {"model": model, "messages": self._messages(request), "stream": stream}
        body["max_completion_tokens" if self._is_openai else "max_tokens"] = max(1, request.max_tokens)
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
        if stream and self._is_openai:
            body["stream_options"] = {"include_usage": True}
        return body

    def _raise_for_status(self, response: httpx2.Response, model: str | None = None) -> None:
        if response.status_code < 400:
            return
        try:
            payload = response.json()
        except (ValueError, httpx2.HTTPError, httpx2.StreamError):
            payload = None
        retry_after = parse_retry_after(response.headers.get("retry-after"))
        log.warning("%s request failed: HTTP %s", self.name, response.status_code)
        raise status_error(response.status_code, service=self.SERVICE, model=model or self.model,
                           retry_after=retry_after, code=_error_code(payload))

    def _bad_payload(self) -> ProviderRequestError:
        return ProviderRequestError(
            f"{self.name}: unexpected response",
            user_message="Сервер AI повернув незрозумілу відповідь — перевір адресу сервера.",
        )

    def generate(self, request: GenerationRequest, cancel: CancelToken | None = None) -> GenerationResult:
        if cancel is not None:
            cancel.raise_if_cancelled()
        body = self._body(request, stream=False)
        try:
            response = self._client.post(f"{self.base_url}/chat/completions", json=body,
                                         headers=self._headers(), timeout=self._timeout)
        except httpx2.HTTPError as exc:
            raise transport_error(exc, service=self.SERVICE) from None
        self._raise_for_status(response, body["model"])
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice.get("message") or {}
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise self._bad_payload() from None
        if cancel is not None:
            cancel.raise_if_cancelled()
        finish = str(choice.get("finish_reason") or "")
        self._check_refusal(finish, message.get("refusal"))
        text = message.get("content") or ""
        if not isinstance(text, str):
            raise self._bad_payload()
        return self._result(strip_reasoning(text), finish, data.get("usage"), request)

    def stream(self, request: GenerationRequest, on_delta: Callable[[str], None],
               cancel: CancelToken | None = None) -> GenerationResult:
        if cancel is not None:
            cancel.raise_if_cancelled()
        body = self._body(request, stream=True)
        parts: list[str] = []
        reasoning = ReasoningFilter()
        finish = ""
        usage: Any = None
        refusal = ""
        try:
            with self._client.stream("POST", f"{self.base_url}/chat/completions", json=body,
                                     headers={**self._headers(), "Accept": "text/event-stream"},
                                     timeout=self._timeout) as response:
                if response.status_code >= 400:
                    response.read()
                    self._raise_for_status(response, body["model"])
                for payload in _sse_data(response.iter_lines(), cancel):
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        raise self._bad_payload() from None
                    if not isinstance(chunk, dict):
                        continue
                    if "error" in chunk:
                        code = _error_code(chunk)
                        log.warning("%s stream reported an error (%s)", self.name, code or "unknown")
                        raise ProviderUnavailableError(f"{self.name} stream error {code}",
                                                       user_message=f"{self.SERVICE}: відповідь перервалася.")
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices or not isinstance(choices[0], dict):
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") if isinstance(delta, dict) else None
                    if isinstance(delta, dict) and delta.get("refusal"):
                        refusal += str(delta["refusal"])
                    if isinstance(piece, str) and piece:
                        parts.append(piece)
                        visible = reasoning.feed(piece)
                        if visible:
                            on_delta(visible)
                    if choices[0].get("finish_reason"):
                        finish = str(choices[0]["finish_reason"])
        except httpx2.HTTPError as exc:
            raise transport_error(exc, service=self.SERVICE) from None
        self._check_refusal(finish, refusal)
        tail = reasoning.finish()
        if tail:
            on_delta(tail)
        return self._result(strip_reasoning("".join(parts)), finish, usage, request)

    def list_models(self) -> list[str]:
        try:
            response = self._client.get(f"{self.base_url}/models", headers=self._headers(),
                                        timeout=self._timeout)
        except httpx2.HTTPError as exc:
            raise transport_error(exc, service=self.SERVICE) from None
        self._raise_for_status(response)
        try:
            items = response.json()["data"]
            return sorted({str(item["id"]) for item in items if isinstance(item, dict) and item.get("id")})
        except (ValueError, KeyError, TypeError):
            raise self._bad_payload() from None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # -- results -------------------------------------------------------------------------

    @staticmethod
    def _check_refusal(finish: str, refusal: Any) -> None:
        if finish == "content_filter" or (isinstance(refusal, str) and refusal.strip()):
            raise ProviderRefusalError("refusal", user_message="Модель відмовилася відповідати на це.")

    def _result(self, text: str, finish: str, usage: Any, request: GenerationRequest) -> GenerationResult:
        if request.json_mode:
            text = clean_json_text(text)
        prompt = completion = cached = 0
        if isinstance(usage, dict):
            prompt = _int(usage.get("prompt_tokens"))
            completion = _int(usage.get("completion_tokens"))
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                cached = min(_int(details.get("cached_tokens")), prompt)
        stop = {"stop": "end_turn", "length": "max_tokens"}.get(finish, finish)
        return GenerationResult(
            text=text, provider=self.name, model=self.model_for(request),
            usage=Usage(input_tokens=prompt - cached, output_tokens=completion, cache_read_tokens=cached),
            stop_reason=stop,
        )


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _sse_data(lines: Iterator[str], cancel: CancelToken | None) -> Iterator[str]:
    """Yield the ``data`` payload of each server-sent event (multi-line data joined)."""
    data: list[str] = []
    for line in lines:
        if cancel is not None and cancel.cancelled:
            raise CancelledError("cancelled")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield "\n".join(data)
