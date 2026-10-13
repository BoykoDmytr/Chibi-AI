"""Claude through the official ``anthropic`` SDK (never raw HTTP).

Request shape, per the current Messages API rules:

* the persona is one cached system block (``cache_control: ephemeral``); the
  volatile operator context and the JSON instruction follow it *after* the
  breakpoint, so mood/time changes never invalidate the cached prefix;
* no ``temperature`` / ``top_p`` / ``top_k`` and no ``thinking``: current
  models reject sampling parameters and run adaptive thinking by default
  (``disabled`` / ``budget_tokens`` are 400s); depth and spend are steered with
  ``output_config.effort`` on the models that support it;
* server-side refusal fallbacks (beta ``server-side-fallback-2026-07-01``,
  ``fallbacks: "default"``) for the models that offer them, so a classifier
  false positive is retried on another model inside the same call;
* ``stop_reason == "refusal"`` is checked before the content is read.

Everything is streamed, also for :meth:`generate`: no HTTP timeout on long
answers, and a :class:`CancelToken` is honoured between stream events.
"""

from __future__ import annotations

import base64
import logging
import re
from collections.abc import Callable
from typing import Any

import anthropic

from chibi.ai.providers.openai_provider import JSON_INSTRUCTION, clean_json_text, parse_retry_after
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
    Purpose,
    Role,
    Usage,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high")
#: Chat replies get room for adaptive thinking plus a few paragraphs.
CHAT_MAX_TOKENS = 1500
FALLBACK_BETA = "server-side-fallback-2026-07-01"
#: Models with server-side refusal fallbacks ("default" routing).
FALLBACK_MODELS = frozenset({"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"})
#: 4.x models that accept ``output_config.effort`` (the whole 5.x family does too).
_EFFORT_4X = frozenset({"claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-4-6"})
_FAMILY_5X = re.compile(r"claude-(opus|sonnet|haiku|fable|mythos)-5(-\d+)?")

_NO_KEY = "Немає API-ключа Anthropic. Додай його в налаштуваннях AI."


def supports_effort(model: str) -> bool:
    return model in _EFFORT_4X or bool(_FAMILY_5X.fullmatch(model))


def uses_fallbacks(model: str) -> bool:
    return model in FALLBACK_MODELS


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None, model: str = DEFAULT_MODEL, *, effort: str = "low",
                 timeout: float = 30.0, max_retries: int = 2, http_client: Any = None) -> None:
        self.model = (model or DEFAULT_MODEL).strip()
        self.effort = effort if effort in EFFORTS else "low"
        key = (api_key or "").strip() or None
        kwargs: dict[str, Any] = {"api_key": key, "timeout": timeout, "max_retries": max_retries}
        if http_client is not None:
            kwargs["http_client"] = http_client
        # api_key=None lets the SDK resolve ANTHROPIC_API_KEY / an `ant auth` profile itself.
        try:
            self._client = anthropic.Anthropic(**kwargs)
        except anthropic.CredentialsError:
            # An explicitly selected profile / config dir that doesn't exist.
            raise ProviderAuthError("Anthropic credentials not found", user_message=_NO_KEY) from None

    def __repr__(self) -> str:  # never show the key
        return f"AnthropicProvider({self.model!r})"

    @property
    def is_local(self) -> bool:
        return False

    @property
    def supports_vision(self) -> bool:
        return True

    # -- request -------------------------------------------------------------------------

    def _system(self, request: GenerationRequest) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if request.system.strip():
            blocks.append({"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}})
        if request.context.strip():
            blocks.append({"type": "text", "text": request.context})
        if request.json_mode:
            blocks.append({"type": "text", "text": JSON_INSTRUCTION})
        return blocks

    @staticmethod
    def _messages(request: GenerationRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for turn in request.turns:
            content: list[dict[str, Any]] = []
            if turn.role == Role.USER:
                for img in turn.images:
                    content.append({"type": "image", "source": {
                        "type": "base64", "media_type": img.media_type,
                        "data": base64.b64encode(img.data).decode("ascii")}})
            if turn.text.strip():
                content.append({"type": "text", "text": turn.text})
            if not content:
                continue  # the API rejects empty text blocks
            if messages and messages[-1]["role"] == turn.role.value:
                messages[-1]["content"].extend(content)
            else:
                messages.append({"role": turn.role.value, "content": content})
        # The conversation must start with the user, and current models reject an
        # assistant prefill at the end.
        while messages and messages[0]["role"] != Role.USER.value:
            messages.pop(0)
        while messages and messages[-1]["role"] != Role.USER.value:
            messages.pop()
        if not messages:
            raise ProviderRequestError("request has no user turn", user_message="Порожній запит до AI.")
        return messages

    def model_for(self, request: GenerationRequest) -> str:
        return (request.model or "").strip() or self.model

    def _params(self, request: GenerationRequest) -> dict[str, Any]:
        max_tokens = max(1, request.max_tokens)
        if request.purpose == Purpose.CHAT:
            max_tokens = max(max_tokens, CHAT_MAX_TOKENS)
        model = self.model_for(request)
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": self._messages(request),
        }
        system = self._system(request)
        if system:
            params["system"] = system
        if supports_effort(model):
            effort = request.effort if request.effort in EFFORTS else self.effort
            params["output_config"] = {"effort": effort}
        if uses_fallbacks(model):
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    # -- calls ---------------------------------------------------------------------------

    def generate(self, request: GenerationRequest, cancel: CancelToken | None = None) -> GenerationResult:
        return self._run(request, None, cancel)

    def stream(self, request: GenerationRequest, on_delta: Callable[[str], None],
               cancel: CancelToken | None = None) -> GenerationResult:
        return self._run(request, on_delta, cancel)

    def _run(self, request: GenerationRequest, on_delta: Callable[[str], None] | None,
             cancel: CancelToken | None) -> GenerationResult:
        if cancel is not None:
            cancel.raise_if_cancelled()
        params = self._params(request)
        api = self._client.beta.messages if "fallbacks" in params else self._client.messages
        try:
            with api.stream(**params) as stream:
                for event in stream:
                    # Every stream event (thinking deltas included) is a cancellation point.
                    if cancel is not None and cancel.cancelled:
                        raise CancelledError("cancelled")
                    if on_delta is not None and event.type == "text" and event.text:
                        on_delta(event.text)
                final = stream.get_final_message()
        except anthropic.AnthropicError as exc:
            raise self._map_error(exc) from None
        except TypeError as exc:
            # The SDK raises TypeError when no key/token/profile can be resolved.
            if "authentication" in str(exc).lower():
                raise ProviderAuthError("no Anthropic credentials", user_message=_NO_KEY) from None
            raise
        # A complete answer is returned even if the token was cancelled after its last
        # event: it was billed in full; the AI service records it and decides.
        # Check the stop reason before touching the content: a refused answer may
        # be empty or a partial that must be discarded.
        if final.stop_reason == "refusal":
            category = getattr(getattr(final, "stop_details", None), "category", None)
            log.info("Claude declined the request (category %s)", category or "unknown")
            raise ProviderRefusalError("refusal", user_message="Claude відмовився відповідати на це.")
        text = "".join(block.text for block in final.content if getattr(block, "type", "") == "text")
        if request.json_mode:
            text = clean_json_text(text)
        usage = final.usage
        return GenerationResult(
            text=text,
            provider=self.name,
            model=str(final.model or self.model_for(request)),
            usage=Usage(
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
            ),
            stop_reason=str(final.stop_reason or ""),
        )

    def list_models(self) -> list[str]:
        try:
            return [model.id for model in self._client.models.list()]
        except anthropic.AnthropicError as exc:
            raise self._map_error(exc) from None
        except TypeError as exc:
            if "authentication" in str(exc).lower():
                raise ProviderAuthError("no Anthropic credentials", user_message=_NO_KEY) from None
            raise

    def close(self) -> None:
        self._client.close()

    # -- errors --------------------------------------------------------------------------

    def _map_error(self, exc: anthropic.AnthropicError) -> ProviderError:
        """Typed SDK error -> provider-neutral error.  The server's message is not
        copied: it can quote the request."""
        status = getattr(exc, "status_code", None)
        etype = getattr(exc, "type", None)
        request_id = getattr(exc, "request_id", None)
        log.warning("Claude request failed: %s (HTTP %s, %s, request id %s)",
                    type(exc).__name__, status, etype, request_id)
        detail = f"anthropic {type(exc).__name__} {status or ''} {etype or ''}".strip()
        model = self.model
        if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)) \
                or etype in ("authentication_error", "permission_error"):
            return ProviderAuthError(detail, user_message="Ключ Anthropic не підходить або не має доступу "
                                                          "до цієї моделі. Перевір його в налаштуваннях.")
        if status == 402 or etype == "billing_error":
            return ProviderAuthError(detail, user_message="Проблема з оплатою в акаунті Anthropic.")
        if isinstance(exc, anthropic.RateLimitError) or etype == "rate_limit_error":
            response = getattr(exc, "response", None)
            retry_after = parse_retry_after(response.headers.get("retry-after")) if response is not None \
                else None
            return ProviderRateLimitError(detail, retry_after=retry_after,
                                          user_message="Забагато запитів до Claude — спробую трохи згодом.")
        if isinstance(exc, anthropic.NotFoundError) or etype == "not_found_error":
            return ProviderRequestError(detail, user_message=f"Модель «{model}» недоступна. "
                                                             "Вибери іншу в налаштуваннях AI.")
        if isinstance(exc, (anthropic.BadRequestError, anthropic.UnprocessableEntityError,
                            anthropic.RequestTooLargeError)) \
                or etype in ("invalid_request_error", "request_too_large"):
            return ProviderRequestError(detail, user_message="Claude відхилив запит (неправильні параметри).")
        if isinstance(exc, anthropic.APITimeoutError):
            return ProviderUnavailableError(detail, user_message="Claude не відповів вчасно.")
        if isinstance(exc, anthropic.APIConnectionError):
            return ProviderUnavailableError(detail,
                                            user_message="Немає з'єднання з Claude. Перевір інтернет.")
        if isinstance(exc, (anthropic.InternalServerError, anthropic.OverloadedError,
                            anthropic.ServiceUnavailableError, anthropic.DeadlineExceededError,
                            anthropic.ConflictError)) \
                or etype in ("overloaded_error", "api_error", "timeout_error") \
                or (isinstance(status, int) and (status >= 500 or status == 408)):
            return ProviderUnavailableError(
                detail, user_message="Claude зараз перевантажений або недоступний. Спробую пізніше.")
        if isinstance(exc, anthropic.APIStatusError):
            return ProviderRequestError(detail, user_message=f"Claude відхилив запит (код {status}).")
        if isinstance(exc, anthropic.CredentialsError):
            # No key, and no usable `ant auth` profile / workload identity either.
            return ProviderAuthError(detail, user_message=_NO_KEY)
        return ProviderUnavailableError(detail, user_message="Claude зараз недоступний.")
