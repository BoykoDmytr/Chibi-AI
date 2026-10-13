"""Provider-neutral types for the AI layer (Phase 2+).

Every provider (Anthropic, OpenAI-compatible, Ollama) takes a
:class:`GenerationRequest` and returns a :class:`GenerationResult` or raises
one of the :class:`ProviderError` subclasses, so the brain, the budget and the
UI never see provider-specific shapes.

Trust boundary - read before adding fields:

* ``system`` is ours: the stable persona prompt.  Never put anything derived
  from the screen, window titles, app names, chat text or memories into it.
* ``context`` is ours too: volatile *operator-side* state (her mood, time of
  day, needs).  Providers place it after the cached system prompt.
* Everything that came from outside (the user's words, an app name, a vision
  observation, a stored memory) travels inside ``turns`` as user content,
  wrapped in tags the system prompt declares to be data, not instructions.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Purpose(StrEnum):
    """Why a request is made; drives queue priority and usage accounting.

    Ordered from most to least important: a chat reply the user is waiting
    for beats a reaction line, which beats autonomous chatter, background
    memory extraction and vision analysis.
    """

    CHAT = "chat"
    REACTION = "reaction"
    INITIATIVE = "initiative"
    VISION = "vision"
    MEMORY = "memory"

    @property
    def rank(self) -> int:
        return list(Purpose).index(self)


@dataclass(frozen=True, slots=True)
class ImageInput:
    """An image for vision requests (already privacy-checked and downscaled)."""

    data: bytes
    media_type: str = "image/png"


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: Role
    text: str
    images: tuple[ImageInput, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    system: str
    turns: tuple[ChatTurn, ...]
    purpose: Purpose
    max_tokens: int = 600
    #: Operator-side volatile state; placed after the cached system prompt.
    context: str = ""
    #: Ask for a single JSON object as the whole reply.
    json_mode: bool = False
    #: Claude effort override ("low" | "medium" | "high"); None = settings.
    effort: str | None = None
    #: Model for this request only (e.g. a cheaper one for screen vision); None = settings.
    model: str | None = None
    #: The request carries what she saw on the screen (a description, a frame): it runs
    #: only on the provider it was checked for, and a cloud one only with cloud vision.
    screen_data: bool = False


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""
    from_cache: bool = False


# -- errors ------------------------------------------------------------------------------


class ProviderError(Exception):
    """Base class.  ``retryable`` tells the runner whether a retry can help."""

    retryable = False

    def __init__(self, message: str = "", *, user_message: str = "") -> None:
        super().__init__(message)
        #: Short Ukrainian explanation suitable for the UI.
        self.user_message = user_message or message


class ProviderUnavailableError(ProviderError):
    """Network error, timeout, 5xx / overloaded, local server not running."""

    retryable = True


class ProviderRateLimitError(ProviderError):
    retryable = True

    def __init__(self, message: str = "", *, retry_after: float | None = None,
                 user_message: str = "") -> None:
        super().__init__(message, user_message=user_message)
        self.retry_after = retry_after


class ProviderAuthError(ProviderError):
    """Missing / invalid key or no permission (401/403)."""


class ProviderRequestError(ProviderError):
    """The request itself is wrong (400/404: unknown model, bad parameter)."""


class ProviderRefusalError(ProviderError):
    """The model declined (Claude ``stop_reason == "refusal"``, also after fallbacks)."""


class BudgetExceededError(Exception):
    """A request/day, request/hour or cost/day limit would be exceeded.
    ``hourly``: only the rolling hour is full (it frees up within the hour)."""

    def __init__(self, message: str, *, user_message: str = "", hourly: bool = False) -> None:
        super().__init__(message)
        self.user_message = user_message or message
        self.hourly = hourly


class CancelledError(Exception):
    """The request was cancelled (user interrupted, app shutting down)."""


class CancelToken:
    """Thread-safe cancellation flag checked by providers between stream chunks."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancelledError("cancelled")


# -- providers ---------------------------------------------------------------------------


@runtime_checkable
class TextProvider(Protocol):
    """Text generation.  Implementations must be safe to call from a worker
    thread and must never touch Qt."""

    name: str  # "anthropic" | "openai" | "ollama" | "fake"
    model: str

    @property
    def is_local(self) -> bool:
        """True when no data leaves this machine (decides the CLOUD_AI check)."""
        ...

    def generate(self, request: GenerationRequest, cancel: CancelToken | None = None) -> GenerationResult: ...

    def stream(self, request: GenerationRequest, on_delta: Callable[[str], None],
               cancel: CancelToken | None = None) -> GenerationResult:
        """Like generate, calling *on_delta* (on the worker thread) with text pieces."""
        ...

    def list_models(self) -> list[str]: ...


@runtime_checkable
class VisionProvider(TextProvider, Protocol):
    """A provider/model that accepts :class:`ImageInput` in user turns."""

    @property
    def supports_vision(self) -> bool: ...
