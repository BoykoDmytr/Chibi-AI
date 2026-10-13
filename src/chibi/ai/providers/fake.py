"""Scripted provider for tests and the headless smoke test.

Replies are taken from ``replies`` in order; an ``Exception`` instance in the
script is raised instead of replying.  When the script runs out,
``default_reply`` is used (``None`` -> :class:`ProviderUnavailableError`).

Test hooks: ``gate`` (a :class:`threading.Event`) holds every call until it is
set - while waiting, cancellation is honoured, so tests can keep the worker
busy and fill the queue; ``started`` is set when a call begins.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable

from chibi.ai.types import (
    CancelledError,
    CancelToken,
    GenerationRequest,
    GenerationResult,
    ProviderUnavailableError,
    Usage,
)


class FakeProvider:
    name = "fake"

    def __init__(self, replies: Iterable[str | Exception] = (), *, model: str = "fake-model",
                 local: bool = True, models: Iterable[str] | None = None,
                 default_reply: str | None = "Привіт! Я тут.", gate: threading.Event | None = None,
                 vision: bool = True, models_error: Exception | None = None) -> None:
        self.model = model
        self.replies: list[str | Exception] = list(replies)
        self.requests: list[GenerationRequest] = []
        self.models = list(models) if models is not None else [model]
        self.default_reply = default_reply
        self.local = local
        self.vision = vision
        self.models_error = models_error
        self.gate = gate
        self.started = threading.Event()
        self.calls = 0
        self.closed = False
        self._lock = threading.Lock()

    @property
    def is_local(self) -> bool:
        return self.local

    @property
    def supports_vision(self) -> bool:
        return self.vision

    def _next(self, request: GenerationRequest, cancel: CancelToken | None) -> str:
        with self._lock:
            self.requests.append(request)
            self.calls += 1
            reply: str | Exception | None = self.replies.pop(0) if self.replies else self.default_reply
        self.started.set()
        if self.gate is not None:
            while not self.gate.wait(0.01):
                if cancel is not None and cancel.cancelled:
                    raise CancelledError("cancelled")
        if cancel is not None:
            cancel.raise_if_cancelled()
        if isinstance(reply, Exception):
            raise reply
        if reply is None:
            raise ProviderUnavailableError("fake script exhausted", user_message="Немає відповіді.")
        return reply

    def _result(self, request: GenerationRequest, text: str) -> GenerationResult:
        prompt = len(request.system) + len(request.context) + sum(len(t.text) for t in request.turns)
        return GenerationResult(
            text=text, provider=self.name, model=self.model,
            usage=Usage(input_tokens=max(1, prompt // 4), output_tokens=max(1, len(text) // 4)),
            stop_reason="end_turn",
        )

    def generate(self, request: GenerationRequest, cancel: CancelToken | None = None) -> GenerationResult:
        return self._result(request, self._next(request, cancel))

    def stream(self, request: GenerationRequest, on_delta: Callable[[str], None],
               cancel: CancelToken | None = None) -> GenerationResult:
        text = self._next(request, cancel)
        # Word-sized pieces (keeping the spaces) like a real stream.
        pieces = [w + " " for w in text.split(" ")]
        pieces[-1] = pieces[-1][:-1]
        for piece in pieces:
            if cancel is not None:
                cancel.raise_if_cancelled()
            if piece:
                on_delta(piece)
        return self._result(request, text)

    def list_models(self) -> list[str]:
        if self.models_error is not None:
            raise self.models_error
        return list(self.models)

    def close(self) -> None:
        self.closed = True
