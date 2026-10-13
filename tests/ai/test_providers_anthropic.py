"""AnthropicProvider against the real SDK, with httpx2.MockTransport serving
Messages API wire format (JSON errors and SSE streams)."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Callable, Iterator

import anthropic
import httpx2
import pytest

from chibi.ai.providers.anthropic_provider import (
    CHAT_MAX_TOKENS,
    FALLBACK_BETA,
    AnthropicProvider,
    supports_effort,
    uses_fallbacks,
)
from chibi.ai.types import (
    CancelledError,
    CancelToken,
    ChatTurn,
    GenerationRequest,
    ImageInput,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderRequestError,
    ProviderUnavailableError,
    Purpose,
    Role,
    TextProvider,
    VisionProvider,
)

SYSTEM = "Ти — Мія, чібі-компаньйонка."
SECRET_CHAT = "мій пароль від банку 12345"


# -- wire-format helpers ----------------------------------------------------------------------


def message_start(model: str = "claude-opus-5-5", input_tokens: int = 25, cache_read: int | None = 100,
                  cache_write: int | None = 0) -> dict:
    usage: dict = {"input_tokens": input_tokens, "output_tokens": 1}
    if cache_read is not None:
        usage["cache_read_input_tokens"] = cache_read
    if cache_write is not None:
        usage["cache_creation_input_tokens"] = cache_write
    return {"type": "message_start", "message": {
        "id": "msg_01", "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": usage}}


def text_events(pieces: list[str], *, stop_reason: str = "end_turn", model: str = "claude-opus-5-5",
                output_tokens: int = 9, stop_details: dict | None = None, **start_kwargs) -> list[dict]:
    events = [message_start(model, **start_kwargs),
              {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
              {"type": "ping"}]
    events += [{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}}
               for piece in pieces]
    delta: dict = {"stop_reason": stop_reason, "stop_sequence": None}
    if stop_details is not None:
        delta["stop_details"] = stop_details
    events += [{"type": "content_block_stop", "index": 0},
               {"type": "message_delta", "delta": delta, "usage": {"output_tokens": output_tokens}},
               {"type": "message_stop"}]
    return events


def sse(events: list[dict]) -> bytes:
    frames = (f"event: {e['type']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events)
    return "".join(frames).encode()


def sse_response(events: list[dict]) -> httpx2.Response:
    return httpx2.Response(200, headers={"content-type": "text/event-stream", "request-id": "req_ok"},
                           content=sse(events))


def error_response(status: int, etype: str, headers: dict | None = None) -> httpx2.Response:
    return httpx2.Response(status, headers={"request-id": "req_err", **(headers or {})},
                           json={"type": "error", "error": {"type": etype, "message": f"echo: {SECRET_CHAT}"},
                                 "request_id": "req_err"})


class Server:
    """Records requests; answers with the next scripted response (or a factory)."""

    def __init__(self, *responses: httpx2.Response | Callable[[httpx2.Request], httpx2.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        nxt = self.responses.pop(0)
        return nxt(request) if callable(nxt) else nxt

    @property
    def body(self) -> dict:
        return json.loads(self.requests[-1].content)

    @property
    def headers(self) -> httpx2.Headers:
        return self.requests[-1].headers


def provider(server: Server, model: str = "claude-opus-5-5", **kwargs) -> AnthropicProvider:
    client = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(server))
    return AnthropicProvider("sk-ant-test", model, http_client=client, max_retries=0, **kwargs)


def chat(text: str = "Привіт!", purpose: Purpose = Purpose.CHAT, **kwargs) -> GenerationRequest:
    return GenerationRequest(system=SYSTEM, turns=(ChatTurn(Role.USER, text),), purpose=purpose, **kwargs)


# -- request shape ----------------------------------------------------------------------------


def test_is_a_text_and_vision_provider():
    p = provider(Server())
    assert isinstance(p, TextProvider) and isinstance(p, VisionProvider)
    assert p.name == "anthropic" and p.is_local is False and p.supports_vision is True
    assert "sk-ant-test" not in repr(p)


def test_default_model_is_opus_5_5():
    server = Server(sse_response(text_events(["ok"])))
    client = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(server))
    p = AnthropicProvider("k", http_client=client, max_retries=0)
    assert p.model == "claude-opus-5-5"
    p.generate(chat())
    assert server.body["model"] == "claude-opus-5-5"


def test_request_body_cached_system_context_and_no_sampling_or_thinking():
    server = Server(sse_response(text_events(["Привіт", ", друже!"])))
    p = provider(server, effort="medium")
    result = p.generate(chat(purpose=Purpose.REACTION, context="Настрій: весела. Ранок.", max_tokens=300))
    body = server.body
    assert body["model"] == "claude-opus-5-5"
    assert body["system"] == [
        {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Настрій: весела. Ранок."},
    ]
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Привіт!"}]}]
    assert body["max_tokens"] == 300
    assert body["output_config"] == {"effort": "medium"}
    assert body["stream"] is True
    for forbidden in ("temperature", "top_p", "top_k", "thinking"):
        assert forbidden not in body
    assert server.headers["x-api-key"] == "sk-ant-test"
    assert result.text == "Привіт, друже!"
    assert result.provider == "anthropic" and result.model == "claude-opus-5-5"
    assert result.stop_reason == "end_turn"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (25, 9)
    assert (result.usage.cache_read_tokens, result.usage.cache_write_tokens) == (100, 0)


def test_request_effort_override_and_invalid_effort():
    server = Server(sse_response(text_events(["a"])), sse_response(text_events(["b"])))
    p = provider(server, effort="bogus")
    assert p.effort == "low"
    p.generate(chat(effort="high"))
    assert server.body["output_config"] == {"effort": "high"}
    p.generate(chat(effort="extreme"))
    assert server.body["output_config"] == {"effort": "low"}


@pytest.mark.parametrize("model,fallbacks", [
    ("claude-opus-5-5", True), ("claude-opus-5", True), ("claude-fable-5-1", True),
    ("claude-sonnet-5-5", True),
    ("claude-haiku-5-5", False), ("claude-sonnet-5", False), ("claude-opus-4-8", False),
    ("claude-sonnet-4-6", False), ("claude-haiku-4-5", False),
])
def test_server_side_fallbacks_only_for_supported_models(model, fallbacks):
    server = Server(sse_response(text_events(["ok"], model=model)))
    provider(server, model).generate(chat())
    request = server.requests[0]
    if fallbacks:
        assert request.url.path == "/v1/messages" and request.url.params.get("beta") == "true"
        assert request.headers["anthropic-beta"] == FALLBACK_BETA
        assert server.body["fallbacks"] == "default"
    else:
        assert "anthropic-beta" not in request.headers
        assert "fallbacks" not in server.body
    assert "betas" not in server.body
    assert uses_fallbacks(model) is fallbacks


@pytest.mark.parametrize("model,effort", [
    ("claude-opus-5-5", True), ("claude-sonnet-5-5", True), ("claude-haiku-5-5", True),
    ("claude-fable-5-1", True),
    ("claude-opus-5", True), ("claude-sonnet-5", True), ("claude-opus-4-8", True), ("claude-opus-4-7", True),
    ("claude-opus-4-6", True), ("claude-sonnet-4-6", True), ("claude-haiku-4-5", False),
    ("claude-3-5-sonnet-latest", False),
])
def test_effort_only_where_supported(model, effort):
    server = Server(sse_response(text_events(["ok"], model=model)))
    provider(server, model).generate(chat())
    assert ("output_config" in server.body) is effort
    assert supports_effort(model) is effort


def test_chat_gets_room_for_thinking_lines_keep_their_cap():
    server = Server(sse_response(text_events(["a"])), sse_response(text_events(["b"])))
    p = provider(server)
    p.generate(chat())
    assert server.body["max_tokens"] == CHAT_MAX_TOKENS == 1500
    p.generate(chat(purpose=Purpose.INITIATIVE))
    assert server.body["max_tokens"] == 600


def test_images_go_before_the_text_as_base64_blocks():
    png = b"\x89PNG\r\n\x1a\nfake-image-bytes"
    server = Server(sse_response(text_events(["Бачу котика"])))
    request = GenerationRequest(
        system=SYSTEM, purpose=Purpose.VISION,
        turns=(ChatTurn(Role.USER, "Що на екрані?", images=(ImageInput(png, "image/png"),
                                                           ImageInput(b"jpg", "image/jpeg"))),))
    provider(server).generate(request)
    content = server.body["messages"][0]["content"]
    assert content[0] == {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                      "data": base64.b64encode(png).decode()}}
    assert content[1]["source"]["media_type"] == "image/jpeg"
    assert content[2] == {"type": "text", "text": "Що на екрані?"}


def test_turns_are_normalised_for_the_api():
    server = Server(sse_response(text_events(["ok"])))
    request = GenerationRequest(system=SYSTEM, purpose=Purpose.CHAT, turns=(
        ChatTurn(Role.ASSISTANT, "Привіт! Я прокинулась."),  # she spoke first: dropped
        ChatTurn(Role.USER, "Привіт"),
        ChatTurn(Role.USER, "Як ти?"),  # merged with the previous user turn
        ChatTurn(Role.ASSISTANT, ""),  # empty: dropped
        ChatTurn(Role.ASSISTANT, "Чудово!"),
        ChatTurn(Role.USER, "А я втомився"),
        ChatTurn(Role.ASSISTANT, "Ой"),  # trailing assistant = prefill, rejected by the API: dropped
    ))
    provider(server).generate(request)
    assert server.body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Привіт"}, {"type": "text", "text": "Як ти?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Чудово!"}]},
        {"role": "user", "content": [{"type": "text", "text": "А я втомився"}]},
    ]


def test_request_without_user_turn_is_rejected_locally():
    server = Server()
    with pytest.raises(ProviderRequestError):
        provider(server).generate(GenerationRequest(system=SYSTEM, turns=(ChatTurn(Role.ASSISTANT, "x"),),
                                                    purpose=Purpose.CHAT))
    assert server.requests == []


def test_empty_system_and_json_mode():
    server = Server(sse_response(text_events(['```json\n{"mood": "весела",', ' "line": "Ура!"}\n```'])))
    request = GenerationRequest(system="", turns=(ChatTurn(Role.USER, "Онови пам'ять"),),
                                purpose=Purpose.MEMORY, json_mode=True)
    result = provider(server).generate(request)
    system = server.body["system"]
    assert len(system) == 1 and "JSON" in system[0]["text"] and "cache_control" not in system[0]
    assert json.loads(result.text) == {"mood": "весела", "line": "Ура!"}


def test_json_instruction_sits_after_the_cache_breakpoint():
    server = Server(sse_response(text_events(["{}"])))
    provider(server).generate(chat(purpose=Purpose.MEMORY, json_mode=True, context="ctx"))
    system = server.body["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"} and system[0]["text"] == SYSTEM
    assert system[1] == {"type": "text", "text": "ctx"}
    assert "JSON" in system[2]["text"] and "cache_control" not in system[2]


# -- streaming / cancellation -------------------------------------------------------------------


def test_stream_delivers_deltas_and_final_text():
    server = Server(sse_response(text_events(["При", "віт", "!"], output_tokens=3)))
    pieces: list[str] = []
    result = provider(server).stream(chat(), pieces.append)
    assert pieces == ["При", "віт", "!"]
    assert result.text == "Привіт!"
    assert result.usage.output_tokens == 3


def test_usage_missing_cache_fields_are_zero():
    server = Server(sse_response(text_events(["ok"], cache_read=None, cache_write=None)))
    result = provider(server, "claude-haiku-4-5").generate(chat())
    assert result.usage.cache_read_tokens == 0 and result.usage.cache_write_tokens == 0


def test_fallback_model_reported_and_fallback_block_ignored():
    events = [message_start("claude-opus-4-8"),
              {"type": "content_block_start", "index": 0, "content_block": {
                  "type": "fallback", "from": {"model": "claude-opus-5-5"},
                  "to": {"model": "claude-opus-4-8"}}},
              {"type": "content_block_stop", "index": 0},
              {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
              {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Тримай!"}},
              {"type": "content_block_stop", "index": 1},
              {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
               "usage": {"output_tokens": 4}},
              {"type": "message_stop"}]
    server = Server(sse_response(events))
    result = provider(server).generate(chat())
    assert result.text == "Тримай!"
    assert result.model == "claude-opus-4-8"


def test_cancellation_mid_stream_stops_reading():
    produced = {"events": 0}

    def body() -> Iterator[bytes]:
        events = text_events([f"слово{i} " for i in range(200)])
        for event in events:
            produced["events"] += 1
            yield sse([event])

    server = Server(lambda request: httpx2.Response(200, headers={"content-type": "text/event-stream"},
                                                    content=body()))
    token = CancelToken()
    pieces: list[str] = []

    def on_delta(piece: str) -> None:
        pieces.append(piece)
        if len(pieces) == 3:
            token.cancel()

    with pytest.raises(CancelledError):
        provider(server).stream(chat(), on_delta, token)
    assert len(pieces) == 3
    assert produced["events"] < 20  # the rest of the stream was never pulled


def test_cancelled_before_start_sends_nothing():
    server = Server()
    token = CancelToken()
    token.cancel()
    with pytest.raises(CancelledError):
        provider(server).generate(chat(), token)
    assert server.requests == []


# -- refusals and errors ------------------------------------------------------------------------


def test_refusal_is_checked_before_content():
    server = Server(sse_response(text_events(["Частковa відповідь"], stop_reason="refusal",
                                             stop_details={"type": "refusal", "category": "cyber",
                                                           "explanation": None})))
    with pytest.raises(ProviderRefusalError) as err:
        provider(server).generate(chat())
    assert err.value.user_message and not err.value.retryable


def test_refusal_with_empty_content():
    events = [message_start(),
              {"type": "message_delta", "delta": {"stop_reason": "refusal", "stop_sequence": None},
               "usage": {"output_tokens": 0}},
              {"type": "message_stop"}]
    with pytest.raises(ProviderRefusalError):
        provider(Server(sse_response(events))).stream(chat(), lambda piece: None)


@pytest.mark.parametrize("status,etype,error_class,retryable", [
    (401, "authentication_error", ProviderAuthError, False),
    (403, "permission_error", ProviderAuthError, False),
    (402, "billing_error", ProviderAuthError, False),
    (400, "invalid_request_error", ProviderRequestError, False),
    (404, "not_found_error", ProviderRequestError, False),
    (413, "request_too_large", ProviderRequestError, False),
    (429, "rate_limit_error", ProviderRateLimitError, True),
    (500, "api_error", ProviderUnavailableError, True),
    (503, "api_error", ProviderUnavailableError, True),
    (529, "overloaded_error", ProviderUnavailableError, True),
])
def test_http_errors_are_mapped(status, etype, error_class, retryable, caplog):
    server = Server(error_response(status, etype, {"retry-after": "7"} if status == 429 else None))
    with caplog.at_level(logging.DEBUG), pytest.raises(error_class) as err:
        provider(server).generate(chat(SECRET_CHAT))
    assert type(err.value) is error_class
    assert err.value.retryable is retryable
    assert err.value.user_message
    assert SECRET_CHAT not in str(err.value) and SECRET_CHAT not in err.value.user_message
    assert SECRET_CHAT not in caplog.text and "sk-ant-test" not in caplog.text
    if status == 429:
        assert err.value.retry_after == 7.0
    if status == 404:
        assert "claude-opus-5-5" in err.value.user_message


def test_rate_limit_without_retry_after():
    with pytest.raises(ProviderRateLimitError) as err:
        provider(Server(error_response(429, "rate_limit_error"))).generate(chat())
    assert err.value.retry_after is None


def test_overloaded_error_event_mid_stream():
    events = [message_start(),
              {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
              {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Хм"}}]
    body = sse(events) + b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error",' \
                         b'"message":"Overloaded"}}\n\n'
    server = Server(httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body))
    pieces: list[str] = []
    with pytest.raises(ProviderUnavailableError):
        provider(server).stream(chat(), pieces.append)
    assert pieces == ["Хм"]


@pytest.mark.parametrize("exc", [httpx2.ConnectError("refused"), httpx2.ReadTimeout("slow")])
def test_network_failures_are_unavailable(exc):
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise type(exc)(str(exc), request=request)

    with pytest.raises(ProviderUnavailableError) as err:
        provider(Server(handler)).generate(chat())
    assert err.value.retryable


@pytest.mark.parametrize("explicit_config_dir", [True, False])
def test_missing_credentials_map_to_auth_error(monkeypatch, tmp_path, explicit_config_dir):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    if explicit_config_dir:  # the SDK raises CredentialsError
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "no-profiles"))
    else:  # the SDK raises TypeError("Could not resolve authentication method...")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    server = Server(sse_response(text_events(["never"])))
    with pytest.raises(ProviderAuthError) as err:
        p = AnthropicProvider(None, max_retries=0,
                              http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(server)))
        p.generate(chat())
    assert "ключ" in err.value.user_message.lower()
    assert server.requests == []


def test_env_key_is_used_when_no_key_is_stored(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    server = Server(sse_response(text_events(["ok"])))
    client = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(server))
    p = AnthropicProvider(None, http_client=client, max_retries=0)
    p.generate(chat())
    assert server.headers["x-api-key"] == "sk-from-env"


# -- models ---------------------------------------------------------------------------------------


def test_list_models():
    page = {"data": [
        {"type": "model", "id": "claude-opus-5-5", "display_name": "Claude Opus 5.5",
         "created_at": "2026-06-01T00:00:00Z"},
        {"type": "model", "id": "claude-haiku-5-5", "display_name": "Claude Haiku 5.5",
         "created_at": "2026-06-01T00:00:00Z"}],
        "has_more": False, "first_id": "claude-opus-5-5", "last_id": "claude-haiku-5-5"}
    server = Server(httpx2.Response(200, json=page))
    assert provider(server).list_models() == ["claude-opus-5-5", "claude-haiku-5-5"]
    assert server.requests[0].url.path == "/v1/models"


def test_list_models_auth_error():
    with pytest.raises(ProviderAuthError):
        provider(Server(error_response(401, "authentication_error"))).list_models()


def test_request_model_override_drives_model_effort_and_fallbacks():
    server = Server(sse_response(text_events(["a"])), sse_response(text_events(["b"])))
    p = provider(server, model="claude-opus-5-5", effort="medium")
    p.generate(chat(purpose=Purpose.VISION, model="claude-haiku-5-5"))
    # Effort and fallbacks follow the model actually requested, not the configured one.
    assert server.body["model"] == "claude-haiku-5-5"
    assert server.body["output_config"] == {"effort": "medium"}
    assert "fallbacks" not in server.body
    p.generate(chat(model="  "))
    assert server.body["model"] == "claude-opus-5-5" and server.body["fallbacks"] == "default"
    assert p.model == "claude-opus-5-5"
