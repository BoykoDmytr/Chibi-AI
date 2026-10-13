"""OpenAIProvider (httpx2) against MockTransport serving /chat/completions JSON and SSE."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator

import httpx2
import pytest

from chibi.ai.providers.openai_provider import (
    OpenAIProvider,
    clean_json_text,
    is_loopback_url,
    is_private_network_url,
    parse_retry_after,
    quiet_http_loggers,
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
)

SYSTEM = "Ти — Мія."
SECRET = "мій секретний текст"


class Server:
    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        nxt = self.responses.pop(0)
        return nxt(request) if callable(nxt) else nxt

    @property
    def body(self) -> dict:
        return json.loads(self.requests[-1].content)


def make(server: Server, base_url: str = "https://api.openai.com/v1", model: str = "gpt-5-mini",
         key: str | None = "sk-openai-test") -> OpenAIProvider:
    client = httpx2.Client(transport=httpx2.MockTransport(server))
    return OpenAIProvider(key, base_url, model, http_client=client)


def completion(text: str, finish: str = "stop", usage: dict | None = None, **message) -> httpx2.Response:
    return httpx2.Response(200, json={
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-5-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text, **message},
                     "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 40, "completion_tokens": 7, "total_tokens": 47},
    })


def sse_chunks(pieces: list[str], *, usage: bool = True, finish: str = "stop") -> list[bytes]:
    chunks = [{"id": "c", "object": "chat.completion.chunk", "choices": [
        {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}]
    chunks += [{"id": "c", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]} for p in pieces]
    chunks.append({"id": "c", "object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    if usage:
        chunks.append({"id": "c", "object": "chat.completion.chunk", "choices": [],
                       "usage": {"prompt_tokens": 30, "completion_tokens": 5,
                                 "prompt_tokens_details": {"cached_tokens": 10}}})
    out = [f"data: {json.dumps(c, ensure_ascii=False)}\n\n".encode() for c in chunks]
    out.append(b"data: [DONE]\n\n")
    return out


def sse_response(chunks: list[bytes]) -> httpx2.Response:
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=b"".join(chunks))


def req(text: str = "Привіт", purpose: Purpose = Purpose.CHAT, **kwargs) -> GenerationRequest:
    return GenerationRequest(system=SYSTEM, turns=(ChatTurn(Role.USER, text),), purpose=purpose, **kwargs)


# -- helpers ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("url,loopback", [
    ("http://localhost:11434", True), ("http://LOCALHOST:1234/v1", True), ("http://127.0.0.1:8080", True),
    ("http://127.8.9.10", True), ("http://[::1]:11434", True), ("http://[::ffff:127.0.0.1]:80", True),
    ("http://app.localhost", True),
    ("https://api.openai.com/v1", False), ("http://192.168.1.5:11434", False), ("http://10.0.0.2", False),
    ("http://localhost.evil.com", False), ("http://127.0.0.1.nip.io", False), ("", False),
    ("not a url", False),
])
def test_is_loopback_url(url, loopback):
    assert is_loopback_url(url) is loopback


@pytest.mark.parametrize("url,private", [
    ("http://localhost:1234", True), ("http://192.168.1.5:1234", True), ("http://10.1.2.3", True),
    ("http://172.16.0.9", True), ("http://studio.local:1234", True), ("http://[fd00::5]:80", True),
    ("https://api.openai.com/v1", False), ("https://openrouter.ai/api/v1", False), ("http://8.8.8.8", False),
])
def test_is_private_network_url(url, private):
    assert is_private_network_url(url) is private


def test_parse_retry_after():
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after(" 1.5 ") == 1.5
    assert parse_retry_after("-3") == 0.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after("nan") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # in the past
    assert parse_retry_after("99999999") == 3600.0


@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', '{"a": 1}'),
    ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ('```\n[1, 2]\n```', "[1, 2]"),
    ('Ось JSON: {"line": "Привіт {друже}"} - готово', '{"line": "Привіт {друже}"}'),
    ("{broken} then {\"ok\": true}", '{"ok": true}'),
    ("зовсім не json", "зовсім не json"),
])
def test_clean_json_text(raw, expected):
    assert clean_json_text(raw) == expected


def test_quiet_http_loggers_respects_explicit_configuration(monkeypatch):
    logger = logging.getLogger("httpcore2")
    monkeypatch.setattr(logger, "level", logging.DEBUG)
    quiet_http_loggers()
    assert logger.level == logging.DEBUG  # explicitly configured: left alone
    assert logging.getLogger("anthropic").getEffectiveLevel() >= logging.WARNING


# -- requests ---------------------------------------------------------------------------------------


def test_openai_body_headers_and_result():
    server = Server(completion("Привіт! Як справи?"))
    p = make(server)
    assert isinstance(p, TextProvider) and p.name == "openai" and p.is_local is False
    result = p.generate(req(context="Настрій: сонна."))
    request = server.requests[0]
    assert request.method == "POST" and str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-openai-test"
    body = server.body
    assert body["model"] == "gpt-5-mini"
    assert body["messages"] == [
        {"role": "system", "content": SYSTEM},
        {"role": "system", "content": "Настрій: сонна."},
        {"role": "user", "content": "Привіт"},
    ]
    assert body["max_completion_tokens"] == 600 and "max_tokens" not in body
    assert body["stream"] is False
    assert "response_format" not in body and "temperature" not in body
    assert result.text == "Привіт! Як справи?"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (40, 7)
    assert result.stop_reason == "end_turn" and result.provider == "openai"


def test_compatible_server_uses_max_tokens_and_no_auth_without_key():
    server = Server(completion("ok"))
    p = make(server, base_url="http://localhost:1234/v1/", model="qwen2.5-7b", key=None)
    assert p.is_local is True
    p.generate(req())
    assert str(server.requests[0].url) == "http://localhost:1234/v1/chat/completions"
    assert server.body["max_tokens"] == 600 and "max_completion_tokens" not in server.body
    assert "authorization" not in server.requests[0].headers


def test_env_key_only_for_openai_host(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-real-openai")
    server = Server(completion("ok"), completion("ok"))
    make(server, key=None).generate(req())
    assert server.requests[0].headers["authorization"] == "Bearer sk-env-real-openai"
    third = make(server, base_url="https://some-proxy.example.com/v1", key=None)
    assert third.has_key is False
    third.generate(req())
    assert "authorization" not in server.requests[1].headers  # the real key never goes elsewhere


def test_images_json_mode_and_history():
    server = Server(completion('```json\n{"seen": "кіт"}\n```'))
    request = GenerationRequest(system=SYSTEM, purpose=Purpose.VISION, json_mode=True, turns=(
        ChatTurn(Role.ASSISTANT, "Привіт!"),
        ChatTurn(Role.USER, "Що тут?", images=(ImageInput(b"png-bytes", "image/png"),)),
    ))
    result = make(server).generate(request)
    body = server.body
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][1]["role"] == "system" and "JSON" in body["messages"][1]["content"]
    assert body["messages"][2] == {"role": "assistant", "content": "Привіт!"}
    assert body["messages"][3] == {"role": "user", "content": [
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"png-bytes").decode()}},
        {"type": "text", "text": "Що тут?"},
    ]}
    assert json.loads(result.text) == {"seen": "кіт"}


def test_stream_sse_with_usage_and_done():
    server = Server(sse_response(sse_chunks(["При", "віт", "!"])))
    pieces: list[str] = []
    result = make(server).stream(req(), pieces.append)
    body = server.body
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert pieces == ["При", "віт", "!"]
    assert result.text == "Привіт!"
    assert result.usage.input_tokens == 20 and result.usage.cache_read_tokens == 10
    assert result.usage.output_tokens == 5


def test_stream_on_compatible_server_has_no_stream_options():
    server = Server(sse_response(sse_chunks(["ok"], usage=False, finish="length")))
    result = make(server, base_url="http://127.0.0.1:8080/v1", model="local").stream(req(), lambda p: None)
    assert "stream_options" not in server.body
    assert result.text == "ok" and result.stop_reason == "max_tokens"
    assert result.usage.input_tokens == 0


def test_stream_ignores_comments_and_event_lines():
    raw = [b": keep-alive\n\n", b"event: message\n", *sse_chunks(["a", "b"], usage=False)]
    server = Server(sse_response(raw))
    assert make(server).stream(req(), lambda p: None).text == "ab"


def test_stream_cancellation_stops_reading():
    produced = {"n": 0}

    def body() -> Iterator[bytes]:
        for chunk in sse_chunks([f"w{i} " for i in range(300)]):
            produced["n"] += 1
            yield chunk

    server = Server(lambda r: httpx2.Response(200, headers={"content-type": "text/event-stream"},
                                              content=body()))
    token = CancelToken()
    seen: list[str] = []

    def on_delta(piece: str) -> None:
        seen.append(piece)
        if len(seen) == 2:
            token.cancel()

    with pytest.raises(CancelledError):
        make(server).stream(req(), on_delta, token)
    assert len(seen) == 2 and produced["n"] < 10


def test_stream_error_chunk():
    error = b'data: {"error": {"message": "boom", "code": "server_error"}}\n\n'
    raw = [*sse_chunks(["Почала"], usage=False)[:2], error]
    with pytest.raises(ProviderUnavailableError):
        make(Server(sse_response(raw))).stream(req(), lambda p: None)


def test_refusals():
    with pytest.raises(ProviderRefusalError):
        make(Server(completion("", refusal="I can't help with that."))).generate(req())
    with pytest.raises(ProviderRefusalError):
        make(Server(completion("", finish="content_filter"))).generate(req())


@pytest.mark.parametrize("status,error_class", [
    (401, ProviderAuthError), (403, ProviderAuthError), (402, ProviderAuthError),
    (400, ProviderRequestError), (404, ProviderRequestError), (422, ProviderRequestError),
    (429, ProviderRateLimitError), (500, ProviderUnavailableError), (503, ProviderUnavailableError),
    (408, ProviderUnavailableError),
])
def test_status_mapping(status, error_class, caplog):
    payload = {"error": {"message": f"echo {SECRET}", "type": "x", "code": "model_not_found"}}
    error = httpx2.Response(status, headers={"retry-after": "3"}, json=payload)
    with caplog.at_level(logging.DEBUG), pytest.raises(error_class) as err:
        make(Server(error)).generate(req(SECRET))
    assert type(err.value) is error_class and err.value.user_message
    assert SECRET not in str(err.value) and SECRET not in caplog.text and "sk-openai-test" not in caplog.text
    if status == 429:
        assert err.value.retry_after == 3.0


def test_stream_status_error():
    with pytest.raises(ProviderAuthError):
        make(Server(httpx2.Response(401, json={"error": {"type": "invalid_api_key"}}))).stream(req(), print)


@pytest.mark.parametrize("exc_type", [httpx2.ConnectError, httpx2.ReadTimeout, httpx2.RemoteProtocolError])
def test_transport_errors(exc_type):
    def handler(request):
        raise exc_type("x", request=request)

    with pytest.raises(ProviderUnavailableError) as err:
        make(Server(handler)).generate(req())
    assert err.value.retryable


def test_html_instead_of_api_is_a_request_error():
    page = httpx2.Response(200, headers={"content-type": "text/html"}, content=b"<html>router login</html>")
    with pytest.raises(ProviderRequestError):
        make(Server(page)).generate(req())


def test_missing_model_and_user_turn():
    with pytest.raises(ProviderRequestError):
        make(Server(), model="").generate(req())
    with pytest.raises(ProviderRequestError):
        make(Server()).generate(GenerationRequest(system="s", turns=(), purpose=Purpose.CHAT))


def test_list_models():
    server = Server(httpx2.Response(200, json={"object": "list", "data": [
        {"id": "gpt-5-mini", "object": "model"}, {"id": "gpt-4.1", "object": "model"}]}))
    assert make(server).list_models() == ["gpt-4.1", "gpt-5-mini"]
    assert server.requests[0].method == "GET" and server.requests[0].url.path == "/v1/models"


def test_repr_hides_key():
    assert "sk-openai-test" not in repr(make(Server()))
