"""OllamaProvider (httpx2) against MockTransport serving /api/chat (JSON + NDJSON), /api/tags, /api/show."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator

import httpx2
import pytest

from chibi.ai.providers.ollama_provider import OllamaProvider
from chibi.ai.types import (
    CancelledError,
    CancelToken,
    ChatTurn,
    GenerationRequest,
    ImageInput,
    ProviderRequestError,
    ProviderUnavailableError,
    Purpose,
    Role,
    TextProvider,
    VisionProvider,
)

TAGS = {"models": [
    {"name": "nomic-embed-text:latest", "model": "nomic-embed-text:latest", "size": 1},
    {"name": "gemma3:4b", "model": "gemma3:4b", "size": 3_300_000_000},
    {"name": "llama3.2:3b", "model": "llama3.2:3b", "size": 2_000_000_000},
]}


class Server:
    """Routes by path; each route has a list of scripted responses."""

    def __init__(self, **routes) -> None:
        self.routes = {"/" + path.replace("_", "/"): list(responses) for path, responses in routes.items()}
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        responses = self.routes[request.url.path]
        nxt = responses.pop(0) if len(responses) > 1 else responses[0]
        return nxt(request) if callable(nxt) else nxt

    def bodies(self, path: str) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]


def make(server: Server, model: str = "gemma3:4b", url: str = "http://localhost:11434") -> OllamaProvider:
    return OllamaProvider(url, model, http_client=httpx2.Client(transport=httpx2.MockTransport(server)))


def chat_json(text: str, **extra) -> httpx2.Response:
    return httpx2.Response(200, json={
        "model": "gemma3:4b", "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": text}, "done": True,
        "done_reason": "stop", "prompt_eval_count": 26, "eval_count": 12, **extra})


def ndjson(pieces: list[str], done_reason: str = "stop") -> list[bytes]:
    lines = [json.dumps({"model": "gemma3:4b", "message": {"role": "assistant", "content": p}, "done": False},
                        ensure_ascii=False) + "\n" for p in pieces]
    lines.append(json.dumps({
        "model": "gemma3:4b", "message": {"role": "assistant", "content": ""}, "done": True,
        "done_reason": done_reason, "prompt_eval_count": 31, "eval_count": len(pieces)}) + "\n")
    return [line.encode() for line in lines]


def req(text: str = "Привіт", purpose: Purpose = Purpose.CHAT, **kwargs) -> GenerationRequest:
    return GenerationRequest(system="Ти — Мія.", turns=(ChatTurn(Role.USER, text),), purpose=purpose,
                             **kwargs)


def test_chat_body_and_result():
    server = Server(api_chat=[chat_json("Привіт-привіт!")])
    p = make(server)
    assert isinstance(p, TextProvider) and isinstance(p, VisionProvider)
    assert p.name == "ollama" and p.is_local is True
    result = p.generate(req(context="Настрій: весела.", max_tokens=123))
    body = server.bodies("/api/chat")[0]
    # The volatile operator context opens the last user turn (prefix reuse), never a system message.
    assert body == {
        "model": "gemma3:4b",
        "messages": [{"role": "system", "content": "Ти — Мія."},
                     {"role": "user", "content": "Настрій: весела.\n\nПривіт"}],
        "stream": False,
        "options": {"num_predict": 123},
    }
    assert result.text == "Привіт-привіт!" and result.model == "gemma3:4b"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (26, 12)
    assert result.stop_reason == "end_turn"


def test_images_and_json_format():
    server = Server(api_chat=[chat_json('```json\n{"ok": true}\n```')])
    request = GenerationRequest(system="s", purpose=Purpose.VISION, json_mode=True, turns=(
        ChatTurn(Role.USER, "Що тут?", images=(ImageInput(b"\x89PNG-bytes"),)),))
    result = make(server).generate(request)
    body = server.bodies("/api/chat")[0]
    assert body["format"] == "json"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"].startswith("Reply with exactly one JSON object")
    assert body["messages"][1]["content"].endswith("\n\nЩо тут?")
    assert body["messages"][1]["images"] == [base64.b64encode(b"\x89PNG-bytes").decode()]
    assert json.loads(result.text) == {"ok": True}


def test_ndjson_stream():
    server = Server(api_chat=[httpx2.Response(200, content=b"".join(ndjson(["При", "віт", "!"], "length")))])
    pieces: list[str] = []
    result = make(server).stream(req(), pieces.append)
    assert server.bodies("/api/chat")[0]["stream"] is True
    assert pieces == ["При", "віт", "!"]
    assert result.text == "Привіт!" and result.stop_reason == "max_tokens"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (31, 3)


def test_stream_cancellation():
    produced = {"n": 0}

    def body() -> Iterator[bytes]:
        for line in ndjson([f"w{i} " for i in range(500)]):
            produced["n"] += 1
            yield line

    server = Server(api_chat=[lambda r: httpx2.Response(200, content=body())])
    token = CancelToken()
    seen: list[str] = []

    def on_delta(piece: str) -> None:
        seen.append(piece)
        token.cancel()

    with pytest.raises(CancelledError):
        make(server).stream(req(), on_delta, token)
    assert seen == ["w0 "] and produced["n"] < 10


def test_stream_error_line():
    lines = [*ndjson(["Хм"])[:1], b'{"error": "model runner has unexpectedly stopped"}\n']
    server = Server(api_chat=[httpx2.Response(200, content=b"".join(lines))])
    with pytest.raises(ProviderUnavailableError):
        make(server).stream(req(), lambda p: None)


def test_empty_model_resolves_to_first_installed_chat_model():
    server = Server(api_tags=[httpx2.Response(200, json=TAGS)], api_chat=[chat_json("ok")])
    p = make(server, model="")
    p.generate(req())
    assert p.model == "gemma3:4b"
    assert server.bodies("/api/chat")[0]["model"] == "gemma3:4b"
    p.generate(req())
    assert [r.url.path for r in server.requests].count("/api/tags") == 1  # resolved once


def test_no_models_installed_hints_at_pull():
    server = Server(api_tags=[httpx2.Response(200, json={"models": []})])
    with pytest.raises(ProviderRequestError) as err:
        make(server, model="").generate(req())
    assert "ollama pull" in err.value.user_message


def test_model_not_found_hints_at_pull():
    missing = {"error": "model \"qwen3:8b\" not found, try pulling it first"}
    server = Server(api_chat=[httpx2.Response(404, json=missing)])
    with pytest.raises(ProviderRequestError) as err:
        make(server, model="qwen3:8b").stream(req(), lambda p: None)
    assert "ollama pull qwen3:8b" in err.value.user_message


def test_connection_refused_means_not_running():
    def refused(request):
        raise httpx2.ConnectError("[Errno 111] Connection refused", request=request)

    server = Server(api_chat=[refused], api_tags=[refused])
    with pytest.raises(ProviderUnavailableError) as err:
        make(server).generate(req())
    assert str(err.value) == "Ollama не запущена" and "Ollama не запущена" in err.value.user_message
    with pytest.raises(ProviderUnavailableError):
        make(server).list_models()


def test_read_timeout_and_server_error():
    def slow(request):
        raise httpx2.ReadTimeout("timed out", request=request)

    with pytest.raises(ProviderUnavailableError) as err:
        make(Server(api_chat=[slow])).generate(req())
    assert "вчасно" in err.value.user_message
    with pytest.raises(ProviderUnavailableError):
        make(Server(api_chat=[httpx2.Response(500, json={"error": "out of memory"})])).generate(req())
    with pytest.raises(ProviderRequestError):
        make(Server(api_chat=[httpx2.Response(400, json={"error": "bad"})])).generate(req())


def test_list_models():
    server = Server(api_tags=[httpx2.Response(200, json=TAGS)])
    assert make(server).list_models() == ["nomic-embed-text:latest", "gemma3:4b", "llama3.2:3b"]


def test_supports_vision_is_cached_and_false_on_error():
    server = Server(api_show=[httpx2.Response(200, json={"capabilities": ["completion", "vision"]})])
    p = make(server)
    assert p.supports_vision is True
    assert p.supports_vision is True
    shows = server.bodies("/api/show")
    assert shows == [{"model": "gemma3:4b"}]

    text_only = Server(api_show=[httpx2.Response(200, json={"capabilities": ["completion"]})])
    assert make(text_only, model="llama3.2:3b").supports_vision is False

    def refused(request):
        raise httpx2.ConnectError("refused", request=request)

    failing = Server(api_show=[refused, httpx2.Response(200, json={"capabilities": ["vision"]})])
    p = make(failing)
    assert p.supports_vision is False  # not running: unknown -> False, not cached
    assert p.supports_vision is True


def test_supports_vision_on_old_ollama_without_capabilities():
    server = Server(api_show=[httpx2.Response(200, json={"details": {"families": ["llama", "clip"]}})])
    assert make(server, model="llava").supports_vision is True


@pytest.mark.parametrize("url,local", [
    ("http://localhost:11434", True), ("http://127.0.0.1:11434", True),
    ("http://192.168.0.10:11434", False), ("https://ollama.example.com", False),
])
def test_is_local_by_host(url, local):
    assert make(Server(), url=url).is_local is local


def test_request_model_override_skips_model_resolution():
    server = Server(api_chat=[chat_json("ok")])
    p = make(server, model="")
    result = p.generate(req(model="llava:7b"))
    assert server.bodies("/api/chat")[0]["model"] == "llava:7b" and result.model == "llava:7b"
    assert not server.bodies("/api/tags") and p.model == ""


def test_reasoning_blocks_are_removed():
    server = Server(api_chat=[httpx2.Response(200, content=b"".join(
        ndjson(["<think>", "hmm, a greeting", "</think>", "\n\nПривіт!"]))),
        chat_json("<think>hmm</think>\nПривіт!")])
    p = make(server)
    pieces: list[str] = []
    assert p.stream(req(), pieces.append).text == "Привіт!"
    assert "".join(pieces) == "Привіт!"
    assert p.generate(req()).text == "Привіт!"


# -- keeping the model loaded, prompt layout, timing ------------------------------------------------


def test_keep_alive_goes_with_every_request_and_none_leaves_ollamas_default():
    lines = b"".join(ndjson(["ok"]))
    server = Server(api_chat=[chat_json("ok"), httpx2.Response(200, content=lines)],
                    api_generate=[httpx2.Response(200, json={"model": "gemma3:4b", "done": True,
                                                             "done_reason": "load"})])
    p = make(server)
    p.keep_alive = "45m"
    p.generate(req())
    p.stream(req(), lambda piece: None)
    p.warm_up()
    assert [b["keep_alive"] for b in server.bodies("/api/chat")] == ["45m", "45m"]
    assert server.bodies("/api/generate") == [{"model": "gemma3:4b", "keep_alive": "45m"}]
    default = Server(api_chat=[chat_json("ok")])
    make(default).generate(req())
    assert "keep_alive" not in default.bodies("/api/chat")[0]
    assert OllamaProvider("http://localhost:11434", "m", keep_alive="30m").keep_alive == "30m"


def test_warm_up_loads_without_a_prompt_and_reports_the_load_time():
    server = Server(api_tags=[httpx2.Response(200, json=TAGS)],
                    api_generate=[httpx2.Response(200, json={"model": "gemma3:4b", "done": True,
                                                             "load_duration": 4_250_000_000})])
    p = make(server, model="")
    p.warm_up(CancelToken())
    body = server.bodies("/api/generate")[0]
    assert body == {"model": "gemma3:4b"}  # no prompt, no messages: nothing is generated
    assert p.last_load_seconds == pytest.approx(4.25)


def test_warm_up_errors_are_the_usual_ones():
    missing = Server(api_generate=[httpx2.Response(404, json={"error": "model not found"})])
    with pytest.raises(ProviderRequestError) as err:
        make(missing, model="qwen3:8b").warm_up()
    assert "ollama pull qwen3:8b" in err.value.user_message

    def refused(request):
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(ProviderUnavailableError):
        make(Server(api_generate=[refused])).warm_up()
    token = CancelToken()
    token.cancel()
    with pytest.raises(CancelledError):
        make(Server(api_generate=[refused])).warm_up(token)


def test_where_the_time_went_is_read_from_the_final_object():
    final = {"load_duration": 2_000_000_000, "prompt_eval_duration": 1_500_000_000}
    server = Server(api_chat=[chat_json("ok", **final), chat_json("ok")])
    p = make(server)
    p.generate(req())
    assert (p.last_load_seconds, p.last_prompt_seconds) == (2.0, 1.5)
    p.generate(req())  # not reported: unknown, not the previous request's numbers
    assert (p.last_load_seconds, p.last_prompt_seconds) == (None, None)
    lines = [*ndjson(["a"])[:-1], json.dumps({"done": True, "done_reason": "stop", "load_duration": 0,
                                               "prompt_eval_duration": 250_000_000}).encode() + b"\n"]
    streamed = make(Server(api_chat=[httpx2.Response(200, content=b"".join(lines))]))
    streamed.stream(req(), lambda piece: None)
    assert (streamed.last_load_seconds, streamed.last_prompt_seconds) == (0.0, 0.25)


def test_operator_context_opens_the_last_user_turn_so_the_prefix_stays_the_same():
    history = (ChatTurn(Role.USER, "<user>Привіт</user>"),
               ChatTurn(Role.ASSISTANT, "<<state=talk>>\nПривіт!"))

    def request(context: str, text: str) -> GenerationRequest:
        return GenerationRequest(system="Ти — Мія.", purpose=Purpose.CHAT, context=context,
                                 turns=(*history, ChatTurn(Role.USER, text)))

    server = Server(api_chat=[chat_json("ok")])
    p = make(server)
    p.generate(request("Зараз 10:01, настрій: весела.", "<user>Як справи?</user>"))
    p.generate(request("Зараз 10:02, настрій: сонна.", "<user>А погода?</user>"))
    first, second = (body["messages"] for body in server.bodies("/api/chat"))
    assert first[:3] == second[:3]  # system prompt and history: an unchanged prefix
    assert first[0] == {"role": "system", "content": "Ти — Мія."}
    assert [m["role"] for m in first] == ["system", "user", "assistant", "user"]
    assert first[3]["content"] == "Зараз 10:01, настрій: весела.\n\n<user>Як справи?</user>"
    assert not any(m["role"] == "system" for m in first[1:])


def test_an_answer_that_arrived_complete_is_returned_although_cancelled_meanwhile():
    """The answer was read in full before the cancellation was looked at: it is returned
    (the AI service delivers it to a caller merely put aside for a conversation instead of
    computing it again, and records it either way)."""
    token = CancelToken()

    def answer(request: httpx2.Request) -> httpx2.Response:
        token.cancel()  # e.g. the user started talking while the last bytes arrived
        return chat_json("Ось.", load_duration=1_000_000_000)

    p = make(Server(api_chat=[answer]))
    result = p.generate(req(), token)
    assert result.text == "Ось." and p.last_load_seconds == 1.0
