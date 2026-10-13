"""Cancelling a request really stops it: against a real local HTTP server, a request
blocked waiting for the server (a model loading, a long prompt) is aborted at once and
the server sees the client go - also when a conversation preempts screen analysis."""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx2
import pytest

from chibi.ai.providers.ollama_provider import OllamaProvider
from chibi.ai.providers.openai_provider import OpenAIProvider, RequestAbort
from chibi.ai.service import ConversationBusyError, _JobToken
from chibi.ai.types import (
    CancelledError,
    CancelToken,
    ChatTurn,
    GenerationRequest,
    ProviderUnavailableError,
    Purpose,
    Role,
)
from chibi.core.events import ConversationActive
from tests.ai.test_service import Harness, pump

HOLD_SECONDS = 10.0
SLOW_SECONDS = 0.6


class _Handler(BaseHTTPRequestHandler):
    server: _SlowServer

    def log_message(self, format, *args) -> None:  # quiet
        pass

    def setup(self) -> None:
        super().setup()
        with self.server.lock:
            self.server.connections += 1

    def do_GET(self) -> None:
        self._serve(b"")

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        self._serve(self.rfile.read(length))

    def _serve(self, body: bytes) -> None:
        with self.server.lock:
            self.server.paths.append(self.path)
            self.server.bodies.append(json.loads(body) if body else None)
        plan = self.server.plan(self.path, json.loads(body) if body else None)
        if plan == "hold":
            self._hold()
            return
        if plan == "stall_stream":
            self.send_response(200)
            self.send_header("content-type", "application/x-ndjson")
            self.end_headers()
            self.wfile.write(b'{"message": {"content": "\\u0425\\u043c"}, "done": false}\n')
            self.wfile.flush()
            self._hold()
            return
        if isinstance(plan, tuple):  # ("slow", reply): the whole answer after a while
            time.sleep(SLOW_SECONDS)
            plan = plan[1]
        payload = json.dumps(plan).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _hold(self) -> None:
        """Wait (like a model reading a long prompt) until the client goes away."""
        started = time.monotonic()
        self.connection.settimeout(0.02)
        while time.monotonic() - started < HOLD_SECONDS:
            try:
                if self.connection.recv(1, socket.MSG_PEEK) == b"":
                    break
            except TimeoutError:
                continue
            except OSError:
                break
        with self.server.lock:
            self.server.disconnects.append(time.monotonic())
        self.close_connection = True


class _KeepAliveHandler(_Handler):
    """HTTP/1.1: the connection stays open for the next request (like a cloud API)."""

    protocol_version = "HTTP/1.1"


class _SlowServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, plan, handler: type[_Handler] = _Handler) -> None:
        super().__init__(("127.0.0.1", 0), handler)
        self.plan = plan
        self.lock = threading.Lock()
        self.connections = 0
        self.paths: list[str] = []
        self.bodies: list[object] = []
        self.disconnects: list[float] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


@pytest.fixture
def serve():
    servers: list[_SlowServer] = []

    def start(plan, handler: type[_Handler] = _Handler) -> _SlowServer:
        server = _SlowServer(plan, handler)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def req(purpose: Purpose = Purpose.VISION) -> GenerationRequest:
    return GenerationRequest(system="Ти — Мія.", turns=(ChatTurn(Role.USER, "Що тут?"),), purpose=purpose)


def cancel_after(token: CancelToken, seconds: float) -> None:
    threading.Timer(seconds, token.cancel).start()


def ollama_reply(text: str = "ok") -> dict:
    return {"model": "m", "message": {"role": "assistant", "content": text}, "done": True,
            "done_reason": "stop"}


OPENAI_REPLY = {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}


def _assert_aborted(server: _SlowServer, started: float) -> None:
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"the request was not aborted ({elapsed:.1f} s)"
    deadline = time.monotonic() + 2.0
    while not server.disconnects and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.disconnects, "the server never saw the client go"
    assert server.disconnects[0] - started < 2.0


@pytest.mark.parametrize("stream", [False, True])
def test_ollama_request_waiting_for_the_server_is_aborted(serve, stream):
    server = serve(lambda path, body: "hold")
    provider = OllamaProvider(server.url, "m")
    token = CancelToken()
    cancel_after(token, 0.2)
    started = time.monotonic()
    with pytest.raises(CancelledError):
        if stream:
            provider.stream(req(), lambda piece: None, token)
        else:
            provider.generate(req(), token)
    _assert_aborted(server, started)
    provider.close()


def test_ollama_stream_stalled_mid_answer_is_aborted(serve):
    server = serve(lambda path, body: "stall_stream")
    provider = OllamaProvider(server.url, "m")
    token = CancelToken()
    pieces: list[str] = []
    cancel_after(token, 0.3)
    started = time.monotonic()
    with pytest.raises(CancelledError):
        provider.stream(req(), pieces.append, token)
    assert pieces == ["Хм"]
    _assert_aborted(server, started)
    provider.close()


def test_ollama_warm_up_is_aborted_too(serve):
    server = serve(lambda path, body: "hold")
    provider = OllamaProvider(server.url, "m", keep_alive="30m")
    token = CancelToken()
    cancel_after(token, 0.2)
    started = time.monotonic()
    with pytest.raises(CancelledError):
        provider.warm_up(token)
    _assert_aborted(server, started)
    assert server.paths == ["/api/generate"] and server.bodies == [{"model": "m", "keep_alive": "30m"}]
    provider.close()


@pytest.mark.parametrize("stream", [False, True])
def test_openai_compatible_request_waiting_for_the_server_is_aborted(serve, stream):
    server = serve(lambda path, body: "hold")
    provider = OpenAIProvider(None, f"{server.url}/v1", "local")
    token = CancelToken()
    cancel_after(token, 0.2)
    started = time.monotonic()
    with pytest.raises(CancelledError):
        if stream:
            provider.stream(req(), lambda piece: None, token)
        else:
            provider.generate(req(), token)
    _assert_aborted(server, started)
    provider.close()


def test_every_request_to_a_local_server_gets_its_own_connection(serve):
    server = serve(lambda path, body: ollama_reply() if path == "/api/chat" else OPENAI_REPLY)
    ollama = OllamaProvider(server.url, "m")
    ollama.generate(req())
    ollama.generate(req(), CancelToken())
    compatible = OpenAIProvider(None, f"{server.url}/v1", "local")
    compatible.generate(req())
    compatible.generate(req(), CancelToken())
    assert server.connections == 4  # no keep-alive: a cancel can always reach the socket
    ollama.close()
    compatible.close()


def test_ollama_on_localhost_is_reached_over_ipv4_without_asking_for_localhost(serve, monkeypatch):
    """Windows resolves "localhost" to ::1 first, and Ollama listens on 127.0.0.1 only: the
    refused ::1 attempt costs 1-2 s there (Windows retries the SYN) - on every request,
    since each opens a fresh connection.  Ollama's own clients connect to 127.0.0.1."""
    server = serve(lambda path, body: ollama_reply("ok"))
    port = server.server_address[1]
    looked_up: list[str] = []
    real = socket.getaddrinfo

    def getaddrinfo(host, *args, **kwargs):
        looked_up.append(host if isinstance(host, str) else host.decode())
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    provider = OllamaProvider(f"http://localhost:{port}", "m", keep_alive="30m")
    try:
        assert provider.is_local
        assert provider.generate(req()).text == "ok"
        provider.warm_up()
        assert server.paths == ["/api/chat", "/api/generate"]
        assert "localhost" not in looked_up
    finally:
        provider.close()


def test_a_finished_request_is_not_affected_by_a_late_cancel(serve):
    server = serve(lambda path, body: ollama_reply("ok"))
    provider = OllamaProvider(server.url, "m")
    token = CancelToken()
    assert provider.generate(req(), token).text == "ok"
    token.cancel()
    time.sleep(0.15)
    assert provider.generate(req()).text == "ok"
    provider.close()


@pytest.mark.parametrize("how", ["interrupted", "withdrawn"])
def test_a_complete_answer_on_a_connection_that_cannot_be_aborted_is_kept(serve, how):
    """A cloud API's keep-alive connection is reused: there is no socket to shut, the
    request runs to its end and is billed - its answer is returned, not thrown away (the
    AI service decides: it records the usage and tells a caller who withdrew it)."""
    calls = []

    def plan(path, body):
        calls.append(path)
        return OPENAI_REPLY if len(calls) == 1 else ("slow", OPENAI_REPLY)

    server = serve(plan, _KeepAliveHandler)
    client = httpx2.Client(timeout=5.0)  # keep-alive, like the client of a cloud API
    provider = OpenAIProvider(None, f"{server.url}/v1", "local", http_client=client)
    try:
        provider.generate(req())  # opens the connection the next request reuses
        token = _JobToken(CancelToken())
        threading.Timer(0.15, token.preempt if how == "interrupted" else token.cancel).start()
        assert provider.generate(req(), token).text == "ok"
        assert server.connections == 1
    finally:
        client.close()


def test_request_abort_raises_on_a_clean_exit_only_when_it_cut_a_socket():
    """A response without a length ends at the shut socket as if it were complete: only
    then is a clean exit the abort's doing.  Nothing shut: the answer is real."""

    class _Stream:
        def __init__(self, sock: socket.socket) -> None:
            self.sock = sock

        def get_extra_info(self, name: str):
            return self.sock if name == "socket" else None

    token = CancelToken()
    with RequestAbort(token) as abort:
        token.cancel()
        time.sleep(0.2)
        assert abort.aborted  # seen, but there was nothing to shut
    left, right = socket.socketpair()
    try:
        token = CancelToken()
        with pytest.raises(CancelledError), RequestAbort(token) as abort:
            abort.extensions["trace"]("connection.connect_tcp.complete", {"return_value": _Stream(left)})
            token.cancel()
            assert right.recv(1) == b""  # shut by the watcher: the peer sees the end
    finally:
        left.close()
        right.close()


def test_request_abort_without_a_token_adds_nothing_and_errors_pass_through():
    abort = RequestAbort(None)
    assert abort.extensions == {}
    with pytest.raises(ProviderUnavailableError), abort:
        raise ProviderUnavailableError("down")
    token = CancelToken()
    with pytest.raises(ProviderUnavailableError), RequestAbort(token):
        raise ProviderUnavailableError("down")  # not cancelled: the error is real
    token.cancel()
    with pytest.raises(CancelledError), RequestAbort(token):
        raise ProviderUnavailableError("aborted")  # the abort's own doing
    with pytest.raises(CancelledError, match="own"), RequestAbort(token):
        raise CancelledError("own")  # passes unchanged
    assert not RequestAbort(token).aborted


def test_a_conversation_frees_the_local_server_for_the_chat_answer(qapp, db, clock, bus, serve):
    """End to end: the screen analysis running on Ollama is aborted the moment the user
    starts talking, so the chat answer is not stuck behind it on the server."""
    server = serve(lambda path, body: ("hold" if body and body["messages"][-1]["content"].endswith("кадр")
                                       else ollama_reply("Слухаю!")))
    provider = OllamaProvider(server.url, "m")
    h = Harness(qapp, db, clock, bus, provider)  # type: ignore[arg-type]
    try:
        vision_errors: list[Exception] = []
        h.service.submit(GenerationRequest(system="s", turns=(ChatTurn(Role.USER, "кадр"),),
                                           purpose=Purpose.VISION),
                         on_done=lambda result: None, on_error=vision_errors.append)
        assert pump(qapp, lambda: len(server.paths) == 1)
        started = time.monotonic()
        bus.publish(ConversationActive(active=True))
        bus.dispatch_pending()
        answers: list[str] = []
        h.service.submit(req(Purpose.CHAT), on_done=lambda result: answers.append(result.text),
                         on_error=lambda exc: answers.append(type(exc).__name__))
        assert pump(qapp, lambda: bool(answers and vision_errors), timeout=3.0)
        assert answers == ["Слухаю!"]
        assert isinstance(vision_errors[0], ConversationBusyError)
        _assert_aborted(server, started)
    finally:
        h.service.shutdown()
