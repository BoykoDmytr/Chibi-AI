import logging
import threading

import pytest

from chibi.core.logging_setup import RedactingFormatter, install_exception_hooks, redact, setup_logging


@pytest.mark.parametrize(
    ("line", "secret"),
    [
        ("key=sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv", "sk-ant-api03"),
        ("using sk-proj-1234567890abcdefghijKLMN", "sk-proj-1234567890"),
        ("Authorization: Bearer abcdef123456789xyz", "abcdef123456789xyz"),
        ("Ocp-Apim-Subscription-Key: 0123456789abcdef0123456789abcdef", "0123456789abcdef"),
        ("password=hunter2hunter2", "hunter2hunter2"),
        ("token ghp_0123456789abcdefghijklmnopqrstuvwxyz", "ghp_0123"),
        ("mail me at olena.k@example.com please", "olena.k@example.com"),
        ("GET https://api.example.com/v1/x?key=abc&user=olena HTTP/1.1", "user=olena"),
        (r"File C:\Users\Олена\AppData\Roaming\x.py", "Олена"),
        ("File /home/olena/chibi/app.py", "olena"),
        ("card 4111 1111 1111 1111 declined", "4111 1111 1111 1111"),
        ("call +380 67 123 45 67", "123 45 67"),
    ],
)
def test_secrets_and_personal_data_are_masked(line, secret):
    assert secret not in redact(line)


@pytest.mark.parametrize(
    "line",
    ["started in 1.25 s", "game stalker2.exe started", "schema v2 migrated",
     "https://api.anthropic.com/v1/messages returned 529", "version 0.1.0", "error 404"],
)
def test_ordinary_lines_are_left_alone(line):
    assert redact(line) == line


def test_redaction_is_idempotent():
    line = "key sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv mail a@b.com"
    assert redact(redact(line)) == redact(line)


def test_tracebacks_are_redacted_too():
    formatter = RedactingFormatter("%(message)s")
    try:
        raise ValueError("bad key sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv for olena@example.com")
    except ValueError:
        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", None, exc_info=True)
        import sys

        record.exc_info = sys.exc_info()
    text = formatter.format(record)
    assert "Traceback" in text
    assert "sk-ant" not in text and "olena@example.com" not in text


def test_setup_logging_writes_redacted_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CHIBI_DATA_DIR", str(tmp_path))
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        setup_logging()
        logging.getLogger("chibi.test").warning("leaked key sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv")
        logging.getLogger("anthropic").debug("request body with chat text")
        for handler in root.handlers:
            handler.flush()
        text = (tmp_path / "logs" / "chibi.log").read_text("utf-8")
        assert "leaked key <key>" in text
        assert "chat text" not in text
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
                handler.close()


def test_thread_exceptions_are_logged(caplog):
    install_exception_hooks()

    def boom():
        raise RuntimeError("worker died")

    with caplog.at_level(logging.CRITICAL, logger="chibi.crash"):
        thread = threading.Thread(target=boom, name="chibi-test-worker")
        thread.start()
        thread.join()
    assert "chibi-test-worker" in caplog.text
