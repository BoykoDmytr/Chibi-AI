"""Logging configuration.

Logs go to a size-capped rotating file in the data directory.  Privacy rule
for every module: never log chat text, window titles, file paths from other
apps or screen contents.  Process names of *non-private* apps may be logged
at DEBUG level only.

Defence in depth for that rule: every record passes :func:`redact` before it
is written - API keys and tokens, e-mail addresses, phone and card numbers,
URL query strings and the Windows user name in paths are masked, including
inside exception tracebacks.  A user can therefore send chibi.log with a bug
report without leaking secrets, even if some module logs carelessly.
"""

from __future__ import annotations

import logging
import logging.handlers
import re
import sys
import threading

from chibi.core.paths import log_dir

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Provider keys / tokens (Anthropic, OpenAI, GitHub, AWS, Google, Slack, Azure-style hex).
    (re.compile(r"\b(?:sk-ant-[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}"
                r"|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{30,}|xox[abprs]-[A-Za-z0-9\-]{10,})"), "<key>"),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/\-]{6,}=*"), r"\1 <redacted>"),
    (re.compile(r"(?i)\b(authorization|api[_-]?key|x-api-key|ocp-apim-subscription-key|token"
                r"|password|secret)(\s*[:=]\s*|\s+)([\"']?)(?!(?:bearer|basic) <redacted>)[^\s\"',;]{6,}"),
     r"\1\2\3<redacted>"),
    (re.compile(r"\b[0-9a-f]{32,}\b", re.IGNORECASE), "<hex>"),
    (re.compile(r"[\w.+\-]{1,64}@[\w\-]{1,63}(?:\.[\w\-]{1,63}){1,8}"), "<email>"),
    # Query strings may carry keys or personal data; keep the endpoint for debugging.
    (re.compile(r"(https?://[^\s?#\"']+)\?[^\s\"']*"), r"\1?<query>"),
    # User profile paths: C:\Users\Олена\... -> C:\Users\<user>\...
    (re.compile(r"(?i)([a-z]:[\\/]+(?:users|documents and settings)[\\/]+)[^\\/\s\"']+"), r"\1<user>"),
    (re.compile(r"(/home/|/Users/)[^/\s\"']+"), r"\1<user>"),
    # Card-like and phone-like digit runs.
    (re.compile(r"(?<![\w.])(?:\d[ \-]?){12,18}\d(?![\w.])"), "<number>"),
    (re.compile(r"(?<![\w.])\+?\d{1,3}[ \-]?\(?\d{2,3}\)?[ \-]?\d{3}[ \-]?\d{2}[ \-]?\d{2}(?![\w.])"),
     "<phone>"),
)


def redact(text: str) -> str:
    """Mask secrets and personal data in a log line (idempotent)."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class RedactingFormatter(logging.Formatter):
    """Redacts the complete formatted record, traceback included."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup_logging(debug: bool = False) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = RedactingFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir() / "chibi.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    if debug or (sys.stderr is not None and sys.stderr.isatty()):
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)

    # HTTP client libraries log full request bodies (chat text, images) at
    # DEBUG; even run_debug.bat must not write those.
    for name in ("anthropic", "httpx", "httpx2", "httpcore", "httpcore2", "urllib3",
                 "huggingface_hub", "faster_whisper"):
        logging.getLogger(name).setLevel(logging.WARNING)
    install_exception_hooks()


def install_exception_hooks() -> None:
    """Unhandled exceptions anywhere (main thread, worker threads, Qt) end up
    in the log - redacted - instead of vanishing with a windowed .exe."""
    log = logging.getLogger("chibi.crash")

    def main_hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("unhandled exception", exc_info=(exc_type, exc, tb))

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else "?"
        log.critical("unhandled exception in thread %s", name,
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = main_hook
    threading.excepthook = thread_hook
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    except ImportError:  # pragma: no cover
        return
    qt_log = logging.getLogger("qt")
    levels = {QtMsgType.QtDebugMsg: logging.DEBUG, QtMsgType.QtInfoMsg: logging.INFO,
              QtMsgType.QtWarningMsg: logging.WARNING, QtMsgType.QtCriticalMsg: logging.ERROR,
              QtMsgType.QtFatalMsg: logging.CRITICAL}

    def qt_handler(mode, _context, message):
        qt_log.log(levels.get(mode, logging.WARNING), "%s", message)

    qInstallMessageHandler(qt_handler)
