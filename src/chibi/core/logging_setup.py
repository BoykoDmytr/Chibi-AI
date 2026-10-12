"""Logging configuration.

Logs go to a size-capped rotating file in the data directory.  Privacy rule
for every module: never log chat text, window titles, file paths from other
apps or screen contents.  Process names of *non-private* apps may be logged
at DEBUG level only.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys

from chibi.core.paths import log_dir


def setup_logging(debug: bool = False) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir() / "chibi.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    if debug or (sys.stderr is not None and sys.stderr.isatty()):
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
