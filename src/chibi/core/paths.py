"""Filesystem locations.

Data lives in ``%APPDATA%\\ChibiCompanion`` on Windows and
``$XDG_DATA_HOME/chibi-companion`` (``~/.local/share/...``) elsewhere.
``CHIBI_DATA_DIR`` overrides it (tests, portable installs).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME_WINDOWS = "ChibiCompanion"
APP_DIR_NAME_POSIX = "chibi-companion"

PACKAGE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = PACKAGE_DIR / "assets"
CHARACTERS_DIR = ASSETS_DIR / "characters"


def data_dir() -> Path:
    override = os.environ.get("CHIBI_DATA_DIR")
    if override:
        path = Path(override)
    elif sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        path = Path(base) / APP_DIR_NAME_WINDOWS
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        path = Path(base) / APP_DIR_NAME_POSIX
    path.mkdir(parents=True, exist_ok=True)
    return path


def database_path() -> Path:
    return data_dir() / "chibi.db"


def log_dir() -> Path:
    path = data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def character_pack_dir(pack: str) -> Path:
    return CHARACTERS_DIR / pack
