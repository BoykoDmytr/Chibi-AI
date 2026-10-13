"""``ChibiCompanion.exe --self-test``: is this installation complete?

Run by CI against the frozen build (and useful for support: "run it with
--self-test and send me the log").  Checks what a broken package would get
wrong - assets, data files, Qt image plugins, the database layer - and that
the optional voice libraries import if they were bundled.  Results go to the
log; the exit code is 0 only if every required check passed.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
from collections.abc import Callable
from importlib import resources

log = logging.getLogger(__name__)

#: Optional libraries: if present in the build, they must import.
OPTIONAL_MODULES = ("numpy", "onnxruntime", "ctranslate2", "faster_whisper", "sounddevice")
DATA_FILES = (
    ("chibi.gaming", "data/games.json"),
    ("chibi.personality", "data/phrases_uk.json"),
    ("chibi.privacy", "data/private_apps.json"),
    ("chibi.activity", "data/app_categories.json"),
    ("chibi.personality", "data/phrases_vision_uk.json"),
    ("chibi.personality", "data/phrases_games_uk.json"),
    ("chibi.tamagotchi", "data/gifts.json"),
    ("chibi.voice", "data/pronunciations_uk.json"),
)


def _check_pack() -> str:
    from chibi.character.animation import CharacterPack
    from chibi.core import paths

    pack = CharacterPack.load(paths.character_pack_dir("mia"))
    return f"pack {pack.name!r} loaded"


def _check_data() -> str:
    for package, name in DATA_FILES:
        json.loads(resources.files(package).joinpath(name).read_text("utf-8"))
    return f"{len(DATA_FILES)} data files"


def _check_images() -> str:
    from PySide6.QtGui import QImageWriter

    formats = {bytes(f).decode() for f in QImageWriter.supportedImageFormats()}
    missing = {"png", "jpg"} - formats
    if missing:
        raise RuntimeError(f"Qt image plugins missing: {sorted(missing)}")
    return "png+jpg"


def _check_database() -> str:
    from chibi.memory.database import MIGRATIONS, Database

    db = Database(":memory:")
    try:
        if db.schema_version != len(MIGRATIONS):
            raise RuntimeError("migrations incomplete")
    finally:
        db.close()
    return f"schema v{len(MIGRATIONS)}"


def _check_tts_text() -> str:
    from chibi.voice.tts_text import normalize_for_tts

    if normalize_for_tts("Граєш уже 2 години в S.T.A.L.K.E.R. 2") != "граєш уже дві години в сталкер два.":
        raise RuntimeError("TTS normaliser produced unexpected text")
    return "ok"


REQUIRED: tuple[tuple[str, Callable[[], str]], ...] = (
    ("character pack", _check_pack),
    ("data files", _check_data),
    ("image formats", _check_images),
    ("database", _check_database),
    ("voice text", _check_tts_text),
)


def run() -> int:
    """Run every check (needs a QApplication); returns the process exit code."""
    failures = 0
    for name, check in REQUIRED:
        try:
            log.info("self-test %-15s OK   %s", name, check())
        except Exception as exc:
            failures += 1
            log.error("self-test %-15s FAIL %s: %s", name, type(exc).__name__, exc)
    for module in OPTIONAL_MODULES:
        try:
            present = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            present = False
        if not present:
            log.info("self-test %-15s --   not bundled", module)
            continue
        try:
            importlib.import_module(module)
            log.info("self-test %-15s OK   imported", module)
        except Exception as exc:  # a bundled library that can't load = broken build
            failures += 1
            log.error("self-test %-15s FAIL %s: %s", module, type(exc).__name__, exc)
    log.info("self-test finished: %s", "OK" if failures == 0 else f"{failures} failure(s)")
    return 0 if failures == 0 else 1
