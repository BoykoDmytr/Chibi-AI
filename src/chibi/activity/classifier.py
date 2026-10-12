"""What kind of thing is the user doing: foreground app -> activity category.

Rules, first match wins:

1. a game from the database or the user's list -> ``GAMING`` (its confidence);
2. a game recognised by window title (Minecraft Java runs as ``javaw.exe``);
3. a known app (``data/app_categories.json``) -> its category at 0.9;
   browsers are the exception: a browser alone is only ``BROWSING`` at 0.6,
   because "Chrome is open" says nothing about *what* is in the tab;
4. a game found by folder/exe-name heuristics -> ``GAMING`` (0.7-0.75).
   Known apps win over heuristics: Blender installed through Steam lives in
   ``steamapps\\common`` but is not a game;
5. window title (only present if the user granted ``WINDOW_TITLES``): video
   sites -> ``WATCHING_VIDEO`` 0.8, developer sites -> ``PROGRAMMING`` 0.55.
   Applies to browsers and unknown apps; a known editor stays an editor even
   if a file named "youtube.py" is open;
6. otherwise ``UNKNOWN`` 0.3, with a display name made from the exe name.

The classifier never returns ``PRIVATE``: the monitor runs the privacy filter
first, and private apps never reach it.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from chibi.activity.backend import ForegroundInfo
from chibi.core.events import ActivityCategory
from chibi.gaming.games_db import GameDatabase, GameMatch, normalize_process, title_from_exe
from chibi.privacy.filters import compile_keywords, normalize_text

log = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).with_name("data") / "app_categories.json"

KNOWN_APP_CONFIDENCE = 0.9
BROWSER_CONFIDENCE = 0.6
VIDEO_TITLE_CONFIDENCE = 0.8
DEV_TITLE_CONFIDENCE = 0.55
UNKNOWN_CONFIDENCE = 0.3

#: JSON group name -> category.  "other" = known by name, no activity category (launchers, system tools).
CATALOG_GROUPS: Mapping[str, ActivityCategory] = {
    "programming": ActivityCategory.PROGRAMMING,
    "browsing": ActivityCategory.BROWSING,
    "watching_video": ActivityCategory.WATCHING_VIDEO,
    "working": ActivityCategory.WORKING,
    "other": ActivityCategory.UNKNOWN,
}


@dataclass(frozen=True)
class KnownApp:
    process_name: str
    category: ActivityCategory
    display_name: str


@dataclass(frozen=True)
class Classification:
    category: ActivityCategory
    confidence: float
    display_name: str
    game: GameMatch | None = None


@dataclass(frozen=True)
class AppCatalog:
    """Known apps and title keywords.  Immutable, so it is safe to share across threads."""

    apps: Mapping[str, KnownApp] = field(default_factory=dict)
    patterns: tuple[KnownApp, ...] = ()  # process_name holds the glob
    video_titles: re.Pattern[str] | None = None
    dev_titles: re.Pattern[str] | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> AppCatalog:
        return _load_catalog(path or DEFAULT_DATA_PATH)

    def lookup(self, process_name: str) -> KnownApp | None:
        name = normalize_process(process_name)
        if not name:
            return None
        app = self.apps.get(name)
        if app is not None:
            return app
        for pattern in self.patterns:
            if fnmatch.fnmatchcase(name, pattern.process_name):
                return KnownApp(name, pattern.category, pattern.display_name)
        return None


def parse_catalog(data: Any) -> AppCatalog:
    """Build a catalog from the JSON document.  Raises ValueError if its shape is wrong."""
    if not isinstance(data, dict) or not isinstance(data.get("categories"), dict):
        raise ValueError("app catalog must be an object with a 'categories' object")
    apps: dict[str, KnownApp] = {}
    for group, entries in data["categories"].items():
        category = CATALOG_GROUPS.get(group)
        if category is None or not isinstance(entries, dict):
            raise ValueError(f"unknown or malformed app group {group!r}")
        for process, display in entries.items():
            name = normalize_process(process)
            if not name or not isinstance(display, str) or not display.strip():
                raise ValueError(f"malformed app entry in group {group!r}")
            if name in apps:
                log.warning("app catalog lists %s twice; keeping the first entry", name)
                continue
            apps[name] = KnownApp(name, category, display.strip())
    patterns: list[KnownApp] = []
    for entry in data.get("patterns", []):
        if not isinstance(entry, dict) or entry.get("category") not in CATALOG_GROUPS:
            raise ValueError("malformed app pattern")
        glob, display = entry.get("glob"), entry.get("name")
        if not isinstance(glob, str) or not glob.strip() or not isinstance(display, str):
            raise ValueError("malformed app pattern")
        patterns.append(KnownApp(glob.strip().lower(), CATALOG_GROUPS[entry["category"]], display.strip()))
    keywords = data.get("title_keywords", {})
    if not isinstance(keywords, dict):
        raise ValueError("title_keywords must be an object")
    return AppCatalog(
        apps=MappingProxyType(apps),
        patterns=tuple(patterns),
        video_titles=compile_keywords(keywords.get("watching_video", [])),
        dev_titles=compile_keywords(keywords.get("programming", [])),
    )


@lru_cache(maxsize=4)
def _load_catalog(path: Path) -> AppCatalog:
    try:
        return parse_catalog(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as exc:
        # Without the catalog every app is UNKNOWN: worse comments, nothing unsafe.
        log.error("app catalog %s is unusable (%s); apps will not be categorised",
                  path.name, type(exc).__name__)
        return AppCatalog()


def _last_match(pattern: re.Pattern[str] | None, text: str) -> int:
    """Start of the last keyword match in *text*, -1 if none."""
    if pattern is None:
        return -1
    last = -1
    for match in pattern.finditer(text):
        last = match.start()
    return last


def display_name_for(process_name: str, exe_path: str | None = None) -> str:
    """Friendly name for an app we know nothing about: "MyTool.exe" -> "My Tool"."""
    name = normalize_process(process_name)
    base = re.split(r"[\\/]+", exe_path)[-1] if exe_path else ""
    return title_from_exe(base if base and base.lower() == name else name)


def _gaming(game: GameMatch) -> Classification:
    return Classification(ActivityCategory.GAMING, game.confidence, game.title, game)


class ActivityClassifier:
    def __init__(self, games: GameDatabase, catalog: AppCatalog | None = None) -> None:
        self._games = games
        self._catalog = catalog if catalog is not None else AppCatalog.load()

    @property
    def games(self) -> GameDatabase:
        return self._games

    def known_app(self, process_name: str) -> KnownApp | None:
        return self._catalog.lookup(process_name)

    def match_game(self, process_name: str, exe_path: str | None = None) -> GameMatch | None:
        """Game for a process, with the same precedence as :meth:`classify`:
        a heuristic guess never overrides an app we know is not a game."""
        game = self._games.match(process_name, exe_path)
        if game is not None and game.source == "heuristic" and self.known_app(process_name) is not None:
            return None
        return game

    def classify(self, info: ForegroundInfo) -> Classification:
        name = normalize_process(info.process_name)
        title = normalize_text(info.title) if info.title else ""

        game = self._games.match(name, info.exe_path)
        if game is not None and game.source != "heuristic":
            return _gaming(game)
        title_game = self._games.match_title(name, title)
        if title_game is not None:
            return _gaming(title_game)

        known = self.known_app(name)
        if known is not None:
            if known.category is ActivityCategory.BROWSING:
                upgraded = self._from_title(title)
                if upgraded is not None:
                    return Classification(upgraded[0], upgraded[1], known.display_name)
                return Classification(ActivityCategory.BROWSING, BROWSER_CONFIDENCE, known.display_name)
            return Classification(known.category, KNOWN_APP_CONFIDENCE, known.display_name)

        if game is not None:
            return _gaming(game)

        display = display_name_for(name, info.exe_path)
        upgraded = self._from_title(title)
        if upgraded is not None:
            return Classification(upgraded[0], upgraded[1], display)
        return Classification(ActivityCategory.UNKNOWN, UNKNOWN_CONFIDENCE, display)

    def _from_title(self, title: str) -> tuple[ActivityCategory, float] | None:
        """Category suggested by the window title.

        Browser titles end with the site name ("How to X - YouTube - Firefox"),
        so when both a video and a developer keyword appear, the later one is
        the site and wins ("youtube-dl - GitHub" is GitHub).
        """
        if not title:
            return None
        video = _last_match(self._catalog.video_titles, title)
        dev = _last_match(self._catalog.dev_titles, title)
        if video < 0 and dev < 0:
            return None
        if video >= dev:
            return ActivityCategory.WATCHING_VIDEO, VIDEO_TITLE_CONFIDENCE
        return ActivityCategory.PROGRAMMING, DEV_TITLE_CONFIDENCE
