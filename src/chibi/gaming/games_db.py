"""Which running program is a game.

Three layers, most certain first:

1. **User games** (settings -> "custom_games"): the user said so, confidence 1.0.
   The settings form asks only for the exe and a title, so a user game has no
   genre (``None``) unless its entry names a valid one: guessing one would
   greet a cozy farm game with action lines, while ``None`` lets reactions use
   the lines written for games of unknown genre.
2. **Built-in database** (``data/games.json``): exact, case-insensitive exe
   names of ~80 popular PC games, confidence 0.95.  Game ids there are stable
   phrase keys (``game_start.<id>``).
3. **Heuristics** for games the database doesn't know: an exe inside a game
   library folder (``steamapps\\common``, ``Epic Games``, ``XboxGames``...)
   scores 0.75; an Unreal Engine ``*-Win64-Shipping.exe`` scores 0.7.
   Library folders also hold launchers, crash handlers, anti-cheat services
   and redistributable installers, so those are rejected by name and by the
   folder they live in.  Their genre is unknown (``None``).

A few games run under a generic process (Minecraft Java is ``javaw.exe``);
the database recognises them by a window-title pattern instead, which the
classifier uses only if the user allowed window titles.

The database is immutable after loading, so the monitor thread can read it
while the main thread builds a replacement after a settings change.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).with_name("data") / "games.json"

GENRES: frozenset[str] = frozenset(
    {
        "shooter", "souls", "rpg", "strategy", "sandbox", "moba", "cozy", "survival",
        "horror", "racing", "sim", "rhythm", "mmo", "action", "party", "card",
    }
)

CUSTOM_CONFIDENCE = 1.0
DB_CONFIDENCE = 0.95
TITLE_HINT_CONFIDENCE = 0.85
LIBRARY_CONFIDENCE = 0.75
SHIPPING_CONFIDENCE = 0.7

#: Folder sequences (lower case) that contain installed games; the folder right
#: after the marker is the game's install folder.
LIBRARY_MARKERS: tuple[tuple[str, ...], ...] = (
    ("steamapps", "common"),
    ("epic games",),
    ("gog galaxy", "games"),
    ("gog games",),
    ("xboxgames",),
    ("riot games",),
    ("ubisoft game launcher", "games"),
    ("ea games",),
    ("amazon games", "library"),
)

#: Unreal Engine names shipping builds ``<Project>-<Platform>-Shipping.exe``.
SHIPPING_SUFFIXES: tuple[str, ...] = ("-win64-shipping.exe", "-wingdk-shipping.exe")

#: Helper programs that live next to games but are not games (lower-case globs).
HELPER_NAME_PATTERNS: tuple[str, ...] = (
    "unitycrashhandler*", "*crashhandler*", "*crashreport*", "*crash_report*", "*crashpad*",
    "*crashsender*", "*crashdump*", "*bugreport*",
    "easyanticheat*", "*anticheat*", "eac_*", "be*service*", "battleye*", "start_protected_game*",
    "*launcher*", "*bootstrap*", "*setup*", "*install*", "*unins*", "*redist*", "*prereq*",
    "dxsetup*", "dotnet*", "*updater*", "update.exe", "*patcher*", "touchup*", "cleanup*",
    "*helper*", "*overlay*", "*.tmp",
    "steam.exe", "steamservice.exe", "steamerrorreporter*",
    "epicgameslauncher*", "epicwebhelper*", "epiconlineservices*", "riotclient*", "leagueclient*",
    "galaxyclient*", "upc.exe", "uplay*", "ubisoftconnect*",
    "socialclub*", "rockstarservice*", "eadesktop*", "eabackgroundservice*", "link2ea*", "origin.exe",
    "wallpaper32.exe", "wallpaper64.exe", "webwallpaper*", "losslessscaling*", "bongocat*",
    "vrserver*", "vrmonitor*", "vrcompositor*", "vrdashboard*", "vrstartup*",
)

#: Folders inside a game library whose programs are never games.
HELPER_DIRS: frozenset[str] = frozenset(
    {
        "_commonredist", "__installer", "_installer", "installer", "installers", "redist",
        "redistributables", "directx", "directxredist", "prerequisites", "prereqs", "support",
        "easyanticheat", "easyanticheat_eos", "battleye", "steamworks shared", "steamvr",
        "wallpaper_engine", "launcher", "riot client", "epic online services",
    }
)


@dataclass(frozen=True)
class GameMatch:
    game_id: str
    title: str
    genre: str | None  # one of GENRES; None = unknown (heuristic guesses, most user games)
    confidence: float
    source: str  # "custom" | "db" | "title" | "heuristic"


@dataclass(frozen=True)
class GameInfo:
    game_id: str
    title: str
    genre: str | None  # always one of GENRES for "db"; None for a user game without one
    processes: tuple[str, ...]
    source: str  # "db" | "custom"


@dataclass(frozen=True)
class TitleHint:
    game_id: str
    process: str
    pattern: re.Pattern[str]


def _split_path(path: str) -> list[str]:
    return [part for part in re.split(r"[\\/]+", path) if part]


def normalize_process(name: object) -> str:
    """Lower-case exe name; a bare name ("Foo") gets ".exe", a path is reduced to its file name."""
    if not isinstance(name, str):
        return ""
    parts = _split_path(name.strip().lower())
    if not parts:
        return ""
    base = parts[-1].strip()
    if base and "." not in base:
        base += ".exe"
    return base


def prettify_name(text: str) -> str:
    """Readable title from a folder or exe stem: "HogwartsLegacy" -> "Hogwarts Legacy"."""
    text = " ".join(text.replace("_", " ").split())
    if text and " " not in text:
        text = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=\d{2,})", " ", text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def title_from_exe(file_name: str) -> str:
    """Title from an exe name: strip ``.exe`` and ``-Win64-Shipping``, split words."""
    stem = re.sub(r"\.exe$", "", file_name.strip(), flags=re.IGNORECASE)
    stem = re.sub(r"-win(?:64|gdk)-shipping$", "", stem, flags=re.IGNORECASE)
    return prettify_name(stem.replace("-", " ")) or stem or file_name


def is_helper_process(process_name: str) -> bool:
    name = process_name.lower()
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in HELPER_NAME_PATTERNS)


def library_location(exe_path: str) -> tuple[str, bool] | None:
    """If *exe_path* lies in a game library: (install folder name, inside a helper folder?).

    None if the path is not inside a library (or is directly in its root).
    """
    parts = _split_path(exe_path)
    lower = [p.casefold() for p in parts]
    for marker in LIBRARY_MARKERS:
        size = len(marker)
        for start in range(len(lower) - size):
            if tuple(lower[start:start + size]) != marker:
                continue
            rest = parts[start + size:]  # [install folder, ..., exe]
            if len(rest) < 2:
                continue
            inner = lower[start + size:-1]
            return rest[0], any(segment in HELPER_DIRS for segment in inner)
    return None


def validate_games_data(data: Any) -> list[str]:
    """Problems in a games.json document (empty list = valid).  Used by tests and by load()."""
    problems: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("games"), list):
        return ["games.json must be an object with a 'games' list"]
    seen_ids: set[str] = set()
    seen_processes: dict[str, str] = {}
    for index, entry in enumerate(data["games"]):
        problems.extend(_entry_problems(index, entry, seen_ids, seen_processes))
    return problems


def _entry_problems(index: int, entry: Any, seen_ids: set[str], seen_processes: dict[str, str]) -> list[str]:
    if not isinstance(entry, dict):
        return [f"games[{index}] is not an object"]
    problems: list[str] = []
    game_id = entry.get("id")
    where = f"games[{index}] ({game_id!r})"
    if not isinstance(game_id, str) or not re.fullmatch(r"[a-z0-9_]+", game_id):
        problems.append(f"{where}: id must match [a-z0-9_]+")
    elif game_id in seen_ids:
        problems.append(f"{where}: duplicate id")
    else:
        seen_ids.add(game_id)
    if not isinstance(entry.get("title"), str) or not entry["title"].strip():
        problems.append(f"{where}: empty title")
    if entry.get("genre") not in GENRES:
        problems.append(f"{where}: unknown genre {entry.get('genre')!r}")
    processes = entry.get("processes")
    if not isinstance(processes, list) or not processes:
        problems.append(f"{where}: processes must be a non-empty list")
        processes = []
    for process in processes:
        if not isinstance(process, str) or process != process.strip().lower() or not process:
            problems.append(f"{where}: process {process!r} must be a lower-case exe name")
        elif process in seen_processes:
            problems.append(f"{where}: process {process!r} already belongs to {seen_processes[process]!r}")
        else:
            seen_processes[process] = str(game_id)
    for hint in entry.get("title_hints", []) or []:
        if not isinstance(hint, dict) or not isinstance(hint.get("process"), str):
            problems.append(f"{where}: title hint needs a process")
            continue
        try:
            re.compile(str(hint.get("pattern", "")))
        except re.error:
            problems.append(f"{where}: title hint pattern does not compile")
        if not hint.get("pattern"):
            problems.append(f"{where}: title hint needs a pattern")
    return problems


@lru_cache(maxsize=4)
def _builtin(path: Path) -> tuple[tuple[GameInfo, ...], tuple[TitleHint, ...]]:
    """Valid entries of the built-in database; invalid ones are logged and skipped."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.error("games database %s is unreadable (%s); only heuristics and user games work",
                  path.name, type(exc).__name__)
        return (), ()
    if not isinstance(data, dict) or not isinstance(data.get("games"), list):
        log.error("games database %s has no 'games' list", path.name)
        return (), ()
    games: list[GameInfo] = []
    hints: list[TitleHint] = []
    seen_ids: set[str] = set()
    seen_processes: dict[str, str] = {}
    for index, entry in enumerate(data["games"]):
        problems = _entry_problems(index, entry, seen_ids, seen_processes)
        if problems:
            log.warning("games database: skipping invalid entry: %s", "; ".join(problems))
            continue
        games.append(
            GameInfo(entry["id"], entry["title"].strip(), entry["genre"], tuple(entry["processes"]), "db")
        )
        for hint in entry.get("title_hints", []) or []:
            hints.append(
                TitleHint(entry["id"], hint["process"].lower(), re.compile(hint["pattern"], re.IGNORECASE))
            )
    return tuple(games), tuple(hints)


class GameDatabase:
    def __init__(self, games: Iterable[GameInfo] = (), title_hints: Iterable[TitleHint] = ()) -> None:
        self._games: dict[str, GameInfo] = {}
        self._by_process: dict[str, GameInfo] = {}
        for info in games:
            # The first game to claim an exe keeps it: a user entry for an exe the
            # database knows would only hide the game's own id (and its phrases).
            processes = [p for p in info.processes if p not in self._by_process]
            if not processes:
                continue
            self._games[info.game_id] = info
            for process in processes:
                self._by_process[process] = info
        self._title_hints: dict[str, list[TitleHint]] = {}
        for hint in title_hints:
            if hint.game_id in self._games:
                self._title_hints.setdefault(hint.process, []).append(hint)

    @classmethod
    def load(
        cls, custom_games: Iterable[Mapping[str, str]] = (), *, path: Path | None = None
    ) -> GameDatabase:
        """Built-in games plus the user's ``[{"process": "foo.exe", "title": "Foo"}]``.

        User games get id ``custom:<process>``; one naming an exe the database
        already knows is ignored (the built-in entry carries more: id, genre).
        An optional ``"genre"`` is kept if it is one of :data:`GENRES`;
        otherwise the genre is ``None`` (unknown).
        """
        builtin, hints = _builtin(path or DEFAULT_DATA_PATH)
        # Built-in entries come first: the first game to claim an exe keeps it.
        return cls([*builtin, *_custom_entries(custom_games)], hints)

    # -- queries ------------------------------------------------------------------

    def match(self, process_name: str, exe_path: str | None = None) -> GameMatch | None:
        name = normalize_process(process_name)
        if not name:
            return None
        info = self._by_process.get(name)
        if info is not None:
            confidence = CUSTOM_CONFIDENCE if info.source == "custom" else DB_CONFIDENCE
            return GameMatch(info.game_id, info.title, info.genre, confidence, info.source)
        return self._heuristic(name, exe_path)

    def match_title(self, process_name: str, title: str | None) -> GameMatch | None:
        """Games recognisable only by window title (see module docstring)."""
        if not title:
            return None
        for hint in self._title_hints.get(normalize_process(process_name), ()):
            if hint.pattern.search(title):
                info = self._games[hint.game_id]
                return GameMatch(info.game_id, info.title, info.genre, TITLE_HINT_CONFIDENCE, "title")
        return None

    def get(self, game_id: str) -> GameInfo | None:
        return self._games.get(game_id)

    def games(self) -> list[GameInfo]:
        return list(self._games.values())

    def __len__(self) -> int:
        return len(self._games)

    def __contains__(self, game_id: object) -> bool:
        return game_id in self._games

    # -- heuristics ---------------------------------------------------------------

    def _heuristic(self, name: str, exe_path: str | None) -> GameMatch | None:
        confidence = 0.0
        folder: str | None = None
        if exe_path:
            location = library_location(exe_path)
            if location is not None:
                folder, in_helper_dir = location
                if in_helper_dir:
                    return None
                confidence = LIBRARY_CONFIDENCE
        if name.endswith(SHIPPING_SUFFIXES):
            confidence = max(confidence, SHIPPING_CONFIDENCE)
        # The (pricier) helper check runs last: process scans call this for every process.
        if confidence == 0.0 or is_helper_process(name):
            return None
        title = prettify_name(folder) if folder else ""
        if not title:
            # Prefer the exe's original spelling (the path keeps CamelCase, the name is lower-cased).
            base = _split_path(exe_path)[-1] if exe_path else name
            title = title_from_exe(base if base.lower() == name else name)
        return GameMatch(f"exe:{name}", title, None, confidence, "heuristic")


def _custom_entries(custom_games: Iterable[Mapping[str, str]]) -> list[GameInfo]:
    entries: list[GameInfo] = []
    for raw in custom_games or ():
        if not isinstance(raw, Mapping):
            continue
        process = normalize_process(raw.get("process"))
        if not process:
            continue
        title = raw.get("title")
        title = title.strip()[:80] if isinstance(title, str) and title.strip() else title_from_exe(process)
        genre = raw.get("genre")
        # Hand-edited settings may hold anything here (a list is not even hashable).
        genre = genre if isinstance(genre, str) and genre in GENRES else None
        entries.append(GameInfo(f"custom:{process}", title, genre, (process,), "custom"))
    return entries
