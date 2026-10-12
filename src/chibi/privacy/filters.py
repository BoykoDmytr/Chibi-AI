"""Privacy filter: which foreground apps the companion must not observe at all.

The activity monitor asks this filter *before* it publishes anything.  A
private app is reported as ``AppInfo.private_app()`` with category
``PRIVATE`` - no name, no title, no game, no milestone - so no consumer
(reactions, logs, a future LLM prompt) can leak what it never received.

Sources of "private":

* built-in process names from ``data/private_apps.json`` (password managers,
  authenticators, messengers, mail, finance) plus the user's own exclusions;
* folder names in the exe path (Tor Browser ships a plain ``firefox.exe``);
* window-title keywords (banking, passwords, sign-in pages, private browsing,
  mail and messengers in a browser tab) - only when a title was read, i.e.
  only if the user granted ``WINDOW_TITLES``.

Title keywords match case-insensitively at the *start* of a word: "bank"
matches "Banking" but "wise" does not match "otherwise", and Ukrainian stems
such as "парол" cover every inflection ("пароль", "паролі", "паролем").

The filter fails closed: if its data cannot be loaded, or a check raises,
the app is treated as private.  Over-hiding costs a missed comment;
under-hiding would cost the user's trust.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from chibi.activity.backend import ForegroundInfo

log = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).with_name("data") / "private_apps.json"

_GLOB_CHARS = frozenset("*?[")


def compile_keywords(keywords: Iterable[str]) -> re.Pattern[str] | None:
    """One case-insensitive regex that finds any of *keywords* at the start of a word.

    Whitespace inside keywords is normalised to single spaces; callers should
    normalise the searched text the same way (see :func:`normalize_text`).
    Returns None for an empty keyword list.
    """
    cleaned = {normalize_text(k).casefold() for k in keywords if isinstance(k, str)}
    cleaned.discard("")
    if not cleaned:
        return None
    # Longest first, so a longer keyword wins over its own prefix (only matters for match spans).
    alternation = "|".join(re.escape(k) for k in sorted(cleaned, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternation})", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Collapse any run of whitespace (incl. non-breaking spaces) into one space."""
    return " ".join(text.split())


def _split_path(path: str) -> list[str]:
    return [part for part in re.split(r"[\\/]+", path) if part]


@dataclass(frozen=True)
class PrivateAppDefaults:
    processes: frozenset[str]
    patterns: tuple[str, ...]
    path_segments: frozenset[str]
    title_keywords: tuple[str, ...]


def _strings(value: Any, what: str) -> Iterator[str]:
    """Yield strings from a list, or from a dict of lists (grouped for readability in the JSON)."""
    if isinstance(value, dict):
        for group in value.values():
            yield from _strings(group, what)
        return
    if not isinstance(value, list):
        raise ValueError(f"{what} must be a list or an object of lists")
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{what} contains a non-string entry")
        if item.strip():
            yield item.strip()


@lru_cache(maxsize=8)
def load_defaults(path: Path = DEFAULT_DATA_PATH) -> PrivateAppDefaults:
    """Parse the built-in private app list.  Raises OSError/ValueError if it is unusable."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("private app list must be a JSON object")
    return PrivateAppDefaults(
        processes=frozenset(s.lower() for s in _strings(data.get("processes", []), "processes")),
        patterns=tuple(s.lower() for s in _strings(data.get("patterns", []), "patterns")),
        path_segments=frozenset(
            s.casefold() for s in _strings(data.get("path_segments", []), "path_segments")
        ),
        title_keywords=tuple(_strings(data.get("title_keywords", []), "title_keywords")),
    )


def _normalize_exclusion(raw: object) -> list[str]:
    """User exclusions are typed by hand: accept "MyBank", "mybank.exe" or a full path."""
    if not isinstance(raw, str):
        return []
    name = _split_path(raw.strip().lower())
    if not name:
        return []
    base = name[-1].strip()
    if not base:
        return []
    if "." not in base and not (_GLOB_CHARS & set(base)):
        return [base, f"{base}.exe"]
    return [base]


class PrivacyFilter:
    def __init__(self, extra_processes: Iterable[str] = (), *, data_path: Path | None = None) -> None:
        self._fail_closed = False
        try:
            defaults = load_defaults(data_path or DEFAULT_DATA_PATH)
        except (OSError, ValueError, TypeError) as exc:
            log.error("private app list is unusable (%s); every app is treated as private",
                      type(exc).__name__)
            self._fail_closed = True
            defaults = PrivateAppDefaults(frozenset(), (), frozenset(), ())

        names = set(defaults.processes)
        patterns = list(defaults.patterns)
        for raw in extra_processes:
            for entry in _normalize_exclusion(raw):
                if _GLOB_CHARS & set(entry):
                    patterns.append(entry)
                else:
                    names.add(entry)
        self._processes = frozenset(names)
        self._patterns = tuple(dict.fromkeys(patterns))
        self._path_segments = defaults.path_segments
        self._title_re = compile_keywords(defaults.title_keywords)

    @property
    def private_processes(self) -> frozenset[str]:
        """Exact process names treated as private (built-in + user exclusions)."""
        return self._processes

    def is_private_process(self, process_name: str) -> bool:
        try:
            if self._fail_closed:
                return True
            key = process_name.strip().lower()
            if not key:
                return True  # an app we cannot identify is not one we may comment on
            if key in self._processes:
                return True
            return any(fnmatch.fnmatchcase(key, pattern) for pattern in self._patterns)
        except Exception:
            return True

    def is_private_title(self, title: str | None) -> bool:
        """True if *title* looks private.  No title means no title-based evidence."""
        if title is None:
            return False
        try:
            if self._fail_closed:
                return True
            text = normalize_text(title)
            if not text or self._title_re is None:
                return False
            return self._title_re.search(text) is not None
        except Exception:
            return True

    def is_private_path(self, exe_path: str | None) -> bool:
        if not exe_path:
            return False
        try:
            if self._fail_closed:
                return True
            parts = _split_path(exe_path)
            if parts and self.is_private_process(parts[-1]):
                return True
            return any(part.casefold() in self._path_segments for part in parts[:-1])
        except Exception:
            return True

    def is_private(self, info: ForegroundInfo) -> bool:
        try:
            if self._fail_closed:
                return True
            return (
                self.is_private_process(info.process_name)
                or self.is_private_path(info.exe_path)
                or (info.title is not None and self.is_private_title(info.title))
            )
        except Exception:
            return True
