"""Offline phrase bank and the selector that picks what she says.

Phase 1 has no language model, so every line comes from
``data/phrases_uk.json``: ``{"version": 1, "phrases": {"<key>": [entry...]}}``
where an entry is ``{"id", "text", "tone", "state", "portrait", "requires"?}``.

* :class:`PhraseBank` loads and validates the file.  A broken *entry* (unknown
  tone or state, duplicate id, empty text, unknown placeholder) is skipped
  with a warning instead of crashing the app: a typo in one line must not
  silence the character.  A missing or unreadable file gives an empty bank
  (logged as an error) - she then simply has nothing to say offline.  Lines
  of a key in :data:`REQUIRED_STATES` always get that key's posture state.
* :class:`PhraseSelector` picks a line for one of several keys:

  - personality sliders weight the tones (no sarcasm at all below 10%);
  - anti-repeat: with ``n`` phrases in a key, a phrase used within the last
    ``min(n - 1, 30)`` uses of that key is excluded and the immediately previous phrase never
    repeats (unless it is the only one); when everything is excluded the
    least-recently-used phrase wins.  History is persisted through
    :class:`~chibi.memory.repositories.PhraseHistoryRepository`, so she does
    not start every session with the same joke;
  - ``requires``: ``"user"`` needs a nickname in the settings, any other
    requirement needs that variable to be passed; phrases whose requirements
    are not met (or whose template fails to render) are skipped.

Main (Qt) thread only - the history repository owns a SQLite connection.
Rendered text is never logged (it can contain game or app names).
"""

from __future__ import annotations

import json
import logging
import random
import string
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chibi.character.states import CharacterState
from chibi.core.clock import Clock
from chibi.core.config import PersonalitySettings, Settings
from chibi.core.events import SpeechRequested
from chibi.memory.repositories import PhraseHistoryRepository
from chibi.personality.text_uk import render

log = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).with_name("data") / "phrases_uk.json"
SUPPORTED_VERSION = 1

TONES: frozenset[str] = frozenset({"warm", "playful", "sarcastic", "caring"})
STATES: frozenset[str] = frozenset(s.value for s in CharacterState)
#: Placeholders a template may use.  ``name`` (her name) is always available.
PLACEHOLDERS: frozenset[str] = frozenset({"name", "user", "game", "app", "duration", "time"})
#: Values for ``requires``: every placeholder except ``name``, which is always filled.
REQUIREMENTS: frozenset[str] = PLACEHOLDERS - {"name"}
#: Keys whose lines accompany a posture change and must carry that posture.
#: "Go to sleep" puts her to sleep at user priority; a line with an action state
#: such as "yawn" would then wake her straight away (the state machine turns any
#: action while asleep into idle), so she would say "лягаю" and stay awake.
REQUIRED_STATES: dict[str, str] = {"interaction.rest.sleep": CharacterState.SLEEP.value}

#: Anti-repeat window cap (uses of the same key).
MAX_REPEAT_WINDOW = 30
#: Below this sarcasm level sarcastic lines are never chosen.
SARCASM_FLOOR = 10

_FORMATTER = string.Formatter()


@dataclass(frozen=True)
class Phrase:
    id: str
    key: str
    text: str
    tone: str
    state: str
    portrait: str
    requires: tuple[str, ...]


@dataclass(frozen=True)
class PickedPhrase:
    """A phrase chosen (but not yet recorded as said) and its rendered text."""

    phrase: Phrase
    text: str

    @property
    def key(self) -> str:
        return self.phrase.key


def template_fields(text: str) -> list[str]:
    """Placeholder names used by ``text``.

    Raises :class:`ValueError` for malformed templates and for anything but
    plain ``{name}`` fields (no attribute access, indexing, conversions or
    format specs), so a bank entry can never reach into Python objects.
    """
    fields: list[str] = []
    for _literal, field_name, format_spec, conversion in _FORMATTER.parse(text):
        if field_name is None:
            continue
        if not field_name.isidentifier():
            raise ValueError(f"unsupported placeholder {{{field_name}}}")
        if format_spec or conversion:
            raise ValueError(f"placeholder {{{field_name}}} must not use a format spec or conversion")
        fields.append(field_name)
    return fields


class PhraseBank:
    """Validated, immutable collection of phrases grouped by key."""

    def __init__(self, phrases: Mapping[str, list[Phrase]] | None = None) -> None:
        self._by_key: dict[str, tuple[Phrase, ...]] = {
            key: tuple(items) for key, items in (phrases or {}).items() if items
        }
        self._by_id: dict[str, Phrase] = {p.id: p for items in self._by_key.values() for p in items}

    @classmethod
    def load(cls, path: Path | None = None) -> PhraseBank:
        """Load the bank from ``path`` (default: the packaged Ukrainian bank)."""
        source = path or DEFAULT_PATH
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.error("phrase bank %s could not be read (%s); she will have nothing to say offline",
                      source.name, exc)
            return cls()
        return cls.from_dict(data, source=source.name)

    @classmethod
    def from_dict(cls, data: Any, source: str = "<dict>") -> PhraseBank:
        """Build a bank from already parsed JSON, skipping invalid entries with a warning."""
        if not isinstance(data, Mapping) or not isinstance(data.get("phrases"), Mapping):
            log.error("phrase bank %s has no 'phrases' object; ignoring it", source)
            return cls()
        version = data.get("version")
        if version != SUPPORTED_VERSION:
            log.warning("phrase bank %s has version %r (expected %d); loading what is understood",
                        source, version, SUPPORTED_VERSION)
        seen_ids: set[str] = set()
        result: dict[str, list[Phrase]] = {}
        skipped = 0
        for key, entries in data["phrases"].items():
            if not isinstance(key, str) or not key.strip() or not isinstance(entries, list):
                log.warning("phrase bank %s: key %r is not a list of phrases; skipped", source, key)
                skipped += 1
                continue
            for index, entry in enumerate(entries):
                phrase, problem = _parse_entry(key, entry, seen_ids)
                if phrase is None:
                    log.warning("phrase bank %s: %s[%d] skipped: %s", source, key, index, problem)
                    skipped += 1
                    continue
                seen_ids.add(phrase.id)
                result.setdefault(key, []).append(phrase)
        bank = cls(result)
        log.info("phrase bank %s: %d phrases in %d keys (%d skipped)", source, len(bank), len(bank.keys()),
                 skipped)
        return bank

    def has(self, key: str) -> bool:
        return key in self._by_key

    def phrases(self, key: str) -> list[Phrase]:
        return list(self._by_key.get(key, ()))

    def keys(self) -> list[str]:
        return list(self._by_key)

    def get(self, phrase_id: str) -> Phrase | None:
        return self._by_id.get(phrase_id)

    def __iter__(self) -> Iterator[Phrase]:
        for items in self._by_key.values():
            yield from items

    def __len__(self) -> int:
        return len(self._by_id)


def _parse_entry(key: str, entry: Any, seen_ids: set[str]) -> tuple[Phrase | None, str]:
    """(phrase, "") or (None, reason)."""
    if not isinstance(entry, Mapping):
        return None, "entry is not an object"
    phrase_id = entry.get("id")
    if not isinstance(phrase_id, str) or not phrase_id.strip():
        return None, "missing id"
    phrase_id = phrase_id.strip()
    if phrase_id in seen_ids:
        return None, f"duplicate id {phrase_id!r}"
    text = entry.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, f"{phrase_id}: empty text"
    tone = entry.get("tone")
    if not isinstance(tone, str) or tone not in TONES:
        return None, f"{phrase_id}: unknown tone {tone!r}"
    state = entry.get("state", "talk")
    if not isinstance(state, str) or state not in STATES:
        return None, f"{phrase_id}: unknown state {state!r}"
    required_state = REQUIRED_STATES.get(key)
    if required_state is not None and state != required_state:
        # Keep the line (its text is fine) but never let it undo the posture.
        log.warning("phrase %s: state %r replaced by %r, required for %s",
                    phrase_id, state, required_state, key)
        state = required_state
    portrait = entry.get("portrait", "")
    if not isinstance(portrait, str):
        return None, f"{phrase_id}: portrait is not a string"
    requires_raw = entry.get("requires", [])
    if not isinstance(requires_raw, list) or not all(isinstance(r, str) for r in requires_raw):
        return None, f"{phrase_id}: requires is not a list of strings"
    unknown = [r for r in requires_raw if r not in REQUIREMENTS]
    if unknown:
        return None, f"{phrase_id}: unknown requirement(s) {unknown}"
    try:
        fields = template_fields(text)
    except ValueError as exc:
        return None, f"{phrase_id}: {exc}"
    unknown_fields = sorted(set(fields) - PLACEHOLDERS)
    if unknown_fields:
        return None, f"{phrase_id}: unknown placeholder(s) {unknown_fields}"
    # A placeholder is itself a requirement: a phrase about {game} without a
    # game would otherwise only be filtered by the render error.
    requires = list(dict.fromkeys(requires_raw))
    requires.extend(f for f in dict.fromkeys(fields) if f in REQUIREMENTS and f not in requires)
    return Phrase(
        id=phrase_id,
        key=key,
        text=text.strip(),
        tone=tone,
        state=state,
        portrait=portrait.strip(),
        requires=tuple(requires),
    ), ""


def tone_weights(personality: PersonalitySettings) -> dict[str, float]:
    """How likely each tone is, from the personality sliders (0..100)."""
    affection = personality.affection / 100.0
    humor = personality.humor / 100.0
    sarcasm = personality.sarcasm
    return {
        "warm": 0.6 + affection,
        "playful": 0.6 + humor,
        "sarcastic": 0.0 if sarcasm < SARCASM_FLOOR else 1.6 * sarcasm / 100.0,
        "caring": 0.8 + affection / 2.0,
    }


class PhraseSelector:
    """Picks, renders and records phrases.  See the module docstring for the rules."""

    def __init__(
        self,
        bank: PhraseBank,
        history: PhraseHistoryRepository,
        settings: Callable[[], Settings],
        clock: Clock,
        rng: random.Random,
    ) -> None:
        self._bank = bank
        self._history = history
        self._settings = settings
        self._clock = clock
        self._rng = rng
        # In-memory mirror of the persisted history: per key the most recent
        # ids (newest first), plus a global use counter per id for LRU.
        self._recent: dict[str, deque[str]] = {}
        self._last_seq: dict[str, int] = {}
        self._last_used_at: dict[str, float] = {}
        self._seq = 0
        self.reload_history()

    @property
    def bank(self) -> PhraseBank:
        return self._bank

    def reload_history(self) -> None:
        """(Re)build the in-memory history from the database, e.g. after "delete history"."""
        self._recent.clear()
        self._last_seq.clear()
        self._last_used_at.clear()
        self._seq = 0
        try:
            rows = self._history.recent(PhraseHistoryRepository.MAX_ROWS)
        except Exception:
            log.exception("could not read phrase history; starting without anti-repeat memory")
            return
        for phrase_id, key, used_at in reversed(rows):  # oldest first
            self._remember(phrase_id, key, used_at)

    def last_used_at(self, key: str) -> float | None:
        """Epoch seconds when a phrase of ``key`` was last said (survives restarts), or None."""
        return self._last_used_at.get(key)

    # -- choosing --------------------------------------------------------------------------

    def pick(self, *keys: str, **variables: Any) -> PickedPhrase | None:
        """Choose a phrase for the first key that has an eligible one, without recording it.

        Used directly when only the phrase's *state* is needed (quiet mode).
        """
        settings = self._settings()
        weights = tone_weights(settings.personality)
        values = self._variables(settings, variables)
        for key in keys:
            candidates: list[tuple[Phrase, str, float]] = []
            for phrase in self._bank.phrases(key):
                weight = weights.get(phrase.tone, 0.0)
                if weight <= 0.0 or not self._requirements_met(phrase, values):
                    continue
                try:
                    text = render(phrase.text, values).strip()
                except (KeyError, ValueError, IndexError, AttributeError):
                    continue
                if text:
                    candidates.append((phrase, text, weight))
            if candidates:
                phrase, text = self._choose(key, candidates, len(self._bank.phrases(key)))
                return PickedPhrase(phrase, text)
        return None

    def say(self, *keys: str, solicited: bool = False, **variables: Any) -> SpeechRequested | None:
        """Pick a line for the first key with an eligible phrase, record it and wrap it as speech.

        Returns None when no key has a usable phrase.  The event is returned,
        not published: the caller decides whether and where it is shown.
        """
        picked = self.pick(*keys, **variables)
        if picked is None:
            return None
        self.record(picked.phrase)
        return SpeechRequested(
            text=picked.text,
            state=picked.phrase.state,
            portrait=picked.phrase.portrait,
            solicited=solicited,
            source=picked.phrase.key,
        )

    def record(self, phrase: Phrase) -> None:
        """Remember that ``phrase`` was said (memory first, then the database)."""
        now = self._clock.now()
        self._remember(phrase.id, phrase.key, now)
        try:
            self._history.add(phrase.id, phrase.key, now)
        except Exception:
            # Anti-repeat still works for this session from memory.
            log.exception("could not persist phrase history")

    # -- internals ---------------------------------------------------------------------------

    def _remember(self, phrase_id: str, key: str, used_at: float) -> None:
        self._seq += 1
        self._last_seq[phrase_id] = self._seq
        self._last_used_at[key] = used_at
        recent = self._recent.get(key)
        if recent is None:
            recent = self._recent[key] = deque(maxlen=MAX_REPEAT_WINDOW + 1)
        recent.appendleft(phrase_id)

    @staticmethod
    def _variables(settings: Settings, variables: Mapping[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {"name": settings.character.name}
        nickname = settings.character.user_nickname.strip()
        if nickname:
            values["user"] = nickname
        for name, value in variables.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            values[name] = value
        return values

    @staticmethod
    def _requirements_met(phrase: Phrase, values: Mapping[str, Any]) -> bool:
        return all(requirement in values for requirement in phrase.requires)

    def _choose(
        self, key: str, candidates: list[tuple[Phrase, str, float]], key_size: int
    ) -> tuple[Phrase, str]:
        """Anti-repeat over the key's *whole* phrase list (``key_size``), so phrases
        that are only temporarily ineligible (no nickname, sarcasm off) still count;
        when that excludes every eligible phrase, the least recently used one wins."""
        recent = list(self._recent.get(key, ()))
        window = min(key_size - 1, MAX_REPEAT_WINDOW)
        excluded = set(recent[:window])
        if recent and len(candidates) > 1:
            excluded.add(recent[0])
        pool = [c for c in candidates if c[0].id not in excluded]
        if not pool:
            pool = self._least_recently_used(candidates, recent[0] if recent else None)
        phrase, text, _weight = self._weighted(pool)
        return phrase, text

    def _least_recently_used(
        self, candidates: list[tuple[Phrase, str, float]], previous: str | None
    ) -> list[tuple[Phrase, str, float]]:
        options = [c for c in candidates if c[0].id != previous] or candidates
        oldest = min(self._last_seq.get(c[0].id, 0) for c in options)
        return [c for c in options if self._last_seq.get(c[0].id, 0) == oldest]

    def _weighted(self, pool: list[tuple[Phrase, str, float]]) -> tuple[Phrase, str, float]:
        total = sum(weight for _p, _t, weight in pool)
        threshold = self._rng.random() * total
        running = 0.0
        for item in pool:
            running += item[2]
            if threshold < running:
                return item
        return pool[-1]
