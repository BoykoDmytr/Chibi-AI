"""User settings.

Settings are plain dataclasses, persisted as JSON by
:class:`chibi.memory.repositories.SettingsRepository`.  Loading is tolerant:
unknown keys are ignored, values of the wrong type fall back to the default
and numbers are clamped to their documented range, so an old or hand-edited
database can never crash the app.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import re
from dataclasses import dataclass, field
from typing import Any, get_type_hints

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _range(default: float, lo: float, hi: float) -> Any:
    return field(default=default, metadata={"range": (lo, hi)})


@dataclass
class CharacterSettings:
    name: str = "Мія"
    #: How she addresses the user.  Empty -> neutral phrasing ("ти").
    user_nickname: str = ""
    pack: str = "mia"
    #: "sprite" (character pack frames) or "vector" (built-in QPainter chibi).
    renderer: str = "sprite"
    #: Display scale; 1.0 = 220 logical px tall.
    scale: float = _range(1.0, 0.5, 2.5)
    always_on_top: bool = True
    lock_position: bool = False
    walking_enabled: bool = True
    #: Fall back to the taskbar edge after being dropped mid-air.
    gravity: bool = True
    #: Mouse clicks pass through her (re-enable from the tray menu).
    click_through: bool = False
    show_bubbles: bool = True

    def __post_init__(self) -> None:
        self.name = self.name.strip()[:24] or "Мія"
        self.user_nickname = self.user_nickname.strip()[:32]
        if self.renderer not in ("sprite", "vector"):
            self.renderer = "sprite"
        # ``pack`` becomes a directory name: never allow path traversal.
        if not re.fullmatch(r"[a-z0-9_-]{1,32}", self.pack):
            self.pack = "mia"


@dataclass
class BehaviorSettings:
    do_not_disturb: bool = False
    #: Hard cap on lines she says without being asked, per rolling hour.
    max_unsolicited_per_hour: int = _range(6, 0, 30)
    #: Minimum gap between two unsolicited lines.
    min_seconds_between_unsolicited: int = _range(90, 20, 3600)
    #: How long "помовч" keeps her quiet.
    quiet_minutes: int = _range(60, 5, 600)
    #: Hide while any app is fullscreen (games, video, presentations).
    hide_on_fullscreen: bool = True
    #: Lower frame rate / pause walking under high CPU load or battery saver.
    auto_eco_mode: bool = True
    #: Fall asleep after this long without interaction (or user input, if permitted).
    sleep_after_idle_minutes: int = _range(10, 2, 240)


@dataclass
class PersonalitySettings:
    """0..100 sliders.  In Phase 1 (no AI) they steer the offline phrase
    selection and how often she speaks; Phase 2 feeds them to the LLM prompt."""

    humor: int = _range(70, 0, 100)
    sarcasm: int = _range(40, 0, 100)
    talkativeness: int = _range(50, 0, 100)
    curiosity: int = _range(60, 0, 100)
    affection: int = _range(60, 0, 100)
    independence: int = _range(50, 0, 100)
    commentary_frequency: int = _range(40, 0, 100)


@dataclass
class ActivitySettings:
    """Only used while Permission.APP_METADATA is granted."""

    poll_interval_seconds: float = _range(3.0, 1.0, 30.0)
    process_scan_interval_seconds: float = _range(15.0, 5.0, 120.0)
    idle_threshold_minutes: int = _range(5, 1, 120)
    #: Greet the user if they were away at least this long.
    return_greeting_minutes: int = _range(10, 1, 240)
    comment_on_games: bool = True
    comment_on_programming: bool = True
    comment_on_video: bool = True
    comment_on_browsing: bool = False
    #: Extra process names (lower case, e.g. "mybank.exe") treated as private.
    excluded_processes: list[str] = field(default_factory=list)
    #: User-added games: [{"process": "foo.exe", "title": "Foo"}].
    custom_games: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Settings:
    schema_version: int = SCHEMA_VERSION
    first_run_completed: bool = False
    character: CharacterSettings = field(default_factory=CharacterSettings)
    behavior: BehaviorSettings = field(default_factory=BehaviorSettings)
    personality: PersonalitySettings = field(default_factory=PersonalitySettings)
    activity: ActivitySettings = field(default_factory=ActivitySettings)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> Settings:
        if not isinstance(data, dict):
            return cls()
        return _load(cls, data)

    def copy(self) -> Settings:
        return copy.deepcopy(self)


def _load(klass: type, data: dict[str, Any]) -> Any:
    hints = get_type_hints(klass)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(klass):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints[f.name]
        if dataclasses.is_dataclass(hint):
            if isinstance(value, dict):
                kwargs[f.name] = _load(hint, value)
            continue
        coerced = _coerce(value, hint, f)
        if coerced is not _INVALID:
            kwargs[f.name] = coerced
        else:
            log.warning("ignoring invalid setting %s.%s=%r", klass.__name__, f.name, value)
    return klass(**kwargs)


_INVALID = object()


def _coerce(value: Any, hint: Any, f: dataclasses.Field) -> Any:
    origin = getattr(hint, "__origin__", None)
    if hint is bool:
        return value if isinstance(value, bool) else _INVALID
    if hint in (int, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _INVALID
        if value != value:  # NaN
            return _INVALID
        lo, hi = f.metadata.get("range", (float("-inf"), float("inf")))
        value = min(max(value, lo), hi)
        return round(value) if hint is int else float(value)
    if hint is str:
        return value if isinstance(value, str) else _INVALID
    if origin is list:
        if not isinstance(value, list):
            return _INVALID
        (item_hint,) = hint.__args__
        if item_hint is str:
            return [v.strip().lower() for v in value if isinstance(v, str) and v.strip()]
        if getattr(item_hint, "__origin__", None) is dict:
            return [
                {str(k): str(v) for k, v in item.items() if isinstance(v, str)}
                for item in value
                if isinstance(item, dict)
            ]
        return _INVALID
    return _INVALID
