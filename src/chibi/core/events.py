"""Structured events exchanged through the :class:`~chibi.core.event_bus.EventBus`.

Events are immutable dataclasses.  Producers may live on worker threads;
consumers always run on the Qt main thread (the bus dispatches there).

Privacy rule: an event that describes the user's activity never carries the
identity of an app the privacy filter marked private.  Such apps are reported
as ``AppInfo.private_app()`` (no process name, no title), so no consumer can
leak what it never received.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum


class Priority(IntEnum):
    """Queue priority; when the bus is full, lower priority events are dropped first."""

    LOW = 0
    NORMAL = 1
    HIGH = 2


class ActivityCategory(StrEnum):
    GAMING = "gaming"
    PROGRAMMING = "programming"
    BROWSING = "browsing"
    WATCHING_VIDEO = "watching_video"
    WORKING = "working"
    IDLE = "idle"
    UNKNOWN = "unknown"
    PRIVATE = "private"  # excluded by the privacy filter: never commented on or stored


class Permission(StrEnum):
    """Separately granted capabilities.  All are OFF on first run."""

    APP_METADATA = "app_metadata"  # foreground process name / category, app launches, input-idle time
    WINDOW_TITLES = "window_titles"  # read window titles transiently for classification (never stored)
    SCREEN_CAPTURE = "screen_capture"  # Phase 3
    CLOUD_VISION = "cloud_vision"  # Phase 3: send frames to a cloud provider
    CLOUD_AI = "cloud_ai"  # Phase 2: send text context to a cloud LLM
    MICROPHONE = "microphone"  # Phase 4


#: Permissions that observe the user; the kill switch revokes all of them.
OBSERVATION_PERMISSIONS: frozenset[Permission] = frozenset(
    {
        Permission.APP_METADATA,
        Permission.WINDOW_TITLES,
        Permission.SCREEN_CAPTURE,
        Permission.CLOUD_VISION,
        Permission.MICROPHONE,
    }
)


class InteractionKind(StrEnum):
    CLICK = "click"  # single click / poke
    PET = "pet"
    FEED = "feed"
    REST = "rest"  # "go to sleep"
    CALL = "call"  # "come here"
    TALK = "talk"  # user opened the chat / asked to talk
    DRAG_START = "drag_start"
    DRAG_END = "drag_end"
    WAKE = "wake"


@dataclass(frozen=True, slots=True)
class AppInfo:
    """Identity of an application as far as the user permitted.

    ``process_name`` is the lower-cased executable name (``code.exe``).
    ``display_name`` is a friendly name (``VS Code``).  Window titles are never
    part of an event.
    """

    process_name: str
    display_name: str
    private: bool = False

    @staticmethod
    def private_app() -> AppInfo:
        return AppInfo(process_name="", display_name="", private=True)


@dataclass(frozen=True, slots=True)
class Event:
    """Base class.  ``priority`` is a class-level default that subclasses override."""

    timestamp: float = field(default_factory=time.time, kw_only=True)

    @property
    def priority(self) -> Priority:
        return Priority.NORMAL


# --- lifecycle -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AppStarted(Event):
    first_run: bool = False
    #: seconds since the app was last closed (None on first run / unknown)
    away_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class AppShuttingDown(Event):
    @property
    def priority(self) -> Priority:
        return Priority.HIGH


@dataclass(frozen=True, slots=True)
class SettingsChanged(Event):
    section: str = ""  # "character" | "behavior" | "personality" | "activity" | ""


# --- privacy -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PermissionChanged(Event):
    permission: Permission
    granted: bool

    @property
    def priority(self) -> Priority:
        return Priority.HIGH


@dataclass(frozen=True, slots=True)
class ObservationStopped(Event):
    """The kill switch was used: every observation permission was revoked."""

    @property
    def priority(self) -> Priority:
        return Priority.HIGH


# --- activity (only produced while Permission.APP_METADATA is granted) -----------


@dataclass(frozen=True, slots=True)
class ForegroundAppChanged(Event):
    app: AppInfo
    category: ActivityCategory
    confidence: float  # 0..1, how sure the classifier is about ``category``


@dataclass(frozen=True, slots=True)
class AppLaunched(Event):
    app: AppInfo
    category: ActivityCategory

    @property
    def priority(self) -> Priority:
        return Priority.LOW


@dataclass(frozen=True, slots=True)
class AppClosed(Event):
    app: AppInfo
    category: ActivityCategory
    ran_seconds: float

    @property
    def priority(self) -> Priority:
        return Priority.LOW


@dataclass(frozen=True, slots=True)
class GameStarted(Event):
    game_id: str  # stable id from the games database ("stalker2") or "exe:<name>"
    title: str  # display title
    process_name: str


@dataclass(frozen=True, slots=True)
class GameEnded(Event):
    game_id: str
    title: str
    duration_seconds: float
    #: The game became private while running (the user excluded it): consumers
    #: must forget the session - no line about it, nothing persisted.  The
    #: title is empty in that case.
    forgotten: bool = False


@dataclass(frozen=True, slots=True)
class ActivityMilestone(Event):
    """The user has been doing the same kind of thing continuously for a while."""

    category: ActivityCategory
    app: AppInfo
    seconds: float


@dataclass(frozen=True, slots=True)
class UserIdle(Event):
    idle_seconds: float


@dataclass(frozen=True, slots=True)
class UserReturned(Event):
    away_seconds: float


# --- environment (no permission needed: carries no information about the user) ---


@dataclass(frozen=True, slots=True)
class FullscreenChanged(Event):
    """Some foreground window covers its whole monitor (game, video, presentation)."""

    active: bool


@dataclass(frozen=True, slots=True)
class SystemLoadChanged(Event):
    high: bool
    cpu_percent: float
    on_battery_saver: bool = False


# --- character / interaction ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CharacterInteraction(Event):
    kind: InteractionKind

    @property
    def priority(self) -> Priority:
        return Priority.HIGH


@dataclass(frozen=True, slots=True)
class InteractionOutcome(Event):
    """How the tamagotchi took an interaction; reactions speak based on this.

    ``outcome`` per kind:
      FEED: "fed" | "full" | "starving_fed"     PET: "happy" | "too_much"
      REST: "sleep" | "not_tired"               CLICK: "poke" | "annoyed"
      WAKE: "ok" | "grumpy"                     others: "ok"
    """

    kind: InteractionKind
    outcome: str

    @property
    def priority(self) -> Priority:
        return Priority.HIGH


@dataclass(frozen=True, slots=True)
class UserMessage(Event):
    """Text the user typed in the chat window.  Never logged."""

    text: str

    @property
    def priority(self) -> Priority:
        return Priority.HIGH


@dataclass(frozen=True, slots=True)
class SpeechRequested(Event):
    """Ask the desktop layer to say something.

    ``state`` is a :class:`chibi.character.states.CharacterState` value name
    (string, to keep this module free of character imports) played while the
    line is shown, e.g. ``"laugh"``.  ``portrait`` names a portrait from the
    character pack (``"angry"``) or is empty for the state's default.
    ``solicited`` is True when the user asked for it (click, chat, menu) - such
    lines bypass the unsolicited-speech budget.
    """

    text: str
    state: str = "talk"
    portrait: str = ""
    solicited: bool = False
    source: str = ""  # which rule produced it, for diagnostics

    @property
    def priority(self) -> Priority:
        return Priority.HIGH if self.solicited else Priority.NORMAL


@dataclass(frozen=True, slots=True)
class QuietModeChanged(Event):
    """The user asked her to be quiet ("помовч") - or lifted it."""

    enabled: bool
    until: float | None = None  # epoch seconds; None = until lifted


@dataclass(frozen=True, slots=True)
class DoNotDisturbChanged(Event):
    enabled: bool


@dataclass(frozen=True, slots=True)
class NeedsChanged(Event):
    """Tamagotchi needs snapshot (0..100 each) after a meaningful change."""

    hunger: float
    energy: float
    happiness: float
    boredom: float
    affection: float
    friendship_level: int


# --- Phase 2: AI, memory, emotion ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AIStatusChanged(Event):
    """The configured AI provider became usable or unusable."""

    online: bool
    provider: str  # "none" | "ollama" | "anthropic" | "openai"
    model: str
    #: Why it is offline, short Ukrainian text for the UI ("немає ключа", "ліміт на сьогодні"...).
    reason: str = ""


@dataclass(frozen=True, slots=True)
class MemoryChanged(Event):
    """Long-term memory was added to, edited or cleared (UI refresh)."""

    kind: str = ""  # memory kind affected, "" = several / all


@dataclass(frozen=True, slots=True)
class EmotionChanged(Event):
    """Emotion snapshot (0..100 each) after a meaningful change."""

    happiness: float
    energy: float
    boredom: float
    affection: float
    curiosity: float
    excitement: float
    hunger: float
    trust: float
    #: CharacterState value that best expresses the mood ("happy", "sad", "curious", "idle"...).
    dominant: str = "idle"


@dataclass(frozen=True, slots=True)
class InitiativeAction(Event):
    """A non-speech autonomous action chosen by the decision engine, for the
    desktop controller: "wave", "approach_cursor", "sit", "sleep", "jump", "dance"."""

    action: str


# --- Phase 4: voice ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VoiceStateChanged(Event):
    """The voice service changed state (UI: microphone indicator, listening pose)."""

    state: str  # chibi.voice.service.VoiceState value: "off" | "idle" | "standby" | "listening" ...
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ConversationActive(Event):
    """The user is talking with her right now: speaking, being recognised, waiting
    for her answer or hearing it, or inside the follow-up window - also a typed
    chat reply in flight.  Published by the app on every change.  While active,
    background AI work (screen analysis, observer decisions, memory extraction,
    AI-phrased reactions) waits, so her answer never queues behind it."""

    active: bool

    @property
    def priority(self) -> Priority:
        return Priority.HIGH


@dataclass(frozen=True, slots=True)
class VoiceTranscript(Event):
    """What the user said to her, recognised locally.  Never logged.

    ``origin`` says how it reached her: ``"push"`` (push-to-talk), ``"named"``
    (the open microphone, addressed by her name) or ``"heard"`` (the open
    microphone without her name: "open" mode, the conversation window) - that
    may be the TV or someone else in the room, so it is never mined for
    long-term memories."""

    text: str
    origin: str = "push"

    @property
    def priority(self) -> Priority:
        return Priority.HIGH


@dataclass(frozen=True, slots=True)
class SpeechPlayback(Event):
    """Her voice started / stopped (the controller holds TALK while playing)."""

    playing: bool
    duration: float = 0.0


# --- Phase 3: screen vision --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VisionStateChanged(Event):
    """Capture indicator: ``state`` is a chibi.vision.service.VisionState value."""

    state: str
    app_title: str = ""
    reason: str = ""

    @property
    def priority(self) -> Priority:
        return Priority.HIGH


@dataclass(frozen=True, slots=True)
class VisionObserved(Event):
    """Something she saw in an allowed window (validated vocabulary only;
    ``description`` is sanitised untrusted text for the AI prompt, never
    stored or logged)."""

    app_title: str
    scene: str
    event: str
    confidence: float
    description: str = ""
    emotion: str = "idle"
    confirmed: bool = False
    #: The reaction policy decided she should comment.
    react: bool = False
