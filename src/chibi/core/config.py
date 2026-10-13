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


#: Longest character description the user can write for her (see ``persona``).
PERSONA_MAX_CHARS = 1500
_PERSONA_STRIP = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\u200b-\u200f\u2028-\u202e\u2066-\u2069]")


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
    #: Her character in the user's own words (background, manner, interests...),
    #: given to the AI model.  The app's own rules (Ukrainian, privacy, the reply
    #: format) still come first; the phrase bank (no AI) cannot follow it.
    persona: str = ""

    def __post_init__(self) -> None:
        text = _PERSONA_STRIP.sub("", self.persona.replace("\r\n", "\n").replace("\r", "\n"))
        self.persona = re.sub(r"\n{3,}", "\n\n", text).strip()[:PERSONA_MAX_CHARS]


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


AI_PROVIDERS = ("none", "ollama", "anthropic", "openai")
AI_EFFORTS = ("low", "medium", "high")


@dataclass
class AISettings:
    """Phase 2: the LLM that phrases her lines and chats.  Rules still decide
    *when* she speaks; the model only decides *how*.  Any provider whose host
    is not this machine needs Permission.CLOUD_AI (data leaves the PC)."""

    provider: str = "none"
    #: Empty -> provider default (anthropic: claude-opus-5-5; ollama: first installed model).
    model: str = ""
    ollama_url: str = "http://localhost:11434"
    #: Any OpenAI-compatible server (OpenAI, LM Studio, ...).
    openai_base_url: str = "https://api.openai.com/v1"
    #: Price per 1M tokens for OpenAI-compatible models (unknown to us); 0 = unknown.
    openai_price_input_per_mtok: float = _range(0.0, 0.0, 1000.0)
    openai_price_output_per_mtok: float = _range(0.0, 0.0, 1000.0)
    #: Claude effort for companion lines (thinking depth / spend).
    effort: str = "low"
    use_for_chat: bool = True
    use_for_reactions: bool = True
    stream: bool = True
    #: Remember facts the user shares in chat (never anything from the screen).
    remember_facts: bool = True
    max_requests_per_hour: int = _range(60, 1, 2000)
    max_requests_per_day: int = _range(400, 1, 20000)
    #: Hard daily spending cap for priced providers, in USD.
    max_cost_per_day_usd: float = _range(1.0, 0.01, 100.0)
    request_timeout_seconds: float = _range(30.0, 5.0, 180.0)
    #: Recent chat turns sent with each request (short-term memory).
    context_turns: int = _range(12, 2, 60)
    #: Relevant long-term memories sent with each request.
    memory_items: int = _range(6, 0, 20)
    #: Ollama: keep the model in memory this long after the last request (its own
    #: default unloads it after 5 minutes, and reloading takes 5-20 s).
    keep_model_loaded_minutes: int = _range(30, 5, 240)

    def __post_init__(self) -> None:
        if self.provider not in AI_PROVIDERS:
            self.provider = "none"
        if self.effort not in AI_EFFORTS:
            self.effort = "low"
        self.model = self.model.strip()[:120]
        for attr, default in (("ollama_url", "http://localhost:11434"),
                              ("openai_base_url", "https://api.openai.com/v1")):
            value = getattr(self, attr).strip().rstrip("/")
            if not re.fullmatch(r"https?://[^\s/]+(/[^\s]*)?", value):
                value = default
            setattr(self, attr, value[:300])


TTS_ENGINES = ("piper", "azure")
PIPER_SPEAKERS = ("tetiana", "lada", "mykyta")
AZURE_VOICES = ("uk-UA-PolinaNeural", "uk-UA-OstapNeural")
STT_MODELS = ("whisper-base", "whisper-small", "whisper-medium")
#: Which of her lines are spoken aloud: everything, only answers to the user, none.
SPEAK_MODES = ("all", "replies", "off")
PTT_MODES = ("toggle", "hold")
#: "push": the key or the microphone button; "name": the microphone stays open and
#: she answers when called by name, then follows the conversation for a while;
#: "open": the microphone stays open and everything she hears is meant for her.
LISTEN_MODES = ("push", "name", "open")


@dataclass
class VoiceSettings:
    """Phase 4.  Recognition is local (Whisper); the voice is local (Piper)
    unless the user picks Azure (cloud: needs Permission.CLOUD_AI).  The
    microphone additionally needs Permission.MICROPHONE."""

    tts_enabled: bool = False
    tts_engine: str = "piper"
    piper_speaker: str = "tetiana"
    azure_region: str = "westeurope"
    azure_voice: str = "uk-UA-PolinaNeural"
    speak: str = "all"
    #: Keep talking while a fullscreen game hides her ("voice reactions in games").
    speak_when_hidden: bool = True
    speed: float = _range(1.0, 0.6, 1.6)
    volume: int = _range(80, 0, 100)
    stt_enabled: bool = False
    stt_model: str = "whisper-small"
    ptt_hotkey: str = "ctrl+alt+space"
    #: "toggle": press, talk, she stops listening after a pause; "hold": walkie-talkie.
    ptt_mode: str = "toggle"
    end_silence_ms: int = _range(800, 400, 2500)
    max_utterance_seconds: int = _range(15, 5, 60)
    #: Experimental: microphone stays open waiting for "Мія, ...".
    #: How she listens (see LISTEN_MODES).  "" only while loading data from before
    #: this setting existed: then ``wake_phrase_enabled`` decides.
    listen_mode: str = ""
    #: "name" mode: after her answer, how long she keeps listening without her name.
    follow_up_seconds: int = _range(20, 0, 120)
    #: Pause the open microphone while a voice-call app (Discord, Teams, Zoom...) is in front.
    pause_in_calls: bool = True
    #: Deprecated: read ``listen_mode``.  Kept in step with it (``listen_mode != "push"``)
    #: so data written by older versions loads as before.
    wake_phrase_enabled: bool = False
    #: Keep the open microphone (name / open modes) while a fullscreen game is in front.
    wake_in_fullscreen: bool = False
    #: Empty -> system default device.
    input_device: str = ""
    output_device: str = ""
    #: CPU threads for recognition/synthesis (keep games smooth).
    cpu_threads: int = _range(2, 1, 8)

    def __post_init__(self) -> None:
        if self.listen_mode not in LISTEN_MODES:
            self.listen_mode = "name" if self.wake_phrase_enabled else "push"
        self.wake_phrase_enabled = self.listen_mode != "push"
        for attr, allowed in (("tts_engine", TTS_ENGINES), ("piper_speaker", PIPER_SPEAKERS),
                              ("azure_voice", AZURE_VOICES), ("speak", SPEAK_MODES),
                              ("stt_model", STT_MODELS), ("ptt_mode", PTT_MODES)):
            if getattr(self, attr) not in allowed:
                setattr(self, attr, allowed[1] if attr == "stt_model" else allowed[0])
        region = self.azure_region.strip().lower()
        self.azure_region = region if re.fullmatch(r"[a-z0-9]{3,40}", region) else "westeurope"
        from chibi.voice.hotkey_win import parse_hotkey  # pure module, no dependencies

        try:
            self.ptt_hotkey = parse_hotkey(self.ptt_hotkey).text
        except ValueError:
            self.ptt_hotkey = "ctrl+alt+space"
        self.input_device = self.input_device.strip()[:200]
        self.output_device = self.output_device.strip()[:200]


VISION_MODES = ("off", "manual", "auto", "game")
#: "allowed": only the apps on the lists; "foreground": whatever window is in front,
#: except private apps (and browsers only while window titles may be read, so
#: private tabs, banking and mail are recognised).
VISION_SCOPES = ("allowed", "foreground")
#: "events": she comments on notable game events; "observer": the model decides,
#: for games and work alike, whether what she sees is worth a word.
VISION_COMMENTARY = ("events", "observer")


@dataclass
class VisionSettings:
    """Phase 3.  Capture needs Permission.SCREEN_CAPTURE; frames go to a cloud
    model only with Permission.CLOUD_VISION.  Only the window in front is ever
    captured - from the allowed lists, or (``scope="foreground"``) any - and
    never one the privacy filter marks private."""

    mode: str = "off"
    interval_seconds: int = _range(30, 10, 600)
    #: Game ids from the games database ("stalker2") or "exe:<process>".
    allowed_games: list[str] = field(default_factory=list)
    #: Non-game apps the user allowed (lower-case process names).
    allowed_processes: list[str] = field(default_factory=list)
    min_confidence: float = _range(0.6, 0.3, 0.95)
    #: "Режим локального аналізу": never send frames to a cloud model.
    local_only: bool = False
    #: Vision model override; empty = the AI provider's model.
    model: str = ""
    #: Look less often while the PC is busy (GAME mode).
    adapt_to_load: bool = True
    #: Which windows she may look at (see VISION_SCOPES).
    scope: str = "allowed"
    #: What makes her speak about what she sees (see VISION_COMMENTARY).
    commentary: str = "events"

    def __post_init__(self) -> None:
        if self.mode not in VISION_MODES:
            self.mode = "off"
        if self.scope not in VISION_SCOPES:
            self.scope = "allowed"
        if self.commentary not in VISION_COMMENTARY:
            self.commentary = "events"
        self.model = self.model.strip()[:120]
        self.allowed_games = [g for g in dict.fromkeys(self.allowed_games)
                              if re.fullmatch(r"[a-z0-9_:.\-]{1,80}", g)]
        self.allowed_processes = [p for p in dict.fromkeys(self.allowed_processes)
                                  if re.fullmatch(r"[a-z0-9_.\- ]{1,80}\.exe", p)]


@dataclass
class GamingSettings:
    """Gaming companion mode (spec 8): how she behaves while the user plays."""

    #: Few or no bubbles over games, short comments, voice first.
    companion_mode: bool = True
    #: 0 = silent during games, 100 = chatty.
    reaction_frequency: int = _range(50, 0, 100)
    #: Allow gameplay hints and plot comments (spoilers).
    hints_and_spoilers: bool = False


@dataclass
class Settings:
    schema_version: int = SCHEMA_VERSION
    first_run_completed: bool = False
    character: CharacterSettings = field(default_factory=CharacterSettings)
    behavior: BehaviorSettings = field(default_factory=BehaviorSettings)
    personality: PersonalitySettings = field(default_factory=PersonalitySettings)
    activity: ActivitySettings = field(default_factory=ActivitySettings)
    ai: AISettings = field(default_factory=AISettings)
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)
    gaming: GamingSettings = field(default_factory=GamingSettings)

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
