"""What the settings / status / chat windows may ask of the application.

The windows depend only on this protocol and the plain view types below, so
they can be built and tested without the rest of the app (tests pass a fake).
The application (``chibi.app.ChibiApp``) implements it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from PySide6.QtGui import QPixmap

from chibi.core.config import Settings
from chibi.memory.repositories import GameTotals
from chibi.privacy.permissions import PermissionManager

if TYPE_CHECKING:  # pragma: no cover - typing only; the windows never import the AI service
    from chibi.ai.service import RequestTiming


@dataclass(frozen=True)
class NeedsView:
    """Tamagotchi needs, 0..100 each (hunger: 0 = full, 100 = starving)."""

    hunger: float
    energy: float
    happiness: float
    boredom: float
    affection: float
    friendship_level: int  # 1..10
    friendship_progress: float  # 0..1 towards the next level
    mood: str  # short Ukrainian description, e.g. "Весела"


@dataclass(frozen=True)
class ChatReply:
    text: str
    state: str = "talk"  # CharacterState value
    portrait: str = ""
    #: side effect the app performed, for the UI to reflect:
    #: "", "quiet_on", "quiet_off", "fed", "petted", "rest"
    action: str = ""


@dataclass(frozen=True)
class StatusView:
    character_name: str
    state: str  # CharacterState value
    observing: bool  # any observation permission active
    activity_supported: bool  # activity monitoring works on this OS
    ai_connected: bool  # always False in Phase 1
    #: Epoch seconds when quiet mode ends; ``inf`` = until lifted; None = not quiet.
    quiet_until: float | None
    do_not_disturb: bool


#: Long-term memory kinds shown as tabs in the memory viewer.
MEMORY_KINDS = ("fact", "preference", "relationship", "game_event", "joke")


@dataclass(frozen=True)
class AIStatusView:
    provider: str  # "none" | "ollama" | "anthropic" | "openai"
    model: str
    online: bool
    reason: str  # why offline, Ukrainian ("" when online)
    cloud: bool  # data leaves this PC (needs Permission.CLOUD_AI)
    requests_today: int
    requests_limit_day: int
    cost_today_usd: float
    cost_limit_usd: float
    cost_known: bool  # False for OpenAI-compatible models without a price set


@dataclass(frozen=True)
class MemoryView:
    id: int
    kind: str  # one of MEMORY_KINDS
    text: str
    source: str  # "chat" | "system" | "game" | "user"
    pinned: bool
    created_at: float
    updated_at: float


class AppFacade(Protocol):
    permissions: PermissionManager

    def settings(self) -> Settings:
        """A copy of the current settings."""
        ...

    def apply_settings(self, settings: Settings) -> None:
        """Validate, persist and apply new settings immediately."""
        ...

    def needs(self) -> NeedsView: ...

    def status(self) -> StatusView: ...

    def feed(self) -> None: ...

    def pet(self) -> None: ...

    def rest(self) -> None: ...

    def send_chat(self, text: str) -> ChatReply:
        """Offline (no-AI) reply in Phase 1; the reply is also spoken on the desktop."""
        ...

    def game_totals(self) -> list[GameTotals]: ...

    def known_games_count(self) -> int: ...

    def portrait(self, name: str) -> QPixmap | None:
        """Portrait from the character pack (``"calm"``, ``"angry"``...), or None."""
        ...

    def data_dir(self) -> Path: ...

    def has_tray(self) -> bool:
        """Whether the tray icon exists.  Without it click-through is refused: her own
        menu would be the only way to turn it off again, and click-through blocks it."""
        ...

    def stop_all_observation(self) -> None: ...

    def wipe_history(self) -> None: ...

    # -- Phase 2: AI and memory (callbacks always run on the Qt main thread) --

    def ai_status(self) -> AIStatusView: ...

    def ai_has_key(self, provider: str) -> bool: ...

    def ai_set_key(self, provider: str, key: str) -> None:
        """Store an API key in the OS credential store ("" deletes it).  Never logged."""
        ...

    def ai_test(self, on_done: Callable[[bool, str], None]) -> None:
        """Try a tiny request with the current settings; on_done(ok, Ukrainian message)."""
        ...

    def ai_list_models(self, on_done: Callable[[list[str], str], None]) -> None:
        """Ask the provider which models exist; on_done(models, error message or "")."""
        ...

    def ai_last_timing(self) -> RequestTiming | None:
        """Where the time of her last chat answer went (queue, model load, first words,
        whole answer - seconds, numbers only), or None before the first one."""
        ...

    def send_chat_stream(self, text: str, on_delta: Callable[[str], None],
                         on_done: Callable[[ChatReply], None]) -> None:
        """AI reply streamed through on_delta when AI is usable; otherwise the
        offline reply arrives through on_done only.  on_done always fires once.
        The app also says the reply on the desktop (her bubble, and her voice
        sentence by sentence while it streams); the window only shows it."""
        ...

    def list_memories(self, kind: str | None = None, query: str = "") -> list[MemoryView]: ...

    def add_memory(self, kind: str, text: str) -> None: ...

    def update_memory(self, memory_id: int, text: str) -> None: ...

    def set_memory_pinned(self, memory_id: int, pinned: bool) -> None: ...

    def delete_memory(self, memory_id: int) -> None: ...

    def clear_memories(self, kind: str | None = None) -> None: ...

    def emotion(self) -> dict[str, float]:
        """The 8 emotion values 0..100: happiness, energy, boredom, affection,
        curiosity, excitement, hunger, trust."""
        ...

    def wipe_all(self) -> None:
        """Delete all data (database, log files, the open chat) and return to
        first-run state (the app restarts its state)."""
        ...
