"""What the settings / status / chat windows may ask of the application.

The windows depend only on this protocol and the plain view types below, so
they can be built and tested without the rest of the app (tests pass a fake).
The application (``chibi.app.ChibiApp``) implements it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PySide6.QtGui import QPixmap

from chibi.core.config import Settings
from chibi.memory.repositories import GameTotals
from chibi.privacy.permissions import PermissionManager


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

    def wipe_all(self) -> None:
        """Delete all data (database, log files, the open chat) and return to
        first-run state (the app restarts its state)."""
        ...
