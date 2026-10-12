"""A recording :class:`~chibi.settings.facade.AppFacade` for the window tests.

Permissions are real (a PermissionManager over the in-memory database) so the
privacy tests exercise the actual gate, not a mock of it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtGui import QPixmap

from chibi.core.config import Settings
from chibi.memory.repositories import GameTotals
from chibi.privacy.permissions import PermissionManager
from chibi.settings.facade import ChatReply, NeedsView, StatusView


class FakeFacade:
    def __init__(self, permissions: PermissionManager, data_dir: Path) -> None:
        self.permissions = permissions
        self._settings = Settings()
        self._data_dir = data_dir
        self.applied: list[Settings] = []
        self.calls: list[str] = []
        self.chat_sent: list[str] = []
        self.replies: list[ChatReply] = []
        self.chat_error: Exception | None = None
        self.totals: list[GameTotals] = []
        self.known_games = 152
        self.portraits: dict[str, QPixmap] = {}
        self.portrait_requests: list[str] = []
        self.needs_view = NeedsView(
            hunger=30.0, energy=80.0, happiness=70.0, boredom=20.0, affection=55.0,
            friendship_level=3, friendship_progress=0.45, mood="Весела",
        )
        self.status_view = StatusView(
            character_name="Мія", state="idle", observing=False, activity_supported=True,
            ai_connected=False, quiet_until=None, do_not_disturb=False,
        )

    # --- AppFacade ---------------------------------------------------------------------

    def settings(self) -> Settings:
        return self._settings.copy()

    def apply_settings(self, settings: Settings) -> None:
        self.applied.append(settings.copy())
        self._settings = settings.copy()

    def needs(self) -> NeedsView:
        return self.needs_view

    def status(self) -> StatusView:
        return replace(
            self.status_view,
            character_name=self._settings.character.name,
            observing=self.permissions.any_observation_active(),
        )

    def feed(self) -> None:
        self.calls.append("feed")

    def pet(self) -> None:
        self.calls.append("pet")

    def rest(self) -> None:
        self.calls.append("rest")

    def send_chat(self, text: str) -> ChatReply:
        self.chat_sent.append(text)
        if self.chat_error is not None:
            raise self.chat_error
        return self.replies.pop(0) if self.replies else ChatReply(text="Привіт!")

    def game_totals(self) -> list[GameTotals]:
        return list(self.totals)

    def known_games_count(self) -> int:
        return self.known_games

    def portrait(self, name: str) -> QPixmap | None:
        self.portrait_requests.append(name)
        return self.portraits.get(name)

    def data_dir(self) -> Path:
        return self._data_dir

    def stop_all_observation(self) -> None:
        self.calls.append("stop_all_observation")
        self.permissions.stop_all_observation()

    def wipe_history(self) -> None:
        self.calls.append("wipe_history")

    def wipe_all(self) -> None:
        self.calls.append("wipe_all")
