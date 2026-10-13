"""A recording :class:`~chibi.settings.facade.AppFacade` for the window tests.

Permissions are real (a PermissionManager over the in-memory database) so the
privacy tests exercise the actual gate, not a mock of it.

Asynchronous Phase 2 calls (``ai_test``, ``ai_list_models``,
``send_chat_stream``) answer immediately by default, like an offline app.  With
``deferred = True`` they park their callbacks in ``pending`` instead, and the
test decides when (and in which order) the answer arrives - the way the AI
worker thread posts results back to the main thread later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from PySide6.QtGui import QPixmap

from chibi.core.config import Settings
from chibi.memory.repositories import GameTotals
from chibi.privacy.permissions import PermissionManager
from chibi.settings.facade import MEMORY_KINDS, AIStatusView, ChatReply, MemoryView, NeedsView, StatusView


class MemoryRefused(ValueError):
    """Like ``chibi.memory.longterm.MemoryRejectedError``."""

    def __init__(self, message: str, *, user_message: str) -> None:
        super().__init__(message)
        self.user_message = user_message


@dataclass
class StreamScript:
    """What ``send_chat_stream`` answers: text pieces, then the final reply."""

    deltas: list[str] = field(default_factory=list)
    reply: ChatReply = field(default_factory=lambda: ChatReply(text="Привіт!"))


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
        self.tray = True
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
        # --- Phase 2
        self.deferred = False
        self.pending: list[tuple[str, tuple]] = []
        self.ai_status_view = AIStatusView(
            provider="none", model="", online=False, reason="AI вимкнено", cloud=False,
            requests_today=0, requests_limit_day=400, cost_today_usd=0.0, cost_limit_usd=1.0,
            cost_known=True,
        )
        self.ai_status_error: Exception | None = None
        self.keys: dict[str, str] = {}
        self.key_calls: list[tuple[str, int]] = []  # (provider, key length) - never the key itself
        self.key_error: Exception | None = None
        self.test_result: tuple[bool, str] = (True, "Працює! claude-opus-5-5 відповідає: «Привіт»")
        self.test_calls = 0
        self.models_result: tuple[list[str], str] = (["llama3.2", "qwen3:8b"], "")
        self.models_calls = 0
        self.settings_at_check: list[Settings] = []
        self.streams: list[StreamScript] = []
        self.stream_error: Exception | None = None
        self.memories: list[MemoryView] = []
        self.memory_calls: list[tuple] = []
        self.memory_error: Exception | None = None
        self._next_memory_id = 1
        self._memory_clock = 1_700_000_000.0
        self.emotion_values: dict[str, float] = {
            "happiness": 70.0, "energy": 80.0, "boredom": 20.0, "affection": 55.0,
            "curiosity": 62.0, "excitement": 35.0, "hunger": 30.0, "trust": 48.0,
        }
        self.emotion_error: Exception | None = None

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

    def has_tray(self) -> bool:
        return self.tray

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

    # --- Phase 2: AI ---------------------------------------------------------------------------

    def _answer(self, kind: str, callback: Callable, *args) -> None:
        if self.deferred:
            self.pending.append((kind, (callback, *args)))
        else:
            callback(*args)

    def resolve(self, kind: str | None = None) -> None:
        """Deliver the oldest parked answer (of *kind*, if given)."""
        for i, (k, call) in enumerate(self.pending):
            if kind is None or k == kind:
                del self.pending[i]
                callback, *args = call
                callback(*args)
                return
        raise AssertionError(f"nothing pending for {kind!r}")

    def ai_status(self) -> AIStatusView:
        if self.ai_status_error is not None:
            raise self.ai_status_error
        return self.ai_status_view

    def ai_has_key(self, provider: str) -> bool:
        return bool(self.keys.get(provider))

    def ai_set_key(self, provider: str, key: str) -> None:
        if self.key_error is not None:
            raise self.key_error
        self.key_calls.append((provider, len(key)))
        if key:
            self.keys[provider] = key
        else:
            self.keys.pop(provider, None)

    def ai_test(self, on_done: Callable[[bool, str], None]) -> None:
        self.test_calls += 1
        self.settings_at_check.append(self._settings.copy())
        self._answer("test", on_done, *self.test_result)

    def ai_list_models(self, on_done: Callable[[list[str], str], None]) -> None:
        self.models_calls += 1
        self.settings_at_check.append(self._settings.copy())
        models, error = self.models_result
        self._answer("models", on_done, list(models), error)

    def send_chat_stream(self, text: str, on_delta: Callable[[str], None],
                         on_done: Callable[[ChatReply], None]) -> None:
        self.chat_sent.append(text)
        if self.chat_error is not None:
            raise self.chat_error
        if self.streams:
            script = self.streams.pop(0)
        else:
            script = StreamScript(reply=self.replies.pop(0) if self.replies else ChatReply(text="Привіт!"))
        for piece in script.deltas:
            self._answer("delta", on_delta, piece)
        self._answer("done", on_done, script.reply)

    # --- Phase 2: memory -------------------------------------------------------------------------

    def _tick(self) -> float:
        self._memory_clock += 60.0
        return self._memory_clock

    def seed_memory(self, kind: str, text: str, *, source: str = "chat", pinned: bool = False) -> MemoryView:
        now = self._tick()
        memory = MemoryView(id=self._next_memory_id, kind=kind, text=text, source=source, pinned=pinned,
                            created_at=now, updated_at=now)
        self._next_memory_id += 1
        self.memories.append(memory)
        return memory

    def list_memories(self, kind: str | None = None, query: str = "") -> list[MemoryView]:
        if self.memory_error is not None:
            raise self.memory_error
        needle = query.casefold().strip()
        found = [m for m in self.memories
                 if (kind is None or m.kind == kind) and (not needle or needle in m.text.casefold())]
        return sorted(found, key=lambda m: (not m.pinned, -m.updated_at, -m.id))

    def add_memory(self, kind: str, text: str) -> None:
        assert kind in MEMORY_KINDS
        self.memory_calls.append(("add", kind, text))
        cleaned = " ".join(text.split())
        # like MemoryStore.add: personal data is silently not stored
        if not cleaned or "пароль" in cleaned.casefold() or any(ch.isdigit() for ch in cleaned):
            return
        self.seed_memory(kind, cleaned, source="user")

    def update_memory(self, memory_id: int, text: str) -> None:
        self.memory_calls.append(("update", memory_id, text))
        if "пароль" in text.casefold():
            raise MemoryRefused("sensitive", user_message="Схоже на особисті дані — таке я не запам'ятовую.")
        for i, m in enumerate(self.memories):
            if m.id == memory_id:
                self.memories[i] = replace(m, text=text.strip(), updated_at=self._tick())

    def set_memory_pinned(self, memory_id: int, pinned: bool) -> None:
        self.memory_calls.append(("pin", memory_id, pinned))
        for i, m in enumerate(self.memories):
            if m.id == memory_id:
                self.memories[i] = replace(m, pinned=pinned)

    def delete_memory(self, memory_id: int) -> None:
        self.memory_calls.append(("delete", memory_id))
        self.memories = [m for m in self.memories if m.id != memory_id]

    def clear_memories(self, kind: str | None = None) -> None:
        self.memory_calls.append(("clear", kind))
        self.memories = [m for m in self.memories if kind is not None and m.kind != kind]

    def emotion(self) -> dict[str, float]:
        if self.emotion_error is not None:
            raise self.emotion_error
        return dict(self.emotion_values)
