"""Shared test doubles for gifts, games and the shop.

The tamagotchi, the wallet, the repository and the memory store are real
(over the in-memory database); only the speech composer is faked, so the
tests see exactly which phrase keys were asked for.
"""

from __future__ import annotations

import random
import sqlite3
from collections.abc import Callable, Sequence
from typing import Any

from chibi.core.clock import FakeClock
from chibi.core.config import Settings
from chibi.core.event_bus import EventBus
from chibi.core.events import Event
from chibi.memory.database import Database
from chibi.memory.longterm import MemoryStore
from chibi.memory.repositories import StateRepository
from chibi.tamagotchi.gifts import GiftService
from chibi.tamagotchi.service import TamagotchiService


class FakeComposer:
    """Records ``compose`` calls; answers ``result`` (or raises ``error``)."""

    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.error: Exception | None = None
        self.calls: list[tuple[tuple[str, ...], bool]] = []
        self.variables: list[dict[str, Any]] = []

    def compose(self, *keys: str, solicited: bool = False, **variables: Any) -> bool:
        self.calls.append((keys, solicited))
        self.variables.append(variables)
        if self.error is not None:
            raise self.error
        return self.result

    @property
    def keys(self) -> list[tuple[str, ...]]:
        return [keys for keys, _solicited in self.calls]


class FlakyRepo(StateRepository):
    """A real repository whose writes to the keys in ``broken`` fail like a full disk."""

    def __init__(self, db: Database, clock: FakeClock | None = None) -> None:
        super().__init__(db, clock)
        self.broken: set[str] = set()
        self.writes: list[str] = []

    def set(self, key: str, value: Any) -> None:
        if key in self.broken:
            raise sqlite3.OperationalError("database or disk is full")
        self.writes.append(key)
        super().set(key, value)


class ScriptedRandom(random.Random):
    """A ``random.Random`` whose ``choice`` / ``randint`` answer from scripts first."""

    def __init__(self, choices: Sequence[Any] = (), ints: Sequence[int] = (), seed: int = 0) -> None:
        super().__init__(seed)
        self.choices = list(choices)
        self.ints = list(ints)
        self.draws = 0

    def choice(self, seq: Sequence[Any]) -> Any:
        self.draws += 1
        if self.choices:
            return self.choices.pop(0)
        return super().choice(seq)

    def randint(self, a: int, b: int) -> int:
        if self.ints:
            return self.ints.pop(0)
        return super().randint(a, b)


class Recorder:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.events: list[Event] = []
        bus.subscribe(Event, self.events.append)

    def drain(self) -> list[Event]:
        self.bus.dispatch_pending()
        out, self.events[:] = list(self.events), []
        return out

    def of(self, kind: type) -> list[Any]:
        return [e for e in self.drain() if isinstance(e, kind)]


class World:
    """Everything a GiftService needs, real except the composer."""

    def __init__(self, db: Database, bus: EventBus, clock: FakeClock, *, repo: StateRepository | None = None,
                 memory: bool = True) -> None:
        self.db = db
        self.bus = bus
        self.clock = clock
        self.repo = repo if repo is not None else FlakyRepo(db, clock)
        self.settings = Settings()
        self.tamagotchi = TamagotchiService(bus, self.repo, clock)
        self.tamagotchi.load()
        self.composer = FakeComposer()
        self.memory = MemoryStore(db, bus, clock) if memory else None
        self.recorder = Recorder(bus)
        bus.dispatch_pending()
        self.recorder.events.clear()

    def gifts(self, **kwargs: Any) -> GiftService:
        return GiftService(self.bus, self.repo, self.clock, self.tamagotchi, self.composer, self.memory,
                           self.settings_getter, **kwargs)

    @property
    def settings_getter(self) -> Callable[[], Settings]:
        return lambda: self.settings
