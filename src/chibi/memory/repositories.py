"""Small typed repositories over :class:`Database`.  Main thread only."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from chibi.core.clock import Clock, SystemClock
from chibi.core.config import Settings
from chibi.core.events import Permission
from chibi.memory.database import Database

log = logging.getLogger(__name__)


class SettingsRepository:
    KEY = "app"

    def __init__(self, db: Database, clock: Clock | None = None) -> None:
        self._db = db
        self._clock = clock or SystemClock()

    def load(self) -> Settings:
        row = self._db.conn.execute("SELECT value FROM settings WHERE key = ?", (self.KEY,)).fetchone()
        if row is None:
            return Settings()
        try:
            return Settings.from_dict(json.loads(row["value"]))
        except (ValueError, TypeError):
            log.warning("stored settings are corrupt; using defaults")
            return Settings()

    def save(self, settings: Settings) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO settings(key, value, updated_at) VALUES(?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (self.KEY, json.dumps(settings.to_dict(), ensure_ascii=False), self._clock.now()),
            )


class StateRepository:
    """JSON key/value store for small pieces of persistent state
    (window position, tamagotchi needs, last shutdown time, ...)."""

    def __init__(self, db: Database, clock: Clock | None = None) -> None:
        self._db = db
        self._clock = clock or SystemClock()

    def get(self, key: str, default: Any = None) -> Any:
        row = self._db.conn.execute("SELECT value FROM kv_state WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except ValueError:
            log.warning("state %r is corrupt; using default", key)
            return default

    def set(self, key: str, value: Any) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO kv_state(key, value, updated_at) VALUES(?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key, json.dumps(value, ensure_ascii=False), self._clock.now()),
            )

    def delete(self, key: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM kv_state WHERE key = ?", (key,))


@dataclass(frozen=True)
class PermissionRecord:
    permission: Permission
    granted: bool
    at: float


class PermissionRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def load_all(self) -> dict[Permission, bool]:
        result = {p: False for p in Permission}
        for row in self._db.conn.execute("SELECT name, granted FROM permissions"):
            try:
                result[Permission(row["name"])] = bool(row["granted"])
            except ValueError:
                log.warning("unknown permission %r in database ignored", row["name"])
        return result

    def save(self, permission: Permission, granted: bool, at: float) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO permissions(name, granted, updated_at) VALUES(?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET granted = excluded.granted, "
                "updated_at = excluded.updated_at",
                (permission.value, int(granted), at),
            )
            conn.execute(
                "INSERT INTO permission_log(name, granted, at) VALUES(?, ?, ?)",
                (permission.value, int(granted), at),
            )

    def history(self, limit: int = 100) -> list[PermissionRecord]:
        rows = self._db.conn.execute(
            "SELECT name, granted, at FROM permission_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for row in rows:
            try:
                out.append(PermissionRecord(Permission(row["name"]), bool(row["granted"]), row["at"]))
            except ValueError:
                continue
        return out


class PhraseHistoryRepository:
    """Which offline phrases were said recently, so she doesn't repeat herself
    across restarts."""

    MAX_ROWS = 2000

    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, phrase_id: str, phrase_key: str, used_at: float) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO phrase_history(phrase_id, phrase_key, used_at) VALUES(?, ?, ?)",
                (phrase_id, phrase_key, used_at),
            )
            conn.execute(
                "DELETE FROM phrase_history WHERE id <= "
                "(SELECT id FROM phrase_history ORDER BY id DESC LIMIT 1 OFFSET ?)",
                (self.MAX_ROWS,),
            )

    def recent(self, limit: int = 200) -> list[tuple[str, str, float]]:
        """Most recent first: (phrase_id, phrase_key, used_at)."""
        rows = self._db.conn.execute(
            "SELECT phrase_id, phrase_key, used_at FROM phrase_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [(r["phrase_id"], r["phrase_key"], r["used_at"]) for r in rows]


@dataclass(frozen=True)
class GameTotals:
    game_id: str
    title: str
    total_seconds: float
    sessions: int
    last_played: float


class GameSessionRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, game_id: str, title: str, started_at: float, ended_at: float) -> None:
        duration = max(0.0, ended_at - started_at)
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO game_sessions(game_id, title, started_at, ended_at, duration) "
                "VALUES(?, ?, ?, ?, ?)",
                (game_id, title, started_at, ended_at, duration),
            )

    def totals(self, limit: int = 20) -> list[GameTotals]:
        rows = self._db.conn.execute(
            "SELECT game_id, MAX(title) AS title, SUM(duration) AS total, COUNT(*) AS n, "
            "MAX(ended_at) AS last FROM game_sessions GROUP BY game_id ORDER BY total DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [GameTotals(r["game_id"], r["title"], r["total"], r["n"], r["last"]) for r in rows]

    def total_for(self, game_id: str) -> tuple[float, int]:
        row = self._db.conn.execute(
            "SELECT COALESCE(SUM(duration), 0) AS total, COUNT(*) AS n FROM game_sessions WHERE game_id = ?",
            (game_id,),
        ).fetchone()
        return float(row["total"]), int(row["n"])
