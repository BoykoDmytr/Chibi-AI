"""SQLite storage.

One connection, used only from the Qt main thread (sqlite3's default
``check_same_thread=True`` turns accidental cross-thread use into an error
instead of a heisenbug).  Worker threads hand data to the main thread through
the event bus.

Schema changes are append-only migrations tracked with ``PRAGMA user_version``.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

MIGRATIONS: list[str] = [
    # v1
    """
    CREATE TABLE settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at REAL NOT NULL
    );
    CREATE TABLE kv_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at REAL NOT NULL
    );
    CREATE TABLE permissions (
        name TEXT PRIMARY KEY,
        granted INTEGER NOT NULL,
        updated_at REAL NOT NULL
    );
    CREATE TABLE permission_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        granted INTEGER NOT NULL,
        at REAL NOT NULL
    );
    CREATE TABLE phrase_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phrase_id TEXT NOT NULL,
        phrase_key TEXT NOT NULL,
        used_at REAL NOT NULL
    );
    CREATE INDEX phrase_history_key ON phrase_history(phrase_key, used_at);
    CREATE TABLE game_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id TEXT NOT NULL,
        title TEXT NOT NULL,
        started_at REAL NOT NULL,
        ended_at REAL NOT NULL,
        duration REAL NOT NULL
    );
    CREATE INDEX game_sessions_game ON game_sessions(game_id);
    """,
]

#: Tables holding history about the user (cleared by "delete history").  The
#: permission audit log is deliberately not one of them: while grants stay
#: active, their record of when they were given must stay too.
HISTORY_TABLES = ("phrase_history", "game_sessions")
#: Every table with user data (cleared by "delete all data").
ALL_TABLES = ("settings", "kv_state", "permissions", "permission_log", *HISTORY_TABLES)


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        # isolation_level=None: no implicit transactions; every write goes
        # through transaction() with an explicit BEGIN/COMMIT.
        self._conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        # Overwrite deleted content with zeros instead of leaving it in free
        # pages: python.org builds of SQLite default to off, and "delete" must
        # mean the bytes are gone, not merely unlinked.
        self._conn.execute("PRAGMA secure_delete = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
        self.migrate()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    @property
    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self) -> None:
        current = self.schema_version
        if current > len(MIGRATIONS):
            raise RuntimeError(
                f"database schema v{current} is newer than this app (v{len(MIGRATIONS)}); "
                "refusing to touch it"
            )
        for version in range(current + 1, len(MIGRATIONS) + 1):
            log.info("migrating database to schema v%d", version)
            with self.transaction() as conn:
                for statement in MIGRATIONS[version - 1].split(";"):
                    if statement.strip():
                        conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {version}")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Atomic block: commits on success, rolls back on any exception."""
        conn = self._conn
        conn.execute("BEGIN")
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()

    def wipe_history(self) -> None:
        """Delete game sessions and spoken lines; settings, permissions and their audit log stay."""
        with self.transaction() as conn:
            for table in HISTORY_TABLES:
                conn.execute(f"DELETE FROM {table}")
        self._vacuum()

    def wipe_all(self) -> None:
        """Delete every row of user data; the app behaves like a first run afterwards."""
        with self.transaction() as conn:
            for table in ALL_TABLES:
                conn.execute(f"DELETE FROM {table}")
        self._vacuum()

    def _vacuum(self) -> None:
        # Deleted rows can linger in free pages / the WAL; make deletion real.
        # In WAL mode VACUUM writes the compacted database into the WAL only, so
        # the main file keeps the old pages until the next checkpoint (normally
        # at close, never if the app is killed): checkpoint again right away.
        if self.path != ":memory:":
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.execute("VACUUM")
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:  # pragma: no cover
            log.exception("closing database failed")
