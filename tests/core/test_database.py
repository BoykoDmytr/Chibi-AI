import sqlite3

import pytest

from chibi.core.config import Settings
from chibi.core.events import Permission
from chibi.memory.database import MIGRATIONS, Database
from chibi.memory.repositories import (
    GameSessionRepository,
    PermissionRepository,
    PhraseHistoryRepository,
    SettingsRepository,
    StateRepository,
)


def test_fresh_database_is_migrated(db):
    assert db.schema_version == len(MIGRATIONS)


def test_settings_persist(tmp_path):
    path = tmp_path / "x.db"
    db = Database(path)
    s = Settings()
    s.character.name = "Кіра"
    s.personality.sarcasm = 77
    SettingsRepository(db).save(s)
    db.close()
    db2 = Database(path)
    loaded = SettingsRepository(db2).load()
    db2.close()
    assert loaded.character.name == "Кіра"
    assert loaded.personality.sarcasm == 77


def test_corrupt_settings_fall_back_to_defaults(db):
    db.conn.execute("INSERT INTO settings(key, value, updated_at) VALUES('app', '{not json', 0)")
    assert SettingsRepository(db).load() == Settings()


def test_state_roundtrip_and_delete(db):
    repo = StateRepository(db)
    assert repo.get("pos", {"x": 1}) == {"x": 1}
    repo.set("pos", {"x": 5, "screen": "DISPLAY1"})
    assert repo.get("pos") == {"x": 5, "screen": "DISPLAY1"}
    repo.delete("pos")
    assert repo.get("pos") is None


def test_transaction_rolls_back_on_error(db):
    repo = StateRepository(db)
    repo.set("a", 1)
    with pytest.raises(RuntimeError), db.transaction() as conn:
        conn.execute("UPDATE kv_state SET value = '2' WHERE key = 'a'")
        raise RuntimeError
    assert repo.get("a") == 1


def test_permissions_and_audit_log(db):
    repo = PermissionRepository(db)
    assert all(v is False for v in repo.load_all().values())
    repo.save(Permission.APP_METADATA, True, 10.0)
    repo.save(Permission.APP_METADATA, False, 20.0)
    assert repo.load_all()[Permission.APP_METADATA] is False
    log = repo.history()
    assert [(r.permission, r.granted, r.at) for r in log] == [
        (Permission.APP_METADATA, False, 20.0),
        (Permission.APP_METADATA, True, 10.0),
    ]


def test_phrase_history_is_capped(db, monkeypatch):
    repo = PhraseHistoryRepository(db)
    monkeypatch.setattr(PhraseHistoryRepository, "MAX_ROWS", 5)
    for i in range(12):
        repo.add(f"p{i}", "k", float(i))
    recent = repo.recent()
    assert len(recent) == 5
    assert recent[0][0] == "p11"


def test_game_sessions_totals(db):
    repo = GameSessionRepository(db)
    repo.add("stalker2", "S.T.A.L.K.E.R. 2", 0, 3600)
    repo.add("stalker2", "S.T.A.L.K.E.R. 2", 5000, 6800)
    repo.add("cs2", "Counter-Strike 2", 0, 600)
    totals = repo.totals()
    assert totals[0].game_id == "stalker2"
    assert totals[0].total_seconds == 5400
    assert totals[0].sessions == 2
    assert repo.total_for("cs2") == (600.0, 1)


def test_wipe_history_keeps_settings(db):
    SettingsRepository(db).save(Settings())
    GameSessionRepository(db).add("g", "G", 0, 1)
    PhraseHistoryRepository(db).add("p", "k", 1)
    db.wipe_history()
    assert GameSessionRepository(db).totals() == []
    assert PhraseHistoryRepository(db).recent() == []
    assert db.conn.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 1


def test_wipe_history_keeps_the_permission_audit_log(db):
    """Grants stay active after "delete history", so their record must stay too."""
    repo = PermissionRepository(db)
    repo.save(Permission.APP_METADATA, True, 10.0)
    db.wipe_history()
    assert repo.load_all()[Permission.APP_METADATA] is True
    history = [(r.permission, r.granted, r.at) for r in repo.history()]
    assert history == [(Permission.APP_METADATA, True, 10.0)]


def test_secure_delete_is_on(db, tmp_path):
    assert db.conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
    on_disk = Database(tmp_path / "s.db")
    try:
        assert on_disk.conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
    finally:
        on_disk.close()


@pytest.mark.parametrize("wipe", ["wipe_history", "wipe_all"])
def test_wiped_rows_leave_no_bytes_on_disk_before_close(tmp_path, wipe):
    """The compacted file must replace the old pages right away, not at the next
    clean close: a crash or kill would otherwise leave the deleted titles in chibi.db.
    secure_delete is forced off here, as in SQLite builds where it defaults off
    and someone turns it off again, so only the vacuum/checkpoint is tested."""
    secret = b"SECRET_GAME_TITLE_XYZ"
    db = Database(tmp_path / "chibi.db")
    try:
        db.conn.execute("PRAGMA secure_delete = 0")
        repo = GameSessionRepository(db)
        for _ in range(5):
            repo.add("x", secret.decode(), 1.0, 500.0)
        getattr(db, wipe)()
        for path in tmp_path.iterdir():
            assert secret not in path.read_bytes(), path.name
    finally:
        db.close()


def test_wipe_all_removes_everything(tmp_path):
    db = Database(tmp_path / "w.db")
    SettingsRepository(db).save(Settings())
    StateRepository(db).set("needs", {"hunger": 3})
    PermissionRepository(db).save(Permission.APP_METADATA, True, 1)
    db.wipe_all()
    for table in ("settings", "kv_state", "permissions", "permission_log"):
        assert db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    db.close()


def test_newer_schema_is_refused(tmp_path):
    path = tmp_path / "future.db"
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {len(MIGRATIONS) + 1}")
    conn.close()
    with pytest.raises(RuntimeError):
        Database(path)


def test_cross_thread_use_is_an_error(db):
    import threading

    errors = []

    def use():
        try:
            db.conn.execute("SELECT 1")
        except sqlite3.ProgrammingError as exc:
            errors.append(exc)

    t = threading.Thread(target=use)
    t.start()
    t.join()
    assert errors, "sqlite connection must be main-thread only"
