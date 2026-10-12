"""End-to-end wiring of ChibiApp (headless): the modules are unit-tested on
their own; this checks that app.py connects them correctly."""

import logging
import logging.handlers
import sqlite3
import time

import pytest
from PySide6.QtCore import QCoreApplication, QLockFile, Qt
from PySide6.QtWidgets import QLabel, QMessageBox

import chibi.app as app_module
from chibi.activity.backend import ForegroundInfo
from chibi.app import ALREADY_RUNNING_TEXT, LAST_SHUTDOWN_KEY, ChibiApp
from chibi.core import paths
from chibi.core.clock import FakeClock
from chibi.core.events import AppStarted, GameEnded, GameStarted, Permission, SpeechRequested
from chibi.memory.database import Database
from chibi.memory.repositories import StateRepository
from chibi.settings.theme import apply_dark_theme
from chibi.tamagotchi.service import KEY_LAST_SEEN


class FakeBackend:
    def __init__(self):
        self.fg = ForegroundInfo(pid=10, process_name="explorer.exe")
        self.procs = {"explorer.exe"}

    def is_supported(self):
        return True

    def foreground(self, read_title):
        return self.fg

    def idle_seconds(self):
        return 0.0

    def running_processes(self):
        return set(self.procs)


class UnsupportedBackend(FakeBackend):
    """No monitor thread: the test publishes the activity events itself."""

    def is_supported(self):
        return False


def pump(ms=200):
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.005)


@pytest.fixture
def make_app(qapp):
    apply_dark_theme(qapp)
    created = []

    def factory(backend=None, clock=None):
        app = ChibiApp(qapp, backend=backend or FakeBackend(), clock=clock)
        created.append(app)
        return app

    yield factory
    for app in created:
        app.close()


def test_first_start_greets_and_shows_character(make_app):
    app = make_app()
    spoken = []
    app.bus.subscribe(SpeechRequested, lambda e: spoken.append(e.source))
    app.start()
    pump(300)
    assert app.controller.is_visible()
    assert spoken and spoken[0] == "greet.first_run"
    assert not app.permissions.any_observation_active()  # nothing observed by default
    assert not app._monitor.is_running()


def test_permission_starts_monitor_and_game_triggers_reaction(make_app):
    backend = FakeBackend()
    app = make_app(backend)
    games, spoken = [], []
    app.bus.subscribe(GameStarted, lambda e: games.append(e.game_id))
    app.bus.subscribe(SpeechRequested, lambda e: spoken.append(e.source))
    app.start()
    pump(200)
    app.permissions.grant(Permission.APP_METADATA)
    pump(200)
    assert app._monitor.is_running()

    backend.procs.add("stalker2-win64-shipping.exe")
    backend.fg = ForegroundInfo(pid=11, process_name="stalker2-win64-shipping.exe")
    app._monitor.poll_once()
    pump(300)
    assert games == ["stalker2"]
    assert any(s.startswith("game_start.") for s in spoken)

    app.stop_all_observation()
    pump(300)
    assert not app._monitor.is_running()
    assert not app.permissions.any_observation_active()


def test_chat_side_effects(make_app):
    app = make_app()
    app.start()
    pump(100)
    app._tamagotchi.needs.hunger = 60
    reply = app.send_chat("поїж, будь ласка")
    assert reply.action == "fed"
    assert app._tamagotchi.needs.hunger < 60
    reply = app.send_chat("помовч")
    assert reply.action == "quiet_on"
    assert app.status().quiet_until is not None
    reply = app.send_chat("можеш говорити")
    assert reply.action == "quiet_off"
    assert app.status().quiet_until is None


def test_settings_apply_persist_and_survive_restart(make_app):
    app = make_app()
    app.start()
    s = app.settings()
    s.character.name = "Кіра"
    s.behavior.do_not_disturb = True
    s.first_run_completed = True
    app.apply_settings(s)
    app._tamagotchi.needs.affection = 77
    app.close()

    again = make_app()
    assert again.settings().character.name == "Кіра"
    assert again.settings().behavior.do_not_disturb is True
    assert abs(again._tamagotchi.needs.affection - 77) < 1
    assert again._first_run is False
    assert again._away_seconds() is not None


def test_quiet_mode_survives_restart(make_app):
    app = make_app()
    app.start()
    app._reactions.set_quiet(30)
    pump(100)
    app.close()
    again = make_app()
    again.start()
    assert again._reactions.quiet_active


def test_wipe_all_returns_to_first_run(make_app):
    app = make_app()
    app.start()
    s = app.settings()
    s.first_run_completed = True
    app.apply_settings(s)
    app.permissions.grant(Permission.APP_METADATA)
    app._game_repo.add("cs2", "Counter-Strike 2", 0, 600)
    app.wipe_all()
    pump(100)
    assert app.settings().first_run_completed is False
    assert not app.permissions.any_observation_active()
    assert app.game_totals() == []


def test_shutdown_keeps_the_database_open_until_close(make_app):
    """aboutToQuit can fire while a nested exec() (the consent box) runs; the
    slot that opened it then continues and reads the database."""
    app = make_app()
    app.start()
    app.show_settings("privacy")
    dialog = app._settings_dialog
    app.shutdown()  # what aboutToQuit runs
    app.shutdown()
    dialog._refresh_history()  # the rest of _on_permission_clicked after box.exec() returned
    assert app.permissions.history() == []
    app.close()  # what main() runs once exec() has returned
    app.close()
    with pytest.raises(sqlite3.ProgrammingError):
        app._db.conn.execute("SELECT 1")


def test_about_to_quit_shuts_down_without_closing(make_app, qapp):
    app = make_app()
    app.start()
    qapp.aboutToQuit.emit()
    assert app._shut_down
    assert app._db.conn.execute("SELECT 1").fetchone()[0] == 1


def test_close_alone_shuts_down_first(make_app):
    app = make_app()
    app.start()
    app.close()
    assert app._shut_down and not app.controller.is_visible()
    with pytest.raises(sqlite3.ProgrammingError):
        app._db.conn.execute("SELECT 1")


def test_without_a_tray_click_through_is_never_applied(make_app):
    app = make_app()
    if app.has_tray():
        pytest.skip("this platform has a system tray")
    s = app.settings()
    s.character.click_through = True
    app.apply_settings(s)
    assert app.settings().character.click_through is False
    # persisted by a session that had a tray: turned off before she appears
    app._settings_repo.save(s)
    app.close()
    again = make_app()
    again.start()
    pump(100)
    assert again.settings().character.click_through is False
    assert not again.controller.window.click_through
    assert not again.controller.window.windowFlags() & Qt.WindowType.WindowTransparentForInput


def test_with_a_tray_click_through_is_kept(make_app, monkeypatch):
    monkeypatch.setattr(app_module.TrayIcon, "available", staticmethod(lambda: True))
    app = make_app()
    assert app.has_tray()
    s = app.settings()
    s.character.click_through = True
    app.apply_settings(s)
    assert app.settings().character.click_through is True


def test_wipe_all_discards_the_chat_and_status_windows(make_app):
    app = make_app()
    app.start()
    pump(100)
    app.show_chat()
    app._chat.input.setText("мій пароль від банку 1234")
    app._chat.send()
    app.show_status()
    old_chat, old_status = app._chat, app._status
    assert old_chat.message_count() > 0
    app.wipe_all()
    assert not old_chat.isVisible() and not old_status.isVisible()
    assert app._chat is None and app._status is None
    pump(100)
    app.show_chat()
    assert app._chat is not old_chat
    assert app._chat.message_count() == 0
    assert not any("1234" in label.text() for label in app._chat.findChildren(QLabel))


def test_wipe_all_deletes_the_log_files(make_app):
    """The log holds timestamped permission changes, like the audit table wipe_all clears."""
    log_dir = paths.log_dir()
    handler = logging.handlers.RotatingFileHandler(log_dir / "chibi.log", backupCount=3, encoding="utf-8")
    old = logging.makeLogRecord({"msg": "permission app_metadata granted", "levelno": logging.INFO})
    new = logging.makeLogRecord({"msg": "after the wipe", "levelno": logging.INFO})
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        handler.handle(old)
        for n in (1, 2):
            (log_dir / f"chibi.log.{n}").write_text("permission window_titles revoked\n", encoding="utf-8")
        (log_dir / "unrelated.txt").write_text("keep me", encoding="utf-8")
        app = make_app()
        app.start()
        app.wipe_all()
        assert not (log_dir / "chibi.log.1").exists() and not (log_dir / "chibi.log.2").exists()
        assert (log_dir / "unrelated.txt").exists()
        handler.handle(new)  # logging goes on into a fresh file
        text = (log_dir / "chibi.log").read_text(encoding="utf-8")
        assert "after the wipe" in text and "granted" not in text
    finally:
        root.removeHandler(handler)
        handler.close()


def test_reopening_restores_a_minimized_window(make_app):
    app = make_app()
    app.start()
    app.show_status()
    pump(50)
    app._status.showMinimized()
    pump(50)
    assert app._status.isMinimized()
    app.show_status()
    pump(50)
    assert not app._status.isMinimized()
    assert app._status.refresh_timer.isActive()


def test_reused_settings_dialog_shows_changes_made_from_the_menus(make_app):
    app = make_app()
    app.start()
    app.show_settings()
    dialog = app._settings_dialog
    app._toggle_dnd()  # tray / context menu while the dialog stays open
    app.show_settings()
    assert app._settings_dialog is dialog
    assert dialog.dnd_check.isChecked()
    dialog.ok_button.click()
    assert app.settings().behavior.do_not_disturb is True


def test_wipe_history_keeps_tracking_a_running_game(make_app):
    clock = FakeClock()
    app = make_app(UnsupportedBackend(), clock=clock)
    app.start()
    app.permissions.grant(Permission.APP_METADATA)
    app.bus.publish(GameStarted(game_id="cs2", title="Counter-Strike 2", process_name="cs2.exe"))
    pump(50)
    clock.advance(300)
    app.wipe_history()
    assert app.game_totals() == []
    clock.advance(7200)
    app.bus.publish(GameEnded(game_id="cs2", title="Counter-Strike 2", duration_seconds=7500.0))
    pump(50)
    totals = app.game_totals()
    assert [(t.game_id, t.sessions) for t in totals] == [("cs2", 1)]
    assert totals[0].total_seconds == pytest.approx(7200)  # only the part after the wipe


def test_away_time_counts_from_the_newest_heartbeat(make_app):
    """After a crash last_shutdown is stale; the tamagotchi's heartbeat is not."""
    clock = FakeClock()
    app = make_app(clock=clock)
    s = app.settings()
    s.first_run_completed = True
    app.apply_settings(s)
    app.start()
    app.close()  # clean exit: last_shutdown written
    clock.advance(22 * 3600 - 120)
    db = Database(paths.database_path())  # the next run's heartbeat, then a crash
    StateRepository(db, clock).set(KEY_LAST_SEEN, clock.now())
    db.close()
    clock.advance(120)

    again = make_app(clock=clock)
    started, spoken = [], []
    again.bus.subscribe(AppStarted, lambda e: started.append(e.away_seconds))
    again.bus.subscribe(SpeechRequested, lambda e: spoken.append(e.source))
    again.start()
    pump(200)
    assert started == [pytest.approx(120)]
    assert not any("return_long" in source for source in spoken)


def test_away_time_ignores_unusable_stamps(make_app):
    clock = FakeClock()
    app = make_app(clock=clock)
    assert app._away_seconds() is None
    app._state.set(LAST_SHUTDOWN_KEY, clock.now() - 3600)
    app._state.set(KEY_LAST_SEEN, True)  # corrupt
    assert app._away_seconds() == pytest.approx(3600)
    app._state.set(KEY_LAST_SEEN, clock.now() + 600)  # the wall clock went backwards
    assert app._away_seconds() == 0.0


def test_monitoring_tells_her_the_user_is_present(make_app):
    clock = FakeClock()
    backend = FakeBackend()
    backend.fg = None  # no ForegroundAppChanged either: only the app can tell her the monitor runs
    app = make_app(backend, clock=clock)
    app.start()
    clock.advance(600)
    assert app.controller.idle_seconds() >= 600  # only clicks on her count
    app.permissions.grant(Permission.APP_METADATA)
    pump(100)
    assert app._monitor.is_running()
    assert app.controller.idle_seconds() == 0.0  # no UserIdle from the monitor: the user is here
    app.stop_all_observation()
    pump(100)
    assert app.controller.idle_seconds() >= 600


def test_activity_events_queued_before_the_kill_switch_are_ignored(make_app):
    app = make_app(UnsupportedBackend())
    spoken = []
    app.bus.subscribe(SpeechRequested, lambda e: spoken.append(e.source))
    app.start()
    pump(100)
    app.permissions.grant(Permission.APP_METADATA)
    pump(50)
    spoken.clear()
    app.bus.publish(GameStarted(game_id="cs2", title="Counter-Strike 2", process_name="cs2.exe"))
    app.stop_all_observation()  # revoked in memory before the queued event is dispatched
    pump(200)
    assert not any(source.startswith("game_start.") for source in spoken)
    assert app.controller._permissions is app.permissions
    assert app._reactions._permissions is app.permissions


def test_second_instance_message_does_not_name_her(monkeypatch, qapp):
    shown = []
    monkeypatch.setattr(app_module, "setup_logging", lambda debug=False: None)
    monkeypatch.setattr(app_module, "_install_excepthook", lambda: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *args: shown.append(args[-1]))
    quit_on_close = qapp.quitOnLastWindowClosed()
    lock = QLockFile(str(paths.data_dir() / "chibi.lock"))
    assert lock.tryLock(200)
    try:
        assert app_module.main([]) == 0
    finally:
        lock.unlock()
        qapp.setQuitOnLastWindowClosed(quit_on_close)
    assert shown == [ALREADY_RUNNING_TEXT]
    assert "Мія" not in ALREADY_RUNNING_TEXT


def test_excluding_a_running_game_discards_its_session_immediately(make_app):
    # Exclude + kill switch in the same instant must not persist the game.
    backend = FakeBackend()
    app = make_app(backend)
    app.start()
    app.permissions.grant(Permission.APP_METADATA)
    pump(100)
    backend.procs.add("cs2.exe")
    backend.fg = ForegroundInfo(pid=12, process_name="cs2.exe")
    app._monitor.poll_once()
    pump(200)
    assert [s.game_id for s in app._sessions.open_sessions()] == ["cs2"]
    s = app.settings()
    s.activity.excluded_processes = ["cs2.exe"]
    app.apply_settings(s)
    app.stop_all_observation()  # flushes whatever is still open
    pump(200)
    assert app.game_totals() == []
