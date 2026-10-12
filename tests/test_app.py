"""End-to-end wiring of ChibiApp (headless): the modules are unit-tested on
their own; this checks that app.py connects them correctly."""

import time

import pytest
from PySide6.QtCore import QCoreApplication

from chibi.activity.backend import ForegroundInfo
from chibi.app import ChibiApp
from chibi.core.events import GameStarted, Permission, SpeechRequested
from chibi.settings.theme import apply_dark_theme


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


def pump(ms=200):
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.005)


@pytest.fixture
def make_app(qapp):
    apply_dark_theme(qapp)
    created = []

    def factory(backend=None):
        app = ChibiApp(qapp, backend=backend or FakeBackend())
        created.append(app)
        return app

    yield factory
    for app in created:
        app.shutdown()


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
    app.shutdown()

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
    app.shutdown()
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


def test_shutdown_is_idempotent_and_closes_db(make_app):
    app = make_app()
    app.start()
    app.shutdown()
    app.shutdown()
    import sqlite3

    with pytest.raises(sqlite3.ProgrammingError):
        app._db.conn.execute("SELECT 1")
