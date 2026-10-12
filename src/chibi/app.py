"""Composition root: builds every service, wires them through the event bus
and runs the Qt event loop.

Startup order matters:
1. QApplication + theme before any widget; single-instance lock.
2. Storage, settings, permissions, then the services that subscribe to the bus.
3. ``controller.start()`` before the event loop runs, so the greeting that
   ``AppStarted`` triggers is not missed.

Shutdown (``aboutToQuit``) stops producers first (monitor thread, screen
polling), then persists state (game sessions, needs, position) and closes
the database last.
"""

from __future__ import annotations

import argparse
import logging
import random
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QLockFile, QObject, QPoint, Qt, QTimer
from PySide6.QtGui import QAction, QGuiApplication, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox

from chibi import __version__
from chibi.activity.backend import ActivityBackend, default_backend
from chibi.activity.classifier import ActivityClassifier
from chibi.activity.monitor import ActivityMonitor
from chibi.character.animation import CharacterPack, PackError
from chibi.character.states import CharacterState, StatePriority
from chibi.core import paths
from chibi.core.clock import SystemClock
from chibi.core.config import Settings
from chibi.core.event_bus import EventBus
from chibi.core.events import (
    AppShuttingDown,
    AppStarted,
    CharacterInteraction,
    DoNotDisturbChanged,
    InteractionKind,
    ObservationStopped,
    Permission,
    PermissionChanged,
    QuietModeChanged,
    SettingsChanged,
)
from chibi.core.logging_setup import setup_logging
from chibi.desktop.controller import CharacterController
from chibi.desktop.qt_bridge import QtEventPump
from chibi.desktop.screen_service import ScreenService
from chibi.desktop.tray import TrayIcon, make_icon_from_pixmap, vector_icon
from chibi.gaming.games_db import GameDatabase
from chibi.gaming.sessions import GameSessionTracker
from chibi.memory.database import Database
from chibi.memory.repositories import (
    GameSessionRepository,
    GameTotals,
    PermissionRepository,
    PhraseHistoryRepository,
    SettingsRepository,
    StateRepository,
)
from chibi.personality.chat_offline import OfflineChatResponder
from chibi.personality.phrases import PhraseBank, PhraseSelector
from chibi.personality.reactions import ReactionEngine
from chibi.privacy.filters import PrivacyFilter
from chibi.privacy.permissions import PermissionManager
from chibi.settings.chat_window import ChatWindow
from chibi.settings.facade import ChatReply, NeedsView, StatusView
from chibi.settings.first_run import FirstRunDialog
from chibi.settings.settings_dialog import SettingsDialog
from chibi.settings.status_window import StatusWindow
from chibi.settings.theme import apply_dark_theme
from chibi.tamagotchi.service import TamagotchiService

log = logging.getLogger(__name__)

TAMAGOTCHI_TICK_MS = 30_000
REACTIONS_TICK_MS = 10_000
QUIET_STATE_KEY = "quiet_until"
LAST_SHUTDOWN_KEY = "last_shutdown"


class ChibiApp(QObject):
    """The application.  Also implements :class:`chibi.settings.facade.AppFacade`
    for the settings, status, chat and first-run windows."""

    def __init__(self, qt_app: QApplication, *, backend: ActivityBackend | None = None,
                 rng: random.Random | None = None) -> None:
        super().__init__()
        self._qt = qt_app
        self._clock = SystemClock()
        self._rng = rng or random.Random()
        self._shut_down = False

        # -- storage & settings ------------------------------------------------
        self._db = Database(paths.database_path())
        self._settings_repo = SettingsRepository(self._db, self._clock)
        self._state = StateRepository(self._db, self._clock)
        self._game_repo = GameSessionRepository(self._db)
        self._settings = self._settings_repo.load()
        self._first_run = not self._settings.first_run_completed

        self.bus = EventBus(max_queue=512)
        self._pump = QtEventPump(self.bus, self)
        self.permissions = PermissionManager(PermissionRepository(self._db), self.bus, self._clock)

        # -- character pack (vector fallback if it can't load) --------------------
        self._pack = self._load_pack(self._settings.character.pack)

        # -- tamagotchi ---------------------------------------------------------------
        self._tamagotchi = TamagotchiService(self.bus, self._state, self._clock)
        self._tamagotchi.load()

        # -- activity (thread starts only while APP_METADATA is granted) --------------
        self._backend = backend or default_backend()
        self._games = GameDatabase.load(self._settings.activity.custom_games)
        self._monitor = ActivityMonitor(
            self._backend, self.permissions, self.bus,
            ActivityClassifier(self._games), PrivacyFilter(self._settings.activity.excluded_processes),
            lambda: self._settings.activity, self._clock,
        )
        self._sessions = GameSessionTracker(self.bus, self._game_repo, self._clock, self.permissions)

        # -- personality ----------------------------------------------------------
        self._selector = PhraseSelector(
            PhraseBank.load(), PhraseHistoryRepository(self._db),
            lambda: self._settings, self._clock, self._rng,
        )
        self._reactions = ReactionEngine(
            self.bus, self._selector, lambda: self._settings, self._clock, self._rng,
            game_genre=self._game_genre, sleeping=self._is_sleeping,
        )
        self._responder = OfflineChatResponder(
            self._selector, self._reactions, lambda: self._settings, self._clock,
            perform=self._perform, needs=self._needs_mapping,
        )

        # -- desktop ------------------------------------------------------------------
        self._controller = CharacterController(
            self.bus, lambda: self._settings, self._clock, self._state, self._pack,
            self._needs_tuple, self._rng, self,
        )
        self._screen = ScreenService(self.bus, lambda: self._settings, self._controller.own_window_ids,
                                     parent=self)
        self._context_menu = self._build_context_menu()
        self._tray = self._build_tray()

        # -- windows (created lazily) ---------------------------------------------------
        self._chat: ChatWindow | None = None
        self._status: StatusWindow | None = None
        self._settings_dialog: SettingsDialog | None = None
        self._first_run_dialog: FirstRunDialog | None = None

        # -- wiring -----------------------------------------------------------------------
        self.bus.subscribe(PermissionChanged, lambda _e: self._sync_observation())
        self.bus.subscribe(ObservationStopped, lambda _e: self._sync_observation())
        self.bus.subscribe(QuietModeChanged, self._on_quiet_changed)
        self._controller.state_changed.connect(self._on_state_changed)
        self._controller.context_menu_requested.connect(self._show_context_menu)
        self._controller.window.scale_requested.connect(self._on_scale_requested)

        self._tamagotchi_timer = QTimer(self)
        self._tamagotchi_timer.setTimerType(Qt.TimerType.VeryCoarseTimer)
        self._tamagotchi_timer.timeout.connect(self._tamagotchi.tick)
        self._reactions_timer = QTimer(self)
        self._reactions_timer.setTimerType(Qt.TimerType.VeryCoarseTimer)
        self._reactions_timer.timeout.connect(self._reactions.tick)

        qt_app.aboutToQuit.connect(self.shutdown)

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        away = self._away_seconds()
        self._controller.start()  # subscribes to the bus: must precede any dispatch
        self._screen.start()
        self._tamagotchi_timer.start(TAMAGOTCHI_TICK_MS)
        self._reactions_timer.start(REACTIONS_TICK_MS)
        self._restore_quiet()
        if self._tray is not None:
            self._tray.show()
        self._sync_observation()
        self.bus.publish(AppStarted(first_run=self._first_run, away_seconds=away))
        if self._first_run:
            # Let her appear and greet first; the dialog is not modal to the desktop.
            QTimer.singleShot(1200, self.show_first_run)
        log.info("Chibi Companion %s started (pack: %s, activity backend supported: %s)",
                 __version__, self._pack.name if self._pack else "vector fallback",
                 self._backend.is_supported())

    def shutdown(self) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        log.info("shutting down")
        for step in (
            lambda: self.bus.publish(AppShuttingDown()),
            lambda: self._monitor.stop(timeout=2.0),
            self._screen.stop,
            self._tamagotchi_timer.stop,
            self._reactions_timer.stop,
            self._sessions.flush,
            self._tamagotchi.save,
            self._controller.shutdown,
            lambda: self._state.set(LAST_SHUTDOWN_KEY, self._clock.now()),
            lambda: self._tray.hide() if self._tray else None,
            self._pump.detach,
            self.bus.close,
            self._db.close,
        ):
            try:
                step()
            except Exception:
                log.exception("shutdown step failed")

    # ---------------------------------------------------------------- AppFacade

    def settings(self) -> Settings:
        return self._settings.copy()

    def apply_settings(self, settings: Settings) -> None:
        new = Settings.from_dict(settings.to_dict())  # validate + clamp
        old = self._settings
        if new.character.pack != old.character.pack:
            # Switching packs at runtime would need a new renderer; apply on restart.
            log.info("character pack change takes effect after restart")
        self._settings = new
        self._settings_repo.save(new)
        changed = [name for name in ("character", "behavior", "personality", "activity")
                   if getattr(new, name) != getattr(old, name)]
        if new.activity != old.activity:
            self._monitor.reconfigure(
                ActivityClassifier(self._reload_games(new)), PrivacyFilter(new.activity.excluded_processes)
            )
        if new.behavior.do_not_disturb != old.behavior.do_not_disturb:
            self.bus.publish(DoNotDisturbChanged(enabled=new.behavior.do_not_disturb))
        for section in changed:
            self.bus.publish(SettingsChanged(section=section))
        if new.character.name != old.character.name and self._tray is not None:
            self._tray.set_name(new.character.name)
        self._refresh_menus()

    def needs(self) -> NeedsView:
        n = self._tamagotchi.needs
        return NeedsView(
            hunger=n.hunger, energy=n.energy, happiness=n.happiness, boredom=n.boredom,
            affection=n.affection, friendship_level=n.friendship_level,
            friendship_progress=n.friendship_progress, mood=self._tamagotchi.mood,
        )

    def status(self) -> StatusView:
        quiet: float | None = self._reactions.quiet_until
        if quiet is None and self._reactions.quiet_active:
            quiet = float("inf")  # quiet until lifted
        return StatusView(
            character_name=self._settings.character.name,
            state=self._controller.state.value,
            observing=self.permissions.any_observation_active(),
            activity_supported=self._backend.is_supported(),
            ai_connected=False,
            quiet_until=quiet,
            do_not_disturb=self._settings.behavior.do_not_disturb,
        )

    def feed(self) -> None:
        self.bus.publish(CharacterInteraction(kind=InteractionKind.FEED))

    def pet(self) -> None:
        self.bus.publish(CharacterInteraction(kind=InteractionKind.PET))

    def rest(self) -> None:
        self.bus.publish(CharacterInteraction(kind=InteractionKind.REST))

    def send_chat(self, text: str) -> ChatReply:
        # Talking to her counts as attention (needs) but must not trigger a
        # second, separate "interaction" line: the reply is the line.
        self._tamagotchi.interact(InteractionKind.TALK, announce=False)
        reply = self._responder.respond(text)
        if reply.text or reply.state:
            self._controller.say(reply.text, reply.state, reply.portrait, solicited=True)
        return reply

    def game_totals(self) -> list[GameTotals]:
        return self._game_repo.totals()

    def known_games_count(self) -> int:
        return len(self._games)

    def portrait(self, name: str) -> QPixmap | None:
        return self._controller.portrait_pixmap(name)

    def data_dir(self) -> Path:
        return paths.data_dir()

    def stop_all_observation(self) -> None:
        self.permissions.stop_all_observation()

    def wipe_history(self) -> None:
        self._sessions.flush()
        self._db.wipe_history()
        self._selector.reload_history()
        log.info("history deleted by the user")

    def wipe_all(self) -> None:
        self.permissions.stop_all_observation()
        self._monitor.stop(timeout=2.0)
        self._sessions.close()
        self._db.wipe_all()
        self._selector.reload_history()
        self._reactions.clear_quiet()
        self._tamagotchi.load()  # nothing stored any more -> fresh defaults
        self._sessions = GameSessionTracker(self.bus, self._game_repo, self._clock, self.permissions)
        self._first_run = True
        self.apply_settings(Settings())
        self.bus.publish(SettingsChanged(section=""))
        log.info("all data deleted by the user")
        # The settings dialog closes itself after this returns; show onboarding again.
        QTimer.singleShot(0, self.show_first_run)

    # ------------------------------------------------------------------ windows

    def show_chat(self) -> None:
        if self._chat is None:
            self._chat = ChatWindow(self)
        self.bus.publish(CharacterInteraction(kind=InteractionKind.TALK))
        self._raise(self._chat)

    def show_status(self) -> None:
        if self._status is None:
            self._status = StatusWindow(self)
        self._raise(self._status)

    def show_settings(self, page: str | None = None) -> None:
        if self._settings_dialog is None or not self._settings_dialog.isVisible():
            if self._settings_dialog is not None:
                self._settings_dialog.deleteLater()
            self._settings_dialog = SettingsDialog(self)
        if page:
            self._settings_dialog.show_page(page)
        self._raise(self._settings_dialog)

    def show_first_run(self) -> None:
        if self._first_run_dialog is not None and self._first_run_dialog.isVisible():
            return
        self._first_run_dialog = FirstRunDialog(self)
        self._first_run_dialog.finished.connect(self._on_first_run_finished)
        self._raise(self._first_run_dialog)

    def _on_first_run_finished(self, _result: int) -> None:
        if not self._settings.first_run_completed:
            # Closed without "Почати": don't ask again on every launch.
            s = self.settings()
            s.first_run_completed = True
            self.apply_settings(s)
        self._first_run = False

    @staticmethod
    def _raise(widget) -> None:
        widget.show()
        widget.raise_()
        widget.activateWindow()

    # ------------------------------------------------------------------- menus

    def _build_context_menu(self) -> QMenu:
        menu = QMenu()
        self._act_talk = self._action(menu, "Поговорити", self.show_chat)
        self._action(menu, "Покликати", self._controller.call_over)
        self._action(menu, "Погладити", self.pet)
        self._action(menu, "Погодувати", self.feed)
        voice = self._action(menu, "Сказати голосом (з'явиться у фазі 4)", lambda: None)
        voice.setEnabled(False)
        menu.addSeparator()
        self._action(menu, "Подивитися стан", self.show_status)
        self._action(menu, "Налаштування", self.show_settings)
        menu.addSeparator()
        self._act_dnd = self._action(menu, "Не турбувати", self._toggle_dnd, checkable=True)
        self._act_stop = self._action(menu, "Зупинити спостереження", self.stop_all_observation)
        menu.addSeparator()
        self._action(menu, "Вийти", self._qt.quit)
        menu.aboutToShow.connect(self._refresh_menus)
        return menu

    def _build_tray(self) -> TrayIcon | None:
        if not TrayIcon.available():
            log.info("no system tray available")
            return None
        menu = QMenu()
        self._act_show = self._action(menu, "Сховати", self._toggle_visible)
        self._action(menu, "Поговорити", self.show_chat)
        self._action(menu, "Подивитися стан", self.show_status)
        self._action(menu, "Налаштування", self.show_settings)
        self._action(menu, "Приватність і дозволи", lambda: self.show_settings("privacy"))
        menu.addSeparator()
        self._act_tray_dnd = self._action(menu, "Не турбувати", self._toggle_dnd, checkable=True)
        self._act_click = self._action(menu, "Пропускати кліки крізь неї", self._toggle_click_through,
                                       checkable=True)
        self._act_lock = self._action(menu, "Заблокувати позицію", self._toggle_lock, checkable=True)
        self._act_tray_stop = self._action(menu, "Зупинити спостереження", self.stop_all_observation)
        menu.addSeparator()
        self._action(menu, "Вийти", self._qt.quit)
        menu.aboutToShow.connect(self._refresh_menus)
        self._tray_menu = menu
        portrait = self._controller.portrait_pixmap("calm")
        icon = make_icon_from_pixmap(portrait) if portrait is not None else vector_icon()
        tray = TrayIcon(menu, icon, self._settings.character.name, self)
        tray.activated.connect(self._on_tray_activated)
        return tray

    def _action(self, menu: QMenu, text: str, slot: Callable[[], object], *,
                checkable: bool = False) -> QAction:
        action = QAction(text, menu)
        action.setCheckable(checkable)
        action.triggered.connect(lambda _checked=False: slot())
        menu.addAction(action)
        return action

    def _refresh_menus(self) -> None:
        observing = self.permissions.any_observation_active()
        dnd = self._settings.behavior.do_not_disturb
        self._act_dnd.setChecked(dnd)
        self._act_stop.setEnabled(observing)
        if self._tray is not None:
            self._act_tray_dnd.setChecked(dnd)
            self._act_tray_stop.setEnabled(observing)
            self._act_click.setChecked(self._settings.character.click_through)
            self._act_lock.setChecked(self._settings.character.lock_position)
            self._act_show.setText("Сховати" if self._controller.is_visible() else "Показати")

    def _show_context_menu(self, pos: QPoint) -> None:
        self._context_menu.popup(pos)

    def _on_tray_activated(self) -> None:
        if not self._controller.is_visible():
            self._controller.set_hidden_by_user(False)
        else:
            self.show_status()

    def _toggle_visible(self) -> None:
        self._controller.set_hidden_by_user(self._controller.is_visible())

    def _toggle_dnd(self) -> None:
        s = self.settings()
        s.behavior.do_not_disturb = not s.behavior.do_not_disturb
        self.apply_settings(s)

    def _toggle_click_through(self) -> None:
        s = self.settings()
        s.character.click_through = not s.character.click_through
        self.apply_settings(s)

    def _toggle_lock(self) -> None:
        s = self.settings()
        s.character.lock_position = not s.character.lock_position
        self.apply_settings(s)

    def _on_scale_requested(self, factor: float) -> None:
        s = self.settings()
        s.character.scale = factor
        self.apply_settings(s)

    # ------------------------------------------------------------------ helpers

    def _sync_observation(self) -> None:
        """Run the monitor thread exactly while app metadata is permitted."""
        wanted = self.permissions.is_granted(Permission.APP_METADATA) and self._backend.is_supported()
        if wanted and not self._monitor.is_running():
            self._monitor.start()
        elif not wanted and self._monitor.is_running():
            self._monitor.stop(timeout=2.0)
        if self._tray is not None:
            self._tray.set_observing(self.permissions.any_observation_active())
        self._refresh_menus()

    def _on_state_changed(self, _state: str) -> None:
        sleeping = self._controller.posture is CharacterState.SLEEP
        if sleeping != self._tamagotchi.sleeping:
            self._tamagotchi.set_sleeping(sleeping)

    def _on_quiet_changed(self, event: QuietModeChanged) -> None:
        if not event.enabled:
            self._state.delete(QUIET_STATE_KEY)
        else:
            self._state.set(QUIET_STATE_KEY, event.until if event.until is not None else "forever")

    def _restore_quiet(self) -> None:
        stored = self._state.get(QUIET_STATE_KEY)
        if stored == "forever":
            self._reactions.set_quiet(None)
        elif isinstance(stored, (int, float)) and stored > self._clock.now():
            self._reactions.set_quiet((stored - self._clock.now()) / 60)
        elif stored is not None:
            self._state.delete(QUIET_STATE_KEY)

    def _away_seconds(self) -> float | None:
        last = self._state.get(LAST_SHUTDOWN_KEY)
        if isinstance(last, (int, float)) and last > 0:
            return max(0.0, self._clock.now() - last)
        return None

    def _perform(self, action: str) -> str:
        """Chat side effects ("поїж", "йди спати"): apply them without a second line."""
        kind = {"feed": InteractionKind.FEED, "pet": InteractionKind.PET,
                "rest": InteractionKind.REST}[action]
        outcome = self._tamagotchi.interact(kind, announce=False)
        if kind is InteractionKind.REST and outcome == "sleep":
            # A few seconds of USER priority, then her own wake rules apply again.
            self._controller.request_state(CharacterState.SLEEP, StatePriority.USER, duration=5.0)
        return outcome

    def _needs_tuple(self) -> tuple[float, float, float, float]:
        n = self._tamagotchi.needs
        return n.hunger, n.energy, n.happiness, n.boredom

    def _needs_mapping(self) -> dict[str, object]:
        return {**self._tamagotchi.needs.to_dict(), "mood": self._tamagotchi.mood}

    def _game_genre(self, game_id: str) -> str | None:
        info = self._games.get(game_id)
        return info.genre if info is not None else None

    def _is_sleeping(self) -> bool:
        controller = getattr(self, "_controller", None)
        return controller is not None and controller.posture is CharacterState.SLEEP

    def _reload_games(self, settings: Settings) -> GameDatabase:
        self._games = GameDatabase.load(settings.activity.custom_games)
        return self._games

    @staticmethod
    def _load_pack(name: str) -> CharacterPack | None:
        try:
            return CharacterPack.load(paths.character_pack_dir(name))
        except PackError as exc:
            log.error("character pack %r could not be loaded (%s); using the vector character", name, exc)
            return None

    # Exposed for tests / smoke runs.
    @property
    def controller(self) -> CharacterController:
        return self._controller


# ---------------------------------------------------------------------- entry point


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="chibi", description="Chibi AI Companion")
    parser.add_argument("--debug", action="store_true", help="verbose logging to the console")
    parser.add_argument("--quit-after", type=float, metavar="SECONDS",
                        help="quit automatically (smoke tests)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def _set_windows_app_id() -> None:
    """Own taskbar/tray identity instead of being grouped under python.exe."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ChibiCompanion.Desktop")
    except Exception:  # pragma: no cover - cosmetic only
        log.debug("could not set AppUserModelID", exc_info=True)


def _install_excepthook() -> None:
    def hook(exc_type, exc, tb):
        log.critical("unhandled exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = hook


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    setup_logging(debug=args.debug)
    _install_excepthook()
    _set_windows_app_id()

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    qt_app = QApplication.instance() or QApplication(sys.argv[:1])
    qt_app.setApplicationName("Chibi Companion")
    qt_app.setOrganizationName("ChibiCompanion")
    qt_app.setApplicationVersion(__version__)
    qt_app.setQuitOnLastWindowClosed(False)  # lives in the tray

    lock = QLockFile(str(paths.data_dir() / "chibi.lock"))
    lock.setStaleLockTime(0)
    if not lock.tryLock(200):
        log.info("another instance is already running")
        QMessageBox.information(None, "Chibi Companion", "Мія вже живе на твоєму робочому столі 🙂")
        return 0

    apply_dark_theme(qt_app)
    try:
        chibi = ChibiApp(qt_app)
    except Exception:
        log.exception("startup failed")
        QMessageBox.critical(None, "Chibi Companion",
                             "Не вдалося запуститися. Подробиці в журналі:\n" + str(paths.log_dir()))
        lock.unlock()
        return 1
    chibi.start()

    # Let Ctrl+C in a debug console quit cleanly (Qt otherwise blocks Python signals).
    signal.signal(signal.SIGINT, lambda *_: qt_app.quit())
    pulse = QTimer()
    pulse.timeout.connect(lambda: None)
    pulse.start(500)

    if args.quit_after:
        QTimer.singleShot(int(args.quit_after * 1000), qt_app.quit)

    started = time.monotonic()
    code = qt_app.exec()
    chibi.shutdown()
    lock.unlock()
    log.info("exited after %.0f s", time.monotonic() - started)
    return code
