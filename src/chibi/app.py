"""Composition root: builds every service, wires them through the event bus
and runs the Qt event loop.

Startup order matters:
1. QApplication + theme before any widget; single-instance lock.
2. Storage, settings, permissions, then the services that subscribe to the bus.
3. ``controller.start()`` before the event loop runs, so the greeting that
   ``AppStarted`` triggers is not missed.

Shutdown has two steps.  ``shutdown()`` runs on ``aboutToQuit``: it stops
producers first (monitor thread, screen polling), then persists state (game
sessions, needs, position).  ``close()`` runs after ``exec()`` has returned and
closes the bus and the database: aboutToQuit can fire while a nested event loop
(a message box) is still on the stack, and the slot that opened it continues
afterwards, so the database must outlive aboutToQuit.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import logging.handlers
import math
import random
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QLockFile, QObject, QPoint, Qt, QTimer
from PySide6.QtGui import QAction, QGuiApplication, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QWidget

from chibi import __version__
from chibi.activity.backend import ActivityBackend, default_backend
from chibi.activity.classifier import ActivityClassifier
from chibi.activity.monitor import ActivityMonitor
from chibi.ai.brain import AIBrain, BrainContext
from chibi.ai.budget import UsageLedger
from chibi.ai.providers.factory import DEFAULT_MODELS, is_cloud, stored_key
from chibi.ai.secrets import KEY_NAMES, default_secret_store
from chibi.ai.service import AIService, RequestTiming
from chibi.ai.types import Purpose
from chibi.character.animation import CharacterPack, PackError
from chibi.character.states import CharacterState, StatePriority
from chibi.core import paths
from chibi.core.clock import Clock, SystemClock
from chibi.core.config import Settings
from chibi.core.event_bus import EventBus
from chibi.core.events import (
    ActivityCategory,
    AIStatusChanged,
    AppShuttingDown,
    AppStarted,
    CharacterInteraction,
    ConversationActive,
    DoNotDisturbChanged,
    ForegroundAppChanged,
    FullscreenChanged,
    InteractionKind,
    ObservationStopped,
    Permission,
    PermissionChanged,
    QuietModeChanged,
    SettingsChanged,
    SpeechRequested,
    UserIdle,
    UserReturned,
    VisionStateChanged,
    VoiceStateChanged,
    VoiceTranscript,
)
from chibi.core.logging_setup import setup_logging
from chibi.desktop.controller import CharacterController
from chibi.desktop.qt_bridge import QtEventPump
from chibi.desktop.screen_service import ScreenService
from chibi.desktop.tray import TrayIcon, make_icon_from_pixmap, vector_icon
from chibi.emotion.engine import EmotionEngine
from chibi.games import GAME_IDS, GAME_TITLES
from chibi.games.hub import GamesHub
from chibi.games.rewards import GameRewards
from chibi.gaming.games_db import GameDatabase
from chibi.gaming.sessions import GameSessionTracker
from chibi.memory.database import Database
from chibi.memory.longterm import MemoryStore, ShortTermMemory
from chibi.memory.repositories import (
    GameSessionRepository,
    GameTotals,
    PermissionRepository,
    PhraseHistoryRepository,
    SettingsRepository,
    StateRepository,
)
from chibi.personality.chat_offline import SUCCESS_OUTCOMES, OfflineChatResponder
from chibi.personality.composer import SpeechComposer
from chibi.personality.initiative import DecisionEngine, InitiativeContext
from chibi.personality.observer import VOICE_PRESENCE_SECONDS, Observer
from chibi.personality.phrases import PhraseBank, PhraseSelector
from chibi.personality.reactions import ReactionEngine
from chibi.personality.text_uk import time_of_day
from chibi.privacy.filters import PrivacyFilter
from chibi.privacy.permissions import PermissionManager
from chibi.settings.chat_window import ChatWindow
from chibi.settings.facade import AIStatusView, ChatReply, MemoryView, NeedsView, StatusView
from chibi.settings.first_run import FirstRunDialog
from chibi.settings.settings_dialog import SettingsDialog
from chibi.settings.status_window import StatusWindow
from chibi.settings.theme import apply_dark_theme
from chibi.tamagotchi.gifts import GiftService
from chibi.tamagotchi.needs import finite_number
from chibi.tamagotchi.service import KEY_LAST_SEEN, TamagotchiService
from chibi.vision.integration import (
    SeenObservation,
    VisionIntegration,
    VisionStatusView,
    is_look_request,
    observer_ai_reason,
)
from chibi.voice.integration import (
    AZURE_KEY_NAME,
    DownloadProgress,
    SpeechStart,
    VoiceIntegration,
    VoiceStatus,
    invites_reply,
)
from chibi.voice.models import ModelSpec

log = logging.getLogger(__name__)

TAMAGOTCHI_TICK_MS = 30_000
REACTIONS_TICK_MS = 10_000
QUIET_STATE_KEY = "quiet_until"
#: The user's "Пауза мікрофона" survives a restart: the open microphone must not reopen by itself.
MIC_PAUSE_STATE_KEY = "open_mic_paused"
LAST_SHUTDOWN_KEY = "last_shutdown"
#: Shown by a second launch.  No name: hers is in the database the running instance owns.
ALREADY_RUNNING_TEXT = "Chibi Companion уже запущено — вона чекає на тебе на робочому столі 🙂"
#: Her chat answer to "подивись на екран" while the frame is being analysed.
LOOKING_REPLY = "Зараз гляну!"
#: A voice command heard while the chat is still answering the previous message.
BUSY_REPLY = "Почула, але ще відповідаю на попереднє — повтори за мить, добре?"
#: A voice request heard (chat window hidden) while she is still answering waits for
#: that answer - the newest one only - and is dropped when it has waited this long.
PENDING_TRANSCRIPT_SECONDS = 60.0
#: A chat answer that never arrived stops holding back voice requests after this long.
CHAT_IN_FLIGHT_SECONDS = 90.0
#: ``ConversationActive(False)`` follows this long after the conversation went quiet, so
#: the next turn of a back-and-forth does not find a screen analysis in its way.
CONVERSATION_HOLD_SECONDS = 2.0
#: Menu item: the user's switch for the always-open microphone (push-to-talk keeps working).
MIC_PAUSE_TEXT = "Пауза мікрофона"
#: Its tooltip while she listens only on push-to-talk (there is nothing to pause).
MIC_PAUSE_PUSH_HINT = "Діє, коли вона слухає постійно: «Налаштування → Голос → Як слухати»."


class ChibiApp(QObject):
    """The application.  Also implements :class:`chibi.settings.facade.AppFacade`
    for the settings, status, chat and first-run windows."""

    def __init__(self, qt_app: QApplication, *, backend: ActivityBackend | None = None,
                 rng: random.Random | None = None, clock: Clock | None = None) -> None:
        super().__init__()
        self._qt = qt_app
        self._clock: Clock = clock or SystemClock()
        self._rng = rng or random.Random()
        self._shut_down = False
        self._closed = False

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
        # -- AI brain (Phase 2): rules decide *when* she speaks, the model only *how* ------
        self._activity: tuple[str, str | None, float] | None = None  # category, app, since
        self._user_away = False
        self._fullscreen = False
        self._last_chat: float | None = None
        #: Chat answers being written (id -> monotonic start): one at a time for the voice.
        self._chats_in_flight: dict[int, float] = {}
        self._chat_ids = 0
        #: The newest voice request waiting for the answer in flight: (text, heard at,
        #: VoiceTranscript.origin).  Memory only, never logged.
        self._pending_transcript: tuple[str, float, str] | None = None
        #: The next chat message may be mined for memories (False while a request the open
        #: microphone overheard goes through the chat window).
        self._chat_remember = True
        #: The message the chat window sends now is a voice request (see send_chat_stream).
        self._chat_spoken = False
        #: When the user last talked to her by voice (monotonic): hands-free is presence too.
        self._last_voice_at: float | None = None
        #: ConversationActive as last published, and since when (monotonic) nothing has kept
        #: it up: False follows CONVERSATION_HOLD_SECONDS later (see _sync_conversation).
        self._conversation = False
        self._conversation_quiet_since: float | None = None
        self._conversation_timer = QTimer(self)
        self._conversation_timer.setSingleShot(True)
        self._conversation_timer.timeout.connect(self._sync_conversation)
        self._ai_model = ""
        self.bus.subscribe(AIStatusChanged, self._on_ai_status)
        self._secrets = default_secret_store(paths.data_dir())
        self._ledger = UsageLedger(self._db, self._clock)
        self._ai = AIService(self.bus, lambda: self._settings, self.permissions, self._secrets,
                             self._ledger, self._clock, parent=self)
        self._memory = MemoryStore(self._db, self.bus, self._clock)
        self._short_term = ShortTermMemory(max_turns=60)
        self._emotion = EmotionEngine(self.bus, self._state, self._clock,
                                      needs=lambda: self._tamagotchi.needs, settings=lambda: self._settings)
        self._emotion.load()
        self._brain = AIBrain(self._ai, self._memory, self._short_term, self._emotion,
                              lambda: self._settings, self._clock, self._brain_context,
                              self._needs_mapping, rng=self._rng)
        self._composer = SpeechComposer(self.bus, self._selector, self._brain, lambda: self._settings,
                                        self._clock)
        self._reactions = ReactionEngine(
            self.bus, self._selector, lambda: self._settings, self._clock, self._rng,
            game_genre=self._game_genre, sleeping=self._is_sleeping, permissions=self.permissions,
            composer=self._composer, idle_chatter=False,
        )
        self._decisions = DecisionEngine(
            self.bus, self._composer, lambda: self._settings, self._clock, self._rng, self._emotion,
            self._initiative_context, self._reactions.can_speak_unsolicited,
            on_spoken=self._reactions.note_unsolicited,
        )
        self._responder = OfflineChatResponder(
            self._selector, self._reactions, lambda: self._settings, self._clock,
            perform=self._perform, needs=self._needs_mapping,
        )

        # -- desktop ------------------------------------------------------------------
        self._controller = CharacterController(
            self.bus, lambda: self._settings, self._clock, self._state, self._pack,
            self._needs_tuple, self._rng, self, permissions=self.permissions,
        )
        self._screen = ScreenService(self.bus, lambda: self._settings, self._controller.own_window_ids,
                                     parent=self)

        # -- windows (created lazily) ---------------------------------------------------
        self._chat: ChatWindow | None = None
        self._status: StatusWindow | None = None
        self._settings_dialog: SettingsDialog | None = None
        self._first_run_dialog: FirstRunDialog | None = None
        self._tray: TrayIcon | None = None
        #: Collects vision notices while the chat asks her to look (shown as her reply).
        self._look_notices: list[str] | None = None

        # -- voice (Phase 4): every line she says can be voiced; the mic feeds the chat --------
        self._voice = VoiceIntegration(
            self.bus, lambda: self._settings, self.permissions, self._secrets, self._composer,
            notify=self._notify_voice, on_change=self._sync_voice_ui, is_visible=self._controller.is_visible,
            clock=self._clock, parent=self,
        )
        self._voice.add_speech_start_listener(self._on_speech_start)
        self._controller.set_speech_router(self._route_speech)

        # -- screen vision (Phase 3): only allowed game windows, only with permission ----------
        self._vision = VisionIntegration(
            self.bus, lambda: self._settings, self.permissions, self._ai, self._composer, self._clock,
            games=lambda: self._games, memory=self._memory, backend=self._backend,
            # Every unsolicited line passes the reaction engine's gate: DND, quiet mode and
            # the shared hourly budget (vision has its own cooldowns, and fullscreen games
            # are what it is for).
            quiet=lambda: not self._reactions.can_speak_unsolicited(ignore_gap=True, ignore_fullscreen=True),
            on_spoken=self._reactions.note_unsolicited, notify=self._vision_notice,
            own_windows=self._controller.own_window_ids,
            # As an observer she takes no automatic frames she could not comment on.
            observer_ai=lambda: observer_ai_reason(self._settings, self._ai.status(),
                                                   self._brain.available(Purpose.INITIATIVE)),
            parent=self,
        )

        # -- observer: with commentary "observer" she decides herself when what she sees is
        #    worth a word (one INITIATIVE request now and then, mostly answered by silence).
        #    Same shared gate and budget as every other unsolicited line. -------------------------
        self._observer = Observer(
            self.bus, self._brain, lambda: self._settings, self._clock, self._rng,
            self._reactions.can_speak_unsolicited, self._reactions.note_unsolicited,
            context=self._brain_context, last_chat=lambda: self._last_chat,
            details=self._vision.details_for, may_share=self._vision.screen_data_may_leave,
            permissions=self.permissions, sleeping=self._is_sleeping,
        )

        # -- gifts and mini-games (Phase 5); before AppStarted, which grants the daily stars ---
        self._gifts = GiftService(self.bus, self._state, self._clock, self._tamagotchi, self._composer,
                                  self._memory, lambda: self._settings)
        self._game_rewards = GameRewards(self._gifts.wallet, self._tamagotchi, self._composer)
        self._games_hub = GamesHub(self._game_rewards, self._gifts, self._composer, rng=self._rng,
                                   name=lambda: self._settings.character.name,
                                   portrait=self._controller.portrait_pixmap, parent=self)

        self._context_menu = self._build_context_menu()
        self._tray = self._build_tray()
        if self._tray is None and self._settings.character.click_through:
            # Persisted from a session that had a tray: applied now, it would lock
            # the user out (see apply_settings).  The controller applies it in start().
            log.info("no tray: click-through turned off")
            self._settings.character.click_through = False
            try:
                self._settings_repo.save(self._settings)
            except Exception:  # memory is what counts; any later apply saves it again
                log.exception("could not save the click-through change")

        # -- wiring -----------------------------------------------------------------------
        self.bus.subscribe(PermissionChanged, lambda _e: self._sync_observation())
        self.bus.subscribe(ObservationStopped, lambda _e: self._sync_observation())
        self.bus.subscribe(QuietModeChanged, self._on_quiet_changed)
        self.bus.subscribe(ForegroundAppChanged, self._on_foreground)
        self.bus.subscribe(UserIdle, lambda _e: self._set_away(True))
        self.bus.subscribe(UserReturned, lambda _e: self._set_away(False))
        self.bus.subscribe(FullscreenChanged, self._on_fullscreen)
        self.bus.subscribe(ObservationStopped, lambda _e: self._forget_activity())
        self.bus.subscribe(VoiceTranscript, self._on_transcript)
        self.bus.subscribe(VoiceStateChanged, lambda _e: self._sync_voice_ui())
        self.bus.subscribe(VisionStateChanged, self._on_vision_state)
        self._controller.state_changed.connect(self._on_state_changed)
        self._controller.context_menu_requested.connect(self._show_context_menu)
        self._controller.window.scale_requested.connect(self._on_scale_requested)

        self._tamagotchi_timer = QTimer(self)
        self._tamagotchi_timer.setTimerType(Qt.TimerType.VeryCoarseTimer)
        self._tamagotchi_timer.timeout.connect(self._tamagotchi.tick)
        self._tamagotchi_timer.timeout.connect(self._emotion.tick)
        self._reactions_timer = QTimer(self)
        self._reactions_timer.setTimerType(Qt.TimerType.VeryCoarseTimer)
        self._reactions_timer.timeout.connect(self._reactions.tick)
        self._reactions_timer.timeout.connect(self._decisions.tick)
        #: Sends the voice request that waited for the previous answer (see _on_transcript).
        self._transcript_timer = QTimer(self)
        self._transcript_timer.setSingleShot(True)
        self._transcript_timer.timeout.connect(self._flush_transcript)

        qt_app.aboutToQuit.connect(self.shutdown)

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        away = self._away_seconds()
        self._controller.start()  # subscribes to the bus: must precede any dispatch
        if self._state.get(MIC_PAUSE_STATE_KEY):
            # Before the voice starts: a microphone the user paused never reopens by itself.
            self._voice.set_open_mic_paused(True, announce=False)
        self._voice.start()
        self._screen.start()
        self._tamagotchi_timer.start(TAMAGOTCHI_TICK_MS)
        self._reactions_timer.start(REACTIONS_TICK_MS)
        self._restore_quiet()
        if self._tray is not None:
            self._tray.show()
        self._sync_observation()
        # No model warm-up here: a local model is loaded when the user starts talking to
        # her or opens the chat - one loaded now would sit in video memory (keep-alive)
        # even if they only start a game.
        self.bus.publish(AppStarted(first_run=self._first_run, away_seconds=away))
        if self._first_run:
            # Let her appear and greet first; the dialog is not modal to the desktop.
            QTimer.singleShot(1200, self.show_first_run)
        log.info("Chibi Companion %s started (pack: %s, activity backend supported: %s)",
                 __version__, self._pack.name if self._pack else "vector fallback",
                 self._backend.is_supported())

    def shutdown(self) -> None:
        """Stop producers and save state (``aboutToQuit``).  Idempotent.

        The bus and the database stay open: Qt emits aboutToQuit from inside
        ``quit()``, possibly while a nested ``exec()`` (a consent box) is still
        running, and the slot that opened it goes on to read the database once
        it returns.  :meth:`close` releases them after the event loop is gone.
        """
        if self._shut_down:
            return
        self._shut_down = True
        log.info("shutting down")
        self._run_steps("shutdown", (
            lambda: self.bus.publish(AppShuttingDown()),
            lambda: self._monitor.stop(timeout=2.0),
            # Before the AI service: an analysis or a voiced line in flight would
            # otherwise wait for a callback that never comes.  (AppShuttingDown is
            # only queued and may never be dispatched during aboutToQuit.)
            self._vision.shutdown,
            self._voice.shutdown,
            self._screen.stop,
            self._tamagotchi_timer.stop,
            self._reactions_timer.stop,
            self._transcript_timer.stop,
            self._conversation_timer.stop,
            self._games_hub.close_all,
            self._gifts.close,
            self._sessions.flush,
            self._decisions.close,
            self._observer.close,  # drops its decision in flight: before the AI service
            self._composer.close,
            self._ai.shutdown,  # joins the worker (<= 2 s) before the database closes
            self._tamagotchi.save,
            self._emotion.save,
            self._controller.shutdown,
            lambda: self._state.set(LAST_SHUTDOWN_KEY, self._clock.now()),
            lambda: self._tray.hide() if self._tray else None,
            self._pump.detach,
        ))

    def close(self) -> None:
        """Final teardown once ``exec()`` has returned: shut down (if aboutToQuit
        did not), then close the bus and the database.  Idempotent."""
        self.shutdown()
        if self._closed:
            return
        self._closed = True
        self._run_steps("close", (self.bus.close, self._db.close))

    @staticmethod
    def _run_steps(what: str, steps: tuple[Callable[[], object], ...]) -> None:
        # One failing step (a full disk, a dead thread) must not skip the rest.
        for step in steps:
            try:
                step()
            except Exception:
                log.exception("%s step failed", what)

    # ---------------------------------------------------------------- AppFacade

    def settings(self) -> Settings:
        return self._settings.copy()

    def apply_settings(self, settings: Settings) -> None:
        new = Settings.from_dict(settings.to_dict())  # validate + clamp
        if self._tray is None and new.character.click_through:
            # Her own menu would be the only way back, and click-through blocks it.
            log.info("no tray: click-through refused")
            new.character.click_through = False
        old = self._settings
        if new.character.pack != old.character.pack:
            # Switching packs at runtime would need a new renderer; apply on restart.
            log.info("character pack change takes effect after restart")
        self._settings = new
        self._settings_repo.save(new)
        changed = [name for name in ("character", "behavior", "personality", "activity", "ai",
                                     "voice", "vision", "gaming")
                   if getattr(new, name) != getattr(old, name)]
        if new.ai != old.ai:
            # Synchronously: the AI page tests the connection right after applying.
            self._ai.reconfigure()
        if new.voice != old.voice or new.character.name != old.character.name:
            # Synchronously too: "Прослухати" applies the edits and speaks at once.
            self._voice.reconfigure()
        if new.activity != old.activity:
            privacy = PrivacyFilter(new.activity.excluded_processes)
            if new.activity.excluded_processes != old.activity.excluded_processes:
                self._discard_private_sessions(privacy)
            self._monitor.reconfigure(ActivityClassifier(self._reload_games(new)), privacy)
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
            ai_connected=self._ai.status()[0],
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
        reply = self._look_reply(text) or self._responder.respond(text)
        self._say_reply(reply)
        return reply

    def send_chat_stream(self, text: str, on_delta: Callable[[str], None],
                         on_done: Callable[[ChatReply], None], *, remember: bool | None = None,
                         spoken: bool | None = None) -> None:
        """Chat with the AI brain when it is available, the offline responder otherwise
        (also when the AI fails mid-reply).  *on_done* runs exactly once.  While the
        answer is in flight, voice requests wait for it (:meth:`_on_transcript`).
        *remember* False: the message is not mined for long-term memories (what the open
        microphone overheard); None: as set for this message (typed text: True).
        *spoken*: the message is a voice request (its answer is timed to her voice);
        None: as set for this message (typed text: False)."""
        if remember is None:
            remember = self._chat_remember
        if spoken is None:
            spoken = self._chat_spoken
        self._tamagotchi.interact(InteractionKind.TALK, announce=False)
        self._last_chat = self._clock.monotonic()
        self._chat_ids += 1
        chat_id = self._chat_ids
        self._chats_in_flight[chat_id] = self._last_chat

        def done(reply: ChatReply) -> None:
            self._chats_in_flight.pop(chat_id, None)
            try:
                on_done(reply)
            finally:
                self._chat_finished()

        try:
            self._answer_chat(text, on_delta, done, remember=remember, spoken=spoken)
        except BaseException:
            if self._chats_in_flight.pop(chat_id, None) is not None:
                self._chat_finished()
            raise
        # After the request is queued: a warm-up for this conversation would only delay it.
        self._sync_conversation()

    def _answer_chat(self, text: str, on_delta: Callable[[str], None],
                     on_done: Callable[[ChatReply], None], *, remember: bool = True,
                     spoken: bool = False) -> None:
        look = self._look_reply(text)
        if look is not None:
            # The answer about the screen comes from vision once the frame is analysed.
            self._say_reply(look)
            on_done(look)
            return
        if not self._brain.available(Purpose.CHAT):
            reply = self._responder.respond(text)
            self._say_reply(reply)
            on_done(reply)
            return
        # She starts talking at the first complete sentence while the model writes on;
        # the stream hands the answer to her voice / bubble exactly once in total.
        stream = self._voice.begin_reply("chat", show=self._show_routed, voice_request=spoken)

        def delta(piece: str) -> None:
            try:
                on_delta(piece)
            finally:
                stream.feed(piece)

        def finish(reply: ChatReply | None) -> None:
            if reply is None:
                # The AI failed (perhaps mid-answer): nothing more of it is said, and
                # the offline answer replaces it.
                stream.abort()
                reply = self._responder.respond(text)
                self._say_reply(reply)
            else:
                reply = self._apply_chat_action(reply)
                stream.finish(reply.text, reply.state, reply.portrait)  # never also _say_reply
            on_done(reply)

        self._brain.chat(text, delta, finish, remember=remember)

    def _chat_busy(self) -> bool:
        """A chat answer is still being written (one lost for too long does not count)."""
        now = self._clock.monotonic()
        for chat_id, started in list(self._chats_in_flight.items()):
            if now - started > CHAT_IN_FLIGHT_SECONDS:
                del self._chats_in_flight[chat_id]
                log.warning("a chat answer never arrived; voice requests are no longer held back")
        return bool(self._chats_in_flight)

    def _chat_finished(self) -> None:
        if self._pending_transcript is not None and not self._chat_busy() and not self._shut_down:
            # Later, not from inside the finished answer's callback.
            self._transcript_timer.start(0)
        self._sync_conversation()

    def _sync_conversation(self) -> None:
        """Publish ``ConversationActive`` whenever it changes, so background AI work
        (screen analysis, observer decisions, memory extraction, AI-phrased lines) never
        delays her answer.  The user is talking with her while the voice says so
        (``VoiceIntegration.conversation_active``: speaking, being recognised, waiting for
        or hearing her answer, the follow-up window) or a chat answer is in flight (typed
        or spoken, from :meth:`send_chat_stream` until it is finished - one lost for
        :data:`CHAT_IN_FLIGHT_SECONDS` stops counting).  ``False`` follows only after
        :data:`CONVERSATION_HOLD_SECONDS` of quiet, so back-to-back turns do not flap.
        Starting a conversation also warms the chat model up (a no-op unless it is
        likely unloaded)."""
        if self._shut_down:
            return
        now = self._clock.monotonic()
        active = self._voice.conversation_active or self._chat_busy()
        wake: float | None = None
        if active:
            self._conversation_quiet_since = None
            if not self._conversation:
                self._conversation = True
                self.bus.publish(ConversationActive(active=True))
                self._ai.warm_up()
            if self._chats_in_flight:  # look again when a lost answer stops counting
                wake = min(self._chats_in_flight.values()) + CHAT_IN_FLIGHT_SECONDS
        elif self._conversation:
            if self._conversation_quiet_since is None:
                self._conversation_quiet_since = now
            wake = self._conversation_quiet_since + CONVERSATION_HOLD_SECONDS
            if now >= wake:
                self._conversation = False
                self._conversation_quiet_since = None
                self.bus.publish(ConversationActive(active=False))
                wake = None
        if wake is None:
            self._conversation_timer.stop()
        else:
            # A little late rather than early: the clock must have passed the moment.
            self._conversation_timer.start(min(2**31 - 1, max(0, math.ceil((wake - now) * 1000)) + 20))

    def _on_speech_start(self, event: SpeechStart) -> None:
        """The user started talking: get the chat model ready while they speak (loading
        takes seconds, about as long as recognising the phrase).  Speech the open
        microphone cannot yet tell is for her ("name" mode: maybe the TV) does not count -
        it would keep the model in video memory all evening; her name, once recognised,
        starts the conversation and warms it up then (:meth:`_sync_conversation`)."""
        if event.for_her:
            self._ai.warm_up()

    def _look_reply(self, text: str) -> ChatReply | None:
        """"Подивись на екран" in the chat: ask vision to look.  Her reply says she is
        looking, or (when she cannot) why not - never a made-up description."""
        if not is_look_request(text):
            return None
        self._look_notices = []
        try:
            looking = self._vision.request()
            notices = self._look_notices
        finally:
            self._look_notices = None
        if looking:
            return ChatReply(text=LOOKING_REPLY, state="curious", portrait="question")
        if notices:
            return ChatReply(text=notices[-1], state="thinking", portrait="thinking")
        return None

    def _say_reply(self, reply: ChatReply) -> None:
        if reply.text or reply.state:
            self._voice.route(SpeechRequested(text=reply.text, state=reply.state, portrait=reply.portrait,
                                              solicited=True, source="chat"), self._show_routed)

    def _show_routed(self, event: SpeechRequested, seconds: float | None) -> None:
        self._controller.say(event.text, event.state, event.portrait, seconds, solicited=event.solicited)

    def _route_speech(self, event: SpeechRequested,
                      show: Callable[[SpeechRequested, float | None], None]) -> None:
        """Every ``SpeechRequested`` passes here (the controller's speech router).  A line
        of her own the user may answer (a remark about the screen, a question) goes with
        the next chat message as context."""
        if event.text.strip() and invites_reply(event):
            self._brain.note_unprompted(event.text)

        def companion_show(routed: SpeechRequested, seconds: float | None) -> None:
            # Gaming companion mode: over a running game a voiced comment is heard,
            # not read - no bubble on top of the game, only her talking.
            if (seconds is not None and not routed.solicited and self._settings.gaming.companion_mode
                    and self._current_game() is not None):
                routed = replace(routed, text="")
            show(routed, seconds)

        self._voice.route(event, companion_show)

    def _apply_chat_action(self, reply: ChatReply) -> ChatReply:
        """Carry out what the model asked for ("поїж", "помовч"); the reply's action
        becomes what really happened, so the chat never claims a refused action."""
        action = reply.action
        done = ""
        if action in ("feed", "pet", "rest"):
            outcome = self._perform(action)
            if outcome in SUCCESS_OUTCOMES[action]:
                done = {"feed": "fed", "pet": "petted", "rest": "rest"}[action]
        elif action == "quiet_on":
            self._reactions.set_quiet(self._settings.behavior.quiet_minutes)
            done = action
        elif action == "quiet_off":
            self._reactions.clear_quiet()
            done = action
        return replace(reply, action=done)

    def ai_status(self) -> AIStatusView:
        online, reason = self._ai.status()
        ai = self._settings.ai
        usage = self._ai.usage_today()
        return AIStatusView(
            provider=ai.provider,
            model=self._ai_model or ai.model or DEFAULT_MODELS.get(ai.provider, ""),
            online=online, reason=reason, cloud=is_cloud(ai),
            requests_today=usage.requests, requests_limit_day=ai.max_requests_per_day,
            cost_today_usd=usage.cost_usd, cost_limit_usd=ai.max_cost_per_day_usd,
            cost_known=usage.cost_known,
        )

    def ai_last_timing(self) -> RequestTiming | None:
        return self._ai.last_chat_timing

    def ai_has_key(self, provider: str) -> bool:
        return provider in KEY_NAMES and stored_key(self._secrets, provider) is not None

    def ai_set_key(self, provider: str, key: str) -> None:
        """Store (or with "" delete) an API key in the OS credential store.  Errors
        propagate to the AI page, which reports them without the key."""
        if provider not in KEY_NAMES:
            raise ValueError(f"no key for provider {provider!r}")
        self._secrets.set(KEY_NAMES[provider], key.strip())
        self._ai.reconfigure()

    def ai_test(self, on_done: Callable[[bool, str], None]) -> None:
        self._ai.test(on_done)

    def ai_list_models(self, on_done: Callable[[list[str], str], None]) -> None:
        self._ai.list_models(on_done)

    def list_memories(self, kind: str | None = None, query: str = "") -> list[MemoryView]:
        return [MemoryView(id=m.id, kind=m.kind, text=m.text, source=m.source, pinned=m.pinned,
                           created_at=m.created_at, updated_at=m.updated_at)
                for m in self._memory.list(kind or None, query)]

    def add_memory(self, kind: str, text: str) -> None:
        self._memory.add(kind, text, source="user")

    def update_memory(self, memory_id: int, text: str) -> None:
        self._memory.update(memory_id, text)  # MemoryRejectedError -> shown by the viewer

    def set_memory_pinned(self, memory_id: int, pinned: bool) -> None:
        self._memory.set_pinned(memory_id, pinned)

    def delete_memory(self, memory_id: int) -> None:
        self._memory.delete(memory_id)

    def clear_memories(self, kind: str | None = None) -> None:
        if kind in (None, "", "fact", "preference"):  # what memory extraction would store
            # Extraction waits while the user talks with her: what was said before this
            # must not come back as a memory afterwards.
            self._brain.discard_pending_memories()
        self._memory.clear(kind)

    def emotion(self) -> dict[str, float]:
        return self._emotion.values()

    # -- voice (settings page "Голос") --------------------------------------------------------

    def voice_status(self) -> VoiceStatus:
        return self._voice.status()

    def voice_models(self) -> list[tuple[ModelSpec, bool]]:
        return self._voice.models()

    def voice_download(self, model_id: str, on_progress: Callable[[int, int], None],
                       on_done: Callable[[bool, str], None]) -> None:
        self._voice.download(model_id, on_progress, on_done)

    def voice_cancel_download(self) -> None:
        self._voice.cancel_download()

    def voice_download_state(self) -> DownloadProgress | None:
        return self._voice.download_state()

    def voice_remove_model(self, model_id: str) -> None:
        self._voice.remove(model_id)

    def voice_has_azure_key(self) -> bool:
        return self._voice.has_azure_key()

    def voice_set_azure_key(self, key: str) -> None:
        self._voice.set_azure_key(key)

    def voice_test(self, on_done: Callable[[bool, str], None]) -> None:
        self._voice.test_voice(on_done=on_done)

    def voice_devices(self) -> tuple[list[str], list[str]]:
        return self._voice.devices()

    # -- screen vision (settings page "Бачення екрана") ------------------------------------------

    def vision_status(self) -> VisionStatusView:
        return self._vision.status_view()

    def vision_recent(self) -> list[SeenObservation]:
        return self._vision.recent_observations()

    def vision_clear_history(self) -> None:
        self._vision.clear_history()
        self._observer.clear()

    def vision_games(self) -> list[tuple[str, str]]:
        return self._vision.game_choices()

    def look_at_screen(self) -> None:
        self._vision.request()

    # -- gifts and mini-games ------------------------------------------------------------------------

    def open_game(self, game_id: str) -> None:
        self._games_hub.open_game(game_id)

    def open_shop(self) -> None:
        self._games_hub.open_shop()

    def game_totals(self) -> list[GameTotals]:
        return self._game_repo.totals()

    def known_games_count(self) -> int:
        return len(self._games)

    def portrait(self, name: str) -> QPixmap | None:
        return self._controller.portrait_pixmap(name)

    def data_dir(self) -> Path:
        return paths.data_dir()

    def has_tray(self) -> bool:
        return self._tray is not None

    def stop_all_observation(self) -> None:
        self.permissions.stop_all_observation()

    def wipe_history(self) -> None:
        self._db.wipe_history()
        self._vision.clear_history()
        self._observer.clear()
        # Games still running stay tracked, from now on: the monitor won't announce
        # them again, so dropping them would lose the rest of the session too.
        self._sessions.restart_open()
        self._selector.reload_history()
        log.info("history deleted by the user")

    def wipe_all(self) -> None:
        self.permissions.stop_all_observation()
        # Cloud AI is not observation, but "everything" includes that consent too:
        # its stored row goes with the database, so the grant in memory must go now.
        for permission in Permission:
            if self.permissions.is_granted(permission):
                self.permissions.revoke(permission)
        self._monitor.stop(timeout=2.0)
        self._sessions.close()
        self._brain.discard_pending_memories()  # nothing said before comes back as a memory
        self._db.wipe_all()
        self._memory.clear()  # drops the in-process search cache of deleted texts
        self._short_term.clear()
        self._vision.clear_history()
        self._observer.clear()
        self._pending_transcript = None
        for name in (*KEY_NAMES.values(), AZURE_KEY_NAME):  # "everything" includes the API keys
            try:
                self._secrets.set(name, "")
            except Exception as exc:
                log.error("could not delete a stored API key: %s", type(exc).__name__)
        self._selector.reload_history()
        self._reactions.clear_quiet()
        self._tamagotchi.load()  # nothing stored any more -> fresh defaults
        self._emotion.load()
        self._games_hub.close_all()
        self._gifts.load()  # wallet, inventory and her keepsakes back to empty
        self._sessions = GameSessionTracker(self.bus, self._game_repo, self._clock, self.permissions)
        self._first_run = True
        self.apply_settings(Settings())
        self.bus.publish(SettingsChanged(section=""))
        # The chat lives only in its window and the status window shows the old
        # needs: drop both, the next open starts afresh.
        for window in (self._chat, self._status):
            if window is not None:
                window.close()
                window.deleteLater()
        self._chat = None
        self._status = None
        failed = _delete_log_files()  # they hold timestamped permission changes too
        if failed:
            log.warning("%d log file(s) could not be deleted", failed)
        log.info("all data deleted by the user")
        # The settings dialog closes itself after this returns; show onboarding again.
        QTimer.singleShot(0, self.show_first_run)

    # ------------------------------------------------------------------ windows

    def show_chat(self) -> None:
        if self._chat is None:
            self._chat = ChatWindow(self, on_mic=self._toggle_voice)
            self._sync_voice_ui()
        self.bus.publish(CharacterInteraction(kind=InteractionKind.TALK))
        self._raise(self._chat)
        self._ai.warm_up()  # the model loads while the user types

    def show_status(self) -> None:
        if self._status is None:
            self._status = StatusWindow(self)
        self._raise(self._status)

    def show_settings(self, page: str | None = None) -> None:
        if self._settings_dialog is None or not self._settings_dialog.isVisible():
            if self._settings_dialog is not None:
                self._settings_dialog.deleteLater()
            self._settings_dialog = SettingsDialog(self)
        else:
            self._settings_dialog.refresh()  # show what changed from the tray meanwhile
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
    def _raise(widget: QWidget) -> None:
        if widget.isMinimized():
            # show() keeps a visible window minimised and activateWindow() only
            # flashes its taskbar button: restore it explicitly.
            state = widget.windowState() & ~Qt.WindowState.WindowMinimized
            widget.setWindowState(state | Qt.WindowState.WindowActive)
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
        self._act_voice = self._action(menu, "Сказати голосом", lambda: self._toggle_voice("menu"))
        self._action(menu, "Подивитися на екран", self.look_at_screen)
        self._add_games_menu(menu)
        self._action(menu, "Подарунки", self.open_shop)
        menu.addSeparator()
        self._action(menu, "Подивитися стан", self.show_status)
        self._action(menu, "Налаштування", self.show_settings)
        menu.addSeparator()
        self._act_dnd = self._action(menu, "Не турбувати", self._toggle_dnd, checkable=True)
        self._act_mic_pause = self._action(menu, MIC_PAUSE_TEXT, self._toggle_mic_pause, checkable=True)
        self._act_stop = self._action(menu, "Зупинити спостереження", self.stop_all_observation)
        menu.addSeparator()
        self._action(menu, "Вийти", self._qt.quit)
        menu.setToolTipsVisible(True)  # why an item is disabled, what the microphone does
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
        self._add_games_menu(menu)
        self._action(menu, "Крамничка подарунків", self.open_shop)
        self._action(menu, "Налаштування", self.show_settings)
        self._action(menu, "Пам'ять", lambda: self.show_settings("memory"))
        self._action(menu, "Приватність і дозволи", lambda: self.show_settings("privacy"))
        menu.addSeparator()
        self._act_tray_dnd = self._action(menu, "Не турбувати", self._toggle_dnd, checkable=True)
        self._act_tray_mic_pause = self._action(menu, MIC_PAUSE_TEXT, self._toggle_mic_pause,
                                                checkable=True)
        self._act_click = self._action(menu, "Пропускати кліки крізь неї", self._toggle_click_through,
                                       checkable=True)
        self._act_lock = self._action(menu, "Заблокувати позицію", self._toggle_lock, checkable=True)
        self._act_tray_stop = self._action(menu, "Зупинити спостереження", self.stop_all_observation)
        menu.addSeparator()
        self._action(menu, "Вийти", self._qt.quit)
        menu.setToolTipsVisible(True)
        menu.aboutToShow.connect(self._refresh_menus)
        self._tray_menu = menu
        portrait = self._controller.portrait_pixmap("calm")
        icon = make_icon_from_pixmap(portrait) if portrait is not None else vector_icon()
        tray = TrayIcon(menu, icon, self._settings.character.name, self)
        tray.activated.connect(self._on_tray_activated)
        return tray

    def _add_games_menu(self, menu: QMenu) -> None:
        games = menu.addMenu("Пограти")
        for game_id in GAME_IDS:
            self._action(games, GAME_TITLES[game_id], lambda g=game_id: self.open_game(g))

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
        voice = self._voice.status()
        self._act_voice.setEnabled(voice.stt_ready)
        # Standby (waiting for her name) is not "listening to you": the item still
        # starts push-to-talk there, so it must not promise to stop the microphone.
        self._act_voice.setText("Перестати слухати" if voice.state == "listening" else "Сказати голосом")
        self._act_voice.setToolTip("" if voice.stt_ready else voice.stt_reason)
        self._sync_mic_pause_actions(voice)
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

    def _toggle_mic_pause(self) -> None:
        """"Пауза мікрофона": the always-open microphone closes until resumed - also
        after a restart (saved); push-to-talk keeps working.  She says so in a short line."""
        paused = not self._voice.open_mic_paused
        self._voice.set_open_mic_paused(paused)
        try:
            if paused:
                self._state.set(MIC_PAUSE_STATE_KEY, True)
            else:
                self._state.delete(MIC_PAUSE_STATE_KEY)
        except Exception:  # the pause itself works; only the restart would forget it
            log.exception("could not save the microphone pause")
        self._refresh_menus()

    def _sync_mic_pause_actions(self, voice: VoiceStatus) -> None:
        """The pause items: usable while she listens on an open microphone ("name" /
        "open"), checked while the user paused it; the tooltip says how she listens
        or why the microphone is paused."""
        open_mic = voice.stt_ready and voice.listen_mode != "push"
        paused = self._voice.open_mic_paused
        if open_mic:
            tip = voice.listening_text
        else:
            tip = MIC_PAUSE_PUSH_HINT if voice.stt_ready else voice.stt_reason
        for name in ("_act_mic_pause", "_act_tray_mic_pause"):
            action = getattr(self, name, None)  # the voice may report before the menus exist
            if action is None:
                continue
            # Enabled while paused too, so a pause left over from another mode can be lifted.
            action.setEnabled(open_mic or paused)
            action.setChecked(paused)
            action.setToolTip(tip)

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
        if not wanted:
            self._forget_activity()
        self._drop_disallowed_transcript()  # microphone revoked / kill switch: what was heard goes
        # While the monitor runs she can tell an active user from an idle one.
        self._controller.set_input_monitoring(self._monitor.is_running())
        if self._tray is not None:
            self._tray.set_observing(self.permissions.any_observation_active())
        self._refresh_menus()

    # -- voice and vision glue ---------------------------------------------------------------

    def _toggle_voice(self, source: str = "button") -> None:
        reason = self._voice.toggle_listening(source)
        if not reason:
            return
        if self._chat is not None and self._chat.isVisible():
            self._chat.add_system_line(reason)
        else:
            self._notify_voice(reason)

    def _sync_voice_ui(self) -> None:
        """The voice integration reconfigured or changed state (or the open microphone
        paused / resumed, or the user started / stopped talking with her): microphone
        button, tray "listening" badge and tooltip, the microphone pause items,
        ``ConversationActive``."""
        self._sync_conversation()
        st = self._voice.status()
        # Paused, switched to push-to-talk or recognition off: a request the open
        # microphone heard before that is dropped, not sent once the answer is done.
        self._drop_disallowed_transcript()
        if self._chat is not None:
            self._chat.set_voice_input(st.stt_ready, st.stt_reason, st.hotkey if st.hotkey_active else "")
            self._chat.set_voice_state(st.state)
        if self._tray is not None:
            # The red dot whenever the microphone is open, also while she waits for her name.
            self._tray.set_listening(st.state in ("listening", "standby"))
            self._tray.set_voice_note(st.listening_text)  # how she listens, or why it is paused
        self._sync_mic_pause_actions(st)

    def _notify_voice(self, message: str) -> None:
        """A voice problem the user should know about (no model, no device...)."""
        if self._tray is not None and self._tray.tray.isVisible():
            self._tray.show_message(self._settings.character.name, message)
        else:
            # Directly, not via SpeechRequested: a message about the voice is not voiced.
            self._controller.say(message, "thinking", solicited=True)

    def _vision_notice(self, message: str) -> None:
        if self._look_notices is not None:  # the chat asked: the notice becomes her reply
            self._look_notices.append(message)
            return
        self._controller.say(message, "thinking", "thinking", solicited=True)

    def _on_transcript(self, event: VoiceTranscript) -> None:
        """What the user said (local recognition) goes to the chat.  Never logged."""
        self._last_voice_at = self._clock.monotonic()  # talking to her hands-free is presence
        self._deliver_transcript(event.text.strip(), event.origin)

    def _transcript_allowed(self, origin: str) -> bool:
        return (self.permissions.is_granted(Permission.MICROPHONE)
                and self._voice.transcript_allowed(origin))

    def _drop_disallowed_transcript(self) -> None:
        """The waiting voice request may no longer be sent (microphone revoked, the open
        microphone paused or switched off, recognition off): forget it."""
        pending = self._pending_transcript
        if pending is not None and not self._transcript_allowed(pending[2]):
            self._pending_transcript = None
            self._transcript_timer.stop()
            log.info("a waiting voice request was dropped: the microphone no longer listens")

    def _deliver_transcript(self, text: str, origin: str = "push") -> None:
        """One chat answer at a time: with the chat window open a request heard while
        it is answering gets :data:`BUSY_REPLY` there (never spoken); otherwise only
        the newest request waits for the answer in flight and is sent after it.  What
        the open microphone overheard without her name (*origin* "heard") is answered
        but never mined for long-term memories."""
        if not self._transcript_allowed(origin):
            # Revoked, paused or switched off while the transcript was queued (or waited).
            if not self.permissions.is_granted(Permission.MICROPHONE):
                self._pending_transcript = None
            return
        if not text:
            return
        remember = origin != "heard"
        chat = self._chat
        if chat is not None and chat.isVisible():
            if chat.is_waiting():
                chat.add_system_line(BUSY_REPLY)
                return
            if self._chat_busy():  # an answer to a request made while the chat was hidden
                self._pending_transcript = (text, self._clock.monotonic(), origin)
                return
            draft = chat.input.text()  # what the user was typing stays
            chat.input.setText(text)
            self._chat_remember = remember
            self._chat_spoken = True
            try:
                chat.send()
            finally:
                self._chat_remember = True
                self._chat_spoken = False
            chat.input.setText(draft)
        elif self._chat_busy():
            # In "name" / "open" mode she can hear the next request before the answer
            # to the previous one is complete: no second answer in parallel.
            self._pending_transcript = (text, self._clock.monotonic(), origin)
        else:
            self.send_chat_stream(text, lambda _delta: None, lambda _reply: None, remember=remember,
                                  spoken=True)

    def _flush_transcript(self) -> None:
        """The answer in flight is complete: send the request that waited for it."""
        pending, self._pending_transcript = self._pending_transcript, None
        if pending is None or self._shut_down:
            return
        text, heard_at, origin = pending
        if self._clock.monotonic() - heard_at > PENDING_TRANSCRIPT_SECONDS:
            log.info("a voice request waited too long for the previous answer and was dropped")
            return
        self._deliver_transcript(text, origin)

    def _on_vision_state(self, event: VisionStateChanged) -> None:
        if self._tray is not None:
            watching = event.state in ("watching", "analyzing")
            self._tray.set_watching(event.app_title if watching else "")

    # -- what she knows about the moment (for the AI brain and the decision engine) ----

    def _on_ai_status(self, event: AIStatusChanged) -> None:
        self._ai_model = event.model

    def _on_foreground(self, event: ForegroundAppChanged) -> None:
        private = event.app.private or event.category is ActivityCategory.PRIVATE
        category = ActivityCategory.PRIVATE.value if private else event.category.value
        app = None if private else (event.app.display_name or None)
        since = self._clock.monotonic()
        if self._activity is not None and self._activity[0] == category:
            since = self._activity[2]
        self._activity = (category, app, since)

    def _set_away(self, away: bool) -> None:
        self._user_away = away

    def _away(self) -> bool:
        """Away from the PC: idle input and not talking to her by voice lately either
        (a hands-free conversation touches neither keyboard nor mouse)."""
        if not self._user_away:
            return False
        voice = self._last_voice_at
        return voice is None or self._clock.monotonic() - voice > VOICE_PRESENCE_SECONDS

    def _on_fullscreen(self, event: FullscreenChanged) -> None:
        self._fullscreen = event.active

    def _forget_activity(self) -> None:
        """Kill switch: what she knew about the user's apps goes too."""
        self._activity = None
        self._user_away = False

    def _current_game(self) -> tuple[str, float] | None:
        """(title, minutes) of a running game, if the monitor reports one."""
        sessions = self._sessions.open_sessions()
        if not sessions:
            return None
        session = sessions[-1]
        since = session.measured_from if session.measured_from is not None else session.started_at
        return session.title, max(0.0, (self._clock.now() - since) / 60)

    def _brain_context(self) -> BrainContext:
        category, app, _since = self._activity or (None, None, 0.0)
        game = self._current_game()
        # What she saw stays out of a chat answer a cloud voice (Azure) would read out
        # unless screen data may go to the cloud.
        voiced_ok = self._voice.screen_content_may_be_voiced()
        observation = self._vision.latest_observation() if voiced_ok else None
        return BrainContext(
            activity_category=category, activity_app=app,
            game_title=game[0] if game else None, game_minutes=game[1] if game else None,
            observation=observation, user_away=self._away(),
        )

    def _initiative_context(self) -> InitiativeContext:
        now = self._clock.monotonic()
        category, _app, since = self._activity or (None, None, now)
        game = self._current_game()
        return InitiativeContext(
            now=now,
            posture=self._controller.posture.value,
            activity_category=category,
            game_running=game is not None,
            game_minutes=game[1] if game else None,
            fullscreen=self._fullscreen,
            user_away=self._away(),
            minutes_since_last_chat=(now - self._last_chat) / 60 if self._last_chat is not None else None,
            time_of_day=time_of_day(datetime.fromtimestamp(self._clock.now())),
            ai_available=self._brain.available(Purpose.INITIATIVE),
            game_title=game[0] if game else None,
            activity_minutes=(now - since) / 60 if category is not None else None,
        )

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

    def _discard_private_sessions(self, privacy: PrivacyFilter) -> None:
        """Drop open game sessions whose game the user has just excluded, right
        away (the monitor would only report it on its next cycle)."""
        for session in self._sessions.open_sessions():
            info = self._games.get(session.game_id)
            processes = info.processes if info is not None else ()
            if not processes and session.game_id.startswith(("exe:", "custom:")):
                processes = (session.game_id.split(":", 1)[1],)
            if processes and all(privacy.is_private_process(p) for p in processes):
                self._sessions.discard(session.game_id)
                log.info("discarded a running game session the user made private")

    def _away_seconds(self) -> float | None:
        """Seconds since she last ran.  The clean-exit stamp alone is stale after a
        crash or power loss (no aboutToQuit), so the tamagotchi's heartbeat, saved
        every minute or so, counts too: whichever is newer."""
        stamps = (finite_number(self._state.get(key)) for key in (LAST_SHUTDOWN_KEY, KEY_LAST_SEEN))
        last = max((stamp for stamp in stamps if stamp is not None and stamp > 0), default=None)
        if last is None:
            return None
        return max(0.0, self._clock.now() - last)

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


def _delete_log_files() -> int:
    """Delete our log files (current and rotated) for "delete all data".  Returns
    how many could not be deleted.

    Each file handler's lock is held meanwhile, so no record is written half-way;
    with its stream set to None the handler opens a fresh file on the next record.
    Only handlers writing into our own log directory are touched.
    """
    try:
        own_dir = paths.log_dir().resolve()
    except OSError:
        return 1
    failed = 0
    for handler in logging.getLogger().handlers:
        if not isinstance(handler, logging.handlers.RotatingFileHandler):
            continue
        base = Path(handler.baseFilename)
        if base.parent.resolve() != own_dir:
            continue
        handler.acquire()
        try:
            stream, handler.stream = handler.stream, None
            if stream is not None:
                with contextlib.suppress(OSError, ValueError):  # a failed flush: delete it anyway
                    stream.close()
            try:
                backups = [p for p in base.parent.glob(base.name + ".*") if p.suffix[1:].isdigit()]
            except OSError:
                backups, failed = [], failed + 1
            for path in (base, *backups):
                try:
                    path.unlink(missing_ok=True)
                except OSError:  # e.g. opened by an editor on Windows
                    failed += 1
        finally:
            handler.release()  # log nothing before this: it would need this handler
    return failed


# ---------------------------------------------------------------------- entry point


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="chibi", description="Chibi AI Companion")
    parser.add_argument("--debug", action="store_true", help="verbose logging to the console")
    parser.add_argument("--quit-after", type=float, metavar="SECONDS",
                        help="quit automatically (smoke tests)")
    parser.add_argument("--self-test", action="store_true",
                        help="check the installation (assets, data, libraries) and exit")
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    setup_logging(debug=args.debug)  # also installs the crash hooks (redacted)
    _set_windows_app_id()

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    qt_app = QApplication.instance() or QApplication(sys.argv[:1])
    qt_app.setApplicationName("Chibi Companion")
    qt_app.setOrganizationName("ChibiCompanion")
    qt_app.setApplicationVersion(__version__)
    qt_app.setQuitOnLastWindowClosed(False)  # lives in the tray

    if args.self_test:
        from chibi import selftest

        return selftest.run()

    lock = QLockFile(str(paths.data_dir() / "chibi.lock"))
    lock.setStaleLockTime(0)
    if not lock.tryLock(200):
        log.info("another instance is already running")
        QMessageBox.information(None, "Chibi Companion", ALREADY_RUNNING_TEXT)
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
    chibi.close()  # shutdown() already ran on aboutToQuit; now nothing else can use the database
    lock.unlock()
    log.info("exited after %.0f s", time.monotonic() - started)
    return code
