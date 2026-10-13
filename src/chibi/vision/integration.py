"""VisionIntegration - wires screen vision into the app (Qt main thread).

:class:`~chibi.vision.service.VisionService` is the pipeline; this class is
the glue the composition root needs, kept out of ``app.py`` so it can be
tested on its own with fakes:

* **What may be looked at** (``resolve_target``): only the window in front,
  only if it belongs to an allowed game (``settings.vision.allowed_games``,
  matched through the games database) or an allowed app
  (``settings.vision.allowed_processes``), never if the privacy filter calls
  it private (process name, exe path and - when the user allowed window
  titles - the title), and never one of our own windows.  The rules are an
  immutable snapshot swapped on settings changes, because the service also
  calls the resolver on its worker thread (privacy re-check after capture).
* **When** (settings + permissions + events): the mode, interval and analyzer
  come from the settings; a 1 s ``QTimer`` drives ``service.tick()`` only while
  vision can actually run (mode not off, a model configured, capture
  permitted) - no timer otherwise.  Revoking ``SCREEN_CAPTURE`` /
  ``CLOUD_VISION`` or the kill switch stops everything at once and forgets
  what she saw; the permission checks themselves are read live from the
  :class:`~chibi.privacy.permissions.PermissionManager` on every step, so they
  take effect even before the bus delivers the event.
* **What she does with it**: every observation is published as
  :class:`~chibi.core.events.VisionObserved`; when the reaction policy says
  "react", one line goes through the :class:`SpeechComposer` (the AI brain
  gets the sanitised description via :meth:`latest_observation`).  A request
  from the user ("подивись на екран") is always answered.
* **Memory**: when a game session ends, one ``game_event`` memory with counts
  only ("S.T.A.L.K.E.R. 2: 3 смерті, 1 перемога за сесію") - never a
  description or anything read from the screen.

Frames are never written to disk; observations live in memory only (a short
list for the settings page and the brain's context) and are forgotten on
stop, revoke, kill switch and "Очистити".  Nothing here logs a window title,
a description or a frame.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import re
import sys
from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import EllipsisType
from typing import Any, Protocol

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

from chibi.activity.backend import ActivityBackend, ForegroundInfo, NullActivityBackend
from chibi.ai.providers.factory import is_cloud
from chibi.core.clock import Clock
from chibi.core.config import AISettings, Settings, VisionSettings
from chibi.core.event_bus import EventBus
from chibi.core.events import (
    AppShuttingDown,
    GameEnded,
    GameStarted,
    ObservationStopped,
    Permission,
    PermissionChanged,
    SettingsChanged,
    SystemLoadChanged,
    VisionObserved,
    VisionStateChanged,
)
from chibi.core.worker import SerialWorker
from chibi.gaming.games_db import GameDatabase, normalize_process, title_from_exe
from chibi.personality.reactions import speakable_title
from chibi.personality.text_uk import normalize as normalize_text
from chibi.personality.text_uk import plural
from chibi.privacy.filters import PrivacyFilter
from chibi.vision.analysis import Observation
from chibi.vision.analyzer import MODEL_OVERRIDE_SUPPORTED, ProviderFrameAnalyzer, VisionAI
from chibi.vision.capture_win import CaptureBackend, CaptureError, exclude_from_capture
from chibi.vision.policy import EVENT_COOLDOWNS, Decision, PolicyConfig, VisionReactionPolicy
from chibi.vision.service import FrameAnalyzer, Target, VisionMode, VisionService, VisionState

log = logging.getLogger(__name__)

TICK_MS = 1000
#: "Подивись на екран" while her own menu / chat is in front: let the user switch back first.
OWN_WINDOW_RETRY_MS = 2000
#: How long a user's request waits for its answer, on top of the AI request timeout
#: (a confirmation frame and some queueing).
MANUAL_ANSWER_EXTRA_SECONDS = 25.0
#: The brain gets an observation only while it is this fresh.
FRESH_OBSERVATION_SECONDS = 120.0
#: Observations kept for the settings page ("Що Мія бачила").
RECENT_LIMIT = 20
#: The same automatic error is shown again only after this long.
NOTICE_REPEAT_SECONDS = 600.0
#: Workers get this long to finish on shutdown.
WORKER_JOIN_SECONDS = 2.0
GAME_MEMORY_IMPORTANCE = 0.4

#: Events counted for the end-of-session memory, with their (one, few, many) forms.
MEMORY_EVENT_FORMS: dict[str, tuple[str, str, str]] = {
    "player_death": ("смерть", "смерті", "смертей"),
    "victory": ("перемога", "перемоги", "перемог"),
    "boss_fight": ("бій із босом", "бої з босами", "боїв із босами"),
    "level_up": ("новий рівень", "нові рівні", "нових рівнів"),
    "achievement": ("досягнення", "досягнення", "досягнень"),
    "new_area": ("нова локація", "нові локації", "нових локацій"),
}

#: Chat phrases that mean "look at my screen" (normalised: lower case, one space).
LOOK_REQUEST = re.compile(
    r"\b(?:подивис[ья]|поглянь|глянь|зирни|подивіться)\b[^.?!]{0,20}\bекран"
    r"|\bщо\s+(?:ти\s+)?(?:там\s+)?бачиш\b"
    r"|\bщо\s+(?:в|у)\s+мене\s+на\s+екрані\b",
    re.IGNORECASE,
)

#: Ukrainian reasons shown when vision cannot run (status line, "подивись на екран").
REASON_OFF = "Бачення екрана вимкнене: увімкни його в Налаштуваннях → «Бачення екрана»."
REASON_UNSUPPORTED = "Бачення екрана працює лише у Windows."
REASON_NO_CAPTURE = "Потрібен дозвіл «Знімки дозволених вікон» (Налаштування → «Бачення екрана»)."
REASON_NO_AI = "Потрібна модель, яка бачить зображення: вибери її на сторінці «AI»."
REASON_LOCAL_ONLY = "Увімкнено «Лише локальний аналіз», а модель на сторінці «AI» працює в хмарі."
REASON_NO_CLOUD_VISION = "Хмарній моделі потрібен окремий дозвіл «Хмарний аналіз екрана»."
REASON_NO_CLOUD_AI = "Хмарній моделі потрібен дозвіл на хмарний AI (сторінка «Приватність»)."
NOTE_NOTHING_ALLOWED = "Список дозволених ігор і програм порожній — їй поки нема на що дивитися."
NOTE_NO_MODEL_OVERRIDE = (
    "Окрема модель для зору поки не підтримується — кадри аналізує модель зі сторінки «AI»."
)
OWN_WINDOW_MESSAGE = "Я бачу лише власне вікно. Перемкнися на гру й попроси ще раз."
LOOK_FAILED_MESSAGE = "Не вдалося роздивитися екран. Спробуй ще раз за хвилинку."


def reactions_per_hour(reaction_frequency: int) -> int:
    """``GamingSettings.reaction_frequency`` (0..100) -> vision reactions per hour: 0 / 6 / 12."""
    return round(max(0, min(100, int(reaction_frequency))) * 12 / 100)


def is_look_request(text: str) -> bool:
    """True for chat messages such as "подивись на екран" or "що ти бачиш?"."""
    return bool(LOOK_REQUEST.search(normalize_text(text)))


def session_summary(title: str, counts: Counter[str] | dict[str, int]) -> str:
    """``"S.T.A.L.K.E.R. 2: 3 смерті, 1 перемога за сесію"`` - counts only, in a fixed order."""
    parts = [f"{counts[event]} {plural(counts[event], forms)}"
             for event, forms in MEMORY_EVENT_FORMS.items() if counts.get(event, 0) > 0]
    return f"{title}: {', '.join(parts)} за сесію"


def default_capture() -> CaptureBackend | None:
    """The Windows window capture, or None where it cannot work."""
    if sys.platform != "win32":
        return None
    try:
        from chibi.vision.capture_win import Win32WindowCapture

        return Win32WindowCapture()
    except (CaptureError, OSError, AttributeError) as exc:  # pragma: no cover - Windows only
        log.warning("window capture unavailable (%s); screen vision is disabled", type(exc).__name__)
        return None


# --- views for the settings page --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeenObservation:
    """One thing she saw, for "Що Мія бачила" (in memory only, no description)."""

    app_title: str
    scene: str  # chibi.vision.analysis.SCENES value
    event: str  # chibi.vision.analysis.EVENTS value
    confidence: float
    at: float  # epoch seconds
    reacted: bool


@dataclass(frozen=True, slots=True)
class VisionStatusView:
    mode: str  # VisionSettings.mode
    state: str  # VisionState value
    app_title: str  # the allowed app being watched ("" if none)
    #: Why vision cannot run now (Ukrainian), "" if it can (or the mode is off).
    blocked: str
    #: The analysing model runs on this PC; None when no model is configured.
    local: bool | None
    #: Extra hint for the page ("" if none).
    note: str = ""


# --- collaborators ------------------------------------------------------------------------------


class Composer(Protocol):
    def compose(self, *keys: str, solicited: bool = False, **variables: Any) -> bool: ...


class MemorySink(Protocol):
    def add(self, kind: str, text: str, source: str = ..., importance: float = ...) -> object: ...


class Worker(Protocol):
    def submit(self, job: Callable[[], None], priority: int = 1) -> None: ...

    def shutdown(self, timeout: float = 3.0) -> None: ...


class ManagedAnalyzer(FrameAnalyzer, Protocol):
    def cancel_current(self) -> None: ...

    def close(self) -> None: ...


#: (ai, post, ai settings, vision settings, allowed) -> analyzer
AnalyzerFactory = Callable[
    [VisionAI, Callable[[Callable[[], None]], None], AISettings, VisionSettings, Callable[[], bool]],
    ManagedAnalyzer,
]


def _provider_analyzer(ai: VisionAI, post: Callable[[Callable[[], None]], None], ai_settings: AISettings,
                       vision_settings: VisionSettings, allowed: Callable[[], bool]) -> ManagedAnalyzer:
    return ProviderFrameAnalyzer(ai, post, ai_settings, vision_settings, allowed=allowed)


class QtInvoker(QObject):
    """Runs callables on the thread that owns it (the Qt main thread), from any
    thread, through a queued signal - never synchronously.  After
    :meth:`close` nothing more runs (shutdown)."""

    _invoke = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._closed = False
        self._invoke.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def post(self, fn: Callable[[], None]) -> None:
        if self._closed:
            return
        with contextlib.suppress(RuntimeError):  # the QObject is gone (teardown)
            self._invoke.emit(fn)

    def close(self) -> None:
        self._closed = True

    @Slot(object)
    def _run(self, fn: Callable[[], None]) -> None:
        if self._closed:
            return
        try:
            fn()
        except Exception:
            log.exception("vision: main-thread callback failed")


@dataclass(frozen=True)
class _Rules:
    """What may be captured: immutable, read by the main and the worker thread."""

    privacy: PrivacyFilter
    games: GameDatabase
    allowed_games: frozenset[str]
    allowed_processes: frozenset[str]
    own_pid: int
    own_windows: frozenset[int]


class VisionIntegration(QObject):
    """See the module docstring.  Construct on the main thread; every public
    method is main-thread only."""

    def __init__(
        self,
        bus: EventBus,
        settings: Callable[[], Settings],
        permissions: Any,
        ai: VisionAI,
        composer: Composer,
        clock: Clock,
        *,
        games: Callable[[], GameDatabase],
        memory: MemorySink | None = None,
        quiet: Callable[[], bool] = lambda: False,
        on_spoken: Callable[[], None] | None = None,
        notify: Callable[[str], None] | None = None,
        backend: ActivityBackend | None = None,
        capture: CaptureBackend | EllipsisType | None = ...,
        worker: Worker | None = None,
        own_windows: Callable[[], Iterable[int]] = tuple,
        own_pid: int | None = None,
        analyzer_factory: AnalyzerFactory | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._bus = bus
        self._settings = settings
        self._permissions = permissions
        self._ai = ai
        self._composer = composer
        self._clock = clock
        self._games = games
        self._memory = memory
        self._quiet = quiet
        self._on_spoken = on_spoken
        self._notify_cb = notify
        self._backend: ActivityBackend = backend if backend is not None else NullActivityBackend()
        self._capture: CaptureBackend | None = (
            default_capture() if isinstance(capture, EllipsisType) else capture)
        self._worker: Worker = worker if worker is not None else SerialWorker("chibi-vision")
        self._own_windows = own_windows
        self._own_pid = os.getpid() if own_pid is None else own_pid
        self._analyzer_factory: AnalyzerFactory = analyzer_factory or _provider_analyzer
        self._closed = False

        self._invoker = QtInvoker(self)
        self._analyzer: ManagedAnalyzer | None = None
        self._analyzer_key: tuple[object, ...] | None = None
        self._allow_spoilers = False
        self._min_confidence = 0.6
        #: "Лише локальний аналіз" (a snapshot: read by the worker thread too).
        self._local_only = True
        self._high_load = False
        self._rules = self._build_rules(self._read_settings())
        self._excluded: set[int] = set()

        #: (what she saw, process it came from); the process is kept only to forget
        #: an app the moment it becomes private, it is never shown.
        self._seen: deque[tuple[SeenObservation, str]] = deque(maxlen=RECENT_LIMIT)
        self._latest: tuple[Observation, float, str] | None = None
        self._counts: dict[str, Counter[str]] = {}
        self._count_process: dict[str, str] = {}
        self._last_counted: dict[tuple[str, str], float] = {}
        self._manual_until: float | None = None
        self._manual_first: Observation | None = None
        self._in_request = False
        self._noticed: dict[str, float] = {}

        self._service = VisionService(
            post=self._invoker.post,
            worker=self._worker,
            capture=self._capture,
            foreground=self._foreground,
            resolve_target=self.resolve_target,
            capture_allowed=self._capture_allowed,
            cloud_allowed=self._cloud_allowed,
            policy=VisionReactionPolicy(PolicyConfig(), clock=clock.monotonic),
            clock=clock.monotonic,
            quiet=self._is_quiet,
            on_state=self._on_state,
            on_observation=self._on_observation,
            on_error=self._on_error,
        )

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self.tick)
        self._retry_timer = QTimer(self)
        self._retry_timer.setSingleShot(True)
        self._retry_timer.setInterval(OWN_WINDOW_RETRY_MS)
        self._retry_timer.timeout.connect(self._retry_request)

        self._unsubscribers = [
            bus.subscribe(SettingsChanged, self._on_settings_changed),
            bus.subscribe(PermissionChanged, self._on_permission_changed),
            bus.subscribe(ObservationStopped, self._on_observation_stopped),
            bus.subscribe(SystemLoadChanged, self._on_load),
            bus.subscribe(GameStarted, self._on_game_started),
            bus.subscribe(GameEnded, self._on_game_ended),
            bus.subscribe(AppShuttingDown, lambda _event: self.shutdown()),
        ]
        self.reconfigure()

    # ------------------------------------------------------------------ public API

    @property
    def service(self) -> VisionService:
        return self._service

    @property
    def timer_active(self) -> bool:
        return self._timer.isActive()

    def reconfigure(self) -> None:
        """Apply the current settings and permissions (idempotent).  The service is
        only reset when mode, interval or the analysing model actually changed, so a
        slider elsewhere never restarts a back-off or forgets a pending confirmation."""
        if self._closed:
            return
        s = self._read_settings()
        self._rules = self._build_rules(s)
        self._local_only = s.vision.local_only
        self._forget_private()
        try:
            mode = VisionMode(s.vision.mode)
        except ValueError:
            mode = VisionMode.OFF
        key: tuple[object, ...] | None = None
        if mode != VisionMode.OFF and not self._blocker(s):
            key = (copy.deepcopy(s.ai), s.vision.model)
        reset = False
        if key != self._analyzer_key:
            old = self._analyzer
            self._analyzer = None
            if key is not None:
                try:
                    self._analyzer = self._analyzer_factory(
                        self._ai, self._invoker.post, s.ai, s.vision, self._frame_may_leave)
                except Exception:
                    log.exception("building the vision analyzer failed")
                    key = None
            self._analyzer_key = key
            if old is not None:
                old.close()
            reset = True
        interval = float(s.vision.interval_seconds)
        if reset or mode != self._service.mode or max(5.0, interval) != self._service.interval:
            self._service.configure(mode, interval=interval, analyzer=self._analyzer)
        self._allow_spoilers = s.gaming.hints_and_spoilers
        self._min_confidence = s.vision.min_confidence
        self._service.policy.config = PolicyConfig(
            min_confidence=s.vision.min_confidence,
            # The user's hard cap on unsolicited lines applies to what she sees too.
            reactions_per_hour=min(reactions_per_hour(s.gaming.reaction_frequency),
                                   s.behavior.max_unsolicited_per_hour),
            allow_spoilers=s.gaming.hints_and_spoilers,
        )
        self._service.high_load = self._high_load and s.vision.adapt_to_load
        self._sync_timer()

    def request(self) -> bool:
        """"Подивись на екран" (her menu, the chat): look at the window in front now.

        If the window in front is one of hers (the menu or the chat the user just
        used), wait a moment and try once more, so the user can switch back.
        Returns False (and says why) if she cannot look at all.
        """
        if self._closed:
            return False
        blocker = self._request_blocker()
        if blocker:
            self._notify(blocker, user=True)
            return False
        self._retry_timer.stop()
        info = self._safe_foreground()
        if info is not None and self._is_own(info, self._rules):
            self._retry_timer.start()
            return True
        return self._request_now()

    def stop(self) -> None:
        """Kill switch / permission revoked: stop now and forget what she saw."""
        self._retry_timer.stop()
        if self._analyzer is not None:
            self._analyzer.cancel_current()
        self._service.stop()
        self._forget()

    def clear_history(self) -> None:
        """"Очистити" on the settings page: forget every observation (frames are never kept)."""
        self._seen.clear()
        self._latest = None
        self._manual_first = None
        self._service.recent.clear()
        self._counts.clear()
        self._count_process.clear()
        self._last_counted.clear()
        log.info("vision history cleared by the user")

    def latest_observation(self) -> str | None:
        """``"<scene>/<event>: <description>"`` of the newest observation younger than
        two minutes, for the AI brain's context (sanitised, untrusted text)."""
        latest = self._latest
        if latest is None or not self._capture_allowed():
            return None
        observation, seen_at, _process = latest
        if self._clock.monotonic() - seen_at > FRESH_OBSERVATION_SECONDS:
            return None
        text = f"{observation.scene}/{observation.event}"
        hide = observation.spoiler and not self._allow_spoilers
        if observation.description and not hide:
            text += f": {observation.description}"
        return text

    def recent_observations(self) -> list[SeenObservation]:
        """Newest first (in memory only)."""
        return [item for item, _process in reversed(self._seen)]

    def game_choices(self) -> list[tuple[str, str]]:
        """(game id, title) of every game the database knows, by title."""
        try:
            games = self._games().games()
        except Exception:
            log.exception("reading the games database failed")
            return []
        return sorted(((g.game_id, g.title) for g in games), key=lambda item: item[1].casefold())

    def status_view(self) -> VisionStatusView:
        s = self._read_settings()
        target = self._service.target
        state = self._service.state
        watching = state in (VisionState.WATCHING, VisionState.ANALYZING)
        notes = []
        if s.vision.mode != VisionMode.OFF and not (s.vision.allowed_games or s.vision.allowed_processes):
            notes.append(NOTE_NOTHING_ALLOWED)
        if s.vision.model.strip() and not MODEL_OVERRIDE_SUPPORTED:
            notes.append(NOTE_NO_MODEL_OVERRIDE)
        local = None if s.ai.provider == "none" else not is_cloud(s.ai)
        return VisionStatusView(
            mode=s.vision.mode, state=state.value,
            app_title=target.title if target is not None and watching else "",
            blocked=self._blocker(s), local=local, note=" ".join(notes),
        )

    def exclude_windows(self) -> None:
        """Hide her own windows (character, bubble) from every capture, so she never
        analyses herself.  Cheap and repeatable: Qt may recreate a native window
        (new handle) when its flags change, so new handles are picked up each tick."""
        try:
            ids = {int(i) for i in self._own_windows() if i}
        except Exception:
            log.exception("reading our own window ids failed")
            return
        for hwnd in ids - self._excluded:
            try:
                exclude_from_capture(hwnd)
            except Exception as exc:  # e.g. a ctypes argument error: never break the tick
                log.warning("could not exclude a window from capture (%s)", type(exc).__name__)
        self._excluded = ids
        if frozenset(ids) != self._rules.own_windows:
            rules = self._rules
            self._rules = _Rules(rules.privacy, rules.games, rules.allowed_games, rules.allowed_processes,
                                 rules.own_pid, frozenset(ids))

    def shutdown(self) -> None:
        """Stop the timers and the worker (<= ~2 s).  Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        self._retry_timer.stop()
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        if self._analyzer is not None:
            self._analyzer.close()
        self._service.stop()
        self._service.configure(VisionMode.OFF, interval=self._service.interval, analyzer=None)
        self._invoker.close()
        try:
            self._worker.shutdown(timeout=WORKER_JOIN_SECONDS)
        except Exception:
            log.exception("stopping the vision worker failed")
        self._forget()
        log.info("vision stopped")

    # ------------------------------------------------------------------ service callbacks

    def _foreground(self) -> ForegroundInfo | None:
        """Main or worker thread.  The title is read only with WINDOW_TITLES (for the
        privacy filter); it is never stored, logged or sent anywhere."""
        read_title = bool(self._permissions.is_granted(Permission.WINDOW_TITLES))
        return self._backend.foreground(read_title)

    def resolve_target(self, info: ForegroundInfo, require_game: bool = False) -> Target | None:
        """What may be captured of the window *info* describes (None = nothing).
        Main or worker thread; fails closed.  ``require_game`` is the service's
        own filter (GAME mode) and is applied by the service."""
        try:
            return self._resolve_with(self._rules, info)
        except Exception:
            log.exception("vision target lookup failed")
            return None

    @classmethod
    def _resolve_with(cls, rules: _Rules, info: ForegroundInfo) -> Target | None:
        if info is None or not info.hwnd or cls._is_own(info, rules):
            return None
        if rules.privacy.is_private(info):
            return None
        name = normalize_process(info.process_name)
        if not name:
            return None
        match = rules.games.match(name, info.exe_path)
        if match is not None and match.game_id in rules.allowed_games:
            return Target(info.hwnd, name, match.title, True)
        if f"exe:{name}" in rules.allowed_games:
            return Target(info.hwnd, name, match.title if match is not None else title_from_exe(name), True)
        if name in rules.allowed_processes:
            title = match.title if match is not None else title_from_exe(name)
            return Target(info.hwnd, name, title, match is not None)
        return None

    @staticmethod
    def _is_own(info: ForegroundInfo, rules: _Rules) -> bool:
        return info.pid == rules.own_pid or (bool(info.hwnd) and info.hwnd in rules.own_windows)

    def _capture_allowed(self) -> bool:
        return bool(self._permissions.is_granted(Permission.SCREEN_CAPTURE))

    def _cloud_allowed(self) -> bool:
        """A frame may go to a cloud model (any thread)."""
        return (not self._local_only and bool(self._permissions.is_granted(Permission.CLOUD_VISION))
                and bool(self._permissions.is_granted(Permission.CLOUD_AI)))

    def _frame_may_leave(self) -> bool:
        """Checked on the main thread right before a frame is handed to the AI service.

        Judged by the app's *current* AI settings, not the analyzer's snapshot: the
        AI service rebuilds its provider the moment settings are applied, before our
        SettingsChanged arrives, so a frame captured for a local model could otherwise
        reach a cloud provider without CLOUD_VISION."""
        if self._closed or self._analyzer is None or not self._capture_allowed():
            return False
        s = self._read_settings()
        if not is_cloud(s.ai):
            return True
        return (not s.vision.local_only and bool(self._permissions.is_granted(Permission.CLOUD_VISION))
                and bool(self._permissions.is_granted(Permission.CLOUD_AI)))

    def _is_quiet(self) -> bool:
        try:
            return bool(self._quiet())
        except Exception:
            log.exception("quiet check failed")
            return True

    def _on_state(self, state: VisionState, reason: str) -> None:
        target = self._service.target
        watching = state in (VisionState.WATCHING, VisionState.ANALYZING)
        title = target.title if target is not None and watching else ""
        if reason == title:  # the service reports the title as the reason for WATCHING
            reason = "watching"
        log.debug("vision state: %s", state.value)
        self._bus.publish(VisionStateChanged(state=state.value, app_title=title, reason=reason))

    def _on_observation(self, observation: Observation, decision: Decision) -> None:
        now = self._clock.monotonic()
        react = decision.action == "react"
        # While a job runs the service keeps its target, so this is the frame's window.
        target = self._service.target
        process = target.process_name if target is not None else ""
        self._latest = (observation, now, process)
        self._seen.append((SeenObservation(
            app_title=observation.app_title, scene=observation.scene, event=observation.event,
            confidence=observation.confidence, at=self._clock.now(), reacted=react,
        ), process))
        self._count(observation, now, process)
        self._bus.publish(VisionObserved(
            app_title=observation.app_title, scene=observation.scene, event=observation.event,
            confidence=observation.confidence, description=observation.description,
            emotion=observation.suggested_emotion, confirmed=observation.confirmed, react=react,
        ))
        manual = self._manual_until is not None and now <= self._manual_until
        if manual and decision.action == "confirm":
            # The service takes a second frame in a few seconds; answer then.
            self._manual_first = observation
            return
        game = speakable_title(observation.app_title) if observation.app_title else None
        variables = {"game": game} if game else {}
        if react:
            self._end_manual()
            said = self._composer.compose(f"vision.{observation.event}", "vision.generic", solicited=manual,
                                          **variables)
            if said and not manual and self._on_spoken is not None:
                try:
                    self._on_spoken()  # counts against the shared unsolicited budget
                except Exception:
                    log.exception("counting a vision line failed")
        elif manual:
            self._end_manual()
            self._composer.compose(*self._answer_keys(observation), solicited=True, **variables)

    def _on_error(self, message: str) -> None:
        user = self._in_request or (self._manual_until is not None
                                    and self._clock.monotonic() <= self._manual_until)
        if user:
            self._end_manual()
        self._notify(message, user=user)

    # ------------------------------------------------------------------ bus handlers

    def _on_settings_changed(self, event: SettingsChanged) -> None:
        if event.section in ("", "vision", "gaming", "ai", "activity", "behavior"):
            self.reconfigure()

    def _on_permission_changed(self, event: PermissionChanged) -> None:
        if event.permission not in (Permission.SCREEN_CAPTURE, Permission.CLOUD_VISION, Permission.CLOUD_AI):
            return
        if not event.granted:
            if event.permission == Permission.CLOUD_AI:
                if self._analyzer is not None and not self._analyzer.is_local:
                    self._analyzer.cancel_current()
            else:
                self.stop()
        self.reconfigure()

    def _on_observation_stopped(self, _event: ObservationStopped) -> None:
        self.stop()
        self.reconfigure()

    def _on_load(self, event: SystemLoadChanged) -> None:
        self._high_load = bool(event.high or event.on_battery_saver)
        self._service.high_load = self._high_load and self._read_settings().vision.adapt_to_load

    def _on_game_started(self, event: GameStarted) -> None:
        self._service.game_session(True)
        self._counts.pop(event.title, None)
        self._count_process.pop(event.title, None)
        self._drop_counted(event.title)

    def _on_game_ended(self, event: GameEnded) -> None:
        self._service.game_session(False)
        self._latest = None
        # A forgotten session (the game became private) has no title: find its
        # counts through the games database, and drop them all the same.
        info = self._safe_game(event.game_id)
        title, counts = "", None
        for candidate in dict.fromkeys(t for t in (event.title, info.title if info else "") if t):
            found = self._counts.pop(candidate, None)
            self._count_process.pop(candidate, None)
            self._drop_counted(candidate)
            if found and counts is None:
                title, counts = candidate, found
        if event.forgotten or not counts:
            return
        self._remember_session(title, counts)

    # ------------------------------------------------------------------ internals

    def _read_settings(self) -> Settings:
        try:
            return self._settings()
        except Exception:
            log.exception("reading the settings failed; vision stays off")
            return Settings()

    def _build_rules(self, s: Settings) -> _Rules:
        try:
            games = self._games()
        except Exception:
            log.exception("reading the games database failed")
            games = GameDatabase()
        previous = getattr(self, "_rules", None)
        return _Rules(
            privacy=PrivacyFilter(s.activity.excluded_processes),
            games=games,
            allowed_games=frozenset(s.vision.allowed_games),
            allowed_processes=frozenset(normalize_process(p) for p in s.vision.allowed_processes),
            own_pid=self._own_pid,
            own_windows=previous.own_windows if previous is not None else frozenset(),
        )

    def _blocker(self, s: Settings) -> str:
        """Why vision cannot run with these settings and permissions ("" = it can, or it is off)."""
        if s.vision.mode == VisionMode.OFF:
            return ""
        if self._capture is None:
            return REASON_UNSUPPORTED
        if not self._capture_allowed():
            return REASON_NO_CAPTURE
        if s.ai.provider == "none":
            return REASON_NO_AI
        if is_cloud(s.ai):
            if s.vision.local_only:
                return REASON_LOCAL_ONLY
            if not self._permissions.is_granted(Permission.CLOUD_VISION):
                return REASON_NO_CLOUD_VISION
            if not self._permissions.is_granted(Permission.CLOUD_AI):
                return REASON_NO_CLOUD_AI
        return ""

    def _request_blocker(self) -> str:
        s = self._read_settings()
        if s.vision.mode == VisionMode.OFF:
            return REASON_OFF
        blocker = self._blocker(s)
        if blocker:
            return blocker
        if self._analyzer is None:
            return REASON_NO_AI
        return ""

    def _sync_timer(self) -> None:
        wanted = (not self._closed and self._service.mode != VisionMode.OFF
                  and self._analyzer is not None and self._capture is not None)
        if wanted and not self._timer.isActive():
            self._timer.start()
            log.info("vision on: mode %s", self._service.mode.value)
        elif not wanted and self._timer.isActive():
            self._timer.stop()
            log.info("vision off")

    def tick(self) -> None:
        """The 1 s timer (public for tests): our windows stay excluded, a user's
        request that never got its confirmation frame is answered, the service ticks."""
        if self._closed:
            return
        self.exclude_windows()
        self._expire_manual()
        self._service.tick()

    def _request_now(self) -> bool:
        wait = self._read_settings().ai.request_timeout_seconds + MANUAL_ANSWER_EXTRA_SECONDS
        self._manual_until = self._clock.monotonic() + wait
        self._manual_first = None
        self._in_request = True
        try:
            started = self._service.request()
        finally:
            self._in_request = False
        if not started:
            self._end_manual()
        return started

    def _retry_request(self) -> None:
        if self._closed:
            return
        blocker = self._request_blocker()
        if blocker:
            self._notify(blocker, user=True)
            return
        info = self._safe_foreground()
        if info is not None and self._is_own(info, self._rules):
            self._notify(OWN_WINDOW_MESSAGE, user=True)
            return
        self._request_now()

    def _expire_manual(self) -> None:
        """A user's request that got no answer in time: answer from its first frame if
        the confirmation frame never came, otherwise say it did not work."""
        if self._manual_until is None or self._clock.monotonic() <= self._manual_until:
            return
        first = self._manual_first
        self._end_manual()
        if first is None:
            self._notify(LOOK_FAILED_MESSAGE, user=True)
            return
        game = speakable_title(first.app_title) if first.app_title else None
        self._composer.compose(*self._answer_keys(first, confirmed=False), solicited=True,
                               **({"game": game} if game else {}))

    def _end_manual(self) -> None:
        self._manual_until = None
        self._manual_first = None

    def _answer_keys(self, observation: Observation, *, confirmed: bool = True) -> tuple[str, ...]:
        """Phrase keys for answering "що ти бачиш?" (the user asked, so she always says something)."""
        keys: list[str] = []
        sure = observation.confidence >= self._min_confidence and confirmed
        spoiler = observation.spoiler and not self._allow_spoilers
        if observation.notable and sure and not spoiler:
            keys.append(f"vision.{observation.event}")
        if observation.event in ("combat", "puzzle") and not spoiler:
            keys.append(f"vision.look.{observation.event}")
        keys += [f"vision.look.{observation.scene}", "vision.look"]
        return tuple(keys)

    def _count(self, observation: Observation, now: float, process: str) -> None:
        """Count confirmed notable events for the end-of-session memory (counts only).
        A death screen seen in several frames is one death: per event kind at most
        once per its reaction cooldown."""
        event = observation.event
        if event not in MEMORY_EVENT_FORMS or not observation.app_title:
            return
        if observation.confidence < self._min_confidence:
            return
        if observation.needs_confirmation and not observation.confirmed:
            return
        key = (observation.app_title, event)
        last = self._last_counted.get(key)
        if last is not None and now - last < EVENT_COOLDOWNS.get(event, 120.0):
            return
        self._last_counted[key] = now
        self._counts.setdefault(observation.app_title, Counter())[event] += 1
        self._count_process[observation.app_title] = process

    def _drop_counted(self, title: str) -> None:
        for key in [k for k in self._last_counted if k[0] == title]:
            del self._last_counted[key]

    def _remember_session(self, title: str, counts: Counter[str]) -> None:
        if self._memory is None or not self._read_settings().ai.remember_facts:
            return
        try:
            self._memory.add("game_event", session_summary(title, counts), source="game",
                             importance=GAME_MEMORY_IMPORTANCE)
        except Exception as exc:
            log.warning("could not store the game session memory (%s)", type(exc).__name__)

    def _safe_game(self, game_id: str) -> Any:
        try:
            return self._games().get(game_id)
        except Exception:
            return None

    def _safe_foreground(self) -> ForegroundInfo | None:
        try:
            return self._foreground()
        except Exception:
            log.exception("foreground lookup failed")
            return None

    def _forget(self) -> None:
        self._seen.clear()
        self._latest = None
        self._counts.clear()
        self._count_process.clear()
        self._last_counted.clear()
        self._end_manual()

    def _forget_private(self) -> None:
        """An app became private (the user excluded it): forget what she saw of it,
        like the rest of the app forgets its game session."""
        privacy = self._rules.privacy

        def private(process: str) -> bool:
            return privacy.is_private_process(process)  # "" (unknown) counts as private

        kept = [entry for entry in self._seen if not private(entry[1])]
        purged = len(kept) != len(self._seen)
        if purged:
            self._seen.clear()
            self._seen.extend(kept)
        if self._latest is not None and private(self._latest[2]):
            self._latest = None
            purged = True
        for title, process in list(self._count_process.items()):
            if private(process):
                self._counts.pop(title, None)
                self._count_process.pop(title, None)
                self._drop_counted(title)
                purged = True
        if purged:
            self._service.recent.clear()  # the model's "previous scene" context
            log.info("forgot what she saw of an app that became private")

    def _notify(self, message: str, *, user: bool) -> None:
        """Tell the user, without repeating the same automatic problem every minute."""
        if not message:
            return
        now = self._clock.monotonic()
        last = self._noticed.get(message)
        if not user and last is not None and now - last < NOTICE_REPEAT_SECONDS:
            return
        self._noticed[message] = now
        if self._notify_cb is None:
            return
        try:
            self._notify_cb(message)
        except Exception:
            log.exception("showing a vision notice failed")

