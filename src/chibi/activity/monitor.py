"""Activity monitor: a worker thread that turns OS observations into events.

Each cycle (every ``poll_interval_seconds``):

1. **Permission gate.**  Without ``APP_METADATA`` the monitor does not touch
   the backend at all, forgets everything it knew and emits nothing.  The
   permission is checked again after gathering and right before *every*
   publish; anything gathered while it was revoked is thrown away.
2. **Foreground app.**  The title is requested only while ``WINDOW_TITLES`` is
   granted and is used only for the privacy filter and the classifier - no
   event carries it.  A private app (password manager, messenger, banking,
   user exclusion, private-browsing title) is published as
   ``AppInfo.private_app()`` / ``PRIVATE`` and nothing else about it.
   ``ForegroundAppChanged`` is emitted when (process, category) changes.
3. **Idle.**  ``UserIdle`` once when input-idle time crosses the threshold,
   ``UserReturned`` (with the longest idle seen) when input resumes.  While a
   game or a video is in front, crossing the threshold does not count: a
   gamepad is not "input" for Windows and nobody touches the mouse during a
   film.
4. **Process scan** (every ``process_scan_interval_seconds``).  The first scan
   is a silent baseline (games already running do start a session).  Later
   scans emit ``AppLaunched``/``AppClosed`` for *known* apps only - never for
   arbitrary processes, never for private ones - and ``GameStarted``/
   ``GameEnded`` for games.  A game recognised only in the foreground (by
   heuristics or title) also starts a session, which ends once its process is
   gone.
5. **Milestones.**  ``ActivityMilestone`` after 1/2/3 hours of the same
   category, once per streak.  An idle period ends the streak; so does a
   switch to another category that lasts longer than a short grace period
   (alt-tabbing to a messenger for half a minute does not reset two hours of
   gaming).

Worker-thread rules (see ARCHITECTURE.md): no Qt, no SQLite; only
``EventBus.publish`` and ``is_granted``.  Waiting uses ``Event.wait`` so
``stop()`` returns promptly.  Errors in a cycle are logged (rate-limited,
without exception messages, which could name apps or paths) and the loop
carries on.
"""

from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from chibi.activity.backend import ActivityBackend, ForegroundInfo
from chibi.activity.classifier import ActivityClassifier
from chibi.core.clock import Clock
from chibi.core.config import ActivitySettings
from chibi.core.event_bus import EventBus
from chibi.core.events import (
    ActivityCategory,
    ActivityMilestone,
    AppClosed,
    AppInfo,
    AppLaunched,
    Event,
    ForegroundAppChanged,
    GameEnded,
    GameStarted,
    Permission,
    UserIdle,
    UserReturned,
)
from chibi.gaming.games_db import GameMatch, normalize_process
from chibi.privacy.filters import PrivacyFilter

log = logging.getLogger(__name__)

#: Continuous-activity milestones (seconds), each reported once per streak.
MILESTONE_SECONDS: tuple[float, ...] = (3600.0, 7200.0, 10800.0)
MILESTONE_CATEGORIES: frozenset[ActivityCategory] = frozenset(
    {
        ActivityCategory.GAMING,
        ActivityCategory.PROGRAMMING,
        ActivityCategory.BROWSING,
        ActivityCategory.WATCHING_VIDEO,
        ActivityCategory.WORKING,
    }
)
#: Input this recent means the user is back.
RETURN_INPUT_SECONDS = 5.0
#: A switch to another category shorter than this does not break a streak.
STREAK_GRACE_SECONDS = 60.0
#: Foreground categories during which a long input-idle time is not "away".
IDLE_EXEMPT_CATEGORIES: frozenset[ActivityCategory] = frozenset(
    {ActivityCategory.GAMING, ActivityCategory.WATCHING_VIDEO}
)
ERROR_LOG_INTERVAL_SECONDS = 60.0
#: Floor for the wait between cycles, so a broken setting can never spin the CPU.
MIN_POLL_SECONDS = 0.05
DEFAULT_SETTINGS = ActivitySettings()


class PermissionChecker(Protocol):
    def is_granted(self, permission: Permission) -> bool: ...


@dataclass
class _Streak:
    category: ActivityCategory
    started: float
    last_seen: float  # last cycle the foreground was in this category
    reached: set[float] = field(default_factory=set)
    #: category the user switched to, and since when (still within the grace period)
    switched_to: ActivityCategory | None = None
    switched_at: float = 0.0


@dataclass
class _GameSession:
    title: str
    started: float
    processes: set[str]


@dataclass
class _RunningApp:
    app: AppInfo
    since: float
    processes: set[str]


@dataclass
class _State:
    """Everything the monitor remembers.  Dropped whenever APP_METADATA is not granted."""

    fg_key: tuple[str, ActivityCategory] | None = None
    fg_app: AppInfo | None = None
    fg_category: ActivityCategory | None = None
    streak: _Streak | None = None
    idle_announced: bool = False
    idle_longest: float = 0.0
    last_idle: float | None = None
    last_scan: float | None = None
    baseline_done: bool = False
    #: (display name, category) -> app: several processes of one app count once
    running_apps: dict[tuple[str, ActivityCategory], _RunningApp] = field(default_factory=dict)
    sessions: dict[str, _GameSession] = field(default_factory=dict)


def _describe(exc: BaseException) -> str:
    """Exception type and location only: messages may contain process names or paths."""
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return type(exc).__name__
    last = frames[-1]
    return f"{type(exc).__name__} at {Path(last.filename).name}:{last.lineno} ({last.name})"


class ActivityMonitor:
    def __init__(
        self,
        backend: ActivityBackend,
        permissions: PermissionChecker,
        bus: EventBus,
        classifier: ActivityClassifier,
        privacy: PrivacyFilter,
        settings: Callable[[], ActivitySettings],
        clock: Clock,
    ) -> None:
        self._backend = backend
        self._permissions = permissions
        self._bus = bus
        self._settings = settings
        self._clock = clock
        self._config_lock = threading.Lock()
        self._classifier = classifier
        self._privacy = privacy
        self._cycle_lock = threading.Lock()  # one cycle at a time (worker vs poll_once)
        self._lifecycle_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state = _State()
        self._last_error_log: float | None = None
        self._suppressed_errors = 0
        self._settings_failing = False

    # -- lifecycle ------------------------------------------------------------------

    def start(self) -> None:
        """Start the worker thread (no-op if running).  Every start observes afresh."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set():
                return
            stop = threading.Event()
            with self._cycle_lock:
                self._state = _State()
            thread = threading.Thread(target=self._run, args=(stop,), name="activity-monitor", daemon=True)
            self._stop_event = stop
            self._thread = thread
            thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        with self._lifecycle_lock:
            thread, stop = self._thread, self._stop_event
            self._thread = None
        stop.set()
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout)
        if thread.is_alive():
            log.warning("activity monitor did not stop within %.1f s", timeout)

    def is_running(self) -> bool:
        with self._lifecycle_lock:
            thread = self._thread
            return thread is not None and thread.is_alive() and not self._stop_event.is_set()

    def reconfigure(self, classifier: ActivityClassifier, privacy: PrivacyFilter) -> None:
        """Swap classifier and privacy filter (after the user changed games or exclusions).

        Thread-safe; takes effect from the next cycle.  Apps that just became
        private are dropped from what the monitor reports at that cycle.
        """
        with self._config_lock:
            self._classifier = classifier
            self._privacy = privacy

    def poll_once(self) -> None:
        """Run one full cycle synchronously (including a process scan when due)."""
        with self._cycle_lock:
            try:
                self._cycle()
            except Exception as exc:
                self._report_error(exc)

    def _run(self, stop: threading.Event) -> None:
        log.info("activity monitor started")
        while not stop.is_set():
            try:
                self.poll_once()
                interval = self._poll_interval()
            except Exception:  # last resort: poll_once already reports cycle errors
                interval = DEFAULT_SETTINGS.poll_interval_seconds
            stop.wait(interval)
        log.info("activity monitor stopped")

    # -- one cycle ----------------------------------------------------------------------

    def _cycle(self) -> None:
        settings = self._read_settings()
        if self._revoked():
            return
        with self._config_lock:
            classifier, privacy = self._classifier, self._privacy
        now = self._clock.monotonic()
        events: list[Event] = []

        # The permission is re-checked after every backend call: once revoked,
        # nothing more is read and what was read is dropped.
        read_titles = self._permissions.is_granted(Permission.WINDOW_TITLES)
        info = self._backend.foreground(read_title=read_titles)
        if self._revoked():
            return
        if info is not None and info.title is not None and not (
            read_titles and self._permissions.is_granted(Permission.WINDOW_TITLES)
        ):
            # Never use a title the user does not (or no longer) allow, even if a backend returned one.
            info = replace(info, title=None)
        self._observe_foreground(info, classifier, privacy, now, events)

        idle = self._backend.idle_seconds()
        if self._revoked():
            return
        self._observe_idle(idle, settings, now, events)

        state = self._state
        if state.last_scan is None or now - state.last_scan >= settings.process_scan_interval_seconds:
            processes = self._backend.running_processes()
            if self._revoked():
                return
            state.last_scan = now
            self._observe_processes(processes, classifier, privacy, now, events)

        self._check_milestones(now, events)
        self._publish(events)

    def _revoked(self) -> bool:
        """True - and everything the monitor knew is forgotten - if APP_METADATA is not granted now."""
        if self._metadata_granted():
            return False
        self._state = _State()
        return True

    def _publish(self, events: list[Event]) -> None:
        if self._revoked():
            return
        for event in events:
            if self._revoked():
                return  # revoked mid-publish: the rest is discarded with the state
            self._bus.publish(event)

    # -- foreground -----------------------------------------------------------------------

    def _observe_foreground(
        self,
        info: ForegroundInfo | None,
        classifier: ActivityClassifier,
        privacy: PrivacyFilter,
        now: float,
        events: list[Event],
    ) -> None:
        if info is None:
            return  # nothing in front (lock screen, secure desktop): keep what we had
        state = self._state
        game: GameMatch | None = None
        if self._is_private(privacy, info):
            app, category, confidence = AppInfo.private_app(), ActivityCategory.PRIVATE, 1.0
            key = ("", ActivityCategory.PRIVATE)
        else:
            result = classifier.classify(info)
            if result.category is ActivityCategory.PRIVATE:  # not expected; fail closed anyway
                app, category, confidence = AppInfo.private_app(), ActivityCategory.PRIVATE, 1.0
                key = ("", ActivityCategory.PRIVATE)
            else:
                process = normalize_process(info.process_name)
                app = AppInfo(process_name=process, display_name=result.display_name)
                category, confidence, game = result.category, result.confidence, result.game
                key = (process, category)
        if key != state.fg_key:
            events.append(ForegroundAppChanged(
                app=app, category=category, confidence=confidence, timestamp=self._clock.now()
            ))
        state.fg_key, state.fg_app, state.fg_category = key, app, category
        self._update_streak(category, now)
        if game is not None:
            self._ensure_session(game, app.process_name, now, events)

    # -- idle -------------------------------------------------------------------------------

    def _observe_idle(
        self, idle: float | None, settings: ActivitySettings, now: float, events: list[Event]
    ) -> None:
        if idle is None or idle != idle or idle < 0:  # unknown / NaN / nonsense
            return
        state = self._state
        threshold = float(settings.idle_threshold_minutes) * 60.0
        if state.idle_announced:
            input_happened = idle < RETURN_INPUT_SECONDS or (
                state.last_idle is not None and idle < state.last_idle - 1.0
            )
            if input_happened:
                events.append(UserReturned(away_seconds=state.idle_longest, timestamp=self._clock.now()))
                state.idle_announced = False
                state.idle_longest = 0.0
                state.streak = None
                if state.fg_category is not None:
                    state.streak = _Streak(state.fg_category, started=now, last_seen=now)
            else:
                state.idle_longest = max(state.idle_longest, idle)
        elif idle >= threshold and state.fg_category not in IDLE_EXEMPT_CATEGORIES:
            events.append(UserIdle(idle_seconds=idle, timestamp=self._clock.now()))
            state.idle_announced = True
            state.idle_longest = idle
            state.streak = None  # being away ends any streak
        state.last_idle = idle

    # -- streaks / milestones ------------------------------------------------------------

    def _update_streak(self, category: ActivityCategory, now: float) -> None:
        state = self._state
        if state.idle_announced:
            return  # no streak while the user is away; returning starts a new one
        streak = state.streak
        if streak is None:
            state.streak = _Streak(category, started=now, last_seen=now)
            return
        if category == streak.category:
            streak.last_seen = now
            streak.switched_to = None
            return
        if streak.switched_to != category:
            streak.switched_to, streak.switched_at = category, now
        if now - streak.last_seen > STREAK_GRACE_SECONDS:
            state.streak = _Streak(category, started=streak.switched_at, last_seen=now)

    def _check_milestones(self, now: float, events: list[Event]) -> None:
        state = self._state
        streak = state.streak
        if (
            streak is None
            or state.idle_announced
            or state.fg_app is None
            or state.fg_category != streak.category
            or streak.category not in MILESTONE_CATEGORIES
        ):
            return
        elapsed = now - streak.started
        crossed = [m for m in MILESTONE_SECONDS if elapsed >= m and m not in streak.reached]
        if not crossed:
            return
        # After a gap (e.g. the PC slept) report only the highest milestone, never a burst.
        top = max(crossed)
        streak.reached.update(m for m in MILESTONE_SECONDS if m <= top)
        events.append(ActivityMilestone(
            category=streak.category, app=state.fg_app, seconds=top, timestamp=self._clock.now()
        ))

    # -- process scan / game sessions ---------------------------------------------------

    def _observe_processes(
        self,
        processes: set[str],
        classifier: ActivityClassifier,
        privacy: PrivacyFilter,
        now: float,
        events: list[Event],
    ) -> None:
        if not processes:
            # Windows always runs processes: an empty list means the scan failed,
            # not that every app (and game) just closed.
            return
        state = self._state
        visible = {
            name for name in (normalize_process(p) for p in processes)
            if name and not self._is_private_process(privacy, name)
        }
        games_now: dict[str, tuple[GameMatch, set[str]]] = {}
        apps_now: dict[tuple[str, ActivityCategory], _RunningApp] = {}
        for name in sorted(visible):
            game = classifier.match_game(name)
            if game is not None:
                games_now.setdefault(game.game_id, (game, set()))[1].add(name)
                key, display = (game.title, ActivityCategory.GAMING), game.title
            else:
                known = classifier.known_app(name)
                if known is None:
                    continue  # unknown processes are never reported
                key, display = (known.display_name, known.category), known.display_name
            apps_now.setdefault(key, _RunningApp(AppInfo(name, display), now, set())).processes.add(name)

        # Game sessions: end those whose processes are all gone (or became private), start new ones.
        # The foreground process is running by definition, even if the scan missed it - otherwise a
        # name mismatch would start and end its session on every cycle.
        alive = set(visible)
        if state.fg_key is not None and state.fg_key[1] is not ActivityCategory.PRIVATE:
            alive.add(state.fg_key[0])
        for game_id in sorted(state.sessions):
            session = state.sessions[game_id]
            if game_id in games_now:
                session.processes |= games_now[game_id][1]
            if not session.processes & alive:
                del state.sessions[game_id]
                events.append(GameEnded(
                    game_id=game_id, title=session.title, duration_seconds=max(0.0, now - session.started),
                    timestamp=self._clock.now(),
                ))
        for game_id in sorted(games_now):
            game, names = games_now[game_id]
            self._ensure_session(game, min(names), now, events)

        if not state.baseline_done:
            # Apps already running when observation starts were not "launched" now.
            state.running_apps = apps_now
            state.baseline_done = True
            return
        for key in list(state.running_apps):
            running = state.running_apps[key]
            if all(self._is_private_process(privacy, p) for p in running.processes):
                del state.running_apps[key]  # became private: forget silently, no "closed" event
            elif key in apps_now:
                running.processes = apps_now[key].processes
        for key in sorted(apps_now.keys() - state.running_apps.keys()):
            state.running_apps[key] = apps_now[key]
            events.append(AppLaunched(app=apps_now[key].app, category=key[1], timestamp=self._clock.now()))
        for key in sorted(state.running_apps.keys() - apps_now.keys()):
            running = state.running_apps.pop(key)
            events.append(AppClosed(
                app=running.app, category=key[1], ran_seconds=max(0.0, now - running.since),
                timestamp=self._clock.now(),
            ))

    def _ensure_session(self, game: GameMatch, process: str, now: float, events: list[Event]) -> None:
        session = self._state.sessions.get(game.game_id)
        if session is not None:
            session.processes.add(process)
            return
        self._state.sessions[game.game_id] = _GameSession(game.title, now, {process})
        events.append(GameStarted(
            game_id=game.game_id, title=game.title, process_name=process, timestamp=self._clock.now()
        ))

    # -- helpers ---------------------------------------------------------------------------

    def _metadata_granted(self) -> bool:
        try:
            return bool(self._permissions.is_granted(Permission.APP_METADATA))
        except Exception:
            return False  # cannot confirm consent -> no observation

    @staticmethod
    def _is_private(privacy: PrivacyFilter, info: ForegroundInfo) -> bool:
        try:
            return bool(privacy.is_private(info))
        except Exception:
            return True

    @staticmethod
    def _is_private_process(privacy: PrivacyFilter, process: str) -> bool:
        try:
            return bool(privacy.is_private_process(process))
        except Exception:
            return True

    def _read_settings(self) -> ActivitySettings:
        try:
            settings = self._settings()
        except Exception as exc:
            if not self._settings_failing:  # once per failure streak, not twice per cycle
                log.warning("could not read activity settings (%s); using defaults", type(exc).__name__)
            self._settings_failing = True
            return DEFAULT_SETTINGS
        self._settings_failing = False
        return settings

    def _poll_interval(self) -> float:
        try:
            interval = float(self._read_settings().poll_interval_seconds)
        except (TypeError, ValueError):
            interval = DEFAULT_SETTINGS.poll_interval_seconds
        if interval != interval:  # NaN
            interval = DEFAULT_SETTINGS.poll_interval_seconds
        return max(MIN_POLL_SECONDS, interval)

    def _report_error(self, exc: BaseException) -> None:
        now = self._clock.monotonic()
        if self._last_error_log is not None and now - self._last_error_log < ERROR_LOG_INTERVAL_SECONDS:
            self._suppressed_errors += 1
            return
        log.warning("activity cycle failed: %s (%d similar errors suppressed)",
                    _describe(exc), self._suppressed_errors)
        log.debug("cycle failure stack:\n%s", "".join(traceback.format_tb(exc.__traceback__)))
        self._last_error_log = now
        self._suppressed_errors = 0
