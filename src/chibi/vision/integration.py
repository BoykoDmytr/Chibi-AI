"""VisionIntegration - wires screen vision into the app (Qt main thread).

:class:`~chibi.vision.service.VisionService` is the pipeline; this class is
the glue the composition root needs, kept out of ``app.py`` so it can be
tested on its own with fakes:

* **What may be looked at** (``resolve_target``): only the window in front,
  never if the privacy filter calls it private (process name, exe path and -
  when the user allowed window titles - the title), and never one of our own
  windows.  Which windows depends on ``settings.vision.scope``:

  - ``"allowed"``: an allowed game (``settings.vision.allowed_games``, matched
    through the games database) or an allowed app
    (``settings.vision.allowed_processes``);
  - ``"foreground"``: any window, except Windows' own surfaces (desktop,
    taskbar, lock screen, Start and search, Task Manager, UAC and credential
    prompts, the Microsoft / work account sign-in windows - :data:`SYSTEM_PROCESSES`)
    and browsers while window titles may
    not be read: without the title the privacy filter cannot tell a private
    tab, a bank or web mail from any other page, so a browser is skipped (and
    the status says why).

  The rules are an immutable snapshot swapped on settings changes, because the
  service also calls the resolver on its worker thread (privacy re-check after
  capture); the window-titles permission is read live.
* **When** (settings + permissions + events): the mode, interval and analyzer
  come from the settings; a 1 s ``QTimer`` drives ``service.tick()`` only while
  vision can actually run (mode not off, a model configured, capture
  permitted) - no timer otherwise.  Revoking ``SCREEN_CAPTURE`` /
  ``CLOUD_VISION`` or the kill switch stops everything at once and forgets
  what she saw; the permission checks themselves are read live from the
  :class:`~chibi.privacy.permissions.PermissionManager` on every step, so they
  take effect even before the bus delivers the event.  With the "foreground"
  scope in AUTO mode she does not look while the user is away
  (``UserIdle`` .. ``UserReturned``, unless they talked to her by voice in the
  last :data:`VOICE_PRESENCE_SECONDS`), and a window that keeps showing the
  same picture is captured less often.  As an observer she takes no automatic
  frames while she could not say anything about them (her own lines may not
  use the AI, it is offline or its limit is reached): the status says why
  (:attr:`VisionStatusView.observer_note`).
* **The conversation comes first** (``ConversationActive``): while the user
  is talking with her no automatic frame is taken, a frame in flight is
  abandoned - its analysis request withdrawn from the AI service, its end
  ignored (no notice, no back-off) - and frames resume a few seconds after the
  conversation (:meth:`VisionService.set_conversation`).  "Подивись на екран"
  said in the conversation itself is answered at once: the user is waiting for
  it, so its frame is analysed as their request (``user_waiting``), not held
  back as background work.
* **What she does with it**: every analysed observation that is still allowed
  when its answer arrives is published as
  :class:`~chibi.core.events.VisionObserved` (see :meth:`_publish`).  With
  ``settings.vision.commentary == "events"``, when the reaction policy says
  "react", one line goes through the :class:`SpeechComposer`; with
  ``"observer"`` this class never comments on its own - the observer decides
  from ``VisionObserved``.  The AI brain gets the sanitised description via
  :meth:`latest_observation`.  A request from the user ("подивись на екран")
  is always answered here, in both modes.
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
import dataclasses
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
from chibi.activity.classifier import AppCatalog
from chibi.ai.providers.factory import is_cloud
from chibi.core.clock import Clock
from chibi.core.config import AISettings, Settings, VisionSettings
from chibi.core.event_bus import EventBus
from chibi.core.events import (
    ActivityCategory,
    AppShuttingDown,
    ConversationActive,
    GameEnded,
    GameStarted,
    ObservationStopped,
    Permission,
    PermissionChanged,
    SettingsChanged,
    SystemLoadChanged,
    UserIdle,
    UserReturned,
    VisionObserved,
    VisionStateChanged,
    VoiceTranscript,
)
from chibi.core.worker import SerialWorker
from chibi.gaming.games_db import GameDatabase, GameMatch, normalize_process, title_from_exe
from chibi.personality.reactions import budget_limit, speakable_title
from chibi.personality.text_uk import normalize as normalize_text
from chibi.personality.text_uk import plural
from chibi.privacy.filters import PrivacyFilter
from chibi.privacy.filters import normalize_text as collapse_spaces
from chibi.vision.analysis import (
    GAME_SCENES,
    PRIVATE_SCENES,
    WORK_EVENTS,
    WORK_SCENES,
    Observation,
    sanitize_description,
)
from chibi.vision.analyzer import ProviderFrameAnalyzer, VisionAI
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
REASON_NO_CAPTURE = "Потрібен дозвіл «Бачити вікна ігор і програм» (Налаштування → «Бачення екрана»)."
REASON_NO_AI = "Потрібна модель, яка бачить зображення: вибери її на сторінці «AI»."
REASON_LOCAL_ONLY = "Увімкнено «Лише локальний аналіз», а модель на сторінці «AI» працює в хмарі."
REASON_NO_CLOUD_VISION = "Хмарній моделі потрібен окремий дозвіл «Хмарний аналіз екрана»."
REASON_NO_CLOUD_AI = "Хмарній моделі потрібен дозвіл на хмарний AI (сторінка «Приватність»)."
NOTE_NOTHING_ALLOWED = "Список дозволених ігор і програм порожній — їй поки нема на що дивитися."
OWN_WINDOW_MESSAGE = "Я бачу лише власне вікно. Перемкнися на гру чи програму й попроси ще раз."
LOOK_FAILED_MESSAGE = "Не вдалося роздивитися екран. Спробуй ще раз за хвилинку."

SCOPE_ALLOWED = "allowed"
SCOPE_FOREGROUND = "foreground"
COMMENTARY_OBSERVER = "observer"

#: Windows' own surfaces: never "what the user is doing", often personal (the lock
#: and sign-in screens, Start and search, notification and quick-settings flyouts,
#: the touch keyboard and emoji panel, Task Manager, Settings, UAC and credential
#: prompts, file pickers).  Explorer is handled apart (see :data:`SHELL_TITLES`).
SYSTEM_PROCESSES = frozenset({
    "lockapp.exe", "logonui.exe", "winlogon.exe", "consent.exe", "credentialuibroker.exe",
    "useraccountcontrolsettings.exe", "startmenuexperiencehost.exe", "searchhost.exe", "searchapp.exe",
    "searchui.exe", "shellexperiencehost.exe", "shellhost.exe", "textinputhost.exe",
    "screenclippinghost.exe", "pickerhost.exe", "taskmgr.exe", "systemsettings.exe", "control.exe",
    "mmc.exe", "smartscreen.exe", "sihost.exe", "dwm.exe", "rundll32.exe", "securityhealthhost.exe",
    "applicationframehost.exe",
    # Sign-in windows (Microsoft / work account, Windows Hello): e-mail and one-time codes.
    "microsoft.aad.brokerplugin.exe", "wwahost.exe", "useroobebroker.exe",
    "credentialenrollmentmanager.exe",
})
#: explorer.exe is the desktop ("Program Manager"), the taskbar (no title), Alt+Tab
#: and File Explorer.  Only a File Explorer window - recognisable by its title, so
#: only while titles may be read - is something she may look at.
EXPLORER = "explorer.exe"
SHELL_TITLES = frozenset({"", "program manager", "task switching", "task view", "перемикання завдань",
                          "представлення завдань"})

# Why the window in front is not looked at (internal codes).
SKIP_OWN = "own"  # one of hers: says nothing about the user's window, keeps the last reason
SKIP_PRIVATE = "private"
SKIP_SYSTEM = "system"
SKIP_BROWSER = "browser"
SKIP_NOT_LISTED = "not_listed"
SKIP_NOT_GAME = "not_game"
SKIP_UNKNOWN = "unknown"
SKIP_AWAY = "away"
SKIP_MUTED = "muted"  # an observer who could not say anything: no automatic frames

#: Status line ("Чекає: ...") for each reason.
SKIP_NOTES: dict[str, str] = {
    SKIP_PRIVATE: "вікно на передньому плані приватне — на нього вона не дивиться",
    SKIP_SYSTEM: "на передньому плані системне вікно (робочий стіл, пошук, блокування, диспетчер "
                 "завдань чи запит безпеки) — його вона не знімає",
    SKIP_BROWSER: "на передньому плані браузер, а без дозволу «Читати заголовки вікон» не відрізнити "
                  "приватну вкладку, банкінг чи пошту, тож браузер вона пропускає",
    SKIP_NOT_LISTED: "програми на передньому плані немає в списку дозволених",
    SKIP_NOT_GAME: "в ігровому режимі вона дивиться лише на ігри",
    SKIP_UNKNOWN: "не вдалося визначити, що за програма на передньому плані",
    SKIP_AWAY: "тебе немає за комп'ютером, тож вона не дивиться й не витрачає ресурси",
    SKIP_MUTED: "сама вона зараз нічого не скаже, тож кадрів не знімає (причина — нижче)",
}
#: Her answer to "подивись на екран" when the window in front is skipped ("foreground" scope).
LOOK_SKIP_MESSAGES: dict[str, str] = {
    SKIP_PRIVATE: "Це вікно приватне — на нього я не дивлюся.",
    SKIP_SYSTEM: "Це системне вікно Windows — його я не знімаю.",
    SKIP_BROWSER: "Браузер я не знімаю без дозволу «Читати заголовки вікон»: інакше не відрізню "
                  "приватну вкладку, банкінг чи пошту.",
    SKIP_UNKNOWN: "Не розумію, що це за вікно, тож краще не дивитимуся.",
}
#: Published observations remembered for :meth:`VisionIntegration.details_for`.
DETAILS_LIMIT = 16
#: Talking to her by voice counts as being at the PC for this long (hands-free: no
#: keyboard or mouse input; the same as ``chibi.personality.observer``).
VOICE_PRESENCE_SECONDS = 300.0
#: Whether the observer can speak is re-checked at most this often (it reads the budget).
OBSERVER_CHECK_SECONDS = 5.0

#: Why she says nothing about the screen herself as an observer (VisionStatusView.observer_note).
OBSERVER_MANUAL = ("Сама вона не коментує: у режимі «Лише за запитом» вона лише відповідає, "
                   "коли попросиш подивитися.")
OBSERVER_NO_AI_LINES = ("Сама вона мовчить і кадрів не знімає: на сторінці «AI» вимкнено «Формулювати "
                        "через AI її власні репліки».")
OBSERVER_AI_OFF = "Сама вона мовчить і кадрів не знімає, поки AI недоступний ({reason})."
OBSERVER_BUDGET = ("Сама вона мовчить і кадрів не знімає: ліміт запитів до AI на цю годину "
                   "вичерпано.")


def observer_ai_reason(settings: Settings, status: tuple[bool, str], initiative_available: bool) -> str:
    """Why the observer cannot ask the model now ("" if it can): the app passes the AI
    service's ``status()`` and ``is_available(Purpose.INITIATIVE)``."""
    if not settings.ai.use_for_reactions:
        return OBSERVER_NO_AI_LINES
    online, reason = status
    if not online:
        return OBSERVER_AI_OFF.format(reason=reason or "немає з'єднання")
    if not initiative_available:
        return OBSERVER_BUDGET
    return ""


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
    app_title: str  # the app being watched ("" if none)
    #: Why vision cannot run now (Ukrainian), "" if it can (or the mode is off).
    blocked: str
    #: The analysing model runs on this PC; None when no model is configured.
    local: bool | None
    #: Extra hint for the page ("" if none).
    note: str = ""
    scope: str = SCOPE_ALLOWED  # VisionSettings.scope
    commentary: str = "events"  # VisionSettings.commentary
    #: Why the window in front is not being looked at (a Ukrainian clause for
    #: "Чекає: ..."), "" if it is, or if there is nothing to explain.
    skipped: str = ""
    #: Commentary "observer": why she says nothing herself now (a Ukrainian sentence),
    #: "" when she can comment.
    observer_note: str = ""


@dataclass(frozen=True, slots=True)
class ObservationDetails:
    """What a published :class:`~chibi.core.events.VisionObserved` does not carry,
    for the observer (:meth:`VisionIntegration.details_for`)."""

    #: The model thinks a friend watching over the shoulder would say something now.
    interesting: bool
    #: Commenting risks a spoiler (the event's description is already empty then,
    #: unless the user allows spoilers).
    spoiler: bool
    #: The window is a game the games database knows (or the user listed as one).
    is_game: bool
    #: A work window or a work event (code, terminal, document, error, build...).
    work: bool
    #: The frame answers the user's own "подивись на екран": she answers it herself,
    #: the observer must not add a second line.
    solicited: bool


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
    scope: str = SCOPE_ALLOWED
    #: Known apps: which ones are browsers, and friendly names ("VS Code").
    catalog: AppCatalog = dataclasses.field(default_factory=AppCatalog)


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
        observer_ai: Callable[[], str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        """*observer_ai* says why the observer cannot ask the model now ("" if it can;
        see :func:`observer_ai_reason`); without it the observer is assumed able to."""
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
        self._observer_ai = observer_ai
        self._observer_checked: tuple[float, str] | None = None
        self._closed = False

        self._invoker = QtInvoker(self)
        self._analyzer: ManagedAnalyzer | None = None
        self._analyzer_key: tuple[object, ...] | None = None
        self._allow_spoilers = False
        self._min_confidence = 0.6
        #: "Лише локальний аналіз" (a snapshot: read by the worker thread too).
        self._local_only = True
        self._high_load = False
        #: The activity monitor said the user is idle (no UserReturned yet).
        self._user_away = False
        #: When the user last talked to her by voice (monotonic): presence too.
        self._voice_at: float | None = None
        self._rules = self._build_rules(self._read_settings())
        self._excluded: set[int] = set()
        #: Why the last window in front (not one of hers) was not looked at ("" = it was).
        #: Written by the resolver on the main and the worker thread; display only.
        self._skip = ""

        #: (what she saw, process it came from); the process is kept only to forget
        #: an app the moment it becomes private, it is never shown.
        self._seen: deque[tuple[SeenObservation, str]] = deque(maxlen=RECENT_LIMIT)
        self._latest: tuple[Observation, float, str] | None = None
        self._counts: dict[str, Counter[str]] = {}
        self._count_process: dict[str, str] = {}
        self._last_counted: dict[tuple[str, str], float] = {}
        self._manual_until: float | None = None
        #: A user's request waiting for its confirmation frame: (first frame, window is a game).
        self._manual_first: tuple[Observation, bool] | None = None
        self._in_request = False
        self._noticed: dict[str, float] = {}
        self._details: deque[tuple[VisionObserved, ObservationDetails]] = deque(maxlen=DETAILS_LIMIT)

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
            still_allowed=self._result_allowed,
            cancel=self._cancel_analysis,
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
            bus.subscribe(UserIdle, self._on_user_idle),
            bus.subscribe(UserReturned, self._on_user_returned),
            bus.subscribe(VoiceTranscript, self._on_voice),
            bus.subscribe(ConversationActive, self._on_conversation),
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
        if s.vision.scope != self._rules.scope:
            self._skip = ""  # a reason from the other scope would mislead
        self._rules = self._build_rules(s)
        self._local_only = s.vision.local_only
        self._forget_private()
        self._drop_disallowed_target()
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
            # The user's cap on unsolicited lines applies to what she sees too - the
            # shared budget as the reaction engine computes it, not the raw slider.
            reactions_per_hour=min(reactions_per_hour(s.gaming.reaction_frequency), budget_limit(s)),
            allow_spoilers=s.gaming.hints_and_spoilers,
        )
        self._service.high_load = self._high_load and s.vision.adapt_to_load
        self._sync_pace(s)
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
        if not self._may_look_at(info):
            return False
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
        self._details.clear()
        self._service.recent.clear()
        self._counts.clear()
        self._count_process.clear()
        self._last_counted.clear()
        log.info("vision history cleared by the user")

    def latest_observation(self) -> str | None:
        """``"<scene>/<event>: <description>"`` of the newest observation younger than
        two minutes, for the AI brain's context (sanitised, untrusted text).  None
        when the current AI provider may not receive screen content: a description
        seen by a local model never reaches a cloud one without ``CLOUD_VISION``, nor
        while "Лише локальний аналіз" is on."""
        latest = self._latest
        if latest is None or not self._screen_data_may_leave():
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

    def details_for(self, event: VisionObserved) -> ObservationDetails | None:
        """More about a :class:`VisionObserved` this integration published (the same
        object, as the bus delivers it), or None if it is not one of the last few."""
        for published, details in reversed(self._details):
            if published is event:
                return details
        return None

    def screen_data_may_leave(self) -> bool:
        """Whether what she saw (a description, a scene) may go to the AI provider the
        app is configured with *now*: always for a local model; for a cloud model only
        with ``CLOUD_VISION`` and without "Лише локальний аналіз" (main thread)."""
        return self._screen_data_may_leave()

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
        if s.vision.mode != VisionMode.OFF and s.vision.scope == SCOPE_ALLOWED \
                and not (s.vision.allowed_games or s.vision.allowed_processes):
            notes.append(NOTE_NOTHING_ALLOWED)
        local = None if s.ai.provider == "none" else not is_cloud(s.ai)
        blocked = self._blocker(s)
        skipped = ""
        if s.vision.mode in (VisionMode.AUTO, VisionMode.GAME) and not blocked and state == VisionState.IDLE:
            skip = SKIP_AWAY if self._service.away else SKIP_MUTED if self._service.muted else self._skip
            skipped = SKIP_NOTES.get(skip, "")
        observer_note = ""
        if s.vision.mode != VisionMode.OFF and s.vision.commentary == COMMENTARY_OBSERVER and not blocked:
            observer_note = OBSERVER_MANUAL if s.vision.mode == VisionMode.MANUAL else self._observer_reason(
                fresh=True)
        return VisionStatusView(
            mode=s.vision.mode, state=state.value,
            app_title=target.title if target is not None and watching else "",
            blocked=blocked, local=local, note=" ".join(notes),
            scope=s.vision.scope, commentary=s.vision.commentary, skipped=skipped,
            observer_note=observer_note,
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
            self._rules = dataclasses.replace(self._rules, own_windows=frozenset(ids))

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
        own filter (GAME mode) and is applied by the service; here it only
        explains the skip."""
        try:
            target, skip = self._resolve_with(self._rules, info, self._titles_allowed())
        except Exception:
            log.exception("vision target lookup failed")
            return None
        if skip != SKIP_OWN:  # her own window says nothing about the user's
            if target is not None and require_game and not target.is_game:
                skip = SKIP_NOT_GAME
            self._skip = skip
        return target

    @classmethod
    def _resolve_with(cls, rules: _Rules, info: ForegroundInfo,
                      titles: bool) -> tuple[Target | None, str]:
        """(target, "") or (None, why not).  *titles*: window titles may be read now."""
        if info is None or not info.hwnd or cls._is_own(info, rules):
            return None, SKIP_OWN
        if rules.privacy.is_private(info):
            return None, SKIP_PRIVATE
        name = normalize_process(info.process_name)
        if not name:
            return None, SKIP_UNKNOWN
        if rules.scope == SCOPE_FOREGROUND:
            return cls._any_window(rules, info, name, titles)
        target = cls._listed(rules, info, name)
        return (target, "") if target is not None else (None, SKIP_NOT_LISTED)

    @staticmethod
    def _listed(rules: _Rules, info: ForegroundInfo, name: str) -> Target | None:
        """The "allowed" scope: games and apps from the user's lists."""
        match = rules.games.match(name, info.exe_path)
        if match is not None and match.game_id in rules.allowed_games:
            return Target(info.hwnd, name, match.title, True)
        if f"exe:{name}" in rules.allowed_games:
            return Target(info.hwnd, name, match.title if match is not None else title_from_exe(name), True)
        if name in rules.allowed_processes:
            title = match.title if match is not None else title_from_exe(name)
            return Target(info.hwnd, name, title, match is not None)
        return None

    @classmethod
    def _any_window(cls, rules: _Rules, info: ForegroundInfo, name: str,
                    titles: bool) -> tuple[Target | None, str]:
        """The "foreground" scope: any (non-private) window but Windows' own surfaces,
        and a browser only while its title can be checked by the privacy filter."""
        skip = cls._foreground_skip(rules, name, info.title if titles else None, titles)
        if skip:
            return None, skip
        match = cls._game_match(rules, name, info.exe_path)
        known = rules.catalog.lookup(name)
        if match is not None:
            title = match.title
        elif known is not None:
            title = known.display_name
        else:
            title = title_from_exe(name)
        return Target(info.hwnd, name, title, match is not None or f"exe:{name}" in rules.allowed_games), ""

    @staticmethod
    def _foreground_skip(rules: _Rules, name: str, title: str | None, titles: bool) -> str:
        """Why a (non-private) window may not be looked at in the "foreground" scope;
        "" if it may.  *title* is None when it was not (or may not be) read."""
        if name in SYSTEM_PROCESSES:
            return SKIP_SYSTEM
        if name == EXPLORER:
            if not titles or title is None or collapse_spaces(title).casefold() in SHELL_TITLES:
                return SKIP_SYSTEM
            return ""
        known = rules.catalog.lookup(name)
        browser = known is not None and known.category is ActivityCategory.BROWSING
        if browser and (not titles or title is None):
            return SKIP_BROWSER
        return ""

    @staticmethod
    def _game_match(rules: _Rules, name: str, exe_path: str | None) -> GameMatch | None:
        """A game for the process; a folder/exe-name guess never makes a known app a
        game (Blender installed through Steam is not one)."""
        match = rules.games.match(name, exe_path)
        if match is not None and match.source == "heuristic" and rules.catalog.lookup(name) is not None:
            return None
        return match

    @classmethod
    def _still_allowed(cls, target: Target, rules: _Rules, titles: bool) -> bool:
        """Would *target* still resolve under *rules*?  Same decision as
        :meth:`_resolve_with` from what a target keeps (its process name): the exe
        path only matters for heuristic matches, whose id is ``exe:<process>``;
        a window title is never kept, so a window that needs one (a browser,
        File Explorer) stays allowed only while titles may be read."""
        name = target.process_name
        if rules.privacy.is_private_process(name):
            return False
        if rules.scope == SCOPE_FOREGROUND:
            if name == EXPLORER:
                return titles
            return not cls._foreground_skip(rules, name, "" if titles else None, titles)
        if name in rules.allowed_processes or f"exe:{name}" in rules.allowed_games:
            return True
        match = rules.games.match(name)
        return match is not None and match.game_id in rules.allowed_games

    @staticmethod
    def _is_own(info: ForegroundInfo, rules: _Rules) -> bool:
        return info.pid == rules.own_pid or (bool(info.hwnd) and info.hwnd in rules.own_windows)

    def _capture_allowed(self) -> bool:
        return bool(self._permissions.is_granted(Permission.SCREEN_CAPTURE))

    def _titles_allowed(self) -> bool:
        """Window titles may be read (any thread)."""
        return bool(self._permissions.is_granted(Permission.WINDOW_TITLES))

    def _result_allowed(self, target: Target) -> bool:
        """Main thread, right before an analysed frame becomes something she knows,
        says or publishes: capture still granted, a cloud model's answer only while
        cloud vision still is, and the window's app still one she may see."""
        if self._closed or not self._capture_allowed():
            return False
        analyzer = self._analyzer
        if analyzer is not None and not analyzer.is_local and not self._cloud_allowed():
            return False
        return self._still_allowed(target, self._rules, self._titles_allowed())

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
        if self._closed or self._analyzer is None:
            return False
        return self._screen_data_may_leave()

    def _screen_data_may_leave(self) -> bool:
        """Something taken from the screen (a frame, or what the model said about one)
        may go to the AI provider the app is configured with *now* (main thread)."""
        if not self._capture_allowed():
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
        """An analysed frame, already re-checked by the service (``_result_allowed``)."""
        observation = self._clean(observation)
        now = self._clock.monotonic()
        react = decision.action == "react"
        # While a job runs the service keeps its target, so this is the frame's window.
        target = self._service.target
        process = target.process_name if target is not None else ""
        is_game = target.is_game if target is not None else False
        manual = self._manual_until is not None and now <= self._manual_until
        # As an observer she never comments here on her own: the observer decides
        # from VisionObserved.  What the user asked for is answered either way.
        speak = react and (manual or not self._observer_mode())
        self._latest = (observation, now, process)
        self._seen.append((SeenObservation(
            app_title=observation.app_title, scene=observation.scene, event=observation.event,
            confidence=observation.confidence, at=self._clock.now(), reacted=speak,
        ), process))
        self._count(observation, now, process)
        self._publish(observation, react=react, is_game=is_game, solicited=manual)
        if manual and decision.action == "confirm":
            # The service takes a second frame in a few seconds; answer then.
            self._manual_first = (observation, is_game)
            return
        variables = self._title_variables(observation, is_game)
        if speak:
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
            # An event that still needed its second frame (it never came, or showed
            # something else) is not claimed as certain.
            confirmed = observation.confirmed or not observation.needs_confirmation
            self._composer.compose(*self._answer_keys(observation, is_game, confirmed=confirmed),
                                   solicited=True, **variables)

    @staticmethod
    def _clean(observation: Observation) -> Observation:
        """The analysis already did this; a different analyzer must not undo it.  The
        description is sanitised (no links, e-mails, numbers...) and correspondence
        (chat, e-mail) is never described nor "interesting"."""
        if observation.scene in PRIVATE_SCENES:
            return dataclasses.replace(observation, description="", interesting=False)
        return dataclasses.replace(observation, description=sanitize_description(observation.description))

    def _publish(self, observation: Observation, *, react: bool, is_game: bool, solicited: bool) -> None:
        """Publish :class:`VisionObserved` for one analysed (and cleaned) frame.

        Only while what she saw may reach the AI provider configured now (the
        observer and the brain hand it to that provider).  A spoiler's
        description stays empty unless the user allows spoilers."""
        if not self._screen_data_may_leave():
            return
        hide = observation.spoiler and not self._allow_spoilers
        event = VisionObserved(
            app_title=observation.app_title, scene=observation.scene, event=observation.event,
            confidence=observation.confidence, description="" if hide else observation.description,
            emotion=observation.suggested_emotion, confirmed=observation.confirmed, react=react,
        )
        self._details.append((event, ObservationDetails(
            interesting=observation.interesting, spoiler=observation.spoiler, is_game=is_game,
            work=observation.work, solicited=solicited,
        )))
        self._bus.publish(event)

    def _on_error(self, message: str) -> None:
        user = self._in_request or (self._manual_until is not None
                                    and self._clock.monotonic() <= self._manual_until)
        if user:
            self._end_manual()
        self._notify(message, user=user)

    # ------------------------------------------------------------------ bus handlers

    def _on_settings_changed(self, event: SettingsChanged) -> None:
        if event.section in ("", "vision", "gaming", "ai", "activity", "behavior", "personality"):
            self.reconfigure()

    def _on_permission_changed(self, event: PermissionChanged) -> None:
        if event.permission in (Permission.WINDOW_TITLES, Permission.APP_METADATA):
            if not event.granted:
                if event.permission == Permission.APP_METADATA:
                    self._user_away = False  # the monitor stops: no UserReturned will come
                # A browser (or File Explorer) being watched is no longer one she may see.
                self.reconfigure()
            return
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
        self._user_away = False
        self.stop()
        self.reconfigure()

    def _on_user_idle(self, _event: UserIdle) -> None:
        self._user_away = True
        self._sync_pace(self._read_settings())

    def _on_user_returned(self, _event: UserReturned) -> None:
        self._user_away = False
        self._sync_pace(self._read_settings())

    def _on_voice(self, _event: VoiceTranscript) -> None:
        """The user talks to her (hands-free: no keyboard, no mouse): they are here."""
        self._voice_at = self._clock.monotonic()
        self._sync_away()

    def _on_conversation(self, event: ConversationActive) -> None:
        """The user started or stopped talking with her: no automatic frame may delay her
        answer (see :meth:`VisionService.set_conversation`)."""
        if self._closed:
            return
        self._service.set_conversation(bool(event.active))

    def _cancel_analysis(self) -> None:
        """Withdraw the analysis request in flight (the service abandoned its frame)."""
        analyzer = self._analyzer
        if analyzer is not None:
            analyzer.cancel_current()

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
            scope=s.vision.scope,
            catalog=AppCatalog.load(),
        )

    def _observer_mode(self) -> bool:
        return self._read_settings().vision.commentary == COMMENTARY_OBSERVER

    def _sync_pace(self, s: Settings) -> None:
        """Any window in AUTO mode: pace like GAME mode, and no frames while the user
        is away.  The "allowed" scope keeps its fixed pace."""
        self._service.paced = s.vision.scope == SCOPE_FOREGROUND and s.vision.mode == VisionMode.AUTO
        self._sync_away()

    def _sync_away(self) -> None:
        """Away: idle input and no voice lately (re-checked every tick: the voice expires)."""
        voice = self._voice_at
        talking = voice is not None and self._clock.monotonic() - voice <= VOICE_PRESENCE_SECONDS
        self._service.away = self._service.paced and self._user_away and not talking

    def _sync_muted(self, s: Settings) -> None:
        """As an observer who could not say anything about them, take no automatic frames."""
        self._service.muted = (s.vision.commentary == COMMENTARY_OBSERVER
                               and s.vision.mode in (VisionMode.AUTO, VisionMode.GAME)
                               and bool(self._observer_reason()))

    def _observer_reason(self, *, fresh: bool = False) -> str:
        """Why the observer cannot ask the model now ("" if it can), checked at most every
        :data:`OBSERVER_CHECK_SECONDS` (it reads the budget) unless *fresh*."""
        if self._observer_ai is None:
            return ""
        now = self._clock.monotonic()
        checked = self._observer_checked
        if not fresh and checked is not None and now - checked[0] < OBSERVER_CHECK_SECONDS:
            return checked[1]
        try:
            reason = str(self._observer_ai() or "")
        except Exception:
            log.exception("checking whether the observer can speak failed")
            reason = ""
        self._observer_checked = (now, reason)
        return reason

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
        self._sync_away()
        self._sync_muted(self._read_settings())
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
        if self._may_look_at(info):
            self._request_now()

    def _may_look_at(self, info: ForegroundInfo | None) -> bool:
        """"Подивись на екран" with the "foreground" scope: say exactly why the window
        in front is skipped.  (The "allowed" scope keeps the service's own answer.)"""
        if info is None or self._rules.scope != SCOPE_FOREGROUND:
            return True
        try:
            target, skip = self._resolve_with(self._rules, info, self._titles_allowed())
        except Exception:
            log.exception("vision target lookup failed")
            target, skip = None, SKIP_UNKNOWN
        if target is not None or skip not in LOOK_SKIP_MESSAGES:
            return True
        self._skip = skip
        self._notify(LOOK_SKIP_MESSAGES[skip], user=True)
        return False

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
        observation, is_game = first
        self._composer.compose(*self._answer_keys(observation, is_game, confirmed=False), solicited=True,
                               **self._title_variables(observation, is_game))

    def _end_manual(self) -> None:
        self._manual_until = None
        self._manual_first = None

    @staticmethod
    def _title_variables(observation: Observation, is_game: bool) -> dict[str, str]:
        """``{game}`` for a game, ``{app}`` for any other window ("Бачу гру «VS Code»"
        would be a lie)."""
        if not observation.app_title:
            return {}
        title = speakable_title(observation.app_title)
        return {"game": title} if is_game else {"app": title}

    def _answer_keys(self, observation: Observation, is_game: bool = True, *,
                     confirmed: bool = True) -> tuple[str, ...]:
        """Phrase keys for answering "що ти бачиш?" (the user asked, so she always says something)."""
        keys: list[str] = []
        sure = observation.confidence >= self._min_confidence and confirmed
        spoiler = observation.spoiler and not self._allow_spoilers
        if observation.notable and sure and not spoiler:
            keys.append(f"vision.{observation.event}")
        if observation.event in ("combat", "puzzle") and not spoiler:
            keys.append(f"vision.look.{observation.event}")
        if observation.event in WORK_EVENTS and sure:
            keys.append(f"vision.look.{observation.event}")
        keys.append(f"vision.look.{observation.scene}")
        if observation.scene in WORK_SCENES or (not is_game and observation.scene not in GAME_SCENES):
            keys.append("vision.look.work")  # "граємо далі" would not fit a document
        keys.append("vision.look")
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
        self._details.clear()
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
            self._details.clear()
            self._service.recent.clear()  # the model's "previous scene" context
            log.info("forgot what she saw of an app that became private")

    def _drop_disallowed_target(self) -> None:
        """The app being watched became private, left the allowed lists, or is no longer
        one she may see (the scope changed; window titles were revoked while a browser
        was in front): abandon its frame in flight - a request still queued for the AI
        never leaves, and an answer never becomes something she knows or says (the
        service ignores late results)."""
        target = self._service.target
        try:
            allowed = target is None or self._still_allowed(target, self._rules, self._titles_allowed())
        except Exception:
            log.exception("vision target re-check failed")
            allowed = False
        if allowed:
            return
        if self._analyzer is not None:
            self._analyzer.cancel_current()
        self._service.stop()

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

