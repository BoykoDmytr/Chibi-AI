"""The character controller: state machine + animation + movement + behavior
wired to the window, the speech bubble and the event bus.

One frame (:meth:`CharacterController.step`):

1. expire finished actions (state machine), show the next queued line;
2. move (walk / fall), finish arrived walks, save the position after landing;
3. poll the cursor (~4 Hz, only while she is visible and awake);
4. once a second ask :class:`AutonomousBehavior` what to do and apply it
   through the state machine, so priorities still arbitrate;
5. advance the clip, move the window, repaint *only* if something visual
   changed, keep the bubble on her head, retune the frame timer.

Frame rate adapts: 30 fps while something moves fast (walking, falling,
dragged, hop/shake/swing effects), 15 fps for calm states, 8 fps asleep,
halved in eco mode and *stopped* while she is hidden - then a 1 s coarse
timer keeps the state machine and behavior ticking at no visible cost.

Speech: a line gets ``clamp(2.2 + 0.055 * len, 3, 12)`` seconds.  Lines
the user asked for (``solicited``) replace whatever is showing; unsolicited
lines wait in a short queue (2, oldest dropped) and are dropped entirely
while she is hidden - and when the user asks for quiet or turns on Do Not
Disturb (they were chosen before that).

Activity events (games, user idle/returned) only change her while app
observation is permitted: one queued just before the kill switch is
dropped, and the kill switch or a revoke forgets running games (the monitor
sends no ``GameEnded`` then) and the queued lines.

Positions are global logical px; the window's bottom-centre region holds
her feet anchor (a few px of margin below for art that reaches under the
feet).  Her position is saved relative to the monitor's work area so it
survives resolution and taskbar changes, and restored only if that place
still exists.
"""

from __future__ import annotations

import contextlib
import logging
import math
import random
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from PySide6.QtCore import QObject, QPoint, QPointF, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QGuiApplication, QPainter, QPixmap, QScreen

from chibi.character.animation import CharacterPack, Clip, FramePlayer
from chibi.character.behavior import AutonomousBehavior, BehaviorContext, Decision
from chibi.character.movement import Movement, Rect
from chibi.character.state_machine import (
    DEFAULT_ACTION_SECONDS,
    FALLBACK_ACTION_SECONDS,
    CharacterStateMachine,
    StateChange,
)
from chibi.character.states import POSTURES, CharacterState, Facing, StatePriority
from chibi.core.clock import Clock
from chibi.core.config import Settings
from chibi.core.event_bus import EventBus
from chibi.core.events import (
    CharacterInteraction,
    DoNotDisturbChanged,
    ForegroundAppChanged,
    FullscreenChanged,
    GameEnded,
    GameStarted,
    InteractionKind,
    InteractionOutcome,
    ObservationStopped,
    Permission,
    PermissionChanged,
    QuietModeChanged,
    SettingsChanged,
    SpeechRequested,
    SystemLoadChanged,
    UserIdle,
    UserReturned,
)
from chibi.desktop import overlays
from chibi.desktop.bubble import SpeechBubble
from chibi.desktop.character_window import CharacterWindow
from chibi.desktop.renderers import (
    CharacterRenderer,
    SpriteRenderer,
    VectorRenderer,
    VisualState,
    vector_clip_for,
)
from chibi.memory.repositories import StateRepository

log = logging.getLogger(__name__)

S = CharacterState
P = StatePriority

#: Logical px tall at scale 1.0.
BASE_HEIGHT = 220.0
#: px/s at scale 1.0.
WALK_SPEED = 70.0
RUN_SPEED = 170.0
#: A "come here" farther than this (px at scale 1.0) is run, not walked.
RUN_DISTANCE = 500.0
#: Closer than this she just waves instead of walking over.
CALL_NEAR = 40.0
#: Overlay size (about a head diameter) as a fraction of her height.
OVERLAY_SIZE = 0.42
#: Half her body width for the walkable range (hair may poke past the edge).
BODY_HALF_WIDTH = 0.3
#: Extra transparent margin around the drawing (rounding, antialiasing).
WINDOW_PAD = 4

#: Walking, falling, dragging, hops: anything below ~30 fps looks choppy.
FPS_FAST = 30
#: Idle breathing / talking move a fraction of a pixel per frame; 10 fps is
#: visually identical to 15 and costs a third less CPU (measured ~3 % -> ~2 %
#: of one core for an app that runs all day).
FPS_CALM = 10
FPS_SLEEP = 5
HIDDEN_TICK_MS = 1000
#: Effects that move a lot per frame and look choppy below 30 fps.
FAST_FX = frozenset({"hop", "shake", "swing"})

DECIDE_INTERVAL = 1.0
CURSOR_POLL_INTERVAL = 0.25
#: Cursor within this many of her heights (horizontally and vertically) is "near".
CURSOR_NEAR = 0.9

SPEECH_BASE_SECONDS = 2.2
SPEECH_SECONDS_PER_CHAR = 0.055
SPEECH_MIN_SECONDS = 3.0
SPEECH_MAX_SECONDS = 12.0
SPEECH_QUEUE_MAX = 2
#: Cap for an explicit ``say(seconds=...)``: a stuck bubble would also block the queue.
SPEECH_MAX_EXPLICIT_SECONDS = 60.0

#: Priority holds for posture requests (see the state machine): long enough
#: to beat ambient behavior for a moment, short enough not to freeze it.
WAKE_HOLD = 3.0
SLEEP_HOLD = 5.0
GAMING_HOLD = 3.0
#: Hold for the IDLE that ends a walk: just long enough to be a valid duration.
POSTURE_SETTLE_HOLD = 0.1

POSITION_KEY = "window_position"
#: Saving after every short stroll would be pointless disk churn.
POSITION_SAVE_INTERVAL = 30.0

_MOVING = frozenset({S.WALK, S.RUN})
_DEFAULT_NEEDS = (30.0, 80.0, 70.0, 20.0)  # hunger, energy, happiness, boredom


def speech_seconds(text: str) -> float:
    """How long a line stays: longer lines longer, within 3..12 s."""
    seconds = SPEECH_BASE_SECONDS + SPEECH_SECONDS_PER_CHAR * len(text)
    return min(max(seconds, SPEECH_MIN_SECONDS), SPEECH_MAX_SECONDS)


# -- screens and positions (pure logic, tested without real monitors) --------------------------


@dataclass(frozen=True)
class ScreenArea:
    """A monitor as the controller sees it: its work area in global logical px."""

    name: str
    area: Rect
    primary: bool = False
    dpr: float = 1.0

    def qrect(self) -> QRect:
        a = self.area
        return QRect(round(a.left), round(a.top), max(1, round(a.width)), max(1, round(a.height)))


def qt_screens() -> list[ScreenArea]:
    """The current monitors from Qt (work areas exclude the taskbar)."""
    if QGuiApplication.instance() is None:
        return []
    primary = QGuiApplication.primaryScreen()
    out: list[ScreenArea] = []
    for screen in QGuiApplication.screens():
        g = screen.availableGeometry()
        if g.width() <= 0 or g.height() <= 0:
            continue
        is_primary = (primary is not None and screen.name() == primary.name()
                      and screen.geometry() == primary.geometry())
        out.append(ScreenArea(screen.name(), Rect(g.x(), g.y(), g.x() + g.width(), g.y() + g.height()),
                              is_primary, float(screen.devicePixelRatio())))
    return out


def _distance_to(area: Rect, x: float, y: float) -> float:
    dx = max(area.left - x, 0.0, x - area.right)
    dy = max(area.top - y, 0.0, y - area.bottom)
    return math.hypot(dx, dy)


def screen_for_point(screens: list[ScreenArea], x: float, y: float) -> ScreenArea | None:
    """The screen containing (x, y), else the nearest one; None if there are no screens."""
    if not screens:
        return None
    for screen in screens:
        if screen.area.contains(x, y):
            return screen
    return min(screens, key=lambda s: _distance_to(s.area, x, y))


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def encode_position(x: float, y: float, screen: ScreenArea) -> dict[str, Any]:
    """Position record for ``StateRepository`` (relative + absolute)."""
    a = screen.area
    rx = (x - a.left) / a.width if a.width > 0 else 0.5
    return {
        "screen": screen.name,
        "rx": round(min(max(rx, 0.0), 1.0), 6),
        "y_from_bottom": round(max(0.0, a.bottom - y), 2),
        "x": round(x, 2),
        "y": round(y, 2),
    }


def decode_position(data: Any, screens: list[ScreenArea]) -> tuple[float, float, ScreenArea] | None:
    """Where a saved position lands on the *current* screens, or None if it
    no longer exists (monitor unplugged, record corrupt).

    The relative form (fraction of the named screen's width, distance above
    its work-area bottom) wins: it survives resolution and taskbar changes.
    The absolute coordinates are only used if they are on some screen now.
    """
    if not isinstance(data, Mapping) or not screens:
        return None
    name = data.get("screen")
    rx = _finite(data.get("rx"))
    y_from_bottom = _finite(data.get("y_from_bottom"))
    if isinstance(name, str) and rx is not None and y_from_bottom is not None and 0.0 <= rx <= 1.0:
        for screen in screens:
            if screen.name == name:
                a = screen.area
                y = a.bottom - min(max(y_from_bottom, 0.0), a.height)
                return a.left + rx * a.width, y, screen
    x, y = _finite(data.get("x")), _finite(data.get("y"))
    if x is not None and y is not None:
        for screen in screens:
            a = screen.area
            # feet exactly on the work-area bottom are on the screen too
            if a.left <= x < a.right and a.top <= y <= a.bottom:
                return x, y, screen
    return None


def default_position(screens: list[ScreenArea], half_width: float) -> tuple[float, float, ScreenArea]:
    """Bottom-right of the primary screen, standing on the taskbar edge."""
    if not screens:
        fallback = ScreenArea("", Rect(0.0, 0.0, 1280.0, 720.0), True)
        screens = [fallback]
    screen = next((s for s in screens if s.primary), screens[0])
    a = screen.area
    x = a.right - max(a.width * 0.1, 2.0 * half_width)
    x = min(max(x, a.left + half_width), a.right - half_width) if a.width > 2 * half_width else a.center_x
    return x, a.bottom, screen


class PermissionChecker(Protocol):
    def is_granted(self, permission: Permission) -> bool: ...


@dataclass(frozen=True)
class _Line:
    text: str
    state: CharacterState | None
    portrait: str
    seconds: float


# -- the controller ----------------------------------------------------------------------------


class CharacterController(QObject):
    context_menu_requested = Signal(QPoint)
    #: Emitted with the new ``CharacterState`` value on every state change.
    state_changed = Signal(str)

    def __init__(
        self,
        bus: EventBus,
        settings: Callable[[], Settings],
        clock: Clock,
        state_repo: StateRepository,
        pack: CharacterPack | None,
        needs: Callable[[], tuple[float, float, float, float]],
        rng: random.Random | None = None,
        parent: QObject | None = None,
        *,
        cursor_pos: Callable[[], QPointF] | None = None,
        screens: Callable[[], list[ScreenArea]] | None = None,
        permissions: PermissionChecker | None = None,
    ) -> None:
        """*permissions* (optional) lets her ignore activity events dispatched
        after ``APP_METADATA`` was revoked (queued just before the kill switch):
        those must not change her any more."""
        super().__init__(parent)
        self._bus = bus
        self._settings_fn = settings
        self._clock = clock
        self._repo = state_repo
        self._pack = pack
        self._needs_fn = needs
        self._rng = rng or random.Random()
        self._cursor_fn = cursor_pos or (lambda: QPointF(QCursor.pos()))
        self._screens_fn = screens or qt_screens
        self._permissions = permissions
        self._permission_check_failed = False
        self._cfg = Settings()
        self._cfg = self._read_settings()

        self._sm = CharacterStateMachine(clock)
        self._sm.add_listener(self._on_state_change)
        self._behavior = AutonomousBehavior(self._rng)
        self._vector = VectorRenderer()
        self._use_vector = self._wants_vector(self._cfg)
        self._renderer: CharacterRenderer = self._make_renderer()
        self._player = FramePlayer(self._clip_for(S.IDLE))
        self._variant_active = False
        self._change_count = 0
        self._facing = Facing.FRONT
        self._anim_t = 0.0

        self._scale = self._cfg.character.scale
        self._height = BASE_HEIGHT * self._scale
        self._half_width = BODY_HALF_WIDTH * self._height
        self._anchor = QPointF(0.0, 0.0)
        self._window_size = (1, 1)
        self._window_pos: QPoint | None = None
        self._movement = Movement(0.0, 0.0)
        self._screen: ScreenArea | None = None

        self._started = False
        self._shutdown = False
        self._hidden_by_user = False
        self._fullscreen = False
        self._load_high = False
        self._battery_saver = False
        self._eco = False

        self._speech_until: float | None = None
        self._queue: deque[_Line] = deque()
        self._portraits: dict[str, QPixmap | None] = {}

        now = clock.monotonic()
        self._last_frame = now
        self._last_interaction = now
        self._input_known = False
        self._user_idle_since: float | None = None
        self._cursor_near = False
        self._cursor_side: Facing | None = None
        self._next_cursor_poll = now
        self._next_decide = now + DECIDE_INTERVAL
        self._games: set[str] = set()
        #: GAMING was rejected (a higher-priority action is showing): retried each behavior tick.
        self._gaming_pending = False
        #: Likewise for leaving GAMING once no game runs: without the retry the
        #: action would end back in GAMING and she would hold the gamepad for good.
        self._ungaming_pending = False
        self._walk_priority = P.AMBIENT
        self._call_walk = False
        self._last_visual_key: tuple | None = None
        self._visual_dirty = True
        self._last_position_save = -math.inf
        self._fps = 0
        self._unsubscribers: list[Callable[[], None]] = []
        self._hooked_screens: list[QScreen] = []
        self._app_screen_hooks = False

        self.window = CharacterWindow(always_on_top=self._cfg.character.always_on_top)
        self.window.set_paint_callback(self._paint)
        self.window.clicked.connect(self._on_clicked)
        self.window.scale_preview.connect(self._preview_scale)
        self.window.drag_started.connect(self._on_drag_started)
        self.window.dragged.connect(self._on_dragged)
        self.window.drag_finished.connect(self._on_drag_finished)
        self.window.context_menu_requested.connect(self.context_menu_requested)
        self.window.display_changed.connect(self._on_display_changed)
        self.bubble = SpeechBubble()

        self._frame_timer = QTimer(self)
        self._frame_timer.timeout.connect(self._on_timer)
        self._hidden_timer = QTimer(self)
        self._hidden_timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._hidden_timer.setInterval(HIDDEN_TICK_MS)
        self._hidden_timer.timeout.connect(self._on_timer)

    # -- read-only state ----------------------------------------------------------------

    @property
    def state(self) -> CharacterState:
        return self._sm.current

    @property
    def posture(self) -> CharacterState:
        return self._sm.posture

    @property
    def state_machine(self) -> CharacterStateMachine:
        return self._sm

    @property
    def movement(self) -> Movement:
        return self._movement

    @property
    def renderer(self) -> CharacterRenderer:
        return self._renderer

    @property
    def player(self) -> FramePlayer:
        return self._player

    @property
    def fps(self) -> int:
        """Current frame rate (0 = frame timer stopped)."""
        return self._fps

    @property
    def eco(self) -> bool:
        return self._eco

    @property
    def height_px(self) -> float:
        return self._height

    @property
    def screen(self) -> ScreenArea | None:
        return self._screen

    @property
    def frame_timer_active(self) -> bool:
        return self._frame_timer.isActive()

    @property
    def queued_lines(self) -> int:
        return len(self._queue)

    def is_visible(self) -> bool:
        return self.window.isVisible()

    def own_window_ids(self) -> set[int]:
        """Native handles of her windows (for the fullscreen check); never creates one."""
        ids = {int(self.window.internalWinId()), int(self.bubble.internalWinId())}
        ids.discard(0)
        return ids

    def idle_seconds(self) -> float:
        """Seconds since the last interaction - or since the user stopped
        typing, when the activity monitor (with permission) tells us."""
        now = self._clock.monotonic()
        since_interaction = max(0.0, now - self._last_interaction)
        if self._input_known:
            if self._user_idle_since is None:
                return 0.0  # the user is at the computer right now
            return max(0.0, min(since_interaction, now - self._user_idle_since))
        return since_interaction

    def visual_state(self) -> VisualState:
        clip = self._player.clip
        facing = self._movement.facing if self._movement.walking else self._facing
        overlay = clip.overlay
        if overlay is None and self._sm.current is S.SLEEP:
            overlay = "zzz"  # packs may not draw Zs themselves; sleeping always shows them
        return VisualState(self._sm.current, facing, self._anim_t, self._player.sample(), clip, overlay)

    # -- lifecycle ------------------------------------------------------------------------

    def start(self) -> None:
        """Place her (restored position or a default), show her and start ticking.
        Subscribes to the bus here, so events before start() are not half-handled."""
        if self._started or self._shutdown:
            return
        self._started = True
        self._last_frame = self._clock.monotonic()
        self._layout()
        self._restore_position()
        bus = self._bus
        self._unsubscribers = [
            bus.subscribe(SpeechRequested, self._on_speech),
            bus.subscribe(FullscreenChanged, self._on_fullscreen),
            bus.subscribe(SystemLoadChanged, self._on_system_load),
            bus.subscribe(UserIdle, self._on_user_idle),
            bus.subscribe(UserReturned, self._on_user_returned),
            bus.subscribe(InteractionOutcome, self._on_outcome),
            bus.subscribe(GameStarted, self._on_game_started),
            bus.subscribe(GameEnded, self._on_game_ended),
            bus.subscribe(SettingsChanged, self._on_settings_changed),
            bus.subscribe(PermissionChanged, self._on_permission_changed),
            bus.subscribe(ObservationStopped, self._on_observation_stopped),
            bus.subscribe(ForegroundAppChanged, self._on_foreground),
            bus.subscribe(QuietModeChanged, self._on_quiet_requested),
            bus.subscribe(DoNotDisturbChanged, self._on_quiet_requested),
        ]
        self._hook_screens()
        self.apply_settings()
        log.info("character started (%s renderer, scale %.2f)", "vector" if self._use_vector else "sprite",
                 self._scale)

    def shutdown(self) -> None:
        """Save her position, stop timers, leave the bus and close the windows."""
        if self._shutdown:
            return
        if self._started:
            self._save_position(force=True)
        self._shutdown = True
        self._frame_timer.stop()
        self._hidden_timer.stop()
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        self._unhook_screens()
        self._sm.remove_listener(self._on_state_change)
        self._queue.clear()
        self.bubble.hide_now()
        self.bubble.close()
        self.window.set_paint_callback(None)
        self.window.close()
        self._fps = 0

    def apply_settings(self) -> None:
        """Re-read settings and apply what changed (renderer, scale, window
        flags, lock, bubbles, eco, visibility)."""
        if self._shutdown:
            return
        cfg = self._read_settings()
        self._cfg = cfg
        c = cfg.character
        relayout = False
        if self._wants_vector(cfg) != self._use_vector:
            self._use_vector = not self._use_vector
            self._renderer = self._make_renderer()
            self._variant_active = False
            self._player = FramePlayer(self._clip_for(self._sm.current))
            relayout = True
        if c.scale != self._scale:
            self._scale = c.scale
            self._height = BASE_HEIGHT * self._scale
            self._half_width = BODY_HALF_WIDTH * self._height
            self._renderer.invalidate()
            relayout = True
        if relayout and self._started:
            self._layout()
            self._update_area()
        self.window.set_scale(c.scale)
        self.window.set_always_on_top(c.always_on_top)
        self.window.set_locked(c.lock_position)
        self.window.set_click_through(c.click_through)
        if not c.show_bubbles:
            self._queue.clear()
            self._speech_until = None
            self.bubble.hide_now()
        elif cfg.behavior.do_not_disturb:
            self._queue.clear()  # queued before DND: she must not say them after it
        mv = self._movement
        if mv.walking and (c.lock_position or (not c.walking_enabled and not self._call_walk)):
            self._end_walk()
        if mv.gravity != c.gravity and not mv.held:
            mv.drop(c.gravity)  # gravity switched on mid-air: she falls now
        self._update_eco()
        self._visual_dirty = True
        self._update_visibility()

    # -- public commands ------------------------------------------------------------------

    def say(self, text: str, state: str = "talk", portrait: str = "", seconds: float | None = None,
            solicited: bool = True) -> None:
        """Show *text* in the bubble while playing *state* (a CharacterState value).

        Empty *text* plays the state only (a gesture).  Unsolicited lines wait
        if a bubble is showing and are dropped while she is hidden.
        """
        if self._shutdown:
            return
        text = (text or "").strip()
        target = self._parse_state(state)
        visible = self.is_visible()
        if not visible and not solicited:
            return
        if seconds is None or not math.isfinite(seconds) or seconds <= 0:
            seconds = speech_seconds(text)
        seconds = min(float(seconds), SPEECH_MAX_EXPLICIT_SECONDS)
        show = bool(text) and visible and self._cfg.character.show_bubbles
        if show and not solicited and self._speaking():
            self._queue.append(_Line(text, target, portrait, seconds))
            while len(self._queue) > SPEECH_QUEUE_MAX:
                self._queue.popleft()
            return
        if solicited:
            self._mark_interaction()
        self._start_line(_Line(text, target, portrait, seconds), solicited, show)

    def request_state(self, state: CharacterState | str, priority: StatePriority = P.USER,
                      duration: float | None = None, variant: str | None = None) -> bool:
        """Ask the state machine for *state* (priorities arbitrate).  *variant*
        plays one of the state's fidget clips once (ignored if unknown)."""
        return self._request(state, priority, duration, variant=variant)

    def call_over(self) -> None:
        """"Come here": she walks (or runs) to the cursor's x on her screen, then waves."""
        if not self._started or self._shutdown:
            return
        self._mark_interaction()
        self._bus.publish(CharacterInteraction(kind=InteractionKind.CALL))
        mv = self._movement
        walk_range = mv.walk_range
        try:
            cursor_x = float(self._cursor_fn().x())
        except Exception:
            log.exception("cursor position unavailable")
            cursor_x = mv.x
        if self._cfg.character.lock_position or walk_range is None or not mv.on_ground or self._sm.dragged:
            self._request(S.WAVE, P.USER)
            return
        target = min(max(cursor_x, walk_range[0]), walk_range[1])
        distance = abs(target - mv.x)
        if distance < CALL_NEAR * self._scale:
            self._request(S.WAVE, P.USER)
            return
        run = distance > RUN_DISTANCE * self._scale
        speed = (RUN_SPEED if run else WALK_SPEED) * self._scale
        if self._request(S.RUN if run else S.WALK, P.USER, distance / speed + 2.0):
            mv.walk_to(target, speed)
            self._call_walk = True
            self._walk_priority = P.USER
            self._update_timer()

    def set_hidden_by_user(self, hidden: bool) -> None:
        self._hidden_by_user = bool(hidden)
        self._update_visibility()

    def set_input_monitoring(self, active: bool) -> None:
        """Whether the activity monitor is running (it reads input idle time).

        While it runs, "no ``UserIdle``" means the user is at the computer: the
        monitor only speaks up once they have been idle for minutes.  Without
        this she would judge an active user by clicks on her alone, fall asleep
        on them and never wake.  If the user is already idle, the monitor's
        first cycle sends ``UserIdle`` and corrects it.  Repeating ``True``
        while known keeps the current idle state (a redundant call must not
        mark an idle user present: no second ``UserIdle`` would follow).
        """
        if not active:
            self._forget_user_input()
        elif not self._input_known:
            self._input_known = True
            self._user_idle_since = None

    def portrait_pixmap(self, name: str) -> QPixmap | None:
        """Portrait *name* from the pack (cached); None without a pack or portrait."""
        if not name or self._pack is None:
            return None
        if name in self._portraits:
            return self._portraits[name]
        path = self._pack.portrait_path(name)
        pixmap: QPixmap | None = None
        if path is not None:
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                log.warning("portrait %r does not load", name)
                pixmap = None
        self._portraits[name] = pixmap
        return pixmap

    # -- the frame --------------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance everything by *dt* seconds (the frame timers call this; tests too)."""
        if self._shutdown:
            return
        if not (isinstance(dt, int | float) and math.isfinite(dt) and dt > 0):
            dt = 0.0
        now = self._clock.monotonic()
        self._last_frame = now
        self._anim_t += dt
        self._sm.tick()
        self._tick_speech(now)
        self._tick_movement(dt)
        if self.is_visible() and not self._sm.asleep:
            if now >= self._next_cursor_poll:
                self._next_cursor_poll = now + CURSOR_POLL_INTERVAL
                self._poll_cursor()
        else:
            self._cursor_near, self._cursor_side = False, None
        if now >= self._next_decide:
            self._next_decide = now + DECIDE_INTERVAL
            self._tick_behavior(now)
        self._advance_animation(dt)
        self._sync_window()
        self._repaint_if_changed()
        self._follow_bubble()
        self._update_timer()

    def _on_timer(self) -> None:
        now = self._clock.monotonic()
        self.step(now - self._last_frame)

    def _tick_speech(self, now: float) -> None:
        if self._speech_until is None or now < self._speech_until:
            return
        self._speech_until = None
        if self._queue and self.is_visible() and self._cfg.character.show_bubbles:
            self._start_line(self._queue.popleft(), solicited=False, show=True)
        else:
            self.bubble.fade_out()

    def _tick_movement(self, dt: float) -> None:
        mv = self._movement
        was_falling = mv.falling
        if dt > 0:
            mv.tick(dt)
        if self._sm.posture in _MOVING and not mv.walking and not self._sm.dragged:
            self._finish_walk()
        if was_falling and not mv.falling and not mv.held:
            self._update_area()
            self._save_position(force=True)

    def _tick_behavior(self, now: float) -> None:
        self._cfg = self._read_settings()
        if self._games and self._gaming_pending and self._request(S.GAMING, P.REACTION, GAMING_HOLD):
            self._gaming_pending = False
        if self._ungaming_pending and (self._games or self._sm.posture is not S.GAMING
                                       or self._request(S.IDLE, P.REACTION, GAMING_HOLD)):
            self._ungaming_pending = False
        ctx = self._behavior_context(now)
        try:
            decision = self._behavior.decide(ctx)
        except Exception:
            log.exception("behavior failed")
            return
        if decision is not None:
            self._apply_decision(decision)

    def _advance_animation(self, dt: float) -> None:
        self._player.advance(dt * 1000.0)
        if self._variant_active and self._player.finished:
            # a one-shot fidget ended: back to the state's main clip
            self._variant_active = False
            self._player = FramePlayer(self._clip_for(self._sm.current))
            self._visual_dirty = True

    # -- behavior -------------------------------------------------------------------------

    def _behavior_context(self, now: float) -> BehaviorContext:
        hunger, energy, happiness, boredom = self._read_needs()
        cfg = self._cfg
        mv = self._movement
        area = mv.area or (self._screen.area if self._screen is not None else Rect(0.0, 0.0, 1280.0, 720.0))
        return BehaviorContext(
            now=now,
            posture=self._sm.posture,
            current=self._sm.current,
            energy=energy,
            boredom=boredom,
            happiness=happiness,
            hunger=hunger,
            walking_enabled=cfg.character.walking_enabled,
            locked=cfg.character.lock_position,
            on_ground=mv.on_ground,
            eco=self._eco,
            independence=cfg.personality.independence,
            curiosity=cfg.personality.curiosity,
            idle_seconds=self.idle_seconds(),
            sleep_after_seconds=cfg.behavior.sleep_after_idle_minutes * 60.0,
            cursor_near=self._cursor_near,
            cursor_side=self._cursor_side,
            area=area,
            x=mv.x,
        )

    def _apply_decision(self, d: Decision) -> None:
        log.debug("decision %s -> %s", d.reason, d.state.value if d.state else None)
        if d.state is None:
            return
        if d.state in _MOVING:
            c = self._cfg.character
            if d.walk_to_x is None or c.lock_position or not c.walking_enabled:
                return
            speed = (RUN_SPEED if d.state is S.RUN else WALK_SPEED) * self._scale
            if self._request(d.state, d.priority, d.duration):
                self._call_walk = False
                self._walk_priority = d.priority
                self._movement.walk_to(d.walk_to_x, speed)
                self._update_timer()
            return
        if self._request(d.state, d.priority, d.duration, variant=d.variant) and d.facing is not None:
            self._facing = d.facing
            self._visual_dirty = True

    def _read_needs(self) -> tuple[float, float, float, float]:
        try:
            values = tuple(float(v) for v in self._needs_fn())
        except Exception:
            log.exception("needs() failed")
            return _DEFAULT_NEEDS
        if len(values) != 4 or not all(math.isfinite(v) for v in values):
            return _DEFAULT_NEEDS
        hunger, energy, happiness, boredom = (min(max(v, 0.0), 100.0) for v in values)
        return hunger, energy, happiness, boredom

    def _poll_cursor(self) -> None:
        try:
            cursor = self._cursor_fn()
            cx, cy = float(cursor.x()), float(cursor.y())
        except Exception:
            log.exception("cursor position unavailable")
            return
        mv = self._movement
        dx = cx - mv.x
        dy = cy - (mv.y - 0.5 * self._height)
        reach = CURSOR_NEAR * self._height
        near = abs(dx) < reach and abs(dy) < reach
        self._cursor_near = near
        self._cursor_side = (Facing.LEFT if dx < 0 else Facing.RIGHT) if near else None

    # -- states ---------------------------------------------------------------------------

    def _request(self, state: CharacterState | str, priority: StatePriority, duration: float | None = None,
                 variant: str | None = None) -> bool:
        target = self._parse_state(state)
        if target is None:
            return False
        before = self._change_count
        if not self._sm.request(target, priority, duration):
            return False
        changed = self._change_count != before
        if variant:
            self._play(self._clip_for(self._sm.current, variant), variant=True)
        elif not changed and not self._variant_active and not self._player.clip.loop:
            self._player.restart()  # the same one-shot again (a second wave) plays again
        self._visual_dirty = True
        return True

    def _on_state_change(self, change: StateChange) -> None:
        self._change_count += 1
        mv = self._movement
        if change.posture not in _MOVING and mv.walking:
            self._facing = mv.facing  # keep looking where she was walking
            mv.stop()
            self._call_walk = False
        self._variant_active = False
        self._play(self._clip_for(change.current))
        self._visual_dirty = True
        self.state_changed.emit(change.current.value)
        if self._started:
            self._update_timer()

    def _play(self, clip: Clip, variant: bool = False) -> None:
        if not variant and clip is self._player.clip and clip.loop:
            return  # same looping clip: keep its phase, no visible restart
        self._player = FramePlayer(clip)
        self._variant_active = variant and not clip.loop

    def _clip_for(self, state: CharacterState, variant: str | None = None) -> Clip:
        if self._use_vector or self._pack is None:
            return vector_clip_for(state, variant)
        return self._pack.clip_for(state, variant)

    def _finish_walk(self) -> None:
        """Her walk ended (arrived or stopped): settle into IDLE, or wave after a call."""
        mv = self._movement
        calling = self._call_walk
        self._call_walk = False
        self._facing = mv.facing
        if calling and mv.arrived and self._request(S.WAVE, P.USER):
            self._save_position()
            return
        # A posture keeps the priority it was requested with unless given a
        # duration: hold it only for a moment, or a USER walk ("come here")
        # would leave her IDLE at USER priority, deaf to her own behavior.
        if not self._request(S.IDLE, self._walk_priority, POSTURE_SETTLE_HOLD):
            # never leave her "walking" on the spot
            self._request(S.IDLE, P.USER, POSTURE_SETTLE_HOLD)
        self._save_position()

    def _end_walk(self) -> None:
        self._movement.stop()
        if self._sm.posture in _MOVING:
            self._finish_walk()

    @staticmethod
    def _parse_state(state: CharacterState | str | None) -> CharacterState | None:
        if state is None or state == "":
            return None
        try:
            return CharacterState(state)
        except ValueError:
            log.warning("unknown character state %r", state)
            return None

    # -- speech -----------------------------------------------------------------------------

    def _speaking(self) -> bool:
        return self._speech_until is not None and self._clock.monotonic() < self._speech_until

    def _start_line(self, line: _Line, solicited: bool, show: bool) -> None:
        priority = P.USER if solicited else P.REACTION
        target = line.state
        if target is S.GAMING and not self._games:
            # A game-start line spoken after its game ended or was forgotten:
            # nothing would ever take the gamepad from her again.
            target = S.TALK
        mv = self._movement
        # While she walks over because she was called, the reply's gesture would
        # stop the walk (actions end walks); she waves on arrival instead.
        busy_walking = self._call_walk and mv.walking and target not in POSTURES
        if target is not None and not busy_walking:
            if target is S.TALK or target in POSTURES:
                # postures too: a finite duration bounds how long the priority is held
                duration = line.seconds
            elif not line.text:
                duration = None
            else:
                default = DEFAULT_ACTION_SECONDS.get(target, FALLBACK_ACTION_SECONDS)
                duration = min(line.seconds, default + 1.0)
            self._request(target, priority, duration)
        if show:
            portrait = line.portrait or self._default_portrait(target)
            if self.bubble.show_text(line.text, self.portrait_pixmap(portrait), line.seconds):
                self._speech_until = self._clock.monotonic() + line.seconds
                self._follow_bubble()

    def _default_portrait(self, state: CharacterState | None) -> str:
        if self._pack is None or state is None:
            return ""
        spec = self._pack.states.get(state)
        return spec.portrait if spec is not None else ""

    def _follow_bubble(self) -> None:
        if not self.bubble.isVisible() or self._window_pos is None:
            return
        head = self._renderer.head_point(self._anchor, self._height, self.visual_state())
        top = QPointF(self._window_pos.x() + head.x(),
                      self._window_pos.y() + head.y() - 0.5 * OVERLAY_SIZE * self._height)
        self.bubble.follow(top, self._screen_qrect())

    # -- bus handlers -----------------------------------------------------------------------

    def _on_speech(self, event: SpeechRequested) -> None:
        self.say(event.text, event.state, event.portrait, None, event.solicited)

    def _on_fullscreen(self, event: FullscreenChanged) -> None:
        self._fullscreen = bool(event.active)
        self._update_visibility()

    def _on_system_load(self, event: SystemLoadChanged) -> None:
        self._load_high = bool(event.high)
        self._battery_saver = bool(event.on_battery_saver)
        self._update_eco()

    def _on_user_idle(self, event: UserIdle) -> None:
        if not self._observing():
            return
        self._input_known = True
        idle = event.idle_seconds if math.isfinite(event.idle_seconds) else 0.0
        self._user_idle_since = self._clock.monotonic() - max(0.0, idle)

    def _on_user_returned(self, _event: UserReturned) -> None:
        if not self._observing():
            return
        self._input_known = True
        self._user_idle_since = None
        # REACTION is enough to wake her (an action at >= REACTION wakes a sleeper).
        self._request(S.WAVE, P.REACTION, variant="greet")

    def _on_foreground(self, _event: ForegroundAppChanged) -> None:
        # The monitor sends this on its first cycle (before any UserIdle of that
        # cycle): proof that it watches input, even if the app never said so.
        if self._observing():
            self.set_input_monitoring(True)

    def _on_quiet_requested(self, event: QuietModeChanged | DoNotDisturbChanged) -> None:
        # Queued unsolicited lines were chosen before "помовч" / DND; saying
        # them right after her "I'll be quiet" would break the promise.
        if event.enabled:
            self._queue.clear()

    def _on_outcome(self, event: InteractionOutcome) -> None:
        self._mark_interaction()
        if event.kind is InteractionKind.REST and event.outcome == "sleep":
            self._request(S.SLEEP, P.USER, SLEEP_HOLD)
        elif event.kind is InteractionKind.WAKE and self._sm.asleep:
            self._request(S.IDLE, P.USER, WAKE_HOLD)

    def _on_game_started(self, event: GameStarted) -> None:
        if not self._observing():
            return  # queued before the kill switch: no GameEnded would ever follow
        self._games.add(event.game_id)
        self._ungaming_pending = False
        self._gaming_pending = not self._request(S.GAMING, P.REACTION, GAMING_HOLD)

    def _on_game_ended(self, event: GameEnded) -> None:
        # Never permission-gated: an end only removes state.  A *forgotten* end
        # (the game was made private while running) ends GAMING the same way.
        self._games.discard(event.game_id)
        if self._games:
            return
        self._gaming_pending = False
        self._leave_gaming()

    def _leave_gaming(self) -> None:
        """No game runs any more: back to IDLE.  A higher-priority action (a chat
        reply, a pat, a drag) rejects the request and would end back in GAMING,
        so a rejected request is retried by the behavior tick."""
        self._ungaming_pending = (self._sm.posture is S.GAMING
                                  and not self._request(S.IDLE, P.REACTION, GAMING_HOLD))

    def _forget_games(self) -> None:
        """Observation stopped or was revoked: the monitor drops its state without
        a ``GameEnded``, so the running games are forgotten here.  A stale id
        would keep her in GAMING and swallow every later game's end."""
        self._games.clear()
        self._gaming_pending = False
        self._leave_gaming()

    def _on_settings_changed(self, _event: SettingsChanged) -> None:
        self.apply_settings()

    def _on_permission_changed(self, event: PermissionChanged) -> None:
        if event.permission is Permission.APP_METADATA and not event.granted:
            self._forget_observation()

    def _on_observation_stopped(self, _event: ObservationStopped) -> None:
        self._forget_observation()

    def _forget_observation(self) -> None:
        """The kill switch or an ``APP_METADATA`` revoke: forget what the monitor
        told her.  Queued unsolicited lines go too - they may be about the game
        she just saw, and "stop watching" means she stops mentioning it."""
        self._forget_user_input()
        self._forget_games()
        self._queue.clear()

    def _observing(self) -> bool:
        """``APP_METADATA`` is granted *now* (dispatch happens after an in-memory
        revoke, so this also catches events queued just before it)."""
        if self._permissions is None:
            return True
        try:
            return bool(self._permissions.is_granted(Permission.APP_METADATA))
        except Exception:
            # Foreground changes arrive on every app switch: log a broken checker once.
            if not self._permission_check_failed:
                self._permission_check_failed = True
                log.exception("permission check failed; ignoring activity events")
            return False

    def _forget_user_input(self) -> None:
        """Monitoring stopped: no UserReturned will come, so stop assuming the
        user is idle (or active) - idle time falls back to interactions."""
        self._input_known = False
        self._user_idle_since = None

    def _mark_interaction(self) -> None:
        self._last_interaction = self._clock.monotonic()

    # -- window signals ---------------------------------------------------------------------

    def _on_clicked(self) -> None:
        if self._shutdown:
            return
        self._mark_interaction()
        if self._sm.asleep:
            self._request(S.IDLE, P.USER, WAKE_HOLD)
            self._bus.publish(CharacterInteraction(kind=InteractionKind.WAKE))
        else:
            self._bus.publish(CharacterInteraction(kind=InteractionKind.CLICK))

    def _on_drag_started(self) -> None:
        if self._shutdown or self._cfg.character.lock_position:
            return
        self._mark_interaction()
        if self._request(S.DRAGGED, P.DRAG):
            self._movement.place(self._movement.x, self._movement.y)
            self._bus.publish(CharacterInteraction(kind=InteractionKind.DRAG_START))
            self._update_timer()

    def _on_dragged(self, feet: QPointF) -> None:
        if self._shutdown or not self._sm.dragged:
            return
        x, y = float(feet.x()), float(feet.y())
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        screen = screen_for_point(self._screens(), x, y - 1.0)
        if screen is not None and screen != self._screen:
            self._set_screen(screen)
        self._movement.place(x, y)
        self._sync_window()
        self._follow_bubble()

    def _on_drag_finished(self) -> None:
        if self._shutdown or not self._sm.dragged:
            return
        self._mark_interaction()
        self._sm.release_drag()
        self._movement.drop(self._cfg.character.gravity)
        self._bus.publish(CharacterInteraction(kind=InteractionKind.DRAG_END))
        self._update_area()
        if not self._movement.falling:
            self._save_position(force=True)
        self._update_timer()

    def _on_display_changed(self) -> None:
        self._renderer.invalidate()
        self._visual_dirty = True
        self._update_area()

    # -- screens, layout, window ------------------------------------------------------------

    def _screens(self) -> list[ScreenArea]:
        try:
            return list(self._screens_fn())
        except Exception:
            log.exception("screen query failed")
            return []

    def _hook_screens(self) -> None:
        app = QGuiApplication.instance()
        if app is None:
            return
        app.screenAdded.connect(self._on_screen_added)
        app.screenRemoved.connect(self._on_screens_changed)
        self._app_screen_hooks = True
        for screen in QGuiApplication.screens():
            self._hook_screen(screen)

    def _hook_screen(self, screen: QScreen) -> None:
        screen.availableGeometryChanged.connect(self._on_screens_changed)
        self._hooked_screens.append(screen)

    def _unhook_screens(self) -> None:
        app = QGuiApplication.instance()
        connections: list[tuple[Any, Callable[..., None]]] = [
            (screen.availableGeometryChanged, self._on_screens_changed) for screen in self._hooked_screens
        ]
        if app is not None and self._app_screen_hooks:
            # Only disconnect what start() connected (shutdown without start is legal).
            connections += [(app.screenAdded, self._on_screen_added),
                            (app.screenRemoved, self._on_screens_changed)]
            self._app_screen_hooks = False
        for signal, slot in connections:
            # RuntimeError/TypeError: the screen (and its signal) is already gone
            with contextlib.suppress(RuntimeError, TypeError):
                signal.disconnect(slot)
        self._hooked_screens.clear()

    def _on_screen_added(self, screen: QScreen) -> None:
        if self._shutdown:
            return
        self._hook_screen(screen)
        self._on_screens_changed()

    def _on_screens_changed(self, *_args: object) -> None:
        if self._shutdown or not self._started:
            return
        self._hooked_screens = [s for s in self._hooked_screens if s in QGuiApplication.screens()]
        self._update_area()

    def _set_screen(self, screen: ScreenArea) -> None:
        old = self._screen
        self._screen = screen
        if old is not None and old.dpr != screen.dpr:
            self._renderer.invalidate()
            self._visual_dirty = True
        self._movement.set_bounds(screen.area, self._half_width, self._height)

    def _update_area(self) -> None:
        """Re-evaluate which monitor she is on and confine her to its work area."""
        if not self._started:
            return
        mv = self._movement
        screen = screen_for_point(self._screens(), mv.x, mv.y - 1.0)
        if screen is None:
            return
        self._set_screen(screen)
        self._sync_window()

    def _screen_qrect(self) -> QRect:
        if self._screen is not None:
            return self._screen.qrect()
        screen = QGuiApplication.primaryScreen()
        return screen.availableGeometry() if screen is not None else QRect(0, 0, 1280, 720)

    def _preview_scale(self, scale: float) -> None:
        """Resize immediately during a Ctrl+wheel gesture.  The setting itself is
        saved once the gesture ends (``scale_requested``); apply_settings then
        sees the same scale and does nothing more."""
        if self._shutdown or scale == self._scale:
            return
        self._scale = scale
        self._height = BASE_HEIGHT * self._scale
        self._half_width = BODY_HALF_WIDTH * self._height
        self._renderer.invalidate()
        if self._started:
            self._layout()
            self._update_area()
        self._visual_dirty = True

    def _layout(self) -> None:
        """Size the window from the renderer's reach plus room for overlays."""
        h = self._height
        e = self._renderer.full_extents(h)
        head = self._renderer.head_bounds(h)
        size = OVERLAY_SIZE * h
        reach_l, reach_r, reach_u, reach_d = overlays.OVERLAY_REACH
        side = max(e.left, e.right, head.left + reach_l * size, head.right + reach_r * size)
        up = max(e.up, head.up + reach_u * size)
        down = max(e.down, head.down + reach_d * size, 0.0)
        half = math.ceil(side) + WINDOW_PAD
        top = math.ceil(up) + WINDOW_PAD
        bottom = math.ceil(down) + WINDOW_PAD
        self._anchor = QPointF(half, top)
        self._window_size = (2 * half, top + bottom)
        self.window.setFixedSize(*self._window_size)
        self.window.set_anchor(self._anchor)
        self._window_pos = None  # force a move on the next sync
        self._visual_dirty = True

    def _sync_window(self) -> None:
        mv = self._movement
        pos = QPoint(round(mv.x) - round(self._anchor.x()), round(mv.y) - round(self._anchor.y()))
        if pos != self._window_pos:
            self._window_pos = pos
            self.window.move(pos)

    def _restore_position(self) -> None:
        screens = self._screens()
        try:
            data = self._repo.get(POSITION_KEY)
        except Exception:
            log.exception("could not read the saved position")
            data = None
        placed = decode_position(data, screens) if data is not None else None
        if placed is None:
            if data is not None:
                log.info("saved position is not on any current screen; using the default")
            placed = default_position(screens, self._half_width)
        x, y, screen = placed
        self._screen = screen
        mv = Movement(x, y)
        mv.gravity = self._cfg.character.gravity
        mv.set_bounds(screen.area, self._half_width, self._height)
        self._movement = mv
        self._sync_window()

    def _save_position(self, force: bool = False) -> None:
        if not self._started or self._shutdown or self._screen is None or self._movement.held:
            return
        now = self._clock.monotonic()
        if not force and now - self._last_position_save < POSITION_SAVE_INTERVAL:
            return
        self._last_position_save = now
        try:
            self._repo.set(POSITION_KEY, encode_position(self._movement.x, self._movement.y, self._screen))
        except Exception:
            log.exception("could not save the position")

    # -- visibility, eco, frame rate --------------------------------------------------------

    def _should_show(self) -> bool:
        if not self._started or self._shutdown or self._hidden_by_user:
            return False
        return not (self._fullscreen and self._cfg.behavior.hide_on_fullscreen)

    def _update_visibility(self) -> None:
        if self._should_show():
            if not self.window.isVisible():
                self._window_pos = None
                self._sync_window()
                self.window.show()
                self._visual_dirty = True
                self.window.update()
        elif self.window.isVisible():
            self.window.hide()
            self.bubble.hide_now()
            self._queue.clear()  # unsolicited lines are not for an empty desk
            self._speech_until = None
        self._update_timer()

    def _update_eco(self) -> None:
        eco = (self._load_high or self._battery_saver) and self._cfg.behavior.auto_eco_mode
        if eco != self._eco:
            self._eco = eco
            log.info("eco mode %s", "on" if eco else "off")
            self._update_timer()

    def _target_fps(self) -> int:
        if not self.is_visible():
            return 0
        mv = self._movement
        clip = self._player.clip
        if mv.walking or mv.falling or self._sm.dragged or not FAST_FX.isdisjoint(clip.fx):
            fps = FPS_FAST
        elif self._sm.current is S.SLEEP:
            fps = FPS_SLEEP
        else:
            fps = FPS_CALM
        return max(4, fps // 2) if self._eco else fps

    def _update_timer(self) -> None:
        if not self._started or self._shutdown:
            return
        fps = self._target_fps()
        if fps <= 0:
            self._frame_timer.stop()
            if not self._hidden_timer.isActive():
                self._hidden_timer.start()
        else:
            self._hidden_timer.stop()
            interval = max(1, round(1000 / fps))
            # Precise only while it matters: coarse timers let Windows coalesce wake-ups.
            timer_type = Qt.TimerType.PreciseTimer if fps >= FPS_FAST else Qt.TimerType.CoarseTimer
            if (not self._frame_timer.isActive() or self._frame_timer.interval() != interval
                    or self._frame_timer.timerType() != timer_type):
                self._frame_timer.setTimerType(timer_type)
                self._frame_timer.start(interval)
        self._fps = fps

    # -- painting ---------------------------------------------------------------------------

    def _repaint_if_changed(self) -> None:
        if not self.window.isVisible():
            return
        v = self.visual_state()
        frame = v.frame
        key = (v.state, v.facing, v.clip.name if v.clip else None, v.overlay,
               frame.frame_id if frame else None, frame.next_frame_id if frame else None,
               round(frame.blend, 2) if frame else 0.0, self._window_size)
        # Effects, overlays and the vector chibi (blinking) change with time itself.
        animated = self._use_vector or bool(v.clip and v.clip.fx) or v.overlay is not None
        if animated or self._visual_dirty or key != self._last_visual_key:
            self._last_visual_key = key
            self._visual_dirty = False
            self.window.update()

    def _paint(self, painter: QPainter) -> None:
        v = self.visual_state()
        self._renderer.paint(painter, self._anchor, self._height, v)
        if v.overlay:
            head = self._renderer.head_point(self._anchor, self._height, v)
            overlays.paint_overlay(painter, v.overlay, head, OVERLAY_SIZE * self._height, v.t,
                                   mirror=v.facing is Facing.LEFT)

    # -- helpers ----------------------------------------------------------------------------

    def _read_settings(self) -> Settings:
        try:
            return self._settings_fn()
        except Exception:
            log.exception("settings() failed; keeping the previous settings")
            return self._cfg

    def _wants_vector(self, cfg: Settings) -> bool:
        return self._pack is None or cfg.character.renderer == "vector"

    def _make_renderer(self) -> CharacterRenderer:
        if self._use_vector or self._pack is None:
            return self._vector
        return SpriteRenderer(self._pack, fallback=self._vector)
