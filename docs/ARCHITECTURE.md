# Architecture

Chibi Companion is a single-process PySide6 app.  Logic is plain Python
(testable headless); Qt is confined to `desktop/` and `settings/`.

```
Windows / timers ──► producers ──► EventBus (bounded, thread-safe) ──► main-thread consumers
                                                                         │
 activity monitor (worker thread) ─┐                                     ├─ TamagotchiService
 screen service (fullscreen, load) ┤   publish()          dispatch on    ├─ ReactionEngine ──► SpeechRequested
 character window (clicks, drags) ─┤ ─────────────► bus ───────────────► ├─ GameSessionTracker
 chat window (UserMessage)  ───────┘   any thread       Qt main thread   └─ CharacterController ──► state machine
                                                                                 │                 animation
                                                                                 └──► window, bubble, tray
```

## Threading rules

* **Main (Qt) thread**: all widgets, the state machine, reactions, tamagotchi,
  SQLite.  The SQLite connection is created with `check_same_thread=True`, so
  touching it from another thread raises instead of corrupting data.
* **Worker threads** (activity monitor): never touch Qt or SQLite.  They only
  call `EventBus.publish()` and `PermissionManager.is_granted()` (both
  thread-safe) and must stop within ~1 s of `stop()`.
* Heavy or blocking work (future AI calls, capture, TTS) never runs on the main
  thread.

## Privacy by construction

* `PermissionManager` is the only gate.  All permissions are off on first run;
  features that don't exist in this build cannot be granted; the kill switch
  revokes every observation permission at once.
* The activity monitor checks `APP_METADATA` **every cycle** and discards
  results gathered while it was revoked.  Window titles are read only if
  `WINDOW_TITLES` is granted, are used transiently for classification and
  never leave the monitor (no event field carries a title).
* The privacy filter runs inside the monitor *before* publishing: private apps
  (password managers, messengers, banking, user exclusions, private-browsing
  titles) are published as `AppInfo.private_app()` with category `PRIVATE` -
  consumers cannot leak what they never receive.
* Nothing about other apps is written to disk except game sessions
  (game id, title, start/end).  No screenshots, no keystrokes, no titles.
* Environment checks that need no permission carry no personal data:
  "is some window fullscreen" (geometry only) and system CPU load.

## Modules

| package        | responsibility |
|----------------|----------------|
| `core`         | events, event bus, clock, settings model, paths, logging |
| `memory`       | SQLite database, migrations, repositories |
| `privacy`      | permissions, privacy filter (private apps / titles) |
| `character`    | states, state machine, animation pack & player, movement, autonomous behavior (pure Python) |
| `desktop`      | Qt: character window, renderers, overlays, speech bubble, tray, Windows helpers, controller |
| `activity`     | foreground app / idle / process scan backends (Windows via ctypes), classifier, monitor thread |
| `gaming`       | games database, game session tracking |
| `tamagotchi`   | needs model (hunger, energy, ...), offline catch-up, interactions |
| `personality`  | Ukrainian phrase bank, reaction engine (cooldowns, budget, quiet mode), offline chat |
| `settings`     | Qt windows: settings, status, chat, first run; dark theme |
| `app.py`       | composition root: builds everything, wires events, lifecycle |

Future phases add `ai`, `emotion`, `vision`, `voice` packages behind the same
bus; see `PLAN.md`.

---

## Module contracts

These signatures are the integration contract between modules.  Types not
spelled out are in the referenced module.

### character (pure Python, no Qt)

`character/states.py` (exists): `CharacterState`, `POSTURES`, `Facing`, `StatePriority`.

`character/animation.py`
```python
class PackError(Exception)

@dataclass(frozen=True)
class FrameInfo:      id: str; path: Path; anchor: tuple[int, int]; scale: float
@dataclass(frozen=True)
class ClipFrame:      frame_id: str; ms: int
@dataclass(frozen=True)
class Clip:
    name: str; frames: tuple[ClipFrame, ...]; loop: bool; facing: Facing
    blend_ms: int; fx: Mapping[str, float]; overlay: str | None
    duration_ms: int  (property)
@dataclass(frozen=True)
class StateSpec:      clip: str; variants: tuple[str, ...]; portrait: str

class CharacterPack:
    name: str; reference_height: int; directory: Path
    frames: Mapping[str, FrameInfo]; clips: Mapping[str, Clip]
    states: Mapping[CharacterState, StateSpec]; portraits: Mapping[str, Path]
    @classmethod
    def load(cls, directory: Path) -> CharacterPack      # validates; raises PackError
    def clip_for(self, state: CharacterState, variant: str | None = None) -> Clip
        # falls back to the IDLE clip if the pack lacks the state
    def portrait_path(self, name: str) -> Path | None

@dataclass(frozen=True)
class FrameSample:
    frame_id: str; next_frame_id: str | None; blend: float   # 0..1 weight of next frame
    finished: bool                                           # non-loop clip reached its end

class FramePlayer:
    def __init__(self, clip: Clip)
    clip: Clip; elapsed_ms: float; finished: bool
    def advance(self, dt_ms: float) -> FrameSample
    def sample(self) -> FrameSample
    def restart(self) -> None
```
Blending: within the last `blend_ms` of a frame's duration the sample blends
toward the next frame (wrapping for loops; never past the end of a non-loop).

`character/state_machine.py`
```python
DEFAULT_ACTION_SECONDS: Mapping[CharacterState, float]

@dataclass(frozen=True)
class StateChange: previous: CharacterState; current: CharacterState; posture: CharacterState

class CharacterStateMachine:
    def __init__(self, clock: Clock)
    current: CharacterState; posture: CharacterState; priority: StatePriority
    def request(self, state, priority=StatePriority.AMBIENT, duration: float | None = None) -> bool
    def release_drag(self) -> None
    def tick(self) -> StateChange | None
    def add_listener(self, callback: Callable[[StateChange], None]) -> None
```
Rules: postures persist until replaced; actions last `duration` (or the
default) then revert to the posture.  A request is accepted if its priority is
`>=` the active priority or the active action has expired.  `DRAGGED` is only
entered with `StatePriority.DRAG` and held until `release_drag()` (posture
becomes IDLE).  While asleep, requests below `REACTION` are rejected; accepted
non-posture requests wake her (posture -> IDLE).  An action requested while
walking/running stops the walk (posture -> IDLE).

`character/movement.py`
```python
@dataclass(frozen=True)
class Rect: left: float; top: float; right: float; bottom: float   # logical px, global

class Movement:
    def __init__(self, x: float, y: float)       # feet anchor, global logical coords
    x; y; facing: Facing; walking: bool; falling: bool; on_ground: bool; arrived: bool
    def set_bounds(self, area: Rect, half_width: float, head_height: float) -> None
    def walk_to(self, x: float, speed: float) -> None
    def stop(self) -> None
    def place(self, x: float, y: float) -> None  # while dragging; keeps her reachable on screen
    def drop(self, gravity: bool) -> None        # after drag; falls to area.bottom if gravity
    def tick(self, dt: float) -> bool            # True if position changed
```

`character/behavior.py`
```python
@dataclass(frozen=True)
class BehaviorContext:
    now: float                 # monotonic seconds
    posture: CharacterState; current: CharacterState
    energy: float; boredom: float; happiness: float; hunger: float     # 0..100
    walking_enabled: bool; locked: bool; on_ground: bool; eco: bool
    independence: int; curiosity: int                                   # 0..100
    idle_seconds: float        # since last interaction or (if permitted) user input
    sleep_after_seconds: float
    cursor_near: bool; cursor_side: Facing | None
    area: Rect; x: float

@dataclass(frozen=True)
class Decision:
    state: CharacterState | None; priority: StatePriority
    duration: float | None = None; walk_to_x: float | None = None
    variant: str | None = None; facing: Facing | None = None; reason: str = ""

class AutonomousBehavior:
    def __init__(self, rng: random.Random)
    def decide(self, ctx: BehaviorContext) -> Decision | None   # called ~1/s
```

### desktop (Qt)

* `qt_bridge.QtEventPump` (exists).
* `renderers.py`: `VisualState(state, facing, t, frame: FrameSample | None, clip: Clip | None)`;
  `CharacterRenderer` protocol with
  `paint(painter, anchor: QPointF, height_px: float, visual: VisualState) -> None` and
  `head_point(anchor, height_px, visual) -> QPointF`; implementations
  `SpriteRenderer(pack)` (cached pre-scaled pixmaps keyed by frame, size,
  mirror, device pixel ratio) and `VectorRenderer()` (original QPainter chibi,
  used when the pack is missing or `renderer == "vector"`).
* `overlays.py`: `paint_overlay(painter, kind, head: QPointF, size: float, t: float)`
  for `anger, tear, exclaim, question, dots, sparkle, hearts, blush, game, zzz`.
* `character_window.py`: frameless, translucent, tool window that never takes
  focus; emits `clicked`, `drag_started`, `dragged(QPointF)`, `drag_finished`,
  `context_menu_requested(QPoint)`, `scale_requested(float)`.
* `bubble.py`: `SpeechBubble.show_text(text, portrait: QPixmap | None, seconds)`,
  `follow(head_global: QPointF, screen_area: QRect)`, `hide_now()`; click-through.
* `tray.py`: `TrayIcon(menu, icon)` with `set_observing(bool)` indicator.
* `platform_win.py`: ctypes helpers (`apply_no_activate(hwnd)`,
  `foreground_fullscreen(own_hwnds) -> bool`), safe no-ops off Windows.
* `screen_service.py`: polls fullscreen (2 s) and CPU / battery saver (5 s)
  on the main thread; publishes `FullscreenChanged`, `SystemLoadChanged`.
* `controller.py`: `CharacterController` owns the frame timer, state machine,
  player, movement and behavior; consumes `SpeechRequested`; publishes
  `CharacterInteraction`; adapts frame rate (smooth while moving, slow while
  idle/asleep, stopped while hidden, halved in eco mode).

### activity / gaming / privacy filter

```python
# activity/backend.py
@dataclass(frozen=True)
class ForegroundInfo: pid: int; process_name: str; exe_path: str | None; title: str | None
class ActivityBackend(Protocol):
    def is_supported(self) -> bool
    def foreground(self, read_title: bool) -> ForegroundInfo | None
    def idle_seconds(self) -> float | None
    def running_processes(self) -> set[str]          # lower-case exe names
class NullActivityBackend; def default_backend() -> ActivityBackend
# activity/windows_backend.py: WindowsActivityBackend (ctypes + psutil)

# privacy/filters.py
class PrivacyFilter:
    def __init__(self, extra_processes: Iterable[str] = ())
    def is_private_process(self, process_name: str) -> bool
    def is_private_title(self, title: str | None) -> bool
    def is_private(self, info: ForegroundInfo) -> bool

# gaming/games_db.py
@dataclass(frozen=True)
class GameMatch: game_id: str; title: str; genre: str; confidence: float; source: str
class GameDatabase:
    @classmethod
    def load(cls, custom_games: list[dict[str, str]] = ()) -> GameDatabase
    def match(self, process_name: str, exe_path: str | None = None) -> GameMatch | None
    def __len__(self) -> int

# activity/classifier.py
@dataclass(frozen=True)
class Classification: category: ActivityCategory; confidence: float; display_name: str; game: GameMatch | None
class ActivityClassifier:
    def __init__(self, games: GameDatabase)
    def classify(self, info: ForegroundInfo) -> Classification

# activity/monitor.py
class ActivityMonitor:
    def __init__(self, backend, permissions, bus, classifier, privacy,
                 settings: Callable[[], ActivitySettings], clock: Clock)
    def start(self) -> None; def stop(self, timeout: float = 2.0) -> None
    def is_running(self) -> bool
    def poll_once(self) -> None          # one synchronous cycle (tests)
    def reconfigure(self, classifier, privacy) -> None   # thread-safe swap after settings change

# gaming/sessions.py
class GameSessionTracker:          # main thread; persists finished sessions
    def __init__(self, bus: EventBus, repo: GameSessionRepository, clock: Clock)
```
Game ids and genres are fixed by `gaming/data/games.json` and used as phrase
keys (`game_start.<game_id>`, `game_start.genre.<genre>`).

### tamagotchi

```python
@dataclass
class Needs: hunger; energy; happiness; boredom; affection: float; friendship_xp: float
    friendship_level: int (property, 1..10); friendship_progress: float (property)
class NeedsModel:   # pure functions over Needs, no clock inside
    def tick(self, needs, dt: float, *, sleeping: bool, user_active: bool) -> Needs
    def offline_catch_up(self, needs, away_seconds: float) -> Needs   # never punishes absence
    def feed / pet / rest / click / wake(...) -> tuple[Needs, str]   # outcome strings, see InteractionOutcome
    def mood(self, needs) -> str                                      # Ukrainian word
class TamagotchiService:
    def __init__(self, bus: EventBus, repo: StateRepository, clock: Clock, model: NeedsModel | None = None)
    needs: Needs; sleeping: bool
    def load(self) -> None; def save(self) -> None; def tick(self) -> None
    def set_sleeping(self, sleeping: bool) -> None
    def interact(self, kind: InteractionKind, *, announce: bool = True) -> str   # returns outcome
    # subscribes to CharacterInteraction -> interact(kind) -> publishes InteractionOutcome (+ NeedsChanged).
    # announce=False is for the chat, which composes its own reply (avoids saying two lines).
```

### personality

```python
# personality/phrases.py
@dataclass(frozen=True)
class Phrase: id: str; key: str; text: str; tone: str; state: str; portrait: str; requires: tuple[str, ...]
class PhraseBank:
    @classmethod
    def load(cls, path: Path | None = None) -> PhraseBank
    def has(self, key: str) -> bool; def phrases(self, key: str) -> list[Phrase]
class PhraseSelector:
    def __init__(self, bank, history: PhraseHistoryRepository, settings: Callable[[], Settings],
                 clock: Clock, rng: random.Random)
    def say(self, *keys: str, solicited: bool = False, **variables) -> SpeechRequested | None
        # first key with an eligible phrase wins; records usage for anti-repeat

# personality/reactions.py
class ReactionEngine:
    def __init__(self, bus, selector, settings: Callable[[], Settings], clock, rng)
    quiet_until: float | None
    def set_quiet(self, minutes: float | None) -> None; def clear_quiet(self) -> None
    def can_speak_unsolicited(self) -> bool
    def tick(self) -> None        # autonomous idle chatter, needs hints, late-night care

# personality/chat_offline.py
class OfflineChatResponder:
    def __init__(self, selector, reactions: ReactionEngine, settings: Callable[[], Settings], clock,
                 perform: Callable[[str], str] | None = None,        # "feed"|"pet"|"rest" -> outcome
                 needs: Callable[[], Mapping[str, float]] | None = None)
    def respond(self, text: str) -> ChatReply
```

### settings (Qt windows)

Depend only on `settings/facade.py` (`AppFacade`, `NeedsView`, `ChatReply`,
`StatusView`): `theme.apply_dark_theme(app)`, `SettingsDialog(facade)`,
`StatusWindow(facade)`, `ChatWindow(facade)`, `FirstRunDialog(facade)`.
