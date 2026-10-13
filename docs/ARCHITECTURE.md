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
* **Worker threads** (activity monitor, AI service worker, voice worker,
  vision worker, hotkey thread, PortAudio callbacks): never touch Qt widgets or
  SQLite.  They call `EventBus.publish()`, `PermissionManager.is_granted()`
  (both thread-safe) or post a callable back to the main thread through a
  queued Qt signal, and they stop within ~1-2 s.
* Heavy or blocking work (AI calls, screen capture and analysis, speech
  recognition and synthesis, model downloads) never runs on the main thread.
  Every asynchronous result carries a *generation* number; cancelling or
  reconfiguring bumps it, so late results are dropped.

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
| `personality`  | Ukrainian phrase bank, reaction engine (cooldowns, budget, quiet mode), offline chat, decision engine, observer |
| `settings`     | Qt windows: settings (AI, memory, voice, vision, gaming, privacy pages), status, chat, first run; dark theme |
| `ai`           | providers (Claude via the official SDK, OpenAI-compatible, Ollama), secrets, budget, AIService worker, prompts, AIBrain |
| `emotion`      | EmotionEngine (eight emotions, decay, persistence) |
| `voice`        | TTS text normaliser, Piper voices on onnxruntime, Whisper STT, VAD, audio I/O, models, push-to-talk hotkey, VoiceService |
| `vision`       | window capture, frame processing, analysis contract (game and work scenes), reaction policy, VisionService |
| `games`        | mini-games (Phase 5) |
| `selftest.py`  | `--self-test`: checks an installation (used by CI on the frozen build) |
| `app.py`       | composition root: builds everything, wires events, lifecycle |

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
* `tray.py`: `TrayIcon(menu, icon)` with `set_observing(bool)` indicator,
  `set_listening(bool)` (red dot: the microphone is open), `set_voice_note(text)`
  (tooltip line: how the open microphone listens or why it is paused) and
  `set_watching(title)`.
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

---

## Phase 2 contracts: AI brain, memory, emotion, decisions

Principles: rules decide *when* she acts, the LLM only decides *how* she
phrases it; every AI failure (offline, budget, refusal, timeout) falls back to
the offline phrase bank; the Qt main thread never waits for a model.  Data
that leaves the PC (any provider whose host is not loopback) requires
`Permission.CLOUD_AI`.  Shared types: `chibi/ai/types.py` (exists).

### ai: providers, secrets, budget, service

```python
# ai/secrets.py
class SecretStore(Protocol):
    secure: bool                                   # True = OS credential store
    def get(self, name: str) -> str | None
    def set(self, name: str, value: str) -> None   # value never logged
    def delete(self, name: str) -> None
class WindowsCredentialStore   # ctypes advapi32 CredReadW/CredWriteW/CredDeleteW/CredFree,
                               # CRED_TYPE_GENERIC, target "ChibiCompanion/<name>"
class FileSecretStore          # non-Windows dev fallback (secure=False, file mode 0600)
class MemorySecretStore        # tests
def default_secret_store(data_dir: Path) -> SecretStore
KEY_NAMES = {"anthropic": "anthropic_api_key", "openai": "openai_api_key"}

# ai/providers/anthropic_provider.py  - official `anthropic` SDK only
class AnthropicProvider(TextProvider):
    def __init__(self, api_key: str | None, model: str = "claude-opus-5-5", *, effort: str = "low",
                 timeout: float = 30.0, max_retries: int = 2, http_client=None)
# ai/providers/openai_provider.py  - httpx2, OpenAI-compatible /chat/completions (+ SSE)
class OpenAIProvider(TextProvider):
    def __init__(self, api_key: str | None, base_url: str, model: str, *, timeout: float = 30.0,
                 http_client=None)
# ai/providers/ollama_provider.py  - httpx2, /api/chat (NDJSON stream), /api/tags, /api/show
class OllamaProvider(TextProvider):
    def __init__(self, base_url: str, model: str, *, timeout: float = 60.0, http_client=None)
# ai/providers/fake.py  - scripted provider for tests and the headless smoke test
class FakeProvider(TextProvider): replies: list[str | Exception]; requests: list[GenerationRequest]
# ai/providers/factory.py
DEFAULT_MODELS = {"anthropic": "claude-opus-5-5"}
def is_cloud(ai: AISettings) -> bool
def build_provider(ai: AISettings, secrets: SecretStore) -> TextProvider | None   # None for "none"

# ai/budget.py
PRICES_PER_MTOK: Mapping[str, tuple[float, float]]   # Claude model id -> (input, output) USD
@dataclass(frozen=True)
class UsageSummary: requests: int; input_tokens: int; output_tokens: int; cost_usd: float; cost_known: bool
class UsageLedger:                     # SQLite table ai_usage; main thread only
    def __init__(self, db: Database, clock: Clock)
    def record(self, provider: str, model: str, purpose: Purpose, usage: Usage, cost_usd: float | None) -> None
    def today(self) -> UsageSummary
    def requests_last_hour(self) -> int
def estimate_cost(provider: str, model: str, usage: Usage, ai: AISettings) -> float | None
class BudgetGuard:
    def __init__(self, ledger: UsageLedger, settings: Callable[[], AISettings])
    def check(self, purpose: Purpose) -> None      # raises BudgetExceededError (Ukrainian user_message)

# ai/service.py  (QObject; the only place that runs model calls - on one worker thread)
class AIService(QObject):
    def __init__(self, bus, settings: Callable[[], Settings], permissions, secrets: SecretStore,
                 ledger: UsageLedger, clock, *, provider_factory=build_provider, parent=None)
    def status(self) -> tuple[bool, str]             # (online, Ukrainian reason when offline)
    def is_available(self, purpose: Purpose) -> bool
    def submit(self, request: GenerationRequest, *, on_done: Callable[[GenerationResult], None],
               on_error: Callable[[Exception], None], on_delta: Callable[[str], None] | None = None,
               cancel: CancelToken | None = None) -> CancelToken
        # callbacks always on the main thread; queue max 4, lower Purpose rank dropped first;
        # identical non-chat requests answered from a 10-minute cache
    def reconfigure(self) -> None                   # settings or key changed
    def test(self, on_done: Callable[[bool, str], None]) -> None
    def list_models(self, on_done: Callable[[list[str], str], None]) -> None
    def usage_today(self) -> UsageSummary
    def shutdown(self) -> None                       # cancels and joins the worker (<= 2 s)
    # publishes AIStatusChanged; circuit breaker: 3 consecutive "unavailable" errors -> offline 120 s
```

### memory and emotion

```python
# memory/longterm.py
@dataclass(frozen=True)
class Memory: id: int; kind: str; text: str; source: str; importance: float; pinned: bool
              created_at: float; updated_at: float; last_used_at: float | None
def looks_sensitive(text: str) -> bool     # card/IBAN numbers, passwords, emails, phones -> never stored
class MemoryStore:                          # main thread; publishes MemoryChanged
    def __init__(self, db: Database, bus: EventBus | None, clock: Clock)
    def add(self, kind: str, text: str, source: str = "chat", importance: float = 0.5) -> Memory | None
        # near-duplicates (same kind) update the existing row; sensitive/empty text -> None
    def update(self, memory_id: int, text: str) -> None
    def set_pinned(self, memory_id: int, pinned: bool) -> None
    def delete(self, memory_id: int) -> None
    def clear(self, kind: str | None = None) -> None
    def list(self, kind: str | None = None, query: str = "", limit: int = 500) -> list[Memory]
    def search(self, query: str, k: int = 6, kinds: Iterable[str] | None = None) -> list[Memory]
        # BM25 over Ukrainian-friendly tokens (lower, apostrophes unified, 5-char prefix stems),
        # pinned + recency boost; updates last_used_at
class ShortTermMemory:                      # current conversation only, never persisted
    def __init__(self, max_turns: int = 60)
    def add(self, role: Role, text: str) -> None
    def turns(self, limit: int) -> tuple[ChatTurn, ...]
    def clear(self) -> None

# emotion/engine.py
EMOTIONS = ("happiness", "energy", "boredom", "affection", "curiosity", "excitement", "hunger", "trust")
class EmotionEngine:   # deterministic; hunger/energy/happiness/boredom/affection come from the tamagotchi
    def __init__(self, bus, repo: StateRepository, clock, needs: Callable[[], Needs],
                 settings: Callable[[], Settings])
    def values(self) -> dict[str, float]
    def dominant(self) -> str                 # CharacterState value
    def describe(self) -> str                 # short Ukrainian mood description for prompts
    def tick(self) -> None                    # decay; persists; publishes EmotionChanged on change >= 3
    # subscribes: GameStarted/GameEnded, InteractionOutcome, UserReturned, ForegroundAppChanged,
    #             ActivityMilestone, NeedsChanged
```

### brain, composer, decisions (personality package)

```python
# ai/prompts.py - stable Ukrainian persona prompt (cacheable: no timestamps), context and event prompts.
#   The user's own description (PersonalitySettings.persona, "Характер своїми словами") is a
#   section between the slider traits and the rules (rules win); prompt_safe_persona() cleans
#   it again, neutralises markup look-alikes and quotes every line with "│ ".  Empty -> the
#   prompt is byte-identical to the one without it.
# ai/brain.py
@dataclass(frozen=True)
class BrainContext: activity_category: str | None; activity_app: str | None; game_title: str | None
                    game_minutes: float | None; observation: str | None; user_away: bool
class AIBrain:
    def __init__(self, ai: AIService, memory: MemoryStore, short_term: ShortTermMemory,
                 emotion: EmotionEngine, settings: Callable[[], Settings], clock,
                 context: Callable[[], BrainContext], needs: Callable[[], Mapping[str, float]])
    def available(self, purpose: Purpose) -> bool
    def chat(self, text: str, on_delta: Callable[[str], None],
             on_done: Callable[[ChatReply | None], None]) -> None   # None -> caller uses offline reply
    def phrase(self, keys: Sequence[str], variables: Mapping[str, str], solicited: bool,
               on_done: Callable[[SpeechRequested | None], None]) -> None
# reply control line (first line of every chat/phrase reply, stripped before display):
#   <<state=happy;action=none>>     state: CharacterState value; action (chat only):
#   none | feed | pet | rest | quiet_on | quiet_off - anything else is ignored

# personality/composer.py
class SpeechComposer:   # one entry point for every line she says
    def __init__(self, bus, selector: PhraseSelector, brain: AIBrain | None,
                 settings: Callable[[], Settings], clock, *, ai_timeout: float = 8.0)
    def compose(self, *keys: str, solicited: bool = False, **variables) -> bool
        # True if a line will be published (now from the phrase bank, or later from the AI with
        # a phrase-bank fallback on failure/timeout); publishes exactly one SpeechRequested

# personality/initiative.py
@dataclass(frozen=True)
class Candidate: name: str; keys: tuple[str, ...]; priority: float; cooldown: float
                 requires: Callable[[InitiativeContext], bool]; emotion_fit: Callable[[Mapping[str, float]], float]
                 interruption_cost: Callable[[InitiativeContext], float]; action: str | None = None
class DecisionEngine:   # replaces ReactionEngine's idle chatter when wired
    def __init__(self, bus, composer: SpeechComposer, settings, clock, rng, emotion: EmotionEngine,
                 context: Callable[[], InitiativeContext], can_speak: Callable[[], bool])
    def tick(self) -> None        # called every ~10 s; picks the best candidate above threshold or nothing
```


---

## As built: Phase 2 contract extensions

Recorded from the implementation (the signatures above are the original
contract; these are additions, all keyword-only unless noted):

* `AIBrain(..., bank: PhraseBank | None = None, rng=None)` - style examples
  come from the phrase bank, not the selector (no history is written).
* `SpeechComposer(..., schedule=None)`; `set_brain()`, `pending`,
  `abandon_pending()`, `close()`.  At most 2 lines wait for the model; a line
  the user asked for waits at most 5 s; quiet / DND / shutdown abandon pending
  lines.
* `ReactionEngine(..., composer=None, idle_chatter=True)`, `note_unsolicited()`:
  with the decision engine wired, idle chatter is off and both engines share
  the unsolicited budget through `note_unsolicited`.
* `DecisionEngine(..., on_spoken=None, candidates=None)`; `InitiativeContext`
  has `game_title`, `activity_minutes`, `seconds_since_return`; `Candidate`
  has `relevance`, `spontaneous`, `variables`.
* Model replies never set a posture (sleep/walk...), and a chat action from
  the model is ignored when the user's own message contains control syntax.
* Phase 6: `AIBrain.consider_comment(observations: Sequence[SeenMoment],
  context: WatchContext, on_done: Callable[[CommentDecision], None]) ->
  CancelToken | None` (the observer; `on_done` exactly once, None when nothing
  was sent); pure helpers `build_observer_turn`, `parse_comment`,
  `observer_sources`, `similarity`.  The system prompt is the same cached
  `build_persona(settings)`; `action=silent` is allowed by the observer turn
  only (`SILENT_REPLY = "<<state=talk;action=silent>>"`).

## Phase 4: voice

```
push-to-talk (RegisterHotKey thread) / mic button ──► VoiceService.listen()
open microphone ("name" / "open": STANDBY) ──┐
microphone (PortAudio thread) ──► EnergyVAD ──► utterance ──► voice worker: WhisperSTT (one at a time)
        ──► main: find_address / fillers / echo filters ──► VoiceTranscript ──► chat (brain)
        ──► streamed reply ──► ReplyStream: a sentence at a time ──► voice worker: TTS (next one prefetched)
        ──► AudioOutput ──► SpeechPlayback(True, duration) + bubble for the audio's duration
```

* `voice/tts_text.py` turns any line into what the Ukrainian Piper voice can
  pronounce (lower case, numbers in words agreeing with the noun, clock times,
  game titles and acronyms from `data/pronunciations_uk.json`, transliteration).
* `voice/piper_onnx.py` runs Piper "text" voices directly on onnxruntime (MIT);
  the GPL `piper-tts` package is not used.  Phoneme ids were verified identical.
* `voice/stt.py`: faster-whisper on numpy arrays (no file decoding), int8 CPU,
  subtitle-credit hallucinations dropped.
* `voice/vad.py`: energy VAD with a minimum-statistics noise floor, hysteresis,
  stuck detection and re-confirmation; the floor carries over between sessions.
* `voice/models.py`: catalog pinned to repository revisions with SHA-256,
  resumable verified downloads into `%LOCALAPPDATA%\ChibiCompanion\models`.
* `voice/hotkey_win.py`: `RegisterHotKey` (no keyboard hook); hold mode polls
  `GetAsyncKeyState` for that key only while it is held.
* `voice/service.py` `VoiceService`: states off / idle / standby (open
  microphone) / listening (push-to-talk) / transcribing / speaking.  The
  microphone is open only while listening or in standby, never while she
  speaks; `mic_allowed()` is checked each time it would open and before
  anything the open microphone heard is delivered, and `revoke_microphone()`
  closes it immediately.

### As built: Phase 6, the voice companion (hands-free conversation)

* `VoiceSettings.listen_mode` (`set_listen_mode`): `"push"` - push-to-talk
  only; `"name"` - the microphone stays open and a phrase is delivered only
  when it addresses her: `find_address(text, NameForms)` finds her name
  anywhere as a separate word and returns the request without it
  ("Як справи, Міє?" -> "Як справи?").  An unambiguous vocative counts
  anywhere, the plain name only at the start / end / next to punctuation,
  other case forms only at the start; a one-letter recognition slip only of
  a vocative / the name of 5+ letters, at the start and set off like a
  vocative (`NameForms.near`: "Алено, ..."; never "вікно" for Віка, "мама"
  for Маша).  `NameForms.for_name` covers common Ukrainian declension, the
  app adds Mia's forms.  Her name alone keeps the open microphone and makes
  the next phrase (within 8 s) the request; one utterance may wait behind the
  one being recognised, kept only for that case.  `"open"` - every phrase is
  for her.  Push-to-talk works in every mode.
* A delivered request's `on_result` status says how it reached her: `"ok"`
  (push-to-talk), `"named"`, `"heard"` (open mode, conversation window) ->
  `VoiceTranscript.origin` `"push"` / `"named"` / `"heard"`.  What was
  `"heard"` (it may be the TV or someone else) is answered but never mined
  for memories (`AIBrain.chat(remember=False)`).
* Conversation window: after a delivered request (also push-to-talk) the
  microphone stays "addressed" while her answer is pending (up to 30 s) or
  being spoken and `follow_up_seconds` after it ends (`note_reply_shown()`
  for text-only answers; 0 turns it off).  A voiced line of her own that
  invites a reply (`voice.integration.invites_reply`: the observer's, a
  question) opens it too (`say(opens_conversation=True)`), and the app hands
  such lines to `AIBrain.note_unprompted`, so the next chat message carries
  them as context.  Pausing the open microphone ends the window.
* Filters on the open microphone: fillers ("угу", "мм"), what
  `stt.clean_segments` drops, a Whisper echo of the open-mic prompt, phrases
  shorter than 350 ms of speech, and the first 0.3 s after the microphone
  reopens (`echo_tail_seconds`: never her own voice).  Recognition is
  single-flight: phrases arriving meanwhile are dropped and counted
  (`dropped_utterances`), so work never queues up.  Her unprompted lines are
  shown as text instead of voiced while the open microphone hears someone,
  right after the user called her, and while she voices an answer sentence
  by sentence (never between two of its sentences).
* Pauses (`VoiceIntegration`, `VoiceStatus.pause_reason`): the user's switch
  (`set_open_mic_paused`, the "Пауза мікрофона" menu item; the app saves it in
  the state table, so a restart keeps it; "I hear you again" only when
  nothing else keeps the microphone paused), a
  private app in front (always), a call app in front (`pause_in_calls`,
  `is_call_app`), do-not-disturb, a fullscreen app unless
  `wake_in_fullscreen`.  The app in front comes from `ForegroundAppChanged`
  (APP_METADATA only).  Push-to-talk keeps working while paused.  If the
  microphone cannot stay open the service falls back to `"push"` and the next
  settings change retries.
* Streamed answers: `VoiceIntegration.begin_reply(source, show=) ->
  ReplyStream` with `feed(delta)` / `finish(text, state, portrait)` /
  `abort()`.  When she can voice it, each complete sentence
  (`SentenceChunker`) is routed as a solicited line while the model writes
  on; the microphone is held closed between sentences (`hold_for_reply`, a
  QTimer watchdog releases a stalled hold after 8 s).  `finish` routes only
  the sentences not routed yet, so the answer reaches her bubble / voice
  exactly once; when she cannot voice it, `finish` routes the whole answer
  once like `route()`.  A user cut-off (click, push-to-talk, her name) stops
  the voicing; the unheard rest is shown once as text at the end.

## Phase 3: screen vision

Pipeline (`vision/service.py`), in this order: permission -> the foreground
window is one she may see (scope `"allowed"`: the allowed lists; scope
`"foreground"`: any window except her own, private ones (by process, exe
path, or title with WINDOW_TITLES), Windows' own surfaces (`SYSTEM_PROCESSES`:
lock and sign-in screens, Start, flyouts, Task Manager, UAC, file pickers),
`explorer.exe` unless its title is readable and not a shell title, the
Microsoft / work account sign-in hosts, and
browsers unless WINDOW_TITLES is granted; display names come from the games
database and the app catalog, never from titles) -> due (mode, interval, back-off) ->
capture that one window (`PrintWindow`, foreground-only `BitBlt` fallback) ->
privacy re-check (same window, still allowed, permission still granted) ->
downscale to 1024 px, dHash dedup -> analysis by a local model, or a cloud
model only with `CLOUD_VISION` -> strict parsing into closed vocabularies
(`vision/analysis.py`) -> `VisionReactionPolicy` (two-frame confirmation for
important events, cooldowns, hourly budget, quiet mode, spoilers, novelty)
-> `still_allowed(target)` on the main thread when the answer arrives (capture
and, for a cloud model, cloud permissions still granted; the app still
allowed; otherwise the observation is dropped).  A frame in flight is
cancelled when the scope changes, the app becomes private or WINDOW_TITLES /
APP_METADATA is revoked.  Frames live in memory only; the last few observations (scene, event, short
sanitised description) are context for her chat, nothing is stored on disk.
Windows Graphics Capture is not used (it needs a Direct3D 11 interop layer);
`CaptureBackend` is the seam for it.

Vision analysis may use its own model (`VisionSettings.model`): it travels as
`GenerationRequest.model`, which every provider honours (`request.model or
self.model`) for the model id, Claude effort/fallback decisions, the result's
`model` (so the ledger prices the right model) and the response cache key.
`vision/integration.py` (`VisionIntegration`) owns the 1 s timer, the analyzer
(rebuilt on settings/permission changes, never created while the frames could
not legally be sent), the manual "подивись на екран" request, notices, the
session summary memory and `latest_observation()` for the brain.

As built for Phase 6:

* Work scenes (`code`, `terminal`, `document`, `spreadsheet`, `browser`,
  `video`, `chat`, `email`, `design`) and work events (`error_on_screen`,
  `build_failed`, `tests_passed`, `waiting`) join the closed vocabularies; a
  game event in a work scene (or the reverse) is distrusted; `chat` / `email`
  are never described.  `sanitize_description` also redacts e-mails,
  @handles, 6+ digit numbers, URLs with queries or paths and user file paths
  (160 characters at most).  Non-game windows are named `{app}`, never
  "гра «VS Code»".
* `VisionSettings.commentary`: `"events"` - she comments on notable game
  events herself (as before); `"observer"` - vision never voices a line on its
  own (manual requests are still answered) and the observer decides.
* `VisionObserved` is published exactly once per analysed, parsed frame
  (every mode and scope), never for skipped duplicates, errors, frames dropped
  by a re-check, results after stop / revoke / kill switch, or while
  `screen_data_may_leave()` is False (cloud provider without CLOUD_VISION, or
  "Лише локальний аналіз").  Fields: display title only, closed-vocabulary
  scene / event / emotion, confidence, sanitised description (empty for
  correspondence and hidden spoilers), `confirmed`, `react`.
  `details_for(event) -> ObservationDetails(interesting, spoiler, is_game, work,
  solicited)` for the last 16 published events (by identity).
* Cost in AUTO mode with scope `"foreground"`: no automatic frames between
  `UserIdle` and `UserReturned` (`service.away`; a `VoiceTranscript` in the
  last 5 minutes counts as presence), identical frames of the same
  window stretch the pause (x0.5 per frame, up to x3), `adapt_to_load`
  doubles the interval, a locked screen has no target.  With commentary
  `"observer"`, no automatic frames while the observer could not speak
  (`observer_ai`: her own lines may not use the AI, offline, limit;
  `service.muted`, `VisionStatusView.observer_note`).  Everywhere:
  `BudgetGuard` gives `Purpose.VISION` at most half of `max_requests_per_hour`
  (counting its own requests), so chat and the observer keep headroom; an
  hourly refusal pauses vision 10 min with an "this hour" message, not
  "today".
* Requests that carry what she saw (`GenerationRequest.screen_data`, or
  images) are bound to the provider current at submit, and a cloud provider
  gets them only while `CLOUD_VISION` is granted and "Лише локальний аналіз"
  is off (re-checked on the AI worker; revoking cloud vision drops them).  A
  line composed from the screen is voiced by Azure only under the same
  condition (shown as text otherwise), and then the chat context carries no
  observation either.

## Phase 5: character polish, gifts, mini-games

* `CharacterState` gains transient actions JUMP, DANCE, EXCITED, PROUD and
  STRETCH (not postures); the renderer has the `jump`, `bounce`, `squash`,
  `sway` and `stretch` effects, a landing squash after a fall, and
  `FramePlayer.ms_to_next_change()` keeps blinks exact at low frame rates.
* The controller handles `InitiativeAction` (wave, jump, dance, sit, sleep,
  approach_cursor) at BEHAVIOR priority with guards (no nap unless tired and
  idle for minutes; in eco mode no dance and a shorter jump) and adds idle variety.
* `tamagotchi/gifts.py`: `Wallet` (daily +5 start bonus stored with its day,
  daily earning cap), `GiftCatalog` (`data/gifts.json`), `GiftService`
  (buy / give, consumables apply needs effects through
  `TamagotchiService.apply_effect`, keepsakes kept in a separate `collection`
  and remembered).  `games/`: pure game logic (`rps`, `cookies`, `guess`),
  Qt windows, `GameRewards` (stars + effects + her reaction once per game)
  and `GamesHub` (opens/closes the windows and the shop).

## App wiring (Phases 3-6)

* Every `SpeechRequested` goes through `CharacterController.set_speech_router`
  -> `ChibiApp._route_speech` -> `VoiceIntegration.route`, which voices it (if
  enabled and allowed) and calls `show(event, seconds)` exactly once, when the
  audio starts; the bubble then lasts as long as the audio.  In gaming
  companion mode a voiced unsolicited line over a running game is shown
  without its bubble.
* Chat answers from the AI brain (typed or spoken) go through
  `VoiceIntegration.begin_reply("chat")`: every streamed piece is fed to the
  `ReplyStream`, so she starts talking at the first complete sentence;
  `finish(reply.text, reply.state, reply.portrait)` after `_apply_chat_action`
  (never also `route()`: the answer is said exactly once).  When the AI
  fails, even mid-answer, the stream is aborted and the offline answer is
  routed instead.  Offline answers and "подивись на екран" take `route()` as
  before.
* `VoiceTranscript` is dropped unless `MICROPHONE` is granted and
  `VoiceIntegration.transcript_allowed(origin)` holds at dispatch time (one
  from the open microphone not once it paused, left the open modes or
  recognition is off - a request waiting for the answer in flight is dropped
  then too); otherwise it goes to the open chat window (keeping the user's draft; while
  that window is answering, `BUSY_REPLY` there, never spoken) or straight to
  `send_chat_stream`.  One chat answer at a time: a request heard while an
  answer is in flight waits for it (only the newest one, at most 60 s); an
  answer lost for 90 s no longer holds requests back; a revoked microphone or
  the kill switch drops the waiting request.
* The tray shows a red dot while the microphone is open (also in standby) and
  names the window she is looking at; its tooltip says how the open microphone
  listens or why it is paused (`VoiceStatus.listening_text`).  Her menu and the
  tray menu have "Пауза мікрофона" next to "Не турбувати" (enabled in the
  "name" / "open" modes, or while paused).
* Observer (`personality/observer.py`), built after the reaction engine, the
  brain and vision: `Observer(bus, brain, settings, clock, rng,
  reactions.can_speak_unsolicited, reactions.note_unsolicited,
  context=_brain_context, last_chat=..., details=vision.details_for,
  may_share=vision.screen_data_may_leave, permissions, sleeping)`.  Active
  only with commentary `"observer"`: it keeps an in-memory log of up to 10
  moments / 15 minutes from `VisionObserved` (never correspondence; a death /
  victory not confirmed by a second frame is not logged at all, another unsure
  event is logged as `other`) and, when every gate in its
  module docstring passes (presence - a voice transcript counts for 5
  minutes -, DND / quiet, nobody talking to her for
  60 s, cooldown 2-10 min after speaking, growing gaps after silences, 6-20
  decisions an hour, the shared unsolicited gate, AI available, data may
  leave), asks `AIBrain.consider_comment` - one `Purpose.INITIATIVE` request
  with escaped `<observation>` data (`screen_data=True`).  The strict parser
  answers silence for `SILENT_REPLY`, malformed output, repeats, sensitive
  text or a number / Latin word that appears in nothing she saw (her name,
  the user's nickname and the persona text count as known); otherwise one or two sentences
  are published as `SpeechRequested(solicited=False, source="observer:ai")`
  and counted against the shared budget.  A newer different frame, an app
  change, the user going away or talking, quiet / DND, a revoked permission,
  a settings change or the kill switch make the decision in flight stale.
  Nothing of it is persisted or logged.
* "Подивись на екран" in the chat calls `VisionIntegration.request()`; the
  reply is "Зараз гляну!" or the real reason she cannot look - never a
  description the model made up.  The answer arrives later as a solicited line.
* Shutdown stops vision, voice and the observer before the AI service (an
  analysis, a voiced line or a decision in flight would otherwise wait for a
  callback that never comes); "Очистити" on the vision page, wipe-history and
  wipe-all also make the observer forget; wipe-all also deletes the Azure key,
  the wallet/inventory, what she saw and a voice request still waiting.
* Interface feedback keys (`voice.*`, `game.start`) never go to the model
  (`prompts.BANK_ONLY_PREFIXES`): "Слухаю!" must be instant and free.

## Packaging

`packaging/ChibiCompanion.spec` (PyInstaller, one folder, windowed, no UPX),
`packaging/installer.iss` (Inno Setup, per-user, no admin).  CI builds both on
windows-latest, runs `ChibiCompanion.exe --self-test` and a 10-second
start/quit, and uploads the installer as an artifact.  Logs are redacted by
`core.logging_setup.RedactingFormatter` (keys, tokens, e-mails, card/phone
numbers, URL queries, user names in paths).
