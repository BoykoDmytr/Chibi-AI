"""VisionIntegration with fakes (capture, foreground, analyzer, composer,
memory): what may be captured, permission revocation and the kill switch,
reactions only through the composer when the policy says so, the game-session
memory (counts only), the brain's observation and the timer lifecycle.  Plus
the vision phrase file."""

from __future__ import annotations

import json
import random
import time
from collections import Counter

import pytest
from PySide6.QtCore import QCoreApplication

import chibi.privacy.permissions as permissions_module
import chibi.vision.integration as integration_module
from chibi.activity.backend import ForegroundInfo
from chibi.ai.providers.factory import is_cloud
from chibi.core.config import Settings
from chibi.core.events import (
    ConversationActive,
    GameEnded,
    GameStarted,
    Permission,
    SettingsChanged,
    SystemLoadChanged,
    UserIdle,
    UserReturned,
    VisionObserved,
    VisionStateChanged,
    VoiceTranscript,
)
from chibi.gaming.games_db import GameDatabase, title_from_exe
from chibi.memory.repositories import PermissionRepository
from chibi.personality.phrases import DEFAULT_PATH, PhraseBank
from chibi.privacy.permissions import PermissionManager
from chibi.vision.analysis import WORK_EVENTS, WORK_SCENES, Observation
from chibi.vision.analyzer import CANCELLED_USER_MESSAGE
from chibi.vision.capture_win import Frame
from chibi.vision.integration import (
    LOOK_FAILED_MESSAGE,
    LOOK_SKIP_MESSAGES,
    MEMORY_EVENT_FORMS,
    OBSERVER_AI_OFF,
    OBSERVER_BUDGET,
    OBSERVER_MANUAL,
    OBSERVER_NO_AI_LINES,
    OWN_WINDOW_MESSAGE,
    OWN_WINDOW_RETRY_MS,
    REASON_LOCAL_ONLY,
    REASON_NO_CAPTURE,
    REASON_NO_CLOUD_VISION,
    REASON_OFF,
    SKIP_AWAY,
    SKIP_BROWSER,
    SKIP_MUTED,
    SKIP_NOT_GAME,
    SKIP_NOT_LISTED,
    SKIP_NOTES,
    SKIP_PRIVATE,
    SKIP_SYSTEM,
    SYSTEM_PROCESSES,
    VOICE_PRESENCE_SECONDS,
    VisionIntegration,
    is_look_request,
    observer_ai_reason,
    reactions_per_hour,
    session_summary,
)
from chibi.vision.service import CONVERSATION_GRACE_SECONDS, AnalyzerError, VisionMode, VisionState
from tests.personality.test_phrase_bank import (
    FANCY_APOSTROPHES,
    GENDERED_BARE,
    GENDERED_USER,
    MIXED_SCRIPT_WORD,
    PORTRAITS,
    PREPOSITION_BEFORE_TITLE,
    RUSSIAN_ONLY,
    RUSSISMS,
)

OWN_PID = 4242
OWN_HWND = 9001
TITLE = "S.T.A.L.K.E.R. 2: Серце Чорнобиля"
GAME = ForegroundInfo(pid=10, process_name="stalker2-win64-shipping.exe", hwnd=111)
SECRET_DESCRIPTION = "На екрані видно чат із Петром"


def obs(event="none", scene="gameplay", confidence=0.9, description="Гравець біжить полем", spoiler=False):
    return Observation(scene=scene, event=event, confidence=confidence, interesting=True,
                       suggested_emotion="curious", description=description, spoiler=spoiler)


class FakeBackend:
    def __init__(self):
        self.fg: ForegroundInfo | None = GAME
        self.read_title: list[bool] = []

    def is_supported(self):
        return True

    def foreground(self, read_title):
        self.read_title.append(read_title)
        return self.fg

    def idle_seconds(self):
        return 0.0

    def running_processes(self):
        return set()


class FakeCapture:
    def __init__(self):
        self.calls: list[int] = []
        self.rng = random.Random(7)

    def capture(self, hwnd):
        self.calls.append(hwnd)
        # Different pixels each time: the service skips frames that look the same.
        return Frame(64, 64, self.rng.randbytes(64 * 64 * 4), hwnd, 0.0, "fake")


class Worker:
    def __init__(self):
        self.jobs = []
        self.shutdowns = 0

    def submit(self, job, priority=1):
        self.jobs.append(job)

    def run(self):
        while self.jobs:
            self.jobs.pop(0)()

    def shutdown(self, timeout=3.0):
        self.shutdowns += 1


class FakeAnalyzer:
    def __init__(self, harness, local, allowed):
        self.h = harness
        self.local = local
        self.allowed = allowed
        self.calls: list[str] = []
        self.urgent: list[bool] = []
        self.cancelled = 0
        self.closed = False

    @property
    def is_local(self):
        return self.local

    def analyze(self, image, media_type, app_title, previous, *, urgent=False, wanted=None):
        self.calls.append(app_title)
        self.urgent.append(urgent)
        result = self.h.results.pop(0) if len(self.h.results) > 1 else self.h.results[0]
        if isinstance(result, Exception):
            raise result
        return result

    def cancel_current(self):
        self.cancelled += 1

    def close(self):
        self.closed = True


class FakeComposer:
    def __init__(self):
        self.calls: list[tuple[tuple[str, ...], bool, dict]] = []

    def compose(self, *keys, solicited=False, **variables):
        self.calls.append((keys, solicited, variables))
        return True


class FakeMemory:
    def __init__(self):
        self.added: list[tuple[str, str, str]] = []

    def add(self, kind, text, source="chat", importance=0.5):
        self.added.append((kind, text, source))
        return object()


class H:
    def __init__(self, qapp, db, bus, clock, monkeypatch, *, mode="game", provider="ollama",
                 allowed_games=("stalker2",), allowed_processes=(), grant=True, local_only=False,
                 scope="allowed", commentary="events", titles=False, observer_ai=None):
        monkeypatch.setattr(permissions_module, "AVAILABLE", frozenset(Permission))
        self.excluded: list[int] = []
        monkeypatch.setattr(integration_module, "exclude_from_capture",
                            lambda hwnd: self.excluded.append(hwnd) or True)
        self.qapp, self.bus, self.clock = qapp, bus, clock
        self.permissions = PermissionManager(PermissionRepository(db), bus, clock)
        self.settings = Settings()
        self.settings.ai.provider = provider
        self.settings.vision.mode = mode
        self.settings.vision.allowed_games = list(allowed_games)
        self.settings.vision.allowed_processes = list(allowed_processes)
        self.settings.vision.local_only = local_only
        self.settings.vision.scope = scope
        self.settings.vision.commentary = commentary
        self.games = GameDatabase.load()
        self.backend = FakeBackend()
        self.capture = FakeCapture()
        self.worker = Worker()
        self.composer = FakeComposer()
        self.memory = FakeMemory()
        self.notices: list[str] = []
        self.spoken: list[int] = []
        self.quiet = False
        self.results: list[Observation | Exception] = [obs()]
        self.analyzers: list[FakeAnalyzer] = []
        self.own = {OWN_HWND}
        self.states: list[VisionStateChanged] = []
        self.observed: list[VisionObserved] = []
        bus.subscribe(VisionStateChanged, self.states.append)
        bus.subscribe(VisionObserved, self.observed.append)
        if grant:
            self.permissions.grant(Permission.SCREEN_CAPTURE)
        if titles:
            self.permissions.grant(Permission.APP_METADATA)
            self.permissions.grant(Permission.WINDOW_TITLES)
        self.integ = VisionIntegration(
            bus, lambda: self.settings, self.permissions, object(), self.composer, clock,
            games=lambda: self.games, memory=self.memory, quiet=lambda: self.quiet,
            on_spoken=lambda: self.spoken.append(1), notify=self.notices.append,
            backend=self.backend, capture=self.capture, worker=self.worker,
            own_windows=lambda: self.own, own_pid=OWN_PID, analyzer_factory=self._factory,
            observer_ai=observer_ai,
        )
        self.dispatch()

    def _factory(self, ai, post, ai_settings, vision_settings, allowed):
        analyzer = FakeAnalyzer(self, not is_cloud(ai_settings), allowed)
        self.analyzers.append(analyzer)
        return analyzer

    @property
    def analyzer(self) -> FakeAnalyzer:
        return self.analyzers[-1]

    def dispatch(self):
        for _ in range(5):
            self.qapp.processEvents()
            self.bus.dispatch_pending()

    def change(self, section="vision", **values):
        target = getattr(self.settings, section)
        for name, value in values.items():
            setattr(target, name, value)
        self.bus.publish(SettingsChanged(section=section))
        self.dispatch()

    def step(self, seconds=1.0):
        self.clock.advance(seconds)
        self.integ.tick()
        self.worker.run()
        self.dispatch()

    def first_frame(self):
        self.step(1)  # sees the game; first frame ~2 s later
        self.step(3)

    def confirm_frame(self):
        self.step(3.5)

    def close(self):
        self.integ.shutdown()


@pytest.fixture
def make(qapp, db, bus, clock, monkeypatch):
    made: list[H] = []

    def factory(**kwargs) -> H:
        h = H(qapp, db, bus, clock, monkeypatch, **kwargs)
        made.append(h)
        return h

    yield factory
    for h in made:
        h.close()


# --- what may be captured ---------------------------------------------------------------------------


def info(process, *, pid=20, hwnd=222, title=None, exe_path=None):
    return ForegroundInfo(pid=pid, process_name=process, exe_path=exe_path, title=title, hwnd=hwnd)


def test_an_allowed_game_resolves_to_its_database_title(make):
    h = make()
    target = h.integ.resolve_target(GAME)
    assert target is not None
    assert (target.hwnd, target.title, target.is_game) == (111, TITLE, True)


def test_unlisted_private_and_own_windows_are_never_targets(make):
    h = make(allowed_games=("stalker2",), allowed_processes=("keepass.exe", "chrome.exe"))
    h.change("activity", excluded_processes=["stalker2-win64-shipping.exe"])
    assert h.integ.resolve_target(GAME) is None  # the user made the game private
    h.change("activity", excluded_processes=[])
    assert h.integ.resolve_target(GAME) is not None
    assert h.integ.resolve_target(info("cs2.exe")) is None  # a game, but not allowed
    assert h.integ.resolve_target(info("notepad.exe")) is None  # not listed at all
    assert h.integ.resolve_target(info("keepass.exe")) is None  # built-in private, even if listed
    # Our own process and our own windows, even when they look like an allowed game.
    assert h.integ.resolve_target(info("stalker2-win64-shipping.exe", pid=OWN_PID)) is None
    h.integ.exclude_windows()
    assert h.integ.resolve_target(info("stalker2-win64-shipping.exe", hwnd=OWN_HWND)) is None
    assert h.integ.resolve_target(info("stalker2-win64-shipping.exe", hwnd=0)) is None  # no window


def test_private_titles_and_private_paths_block_an_allowed_app(make):
    h = make(allowed_games=(), allowed_processes=("chrome.exe", "firefox.exe"))
    target = h.integ.resolve_target(info("chrome.exe", title="YouTube — Google Chrome"))
    assert target is not None and target.title == "Chrome" and not target.is_game
    assert h.integ.resolve_target(info("chrome.exe", title="Monobank — особистий кабінет")) is None
    assert h.integ.resolve_target(info("chrome.exe", title="Нове вікно (інкогніто)")) is None
    tor = r"C:\Users\u\Tor Browser\Browser\firefox.exe"
    assert h.integ.resolve_target(info("firefox.exe", exe_path=tor)) is None


def test_exe_ids_and_allowed_apps_that_are_games(make):
    h = make(allowed_games=("exe:indie.exe",), allowed_processes=("cs2.exe",))
    indie = h.integ.resolve_target(info("indie.exe"))
    assert indie is not None and indie.title == "Indie" and indie.is_game
    cs2 = h.integ.resolve_target(info("cs2.exe"))
    assert cs2 is not None and cs2.title == "Counter-Strike 2" and cs2.is_game


def test_window_titles_are_read_only_with_their_permission(make):
    h = make()
    h.first_frame()
    assert h.backend.read_title and not any(h.backend.read_title)
    h.permissions.grant(Permission.APP_METADATA)
    h.permissions.grant(Permission.WINDOW_TITLES)
    h.backend.read_title.clear()
    h.step(100)
    assert h.backend.read_title and all(h.backend.read_title)


def test_her_own_windows_are_excluded_from_capture_once_each(make):
    h = make()
    h.step(1)
    h.step(1)
    assert h.excluded == [OWN_HWND]
    h.own = {OWN_HWND, 9002}  # Qt recreated a native window
    h.step(1)
    assert h.excluded == [OWN_HWND, 9002]


# --- the pipeline -----------------------------------------------------------------------------------


def test_watching_state_and_observations_are_published(make):
    h = make()
    h.step(1)
    assert h.states[-1].state == VisionState.WATCHING.value
    assert h.states[-1].app_title == TITLE and h.states[-1].reason == "watching"
    h.step(3)
    assert h.capture.calls == [111]
    assert h.analyzer.calls == [TITLE]
    (seen,) = h.observed
    assert (seen.app_title, seen.scene, seen.event, seen.react) == (TITLE, "gameplay", "none", False)


def test_reactions_go_through_the_composer_only_when_the_policy_says_react(make):
    h = make()
    h.results = [obs("player_death", "death_screen")]
    h.first_frame()
    assert h.observed[-1].event == "player_death" and not h.observed[-1].react  # needs a second frame
    assert h.composer.calls == []
    h.confirm_frame()
    assert h.observed[-1].react and h.observed[-1].confirmed
    assert h.composer.calls == [(("vision.player_death", "vision.generic"), False, {"game": TITLE})]
    # Nothing notable afterwards: no line.
    h.results = [obs("none")]
    h.step(200)
    assert len(h.composer.calls) == 1


def test_quiet_mode_and_zero_frequency_keep_her_silent(make):
    h = make()
    h.quiet = True
    h.results = [obs("victory", "victory_screen")]
    h.first_frame()
    h.confirm_frame()
    assert h.observed[-1].confirmed and not h.observed[-1].react
    h.quiet = False
    h.change("gaming", reaction_frequency=0)
    assert h.integ.service.policy.config.reactions_per_hour == 0
    h.step(300)
    h.confirm_frame()
    assert h.composer.calls == []


def test_settings_feed_the_policy(make):
    h = make()
    h.change("behavior", max_unsolicited_per_hour=30)
    h.change("gaming", reaction_frequency=100, hints_and_spoilers=True)
    h.change("vision", min_confidence=0.8)
    config = h.integ.service.policy.config
    assert (config.reactions_per_hour, config.allow_spoilers, config.min_confidence) == (12, True, 0.8)
    # The hard cap on unsolicited lines ("0 — мовчить") applies to vision too.
    h.change("behavior", max_unsolicited_per_hour=4)
    assert h.integ.service.policy.config.reactions_per_hour == 4
    h.change("behavior", max_unsolicited_per_hour=0)
    assert h.integ.service.policy.config.reactions_per_hour == 0
    # The shared budget is scaled by "how often she comments": 6 x 0.4 -> 2 lines an hour,
    # and what she sees must not get more than that on its own.
    h.change("behavior", max_unsolicited_per_hour=6)
    h.change("personality", commentary_frequency=0)
    assert h.integ.service.policy.config.reactions_per_hour == 2


def test_spoken_reactions_count_against_the_shared_budget(make):
    h = make()
    h.results = [obs("victory", "victory_screen")]
    h.first_frame()
    h.confirm_frame()
    assert len(h.composer.calls) == 1 and h.spoken == [1]
    # An answer the user asked for is not unsolicited.
    h.change(mode="manual")
    h.results = [obs("none", "menu")]
    assert h.integ.request()
    h.worker.run()
    h.dispatch()
    assert len(h.composer.calls) == 2 and h.spoken == [1]


def test_permission_revoke_stops_everything_at_once(make):
    h = make()
    h.first_frame()
    assert h.integ.recent_observations() and h.integ.latest_observation()
    analyzer = h.analyzer
    h.clock.advance(100)
    h.integ.tick()  # a capture job is queued...
    assert h.worker.jobs
    h.permissions.revoke(Permission.SCREEN_CAPTURE)
    h.worker.run()  # ...and finds the permission gone before it captures anything
    assert h.capture.calls == [111]
    h.dispatch()
    assert h.integ.service.state == VisionState.OFF
    assert not h.integ.timer_active
    assert analyzer.cancelled >= 1 and analyzer.closed
    assert h.integ.recent_observations() == [] and h.integ.latest_observation() is None
    assert h.states[-1].state == VisionState.OFF.value
    h.step(100)
    assert h.capture.calls == [111]


def test_a_result_arriving_after_revoke_is_dropped(make):
    h = make()
    h.step(1)
    h.clock.advance(3)
    h.integ.tick()
    h.worker.run()  # analysed on the worker; the result waits for the main thread
    h.permissions.revoke(Permission.SCREEN_CAPTURE)
    h.dispatch()
    assert h.observed == [] and h.integ.latest_observation() is None


def test_kill_switch_stops_and_forgets(make):
    h = make()
    h.results = [obs("player_death", "death_screen")]
    h.bus.publish(GameStarted(game_id="stalker2", title=TITLE, process_name=GAME.process_name))
    h.dispatch()
    h.first_frame()
    h.confirm_frame()
    h.permissions.stop_all_observation()
    h.dispatch()
    assert h.integ.service.state == VisionState.OFF and not h.integ.timer_active
    assert h.integ.recent_observations() == [] and h.integ.latest_observation() is None
    # The session's counts were forgotten too: no memory at the end of the game.
    h.bus.publish(GameEnded(game_id="stalker2", title=TITLE, duration_seconds=600))
    h.dispatch()
    assert h.memory.added == []


# --- cloud ------------------------------------------------------------------------------------------


def test_cloud_model_needs_cloud_vision_and_respects_local_only(make):
    h = make(provider="anthropic", local_only=True)
    assert h.analyzers == [] and not h.integ.timer_active
    assert h.integ.status_view().blocked == REASON_LOCAL_ONLY
    h.change("vision", local_only=False)
    assert h.analyzers == [] and h.integ.status_view().blocked == REASON_NO_CLOUD_VISION
    h.permissions.grant(Permission.CLOUD_AI)
    h.permissions.grant(Permission.CLOUD_VISION)
    h.dispatch()
    assert h.analyzers and not h.analyzer.is_local and h.integ.timer_active
    assert h.integ.status_view().local is False
    h.first_frame()
    assert h.analyzer.calls == [TITLE]
    analyzer = h.analyzer
    assert analyzer.allowed() is True
    h.permissions.revoke(Permission.CLOUD_VISION)
    assert analyzer.allowed() is False  # the last gate closes before the event is even delivered
    h.dispatch()
    assert analyzer.closed and analyzer.cancelled >= 1 and not h.integ.timer_active
    h.step(200)
    assert analyzer.calls == [TITLE]


def test_the_last_gate_follows_the_current_ai_settings_not_the_analyzer(make):
    # The AI service switches provider the moment settings are applied, before our
    # SettingsChanged arrives: a frame captured for a local model must not reach the cloud.
    h = make(provider="ollama")
    analyzer = h.analyzer
    assert analyzer.is_local and analyzer.allowed() is True
    h.settings.ai.provider = "anthropic"  # applied, event not delivered yet
    assert analyzer.allowed() is False
    h.permissions.grant(Permission.CLOUD_AI)
    h.permissions.grant(Permission.CLOUD_VISION)
    assert analyzer.allowed() is True
    h.settings.vision.local_only = True
    assert analyzer.allowed() is False


def test_an_app_that_becomes_private_is_forgotten(make):
    h = make(mode="auto", allowed_processes=("obs64.exe",))
    h.bus.publish(GameStarted(game_id="stalker2", title=TITLE, process_name=GAME.process_name))
    h.dispatch()
    h.results = [obs("player_death", "death_screen")]
    h.first_frame()
    h.confirm_frame()
    h.backend.fg = info("obs64.exe", hwnd=333)
    h.results = [obs("none")]
    h.step(300)  # a new window settles first
    h.step(3)
    assert [s.app_title for s in h.integ.recent_observations()] == ["Obs 64", TITLE, TITLE]
    h.change("activity", excluded_processes=["stalker2-win64-shipping.exe"])
    assert [s.app_title for s in h.integ.recent_observations()] == ["Obs 64"]
    assert len(h.integ.service.recent) == 0
    end_session(h)
    assert h.memory.added == []  # its counts went too


@pytest.mark.parametrize("section, values", [
    ("activity", {"excluded_processes": ["stalker2-win64-shipping.exe"]}),  # made private
    ("vision", {"allowed_games": []}),  # taken off the allowed list
])
def test_a_frame_in_flight_when_its_app_stops_being_allowed_is_dropped(make, section, values):
    # The model is still looking at the frame when the user changes their mind:
    # the request is abandoned and the answer never becomes something she knows or says.
    h = make(mode="auto")
    analyzer = h.analyzer
    original = analyzer.analyze

    def analyze(*args, **kwargs):
        h.change(section, **values)
        return original(*args, **kwargs)

    analyzer.analyze = analyze
    h.results = [obs("victory", "victory_screen", description=SECRET_DESCRIPTION)]
    h.first_frame()
    assert h.capture.calls
    assert h.observed == [] and h.composer.calls == []
    assert h.integ.recent_observations() == [] and h.integ.latest_observation() is None
    assert analyzer.cancelled >= 1  # a request still queued for the AI never leaves
    h.step(300)
    assert len(h.capture.calls) == 1


def test_what_she_saw_reaches_a_cloud_brain_only_under_the_cloud_vision_rules(make):
    # Seen by a local model; then the AI page switches to a cloud provider (applied at
    # once, the settings event still queued).  The description is screen content: it
    # follows the same rules as a frame.
    h = make(provider="ollama")
    h.first_frame()
    assert h.integ.latest_observation() == "gameplay/none: Гравець біжить полем"
    h.permissions.grant(Permission.CLOUD_AI)
    h.settings.ai.provider = "anthropic"
    assert h.integ.latest_observation() is None  # no CLOUD_VISION
    h.permissions.grant(Permission.CLOUD_VISION)
    assert h.integ.latest_observation() is not None
    h.settings.vision.local_only = True
    assert h.integ.latest_observation() is None
    h.settings.ai.provider = "ollama"
    assert h.integ.latest_observation() is not None


# --- game memory ------------------------------------------------------------------------------------


def play_session(h: H, *events: str) -> None:
    h.bus.publish(GameStarted(game_id="stalker2", title=TITLE, process_name=GAME.process_name))
    h.dispatch()
    for event in events:
        h.results = [obs(event, description=SECRET_DESCRIPTION)]
        h.step(300)
        h.confirm_frame()


def end_session(h: H, **kwargs) -> None:
    h.bus.publish(GameEnded(game_id="stalker2", title=kwargs.pop("title", TITLE), duration_seconds=900,
                            **kwargs))
    h.dispatch()


def test_game_memory_is_written_once_with_counts_only(make):
    h = make()
    h.step(1)
    play_session(h, "player_death", "player_death", "victory", "loot")
    end_session(h)
    assert h.memory.added == [("game_event", f"{TITLE}: 2 смерті, 1 перемога за сесію", "game")]
    assert SECRET_DESCRIPTION not in h.memory.added[0][1]
    end_session(h)
    assert len(h.memory.added) == 1


def test_a_death_screen_seen_in_many_frames_is_one_death(make):
    h = make()
    h.step(1)
    h.bus.publish(GameStarted(game_id="stalker2", title=TITLE, process_name=GAME.process_name))
    h.results = [obs("player_death", "death_screen")]
    h.dispatch()
    h.step(3)
    for _ in range(4):
        h.confirm_frame()
    end_session(h)
    assert h.memory.added == [("game_event", f"{TITLE}: 1 смерть за сесію", "game")]


@pytest.mark.parametrize("case", ["no_events", "forgotten", "no_remember_facts"])
def test_no_game_memory(make, case):
    h = make()
    h.step(1)
    if case == "no_remember_facts":
        h.settings.ai.remember_facts = False
    play_session(h, *(() if case == "no_events" else ("victory",)))
    if case == "forgotten":
        end_session(h, title="", forgotten=True)
        end_session(h)  # its counts are gone, not waiting for the next end
    else:
        end_session(h)
    assert h.memory.added == []


def test_session_summary_forms():
    assert session_summary("Гра", Counter(player_death=1)) == "Гра: 1 смерть за сесію"
    assert session_summary("Гра", Counter(player_death=5, victory=3, boss_fight=2)) == \
        "Гра: 5 смертей, 3 перемоги, 2 бої з босами за сесію"
    assert session_summary("Гра", Counter(level_up=21, achievement=11, new_area=4)) == \
        "Гра: 21 новий рівень, 11 досягнень, 4 нові локації за сесію"
    assert set(MEMORY_EVENT_FORMS) <= {"player_death", "victory", "boss_fight", "level_up", "achievement",
                                       "new_area"}


# --- the brain's observation ---------------------------------------------------------------------------


def test_latest_observation_is_fresh_for_two_minutes(make):
    h = make()
    assert h.integ.latest_observation() is None
    h.first_frame()
    assert h.integ.latest_observation() == "gameplay/none: Гравець біжить полем"
    h.clock.advance(119)
    assert h.integ.latest_observation() is not None
    h.clock.advance(2)
    assert h.integ.latest_observation() is None


def test_spoilers_never_reach_the_brain_unless_allowed(make):
    h = make()
    h.results = [obs("other", "cutscene", description="Головний лиходій — батько героя", spoiler=True)]
    h.first_frame()
    assert h.integ.latest_observation() == "cutscene/other"
    h.change("gaming", hints_and_spoilers=True)
    assert h.integ.latest_observation() == "cutscene/other: Головний лиходій — батько героя"


def test_clear_history_forgets_observations(make):
    h = make()
    h.first_frame()
    (seen,) = h.integ.recent_observations()
    assert (seen.app_title, seen.scene, seen.event) == (TITLE, "gameplay", "none")
    assert not hasattr(seen, "description")  # the settings list never shows model text
    h.integ.clear_history()
    assert h.integ.recent_observations() == [] and h.integ.latest_observation() is None
    assert len(h.integ.service.recent) == 0


# --- timer -------------------------------------------------------------------------------------------


def test_no_timer_while_off_or_not_permitted(make):
    h = make(mode="off")
    assert not h.integ.timer_active
    h.change(mode="game")
    assert h.integ.timer_active
    h.change(mode="off")
    assert not h.integ.timer_active and h.integ.service.state == VisionState.OFF


def test_timer_follows_the_capture_permission_and_the_ai(make):
    h = make(grant=False)
    assert not h.integ.timer_active
    assert h.integ.status_view().blocked == REASON_NO_CAPTURE
    h.permissions.grant(Permission.SCREEN_CAPTURE)
    h.dispatch()
    assert h.integ.timer_active and h.integ.service.mode == VisionMode.GAME
    h.change("ai", provider="none")
    assert not h.integ.timer_active


def test_unrelated_settings_do_not_reset_the_service(make):
    h = make()
    first = h.analyzer
    h.change("gaming", reaction_frequency=80)
    h.change("character", name="Ліна")
    assert h.analyzers == [first]
    h.change("vision", interval_seconds=60)
    assert h.integ.service.interval == 60.0 and h.analyzers == [first]
    h.change("ai", model="llava")
    assert len(h.analyzers) == 2 and first.closed


def test_load_changes_the_pace_only_when_adapting(make):
    h = make()
    h.bus.publish(SystemLoadChanged(high=True, cpu_percent=95.0))
    h.dispatch()
    assert h.integ.service.high_load
    h.change(adapt_to_load=False)
    assert not h.integ.service.high_load


# --- "подивись на екран" --------------------------------------------------------------------------------


def test_request_answers_even_when_nothing_happens(make):
    h = make(mode="manual")
    h.results = [obs("none", "menu")]
    assert h.integ.request()
    h.worker.run()
    h.dispatch()
    assert h.composer.calls == [(("vision.look.menu", "vision.look"), True, {"game": TITLE})]


def test_request_waits_for_the_confirmation_frame(make):
    h = make(mode="manual")
    h.results = [obs("player_death", "death_screen")]
    assert h.integ.request()
    h.worker.run()
    h.dispatch()
    assert h.composer.calls == []
    h.confirm_frame()
    assert h.composer.calls == [(("vision.player_death", "vision.generic"), True, {"game": TITLE})]


def test_request_whose_confirmation_frame_disagrees_is_answered_at_once(make):
    h = make(mode="manual")
    h.results = [obs("player_death", "death_screen"), obs("victory", "victory_screen")]
    assert h.integ.request()
    h.worker.run()
    h.dispatch()
    h.confirm_frame()
    # Answered now (not a minute later), and never as if either event were certain.
    assert h.composer.calls == [(("vision.look.victory_screen", "vision.look"), True, {"game": TITLE})]
    h.step(200)
    assert len(h.composer.calls) == 1 and h.notices == []


def test_request_with_her_own_window_in_front_retries_once(make):
    h = make(mode="manual")
    h.backend.fg = info("python.exe", pid=OWN_PID, hwnd=5)
    assert h.integ.request()
    assert h.integ._retry_timer.isActive() and h.integ._retry_timer.interval() == OWN_WINDOW_RETRY_MS == 2000
    assert h.capture.calls == []
    h.backend.fg = GAME  # the user switched back to the game
    h.integ._retry_timer.setInterval(1)
    deadline = time.monotonic() + 1.0
    while not h.worker.jobs and time.monotonic() < deadline:
        QCoreApplication.processEvents()
    h.worker.run()
    h.dispatch()
    assert h.capture.calls == [111] and h.composer.calls


def test_request_gives_up_if_her_window_is_still_in_front(make):
    h = make(mode="manual")
    h.backend.fg = info("python.exe", pid=OWN_PID, hwnd=5)
    assert h.integ.request()
    h.integ._retry_timer.setInterval(1)
    deadline = time.monotonic() + 1.0
    while not h.notices and time.monotonic() < deadline:
        QCoreApplication.processEvents()
    assert h.notices == [OWN_WINDOW_MESSAGE] and h.capture.calls == []


def test_a_request_that_never_gets_an_answer_is_explained(make):
    h = make(mode="auto")
    assert h.integ.request()  # the job is queued but never comes back
    h.clock.advance(h.settings.ai.request_timeout_seconds + 26)
    h.integ.tick()
    assert h.notices == [LOOK_FAILED_MESSAGE]
    h.integ.tick()
    assert h.notices == [LOOK_FAILED_MESSAGE]


def test_window_exclusion_errors_never_break_the_tick(make, monkeypatch):
    import ctypes

    def broken(hwnd):
        raise ctypes.ArgumentError("int too long to convert")

    h = make()
    monkeypatch.setattr(integration_module, "exclude_from_capture", broken)
    h.step(1)
    assert h.integ.service.state == VisionState.WATCHING


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"mode": "off"}, REASON_OFF),
    ({"grant": False}, REASON_NO_CAPTURE),
])
def test_request_explains_why_it_cannot_look(make, kwargs, message):
    h = make(**kwargs)
    assert not h.integ.request()
    assert h.notices == [message]


def test_request_for_an_unlisted_window_is_explained(make):
    h = make(mode="auto")
    h.backend.fg = info("notepad.exe")
    assert not h.integ.request()
    assert len(h.notices) == 1 and "не в списку" in h.notices[0]


@pytest.mark.parametrize("text", ["Подивись на екран!", "мія, подивися на мій екран", "Що ти бачиш?",
                                  "а що бачиш", "глянь на екран, будь ласка", "що в мене на екрані?"])
def test_look_requests_are_recognised(text):
    assert is_look_request(text)


@pytest.mark.parametrize("text", ["привіт", "подивись, яка погода", "екран погас", "я бачу кота"])
def test_other_messages_are_not_look_requests(text):
    assert not is_look_request(text)


# --- notices, status, games, shutdown --------------------------------------------------------------------


def test_automatic_errors_are_not_repeated(make):
    h = make()
    message = "Не можу роздивитися екран: Ollama не запущена."
    error = AnalyzerError("down", user_message=message, retry_after=30)
    h.results = [error]
    h.first_frame()
    assert h.notices == [message]
    h.change(interval_seconds=40)  # resets the service, which would report it again
    h.step(100)
    h.step(100)
    assert len(h.notices) == 1
    h.clock.advance(700)
    h.change(interval_seconds=50)
    h.step(5)
    assert len(h.notices) == 2


def test_status_view(make):
    h = make(allowed_games=())
    view = h.integ.status_view()
    assert view.mode == "game" and view.local is True and view.blocked == ""
    assert "порожній" in view.note
    h.change(allowed_games=["stalker2"], model="llava")
    h.step(1)
    view = h.integ.status_view()
    assert view.state == "watching" and view.app_title == TITLE and view.note == ""


def test_game_choices_are_sorted_and_include_custom_games(make):
    h = make()
    h.games = GameDatabase.load([{"process": "zzz.exe", "title": "Аркада"}])
    choices = h.integ.game_choices()
    titles = [title for _id, title in choices]
    assert titles == sorted(titles, key=str.casefold)
    assert ("custom:zzz.exe", "Аркада") in choices and ("stalker2", TITLE) in choices


def test_reactions_per_hour_mapping():
    assert [reactions_per_hour(v) for v in (0, 50, 100, -5, 500)] == [0, 6, 12, 0, 12]


def test_shutdown_stops_worker_timer_and_events(make):
    h = make()
    h.first_frame()
    analyzer = h.analyzer
    h.integ.shutdown()
    h.dispatch()
    assert h.worker.shutdowns == 1 and not h.integ.timer_active and analyzer.closed
    assert h.states[-1].state == VisionState.OFF.value
    assert h.integ.recent_observations() == []
    h.change(mode="auto")
    assert not h.integ.timer_active and len(h.analyzers) == 1
    assert not h.integ.request()
    h.integ.shutdown()  # idempotent
    assert h.worker.shutdowns == 1


# --- the phrase file --------------------------------------------------------------------------------------

VISION_FILE = DEFAULT_PATH.with_name("phrases_vision_uk.json")
REQUIRED_KEYS = ("vision.player_death", "vision.victory", "vision.boss_fight", "vision.level_up",
                 "vision.achievement", "vision.new_area", "vision.low_health", "vision.loot",
                 "vision.generic")


def test_vision_phrases_load_and_cover_every_key():
    raw = json.loads(VISION_FILE.read_text(encoding="utf-8"))
    entries = [entry for items in raw["phrases"].values() for entry in items]
    bank = PhraseBank.from_dict(raw)
    assert len(bank) == len(entries)  # nothing skipped
    for key in (*REQUIRED_KEYS, "vision.look"):
        assert len(bank.phrases(key)) >= 6, key
        assert any(p.tone != "sarcastic" for p in bank.phrases(key)), key
        assert any("game" not in p.requires for p in bank.phrases(key)), key
    texts = [p.text for p in bank.phrases("vision.player_death")]
    assert "Ахах, та що ж ти робиш?! Навіть я б краще зіграла!" in texts
    assert "Гаразд-гаразд, наступного разу точно вийде!" in texts
    # Merged into the default bank automatically.
    assert PhraseBank.load().has("vision.generic")


def test_vision_phrases_follow_the_language_rules():
    raw = json.loads(VISION_FILE.read_text(encoding="utf-8"))
    ids = Counter()
    for key, items in raw["phrases"].items():
        assert key.startswith("vision.")
        for entry in items:
            text = entry["text"]
            ids[entry["id"]] += 1
            assert entry["id"].startswith(key + ".")
            assert len(text) <= 140 and text[-1] in ".!?…»)", entry["id"]
            for rule in (GENDERED_USER, GENDERED_BARE, RUSSIAN_ONLY, FANCY_APOSTROPHES, RUSSISMS,
                         MIXED_SCRIPT_WORD, PREPOSITION_BEFORE_TITLE):
                assert not rule.search(text), (entry["id"], rule.pattern[:30])
            assert entry["portrait"] in PORTRAITS, entry["id"]
            assert ("{game}" in text) == ("game" in entry.get("requires", [])), entry["id"]
            for hint in ("натисни", "спробуй піти", "ключ від", "відповідь —"):
                assert hint not in text.lower(), entry["id"]  # no gameplay hints
    assert max(ids.values()) == 1


def test_vision_phrases_answer_about_every_work_scene_and_event():
    bank = PhraseBank.from_dict(json.loads(VISION_FILE.read_text(encoding="utf-8")))
    keys = [f"vision.look.{name}" for name in (*sorted(WORK_SCENES), *sorted(WORK_EVENTS), "work")]
    for key in keys:
        phrases = bank.phrases(key)
        assert 3 <= len(phrases) <= 5, key
        assert sum("app" not in p.requires for p in phrases) >= 2, key  # works without a title
        assert all("game" not in p.requires for p in phrases), key  # never "гра" about a document


# --- "foreground" scope: any window in front ------------------------------------------------------------


BROWSER_TAB = info("chrome.exe", hwnd=777, title="Документація Python — Google Chrome")


def grant_titles(h: H) -> None:
    h.permissions.grant(Permission.APP_METADATA)
    h.permissions.grant(Permission.WINDOW_TITLES)
    h.dispatch()


def test_any_window_in_front_resolves_with_a_friendly_name(make):
    h = make(mode="auto", scope="foreground", allowed_games=("exe:indie.exe",))
    notepad = h.integ.resolve_target(info("notepad.exe"))
    assert notepad is not None and (notepad.title, notepad.is_game) == ("Блокнот", False)
    assert h.integ.resolve_target(info("code.exe")).title == "VS Code"
    tool = h.integ.resolve_target(info("mytool.exe"))
    assert tool is not None and (tool.title, tool.is_game) == (title_from_exe("mytool.exe"), False)
    game = h.integ.resolve_target(GAME)  # a game she knows, listed or not
    assert game is not None and (game.title, game.is_game) == (TITLE, True)
    indie = h.integ.resolve_target(info("indie.exe"))
    assert indie is not None and indie.is_game  # the user's list still says what is a game


def test_any_window_never_means_private_own_or_system_windows(make):
    h = make(mode="auto", scope="foreground")
    assert h.integ.resolve_target(info("keepass.exe")) is None
    assert h.integ.resolve_target(info("telegram.exe")) is None
    assert h.integ.resolve_target(info("code.exe", pid=OWN_PID)) is None
    h.integ.exclude_windows()
    assert h.integ.resolve_target(info("code.exe", hwnd=OWN_HWND)) is None
    for process in ("lockapp.exe", "logonui.exe", "taskmgr.exe", "consent.exe", "searchhost.exe",
                    "startmenuexperiencehost.exe", "shellexperiencehost.exe", "systemsettings.exe"):
        assert process in SYSTEM_PROCESSES
        assert h.integ.resolve_target(info(process)) is None, process
    h.change("activity", excluded_processes=["notepad.exe"])
    assert h.integ.resolve_target(info("notepad.exe")) is None  # the user's own exclusions


def test_explorer_only_as_a_file_window_and_only_with_titles(make):
    h = make(mode="auto", scope="foreground")
    assert h.integ.resolve_target(info("explorer.exe", title="Проєкти")) is None  # title may not be read
    grant_titles(h)
    for title in ("Program Manager", "", None):  # the desktop, the taskbar, unknown
        assert h.integ.resolve_target(info("explorer.exe", title=title)) is None, title
    target = h.integ.resolve_target(info("explorer.exe", title="Проєкти"))
    assert target is not None and target.title == "Провідник"


def test_browsers_only_while_window_titles_may_be_read(make):
    h = make(mode="auto", scope="foreground")
    h.backend.fg = BROWSER_TAB
    h.step(1)
    h.step(3)
    assert h.capture.calls == [] and h.integ.resolve_target(BROWSER_TAB) is None
    assert h.integ.status_view().skipped == SKIP_NOTES[SKIP_BROWSER]
    grant_titles(h)
    h.step(1)
    h.step(3)
    assert h.capture.calls == [777] and h.analyzer.calls == ["Google Chrome"]
    # With titles the privacy filter recognises a bank, mail, a private window.
    for title in ("Monobank — особистий кабінет", "Вхідні — Gmail", "Нове вікно (інкогніто)", "Telegram Web"):
        assert h.integ.resolve_target(info("chrome.exe", title=title)) is None, title


def test_a_work_window_is_watched_and_published_with_a_sanitised_description(make):
    h = make(mode="auto", scope="foreground", allowed_games=())
    assert h.integ.status_view().note == ""  # empty lists do not matter for this scope
    h.backend.fg = info("code.exe", hwnd=444)
    h.results = [obs("error_on_screen", "code", description="Помилка в коді, пише ivan@example.com")]
    h.first_frame()
    assert h.capture.calls == [444] and h.analyzer.calls == ["VS Code"]
    assert h.states[-1].app_title == "VS Code"
    (seen,) = h.observed
    assert (seen.app_title, seen.scene, seen.event) == ("VS Code", "code", "error_on_screen")
    assert not seen.react
    assert seen.description == "Помилка в коді, пише"
    assert h.composer.calls == []  # a work event is no game event: "events" mode stays silent


@pytest.mark.parametrize(("front", "skip"), [
    (info("keepass.exe"), SKIP_PRIVATE),
    (info("taskmgr.exe"), SKIP_SYSTEM),
    (info("chrome.exe"), SKIP_BROWSER),
])
def test_status_says_why_the_window_in_front_is_skipped(make, front, skip):
    h = make(mode="auto", scope="foreground")
    h.backend.fg = front
    h.step(1)
    view = h.integ.status_view()
    assert (view.scope, view.state, view.skipped) == ("foreground", "idle", SKIP_NOTES[skip])
    # Her own window (the settings dialog itself) does not hide the reason.
    h.backend.fg = info("python.exe", pid=OWN_PID, hwnd=5)
    h.step(1)
    assert h.integ.status_view().skipped == SKIP_NOTES[skip]
    h.backend.fg = info("code.exe")
    h.step(1)
    view = h.integ.status_view()
    assert (view.state, view.skipped, view.app_title) == ("watching", "", "VS Code")


def test_status_reasons_for_the_lists_and_game_mode(make):
    h = make(mode="auto")
    h.backend.fg = info("notepad.exe")
    h.step(1)
    view = h.integ.status_view()
    assert (view.scope, view.skipped) == ("allowed", SKIP_NOTES[SKIP_NOT_LISTED])
    h.change(scope="foreground", mode="game")
    assert h.integ.status_view().skipped == ""  # a reason from the other scope is dropped
    h.step(1)
    assert h.integ.status_view().skipped == SKIP_NOTES[SKIP_NOT_GAME]
    h.change(mode="off")
    assert h.integ.status_view().skipped == ""


@pytest.mark.parametrize(("front", "skip"), [
    (info("keepass.exe"), SKIP_PRIVATE),
    (info("taskmgr.exe"), SKIP_SYSTEM),
    (info("chrome.exe"), SKIP_BROWSER),
])
def test_request_says_exactly_why_a_window_is_skipped(make, front, skip):
    h = make(mode="manual", scope="foreground")
    h.backend.fg = front
    assert not h.integ.request()
    assert h.notices == [LOOK_SKIP_MESSAGES[skip]] and h.capture.calls == []


def test_revoking_window_titles_drops_a_browser_frame_in_flight(make):
    h = make(mode="auto", scope="foreground", titles=True)
    h.backend.fg = BROWSER_TAB
    analyzer = h.analyzer
    original = analyzer.analyze

    def analyze(*args, **kwargs):
        h.permissions.revoke(Permission.WINDOW_TITLES)
        h.dispatch()
        return original(*args, **kwargs)

    analyzer.analyze = analyze
    h.results = [obs("none", "browser", description=SECRET_DESCRIPTION)]
    h.first_frame()
    assert h.capture.calls == [777] and analyzer.cancelled >= 1
    assert h.observed == [] and h.integ.recent_observations() == []
    h.step(300)
    assert h.capture.calls == [777]


def test_the_last_check_before_publishing_reads_the_titles_permission_live(make):
    # The answer arrives on the main thread before the bus delivers the revoke.
    h = make(mode="auto", scope="foreground", titles=True)
    h.backend.fg = BROWSER_TAB
    h.step(1)
    h.clock.advance(3)
    h.integ.tick()
    h.worker.run()
    h.permissions.revoke(Permission.WINDOW_TITLES)
    h.qapp.processEvents()
    assert h.integ.recent_observations() == [] and h.integ.latest_observation() is None
    h.dispatch()
    assert h.observed == []


def test_switching_back_to_the_lists_drops_a_frame_of_an_unlisted_window(make):
    h = make(mode="auto", scope="foreground", allowed_games=())
    h.backend.fg = info("code.exe", hwnd=444)
    analyzer = h.analyzer
    original = analyzer.analyze

    def analyze(*args, **kwargs):
        h.change(scope="allowed")
        return original(*args, **kwargs)

    analyzer.analyze = analyze
    h.first_frame()
    assert h.capture.calls == [444] and analyzer.cancelled >= 1
    assert h.observed == [] and h.integ.recent_observations() == []


def test_a_cloud_answer_after_cloud_vision_was_revoked_is_dropped(make):
    h = make(provider="anthropic")
    h.permissions.grant(Permission.CLOUD_AI)
    h.permissions.grant(Permission.CLOUD_VISION)
    h.dispatch()
    h.step(1)
    h.clock.advance(3)
    h.integ.tick()
    h.worker.run()
    assert h.analyzer.calls == [TITLE]
    h.permissions.revoke(Permission.CLOUD_VISION)
    h.qapp.processEvents()  # the answer arrives before the bus delivers the revoke
    assert h.integ.recent_observations() == []
    h.dispatch()
    assert h.observed == []


def test_nothing_is_published_while_what_she_saw_may_not_reach_the_current_ai(make):
    h = make(provider="ollama")
    h.step(1)
    h.clock.advance(3)
    h.integ.tick()
    h.worker.run()
    h.permissions.grant(Permission.CLOUD_AI)
    h.settings.ai.provider = "anthropic"  # applied; the settings event is not delivered yet
    assert not h.integ.screen_data_may_leave()
    h.dispatch()
    assert h.observed == []


# --- "observer" commentary --------------------------------------------------------------------------------


def test_as_an_observer_she_never_comments_on_her_own(make):
    h = make(commentary="observer")
    h.results = [obs("player_death", "death_screen")]
    h.first_frame()
    h.confirm_frame()
    last = h.observed[-1]
    assert last.react and last.confirmed  # the event policy's hint, for the observer
    assert h.composer.calls == [] and h.spoken == []
    assert not any(seen.reacted for seen in h.integ.recent_observations())
    details = h.integ.details_for(last)
    assert details is not None and details.is_game and not details.work and not details.solicited
    h.change(commentary="events")  # back to events: she comments again
    h.results = [obs("victory", "victory_screen")]
    h.step(300)
    h.confirm_frame()
    assert h.composer.calls == [(("vision.victory", "vision.generic"), False, {"game": TITLE})]


def test_every_analysed_frame_reaches_the_observer(make):
    h = make(mode="auto", scope="foreground", commentary="observer")
    h.backend.fg = info("code.exe", hwnd=444)
    h.results = [obs("none", "code", description="Пише код"),
                 obs("tests_passed", "terminal", description="Тести пройшли")]
    h.first_frame()
    h.step(31)
    assert [(o.scene, o.event, o.description) for o in h.observed] == [
        ("code", "none", "Пише код"), ("terminal", "tests_passed", "Тести пройшли")]
    for event in h.observed:
        details = h.integ.details_for(event)
        assert details is not None and details.work and details.interesting
        assert not details.is_game and not details.solicited
    assert h.composer.calls == []
    assert h.integ.details_for(VisionObserved("VS Code", "code", "none", 0.5)) is None  # not hers


def test_as_an_observer_she_still_answers_when_asked(make):
    h = make(mode="manual", commentary="observer")
    h.results = [obs("none", "menu")]
    assert h.integ.request()
    h.worker.run()
    h.dispatch()
    assert h.composer.calls == [(("vision.look.menu", "vision.look"), True, {"game": TITLE})]
    details = h.integ.details_for(h.observed[-1])
    assert details is not None and details.solicited  # the observer must not answer it twice
    h.results = [obs("player_death", "death_screen")]
    assert h.integ.request()
    h.worker.run()
    h.dispatch()
    h.confirm_frame()
    assert h.composer.calls[-1] == (("vision.player_death", "vision.generic"), True, {"game": TITLE})


def test_published_descriptions_hide_spoilers_and_correspondence(make):
    h = make(mode="auto", scope="foreground")
    h.results = [obs("other", "cutscene", description="Лиходій — батько героя", spoiler=True)]
    h.first_frame()
    assert h.observed[-1].description == "" and h.integ.details_for(h.observed[-1]).spoiler
    h.backend.fg = info("notepad.exe", hwnd=555)
    h.results = [obs("none", "chat", description="Лист від Олени про зустріч")]
    h.step(1)
    h.step(3)
    assert (h.observed[-1].scene, h.observed[-1].description) == ("chat", "")
    assert not h.integ.details_for(h.observed[-1]).interesting
    assert h.integ.latest_observation() == "chat/none"  # nor for the brain
    h.change("gaming", hints_and_spoilers=True)
    h.integ.clear_history()
    assert h.integ.details_for(h.observed[-1]) is None  # forgotten with the rest


# --- cost: away, pace ---------------------------------------------------------------------------------------


def test_no_frames_while_the_user_is_away(make):
    h = make(mode="auto", scope="foreground")
    h.backend.fg = info("code.exe", hwnd=444)
    h.first_frame()
    assert h.integ.service.paced and h.capture.calls == [444]
    h.bus.publish(UserIdle(idle_seconds=600.0))
    h.dispatch()
    h.step(1)
    assert h.integ.service.away
    assert h.integ.status_view().skipped == SKIP_NOTES[SKIP_AWAY]
    h.step(600)
    assert h.capture.calls == [444]
    # A request still works: whoever asks is obviously there.
    assert h.integ.request()
    h.worker.run()
    h.dispatch()
    assert h.capture.calls == [444, 444]
    h.bus.publish(UserReturned(away_seconds=600.0))
    h.dispatch()
    h.step(1)
    assert not h.integ.service.away and h.integ.service.state == VisionState.WATCHING
    h.step(30)  # the regular pace again (the requested frame was just taken)
    assert len(h.capture.calls) == 3


def test_away_and_pace_only_apply_to_any_window_in_auto_mode(make):
    h = make(mode="auto")
    h.bus.publish(UserIdle(idle_seconds=600.0))
    h.dispatch()
    assert not h.integ.service.away and not h.integ.service.paced  # the lists: as before
    h.change(scope="foreground")
    assert h.integ.service.away and h.integ.service.paced
    h.change(mode="game")
    assert not h.integ.service.away and not h.integ.service.paced
    h.change(mode="auto")
    assert h.integ.service.away
    h.permissions.grant(Permission.APP_METADATA)
    h.permissions.revoke(Permission.APP_METADATA)  # the monitor stops: no UserReturned will come
    h.dispatch()
    assert not h.integ.service.away


# --- answers about work windows -----------------------------------------------------------------------------


def test_answers_about_work_windows_name_the_app_not_a_game(make):
    h = make(mode="manual", scope="foreground")
    h.backend.fg = info("notepad.exe", hwnd=444)
    h.results = [obs("none", "document")]
    assert h.integ.request()
    h.worker.run()
    h.dispatch()
    assert h.composer.calls == [
        (("vision.look.document", "vision.look.work", "vision.look"), True, {"app": "Блокнот"})]


def test_a_work_event_is_named_only_when_the_model_is_sure(make):
    h = make(mode="manual", scope="foreground")
    h.backend.fg = info("code.exe", hwnd=444)
    for confidence in (0.9, 0.3):
        h.results = [obs("error_on_screen", "code", confidence=confidence)]
        assert h.integ.request()
        h.worker.run()
        h.dispatch()
    assert [keys for keys, _solicited, _vars in h.composer.calls] == [
        ("vision.look.error_on_screen", "vision.look.code", "vision.look.work", "vision.look"),
        ("vision.look.code", "vision.look.work", "vision.look"),
    ]


def test_any_window_frame_in_flight_when_its_app_becomes_private_is_dropped(make):
    h = make(mode="auto", scope="foreground", commentary="observer")
    h.backend.fg = info("code.exe", hwnd=444)
    analyzer = h.analyzer
    original = analyzer.analyze

    def analyze(*args, **kwargs):
        h.change("activity", excluded_processes=["code.exe"])
        return original(*args, **kwargs)

    analyzer.analyze = analyze
    h.results = [obs("none", "code", description=SECRET_DESCRIPTION)]
    h.first_frame()
    assert h.capture.calls == [444] and analyzer.cancelled >= 1
    assert h.observed == [] and h.integ.latest_observation() is None
    h.step(300)
    assert h.capture.calls == [444]
    assert h.integ.status_view().skipped == SKIP_NOTES[SKIP_PRIVATE]


# --- review fixes -----------------------------------------------------------------------------------


@pytest.mark.parametrize("process", ["Microsoft.AAD.BrokerPlugin.exe", "WWAHost.exe", "UserOOBEBroker.exe",
                                     "CredentialEnrollmentManager.exe"])
def test_sign_in_windows_are_never_captured(make, process):
    h = make(mode="auto", scope="foreground")
    h.backend.fg = info(process, hwnd=777)
    h.step(1)
    h.step(5)
    assert h.integ.resolve_target(info(process)) is None
    assert h.capture.calls == [] and h.integ.status_view().skipped == SKIP_NOTES[SKIP_SYSTEM]


def test_talking_to_her_hands_free_keeps_her_looking(make):
    h = make(mode="auto", scope="foreground")
    h.backend.fg = info("code.exe", hwnd=444)
    h.first_frame()
    h.bus.publish(UserIdle(idle_seconds=600.0))  # no keyboard or mouse...
    h.bus.publish(VoiceTranscript(text="а що тут не так?", origin="named"))  # ... but talking to her
    h.dispatch()
    h.step(1)
    assert not h.integ.service.away
    h.step(30)
    assert len(h.capture.calls) == 2
    h.step(VOICE_PRESENCE_SECONDS)  # silent since: away
    assert h.integ.service.away


def test_observer_reasons():
    s = Settings()
    assert observer_ai_reason(s, (True, ""), True) == ""
    assert observer_ai_reason(s, (True, ""), False) == OBSERVER_BUDGET
    assert observer_ai_reason(s, (False, "сервіс недоступний"), False) == \
        OBSERVER_AI_OFF.format(reason="сервіс недоступний")
    s.ai.use_for_reactions = False
    assert observer_ai_reason(s, (True, ""), True) == OBSERVER_NO_AI_LINES


def test_an_observer_who_cannot_speak_takes_no_automatic_frames_and_says_why(make):
    reason = [OBSERVER_NO_AI_LINES]
    h = make(mode="auto", scope="foreground", commentary="observer", observer_ai=lambda: reason[0])
    h.backend.fg = info("code.exe", hwnd=444)
    for _ in range(40):
        h.step(1)
    assert h.capture.calls == []  # nothing would use the frames: none are paid for
    view = h.integ.status_view()
    assert view.observer_note == OBSERVER_NO_AI_LINES and view.skipped == SKIP_NOTES[SKIP_MUTED]
    assert h.integ.request()  # the user may still ask
    h.worker.run()
    h.dispatch()
    assert h.capture.calls == [444]
    reason[0] = ""
    for _ in range(40):
        h.step(1)
    assert len(h.capture.calls) >= 2
    assert h.integ.status_view().observer_note == ""
    h.change(mode="manual")
    assert h.integ.status_view().observer_note == OBSERVER_MANUAL


def test_events_commentary_never_waits_for_the_observer(make):
    h = make(mode="auto", scope="foreground", observer_ai=lambda: OBSERVER_NO_AI_LINES)
    h.backend.fg = info("code.exe", hwnd=444)
    h.first_frame()
    assert h.capture.calls == [444] and h.integ.status_view().observer_note == ""


def test_her_answers_about_private_windows_never_claim_she_did_not_look():
    # The frame was analysed (perhaps by a cloud model): "я не читаю" would be untrue.
    from chibi.personality.phrases import PhraseBank

    bank = PhraseBank.load()
    untrue = ("не читаю", "не читаю.", "відвертаюся", "не підглядати", "не розглядаю", "не для моїх очей")
    for key in ("vision.look.chat", "vision.look.email", "vision.look.document"):
        for phrase in bank.phrases(key):
            assert not any(claim in phrase.text.casefold() for claim in untrue), phrase.id


# --- the user's conversation comes first ------------------------------------------------------------


def talk(h: H, active: bool) -> None:
    h.bus.publish(ConversationActive(active=active))
    h.dispatch()


def test_a_conversation_holds_her_frames_and_abandons_the_one_in_flight(make):
    h = make(mode="auto", scope="foreground", commentary="observer")
    h.backend.fg = info("code.exe", hwnd=444)
    h.first_frame()
    assert h.capture.calls == [444] and len(h.observed) == 1
    h.clock.advance(100)
    h.integ.tick()  # the next frame is on its way...
    talk(h, True)  # ...when the user starts talking with her
    assert h.analyzer.cancelled == 1  # its request is withdrawn from the AI service
    h.worker.run()
    h.dispatch()
    for _ in range(60):
        h.step(1)
    assert h.capture.calls == [444] and len(h.observed) == 1  # nothing while they talk
    talk(h, False)
    h.step(CONVERSATION_GRACE_SECONDS - 1)
    assert h.capture.calls == [444]
    h.step(1)
    assert h.capture.calls == [444, 444]  # back right after the grace, no back-off
    assert h.notices == [] and h.integ.service.state != VisionState.PAUSED


def test_an_analysis_cut_short_by_a_conversation_is_never_an_error_notice(make):
    h = make()
    analyzer = h.analyzer
    original = analyzer.analyze

    def analyze(*args, **kwargs):
        if not analyzer.calls:
            analyzer.calls.append("interrupted")
            talk(h, True)  # the conversation starts while the model looks at the frame
            # ...and the AI service reports the withdrawn request as cancelled.
            raise AnalyzerError("analysis cancelled", user_message=CANCELLED_USER_MESSAGE)
        return original(*args, **kwargs)

    analyzer.analyze = analyze
    h.first_frame()
    assert analyzer.cancelled == 1 and h.observed == []
    assert h.notices == [] and h.integ.service.state != VisionState.PAUSED
    talk(h, False)
    h.step(CONVERSATION_GRACE_SECONDS)
    assert len(h.observed) == 1 and h.notices == []  # no 60 s pause after a "failure"


def test_look_at_my_screen_in_the_conversation_is_answered_at_once(make):
    h = make(mode="auto", scope="foreground", commentary="observer")
    h.backend.fg = info("code.exe", hwnd=444)
    talk(h, True)
    h.results = [obs("none", "code", description="код у редакторі")]
    assert h.integ.request()
    h.worker.run()
    h.dispatch()
    # The user is waiting for this one: analysed now, marked so the AI service runs it.
    assert h.capture.calls == [444] and h.analyzer.urgent == [True]
    assert h.composer.calls and h.composer.calls[-1][1] is True  # her (solicited) answer
    details = h.integ.details_for(h.observed[-1])
    assert details is not None and details.solicited
    assert h.notices == []
    for _ in range(60):
        h.step(1)
    assert h.capture.calls == [444]  # nothing automatic while the conversation lasts


def test_the_second_frame_of_a_request_in_the_conversation_is_taken(make):
    h = make(mode="manual")
    talk(h, True)
    h.results = [obs("player_death", "death_screen")]
    assert h.integ.request()
    h.worker.run()
    h.dispatch()
    assert h.composer.calls == []  # waits for the confirmation frame
    h.confirm_frame()
    assert h.analyzer.urgent == [True, True]
    assert h.composer.calls == [(("vision.player_death", "vision.generic"), True, {"game": TITLE})]


def test_after_shutdown_a_conversation_changes_nothing(make):
    h = make()
    h.integ.shutdown()
    talk(h, True)
    talk(h, False)
    assert h.notices == [] and not h.integ.service.conversation
