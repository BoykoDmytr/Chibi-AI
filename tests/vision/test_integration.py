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
    GameEnded,
    GameStarted,
    Permission,
    SettingsChanged,
    SystemLoadChanged,
    VisionObserved,
    VisionStateChanged,
)
from chibi.gaming.games_db import GameDatabase
from chibi.memory.repositories import PermissionRepository
from chibi.personality.phrases import DEFAULT_PATH, PhraseBank
from chibi.privacy.permissions import PermissionManager
from chibi.vision.analysis import Observation
from chibi.vision.capture_win import Frame
from chibi.vision.integration import (
    LOOK_FAILED_MESSAGE,
    MEMORY_EVENT_FORMS,
    OWN_WINDOW_MESSAGE,
    OWN_WINDOW_RETRY_MS,
    REASON_LOCAL_ONLY,
    REASON_NO_CAPTURE,
    REASON_NO_CLOUD_VISION,
    REASON_OFF,
    VisionIntegration,
    is_look_request,
    reactions_per_hour,
    session_summary,
)
from chibi.vision.service import AnalyzerError, VisionMode, VisionState
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
        self.cancelled = 0
        self.closed = False

    @property
    def is_local(self):
        return self.local

    def analyze(self, image, media_type, app_title, previous):
        self.calls.append(app_title)
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
                 allowed_games=("stalker2",), allowed_processes=(), grant=True, local_only=False):
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
        self.integ = VisionIntegration(
            bus, lambda: self.settings, self.permissions, object(), self.composer, clock,
            games=lambda: self.games, memory=self.memory, quiet=lambda: self.quiet,
            on_spoken=lambda: self.spoken.append(1), notify=self.notices.append,
            backend=self.backend, capture=self.capture, worker=self.worker,
            own_windows=lambda: self.own, own_pid=OWN_PID, analyzer_factory=self._factory,
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
    assert view.state == "watching" and view.app_title == TITLE
    assert "не підтримується" in view.note


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
