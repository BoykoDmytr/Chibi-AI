"""VisionService with fakes: the pipeline order, privacy guarantees and
anti-spam behaviour.  Leak tests check that a frame never reaches capture /
analysis / disk when it must not."""

import builtins
import random

import pytest

from chibi.activity.backend import ForegroundInfo
from chibi.vision.analysis import Observation
from chibi.vision.capture_win import CaptureError, Frame
from chibi.vision.policy import PolicyConfig, VisionReactionPolicy
from chibi.vision.service import AnalyzerError, Target, VisionMode, VisionService, VisionState

GAME = ForegroundInfo(pid=1, process_name="stalker2.exe", hwnd=111)
BANK = ForegroundInfo(pid=2, process_name="mybank.exe", hwnd=222)
NOTEPAD = ForegroundInfo(pid=3, process_name="notepad.exe", hwnd=333)
ALLOWED = {"stalker2.exe": ("S.T.A.L.K.E.R. 2", True), "notepad.exe": ("Блокнот", False)}


def obs(event="none", scene="gameplay", confidence=0.9):
    return Observation(scene=scene, event=event, confidence=confidence, interesting=True,
                       suggested_emotion="curious", description="опис", spoiler=False)


class Worker:
    def __init__(self):
        self.jobs = []

    def submit(self, job, priority):
        self.jobs.append(job)

    def run(self):
        while self.jobs:
            self.jobs.pop(0)()


class Capture:
    def __init__(self):
        self.calls = []
        self.error = None

    def capture(self, hwnd):
        self.calls.append(hwnd)
        if self.error:
            raise self.error
        return Frame(4, 4, b"\x10" * 64, hwnd, 0.0, "fake")


class Analyzer:
    def __init__(self, results=None, local=True):
        self.results = list(results or [obs()])
        self.calls = []
        self.local = local

    @property
    def is_local(self):
        return self.local

    def analyze(self, image, media_type, app_title, previous):
        self.calls.append((image, media_type, app_title, previous))
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, Exception):
            raise result
        return result


class H:
    def __init__(self, *, mode=VisionMode.GAME, analyzer=None, capture_ok=True, cloud_ok=False):
        self.now = 1000.0
        self.worker = Worker()
        self.capture = Capture()
        self.analyzer = analyzer or Analyzer()
        self.front = GAME
        self.front_during_capture = None
        self.capture_ok = capture_ok
        self.cloud_ok = cloud_ok
        self.quiet = False
        self.states, self.seen, self.errors = [], [], []
        rng = random.Random(0)
        self.hashes = (rng.getrandbits(64) for _ in range(10**5))  # distinct frames by default
        self.service = VisionService(
            post=lambda fn: fn(),
            worker=self.worker,
            capture=self.capture,
            foreground=self._foreground,
            resolve_target=self._resolve,
            capture_allowed=lambda: self.capture_ok,
            cloud_allowed=lambda: self.cloud_ok,
            policy=VisionReactionPolicy(PolicyConfig(), clock=lambda: self.now),
            clock=lambda: self.now,
            quiet=lambda: self.quiet,
            on_state=lambda s, r: self.states.append(s),
            on_observation=lambda o, d: self.seen.append((o, d)),
            on_error=self.errors.append,
            process_frame=lambda frame: (b"jpeg", "image/jpeg", next(self.hashes)),
        )
        self.service.configure(mode, interval=30, analyzer=self.analyzer)
        self.service.tick()  # sees the window in front; first frame ~2 s later

    def _foreground(self):
        if self.front_during_capture is not None and self.capture.calls:
            return self.front_during_capture
        return self.front

    def _resolve(self, info, require_game):
        entry = ALLOWED.get(info.process_name)
        if entry is None:
            return None
        title, is_game = entry
        return Target(info.hwnd, info.process_name, title, is_game)

    def advance(self, seconds):
        self.now += seconds
        self.service.tick()
        self.worker.run()


def test_off_mode_never_captures():
    h = H(mode=VisionMode.OFF)
    h.advance(100)
    assert h.capture.calls == [] and h.service.state == VisionState.OFF


def test_game_mode_watches_an_allowed_game_and_analyses_periodically():
    h = H()
    h.advance(1)
    assert h.service.state == VisionState.WATCHING
    h.advance(3)
    assert h.capture.calls == [111]
    assert h.analyzer.calls[0][2] == "S.T.A.L.K.E.R. 2"
    h.advance(10)
    assert len(h.capture.calls) == 1  # interval not reached
    h.advance(25)
    assert len(h.capture.calls) == 2
    assert h.seen[0][0].app_title == "S.T.A.L.K.E.R. 2"


def test_private_or_unlisted_apps_are_never_captured():
    h = H(mode=VisionMode.AUTO)
    for front in (BANK, ForegroundInfo(pid=9, process_name="chrome.exe", hwnd=9)):
        h.front = front
        h.advance(100)
    assert h.capture.calls == []
    assert h.analyzer.calls == []


def test_game_mode_ignores_allowed_non_games():
    h = H(mode=VisionMode.GAME)
    h.front = NOTEPAD
    h.advance(100)
    assert h.capture.calls == []
    h = H(mode=VisionMode.AUTO)
    h.front = NOTEPAD
    h.advance(1)  # notices the new window
    h.advance(3)
    assert h.capture.calls == [333]


def test_window_switch_during_capture_drops_the_frame():
    h = H()
    h.front_during_capture = BANK  # the user alt-tabbed to the bank while capturing
    h.advance(3)
    assert h.capture.calls == [111]
    assert h.analyzer.calls == []
    assert h.seen == []


def test_permission_revoked_during_analysis_discards_the_result():
    h = H()
    h.advance(1)
    h.now += 3
    h.service.tick()  # job queued
    h.capture_ok = False  # revoked before the worker ran
    h.worker.run()
    assert h.analyzer.calls == [] and h.seen == []


def test_permission_revoked_after_analysis_discards_the_result():
    h = H()

    class Revoking(Analyzer):
        def analyze(self, *args):
            h.capture_ok = False
            return super().analyze(*args)

    h = H(analyzer=Revoking())
    h.advance(3)
    assert len(h.analyzer.calls) == 1
    assert h.seen == []


def test_tick_stands_down_when_permission_disappears():
    h = H()
    h.advance(1)
    assert h.service.active
    h.capture_ok = False
    h.advance(1)
    assert not h.service.active


def test_cloud_analyzer_needs_cloud_permission():
    h = H(analyzer=Analyzer(local=False), cloud_ok=False)
    h.advance(3)
    assert h.capture.calls == [111]
    assert h.analyzer.calls == []
    h.cloud_ok = True
    h.advance(31)
    assert len(h.analyzer.calls) == 1


def test_identical_frames_are_not_analysed_twice():
    h = H(mode=VisionMode.AUTO)
    h.hashes = iter([7, 7, 7, 0xFFFF_FFFF_0000])
    h.advance(3)
    h.advance(31)
    h.advance(31)
    assert len(h.capture.calls) == 3 and len(h.analyzer.calls) == 1
    h.advance(40)
    assert len(h.analyzer.calls) == 2


def test_important_event_is_confirmed_with_a_second_frame_then_reacted_to():
    death = obs("player_death", "death_screen", 0.92)
    h = H(analyzer=Analyzer([death, death]))
    h.advance(3)
    assert h.seen[0][1].action == "confirm"
    h.advance(3.5)  # confirmation frame, before the normal interval
    assert len(h.capture.calls) == 2
    observation, decision = h.seen[1]
    assert decision.action == "react" and observation.confirmed


def test_unconfirmed_event_does_not_react():
    h = H(analyzer=Analyzer([obs("player_death", "death_screen"), obs("none")]))
    h.advance(3)
    h.advance(3.5)
    assert [d.action for _, d in h.seen] == ["confirm", "ignore"]


def test_a_model_that_keeps_changing_its_mind_does_not_bypass_the_interval():
    # On an end-of-match screen the model may say "level up", then "achievement", then
    # "level up"...  Each answer is a new event awaiting confirmation; chaining forced
    # confirmation frames every 3 s would analyse (and pay for) ~10x more frames.
    level_up = obs("level_up", "gameplay", 0.9)
    achievement = obs("achievement", "gameplay", 0.9)
    h = H(analyzer=Analyzer([level_up, achievement] * 50))
    for _ in range(60):
        h.advance(1)
    assert len(h.analyzer.calls) <= 4  # each regular frame + at most one confirmation
    assert [d.action for _, d in h.seen[:2]] == ["confirm", "ignore"]
    assert all(d.action != "react" for _, d in h.seen)


def test_quiet_mode_still_observes_but_never_reacts():
    death = obs("player_death", "death_screen", 0.95)
    h = H(analyzer=Analyzer([death, death]))
    h.quiet = True
    h.advance(3)
    h.advance(3.5)
    assert h.seen[-1][1].reason == "quiet"
    assert h.service.recent  # context for chat is still kept


def test_black_frames_report_once_and_back_off_then_pause():
    h = H()
    h.capture.error = CaptureError("black")
    h.advance(3)
    assert len(h.errors) == 1 and "Вікно без рамки" in h.errors[0]
    h.advance(31)
    assert len(h.capture.calls) == 1  # backed off (x4)
    for _ in range(4):
        h.advance(200)
    assert h.service.state == VisionState.PAUSED
    assert len(h.errors) == 1


def test_analyzer_failure_pauses_and_reports_budget():
    h = H(analyzer=Analyzer([AnalyzerError("x", user_message="Ліміт ШІ на сьогодні вичерпано.",
                                           retry_after=600, budget=True)]))
    h.advance(3)
    assert h.service.state == VisionState.PAUSED
    assert h.errors == ["Ліміт ШІ на сьогодні вичерпано."]
    h.advance(100)
    assert len(h.capture.calls) == 1


def test_every_failed_request_says_why():
    # Automatic failures are reported once; a user who asks again must not be
    # left with "Зараз гляну!" and silence (the reason only after a minute).
    budget = AnalyzerError("x", user_message="Ліміт ШІ на сьогодні вичерпано.", retry_after=600, budget=True)
    h = H(mode=VisionMode.MANUAL, analyzer=Analyzer([budget]))
    for expected in (1, 2):
        assert h.service.request()
        h.worker.run()
        assert h.errors == ["Ліміт ШІ на сьогодні вичерпано."] * expected
    h = H(mode=VisionMode.MANUAL)
    h.capture.error = CaptureError("black")
    for expected in (1, 2):
        assert h.service.request()
        h.worker.run()
        assert len(h.errors) == expected and "Вікно без рамки" in h.errors[-1]


def test_manual_mode_only_on_request():
    h = H(mode=VisionMode.MANUAL)
    h.front = NOTEPAD
    h.advance(100)
    assert h.capture.calls == []
    assert h.service.request()
    h.worker.run()
    assert h.capture.calls == [333]
    h.front = BANK
    assert not h.service.request()
    assert "дозволених" in h.errors[-1]


def test_request_explains_when_disabled_or_not_permitted():
    h = H(mode=VisionMode.OFF)
    assert not h.service.request()
    h = H(capture_ok=False)
    assert not h.service.request()
    assert "Приватності" in h.errors[-1]


def test_stop_forgets_everything_and_ignores_late_results():
    h = H()
    h.advance(1)
    h.now += 3
    h.service.tick()
    h.service.stop()
    h.worker.run()
    assert h.seen == [] and not h.service.recent
    assert h.service.target is None


def test_game_mode_adapts_to_load_and_quiet_periods():
    h = H()
    h.service.high_load = True
    h.advance(3)
    h.advance(31)
    assert len(h.capture.calls) == 1  # 60 s under load
    h.advance(30)
    assert len(h.capture.calls) == 2


def test_frames_never_touch_the_disk(monkeypatch, qapp):
    """The real frame processing (QImage, JPEG in memory) with open() trapped."""
    from chibi.vision.service import _default_process_frame

    def no_open(*args, **kwargs):
        raise AssertionError(f"file opened: {args}")

    frame = Frame(64, 48, bytes(range(256)) * 48, 1, 0.0, "fake")
    monkeypatch.setattr(builtins, "open", no_open)
    data, media_type, frame_hash = _default_process_frame(frame)
    assert data and media_type in ("image/jpeg", "image/png") and isinstance(frame_hash, int)


def test_recent_observations_are_bounded():
    h = H(mode=VisionMode.AUTO)
    for _ in range(12):
        h.advance(31)
    assert len(h.service.recent) == 5


@pytest.mark.parametrize("mode", [VisionMode.AUTO, VisionMode.GAME])
def test_game_session_end_clears_context(mode):
    h = H(mode=mode)
    h.advance(3)
    assert h.service.recent
    h.service.game_session(False)
    assert not h.service.recent
