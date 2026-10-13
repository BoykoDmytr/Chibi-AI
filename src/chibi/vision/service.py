"""Screen vision: the pipeline from "a window she may see is in front" to "she saw it".

Main-thread state machine (like :mod:`chibi.voice.service`); capture and
analysis run on one worker thread and post results back.  The pipeline is
the one the spec prescribes, in this order:

1. permission check (Permission.SCREEN_CAPTURE; CLOUD_VISION before any
   frame goes to a cloud model)
2. the foreground window must be one she may see (``resolve_target``: an
   allowed app, or any non-private window for the "foreground" scope)
3. is a new frame needed (mode, interval, back-off, busy, user away)?
4. capture that one window (never the whole screen)
5. privacy re-check after capture: same window still in front, still
   allowed, permission still granted - otherwise the frame is dropped
6. downscale (<= 1024 px) and skip if it looks like the last analysed frame
   of the same window
7. analysis by a local model or, with CLOUD_VISION, a cloud model
8. strict parsing of the structured answer (:mod:`chibi.vision.analysis`)
9. re-check before the result is used (``still_allowed``: permissions and
   privacy rules may have changed while the model looked)
10. confidence / confirmation / anti-spam rules (:mod:`chibi.vision.policy`)
11. the app decides how she reacts (``on_observation``)

Frames are never written to disk; the last few *observations* (scene, event,
short description) are kept in memory as context for her chat answers.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

from chibi.activity.backend import ForegroundInfo
from chibi.vision.analysis import AnalysisError, Observation
from chibi.vision.capture_win import CaptureBackend, CaptureError
from chibi.vision.policy import Decision, VisionReactionPolicy

log = logging.getLogger(__name__)


class VisionMode(StrEnum):
    OFF = "off"  # "Vision Disabled": no capture at all
    MANUAL = "manual"  # only when the user asks
    AUTO = "auto"  # periodically, while a window she may see is in front
    GAME = "game"  # periodically, allowed games only, adapting to load


class VisionState(StrEnum):
    OFF = "off"
    IDLE = "idle"  # enabled, nothing she may see in front (or the user is away)
    WATCHING = "watching"  # a window she may see is in front; frames are taken periodically
    ANALYZING = "analyzing"  # capture / analysis in progress
    PAUSED = "paused"  # repeated errors or the AI budget is spent


@dataclass(frozen=True, slots=True)
class Target:
    hwnd: int
    process_name: str
    title: str  # display title (games database / app catalog / user list), never the window title
    is_game: bool


class FrameAnalyzer(Protocol):
    """Sends one encoded frame to a vision model; called on the worker thread."""

    @property
    def is_local(self) -> bool: ...

    def analyze(self, image: bytes, media_type: str, app_title: str,
                previous: Observation | None) -> Observation: ...


class AnalyzerError(RuntimeError):
    """Provider failure.  ``retry_after`` (seconds) when it's worth waiting;
    ``budget`` when the AI spending limit was reached."""

    def __init__(self, message: str, *, user_message: str = "", retry_after: float = 60.0,
                 budget: bool = False) -> None:
        super().__init__(message)
        self.user_message = user_message or message
        self.retry_after = retry_after
        self.budget = budget


#: Hamming distance under which two frames count as the same picture.
SAME_FRAME_DISTANCE = 5
CONFIRM_DELAY = 3.0
MAX_CAPTURE_FAILURES = 4
#: Paced AUTO mode: each identical frame in a row stretches the interval by this much (up to 3x).
SAME_FRAME_SLOWDOWN = 0.5


class VisionService:
    def __init__(
        self,
        *,
        post: Callable[[Callable[[], None]], None],
        worker: Any,
        capture: CaptureBackend | None,
        foreground: Callable[[], ForegroundInfo | None],
        resolve_target: Callable[[ForegroundInfo, bool], Target | None],
        capture_allowed: Callable[[], bool],
        cloud_allowed: Callable[[], bool],
        policy: VisionReactionPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
        quiet: Callable[[], bool] = lambda: False,
        on_state: Callable[[VisionState, str], None] | None = None,
        on_observation: Callable[[Observation, Decision], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        process_frame: Callable[[Any], tuple[bytes, str, int]] | None = None,
        still_allowed: Callable[[Target], bool] | None = None,
    ) -> None:
        self._post = post
        self._worker = worker
        self._capture = capture
        self._foreground = foreground
        self._resolve = resolve_target
        self._capture_allowed = capture_allowed
        self._cloud_allowed = cloud_allowed
        self.policy = policy or VisionReactionPolicy(clock=clock)
        self._clock = clock
        self._quiet = quiet
        self._on_state = on_state
        self._on_observation = on_observation
        self._on_error = on_error
        self._process_frame = process_frame or _default_process_frame
        self._still_allowed = still_allowed

        self.mode = VisionMode.OFF
        self.interval = 30.0
        self.analyzer: FrameAnalyzer | None = None
        self.high_load = False
        #: AUTO mode paces itself (any-window scope): slower under load and while
        #: the window keeps showing the same picture.  GAME mode always does.
        self.paced = False
        #: The user is away (the activity monitor reported them idle): no automatic
        #: frames.  A request from the user still works - they are obviously back.
        #: (A locked screen needs no flag: there is no window in front to resolve.)
        self.away = False
        self._state = VisionState.OFF
        self._gen = 0
        self._busy = False
        self._target: Target | None = None
        self._next_due = 0.0
        self._confirm_due: float | None = None
        self._backoff_until = 0.0
        self._quiet_streak = 0
        self._capture_failures = 0
        self._reported: set[str] = set()
        #: dHash of the last analysed frame and the window it came from.
        self._last_hash: int | None = None
        self._last_hash_hwnd = 0
        self._same_streak = 0
        self.recent: deque[Observation] = deque(maxlen=5)

    # -- configuration ---------------------------------------------------------------------

    @property
    def state(self) -> VisionState:
        return self._state

    @property
    def target(self) -> Target | None:
        return self._target

    @property
    def active(self) -> bool:
        """Capture may happen right now (drives the on-screen indicator)."""
        return self._state in (VisionState.WATCHING, VisionState.ANALYZING)

    def configure(self, mode: VisionMode, *, interval: float, analyzer: FrameAnalyzer | None) -> None:
        self.mode = mode
        self.interval = max(5.0, interval)
        self.analyzer = analyzer
        self._gen += 1
        self._busy = False
        self._backoff_until = 0.0
        self._capture_failures = 0
        self._reported.clear()
        if mode == VisionMode.OFF or analyzer is None or self._capture is None:
            self._target = None
            self.recent.clear()
            self._set_state(VisionState.OFF, "disabled")
        else:
            self._set_state(VisionState.IDLE, "configured")
            self._next_due = self._clock() + 2.0

    def stop(self) -> None:
        """Kill switch / permission revoked: stop now and forget what she saw."""
        self._gen += 1
        self._busy = False
        self._target = None
        self._confirm_due = None
        self._last_hash = None
        self._same_streak = 0
        self.recent.clear()
        if self.mode != VisionMode.OFF:
            self._set_state(VisionState.IDLE if self._enabled() else VisionState.OFF, "stopped")

    def game_session(self, active: bool) -> None:
        """A game started or ended: novelty and dedup start over."""
        self.policy.reset_session()
        self._last_hash = None
        self._quiet_streak = 0
        self._same_streak = 0
        if not active:
            self.recent.clear()

    # -- scheduling -----------------------------------------------------------------------------

    def tick(self) -> None:
        """Call about once a second from the main thread."""
        if not self._enabled():
            if self._state in (VisionState.WATCHING, VisionState.ANALYZING, VisionState.PAUSED):
                # Permission gone without stop() (defence in depth): stand down.
                self._gen += 1
                self._busy = False
                self._target = None
                self._confirm_due = None
                self._set_state(VisionState.IDLE if self.mode != VisionMode.OFF else VisionState.OFF,
                                "not allowed")
            return
        if self._busy:
            return
        now = self._clock()
        if self._state == VisionState.PAUSED and now < self._backoff_until:
            return
        if self.away and self.mode != VisionMode.MANUAL:
            # Nobody to watch: no frames (and no cost) until the user is back.
            self._target = None
            self._confirm_due = None
            if self._state in (VisionState.WATCHING, VisionState.PAUSED):
                self._set_state(VisionState.IDLE, "user away")
            return
        if self.mode == VisionMode.MANUAL:
            # Only the confirmation frame of a manual request is ever automatic.
            if self._confirm_due is not None and now >= self._confirm_due:
                self._confirm_due = None
                target = self._current_target(require_game=False)
                if target is not None and target == self._target:
                    self._start(target, force=True, reason="confirm")
            return
        target = self._current_target(require_game=self.mode == VisionMode.GAME)
        if target is None:
            self._target = None
            self._confirm_due = None
            if self._state in (VisionState.WATCHING, VisionState.PAUSED):
                self._set_state(VisionState.IDLE, "no allowed app in front")
            return
        if target != self._target:
            self._target = target
            self._confirm_due = None
            self._next_due = now + 2.0  # let the window settle
            self._capture_failures = 0
            self._same_streak = 0
        if self._state != VisionState.WATCHING:
            self._set_state(VisionState.WATCHING, target.title)
        if self._confirm_due is not None and now >= self._confirm_due:
            self._confirm_due = None
            self._start(target, force=True, reason="confirm")
            return
        if now >= self._next_due and now >= self._backoff_until:
            self._start(target, force=False, reason="interval")

    def request(self) -> bool:
        """Manual vision ("подивись на екран"): analyse the window in front now.

        Works in every mode except OFF; the app in front must still be allowed
        and not private.  Returns False (and explains) if it can't.
        """
        if self.mode == VisionMode.OFF or self.analyzer is None or self._capture is None:
            self._error("Бачення екрана вимкнене (Налаштування → Бачення екрана).")
            return False
        if not self._capture_allowed():
            self._error("Захоплення екрана не дозволене в «Приватності».")
            return False
        if self._busy:
            return True
        target = self._current_target(require_game=False)
        if target is None:
            self._error("Це вікно не в списку дозволених для аналізу (або воно приватне).")
            return False
        self._target = target
        self._start(target, force=True, reason="manual")
        return True

    def _enabled(self) -> bool:
        return (self.mode != VisionMode.OFF and self.analyzer is not None
                and self._capture is not None and self._capture_allowed())

    def _current_target(self, *, require_game: bool) -> Target | None:
        try:
            info = self._foreground()
        except Exception:
            log.exception("foreground lookup failed")
            return None
        if info is None or not info.hwnd:
            return None
        target = self._resolve(info, require_game)
        if target is None or (require_game and not target.is_game):
            return None
        return target

    def _effective_interval(self) -> float:
        interval = self.interval
        if self.mode == VisionMode.GAME:
            if self.high_load:
                interval *= 2.0
            # Nothing happening for a while: look less often (up to 3x).
            interval *= min(3.0, 1.0 + 0.25 * self._quiet_streak)
        elif self.mode == VisionMode.AUTO and self.paced:
            if self.high_load:
                interval *= 2.0
            # The window keeps showing the same picture (a document being read, a
            # paused video): capture less often (up to 3x).  A quiet but changing
            # work window is not slowed down - the observer needs to see it.
            interval *= min(3.0, 1.0 + SAME_FRAME_SLOWDOWN * self._same_streak)
        return interval

    # -- the pipeline ---------------------------------------------------------------------------

    def _start(self, target: Target, *, force: bool, reason: str) -> None:
        analyzer = self.analyzer
        capture = self._capture
        if analyzer is None or capture is None:
            return
        self._busy = True
        gen = self._gen
        previous = self.recent[-1] if self.recent else None
        # Dedup only against the last frame of the same window.
        last_hash = None if force or target.hwnd != self._last_hash_hwnd else self._last_hash
        self._set_state(VisionState.ANALYZING, reason)

        def job() -> None:
            outcome = self._pipeline(capture, analyzer, target, previous, last_hash)
            self._post(lambda: self._finished(gen, target, reason, outcome))

        self._worker.submit(job, 1)

    def _pipeline(self, capture: CaptureBackend, analyzer: FrameAnalyzer, target: Target,
                  previous: Observation | None, last_hash: int | None) -> tuple[str, Any]:
        """Worker thread.  Returns (kind, payload)."""
        if not self._capture_allowed():
            return "dropped", "permission revoked"
        try:
            frame = capture.capture(target.hwnd)
        except CaptureError as exc:
            return "capture_error", exc
        except Exception as exc:  # a ctypes surprise must not kill the worker
            log.exception("capture failed")
            return "capture_error", CaptureError("failed", str(exc))
        # Privacy re-check: the user may have switched windows during the capture.
        try:
            info = self._foreground()
        except Exception:
            info = None
        if info is None or info.hwnd != target.hwnd or not self._capture_allowed() \
                or self._resolve(info, False) is None:
            return "dropped", "window changed"
        try:
            image, media_type, frame_hash = self._process_frame(frame)
        except Exception as exc:
            log.exception("frame processing failed")
            return "capture_error", CaptureError("failed", str(exc))
        finally:
            del frame
        if last_hash is not None and (frame_hash ^ last_hash).bit_count() <= SAME_FRAME_DISTANCE:
            return "same", frame_hash
        if not analyzer.is_local and not self._cloud_allowed():
            return "dropped", "cloud vision not allowed"
        try:
            observation = analyzer.analyze(image, media_type, target.title, previous)
        except AnalyzerError as exc:
            return "analyzer_error", exc
        except AnalysisError as exc:
            return "bad_answer", exc
        except Exception as exc:
            log.exception("vision analysis failed")
            return "analyzer_error", AnalyzerError(str(exc), user_message="Аналіз кадру не вдався.")
        return "observation", (observation, frame_hash)

    def _finished(self, gen: int, target: Target, reason: str, outcome: tuple[str, Any]) -> None:
        if gen != self._gen:
            return  # stopped or reconfigured meanwhile: forget it
        self._busy = False
        now = self._clock()
        kind, payload = outcome
        self._next_due = now + self._effective_interval()
        if kind == "observation" and not self._capture_allowed():
            kind, payload = "dropped", "permission revoked"
        if kind == "observation" and not self._may_use(target):
            kind, payload = "dropped", "no longer allowed"
        if kind == "observation":
            observation, frame_hash = payload
            self._capture_failures = 0
            self._same_streak = 0
            self._last_hash = frame_hash
            self._last_hash_hwnd = target.hwnd
            observation = replace(observation, app_title=target.title, at=time.time())
            self.recent.append(observation)
            decision = self.policy.consider(observation, quiet=self._quiet())
            if decision.action == "confirm" and reason == "confirm":
                # The confirmation frame shows yet another event: the model is unsure.
                # No third frame - a model that keeps changing its mind would otherwise
                # be asked every few seconds, whatever the interval; the next regular
                # frame starts over.
                decision = Decision("ignore", "unconfirmed", decision.score, observation=decision.observation)
            elif decision.action == "confirm":
                self._confirm_due = now + CONFIRM_DELAY
            self._quiet_streak = 0 if observation.notable else self._quiet_streak + 1
            if self._on_observation is not None:
                self._on_observation(decision.observation or observation, decision)
        elif kind == "same":
            self._quiet_streak += 1
            self._same_streak += 1
        elif kind == "capture_error":
            self._capture_failures += 1
            error: CaptureError = payload
            self._report_once(f"capture:{error.reason}", error.user_message, always=reason == "manual")
            if error.reason == "black":
                self._next_due = now + self._effective_interval() * 4
            if self._capture_failures >= MAX_CAPTURE_FAILURES:
                self._pause(now + 300, "capture keeps failing")
                return
        elif kind == "analyzer_error":
            error_a: AnalyzerError = payload
            self._report_once("budget" if error_a.budget else "analyzer", error_a.user_message,
                              always=reason == "manual")
            self._pause(now + max(30.0, error_a.retry_after), "analysis failed")
            return
        elif kind == "bad_answer":
            log.info("vision model gave an unusable answer: %s", payload)
        elif kind == "dropped":
            log.debug("frame dropped: %s", payload)
        if reason == "manual" and kind != "observation" and kind != "capture_error":
            self._error("Не вдалося роздивитися екран. Спробуй ще раз за хвилинку.")
        state = VisionState.WATCHING if self._target is not None and self.mode != VisionMode.MANUAL \
            else VisionState.IDLE
        self._set_state(state, kind)

    def _may_use(self, target: Target) -> bool:
        """The re-check before a result becomes something she knows (main thread)."""
        if self._still_allowed is None:
            return True
        try:
            return bool(self._still_allowed(target))
        except Exception:
            log.exception("vision re-check failed")
            return False

    def _pause(self, until: float, reason: str) -> None:
        self._backoff_until = until
        self._set_state(VisionState.PAUSED, reason)

    def _report_once(self, key: str, message: str, *, always: bool = False) -> None:
        """Automatic failures are reported once; ``always`` when the user asked
        (a request must get its answer, not silence)."""
        if key in self._reported and not always:
            return
        self._reported.add(key)
        self._error(message)

    def _set_state(self, state: VisionState, reason: str) -> None:
        if state == self._state:
            return
        self._state = state
        if self._on_state is not None:
            self._on_state(state, reason)

    def _error(self, message: str) -> None:
        log.info("vision: %s", message)
        if self._on_error is not None:
            self._on_error(message)


def _default_process_frame(frame: Any) -> tuple[bytes, str, int]:
    """Downscale, hash and encode in memory (worker thread)."""
    from chibi.vision.imaging import dhash, downscale, encode_jpeg, frame_to_image, image_media_type

    small = downscale(frame_to_image(frame))
    data = encode_jpeg(small)
    return data, image_media_type(data), dhash(small)
