"""Energy-based voice activity detection: when did the user start / stop talking?

Push-to-talk opens the microphone; this detector decides where the
utterance begins (so the silence before it is not sent to recognition) and
when it is over (a pause of ``end_silence_ms``), so the user doesn't have to
press the key a second time.  An adaptive noise floor (minimum
statistics) copes with fans and keyboards; hysteresis keeps soft word
endings from being cut off.  The user may start talking the instant the
microphone opens, so the floor is not taken from the first frames: it starts
from the last session's value (or a quiet-room default), and every finished
utterance is re-checked against the floor learnt meanwhile - steady noise
without peaks is never reported as speech.

Pure numpy, no I/O: feed float32 mono samples at ``sample_rate``.  Not
thread-safe: one thread feeds it (the audio callback or a worker).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VADConfig:
    sample_rate: int = 16_000
    frame_ms: int = 30
    #: Continuous speech needed before an utterance starts (ignores clicks).
    start_ms: int = 120
    #: Pause that ends the utterance.
    end_silence_ms: int = 800
    #: Shorter utterances are discarded (a cough, a key clack).
    min_speech_ms: int = 250
    max_utterance_seconds: float = 15.0
    #: Audio kept from before the start was detected (first syllable).
    pre_roll_ms: int = 300
    #: Silence kept after the speech (soft endings).
    tail_ms: int = 250
    #: Speech when the frame RMS exceeds noise_floor * ratio ...
    threshold_ratio: float = 3.0
    #: ... and once speaking, while it exceeds noise_floor * ratio * hold.
    hold_ratio: float = 0.6
    #: Absolute minimum RMS for speech (about -44 dBFS).
    min_rms: float = 0.006
    #: The noise floor is the 10th percentile of frame energy over this window
    #: ("minimum statistics": speech always has dips, steady noise doesn't).
    noise_window_ms: int = 1500
    #: Speech always dips between words; a signal loud for this long without a
    #: single dip is noise that slipped in before the floor was learnt.
    stuck_ms: int = 2000


@dataclass(frozen=True, slots=True)
class VADEvent:
    kind: str  # "start" | "end"
    #: The utterance (float32), for "end" events.
    audio: np.ndarray | None = None
    #: "silence" | "max_length" | "finished"
    reason: str = ""


class EnergyVAD:
    def __init__(self, config: VADConfig | None = None, initial_floor: float | None = None) -> None:
        self.config = config or VADConfig()
        c = self.config
        self._initial_floor = initial_floor if initial_floor and initial_floor > 0 else \
            c.min_rms / c.threshold_ratio
        self._window: deque[float] = deque(maxlen=max(4, math.ceil(c.noise_window_ms / c.frame_ms)))
        self._calibrated_after = max(4, math.ceil(240 / c.frame_ms))
        self._stuck_frames = max(1, math.ceil(c.stuck_ms / c.frame_ms))
        self._frame = max(1, c.sample_rate * c.frame_ms // 1000)
        self._start_frames = max(1, math.ceil(c.start_ms / c.frame_ms))
        self._end_frames = max(1, math.ceil(c.end_silence_ms / c.frame_ms))
        self._min_frames = max(1, math.ceil(c.min_speech_ms / c.frame_ms))
        self._max_frames = max(1, int(c.max_utterance_seconds * 1000 / c.frame_ms))
        self._tail_frames = max(0, math.ceil(c.tail_ms / c.frame_ms))
        self._pre_roll: deque[np.ndarray] = deque(maxlen=max(1, math.ceil(c.pre_roll_ms / c.frame_ms)))
        self.reset()

    # -- state ------------------------------------------------------------------------------

    def reset(self) -> None:
        """Forget everything but the learnt noise floor."""
        self._pending = np.zeros(0, dtype=np.float32)
        self._pre_roll.clear()
        self._speech_run = 0
        self._in_speech = False
        self._frames: list[np.ndarray] = []
        self._rms: list[float] = []
        self._voiced = 0
        self._silence_run = 0
        self._loud_run = 0
        if not hasattr(self, "_floor"):
            self._floor = self._initial_floor
        self.level = 0.0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def noise_floor(self) -> float:
        return self._floor

    # -- processing ----------------------------------------------------------------------------

    def feed(self, samples: np.ndarray) -> list[VADEvent]:
        """Process a chunk of float32 samples; returns 0..n events."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        data = np.concatenate([self._pending, samples]) if self._pending.size else samples
        events: list[VADEvent] = []
        n_frames = data.size // self._frame
        for i in range(n_frames):
            frame = data[i * self._frame:(i + 1) * self._frame]
            event = self._process(frame)
            if event is not None:
                events.append(event)
        self._pending = data[n_frames * self._frame:].copy()
        return events

    def finish(self) -> np.ndarray | None:
        """Push-to-talk released: the utterance so far, or None if there was
        no (long enough) speech.  Resets the detector."""
        audio = None
        if self._in_speech and self._voiced >= self._min_frames and self._confirmed():
            audio = self._utterance()
        elif not self._in_speech and self._speech_run >= self._min_frames:
            audio = np.concatenate(list(self._pre_roll)) if self._pre_roll else None
        floor = self._floor
        self.reset()
        self._floor = floor
        return audio

    def _process(self, frame: np.ndarray) -> VADEvent | None:
        c = self.config
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        self.level = min(1.0, max(0.0, (20 * math.log10(max(rms, 1e-6)) + 60) / 60))
        self._window.append(rms)
        # The floor follows the background only between utterances: inside
        # one it would climb to the level of the voice itself.
        if not self._in_speech and len(self._window) >= self._calibrated_after:
            self._floor = self._window_floor()
        floor = self._floor
        threshold = max(c.min_rms, floor * c.threshold_ratio)

        if not self._in_speech:
            self._pre_roll.append(frame)
            if rms > threshold:
                self._speech_run += 1
                if self._speech_run >= self._start_frames:
                    self._in_speech = True
                    self._frames = list(self._pre_roll)
                    self._rms = [float(np.sqrt(np.mean(f.astype(np.float64) ** 2))) for f in self._frames]
                    self._voiced = self._speech_run
                    self._silence_run = 0
                    self._pre_roll.clear()
                    return VADEvent("start")
            else:
                self._speech_run = 0
            return None

        self._frames.append(frame)
        self._rms.append(rms)
        if rms > max(c.min_rms * c.hold_ratio, threshold * c.hold_ratio):
            self._voiced += 1
            self._silence_run = 0
            self._loud_run += 1
            if self._loud_run >= self._stuck_frames:
                self._floor = self._window_floor()
                self._loud_run = 0
        else:
            self._silence_run += 1
            self._loud_run = 0
        if self._silence_run >= self._end_frames:
            return self._end("silence")
        if len(self._frames) >= self._max_frames:
            return self._end("max_length")
        return None

    def _window_floor(self) -> float:
        return max(float(np.percentile(self._window, 10)), 1e-5)

    def _confirmed(self) -> bool:
        """Re-check against the floor learnt by now: real speech has enough
        frames well above it; noise that merely looked loud at first doesn't."""
        threshold = max(self.config.min_rms, self._floor * self.config.threshold_ratio)
        return sum(r > threshold for r in self._rms) >= self._min_frames

    def _end(self, reason: str) -> VADEvent | None:
        enough = self._voiced >= self._min_frames and self._confirmed()
        audio = self._utterance() if enough else None
        floor = self._floor
        self.reset()
        self._floor = floor
        if audio is None:
            return None  # too short: a click or a cough
        return VADEvent("end", audio=audio, reason=reason)

    def _utterance(self) -> np.ndarray:
        frames = self._frames
        # Drop the trailing silence but keep a short tail.
        drop = max(0, self._silence_run - self._tail_frames)
        if drop:
            frames = frames[:-drop]
        # Drop leading noise that started the utterance before the floor was
        # learnt, keeping the pre-roll before the first real speech frame.
        threshold = max(self.config.min_rms, self._floor * self.config.threshold_ratio)
        first = next((i for i, r in enumerate(self._rms[:len(frames)]) if r > threshold), 0)
        frames = frames[max(0, first - self._pre_roll.maxlen):]
        return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)


def to_mono_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mix down and resample to 16 kHz float32 (what Whisper expects).

    A moving-average low-pass before linear interpolation keeps aliasing
    out of the speech band when the device runs at 44.1/48 kHz.
    """
    audio = np.asarray(samples)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate == 16_000 or audio.size == 0:
        return audio
    ratio = sample_rate / 16_000
    if ratio > 1:
        width = round(ratio)
        if width > 1:
            kernel = np.ones(width, dtype=np.float32) / width
            audio = np.convolve(audio, kernel, mode="same")
    n_out = int(audio.size / ratio)
    positions = np.arange(n_out, dtype=np.float64) * ratio
    return np.interp(positions, np.arange(audio.size), audio).astype(np.float32)
