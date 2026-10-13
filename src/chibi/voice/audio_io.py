"""Microphone and speaker access through ``sounddevice`` (PortAudio).

Thin by design - the logic lives in :mod:`chibi.voice.service` and is
tested with fakes; this module only opens and closes streams.  Callbacks
run on PortAudio's audio thread and must hand work to the main thread.

The microphone stream exists only while the user is talking to her (push to
talk) or, if they enabled it, while waiting for the wake phrase.  Audio is
kept in memory only and never written to disk.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, Protocol

from chibi.voice.tts import AudioClip
from chibi.voice.vad import to_mono_16k

log = logging.getLogger(__name__)


class AudioUnavailable(RuntimeError):
    def __init__(self, message: str, *, user_message: str = "") -> None:
        super().__init__(message)
        self.user_message = user_message or message


class AudioOutput(Protocol):
    def play(self, clip: AudioClip, volume: float, on_done: Callable[[bool], None]) -> None:
        """Start playing; *on_done(completed)* is called once, from any thread,
        when playback ends (completed=False if stopped)."""

    def stop(self) -> None: ...


class AudioInput(Protocol):
    def start(self, on_audio: Callable[[Any], None]) -> None:
        """Open the microphone; *on_audio* receives float32 mono 16 kHz chunks
        on the audio thread."""

    def stop(self) -> None: ...

    @property
    def active(self) -> bool: ...


def _sounddevice() -> Any:
    try:
        import sounddevice
    except (ImportError, OSError) as exc:  # OSError: PortAudio library missing
        raise AudioUnavailable(f"sounddevice unavailable: {exc}",
                               user_message="Звук недоступний: не встановлено модуль голосу.") from exc
    return sounddevice


def available() -> bool:
    try:
        _sounddevice()
    except AudioUnavailable:
        return False
    return True


def list_devices(sd: Any = None) -> tuple[list[str], list[str]]:
    """(input names, output names) of the default host API, for the settings."""
    sd = sd or _sounddevice()
    try:
        devices = sd.query_devices()
        hostapi = sd.default.hostapi
    except Exception:
        log.exception("cannot list audio devices")
        return [], []
    inputs, outputs = [], []
    for dev in devices:
        if hostapi is not None and hostapi >= 0 and dev.get("hostapi") != hostapi:
            continue
        name = str(dev.get("name", "")).strip()
        if not name:
            continue
        if dev.get("max_input_channels", 0) > 0 and name not in inputs:
            inputs.append(name)
        if dev.get("max_output_channels", 0) > 0 and name not in outputs:
            outputs.append(name)
    return inputs, outputs


def _device_index(sd: Any, name: str, kind: str) -> int | None:
    if not name:
        return None
    channels = "max_input_channels" if kind == "input" else "max_output_channels"
    try:
        for index, dev in enumerate(sd.query_devices()):
            if dev.get("name") == name and dev.get(channels, 0) > 0:
                return index
    except Exception:
        log.exception("cannot look up audio device")
    log.warning("audio device not found, using the default")
    return None


class SoundDeviceOutput:
    def __init__(self, device: str = "", sd: Any = None) -> None:
        self._sd = sd
        self.device = device
        self._stream: Any = None
        self._lock = threading.Lock()
        self._interrupted = False

    def play(self, clip: AudioClip, volume: float, on_done: Callable[[bool], None]) -> None:
        import numpy as np

        sd = self._sd or _sounddevice()
        self.stop()
        data = np.clip(clip.samples.astype(np.float32) * max(0.0, min(volume, 1.0)), -32768, 32767)
        data = data.astype(np.int16)
        position = [0]
        done = [False]
        interrupted = [False]

        def callback(outdata, frames, _time, _status):
            start = position[0]
            chunk = data[start:start + frames]
            outdata[: len(chunk), 0] = chunk
            if len(chunk) < frames:
                outdata[len(chunk):, 0] = 0
                position[0] = len(data)
                raise sd.CallbackStop
            position[0] = start + frames

        def finished():
            if not done[0]:
                done[0] = True
                on_done(not interrupted[0])

        try:
            stream = sd.OutputStream(
                samplerate=clip.sample_rate, channels=1, dtype="int16",
                device=_device_index(sd, self.device, "output"),
                callback=callback, finished_callback=finished,
            )
            stream.start()
        except Exception as exc:
            raise AudioUnavailable(f"cannot open audio output: {exc}",
                                   user_message="Не вдалося відтворити звук: перевір динаміки.") from exc
        with self._lock:
            self._stream = stream
            self._interrupt_flag = interrupted

    def stop(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
            flag = getattr(self, "_interrupt_flag", None)
        if stream is None:
            return
        if flag is not None:
            flag[0] = True
        try:
            stream.abort()
            stream.close()
        except Exception:
            log.debug("closing output stream failed", exc_info=True)


class SoundDeviceInput:
    """16 kHz mono float32 microphone stream (resampled if the device can't)."""

    def __init__(self, device: str = "", sd: Any = None) -> None:
        self._sd = sd
        self.device = device
        self._stream: Any = None

    @property
    def active(self) -> bool:
        return self._stream is not None

    def start(self, on_audio: Callable[[Any], None]) -> None:
        sd = self._sd or _sounddevice()
        if self._stream is not None:
            return
        index = _device_index(sd, self.device, "input")
        last_error: Exception | None = None
        for rate in (16_000, None):
            try:
                if rate is None:
                    info = sd.query_devices(index, "input")
                    rate = int(info["default_samplerate"])

                def callback(indata, _frames, _time, _status, rate=rate):
                    on_audio(to_mono_16k(indata[:, 0].copy(), rate))

                stream = sd.InputStream(samplerate=rate, channels=1, dtype="float32",
                                        blocksize=int(rate * 0.03), device=index, callback=callback)
                stream.start()
            except Exception as exc:  # PortAudioError, ValueError
                last_error = exc
                continue
            self._stream = stream
            return
        raise AudioUnavailable(f"cannot open microphone: {last_error}",
                               user_message="Не вдалося увімкнути мікрофон: перевір, чи він "
                                            "підключений і чи Windows дозволяє програмам доступ до нього.")

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            log.debug("closing input stream failed", exc_info=True)
