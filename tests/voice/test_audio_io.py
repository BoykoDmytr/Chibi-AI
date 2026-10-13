"""audio_io against a fake ``sounddevice`` module (no audio hardware in CI)."""

from typing import ClassVar

import numpy as np
import pytest

from chibi.voice.audio_io import AudioUnavailable, SoundDeviceInput, SoundDeviceOutput, list_devices
from chibi.voice.tts import AudioClip


class CallbackStop(Exception):
    pass


class FakeOutputStream:
    def __init__(self, samplerate, channels, dtype, device, callback, finished_callback):
        self.samplerate = samplerate
        self.device = device
        self.callback = callback
        self.finished = finished_callback
        self.started = self.closed = self.aborted = False

    def start(self):
        self.started = True

    def pump(self, frames=1024):
        """Simulate the audio thread pulling data until the clip ends."""
        while True:
            out = np.zeros((frames, 1), dtype=np.int16)
            try:
                self.callback(out, frames, None, None)
            except CallbackStop:
                self.finished()
                return out
            yield out

    def abort(self):
        self.aborted = True
        self.finished()

    def close(self):
        self.closed = True


class FakeInputStream:
    fail_rates: ClassVar[set[int]] = set()

    def __init__(self, samplerate, channels, dtype, blocksize, device, callback):
        if samplerate in self.fail_rates:
            raise ValueError("Invalid sample rate")
        self.samplerate = samplerate
        self.callback = callback
        self.device = device
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def close(self):
        pass


class FakeSD:
    CallbackStop = CallbackStop

    def __init__(self):
        self.outputs = []
        self.inputs = []
        self.default = type("D", (), {"hostapi": 0})()

    def OutputStream(self, **kwargs):
        stream = FakeOutputStream(**kwargs)
        self.outputs.append(stream)
        return stream

    def InputStream(self, **kwargs):
        stream = FakeInputStream(**kwargs)
        self.inputs.append(stream)
        return stream

    def query_devices(self, index=None, kind=None):
        devices = [
            {"name": "Mic A", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0,
             "default_samplerate": 48000.0},
            {"name": "Speakers", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2,
             "default_samplerate": 48000.0},
            {"name": "Mic A", "hostapi": 1, "max_input_channels": 1, "max_output_channels": 0,
             "default_samplerate": 44100.0},
        ]
        if index is None and kind is None:
            return devices
        return devices[index or 0]


@pytest.fixture
def sd():
    FakeInputStream.fail_rates = set()
    return FakeSD()


def test_list_devices_uses_default_host_api_without_duplicates(sd):
    assert list_devices(sd) == (["Mic A"], ["Speakers"])


def test_output_plays_whole_clip_with_volume_and_reports_completion(sd):
    out = SoundDeviceOutput(sd=sd)
    done = []
    clip = AudioClip(np.full(2500, 10000, dtype=np.int16), 22050)
    out.play(clip, 0.5, done.append)
    stream = sd.outputs[0]
    assert stream.started and stream.samplerate == 22050
    chunks = list(stream.pump())
    played = np.concatenate([c[:, 0] for c in chunks])
    assert (played[:2048] == 5000).all()
    assert done == [True]


def test_output_stop_reports_interruption_once(sd):
    out = SoundDeviceOutput(sd=sd)
    done = []
    out.play(AudioClip(np.ones(100000, dtype=np.int16), 22050), 1.0, done.append)
    out.stop()
    out.stop()
    assert done == [False]
    assert sd.outputs[0].aborted and sd.outputs[0].closed


def test_new_clip_interrupts_the_previous_one(sd):
    out = SoundDeviceOutput(sd=sd)
    first, second = [], []
    out.play(AudioClip(np.ones(100000, dtype=np.int16), 22050), 1.0, first.append)
    out.play(AudioClip(np.ones(10, dtype=np.int16), 22050), 1.0, second.append)
    assert first == [False]
    list(sd.outputs[1].pump())
    assert second == [True]


def test_output_selects_named_device(sd):
    out = SoundDeviceOutput(device="Speakers", sd=sd)
    out.play(AudioClip(np.ones(10, dtype=np.int16), 22050), 1.0, lambda c: None)
    assert sd.outputs[0].device == 1


def test_input_delivers_16k_mono_float_chunks(sd):
    mic = SoundDeviceInput(sd=sd)
    got = []
    mic.start(got.append)
    stream = sd.inputs[0]
    assert stream.samplerate == 16000 and mic.active
    stream.callback(np.ones((480, 1), dtype=np.float32) * 0.5, 480, None, None)
    assert got[0].dtype == np.float32 and got[0].shape == (480,)
    mic.stop()
    assert stream.stopped and not mic.active


def test_input_falls_back_to_device_rate_and_resamples(sd):
    FakeInputStream.fail_rates = {16000}
    mic = SoundDeviceInput(sd=sd)
    got = []
    mic.start(got.append)
    stream = sd.inputs[-1]
    assert stream.samplerate == 48000
    stream.callback(np.zeros((1440, 1), dtype=np.float32), 1440, None, None)
    assert got[0].shape == (480,)


def test_input_failure_raises_friendly_error(sd):
    FakeInputStream.fail_rates = {16000, 48000}
    with pytest.raises(AudioUnavailable) as info:
        SoundDeviceInput(sd=sd).start(lambda chunk: None)
    assert "мікрофон" in info.value.user_message
