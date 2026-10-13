import numpy as np
import pytest

from chibi.voice.vad import EnergyVAD, VADConfig, to_mono_16k

SR = 16_000


def noise(seconds, level=0.002, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(SR * seconds)) * level).astype(np.float32)


def speech(seconds, amplitude=0.2, seed=1):
    """Voiced-ish signal: a 180 Hz tone with harmonics and syllable-rate wobble."""
    t = np.arange(int(SR * seconds)) / SR
    tone = np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 360 * t)
    envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 4 * t)
    return (amplitude * tone * envelope).astype(np.float32) + noise(seconds, seed=seed)


def feed_in_chunks(vad, audio, chunk=512):
    events = []
    for i in range(0, audio.size, chunk):
        events += vad.feed(audio[i:i + chunk])
    return events


def test_detects_start_and_end_of_an_utterance():
    vad = EnergyVAD()
    audio = np.concatenate([noise(1.0), speech(1.5), noise(1.5)])
    events = feed_in_chunks(vad, audio)
    assert [e.kind for e in events] == ["start", "end"]
    end = events[1]
    assert end.reason == "silence"
    # 1.5 s of speech + pre-roll (0.3) + tail (0.25), give or take a frame or two.
    assert 1.6 <= end.audio.size / SR <= 2.2
    assert not vad.in_speech


def test_pre_roll_keeps_the_first_syllable():
    vad = EnergyVAD()
    audio = np.concatenate([noise(1.0), speech(1.0), noise(1.2)])
    end = next(e for e in feed_in_chunks(vad, audio) if e.kind == "end")
    # The utterance starts in the noise just before the speech onset.
    first_frames = end.audio[: int(0.1 * SR)]
    assert np.sqrt(np.mean(first_frames ** 2)) < 0.01


def test_clicks_and_short_bursts_are_ignored():
    vad = EnergyVAD()
    audio = np.concatenate([noise(1.0), speech(0.06), noise(1.0), speech(0.18), noise(1.5)])
    assert [e for e in feed_in_chunks(vad, audio) if e.kind == "end"] == []


def test_short_pause_inside_a_sentence_does_not_end_it():
    vad = EnergyVAD(VADConfig(end_silence_ms=800))
    audio = np.concatenate([noise(0.5), speech(0.8), noise(0.4), speech(0.8), noise(1.5)])
    ends = [e for e in feed_in_chunks(vad, audio) if e.kind == "end"]
    assert len(ends) == 1
    assert ends[0].audio.size / SR > 1.9


def syllables(seconds):
    """Speech-like bursts with dips between them (real speech is never flat)."""
    audio = speech(seconds)
    period = int(0.28 * SR)
    for start in range(int(0.2 * SR), audio.size, period):
        audio[start:start + int(0.08 * SR)] *= 0.08
    return audio


def test_max_length_cuts_endless_speech():
    vad = EnergyVAD(VADConfig(max_utterance_seconds=2.0))
    audio = np.concatenate([noise(0.5), syllables(5.0)])
    ends = [e for e in feed_in_chunks(vad, audio) if e.kind == "end"]
    assert ends and ends[0].reason == "max_length"
    assert ends[0].audio.size / SR == pytest.approx(2.0, abs=0.05)


def test_adapts_to_a_noisy_room():
    loud = 0.02
    vad = EnergyVAD()
    talking = noise(1.2, level=loud, seed=5) + speech(1.2, amplitude=0.25)
    audio = np.concatenate([noise(2.0, level=loud), talking, noise(1.5, level=loud)])
    ends = [e for e in feed_in_chunks(vad, audio) if e.kind == "end"]
    # Before the floor is learnt the noise may look like a start, but only the
    # real speech is reported as an utterance.
    assert len(ends) == 1
    assert 1.1 <= ends[0].audio.size / SR <= 2.0
    assert vad.noise_floor > 0.01


def test_steady_noise_alone_never_becomes_an_utterance():
    vad = EnergyVAD()
    assert [e for e in feed_in_chunks(vad, noise(8.0, level=0.03)) if e.kind == "end"] == []
    assert not vad.in_speech  # the false start was recognised as noise


def test_learnt_floor_carries_over_to_the_next_session():
    first = EnergyVAD()
    feed_in_chunks(first, noise(4.0, level=0.03))
    second = EnergyVAD(initial_floor=first.noise_floor)
    assert feed_in_chunks(second, noise(1.0, level=0.03, seed=3)) == []


def test_speech_right_after_the_microphone_opens_is_heard():
    vad = EnergyVAD()
    events = feed_in_chunks(vad, np.concatenate([speech(1.2), noise(1.2)]))
    assert [e.kind for e in events] == ["start", "end"]


def test_finish_returns_speech_so_far_for_push_to_talk_release():
    vad = EnergyVAD()
    feed_in_chunks(vad, np.concatenate([noise(0.5), speech(1.0)]))
    assert vad.in_speech
    audio = vad.finish()
    assert audio is not None and audio.size / SR > 0.9
    assert not vad.in_speech
    assert vad.finish() is None


def test_finish_without_speech_returns_none():
    vad = EnergyVAD()
    feed_in_chunks(vad, noise(1.0))
    assert vad.finish() is None


def test_level_meter_is_between_zero_and_one():
    vad = EnergyVAD()
    vad.feed(noise(0.1))
    quiet = vad.level
    vad.feed(speech(0.1, amplitude=0.5))
    assert 0.0 <= quiet < vad.level <= 1.0


def test_long_speech_with_dips_is_not_mistaken_for_noise():
    vad = EnergyVAD()
    ends = [e for e in feed_in_chunks(vad, np.concatenate([noise(0.5), syllables(6.0), noise(1.2)]))
            if e.kind == "end"]
    assert len(ends) == 1 and ends[0].audio.size / SR > 5.5


def test_odd_chunk_sizes_are_buffered():
    vad = EnergyVAD()
    audio = np.concatenate([noise(0.7), speech(1.0), noise(1.2)])
    assert [e.kind for e in feed_in_chunks(vad, audio, chunk=137)] == ["start", "end"]


def test_to_mono_16k_resamples_and_converts():
    t = np.arange(48_000) / 48_000
    stereo = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32)
    out = to_mono_16k(stereo, 48_000)
    assert out.dtype == np.float32 and out.size == 16_000
    assert np.max(np.abs(out)) == pytest.approx(1.0, abs=0.05)
    pcm = (np.ones(160) * 16384).astype(np.int16)
    assert to_mono_16k(pcm, 16_000)[0] == pytest.approx(0.5)
    assert to_mono_16k(np.zeros(0, dtype=np.float32), 44_100).size == 0
