import sys
import types
from typing import ClassVar

import numpy as np
import pytest

from chibi.voice.stt import Segment, STTError, WhisperSTT, clean_segments


def test_keeps_normal_speech():
    text, conf = clean_segments([Segment("Мія, котра година?", avg_logprob=-0.2)])
    assert text == "Мія, котра година?"
    assert 0.7 < conf <= 1.0


@pytest.mark.parametrize(
    "phantom",
    ["Дякую за перегляд!", "Субтитри створені спільнотою Amara.org", "Продовження наступне...",
     "Спасибо за просмотр!", "[Музика]", "♪♪♪", "...", "Редактор субтитрів А. Синецька",
     "Субтитры сделал DimaTorzok"],
)
def test_drops_known_hallucinations(phantom):
    assert clean_segments([Segment(phantom)]) == ("", 0.0)


def test_drops_phantom_but_keeps_real_words_in_other_segments():
    text, _ = clean_segments([Segment("Мія, привіт."), Segment("Дякую за перегляд!")])
    assert text == "Мія, привіт."


def test_real_sentence_mentioning_music_is_kept():
    text, _ = clean_segments([Segment("Увімкни мені музику, будь ласка")])
    assert text == "Увімкни мені музику, будь ласка"


def test_drops_segments_whisper_marks_as_no_speech():
    segs = [Segment("Ну що ж", no_speech_prob=0.9, avg_logprob=-1.5)]
    assert clean_segments(segs)[0] == ""
    # High no-speech probability alone (confident text) is kept.
    segs = [Segment("Так, давай", no_speech_prob=0.9, avg_logprob=-0.3)]
    assert clean_segments(segs)[0] == "Так, давай"


def test_drops_repetition_loops():
    assert clean_segments([Segment("так " * 30, compression_ratio=3.1)])[0] == ""


def test_missing_model_reports_a_user_message(tmp_path):
    stt = WhisperSTT(tmp_path / "nothing")
    with pytest.raises(STTError) as info:
        stt.transcribe(np.zeros(16_000, dtype=np.float32))
    assert "Модель" in info.value.user_message


def test_too_short_audio_is_not_sent_to_the_model(tmp_path):
    stt = WhisperSTT(tmp_path)
    result = stt.transcribe(np.zeros(1000, dtype=np.float32))
    assert result.text == "" and not stt.loaded


def test_transcribe_uses_array_input_and_cleans(tmp_path, monkeypatch):
    (tmp_path / "model.bin").write_bytes(b"x")
    calls = {}

    class FakeModel:
        def __init__(self, path, **kwargs):
            calls["init"] = (path, kwargs)

        def transcribe(self, audio, **kwargs):
            calls["audio"] = audio
            calls["kwargs"] = kwargs
            seg = types.SimpleNamespace(text=" Привіт, Міє! ", no_speech_prob=0.01, avg_logprob=-0.1,
                                        compression_ratio=1.1)
            phantom = types.SimpleNamespace(text="Дякую за перегляд!", no_speech_prob=0.5,
                                            avg_logprob=-0.8, compression_ratio=1.0)
            return iter([seg, phantom]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeModel))
    stt = WhisperSTT(tmp_path, threads=2)
    result = stt.transcribe(np.zeros(32_000, dtype=np.float32), prompt="Мія")
    assert result.text == "Привіт, Міє!"
    assert result.audio_seconds == pytest.approx(2.0)
    assert calls["init"][1]["compute_type"] == "int8"
    assert calls["init"][1]["device"] == "cpu"
    assert isinstance(calls["audio"], np.ndarray) and calls["audio"].dtype == np.float32
    assert calls["kwargs"]["language"] == "uk"
    assert calls["kwargs"]["initial_prompt"] == "Мія"
    assert stt.loaded
    stt.unload()
    assert not stt.loaded


def test_user_talking_about_subtitles_is_kept():
    assert clean_segments([Segment("Увімкни субтитри українською")])[0] == "Увімкни субтитри українською"


class RecordingModel:
    """faster_whisper.WhisperModel stand-in that records every transcription."""

    instances: ClassVar[list] = []

    def __init__(self, path, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.fail_first = False
        RecordingModel.instances.append(self)

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("first call failed")
        seg = types.SimpleNamespace(text=" Так ", no_speech_prob=0.01, avg_logprob=-0.1,
                                    compression_ratio=1.0)
        return iter([seg]), None


@pytest.fixture
def recording(tmp_path, monkeypatch):
    (tmp_path / "model.bin").write_bytes(b"x")
    RecordingModel.instances.clear()
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=RecordingModel))
    return tmp_path


def test_warming_up_after_loading_so_the_first_request_is_not_the_slow_one(recording):
    stt = WhisperSTT(recording)
    stt.warm_up()  # not loaded: nothing to warm up (and no load on the side)
    assert RecordingModel.instances == []
    stt.load()
    model = RecordingModel.instances[-1]
    assert model.calls == []  # loading alone: a job of its own, quick
    stt.warm_up()
    assert len(model.calls) == 1  # one throwaway transcription of quiet audio
    audio, kwargs = model.calls[0]
    assert audio.dtype == np.float32 and audio.size == 16_000 and float(np.abs(audio).max()) < 0.01
    assert kwargs["initial_prompt"] is None and kwargs["max_new_tokens"] <= 64
    stt.warm_up()  # once is enough
    stt.transcribe(np.zeros(32_000, dtype=np.float32))
    assert len(model.calls) == 2


def test_no_warm_up_after_a_real_request(recording):
    stt = WhisperSTT(recording)
    stt.load()
    stt.transcribe(np.zeros(32_000, dtype=np.float32))
    stt.warm_up()  # the user's request already did what the warm-up is for
    assert len(RecordingModel.instances[-1].calls) == 1


def test_warm_up_can_be_switched_off_and_never_breaks_loading(recording):
    stt = WhisperSTT(recording, warm_up=False)
    stt.load()
    stt.warm_up()
    assert RecordingModel.instances[-1].calls == []

    class FailingFirst(RecordingModel):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            self.fail_first = True

    sys.modules["faster_whisper"].WhisperModel = FailingFirst
    stt = WhisperSTT(recording)
    stt.load()
    stt.warm_up()  # failed: never fatal
    assert stt.loaded
    assert stt.transcribe(np.zeros(32_000, dtype=np.float32)).text == "Так"


def test_decoding_options_greedy_and_capped_by_the_phrase_length(recording):
    stt = WhisperSTT(recording, warm_up=False)
    stt.transcribe(np.zeros(2 * 16_000, dtype=np.float32), prompt="Мія, привіт.")
    stt.transcribe(np.zeros(60 * 16_000, dtype=np.float32))
    (_a, short), (_b, long_) = RecordingModel.instances[-1].calls
    assert short["beam_size"] == 1 and short["temperature"] == 0.0
    assert short["without_timestamps"] and not short["condition_on_previous_text"] and not short["vad_filter"]
    assert short["max_new_tokens"] == 24 + 2 * 24  # ~7 tokens a second of speech: room to spare
    assert long_["max_new_tokens"] is None  # the model's own limit per 30-second window
    assert short["initial_prompt"] == "Мія, привіт." and long_["initial_prompt"] is None


class HalvingModel(RecordingModel):
    """Decodes like faster-whisper 1.2 on CTranslate2 4.x, measured with "small": with
    ``max_new_tokens`` set it stops after about (prompt tokens + max_new_tokens) / 2
    tokens (a 25 s clip: 155 / 117 / 80 tokens for caps of 300 / 224 / 150, 161
    uncapped); uncapped, a 30-second window gives at most 224.  The "speech" here runs
    at 9 tokens a second, the fastest rate measured."""

    RATE = 9

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        spoken = round(self.RATE * audio.size / 16_000)
        prompt = 4 + len((kwargs.get("initial_prompt") or "").split())  # start tokens + the prompt
        cap = kwargs.get("max_new_tokens")
        limit = (prompt + cap) // 2 if cap is not None else 224
        if cap is not None and prompt + cap > 448:
            raise ValueError("the prompt and max_new_tokens exceed the model's max_length")
        seg = types.SimpleNamespace(text=" ".join(["слово"] * min(spoken, limit)), no_speech_prob=0.01,
                                    avg_logprob=-0.1, compression_ratio=1.0)
        return iter([seg]), None


@pytest.mark.parametrize("prompt", ["", "Мія, привіт."])
def test_the_token_cap_never_cuts_off_a_long_request(recording, monkeypatch, prompt):
    """The end of a long request is often the question itself: up to the model's own
    window limit (224 tokens, ~24 s at the fastest rate) every word must come through."""
    monkeypatch.setattr(sys.modules["faster_whisper"], "WhisperModel", HalvingModel)
    stt = WhisperSTT(recording, warm_up=False)
    for seconds in (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24):
        text = stt.transcribe(np.zeros(seconds * 16_000, dtype=np.float32), prompt=prompt).text
        assert len(text.split()) == HalvingModel.RATE * seconds, seconds


def test_short_noisy_clips_stay_capped(recording):
    """A hallucination loop on a short noisy clip ("так так так ...") stops early: the
    open microphone recognises one phrase at a time, and a long loop would make the
    next real request wait."""
    stt = WhisperSTT(recording, warm_up=False)
    stt.transcribe(np.zeros(2 * 16_000, dtype=np.float32), prompt="Мія, привіт.")
    (_audio, kwargs), = RecordingModel.instances[-1].calls
    assert kwargs["max_new_tokens"] is not None and kwargs["max_new_tokens"] <= 100


def test_no_warm_up_when_a_waiting_request_loads_the_model(recording):
    stt = WhisperSTT(recording)  # never preloaded: the user's request loads it
    stt.transcribe(np.zeros(32_000, dtype=np.float32))
    assert len(RecordingModel.instances[-1].calls) == 1  # just the request
