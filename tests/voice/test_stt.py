import sys
import types

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
