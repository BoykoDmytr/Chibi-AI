import json
import unicodedata

import httpx2
import numpy as np
import pytest

from chibi.voice.piper_onnx import BOS, EOS, PAD, VoiceConfig, VoiceModelError
from chibi.voice.tts import AudioClip, AzureTTS, PiperTTS, TTSError

ID_MAP = {BOS: [1], EOS: [2], PAD: [0], " ": [3], "а": [10], "б": [11], "и": [12], "й": [99],
          "̆": [13], "і": [14], "̈": [15], "!": [4]}
CONFIG = {
    "audio": {"sample_rate": 22050},
    "phoneme_type": "text",
    "phoneme_id_map": ID_MAP,
    "speaker_id_map": {"lada": 0, "tetiana": 2},
    "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8},
}


def test_voice_config_maps_nfd_code_points_with_padding():
    cfg = VoiceConfig.from_json(CONFIG)
    # "й" is decomposed (NFD) into "и" + combining breve, exactly as the voice was trained.
    assert cfg.text_to_ids("ай!") == [1, 0, 10, 0, 12, 0, 13, 0, 4, 0, 2]
    # Control symbols and unknown characters never reach the model.
    assert cfg.text_to_ids("а^_$ж") == [1, 0, 10, 0, 2]
    assert cfg.text_to_ids("ї") == [1, 0, 14, 0, 15, 0, 2]
    assert unicodedata.normalize("NFD", "ї") == "ї"


@pytest.mark.parametrize("bad", [{**CONFIG, "phoneme_type": "espeak"},
                                 {**CONFIG, "phoneme_id_map": {"а": [1]}}])
def test_unsupported_voice_configs_are_rejected(bad):
    with pytest.raises(VoiceModelError):
        VoiceConfig.from_json(bad)


class FakeVoice:
    sample_rate = 22050

    def __init__(self, path, threads):
        self.path = path
        self.threads = threads
        self.calls = []

    def synthesize(self, text, speaker=None, speed=1.0):
        self.calls.append((text, speaker, speed))
        return np.ones(int(self.sample_rate * 0.5), dtype=np.int16)


@pytest.fixture
def voice_dir(tmp_path):
    (tmp_path / "v.onnx").write_bytes(b"fake")
    (tmp_path / "v.onnx.json").write_text(json.dumps(CONFIG))
    return tmp_path


def test_piper_tts_normalises_splits_and_joins(voice_dir):
    voices = []

    def factory(path, threads):
        voices.append(FakeVoice(path, threads))
        return voices[-1]

    tts = PiperTTS(voice_dir, "v", speaker="tetiana", speed=1.2, threads=3, voice_factory=factory)
    long_text = "Ти вже 2 години тут! " + "Дуже довге речення про все на світі, " * 8 + "кінець."
    clip = tts.synthesize(long_text)
    voice = voices[0]
    assert voice.threads == 3
    texts = [c[0] for c in voice.calls]
    assert texts[0].startswith("ти вже дві години тут!")
    assert len(texts) >= 2
    assert all(c[1] == "tetiana" and c[2] == 1.2 for c in voice.calls)
    gap = int(22050 * 0.12)
    assert len(clip.samples) == len(texts) * 11025 + (len(texts) - 1) * gap
    assert clip.sample_rate == 22050
    assert tts.is_local


def test_piper_tts_unspeakable_text_gives_empty_clip(voice_dir):
    tts = PiperTTS(voice_dir, "v", voice_factory=FakeVoice)
    assert tts.synthesize("🙂🙂").empty


def test_piper_tts_missing_model_explains(tmp_path):
    with pytest.raises(TTSError) as info:
        PiperTTS(tmp_path, "nothing").synthesize("привіт")
    assert "не завантажено" in info.value.user_message


def test_audio_clip_duration():
    assert AudioClip(np.zeros(44100, dtype=np.int16), 22050).duration == 2.0
    assert AudioClip(np.zeros(0, dtype=np.int16), 22050).empty


def azure(handler, key="secret-key", **kwargs):
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    return AzureTTS(lambda: key, region="westeurope", client=client, **kwargs)


def test_azure_sends_ssml_and_returns_pcm():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = request.content.decode("utf-8")
        return httpx2.Response(200, content=np.arange(100, dtype="<i2").tobytes())

    tts = azure(handler, voice="uk-UA-PolinaNeural", speed=1.2)
    clip = tts.synthesize("Привіт <друже> & *сміється* 🙂 о 14:30")
    assert seen["url"] == "https://westeurope.tts.speech.microsoft.com/cognitiveservices/v1"
    assert seen["headers"]["ocp-apim-subscription-key"] == "secret-key"
    assert seen["headers"]["x-microsoft-outputformat"] == "raw-24khz-16bit-mono-pcm"
    body = seen["body"]
    assert "uk-UA-PolinaNeural" in body and "rate='+20%'" in body
    assert "&lt;друже&gt; &amp;" in body  # escaped, not markup
    assert "сміється" not in body and "🙂" not in body
    assert "14:30" in body  # Azure reads numbers itself
    assert clip.sample_rate == 24000 and len(clip.samples) == 100
    assert not tts.is_local


@pytest.mark.parametrize(("status", "retryable", "fragment"),
                         [(401, False, "Ключ"), (403, False, "Ключ"), (429, True, "забагато"),
                          (500, True, ""), (400, False, "")])
def test_azure_errors(status, retryable, fragment):
    tts = azure(lambda request: httpx2.Response(status))
    with pytest.raises(TTSError) as info:
        tts.synthesize("привіт")
    assert info.value.retryable is retryable
    assert fragment in info.value.user_message


def test_azure_without_key_or_with_bad_region():
    with pytest.raises(TTSError) as info:
        azure(lambda r: httpx2.Response(200), key="").synthesize("привіт")
    assert "ключа" in info.value.user_message
    tts = azure(lambda r: httpx2.Response(200))
    tts.region = "evil.example.com/x"
    with pytest.raises(TTSError):
        tts.synthesize("привіт")


def test_azure_network_error_is_retryable():
    def handler(request):
        raise httpx2.ConnectError("down")

    with pytest.raises(TTSError) as info:
        azure(handler).synthesize("привіт")
    assert info.value.retryable


def test_azure_empty_text_makes_no_request():
    def handler(request):
        raise AssertionError("no request expected")

    assert azure(handler).synthesize("🙂 *кліпає*").empty
