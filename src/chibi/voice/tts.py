"""Text-to-speech engines.

* :class:`PiperTTS` - local, offline, free: the Piper Ukrainian voice run
  with onnxruntime (see :mod:`chibi.voice.piper_onnx`).  Three speakers:
  Тетяна (default, clearest in our round-trip test), Лада, Микита.
* :class:`AzureTTS` - Microsoft Azure AI Speech, the most natural Ukrainian
  voices (Поліна, Остап).  Cloud: her lines leave the PC, so it needs
  Permission.CLOUD_AI and the user's own Azure key.

Engines take raw text and return an :class:`AudioClip`; they never touch Qt
or audio devices and are called from the voice worker thread.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from xml.sax.saxutils import escape

from chibi.voice.tts_text import PronunciationLexicon, normalize_for_tts, split_for_tts

log = logging.getLogger(__name__)

#: Pause inserted between synthesised sentences.
SENTENCE_GAP_SECONDS = 0.12

PIPER_SPEAKERS: dict[str, str] = {"tetiana": "Тетяна", "lada": "Лада", "mykyta": "Микита (чоловічий)"}
AZURE_VOICES: dict[str, str] = {"uk-UA-PolinaNeural": "Поліна", "uk-UA-OstapNeural": "Остап (чоловічий)"}


@dataclass(frozen=True, slots=True)
class AudioClip:
    """Mono int16 samples (numpy array)."""

    samples: Any
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0

    @property
    def empty(self) -> bool:
        return len(self.samples) == 0


class TTSError(RuntimeError):
    def __init__(self, message: str, *, user_message: str = "", retryable: bool = False) -> None:
        super().__init__(message)
        self.user_message = user_message or "Не вдалося озвучити репліку."
        self.retryable = retryable


class TTSEngine(Protocol):
    name: str

    @property
    def is_local(self) -> bool: ...

    def synthesize(self, text: str) -> AudioClip: ...


class PiperTTS:
    name = "piper"

    def __init__(self, model_dir: Path | str, model_id: str, *, speaker: str = "tetiana",
                 speed: float = 1.0, threads: int = 2,
                 lexicon: PronunciationLexicon | None = None,
                 voice_factory: Callable[..., Any] | None = None) -> None:
        self.model_path = Path(model_dir) / f"{model_id}.onnx"
        self.speaker = speaker
        self.speed = speed
        self.threads = threads
        self.lexicon = lexicon
        self._voice_factory = voice_factory
        self._voice: Any = None
        self._lock = threading.Lock()

    @property
    def is_local(self) -> bool:
        return True

    def load(self) -> None:
        with self._lock:
            if self._voice is not None:
                return
            if not self.model_path.is_file():
                raise TTSError(f"no voice model at {self.model_path}",
                               user_message="Голос ще не завантажено (Налаштування → Голос).")
            factory = self._voice_factory
            if factory is None:
                from chibi.voice.piper_onnx import PiperOnnxVoice, VoiceModelError

                try:
                    self._voice = PiperOnnxVoice(self.model_path, threads=self.threads)
                except VoiceModelError as exc:
                    raise TTSError(str(exc), user_message="Не вдалося завантажити голос.") from exc
            else:
                self._voice = factory(self.model_path, threads=self.threads)

    def synthesize(self, text: str) -> AudioClip:
        import numpy as np

        self.load()
        normalized = normalize_for_tts(text, self.lexicon)
        pieces = split_for_tts(normalized)
        rate = self._voice.sample_rate
        if not pieces:
            return AudioClip(np.zeros(0, dtype=np.int16), rate)
        gap = np.zeros(int(rate * SENTENCE_GAP_SECONDS), dtype=np.int16)
        chunks: list[Any] = []
        for i, piece in enumerate(pieces):
            if i:
                chunks.append(gap)
            chunks.append(self._voice.synthesize(piece, speaker=self.speaker, speed=self.speed))
        return AudioClip(np.concatenate(chunks).astype(np.int16), rate)


class AzureTTS:
    """Azure AI Speech REST API (``/cognitiveservices/v1``), 24 kHz PCM.

    *get_key* is called for each request so a key changed in the settings
    is used at once and the key is never kept in this object.
    """

    name = "azure"
    SAMPLE_RATE = 24_000

    def __init__(self, get_key: Callable[[], str | None], *, region: str,
                 voice: str = "uk-UA-PolinaNeural", speed: float = 1.0, timeout: float = 15.0,
                 client: Any = None) -> None:
        self._get_key = get_key
        self.region = region
        self.voice = voice
        self.speed = speed
        self.timeout = timeout
        self._client = client

    @property
    def is_local(self) -> bool:
        return False

    @property
    def endpoint(self) -> str:
        return f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"

    def ssml(self, text: str) -> str:
        rate = f"{round((self.speed - 1.0) * 100):+d}%"
        return (
            "<speak version='1.0' xml:lang='uk-UA'>"
            f"<voice name='{escape(self.voice, {chr(39): '&apos;'})}'>"
            f"<prosody rate='{rate}'>{escape(_clean_for_cloud(text))}</prosody>"
            "</voice></speak>"
        )

    def synthesize(self, text: str) -> AudioClip:
        import numpy as np

        if not _clean_for_cloud(text):
            return AudioClip(np.zeros(0, dtype=np.int16), self.SAMPLE_RATE)
        key = self._get_key()
        if not key:
            raise TTSError("no Azure key", user_message="Немає ключа Azure Speech.")
        if not re.fullmatch(r"[a-z0-9]{3,40}", self.region or ""):
            raise TTSError(f"bad region {self.region!r}", user_message="Невірний регіон Azure.")
        client = self._client
        own = client is None
        if own:
            import httpx2

            client = httpx2.Client(timeout=self.timeout)
        try:
            response = client.post(
                self.endpoint,
                content=self.ssml(text).encode("utf-8"),
                headers={
                    "Ocp-Apim-Subscription-Key": key,
                    "Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": "raw-24khz-16bit-mono-pcm",
                    "User-Agent": "ChibiCompanion",
                },
            )
        except Exception as exc:  # transport errors
            raise TTSError(f"Azure TTS request failed: {exc}",
                           user_message="Немає з'єднання з Azure.", retryable=True) from exc
        finally:
            if own:
                client.close()
        if response.status_code in (401, 403):
            raise TTSError(f"Azure TTS HTTP {response.status_code}",
                           user_message="Ключ Azure Speech не підходить для цього регіону.")
        if response.status_code == 429:
            raise TTSError("Azure TTS rate limited", user_message="Azure: забагато запитів.",
                           retryable=True)
        if response.status_code != 200:
            raise TTSError(f"Azure TTS HTTP {response.status_code}",
                           retryable=response.status_code >= 500)
        data = response.content
        samples = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.int16)
        return AudioClip(samples, self.SAMPLE_RATE)


_EMOJI_AND_ACTIONS = re.compile(r"\*[^*\n]{1,60}\*|[\U0001F000-\U0001FAFF☀-➿️]")


def _clean_for_cloud(text: str) -> str:
    """Azure reads numbers and Latin itself; only drop what must not be spoken."""
    return re.sub(r"\s+", " ", _EMOJI_AND_ACTIONS.sub(" ", text)).strip()
