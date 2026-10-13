"""Speech recognition: local Whisper (faster-whisper / CTranslate2) in Ukrainian.

Audio never leaves the PC.  The model runs on the CPU with int8 weights;
"small" transcribes ~5 s of speech in ~1-3 s on a 4-core desktop.

Audio is passed as a float32 16 kHz numpy array - never as a file - which
also sidesteps faster-whisper's PyAV decoder (an incompatible ``av`` release
breaks ``transcribe("file.wav")`` with ``unexpected keyword 'metadata_errors'``).

Whisper invents text on silence and noise ("Дякую за перегляд!",
"Субтитри..." - its training data were subtitled videos);
:func:`clean_segments` drops such segments so Mia never answers a phantom.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from chibi.personality.text_uk import normalize as normalize_uk

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    #: Seconds of audio that were recognised.
    audio_seconds: float
    #: Rough 0..1 confidence (mean token probability of the kept segments).
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class Segment:
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0


class STTEngine(Protocol):
    """Implementations are called from one worker thread at a time."""

    def transcribe(self, audio: Any, prompt: str = "") -> Transcript: ...


class STTError(RuntimeError):
    def __init__(self, message: str, *, user_message: str = "") -> None:
        super().__init__(message)
        self.user_message = user_message or "Не вдалося розпізнати мовлення."


# Whisper hallucinates on silence and noise, mostly subtitle credits and
# outros from its training videos (Ukrainian and Russian).  Matched against
# normalised text (see chibi.personality.text_uk.normalize).
#: Any segment containing one of these is dropped whole (credits fill a segment).
CREDIT_PATTERNS: tuple[str, ...] = (
    r"дякую за перегляд", r"дякуємо за перегляд", r"дякую за підписку",
    r"підписуйте?(?:сь|ся) на (?:наш )?канал", r"ставте лайки",
    r"субтитр\w* (?:зроб|створ|підгот|переклав|переклад)", r"редактор субтитр\w*",
    r"коректор\b", r"продовження (?:наступне|буде)", r"до зустрічі в наступному відео",
    r"спасибо за просмотр", r"продолжение следует", r"субтитры (?:сдел|созд|подгот)",
    r"редактор субтитров", r"подписывайтесь на канал", r"amara", r"dimatorzok",
)
#: A segment that is nothing but one of these is a sound, not speech ("Музика.").
SOUND_WORDS: frozenset[str] = frozenset({"музика", "музыка", "оплески", "аплодисменти", "сміх"})
_CREDITS_RE = re.compile("|".join(CREDIT_PATTERNS))
_NON_SPEECH = re.compile(r"^[\s\W_]*$|^\s*[\[(♪*].*[\])♪*]\s*$")


def clean_segments(segments: Iterable[Segment], *, no_speech_threshold: float = 0.6,
                   logprob_threshold: float = -1.0, max_compression: float = 2.4) -> tuple[str, float]:
    """Keep the believable segments; returns (text, confidence 0..1).

    Short real answers ("так", "дякую") survive unless Whisper itself
    doubts there was speech (the no-speech check).
    """
    import math

    kept: list[str] = []
    probs: list[float] = []
    for seg in segments:
        text = seg.text.strip()
        if not text or _NON_SPEECH.match(text):
            continue
        if seg.compression_ratio > max_compression:
            continue  # "так так так так так..." loops
        normalized = normalize_uk(text)
        if _CREDITS_RE.search(normalized):
            continue
        doubtful = seg.no_speech_prob > no_speech_threshold and seg.avg_logprob < logprob_threshold
        if doubtful:
            continue  # Whisper itself thinks there was no speech
        if normalized in SOUND_WORDS:
            continue
        kept.append(text)
        probs.append(math.exp(min(0.0, seg.avg_logprob)))
    text = " ".join(kept).strip()
    confidence = sum(probs) / len(probs) if probs else 0.0
    return text, confidence


class WhisperSTT:
    """faster-whisper with lazy loading.  ``model_dir`` holds a CTranslate2
    Whisper model (config.json, model.bin, tokenizer.json, vocabulary.txt)."""

    def __init__(self, model_dir: Path | str, *, threads: int = 2, beam_size: int = 1,
                 language: str = "uk") -> None:
        self.model_dir = Path(model_dir)
        self.threads = max(1, min(threads, os.cpu_count() or 1))
        self.beam_size = max(1, beam_size)
        self.language = language
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model (seconds; call from a worker thread)."""
        with self._lock:
            if self._model is not None:
                return
            if not (self.model_dir / "model.bin").is_file():
                raise STTError(f"no model in {self.model_dir}",
                               user_message="Модель розпізнавання мовлення не завантажена.")
            try:
                from faster_whisper import WhisperModel  # optional dependency (voice extra)
            except ImportError as exc:
                raise STTError("faster-whisper is not installed",
                               user_message="Розпізнавання мовлення не встановлено.") from exc
            try:
                self._model = WhisperModel(str(self.model_dir), device="cpu", compute_type="int8",
                                           cpu_threads=self.threads, num_workers=1)
            except Exception as exc:  # CTranslate2 raises RuntimeError / ValueError
                raise STTError(f"cannot load whisper model: {exc}",
                               user_message="Не вдалося завантажити модель розпізнавання.") from exc
            log.info("speech recognition model loaded from %s", self.model_dir.name)

    def unload(self) -> None:
        with self._lock:
            self._model = None

    def transcribe(self, audio: Any, prompt: str = "") -> Transcript:
        import numpy as np

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        seconds = samples.size / SAMPLE_RATE
        if samples.size < SAMPLE_RATE // 4:
            return Transcript("", seconds, 0.0)
        self.load()
        try:
            segments, _info = self._model.transcribe(
                samples,
                language=self.language,
                task="transcribe",
                beam_size=self.beam_size,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt=prompt or None,
                vad_filter=False,  # our VAD already trimmed the silence
                without_timestamps=True,
            )
            raw = [Segment(s.text, s.no_speech_prob, s.avg_logprob, s.compression_ratio)
                   for s in segments]
        except Exception as exc:
            raise STTError(f"transcription failed: {exc}") from exc
        text, confidence = clean_segments(raw)
        return Transcript(text, seconds, confidence)
