"""Speech recognition: local Whisper (faster-whisper / CTranslate2) in Ukrainian.

Audio never leaves the PC.  The model runs on the CPU with int8 weights;
"small" transcribes ~5 s of speech in ~1-3 s on a 4-core desktop.

Audio is passed as a float32 16 kHz numpy array - never as a file - which
also sidesteps faster-whisper's PyAV decoder (an incompatible ``av`` release
breaks ``transcribe("file.wav")`` with ``unexpected keyword 'metadata_errors'``).

Whisper invents text on silence and noise ("Дякую за перегляд!",
"Субтитри..." - its training data were subtitled videos);
:func:`clean_segments` drops such segments so Mia never answers a phantom.

Decoding options, measured with "small" (int8) on 54 Ukrainian requests
(three Piper voices, two speeds, light noise; 2-6 s each): greedy decoding
(``beam_size=1``) is both the fastest and the most accurate - beam 2 / 5
took 9 % / 16 % longer and made more word errors (28.7 % / 28.9 % against
25.9 %).  Timestamps, Whisper's own VAD (our VAD has trimmed the audio
already), conditioning on previous text and the temperature fallback
changed neither speed nor accuracy on these phrases (the fallback only
costs time - a whole extra decoding - on audio it struggles with), so they
stay off; without the prompt (her name) accuracy dropped to 28.1 %.  More
threads help most: 4 instead of 2 took 18 % less time at the same accuracy
(``VoiceSettings.cpu_threads``, on the voice page).
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

#: Decoding cap for short clips: a hallucination loop on a short noisy clip ("так
#: так так ...") stops early instead of running on for seconds while the next real
#: request waits.  Measured with faster-whisper 1.2 / CTranslate2 4.8 ("small"):
#: with ``max_new_tokens`` set, decoding stops after about (prompt tokens +
#: max_new_tokens) / 2 tokens - not max_new_tokens - so the cap leaves room for 12
#: tokens a second plus 14, and Ukrainian speech is ~7 Whisper tokens a second (9
#: at most in our samples).  Longer clips are not capped at all: their limit is the
#: model's own (224 tokens per 30-second window), and a cap there cut the end of a
#: request off - often the question itself.
TOKENS_PER_SECOND = 24
TOKENS_BASE = 24
CAPPED_SECONDS = 8.0
#: Seconds of near-silence transcribed once right after loading, in the background:
#: the first transcription in a process is often slower (CTranslate2 sets itself up;
#: 0.4 s on average and up to 2.4 s more in our measurements), and the user's first
#: request should not pay for that.
WARM_UP_SECONDS = 1.0


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
                 language: str = "uk", warm_up: bool = True) -> None:
        """*warm_up*: :meth:`warm_up` runs one throwaway recognition (off: a no-op)."""
        self.model_dir = Path(model_dir)
        self.threads = max(1, min(threads, os.cpu_count() or 1))
        self.beam_size = max(1, beam_size)
        self.language = language
        self.warm_up_enabled = warm_up
        self._model: Any = None
        self._lock = threading.Lock()
        #: The model has recognised something since it was loaded (warm-up or request).
        self._warmed = False

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model (seconds; the background preload calls this on the voice worker)."""
        self._load()

    def warm_up(self) -> None:
        """One throwaway recognition, so the user's first request is not the slow one.
        The background preload runs it as a job of its own after loading the voice too,
        so a request or a voice sample that comes in meanwhile goes first.  A no-op when
        the model is not loaded, warming up is off, or it has recognised something
        already."""
        with self._lock:
            model = self._model
            if model is None or not self.warm_up_enabled or self._warmed:
                return
            self._warmed = True
        self._warm_up(model)

    def _load(self) -> None:
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
                model = WhisperModel(str(self.model_dir), device="cpu", compute_type="int8",
                                     cpu_threads=self.threads, num_workers=1)
            except Exception as exc:  # CTranslate2 raises RuntimeError / ValueError
                raise STTError(f"cannot load whisper model: {exc}",
                               user_message="Не вдалося завантажити модель розпізнавання.") from exc
            self._model = model
            self._warmed = False
            log.info("speech recognition model loaded from %s", self.model_dir.name)

    def _warm_up(self, model: Any) -> None:
        """One throwaway transcription, so the first real one is not the slow one."""
        import numpy as np

        quiet = np.random.default_rng(0).standard_normal(round(SAMPLE_RATE * WARM_UP_SECONDS)) * 1e-3
        try:
            segments, _info = model.transcribe(quiet.astype(np.float32), **self._options(WARM_UP_SECONDS),
                                               initial_prompt=None)
            for _segment in segments:  # decoding happens while iterating
                pass
        except Exception as exc:  # never fatal: the real request simply pays the setup
            log.debug("speech recognition warm-up failed: %s", type(exc).__name__)

    def _options(self, seconds: float) -> dict[str, Any]:
        """Decoding options (see the measurements in the module docstring)."""
        return dict(
            language=self.language,
            task="transcribe",
            beam_size=self.beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,  # our VAD already trimmed the silence
            without_timestamps=True,
            max_new_tokens=(TOKENS_BASE + round(TOKENS_PER_SECOND * seconds)
                            if seconds <= CAPPED_SECONDS else None),
        )

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._warmed = False

    def transcribe(self, audio: Any, prompt: str = "") -> Transcript:
        import numpy as np

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        seconds = samples.size / SAMPLE_RATE
        if samples.size < SAMPLE_RATE // 4:
            return Transcript("", seconds, 0.0)
        # Loaded here only if the background preload has not run yet: no warm-up
        # then - the user is waiting for this very transcription.
        self._load()
        self._warmed = True
        try:
            segments, _info = self._model.transcribe(samples, **self._options(seconds),
                                                     initial_prompt=prompt or None)
            raw = [Segment(s.text, s.no_speech_prob, s.avg_logprob, s.compression_ratio)
                   for s in segments]
        except Exception as exc:
            raise STTError(f"transcription failed: {exc}") from exc
        text, confidence = clean_segments(raw)
        return Transcript(text, seconds, confidence)
