"""Minimal runner for Piper ONNX voices whose phonemes are plain text.

Why not the ``piper-tts`` package: it is GPL-3.0 (this app is not) and ships
espeak-ng, which "text" voices such as ``uk_UA-ukrainian_tts`` never use.
For those voices the model input is simply the NFD code points of the text,
each mapped through ``phoneme_id_map`` and followed by the pad symbol,
framed by begin/end symbols - the voice config file documents all of it.
Only ``onnxruntime`` (MIT) and ``numpy`` are needed.

Thread-safety: one instance may be used from one worker thread at a time.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

BOS, EOS, PAD = "^", "$", "_"


class VoiceModelError(RuntimeError):
    """The voice files are missing, corrupt or of an unsupported kind."""


@dataclass(frozen=True, slots=True)
class VoiceConfig:
    sample_rate: int
    phoneme_id_map: dict[str, list[int]]
    speakers: dict[str, int]
    noise_scale: float
    length_scale: float
    noise_w: float

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> VoiceConfig:
        if data.get("phoneme_type") != "text":
            raise VoiceModelError(
                f"unsupported voice: phoneme_type={data.get('phoneme_type')!r} (only 'text' voices)"
            )
        id_map = data.get("phoneme_id_map")
        if not isinstance(id_map, dict) or not all(k in id_map for k in (BOS, EOS, PAD)):
            raise VoiceModelError("voice config has no usable phoneme_id_map")
        inference = data.get("inference", {})
        return cls(
            sample_rate=int(data["audio"]["sample_rate"]),
            phoneme_id_map={str(k): [int(i) for i in v] for k, v in id_map.items()},
            speakers={str(k): int(v) for k, v in (data.get("speaker_id_map") or {}).items()},
            noise_scale=float(inference.get("noise_scale", 0.667)),
            length_scale=float(inference.get("length_scale", 1.0)),
            noise_w=float(inference.get("noise_w", 0.8)),
        )

    def text_to_ids(self, text: str) -> list[int]:
        """NFD code points -> ids: BOS PAD (id PAD)* EOS.  Unknown characters
        are skipped (the normaliser should already have removed them)."""
        id_map = self.phoneme_id_map
        ids = [*id_map[BOS], *id_map[PAD]]
        for ch in unicodedata.normalize("NFD", text):
            mapped = id_map.get(ch)
            if mapped is None or ch in (BOS, EOS, PAD):
                continue
            ids.extend(mapped)
            ids.extend(id_map[PAD])
        ids.extend(id_map[EOS])
        return ids


class PiperOnnxVoice:
    """Loads ``<name>.onnx`` + ``<name>.onnx.json`` and synthesises int16 audio."""

    def __init__(self, model_path: Path | str, config_path: Path | str | None = None,
                 threads: int = 2) -> None:
        model_path = Path(model_path)
        config_path = Path(config_path) if config_path else model_path.with_name(model_path.name + ".json")
        try:
            self.config = VoiceConfig.from_json(json.loads(config_path.read_text("utf-8")))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise VoiceModelError(f"cannot read voice config {config_path.name}: {exc}") from exc
        try:
            import onnxruntime as ort  # optional dependency (voice extra)

            options = ort.SessionOptions()
            # Keep synthesis from starving a running game of CPU.
            options.intra_op_num_threads = max(1, threads)
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except ImportError as exc:
            raise VoiceModelError("onnxruntime is not installed") from exc
        except Exception as exc:  # onnxruntime raises its own exception types
            raise VoiceModelError(f"cannot load voice model {model_path.name}: {exc}") from exc
        self._input_names = {i.name for i in self._session.get_inputs()}

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def speakers(self) -> dict[str, int]:
        return dict(self.config.speakers)

    def synthesize(self, text: str, speaker: str | None = None, speed: float = 1.0,
                   volume: float = 1.0):
        """Return mono int16 samples (numpy array) for already-normalised *text*.

        *speed* > 1 talks faster (Piper's ``length_scale`` is its inverse).
        Audio is peak-normalised, then scaled by *volume* (0..1).
        """
        import numpy as np

        ids = self.config.text_to_ids(text)
        if len(ids) <= 3:
            return np.zeros(0, dtype=np.int16)
        speed = min(max(speed, 0.5), 2.0)
        inputs: dict[str, Any] = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array(
                [self.config.noise_scale, self.config.length_scale / speed, self.config.noise_w],
                dtype=np.float32,
            ),
        }
        if "sid" in self._input_names:
            sid = self.config.speakers.get(speaker or "", 0)
            inputs["sid"] = np.array([sid], dtype=np.int64)
        audio = self._session.run(None, inputs)[0].squeeze().astype(np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak < 1e-8:
            return np.zeros(audio.shape, dtype=np.int16)
        audio = np.clip(audio / peak * min(max(volume, 0.0), 1.0), -1.0, 1.0)
        return (audio * 32767).astype(np.int16)
