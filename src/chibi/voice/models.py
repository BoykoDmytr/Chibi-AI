"""Voice models: what can be downloaded, where it lives, how it is verified.

Models are not bundled (hundreds of MB); the user downloads them from the
Voice settings page.  Every file is pinned to a repository *revision* and
checked against its SHA-256 before it is used, so a changed or tampered
file on the server can never be loaded.  Downloads resume after an
interruption and are written to ``*.part`` first, then renamed atomically.

Location: ``%LOCALAPPDATA%\\ChibiCompanion\\models`` on Windows (large files
do not belong in the roaming profile), next to the data directory
elsewhere; ``CHIBI_DATA_DIR`` (tests, portable installs) moves it too.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from chibi.core.paths import data_dir

log = logging.getLogger(__name__)

_PIPER_REV = "c10ece1aade47bb51c153c893d14e5bf8e5b7117"
_PIPER = f"https://huggingface.co/rhasspy/piper-voices/resolve/{_PIPER_REV}"
_WHISPER = "https://huggingface.co/Systran/faster-whisper-{size}/resolve/{rev}"
_TOKENIZER_SHA = "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab"
_VOCAB_SHA = "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"


@dataclass(frozen=True, slots=True)
class ModelFile:
    name: str  # file name inside the model directory
    url: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    kind: str  # "tts" | "stt"
    title: str  # Ukrainian, for the UI
    description: str
    license: str
    files: tuple[ModelFile, ...]

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


def _whisper(size: str, rev: str, model_bin: int, model_sha: str, config_sha: str,
             config_size: int, title: str, description: str) -> ModelSpec:
    base = _WHISPER.format(size=size, rev=rev)
    return ModelSpec(
        id=f"whisper-{size}",
        kind="stt",
        title=title,
        description=description,
        license="MIT (OpenAI Whisper, CTranslate2 conversion by Systran)",
        files=(
            ModelFile("model.bin", f"{base}/model.bin", model_bin, model_sha),
            ModelFile("config.json", f"{base}/config.json", config_size, config_sha),
            ModelFile("tokenizer.json", f"{base}/tokenizer.json", 2_203_239, _TOKENIZER_SHA),
            ModelFile("vocabulary.txt", f"{base}/vocabulary.txt", 459_861, _VOCAB_SHA),
        ),
    )


CATALOG: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in (
        ModelSpec(
            id="uk_UA-ukrainian_tts-medium",
            kind="tts",
            title="Український голос (Piper, 3 голоси)",
            description="Тетяна, Лада (жіночі) та Микита (чоловічий). Працює без інтернету.",
            license="MIT (Piper voices); dataset CC0",
            files=(
                ModelFile(
                    "uk_UA-ukrainian_tts-medium.onnx",
                    f"{_PIPER}/uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx",
                    76_735_663,
                    "7920419ac5f6fd8b6450520f24b52ed5a319cb53dd018fbcd71c9e079cbac84f",
                ),
                ModelFile(
                    "uk_UA-ukrainian_tts-medium.onnx.json",
                    f"{_PIPER}/uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx.json",
                    2_002,
                    "4e96e72917ca9b94edc77d6ccfee03a73f450ba2fc1ca93c2e562bc014e5aa55",
                ),
            ),
        ),
        _whisper("base", "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66", 145_217_532,
                 "d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9",
                 "56a6d8110d311f19c8f0471e562832c7527f146b567275bfca59fcf7c184da9a", 2_309,
                 "Розпізнавання: швидке (base)",
                 "Найлегша модель: швидка, але частіше помиляється в українській."),
        _whisper("small", "536b0662742c02347bc0e980a01041f333bce120", 483_546_902,
                 "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671",
                 "b55496ac7940a7ae47d2c01eab40edfd8701feec1229d9cce3b40014383fb828", 2_370,
                 "Розпізнавання: збалансоване (small)",
                 "Рекомендовано: добре розуміє українську, 1-3 с на фразу."),
        _whisper("medium", "08e178d48790749d25932bbc082711ddcfdfbc4f", 1_527_906_378,
                 "9b45e1009dcc4ab601eff815b61d80e60ce3fd8c74c1a14f4a282258286b51ae",
                 "3622a2ddc41ec0e0fd4e68c13c6830f03b90c38d89aaad184de02c8c642cf807", 2_257,
                 "Розпізнавання: точне (medium)",
                 "Найточніша, але повільна на процесорі (кілька секунд на фразу)."),
    )
}

DEFAULT_TTS_MODEL = "uk_UA-ukrainian_tts-medium"
DEFAULT_STT_MODEL = "whisper-small"
_MARKER = "installed.json"


def models_root() -> Path:
    if os.environ.get("CHIBI_DATA_DIR"):
        root = data_dir() / "models"
    elif sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        root = Path(local) / "ChibiCompanion" / "models"
    else:
        root = data_dir() / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def model_dir(spec: ModelSpec, root: Path | None = None) -> Path:
    return (root or models_root()) / spec.id


def is_installed(spec: ModelSpec, root: Path | None = None) -> bool:
    """All files present with the right size and verified at install time."""
    directory = model_dir(spec, root)
    try:
        marker = json.loads((directory / _MARKER).read_text("utf-8"))
    except (OSError, ValueError):
        return False
    verified = marker.get("sha256", {}) if isinstance(marker, dict) else {}
    for f in spec.files:
        path = directory / f.name
        if verified.get(f.name) != f.sha256:
            return False
        try:
            if path.stat().st_size != f.size:
                return False
        except OSError:
            return False
    return True


def remove(spec: ModelSpec, root: Path | None = None) -> None:
    shutil.rmtree(model_dir(spec, root), ignore_errors=True)


class DownloadError(RuntimeError):
    def __init__(self, message: str, *, user_message: str = "") -> None:
        super().__init__(message)
        self.user_message = user_message or "Не вдалося завантажити модель."


class DownloadCancelled(DownloadError):
    def __init__(self) -> None:
        super().__init__("cancelled", user_message="Завантаження скасовано.")


#: (bytes done, bytes total) - called from the download thread.
ProgressCallback = Callable[[int, int], None]


def download(spec: ModelSpec, root: Path | None = None, *,
             on_progress: ProgressCallback | None = None,
             cancel: threading.Event | None = None,
             client: object | None = None,
             timeout: float = 30.0) -> Path:
    """Download and verify every file of *spec*; returns the model directory.

    Blocking - run it on a worker thread.  *client* is an ``httpx2.Client``
    (injected by tests); by default one with redirects enabled is created.
    """
    directory = model_dir(spec, root)
    directory.mkdir(parents=True, exist_ok=True)
    total = spec.size
    done = 0
    own_client = client is None
    if own_client:
        import httpx2

        client = httpx2.Client(follow_redirects=True, timeout=httpx2.Timeout(timeout, read=timeout))
    verified: dict[str, str] = {}
    try:
        for f in spec.files:
            target = directory / f.name
            if _file_ok(target, f):
                done += f.size
                verified[f.name] = f.sha256
                if on_progress:
                    on_progress(done, total)
                continue

            def progress(n: int, base: int = done) -> None:
                if on_progress:
                    on_progress(base + n, total)

            _download_file(client, f, target, progress, cancel)
            done += f.size
            verified[f.name] = f.sha256
        (directory / _MARKER).write_text(
            json.dumps({"id": spec.id, "sha256": verified}, indent=1), encoding="utf-8"
        )
    finally:
        if own_client:
            client.close()  # type: ignore[attr-defined]
    log.info("voice model %s installed", spec.id)
    return directory


def _file_ok(path: Path, f: ModelFile) -> bool:
    try:
        if path.stat().st_size != f.size:
            return False
    except OSError:
        return False
    return _sha256(path) == f.sha256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_file(client, f: ModelFile, target: Path, progress: Callable[[int], None],
                   cancel: threading.Event | None) -> None:
    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    have = 0
    if part.exists():
        have = part.stat().st_size
        if have > f.size:
            part.unlink()
            have = 0
        else:
            with part.open("rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(block)
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        with client.stream("GET", f.url, headers=headers) as response:
            if have and response.status_code == 200:
                # Server ignored the range: start over.
                have = 0
                digest = hashlib.sha256()
                part.unlink(missing_ok=True)
            elif response.status_code == 416 and have == f.size:
                pass  # already complete
            elif response.status_code not in (200, 206):
                raise DownloadError(f"HTTP {response.status_code} for {f.name}",
                                    user_message=f"Сервер відповів помилкою {response.status_code}.")
            if response.status_code != 416:
                with part.open("ab") as out:
                    progress(have)
                    for chunk in response.iter_bytes():
                        if cancel is not None and cancel.is_set():
                            raise DownloadCancelled()
                        out.write(chunk)
                        digest.update(chunk)
                        have += len(chunk)
                        if have > f.size:
                            raise DownloadError(f"{f.name} is larger than expected")
                        progress(have)
    except DownloadError:
        raise
    except Exception as exc:  # httpx2 transport errors, OSError
        raise DownloadError(f"download of {f.name} failed: {exc}",
                            user_message="Немає з'єднання або завантаження перервалося. "
                                         "Спробуй ще раз - продовжиться з того ж місця.") from exc
    if have != f.size or digest.hexdigest() != f.sha256:
        part.unlink(missing_ok=True)
        raise DownloadError(f"{f.name}: checksum mismatch",
                            user_message="Файл моделі пошкоджений або підмінений; його видалено.")
    os.replace(part, target)
