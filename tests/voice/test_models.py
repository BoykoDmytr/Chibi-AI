import hashlib
import threading

import httpx2
import pytest

from chibi.voice import models
from chibi.voice.models import (
    CATALOG,
    DownloadCancelled,
    DownloadError,
    ModelFile,
    ModelSpec,
    download,
    is_installed,
    model_dir,
    remove,
)

ALPHA = b"alpha-model-bytes" * 1000
BETA = b'{"config": true}'


def spec_for(files: dict[str, bytes], bad_sha: str | None = None) -> ModelSpec:
    return ModelSpec(
        id="test-model",
        kind="tts",
        title="t",
        description="d",
        license="MIT",
        files=tuple(
            ModelFile(name, f"https://models.test/{name}", len(data),
                      bad_sha if bad_sha and name == "alpha.bin" else hashlib.sha256(data).hexdigest())
            for name, data in files.items()
        ),
    )


def server(files: dict[str, bytes], log: list | None = None, support_range=True, fail_after=None):
    def handler(request: httpx2.Request) -> httpx2.Response:
        name = request.url.path.lstrip("/")
        if log is not None:
            log.append((name, request.headers.get("range")))
        data = files.get(name)
        if data is None:
            return httpx2.Response(404)
        rng = request.headers.get("range")
        if rng and support_range:
            start = int(rng.split("=")[1].rstrip("-"))
            if start >= len(data):
                return httpx2.Response(416)
            return httpx2.Response(206, content=data[start:])
        if fail_after is not None:
            return httpx2.Response(200, content=data[:fail_after], headers={"content-length": str(len(data))})
        return httpx2.Response(200, content=data)

    return httpx2.Client(transport=httpx2.MockTransport(handler))


FILES = {"alpha.bin": ALPHA, "beta.json": BETA}


def test_download_verifies_and_marks_installed(tmp_path):
    spec = spec_for(FILES)
    progress = []
    path = download(spec, tmp_path, client=server(FILES), on_progress=lambda d, t: progress.append((d, t)))
    assert path == model_dir(spec, tmp_path)
    assert (path / "alpha.bin").read_bytes() == ALPHA
    assert is_installed(spec, tmp_path)
    assert progress[-1] == (spec.size, spec.size)
    assert all(d <= t for d, t in progress)
    assert not list(path.glob("*.part"))


def test_checksum_mismatch_deletes_the_file(tmp_path):
    spec = spec_for(FILES, bad_sha="0" * 64)
    with pytest.raises(DownloadError) as info:
        download(spec, tmp_path, client=server(FILES))
    assert "пошкоджений" in info.value.user_message
    directory = model_dir(spec, tmp_path)
    assert not (directory / "alpha.bin").exists()
    assert not (directory / "alpha.bin.part").exists()
    assert not is_installed(spec, tmp_path)


def test_http_error_is_reported(tmp_path):
    spec = spec_for(FILES)
    with pytest.raises(DownloadError):
        download(spec, tmp_path, client=server({"beta.json": BETA}))


def test_resumes_a_partial_download_with_range(tmp_path):
    spec = spec_for(FILES)
    directory = model_dir(spec, tmp_path)
    directory.mkdir(parents=True)
    (directory / "alpha.bin.part").write_bytes(ALPHA[:5000])
    log = []
    download(spec, tmp_path, client=server(FILES, log))
    assert ("alpha.bin", "bytes=5000-") in log
    assert (directory / "alpha.bin").read_bytes() == ALPHA


def test_restarts_when_server_ignores_range(tmp_path):
    spec = spec_for(FILES)
    directory = model_dir(spec, tmp_path)
    directory.mkdir(parents=True)
    (directory / "alpha.bin.part").write_bytes(b"garbage" * 10)
    download(spec, tmp_path, client=server(FILES, support_range=False))
    assert (directory / "alpha.bin").read_bytes() == ALPHA


def test_complete_part_file_gets_416_and_is_accepted(tmp_path):
    spec = spec_for(FILES)
    directory = model_dir(spec, tmp_path)
    directory.mkdir(parents=True)
    (directory / "alpha.bin.part").write_bytes(ALPHA)
    download(spec, tmp_path, client=server(FILES))
    assert is_installed(spec, tmp_path)


def test_already_present_files_are_not_downloaded_again(tmp_path):
    spec = spec_for(FILES)
    download(spec, tmp_path, client=server(FILES))
    log = []
    download(spec, tmp_path, client=server(FILES, log))
    assert log == []


def test_cancel_stops_and_keeps_partial(tmp_path):
    spec = spec_for(FILES)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(DownloadCancelled):
        download(spec, tmp_path, client=server(FILES), cancel=cancel)
    assert not is_installed(spec, tmp_path)


def test_tampered_file_after_install_is_not_installed(tmp_path):
    spec = spec_for(FILES)
    download(spec, tmp_path, client=server(FILES))
    (model_dir(spec, tmp_path) / "alpha.bin").write_bytes(b"x")
    assert not is_installed(spec, tmp_path)


def test_remove(tmp_path):
    spec = spec_for(FILES)
    download(spec, tmp_path, client=server(FILES))
    remove(spec, tmp_path)
    assert not model_dir(spec, tmp_path).exists()
    remove(spec, tmp_path)  # idempotent


def test_catalog_is_pinned_and_complete():
    assert models.DEFAULT_TTS_MODEL in CATALOG
    assert models.DEFAULT_STT_MODEL in CATALOG
    for spec in CATALOG.values():
        assert spec.kind in ("tts", "stt")
        assert spec.title and spec.license
        for f in spec.files:
            assert f.url.startswith("https://huggingface.co/")
            assert "/resolve/main/" not in f.url, "every file must be pinned to a revision"
            assert len(f.sha256) == 64 and f.size > 0
        if spec.kind == "stt":
            assert {f.name for f in spec.files} >= {"model.bin", "config.json", "tokenizer.json"}


def test_models_root_honours_data_dir_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CHIBI_DATA_DIR", str(tmp_path / "d"))
    assert models.models_root() == tmp_path / "d" / "models"
