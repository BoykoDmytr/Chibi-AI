from __future__ import annotations

import ctypes
import json
import logging
import os
import secrets as pysecrets
import sys

import pytest

from chibi.ai import secrets as secrets_mod
from chibi.ai.secrets import (
    CRED_PERSIST_LOCAL_MACHINE,
    CRED_TYPE_GENERIC,
    CREDENTIALW,
    ERROR_NOT_FOUND,
    KEY_NAMES,
    FileSecretStore,
    MemorySecretStore,
    SecretStore,
    SecretStoreError,
    WindowsCredentialStore,
    default_secret_store,
)

KEY = "sk-ant-api03-very-secret-value"


# -- memory store ----------------------------------------------------------------------------


def test_memory_store_roundtrip_and_delete():
    store = MemorySecretStore()
    assert isinstance(store, SecretStore)
    assert store.secure is False
    assert store.get("anthropic_api_key") is None
    store.set("anthropic_api_key", f"  {KEY}\n")
    assert store.get("anthropic_api_key") == KEY
    store.delete("anthropic_api_key")
    assert store.get("anthropic_api_key") is None
    store.delete("anthropic_api_key")  # deleting a missing secret is fine


def test_memory_store_empty_value_deletes():
    store = MemorySecretStore({"openai_api_key": KEY})
    store.set("openai_api_key", "   ")
    assert store.get("openai_api_key") is None


@pytest.mark.parametrize("bad", ["", "../x", "Anthropic", "a b", "x" * 65, "ключ"])
def test_invalid_names_are_rejected(bad):
    store = MemorySecretStore()
    with pytest.raises(ValueError):
        store.set(bad, KEY)
    with pytest.raises(ValueError):
        store.get(bad)


def test_key_names_contract():
    assert KEY_NAMES == {"anthropic": "anthropic_api_key", "openai": "openai_api_key"}


def test_repr_never_shows_values(tmp_path):
    store = MemorySecretStore({"anthropic_api_key": KEY})
    assert KEY not in repr(store)
    file_store = FileSecretStore(tmp_path)
    file_store.set("anthropic_api_key", KEY)
    assert KEY not in repr(file_store)


# -- file store ------------------------------------------------------------------------------


def test_file_store_roundtrip(tmp_path):
    store = FileSecretStore(tmp_path)
    assert store.secure is False
    assert store.get("anthropic_api_key") is None
    store.set("anthropic_api_key", KEY)
    store.set("openai_api_key", "sk-openai")
    assert FileSecretStore(tmp_path).get("anthropic_api_key") == KEY
    store.delete("anthropic_api_key")
    assert store.get("anthropic_api_key") is None
    assert store.get("openai_api_key") == "sk-openai"
    stored = json.loads((tmp_path / "secrets.json").read_text(encoding="utf-8"))
    assert stored == {"openai_api_key": "sk-openai"}
    # no temp files left behind
    assert sorted(p.name for p in tmp_path.iterdir()) == ["secrets.json"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_file_store_is_owner_only(tmp_path):
    store = FileSecretStore(tmp_path)
    store.set("anthropic_api_key", KEY)
    assert (tmp_path / "secrets.json").stat().st_mode & 0o777 == 0o600
    # Even if someone loosened it, the next write tightens it again.
    os.chmod(tmp_path / "secrets.json", 0o644)
    store.set("openai_api_key", "x")
    assert (tmp_path / "secrets.json").stat().st_mode & 0o777 == 0o600


def test_file_store_corrupt_file_is_treated_as_empty_without_leaking(tmp_path, caplog):
    (tmp_path / "secrets.json").write_text('{"anthropic_api_key": "' + KEY, encoding="utf-8")
    store = FileSecretStore(tmp_path)
    with caplog.at_level(logging.DEBUG):
        assert store.get("anthropic_api_key") is None
        store.set("openai_api_key", "fresh")
    assert store.get("openai_api_key") == "fresh"
    assert KEY not in caplog.text


def test_file_store_creates_missing_directory(tmp_path):
    store = FileSecretStore(tmp_path / "nested" / "dir")
    store.set("anthropic_api_key", KEY)
    assert store.get("anthropic_api_key") == KEY


def test_file_store_values_never_logged(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG):
        store = FileSecretStore(tmp_path)
        store.set("anthropic_api_key", KEY)
        store.get("anthropic_api_key")
        store.delete("anthropic_api_key")
    assert KEY not in caplog.text


# -- Windows credential store: marshalling against a fake advapi32 ----------------------------


class FakeAdvapi:
    """Behaves like advapi32's Cred* functions, operating on the real ctypes structs."""

    def __init__(self) -> None:
        self.vault: dict[str, tuple[int, int, bytes, str | None]] = {}
        self.last_error = 0
        self.freed = 0
        self.fail_with: int | None = None
        self._keep: list[object] = []

    def get_last_error(self) -> int:
        return self.last_error

    def CredWriteW(self, pcred, flags):
        assert flags == 0
        if self.fail_with is not None:
            self.last_error = self.fail_with
            return 0
        cred = pcred._obj
        assert isinstance(cred, CREDENTIALW)
        blob = ctypes.string_at(cred.CredentialBlob, cred.CredentialBlobSize)
        self.vault[cred.TargetName] = (cred.Type, cred.Persist, blob, cred.UserName)
        return 1

    def CredReadW(self, target, cred_type, flags, ppcred):
        assert cred_type == CRED_TYPE_GENERIC and flags == 0
        if self.fail_with is not None:
            self.last_error = self.fail_with
            return 0
        if target not in self.vault:
            self.last_error = ERROR_NOT_FOUND
            return 0
        _type, _persist, blob, _user = self.vault[target]
        buffer = (ctypes.c_ubyte * max(1, len(blob))).from_buffer_copy(blob or b"\0")
        cred = CREDENTIALW()
        cred.Type = CRED_TYPE_GENERIC
        cred.CredentialBlobSize = len(blob)
        cred.CredentialBlob = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
        self._keep.extend([buffer, cred])
        ppcred._obj.contents = cred
        return 1

    def CredFree(self, pcred):
        self.freed += 1

    def CredDeleteW(self, target, cred_type, flags):
        assert cred_type == CRED_TYPE_GENERIC and flags == 0
        if target not in self.vault:
            self.last_error = ERROR_NOT_FOUND
            return 0
        del self.vault[target]
        return 1


def test_windows_store_marshals_utf16_blob_and_target():
    api = FakeAdvapi()
    store = WindowsCredentialStore(api)
    assert store.secure is True
    store.set("anthropic_api_key", KEY)
    cred_type, persist, blob, user = api.vault["ChibiCompanion/anthropic_api_key"]
    assert cred_type == CRED_TYPE_GENERIC
    assert persist == CRED_PERSIST_LOCAL_MACHINE
    assert blob == KEY.encode("utf-16-le")
    assert user == "ChibiCompanion"
    assert store.get("anthropic_api_key") == KEY
    assert api.freed == 1  # every successful read is freed


def test_windows_store_missing_is_none_and_delete_is_idempotent():
    api = FakeAdvapi()
    store = WindowsCredentialStore(api)
    assert store.get("openai_api_key") is None
    assert api.freed == 0
    store.delete("openai_api_key")
    store.set("openai_api_key", "sk-1")
    store.delete("openai_api_key")
    assert api.vault == {}


def test_windows_store_unicode_and_empty_value():
    api = FakeAdvapi()
    store = WindowsCredentialStore(api)
    store.set("openai_api_key", "ключ-🔑")
    assert store.get("openai_api_key") == "ключ-🔑"
    store.set("openai_api_key", "")
    assert "ChibiCompanion/openai_api_key" not in api.vault


def test_windows_store_errors_carry_code_not_value():
    api = FakeAdvapi()
    store = WindowsCredentialStore(api)
    api.fail_with = 5  # ERROR_ACCESS_DENIED
    with pytest.raises(SecretStoreError) as err:
        store.set("anthropic_api_key", KEY)
    assert "5" in str(err.value) and KEY not in str(err.value)
    with pytest.raises(SecretStoreError):
        store.get("anthropic_api_key")


def test_windows_store_rejects_oversized_and_bad_names():
    store = WindowsCredentialStore(FakeAdvapi())
    with pytest.raises(ValueError):
        store.set("anthropic_api_key", "x" * 2000)
    with pytest.raises(ValueError):
        store.get("../other_app")


def test_windows_store_ignores_odd_sized_blob():
    api = FakeAdvapi()
    api.vault["ChibiCompanion/anthropic_api_key"] = (1, 2, b"abc", None)
    assert WindowsCredentialStore(api).get("anthropic_api_key") is None


def test_credentialw_layout_matches_wincred():
    names = [name for name, _ in CREDENTIALW._fields_]
    assert names == ["Flags", "Type", "TargetName", "Comment", "LastWritten", "CredentialBlobSize",
                     "CredentialBlob", "Persist", "AttributeCount", "Attributes", "TargetAlias", "UserName"]
    if sys.platform == "win32" and ctypes.sizeof(ctypes.c_void_p) == 8:
        assert ctypes.sizeof(CREDENTIALW) == 80


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows behaviour")
def test_windows_store_unavailable_off_windows(tmp_path):
    with pytest.raises(OSError):
        WindowsCredentialStore()
    store = default_secret_store(tmp_path)
    assert isinstance(store, FileSecretStore)
    assert store.secure is False


@pytest.mark.skipif(sys.platform != "win32", reason="needs the real Windows Credential Manager")
def test_real_windows_credential_manager_roundtrip(tmp_path):
    store = default_secret_store(tmp_path)
    assert isinstance(store, WindowsCredentialStore)
    name = f"test_{pysecrets.token_hex(6)}"
    value = f"sk-test-{pysecrets.token_hex(16)}-ключ"
    try:
        assert store.get(name) is None
        store.set(name, value)
        assert store.get(name) == value
        store.set(name, value + "2")  # overwrite
        assert store.get(name) == value + "2"
    finally:
        store.delete(name)
    assert store.get(name) is None
    store.delete(name)  # already gone: no error


def test_module_imports_cleanly_everywhere():
    assert secrets_mod.TARGET_PREFIX == "ChibiCompanion/"
