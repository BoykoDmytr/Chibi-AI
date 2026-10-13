"""Where API keys live.

* Windows: the user's Credential Manager (``advapi32`` ``CredWriteW`` /
  ``CredReadW``), target ``ChibiCompanion/<name>``.  Windows encrypts it with
  the user's logon credentials, it never lands in our data directory, and
  "delete all data" cannot leave a copy behind in the SQLite file.
* Elsewhere (development on Linux/macOS): a JSON file in the data directory
  with mode 0600.  ``secure`` is False so the UI can say so.
* Tests: :class:`MemorySecretStore`.

Values are never logged, never put into exception messages and never
returned by ``repr``.  Names are a small fixed vocabulary (see
:data:`KEY_NAMES`), validated so they cannot address another app's
credential.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import logging
import os
import re
import sys
import tempfile
import threading
from ctypes import wintypes
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

#: provider -> secret name of its API key.
KEY_NAMES: dict[str, str] = {"anthropic": "anthropic_api_key", "openai": "openai_api_key"}

#: Prefix of every credential we create in the Windows Credential Manager.
TARGET_PREFIX = "ChibiCompanion/"

_NAME_RE = re.compile(r"[a-z0-9_]{1,64}")


class SecretStoreError(RuntimeError):
    """The backing store failed (OS error code or unreadable file); never carries the value."""


@runtime_checkable
class SecretStore(Protocol):
    #: True when backed by an OS credential store (encrypted per user).
    secure: bool

    def get(self, name: str) -> str | None: ...

    def set(self, name: str, value: str) -> None:
        """Store *value* (an empty value deletes the secret).  Never logged."""
        ...

    def delete(self, name: str) -> None: ...


def _check_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError("invalid secret name")
    return name


def _clean_value(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("secret value must be a string")
    # Keys pasted from a web page often carry a trailing newline or spaces.
    return value.strip()


# -- in-memory (tests) --------------------------------------------------------------------


class MemorySecretStore:
    """Process-local store for tests and the headless smoke test."""

    secure = False

    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, str] = {}
        for name, value in (initial or {}).items():
            self.set(name, value)

    def get(self, name: str) -> str | None:
        _check_name(name)
        with self._lock:
            return self._values.get(name)

    def set(self, name: str, value: str) -> None:
        _check_name(name)
        value = _clean_value(value)
        with self._lock:
            if value:
                self._values[name] = value
            else:
                self._values.pop(name, None)

    def delete(self, name: str) -> None:
        _check_name(name)
        with self._lock:
            self._values.pop(name, None)

    def __repr__(self) -> str:  # never show values
        return f"MemorySecretStore({len(self._values)} secrets)"


# -- file (non-Windows development fallback) ------------------------------------------------


class FileSecretStore:
    """JSON file readable only by the owner (0600 on POSIX).

    Not encrypted - a dev fallback for platforms without a credential store we
    support; the settings UI shows ``secure=False`` as a warning.
    """

    secure = False
    FILE_NAME = "secrets.json"

    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir) / self.FILE_NAME
        self._lock = threading.Lock()

    def get(self, name: str) -> str | None:
        _check_name(name)
        with self._lock:
            value = self._load().get(name)
        return value if isinstance(value, str) and value else None

    def set(self, name: str, value: str) -> None:
        _check_name(name)
        value = _clean_value(value)
        with self._lock:
            data = self._load()
            if value:
                data[name] = value
            else:
                data.pop(name, None)
            self._save(data)

    def delete(self, name: str) -> None:
        _check_name(name)
        with self._lock:
            data = self._load()
            if name in data:
                del data[name]
                self._save(data)

    def _load(self) -> dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise SecretStoreError(f"cannot read secret file ({type(exc).__name__})") from None
        try:
            data = json.loads(raw)
        except ValueError:
            # Never echo the content: it holds keys.
            log.warning("secret file is corrupt; treating it as empty")
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write a private temp file and atomically replace: a crash mid-write
        # can't leave a truncated file, and the content is never world-readable.
        fd, tmp = tempfile.mkstemp(prefix=".secrets-", suffix=".tmp", dir=self.path.parent)
        try:
            if os.name == "posix":
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        if os.name == "posix":
            os.chmod(self.path, 0o600)

    def __repr__(self) -> str:
        return f"FileSecretStore({str(self.path)!r})"


# -- Windows Credential Manager ---------------------------------------------------------------

CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2
ERROR_NOT_FOUND = 1168
#: CRED_MAX_CREDENTIAL_BLOB_SIZE (5 * 512 bytes).
CRED_MAX_BLOB_BYTES = 5 * 512


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


class CREDENTIAL_ATTRIBUTEW(ctypes.Structure):
    _fields_ = [
        ("Keyword", wintypes.LPWSTR),
        ("Flags", wintypes.DWORD),
        ("ValueSize", wintypes.DWORD),
        ("Value", ctypes.POINTER(ctypes.c_ubyte)),
    ]


class CREDENTIALW(ctypes.Structure):
    """``wincred.h`` CREDENTIALW; field order and types must match exactly."""

    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.POINTER(CREDENTIAL_ATTRIBUTEW)),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


PCREDENTIALW = ctypes.POINTER(CREDENTIALW)


class _Advapi32:
    """The four advapi32 functions with explicit signatures.

    Loaded through a private ``WinDLL`` with ``use_last_error=True`` so
    ``GetLastError`` is captured right after each call (ctypes may make other
    API calls in between otherwise).
    """

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("Windows Credential Manager is only available on Windows")
        dll = ctypes.WinDLL("advapi32", use_last_error=True)  # type: ignore[attr-defined]
        self.CredWriteW = dll.CredWriteW
        self.CredWriteW.argtypes = [PCREDENTIALW, wintypes.DWORD]
        self.CredWriteW.restype = wintypes.BOOL
        self.CredReadW = dll.CredReadW
        self.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                   ctypes.POINTER(PCREDENTIALW)]
        self.CredReadW.restype = wintypes.BOOL
        self.CredDeleteW = dll.CredDeleteW
        self.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        self.CredDeleteW.restype = wintypes.BOOL
        self.CredFree = dll.CredFree
        self.CredFree.argtypes = [ctypes.c_void_p]
        self.CredFree.restype = None

    @staticmethod
    def get_last_error() -> int:
        return ctypes.get_last_error()  # type: ignore[attr-defined]


class WindowsCredentialStore:
    """Generic credentials in the current user's Windows Credential Manager.

    *api* is injectable so the marshalling logic is testable off Windows with a
    fake exposing the same four functions and ``get_last_error``.
    """

    secure = True

    def __init__(self, api: Any = None) -> None:
        self._api = api if api is not None else _Advapi32()
        self._lock = threading.Lock()

    @staticmethod
    def target(name: str) -> str:
        return TARGET_PREFIX + _check_name(name)

    def get(self, name: str) -> str | None:
        target = self.target(name)
        pcred = PCREDENTIALW()
        with self._lock:
            ok = self._api.CredReadW(target, CRED_TYPE_GENERIC, 0, ctypes.byref(pcred))
            if not ok:
                error = self._api.get_last_error()
                if error == ERROR_NOT_FOUND:
                    return None
                raise SecretStoreError(f"CredReadW failed (error {error})")
            try:
                cred = pcred.contents
                size = int(cred.CredentialBlobSize)
                blob = ctypes.string_at(cred.CredentialBlob, size) if size and cred.CredentialBlob else b""
            finally:
                self._api.CredFree(pcred)
        if len(blob) % 2:
            log.warning("credential %s has an odd-sized blob; ignoring it", name)
            return None
        try:
            value = blob.decode("utf-16-le")
        except UnicodeDecodeError:
            log.warning("credential %s is not UTF-16; ignoring it", name)
            return None
        return value or None

    def set(self, name: str, value: str) -> None:
        target = self.target(name)
        value = _clean_value(value)
        if not value:
            self.delete(name)
            return
        blob = value.encode("utf-16-le")
        if len(blob) > CRED_MAX_BLOB_BYTES:
            raise ValueError("secret is too long for the Windows Credential Manager")
        buffer = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
        cred = CREDENTIALW()
        cred.Flags = 0
        cred.Type = CRED_TYPE_GENERIC
        cred.TargetName = target
        cred.Comment = None
        cred.CredentialBlobSize = len(blob)
        cred.CredentialBlob = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
        cred.Persist = CRED_PERSIST_LOCAL_MACHINE
        cred.AttributeCount = 0
        cred.Attributes = None
        cred.TargetAlias = None
        cred.UserName = "ChibiCompanion"
        try:
            with self._lock:
                ok = self._api.CredWriteW(ctypes.byref(cred), 0)
                error = 0 if ok else self._api.get_last_error()
        finally:
            # Don't leave a plaintext copy in our own heap longer than needed.
            ctypes.memset(buffer, 0, len(blob))
        if not ok:
            raise SecretStoreError(f"CredWriteW failed (error {error})")

    def delete(self, name: str) -> None:
        target = self.target(name)
        with self._lock:
            ok = self._api.CredDeleteW(target, CRED_TYPE_GENERIC, 0)
            if ok:
                return
            error = self._api.get_last_error()
        if error != ERROR_NOT_FOUND:
            raise SecretStoreError(f"CredDeleteW failed (error {error})")

    def __repr__(self) -> str:
        return "WindowsCredentialStore()"


def default_secret_store(data_dir: Path) -> SecretStore:
    """Windows Credential Manager on Windows, the 0600 file store elsewhere."""
    if sys.platform == "win32":
        try:
            return WindowsCredentialStore()
        except OSError:
            log.exception("Windows Credential Manager unavailable; falling back to a file")
    return FileSecretStore(data_dir)
