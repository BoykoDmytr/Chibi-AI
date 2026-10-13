"""Permission manager - the single gate for every observation feature.

Privacy is enforced here rather than by UI toggles alone:

* every permission is OFF until the user grants it, and grants are persisted
  with an audit log the user can review;
* features that do not exist in this version (screen capture, cloud AI,
  microphone) *cannot* be granted - there is no code path that could use them;
* dependent permissions are revoked together with their base
  (window titles need app metadata, cloud vision needs screen capture);
* ``stop_all_observation`` (the kill switch) revokes every observation
  permission at once;
* ``is_granted`` is thread-safe so worker threads re-check it on every poll
  and discard results the moment a permission is revoked.
"""

from __future__ import annotations

import logging
import threading

from chibi.core.clock import Clock, SystemClock
from chibi.core.event_bus import EventBus
from chibi.core.events import (
    OBSERVATION_PERMISSIONS,
    ObservationStopped,
    Permission,
    PermissionChanged,
)
from chibi.memory.repositories import PermissionRecord, PermissionRepository

log = logging.getLogger(__name__)

#: Permissions whose feature exists in this build.
AVAILABLE: frozenset[Permission] = frozenset(
    {Permission.APP_METADATA, Permission.WINDOW_TITLES, Permission.CLOUD_AI}
)

#: permission -> permission it requires
REQUIRES: dict[Permission, Permission] = {
    Permission.WINDOW_TITLES: Permission.APP_METADATA,
    Permission.CLOUD_VISION: Permission.SCREEN_CAPTURE,
}


class PermissionUnavailableError(ValueError):
    """The feature behind this permission is not part of this version."""


class PermissionManager:
    def __init__(self, repo: PermissionRepository, bus: EventBus | None = None,
                 clock: Clock | None = None) -> None:
        self._repo = repo
        self._bus = bus
        self._clock = clock or SystemClock()
        self._lock = threading.Lock()
        stored = repo.load_all()
        # Never resurrect a grant for a feature this build doesn't have.
        self._granted = {p: bool(stored.get(p)) and p in AVAILABLE for p in Permission}
        # Nor a dependent grant whose base is off (e.g. a partially failed
        # revoke left window_titles=1, app_metadata=0 on disk).
        for dependent, base in REQUIRES.items():
            if self._granted[dependent] and not self._granted[base]:
                self._granted[dependent] = False

    def is_granted(self, permission: Permission) -> bool:
        """Thread-safe."""
        with self._lock:
            return self._granted[permission]

    def snapshot(self) -> dict[Permission, bool]:
        with self._lock:
            return dict(self._granted)

    def is_available(self, permission: Permission) -> bool:
        return permission in AVAILABLE

    def any_observation_active(self) -> bool:
        with self._lock:
            return any(self._granted[p] for p in OBSERVATION_PERMISSIONS)

    def grant(self, permission: Permission) -> None:
        if permission not in AVAILABLE:
            raise PermissionUnavailableError(f"{permission.value} is not available in this version")
        base = REQUIRES.get(permission)
        if base is not None and not self.is_granted(base):
            raise PermissionError(f"{permission.value} requires {base.value}")
        self._set(permission, True)

    def revoke(self, permission: Permission) -> None:
        # Revoke dependents first so nothing ever holds a grant without its base.
        for dependent, base in REQUIRES.items():
            if base is permission:
                self.revoke(dependent)
        self._set(permission, False)

    def set(self, permission: Permission, granted: bool) -> None:
        if granted:
            self.grant(permission)
        else:
            self.revoke(permission)

    def stop_all_observation(self) -> None:
        """Kill switch: revoke every observation permission immediately."""
        # Dependents first, so no dependent is ever granted without its base.
        for permission in sorted(OBSERVATION_PERMISSIONS, key=lambda p: p in REQUIRES, reverse=True):
            self._set(permission, False, announce=False)
        log.info("all observation stopped by the user")
        if self._bus is not None:
            self._bus.publish(ObservationStopped())
            for permission in OBSERVATION_PERMISSIONS:
                self._bus.publish(PermissionChanged(permission=permission, granted=False))

    def history(self, limit: int = 100) -> list[PermissionRecord]:
        return self._repo.history(limit)

    def _set(self, permission: Permission, granted: bool, announce: bool = True) -> None:
        with self._lock:
            changed = self._granted[permission] != granted
            self._granted[permission] = granted
        if not changed:
            return
        # Persist after the in-memory flip: revocation takes effect for worker
        # threads even if the disk write fails.
        try:
            self._repo.save(permission, granted, self._clock.now())
        except Exception:
            log.exception("could not persist permission %s", permission.value)
        log.info("permission %s %s", permission.value, "granted" if granted else "revoked")
        if announce and self._bus is not None:
            self._bus.publish(PermissionChanged(permission=permission, granted=granted))
