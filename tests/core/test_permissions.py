import threading

import pytest

import chibi.privacy.permissions as permissions_module

from chibi.core.event_bus import EventBus
from chibi.core.events import ObservationStopped, Permission, PermissionChanged
from chibi.memory.repositories import PermissionRepository
from chibi.privacy.permissions import PermissionManager, PermissionUnavailableError


@pytest.fixture
def manager(db, clock):
    return PermissionManager(PermissionRepository(db), EventBus(), clock)


def test_everything_is_off_on_first_run(manager):
    assert not any(manager.snapshot().values())
    assert not manager.any_observation_active()


def test_grant_persists_and_announces(db, clock):
    bus = EventBus()
    seen = []
    bus.subscribe(PermissionChanged, seen.append)
    m = PermissionManager(PermissionRepository(db), bus, clock)
    m.grant(Permission.APP_METADATA)
    bus.dispatch_pending()
    assert m.is_granted(Permission.APP_METADATA)
    assert [(e.permission, e.granted) for e in seen] == [(Permission.APP_METADATA, True)]
    # a new manager over the same db sees the grant
    assert PermissionManager(PermissionRepository(db)).is_granted(Permission.APP_METADATA)


def test_every_feature_is_available_since_phase_4():
    assert set(Permission) <= permissions_module.AVAILABLE


def test_unavailable_features_cannot_be_granted(manager, monkeypatch):
    # A build without a feature (e.g. a stripped one) must refuse its permission.
    monkeypatch.setattr(permissions_module, "AVAILABLE", frozenset({Permission.APP_METADATA}))
    for p in (Permission.SCREEN_CAPTURE, Permission.CLOUD_VISION, Permission.MICROPHONE):
        with pytest.raises(PermissionUnavailableError):
            manager.grant(p)
        assert not manager.is_granted(p)


def test_stored_grant_for_unavailable_feature_is_ignored(db, monkeypatch):
    monkeypatch.setattr(permissions_module, "AVAILABLE", frozenset({Permission.APP_METADATA}))
    PermissionRepository(db).save(Permission.MICROPHONE, True, 1.0)
    assert not PermissionManager(PermissionRepository(db)).is_granted(Permission.MICROPHONE)


def test_dependent_permission_requires_base(manager):
    with pytest.raises(PermissionError):
        manager.grant(Permission.WINDOW_TITLES)
    manager.grant(Permission.APP_METADATA)
    manager.grant(Permission.WINDOW_TITLES)
    assert manager.is_granted(Permission.WINDOW_TITLES)


def test_revoking_base_revokes_dependents(manager):
    manager.grant(Permission.APP_METADATA)
    manager.grant(Permission.WINDOW_TITLES)
    manager.revoke(Permission.APP_METADATA)
    assert not manager.is_granted(Permission.WINDOW_TITLES)
    assert not manager.is_granted(Permission.APP_METADATA)


def test_kill_switch_revokes_everything_and_announces(db, clock):
    bus = EventBus()
    stopped = []
    bus.subscribe(ObservationStopped, stopped.append)
    m = PermissionManager(PermissionRepository(db), bus, clock)
    m.grant(Permission.APP_METADATA)
    m.grant(Permission.WINDOW_TITLES)
    m.stop_all_observation()
    bus.dispatch_pending()
    assert not m.any_observation_active()
    assert len(stopped) == 1
    assert not PermissionManager(PermissionRepository(db)).any_observation_active()


def test_history_is_auditable(manager):
    manager.grant(Permission.APP_METADATA)
    manager.revoke(Permission.APP_METADATA)
    assert [r.granted for r in manager.history()] == [False, True]


def test_is_granted_is_safe_from_worker_threads(manager):
    manager.grant(Permission.APP_METADATA)
    results = []
    t = threading.Thread(target=lambda: results.append(manager.is_granted(Permission.APP_METADATA)))
    t.start()
    t.join()
    assert results == [True]


def test_revocation_takes_effect_even_if_persisting_fails(db, clock, monkeypatch):
    m = PermissionManager(PermissionRepository(db), None, clock)
    m.grant(Permission.APP_METADATA)

    def broken_save(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(m._repo, "save", broken_save)
    m.revoke(Permission.APP_METADATA)
    assert not m.is_granted(Permission.APP_METADATA)


def test_stored_dependent_without_base_is_not_resurrected(db):
    PermissionRepository(db).save(Permission.WINDOW_TITLES, True, 1.0)
    PermissionRepository(db).save(Permission.APP_METADATA, False, 2.0)
    m = PermissionManager(PermissionRepository(db))
    assert not m.is_granted(Permission.WINDOW_TITLES)


def test_cloud_ai_is_available_from_phase_2_and_off_by_default(manager):
    assert not manager.is_granted(Permission.CLOUD_AI)
    manager.grant(Permission.CLOUD_AI)
    assert manager.is_granted(Permission.CLOUD_AI)
