from __future__ import annotations

import pytest

from chibi.memory.repositories import PermissionRepository
from chibi.privacy.permissions import PermissionManager
from tests.settings.fakes import FakeFacade


@pytest.fixture
def permissions(db, bus, clock):
    return PermissionManager(PermissionRepository(db), bus, clock)


@pytest.fixture
def facade(permissions, tmp_path):
    return FakeFacade(permissions, tmp_path / "data")


@pytest.fixture
def track(qapp):
    """Register top-level widgets so they are closed and deleted after the test."""
    from PySide6.QtCore import QCoreApplication, QEvent

    created = []

    def _track(widget):
        created.append(widget)
        return widget

    yield _track
    for widget in created:
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
