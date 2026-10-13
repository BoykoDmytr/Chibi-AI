from __future__ import annotations

import pytest


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
        try:
            widget.close()
            widget.deleteLater()
        except RuntimeError:  # already deleted by the code under test
            pass
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
