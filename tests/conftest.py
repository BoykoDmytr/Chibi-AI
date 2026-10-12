import os

# Must be set before any Qt import: tests run headless (CI, containers).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from chibi.core.clock import FakeClock
from chibi.core.event_bus import EventBus
from chibi.memory.database import Database


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Never touch the real user data directory from tests."""
    monkeypatch.setenv("CHIBI_DATA_DIR", str(tmp_path / "chibi-data"))


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def db():
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def bus():
    return EventBus()
