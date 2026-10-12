"""Delivers :class:`~chibi.core.event_bus.EventBus` events on the Qt main thread.

Worker threads call ``bus.publish``; the bus invokes ``on_pending`` from the
publishing thread; we turn that into one queued Qt signal (coalesced, so a
burst of events costs one wake-up) and dispatch on the main thread.
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, Signal, Slot

from chibi.core.event_bus import EventBus


class QtEventPump(QObject):
    _wake = Signal()

    def __init__(self, bus: EventBus, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._bus = bus
        self._scheduled = threading.Event()
        self._wake.connect(self._drain, Qt.ConnectionType.QueuedConnection)
        bus.set_on_pending(self._request)
        if bus.pending:
            self._request()

    def _request(self) -> None:
        """Called from any thread."""
        if not self._scheduled.is_set():
            self._scheduled.set()
            self._wake.emit()

    @Slot()
    def _drain(self) -> None:
        self._scheduled.clear()
        self._bus.dispatch_pending()

    def detach(self) -> None:
        self._bus.set_on_pending(None)
