"""Thread-safe, bounded publish/subscribe bus.

* ``publish`` may be called from any thread.  It never blocks and never runs
  handlers itself; it only enqueues.
* ``dispatch_pending`` runs handlers and must be called on one thread - the Qt
  main thread in the app (see :class:`chibi.desktop.qt_bridge.QtEventPump`).
* The queue is bounded.  When it is full, the oldest queued event with a lower
  priority than the new one is dropped; if there is none, the new event is
  dropped.  Drops are counted, never silent in the logs.
* A failing handler is logged and does not affect other handlers.

The bus is deliberately free of Qt so the core logic is testable headless.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from typing import Any, TypeVar

from chibi.core.events import Event, Priority

log = logging.getLogger(__name__)

E = TypeVar("E", bound=Event)
Handler = Callable[[Any], None]


class EventBus:
    def __init__(self, max_queue: int = 256, on_pending: Callable[[], None] | None = None) -> None:
        if max_queue < 1:
            raise ValueError("max_queue must be >= 1")
        self._max_queue = max_queue
        self._queue: deque[Event] = deque()
        self._lock = threading.Lock()
        self._handlers: dict[type, list[Handler]] = {}
        self._on_pending = on_pending
        self._dropped = 0
        self._closed = False

    # -- configuration ----------------------------------------------------------

    def set_on_pending(self, callback: Callable[[], None] | None) -> None:
        """Callback invoked (from the publishing thread) when the queue goes non-empty."""
        self._on_pending = callback

    def subscribe(self, event_type: type[E], handler: Callable[[E], None]) -> Callable[[], None]:
        """Register *handler* for *event_type* and its subclasses.  Returns an unsubscribe function."""
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

        def unsubscribe() -> None:
            with self._lock:
                handlers = self._handlers.get(event_type, [])
                if handler in handlers:
                    handlers.remove(handler)

        return unsubscribe

    # -- producing ----------------------------------------------------------------

    def publish(self, event: Event) -> bool:
        """Enqueue *event*.  Thread-safe.  Returns False if the event was dropped."""
        notify = False
        with self._lock:
            if self._closed:
                return False
            if len(self._queue) >= self._max_queue:
                victim = self._find_victim(event)
                if victim is None:
                    self._dropped += 1
                    log.warning("event bus full, dropped %s (total dropped %d)",
                                type(event).__name__, self._dropped)
                    return False
                self._queue.remove(victim)
                self._dropped += 1
                log.warning("event bus full, dropped queued %s (total dropped %d)",
                            type(victim).__name__, self._dropped)
            was_empty = not self._queue
            self._queue.append(event)
            notify = was_empty
        if notify and self._on_pending is not None:
            try:
                self._on_pending()
            except Exception:  # pragma: no cover - callback is our own Qt bridge
                log.exception("on_pending callback failed")
        return True

    def _find_victim(self, incoming: Event) -> Event | None:
        """Oldest queued event whose priority is lower than the incoming one."""
        lowest: Event | None = None
        for queued in self._queue:
            if queued.priority < incoming.priority and (lowest is None or queued.priority < lowest.priority):
                lowest = queued
                if lowest.priority == Priority.LOW:
                    break
        return lowest

    # -- consuming ----------------------------------------------------------------

    def dispatch_pending(self, max_events: int = 200) -> int:
        """Run handlers for up to *max_events* queued events.  Returns how many were dispatched.

        If events remain afterwards (more than *max_events* were queued, or
        handlers published new ones), ``on_pending`` is invoked again so the
        owner schedules another round instead of starving its event loop.
        """
        dispatched = 0
        while dispatched < max_events:
            with self._lock:
                if not self._queue:
                    break
                event = self._queue.popleft()
                handlers = self._handlers_for(type(event))
            for handler in handlers:
                try:
                    handler(event)
                except Exception:
                    log.exception("handler %r failed for %s", handler, type(event).__name__)
            dispatched += 1
        with self._lock:
            remaining = bool(self._queue) and not self._closed
        if remaining and self._on_pending is not None:
            self._on_pending()
        return dispatched

    def _handlers_for(self, event_type: type) -> list[Handler]:
        result: list[Handler] = []
        for klass in event_type.__mro__:
            result.extend(self._handlers.get(klass, ()))
        return result

    # -- state ----------------------------------------------------------------------

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def close(self) -> None:
        """Stop accepting events and forget queued ones."""
        with self._lock:
            self._closed = True
            self._queue.clear()
