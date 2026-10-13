"""A single background thread running jobs by priority.

Used for slow work that must never block the Qt main thread (speech
recognition and synthesis, screen analysis).  Jobs report their results by
posting back to the main thread themselves; an exception escaping a job is
logged and the worker keeps running.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
from collections.abc import Callable

log = logging.getLogger(__name__)


class SerialWorker:
    """Lower priority numbers run first; FIFO within a priority.  The thread
    starts lazily on the first job and is a daemon (never blocks exit)."""

    def __init__(self, name: str = "chibi-worker") -> None:
        self._name = name
        self._queue: queue.PriorityQueue[tuple[int, int, Callable[[], None] | None]] = queue.PriorityQueue()
        self._counter = itertools.count()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def submit(self, job: Callable[[], None], priority: int = 1) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
                self._thread.start()
        self._queue.put((priority, next(self._counter), job))

    def shutdown(self, timeout: float = 3.0) -> None:
        thread = self._thread
        if thread is None:
            return
        self._queue.put((-1, next(self._counter), None))
        thread.join(timeout)
        self._thread = None

    def _run(self) -> None:
        while True:
            _priority, _n, job = self._queue.get()
            if job is None:
                return
            try:
                job()
            except Exception:  # jobs report their own errors; this is a last resort
                log.exception("%s: job failed", self._name)
