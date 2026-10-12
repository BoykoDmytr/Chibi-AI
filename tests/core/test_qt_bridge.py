import threading

from chibi.core.event_bus import EventBus
from chibi.core.events import UserIdle
from chibi.desktop.qt_bridge import QtEventPump


def test_events_from_worker_threads_are_dispatched_on_main_thread(qapp):
    from PySide6.QtCore import QCoreApplication, QElapsedTimer

    bus = EventBus()
    pump = QtEventPump(bus)
    main = threading.get_ident()
    seen = []
    bus.subscribe(UserIdle, lambda e: seen.append((e.idle_seconds, threading.get_ident())))

    t = threading.Thread(target=lambda: [bus.publish(UserIdle(idle_seconds=i)) for i in range(50)])
    t.start()
    t.join()

    timer = QElapsedTimer()
    timer.start()
    while len(seen) < 50 and timer.elapsed() < 3000:
        QCoreApplication.processEvents()
    pump.detach()
    assert len(seen) == 50
    assert {tid for _, tid in seen} == {main}
