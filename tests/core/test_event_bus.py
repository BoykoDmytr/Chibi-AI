import threading

from chibi.core.event_bus import EventBus
from chibi.core.events import (
    ActivityCategory,
    AppInfo,
    AppLaunched,
    CharacterInteraction,
    Event,
    InteractionKind,
    UserIdle,
)


def launched(n: int = 0) -> AppLaunched:
    return AppLaunched(app=AppInfo(f"a{n}.exe", f"A{n}"), category=ActivityCategory.UNKNOWN)


def test_publish_does_not_dispatch_until_pumped():
    bus = EventBus()
    seen = []
    bus.subscribe(UserIdle, seen.append)
    bus.publish(UserIdle(idle_seconds=5))
    assert seen == []
    assert bus.dispatch_pending() == 1
    assert seen == [UserIdle(idle_seconds=5, timestamp=seen[0].timestamp)]


def test_handlers_receive_subclasses_of_subscribed_type():
    bus = EventBus()
    seen = []
    bus.subscribe(Event, seen.append)
    bus.publish(UserIdle(idle_seconds=1))
    bus.publish(launched())
    bus.dispatch_pending()
    assert [type(e) for e in seen] == [UserIdle, AppLaunched]


def test_unsubscribe():
    bus = EventBus()
    seen = []
    unsubscribe = bus.subscribe(UserIdle, seen.append)
    unsubscribe()
    bus.publish(UserIdle(idle_seconds=1))
    bus.dispatch_pending()
    assert seen == []


def test_failing_handler_does_not_break_others():
    bus = EventBus()
    seen = []

    def boom(_):
        raise RuntimeError("boom")

    bus.subscribe(UserIdle, boom)
    bus.subscribe(UserIdle, seen.append)
    bus.publish(UserIdle(idle_seconds=1))
    bus.dispatch_pending()
    assert len(seen) == 1


def test_queue_is_bounded_and_drops_low_priority_first():
    bus = EventBus(max_queue=3)
    for i in range(3):
        assert bus.publish(launched(i))  # LOW priority
    # NORMAL event evicts the oldest LOW one
    assert bus.publish(UserIdle(idle_seconds=1))
    assert bus.pending == 3
    assert bus.dropped == 1
    seen = []
    bus.subscribe(Event, seen.append)
    bus.dispatch_pending()
    assert [getattr(e, "app", None) and e.app.process_name for e in seen[:2]] == ["a1.exe", "a2.exe"]
    assert isinstance(seen[2], UserIdle)


def test_full_queue_rejects_event_without_lower_priority_victim():
    bus = EventBus(max_queue=2)
    bus.publish(CharacterInteraction(kind=InteractionKind.PET))  # HIGH
    bus.publish(CharacterInteraction(kind=InteractionKind.FEED))  # HIGH
    assert not bus.publish(UserIdle(idle_seconds=1))  # NORMAL can't evict HIGH
    assert bus.pending == 2
    assert bus.dropped == 1


def test_on_pending_fires_only_on_empty_to_nonempty_transition():
    calls = []
    bus = EventBus(on_pending=lambda: calls.append(1))
    bus.publish(UserIdle(idle_seconds=1))
    bus.publish(UserIdle(idle_seconds=2))
    assert len(calls) == 1
    bus.dispatch_pending()
    bus.publish(UserIdle(idle_seconds=3))
    assert len(calls) == 2


def test_dispatch_respects_max_and_reschedules():
    calls = []
    bus = EventBus(on_pending=lambda: calls.append(1))
    for i in range(5):
        bus.publish(UserIdle(idle_seconds=i))
    assert bus.dispatch_pending(max_events=2) == 2
    assert bus.pending == 3
    assert len(calls) == 2  # initial + "more remain"


def test_events_published_by_handlers_are_delivered():
    bus = EventBus()
    seen = []

    def chain(event):
        seen.append(event)
        if event.idle_seconds < 3:
            bus.publish(UserIdle(idle_seconds=event.idle_seconds + 1))

    bus.subscribe(UserIdle, chain)
    bus.publish(UserIdle(idle_seconds=1))
    bus.dispatch_pending()
    assert [e.idle_seconds for e in seen] == [1, 2, 3]


def test_publish_is_thread_safe():
    bus = EventBus(max_queue=10_000)

    def worker(n):
        for i in range(500):
            bus.publish(UserIdle(idle_seconds=n * 1000 + i))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert bus.pending == 4000
    assert bus.dispatch_pending(max_events=10_000) == 4000


def test_closed_bus_rejects_events():
    bus = EventBus()
    bus.publish(UserIdle(idle_seconds=1))
    bus.close()
    assert bus.pending == 0
    assert not bus.publish(UserIdle(idle_seconds=2))
