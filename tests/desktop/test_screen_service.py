from __future__ import annotations

from collections.abc import Iterator

import pytest

from chibi.core.config import Settings
from chibi.core.events import FullscreenChanged, SystemLoadChanged
from chibi.desktop.screen_service import ScreenService


class Probe:
    """Returns queued values in order (the last one repeats)."""

    def __init__(self, *values: object) -> None:
        self.values = list(values)
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        value = self.values[0] if len(self.values) == 1 else self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture
def events(bus) -> Iterator[list[object]]:
    seen: list[object] = []
    bus.subscribe(FullscreenChanged, seen.append)
    bus.subscribe(SystemLoadChanged, seen.append)
    yield seen


def make(qapp, bus, cpu=None, battery=None, fullscreen=None,
         settings: Settings | None = None) -> ScreenService:
    cfg = settings or Settings()
    return ScreenService(bus, lambda: cfg, lambda: {1}, fullscreen_probe=fullscreen or Probe(False),
                         cpu_probe=cpu or Probe(10.0), battery_probe=battery or Probe(None))


def load_events(events: list[object]) -> list[tuple[bool, bool]]:
    return [(e.high, e.on_battery_saver) for e in events if isinstance(e, SystemLoadChanged)]


def test_cpu_hysteresis(qapp, bus, events):
    cpu = Probe(0.0, 90.0, 50.0, 90.0, 91.0, 70.0, 61.0, 59.0, 59.0)
    service = make(qapp, bus, cpu=cpu)
    service.start()  # primes (the first value is meaningless)
    service.stop()
    assert cpu.calls == 1
    expected_high = [False, False, False, True, True, True, False, False]
    for want in expected_high:
        service.poll_load()
        assert service.high_load is want
    bus.dispatch_pending()
    assert load_events(events) == [(True, False), (False, False)]
    assert [e.cpu_percent for e in events] == [91.0, 59.0]


def test_single_spike_is_not_high(qapp, bus, events):
    service = make(qapp, bus, cpu=Probe(95.0, 40.0, 95.0, 40.0))
    for _ in range(4):
        service.poll_load()
    bus.dispatch_pending()
    assert events == []


def test_battery_saver(qapp, bus, events):
    battery = Probe((False, 15.0), (False, 21.0), (True, 10.0), (False, 20.0))
    service = make(qapp, bus, battery=battery)
    results = []
    for _ in range(4):
        service.poll_load()
        results.append(service.battery_saver)
    assert results == [True, False, False, True]
    bus.dispatch_pending()
    assert load_events(events) == [(False, True), (False, False), (False, True)]


def test_no_battery_and_failing_probes_are_harmless(qapp, bus, events):
    service = make(qapp, bus, cpu=Probe(RuntimeError("no cpu")), battery=Probe(OSError("no battery")),
                   fullscreen=Probe(RuntimeError("boom")))
    service.poll_load()
    assert service.poll_fullscreen() is False
    bus.dispatch_pending()
    assert events == []


def test_fullscreen_published_on_change_only(qapp, bus, events):
    probe = Probe(False, True, True, False, False)
    service = make(qapp, bus, fullscreen=probe)
    for _ in range(5):
        service.poll_fullscreen()
    bus.dispatch_pending()
    assert [e.active for e in events if isinstance(e, FullscreenChanged)] == [True, False]


def test_eco_disabled_stops_sampling_and_reports_normal(qapp, bus, events):
    cfg = Settings()
    cpu = Probe(95.0)
    service = make(qapp, bus, cpu=cpu, settings=cfg)
    service.poll_load()
    service.poll_load()
    assert service.high_load
    cfg.behavior.auto_eco_mode = False
    calls = cpu.calls
    service.poll_load()
    assert cpu.calls == calls
    assert not service.high_load
    bus.dispatch_pending()
    assert load_events(events) == [(True, False), (False, False)]


def test_timers_start_and_stop(qapp, bus):
    service = make(qapp, bus)
    service.start()
    assert service.running
    assert service._fullscreen_timer.isActive()  # an injected probe always runs
    assert service._fullscreen_timer.interval() == 2000
    assert service._load_timer.interval() == 5000
    service.stop()
    assert not service.running
    assert not service._fullscreen_timer.isActive()


def test_default_fullscreen_probe_off_windows(qapp, bus):
    import sys

    service = ScreenService(bus, Settings, lambda: set(), cpu_probe=Probe(1.0), battery_probe=Probe(None))
    if sys.platform != "win32":
        service.start()
        assert not service._fullscreen_timer.isActive()  # nothing to detect off Windows
        service.stop()
    assert service.poll_fullscreen() in (True, False)
