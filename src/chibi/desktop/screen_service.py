"""Environment polling on the main thread: fullscreen apps and system load.

Two cheap timers:

* every 2 s - does the foreground window cover its monitor?  Publishes
  :class:`FullscreenChanged` on change (she hides, reactions stay quiet).
* every 5 s - CPU load and battery saver.  Publishes
  :class:`SystemLoadChanged` on change (eco mode: lower frame rate, no walks).

Both checks are geometry / system-wide numbers only, so they need no
permission and carry nothing about the user.  CPU uses hysteresis - high
only after two consecutive samples above 85 %, normal again below 60 % -
so a compile spike does not flip her into eco mode and back every 5 s.

The probes are injectable, which keeps the service testable without
Windows or a busy CPU.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer

from chibi.core.config import Settings
from chibi.core.event_bus import EventBus
from chibi.core.events import FullscreenChanged, SystemLoadChanged
from chibi.desktop import platform_win

log = logging.getLogger(__name__)

FULLSCREEN_INTERVAL_MS = 2000
LOAD_INTERVAL_MS = 5000
#: CPU above this on HIGH_SAMPLES consecutive samples means high load...
HIGH_CPU_PERCENT = 85.0
HIGH_SAMPLES = 2
#: ...and below this it is normal again.
NORMAL_CPU_PERCENT = 60.0
#: On battery at or below this charge counts as battery saver.
BATTERY_SAVER_PERCENT = 20.0

#: (plugged_in, percent) or None when there is no battery.
BatteryReading = tuple[bool, float] | None


def _psutil_cpu() -> float:
    import psutil

    return float(psutil.cpu_percent(interval=None))


def _psutil_battery() -> BatteryReading:
    import psutil

    sensors = getattr(psutil, "sensors_battery", None)
    if sensors is None:
        return None
    battery = sensors()
    if battery is None or battery.power_plugged is None:
        return None
    return bool(battery.power_plugged), float(battery.percent)


class ScreenService(QObject):
    def __init__(
        self,
        bus: EventBus,
        settings: Callable[[], Settings],
        own_window_ids: Callable[[], set[int]],
        *,
        fullscreen_probe: Callable[[], bool] | None = None,
        cpu_probe: Callable[[], float] | None = None,
        battery_probe: Callable[[], BatteryReading] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._bus = bus
        self._settings = settings
        self._own_window_ids = own_window_ids
        self._fullscreen_probe = fullscreen_probe or self._default_fullscreen_probe
        # Off Windows there is nothing to detect; don't run a timer for it.
        self._fullscreen_enabled = fullscreen_probe is not None or platform_win.supported()
        self._cpu_probe = cpu_probe or _psutil_cpu
        self._battery_probe = battery_probe or _psutil_battery
        self._fullscreen = False
        self._high = False
        self._high_streak = 0
        self._battery_saver = False
        self._last_cpu = 0.0
        self._warned: set[str] = set()

        self._fullscreen_timer = QTimer(self)
        self._fullscreen_timer.setInterval(FULLSCREEN_INTERVAL_MS)
        self._fullscreen_timer.timeout.connect(self.poll_fullscreen)
        self._load_timer = QTimer(self)
        self._load_timer.setInterval(LOAD_INTERVAL_MS)
        self._load_timer.timeout.connect(self.poll_load)

    # -- state ----------------------------------------------------------------------

    @property
    def fullscreen(self) -> bool:
        return self._fullscreen

    @property
    def high_load(self) -> bool:
        return self._high

    @property
    def battery_saver(self) -> bool:
        return self._battery_saver

    @property
    def running(self) -> bool:
        return self._load_timer.isActive()

    # -- lifecycle ------------------------------------------------------------------

    def start(self) -> None:
        # The first cpu_percent(interval=None) compares against process start
        # and is meaningless; prime it so the first real sample covers 5 s.
        self._sample_cpu()
        if self._fullscreen_enabled:
            self._fullscreen_timer.start()
        self._load_timer.start()

    def stop(self) -> None:
        self._fullscreen_timer.stop()
        self._load_timer.stop()

    # -- polling --------------------------------------------------------------------

    def poll_fullscreen(self) -> bool:
        try:
            active = bool(self._fullscreen_probe())
        except Exception as exc:
            self._warn_once("fullscreen", "fullscreen probe failed: %s", exc)
            active = False
        if active != self._fullscreen:
            self._fullscreen = active
            log.debug("fullscreen %s", "on" if active else "off")
            self._bus.publish(FullscreenChanged(active=active))
        return active

    def poll_load(self) -> None:
        if not self._eco_wanted():
            # Eco mode is off: report "normal" once and stop sampling.
            self._high_streak = 0
            self._set_load(False, False, self._last_cpu)
            return
        cpu = self._sample_cpu()
        if cpu is not None:
            self._last_cpu = cpu
            if cpu > HIGH_CPU_PERCENT:
                self._high_streak += 1
            else:
                self._high_streak = 0
            if not self._high and self._high_streak >= HIGH_SAMPLES:
                high = True
            elif self._high and cpu < NORMAL_CPU_PERCENT:
                high = False
            else:
                high = self._high
        else:
            high = self._high
        self._set_load(high, self._read_battery_saver(), self._last_cpu)

    # -- internals ------------------------------------------------------------------

    def _default_fullscreen_probe(self) -> bool:
        try:
            own = self._own_window_ids()
        except Exception as exc:
            self._warn_once("own_ids", "own_window_ids() failed: %s", exc)
            own = set()
        return platform_win.foreground_fullscreen(own)

    def _eco_wanted(self) -> bool:
        try:
            return bool(self._settings().behavior.auto_eco_mode)
        except Exception as exc:
            self._warn_once("settings", "settings() failed: %s", exc)
            return True

    def _sample_cpu(self) -> float | None:
        try:
            value = float(self._cpu_probe())
        except Exception as exc:
            self._warn_once("cpu", "CPU probe failed: %s", exc)
            return None
        return value if math.isfinite(value) else None

    def _read_battery_saver(self) -> bool:
        try:
            reading = self._battery_probe()
        except Exception as exc:
            self._warn_once("battery", "battery probe failed: %s", exc)
            return False
        if reading is None:
            return False
        plugged, percent = reading
        return (not plugged) and math.isfinite(percent) and percent <= BATTERY_SAVER_PERCENT

    def _set_load(self, high: bool, saver: bool, cpu: float) -> None:
        if (high, saver) == (self._high, self._battery_saver):
            return
        self._high = high
        self._battery_saver = saver
        log.info("system load %s (cpu %.0f%%, battery saver %s)", "high" if high else "normal", cpu, saver)
        self._bus.publish(SystemLoadChanged(high=high, cpu_percent=cpu, on_battery_saver=saver))

    def _warn_once(self, key: str, message: str, *args: object) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(message, *args)
