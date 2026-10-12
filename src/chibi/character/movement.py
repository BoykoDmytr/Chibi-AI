"""Where she stands: walking, falling and being dragged around the desktop.

Coordinates are global *logical* pixels (Qt's device-independent pixels, the
same space as ``QScreen.availableGeometry()``), with y growing downwards.
``(x, y)`` is her feet anchor, so "standing on the taskbar" is simply
``y == area.bottom``.  Monitors left of / above the primary one have negative
coordinates; nothing here assumes the origin is on screen.

The controller owns the mapping to monitors: it calls :meth:`Movement.set_bounds`
with the work area of the monitor she is on (again whenever she is dragged to
another monitor, the taskbar moves or the display scale changes).

Physics is deliberately simple and frame-rate independent: ``tick`` splits a
large ``dt`` into steps of at most 0.1 s so a stalled frame timer cannot make
her tunnel through the floor or overshoot a walk target.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from chibi.character.states import Facing

log = logging.getLogger(__name__)

#: px/s^2 - a little "floatier" than real gravity at chibi scale, reads better.
GRAVITY = 2600.0
#: px/s - caps the fall speed so a drop from a 4K monitor's top still looks soft.
TERMINAL_VELOCITY = 2200.0
#: She bounces once with this fraction of the impact speed...
BOUNCE_FACTOR = 0.22
#: ...but only if she hit the floor faster than this (px/s); gentle drops just land.
BOUNCE_MIN_IMPACT = 500.0
#: Closer than this to the ground counts as standing on it (px).
GROUND_EPSILON = 0.5
#: A walk is finished once she is within this distance of the target (px).
ARRIVE_EPSILON = 1.0
#: Longest physics step (s); bigger ``dt`` values are split.
MAX_STEP = 0.1
#: Longest stretch simulated by one ``tick`` (s).  After a system suspend the
#: frame timer may report hours; simulating all of it would stall a frame for
#: nothing (a fall settles in ~2 s), so the rest simply continues next tick.
MAX_TICK_SECONDS = 60.0
#: While dragged, at least this fraction of her body width stays on the
#: monitor so she can always be grabbed again.
MIN_VISIBLE_FRACTION = 0.4


@dataclass(frozen=True)
class Rect:
    """A screen area in global logical px (``right``/``bottom`` are exclusive edges)."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Rect needs finite coordinates, got {values}")
        if self.right < self.left or self.bottom < self.top:
            raise ValueError(f"inverted Rect {values}")

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


class Movement:
    """Position and motion of the feet anchor.

    Plain attributes so the controller and renderer can read them every
    frame without call overhead; they are written only by this class.
    """

    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.facing: Facing = Facing.FRONT
        self.walking = False
        self.falling = False
        #: Standing on the floor (the area's bottom).  Until the first
        #: ``set_bounds`` there is no floor, so she counts as standing.
        self.on_ground = True
        #: The last walk reached its target (reset by walk_to / stop / place).
        self.arrived = False
        #: Whether she falls when above the floor; set by :meth:`drop`.
        self.gravity = True
        #: Being dragged: physics and walking are suspended.
        self.held = False
        self.vy = 0.0

        self._area: Rect | None = None
        self._half_width = 0.0
        self._head_height = 0.0
        self._min_x = -math.inf
        self._max_x = math.inf
        self._target: float | None = None
        self._speed = 0.0
        self._bounced = False

    # -- bounds -------------------------------------------------------------------

    @property
    def area(self) -> Rect | None:
        return self._area

    @property
    def ground(self) -> float | None:
        return self._area.bottom if self._area is not None else None

    @property
    def walk_range(self) -> tuple[float, float] | None:
        """Allowed feet x while walking: the whole body stays on the monitor."""
        return (self._min_x, self._max_x) if self._area is not None else None

    @property
    def target_x(self) -> float | None:
        return self._target

    def set_bounds(self, area: Rect, half_width: float, head_height: float) -> None:
        """Confine her to *area* (the work area of her monitor).

        *half_width* is half her body width and *head_height* the distance
        from feet to the top of her head, both in logical px at the current
        scale.  If she was standing on the old floor she stays standing on the
        new one (taskbar moved, other monitor); a character placed mid-air is
        kept where she is, inside the area.
        """
        first = self._area is None
        standing = not first and self.on_ground and not self.falling and not self.held
        self._area = area
        self._half_width = max(0.0, float(half_width))
        self._head_height = max(0.0, float(head_height))
        lo = area.left + self._half_width
        hi = area.right - self._half_width
        if lo > hi:  # narrower than she is: centre her
            lo = hi = area.center_x
        self._min_x, self._max_x = lo, hi

        if self.held:
            self.x, self.y = self._drag_clamp(self.x, self.y)
        else:
            self.x = _clamp(self.x, lo, hi)
            if standing or self.y > area.bottom:
                self.y = area.bottom
            else:
                self.y = max(self.y, self._head_floor())
        if self._target is not None:
            self._target = _clamp(self._target, lo, hi)
        self._settle()

    # -- commands -----------------------------------------------------------------

    def walk_to(self, x: float, speed: float) -> None:
        """Start walking towards *x* (clamped to the walkable range) at *speed* px/s."""
        if not math.isfinite(x):
            log.warning("ignoring walk to non-finite x")
            return
        if not (math.isfinite(speed) and speed > 0):
            raise ValueError(f"walking speed must be a positive number, got {speed!r}")
        target = _clamp(float(x), self._min_x, self._max_x)
        self.facing = Facing.RIGHT if target > self.x else Facing.LEFT
        self._target = target
        self._speed = float(speed)
        self.walking = True
        self.arrived = False

    def stop(self) -> None:
        """Abandon the current walk.  A fall in progress continues; only the walk ends."""
        self.walking = False
        self.arrived = False
        self._target = None

    def place(self, x: float, y: float) -> None:
        """Put her at (*x*, *y*) while she is dragged.

        Keeps ~40 % of her body width inside the area and her head below its
        top edge, so she can never be dragged out of reach.
        """
        self.held = True
        self.walking = False
        self.arrived = False
        self._target = None
        self.falling = False
        self.vy = 0.0
        self._bounced = False
        if math.isfinite(x) and math.isfinite(y):
            self.x, self.y = self._drag_clamp(float(x), float(y))
        self.on_ground = False

    def drop(self, gravity: bool) -> None:
        """The drag ended.  With *gravity* she falls to the floor; without it she
        stays where she was let go (standing mid-screen is a feature then)."""
        self.held = False
        self.gravity = bool(gravity)
        self.vy = 0.0
        self._bounced = False
        self._settle()

    # -- simulation ---------------------------------------------------------------

    def tick(self, dt: float) -> bool:
        """Advance by *dt* seconds.  Returns True if her position changed."""
        if not dt > 0:  # zero, negative or NaN
            return False
        x0, y0 = self.x, self.y
        remaining = min(dt, MAX_TICK_SECONDS)
        while remaining > 1e-9 and (self.walking or self.falling) and not self.held:
            step = min(remaining, MAX_STEP)
            remaining -= step
            if self.falling:
                self._fall(step)
            elif self.walking:
                self._walk(step)
        return self.x != x0 or self.y != y0

    # -- internals ----------------------------------------------------------------

    def _head_floor(self) -> float:
        """Smallest feet y that keeps her head inside the area."""
        if self._area is None:
            return -math.inf
        return min(self._area.top + self._head_height, self._area.bottom)

    def _drag_clamp(self, x: float, y: float) -> tuple[float, float]:
        area = self._area
        if area is None:
            return x, y
        overhang = self._half_width * (1.0 - 2.0 * MIN_VISIBLE_FRACTION)
        lo, hi = area.left - overhang, area.right + overhang
        if lo > hi:
            lo = hi = area.center_x
        return _clamp(x, lo, hi), _clamp(y, self._head_floor(), area.bottom)

    def _settle(self) -> None:
        """Recompute ground contact after a position or mode change."""
        if self.held:
            return
        if self._area is None:  # no floor known yet: she stands where she is
            self.on_ground = True
            self.falling = False
            return
        ground = self._area.bottom
        if abs(self.y - ground) < GROUND_EPSILON and not (self.falling and self.vy < 0):
            self.y = ground
            self.on_ground = True
            self.falling = False
            self.vy = 0.0
        else:
            self.on_ground = False
            if self.gravity and self.y < ground and not self.falling:
                self.falling = True
                self.vy = 0.0
                self._bounced = False

    def _fall(self, dt: float) -> None:
        if self._area is None:
            self.falling = False
            return
        ground = self._area.bottom
        self.vy = min(self.vy + GRAVITY * dt, TERMINAL_VELOCITY)
        self.y += self.vy * dt
        if self.y < ground:
            self.on_ground = False
            return
        impact = self.vy
        self.y = ground
        if impact > BOUNCE_MIN_IMPACT and not self._bounced:
            self.vy = -impact * BOUNCE_FACTOR
            self._bounced = True
            self.on_ground = False
        else:
            self.vy = 0.0
            self.falling = False
            self._bounced = False
            self.on_ground = True

    def _walk(self, dt: float) -> None:
        target = self._target
        if target is None:
            self.walking = False
            return
        dx = target - self.x
        step = self._speed * dt
        if abs(dx) <= step + ARRIVE_EPSILON:
            self.x = target
            self.walking = False
            self.arrived = True
            self._target = None
            return
        self.facing = Facing.RIGHT if dx > 0 else Facing.LEFT
        self.x += math.copysign(step, dx)
