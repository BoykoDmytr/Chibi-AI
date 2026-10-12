from __future__ import annotations

import math
from itertools import pairwise

import pytest

from chibi.character import movement as movement_module
from chibi.character.movement import (
    BOUNCE_FACTOR,
    GRAVITY,
    MIN_VISIBLE_FRACTION,
    TERMINAL_VELOCITY,
    Movement,
    Rect,
)
from chibi.character.states import Facing

PRIMARY = Rect(0, 0, 1920, 1040)  # 1080p with a 40 px taskbar
LEFT_MONITOR = Rect(-1920, 0, 0, 1080)  # left of the primary: negative x
UPPER_MONITOR = Rect(0, -1080, 1920, -40)  # above the primary: negative y
HW = 60.0  # half body width
HEAD = 200.0  # feet -> top of head


def standing(x: float = 960, area: Rect = PRIMARY) -> Movement:
    m = Movement(x, area.bottom)
    m.set_bounds(area, HW, HEAD)
    return m


def run(m: Movement, seconds: float, dt: float = 1 / 60) -> list[tuple[float, float, float, bool]]:
    """Tick for *seconds*; returns (t, y, vy, falling) after every tick."""
    trace = []
    steps = round(seconds / dt)
    for i in range(1, steps + 1):
        m.tick(dt)
        trace.append((i * dt, m.y, m.vy, m.falling))
    return trace


# -- Rect --------------------------------------------------------------------------------------


def test_rect_geometry() -> None:
    assert (PRIMARY.width, PRIMARY.height, PRIMARY.center_x) == (1920, 1040, 960)
    assert LEFT_MONITOR.contains(-1, 10) and not LEFT_MONITOR.contains(0, 10)


@pytest.mark.parametrize("args", [(10, 0, 0, 10), (0, 10, 10, 0), (0, 0, math.nan, 10), (0, 0, math.inf, 1)])
def test_rect_rejects_invalid(args: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError):
        Rect(*args)


# -- bounds ------------------------------------------------------------------------------------


def test_set_bounds_computes_walk_range_and_clamps() -> None:
    m = Movement(5000, 1040)
    m.set_bounds(PRIMARY, HW, HEAD)
    assert m.walk_range == (HW, 1920 - HW)
    assert m.x == 1920 - HW
    assert m.ground == 1040
    assert m.on_ground and not m.falling


def test_set_bounds_snaps_from_below_ground() -> None:
    m = Movement(500, 1500)
    m.set_bounds(PRIMARY, HW, HEAD)
    assert m.y == 1040 and m.on_ground


def test_first_bounds_with_her_in_the_air_makes_her_fall() -> None:
    m = Movement(500, 300)
    m.set_bounds(PRIMARY, HW, HEAD)
    assert m.y == 300 and m.falling and not m.on_ground
    run(m, 3)
    assert m.y == 1040 and m.on_ground and not m.falling


def test_first_bounds_in_the_air_without_gravity_stays() -> None:
    m = Movement(500, 300)
    m.gravity = False
    m.set_bounds(PRIMARY, HW, HEAD)
    assert m.y == 300 and not m.falling and not m.on_ground


def test_narrow_area_centres_her() -> None:
    narrow = Rect(100, 0, 180, 1040)  # 80 px wide, she is 120 px wide
    m = Movement(0, 1040)
    m.set_bounds(narrow, HW, HEAD)
    assert m.walk_range == (140, 140)
    assert m.x == 140
    m.walk_to(1000, 100)
    run(m, 1)
    assert m.x == 140 and m.arrived


def test_standing_character_moves_to_new_monitor_floor() -> None:
    m = standing(300)
    m.set_bounds(LEFT_MONITOR, HW, HEAD)
    assert m.x == -HW  # clamped into the left monitor's walk range
    assert m.y == 1080 and m.on_ground and not m.falling
    m.walk_to(-1500, 200)
    assert m.facing is Facing.LEFT
    run(m, 10)
    assert m.x == -1500 and m.arrived
    m.set_bounds(UPPER_MONITOR, HW, HEAD)
    assert m.x == HW and m.y == -40 and m.on_ground


def test_taskbar_change_keeps_her_standing() -> None:
    m = standing(500)
    m.set_bounds(Rect(0, 0, 1920, 1080), HW, HEAD)  # taskbar hidden: floor moved down
    assert m.y == 1080 and m.on_ground and not m.falling
    m.set_bounds(Rect(0, 0, 1920, 1000), HW, HEAD)  # taskbar grew: floor moved up
    assert m.y == 1000 and m.on_ground


def test_bounds_keep_mid_air_character_inside() -> None:
    m = standing(500)
    m.place(500, 400)
    m.drop(gravity=False)
    m.set_bounds(Rect(0, 600, 1920, 1040), HW, HEAD)  # smaller monitor area: head must stay visible
    assert m.y == 800 and not m.on_ground and not m.falling
    m.set_bounds(Rect(0, 0, 1920, 700), HW, HEAD)  # floor above her feet: snap to it
    assert m.y == 700 and m.on_ground


def test_bounds_change_clamps_walk_target() -> None:
    m = standing(500)
    m.walk_to(1800, 100)
    m.set_bounds(Rect(0, 0, 1000, 1040), HW, HEAD)
    assert m.target_x == 1000 - HW
    run(m, 10)
    assert m.x == 1000 - HW and m.arrived


def test_bounds_change_while_dragged_uses_drag_limits() -> None:
    m = standing(500)
    m.place(1900, 100)
    m.set_bounds(LEFT_MONITOR, HW, HEAD)
    overhang = HW * (1 - 2 * MIN_VISIBLE_FRACTION)
    assert m.x == pytest.approx(overhang)
    assert m.y == HEAD
    assert m.held and not m.falling


# -- walking -----------------------------------------------------------------------------------


def test_walk_right_at_speed_and_arrive() -> None:
    m = standing(500)
    m.walk_to(800, 100)
    assert m.walking and not m.arrived and m.facing is Facing.RIGHT
    assert m.tick(1.0)
    assert m.x == pytest.approx(600)
    assert m.tick(1.5)
    assert m.x == pytest.approx(750)
    m.tick(0.6)
    assert m.x == 800 and m.arrived and not m.walking
    assert not m.tick(1.0)  # nothing moves any more
    assert m.facing is Facing.RIGHT  # keeps looking where she went


def test_walk_left() -> None:
    m = standing(500)
    m.walk_to(100, 50)
    assert m.facing is Facing.LEFT
    run(m, 9)
    assert m.x == 100 and m.arrived


def test_arrives_within_one_pixel() -> None:
    m = standing(500)
    m.walk_to(500.8, 100)
    m.tick(0.01)
    assert m.x == 500.8 and m.arrived


def test_walk_target_is_clamped_to_range() -> None:
    m = standing(500)
    m.walk_to(-400, 1000)
    assert m.target_x == HW
    run(m, 2)
    assert m.x == HW


def test_walk_to_own_position_faces_left_and_arrives() -> None:
    m = standing(500)
    m.walk_to(500, 100)
    assert m.facing is Facing.LEFT
    m.tick(0.016)
    assert m.arrived and not m.walking


def test_large_dt_does_not_overshoot() -> None:
    m = standing(500)
    m.walk_to(600, 50)
    assert m.tick(5.0)
    assert m.x == 600 and m.arrived


def test_huge_dt_is_bounded() -> None:
    m = standing(100)
    m.walk_to(1800, 10)  # 170 s walk
    m.tick(math.inf)
    assert m.walking and m.x == pytest.approx(100 + 10 * movement_module.MAX_TICK_SECONDS)


def test_walk_speed_must_be_positive() -> None:
    m = standing(500)
    for bad in (0, -5, math.nan, math.inf):
        with pytest.raises(ValueError):
            m.walk_to(700, bad)


def test_non_finite_target_is_ignored() -> None:
    m = standing(500)
    m.walk_to(math.nan, 100)
    assert not m.walking


def test_stop() -> None:
    m = standing(500)
    m.walk_to(900, 100)
    m.tick(1)
    m.stop()
    assert not m.walking and not m.arrived and m.target_x is None
    assert not m.tick(1)
    assert m.x == pytest.approx(600)


@pytest.mark.parametrize("dt", [0, -1, math.nan])
def test_invalid_dt(dt: float) -> None:
    m = standing(500)
    m.walk_to(900, 100)
    assert not m.tick(dt)
    assert m.x == 500


def test_without_bounds_she_walks_unclamped() -> None:
    m = Movement(0, 0)
    assert m.walk_range is None and m.ground is None and m.on_ground
    m.walk_to(-5000, 1000)
    run(m, 6)
    assert m.x == -5000
    m.place(10, 20)
    m.drop(gravity=True)
    assert (m.x, m.y) == (10, 20) and m.on_ground and not m.falling


# -- gravity -----------------------------------------------------------------------------------


def test_fall_bounce_and_land_timing() -> None:
    m = standing(500)
    m.place(500, 200)  # head floor: y = top + HEAD
    m.drop(gravity=True)
    assert m.falling and not m.on_ground
    trace = run(m, 2.0, dt=1 / 120)

    height = 1040 - 200
    t_impact = math.sqrt(2 * height / GRAVITY)  # stays under terminal velocity
    first_landing = next(t for t, y, _, _ in trace if y == 1040)
    assert first_landing == pytest.approx(t_impact, abs=0.02)

    after = [(t, y, vy) for t, y, vy, _ in trace if t > first_landing]
    assert min(y for _, y, _ in after) < 1040 - 30  # she bounced up visibly
    bounce_speed = BOUNCE_FACTOR * GRAVITY * t_impact
    settle = next(t for t, y, vy, falling in trace if t > first_landing + 0.05 and not falling)
    assert settle == pytest.approx(t_impact + 2 * bounce_speed / GRAVITY, abs=0.03)
    assert m.y == 1040 and m.on_ground and not m.falling and m.vy == 0


def test_terminal_velocity() -> None:
    tall = Rect(0, 0, 1920, 4000)
    m = Movement(500, 4000)
    m.set_bounds(tall, HW, HEAD)
    m.place(500, 200)
    m.drop(gravity=True)
    trace = run(m, 3)
    assert max(vy for _, _, vy, _ in trace) == TERMINAL_VELOCITY
    assert m.on_ground


def test_gentle_drop_does_not_bounce() -> None:
    m = standing(500)
    m.place(500, 1020)  # 20 px above the floor: impact ~322 px/s
    m.drop(gravity=True)
    trace = run(m, 1)
    landed = next(i for i, (_, y, _, _) in enumerate(trace) if y == 1040)
    assert all(y == 1040 for _, y, _, _ in trace[landed:])
    assert not m.falling and m.on_ground


def test_bounces_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(movement_module, "BOUNCE_FACTOR", 0.6)  # second impact would exceed 500 px/s
    m = standing(500)
    m.place(500, 200)
    m.drop(gravity=True)
    trace = run(m, 4, dt=1 / 120)
    landings = sum(1 for (_, y0, _, _), (_, y1, _, _) in pairwise(trace)
                   if y0 < 1040 and y1 == 1040)
    assert landings == 2  # first impact + the one bounce, then she stays down
    assert m.on_ground and not m.falling


def test_drop_without_gravity_stays_mid_screen() -> None:
    m = standing(500)
    m.place(700, 500)
    m.drop(gravity=False)
    assert (m.x, m.y) == (700, 500)
    assert not m.on_ground and not m.falling and not m.held
    assert not m.tick(1.0)
    m.walk_to(900, 100)  # she may stroll along her imaginary floor
    run(m, 3)
    assert (m.x, m.y) == (900, 500)


def test_drop_near_ground_snaps() -> None:
    m = standing(500)
    m.place(500, 1039.7)
    m.drop(gravity=False)
    assert m.y == 1040 and m.on_ground and not m.falling


def test_drop_on_the_ground() -> None:
    m = standing(500)
    m.place(500, 1040)
    assert not m.on_ground  # held
    m.drop(gravity=True)
    assert m.on_ground and not m.falling


def test_walk_waits_for_landing() -> None:
    m = standing(500)
    m.place(500, 600)
    m.drop(gravity=True)
    m.walk_to(900, 100)
    m.tick(0.3)
    assert m.x == 500 and m.falling
    run(m, 6)
    assert m.x == 900 and m.y == 1040


# -- dragging ----------------------------------------------------------------------------------


def visible_width(m: Movement, area: Rect) -> float:
    return min(m.x + HW, area.right) - max(m.x - HW, area.left)


@pytest.mark.parametrize("area", [PRIMARY, LEFT_MONITOR, UPPER_MONITOR])
def test_place_keeps_her_reachable(area: Rect) -> None:
    m = standing(area.center_x, area)
    for x, y in [(area.left - 5000, area.top - 5000), (area.right + 5000, area.bottom + 5000),
                 (area.left, area.top), (area.right, area.bottom)]:
        m.place(x, y)
        assert visible_width(m, area) >= 2 * HW * MIN_VISIBLE_FRACTION - 1e-9
        assert area.top + HEAD <= m.y <= area.bottom
    m.place(area.left - 5000, area.top - 5000)
    assert m.x == pytest.approx(area.left - HW * (1 - 2 * MIN_VISIBLE_FRACTION))
    assert m.y == area.top + HEAD
    m.place(area.center_x, area.top + 500)
    assert (m.x, m.y) == (area.center_x, area.top + 500)  # free inside the area


def test_place_on_a_tiny_area() -> None:
    tiny = Rect(0, 0, 30, 100)  # shorter than her head height, narrower than her
    m = Movement(0, 100)
    m.set_bounds(tiny, HW, HEAD)
    for x in (-500, 500):
        m.place(x, -500)
        # 40 % of her body is wider than the area: the best possible is covering all of it
        assert visible_width(m, tiny) == tiny.width
        assert m.y == 100


def test_place_stops_walking_and_falling() -> None:
    m = standing(500)
    m.walk_to(900, 100)
    m.place(600, 300)
    assert m.held and not m.walking and not m.falling and not m.arrived
    assert not m.tick(1.0)  # held: no physics
    assert (m.x, m.y) == (600, 300)
    m.drop(gravity=True)
    m.tick(0.1)
    m.place(650, 300)  # grabbed again mid-fall
    assert not m.falling and m.vy == 0
