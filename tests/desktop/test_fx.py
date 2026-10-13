"""Procedural effects added in Phase 5 (jump, bounce, squash, sway, stretch, the
landing squash) - pure math on :func:`fx_transform` and its wave helpers - and the
new states drawn by both renderers."""

from __future__ import annotations

import itertools
import math
import random

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QTransform

from chibi.character.animation import CharacterPack, FramePlayer
from chibi.character.state_machine import DEFAULT_ACTION_SECONDS
from chibi.character.states import CharacterState, Facing
from chibi.desktop import renderers
from chibi.desktop.renderers import (
    AIR_STRETCH,
    BOUNCE_HZ,
    FX_KEYS,
    IMPULSE_SQUASH_MAX,
    IMPULSE_STRETCH_MAX,
    JUMP_PERIOD,
    STRETCH_PERIOD,
    SWAY_HZ,
    SpriteRenderer,
    VectorRenderer,
    VisualState,
    blink_change_in,
    fx_transform,
    hop_cycle,
    jump_cycle,
    stretch_envelope,
    vector_clip_for,
)
from tests.desktop.helpers import alpha_bbox, count_different
from tests.desktop.test_renderers import ANCHOR, HEIGHT, assert_inside_extents, paint

S = CharacterState
NEW_STATES = (S.JUMP, S.DANCE, S.EXCITED, S.PROUD, S.STRETCH)
#: The crouch, air and landing fractions of a jump cycle (see renderers.jump_cycle).
CROUCH, AIR, LAND = 0.18, 0.50, 0.17


def mapped(tr: QTransform, x: float, y: float) -> tuple[float, float]:
    p = tr.map(QPointF(x, y))
    return p.x(), p.y()


def max_step(f, start: float, stop: float, n: int = 4000) -> float:
    """Largest change of f between neighbouring samples: small means continuous."""
    xs = [start + (stop - start) * i / n for i in range(n + 1)]
    values = [f(x) for x in xs]
    return max(abs(b - a) for a, b in itertools.pairwise(values))


# -- the original effects are unchanged ----------------------------------------------------------


def _old_transform(fx: dict[str, float], t: float, height: float, pivot_y: float) -> QTransform:
    """The Phase 1 implementation, verbatim in substance."""
    two_pi = 2.0 * math.pi
    breathe = math.sin(two_pi * t / 3.4)
    bob = abs(math.sin(two_pi * t * 1.8))
    hop = max(0.0, math.sin(two_pi * t * 1.5))
    shake = math.sin(two_pi * t * 13.0)
    tilt = math.sin(two_pi * t / 2.6)
    swing = math.sin(two_pi * t * 1.1)
    tr = QTransform()
    dx = fx.get("shake", 0.0) * height * shake
    dy = -(fx.get("bob", 0.0) * height * bob + fx.get("hop", 0.0) * height * hop)
    if dx or dy:
        tr.translate(dx, dy)
    if fx.get("tilt", 0.0) * tilt:
        tr.rotate(fx.get("tilt", 0.0) * tilt)
    if fx.get("swing", 0.0) * swing:
        tr.translate(0.0, pivot_y)
        tr.rotate(fx.get("swing", 0.0) * swing)
        tr.translate(0.0, -pivot_y)
    a = fx.get("breathe", 0.0)
    if a and breathe:
        tr.scale(1.0 - a / 2.0 * breathe, 1.0 + a * breathe)
    return tr


@pytest.mark.parametrize("fx", [
    {"breathe": 0.012}, {"bob": 0.01}, {"hop": 0.035}, {"shake": 0.006, "bob": 0.01},
    {"tilt": 3.0}, {"swing": 9.0}, {"bob": 0.008, "breathe": 0.008}, {"hop": 0.02, "tilt": 4.0},
])
def test_original_effects_are_unchanged(fx):
    for t in (0.0, 0.13, 0.5, 1.37, 2.71, 100.3):
        new = fx_transform(fx, t, 220.0, -230.0, clip_t=0.4)
        old = _old_transform(fx, t, 220.0, -230.0)
        for point in ((0.0, 0.0), (50.0, -220.0), (-70.0, 10.0)):
            assert mapped(new, *point) == pytest.approx(mapped(old, *point), abs=1e-9), (fx, t, point)


def test_fx_keys_cover_every_effect_of_the_pack(mia_pack: CharacterPack):
    used = {key for clip in mia_pack.clips.values() for key in clip.fx}
    assert used <= set(FX_KEYS)
    assert {"jump", "bounce", "squash", "sway", "stretch"} <= used


def test_unknown_or_lone_squash_does_nothing():
    assert fx_transform({"wobble": 3.0}, 0.7, 100.0, -100.0).isIdentity()
    # squash only deforms in step with a jump, hop or bounce
    assert fx_transform({"squash": 0.2}, 0.7, 100.0, -100.0, clip_t=0.1).isIdentity()


# -- hop / jump / stretch waves -------------------------------------------------------------------


def test_hop_cycle_matches_the_hop_arc_and_squashes_between_hops():
    for i in range(200):
        p = i / 200
        lift, deform = hop_cycle(p)
        assert lift == pytest.approx(max(0.0, math.sin(2 * math.pi * p)), abs=1e-12)
        assert -AIR_STRETCH - 1e-12 <= deform <= 1.0 + 1e-12
        if p >= 0.5:
            assert lift == 0.0 and deform >= 0.0  # on the ground: squashed (or neutral)
        else:
            assert deform <= 0.0  # in the air: stretched (or neutral)
    assert hop_cycle(0.75) == pytest.approx((0.0, 1.0))  # deepest squash between two hops
    assert hop_cycle(0.125)[1] == pytest.approx(-AIR_STRETCH)  # fastest rise: most stretched
    assert hop_cycle(0.25) == pytest.approx((1.0, 0.0))  # apex: round again
    assert hop_cycle(3.75) == hop_cycle(0.75) and hop_cycle(-0.25) == pytest.approx(hop_cycle(0.75))


def test_jump_cycle_phases():
    # crouch: on the ground, squashing then releasing
    assert jump_cycle(0.0) == (0.0, 0.0)
    lift, deform = jump_cycle(CROUCH / 2)
    assert lift == 0.0 and deform == pytest.approx(1.0)
    # air: a full arc, round at the apex
    lift, deform = jump_cycle(CROUCH + AIR / 2)
    assert lift == pytest.approx(1.0) and deform == pytest.approx(0.0, abs=1e-12)
    lift, deform = jump_cycle(CROUCH + AIR / 4)
    assert lift == pytest.approx(math.sin(math.pi / 4)) and deform == pytest.approx(-AIR_STRETCH)
    # landing squash, then rest
    lift, deform = jump_cycle(CROUCH + AIR + LAND / 2)
    assert lift == 0.0 and deform == pytest.approx(1.0)
    assert jump_cycle(0.95) == (0.0, 0.0)
    # in the air exactly as long as the clip's airborne frame
    airborne = [i for i in range(1000) if jump_cycle(i / 1000)[0] > 0]
    assert airborne[0] / 1000 == pytest.approx(CROUCH, abs=0.002)
    assert airborne[-1] / 1000 == pytest.approx(CROUCH + AIR, abs=0.002)


def test_waves_are_continuous():
    for part in (0, 1):
        assert max_step(lambda p, k=part: jump_cycle(p)[k], 0.0, 2.0) < 0.01
        assert max_step(lambda p, k=part: hop_cycle(p)[k], 0.0, 2.0) < 0.01
    assert max_step(stretch_envelope, 0.0, 2 * STRETCH_PERIOD) < 0.01


def test_stretch_envelope():
    assert stretch_envelope(0.0) == 0.0
    ups = [stretch_envelope(x / 100) for x in range(101)]
    assert all(b >= a for a, b in itertools.pairwise(ups))  # eases up over the first second
    assert stretch_envelope(1.0) == pytest.approx(1.0)
    assert stretch_envelope(1.4) == 1.0  # held
    assert 0.0 < stretch_envelope(2.3) < 1.0  # easing down
    assert stretch_envelope(2.8) == pytest.approx(0.0, abs=1e-12)
    assert stretch_envelope(3.0) == 0.0  # rest
    assert stretch_envelope(STRETCH_PERIOD + 1.4) == 1.0  # repeats
    assert stretch_envelope(-1.8) == stretch_envelope(STRETCH_PERIOD - 1.8)


# -- the transforms -------------------------------------------------------------------------------


def test_jump_runs_on_clip_time():
    fx = {"jump": 0.1}
    apex = (CROUCH + AIR / 2) * JUMP_PERIOD
    # the clip just started: on the ground whatever the global time
    for t in (0.0, 0.31, 12.7):
        assert mapped(fx_transform(fx, t, 100.0, -100.0, clip_t=0.0), 0.0, 0.0) == pytest.approx((0, 0))
    assert mapped(fx_transform(fx, 3.3, 100.0, -100.0, clip_t=apex), 0.0, 0.0) == pytest.approx((0, -10))
    # without a clip time the global time is used
    assert mapped(fx_transform(fx, apex, 100.0, -100.0), 0.0, 0.0) == pytest.approx((0, -10))
    assert mapped(fx_transform(fx, apex, 100.0, -100.0, clip_t=math.nan), 0.0, 0.0) == pytest.approx((0, -10))


def test_squash_follows_the_jump_about_her_feet():
    fx = {"jump": 0.1, "squash": 0.1}
    crouch = fx_transform(fx, 0.0, 100.0, -100.0, clip_t=CROUCH / 2 * JUMP_PERIOD)
    assert mapped(crouch, 0.0, 0.0) == pytest.approx((0.0, 0.0))  # feet stay on the floor
    assert mapped(crouch, 10.0, -100.0) == pytest.approx((10.5, -90.0))  # 10 % shorter, 5 % wider
    rising = fx_transform(fx, 0.0, 100.0, -100.0, clip_t=(CROUCH + AIR / 4) * JUMP_PERIOD)
    lift = 10.0 * math.sin(math.pi / 4)
    assert mapped(rising, 0.0, 0.0) == pytest.approx((0.0, -lift))
    k = 0.1 * AIR_STRETCH  # stretched on the way up
    assert mapped(rising, 10.0, -100.0) == pytest.approx((10.0 * (1 - k / 2), -lift - 100.0 * (1 + k)))


def test_bounce_is_quick_small_hops_with_squash():
    fx = {"bounce": 0.05}
    top = 1.0 / (4 * BOUNCE_HZ)
    assert mapped(fx_transform(fx, top, 100.0, -100.0), 0.0, 0.0) == pytest.approx((0.0, -5.0))
    assert mapped(fx_transform(fx, 3 * top, 100.0, -100.0), 0.0, 0.0) == pytest.approx((0.0, 0.0))
    squashed = fx_transform({"bounce": 0.05, "squash": 0.1}, 3 * top, 100.0, -100.0)
    assert mapped(squashed, 0.0, -100.0) == pytest.approx((0.0, -90.0))  # between hops, on the ground


def test_squash_follows_hop_when_there_is_no_jump():
    on_ground = 0.75 / 1.5  # second half of a hop cycle, deepest squash
    tr = fx_transform({"hop": 0.03, "squash": 0.1}, on_ground, 100.0, -100.0)
    assert mapped(tr, 0.0, -100.0) == pytest.approx((0.0, -90.0))


def test_sway_rocks_about_the_feet_on_the_bob_beat():
    fx = {"sway": 6.0}
    t = 1.0 / (4 * SWAY_HZ)  # fully to one side
    tr = fx_transform(fx, t, 100.0, -100.0)
    assert mapped(tr, 0.0, 0.0) == pytest.approx((0.0, 0.0))
    x, y = mapped(tr, 0.0, -100.0)
    assert x == pytest.approx(100.0 * math.sin(math.radians(6.0)))
    assert y == pytest.approx(-100.0 * math.cos(math.radians(6.0)))
    # the extremes and the centre crossings fall on bob's beats (where bob touches down)
    for beat in range(8):
        t_beat = beat / (4 * SWAY_HZ)
        assert abs(math.sin(2 * math.pi * 1.8 * t_beat)) == pytest.approx(0.0, abs=1e-9)


def test_stretch_grows_upwards_from_the_feet():
    tr = fx_transform({"stretch": 0.1}, 77.0, 100.0, -100.0, clip_t=1.4)
    assert mapped(tr, 0.0, 0.0) == pytest.approx((0.0, 0.0))
    assert mapped(tr, 10.0, -100.0) == pytest.approx((9.5, -110.0))
    assert fx_transform({"stretch": 0.1}, 77.0, 100.0, -100.0, clip_t=0.0).isIdentity()


def test_landing_squash_argument_is_clamped():
    squashed = fx_transform({}, 0.0, 100.0, -100.0, squash=1.0)
    assert mapped(squashed, 0.0, -100.0) == pytest.approx((0.0, -100.0 * (1 - IMPULSE_SQUASH_MAX)))
    assert mapped(squashed, 10.0, 0.0) == pytest.approx((10.0 * (1 + IMPULSE_SQUASH_MAX / 2), 0.0))
    stretched = fx_transform({"breathe": 0.0}, 0.0, 100.0, -100.0, squash=-1.0)
    assert mapped(stretched, 0.0, -100.0) == pytest.approx((0.0, -100.0 * (1 + IMPULSE_STRETCH_MAX)))
    assert fx_transform({}, 0.0, 100.0, -100.0, squash=math.nan).isIdentity()
    assert fx_transform({}, math.nan, 100.0, -100.0, squash=0.0).isIdentity()
    # applied on top of a clip's effects
    with_fx = fx_transform({"bob": 0.01}, 0.0, 100.0, -100.0, squash=0.1)
    assert mapped(with_fx, 0.0, -100.0) == pytest.approx((0.0, -90.0))


def test_absurd_pack_values_never_flip_her():
    tr = fx_transform({"jump": 0.1, "squash": 50.0}, 0.0, 100.0, -100.0, clip_t=CROUCH / 2 * JUMP_PERIOD)
    assert tr.m22() > 0.0 and tr.m11() > 0.0
    tr = fx_transform({"stretch": -9.0}, 0.0, 100.0, -100.0, clip_t=1.4)
    assert tr.m22() > 0.0 and tr.m11() > 0.0


# -- extents --------------------------------------------------------------------------------------


def test_extents_include_the_jump_and_the_landing_squash(qapp, mia_pack: CharacterPack):
    sprite = SpriteRenderer(mia_pack)
    e = sprite.full_extents(HEIGHT)
    jump = mia_pack.clip_for(S.JUMP)
    k = HEIGHT / mia_pack.reference_height
    tallest = max(mia_pack.frames[f.frame_id].anchor[1] for f in jump.frames) * k
    assert e.up >= tallest + jump.fx["jump"] * HEIGHT - 0.01
    widest = max(e.left, e.right)
    for state in (S.IDLE, S.SIT):
        clip = mia_pack.clip_for(state)
        for f in clip.frames:
            info = mia_pack.frames[f.frame_id]
            assert widest >= info.anchor[0] * k * (1 + IMPULSE_SQUASH_MAX / 2) - 0.01


@pytest.mark.parametrize("state", NEW_STATES)
def test_new_states_stay_inside_extents_at_every_phase(qapp, mia_pack: CharacterPack, state):
    sprite, vector = SpriteRenderer(mia_pack), VectorRenderer()
    rng = random.Random(5)
    for renderer, clip in ((sprite, mia_pack.clip_for(state)), (vector, vector_clip_for(state))):
        for _ in range(12):
            t = rng.uniform(0.0, 10.0)
            clip_t = rng.uniform(0.0, clip.duration_ms / 1000.0)
            squash = rng.choice((0.0, IMPULSE_SQUASH_MAX, -IMPULSE_STRETCH_MAX))
            facing = rng.choice(list(Facing))
            sample = FramePlayer(clip).advance(clip_t * 1000.0)
            visual = VisualState(state, facing, t, sample, clip, clip.overlay, clip_t, squash)
            label = f"{state.value} t={t:.2f} clip_t={clip_t:.2f}"
            assert_inside_extents(paint(renderer, visual), renderer, label)


def test_jump_lifts_her_off_the_floor(qapp, mia_pack: CharacterPack):
    sprite = SpriteRenderer(mia_pack)
    clip = mia_pack.clip_for(S.JUMP)

    def bottom(clip_t: float) -> int:
        sample = FramePlayer(clip).advance(clip_t * 1000.0)
        image = paint(sprite, VisualState(S.JUMP, Facing.FRONT, 0.0, sample, clip, None, clip_t))
        return alpha_bbox(image)[3]

    on_floor = bottom(0.0)
    apex = bottom((CROUCH + AIR / 2) * JUMP_PERIOD)
    assert on_floor - apex >= 0.8 * clip.fx["jump"] * HEIGHT


# -- clips, durations and the vector chibi ---------------------------------------------------------


def test_durations_are_whole_effect_cycles(mia_pack: CharacterPack):
    assert DEFAULT_ACTION_SECONDS[S.JUMP] == pytest.approx(2 * JUMP_PERIOD)
    assert DEFAULT_ACTION_SECONDS[S.STRETCH] == pytest.approx(STRETCH_PERIOD)
    assert DEFAULT_ACTION_SECONDS[S.DANCE] * SWAY_HZ == pytest.approx(4.0, abs=0.05)
    # the pack's frames are timed to the same cycles
    assert mia_pack.clips["jump"].duration_ms == round(JUMP_PERIOD * 1000)
    assert mia_pack.clips["stretch"].duration_ms == round(STRETCH_PERIOD * 1000)
    assert mia_pack.clips["dance"].duration_ms == pytest.approx(1000 / SWAY_HZ, abs=2)
    assert mia_pack.clips["excited"].duration_ms == pytest.approx(2 * 1000 / BOUNCE_HZ, abs=1)
    assert not mia_pack.clips["stretch"].loop  # one stretch, then she relaxes


@pytest.mark.parametrize("state", NEW_STATES)
def test_vector_clips_mirror_the_pack(mia_pack: CharacterPack, state):
    pack_clip, vector_clip = mia_pack.clip_for(state), vector_clip_for(state)
    assert vector_clip.name == state.value
    assert dict(vector_clip.fx) == dict(pack_clip.fx)
    assert vector_clip.overlay == pack_clip.overlay
    assert vector_clip.loop == pack_clip.loop


def test_vector_new_states_have_their_own_pose(qapp):
    vector = VectorRenderer()
    images = {s: paint(vector, VisualState(s, Facing.FRONT, 0.6, None, vector_clip_for(s), None, 0.6))
              for s in (S.IDLE, S.HAPPY, *NEW_STATES)}
    for a in images:
        for b in images:
            if a is not b:
                assert count_different(images[a], images[b]) > 50, (a, b)


def test_vector_dance_alternates_arms(qapp):
    vector = VectorRenderer()
    clip = vector_clip_for(S.DANCE)

    def at(t: float):
        return paint(vector, VisualState(S.DANCE, Facing.FRONT, t, None, clip, None, t))

    quarter = 1.0 / (4 * SWAY_HZ)
    assert count_different(at(quarter), at(3 * quarter)) > 300


def test_vector_head_point_for_new_states_is_on_the_head(qapp):
    vector = VectorRenderer()
    for state in NEW_STATES:
        visual = VisualState(state, Facing.FRONT, 0.4, None, vector_clip_for(state))
        head = vector.head_point(ANCHOR, HEIGHT, visual)
        assert ANCHOR.y() - HEIGHT < head.y() < ANCHOR.y() - 0.45 * HEIGHT


# -- blinking -------------------------------------------------------------------------------------


def test_blink_change_in_points_at_the_next_edge():
    rng = random.Random(3)
    for _ in range(300):
        t = rng.uniform(0.0, 50.0)
        due = blink_change_in(t)
        assert due is not None and 0.0 < due <= 3.9
        before, after = renderers._blinking(t + due - 1e-6), renderers._blinking(t + due + 1e-6)
        assert before == renderers._blinking(t) and after != before
    assert blink_change_in(math.nan) is None
    assert blink_change_in(0.0) == pytest.approx(2.6)  # first blink at t = 2.6 s
    assert blink_change_in(2.65) == pytest.approx(0.08)  # eyes open again 130 ms after
