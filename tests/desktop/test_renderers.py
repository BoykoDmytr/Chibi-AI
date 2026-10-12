from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage, QPainter, QTransform

from chibi.character.animation import CharacterPack, Clip, ClipFrame, FramePlayer, FrameSample
from chibi.character.states import CharacterState, Facing
from chibi.desktop.renderers import (
    VECTOR_CLIPS,
    CharacterRenderer,
    Extents,
    SpriteRenderer,
    VectorRenderer,
    VisualState,
    fx_transform,
    needs_mirror,
    vector_clip_for,
)
from tests.desktop.helpers import MIA_DIR, alpha_at, alpha_bbox, blank_image, count_different

S = CharacterState
HEIGHT = 180.0
CANVAS = 520
ANCHOR = QPointF(CANVAS / 2, CANVAS - 80)
TIMES = (0.0, 0.43, 1.37, 2.71)


def paint(renderer: CharacterRenderer, visual: VisualState, *, height: float = HEIGHT, dpr: float = 1.0,
          anchor: QPointF = ANCHOR, size: int = CANVAS) -> QImage:
    image = blank_image(round(size * dpr), round(size * dpr), dpr)
    painter = QPainter(image)
    renderer.paint(painter, anchor, height, visual)
    painter.end()
    return image


def sprite_visual(pack: CharacterPack, state: CharacterState, t: float, facing: Facing = Facing.FRONT,
                  variant: str | None = None) -> VisualState:
    clip = pack.clip_for(state, variant)
    sample = FramePlayer(clip).advance(t * 1000.0)
    return VisualState(state, facing, t, sample, clip, clip.overlay)


def vector_visual(state: CharacterState, t: float, facing: Facing = Facing.FRONT,
                  variant: str | None = None) -> VisualState:
    clip = vector_clip_for(state, variant)
    return VisualState(state, facing, t, FramePlayer(clip).advance(t * 1000.0), clip, clip.overlay)


def assert_inside_extents(image: QImage, renderer: CharacterRenderer, label: str) -> None:
    box = alpha_bbox(image)
    assert box is not None, f"{label}: nothing painted"
    e = renderer.full_extents(HEIGHT)
    left, top, right, bottom = box
    tol = 1.5
    assert ANCHOR.x() - left <= e.left + tol, label
    assert right + 1 - ANCHOR.x() <= e.right + tol, label
    assert ANCHOR.y() - top <= e.up + tol, label
    assert bottom + 1 - ANCHOR.y() <= e.down + tol, label


@pytest.fixture(scope="module")
def sprite(mia_pack: CharacterPack) -> SpriteRenderer:
    return SpriteRenderer(mia_pack)


@pytest.fixture(scope="module")
def vector() -> VectorRenderer:
    return VectorRenderer()


# -- every state, both renderers ------------------------------------------------------------------


@pytest.mark.parametrize("state", list(CharacterState))
def test_sprite_paints_every_state_inside_extents(qapp, sprite, mia_pack, state):
    for t in TIMES:
        for facing in (Facing.FRONT, Facing.LEFT, Facing.RIGHT):
            image = paint(sprite, sprite_visual(mia_pack, state, t, facing))
            assert_inside_extents(image, sprite, f"{state.value} t={t} {facing.value}")


@pytest.mark.parametrize("state", list(CharacterState))
def test_vector_paints_every_state_inside_extents(qapp, vector, state):
    for t in (*TIMES, 0.2, 3.3):
        for facing in (Facing.FRONT, Facing.LEFT, Facing.RIGHT):
            image = paint(vector, vector_visual(state, t, facing))
            assert_inside_extents(image, vector, f"{state.value} t={t} {facing.value}")


@pytest.mark.parametrize("variant", ["idle_look", "idle_smile"])
def test_fidget_variants_paint(qapp, sprite, vector, mia_pack, variant):
    assert_inside_extents(paint(sprite, sprite_visual(mia_pack, S.IDLE, 0.5, Facing.RIGHT, variant)), sprite,
                          variant)
    assert_inside_extents(paint(vector, vector_visual(S.IDLE, 0.5, Facing.RIGHT, variant)), vector, variant)


def test_visual_without_frame_or_clip_still_paints(qapp, sprite, vector):
    for renderer in (sprite, vector):
        image = paint(renderer, VisualState(S.HAPPY))
        assert alpha_bbox(image) is not None


def test_extents_are_positive_and_scale_linearly(qapp, sprite, vector):
    for renderer in (sprite, vector):
        left, right, up = renderer.extents(100.0)
        assert left > 0 and right > 0 and up > 90  # at least her height above the feet
        double = renderer.full_extents(200.0)
        single = renderer.full_extents(100.0)
        assert double.up == pytest.approx(2 * single.up)
        assert double.left == pytest.approx(2 * single.left)


def test_extents_include_fx_headroom(qapp, mia_pack):
    """The happy clip hops (3.5 % of her height): its top must be covered."""
    sprite = SpriteRenderer(mia_pack)
    e = sprite.full_extents(HEIGHT)
    happy = mia_pack.clip_for(S.HAPPY)
    k = HEIGHT / mia_pack.reference_height
    tallest = max(mia_pack.frames[f.frame_id].anchor[1] for f in happy.frames) * k
    assert e.up >= tallest + happy.fx["hop"] * HEIGHT - 0.01


def test_head_point_is_inside_the_head_region(qapp, sprite, vector, mia_pack):
    cases = ((sprite, sprite_visual(mia_pack, S.IDLE, 0.0)), (vector, vector_visual(S.IDLE, 0.0)))
    for renderer, visual in cases:
        head = renderer.head_point(ANCHOR, HEIGHT, visual)
        # above the body, below the top of the hair, roughly centred
        assert ANCHOR.y() - HEIGHT < head.y() < ANCHOR.y() - 0.45 * HEIGHT
        assert abs(head.x() - ANCHOR.x()) < 0.25 * HEIGHT
        bounds = renderer.head_bounds(HEIGHT)
        assert -bounds.left - 0.01 <= head.x() - ANCHOR.x() <= bounds.right + 0.01
        assert -bounds.up - 0.01 <= head.y() - ANCHOR.y() <= bounds.down + 0.01


def test_sleeping_head_is_low_and_pose_is_wide(qapp, vector, sprite, mia_pack):
    sleeping = vector_visual(S.SLEEP, 0.5)
    head = vector.head_point(ANCHOR, HEIGHT, sleeping)
    assert ANCHOR.y() - head.y() < 0.4 * HEIGHT
    left, top, right, bottom = alpha_bbox(paint(vector, sleeping))
    assert right - left > bottom - top  # lying down: wider than tall
    standing = alpha_bbox(paint(vector, vector_visual(S.IDLE, 0.5)))
    assert standing[3] - standing[1] > standing[2] - standing[0]
    sprite_head = sprite.head_point(ANCHOR, HEIGHT, sprite_visual(mia_pack, S.SLEEP, 0.5))
    assert ANCHOR.y() - sprite_head.y() < 0.6 * HEIGHT


# -- mirroring -------------------------------------------------------------------------------------


def test_needs_mirror_rules():
    assert needs_mirror(Facing.LEFT, Facing.RIGHT)
    assert needs_mirror(Facing.RIGHT, Facing.LEFT)
    assert not needs_mirror(Facing.LEFT, Facing.LEFT)
    assert not needs_mirror(Facing.LEFT, Facing.FRONT)
    assert not needs_mirror(Facing.FRONT, Facing.LEFT)
    assert not needs_mirror(Facing.FRONT, Facing.RIGHT)


def test_sprite_mirroring_flips_the_walk(qapp, sprite, mia_pack):
    walk = mia_pack.clip_for(S.WALK)
    assert walk.facing is Facing.LEFT
    as_drawn = paint(sprite, sprite_visual(mia_pack, S.WALK, 0.0, Facing.LEFT))
    flipped = paint(sprite, sprite_visual(mia_pack, S.WALK, 0.0, Facing.RIGHT))
    a, b = alpha_bbox(as_drawn), alpha_bbox(flipped)
    # the bounding box mirrors around the anchor (1-2 px of rounding allowed)
    assert abs((ANCHOR.x() - a[0]) - (b[2] + 1 - ANCHOR.x())) <= 2
    assert abs((a[2] + 1 - ANCHOR.x()) - (ANCHOR.x() - b[0])) <= 2
    assert a[1] == b[1] and a[3] == b[3]
    # and the pixels are the mirror image
    mirrored = as_drawn.transformed(QTransform.fromScale(-1, 1))
    shift = round(2 * ANCHOR.x() - CANVAS)
    expected = blank_image(CANVAS, CANVAS)
    painter = QPainter(expected)
    painter.drawImage(shift, 0, mirrored)
    painter.end()
    assert count_different(expected, flipped) < 0.03 * CANVAS * CANVAS


def test_front_art_never_mirrors(qapp, sprite, mia_pack):
    left = paint(sprite, sprite_visual(mia_pack, S.IDLE, 0.0, Facing.LEFT))
    right = paint(sprite, sprite_visual(mia_pack, S.IDLE, 0.0, Facing.RIGHT))
    assert count_different(left, right) == 0


def test_vector_mirrors_for_left(qapp, vector):
    right = paint(vector, vector_visual(S.POINT, 0.0, Facing.RIGHT))
    left = paint(vector, vector_visual(S.POINT, 0.0, Facing.LEFT))
    rb, lb = alpha_bbox(right), alpha_bbox(left)
    assert rb[2] - ANCHOR.x() > ANCHOR.x() - rb[0]  # pointing arm sticks out to the right
    assert ANCHOR.x() - lb[0] > lb[2] - ANCHOR.x()  # ... and to the left when mirrored


# -- cross-fade ------------------------------------------------------------------------------------


def test_crossfade_paints_current_opaque_and_next_on_top(qapp, mia_pack):
    sprite = SpriteRenderer(mia_pack)
    frames = (ClipFrame("front_0", 100), ClipFrame("sleep_0", 100))
    clip = Clip("pair", frames, loop=True, facing=Facing.FRONT)

    def render(sample: FrameSample) -> QImage:
        return paint(sprite, VisualState(S.IDLE, Facing.FRONT, 0.0, sample, clip))

    only_a = render(FrameSample("front_0", None, 0.0, False))
    only_b = render(FrameSample("sleep_0", None, 0.0, False))
    blended = render(FrameSample("front_0", "sleep_0", 0.5, False))
    solid_a = only_b_pixel = None
    for y in range(0, CANVAS, 3):
        for x in range(0, CANVAS, 3):
            a, b = alpha_at(only_a, x, y), alpha_at(only_b, x, y)
            # The current frame is drawn at full opacity: blending never lowers its alpha.
            assert alpha_at(blended, x, y) >= a - 1, (x, y)
            if solid_a is None and a >= 250:
                solid_a = (x, y)
            if only_b_pixel is None and a == 0 and b >= 250:
                only_b_pixel = (x, y)
    assert solid_a is not None and only_b_pixel is not None
    assert alpha_at(blended, *solid_a) >= 250
    assert 90 <= alpha_at(blended, *only_b_pixel) <= 170  # the next frame is half faded in
    assert alpha_at(render(FrameSample("front_0", "sleep_0", 1.0, False)), *only_b_pixel) >= 250


# -- caching / DPI / broken frames -------------------------------------------------------------


def test_pixmaps_are_cached_per_size_and_bounded(qapp, mia_pack):
    sprite = SpriteRenderer(mia_pack)
    visual = sprite_visual(mia_pack, S.IDLE, 0.0)
    paint(sprite, visual)
    assert sprite.cached_pixmaps == 1
    paint(sprite, visual)
    assert sprite.cached_pixmaps == 1  # same frame, size, mirror and DPR: reused
    paint(sprite, visual, height=200.0)
    assert sprite.cached_pixmaps == 2
    for frame_id in mia_pack.frames:
        for mirrored in (False, True):
            sprite._pixmap(frame_id, HEIGHT, mirrored, 1.0)
    assert sprite.cached_pixmaps <= SpriteRenderer.PIXMAP_CACHE_ENTRIES
    sprite.invalidate()
    assert sprite.cached_pixmaps == 0


def test_high_dpi_renders_at_device_resolution(qapp, mia_pack):
    sprite = SpriteRenderer(mia_pack)
    visual = sprite_visual(mia_pack, S.IDLE, 0.0)
    low = alpha_bbox(paint(sprite, visual, dpr=1.0))
    high = alpha_bbox(paint(sprite, visual, dpr=2.0))
    # Same logical size, twice the device pixels: crisp, not upscaled.
    assert abs((high[2] - high[0]) - 2 * (low[2] - low[0])) <= 3
    assert abs((high[3] - high[1]) - 2 * (low[3] - low[1])) <= 3
    keys = list(sprite._pixmaps._items)
    assert {k[4] for k in keys} == {1.0, 2.0}
    big = next(p for k, p in sprite._pixmaps._items.items() if k[4] == 2.0)
    assert big.devicePixelRatio() == 2.0


def test_broken_frame_falls_back_to_vector(qapp, tmp_path: Path, caplog):
    pack_dir = tmp_path / "mia_copy"
    shutil.copytree(MIA_DIR, pack_dir)
    (pack_dir / "frames" / "front_0.png").write_bytes(b"this is not a png")
    pack = CharacterPack.load(pack_dir)
    sprite = SpriteRenderer(pack)
    caplog.set_level(logging.WARNING, logger="chibi.desktop.renderers")
    visual = VisualState(S.IDLE, Facing.FRONT, 0.0, FrameSample("front_0", None, 0.0, False),
                         pack.clip_for(S.IDLE))
    image = paint(sprite, visual)
    assert alpha_bbox(image) is not None  # the vector chibi stands in
    reference = paint(VectorRenderer(), visual)
    assert count_different(image, reference) == 0
    assert any("front_0" in r.getMessage() for r in caplog.records)
    paint(sprite, visual)  # logged once, not every frame
    assert sum("front_0" in r.getMessage() for r in caplog.records) == 1


def test_zero_or_invalid_height_paints_nothing(qapp, sprite, vector, mia_pack):
    for renderer in (sprite, vector):
        for height in (0.0, -5.0, math.nan):
            image = paint(renderer, sprite_visual(mia_pack, S.IDLE, 0.0), height=height)
            assert alpha_bbox(image) is None


# -- the vector chibi expresses states ------------------------------------------------------------


def test_vector_expressions_differ(qapp, vector):
    images = {state: paint(vector, vector_visual(state, 0.6)) for state in
              (S.IDLE, S.HAPPY, S.ANGRY, S.SAD, S.SURPRISED, S.SLEEP, S.LAUGH, S.SHY, S.WAVE)}
    assert count_different(images[S.HAPPY], images[S.ANGRY]) > 400
    for a in images:
        for b in images:
            if a is not b:
                assert count_different(images[a], images[b]) > 50, (a, b)


def test_vector_walk_animates_legs(qapp, vector):
    a = paint(vector, vector_visual(S.WALK, 0.0, Facing.RIGHT))
    b = paint(vector, vector_visual(S.WALK, 0.14, Facing.RIGHT))
    assert count_different(a, b) > 50


def test_vector_blinks(qapp, vector):
    open_eyes = paint(vector, vector_visual(S.IDLE, 0.0))
    blinking = paint(vector, vector_visual(S.IDLE, 2.62))  # fmod(t + 1.3, 3.9) < 0.13
    assert count_different(open_eyes, blinking) > 20


def test_vector_is_not_a_rectangle(qapp, vector):
    image = paint(vector, vector_visual(S.IDLE, 0.0))
    left, top, right, bottom = alpha_bbox(image)
    # corners of the bounding box are empty: a rounded character, not a block
    for x, y in ((left, top), (right, top), (left, bottom), (right, bottom)):
        assert alpha_at(image, x, y) < 40


def test_vector_clip_table_covers_every_state():
    for state in CharacterState:
        clip = vector_clip_for(state)
        assert clip.frames and clip.name in VECTOR_CLIPS
    assert vector_clip_for(S.IDLE, "idle_look").name == "idle_look"
    assert not vector_clip_for(S.IDLE, "idle_look").loop
    assert vector_clip_for(S.IDLE, "nope").name == "idle"
    assert vector_clip_for(S.TALK, "idle_look").name == "talk"  # variant of another state
    assert vector_clip_for(S.SLEEP).overlay == "zzz"


# -- effects ------------------------------------------------------------------------------------------


def test_fx_transform_identity_without_fx():
    assert fx_transform({}, 1.0, 200.0, -200.0).isIdentity()
    assert fx_transform({"breathe": 0.01}, math.nan, 200.0, -200.0).isIdentity()


def test_fx_breathe_scales_about_the_feet():
    tr = fx_transform({"breathe": 0.1}, 3.4 / 4, 100.0, -100.0)  # sin = 1
    feet = tr.map(QPointF(0.0, 0.0))
    top = tr.map(QPointF(0.0, -100.0))
    assert feet.x() == pytest.approx(0.0) and feet.y() == pytest.approx(0.0)
    assert top.y() == pytest.approx(-110.0)
    assert tr.map(QPointF(10.0, 0.0)).x() == pytest.approx(9.5)


def test_fx_swing_pivots_on_the_head_top():
    tr = fx_transform({"swing": 9.0}, 1.0 / (4 * 1.1), 100.0, -100.0)  # sin = 1
    pivot = tr.map(QPointF(0.0, -100.0))
    assert pivot.x() == pytest.approx(0.0, abs=1e-6) and pivot.y() == pytest.approx(-100.0)
    feet = tr.map(QPointF(0.0, 0.0))
    assert feet.x() == pytest.approx(-100.0 * math.sin(math.radians(9.0)), rel=1e-3)


def test_fx_hop_and_shake_translate():
    hop = fx_transform({"hop": 0.1}, 1.0 / (4 * 1.5), 200.0, -200.0)
    assert hop.map(QPointF(0.0, 0.0)).y() == pytest.approx(-20.0)
    shake = fx_transform({"shake": 0.01}, 1.0 / (4 * 13.0), 200.0, -200.0)
    assert shake.map(QPointF(0.0, 0.0)).x() == pytest.approx(2.0)


def test_extents_union_and_scale():
    a = Extents(1, 2, 3, 4).union(Extents(4, 1, 5, -1))
    assert a == Extents(4, 2, 5, 4)
    assert a.scaled(2) == Extents(8, 4, 10, 8)


def test_sprite_ignores_a_foreign_clip(qapp, sprite):
    """A clip whose frames are not in the pack (e.g. a vector clip) falls back to the pack's own."""
    visual = vector_visual(S.HAPPY, 0.3)
    image = paint(sprite, visual)
    assert alpha_bbox(image) is not None
    head = sprite.head_point(ANCHOR, HEIGHT, visual)
    assert ANCHOR.y() - HEIGHT < head.y() < ANCHOR.y()
