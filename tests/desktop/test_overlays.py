from __future__ import annotations

import math

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainter

from chibi.desktop.overlays import OVERLAY_KINDS, OVERLAY_REACH, paint_overlay
from tests.desktop.helpers import alpha_bbox, blank_image, count_different

SIZE = 100.0
CANVAS = 400
HEAD = QPointF(200.0, 220.0)


def render(kind: str | None, t: float, mirror: bool = False, size: float = SIZE):
    image = blank_image(CANVAS, CANVAS)
    painter = QPainter(image)
    paint_overlay(painter, kind, HEAD, size, t, mirror)
    assert painter.transform().isIdentity()  # state restored
    painter.end()
    return image


@pytest.mark.parametrize("kind", OVERLAY_KINDS)
def test_every_overlay_paints_inside_its_reach(qapp, kind):
    left, right, up, down = (r * SIZE for r in OVERLAY_REACH)
    painted_any = False
    for i in range(40):
        t = i * 0.137
        for mirror in (False, True):
            box = alpha_bbox(render(kind, t, mirror))
            if box is None:
                continue  # some overlays fade out for a moment of their cycle
            painted_any = True
            bl, bt, br, bb = box
            assert HEAD.x() - bl <= left + 1, (kind, t, mirror)
            assert br + 1 - HEAD.x() <= right + 1, (kind, t, mirror)
            assert HEAD.y() - bt <= up + 1, (kind, t, mirror)
            assert bb + 1 - HEAD.y() <= down + 1, (kind, t, mirror)
    assert painted_any


@pytest.mark.parametrize("kind", OVERLAY_KINDS)
def test_overlays_animate(qapp, kind):
    frames = [render(kind, t) for t in (0.1, 0.5, 0.9, 1.3)]
    assert any(count_different(frames[0], other) > 0 for other in frames[1:])


def test_mirror_moves_the_effect_to_the_other_side(qapp):
    right = alpha_bbox(render("anger", 0.2))
    left = alpha_bbox(render("anger", 0.2, mirror=True))
    assert right[0] > HEAD.x()
    assert left[2] < HEAD.x()


def test_glyphs_are_not_mirrored(qapp):
    """A mirrored '?' reads as a typo: only the position flips."""
    normal = render("question", 0.0)
    mirrored = render("question", 0.0, mirror=True)
    shift = round((alpha_bbox(normal)[0] + alpha_bbox(normal)[2]) / 2
                  - (alpha_bbox(mirrored)[0] + alpha_bbox(mirrored)[2]) / 2)
    moved = blank_image(CANVAS, CANVAS)
    painter = QPainter(moved)
    painter.drawImage(shift, 0, mirrored)
    painter.end()
    assert count_different(moved, normal) < 60


@pytest.mark.parametrize("kind", [None, "", "unknown_effect"])
def test_unknown_or_empty_kind_draws_nothing(qapp, kind):
    assert alpha_bbox(render(kind, 0.5)) is None


@pytest.mark.parametrize("size", [0.0, -3.0, math.nan, math.inf])
def test_bad_size_draws_nothing(qapp, size):
    assert alpha_bbox(render("hearts", 0.5, size=size)) is None


def test_bad_time_draws_nothing(qapp):
    assert alpha_bbox(render("zzz", math.nan)) is None


def test_overlay_scales_with_size(qapp):
    small = alpha_bbox(render("game", 0.0, size=50.0))
    big = alpha_bbox(render("game", 0.0, size=100.0))
    assert (big[2] - big[0]) == pytest.approx(2 * (small[2] - small[0]), abs=3)
