"""Small animated effects drawn around her head: anger vein, tear, "!", "?",
thinking dots, sparkles, hearts, blush, a gamepad while gaming and rising Zs.

Overlays are drawn by the controller on top of whatever the renderer drew,
so every renderer (sprite pack, vector fallback, a future Live2D one) gets
them for free.  A clip names its overlay in the pack manifest
(``"overlay": "anger"``).

Geometry is expressed in units of *size* (about one head diameter) around
the head point, so the effects scale with the character.  Glyphs are never
mirrored - a mirrored "?" reads as a typo - only their *positions* flip when
she faces the other way.  Everything is a vector path: no font is needed,
which keeps the look identical on every machine (and in headless tests).
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QRadialGradient

log = logging.getLogger(__name__)

OVERLAY_KINDS: tuple[str, ...] = (
    "anger", "tear", "exclaim", "question", "dots", "sparkle", "hearts", "blush", "game", "zzz",
)

#: Furthest any overlay reaches from the head point, in units of *size*:
#: (left, right, up, down).  Left equals right because positions mirror.
#: The controller sizes the window with it, so nothing is ever clipped
#: (tests paint every kind over a full cycle and check the bounding box).
OVERLAY_REACH: tuple[float, float, float, float] = (0.82, 0.82, 1.06, 0.74)

# Units: 100 == size.  The painter is scaled once per overlay.
_UNIT = 100.0

_WHITE = QColor(255, 255, 255)


def paint_overlay(painter: QPainter, kind: str | None, head: QPointF, size: float, t: float,
                  mirror: bool = False) -> None:
    """Draw overlay *kind* around *head* (the head centre, in painter coordinates).

    *size* is roughly the head diameter in px, *t* the animation time in
    seconds (overlays loop on it) and *mirror* flips positions to the other
    side for a character facing left.  Unknown kinds are ignored so a pack
    written for a newer build still loads.
    """
    if not kind:
        return
    fn = _PAINTERS.get(kind)
    if fn is None:
        log.debug("unknown overlay %r ignored", kind)
        return
    if not (math.isfinite(size) and size > 0 and math.isfinite(t)):
        return
    if not (math.isfinite(head.x()) and math.isfinite(head.y())):
        return
    painter.save()
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.translate(head)
        painter.scale(size / _UNIT, size / _UNIT)
        fn(painter, t, -1.0 if mirror else 1.0)
    finally:
        painter.restore()


# -- helpers ----------------------------------------------------------------------------------


def _frac(x: float) -> float:
    return x - math.floor(x)


def _pen(color: QColor, width: float) -> QPen:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _with_alpha(color: QColor, alpha: float) -> QColor:
    c = QColor(color)
    c.setAlphaF(max(0.0, min(1.0, alpha)) * color.alphaF())
    return c


def _stroke_outlined(painter: QPainter, path: QPainterPath, color: QColor, width: float,
                     outline: float = 3.2) -> None:
    """A stroke with a white halo: readable over dark hair and light desktops alike."""
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(_pen(_with_alpha(_WHITE, color.alphaF()), width + outline))
    painter.drawPath(path)
    painter.setPen(_pen(color, width))
    painter.drawPath(path)


@functools.cache
def _gamepad_path() -> QPainterPath:
    body = QPainterPath()
    body.addRoundedRect(QRectF(-15.0, -7.5, 30.0, 14.0), 7.0, 7.0)
    body.addEllipse(QPointF(-10.0, 4.0), 6.5, 6.5)
    body.addEllipse(QPointF(10.0, 4.0), 6.5, 6.5)
    return body.simplified()


def _heart_path() -> QPainterPath:
    path = QPainterPath(QPointF(0.0, 4.0))
    path.cubicTo(-10.0, -3.0, -5.5, -12.0, 0.0, -6.5)
    path.cubicTo(5.5, -12.0, 10.0, -3.0, 0.0, 4.0)
    return path


def _star_path(r: float) -> QPainterPath:
    """Four-pointed twinkle with concave sides."""
    path = QPainterPath(QPointF(0.0, -r))
    path.quadTo(0.0, 0.0, r, 0.0)
    path.quadTo(0.0, 0.0, 0.0, r)
    path.quadTo(0.0, 0.0, -r, 0.0)
    path.quadTo(0.0, 0.0, 0.0, -r)
    return path


# -- the overlays -----------------------------------------------------------------------------


def _anger(painter: QPainter, t: float, sx: float) -> None:
    """The manga "cross vein": four bent strokes around a centre, pulsing."""
    pulse = 1.0 + 0.14 * abs(math.sin(2 * math.pi * t * 1.6))
    painter.translate(36.0 * sx, -40.0)
    painter.scale(pulse, pulse)
    path = QPainterPath()
    for k in range(4):
        angle = math.radians(90 * k)
        c, s = math.cos(angle), math.sin(angle)

        def rot(x: float, y: float, c: float = c, s: float = s) -> QPointF:
            return QPointF(x * c - y * s, x * s + y * c)

        path.moveTo(rot(3.2, -11.5))
        path.quadTo(rot(3.2, -3.2), rot(11.5, -3.2))
    _stroke_outlined(painter, path, QColor(229, 57, 80), 4.0, outline=3.6)


def _tear(painter: QPainter, t: float, sx: float) -> None:
    """A drop sliding down from the eye, fading out at the bottom."""
    for offset in (0.0, 0.5):
        p = _frac(t / 1.9 + offset)
        alpha = min(1.0, p / 0.12) * (1.0 if p < 0.7 else max(0.0, (1.0 - p) / 0.3))
        if alpha <= 0.01:
            continue
        painter.save()
        painter.translate((24.0 + 3.0 * p) * sx, 30.0 + 30.0 * p)
        k = 0.75 + 0.3 * p
        painter.scale(k, k)
        path = QPainterPath(QPointF(0.0, -7.0))
        path.quadTo(4.6, -1.0, 4.6, 2.2)
        path.arcTo(QRectF(-4.6, -2.4, 9.2, 9.2), 0.0, -180.0)
        path.quadTo(-4.6, -1.0, 0.0, -7.0)
        grad = QRadialGradient(QPointF(-1.2, 0.5), 7.0)
        grad.setColorAt(0.0, _with_alpha(QColor(205, 236, 255), alpha))
        grad.setColorAt(1.0, _with_alpha(QColor(74, 163, 240), alpha))
        painter.setPen(_pen(_with_alpha(QColor(52, 120, 196), alpha), 0.9))
        painter.setBrush(QBrush(grad))
        painter.drawPath(path)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_with_alpha(_WHITE, alpha * 0.9))
        painter.drawEllipse(QPointF(-1.5, 1.0), 1.1, 1.5)
        painter.restore()


def _exclaim(painter: QPainter, t: float, sx: float) -> None:
    bounce = 1.0 + 0.1 * math.sin(2 * math.pi * t * 2.4)
    painter.translate(46.0 * sx, -62.0)
    painter.scale(bounce, bounce)
    painter.rotate(8.0 * sx)
    bar = QPainterPath(QPointF(-4.6, -16.0))
    bar.quadTo(0.0, -19.5, 4.6, -16.0)
    bar.lineTo(2.2, 4.0)
    bar.quadTo(0.0, 5.6, -2.2, 4.0)
    bar.closeSubpath()
    bar.addEllipse(QPointF(0.0, 10.5), 3.5, 3.5)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(_pen(_WHITE, 4.2))
    painter.drawPath(bar)
    painter.setPen(_pen(QColor(196, 40, 40), 0.9))
    painter.setBrush(QColor(255, 90, 79))
    painter.drawPath(bar)


def _question(painter: QPainter, t: float, sx: float) -> None:
    painter.translate(46.0 * sx, -60.0)
    painter.rotate(10.0 * math.sin(2 * math.pi * t * 0.9))
    hook = QPainterPath(QPointF(-7.0, -9.0))
    hook.cubicTo(-7.0, -19.5, 8.5, -19.5, 8.5, -9.5)
    hook.cubicTo(8.5, -3.0, 0.5, -3.5, 0.5, 3.0)
    color = QColor(79, 134, 232)
    _stroke_outlined(painter, hook, color, 5.0, outline=3.8)
    painter.setPen(_pen(_WHITE, 1.9))
    painter.setBrush(color)
    painter.drawEllipse(QPointF(0.5, 10.5), 3.4, 3.4)


def _dots(painter: QPainter, t: float, sx: float) -> None:
    """Thinking "...": the dots appear one after another, then all fade."""
    p = _frac(t / 1.8)
    fade = 1.0 if p < 0.82 else max(0.0, (1.0 - p) / 0.18)
    color = QColor(122, 138, 170)
    for i, (x, y, r) in enumerate(((22.0, -56.0, 4.0), (36.0, -64.0, 5.0), (51.0, -72.0, 6.0))):
        start = 0.08 + 0.22 * i
        alpha = min(1.0, max(0.0, (p - start) / 0.08)) * fade
        if alpha <= 0.01:
            continue
        painter.setPen(_pen(_with_alpha(_WHITE, alpha), 2.2))
        painter.setBrush(_with_alpha(color, alpha))
        painter.drawEllipse(QPointF(x * sx, y), r, r)


def _sparkle(painter: QPainter, t: float, sx: float) -> None:
    for x, y, r, phase in ((-52.0, -40.0, 9.0, 0.0), (54.0, -24.0, 6.5, 0.33), (34.0, -66.0, 7.5, 0.66)):
        k = 0.3 + 0.7 * abs(math.sin(math.pi * (t * 1.2 + phase)))
        painter.save()
        painter.translate(x * sx, y)
        painter.scale(k, k)
        grad = QRadialGradient(QPointF(0.0, 0.0), r)
        grad.setColorAt(0.0, _WHITE)
        grad.setColorAt(0.45, QColor(255, 236, 140))
        grad.setColorAt(1.0, QColor(255, 196, 40))
        painter.setPen(_pen(QColor(236, 160, 40), 0.8))
        painter.setBrush(QBrush(grad))
        painter.drawPath(_star_path(r))
        painter.restore()


def _hearts(painter: QPainter, t: float, sx: float) -> None:
    heart = _heart_path()
    for i, phase in enumerate((0.0, 0.34, 0.67)):
        p = _frac(t / 2.2 + phase)
        alpha = math.sin(math.pi * p)
        if alpha <= 0.02:
            continue
        x = 28.0 + 9.0 * math.sin(2 * math.pi * (p * 1.3 + i * 0.4))
        y = -18.0 - 66.0 * p
        k = 0.6 + 0.5 * p
        painter.save()
        painter.translate(x * sx, y)
        painter.scale(k, k)
        painter.setPen(_pen(_with_alpha(QColor(224, 67, 106), alpha), 1.0))
        painter.setBrush(_with_alpha(QColor(255, 107, 143), alpha))
        painter.drawPath(heart)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_with_alpha(_WHITE, alpha * 0.85))
        painter.drawEllipse(QPointF(-3.6, -4.2), 1.5, 1.0)
        painter.restore()


def _blush(painter: QPainter, t: float, sx: float) -> None:
    strength = 0.78 + 0.22 * math.sin(2 * math.pi * t * 0.8)
    for side in (-1.0, 1.0):
        cx, cy = 23.0 * side, 37.0
        grad = QRadialGradient(QPointF(cx, cy), 10.0)
        grad.setColorAt(0.0, _with_alpha(QColor(255, 120, 150), 0.75 * strength))
        grad.setColorAt(1.0, _with_alpha(QColor(255, 120, 150), 0.0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(grad))
        painter.drawEllipse(QPointF(cx, cy), 10.0, 5.5)
        painter.setPen(_pen(_with_alpha(QColor(236, 84, 120), 0.85 * strength), 1.3))
        for dx in (-3.5, 0.0, 3.5):
            painter.drawLine(QPointF(cx + dx - 1.4, cy + 2.0), QPointF(cx + dx + 1.4, cy - 2.0))


def _game(painter: QPainter, t: float, sx: float) -> None:
    """A little gamepad bobbing next to her head."""
    bob = -4.0 * abs(math.sin(2 * math.pi * t * 1.4))
    painter.translate(50.0 * sx, -56.0 + bob)
    painter.rotate(-8.0 * sx)
    body = _gamepad_path()
    painter.setPen(_pen(_WHITE, 3.6))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(body)
    painter.setPen(_pen(QColor(63, 55, 201), 1.1))
    painter.setBrush(QColor(108, 99, 255))
    painter.drawPath(body)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(_WHITE)
    painter.drawRoundedRect(QRectF(-11.6, -2.0, 7.2, 2.4), 0.8, 0.8)
    painter.drawRoundedRect(QRectF(-9.2, -4.4, 2.4, 7.2), 0.8, 0.8)
    painter.setBrush(QColor(255, 209, 102))
    painter.drawEllipse(QPointF(7.0, -2.2), 1.9, 1.9)
    painter.setBrush(QColor(239, 71, 111))
    painter.drawEllipse(QPointF(10.8, 1.2), 1.9, 1.9)


def _zzz(painter: QPainter, t: float, sx: float) -> None:
    """Three Zs rising and growing, staggered, from her head."""
    color = QColor(106, 127, 216)
    for i in range(3):
        p = _frac(t / 2.6 + i / 3.0)
        alpha = min(1.0, p / 0.15) * (1.0 - p) ** 0.6
        if alpha <= 0.02:
            continue
        s = 7.0 + 7.0 * p
        x = (24.0 + 38.0 * p + 4.0 * math.sin(2 * math.pi * p)) * sx
        y = -32.0 - 60.0 * p
        z = QPainterPath(QPointF(x - s / 2, y - s / 2))
        z.lineTo(x + s / 2, y - s / 2)
        z.lineTo(x - s / 2, y + s / 2)
        z.lineTo(x + s / 2, y + s / 2)
        _stroke_outlined(painter, z, _with_alpha(color, alpha), s * 0.26, outline=2.8)


_PAINTERS: dict[str, Callable[[QPainter, float, float], None]] = {
    "anger": _anger,
    "tear": _tear,
    "exclaim": _exclaim,
    "question": _question,
    "dots": _dots,
    "sparkle": _sparkle,
    "hearts": _hearts,
    "blush": _blush,
    "game": _game,
    "zzz": _zzz,
}
