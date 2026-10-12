"""Drawing the character: sprite frames from a character pack, or the vector fallback.

The controller only speaks :class:`VisualState` (state, facing, time, frame
sample, clip); a renderer turns that into pixels.  Two implementations:

* :class:`SpriteRenderer` draws the pack's PNG frames.  Frames are scaled
  once per (frame, size, mirror, device pixel ratio) with smooth filtering
  and kept in a small LRU of pixmaps rendered at *device* resolution, so
  painting a frame is a plain blit and stays crisp on 150 %/200 % displays.
* :class:`VectorRenderer` is an original chibi drawn with ``QPainterPath``s.
  It is the fallback when the pack is missing or broken (and a setting), so
  it has to express every :class:`CharacterState` on its own.

Procedural effects (``clip.fx``: breathe, bob, hop, shake, tilt, swing) are
painter transforms around the feet anchor, shared by both renderers via
:func:`fx_transform`.  :meth:`CharacterRenderer.extents` reports how far
anything can reach from the anchor *including* those effects, which is what
the controller sizes the transparent window with - nothing is ever clipped.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QImageReader,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
    QTransform,
)

from chibi.character.animation import CharacterPack, Clip, ClipFrame, FrameSample
from chibi.character.states import CharacterState, Facing

log = logging.getLogger(__name__)

S = CharacterState


@dataclass(frozen=True)
class VisualState:
    """Everything a renderer needs for one frame.

    ``t`` is a continuous animation time in seconds (procedural effects and
    the vector chibi's blinking loop on it).  ``frame``/``clip`` come from
    the controller's :class:`~chibi.character.animation.FramePlayer`;
    ``overlay`` is drawn by the controller, renderers ignore it.
    """

    state: CharacterState
    facing: Facing = Facing.FRONT
    t: float = 0.0
    frame: FrameSample | None = None
    clip: Clip | None = None
    overlay: str | None = None


@dataclass(frozen=True)
class Extents:
    """Distances from the feet anchor in px: how far drawing can reach to the
    left, right, up and down.  ``down`` may be negative for :meth:`head_bounds`
    (the head point is always above the feet)."""

    left: float
    right: float
    up: float
    down: float

    def union(self, other: Extents) -> Extents:
        return Extents(max(self.left, other.left), max(self.right, other.right),
                       max(self.up, other.up), max(self.down, other.down))

    def scaled(self, k: float) -> Extents:
        return Extents(self.left * k, self.right * k, self.up * k, self.down * k)


@runtime_checkable
class CharacterRenderer(Protocol):
    def paint(self, painter: QPainter, anchor: QPointF, height_px: float, visual: VisualState) -> None:
        """Draw the character with her feet at *anchor*, *height_px* tall."""
        ...

    def head_point(self, anchor: QPointF, height_px: float, visual: VisualState) -> QPointF:
        """Centre of her head (overlays are drawn and the bubble points there).
        Procedural effects are not applied: the bubble should not jitter."""
        ...

    def extents(self, height_px: float) -> tuple[float, float, float]:
        """(left, right, up): max reach from the anchor over everything drawable."""
        ...

    def full_extents(self, height_px: float) -> Extents:
        """Like :meth:`extents` plus how far below the anchor drawing can reach."""
        ...

    def head_bounds(self, height_px: float) -> Extents:
        """Range of :meth:`head_point` offsets from the anchor over every state."""
        ...

    def invalidate(self) -> None:
        """Drop cached pixmaps (display scale / screen changed)."""
        ...


# -- procedural effects -------------------------------------------------------------------------

#: Effects understood by :func:`fx_transform`; others in a manifest are ignored.
FX_KEYS: tuple[str, ...] = ("breathe", "bob", "hop", "shake", "tilt", "swing")


def _fx_matrix(fx: Mapping[str, float], breathe: float, bob: float, hop: float, shake: float,
               tilt: float, swing: float, height: float, pivot_y: float) -> QTransform:
    """The effect transform for explicit wave values (sines already evaluated).

    Order (outermost first): translate (shake/bob/hop), tilt about the feet,
    swing about the head top, breathe scale about the feet.
    """
    tr = QTransform()
    dx = fx.get("shake", 0.0) * height * shake
    dy = -(fx.get("bob", 0.0) * height * bob + fx.get("hop", 0.0) * height * hop)
    if dx or dy:
        tr.translate(dx, dy)
    tilt_deg = fx.get("tilt", 0.0) * tilt
    if tilt_deg:
        tr.rotate(tilt_deg)
    swing_deg = fx.get("swing", 0.0) * swing
    if swing_deg:
        tr.translate(0.0, pivot_y)
        tr.rotate(swing_deg)
        tr.translate(0.0, -pivot_y)
    a = fx.get("breathe", 0.0)
    if a and breathe:
        tr.scale(1.0 - a / 2.0 * breathe, 1.0 + a * breathe)
    return tr


def fx_transform(fx: Mapping[str, float], t: float, height_px: float, pivot_y: float) -> QTransform:
    """Transform (relative to the feet anchor) for a clip's *fx* at time *t*.

    Amplitudes of breathe/bob/hop/shake are fractions of the figure height,
    tilt/swing are degrees.  *pivot_y* is the head top (negative, px above
    the feet) that ``swing`` rotates about - she dangles from where she is
    held while dragged.
    """
    if not fx or not math.isfinite(t):
        return QTransform()
    two_pi = 2.0 * math.pi
    return _fx_matrix(
        fx,
        breathe=math.sin(two_pi * t / 3.4),
        bob=abs(math.sin(two_pi * t * 1.8)),
        hop=max(0.0, math.sin(two_pi * t * 1.5)),
        shake=math.sin(two_pi * t * 13.0),
        tilt=math.sin(two_pi * t / 2.6),
        swing=math.sin(two_pi * t * 1.1),
        height=height_px,
        pivot_y=pivot_y,
    )


def _fx_extremes(fx: Mapping[str, float]) -> Iterator[tuple[float, ...]]:
    """Wave-value combinations that bound every pose the effects can produce.

    Each wave is sampled at its extremes (rotations also in between, since a
    rotated corner can peak inside the range); keys a clip does not use stay 0.
    """
    rot = (-1.0, -0.5, 0.0, 0.5, 1.0)
    breathe = (-1.0, 0.0, 1.0) if fx.get("breathe") else (0.0,)
    bob = (0.0, 1.0) if fx.get("bob") else (0.0,)
    hop = (0.0, 1.0) if fx.get("hop") else (0.0,)
    shake = (-1.0, 0.0, 1.0) if fx.get("shake") else (0.0,)
    tilt = rot if fx.get("tilt") else (0.0,)
    swing = rot if fx.get("swing") else (0.0,)
    return product(breathe, bob, hop, shake, tilt, swing)


def _fx_bounds(box: Extents, fx: Mapping[str, float], height: float, pivot_y: float) -> Extents:
    """Bounding extents of *box* under every transform the clip's effects can apply."""
    corners = (QPointF(-box.left, -box.up), QPointF(box.right, -box.up),
               QPointF(-box.left, box.down), QPointF(box.right, box.down))
    left = right = up = down = -math.inf
    for values in _fx_extremes(fx):
        tr = _fx_matrix(fx, *values, height=height, pivot_y=pivot_y)
        for corner in corners:
            p = tr.map(corner)
            left = max(left, -p.x())
            right = max(right, p.x())
            up = max(up, -p.y())
            down = max(down, p.y())
    return Extents(left, right, up, down)


def needs_mirror(art: Facing, facing: Facing) -> bool:
    """Art drawn looking one way is mirrored to look the other way.  Front
    art never mirrors (her hair clip would jump sides for nothing)."""
    return art is not Facing.FRONT and facing is not Facing.FRONT and art is not facing


def _device_ratio(painter: QPainter) -> float:
    device = painter.device()
    try:
        dpr = float(device.devicePixelRatioF()) if device is not None else 1.0
    except (AttributeError, RuntimeError):
        dpr = 1.0
    return dpr if math.isfinite(dpr) and dpr > 0 else 1.0


# -- sprite renderer ---------------------------------------------------------------------------

#: Head centre below the top of a frame, as a fraction of the figure height
#: (frames are trimmed, so their top edge is the top of her hair).
HEAD_CENTER_FROM_TOP = 0.30


class _PixmapLRU:
    """Least-recently-used pixmaps bounded by count *and* bytes: 64 entries
    are ~12 MB at 1x but would be ~300 MB at scale 2.5 on a 200 % display."""

    def __init__(self, max_entries: int, max_bytes: int) -> None:
        self._items: OrderedDict[tuple, QPixmap] = OrderedDict()
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._bytes = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def bytes(self) -> int:
        return self._bytes

    def get(self, key: tuple) -> QPixmap | None:
        pixmap = self._items.get(key)
        if pixmap is not None:
            self._items.move_to_end(key)
        return pixmap

    def put(self, key: tuple, pixmap: QPixmap) -> None:
        old = self._items.pop(key, None)
        if old is not None:
            self._bytes -= self._cost(old)
        self._items[key] = pixmap
        self._bytes += self._cost(pixmap)
        while len(self._items) > 1 and (len(self._items) > self._max_entries
                                        or self._bytes > self._max_bytes):
            _, evicted = self._items.popitem(last=False)
            self._bytes -= self._cost(evicted)

    def clear(self) -> None:
        self._items.clear()
        self._bytes = 0

    @staticmethod
    def _cost(pixmap: QPixmap) -> int:
        return pixmap.width() * pixmap.height() * 4


class SpriteRenderer:
    """Draws frames of a :class:`CharacterPack`.

    Display factor = ``height_px / pack.reference_height * frame.scale``.
    PNGs are decoded lazily; a frame that does not decode is drawn by the
    vector fallback instead (logged once), so a damaged pack degrades to a
    working character rather than an invisible one.
    """

    PIXMAP_CACHE_ENTRIES = 64
    PIXMAP_CACHE_BYTES = 96 * 1024 * 1024
    #: Decoded source images kept around (a cross-fade needs two at once).
    SOURCE_CACHE_ENTRIES = 6

    def __init__(self, pack: CharacterPack, fallback: VectorRenderer | None = None) -> None:
        self._pack = pack
        self._fallback = fallback or VectorRenderer()
        self._pixmaps = _PixmapLRU(self.PIXMAP_CACHE_ENTRIES, self.PIXMAP_CACHE_BYTES)
        self._sources: OrderedDict[str, QImage] = OrderedDict()
        self._sizes: dict[str, tuple[int, int]] = {}
        self._broken: set[str] = set()
        self._unit_extents: Extents | None = None
        self._unit_head: Extents | None = None

    @property
    def pack(self) -> CharacterPack:
        return self._pack

    @property
    def cached_pixmaps(self) -> int:
        return len(self._pixmaps)

    def invalidate(self) -> None:
        self._pixmaps.clear()
        self._sources.clear()

    # -- geometry --------------------------------------------------------------------

    def _frame_size(self, frame_id: str) -> tuple[int, int]:
        """Image size from the PNG header (no decode).  A file whose header
        cannot be read is marked broken and sized from its anchor."""
        size = self._sizes.get(frame_id)
        if size is not None:
            return size
        info = self._pack.frames[frame_id]
        qsize = QImageReader(str(info.path)).size()
        if qsize.isValid() and qsize.width() > 0 and qsize.height() > 0:
            size = (qsize.width(), qsize.height())
        else:
            self._mark_broken(frame_id, "unreadable image header")
            size = (max(1, 2 * info.anchor[0]), max(1, info.anchor[1]))
        self._sizes[frame_id] = size
        return size

    def _frame_box(self, frame_id: str, height: float) -> tuple[Extents, float]:
        """Unmirrored extents of one frame and its head-top pivot (both px)."""
        info = self._pack.frames[frame_id]
        w, h = self._frame_size(frame_id)
        k = height / self._pack.reference_height * info.scale
        ax, ay = info.anchor
        return Extents(ax * k, (w - ax) * k, ay * k, (h - ay) * k), -ay * k

    def _compute_unit(self) -> None:
        extents = Extents(0.0, 0.0, 0.0, 0.0)
        head = Extents(-math.inf, -math.inf, -math.inf, -math.inf)
        for clip in self._pack.clips.values():
            mirrorable = clip.facing is not Facing.FRONT
            for frame_id in {f.frame_id for f in clip.frames}:
                box, pivot = self._frame_box(frame_id, 1.0)
                if mirrorable:
                    side = max(box.left, box.right)
                    box = Extents(side, side, box.up, box.down)
                extents = extents.union(_fx_bounds(box, clip.fx, 1.0, pivot))
                dx, dy = self._head_offset(frame_id, 1.0)
                xs = (dx, -dx) if mirrorable else (dx,)
                for x in xs:
                    head = head.union(Extents(-x, x, -dy, dy))
        self._unit_extents = extents
        self._unit_head = head

    def full_extents(self, height_px: float) -> Extents:
        if self._unit_extents is None:
            self._compute_unit()
        assert self._unit_extents is not None
        return self._unit_extents.scaled(height_px)

    def extents(self, height_px: float) -> tuple[float, float, float]:
        e = self.full_extents(height_px)
        return e.left, e.right, e.up

    def head_bounds(self, height_px: float) -> Extents:
        if self._unit_head is None:
            self._compute_unit()
        assert self._unit_head is not None
        return self._unit_head.scaled(height_px)

    def _head_offset(self, frame_id: str, height: float) -> tuple[float, float]:
        info = self._pack.frames[frame_id]
        w, _h = self._frame_size(frame_id)
        k = height / self._pack.reference_height * info.scale
        ax, ay = info.anchor
        return (w / 2.0 - ax) * k, -ay * k + HEAD_CENTER_FROM_TOP * height * info.scale

    def _resolve(self, visual: VisualState) -> tuple[Clip, FrameSample]:
        clip = visual.clip
        if clip is None or not clip.frames or clip.frames[0].frame_id not in self._pack.frames:
            # no clip, or one from another source (e.g. a vector clip): use the pack's own
            clip = self._pack.clip_for(visual.state)
        sample = visual.frame
        if sample is None or sample.frame_id not in self._pack.frames:
            sample = FrameSample(clip.frames[0].frame_id, None, 0.0, False)
        return clip, sample

    def head_point(self, anchor: QPointF, height_px: float, visual: VisualState) -> QPointF:
        clip, sample = self._resolve(visual)
        dx, dy = self._head_offset(sample.frame_id, height_px)
        if needs_mirror(clip.facing, visual.facing):
            dx = -dx
        return QPointF(anchor.x() + dx, anchor.y() + dy)

    # -- painting --------------------------------------------------------------------

    def paint(self, painter: QPainter, anchor: QPointF, height_px: float, visual: VisualState) -> None:
        if not (math.isfinite(height_px) and height_px > 0):
            return
        clip, sample = self._resolve(visual)
        mirrored = needs_mirror(clip.facing, visual.facing)
        dpr = _device_ratio(painter)
        base = self._pixmap(sample.frame_id, height_px, mirrored, dpr)
        if base is None:
            self._fallback.paint(painter, anchor, height_px, visual)
            return
        nxt = None
        next_id = sample.next_frame_id
        if next_id is not None and sample.blend > 0.0 and next_id in self._pack.frames:
            nxt = self._pixmap(next_id, height_px, mirrored, dpr)

        info = self._pack.frames[sample.frame_id]
        pivot = -info.anchor[1] * height_px / self._pack.reference_height * info.scale
        tr = fx_transform(clip.fx, visual.t, height_px, pivot)
        painter.save()
        try:
            painter.translate(anchor)
            snap = tr.isIdentity()
            if not snap:
                painter.setTransform(tr, True)
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            self._blit(painter, sample.frame_id, base, height_px, mirrored, dpr, anchor if snap else None)
            if nxt is not None:
                # Current frame stays opaque and the next one fades in on top:
                # a symmetric cross-fade would dip the translucent window's alpha.
                painter.setOpacity(painter.opacity() * min(1.0, sample.blend))
                self._blit(painter, next_id, nxt, height_px, mirrored, dpr, anchor if snap else None)
        finally:
            painter.restore()

    def _blit(self, painter: QPainter, frame_id: str, pixmap: QPixmap, height: float, mirrored: bool,
              dpr: float, snap_anchor: QPointF | None) -> None:
        info = self._pack.frames[frame_id]
        w, _h = self._frame_size(frame_id)
        k = height / self._pack.reference_height * info.scale
        ax, ay = info.anchor
        left = -(w - ax) * k if mirrored else -ax * k
        top = -ay * k
        if snap_anchor is not None:
            # Untransformed: land on whole device pixels so the blit is 1:1 (crisp).
            gx = math.floor((snap_anchor.x() + left) * dpr + 0.5) / dpr
            gy = math.floor((snap_anchor.y() + top) * dpr + 0.5) / dpr
            left, top = gx - snap_anchor.x(), gy - snap_anchor.y()
        painter.drawPixmap(QPointF(left, top), pixmap)

    def _pixmap(self, frame_id: str, height: float, mirrored: bool, dpr: float) -> QPixmap | None:
        if frame_id in self._broken or frame_id not in self._pack.frames:
            return None
        info = self._pack.frames[frame_id]
        w, h = self._frame_size(frame_id)
        if frame_id in self._broken:
            return None
        k = height / self._pack.reference_height * info.scale * dpr
        w_dev, h_dev = max(1, round(w * k)), max(1, round(h * k))
        key = (frame_id, w_dev, h_dev, mirrored, round(dpr, 3))
        pixmap = self._pixmaps.get(key)
        if pixmap is not None:
            return pixmap
        source = self._source(frame_id)
        if source is None:
            return None
        image = source.scaled(w_dev, h_dev, Qt.AspectRatioMode.IgnoreAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        if mirrored:
            image = image.transformed(QTransform.fromScale(-1.0, 1.0))
        pixmap = QPixmap.fromImage(image)
        pixmap.setDevicePixelRatio(dpr)
        self._pixmaps.put(key, pixmap)
        return pixmap

    def _source(self, frame_id: str) -> QImage | None:
        image = self._sources.get(frame_id)
        if image is not None:
            self._sources.move_to_end(frame_id)
            return image
        path: Path = self._pack.frames[frame_id].path
        image = QImage(str(path))
        if image.isNull():
            self._mark_broken(frame_id, "image does not decode")
            return None
        image = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
        self._sources[frame_id] = image
        while len(self._sources) > self.SOURCE_CACHE_ENTRIES:
            self._sources.popitem(last=False)
        return image

    def _mark_broken(self, frame_id: str, why: str) -> None:
        if frame_id not in self._broken:
            self._broken.add(frame_id)
            log.warning("character frame %r unusable (%s); drawing the vector fallback for it", frame_id, why)


# -- vector renderer ---------------------------------------------------------------------------

# Palette (original design: dark-brown wavy hair, black oversized cat hoodie).
_HAIR = QColor(74, 44, 34)
_HAIR_DARK = QColor(46, 27, 21)
_HAIR_LIGHT = QColor(125, 84, 64)
_HAIR_LINE = QColor(38, 22, 17)
_SKIN = QColor(255, 228, 212)
_SKIN_SHADE = QColor(244, 196, 178)
_SKIN_LINE = QColor(196, 140, 120)
_HOODIE = QColor(34, 31, 38)
_HOODIE_LIGHT = QColor(58, 53, 64)
_HOODIE_LINE = QColor(14, 12, 17)
_CAT_PRINT = QColor(150, 120, 104)
_CAT_DARK = QColor(70, 50, 42)
_IRIS_TOP = QColor(58, 32, 20)
_IRIS_MID = QColor(122, 70, 40)
_IRIS_BOTTOM = QColor(204, 142, 86)
_PUPIL = QColor(34, 17, 10)
_LASH = QColor(38, 20, 13)
_BROW = QColor(58, 34, 25, 210)
_MOUTH_LINE = QColor(140, 58, 58)
_MOUTH_INNER = QColor(150, 44, 58)
_TONGUE = QColor(255, 138, 140)
_BLUSH = QColor(255, 130, 150)
_CLIP_WHITE = QColor(255, 250, 245)
_CLIP_LINE = QColor(205, 185, 173)
_PILLOW = QColor(248, 236, 224)
_PILLOW_LINE = QColor(214, 196, 182)
_WHITE = QColor(255, 255, 255)

#: Unit box (height = 100) that contains every vector pose before effects:
#: the lying pose is the widest, the ahoge the highest point.
_VECTOR_BOX = Extents(52.0, 52.0, 101.5, 2.0)
#: Head centres: standing, sitting, lying (x mirrors).
_HEAD_STAND = QPointF(0.0, -69.0)
_SIT_DROP = 9.0
_HEAD_SLEEP = QPointF(-17.0, -26.0)
_RUN_LEAN_DEG = 7.0


def _vclip(name: str, fx: Mapping[str, float] | None = None, overlay: str | None = None,
           loop: bool = True, ms: int = 1000) -> Clip:
    return Clip(name=name, frames=(ClipFrame("vector", ms),), loop=loop, facing=Facing.FRONT,
                fx=MappingProxyType(dict(fx or {})), overlay=overlay)


#: Clips the controller plays when the vector renderer is active (one dummy
#: frame each: the drawing is procedural; ``loop=False`` + ``ms`` make the
#: fidget variants end like the pack's do).
VECTOR_CLIPS: Mapping[str, Clip] = MappingProxyType({
    "idle": _vclip("idle", {"breathe": 0.012}),
    "idle_look": _vclip("idle_look", {"breathe": 0.012}, loop=False, ms=2400),
    "idle_smile": _vclip("idle_smile", {"breathe": 0.012}, loop=False, ms=3100),
    "walk": _vclip("walk", {"bob": 0.012}),
    "run": _vclip("run", {"bob": 0.018}),
    "sit": _vclip("sit", {"breathe": 0.01}),
    "sleep": _vclip("sleep", {"breathe": 0.018}, "zzz"),
    "gaming": _vclip("gaming", {"breathe": 0.008}, "game"),
    "talk": _vclip("talk", {"bob": 0.006, "breathe": 0.008}),
    "laugh": _vclip("laugh", {"shake": 0.006, "bob": 0.01}),
    "happy": _vclip("happy", {"hop": 0.035}, "sparkle"),
    "angry": _vclip("angry", {"shake": 0.01}, "anger"),
    "sad": _vclip("sad", {"breathe": 0.01}, "tear"),
    "surprised": _vclip("surprised", {"hop": 0.03}, "exclaim"),
    "curious": _vclip("curious", {"tilt": 4.0}, "question"),
    "thinking": _vclip("thinking", {"breathe": 0.01}, "dots"),
    "wave": _vclip("wave", {"bob": 0.006}),
    "greet": _vclip("greet", {"hop": 0.02}, loop=False, ms=2000),
    "eat": _vclip("eat", {"bob": 0.01}, "hearts"),
    "shy": _vclip("shy", {"breathe": 0.01}, "blush"),
    "yawn": _vclip("yawn", {"breathe": 0.016}),
    "bye": _vclip("bye", {"bob": 0.006}),
    "point": _vclip("point", None, "exclaim"),
    "dragged": _vclip("dragged", {"swing": 9.0}),
})

#: Fidget variants per state, mirroring the mia pack.
VECTOR_VARIANTS: Mapping[CharacterState, tuple[str, ...]] = MappingProxyType({
    S.IDLE: ("idle_look", "idle_smile"),
    S.WAVE: ("greet",),
})


def vector_clip_for(state: CharacterState, variant: str | None = None) -> Clip:
    """Clip for *state* in vector mode (same contract as ``CharacterPack.clip_for``)."""
    if variant is not None and variant in VECTOR_VARIANTS.get(state, ()):
        return VECTOR_CLIPS[variant]
    return VECTOR_CLIPS.get(state.value, VECTOR_CLIPS["idle"])


@dataclass(frozen=True)
class _Look:
    """One frame's expression and pose for the vector chibi."""

    eyes: str = "open"  # open | happy | closed | wide | half | angry | sad | squeeze | down
    brows: str = "normal"  # normal | angry | sad | raised | curious | none
    mouth: str = "smile"  # smile | cat | open_smile | talk | o | big_o | pout | wavy | flat | chew | tongue
    blush: float = 0.35
    arms: str = "down"  # down | swing | wave | bye | up | held | point | chin | mouth | snack | hold
    #                     | together | crossed | lap | surprise
    legs: str = "stand"  # stand | walk | run | dangle | sit
    pose: str = "stand"  # stand | sit | sleep
    look_x: float = 0.0  # iris offset, -1..1 (positive: toward her facing)
    look_y: float = 0.0
    prop: str | None = None  # gamepad | snack
    sparkle_eyes: bool = False
    talk_open: float = 0.0


def _wave01(t: float, hz: float) -> float:
    return 0.5 + 0.5 * math.sin(2.0 * math.pi * hz * t)


def _blinking(t: float) -> bool:
    """A short blink every ~4 s (offset so she doesn't blink right at t=0)."""
    return math.fmod(t + 1.3, 3.9) < 0.13


def _look_for(state: CharacterState, clip_name: str, t: float) -> _Look:
    blink = _blinking(t)
    open_eyes = "closed" if blink else "open"
    if clip_name == "idle_smile":
        return _Look(eyes="happy" if blink else "open", mouth="cat", blush=0.6)
    if clip_name == "idle_look":
        return _Look(eyes=open_eyes, look_x=1.0)
    if clip_name == "greet":
        return _Look(eyes="happy", mouth="open_smile", arms="wave", blush=0.6)
    table: dict[CharacterState, _Look] = {
        S.IDLE: _Look(eyes=open_eyes),
        S.WALK: _Look(eyes=open_eyes, arms="swing", legs="walk", look_x=0.8),
        S.RUN: _Look(eyes=open_eyes, mouth="open_smile", arms="swing", legs="run", look_x=1.0),
        S.SIT: _Look(eyes=open_eyes, arms="lap", legs="sit", pose="sit"),
        S.SLEEP: _Look(eyes="closed", brows="none", mouth="cat", pose="sleep", blush=0.45),
        S.GAMING: _Look(eyes="half", mouth="tongue", arms="hold", look_y=1.0, prop="gamepad"),
        S.TALK: _Look(eyes=open_eyes, mouth="talk", talk_open=_wave01(t, 4.2)),
        S.LAUGH: _Look(eyes="happy", mouth="open_smile", arms="together", blush=0.8),
        S.HAPPY: _Look(eyes=open_eyes, mouth="open_smile", arms="up", blush=0.7, sparkle_eyes=True),
        S.ANGRY: _Look(eyes="angry", brows="angry", mouth="pout", arms="crossed", blush=0.55),
        S.SAD: _Look(eyes="sad", brows="sad", mouth="wavy", arms="together", look_y=1.0),
        S.SURPRISED: _Look(eyes="wide", brows="raised", mouth="o", arms="surprise"),
        S.CURIOUS: _Look(eyes=open_eyes, brows="curious", mouth="o", look_x=1.0),
        S.THINKING: _Look(eyes=open_eyes, mouth="flat", arms="chin", look_x=0.8, look_y=-1.0),
        S.WAVE: _Look(eyes=open_eyes, mouth="open_smile", arms="wave", blush=0.5),
        S.EAT: _Look(eyes="happy", mouth="chew", arms="snack", blush=0.6, prop="snack"),
        S.SHY: _Look(eyes="down", brows="sad", mouth="wavy", arms="together", blush=1.0, look_x=-0.6),
        S.YAWN: _Look(eyes="squeeze", brows="sad", mouth="big_o", arms="mouth"),
        S.BYE: _Look(eyes=open_eyes, brows="sad", mouth="smile", arms="bye"),
        S.POINT: _Look(eyes=open_eyes, brows="raised", mouth="open_smile", arms="point", look_x=1.0),
        S.DRAGGED: _Look(eyes="squeeze", mouth="wavy", arms="held", legs="dangle", blush=0.55),
    }
    return table.get(state, _Look(eyes=open_eyes))


def _pen(color: QColor, width: float) -> QPen:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _alpha(color: QColor, a: float) -> QColor:
    c = QColor(color)
    c.setAlphaF(max(0.0, min(1.0, a)) * color.alphaF())
    return c


def _union(*paths: QPainterPath) -> QPainterPath:
    out = QPainterPath()
    out.setFillRule(Qt.FillRule.WindingFill)
    for p in paths:
        out.addPath(p)
    return out.simplified()


def _ellipse(cx: float, cy: float, rx: float, ry: float) -> QPainterPath:
    path = QPainterPath()
    path.addEllipse(QPointF(cx, cy), rx, ry)
    return path


def _build_back_hair() -> QPainterPath:
    """Long wavy hair behind her: a central mass plus curls along the edge."""
    parts = [_ellipse(0.0, -70.0, 24.5, 24.0)]
    mass = QPainterPath(QPointF(-23.0, -72.0))
    mass.lineTo(23.0, -72.0)
    mass.quadTo(30.0, -50.0, 29.0, -24.0)
    mass.lineTo(-29.0, -24.0)
    mass.quadTo(-30.0, -50.0, -23.0, -72.0)
    mass.closeSubpath()
    parts.append(mass)
    for side in (-1.0, 1.0):
        for x, y, r in ((25.5, -63.0, 4.8), (28.0, -55.0, 5.4), (29.5, -46.0, 5.8), (31.0, -37.0, 6.0),
                        (29.5, -28.5, 6.0), (24.5, -22.5, 5.6), (17.0, -21.0, 5.0)):
            parts.append(_ellipse(side * x, y, r, r * 1.1))
    return _union(*parts)


def _build_bangs() -> QPainterPath:
    """Hair cap with a pointed fringe and two side locks framing the face."""
    path = QPainterPath(QPointF(-18.6, -60.2))
    path.arcTo(QRectF(-21.5, -92.5, 43.0, 43.0), 210.0, -240.0)
    # right side lock down to the cheek, then the fringe from right to left
    path.quadTo(22.5, -54.0, 21.0, -47.5)
    path.quadTo(17.0, -53.0, 16.4, -60.0)
    fringe = ((15.0, -70.5), (11.5, -77.0), (8.0, -71.0), (4.0, -77.5), (0.5, -71.8), (-3.5, -77.5),
              (-7.5, -71.0), (-11.0, -77.0), (-14.8, -70.8))
    prev = QPointF(16.4, -60.0)
    for i, (x, y) in enumerate(fringe):
        point = QPointF(x, y)
        tip = i % 2 == 0
        # Strands curve: towards a tip the control sits above it, back up to a notch it bends outwards.
        ctrl = QPointF(prev.x() * 0.4 + x * 0.6, y - 4.0) if tip else QPointF(prev.x(), (prev.y() + y) / 2)
        path.quadTo(ctrl, point)
        prev = point
    path.quadTo(-16.5, -66.0, -16.4, -60.0)
    path.quadTo(-17.0, -53.0, -21.0, -47.5)
    path.quadTo(-22.5, -54.0, -18.6, -60.2)
    path.closeSubpath()
    return path


def _build_body() -> QPainterPath:
    """The oversized hoodie from the shoulders to the hem."""
    path = QPainterPath(QPointF(-11.0, -49.0))
    path.quadTo(0.0, -51.5, 11.0, -49.0)
    path.quadTo(15.5, -46.5, 16.0, -38.0)
    path.lineTo(18.0, -17.0)
    path.quadTo(18.5, -12.0, 13.0, -12.0)
    path.lineTo(-13.0, -12.0)
    path.quadTo(-18.5, -12.0, -18.0, -17.0)
    path.lineTo(-16.0, -38.0)
    path.quadTo(-15.5, -46.5, -11.0, -49.0)
    path.closeSubpath()
    return path


def _build_hair_clip() -> QPainterPath:
    """The cat-face hair clip's outline (head + ears), centred on the origin."""
    head = QPainterPath()
    head.addEllipse(QPointF(0.0, 0.0), 4.6, 4.6 * 0.9)
    for side in (-1.0, 1.0):
        ear = QPainterPath(QPointF(side * 4.2, -1.0))
        ear.lineTo(side * 3.6, -6.4)
        ear.lineTo(side * 0.9, -3.6)
        ear.closeSubpath()
        head.addPath(ear)
    return head.simplified()


def _build_sleep_hair() -> QPainterPath:
    """Hair spread around her head and over the pillow while she sleeps."""
    return _union(
        _ellipse(-17.0, -28.0, 22.0, 20.0),
        _ellipse(-35.0, -14.0, 7.5, 7.0), _ellipse(-38.0, -26.0, 7.0, 7.0),
        _ellipse(-32.0, -39.0, 7.0, 7.0), _ellipse(-20.0, -46.0, 7.0, 6.0),
        _ellipse(-3.0, -12.5, 14.0, 7.5), _ellipse(9.0, -9.5, 9.0, 5.5),
        _ellipse(-22.0, -10.0, 14.0, 6.0),
    )


def _build_gamepad_body() -> QPainterPath:
    body = QPainterPath()
    body.addRoundedRect(QRectF(-8.5, -3.6, 17.0, 7.2), 3.4, 3.4)
    body.addEllipse(QPointF(-6.0, 2.4), 3.4, 3.4)
    body.addEllipse(QPointF(6.0, 2.4), 3.4, 3.4)
    return body.simplified()


class VectorRenderer:
    """An original chibi girl drawn entirely with QPainter paths.

    Coordinates are designed in units where her height is 100, feet at the
    origin; the painter is scaled per call.  The base drawing looks to the
    *right* when facing RIGHT and is mirrored for LEFT; FRONT looks ahead.
    Paths that never change are built once.
    """

    def __init__(self) -> None:
        self._back_hair = _build_back_hair()
        self._bangs = _build_bangs()
        self._body = _build_body()
        self._clip_shape = _build_hair_clip()
        self._sleep_hair = _build_sleep_hair()
        self._gamepad_body = _build_gamepad_body()

    def invalidate(self) -> None:
        """Nothing is cached per size or screen."""

    # -- geometry --------------------------------------------------------------------

    @staticmethod
    def _clip_of(visual: VisualState) -> Clip:
        return visual.clip or vector_clip_for(visual.state)

    def full_extents(self, height_px: float) -> Extents:
        unit = Extents(0.0, 0.0, 0.0, 0.0)
        box = _VECTOR_BOX.scaled(0.01)
        fx_seen: set[tuple[tuple[str, float], ...]] = set()
        for clip in VECTOR_CLIPS.values():
            key = tuple(sorted(clip.fx.items()))
            if key in fx_seen:
                continue
            fx_seen.add(key)
            unit = unit.union(_fx_bounds(box, clip.fx, 1.0, -1.0))
        return unit.scaled(height_px)

    def extents(self, height_px: float) -> tuple[float, float, float]:
        e = self.full_extents(height_px)
        return e.left, e.right, e.up

    def head_bounds(self, height_px: float) -> Extents:
        lean = abs(_HEAD_STAND.y()) * math.sin(math.radians(_RUN_LEAN_DEG))
        side = max(abs(_HEAD_SLEEP.x()), lean)
        up = abs(_HEAD_STAND.y())
        down = _HEAD_SLEEP.y()  # negative: always above the feet
        return Extents(side, side, up, down).scaled(height_px / 100.0)

    def _pose_transform(self, look: _Look, facing: Facing) -> QTransform:
        """Unit-space transform for mirroring, the run lean and sitting."""
        tr = QTransform()
        if facing is Facing.LEFT:
            tr.scale(-1.0, 1.0)
        if look.legs == "run":
            tr.rotate(_RUN_LEAN_DEG)
        return tr

    def head_point(self, anchor: QPointF, height_px: float, visual: VisualState) -> QPointF:
        clip = self._clip_of(visual)
        look = _look_for(visual.state, clip.name, visual.t)
        if look.pose == "sleep":
            head = _HEAD_SLEEP
        elif look.pose == "sit":
            head = QPointF(_HEAD_STAND.x(), _HEAD_STAND.y() + _SIT_DROP)
        else:
            head = _HEAD_STAND
        p = self._pose_transform(look, visual.facing).map(head)
        k = height_px / 100.0
        return QPointF(anchor.x() + p.x() * k, anchor.y() + p.y() * k)

    # -- painting --------------------------------------------------------------------

    def paint(self, painter: QPainter, anchor: QPointF, height_px: float, visual: VisualState) -> None:
        if not (math.isfinite(height_px) and height_px > 0):
            return
        clip = self._clip_of(visual)
        t = visual.t if math.isfinite(visual.t) else 0.0
        look = _look_for(visual.state, clip.name, t)
        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.translate(anchor)
            pivot = -height_px if look.pose != "sleep" else -0.5 * height_px
            tr = fx_transform(clip.fx, t, height_px, pivot)
            if not tr.isIdentity():
                painter.setTransform(tr, True)
            painter.scale(height_px / 100.0, height_px / 100.0)
            painter.setTransform(self._pose_transform(look, visual.facing), True)
            front = visual.facing is Facing.FRONT
            if look.pose == "sleep":
                self._paint_sleeping(painter, look, t)
            else:
                self._paint_standing(painter, look, t, front)
        finally:
            painter.restore()

    # standing / sitting ------------------------------------------------------------

    def _paint_standing(self, p: QPainter, look: _Look, t: float, front: bool) -> None:
        sitting = look.pose == "sit"
        walk_phase = 2.0 * math.pi * t * (2.6 if look.legs == "run" else 1.8)
        sway = 0.9 * math.sin(2.0 * math.pi * t * 0.55)
        if look.legs in ("walk", "run"):
            sway += 1.6 * math.sin(walk_phase)
        look_x = 0.0 if front and look.look_x > 0 else look.look_x

        if not sitting:
            self._legs(p, look.legs, walk_phase, t)
        p.save()
        if sitting:
            p.translate(0.0, _SIT_DROP)
        self._hair_back(p, sway)
        self._hood(p)
        self._torso(p)
        if look.arms == "crossed":
            self._arms(p, look.arms, t, walk_phase)
        self._head(p, look, t, look_x)
        if look.arms != "crossed":
            self._arms(p, look.arms, t, walk_phase)
        if look.prop == "gamepad":
            self._gamepad(p, QPointF(0.0, -33.5))
        p.restore()
        if sitting:
            # her feet stick out in front of the hoodie hem when she sits
            self._feet_sitting(p)

    def _hair_back(self, p: QPainter, sway: float) -> None:
        # The hair hangs from the head: shear so the ends sway, the crown stays put.
        shear = QTransform()
        k = sway / 66.0
        shear.translate(k * 90.0, 0.0)
        shear.shear(k, 0.0)
        path = shear.map(self._back_hair)
        grad = QLinearGradient(QPointF(0.0, -95.0), QPointF(0.0, -20.0))
        grad.setColorAt(0.0, _HAIR)
        grad.setColorAt(1.0, _HAIR_DARK)
        p.setPen(_pen(_HAIR_LINE, 0.7))
        p.setBrush(QBrush(grad))
        p.drawPath(path)
        # a few inner strand lines for the wavy texture
        p.setPen(_pen(_alpha(_HAIR_LIGHT, 0.45), 0.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        for side in (-1.0, 1.0):
            strand = QPainterPath(QPointF(side * 22.0, -60.0))
            strand.cubicTo(side * 27.0, -50.0, side * 22.0, -42.0, side * 27.0 + sway * 0.5, -30.0)
            p.drawPath(strand)

    def _hood(self, p: QPainter) -> None:
        p.setPen(_pen(_HOODIE_LINE, 0.6))
        p.setBrush(_HOODIE_LIGHT)
        p.drawEllipse(QPointF(0.0, -48.5), 15.5, 6.0)

    def _legs(self, p: QPainter, mode: str, phase: float, t: float) -> None:
        for side in (-1.0, 1.0):
            x = 5.0 * side
            lift = 0.0
            dx = 0.0
            tilt = 0.0
            if mode in ("walk", "run"):
                amp = 3.4 if mode == "run" else 2.2
                s = math.sin(phase + (0.0 if side > 0 else math.pi))
                dx = amp * s
                lift = (3.6 if mode == "run" else 2.4) * max(0.0, s)
            elif mode == "dangle":
                swing = math.sin(2.0 * math.pi * t * 1.1 + (0.0 if side > 0 else 1.4))
                dx = 1.4 * side + 1.2 * swing
                tilt = 18.0 * side
            # leg (skin) then the sock with its foot
            p.setPen(_pen(_SKIN_LINE, 0.5))
            p.setBrush(_SKIN)
            p.drawRoundedRect(QRectF(x - 2.3 + dx * 0.4, -15.0, 4.6, 7.0 - lift * 0.4), 1.8, 1.8)
            p.save()
            p.translate(x + dx, -lift)
            if tilt:
                p.rotate(tilt)
            p.setPen(_pen(_HOODIE_LINE, 0.6))
            p.setBrush(_HOODIE)
            p.drawRoundedRect(QRectF(-2.8, -9.5, 5.6, 8.5), 2.0, 2.0)
            p.drawEllipse(QPointF(0.6 * side, -1.9), 4.2, 2.3)
            self._mini_cat(p, QPointF(0.2 * side, -6.0), 1.25)
            p.restore()

    def _feet_sitting(self, p: QPainter) -> None:
        for side in (-1.0, 1.0):
            x = 7.5 * side
            p.setPen(_pen(_HOODIE_LINE, 0.6))
            p.setBrush(_HOODIE)
            p.drawEllipse(QPointF(x, -4.4), 4.8, 4.2)
            self._mini_cat(p, QPointF(x, -4.6), 1.5)

    def _mini_cat(self, p: QPainter, c: QPointF, r: float) -> None:
        """Tiny cat face printed on socks."""
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_CAT_PRINT)
        ears = QPainterPath(QPointF(c.x() - r, c.y() - 0.2 * r))
        ears.lineTo(c.x() - 0.75 * r, c.y() - 1.45 * r)
        ears.lineTo(c.x() - 0.1 * r, c.y() - 0.7 * r)
        ears.lineTo(c.x() + 0.1 * r, c.y() - 0.7 * r)
        ears.lineTo(c.x() + 0.75 * r, c.y() - 1.45 * r)
        ears.lineTo(c.x() + r, c.y() - 0.2 * r)
        ears.closeSubpath()
        p.drawPath(ears)
        p.drawEllipse(c, r, 0.85 * r)

    def _torso(self, p: QPainter) -> None:
        grad = QLinearGradient(QPointF(-16.0, -50.0), QPointF(16.0, -12.0))
        grad.setColorAt(0.0, _HOODIE_LIGHT)
        grad.setColorAt(1.0, _HOODIE)
        p.setPen(_pen(_HOODIE_LINE, 0.7))
        p.setBrush(QBrush(grad))
        p.drawPath(self._body)
        # ribbed hem and the cat face printed on the chest
        p.setPen(_pen(_alpha(_HOODIE_LIGHT, 0.9), 0.6))
        p.drawLine(QPointF(-16.5, -15.5), QPointF(16.5, -15.5))
        self._chest_cat(p, QPointF(0.0, -33.0))

    def _chest_cat(self, p: QPainter, c: QPointF) -> None:
        r = 4.4
        self._mini_cat(p, c, r)
        p.setBrush(_CAT_DARK)
        p.drawEllipse(QPointF(c.x() - 1.6, c.y() - 0.4), 0.55, 0.65)
        p.drawEllipse(QPointF(c.x() + 1.6, c.y() - 0.4), 0.55, 0.65)
        p.setPen(_pen(_CAT_DARK, 0.4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        mouth = QPainterPath(QPointF(c.x() - 1.1, c.y() + 1.0))
        mouth.quadTo(c.x() - 0.55, c.y() + 1.7, c.x(), c.y() + 1.0)
        mouth.quadTo(c.x() + 0.55, c.y() + 1.7, c.x() + 1.1, c.y() + 1.0)
        p.drawPath(mouth)

    # arms ------------------------------------------------------------------------------

    def _arm_points(self, mode: str, side: float, t: float, phase: float) -> tuple[QPointF, QPointF] | None:
        """(elbow, hand) for one arm; *side* +1 is her right side in the drawing."""

        def pt(x: float, y: float) -> QPointF:
            return QPointF(side * x, y)

        if mode == "swing":
            s = math.sin(phase + (math.pi if side > 0 else 0.0))
            return pt(15.5, -38.0), pt(16.5 + 1.0 * s, -28.0 - 1.8 * max(0.0, s))
        if mode in ("wave", "bye") and side > 0:
            hz, amp = (2.2, 24.0) if mode == "wave" else (1.4, 16.0)
            a = math.radians(-72.0 + amp * math.sin(2.0 * math.pi * t * hz))
            elbow = pt(22.0, -50.0)
            return elbow, QPointF(elbow.x() + 13.0 * math.cos(a), elbow.y() + 13.0 * math.sin(a))
        if mode == "up":
            return pt(20.5, -52.0), pt(24.5, -63.5)
        if mode == "held":
            return pt(17.5, -61.0), pt(15.5, -75.0)
        if mode == "surprise":
            return pt(19.5, -40.0), pt(20.0, -52.0)
        if mode == "point" and side > 0:
            return pt(21.0, -43.5), pt(31.5, -45.0)
        if mode == "chin" and side > 0:
            return pt(17.0, -36.0), pt(7.5, -48.5)
        if mode == "mouth" and side > 0:
            return pt(18.0, -38.0), pt(6.8, -50.5)
        if mode == "snack" and side > 0:
            return pt(17.0, -37.0), pt(8.0, -48.0)
        if mode == "hold":
            return pt(14.5, -35.0), pt(6.5, -33.0)
        if mode == "together":
            return pt(14.0, -35.0), pt(3.4, -30.5)
        if mode == "crossed":
            return pt(14.5, -37.0), pt(-7.5, -38.5)
        if mode == "lap":
            return pt(15.0, -33.0), pt(7.0, -24.0)
        return pt(15.0, -38.0), pt(16.5, -27.5)

    def _arms(self, p: QPainter, mode: str, t: float, phase: float) -> None:
        # Left arm first: in "crossed" the right arm lies on top.
        for side in (-1.0, 1.0):
            points = self._arm_points(mode, side, t, phase)
            if points is None:
                continue
            elbow, hand = points
            shoulder = QPointF(side * 11.5, -46.0)
            sleeve = QPainterPath(shoulder)
            sleeve.quadTo(elbow, hand)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(_pen(_HOODIE_LINE, 8.6))
            p.drawPath(sleeve)
            p.setPen(_pen(_HOODIE, 7.2))
            p.drawPath(sleeve)
            # sweater paw: the cuff swallows the hand, only fingertips peek out
            p.setPen(_pen(_SKIN_LINE, 0.4))
            p.setBrush(_SKIN)
            direction = QPointF(hand.x() - elbow.x(), hand.y() - elbow.y())
            length = math.hypot(direction.x(), direction.y()) or 1.0
            tip = QPointF(hand.x() + direction.x() / length * 3.4, hand.y() + direction.y() / length * 3.4)
            p.drawEllipse(tip, 2.0, 2.0)
            if mode == "point" and side > 0:
                p.drawEllipse(QPointF(tip.x() + 2.4, tip.y() - 0.6), 1.9, 1.1)
            p.setPen(_pen(_HOODIE_LINE, 0.6))
            p.setBrush(_HOODIE)
            p.drawEllipse(hand, 3.9, 3.9)
            p.setPen(_pen(_alpha(_HOODIE_LIGHT, 0.9), 0.6))
            p.drawLine(QPointF(hand.x() - 2.6, hand.y() + 1.2), QPointF(hand.x() + 2.6, hand.y() + 1.2))
            if mode == "snack" and side > 0:
                self._onigiri(p, QPointF(hand.x() - 3.5, hand.y() - 3.0))

    def _gamepad(self, p: QPainter, c: QPointF) -> None:
        p.setPen(_pen(QColor(63, 55, 201), 0.6))
        p.setBrush(QColor(108, 99, 255))
        p.drawPath(self._gamepad_body.translated(c))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_WHITE)
        p.drawRect(QRectF(c.x() - 6.6, c.y() - 0.6, 3.6, 1.2))
        p.drawRect(QRectF(c.x() - 5.4, c.y() - 1.8, 1.2, 3.6))
        p.setBrush(QColor(255, 209, 102))
        p.drawEllipse(QPointF(c.x() + 4.0, c.y() - 0.8), 1.0, 1.0)
        p.setBrush(QColor(239, 71, 111))
        p.drawEllipse(QPointF(c.x() + 6.0, c.y() + 0.8), 1.0, 1.0)

    def _onigiri(self, p: QPainter, c: QPointF) -> None:
        tri = QPainterPath(QPointF(c.x(), c.y() - 4.2))
        tri.quadTo(c.x() + 1.2, c.y() - 4.6, c.x() + 4.0, c.y() + 2.4)
        tri.quadTo(c.x() + 4.2, c.y() + 3.6, c.x(), c.y() + 3.4)
        tri.quadTo(c.x() - 4.2, c.y() + 3.6, c.x() - 4.0, c.y() + 2.4)
        tri.quadTo(c.x() - 1.2, c.y() - 4.6, c.x(), c.y() - 4.2)
        p.setPen(_pen(QColor(190, 180, 170), 0.5))
        p.setBrush(_WHITE)
        p.drawPath(tri)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(40, 60, 45))
        p.drawRect(QRectF(c.x() - 1.8, c.y() + 0.6, 3.6, 2.9))

    # head ------------------------------------------------------------------------------

    def _head(self, p: QPainter, look: _Look, t: float, look_x: float) -> None:
        # neck and face
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_SKIN_SHADE)
        p.drawRect(QRectF(-3.4, -51.0, 6.8, 4.0))
        face = QRadialGradient(QPointF(0.0, -62.0), 22.0)
        face.setColorAt(0.0, _SKIN)
        face.setColorAt(0.85, _SKIN)
        face.setColorAt(1.0, _SKIN_SHADE)
        p.setPen(_pen(_SKIN_LINE, 0.5))
        p.setBrush(QBrush(face))
        p.drawEllipse(QPointF(0.0, -65.0), 18.6, 17.2)
        self._blush(p, look.blush)
        for side in (-1.0, 1.0):
            self._eye(p, look, side, look_x, t)
        self._mouth(p, look, t)
        self._hair_front(p)
        self._brows(p, look.brows)
        self._hair_clip(p, QPointF(15.0, -82.5))

    def _hair_front(self, p: QPainter) -> None:
        grad = QLinearGradient(QPointF(0.0, -93.0), QPointF(0.0, -60.0))
        grad.setColorAt(0.0, _HAIR_LIGHT)
        grad.setColorAt(0.35, _HAIR)
        grad.setColorAt(1.0, _HAIR_DARK)
        p.setPen(_pen(_HAIR_LINE, 0.6))
        p.setBrush(QBrush(grad))
        p.drawPath(self._bangs)
        # glossy "angel ring" and the ahoge
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(_pen(_alpha(_HAIR_LIGHT, 0.7), 1.4))
        ring = QPainterPath(QPointF(-11.5, -83.0))
        ring.quadTo(0.0, -88.5, 11.5, -83.0)
        p.drawPath(ring)
        p.setPen(_pen(_HAIR, 1.5))
        ahoge = QPainterPath(QPointF(1.0, -91.5))
        ahoge.cubicTo(1.5, -97.5, -6.5, -100.5, -4.5, -95.5)
        p.drawPath(ahoge)

    def _hair_clip(self, p: QPainter, c: QPointF) -> None:
        """White cat-face hair clip."""
        p.setPen(_pen(_CLIP_LINE, 0.55))
        p.setBrush(_CLIP_WHITE)
        p.drawPath(self._clip_shape.translated(c))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_CAT_DARK)
        p.drawEllipse(QPointF(c.x() - 1.6, c.y() - 0.2), 0.55, 0.7)
        p.drawEllipse(QPointF(c.x() + 1.6, c.y() - 0.2), 0.55, 0.7)
        p.setBrush(_alpha(_BLUSH, 0.8))
        p.drawEllipse(QPointF(c.x() - 2.9, c.y() + 1.3), 0.9, 0.55)
        p.drawEllipse(QPointF(c.x() + 2.9, c.y() + 1.3), 0.9, 0.55)
        p.setPen(_pen(_CAT_DARK, 0.35))
        p.setBrush(Qt.BrushStyle.NoBrush)
        mouth = QPainterPath(QPointF(c.x() - 0.9, c.y() + 1.0))
        mouth.quadTo(c.x() - 0.45, c.y() + 1.6, c.x(), c.y() + 1.0)
        mouth.quadTo(c.x() + 0.45, c.y() + 1.6, c.x() + 0.9, c.y() + 1.0)
        p.drawPath(mouth)

    def _blush(self, p: QPainter, strength: float) -> None:
        if strength <= 0:
            return
        for side in (-1.0, 1.0):
            c = QPointF(side * 12.6, -57.5)
            grad = QRadialGradient(c, 4.6)
            grad.setColorAt(0.0, _alpha(_BLUSH, 0.85 * strength))
            grad.setColorAt(1.0, _alpha(_BLUSH, 0.0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(grad))
            p.drawEllipse(c, 4.6, 2.8)

    def _eye(self, p: QPainter, look: _Look, side: float, look_x: float, t: float) -> None:
        cx, cy = side * 8.6, -63.0
        mode = look.eyes
        if mode in ("happy", "closed", "squeeze"):
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(_pen(_LASH, 1.35))
            path = QPainterPath()
            if mode == "happy":  # ^ ^
                path.moveTo(cx - 4.6, cy + 1.2)
                path.quadTo(cx, cy - 5.0, cx + 4.6, cy + 1.2)
            elif mode == "closed":  # peaceful curves with lashes
                path.moveTo(cx - 4.8, cy - 0.6)
                path.quadTo(cx, cy + 3.6, cx + 4.8, cy - 0.6)
                outer = cx + side * 4.8
                path.moveTo(outer, cy - 0.6)
                path.lineTo(outer + side * 1.3, cy - 1.6)
            else:  # > <
                inner, outer = cx - side * 3.6, cx + side * 3.8
                path.moveTo(outer, cy - 3.6)
                path.lineTo(inner, cy)
                path.lineTo(outer, cy + 3.4)
            p.drawPath(path)
            return

        wide = mode == "wide"
        rx, ry = (5.4, 6.6) if wide else (4.9, 6.0)
        ix = cx + look_x * 1.1
        iy = cy + look.look_y * 0.9
        if wide:
            # white of the eye around a smaller iris: surprise reads instantly
            p.setPen(_pen(_LASH, 0.5))
            p.setBrush(_WHITE)
            p.drawEllipse(QPointF(cx, cy), rx, ry)
        iris = QLinearGradient(QPointF(ix, iy - ry), QPointF(ix, iy + ry))
        iris.setColorAt(0.0, _IRIS_TOP)
        iris.setColorAt(0.5, _IRIS_MID)
        iris.setColorAt(1.0, _IRIS_BOTTOM)
        irx, iry = (3.3, 4.2) if wide else (rx, ry)
        p.save()
        if not wide:
            clip = QPainterPath()
            clip.addEllipse(QPointF(cx, cy), rx + 0.3, ry + 0.3)
            p.setClipPath(clip, Qt.ClipOperation.IntersectClip)
        p.setPen(_pen(_LASH, 0.45))
        p.setBrush(QBrush(iris))
        p.drawEllipse(QPointF(ix, iy), irx, iry)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_PUPIL)
        p.drawEllipse(QPointF(ix, iy + 0.5), irx * 0.47, iry * 0.5)
        p.setBrush(_WHITE)
        p.drawEllipse(QPointF(ix + 1.5, iy - 2.4), 1.7 if not wide else 1.2, 1.7 if not wide else 1.2)
        p.setBrush(_alpha(_WHITE, 0.85))
        p.drawEllipse(QPointF(ix - 1.7, iy + 2.5), 0.85, 0.85)
        if look.sparkle_eyes:
            star = QPainterPath(QPointF(ix - 1.4, iy - 3.6))
            for dx, dy in ((0.0, 1.6), (1.6, 0.0), (0.0, -1.6), (-1.6, 0.0)):
                star.quadTo(ix - 1.4, iy - 2.0, ix - 1.4 + dx, iy - 2.0 + dy)
            p.drawPath(star)
        p.restore()

        # eyelids for half-closed moods: skin over the top of the eye, lash on the lid edge
        inner_x, outer_x = cx - side * (rx + 0.6), cx + side * (rx + 0.6)
        lid: tuple[float, float] | None = None
        if mode == "half":
            lid = (cy - 1.6, cy - 1.6)
        elif mode == "angry":
            lid = (cy - 0.6, cy - 4.4)
        elif mode == "sad":
            lid = (cy - 4.6, cy - 1.0)
        elif mode == "down":
            lid = (cy - 2.6, cy - 2.6)
        if lid is not None:
            inner_y, outer_y = lid
            cover = QPainterPath(QPointF(inner_x, cy - ry - 2.5))
            cover.lineTo(outer_x, cy - ry - 2.5)
            cover.lineTo(outer_x, outer_y)
            cover.lineTo(inner_x, inner_y)
            cover.closeSubpath()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_SKIN)
            p.drawPath(cover)
            p.setPen(_pen(_LASH, 1.4))
            p.drawLine(QPointF(inner_x, inner_y), QPointF(outer_x, outer_y))
        else:
            lash = QPainterPath(QPointF(cx - side * (rx + 0.5), cy - 1.4))
            lash.quadTo(cx - side * 1.0, cy - ry - 2.2, cx + side * (rx + 0.6), cy - 2.6)
            lash.lineTo(cx + side * (rx + 1.8), cy - 3.4)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(_pen(_LASH, 1.5))
            p.drawPath(lash)

    def _brows(self, p: QPainter, mode: str) -> None:
        if mode == "none":
            return
        p.setBrush(Qt.BrushStyle.NoBrush)
        for side in (-1.0, 1.0):
            inner, outer = side * 4.6, side * 12.6
            width = 1.2
            if mode == "angry":
                a, b, ctrl_dy = -71.0, -75.8, 0.0
                width = 1.6
            elif mode == "sad":
                a, b, ctrl_dy = -76.0, -72.2, -0.4
            elif mode == "raised" or (mode == "curious" and side > 0):
                a, b, ctrl_dy = -76.6, -76.4, -1.8
            else:
                a, b, ctrl_dy = -73.4, -73.6, -1.4
            path = QPainterPath(QPointF(inner, a))
            path.quadTo((inner + outer) / 2, (a + b) / 2 + ctrl_dy, outer, b)
            p.setPen(_pen(_BROW, width))
            p.drawPath(path)

    def _mouth(self, p: QPainter, look: _Look, t: float) -> None:
        mode = look.mouth
        if mode == "chew":
            mode = "cat" if math.fmod(t, 0.5) < 0.25 else "o"
        p.setBrush(Qt.BrushStyle.NoBrush)
        line = _pen(_MOUTH_LINE, 0.9)
        y = -54.0
        if mode == "smile":
            path = QPainterPath(QPointF(-2.6, y - 0.2))
            path.quadTo(0.0, y + 1.9, 2.6, y - 0.2)
            p.setPen(line)
            p.drawPath(path)
        elif mode in ("cat", "tongue"):
            path = QPainterPath(QPointF(-3.0, y - 0.4))
            path.quadTo(-1.5, y + 1.6, 0.0, y)
            path.quadTo(1.5, y + 1.6, 3.0, y - 0.4)
            if mode == "tongue":
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(_TONGUE)
                p.drawEllipse(QPointF(1.5, y + 1.3), 1.1, 1.0)
                p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(line)
            p.drawPath(path)
        elif mode == "open_smile":
            path = QPainterPath(QPointF(-3.5, y - 1.0))
            path.quadTo(0.0, y - 0.2, 3.5, y - 1.0)
            path.quadTo(2.9, y + 3.8, 0.0, y + 3.9)
            path.quadTo(-2.9, y + 3.8, -3.5, y - 1.0)
            self._fill_mouth(p, path, tongue_y=y + 2.6)
        elif mode == "talk":
            o = max(0.0, min(1.0, look.talk_open))
            path = QPainterPath()
            path.addEllipse(QPointF(0.0, y + 0.4), 2.0 + 0.6 * o, 0.6 + 1.9 * o)
            self._fill_mouth(p, path, tongue_y=y + 0.4 + 1.2 * o if o > 0.4 else None)
        elif mode == "o":
            path = QPainterPath()
            path.addEllipse(QPointF(0.0, y + 0.6), 1.6, 1.9)
            self._fill_mouth(p, path, tongue_y=None)
        elif mode == "big_o":
            path = QPainterPath()
            path.addEllipse(QPointF(0.0, y + 1.2), 3.0, 3.9)
            self._fill_mouth(p, path, tongue_y=y + 3.4)
        elif mode == "pout":
            path = QPainterPath(QPointF(-2.4, y + 1.2))
            path.quadTo(0.0, y - 1.2, 2.4, y + 1.2)
            p.setPen(_pen(_MOUTH_LINE, 1.05))
            p.drawPath(path)
        elif mode == "wavy":
            path = QPainterPath(QPointF(-3.2, y))
            path.cubicTo(-2.1, y - 1.4, -1.1, y + 1.4, 0.0, y)
            path.cubicTo(1.1, y - 1.4, 2.1, y + 1.4, 3.2, y)
            p.setPen(line)
            p.drawPath(path)
        else:  # flat
            p.setPen(line)
            p.drawLine(QPointF(-1.4, y + 0.2), QPointF(2.2, y - 0.3))

    def _fill_mouth(self, p: QPainter, path: QPainterPath, tongue_y: float | None) -> None:
        p.setPen(_pen(_MOUTH_LINE, 0.6))
        p.setBrush(_MOUTH_INNER)
        p.drawPath(path)
        if tongue_y is not None:
            p.save()
            p.setClipPath(path, Qt.ClipOperation.IntersectClip)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_TONGUE)
            p.drawEllipse(QPointF(0.0, tongue_y), 2.0, 1.3)
            p.restore()

    # lying down ------------------------------------------------------------------------

    def _paint_sleeping(self, p: QPainter, look: _Look, t: float) -> None:
        """Curled up on a pillow, head on the left, long hair spilling over it."""
        # pillow
        grad = QLinearGradient(QPointF(0.0, -15.0), QPointF(0.0, 0.0))
        grad.setColorAt(0.0, _PILLOW)
        grad.setColorAt(1.0, QColor(232, 214, 198))
        p.setPen(_pen(_PILLOW_LINE, 0.6))
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(QRectF(-45.0, -15.0, 44.0, 15.0), 7.5, 7.5)
        # hair spread around the head and over the pillow
        hair = self._sleep_hair
        hgrad = QLinearGradient(QPointF(0.0, -52.0), QPointF(0.0, -4.0))
        hgrad.setColorAt(0.0, _HAIR)
        hgrad.setColorAt(1.0, _HAIR_DARK)
        p.setPen(_pen(_HAIR_LINE, 0.7))
        p.setBrush(QBrush(hgrad))
        p.drawPath(hair)
        # body in the hoodie, knees tucked
        p.save()
        p.translate(14.0, -10.0)
        p.rotate(-6.0)
        body = QPainterPath()
        body.addRoundedRect(QRectF(-16.0, -10.0, 33.0, 19.0), 9.0, 9.0)
        bgrad = QLinearGradient(QPointF(0.0, -10.0), QPointF(0.0, 9.0))
        bgrad.setColorAt(0.0, _HOODIE_LIGHT)
        bgrad.setColorAt(1.0, _HOODIE)
        p.setPen(_pen(_HOODIE_LINE, 0.7))
        p.setBrush(QBrush(bgrad))
        p.drawPath(body)
        self._chest_cat(p, QPointF(-3.0, -1.5))
        p.restore()
        p.setPen(_pen(_HOODIE_LINE, 0.6))
        p.setBrush(_HOODIE)
        p.drawEllipse(QPointF(34.5, -4.4), 6.0, 3.8)
        p.drawEllipse(QPointF(29.0, -2.8), 5.4, 3.1)
        self._mini_cat(p, QPointF(35.5, -4.6), 1.4)
        # face resting on the pillow, tilted
        p.save()
        p.translate(-17.0, -25.0)
        p.rotate(14.0)
        p.setPen(_pen(_SKIN_LINE, 0.5))
        p.setBrush(_SKIN)
        p.drawEllipse(QPointF(0.0, 0.0), 14.6, 13.4)
        for side in (-1.0, 1.0):
            c = QPointF(side * 9.8, 4.6)
            g = QRadialGradient(c, 3.8)
            g.setColorAt(0.0, _alpha(_BLUSH, 0.8 * look.blush))
            g.setColorAt(1.0, _alpha(_BLUSH, 0.0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(g))
            p.drawEllipse(c, 3.8, 2.4)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(_pen(_LASH, 1.2))
        for side in (-1.0, 1.0):
            eye = QPainterPath(QPointF(side * 6.6 - 3.6, 0.6))
            eye.quadTo(side * 6.6, 3.6, side * 6.6 + 3.6, 0.6)
            p.drawPath(eye)
        # small mouth, slightly open with each breath
        breath = _wave01(t, 1.0 / 3.4)
        p.setPen(_pen(_MOUTH_LINE, 0.7))
        p.setBrush(_MOUTH_INNER if breath > 0.6 else Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(0.0, 8.0), 1.0 + 0.4 * breath, 0.5 + 0.6 * breath)
        # fringe over the forehead
        bangs = QPainterPath(QPointF(-15.0, 1.0))
        bangs.quadTo(-17.0, -16.0, 0.0, -16.5)
        bangs.quadTo(17.0, -16.0, 15.0, 1.0)
        for x, y in ((11.0, -5.0), (7.5, -2.5), (4.0, -6.0), (0.5, -2.8), (-3.0, -6.0), (-6.5, -2.6),
                     (-10.0, -5.6), (-13.0, -1.0)):
            bangs.quadTo((x + bangs.currentPosition().x()) / 2, y - 3.0, x, y)
        bangs.closeSubpath()
        fgrad = QLinearGradient(QPointF(0.0, -16.0), QPointF(0.0, 0.0))
        fgrad.setColorAt(0.0, _HAIR_LIGHT)
        fgrad.setColorAt(1.0, _HAIR)
        p.setPen(_pen(_HAIR_LINE, 0.6))
        p.setBrush(QBrush(fgrad))
        p.drawPath(bangs)
        p.restore()
        # sleeve hugging the pillow
        sleeve = QPainterPath(QPointF(2.0, -16.0))
        sleeve.quadTo(-4.0, -6.0, -10.0, -7.5)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(_pen(_HOODIE_LINE, 8.2))
        p.drawPath(sleeve)
        p.setPen(_pen(_HOODIE, 6.8))
        p.drawPath(sleeve)
        p.setPen(_pen(_HOODIE_LINE, 0.6))
        p.setBrush(_HOODIE)
        p.drawEllipse(QPointF(-10.0, -7.5), 3.6, 3.6)
        self._hair_clip(p, QPointF(-7.0, -43.0))
