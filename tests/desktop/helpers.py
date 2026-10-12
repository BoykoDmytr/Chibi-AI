"""Helpers shared by the desktop tests (image inspection, fakes, a stepping loop)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QImage

from chibi.core.paths import character_pack_dir

MIA_DIR = character_pack_dir("mia")


def blank_image(width: int, height: int, dpr: float = 1.0) -> QImage:
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(0, 0, 0, 0))
    image.setDevicePixelRatio(dpr)
    return image


def alpha_bbox(image: QImage, threshold: int = 0) -> tuple[int, int, int, int] | None:
    """(left, top, right, bottom) of pixels with alpha > threshold, inclusive, device px."""
    img = image.convertToFormat(QImage.Format.Format_RGBA8888)
    width, height, bpl = img.width(), img.height(), img.bytesPerLine()
    data = bytes(img.constBits())[: bpl * height]
    keep = bytes(range(threshold + 1))
    left, top, right, bottom = width, height, -1, -1
    for y in range(height):
        alpha = data[y * bpl: y * bpl + width * 4][3::4]
        stripped = alpha.lstrip(keep)
        if not stripped:
            continue
        top = min(top, y)
        bottom = y
        left = min(left, width - len(stripped))
        right = max(right, len(alpha.rstrip(keep)) - 1)
    if right < 0:
        return None
    return left, top, right, bottom


def alpha_at(image: QImage, x: int, y: int) -> int:
    return image.pixelColor(x, y).alpha()


def count_different(a: QImage, b: QImage) -> int:
    """Number of pixels that differ between two same-sized images."""
    ia = a.convertToFormat(QImage.Format.Format_RGBA8888)
    ib = b.convertToFormat(QImage.Format.Format_RGBA8888)
    assert ia.size() == ib.size()
    da, db = bytes(ia.constBits()), bytes(ib.constBits())
    return sum(1 for i in range(0, len(da), 4) if da[i:i + 4] != db[i:i + 4])


class FakeScreens:
    """Injectable monitor list for the controller (tests can't add real screens)."""

    def __init__(self, screens: list[Any]) -> None:
        self.screens = screens

    def __call__(self) -> list[Any]:
        return list(self.screens)


class FakeCursor:
    def __init__(self, x: float = -5000.0, y: float = -5000.0) -> None:
        self.pos = QPointF(x, y)

    def __call__(self) -> QPointF:
        return QPointF(self.pos)


def run(ctrl: Any, clock: Any, seconds: float, dt: float = 0.1,
        until: Callable[[], bool] | None = None) -> float:
    """Advance the fake clock and the controller together; returns simulated seconds."""
    elapsed = 0.0
    while elapsed < seconds - 1e-9:
        clock.advance(dt)
        ctrl.step(dt)
        elapsed += dt
        if until is not None and until():
            break
    return elapsed
