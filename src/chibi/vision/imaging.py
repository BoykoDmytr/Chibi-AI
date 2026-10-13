"""Frame processing for vision: downscale, de-duplicate, encode.

QImage is used (already a dependency, no Pillow at runtime); it is
reentrant, so this runs on the vision worker thread without a GUI.
Frames stay in memory: nothing here writes to disk.
"""

from __future__ import annotations

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtGui import QImage

from chibi.vision.capture_win import Frame

#: Long side of the image sent for analysis (cost and latency; models
#: downscale larger inputs anyway).
MAX_SIDE = 1024


def frame_to_image(frame: Frame) -> QImage:
    image = QImage(frame.data, frame.width, frame.height, frame.width * 4, QImage.Format.Format_RGB32)
    return image.copy()  # own the pixels: frame.data may go away


def downscale(image: QImage, max_side: int = MAX_SIDE) -> QImage:
    if max(image.width(), image.height()) <= max_side:
        return image
    return image.scaled(max_side, max_side, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)


def encode_jpeg(image: QImage, quality: int = 80) -> bytes:
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "JPG", quality):
        # No JPEG plugin (unusual) - PNG is built into QtGui.
        data.clear()
        buffer.close()
        buffer = QBuffer(data)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        image.save(buffer, "PNG")
    buffer.close()
    return bytes(data.data())


def image_media_type(data: bytes) -> str:
    return "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"


def dhash(image: QImage) -> int:
    """64-bit difference hash: robust to scaling and compression, changes when
    the scene changes.  Hamming distance <= ~6 means "the same picture"."""
    small = image.scaled(9, 8, Qt.AspectRatioMode.IgnoreAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    gray = small.convertToFormat(QImage.Format.Format_Grayscale8)
    value = 0
    bit = 0
    for y in range(8):
        for x in range(8):
            if gray.pixelColor(x, y).value() > gray.pixelColor(x + 1, y).value():
                value |= 1 << bit
            bit += 1
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()
