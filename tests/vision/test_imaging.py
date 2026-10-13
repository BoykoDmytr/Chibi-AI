import sys

import pytest

from chibi.vision.capture_win import CaptureError, Frame, Win32WindowCapture, is_mostly_black
from chibi.vision.imaging import (
    dhash,
    downscale,
    encode_jpeg,
    frame_to_image,
    hamming,
    image_media_type,
)


def solid(width, height, bgr=(0, 0, 0)):
    b, g, r = bgr
    return bytes([b, g, r, 255]) * (width * height)


def gradient(width, height, shift=0):
    data = bytearray()
    for y in range(height):
        for x in range(width):
            v = (x * 255 // max(1, width - 1) + shift) % 256
            data += bytes([v, (y * 3) % 256, 255 - v, 255])
    return bytes(data)


def frame(data, width, height):
    return Frame(width, height, data, hwnd=1, captured_at=0.0, method="fake")


def test_black_detection():
    assert is_mostly_black(solid(64, 64), 64, 64)
    assert not is_mostly_black(gradient(64, 64), 64, 64)
    # A dark scene with a lit HUD is not "black".
    dark = bytearray(solid(100, 100, (3, 3, 3)))
    for i in range(0, 100 * 4 * 5, 4):  # first five rows bright
        dark[i:i + 3] = b"\xff\xff\xff"
    assert not is_mostly_black(bytes(dark), 100, 100)
    assert is_mostly_black(b"", 10, 10)


def test_frame_to_image_and_downscale(qapp):
    img = frame_to_image(frame(gradient(2048, 1152), 2048, 1152))
    assert (img.width(), img.height()) == (2048, 1152)
    small = downscale(img)
    assert (small.width(), small.height()) == (1024, 576)
    assert downscale(small) is small


def test_jpeg_encoding(qapp):
    img = frame_to_image(frame(gradient(320, 200), 320, 200))
    data = encode_jpeg(img)
    assert data[:3] == b"\xff\xd8\xff"
    assert image_media_type(data) == "image/jpeg"


def test_dhash_dedup(qapp):
    a = frame_to_image(frame(gradient(320, 200), 320, 200))
    a_small = downscale(a, 160)
    b = frame_to_image(frame(gradient(320, 200, shift=128), 320, 200))
    assert hamming(dhash(a), dhash(a_small)) <= 6  # same picture, other size
    assert hamming(dhash(a), dhash(b)) > 10  # different picture


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows behaviour")
def test_window_capture_is_windows_only():
    with pytest.raises(CaptureError) as info:
        Win32WindowCapture()
    assert info.value.reason == "unsupported"
    assert "Windows" in info.value.user_message
