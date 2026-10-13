"""Build packaging/chibi.ico from Mia's portrait (same crop as the tray icon).

    python tools/make_icon.py

Needs Pillow (dev extra).  The .ico is committed, so builds don't need this.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
PORTRAIT = ROOT / "src" / "chibi" / "assets" / "characters" / "mia" / "portraits" / "smile.png"
TARGET = ROOT / "packaging" / "chibi.ico"
SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)


def face(portrait: Image.Image) -> Image.Image:
    side = min(portrait.size)
    crop = max(1, round(side * 0.82))
    x = max(0, (portrait.width - crop) // 2)
    return portrait.crop((x, 0, x + crop, crop))


def rounded(image: Image.Image, size: int) -> Image.Image:
    scaled = image.resize((size, size), Image.Resampling.LANCZOS)
    mask = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size * 4 - 1, size * 4 - 1), radius=size, fill=255)
    mask = mask.resize((size, size), Image.Resampling.LANCZOS)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    alpha = Image.composite(scaled.getchannel("A"), Image.new("L", (size, size), 0), mask)
    out.paste(scaled, (0, 0))
    out.putalpha(alpha)
    return out


def main() -> int:
    portrait = Image.open(PORTRAIT).convert("RGBA")
    base = face(portrait)
    images = [rounded(base, s) for s in SIZES]
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    images[-1].save(TARGET, format="ICO", sizes=[(s, s) for s in SIZES], append_images=images[:-1])
    print(f"wrote {TARGET} ({TARGET.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
