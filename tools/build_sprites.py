"""Build the character sprite pack from the source sprite sheets.

The source art is nine PNG sheets (1536x1024, RGBA):

* eight 4x2 sheets with full-body frames (``...-1.png`` .. ``...-8.png``);
* one labelled 8x5 "emotions and poses" master sheet (any other name).

The 4x2 sheets are *not* laid out on an exact grid: hair regularly crosses
the 384 px column borders and the bottom row sits higher than y=512.  Cutting
on a fixed grid would slice hair and bleed neighbours in, so figures are
separated by nearest-body assignment instead (see ``_split_figures``).

The master sheet has text labels, coloured cell borders and a faint tinted
cell background baked into its alpha channel; those are removed here.  Its
cells are cut by their borders and are ~2.7x smaller, so they are exported as *portraits*
(used in speech bubbles and windows), never as desktop body frames.

Usage::

    python tools/build_sprites.py --src <dir with the 9 PNGs> \
        --out src/chibi/assets/characters/mia [--debug <dir>]

Dev-only requirements: Pillow, numpy, scipy (``pip install -e .[dev]``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SHEET_NAMES = {
    1: "front",
    2: "walk",
    3: "run",
    4: "gesture",
    5: "sleep",
    6: "wave",
    7: "cheer",
    8: "sit",
}

# Master sheet layout (measured from the alpha channel of the supplied sheet).
MASTER_ROWS = {
    "idle": (26, 217),
    "walk": (242, 437),
    "actions": (462, 654),
    "emotions": (679, 853),
    "extras": (875, 1024),
}
MASTER_CELL_W = 192
MASTER_LABEL_H = 21
MASTER_INSET = 4
MASTER_NAMES = {
    "idle": ["calm", "breathing", "wink", "smile", "wink_heart", "joyful", "surprised", "thinking"],
    "walk": ["step1", "step2", "step3", "step4", "back", "run1", "run2", "stop"],
    "actions": ["sit_down", "sit_cat", "stand_up", "jump", "greet", "bye", "point", "typing"],
    "emotions": ["joy", "laugh", "wow", "angry", "offended", "shy", "excited", "tired"],
    "extras": ["eat", "drink", "music", "phone", "idea", "question", "facepalm", "sleep_cat"],
}
# The master sheet is used for portraits only, never for desktop body frames:
# * almost every cell is cut by the cell's bottom edge (25-35 % of the bottom
#   rows are opaque), which would look like a sliced-off body on the desktop;
# * matching the full-body sheets needs ~2.7x upscaling (eye-spacing match),
#   which is visibly blurry next to the native 384x512 frames.
# The manifest keeps a per-frame "scale" so properly sized art can be added.
# Master cells exported as bust portraits (cropped from the top when taller).
MASTER_PORTRAITS = {
    "idle": ["calm", "wink", "smile", "wink_heart", "joyful", "surprised", "thinking"],
    "actions": ["greet", "bye", "point", "typing"],
    "emotions": ["joy", "laugh", "wow", "angry", "offended", "shy", "excited", "tired"],
    "extras": ["eat", "drink", "music", "phone", "idea", "question", "facepalm", "sleep_cat"],
}


def F(frame: str, ms: int) -> dict:
    return {"id": frame, "ms": ms}


# Animation clips.  "facing" is the direction the art looks in, so the
# renderer knows when to mirror.  "fx" are procedural modifiers applied on
# top of the frames; "overlay" is a drawn effect (anger mark, tear, ...).
CLIPS: dict[str, dict] = {
    "idle": {
        "frames": [
            F("front_0", 3200), F("front_1", 160), F("front_0", 2600), F("front_1", 140),
            F("front_0", 900), F("front_5", 2800), F("front_6", 170), F("front_5", 2200),
        ],
        "loop": True, "facing": "front", "fx": {"breathe": 0.012},
    },
    "idle_look": {"frames": [F("front_7", 2400)], "loop": False, "facing": "left",
                  "fx": {"breathe": 0.012}},
    "idle_smile": {"frames": [F("gesture_0", 2200), F("front_3", 900)], "loop": False,
                   "facing": "front", "fx": {"breathe": 0.012}},
    "walk": {"frames": [F(f"walk_{i}", 110) for i in range(8)], "loop": True,
             "facing": "left", "fx": {"bob": 0.010}},
    "run": {"frames": [F(f"run_{i}", 75) for i in range(8)], "loop": True,
            "facing": "right", "fx": {"bob": 0.016}},
    "sit": {"frames": [F("sit_0", 3600), F("sit_6", 2600), F("sit_0", 2400), F("sit_4", 1600)],
            "loop": True, "facing": "front", "fx": {"breathe": 0.010}, "blend_ms": 220},
    "yawn": {"frames": [F("sit_3", 1800)], "loop": False, "facing": "front",
             "fx": {"breathe": 0.016}},
    "sleep": {"frames": [F(f"sleep_{i}", 3000) for i in (0, 1, 3, 7, 4, 6)], "loop": True,
              "facing": "front", "fx": {"breathe": 0.018}, "blend_ms": 1100},
    "talk": {"frames": [F("gesture_0", 420), F("gesture_2", 380), F("gesture_4", 520),
                        F("gesture_6", 380), F("gesture_1", 450), F("gesture_3", 420)],
             "loop": True, "facing": "front", "fx": {"bob": 0.008, "breathe": 0.008},
             "blend_ms": 90},
    "laugh": {"frames": [F("cheer_0", 260), F("cheer_3", 260), F("cheer_5", 260), F("cheer_3", 260)],
              "loop": True, "facing": "front", "fx": {"shake": 0.006, "bob": 0.010}},
    "happy": {"frames": [F("cheer_1", 300), F("cheer_2", 300), F("cheer_4", 300),
                         F("cheer_6", 300), F("cheer_7", 300)],
              "loop": True, "facing": "front", "fx": {"hop": 0.035}, "overlay": "sparkle"},
    "angry": {"frames": [F("front_2", 2000)], "loop": True, "facing": "front",
              "fx": {"shake": 0.010}, "overlay": "anger"},
    "sad": {"frames": [F("sit_5", 3000)], "loop": True, "facing": "front",
            "fx": {"breathe": 0.010}, "overlay": "tear"},
    "surprised": {"frames": [F("gesture_6", 600), F("gesture_2", 1100)], "loop": False,
                  "facing": "front", "fx": {"hop": 0.030}, "overlay": "exclaim"},
    "curious": {"frames": [F("front_7", 2200), F("front_0", 600)], "loop": True,
                "facing": "left", "fx": {"tilt": 3.0}, "overlay": "question"},
    "gaming": {"frames": [F("sit_0", 2200), F("sit_2", 900), F("sit_1", 800), F("sit_0", 1800),
                          F("sit_7", 900)],
               "loop": True, "facing": "front", "fx": {"breathe": 0.008}, "overlay": "game",
               "blend_ms": 150},
    "thinking": {"frames": [F("gesture_5", 2200), F("front_7", 1400)], "loop": True,
                 "facing": "front", "fx": {"breathe": 0.010}, "overlay": "dots", "blend_ms": 200},
    "wave": {"frames": [F("wave_1", 300), F("wave_3", 300), F("wave_4", 300), F("wave_3", 300),
                        F("wave_1", 300), F("wave_5", 450)],
             "loop": False, "facing": "front", "fx": {"bob": 0.006}},
    "dragged": {"frames": [F("cheer_1", 360), F("cheer_6", 360)], "loop": True,
                "facing": "front", "fx": {"swing": 9.0}},
    "eat": {"frames": [F("cheer_5", 500), F("cheer_0", 500)], "loop": True, "facing": "front",
            "fx": {"bob": 0.010}, "overlay": "hearts"},
    "shy": {"frames": [F("gesture_5", 1800), F("front_6", 1200)], "loop": True, "facing": "front",
            "fx": {"breathe": 0.010}, "overlay": "blush"},
    "greet": {"frames": [F("wave_2", 500), F("wave_6", 500), F("wave_7", 500), F("wave_6", 500)],
              "loop": False, "facing": "front", "fx": {"hop": 0.02}},
    "bye": {"frames": [F("wave_3", 350), F("wave_4", 350), F("wave_3", 350), F("wave_4", 350),
                       F("wave_1", 500)], "loop": False, "facing": "front", "fx": {"bob": 0.006}},
    "point": {"frames": [F("gesture_4", 1600)], "loop": False, "facing": "front",
              "overlay": "exclaim"},

}

# Blink frames are swapped in for ~150 ms, so their head must sit exactly where
# the partner frame's head is (the generator drew them up to 11 px apart, which
# reads as a head twitch on every blink).  Their anchors are re-aligned to the
# partner by matching the upper body silhouette.
BLINK_ALIGN = {"front_1": "front_0", "front_6": "front_5"}

# Character state -> clips.  "variants" are occasional idle fidgets.
STATES: dict[str, dict] = {
    "idle": {"clip": "idle", "variants": ["idle_look", "idle_smile"], "portrait": "calm"},
    "walk": {"clip": "walk", "portrait": "calm"},
    "run": {"clip": "run", "portrait": "excited"},
    "sit": {"clip": "sit", "portrait": "smile"},
    "sleep": {"clip": "sleep", "portrait": "sleep_cat"},
    "talk": {"clip": "talk", "portrait": "smile"},
    "laugh": {"clip": "laugh", "portrait": "laugh"},
    "happy": {"clip": "happy", "portrait": "joy"},
    "angry": {"clip": "angry", "portrait": "angry"},
    "sad": {"clip": "sad", "portrait": "offended"},
    "surprised": {"clip": "surprised", "portrait": "wow"},
    "curious": {"clip": "curious", "portrait": "question"},
    "gaming": {"clip": "gaming", "portrait": "typing"},
    "thinking": {"clip": "thinking", "portrait": "thinking"},
    "wave": {"clip": "wave", "variants": ["greet"], "portrait": "wink"},
    "dragged": {"clip": "dragged", "portrait": "wow"},
    "eat": {"clip": "eat", "portrait": "eat"},
    "shy": {"clip": "shy", "portrait": "shy"},
    "yawn": {"clip": "yawn", "portrait": "tired"},
    "bye": {"clip": "bye", "portrait": "bye"},
    "point": {"clip": "point", "portrait": "point"},
}


@dataclass
class Frame:
    image: Image.Image
    anchor: tuple[int, int]  # (x, y) of the ground contact point, in frame pixels
    height: int  # figure height in px (top of hair to ground)


def _find_sources(src: Path) -> tuple[dict[int, Path], Path]:
    sheets: dict[int, Path] = {}
    others: list[Path] = []
    for path in sorted(src.glob("*.png")):
        m = re.search(r"-(\d)\.png$", path.name)
        if m and 1 <= int(m.group(1)) <= 8:
            sheets[int(m.group(1))] = path
        else:
            others.append(path)
    if len(sheets) != 8 or len(others) != 1:
        raise SystemExit(
            f"expected 8 numbered sheets (-1..-8) and 1 master sheet in {src}, "
            f"found sheets={sorted(sheets)} others={[p.name for p in others]}"
        )
    return sheets, others[0]


def _drop_specks(alpha: np.ndarray, min_area: int, min_peak: int) -> np.ndarray:
    """Zero out small or faint isolated alpha islands (generator noise)."""
    lab, n = ndimage.label(alpha > 0, structure=np.ones((3, 3)))
    if n == 0:
        return alpha
    idx = np.arange(1, n + 1)
    area = ndimage.sum(np.ones_like(alpha), lab, idx)
    peak = ndimage.maximum(alpha, lab, idx)
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = (area >= min_area) & (peak >= min_peak)
    out = alpha.copy()
    out[~keep[lab]] = 0
    return out


def _ground_and_anchor_x(alpha: np.ndarray, lying: bool) -> tuple[int, int, int]:
    """Return (top, ground_y, anchor_x) for a figure alpha mask."""
    solid = alpha > 128
    rows = np.where(solid.sum(axis=1) >= 6)[0]
    top = int(np.where(solid.any(axis=1))[0][0])
    ground = int(rows.max())
    if lying:
        xs = np.where(solid.any(axis=0))[0]
        return top, ground, round((xs.min() + xs.max()) / 2)
    # Legs/feet band: the part of the body that is locked to the ground.
    height = ground - top
    band = solid[max(top, ground - int(0.09 * height)): ground + 1]
    _, xs = np.where(band)
    return top, ground, round(float(np.median(xs)))


def _split_figures(rgba: np.ndarray, sleeping: bool) -> list[np.ndarray]:
    """Separate the 8 figures of a 4x2 sheet without a fixed grid cut.

    Hair in these sheets regularly crosses the cell borders.  Assigning
    pixels to the nearest body would slice every such curl at the midline
    (and hand it to the neighbour's frame), so ownership is decided per
    connected component instead: a figure whose silhouette doesn't touch a
    neighbour keeps all of its hair, uncut.  Only silhouettes that really
    touch are split - by nearest body, with a soft feathered fade along the
    seam instead of a ruler-straight edge.  Faint edge pixels follow the
    strong pixel they surround.
    """
    alpha = rgba[..., 3]
    _, w = alpha.shape
    seeds = np.zeros_like(alpha, dtype=np.int32)
    cw = w // 4
    bands = [(200, 470), (600, 900)] if sleeping else [(120, 440), (600, 920)]
    k = 0
    for y0, y1 in bands:
        for c in range(4):
            k += 1
            x0, x1 = c * cw + 95, (c + 1) * cw - 95
            box = alpha[y0:y1, x0:x1] >= 200
            if box.sum() < 500:
                raise SystemExit(f"figure core {k} is empty; sheet layout changed?")
            seeds[y0:y1, x0:x1][box] = k

    # Distance of every pixel to each body core (for splitting touching figures).
    core_dist = np.stack([ndimage.distance_transform_edt(seeds != i) for i in range(1, 9)])
    order = np.argsort(core_dist, axis=0)
    nearest = order[0] + 1
    rows, cols = np.indices(alpha.shape)
    d_own = core_dist[order[0], rows, cols]
    d_next = core_dist[order[1], rows, cols]

    strong = alpha >= 40
    lab, _ = ndimage.label(strong, structure=np.ones((3, 3)))
    owner = np.zeros_like(alpha, dtype=np.int32)
    feather = np.ones(alpha.shape, dtype=np.float32)
    seam = 26.0  # px over which a forced split fades out
    objects = ndimage.find_objects(lab)
    for comp, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        mask = lab[sl] == comp
        cores = set(np.unique(seeds[sl][mask])) - {0}
        if len(cores) == 1:
            owner[sl][mask] = cores.pop()
        elif len(cores) > 1:
            # Touching silhouettes: split by nearest body, feather the seam.
            owner[sl][mask] = nearest[sl][mask]
            margin = (d_next[sl] - d_own[sl])[mask]
            t = np.clip(margin / seam, 0.0, 1.0)
            feather[sl][mask] = t * t * (3 - 2 * t)
        else:
            # No body inside (zzz, sparkles, detached curls): whole piece to the
            # body that is nearest on average - never split.
            votes = np.bincount(nearest[sl][mask].ravel(), minlength=9)
            owner[sl][mask] = int(votes.argmax())

    # Faint pixels (soft edges, glow) belong to the strong pixel they touch.
    weak = (alpha > 0) & ~strong
    dist, (iy, ix) = ndimage.distance_transform_edt(~strong, return_indices=True)
    owner[weak] = owner[iy[weak], ix[weak]]
    feather[weak] = feather[iy[weak], ix[weak]]
    owner[weak & (dist > 24)] = 0

    out = []
    for i in range(1, 9):
        a = np.where(owner == i, alpha.astype(np.float32) * feather, 0.0)
        out.append(np.clip(np.rint(a), 0, 255).astype(np.uint8))
    return out


def _crop(rgba: np.ndarray, alpha: np.ndarray, pad: int = 2) -> tuple[Image.Image, int, int]:
    ys, xs = np.where(alpha > 0)
    y0, y1 = max(0, ys.min() - pad), min(alpha.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(alpha.shape[1], xs.max() + pad + 1)
    out = rgba[y0:y1, x0:x1].copy()
    out[..., 3] = alpha[y0:y1, x0:x1]
    out[out[..., 3] == 0, :3] = 0  # don't ship hidden background colour
    return Image.fromarray(out, "RGBA"), int(x0), int(y0)


def extract_sheet(path: Path, sheet_no: int) -> list[Frame]:
    rgba = np.array(Image.open(path).convert("RGBA"))
    sleeping = SHEET_NAMES[sheet_no] == "sleep"
    frames = []
    for fig_alpha in _split_figures(rgba, sleeping):
        fig_alpha = _drop_specks(fig_alpha, min_area=40, min_peak=70)
        top, ground, ax = _ground_and_anchor_x(fig_alpha, lying=sleeping)
        img, x0, y0 = _crop(rgba, fig_alpha)
        frames.append(Frame(img, (ax - x0, ground - y0), ground - top))
    return frames


def _master_cell(rgba: np.ndarray, row: str, col: int) -> tuple[np.ndarray, np.ndarray]:
    y0, y1 = MASTER_ROWS[row]
    x0 = col * MASTER_CELL_W + MASTER_INSET
    x1 = (col + 1) * MASTER_CELL_W - MASTER_INSET
    cell = rgba[y0 + MASTER_LABEL_H: y1 - 2, x0:x1].copy()
    alpha = cell[..., 3].copy()
    alpha[alpha < 24] = 0  # tinted cell background
    # Coloured cell borders: near-solid lines in the outer columns / top rows.
    # (Bottom rows are skipped: busts are legitimately cut there.)
    for x in list(range(8)) + list(range(alpha.shape[1] - 8, alpha.shape[1])):
        if (alpha[:, x] > 60).mean() > 0.30:
            alpha[:, x] = 0
    for y in range(6):
        if (alpha[y] > 60).mean() > 0.30:
            alpha[y] = 0
    # Remove border lines / glow: thin components touching the crop edge,
    # small specks and faint (low mean alpha) haze.
    lab, _ = ndimage.label(alpha > 0, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(lab)
    ch, cw = alpha.shape
    for i, sl in enumerate(objs, start=1):
        comp = lab[sl] == i
        area = int(comp.sum())
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        touches = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == ch or sl[1].stop == cw
        mean_a = float(alpha[sl][comp].mean())
        if area < 30 or mean_a < 70 or (touches and min(hh, ww) <= 6):
            alpha[sl][comp] = 0
    return cell, alpha


def extract_portraits(path: Path) -> dict[str, Image.Image]:
    rgba = np.array(Image.open(path).convert("RGBA"))
    portraits: dict[str, Image.Image] = {}
    for row, names in MASTER_PORTRAITS.items():
        for name in names:
            cell, alpha = _master_cell(rgba, row, MASTER_NAMES[row].index(name))
            img, _, _ = _crop(cell, alpha)
            portraits[name] = img.crop((0, 0, img.width, min(img.height, int(img.width * 0.95))))
    return portraits


def _align_to(frame: Frame, partner: Frame, radius: int = 16) -> tuple[int, int]:
    """Anchor for *frame* that best overlays its upper body on *partner*."""
    size = 700

    def place(fr: Frame) -> np.ndarray:
        a = np.asarray(fr.image)[..., 3].astype(np.float32) / 255.0
        canvas = np.zeros((size, size), np.float32)
        ox, oy = size // 2 - fr.anchor[0], size - 40 - fr.anchor[1]
        canvas[oy:oy + a.shape[0], ox:ox + a.shape[1]] = a
        return canvas

    ref, mov = place(partner), place(frame)
    rows = slice(size - 40 - int(partner.height), size - 40 - int(partner.height * 0.25))
    best = (float("inf"), 0, 0)
    for dy in range(-radius, radius + 1):
        shifted = np.roll(mov, dy, axis=0)
        for dx in range(-radius, radius + 1):
            err = float(((ref[rows] - np.roll(shifted, dx, axis=1)[rows]) ** 2).sum())
            if err < best[0]:
                best = (err, dx, dy)
    _, dx, dy = best
    return frame.anchor[0] - dx, frame.anchor[1] - dy


def _save_png(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=True)


def build(src: Path, out: Path, debug: Path | None) -> dict:
    sheets, master = _find_sources(src)
    frames: dict[str, Frame] = {}
    for no, path in sorted(sheets.items()):
        for i, fr in enumerate(extract_sheet(path, no)):
            frames[f"{SHEET_NAMES[no]}_{i}"] = fr
    for fid, partner in BLINK_ALIGN.items():
        frames[fid].anchor = _align_to(frames[fid], frames[partner])
    standing = [frames[f"front_{i}"].height for i in range(8)]
    reference_height = int(np.median(standing))

    portraits = extract_portraits(master)

    used = {f["id"] for clip in CLIPS.values() for f in clip["frames"]}
    missing = used - frames.keys()
    if missing:
        raise SystemExit(f"clips reference unknown frames: {sorted(missing)}")

    out.mkdir(parents=True, exist_ok=True)
    for old in list((out / "frames").glob("*.png")) + list((out / "portraits").glob("*.png")):
        old.unlink()
    manifest_frames = {}
    for fid in sorted(used):
        fr = frames[fid]
        _save_png(fr.image, out / "frames" / f"{fid}.png")
        manifest_frames[fid] = {
            "file": f"frames/{fid}.png",
            "anchor": list(fr.anchor),
            "scale": 1.0,
        }
    manifest_portraits = {}
    for name, img in sorted(portraits.items()):
        _save_png(img, out / "portraits" / f"{name}.png")
        manifest_portraits[name] = f"portraits/{name}.png"
    for state in STATES.values():
        for clip in [state["clip"], *state.get("variants", [])]:
            if clip not in CLIPS:
                raise SystemExit(f"state references unknown clip {clip}")
        if state["portrait"] not in manifest_portraits:
            raise SystemExit(f"state references unknown portrait {state['portrait']}")

    manifest = {
        "format": 1,
        "name": "Mia",
        "description": "Chibi girl in a black cat hoodie. Art supplied by the project owner "
                       "(generated with ChatGPT image generation); see ASSETS.md.",
        "reference_height": reference_height,
        "frames": manifest_frames,
        "clips": CLIPS,
        "states": STATES,
        "portraits": manifest_portraits,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    if debug:
        _debug_sheets(frames, portraits, reference_height, debug)
    return manifest


def _debug_sheets(frames, portraits, ref_h, debug: Path) -> None:
    """Contact sheet (frames on a light background, anchor crosshair, common
    ground line) and per-clip onion skins to eyeball alignment jitter."""
    debug.mkdir(parents=True, exist_ok=True)
    target = 220
    k = target / ref_h
    ids = sorted(frames)
    cols = 10
    cell_w, cell_h = 200, 290
    rows = (len(ids) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * cell_w, rows * cell_h), (232, 238, 248, 255))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    for n, fid in enumerate(ids):
        fr = frames[fid]
        s = k
        img = fr.image.resize((max(1, int(fr.image.width * s)), max(1, int(fr.image.height * s))),
                              Image.LANCZOS)
        ax, ay = fr.anchor[0] * s, fr.anchor[1] * s
        cx, gy = (n % cols) * cell_w + cell_w // 2, (n // cols) * cell_h + cell_h - 30
        sheet.alpha_composite(img, (int(cx - ax), int(gy - ay)))
        draw.line([(cx - 90, gy), (cx + 90, gy)], fill=(255, 0, 0, 255))
        draw.line([(cx, gy - 6), (cx, gy + 6)], fill=(255, 0, 0, 255))
        draw.text((cx - 90, gy + 8), fid, fill=(0, 0, 0, 255))
    sheet.convert("RGB").save(debug / "contact_sheet.png")

    for name, clip in CLIPS.items():
        ids = list(dict.fromkeys(f["id"] for f in clip["frames"]))
        if len(ids) < 2:
            continue
        canvas = Image.new("RGBA", (300, 300), (232, 238, 248, 255))
        for fid in ids:
            fr = frames[fid]
            s = k
            img = fr.image.resize((int(fr.image.width * s), int(fr.image.height * s)), Image.LANCZOS)
            a = np.array(img)
            a[..., 3] = (a[..., 3] * (1.0 / len(ids)) * 1.6).clip(0, 255).astype(np.uint8)
            canvas.alpha_composite(Image.fromarray(a), (int(150 - fr.anchor[0] * s),
                                                        int(270 - fr.anchor[1] * s)))
        canvas.convert("RGB").save(debug / f"onion_{name}.png")

    strip = Image.new("RGBA", (130 * len(portraits), 140), (232, 238, 248, 255))
    for n, (_name, img) in enumerate(sorted(portraits.items())):
        im = img.copy()
        im.thumbnail((124, 124), Image.LANCZOS)
        strip.alpha_composite(im, (n * 130 + 3, 3))
    strip.convert("RGB").save(debug / "portraits.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--debug", type=Path)
    args = ap.parse_args(argv)
    manifest = build(args.src, args.out, args.debug)
    print(f"wrote {len(manifest['frames'])} frames, {len(manifest['portraits'])} portraits, "
          f"{len(manifest['clips'])} clips (reference height {manifest['reference_height']} px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
