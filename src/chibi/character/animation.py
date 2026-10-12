"""Character packs (sprite frames + clips) and frame playback.

A *character pack* is a directory with a ``manifest.json`` (written by
``tools/build_sprites.py``) plus the PNG files it references::

    {"format": 1, "name": "Mia", "reference_height": 443,
     "frames":    {"front_0": {"file": "frames/front_0.png", "anchor": [191, 461], "scale": 1.0}},
     "clips":     {"idle": {"frames": [{"id": "front_0", "ms": 3200}], "loop": true,
                            "facing": "front", "blend_ms": 0, "fx": {"breathe": 0.012},
                            "overlay": null}},
     "states":    {"idle": {"clip": "idle", "variants": ["idle_look"], "portrait": "calm"}},
     "portraits": {"calm": "portraits/calm.png"}}

The manifest is validated completely at load time so a broken or hand-edited
pack fails once, with a message that names the offending entry, instead of
crashing the frame timer later.  Packs may come from users, so every file
path is resolved inside the pack directory and anything that escapes it
(``..``, absolute paths, drive letters, symlinks pointing outside) is
rejected.

Images are *not* decoded here: this module is pure Python (no Qt), so the
renderer still has to cope with a file that exists but does not decode.

:class:`FramePlayer` turns a :class:`Clip` plus elapsed time into a
:class:`FrameSample` (which frame to draw and how much of the next one to
cross-fade in).  It is deterministic and clock-free: the caller passes the
elapsed milliseconds, which keeps it trivially testable.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

from chibi.character.states import CharacterState, Facing

log = logging.getLogger(__name__)

#: Manifest format this loader understands.
PACK_FORMAT = 1
MANIFEST_NAME = "manifest.json"
#: A real manifest is ~30 KB; refuse absurd files instead of parsing them.
MAX_MANIFEST_BYTES = 4 * 1024 * 1024


class PackError(Exception):
    """The character pack is missing, malformed or unsafe."""


@dataclass(frozen=True)
class FrameInfo:
    """One body image.  ``anchor`` is the feet point in frame pixels; ``scale``
    corrects art drawn at a different size than the pack's reference height."""

    id: str
    path: Path
    anchor: tuple[int, int]
    scale: float


@dataclass(frozen=True)
class ClipFrame:
    frame_id: str
    ms: int


def _effective_ms(ms: float) -> int:
    """Frame duration as played.  ``ms <= 0`` would make a frame invisible and
    break the time -> frame lookup (zero-length spans), so it counts as 1 ms."""
    return max(1, int(ms))


@dataclass(frozen=True)
class Clip:
    """A named animation.  ``facing`` is the direction the *art* looks in, so
    the renderer knows when it has to mirror; ``fx`` are procedural modifiers
    (breathe/bob/hop/shake as a fraction of figure height, tilt/swing in
    degrees) and ``overlay`` a drawn effect such as ``"anger"``."""

    name: str
    frames: tuple[ClipFrame, ...]
    loop: bool
    facing: Facing
    blend_ms: int = 0
    # Excluded from the hash: a mapping is unhashable, and the name + frames
    # already identify a clip for caching purposes.
    fx: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}), hash=False)
    overlay: str | None = None

    @property
    def duration_ms(self) -> int:
        """Total play time, counting ``ms <= 0`` frames as 1 ms like the player does."""
        return sum(_effective_ms(f.ms) for f in self.frames)


@dataclass(frozen=True)
class StateSpec:
    """How a :class:`CharacterState` is shown: its main clip, optional fidget
    variants and the portrait used in speech bubbles ("" = none)."""

    clip: str
    variants: tuple[str, ...] = ()
    portrait: str = ""


# eq=False: the pack is an immutable bag of mappings; identity is the right
# notion of equality and keeps it hashable.
@dataclass(frozen=True, eq=False)
class CharacterPack:
    name: str
    reference_height: int
    directory: Path
    frames: Mapping[str, FrameInfo]
    clips: Mapping[str, Clip]
    states: Mapping[CharacterState, StateSpec]
    portraits: Mapping[str, Path]
    #: Clip played for states the pack does not map (the IDLE state's clip).
    idle_clip: str = "idle"
    description: str = ""

    # -- loading ------------------------------------------------------------------

    @classmethod
    def load(cls, directory: Path) -> CharacterPack:
        """Load and fully validate the pack in *directory*.  Raises :class:`PackError`."""
        directory = Path(directory)
        if not directory.is_dir():
            raise PackError(f"character pack directory not found: {directory}")
        root = directory.resolve()
        data = _read_manifest(root / MANIFEST_NAME)

        fmt = data.get("format")
        if fmt != PACK_FORMAT or isinstance(fmt, bool):
            raise PackError(f"{MANIFEST_NAME}: unsupported pack format {fmt!r} (expected {PACK_FORMAT})")

        name = data.get("name", root.name)
        if not isinstance(name, str) or not name.strip():
            raise PackError(f"{MANIFEST_NAME}: 'name' must be a non-empty string")
        description = data.get("description", "")
        if not isinstance(description, str):
            raise PackError(f"{MANIFEST_NAME}: 'description' must be a string")
        reference_height = _number(data.get("reference_height"), "'reference_height'")
        if reference_height < 1:
            raise PackError(f"{MANIFEST_NAME}: 'reference_height' must be >= 1, got {reference_height!r}")

        frames = _parse_frames(root, data.get("frames"))
        clips = _parse_clips(data.get("clips"), frames)
        portraits = _parse_portraits(root, data.get("portraits", {}))
        states = _parse_states(data.get("states"), clips, portraits)

        if CharacterState.IDLE in states:
            idle_clip = states[CharacterState.IDLE].clip
        elif "idle" in clips:
            idle_clip = "idle"
        else:
            raise PackError(f"{MANIFEST_NAME}: the pack needs an 'idle' state or an 'idle' clip "
                            "(it is the fallback for every state the pack does not map)")

        pack = cls(
            name=name.strip(),
            reference_height=round(reference_height),
            directory=root,
            frames=MappingProxyType(frames),
            clips=MappingProxyType(clips),
            states=MappingProxyType(states),
            portraits=MappingProxyType(portraits),
            idle_clip=idle_clip,
            description=description,
        )
        log.info("loaded character pack %r: %d frames, %d clips, %d states, %d portraits",
                 pack.name, len(frames), len(clips), len(states), len(portraits))
        return pack

    # -- queries ------------------------------------------------------------------

    def has_state(self, state: CharacterState) -> bool:
        return state in self.states

    def variants_for(self, state: CharacterState) -> tuple[str, ...]:
        spec = self.states.get(state)
        return spec.variants if spec is not None else ()

    def clip_for(self, state: CharacterState, variant: str | None = None) -> Clip:
        """The clip to play for *state*.

        *variant* selects one of the state's fidget clips (e.g. ``"idle_look"``)
        and is ignored if the state does not list it, so behavior code can ask
        for a variant without knowing the pack.  States the pack does not map
        fall back to the idle clip: a pack made for an older build still shows
        *something* for states added later.
        """
        spec = self.states.get(state)
        if spec is None:
            return self.clips[self.idle_clip]
        if variant is not None and variant in spec.variants:
            return self.clips[variant]
        return self.clips[spec.clip]

    def portrait_path(self, name: str) -> Path | None:
        return self.portraits.get(name) if name else None

    def portrait_for(self, state: CharacterState) -> Path | None:
        """The state's default portrait, if the pack has one."""
        spec = self.states.get(state)
        return self.portrait_path(spec.portrait) if spec is not None else None


# -- manifest parsing -------------------------------------------------------------------


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        raise PackError(f"{MANIFEST_NAME} not found in {path.parent}") from None
    except OSError as exc:
        raise PackError(f"cannot read {path}: {exc}") from exc
    if size > MAX_MANIFEST_BYTES:
        raise PackError(f"{path} is too large ({size} bytes)")
    try:
        # utf-8-sig: tolerate the BOM Windows editors like to add.
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise PackError(f"cannot read {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PackError(f"{MANIFEST_NAME} is not valid JSON: {exc.msg} "
                        f"(line {exc.lineno}, column {exc.colno})") from exc
    if not isinstance(data, dict):
        raise PackError(f"{MANIFEST_NAME} must contain a JSON object, got {type(data).__name__}")
    return data


def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PackError(f"{what} must be an object, got {_kind(value)}")
    return value


def _kind(value: Any) -> str:
    return "nothing" if value is None else type(value).__name__


def _number(value: Any, what: str) -> float:
    # bool is an int subclass; "loop": true where a number belongs is a typo, not 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PackError(f"{what} must be a finite number, got {value!r}")
    return float(value)


def _string(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise PackError(f"{what} must be a non-empty string, got {value!r}")
    return value


def _resolve_inside(root: Path, raw: Any, what: str) -> Path:
    """Resolve *raw* (a manifest path) inside the pack directory *root*.

    The check is done on both POSIX and Windows interpretations of the string
    so a pack that is safe on the build machine is safe on the user's too
    (``..\\..\\secret`` is harmless on Linux but escapes on Windows).
    """
    rel = _string(raw, f"{what}: 'file'")
    if "\x00" in rel:
        raise PackError(f"{what}: path {rel!r} contains a NUL character")
    win = PureWindowsPath(rel)
    posix = PurePosixPath(rel.replace("\\", "/"))
    if win.drive or win.root or posix.is_absolute():
        raise PackError(f"{what}: absolute path {rel!r} is not allowed; paths are relative to the pack")
    if ".." in posix.parts:
        raise PackError(f"{what}: path {rel!r} escapes the pack directory")
    path = root.joinpath(*posix.parts).resolve()
    # A symlink inside the pack could still point elsewhere: check the real path.
    if not path.is_relative_to(root):
        raise PackError(f"{what}: path {rel!r} resolves outside the pack directory")
    if not path.is_file():
        raise PackError(f"{what}: file {rel!r} not found in {root}")
    return path


def _parse_frames(root: Path, raw: Any) -> dict[str, FrameInfo]:
    frames_raw = _mapping(raw, "'frames'")
    if not frames_raw:
        raise PackError("'frames' is empty: a pack needs at least one frame")
    frames: dict[str, FrameInfo] = {}
    for frame_id, spec_raw in frames_raw.items():
        what = f"frame {frame_id!r}"
        spec = _mapping(spec_raw, what)
        path = _resolve_inside(root, spec.get("file"), what)
        anchor = spec.get("anchor")
        if not isinstance(anchor, list | tuple) or len(anchor) != 2:
            raise PackError(f"{what}: 'anchor' must be [x, y], got {anchor!r}")
        ax = _number(anchor[0], f"{what}: anchor x")
        ay = _number(anchor[1], f"{what}: anchor y")
        scale = _number(spec.get("scale", 1.0), f"{what}: 'scale'")
        if scale <= 0:
            raise PackError(f"{what}: 'scale' must be > 0, got {scale!r}")
        frames[frame_id] = FrameInfo(id=frame_id, path=path, anchor=(round(ax), round(ay)), scale=scale)
    return frames


def _parse_clips(raw: Any, frames: Mapping[str, FrameInfo]) -> dict[str, Clip]:
    clips_raw = _mapping(raw, "'clips'")
    if not clips_raw:
        raise PackError("'clips' is empty: a pack needs at least an idle clip")
    clips: dict[str, Clip] = {}
    for name, spec_raw in clips_raw.items():
        what = f"clip {name!r}"
        spec = _mapping(spec_raw, what)

        frame_list = spec.get("frames")
        if not isinstance(frame_list, list) or not frame_list:
            raise PackError(f"{what}: 'frames' must be a non-empty list")
        clip_frames: list[ClipFrame] = []
        for i, item_raw in enumerate(frame_list):
            item = _mapping(item_raw, f"{what} frame #{i}")
            frame_id = _string(item.get("id"), f"{what} frame #{i}: 'id'")
            if frame_id not in frames:
                raise PackError(f"{what} frame #{i}: unknown frame {frame_id!r} (not in 'frames')")
            ms = _number(item.get("ms"), f"{what} frame #{i}: 'ms'")
            clip_frames.append(ClipFrame(frame_id=frame_id, ms=round(ms)))

        loop = spec.get("loop", True)
        if not isinstance(loop, bool):
            raise PackError(f"{what}: 'loop' must be true or false, got {loop!r}")
        facing_raw = spec.get("facing", Facing.FRONT.value)
        try:
            facing = Facing(facing_raw)
        except ValueError:
            raise PackError(f"{what}: 'facing' must be one of "
                            f"{[f.value for f in Facing]}, got {facing_raw!r}") from None
        blend_ms = _number(spec.get("blend_ms", 0), f"{what}: 'blend_ms'")
        if blend_ms < 0:
            raise PackError(f"{what}: 'blend_ms' must be >= 0, got {blend_ms!r}")
        fx_raw = _mapping(spec.get("fx", {}), f"{what}: 'fx'")
        fx = {str(key): _number(value, f"{what}: fx {key!r}") for key, value in fx_raw.items()}
        overlay = spec.get("overlay")
        if overlay is not None and (not isinstance(overlay, str) or not overlay):
            raise PackError(f"{what}: 'overlay' must be a non-empty string or null, got {overlay!r}")

        clips[name] = Clip(
            name=name,
            frames=tuple(clip_frames),
            loop=loop,
            facing=facing,
            blend_ms=round(blend_ms),
            fx=MappingProxyType(fx),
            overlay=overlay,
        )
    return clips


def _parse_portraits(root: Path, raw: Any) -> dict[str, Path]:
    portraits_raw = _mapping(raw, "'portraits'")
    return {
        name: _resolve_inside(root, file, f"portrait {name!r}") for name, file in portraits_raw.items()
    }


def _parse_states(raw: Any, clips: Mapping[str, Clip],
                  portraits: Mapping[str, Path]) -> dict[CharacterState, StateSpec]:
    states_raw = _mapping(raw, "'states'")
    states: dict[CharacterState, StateSpec] = {}
    for key, spec_raw in states_raw.items():
        try:
            state = CharacterState(key)
        except ValueError:
            # A newer pack may describe states this build does not have yet.
            log.warning("character pack: skipping unknown state %r", key)
            continue
        what = f"state {key!r}"
        spec = _mapping(spec_raw, what)
        clip = _string(spec.get("clip"), f"{what}: 'clip'")
        if clip not in clips:
            raise PackError(f"{what}: unknown clip {clip!r} (not in 'clips')")
        variants_raw = spec.get("variants", [])
        if not isinstance(variants_raw, list):
            raise PackError(f"{what}: 'variants' must be a list of clip names, got {variants_raw!r}")
        variants: list[str] = []
        for variant_raw in variants_raw:
            variant = _string(variant_raw, f"{what}: variant")
            if variant not in clips:
                raise PackError(f"{what}: unknown variant clip {variant!r} (not in 'clips')")
            variants.append(variant)
        portrait = spec.get("portrait", "")
        if not isinstance(portrait, str):
            raise PackError(f"{what}: 'portrait' must be a string, got {portrait!r}")
        if portrait and portrait not in portraits:
            raise PackError(f"{what}: unknown portrait {portrait!r} (not in 'portraits')")
        states[state] = StateSpec(clip=clip, variants=tuple(variants), portrait=portrait)
    return states


# -- playback ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameSample:
    """What to draw right now.

    ``next_frame_id`` is set only while cross-fading; the renderer draws
    ``frame_id`` at ``1 - blend`` and ``next_frame_id`` at ``blend`` opacity.
    """

    frame_id: str
    next_frame_id: str | None
    blend: float  # 0..1 weight of next_frame_id
    finished: bool  # a non-loop clip reached its end (and holds its last frame)


class FramePlayer:
    """Plays one :class:`Clip`.  Clock-free: the caller feeds elapsed time.

    Looping clips wrap (and ``elapsed_ms`` is kept inside one cycle so a clip
    that loops for days does not lose float precision); non-loop clips hold
    their last frame and report ``finished`` so the controller knows a
    one-shot fidget is over.
    """

    def __init__(self, clip: Clip) -> None:
        if not clip.frames:
            raise ValueError(f"clip {clip.name!r} has no frames")
        self._clip = clip
        self._durations = [_effective_ms(f.ms) for f in clip.frames]
        self._starts: list[int] = []
        total = 0
        for duration in self._durations:
            self._starts.append(total)
            total += duration
        self._total = total
        self.elapsed_ms: float = 0.0

    @property
    def clip(self) -> Clip:
        """Read-only: create a new player to switch clips (timing tables are per clip)."""
        return self._clip

    @property
    def duration_ms(self) -> int:
        return self._total

    @property
    def finished(self) -> bool:
        return not self._clip.loop and self.elapsed_ms >= self._total

    @property
    def frame_index(self) -> int:
        """Index into ``clip.frames`` of the frame shown now."""
        if self.finished:
            return len(self._durations) - 1
        return bisect.bisect_right(self._starts, self._position()) - 1

    def restart(self) -> None:
        self.elapsed_ms = 0.0

    def advance(self, dt_ms: float) -> FrameSample:
        """Move time forward by *dt_ms* (any size; non-positive or NaN is ignored)."""
        if dt_ms > 0:  # also False for NaN
            if self._clip.loop:
                if math.isfinite(dt_ms):
                    self.elapsed_ms = math.fmod(self._position() + dt_ms, self._total)
            else:
                self.elapsed_ms = min(self.elapsed_ms + dt_ms, float(self._total))
        return self.sample()

    def sample(self) -> FrameSample:
        frames = self._clip.frames
        if self.finished:
            return FrameSample(frames[-1].frame_id, None, 0.0, True)
        t = self._position()
        index = bisect.bisect_right(self._starts, t) - 1
        duration = self._durations[index]
        into = t - self._starts[index]
        # Blending at most half a frame keeps every frame fully visible for a moment.
        window = min(float(self._clip.blend_ms), duration / 2)
        if window > 0 and into > duration - window:
            nxt = index + 1
            if nxt >= len(frames):
                nxt = 0 if self._clip.loop else -1
            if nxt >= 0 and frames[nxt].frame_id != frames[index].frame_id:
                blend = min(1.0, (into - (duration - window)) / window)
                return FrameSample(frames[index].frame_id, frames[nxt].frame_id, blend, False)
        return FrameSample(frames[index].frame_id, None, 0.0, False)

    def _position(self) -> float:
        """``elapsed_ms`` mapped into the clip's timeline (it is a public attribute,
        so it may have been set to anything)."""
        t = self.elapsed_ms
        if not math.isfinite(t) or t < 0:
            return 0.0
        if self._clip.loop:
            return math.fmod(t, self._total)
        return min(t, float(self._total))
