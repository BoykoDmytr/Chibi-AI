from __future__ import annotations

import json
import logging
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chibi.character.animation import (
    CharacterPack,
    Clip,
    ClipFrame,
    FramePlayer,
    FrameSample,
    PackError,
)
from chibi.character.states import CharacterState, Facing
from chibi.core.paths import character_pack_dir

MIA_DIR = character_pack_dir("mia")
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16  # existence is all the loader checks


# -- the real pack -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def mia() -> CharacterPack:
    return CharacterPack.load(MIA_DIR)


@pytest.fixture(scope="module")
def mia_manifest() -> dict[str, Any]:
    return json.loads((MIA_DIR / "manifest.json").read_text(encoding="utf-8"))


def test_mia_pack_loads(mia: CharacterPack, mia_manifest: dict[str, Any]) -> None:
    assert mia.name == "Mia"
    assert mia.reference_height == mia_manifest["reference_height"] > 0
    assert mia.directory == MIA_DIR.resolve()
    assert set(mia.frames) == set(mia_manifest["frames"])
    assert set(mia.clips) == set(mia_manifest["clips"])
    assert set(mia.portraits) == set(mia_manifest["portraits"])
    for frame_id, info in mia.frames.items():
        assert info.id == frame_id
        assert info.path.is_file()
        assert info.path.is_relative_to(mia.directory)
        assert info.anchor == tuple(mia_manifest["frames"][frame_id]["anchor"])
        assert info.scale > 0


def test_mia_maps_every_character_state(mia: CharacterPack, mia_manifest: dict[str, Any]) -> None:
    assert len(CharacterState) == 26
    assert set(mia.states) == set(CharacterState)
    for state in CharacterState:
        spec = mia_manifest["states"][state.value]
        clip = mia.clip_for(state)
        assert clip.name == spec["clip"]
        assert clip.frames and clip.duration_ms > 0
        assert all(f.frame_id in mia.frames for f in clip.frames)
        portrait = mia.portrait_path(spec["portrait"])
        assert portrait is not None and portrait.is_file()
        assert mia.portrait_for(state) == portrait
        for variant in spec.get("variants", []):
            assert mia.clip_for(state, variant).name == variant
        # every clip of the real pack is playable
        player = FramePlayer(clip)
        assert player.advance(clip.duration_ms * 2.5).frame_id in mia.frames


def test_mia_clip_details(mia: CharacterPack) -> None:
    sit = mia.clips["sit"]
    assert sit.loop and sit.facing is Facing.FRONT and sit.blend_ms == 220
    assert sit.fx == {"breathe": 0.010}
    assert mia.clips["walk"].facing is Facing.LEFT
    assert mia.clips["walk"].duration_ms == 8 * 110
    assert mia.clips["run"].facing is Facing.RIGHT
    assert mia.clips["happy"].overlay == "sparkle"
    assert mia.clips["walk"].overlay is None
    assert not mia.clips["idle_look"].loop
    assert mia.clips["dragged"].fx == {"swing": 9.0}


def test_mia_variants(mia: CharacterPack) -> None:
    assert mia.variants_for(CharacterState.IDLE) == ("idle_look", "idle_smile")
    assert mia.clip_for(CharacterState.IDLE, "idle_smile").name == "idle_smile"
    assert mia.clip_for(CharacterState.WAVE, "greet").name == "greet"
    # a variant of another state, or an unknown one, plays the state's main clip
    assert mia.clip_for(CharacterState.IDLE, "greet").name == "idle"
    assert mia.clip_for(CharacterState.IDLE, "nope").name == "idle"
    assert mia.clip_for(CharacterState.IDLE, None).name == "idle"
    assert mia.variants_for(CharacterState.TALK) == ()


def test_mia_clips_are_hashable_and_immutable(mia: CharacterPack) -> None:
    clip = mia.clips["idle"]
    assert hash(clip) == hash(mia.clips["idle"])
    with pytest.raises(TypeError):
        clip.fx["breathe"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        mia.clips["x"] = clip  # type: ignore[index]


def test_portrait_path_unknown(mia: CharacterPack) -> None:
    assert mia.portrait_path("no_such_portrait") is None
    assert mia.portrait_path("") is None


# -- crafted packs ------------------------------------------------------------------------


def _base_manifest() -> dict[str, Any]:
    return {
        "format": 1,
        "name": "Test",
        "reference_height": 400,
        "frames": {
            "a": {"file": "frames/a.png", "anchor": [10, 90], "scale": 1.0},
            "b": {"file": "frames/b.png", "anchor": [12.6, 88], "scale": 0.5},
        },
        "clips": {
            "idle": {"frames": [{"id": "a", "ms": 500}, {"id": "b", "ms": 300}], "loop": True,
                     "facing": "front", "blend_ms": 100, "fx": {"breathe": 0.01}},
            "idle_look": {"frames": [{"id": "b", "ms": 800}], "loop": False, "facing": "left"},
            "wave": {"frames": [{"id": "a", "ms": 200}], "loop": False, "overlay": "sparkle"},
        },
        "states": {
            "idle": {"clip": "idle", "variants": ["idle_look"], "portrait": "calm"},
            "wave": {"clip": "wave", "portrait": "calm"},
        },
        "portraits": {"calm": "portraits/calm.png"},
    }


def make_pack(tmp_path: Path, mutate: Callable[[dict[str, Any]], Any] | None = None,
              files: tuple[str, ...] = ("frames/a.png", "frames/b.png", "portraits/calm.png"),
              raw: str | None = None) -> Path:
    root = tmp_path / "pack"
    for rel in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG)
    root.mkdir(parents=True, exist_ok=True)
    if raw is None:
        manifest = _base_manifest()
        if mutate is not None:
            mutate(manifest)
        raw = json.dumps(manifest)
    (root / "manifest.json").write_text(raw, encoding="utf-8")
    return root


def test_crafted_pack_loads_with_defaults(tmp_path: Path) -> None:
    pack = CharacterPack.load(make_pack(tmp_path))
    assert pack.name == "Test" and pack.reference_height == 400
    assert pack.frames["b"].anchor == (13, 88)  # rounded to whole pixels
    assert pack.frames["b"].scale == 0.5
    wave = pack.clips["wave"]
    assert wave.facing is Facing.FRONT  # default
    assert wave.blend_ms == 0 and dict(wave.fx) == {} and wave.overlay == "sparkle"
    assert wave.frames == (ClipFrame("a", 200),)
    assert pack.clips["idle_look"].facing is Facing.LEFT
    assert pack.portrait_path("calm") == (tmp_path / "pack" / "portraits" / "calm.png").resolve()


def test_unmapped_state_falls_back_to_idle(tmp_path: Path) -> None:
    pack = CharacterPack.load(make_pack(tmp_path))
    assert not pack.has_state(CharacterState.SAD)
    assert pack.clip_for(CharacterState.SAD).name == "idle"
    assert pack.clip_for(CharacterState.SAD, "idle_look").name == "idle"
    assert pack.portrait_for(CharacterState.SAD) is None
    assert pack.clip_for(CharacterState.WAVE).name == "wave"


def test_idle_fallback_uses_idle_state_clip(tmp_path: Path) -> None:
    def mutate(m: dict[str, Any]) -> None:
        m["clips"]["calm"] = m["clips"].pop("idle")
        m["states"]["idle"]["clip"] = "calm"

    pack = CharacterPack.load(make_pack(tmp_path, mutate))
    assert pack.idle_clip == "calm"
    assert pack.clip_for(CharacterState.ANGRY).name == "calm"


def test_idle_clip_without_idle_state_is_enough(tmp_path: Path) -> None:
    pack = CharacterPack.load(make_pack(tmp_path, lambda m: m["states"].pop("idle")))
    assert pack.clip_for(CharacterState.IDLE).name == "idle"


def test_unknown_states_are_skipped_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    def mutate(m: dict[str, Any]) -> None:
        m["states"]["moonwalk"] = {"clip": "no_such_clip_either"}

    with caplog.at_level(logging.WARNING, logger="chibi.character.animation"):
        pack = CharacterPack.load(make_pack(tmp_path, mutate))
    assert set(pack.states) == {CharacterState.IDLE, CharacterState.WAVE}
    assert "moonwalk" in caplog.text


def test_manifest_with_bom_is_accepted(tmp_path: Path) -> None:
    root = make_pack(tmp_path)
    text = (root / "manifest.json").read_text(encoding="utf-8")
    (root / "manifest.json").write_text("\ufeff" + text, encoding="utf-8")
    assert CharacterPack.load(root).name == "Test"


def _set(path: str, value: Any) -> Callable[[dict[str, Any]], None]:
    """Mutation setting manifest[path...] = value; path segments separated by '/'."""

    def mutate(m: dict[str, Any]) -> None:
        *parents, last = path.split("/")
        node: Any = m
        for key in parents:
            node = node[int(key)] if isinstance(node, list) else node[key]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value

    return mutate


def _delete(path: str) -> Callable[[dict[str, Any]], None]:
    def mutate(m: dict[str, Any]) -> None:
        *parents, last = path.split("/")
        node: Any = m
        for key in parents:
            node = node[key]
        del node[last]

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_set("format", 2), "unsupported pack format 2"),
        (_delete("format"), "unsupported pack format None"),
        (_set("format", True), "unsupported pack format True"),
        (_set("name", ""), "'name'"),
        (_delete("reference_height"), "'reference_height' must be a finite number"),
        (_set("reference_height", 0), "'reference_height' must be >= 1"),
        (_set("frames", {}), "'frames' is empty"),
        (_set("frames", []), "'frames' must be an object"),
        (_set("frames/a/anchor", [1]), "frame 'a': 'anchor' must be [x, y]"),
        (_set("frames/a/anchor", [1, "x"]), "frame 'a': anchor y must be a finite number"),
        (_set("frames/a/scale", 0), "frame 'a': 'scale' must be > 0"),
        (_delete("frames/a/file"), "frame 'a': 'file' must be a non-empty string"),
        (_set("frames/a/file", "frames/missing.png"), "frame 'a': file 'frames/missing.png' not found"),
        (_set("clips/idle/frames/0/id", "zzz"), "clip 'idle' frame #0: unknown frame 'zzz'"),
        (_set("clips/idle/frames/0/ms", "fast"), "clip 'idle' frame #0: 'ms' must be a finite number"),
        (_set("clips/idle/frames/0/ms", True), "'ms' must be a finite number"),
        (_set("clips/idle/frames", []), "clip 'idle': 'frames' must be a non-empty list"),
        (_set("clips/idle/loop", "yes"), "clip 'idle': 'loop' must be true or false"),
        (_set("clips/idle/facing", "up"), "clip 'idle': 'facing' must be one of"),
        (_set("clips/idle/blend_ms", -1), "clip 'idle': 'blend_ms' must be >= 0"),
        (_set("clips/idle/fx", {"bob": "big"}), "clip 'idle': fx 'bob' must be a finite number"),
        (_set("clips/idle/overlay", 5), "clip 'idle': 'overlay' must be a non-empty string or null"),
        (_set("clips", {}), "'clips' is empty"),
        (_set("states/idle/clip", "nope"), "state 'idle': unknown clip 'nope'"),
        (_set("states/idle/variants", ["nope"]), "state 'idle': unknown variant clip 'nope'"),
        (_set("states/idle/variants", "idle_look"), "state 'idle': 'variants' must be a list"),
        (_set("states/wave/portrait", "grin"), "state 'wave': unknown portrait 'grin'"),
        (_set("states/wave/portrait", 3), "state 'wave': 'portrait' must be a string"),
        (_set("states", None), "'states' must be an object, got nothing"),
        (_set("portraits/calm", "portraits/missing.png"), "portrait 'calm': file 'portraits/missing.png'"),
    ],
)
def test_validation_errors(tmp_path: Path, mutate: Callable[[dict[str, Any]], None], message: str) -> None:
    with pytest.raises(PackError) as err:
        CharacterPack.load(make_pack(tmp_path, mutate))
    assert message in str(err.value)


def test_pack_without_any_idle_is_rejected(tmp_path: Path) -> None:
    def mutate(m: dict[str, Any]) -> None:
        del m["states"]["idle"]
        m["clips"]["calm"] = m["clips"].pop("idle")

    with pytest.raises(PackError, match="needs an 'idle' state or an 'idle' clip"):
        CharacterPack.load(make_pack(tmp_path, mutate))


def test_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(PackError, match="directory not found"):
        CharacterPack.load(tmp_path / "nope")


def test_missing_manifest(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(PackError, match=r"manifest\.json not found"):
        CharacterPack.load(tmp_path / "empty")


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("{not json", "not valid JSON"),
        ("", "not valid JSON"),
        ("[1, 2]", "must contain a JSON object, got list"),
    ],
)
def test_invalid_json(tmp_path: Path, raw: str, message: str) -> None:
    with pytest.raises(PackError, match=message):
        CharacterPack.load(make_pack(tmp_path, raw=raw))


def test_non_utf8_manifest(tmp_path: Path) -> None:
    root = make_pack(tmp_path)
    (root / "manifest.json").write_bytes(b'{"name": "\xff\xfe"}')
    with pytest.raises(PackError, match="cannot read"):
        CharacterPack.load(root)


# -- path traversal guard -------------------------------------------------------------


@pytest.fixture
def secret(tmp_path: Path) -> Path:
    """A real file just outside the pack, so only the guard can stop the load."""
    path = tmp_path / "secret.png"
    path.write_bytes(PNG)
    return path


@pytest.mark.parametrize(
    "bad",
    [
        "../secret.png",
        "frames/../../secret.png",
        "..\\secret.png",
        "frames\\..\\..\\secret.png",
    ],
)
def test_relative_traversal_is_rejected(tmp_path: Path, secret: Path, bad: str) -> None:
    with pytest.raises(PackError, match="escapes the pack directory"):
        CharacterPack.load(make_pack(tmp_path, _set("frames/a/file", bad)))


@pytest.mark.parametrize("bad", ["/etc/passwd", "C:\\Windows\\win.ini", "C:secret.png",
                                 "\\\\server\\share\\x.png", "\\secret.png"])
def test_absolute_paths_are_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(PackError, match="absolute path"):
        CharacterPack.load(make_pack(tmp_path, _set("frames/a/file", bad)))


def test_absolute_path_to_existing_file_is_rejected(tmp_path: Path, secret: Path) -> None:
    with pytest.raises(PackError, match="absolute path"):
        CharacterPack.load(make_pack(tmp_path, _set("portraits/calm", str(secret))))


def test_portrait_traversal_is_rejected(tmp_path: Path, secret: Path) -> None:
    with pytest.raises(PackError, match=r"portrait 'calm'.*escapes"):
        CharacterPack.load(make_pack(tmp_path, _set("portraits/calm", "../secret.png")))


def test_symlink_escaping_the_pack_is_rejected(tmp_path: Path, secret: Path) -> None:
    root = make_pack(tmp_path)
    link = root / "frames" / "link.png"
    try:
        os.symlink(secret, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")
    manifest = _base_manifest()
    manifest["frames"]["a"]["file"] = "frames/link.png"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PackError, match="resolves outside the pack"):
        CharacterPack.load(root)


@pytest.mark.skipif(sys.platform == "win32", reason="backslash is a separator on Windows")
def test_backslash_paths_inside_the_pack_work(tmp_path: Path) -> None:
    """Pack authors on Windows write backslashes; they must resolve on every OS."""
    pack = CharacterPack.load(make_pack(tmp_path, _set("frames/a/file", "frames\\a.png")))
    assert pack.frames["a"].path == (tmp_path / "pack" / "frames" / "a.png").resolve()


# -- FramePlayer ----------------------------------------------------------------------------


def make_clip(*frames: tuple[str, int], loop: bool = True, blend_ms: int = 0) -> Clip:
    return Clip(name="test", frames=tuple(ClipFrame(f, ms) for f, ms in frames), loop=loop,
                facing=Facing.FRONT, blend_ms=blend_ms)


ABC = (("a", 100), ("b", 200), ("c", 300))


def at(clip: Clip, ms: float) -> FrameSample:
    player = FramePlayer(clip)
    return player.advance(ms)


def test_player_starts_on_first_frame() -> None:
    player = FramePlayer(make_clip(*ABC))
    assert player.sample() == FrameSample("a", None, 0.0, False)
    assert player.elapsed_ms == 0.0 and not player.finished
    assert player.duration_ms == 600 == player.clip.duration_ms


@pytest.mark.parametrize(
    ("ms", "frame"),
    [(0, "a"), (99.9, "a"), (100, "b"), (299.9, "b"), (300, "c"), (599.9, "c"),
     (600, "a"), (650, "a"), (700, "b"), (600 * 1000 + 350, "c")],
)
def test_loop_timing_and_wrap(ms: float, frame: str) -> None:
    sample = at(make_clip(*ABC), ms)
    assert sample.frame_id == frame and not sample.finished


def test_loop_keeps_elapsed_within_one_cycle() -> None:
    player = FramePlayer(make_clip(*ABC))
    for _ in range(10_000):
        player.advance(37)
    assert 0 <= player.elapsed_ms < 600
    assert player.elapsed_ms == pytest.approx((37 * 10_000) % 600)


def test_incremental_advance_matches_single_advance() -> None:
    clip = make_clip(*ABC, blend_ms=40)
    stepped = FramePlayer(clip)
    for steps in range(1, 200):
        got = stepped.advance(13)
        want = at(clip, steps * 13)
        assert (got.frame_id, got.next_frame_id, got.finished) == (
            want.frame_id, want.next_frame_id, want.finished)
        assert got.blend == pytest.approx(want.blend, abs=1e-9)


def test_non_loop_holds_last_frame_and_finishes() -> None:
    player = FramePlayer(make_clip(*ABC, loop=False))
    assert player.advance(599).frame_id == "c" and not player.finished
    sample = player.advance(1)
    assert sample == FrameSample("c", None, 0.0, True)
    assert player.finished and player.elapsed_ms == 600
    assert player.advance(10_000) == FrameSample("c", None, 0.0, True)
    assert player.frame_index == 2


def test_non_loop_dt_larger_than_clip() -> None:
    assert at(make_clip(*ABC, loop=False), 5_000) == FrameSample("c", None, 0.0, True)


def test_restart() -> None:
    player = FramePlayer(make_clip(*ABC, loop=False))
    player.advance(1_000)
    player.restart()
    assert not player.finished and player.elapsed_ms == 0
    assert player.sample().frame_id == "a"


@pytest.mark.parametrize("ms", [0, -5])
def test_non_positive_frame_durations_count_as_one_ms(ms: int) -> None:
    clip = make_clip(("a", ms), ("b", ms), ("c", 10))
    assert clip.duration_ms == 12
    player = FramePlayer(clip)
    assert player.sample().frame_id == "a"
    assert player.advance(1).frame_id == "b"
    assert player.advance(1).frame_id == "c"
    assert player.advance(10).frame_id == "a"  # wrapped


@pytest.mark.parametrize("dt", [0, -10, math.nan, -math.inf])
def test_invalid_dt_is_ignored(dt: float) -> None:
    player = FramePlayer(make_clip(*ABC))
    player.advance(150)
    assert player.advance(dt).frame_id == "b"
    assert player.elapsed_ms == 150


def test_infinite_dt() -> None:
    looping = FramePlayer(make_clip(*ABC))
    looping.advance(150)
    assert looping.advance(math.inf).frame_id == "b"  # cannot place infinity in a loop: ignored
    once = FramePlayer(make_clip(*ABC, loop=False))
    assert once.advance(math.inf).finished


def test_empty_clip_is_rejected() -> None:
    with pytest.raises(ValueError):
        FramePlayer(make_clip())


def test_externally_set_elapsed_is_tolerated() -> None:
    player = FramePlayer(make_clip(*ABC))
    player.elapsed_ms = -50
    assert player.sample().frame_id == "a"
    player.elapsed_ms = 1_350  # beyond one cycle
    assert player.sample().frame_id == "b"
    assert player.frame_index == 1


def test_blend_rises_towards_next_frame() -> None:
    clip = make_clip(("a", 1000), ("b", 1000), blend_ms=200)
    assert at(clip, 799) == FrameSample("a", None, 0.0, False)
    assert at(clip, 800) == FrameSample("a", None, 0.0, False)
    half = at(clip, 900)
    assert half.frame_id == "a" and half.next_frame_id == "b"
    assert half.blend == pytest.approx(0.5)
    assert at(clip, 999).blend == pytest.approx(0.995)
    assert at(clip, 1000) == FrameSample("b", None, 0.0, False)
    blends = [at(clip, ms).blend for ms in range(801, 1000, 10)]
    assert blends == sorted(blends) and all(0 < b < 1 for b in blends)


def test_blend_wraps_for_loops() -> None:
    clip = make_clip(("a", 1000), ("b", 1000), blend_ms=200)
    sample = at(clip, 1900)
    assert sample.frame_id == "b" and sample.next_frame_id == "a"
    assert sample.blend == pytest.approx(0.5)


def test_blend_never_past_end_of_non_loop() -> None:
    clip = make_clip(("a", 1000), ("b", 1000), loop=False, blend_ms=200)
    assert at(clip, 900).next_frame_id == "b"
    assert at(clip, 1950) == FrameSample("b", None, 0.0, False)


def test_blend_window_capped_at_half_a_frame() -> None:
    clip = make_clip(("a", 100), ("b", 100), blend_ms=500)
    assert at(clip, 40).next_frame_id is None
    assert at(clip, 50).next_frame_id is None
    assert at(clip, 75).blend == pytest.approx(0.5)
    assert at(clip, 75).next_frame_id == "b"


def test_no_blend_between_identical_frames() -> None:
    single = make_clip(("a", 1000), blend_ms=300)
    assert at(single, 950) == FrameSample("a", None, 0.0, False)
    repeat = make_clip(("a", 500), ("a", 500), ("b", 500), blend_ms=100)
    assert at(repeat, 450).next_frame_id is None
    assert at(repeat, 950).next_frame_id == "b"


def test_no_blend_when_disabled() -> None:
    assert at(make_clip(*ABC), 99).next_frame_id is None


def test_player_clip_is_read_only() -> None:
    player = FramePlayer(make_clip(*ABC))
    with pytest.raises(AttributeError):
        player.clip = make_clip(("z", 1))  # type: ignore[misc]


# -- FramePlayer.ms_to_next_change (frame-accurate ticks at low frame rates) ----------------------


@pytest.mark.parametrize(("ms", "due"), [(0, 100), (40, 60), (100, 200), (299, 1), (300, 300), (550, 50)])
def test_ms_to_next_change_in_a_loop(ms: float, due: float) -> None:
    player = FramePlayer(make_clip(*ABC))
    player.advance(ms)
    assert player.ms_to_next_change() == pytest.approx(due)


def test_ms_to_next_change_skips_repeated_images() -> None:
    # a b a(100) a(100): from the third frame the next *different* image is b, after the wrap
    player = FramePlayer(make_clip(("a", 100), ("b", 50), ("a", 100), ("a", 100)))
    player.advance(150)
    assert player.ms_to_next_change() == pytest.approx(200 + 100)


def test_ms_to_next_change_points_at_the_start_of_a_cross_fade() -> None:
    player = FramePlayer(make_clip(("a", 100), ("b", 100), blend_ms=40))
    assert player.ms_to_next_change() == pytest.approx(60)
    player.advance(70)  # inside the cross-fade: the picture changes continuously
    assert player.ms_to_next_change() is None
    player.advance(40)
    assert player.ms_to_next_change() == pytest.approx(50)


def test_ms_to_next_change_for_one_shots_and_static_clips() -> None:
    player = FramePlayer(make_clip(*ABC, loop=False))
    player.advance(450)
    assert player.ms_to_next_change() == pytest.approx(150)  # the clip ends: a fidget hands back
    player.advance(150)
    assert player.finished and player.ms_to_next_change() is None
    assert FramePlayer(make_clip(("a", 500))).ms_to_next_change() is None
    assert FramePlayer(make_clip(("a", 500), ("a", 300))).ms_to_next_change() is None
    single_shot = FramePlayer(make_clip(("a", 500), loop=False))
    assert single_shot.ms_to_next_change() == pytest.approx(500)


def test_ms_to_next_change_lands_on_the_mia_blinks(mia: CharacterPack) -> None:
    player = FramePlayer(mia.clips["idle"])
    blinks = []
    for _ in range(40):
        due = player.ms_to_next_change()
        assert due is not None and due > 0
        player.advance(due)
        if player.sample().frame_id in ("front_1", "front_6"):
            blinks.append((player.sample().frame_id, player.ms_to_next_change()))
    # every blink is reached exactly at its start and lasts its full length
    assert blinks and {ms for _, ms in blinks} <= {160.0, 140.0, 170.0}


# -- Phase 5 states in the real pack --------------------------------------------------------------


def test_mia_new_states(mia: CharacterPack) -> None:
    expected = {
        CharacterState.JUMP: ("jump", "joyful"),
        CharacterState.DANCE: ("dance", "music"),
        CharacterState.EXCITED: ("excited", "excited"),
        CharacterState.PROUD: ("proud", "wink"),
        CharacterState.STRETCH: ("stretch", "smile"),
    }
    for state, (clip, portrait) in expected.items():
        spec = mia.states[state]
        assert (spec.clip, spec.portrait) == (clip, portrait)
        assert mia.portrait_for(state) is not None and mia.portrait_for(state).is_file()
    assert mia.clips["jump"].fx == {"jump": 0.12, "squash": 0.10}
    assert mia.clips["dance"].overlay == "hearts"
    assert mia.clips["excited"].overlay == "sparkle"
    assert not mia.clips["stretch"].loop


def test_manifest_matches_the_build_tool() -> None:
    """manifest.json's clips and states are generated from tools/build_sprites.py's
    tables (the raw sheets are not in the repository, so the manifest is rewritten
    from them): both must say the same."""
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    pytest.importorskip("scipy")
    import build_sprites

    manifest = json.loads((MIA_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["clips"] == build_sprites.CLIPS
    assert manifest["states"] == build_sprites.STATES
    used = {f["id"] for clip in build_sprites.CLIPS.values() for f in clip["frames"]}
    assert used == set(manifest["frames"])  # the new clips need no new art
