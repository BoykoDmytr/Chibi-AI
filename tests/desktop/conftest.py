"""Shared fixtures for the desktop (Qt) tests.  Everything runs offscreen."""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from chibi.character.animation import CharacterPack
from chibi.character.movement import Rect
from chibi.core.config import Settings
from chibi.memory.repositories import StateRepository
from tests.desktop.helpers import MIA_DIR, FakeCursor, FakeScreens


@pytest.fixture(scope="session")
def mia_pack() -> CharacterPack:
    return CharacterPack.load(MIA_DIR)


@pytest.fixture
def tiny_pack_dir(tmp_path: Path) -> Path:
    """A minimal valid pack: two real frames, an idle clip without effects."""
    root = tmp_path / "tiny"
    (root / "frames").mkdir(parents=True)
    (root / "portraits").mkdir()
    for frame in ("front_0", "front_1"):
        shutil.copy(MIA_DIR / "frames" / f"{frame}.png", root / "frames" / f"{frame}.png")
    shutil.copy(MIA_DIR / "portraits" / "calm.png", root / "portraits" / "calm.png")
    source = json.loads((MIA_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "format": 1,
        "name": "Tiny",
        "reference_height": source["reference_height"],
        "frames": {k: source["frames"][k] for k in ("front_0", "front_1")},
        "clips": {"idle": {"frames": [{"id": "front_0", "ms": 1000}, {"id": "front_1", "ms": 1000}],
                           "loop": True, "facing": "front"}},
        "states": {"idle": {"clip": "idle", "portrait": "calm"}},
        "portraits": {"calm": "portraits/calm.png"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def make_controller(qapp, bus, clock, db, settings, mia_pack) -> Iterator[Callable[..., Any]]:
    """Build (and later shut down) controllers wired to fakes.  Started by default."""
    from chibi.desktop.controller import CharacterController, ScreenArea

    created: list[CharacterController] = []

    def build(*, pack: Any = "mia", start: bool = True, screens: FakeScreens | None = None,
              cursor: FakeCursor | None = None, needs: tuple[float, float, float, float] = (20, 80, 70, 20),
              repo: StateRepository | None = None, seed: int = 1, cfg: Settings | None = None,
              ) -> CharacterController:
        used_settings = cfg or settings
        ctrl = CharacterController(
            bus,
            lambda: used_settings,
            clock,
            repo or StateRepository(db, clock),
            mia_pack if pack == "mia" else pack,
            lambda: needs,
            random.Random(seed),
            cursor_pos=cursor or FakeCursor(),
            screens=screens or FakeScreens([ScreenArea("A", Rect(0, 0, 1920, 1040), primary=True)]),
        )
        created.append(ctrl)
        if start:
            ctrl.start()
        return ctrl

    yield build
    for ctrl in created:
        ctrl.shutdown()
