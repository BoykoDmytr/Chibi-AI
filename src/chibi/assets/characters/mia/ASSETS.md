# Character pack "mia"

* **Source art**: nine PNG sprite sheets supplied by the project owner
  (generated with ChatGPT image generation, Oct 2026).  Not third-party
  sprites.  The raw sheets are not committed (22 MB); the frames here were
  produced from them by `tools/build_sprites.py`.
* **Body frames** (`frames/`): 59 frames from the eight 4x2 full-body sheets
  (front/idle, walk, run, gestures, sleep, wave, cheer, sit), separated by
  nearest-body assignment, trimmed, anchored at the feet; blink frames are
  re-aligned to their partner frame.
* **Portraits** (`portraits/`): 27 waist-up crops from the labelled
  "emotions and poses" master sheet (angry, offended, shy, eat, drink,
  typing, idea, ...).  Used in speech bubbles and windows only: the master
  cells are cut by their borders and are ~2.7x smaller than the body frames,
  so they would look sliced-off and blurry on the desktop.
* **manifest.json**: frames (file, anchor, scale), clips (frames + durations,
  loop, facing, crossfade, procedural fx, overlay), state -> clip mapping and
  portraits.  Edit it to retime animations; re-run the tool to regenerate.

To rebuild after replacing the sheets:

    pip install -e .[dev]
    python tools/build_sprites.py --src <folder with the 9 PNGs> --out src/chibi/assets/characters/mia
