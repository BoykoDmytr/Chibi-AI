# Chibi AI Companion — plan

A chibi girl who lives on the Windows desktop: walks, sits, sleeps, reacts to
the user's games and apps (only with permission), talks in Ukrainian, has
tamagotchi needs, and later gains an AI brain, screen vision and voice.

## Architecture (short)

* Python 3.12+, PySide6, single process.  Details and module contracts:
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
* `EventBus` (bounded, thread-safe) connects producers (activity monitor
  thread, window, timers) to main-thread consumers (tamagotchi, reactions,
  character controller).  The Qt main thread never blocks.
* Deterministic game logic (needs, state machine, behavior, reaction budget)
  is plain Python with an injectable clock; AI (Phase 2) will only phrase
  lines, never decide every twitch.
* Privacy is architectural: one `PermissionManager` gate, everything off by
  default, unavailable features can't be granted, the privacy filter runs
  inside the monitor before anything is published, kill switch.
* Renderer is behind a `CharacterRenderer` protocol (sprite pack today, vector
  fallback; Live2D later) — logic only speaks `CharacterState`.

## Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Desktop MVP: sprite character with 21 states, transparent no-focus window, drag, multi-monitor/DPI, bubbles with portraits, tray + context menu, state machine, autonomous wandering/sitting/sleeping, tamagotchi needs, SQLite, opt-in activity monitor (games, IDEs, browsers, idle/return), offline Ukrainian reactions with budget/cooldowns/quiet mode, settings/status/chat/first-run UI, tests, CI on Windows | **done** (needs a manual check on real Windows) |
| 2 | AI brain: `TextProvider` for OpenAI / Anthropic / Ollama (timeouts, retries, cost & context limits, streaming, caching), keys in Windows Credential Manager, personality prompt from sliders, Emotion Engine (adds curiosity, excitement, trust), short/long-term memory with viewer/editor, AI phrasing of reaction triggers, Autonomous Decision Engine scoring | planned |
| 3 | Screen vision: Windows Graphics Capture adapter, per-app allow-list, capture indicator + instant stop, privacy check before capture, frame dedup (perceptual hash), rate & budget limits, `VisionProvider` (local first, cloud only with separate consent), validated observations (multi-frame confirmation), Gaming Companion mode | planned |
| 4 | Voice: local Ukrainian STT (faster-whisper), push-to-talk, optional wake phrase, TTS with Ukrainian female voice, interruption, talking animation sync, voice comments in games | planned |
| 5 | Polish: more animation, mini-games, gifts, optimisation, sanitized error reports, PyInstaller `.exe`, installer docs | planned |

## Phase 1 progress

- [x] Sprite pack built from the owner's sheets (`tools/build_sprites.py`)
- [x] Core: events, bus, clock, settings, SQLite + migrations, permissions
- [x] Character logic: animation player, state machine, movement, behavior
- [x] Desktop: window, renderers (sprite + vector fallback), overlays, bubble, tray, controller
- [x] Activity monitor (Windows backend via ctypes), classifier, games DB (85 games), privacy filter
- [x] Tamagotchi needs + offline catch-up (never punishes absence)
- [x] Ukrainian phrase bank (626 lines / 94 situations), reaction engine, offline chat
- [x] Settings / status / chat / first-run windows
- [x] App integration, ~2450 tests green, headless end-to-end smoke test, CPU measured
- [x] Six-lens adversarial review: 42 confirmed findings fixed, regression-checked
- [x] Offline chat: golden corpus (~500 cases) + two blind tests on fresh messages (32 % -> 12 % errors)
- [x] CI on windows-latest: full suite + native-window tests pass against the real Win32 API
- [ ] Manual check on a real Windows desktop: transparency/click-through, no focus stealing,
      fullscreen/borderless hiding, tray, multi-monitor with mixed DPI, real CPU usage

## Known gaps carried into Phase 2+

* While a fullscreen game runs she hides and stays quiet; game-start lines only
  show for windowed games (voice in Phase 4 fixes this).
* Sprite frames are separate AI-generated poses, not in-betweens: motion is
  pose-to-pose with crossfades.  Smooth animation needs proper sheets or Live2D.
* Perching on other windows' edges is not implemented yet.
* Offline chat intents are regex rules: ~12 % misreads on fresh phrasing in the
  last blind test (2.5 % wrong side effects).  Phase 2 should classify intents
  with the LLM (rules stay as the offline fallback).
* A game session flushed by revoke/shutdown right after the PC slept with the
  game open still counts the sleep time (statistics only).
