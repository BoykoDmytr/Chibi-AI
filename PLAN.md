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
| 1 | Desktop MVP: sprite character with 21 states, transparent no-focus window, drag, multi-monitor/DPI, bubbles with portraits, tray + context menu, state machine, autonomous wandering/sitting/sleeping, tamagotchi needs, SQLite, opt-in activity monitor (games, IDEs, browsers, idle/return), offline Ukrainian reactions with budget/cooldowns/quiet mode, settings/status/chat/first-run UI, tests, CI on Windows | **in progress** |
| 2 | AI brain: `TextProvider` for OpenAI / Anthropic / Ollama (timeouts, retries, cost & context limits, streaming, caching), keys in Windows Credential Manager, personality prompt from sliders, Emotion Engine (adds curiosity, excitement, trust), short/long-term memory with viewer/editor, AI phrasing of reaction triggers, Autonomous Decision Engine scoring | planned |
| 3 | Screen vision: Windows Graphics Capture adapter, per-app allow-list, capture indicator + instant stop, privacy check before capture, frame dedup (perceptual hash), rate & budget limits, `VisionProvider` (local first, cloud only with separate consent), validated observations (multi-frame confirmation), Gaming Companion mode | planned |
| 4 | Voice: local Ukrainian STT (faster-whisper), push-to-talk, optional wake phrase, TTS with Ukrainian female voice, interruption, talking animation sync, voice comments in games | planned |
| 5 | Polish: more animation, mini-games, gifts, optimisation, sanitized error reports, PyInstaller `.exe`, installer docs | planned |

## Phase 1 progress

- [x] Sprite pack built from the owner's sheets (`tools/build_sprites.py`)
- [x] Core: events, bus, clock, settings, SQLite + migrations, permissions
- [ ] Character logic: animation player, state machine, movement, behavior
- [ ] Desktop: window, renderers (sprite + vector fallback), overlays, bubble, tray, controller
- [ ] Activity monitor (Windows backend), classifier, games DB, privacy filter
- [ ] Tamagotchi needs + offline catch-up (never punishes absence)
- [ ] Ukrainian phrase bank, reaction engine, offline chat
- [ ] Settings / status / chat / first-run windows
- [ ] App integration, tests green, headless smoke test
- [ ] Verified on real Windows (CI covers Win32 API calls; UI behaviour needs a manual check)
