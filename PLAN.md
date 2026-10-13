# Chibi AI Companion — plan

A chibi girl who lives on the Windows desktop: walks, sits, sleeps, reacts to
the user's games and apps (only with permission), talks in Ukrainian by text
and voice, can look at an allowed game window, has tamagotchi needs, long-term
memory, gifts and mini-games.

## Architecture (short)

* Python 3.12+, PySide6, single process.  Details and module contracts:
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
* `EventBus` (bounded, thread-safe) connects producers (activity monitor,
  AI / voice / vision workers, window, timers) to main-thread consumers.  The
  Qt main thread never blocks; workers hand results back through queued
  signals.
* Rules decide *when* she speaks (budgets, cooldowns, quiet mode, the
  decision engine); the model only decides *how*.  Everything works without
  AI from the phrase bank.
* Privacy is architectural: one `PermissionManager` gate, everything off by
  default, a separate consent for every capability (app names, window titles,
  window capture, cloud vision, microphone, cloud AI), the privacy filter runs
  before anything is published, the kill switch revokes all observation, and
  each feature re-checks its permission at the moment it acts.
* Renderer behind a `CharacterRenderer` protocol (sprite pack, vector
  fallback; Live2D possible later) — logic only speaks `CharacterState`.

## Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Desktop MVP: sprite character, transparent no-focus window, drag, multi-monitor/DPI, bubbles with portraits, tray + context menu, state machine, autonomous behaviour, tamagotchi, SQLite, opt-in activity monitor, offline Ukrainian reactions with budget/cooldowns/quiet mode, settings/status/chat/first-run UI | **done** |
| 2 | AI brain: Claude (official SDK), OpenAI-compatible and Ollama providers (timeouts, retries, circuit breaker, streaming, caching, daily request/cost limits), keys in Windows Credential Manager, personality prompt from sliders, emotion engine, short/long-term memory with viewer/editor, AI phrasing of reaction triggers, autonomous decision engine | **done** |
| 3 | Screen vision: window capture (PrintWindow + foreground BitBlt fallback), per-app allow-list, capture indicator + instant stop, privacy re-check before every frame, black-frame detection, perceptual-hash dedup, rate/budget limits, local-first analysis (cloud only with its own consent, optional cheaper model), closed-vocabulary observations with two-frame confirmation, gaming companion mode | **done** |
| 4 | Voice: local Ukrainian STT (faster-whisper), push-to-talk via `RegisterHotKey` (toggle / hold), experimental wake phrase, Piper TTS on onnxruntime (no GPL) + optional Azure Neural, Ukrainian text normaliser, interruption, talking animation synced to audio, voice comments in games, verified model downloads | **done** |
| 5 | Polish: new states (jump, dance, excited, proud, stretch) and effects, idle variety, mini-games (rock-paper-scissors, cookie catch, number guess), gifts shop with a star wallet, redacted logs + crash hooks, PyInstaller one-folder `.exe`, per-user Inno Setup installer, `--self-test`, install docs | **done** |

Version **0.9.0**: feature-complete release candidate.  1.0 after the manual
check below.

## Verified

- [x] ~4550 automated tests (headless), ruff clean; CI on windows-latest and
      ubuntu with Python 3.12 / 3.13
- [x] Windows CI: Win32 activity calls, `PrintWindow` capture of a real window,
      `RegisterHotKey` registration, frozen `.exe` self-test + start/quit,
      installer build
- [x] Real engines: Piper TTS → Whisper STT round trip in Ukrainian; voice
      service end to end with the real models
- [x] Real LLM: the whole Phase 2 stack against a local OpenAI-compatible
      server (Qwen2.5-1.5B); its failure modes produced the action-evidence
      gate, grounded memories and HTML stripping
- [x] Adversarial reviews after Phases 1, 2 and 3-5; confirmed findings fixed
      with regression tests

## Needs a manual check on a real Windows PC

- [ ] Transparency / click-through, no focus stealing, hiding over fullscreen
      and borderless games, tray, multi-monitor with mixed DPI, CPU usage
- [ ] Real microphone and speakers (sounddevice / PortAudio), push-to-talk in
      a game, latency of `whisper-small` on the user's CPU
- [ ] Window capture of the user's actual games (some DirectX/Vulkan games
      return black frames to `PrintWindow`; she reports it instead of guessing)
- [ ] Claude / OpenAI with a real key, Azure TTS with a real key

## Known gaps / possible next steps

* Windows Graphics Capture instead of `PrintWindow` for games that return
  black frames (needs WinRT bindings; PrintWindow was the spec's allowed
  first step).
* A GitHub Release job on version tags, so the installer can be downloaded
  without signing in to GitHub; code signing to silence SmartScreen.
* Sprite frames are separate poses, not in-betweens: motion is pose-to-pose
  with crossfades and procedural motion.  Smooth animation needs proper sheets
  or Live2D.
* Perching on other windows' edges is not implemented.
* Offline chat intents are regex rules (~12 % misreads on fresh phrasing);
  with AI the model handles the conversation and rules stay as the fallback.
* Small local models (< ~7B) write poor Ukrainian; the docs recommend Claude
  or a 7-8B+ local model.
