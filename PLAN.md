# Chibi AI Companion — plan

A chibi girl who lives on the Windows desktop: walks, sits, sleeps, reacts to
the user's games and apps (only with permission), talks in Ukrainian by text
and voice, can look at the window in front, has tamagotchi needs, long-term
memory, gifts and mini-games - and, as a voice companion, listens hands-free,
plays a character the user describes and decides herself when what she sees
is worth a word.

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
| 6 | Voice companion: hands-free listening (push / name anywhere in a phrase / open), conversation window, automatic microphone pauses + a menu switch, answers voiced sentence by sentence while the model writes, one answer at a time for voice requests; the user's own character description in the prompt; vision of any non-private foreground window with work scenes and stricter redaction; observer mode (she decides when to speak, mostly silence, shared budget) | **done** (needs the manual check below) |

Version **0.9.0**: feature-complete release candidate.  1.0 after the manual
check below.

## Verified

- [x] ~4900 automated tests (headless), ruff clean; CI on windows-latest and
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

## Phase 6 checklist (voice companion)

- [x] Settings contract: `personality.persona`, `voice.listen_mode` /
      `follow_up_seconds` / `pause_in_calls` / `wake_in_fullscreen`,
      `vision.scope` / `vision.commentary`
- [x] Voice: name found anywhere in a phrase (Ukrainian vocatives), open mode,
      conversation window, filler / echo / hallucination filters, single-flight
      recognition, she never hears herself, microphone pauses (user switch,
      private app, call app, DND, fullscreen), honest consent per mode
- [x] Streamed answers: `ReplyStream` (sentence by sentence, said exactly once,
      cut-off shown as text); wired for typed and spoken chat
- [x] One chat answer at a time for voice requests (newest waits; lost answers
      time out)
- [x] "Пауза мікрофона" in her menu and the tray; the tray tooltip says why
      the microphone is paused
- [x] Persona: "Характер своїми словами" editor with presets; quoted,
      neutralised prompt section; rules win
- [x] Vision: scope "foreground" (private / system / shell / browser rules),
      work scenes and events, stricter description redaction, re-check when
      the answer arrives, pacing and away hold, `VisionObserved` for every
      analysed frame
- [x] Observer: in-memory log, strict gates and budgets, `consider_comment`
      with strict parsing, stale decisions dropped, forgotten on revoke /
      kill switch / wipe
- [x] README (modes, trade-offs, privacy, cost estimates), ARCHITECTURE
- [ ] Manual check on Windows (see below)

## Needs a manual check on a real Windows PC

- [ ] Transparency / click-through, no focus stealing, hiding over fullscreen
      and borderless games, tray, multi-monitor with mixed DPI, CPU usage
- [ ] Real microphone and speakers (sounddevice / PortAudio), push-to-talk in
      a game, latency of `whisper-small` on the user's CPU
- [ ] Window capture of the user's actual games (some DirectX/Vulkan games
      return black frames to `PrintWindow`; she reports it instead of guessing)
- [ ] Claude / OpenAI with a real key, Azure TTS with a real key
- [ ] Phase 6: false-trigger rate of "name" mode with a TV or other people in
      the room; VAD with background noise; CPU load of the open microphone;
      call apps and fullscreen pausing; latency to the first spoken sentence
- [ ] Phase 6: real vision models (Claude, `gemma3` / `qwen2.5vl` in Ollama)
      on work windows; how often the observer stays silent and whether it ever
      says something not on screen; whether models keep to the persona and
      its rules; Windows shell process names for scope "foreground"

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
* Voice is turn-based, not full duplex: the microphone is closed while she
  speaks (no barge-in by voice; a click or push-to-talk interrupts).  Echo
  cancellation would allow it.
* Call apps are detected only in the foreground; a background call during a
  game needs the manual microphone pause.
* The observer sees only analysed frames (no motion between them); a person's
  name on screen cannot be redacted by the description filter.
* `VisionObserved` carries the extra details (`interesting`, `spoiler`,
  `is_game`, `solicited`) through `VisionIntegration.details_for`; putting them
  on the event itself (core/events.py) would be simpler.
