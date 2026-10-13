"""Voice wired into the app: engines from the settings, the push-to-talk
hotkey, speech routing, model downloads.

:class:`VoiceIntegration` is the one object ``app.py`` builds for Phase 4.
It owns a :class:`~chibi.voice.service.VoiceService` and decides, from the
settings, the permissions and the optional libraries actually installed,
which engines exist:

* **Voice** - Piper (local) when its model is downloaded, or Azure (cloud:
  only with ``Permission.CLOUD_AI`` and a stored key).  The Azure key is read
  for every request on the worker thread, *after* re-checking the
  permission, so a line queued just before a revoke never leaves the PC.
* **Recognition** - local Whisper, only while ``Permission.MICROPHONE`` is
  granted (a revoke also frees the model's memory).  Revoking the
  permission or the kill switch closes the microphone at once, drops a
  transcription still running and unregisters the hotkey.
* **Optional libraries** (the ``voice`` extra: numpy, onnxruntime,
  faster-whisper, sounddevice) are detected with ``find_spec`` and imported
  only when used: without them the app runs and the page says, in
  Ukrainian, what to install.

Speech routing: the controller hands every :class:`SpeechRequested` to
:meth:`VoiceIntegration.route` together with a ``show`` callback.  A line
that should be voiced is queued on the service and its bubble appears when
its audio starts, for as long as the audio lasts; a line that is not voiced
- or whose synthesis fails, or that is dropped - is shown as text at once.
No line is ever lost, except one the user cut off (a click on her, the
push-to-talk key).  Like the controller's own queue, voiced lines she chose
by herself are dropped - even mid-sentence - when the user asks for quiet,
turns on Do Not Disturb or stops observation; answers to the user stay.

Threading: everything here runs on the Qt main thread.  Work happens on the
service's worker, the hotkey thread and a download worker; they come back
through :meth:`_post` (a queued ``Signal(object)``, the same pattern as
``chibi.ai.service``).  Nothing here logs what was said or heard.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from PySide6.QtCore import QObject, Qt, Signal, Slot

from chibi.core.clock import Clock, SystemClock
from chibi.core.config import Settings, VoiceSettings
from chibi.core.event_bus import EventBus
from chibi.core.events import (
    AppShuttingDown,
    CharacterInteraction,
    DoNotDisturbChanged,
    FullscreenChanged,
    InteractionKind,
    ObservationStopped,
    Permission,
    PermissionChanged,
    QuietModeChanged,
    SettingsChanged,
    SpeechPlayback,
    SpeechRequested,
    VoiceStateChanged,
    VoiceTranscript,
)
from chibi.core.worker import SerialWorker
from chibi.voice import hotkey_win, models
from chibi.voice.models import CATALOG, DEFAULT_TTS_MODEL, DownloadError, ModelSpec

if TYPE_CHECKING:  # pragma: no cover - typing only; these modules need numpy at import time
    from chibi.voice.service import VoiceService
    from chibi.voice.stt import STTEngine
    from chibi.voice.tts import TTSEngine

log = logging.getLogger(__name__)

#: Secret name of the Azure Speech key in the app's SecretStore.
AZURE_KEY_NAME = "azure_speech_key"

# -- reasons shown in the UI (Ukrainian) ------------------------------------------------------------

VOICE_MODULE_MISSING = "Голосовий модуль не встановлено (pip install .[voice])"
TTS_OFF = "Голос вимкнено."
PIPER_MODEL_MISSING = "Голос ще не завантажено (Налаштування → Голос)."
AZURE_NEEDS_CLOUD = "Azure: потрібен дозвіл на хмарний AI (Налаштування → Приватність)."
AZURE_NO_KEY = "Azure: додай ключ Azure Speech (Налаштування → Голос)."
TTS_BUILD_FAILED = "Не вдалося підготувати голос."
STT_OFF = "Розпізнавання мовлення вимкнено."
STT_MODEL_MISSING = "Модель розпізнавання ще не завантажено (Налаштування → Голос)."
MIC_NOT_ALLOWED = "Мікрофон вимкнено в «Приватності»."
STT_BUILD_FAILED = "Не вдалося підготувати розпізнавання мовлення."
HOTKEY_WINDOWS_ONLY = "Гаряча клавіша працює лише у Windows — говори кнопкою мікрофона в чаті."
HOTKEY_FAILED = "Не вдалося зареєструвати гарячу клавішу."
VOICE_UNAVAILABLE = "Голос зараз недоступний."
#: The "Прослухати" line; ``{name}`` is her name.
SAMPLE_LINE = "Привіт! Я {name}. Ось так звучатиме мій голос."

#: Libraries each engine needs (import names).  numpy is needed by the
#: voice service itself (voice activity detection, audio buffers).
PIPER_LIBRARIES = ("numpy", "onnxruntime", "sounddevice")
AZURE_LIBRARIES = ("numpy", "sounddevice")
STT_LIBRARIES = ("numpy", "faster_whisper", "sounddevice")
SERVICE_LIBRARIES = ("numpy",)

#: Recognition result status -> phrase key of her short reaction.
RESULT_LINES: dict[str, str] = {
    "no_speech": "voice.not_heard",
    "not_understood": "voice.not_understood",
    "error": "voice.error",
}
LISTENING_LINE = "voice.listening"
#: Phrase keys never spoken aloud: they announce an open microphone, which
#: would hear her own voice.
NEVER_SPOKEN: frozenset[str] = frozenset({LISTENING_LINE})
#: A click, a pat or picking her up while she talks cuts her off.
INTERRUPTING: frozenset[InteractionKind] = frozenset(
    {InteractionKind.CLICK, InteractionKind.PET, InteractionKind.DRAG_START})
#: Settings sections that change the engines or the service (her name is in
#: the recognition prompt and the wake phrase).  Routing reads the rest live.
WATCHED_SECTIONS: frozenset[str] = frozenset({"", "voice", "character"})

#: The same notification is not repeated within this many seconds.
NOTIFY_REPEAT_SECONDS = 60.0
#: Download progress is posted to the main thread at most this often.
PROGRESS_INTERVAL = 0.1
#: The wake phrase ("Міє, ...") is short: longer background chatter is cut here.
WAKE_MAX_SECONDS = 8.0


def module_available(name: str) -> bool:
    """True if *name* can be imported - without importing it (no DLLs loaded)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _log_failure(what: str, exc: BaseException) -> None:
    """Log without the exception message (it could quote a line or a path); frames only."""
    frames = "".join(traceback.format_tb(exc.__traceback__)) if exc.__traceback__ else ""
    log.error("%s failed: %s\n%s", what, type(exc).__name__, frames.rstrip())


# -- what the app passes in -----------------------------------------------------------------------


class PermissionReader(Protocol):
    def is_granted(self, permission: Permission) -> bool: ...


class SecretAccess(Protocol):
    def get(self, name: str) -> str | None: ...

    def set(self, name: str, value: str) -> None: ...


class LineComposer(Protocol):
    """:class:`chibi.personality.composer.SpeechComposer`."""

    def compose(self, *keys: str, solicited: bool = False, **variables: Any) -> bool: ...


class EngineFactory(Protocol):
    """Builds the engines (a seam for tests; :class:`DefaultEngines` in the app)."""

    def piper(self, voice: VoiceSettings) -> TTSEngine: ...

    def azure(self, voice: VoiceSettings, get_key: Callable[[], str | None]) -> TTSEngine: ...

    def whisper(self, voice: VoiceSettings) -> STTEngine: ...


class DefaultEngines:
    """The real engines.  Imports happen here, on first use, so a missing
    optional library only disables voice instead of breaking the app."""

    def __init__(self, models_root: Path | None = None) -> None:
        self._root = models_root

    def piper(self, voice: VoiceSettings) -> TTSEngine:
        from chibi.voice.tts import PiperTTS
        from chibi.voice.tts_text import default_lexicon

        spec = CATALOG[DEFAULT_TTS_MODEL]
        # Game titles need no entries of their own: the lexicon knows the
        # common names and transliterates any other Latin word by rule.
        return PiperTTS(models.model_dir(spec, self._root), spec.id, speaker=voice.piper_speaker,
                        speed=voice.speed, threads=voice.cpu_threads, lexicon=default_lexicon())

    def azure(self, voice: VoiceSettings, get_key: Callable[[], str | None]) -> TTSEngine:
        from chibi.voice.tts import AzureTTS

        return AzureTTS(get_key, region=voice.azure_region, voice=voice.azure_voice, speed=voice.speed)

    def whisper(self, voice: VoiceSettings) -> STTEngine:
        from chibi.voice.stt import WhisperSTT

        return WhisperSTT(models.model_dir(CATALOG[voice.stt_model], self._root), threads=voice.cpu_threads)


def _sound_output(device: str) -> Any:
    from chibi.voice.audio_io import SoundDeviceOutput

    return SoundDeviceOutput(device)


def _sound_input(device: str) -> Any:
    from chibi.voice.audio_io import SoundDeviceInput

    return SoundDeviceInput(device)


def _list_devices() -> tuple[list[str], list[str]]:
    from chibi.voice.audio_io import list_devices

    return list_devices()


class _GatedInput:
    """A microphone stream whose audio is dropped, on the audio thread, the
    moment the permission is revoked - before the main thread has even
    dispatched the revoke and closed the stream."""

    def __init__(self, stream: Any, allowed: Callable[[], bool]) -> None:
        self._stream = stream
        self._allowed = allowed

    def start(self, on_audio: Callable[[Any], None]) -> None:
        allowed = self._allowed

        def gated(chunk: Any) -> None:  # audio thread
            if allowed():
                on_audio(chunk)

        self._stream.start(gated)

    def stop(self) -> None:
        self._stream.stop()

    @property
    def active(self) -> bool:
        return bool(self._stream.active)


# -- views for the UI ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class VoiceStatus:
    """What the settings page, the chat's microphone button and the menus show."""

    tts_ready: bool
    stt_ready: bool
    #: Why the voice / recognition is unavailable ("" when ready), Ukrainian.
    tts_reason: str
    stt_reason: str
    #: ``chibi.voice.service.VoiceState`` value ("off" without the voice module).
    state: str
    #: Push-to-talk combination as shown to the user ("Ctrl + Alt + Пробіл").
    hotkey: str
    hotkey_active: bool
    #: Why the hotkey is not registered although recognition works ("" otherwise).
    hotkey_reason: str = ""

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(r for r in (self.tts_reason, self.stt_reason, self.hotkey_reason) if r)


@dataclass(frozen=True)
class DownloadProgress:
    model_id: str
    done: int
    total: int


@dataclass
class _Download:
    model_id: str
    cancel: threading.Event
    on_progress: Callable[[int, int], None]
    on_done: Callable[[bool, str], None]
    done: int = 0
    total: int = 0
    last_post: float = field(default=-1e9)


@dataclass
class _Routed:
    """One line handed to :meth:`VoiceIntegration.route`."""

    event: SpeechRequested
    show: Callable[[SpeechRequested, float | None], None]
    shown: bool = False
    #: Its audio is playing; ``announced`` once SpeechPlayback(True) was published.
    playing: bool = False
    announced: bool = False
    done: bool = False
    #: Dropped by quiet / DND / stop watching: never spoken, never shown.
    cancelled: bool = False
    #: Bumped when the line is queued again; callbacks of an older attempt are ignored.
    attempt: int = 0


class VoiceIntegration(QObject):
    """See the module docstring.  Main thread only (except where noted)."""

    _invoke = Signal(object)

    def __init__(
        self,
        bus: EventBus,
        settings: Callable[[], Settings],
        permissions: PermissionReader,
        secrets: SecretAccess,
        composer: LineComposer | None = None,
        *,
        notify: Callable[[str], None] | None = None,
        on_change: Callable[[], None] | None = None,
        is_visible: Callable[[], bool] | None = None,
        clock: Clock | None = None,
        engines: EngineFactory | None = None,
        models_root: Path | None = None,
        installed: Callable[[ModelSpec], bool] | None = None,
        libraries: Callable[[str], bool] = module_available,
        output_factory: Callable[[str], Any] = _sound_output,
        input_factory: Callable[[str], Any] = _sound_input,
        list_devices: Callable[[], tuple[list[str], list[str]]] = _list_devices,
        hotkey_factory: Callable[..., Any] = hotkey_win.GlobalHotkey,
        hotkey_supported: Callable[[], bool] = hotkey_win.supported,
        voice_worker: Any = None,
        download_worker: Any = None,
        downloader: Callable[..., Any] = models.download,
        remover: Callable[[ModelSpec, Path | None], None] = models.remove,
        post: Callable[[Callable[[], None]], None] | None = None,
        parent: QObject | None = None,
    ) -> None:
        """*notify* shows a short Ukrainian message to the user (tray balloon,
        status line); it is called at most once a minute per message.
        *on_change* runs after every (re)configuration - :meth:`status` may
        differ now (the app updates the chat's microphone button).
        *is_visible* tells whether she is on screen (unsolicited lines are not
        voiced while the user has hidden her).  *post* replaces the queued Qt
        invoker (tests run callbacks inline).  Everything else is a seam for
        tests and defaults to the real thing."""
        super().__init__(parent)
        self._bus = bus
        self._settings = settings
        self._permissions = permissions
        self._secrets = secrets
        self._composer = composer
        self._notify = notify
        self._on_change = on_change
        self._is_visible = is_visible
        self._clock: Clock = clock or SystemClock()
        self._models_root = models_root
        self._engines: EngineFactory = engines or DefaultEngines(models_root)
        self._installed = installed or (lambda spec: models.is_installed(spec, models_root))
        self._libraries = libraries
        self._output_factory = output_factory
        self._input_factory = input_factory
        self._list_devices = list_devices
        self._hotkey_factory = hotkey_factory
        self._hotkey_supported = hotkey_supported
        self._voice_worker = voice_worker
        self._download_worker = (download_worker if download_worker is not None
                                 else SerialWorker("chibi-models"))
        self._downloader = downloader
        self._remover = remover
        self._post_override = post

        self._started = False
        self._closed = False
        self._service: VoiceService | None = None
        self._output: Any = None
        self._tts: TTSEngine | None = None
        self._stt: STTEngine | None = None
        self._tts_key: tuple[Any, ...] | None = None
        self._stt_key: tuple[Any, ...] | None = None
        self._tts_reason = TTS_OFF
        self._stt_reason = STT_OFF
        self._wake: bool | None = None
        self._suspended: bool | None = None
        self._fullscreen = False
        #: >0 while the integration itself (not the user) interrupts her.
        self._system_interrupt = 0
        self._last_error = ""
        self._notified: dict[str, float] = {}

        self._hotkey: Any = None
        self._hotkey_key: tuple[str, str] | None = None
        self._hotkey_failed: tuple[str, str] | None = None
        self._hotkey_reason = ""

        self._download: _Download | None = None
        #: Lines queued on the service and not finished yet.
        self._live: list[_Routed] = []

        self._invoke.connect(self._run_on_main, Qt.ConnectionType.QueuedConnection)
        self._unsubscribers: list[Callable[[], None]] = [
            bus.subscribe(PermissionChanged, self._on_permission_changed),
            bus.subscribe(ObservationStopped, self._on_observation_stopped),
            bus.subscribe(FullscreenChanged, self._on_fullscreen),
            bus.subscribe(SettingsChanged, self._on_settings_changed),
            bus.subscribe(CharacterInteraction, self._on_interaction),
            bus.subscribe(QuietModeChanged, self._on_quiet),
            bus.subscribe(DoNotDisturbChanged, self._on_quiet),
            bus.subscribe(AppShuttingDown, lambda _event: self.shutdown()),
        ]

    # -- lifecycle ---------------------------------------------------------------------------------

    @property
    def service(self) -> VoiceService | None:
        """The voice service, once the voice module is installed and started."""
        return self._service

    def start(self) -> None:
        """Build the engines for the current settings and register the hotkey."""
        if self._started or self._closed:
            return
        self._started = True
        self.reconfigure()

    def shutdown(self) -> None:
        """Stop everything: hotkey, microphone, playback, a running download.  Idempotent."""
        if self._closed:
            return
        self._closed = True
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        self._stop_hotkey()
        if self._download is not None:
            self._download.cancel.set()
        service, self._service = self._service, None
        if service is not None:
            try:
                service.shutdown()
            except Exception as exc:
                _log_failure("stopping the voice service", exc)
        shutdown = getattr(self._download_worker, "shutdown", None)
        if shutdown is not None:
            try:
                shutdown(1.0)
            except Exception as exc:
                _log_failure("stopping the download worker", exc)

    # -- configuration -----------------------------------------------------------------------------

    def reconfigure(self) -> None:
        """Apply the current settings and permissions.  Engines are rebuilt only
        when something they depend on changed (a rebuild interrupts her)."""
        if not self._started or self._closed:
            return
        try:
            cfg = self._settings()
        except Exception as exc:
            _log_failure("reading the voice settings", exc)
            return
        v = cfg.voice
        tts_key, self._tts_reason = self._plan_tts(v)
        stt_key, self._stt_reason = self._plan_stt(v)
        service = self._service
        if service is None and (tts_key is not None or stt_key is not None):
            service = self._create_service(v)
            if service is None:  # the libraries are there (planned), so something else broke
                tts_key = stt_key = None
                self._tts_reason = self._tts_reason or VOICE_UNAVAILABLE
                self._stt_reason = self._stt_reason or VOICE_UNAVAILABLE
        if service is None:
            self._tts_key = self._stt_key = None
            self._tts = self._stt = None
            self._sync_hotkey(v)
            self._changed()
            return

        tts, stt = self._tts, self._stt
        if tts_key != self._tts_key:
            tts = self._build_tts(tts_key, v) if tts_key is not None else None
            if tts_key is not None and tts is None:
                tts_key, self._tts_reason = None, TTS_BUILD_FAILED
        if stt_key != self._stt_key:
            stt = self._build_stt(v) if stt_key is not None else None
            if stt_key is not None and stt is None:
                stt_key, self._stt_reason = None, STT_BUILD_FAILED
        changed = tts_key != self._tts_key or stt_key != self._stt_key

        self._apply_service_options(service, cfg)
        wake = v.wake_phrase_enabled and stt is not None
        suspended = self._fullscreen and not v.wake_in_fullscreen
        if changed:
            self._tts_key, self._stt_key = tts_key, stt_key
            self._tts, self._stt = tts, stt
            # Close a wake-phrase microphone first: it must never be (re)opened
            # with the old engine or an outdated choice.
            self._set_wake(service, False)
            self._set_suspended(service, suspended)
            self._system_interrupt += 1
            try:
                service.configure(tts=tts, stt=stt)
            finally:
                self._system_interrupt -= 1
            self._set_wake(service, wake)
            service.preload()
        else:
            self._set_suspended(service, suspended)
            self._set_wake(service, wake)
        self._sync_hotkey(v)
        self._changed()

    def _changed(self) -> None:
        if self._on_change is None:
            return
        try:
            self._on_change()
        except Exception as exc:
            _log_failure("the voice change callback", exc)
    def _plan_tts(self, v: VoiceSettings) -> tuple[tuple[Any, ...] | None, str]:
        """(engine identity, "") when a voice can be built, else (None, reason)."""
        if not v.tts_enabled:
            return None, TTS_OFF
        if v.tts_engine == "azure":
            if not self._have(AZURE_LIBRARIES):
                return None, VOICE_MODULE_MISSING
            if not self._granted(Permission.CLOUD_AI):
                return None, AZURE_NEEDS_CLOUD
            if not self.has_azure_key():
                return None, AZURE_NO_KEY
            return ("azure", v.azure_region, v.azure_voice, v.speed), ""
        if not self._have(PIPER_LIBRARIES):
            return None, VOICE_MODULE_MISSING
        if not self._is_installed(CATALOG[DEFAULT_TTS_MODEL]):
            return None, PIPER_MODEL_MISSING
        return ("piper", v.piper_speaker, v.speed, v.cpu_threads), ""

    def _plan_stt(self, v: VoiceSettings) -> tuple[tuple[Any, ...] | None, str]:
        if not v.stt_enabled:
            return None, STT_OFF
        if not self._have(STT_LIBRARIES):
            return None, VOICE_MODULE_MISSING
        spec = CATALOG.get(v.stt_model)
        if spec is None or not self._is_installed(spec):
            return None, STT_MODEL_MISSING
        if not self._granted(Permission.MICROPHONE):
            # Not even loaded: hundreds of MB of memory for a microphone she may not use.
            return None, MIC_NOT_ALLOWED
        return ("whisper", v.stt_model, v.cpu_threads), ""

    def _create_service(self, v: VoiceSettings) -> VoiceService | None:
        if not self._have(SERVICE_LIBRARIES):
            return None
        try:
            from chibi.voice.service import VoiceService

            self._output = self._output_factory(v.output_device)
            self._service = VoiceService(
                post=self._post,
                output=self._output,
                input_factory=lambda: _GatedInput(self._input_factory(self._input_device()),
                                                  self._audio_allowed),
                worker=self._voice_worker,
                clock=self._clock.monotonic,
                mic_allowed=self._mic_allowed,
                on_state=self._on_state,
                on_result=self._on_result,
                on_error=self._on_error,
                on_wake=self._on_wake,
            )
        except Exception as exc:
            _log_failure("creating the voice service", exc)
            self._service = None
        return self._service

    def _build_tts(self, key: tuple[Any, ...], v: VoiceSettings) -> TTSEngine | None:
        try:
            if key[0] == "azure":
                return self._engines.azure(v, self._azure_key)
            return self._engines.piper(v)
        except Exception as exc:
            _log_failure("creating the voice", exc)
            return None

    def _build_stt(self, v: VoiceSettings) -> STTEngine | None:
        try:
            return self._engines.whisper(v)
        except Exception as exc:
            _log_failure("creating speech recognition", exc)
            return None

    def _apply_service_options(self, service: VoiceService, cfg: Settings) -> None:
        from chibi.voice.service import WAKE_WORDS
        from chibi.voice.vad import VADConfig

        v = cfg.voice
        name = cfg.character.name
        service.volume = max(0.0, min(v.volume / 100.0, 1.0))
        service.prompt = f"{name}, привіт."
        lowered = name.strip().lower()
        service.wake_names = WAKE_WORDS if not lowered or lowered in WAKE_WORDS else (lowered, *WAKE_WORDS)
        service.listen_config = VADConfig(end_silence_ms=v.end_silence_ms,
                                          max_utterance_seconds=float(v.max_utterance_seconds))
        service.wake_config = VADConfig(end_silence_ms=v.end_silence_ms,
                                        max_utterance_seconds=min(float(v.max_utterance_seconds),
                                                                  WAKE_MAX_SECONDS))
        if self._output is not None and hasattr(self._output, "device"):
            self._output.device = v.output_device

    def _set_wake(self, service: VoiceService, enabled: bool) -> None:
        if enabled != self._wake:
            self._wake = enabled
            service.set_wake(enabled)

    def _set_suspended(self, service: VoiceService, suspended: bool) -> None:
        if suspended != self._suspended:
            self._suspended = suspended
            service.set_suspended(suspended)

    # -- speech routing ----------------------------------------------------------------------------

    def route(self, event: SpeechRequested, show: Callable[[SpeechRequested, float | None], None]) -> None:
        """Voice *event* or show it as text right away.  ``show(event, seconds)``
        puts the line in her bubble; ``seconds`` is the audio's length, or None
        for the usual reading time."""
        line = _Routed(event, show)
        service = self._service
        if service is None or not self._should_voice(event) or not self._queue_line(service, line):
            self._show(line, None)

    def _queue_line(self, service: VoiceService, line: _Routed) -> bool:
        line.attempt += 1
        attempt = line.attempt
        if line not in self._live:
            self._live.append(line)
        try:
            accepted = service.say(line.event.text, solicited=line.event.solicited,
                                   on_start=lambda seconds: self._line_started(line, attempt, seconds),
                                   on_done=lambda outcome: self._line_done(line, attempt, outcome))
        except Exception as exc:
            _log_failure("queueing a voiced line", exc)
            accepted = False
        if not accepted and line.attempt == attempt and not line.done:
            self._forget(line)
        return accepted

    def _should_voice(self, event: SpeechRequested) -> bool:
        service = self._service
        if not self._started or self._closed or service is None or not service.can_speak:
            return False
        if not event.text.strip() or event.source.split(":", 1)[0] in NEVER_SPOKEN:
            return False
        try:
            cfg = self._settings()
        except Exception as exc:
            _log_failure("reading the voice settings", exc)
            return False
        v = cfg.voice
        if v.speak == "off" or (v.speak == "replies" and not event.solicited):
            return False
        if self._fullscreen and cfg.behavior.hide_on_fullscreen:
            return v.speak_when_hidden  # a fullscreen game hides her: voice only, if wanted
        if self._is_visible is not None:
            try:
                visible = bool(self._is_visible())
            except Exception:
                visible = True
            if not visible:
                return event.solicited  # the user hid her: only answers
        return True

    def _line_started(self, line: _Routed, attempt: int, seconds: float) -> None:
        if attempt != line.attempt or line.done:
            return
        line.playing = True
        if line.cancelled:
            # Stopped as soon as the service has finished starting it (never
            # re-enter the service from inside its own callback).
            self._post(self._skip_cancelled)
            return
        line.announced = True
        self._show(line, seconds if seconds > 0 else None)
        self._bus.publish(SpeechPlayback(playing=True, duration=max(0.0, float(seconds))))

    def _line_done(self, line: _Routed, attempt: int, outcome: str) -> None:
        if attempt != line.attempt or line.done:
            return  # this attempt was superseded (the line was queued again)
        self._forget(line)
        line.playing = False
        if line.announced:
            line.announced = False
            self._bus.publish(SpeechPlayback(playing=False))
        if line.cancelled:
            return  # "помовч" / DND / stop watching: not said, not shown
        if outcome == "interrupted" and not self._system_interrupt:
            return  # the user cut her off: the line is not shown either
        # Failed, dropped, an empty clip, or interrupted by a settings change:
        # never lose the line - show it as text.
        self._show(line, None)

    def _forget(self, line: _Routed) -> None:
        line.done = True
        with contextlib.suppress(ValueError):
            self._live.remove(line)

    def _drop_unsolicited(self) -> None:
        """Cancel the lines she chose herself; answers to the user stay queued."""
        cancelled = False
        for line in self._live:
            if not line.event.solicited and not line.done:
                line.cancelled = True
                cancelled = True
        if cancelled and any(line.cancelled and line.playing for line in self._live):
            self._skip_cancelled()  # mid-sentence: stop now

    def _skip_cancelled(self) -> None:
        """Stop the cancelled line being spoken; the answers queued behind it
        are queued again (the service can only drop its whole queue)."""
        service = self._service
        if service is None or self._closed:
            return
        if not any(line.cancelled and line.playing and not line.done for line in self._live):
            return  # it ended meanwhile (or the user cut everything off)
        keep = [line for line in self._live if not line.cancelled and not line.done]
        for line in keep:
            line.attempt += 1  # their callbacks from the interrupted queue no longer count
        service.interrupt()
        for line in keep:
            if not self._queue_line(service, line):
                self._forget(line)
                self._show(line, None)

    @staticmethod
    def _show(line: _Routed, seconds: float | None) -> None:
        if line.shown:
            return
        line.shown = True
        try:
            line.show(line.event, seconds)
        except Exception as exc:
            _log_failure("showing a line", exc)

    def interrupt(self) -> None:
        """The user cut her off (clicked her while she speaks)."""
        if self._service is not None:
            self._service.interrupt()

    # -- listening --------------------------------------------------------------------------------

    def toggle_listening(self, source: str = "button") -> str:
        """The chat's microphone button / a menu item.  Returns "" if handled,
        otherwise why it can't listen (Ukrainian)."""
        service = self._service
        if self._closed or service is None or not service.can_listen:
            return self._stt_reason or VOICE_UNAVAILABLE
        service.toggle_listening(source)
        return ""

    def _mic_allowed(self) -> bool:
        if self._closed or not self._granted(Permission.MICROPHONE):
            return False
        try:
            return bool(self._settings().voice.stt_enabled)
        except Exception:
            return False

    def _audio_allowed(self) -> bool:
        """Audio thread: thread-safe permission check, no settings access."""
        return not self._closed and self._granted(Permission.MICROPHONE)

    def _input_device(self) -> str:
        try:
            return self._settings().voice.input_device
        except Exception:
            return ""

    def _close_microphone(self) -> None:
        """Permission revoked or kill switch: microphone and hotkey off *now*.
        A transcription still running is dropped as well - its audio was heard
        before the revoke, but the text must not arrive after it."""
        self._stop_hotkey()
        self._hotkey_failed = None
        service = self._service
        if service is None:
            return
        try:
            service.cancel_listening()
            service.revoke_microphone()
        except Exception as exc:
            _log_failure("closing the microphone", exc)

    # -- service callbacks (main thread) -------------------------------------------------------------

    def _on_state(self, state: Any, reason: str) -> None:
        self._bus.publish(VoiceStateChanged(state=str(getattr(state, "value", state)), reason=reason))

    def _on_result(self, text: str, status: str) -> None:
        if status == "ok":
            text = text.strip()
            if text and self._mic_allowed():  # revoked meanwhile: what was heard is dropped
                self._bus.publish(VoiceTranscript(text=text))
            return
        key = RESULT_LINES.get(status)
        if key is not None:
            self._say_line(key)

    def _on_error(self, message: str) -> None:
        self._last_error = message
        self._notify_once(message)

    def _on_wake(self) -> None:
        self._say_line(LISTENING_LINE)  # a bubble only: see NEVER_SPOKEN

    def _say_line(self, key: str) -> None:
        if self._composer is None or self._closed:
            return
        try:
            self._composer.compose(key, solicited=True)
        except Exception as exc:
            _log_failure(f"composing {key}", exc)

    def _notify_once(self, message: str) -> None:
        message = (message or "").strip()
        if not message or self._notify is None or self._closed:
            return
        now = self._clock.monotonic()
        last = self._notified.get(message)
        if last is not None and now - last < NOTIFY_REPEAT_SECONDS:
            return
        if len(self._notified) > 32:
            self._notified = {m: t for m, t in self._notified.items() if now - t < NOTIFY_REPEAT_SECONDS}
        self._notified[message] = now
        try:
            self._notify(message)
        except Exception as exc:
            _log_failure("showing a voice notification", exc)

    # -- bus ----------------------------------------------------------------------------------------

    def _on_permission_changed(self, event: PermissionChanged) -> None:
        if event.permission is Permission.APP_METADATA and not self._granted(Permission.APP_METADATA):
            self._drop_unsolicited()  # they may be about the game she just saw (as the controller does)
        if event.permission is Permission.MICROPHONE and not self._granted(Permission.MICROPHONE):
            self._close_microphone()
        if event.permission in (Permission.MICROPHONE, Permission.CLOUD_AI):
            self.reconfigure()

    def _on_observation_stopped(self, _event: ObservationStopped) -> None:
        self._close_microphone()
        self._drop_unsolicited()
        self.reconfigure()

    def _on_fullscreen(self, event: FullscreenChanged) -> None:
        self._fullscreen = bool(event.active)
        service = self._service
        if service is None or self._closed:
            return
        try:
            wake_in_fullscreen = self._settings().voice.wake_in_fullscreen
        except Exception:
            wake_in_fullscreen = False
        self._set_suspended(service, self._fullscreen and not wake_in_fullscreen)

    def _on_settings_changed(self, event: SettingsChanged) -> None:
        if event.section in WATCHED_SECTIONS:
            self.reconfigure()

    def _on_quiet(self, event: QuietModeChanged | DoNotDisturbChanged) -> None:
        if event.enabled:
            self._drop_unsolicited()

    def _on_interaction(self, event: CharacterInteraction) -> None:
        service = self._service
        if event.kind in INTERRUPTING and service is not None and service.state == "speaking":
            service.interrupt()

    # -- push-to-talk hotkey ------------------------------------------------------------------------

    def _sync_hotkey(self, v: VoiceSettings) -> None:
        service = self._service
        usable = (self._started and not self._closed and service is not None and self._stt is not None
                  and self._granted(Permission.MICROPHONE))
        supported = self._safe_supported()
        if not usable or not supported:
            self._stop_hotkey()
            self._hotkey_failed = None
            self._hotkey_reason = HOTKEY_WINDOWS_ONLY if usable else ""
            return
        key = (v.ptt_hotkey, v.ptt_mode)
        if self._hotkey is not None and self._hotkey_key == key:
            return
        self._stop_hotkey()
        if self._hotkey_failed == key:
            return  # the user was told; tried again when the combination changes
        hold = v.ptt_mode == "hold"
        try:
            hotkey = self._hotkey_factory(
                v.ptt_hotkey,
                on_press=lambda: self._post(self._hotkey_pressed),
                on_release=(lambda: self._post(self._hotkey_released)) if hold else None,
            )
            hotkey.start()
        except hotkey_win.HotkeyError as exc:
            self._hotkey_failed = key
            self._hotkey_reason = exc.user_message
            log.warning("push-to-talk hotkey not registered: %s", exc)
            self._notify_once(exc.user_message)
            return
        except Exception as exc:
            _log_failure("registering the push-to-talk hotkey", exc)
            self._hotkey_failed = key
            self._hotkey_reason = HOTKEY_FAILED
            self._notify_once(HOTKEY_FAILED)
            return
        self._hotkey = hotkey
        self._hotkey_key = key
        self._hotkey_reason = ""

    def _safe_supported(self) -> bool:
        try:
            return bool(self._hotkey_supported())
        except Exception:
            return False

    def _stop_hotkey(self) -> None:
        hotkey, self._hotkey = self._hotkey, None
        self._hotkey_key = None
        if hotkey is None:
            return
        try:
            hotkey.stop()
        except Exception as exc:
            _log_failure("unregistering the push-to-talk hotkey", exc)

    def _hotkey_pressed(self) -> None:
        service = self._service
        if self._hotkey is None or self._hotkey_key is None or service is None or self._closed:
            return  # stopped meanwhile (revoke, kill switch)
        if self._hotkey_key[1] == "hold":
            service.listen("hotkey")
        else:
            service.toggle_listening("hotkey")

    def _hotkey_released(self) -> None:
        service = self._service
        if service is not None and not self._closed:
            service.stop_listening()

    # -- the settings page ---------------------------------------------------------------------------

    def status(self) -> VoiceStatus:
        service = self._service
        try:
            hotkey_text = self._settings().voice.ptt_hotkey
            title = hotkey_win.parse_hotkey(hotkey_text).title
        except Exception:
            title = ""
        tts_ready = service is not None and self._tts is not None and not self._closed
        stt_ready = service is not None and self._stt is not None and not self._closed
        return VoiceStatus(
            tts_ready=tts_ready,
            stt_ready=stt_ready,
            tts_reason="" if tts_ready else (self._tts_reason or VOICE_UNAVAILABLE),
            stt_reason="" if stt_ready else (self._stt_reason or VOICE_UNAVAILABLE),
            state=str(getattr(service.state, "value", "off")) if service is not None else "off",
            hotkey=title,
            hotkey_active=self._hotkey is not None,
            hotkey_reason=self._hotkey_reason if stt_ready else "",
        )

    def test_voice(self, text: str = "", on_done: Callable[[bool, str], None] | None = None) -> None:
        """"Прослухати": speak a sample line now, whatever ``speak`` says.
        *on_done(ok, message)* when it has played (or why it could not)."""

        def report(ok: bool, message: str) -> None:
            if on_done is not None:
                try:
                    on_done(ok, message)
                except Exception as exc:
                    _log_failure("reporting the voice test", exc)

        service = self._service
        if not self._started or self._closed or service is None or not service.can_speak:
            report(False, self._tts_reason or VOICE_UNAVAILABLE)
            return
        try:
            name = self._settings().character.name
        except Exception:
            name = "Мія"
        sample = text.strip() or SAMPLE_LINE.format(name=name)
        self._last_error = ""

        def done(outcome: str) -> None:
            if outcome == "played":
                report(True, "")
            elif outcome == "interrupted":
                report(False, "Перервано.")
            else:
                report(False, self._last_error or "Не вдалося озвучити репліку.")

        self._system_interrupt += 1  # lines cut off for the sample are shown as text
        try:
            service.interrupt()
        finally:
            self._system_interrupt -= 1
        if not service.say(sample, solicited=True, on_done=done):
            report(False, VOICE_UNAVAILABLE)

    def has_azure_key(self) -> bool:
        try:
            return bool(self._secrets.get(AZURE_KEY_NAME))
        except Exception as exc:
            _log_failure("reading the Azure key state", exc)
            return False

    def set_azure_key(self, key: str) -> None:
        """Store (or with "" delete) the Azure Speech key.  Errors propagate to
        the page, which reports them without the key."""
        self._secrets.set(AZURE_KEY_NAME, key.strip())
        self.reconfigure()

    def _azure_key(self) -> str | None:
        """Called by AzureTTS on the voice worker for every request."""
        if self._closed or not self._granted(Permission.CLOUD_AI):
            return None
        try:
            return self._secrets.get(AZURE_KEY_NAME)
        except Exception:
            return None

    def devices(self) -> tuple[list[str], list[str]]:
        """(microphones, speakers) by name; empty without the voice module."""
        if not self._have(("numpy", "sounddevice")):
            return [], []
        try:
            inputs, outputs = self._list_devices()
        except Exception as exc:
            _log_failure("listing audio devices", exc)
            return [], []
        return list(inputs), list(outputs)

    # -- models ---------------------------------------------------------------------------------------

    def models(self) -> list[tuple[ModelSpec, bool]]:
        """Every downloadable model and whether it is installed."""
        return [(spec, self._is_installed(spec)) for spec in CATALOG.values()]

    def download_state(self) -> DownloadProgress | None:
        d = self._download
        return DownloadProgress(d.model_id, d.done, d.total) if d is not None else None

    def download(self, model_id: str, on_progress: Callable[[int, int], None],
                 on_done: Callable[[bool, str], None]) -> None:
        """Download a model in the background - only ever on the user's click.
        Callbacks run on the main thread; *on_done* exactly once."""
        spec = CATALOG.get(model_id)
        if spec is None or self._closed:
            self._call(on_done, False, "Цю модель не можна завантажити.")
            return
        if self._download is not None:
            busy = ("Ця модель уже завантажується." if self._download.model_id == model_id
                    else "Зачекай, поки завантажиться інша модель.")
            self._call(on_done, False, busy)
            return
        state = _Download(spec.id, threading.Event(), on_progress, on_done, total=spec.size)
        self._download = state
        root = self._models_root

        def progress(done: int, total: int) -> None:  # download thread
            now = time.monotonic()
            if done < total and now - state.last_post < PROGRESS_INTERVAL:
                return
            state.last_post = now
            self._post(lambda: self._download_progress(state, done, total))

        def job() -> None:  # download thread
            try:
                self._downloader(spec, root, on_progress=progress, cancel=state.cancel)
            except DownloadError as exc:
                ok, message = False, exc.user_message
            except Exception as exc:
                _log_failure(f"downloading {spec.id}", exc)
                ok, message = False, "Не вдалося завантажити модель."
            else:
                ok, message = True, "Завантажено ✓"
            self._post(lambda: self._download_finished(state, ok, message))

        log.info("downloading voice model %s (asked by the user)", spec.id)
        try:
            self._download_worker.submit(job)
        except Exception as exc:
            _log_failure("starting a download", exc)
            self._download = None
            self._call(on_done, False, "Не вдалося почати завантаження.")

    def cancel_download(self) -> None:
        if self._download is not None:
            self._download.cancel.set()

    def _download_progress(self, state: _Download, done: int, total: int) -> None:
        if self._download is not state:
            return
        state.done, state.total = int(done), int(total)
        self._call(state.on_progress, state.done, state.total)

    def _download_finished(self, state: _Download, ok: bool, message: str) -> None:
        if self._download is state:
            self._download = None
        if ok:
            self.reconfigure()  # the engine may be usable now
        self._call(state.on_done, ok, message)

    def remove(self, model_id: str) -> None:
        """Delete a downloaded model (an engine using it is released first)."""
        spec = CATALOG.get(model_id)
        if spec is None or (self._download is not None and self._download.model_id == model_id):
            return
        service = self._service
        uses_it = ((spec.kind == "tts" and self._tts_key is not None and self._tts_key[0] == "piper")
                   or (spec.kind == "stt" and self._stt_key is not None and self._stt_key[1] == spec.id))
        if uses_it and service is not None:
            if spec.kind == "tts":
                self._tts, self._tts_key = None, None
            else:
                self._stt, self._stt_key = None, None
            self._system_interrupt += 1
            try:
                service.configure(tts=self._tts, stt=self._stt)
            finally:
                self._system_interrupt -= 1
        directory = models.model_dir(spec, self._models_root)
        # The marker goes first: should Windows refuse to delete a file still
        # mapped by the engine, the model at least no longer counts as installed.
        with contextlib.suppress(OSError):
            (directory / "installed.json").unlink(missing_ok=True)
        try:
            self._remover(spec, self._models_root)
        except Exception as exc:
            _log_failure(f"removing {spec.id}", exc)
        log.info("voice model %s removed by the user", spec.id)
        self.reconfigure()

    # -- helpers ----------------------------------------------------------------------------------

    def _have(self, names: tuple[str, ...]) -> bool:
        try:
            return all(self._libraries(name) for name in names)
        except Exception:
            return False

    def _granted(self, permission: Permission) -> bool:
        try:
            return bool(self._permissions.is_granted(permission))
        except Exception:
            return False

    def _is_installed(self, spec: ModelSpec) -> bool:
        try:
            return bool(self._installed(spec))
        except Exception as exc:
            _log_failure(f"checking {spec.id}", exc)
            return False

    @staticmethod
    def _call(callback: Callable[..., Any], *args: Any) -> None:
        try:
            callback(*args)
        except Exception as exc:
            _log_failure("a voice page callback", exc)

    def _post(self, fn: Callable[[], None]) -> None:
        """Run *fn* on the main thread.  Any thread; never synchronous with Qt."""
        if self._post_override is not None:
            self._post_override(fn)
            return
        with contextlib.suppress(RuntimeError):  # the QObject is gone (app teardown)
            self._invoke.emit(fn)

    @Slot(object)
    def _run_on_main(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            _log_failure("a voice callback", exc)
