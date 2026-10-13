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
  A line composed by the model from what she saw on the screen (the
  observer, vision's AI lines) is voiced by Azure only while screen data may
  go to the cloud (``CLOUD_VISION``, not "Лише локальний аналіз"); otherwise
  it is shown as text (:meth:`VoiceIntegration.screen_content_may_be_voiced`).
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
No line is ever lost, except her own lines the user cut off (a click on her,
the push-to-talk key); an answer to the user that was cut off before it was
heard is shown as text.  Like the controller's own queue, voiced lines she chose
by herself are dropped - even mid-sentence - when the user asks for quiet,
turns on Do Not Disturb or stops observation; answers to the user stay.

Listening (``VoiceSettings.listen_mode``): push-to-talk always works; in the
"name" and "open" modes the microphone stays open (see
:mod:`chibi.voice.service`).  The open microphone pauses - push-to-talk
still works - while Do Not Disturb is on, while a fullscreen app is in front
(unless ``wake_in_fullscreen``), while the app in front is marked private by
the privacy filter (messengers, mail, banking, password managers: never),
while it is a voice-call app (``pause_in_calls``), and while the user paused
it (:meth:`VoiceIntegration.set_open_mic_paused`).  The app in front is
known only while ``Permission.APP_METADATA`` is granted.
:attr:`VoiceStatus.pause_reason` says why, in Ukrainian.

What the user said arrives as :class:`VoiceTranscript` with its ``origin``
("push", "named", "heard"); :meth:`VoiceIntegration.transcript_allowed`
tells the app whether one may still be sent (a request heard on the open
microphone is dropped once it pauses, changes mode or recognition is off).
A question or remark of hers that invites a reply (:func:`invites_reply`)
opens the conversation window once heard, so the user can answer without
her name.

Streamed answers: :meth:`VoiceIntegration.begin_reply` returns a
:class:`ReplyStream` that voices a chat answer sentence by sentence while
the model is still writing it (see its docstring for the contract).

How fast she answers: every voice request is timed from the end of the
user's phrase to the start of her voice (recognition, the model's first
words, the first part to say, synthesis); the last one is
:attr:`VoiceIntegration.last_latency` (also ``VoiceStatus.latency``), logged
as one line of numbers.  :attr:`VoiceIntegration.conversation_active` says
whether the user is talking with her right now (``on_change`` runs whenever it
flips) - the app publishes ``ConversationActive`` from it, so background AI
work never delays her answer - and
:meth:`VoiceIntegration.add_speech_start_listener` tells the app the moment
the user starts talking.

Threading: everything here runs on the Qt main thread.  Work happens on the
service's worker, the hotkey thread and a download worker; they come back
through :meth:`_post` (a queued ``Signal(object)``, the same pattern as
``chibi.ai.service``).  Nothing here logs what was said or heard.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
import re
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from PySide6.QtCore import QCoreApplication, QObject, Qt, QTimer, Signal, Slot

from chibi.core.clock import Clock, SystemClock
from chibi.core.config import Settings, VoiceSettings
from chibi.core.event_bus import EventBus
from chibi.core.events import (
    ActivityCategory,
    AppShuttingDown,
    CharacterInteraction,
    DoNotDisturbChanged,
    ForegroundAppChanged,
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
from chibi.voice.tts_text import FIRST_CLAUSE_CHARS, SentenceChunker

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

# -- the open microphone ("name" / "open" listen modes) ---------------------------------------------

#: Why the open microphone is paused (``VoiceStatus.pause_reason``), most important first.
PAUSED_BY_USER = "Мікрофон на паузі."
PAUSED_PRIVATE = "Мікрофон на паузі: відкрито приватну програму."
PAUSED_CALL = "Мікрофон на паузі: відкрито програму для дзвінків."
PAUSED_DND = "Мікрофон на паузі: «Не турбувати»."
PAUSED_FULLSCREEN = "Мікрофон на паузі: програма на весь екран."
#: What the open microphone does (``VoiceStatus.listening_text``).
LISTENING_NAME = "Слухає постійно й відгукується на своє ім'я."
LISTENING_OPEN = "Слухає постійно — усе, що чує, для неї."
#: Her short reply when the user pauses / resumes the open microphone.
MIC_PAUSED_LINE = "voice.mic_paused"
MIC_RESUMED_LINE = "voice.mic_resumed"

#: Voice-call apps (lower-case exe names without ".exe"): with ``pause_in_calls`` the
#: open microphone pauses while one is in front.  Most are private apps anyway.
CALL_APPS: frozenset[str] = frozenset({
    "discord", "discordptb", "discordcanary", "vesktop", "teams", "ms-teams", "msteams", "lync",
    "zoom", "skype", "skypeapp", "slack", "webex", "webexmta", "ciscocollabhost", "ciscowebexstart",
    "atmgr", "whatsapp", "whatsapp.root", "viber", "signal", "telegram", "ayugram", "element",
    "mumble", "teamspeak", "ts3client_win64", "ts3client_win32", "guilded", "jitsi meet",
    "ringcentral", "googlemeet", "facebook messenger", "messenger", "wechat", "weixin", "line",
})
CALL_APP_PREFIXES = ("discord", "telegram", "teamspeak", "ts3client", "webex", "whatsapp", "skype")

# -- streamed answers -------------------------------------------------------------------------------

#: Between two sentences of an answer still being written, the microphone stays
#: closed (she is mid-answer) for at most this long after the last sentence arrived.
REPLY_HOLD_SECONDS = 8.0
#: Reading speed assumed for an answer shown without voice (conversation window).
READ_CHARS_PER_SECOND = 15.0
#: A voice request's timing waits this long for the chat answer it gets (see VoiceLatency).
LATENCY_TRACE_SECONDS = 120.0

#: ``SpeechRequested.source`` of the observer's lines (``chibi.ai.brain.OBSERVER_SOURCE``).
OBSERVER_SOURCE = "observer:ai"
#: VoiceService result status -> ``VoiceTranscript.origin``.
TRANSCRIPT_ORIGINS: dict[str, str] = {"ok": "push", "named": "named", "heard": "heard"}
_QUESTION_END = re.compile(r"\?[^\w]*$")
_MARKUP = re.compile(r"</?(?:b|i|u|em|strong|p|span|div|br|s|small|big|font)\b[^<>]{0,60}>", re.IGNORECASE)


def invites_reply(event: SpeechRequested) -> bool:
    """A line of her own that the user may answer: the observer's (she started
    a conversation about the screen) or a question."""
    if event.solicited:
        return False
    return event.source == OBSERVER_SOURCE or bool(_QUESTION_END.search(event.text.strip()))


def carries_screen_content(event: SpeechRequested) -> bool:
    """A line the model composed from what she saw on the screen."""
    source = event.source or ""
    return source == OBSERVER_SOURCE or (source.startswith("vision.") and source.endswith(":ai"))


def is_call_app(process_name: str) -> bool:
    """True for a voice-call app ("Discord.exe", "ms-teams.exe", "Zoom.exe")."""
    name = (process_name or "").strip().lower()
    name = name.removesuffix(".exe")
    return bool(name) and (name in CALL_APPS or name.startswith(CALL_APP_PREFIXES))


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
    #: How she listens right now: "push", "name" or "open" ("push" while recognition
    #: is not ready, or after the microphone failed to stay open).
    listen_mode: str = "push"
    #: Why the open microphone is paused, Ukrainian ("" when it is not, or in "push" mode).
    pause_reason: str = ""
    #: Timing of the last voice request she answered aloud (None before the first).
    latency: VoiceLatency | None = None

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(r for r in (self.tts_reason, self.stt_reason, self.hotkey_reason) if r)

    @property
    def listening_text(self) -> str:
        """One line for the chat, the tray and the menus about the open microphone
        ("" in "push" mode): what it does, or why it is paused."""
        if not self.stt_ready or self.listen_mode not in ("name", "open"):
            return ""
        if self.pause_reason:
            return self.pause_reason
        return LISTENING_NAME if self.listen_mode == "name" else LISTENING_OPEN


@dataclass(frozen=True, slots=True)
class VoiceLatency:
    """Where the time went for the last voice request she answered aloud (seconds):
    from the end of the user's phrase to the start of her voice.  The stages add up
    to ``total``; ``pause`` - the silence waited for before the phrase counted as
    finished ("Пауза в кінці фрази") - comes before it and is not in ``total``."""

    #: Speech end -> the recognised text is ready (waiting for the voice worker included).
    recognition: float
    #: Text ready -> the first piece of the answer arrives from the model.
    model: float
    #: First piece from the model -> the first part she can say is complete.
    first_chunk: float
    #: That part ready -> her audio starts (synthesis, waiting for the voice worker).
    synthesis: float
    #: Speech end -> her audio starts.
    total: float
    pause: float = 0.0

    @property
    def dominant(self) -> str:
        """The slowest part: "recognition", "model" (the model's wait and its first
        words together) or "synthesis"."""
        parts = {"recognition": self.recognition, "model": self.model + self.first_chunk,
                 "synthesis": self.synthesis}
        return max(parts, key=lambda name: parts[name])


@dataclass(frozen=True, slots=True)
class SpeechStart:
    """The user started talking (see :meth:`VoiceIntegration.add_speech_start_listener`)."""

    #: "push" (push-to-talk, the microphone button) or "open_mic" (the open microphone).
    source: str
    #: The phrase is for her: always for push-to-talk; on the open microphone in "open"
    #: mode, in the conversation window and right after her name - in "name" mode it is
    #: not known before it has been recognised (it may be the TV).
    for_her: bool


@dataclass
class _LatencyTrace:
    """One voice request on its way to her voice (``clock.monotonic`` seconds)."""

    speech_end: float
    transcribed: float
    pause: float = 0.0
    first_text: float | None = None
    first_chunk: float | None = None

    def finish(self, audio_start: float) -> VoiceLatency:
        first_text = self.first_text if self.first_text is not None else audio_start
        first_chunk = self.first_chunk if self.first_chunk is not None else first_text
        transcribed = max(self.transcribed, self.speech_end)
        first_text = max(first_text, transcribed)
        first_chunk = max(first_chunk, first_text)
        audio_start = max(audio_start, first_chunk)
        return VoiceLatency(recognition=transcribed - self.speech_end, model=first_text - transcribed,
                            first_chunk=first_chunk - first_text, synthesis=audio_start - first_chunk,
                            total=audio_start - self.speech_end, pause=self.pause)


def recommended_threads(cpu_count: int | None = None) -> int:
    """CPU threads recommended for recognition and the voice on this PC: half the
    logical processors, 2 to 6 (the rest stays free for a game)."""
    count = cpu_count if cpu_count is not None else (os.cpu_count() or 4)
    return max(2, min(6, count // 2))


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
    #: The streamed answer this line is a sentence of.
    stream: ReplyStream | None = None


ShowFn = Callable[[SpeechRequested, "float | None"], None]


def _same_words(text: str, name: str = "") -> str:
    """A sentence as compared by :meth:`ReplyStream._remainder`: markup and a leading
    "Мія:" / "Мія —" (artefacts the final answer no longer has) dropped, spaces collapsed."""
    text = " ".join(_MARKUP.sub("", text).split())
    if name and text.casefold().startswith(name.casefold()):
        rest = text[len(name):].lstrip()
        if rest.startswith((":", "—")):
            text = rest[1:].lstrip()
    return text


class ReplyStream:
    """One answer of hers on its way to her voice and bubble while the model
    is still writing it.  Made by :meth:`VoiceIntegration.begin_reply`.

    Contract (main thread only):

    * ``feed(delta)`` for every piece of the answer's body, in order; then
      exactly one of ``finish(text, state, portrait)`` (the final answer) or
      ``abort()`` (the answer was abandoned).  Calls after the first
      ``finish``/``abort`` do nothing.
    * When she can voice the answer (decided by :meth:`~VoiceIntegration.begin_reply`:
      voice ready, ``speak`` allows answers...), every complete sentence
      (:class:`~chibi.voice.tts_text.SentenceChunker`) is routed as a solicited
      line as soon as it is written: she starts talking while the model
      writes on, and the microphone stays closed between the sentences.  The
      first piece may end at a clause boundary ("Ну що ж, давай подумаємо
      разом," - at least :data:`~chibi.voice.tts_text.FIRST_CLAUSE_CHARS`
      characters), so she starts sooner on a long first sentence; its rest is
      the next piece.  ``finish`` routes what is left with the answer's state
      and portrait - or, when nothing is left, only her gesture (empty text,
      that state).
    * When she cannot voice it, ``feed`` does nothing and ``finish`` routes
      the whole answer once, like :meth:`~VoiceIntegration.route`.
    * The answer reaches her bubble / voice **exactly once** in total:
      ``finish`` compares the final text sentence by sentence with what was
      already routed and routes only the rest.  If the user cuts her off (a
      click on her, push-to-talk, her name), nothing more is voiced; the part
      not heard yet is shown once, as text, when the answer is complete.
    * ``abort()``: nothing more is routed - never route the aborted text
      again (route the replacement, e.g. the offline answer, as usual).
      Sentences queued but not started are dropped (the chat window still
      has them); the sentence she is saying finishes.
    """

    def __init__(self, owner: VoiceIntegration, show: ShowFn, source: str, voiced: bool,
                 name: str = "") -> None:
        self._owner = owner
        self._show = show
        self._source = source
        self._voiced = voiced
        #: Her name: a "Мія:" prefix the final answer drops is not "new" text.
        self._name = name.strip()
        self._chunker = SentenceChunker(first_clause=FIRST_CLAUSE_CHARS)
        #: Every sentence handed on (voiced, shown or kept for later), in order.
        self._routed: list[str] = []
        #: The first routed piece ended at a clause boundary (the start of a sentence).
        self._clause_first = False
        #: Timing of the voice request this answers (see :class:`VoiceLatency`).
        self._trace: _LatencyTrace | None = None
        self._lines: list[_Routed] = []
        #: Sentences not heard because the user cut her off (shown once at the end).
        self._unsaid: list[str] = []
        self._state = "talk"
        self._portrait = ""
        self._cut = False
        self._finished = False
        self._closed = False

    @property
    def voiced(self) -> bool:
        """Sentences are spoken while the answer is being written."""
        return self._voiced

    @property
    def spoken(self) -> bool:
        """Some of the answer has been handed to her voice / bubble already."""
        return bool(self._routed)

    @property
    def closed(self) -> bool:
        return self._closed

    def feed(self, delta: str) -> None:
        if self._closed or not self._voiced or not delta:
            return
        if self._trace is not None and self._trace.first_text is None and delta.strip():
            self._trace.first_text = self._owner._now()
        for sentence in self._chunker.feed(delta):
            if not self._routed and self._chunker.clause_first:
                self._clause_first = True
            self._route(sentence)

    def finish(self, text: str, state: str = "talk", portrait: str = "") -> None:
        if self._closed:
            return
        self._closed = self._finished = True
        self._state, self._portrait = state or "talk", portrait or ""
        text = text or ""
        if not self._voiced:
            self._owner._stream_closed(self)
            if text.strip() or state:
                self._owner.route(self._event(text), self._show)
            return
        if self._trace is not None and self._trace.first_text is None:
            self._trace.first_text = self._owner._now()  # the answer came in one piece
        rest = self._remainder(text)
        if rest:
            self._route(rest)
        elif state and state != "talk" and not self._cut:
            self._owner.route(self._event(""), self._show)  # her expression, nothing to repeat
        self._owner._stream_closed(self)
        self._maybe_flush()

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._trace = None
        self._unsaid.clear()
        for line in self._lines:
            if not line.done and not line.playing:
                line.cancelled = True
        self._owner._stream_closed(self)

    # -- internals (VoiceIntegration calls these) --------------------------------------------

    def _event(self, text: str) -> SpeechRequested:
        return SpeechRequested(text=text, state=self._state, portrait=self._portrait,
                               solicited=True, source=self._source)

    def _route(self, sentence: str) -> None:
        self._routed.append(sentence)
        if self._cut:
            self._unsaid.append(sentence)
            return
        if self._trace is not None and self._trace.first_chunk is None:
            self._trace.first_chunk = self._owner._now()
        self._lines.append(self._owner._route_stream_line(self, self._event(sentence), self._show))

    def _remainder(self, text: str) -> str:
        """The sentences of the final *text* not routed yet.  A first piece that
        ended at a clause counts together with the piece after it (the rest of
        that sentence); if that rest was not routed yet, only the words after
        the clause are left of its sentence."""
        chunker = SentenceChunker()
        sentences = chunker.feed(text) + chunker.flush()
        pieces = [_same_words(s, self._name) for s in self._routed]
        clause = ""
        if self._clause_first and pieces:
            if len(pieces) > 1:
                pieces[:2] = [_same_words(f"{self._routed[0]} {self._routed[1]}", self._name)]
            else:
                clause = pieces.pop(0)
        routed = Counter(pieces)
        rest = []
        for sentence in sentences:
            key = _same_words(sentence, self._name)
            if routed[key] > 0:
                routed[key] -= 1
            elif clause and key.startswith(clause) and key[len(clause):len(clause) + 1] in ("", " "):
                tail = key[len(clause):].strip()
                clause = ""
                if tail:
                    rest.append(tail)
            else:
                rest.append(sentence)
        return " ".join(rest)

    def _cut_off(self, line: _Routed | None = None) -> None:
        """The user cut her off: *line* (not shown yet) was not heard."""
        self._trace = None
        if line is not None and not line.shown:
            self._unsaid.append(line.event.text)
        if not self._cut:
            self._cut = True
            if not self._closed:
                self._owner._release_hold()
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        """Once the answer is complete and nothing of it is still queued, show
        what she did not get to say - once, as text."""
        if not (self._cut and self._finished and self._unsaid):
            return
        if any(not line.done for line in self._lines):
            return
        text = " ".join(self._unsaid)
        self._unsaid.clear()
        self._owner._show_now(self._event(text), self._show)


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
        differ now (the app updates the chat's microphone button) - whenever
        :attr:`conversation_active` flips and when :attr:`last_latency` changes.
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
        self._suspended: bool | None = None
        self._fullscreen = False
        # What pauses the open microphone (see the module docstring).
        self._dnd = False
        self._foreground_private = False
        self._foreground_call = False
        self._user_paused = False
        self._pause_reason = ""
        #: Streamed answers not finished / aborted yet.
        self._streams: list[ReplyStream] = []
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._check_hold)
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

        # -- how fast she answers and whether the user is talking with her -----------------
        #: The last voice request, until a chat answer to it begins (see begin_reply).
        self._pending_trace: _LatencyTrace | None = None
        self._last_latency: VoiceLatency | None = None
        self._speech_listeners: list[Callable[[SpeechStart], None]] = []
        #: conversation_active as last reported through on_change.
        self._conversation_was = False
        self._conversation_timer = QTimer(self)
        self._conversation_timer.setSingleShot(True)
        self._conversation_timer.timeout.connect(self._check_conversation)

        self._invoke.connect(self._run_on_main, Qt.ConnectionType.QueuedConnection)
        self._unsubscribers: list[Callable[[], None]] = [
            bus.subscribe(PermissionChanged, self._on_permission_changed),
            bus.subscribe(ObservationStopped, self._on_observation_stopped),
            bus.subscribe(FullscreenChanged, self._on_fullscreen),
            bus.subscribe(SettingsChanged, self._on_settings_changed),
            bus.subscribe(CharacterInteraction, self._on_interaction),
            bus.subscribe(QuietModeChanged, self._on_quiet),
            bus.subscribe(DoNotDisturbChanged, self._on_dnd),
            bus.subscribe(ForegroundAppChanged, self._on_foreground),
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
        self._streams.clear()
        self._hold_timer.stop()
        self._conversation_timer.stop()
        self._speech_listeners.clear()
        self._pending_trace = None
        self._conversation_was = False
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
        self._dnd = bool(cfg.behavior.do_not_disturb)
        mode = v.listen_mode if stt is not None else "push"
        if changed:
            self._tts_key, self._stt_key = tts_key, stt_key
            self._tts, self._stt = tts, stt
            # Close an open microphone first: it must never be (re)opened with
            # the old engine or an outdated choice.
            self._set_listen_mode(service, "push")
            self._sync_pause(service, cfg)
            self._system_interrupt += 1
            try:
                service.configure(tts=tts, stt=stt)
            finally:
                self._system_interrupt -= 1
            self._set_listen_mode(service, mode)
            service.preload()
        else:
            self._sync_pause(service, cfg)
            self._set_listen_mode(service, mode)
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
                on_speech_start=self._on_speech_start,
                on_activity=self._check_conversation,
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
        from chibi.voice.service import MIA_FORMS, OPEN_MIC_MIN_SPEECH_MS, NameForms
        from chibi.voice.vad import VADConfig

        v = cfg.voice
        name = cfg.character.name
        service.volume = max(0.0, min(v.volume / 100.0, 1.0))
        service.prompt = f"{name}, привіт."
        service.standby_prompt = f"Розмова українською. {name.strip() or 'Мія'} слухає."
        # Her own name, and Mia's forms too: the pack is Mia, and so is the habit.
        service.names = NameForms.for_name(name) | MIA_FORMS
        service.listen_config = VADConfig(end_silence_ms=v.end_silence_ms,
                                          max_utterance_seconds=float(v.max_utterance_seconds))
        # A request said in one breath on the open microphone can be as long as a
        # push-to-talk one; only very short sounds are dropped.
        service.wake_config = VADConfig(end_silence_ms=v.end_silence_ms,
                                        max_utterance_seconds=float(v.max_utterance_seconds),
                                        min_speech_ms=OPEN_MIC_MIN_SPEECH_MS)
        service.follow_up_seconds = float(v.follow_up_seconds)
        if self._output is not None and hasattr(self._output, "device"):
            self._output.device = v.output_device

    @staticmethod
    def _set_listen_mode(service: VoiceService, mode: str) -> None:
        # Compared with the service, not a cache: after the microphone failed to
        # stay open (the service fell back to "push") a settings change retries.
        if service.listen_mode != mode:
            service.set_listen_mode(mode)

    def _set_suspended(self, service: VoiceService, suspended: bool) -> None:
        if suspended != self._suspended:
            self._suspended = suspended
            service.set_suspended(suspended)

    def _open_mic_pause(self, v: VoiceSettings) -> str:
        """Why the open microphone should be paused now ("" if it should not)."""
        if self._user_paused:
            return PAUSED_BY_USER
        if self._foreground_private:
            return PAUSED_PRIVATE  # never listen there, whatever the settings say
        if self._foreground_call and v.pause_in_calls:
            return PAUSED_CALL
        if self._dnd:
            return PAUSED_DND
        if self._fullscreen and not v.wake_in_fullscreen:
            return PAUSED_FULLSCREEN
        return ""

    def _sync_pause(self, service: VoiceService | None, cfg: Settings) -> bool:
        """Apply the pause; True if its reason changed."""
        reason = self._open_mic_pause(cfg.voice)
        changed = reason != self._pause_reason
        self._pause_reason = reason
        if service is not None:
            self._set_suspended(service, bool(reason))
        return changed

    def _refresh_pause(self) -> None:
        """Something that pauses the open microphone changed (not the settings)."""
        if not self._started or self._closed:
            return
        try:
            cfg = self._settings()
        except Exception as exc:
            _log_failure("reading the voice settings", exc)
            return
        if self._sync_pause(self._service, cfg):
            self._changed()

    @property
    def open_mic_paused(self) -> bool:
        """The user paused the open microphone (:meth:`set_open_mic_paused`)."""
        return self._user_paused

    def set_open_mic_paused(self, paused: bool, *, announce: bool = True) -> None:
        """The user's "pause the microphone" switch (a menu item): the open
        microphone closes at once and stays closed until resumed; push-to-talk
        keeps working.  The app saves it, so it survives a restart.  With
        *announce* she says so (a short line) when it changes something she
        does - "I hear you again" only if nothing else keeps it paused (Do
        Not Disturb, a call...): then that reason is shown instead."""
        paused = bool(paused)
        if paused == self._user_paused or self._closed:
            return
        self._user_paused = paused
        self._refresh_pause()
        service = self._service
        if not announce or service is None or service.listen_mode == "push" or self._stt is None:
            return
        if paused:
            self._say_line(MIC_PAUSED_LINE)
        elif not self._pause_reason:
            self._say_line(MIC_RESUMED_LINE)
        else:
            self._notify_once(self._pause_reason)  # still closed: never claim to hear

    # -- speech routing ----------------------------------------------------------------------------

    def route(self, event: SpeechRequested, show: Callable[[SpeechRequested, float | None], None]) -> None:
        """Voice *event* or show it as text right away.  ``show(event, seconds)``
        puts the line in her bubble; ``seconds`` is the audio's length, or None
        for the usual reading time."""
        line = _Routed(event, show)
        service = self._service
        if service is None or not self._should_voice(event) or not self._queue_line(service, line):
            self._show_unvoiced(line)

    def begin_reply(self, source: str = "chat", *, show: ShowFn, voice_request: bool = False) -> ReplyStream:
        """Start voicing an answer that is still being written (see
        :class:`ReplyStream`).  *show* is the same callback :meth:`route` takes.
        *voice_request*: it answers the last voice request (its time to her voice is
        measured); an answer to typed text never takes that request's timing - the
        request may have been answered without a stream ("подивись на екран", offline)
        or still wait for its turn."""
        voiced = self._should_voice(SpeechRequested(text="…", solicited=True, source=source))
        try:
            name = self._settings().character.name
        except Exception:
            name = ""
        stream = ReplyStream(self, show, source, voiced, name)
        trace = None
        if voice_request:
            trace, self._pending_trace = self._pending_trace, None
        if voiced and source == "chat" and trace is not None \
                and self._now() - trace.transcribed <= LATENCY_TRACE_SECONDS:
            stream._trace = trace  # the answer to the last voice request
        if voiced:
            self._streams.append(stream)
            self._conversation_changed()
        return stream

    def _route_stream_line(self, stream: ReplyStream, event: SpeechRequested, show: ShowFn) -> _Routed:
        """One sentence of a streamed answer: voiced at once, or shown."""
        line = _Routed(event, show, stream=stream)
        service = self._service
        if service is not None and self._should_voice(event) and self._queue_line(service, line):
            if not stream.closed:
                self._hold(service)
        else:
            stream._trace = None  # shown, not voiced: no time to her voice to measure
            self._show_unvoiced(line)
        return line

    def _show_now(self, event: SpeechRequested, show: ShowFn) -> None:
        """Show *event* as text (never voiced)."""
        self._show_unvoiced(_Routed(event, show))

    def _show_unvoiced(self, line: _Routed) -> None:
        self._show(line, None)
        event, service = line.event, self._service
        if event.solicited and event.text.strip() and service is not None and not self._closed:
            # An answer shown without voice: the conversation goes on from now.
            try:
                service.note_reply_shown(len(event.text) / READ_CHARS_PER_SECOND)
            except Exception as exc:
                _log_failure("noting a shown answer", exc)

    def _hold(self, service: VoiceService) -> None:
        """Mid-answer: keep the microphone closed until the next sentence."""
        service.hold_for_reply(REPLY_HOLD_SECONDS)
        if QCoreApplication.instance() is not None:
            self._hold_timer.start(round((REPLY_HOLD_SECONDS + 0.5) * 1000))

    def _release_hold(self) -> None:
        if any(not stream._cut for stream in self._streams):
            return  # another answer is still being written and spoken
        self._hold_timer.stop()
        service = self._service
        if service is not None and not self._closed:
            service.release_reply_hold()

    def _check_hold(self) -> None:
        """The hold timer: a stalled answer must not keep the microphone closed."""
        service = self._service
        if service is not None and not self._closed:
            service.check_hold()

    def _stream_closed(self, stream: ReplyStream) -> None:
        with contextlib.suppress(ValueError):
            self._streams.remove(stream)
        self._release_hold()
        self._conversation_changed()

    def _cut_streams(self) -> None:
        """The user cut her off or started talking: answers being written are
        no longer voiced (what she does not get to say is shown as text)."""
        for stream in list(self._streams):
            stream._cut_off()
        self._conversation_changed()

    def _queue_line(self, service: VoiceService, line: _Routed) -> bool:
        line.attempt += 1
        attempt = line.attempt
        if line not in self._live:
            self._live.append(line)
        try:
            accepted = service.say(line.event.text, solicited=line.event.solicited,
                                   on_start=lambda seconds: self._line_started(line, attempt, seconds),
                                   on_done=lambda outcome: self._line_done(line, attempt, outcome),
                                   is_cancelled=lambda: line.cancelled,
                                   prefetch=line.stream is not None,
                                   opens_conversation=invites_reply(line.event))
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
        if carries_screen_content(event) and not self._screen_may_reach_voice(cfg):
            return False  # what she saw goes to a cloud voice only with cloud vision allowed
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

    def screen_content_may_be_voiced(self) -> bool:
        """Whether a line drawn from what she saw on the screen may be voiced now: always
        by a local voice (or none); by a cloud voice (Azure) only while screen data may
        go to the cloud - ``CLOUD_VISION`` (and ``CLOUD_AI``) granted and "Лише
        локальний аналіз" off.  The app also keeps what she saw out of a chat answer
        that a cloud voice would read out otherwise."""
        try:
            cfg = self._settings()
        except Exception as exc:
            _log_failure("reading the voice settings", exc)
            return False
        return cfg.voice.speak == "off" or self._screen_may_reach_voice(cfg)

    def _screen_may_reach_voice(self, cfg: Settings) -> bool:
        key = self._tts_key
        if key is None or key[0] != "azure":
            return True
        return (not cfg.vision.local_only and self._granted(Permission.CLOUD_VISION)
                and self._granted(Permission.CLOUD_AI))

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
        if line.stream is not None and line.stream._trace is not None:
            trace, line.stream._trace = line.stream._trace, None
            self._record_latency(trace.finish(self._now()))

    def _line_done(self, line: _Routed, attempt: int, outcome: str) -> None:
        if attempt != line.attempt or line.done:
            return  # this attempt was superseded (the line was queued again)
        if line.stream is not None:
            line.stream._trace = None  # its first line ended without being heard
        self._forget(line)
        line.playing = False
        if line.announced:
            line.announced = False
            self._bus.publish(SpeechPlayback(playing=False))
        if line.cancelled or self._closed:
            return  # "помовч" / DND / stop watching / an abandoned answer / app exit
        user_cut = outcome == "interrupted" and not self._system_interrupt
        if user_cut and line.stream is not None:
            line.stream._cut_off(line)  # the rest of that answer is shown once, at its end
            return
        if user_cut and not line.event.solicited:
            return  # the user cut her off: her own lines are not shown either
        # Failed, dropped, an empty clip, interrupted by a settings change - or an
        # answer still waiting behind the line the user cut off (the service drops
        # its whole queue): never lose the line - show it as text.  One already
        # showing (its audio had started) stays as it is.
        self._show(line, None)
        if line.stream is not None:
            line.stream._maybe_flush()

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
        if any(not stream._cut for stream in self._streams):
            self._hold(service)  # an answer is still being written

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
            self._cut_streams()

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

    def transcript_allowed(self, origin: str) -> bool:
        """May a request heard as *origin* (``VoiceTranscript.origin``) still be sent to
        her?  Never without the microphone permission or with recognition off; one
        heard on the open microphone ("named" / "heard") no longer once it pauses (the
        user's switch, DND, a call or private app, fullscreen) or leaves the "name" /
        "open" modes - what it heard just before is dropped, not sent later."""
        if not self._mic_allowed():
            return False
        if origin == "push":
            return True
        service = self._service
        return (service is not None and self._stt is not None and not self._pause_reason
                and service.listen_mode != "push")

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
        value = str(getattr(state, "value", state))
        if value == "listening":
            self._cut_streams()  # the user talks to her now: the old answer is not voiced on
        self._bus.publish(VoiceStateChanged(state=value, reason=reason))

    def _on_result(self, text: str, status: str) -> None:
        origin = TRANSCRIPT_ORIGINS.get(status)
        if origin is not None:
            text = text.strip()
            # Revoked, paused or switched meanwhile: what was heard is dropped.
            if text and self.transcript_allowed(origin):
                self._pending_trace = self._trace_of_delivery()
                self._bus.publish(VoiceTranscript(text=text, origin=origin))
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
            self._forget_foreground()
        if event.permission is Permission.MICROPHONE and not self._granted(Permission.MICROPHONE):
            self._close_microphone()
        if event.permission in (Permission.MICROPHONE, Permission.CLOUD_AI):
            self.reconfigure()

    def _on_observation_stopped(self, _event: ObservationStopped) -> None:
        self._close_microphone()
        self._drop_unsolicited()
        self._foreground_private = self._foreground_call = False
        self.reconfigure()

    def _on_fullscreen(self, event: FullscreenChanged) -> None:
        self._fullscreen = bool(event.active)
        self._refresh_pause()

    def _on_foreground(self, event: ForegroundAppChanged) -> None:
        """Only published while APP_METADATA is granted; private apps arrive nameless."""
        if not self._granted(Permission.APP_METADATA):
            return  # queued before a revoke
        private = event.app.private or event.category is ActivityCategory.PRIVATE
        self._foreground_private = private
        self._foreground_call = not private and is_call_app(event.app.process_name)
        self._refresh_pause()

    def _forget_foreground(self) -> None:
        """APP_METADATA revoked: the app in front is unknown from now on."""
        if self._foreground_private or self._foreground_call:
            self._foreground_private = self._foreground_call = False
            self._refresh_pause()

    def _on_dnd(self, event: DoNotDisturbChanged) -> None:
        self._dnd = bool(event.enabled)
        if event.enabled:
            self._drop_unsolicited()
        self._refresh_pause()

    def _on_settings_changed(self, event: SettingsChanged) -> None:
        if event.section in WATCHED_SECTIONS:
            self.reconfigure()

    def _on_quiet(self, event: QuietModeChanged) -> None:
        if event.enabled:
            self._drop_unsolicited()

    def _on_interaction(self, event: CharacterInteraction) -> None:
        service = self._service
        if event.kind in INTERRUPTING and service is not None and service.state == "speaking":
            service.interrupt()
            self._cut_streams()

    # -- the conversation: is the user talking with her, and how fast does she answer -------------

    @property
    def conversation_active(self) -> bool:
        """The user is talking with her right now - while it is, background AI work
        must not delay her answer.  True while push-to-talk listens, a phrase is
        being recognised, a delivered request waits for her answer, her answer is
        being written (a voiced :class:`ReplyStream`) or spoken, her name alone was
        just heard, the conversation window is open, or - in "open" mode - the open
        microphone hears speech (see ``VoiceService.conversation_active``).
        ``on_change`` runs whenever this flips, also when a wait runs out."""
        service = self._service
        if self._closed or service is None:
            return False
        try:
            active = bool(getattr(service, "conversation_active", False))
        except Exception as exc:
            _log_failure("reading the conversation state", exc)
            active = False
        return active or any(not stream._cut for stream in self._streams)

    def add_speech_start_listener(self, callback: Callable[[SpeechStart], None]) -> Callable[[], None]:
        """Call *callback* (main thread) whenever the user starts talking: push-to-talk
        or the microphone button opens the microphone, or the open microphone hears
        speech begin (:class:`SpeechStart` says whether it is known to be for her).
        The app can warm the model up meanwhile.  Returns a function that removes it."""
        self._speech_listeners.append(callback)

        def remove() -> None:
            with contextlib.suppress(ValueError):
                self._speech_listeners.remove(callback)

        return remove

    @property
    def last_latency(self) -> VoiceLatency | None:
        """Timing of the last voice request she answered aloud (None before the first)."""
        return self._last_latency

    def _on_speech_start(self, source: str, for_her: bool) -> None:
        if self._closed:
            return
        event = SpeechStart(source=source, for_her=bool(for_her))
        for listener in list(self._speech_listeners):
            try:
                listener(event)
            except Exception as exc:
                _log_failure("a speech start listener", exc)

    def _conversation_changed(self) -> None:
        """Something of the integration's own (an answer being written) changed:
        re-check once the service has finished what it is doing."""
        note = getattr(self._service, "note_activity", None)
        if note is not None and not self._closed:
            note()
        else:
            self._check_conversation()

    def _check_conversation(self) -> None:
        """Report a flip of :attr:`conversation_active` through ``on_change`` and wake up
        again when a wait or a window runs out."""
        if self._closed:
            return
        active = self.conversation_active
        self._schedule_conversation_check()
        if active != self._conversation_was:
            self._conversation_was = active
            self._changed()

    def _schedule_conversation_check(self) -> None:
        deadline_of = getattr(self._service, "conversation_deadline", None)
        deadline = None
        if deadline_of is not None:
            try:
                deadline = deadline_of()
            except Exception as exc:
                _log_failure("reading the conversation deadline", exc)
        if deadline is None:
            self._conversation_timer.stop()
            return
        if QCoreApplication.instance() is not None:
            wait = max(0.0, deadline - self._now())
            self._conversation_timer.start(min(2**31 - 1, round(wait * 1000) + 50))

    def _now(self) -> float:
        return self._clock.monotonic()

    def _trace_of_delivery(self) -> _LatencyTrace | None:
        """The timing of the request just delivered (see :class:`VoiceLatency`)."""
        delivery = getattr(self._service, "last_delivery", None)
        if delivery is None:
            return None
        return _LatencyTrace(speech_end=delivery.speech_end, transcribed=delivery.transcribed,
                             pause=delivery.pause)

    def _record_latency(self, latency: VoiceLatency) -> None:
        self._last_latency = latency
        # Numbers only: never what was said.
        log.info("voice answer latency %.2f s: recognition %.2f, model %.2f, first part %.2f, "
                 "synthesis %.2f (end-of-phrase pause %.2f before it)", latency.total,
                 latency.recognition, latency.model, latency.first_chunk, latency.synthesis,
                 latency.pause)
        self._changed()

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
            service.listen("hotkey", hold=True)
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
        listen_mode = str(service.listen_mode) if stt_ready and service is not None else "push"
        return VoiceStatus(
            tts_ready=tts_ready,
            stt_ready=stt_ready,
            tts_reason="" if tts_ready else (self._tts_reason or VOICE_UNAVAILABLE),
            stt_reason="" if stt_ready else (self._stt_reason or VOICE_UNAVAILABLE),
            state=str(getattr(service.state, "value", "off")) if service is not None else "off",
            hotkey=title,
            hotkey_active=self._hotkey is not None,
            hotkey_reason=self._hotkey_reason if stt_ready else "",
            listen_mode=listen_mode,
            pause_reason=self._pause_reason if listen_mode != "push" else "",
            latency=self._last_latency,
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
