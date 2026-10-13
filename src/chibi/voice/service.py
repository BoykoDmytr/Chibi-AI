"""The voice conversation: listen -> recognise -> (brain answers) -> speak.

:class:`VoiceService` is a state machine driven from the Qt main thread.
Slow work (recognition, synthesis) runs on one background worker thread;
audio callbacks run on PortAudio's thread.  Both hand results back through
the injected ``post`` function, which must run the callable on the main
thread (the app passes a queued Qt invoker; tests run it inline).  Every
asynchronous result carries a *generation* number: when the user interrupts
or cancels, the generation changes and late results are ignored.

States::

    OFF ---(engines configured)---> IDLE <--> STANDBY (open microphone: "name" / "open" mode)
    IDLE/STANDBY/SPEAKING --listen()--> LISTENING --utterance--> TRANSCRIBING --> IDLE/STANDBY
    IDLE/STANDBY --say()--> SPEAKING --(queue empty)--> IDLE/STANDBY

How she listens (``listen_mode``):

* ``"push"`` - only push-to-talk (the hotkey, the chat's microphone button);
* ``"name"`` - the microphone stays open (STANDBY); every utterance is
  recognised locally and delivered only when it addresses her by name
  (:func:`find_address`: "Міє, котра година?", "Як справи, Міє?").  After a
  delivered request a *conversation window* opens: while her answer is
  pending or being spoken, and ``follow_up_seconds`` after she finishes,
  anything said is meant for her, name or not.  A question or remark of
  her own that invites a reply (``say(..., opens_conversation=True)``) opens
  it too, once it has been heard;
* ``"open"`` - the microphone stays open and every utterance is for her.

In both open modes pure fillers ("угу", "ага", "мм") and what the
hallucination filter drops are ignored.  Her name alone keeps the open
microphone listening and makes the next phrase (within
:data:`CALLED_SECONDS`) the request - also when it was said while her name
was still being recognised.  At most one utterance is recognised at a time
and at most one waits behind it, kept only for that case; any other is
dropped (and counted) - work never piles up.  Push-to-talk works in every
mode.

A delivered request reaches ``on_result(text, status)`` with the status
saying how it reached her: ``"ok"`` (push-to-talk), ``"named"`` (the open
microphone, addressed by name) or ``"heard"`` (the open microphone, not
addressed: "open" mode, the conversation window).  :attr:`VoiceService.last_delivery`
says when it was said and recognised (the start of her answer's latency trace).

Is the user talking with her?  :attr:`VoiceService.conversation_active` says so
(push-to-talk, recognition, a request waiting for her answer, her answer, her
name alone, the conversation window; in "open" mode also speech on the open
microphone).  ``on_activity`` fires whenever it may have changed - once per
operation, so a passing intermediate state never shows - and
:meth:`~VoiceService.conversation_deadline` is the next moment it may change by
time alone.  ``on_speech_start(source, for_her)`` fires the moment the user
starts talking (push-to-talk pressed, speech begins on the open microphone), so
the app can get the model ready while the user is still speaking.

Rules that protect the user:

* the microphone is open only in LISTENING (push-to-talk) and STANDBY;
  ``mic_allowed()`` (Permission.MICROPHONE) is checked every time it would
  open and before anything the open microphone heard is delivered, and
  :meth:`revoke_microphone` closes it at once;
* she never hears herself: the microphone is closed while she speaks
  (also between the sentences of an answer still being written, see
  :meth:`hold_for_reply`) and the first ``echo_tail_seconds`` after it
  reopens are ignored; pressing push-to-talk while she speaks interrupts her;
* unsolicited lines never pile up: a full queue drops them, stale ones are
  skipped, and none are queued while the user is talking to her, right
  after the user called her, or while she is voicing an answer sentence by
  sentence (never between two of its sentences);
* pausing the open microphone (:meth:`set_suspended`) also ends the
  conversation window: after a resume only her name opens it again.

Nothing here logs what was said or heard.
"""

from __future__ import annotations

import functools
import itertools
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from chibi.core.worker import SerialWorker
from chibi.personality.text_uk import normalize as normalize_uk
from chibi.voice.audio_io import AudioInput, AudioOutput, AudioUnavailable
from chibi.voice.stt import Segment, STTEngine, STTError, clean_segments
from chibi.voice.tts import AudioClip, TTSEngine, TTSError
from chibi.voice.vad import EnergyVAD, VADConfig

log = logging.getLogger(__name__)


class VoiceState(StrEnum):
    OFF = "off"
    IDLE = "idle"
    STANDBY = "standby"  # open microphone: waiting for her name / any phrase
    LISTENING = "listening"  # push-to-talk: the user is talking to her
    TRANSCRIBING = "transcribing"
    SPEAKING = "speaking"


class Outcome(StrEnum):
    PLAYED = "played"
    INTERRUPTED = "interrupted"  # the user cut her off: don't show the line either
    FAILED = "failed"  # synthesis/playback error: show the text instead
    DROPPED = "dropped"  # skipped (stale, cancelled, queue full, user talking): show the text instead


#: Listen modes (the same values as ``chibi.core.config.LISTEN_MODES``).
PUSH, NAME, OPEN = "push", "name", "open"
LISTEN_MODES = (PUSH, NAME, OPEN)

#: Audio right after the microphone reopens is ignored: the tail of her own voice.
ECHO_TAIL_SECONDS = 0.3
#: After a request, the conversation stays open this long for her answer to start.
REPLY_WAIT_SECONDS = 30.0
#: Open-microphone utterances shorter than this (voiced) are a clack or a cough.
OPEN_MIC_MIN_SPEECH_MS = 350
#: After her name alone, the next phrase said within this long is the request.
CALLED_SECONDS = 8.0

#: ``on_result`` statuses of a delivered request (see the module docstring).
DELIVERED_PUSH = "ok"
DELIVERED_NAMED = "named"
DELIVERED_HEARD = "heard"
DELIVERED = frozenset({DELIVERED_PUSH, DELIVERED_NAMED, DELIVERED_HEARD})

#: ``on_speech_start`` sources: push-to-talk / the microphone button, or the open microphone.
SPEECH_PUSH = "push"
SPEECH_OPEN_MIC = "open_mic"


@dataclass(frozen=True, slots=True)
class Delivery:
    """When a delivered request was said and recognised (``clock`` seconds): see
    :attr:`VoiceService.last_delivery`."""

    #: The phrase was over: the end-of-phrase pause detected, or the key released.
    speech_end: float
    #: Its text was ready (main thread), just before ``on_result``.
    transcribed: float
    #: Silence waited for before ``speech_end`` (``end_silence_ms``; 0 for a released key).
    pause: float = 0.0


@dataclass
class _Utterance:
    text: str
    solicited: bool
    on_start: Callable[[float], None] | None
    on_done: Callable[[Outcome], None] | None
    created: float
    is_cancelled: Callable[[], bool] | None = None
    #: Synthesise while the line before it plays (the sentences of a streamed answer).
    prefetch: bool = False
    #: Once heard, what the user says next is for her (a question of hers).
    opens_conversation: bool = False
    id: int = field(default_factory=itertools.count().__next__)
    resolved: bool = False
    #: Synthesis: "" not started, "pending", "done" (``clip`` or ``error`` set).
    synth: str = ""
    clip: AudioClip | None = None
    error: str | None = None

    def cancelled(self) -> bool:
        if self.is_cancelled is None:
            return False
        try:
            return bool(self.is_cancelled())
        except Exception:
            log.exception("utterance cancel check failed")
            return False

    def finish(self, outcome: Outcome) -> None:
        if self.resolved:
            return
        self.resolved = True
        self.clip = None
        if self.on_done is not None:
            try:
                self.on_done(outcome)
            except Exception:
                log.exception("utterance callback failed")


class VoiceWorker(SerialWorker):
    """Recognition before synthesis before background loading."""

    TRANSCRIBE = 0
    SYNTHESIZE = 1
    BACKGROUND = 2

    def __init__(self) -> None:
        super().__init__("chibi-voice")


# -- her name in what was heard -----------------------------------------------------------------------


@dataclass(frozen=True)
class NameForms:
    """The forms of her name that mean "this is for her" (normalised words).

    * ``strong`` - an unambiguous vocative ("Міє"): addresses her wherever it stands;
    * ``weak`` - the nominative used as a vocative ("Мія") or a vocative that
      is also another case ("Аню"): only at the start, at the end, or set off
      by punctuation ("Як справи, Мія?"), so "подивись, що Мія робить" is not
      a request;
    * ``lead`` - other case forms ("Мію", "Мії"): only at the start of the
      phrase, where recognition sometimes mishears the vocative;
    * ``near`` - forms whose one-letter recognition slip still counts ("Алено"
      for "Олено"): only the vocative and the name itself, only names of at
      least :data:`NEAR_MIN_LETTERS` letters, and only at the start and set
      off like a vocative ("Алено, ..."; see :func:`find_address`).
    """

    strong: frozenset[str] = frozenset()
    weak: frozenset[str] = frozenset()
    lead: frozenset[str] = frozenset()
    near: frozenset[str] = frozenset()

    @property
    def every(self) -> frozenset[str]:
        return self.strong | self.weak | self.lead

    def __or__(self, other: NameForms) -> NameForms:
        return NameForms(self.strong | other.strong, self.weak | other.weak, self.lead | other.lead,
                         self.near | other.near)

    @classmethod
    def for_name(cls, name: str) -> NameForms:
        """Forms of a Ukrainian first name by the common declension patterns
        (Марія -> Маріє, Олена -> Олено, Аня -> Аню, Тарас -> Тарасе)."""
        words = normalize_uk(name).split()
        if not words:
            return cls()
        n = words[0]
        strong: set[str] = set()
        weak: set[str] = {n}
        lead: set[str] = set()
        stem = n[:-1]
        if len(n) < 2:
            pass
        elif n.endswith(("ія", "ея", "ая")):  # Мія, Марія, Софія: Міє, Мію, Мії, Мією
            strong.add(stem + "є")
            lead.update({stem + "ю", stem + "ї", stem + "єю"})
        elif n.endswith("я"):  # Аня, Катя: Аню (also the accusative), Ані, Анею
            weak.add(stem + "ю")
            lead.update({stem + "і", stem + "ею"})
        elif n.endswith("а"):  # Олена, Кіра: Олено, Олену, Олени, Олені, Оленою
            strong.add(stem + "о")
            lead.update({stem + "у", stem + "и", stem + "і", stem + "ою"})
        elif n.endswith(("й", "ь")):  # Андрій, Василь: Андрію, Василю (also the dative)
            weak.add(stem + "ю")
            lead.update({stem + "я", stem + "ем", stem + "єм"})
        elif n[-1] in "кгх":  # Марк: Марку
            weak.add(n + "у")
            lead.update({n + "а", n + "ом"})
        elif n[-1] not in "аеєиіїоуюяь'" and _CYRILLIC.search(n):  # Тарас: Тарасе
            strong.add(n + "е")
            lead.update({n + "а", n + "у", n + "ом", n + "ові"})
        near = {form for form in (*strong, n) if len(form) >= NEAR_MIN_LETTERS}
        return cls(frozenset(strong), frozenset(weak), frozenset(lead), frozenset(near))


#: A recognition slip of her name counts only for forms this long: in a shorter one a
#: one-letter difference is usually another word ("Віка" / "вікно", "Маша" / "мама",
#: "Кіра" / "кіно", "Ніка" / "ніби").
NEAR_MIN_LETTERS = 5


_CYRILLIC = re.compile(r"[а-яґєії]")

#: Her default name, with the spellings recognition produces for it.
MIA_FORMS = NameForms.for_name("Мія") | NameForms(weak=frozenset({"мия", "mia", "міа"}))

#: Words that may come before her name at the start ("Гей, Міє!", "Слухай, Мія, ...").
_LEAD_FILLERS = frozenset({"о", "ей", "гей", "слухай", "агов", "ну", "а"})
#: A phrase made only of these is not a request ("угу", "ну", "мм").
FILLER_WORDS = frozenset({
    "угу", "ага", "мгм", "м", "мм", "ммм", "е", "ее", "еее", "ем", "ну", "а", "аа", "ааа", "о", "оо",
    "ой", "ох", "ах", "ех", "ух", "уф", "хм", "гм", "еге",
})

_TOKEN = re.compile(r"[^\W_]+(?:['’ʼ`-][^\W_]+)*")
#: Punctuation that sets a vocative off ("Як справи, Мія?").
_SET_OFF = re.compile(r"[,.!?…;:—–-]")
_SENTENCE_END = re.compile(r"[.!?…;:]")


def is_filler(text: str) -> bool:
    """True if *text* is empty or only fillers ("угу", "ну ну", "Ммм...")."""
    return all(word in FILLER_WORDS for word in normalize_uk(text).split())


def find_address(text: str, names: NameForms = MIA_FORMS) -> str | None:
    """None if *text* does not address her by name; otherwise the request
    without the vocative ("" when the user only called her)::

        "Міє, котра година?"     -> "Котра година?"
        "Як справи, Міє?"        -> "Як справи?"
        "А ти, Міє, що думаєш?"  -> "А ти що думаєш?"
        "Гей, Міє!"              -> ""
        "Подивись, що Мія робить" -> None
    """
    tokens = list(_TOKEN.finditer(text))
    words = [normalize_uk(token.group()) for token in tokens]
    last = len(tokens) - 1
    for i, word in enumerate(words):
        if i < 3 and all(w in _LEAD_FILLERS for w in words[:i]):
            # At the start every form counts.  A recognition slip only of a long vocative
            # (or the name itself) that is set off like one: "Алено, ..." / "Алено!".
            after = text[tokens[i].end():tokens[i + 1].start()] if i < last else ""
            slip = (i == last or bool(_SET_OFF.search(after))) and any(_close(word, n) for n in names.near)
            if word in names.every or slip:
                return _clean_request(text[tokens[i].end():])
            continue
        if word in names.strong:
            hit = True
        elif word in names.weak:
            before = text[tokens[i - 1].end():tokens[i].start()]
            after = text[tokens[i].end():tokens[i + 1].start()] if i < last else ""
            hit = i == last or bool(_SET_OFF.search(before) or _SET_OFF.search(after))
        else:
            hit = False
        if hit:
            return _without(text, tokens, i)
    return None


def _without(text: str, tokens: list[re.Match[str]], i: int) -> str:
    """*text* without the vocative ``tokens[i]`` (not the first word)."""
    prev, name = tokens[i - 1], tokens[i]
    if i == len(tokens) - 1:  # "Як справи, Міє?" -> "Як справи?"
        return _clean_request(text[:prev.end()] + text[name.end():].strip())
    following = tokens[i + 1]
    before = text[prev.end():name.start()]
    after = text[name.end():following.start()]
    rest = text[following.start():]
    if _SENTENCE_END.search(after):  # "А ти, Міє? Що ..." keeps the question mark
        glue = after.replace(",", "").strip() + " "
    elif _SENTENCE_END.search(before):  # "Добре. Міє, котра ..." -> "Добре. Котра ..."
        glue = before.replace(",", "").rstrip() + " "
        rest = rest[:1].upper() + rest[1:]
    else:  # "А ти, Міє, що думаєш?" -> "А ти що думаєш?"
        glue = " "
    return _clean_request(text[:prev.end()] + glue + rest)


def _clean_request(text: str) -> str:
    text = text.strip().lstrip(",.!?…;:—–- \t").strip()
    if not _TOKEN.search(text):
        return ""
    return text[:1].upper() + text[1:]


def _close(a: str, b: str) -> bool:
    """Edit distance <= 1 (one typo-like recognition error)."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) <= 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def _words(text: str) -> list[str]:
    return normalize_uk(text).split()


def _echoes(text: str, prompt: str) -> bool:
    """Whisper sometimes "hears" its own prompt in noise: two or more words of
    *text* that are, in order, part of *prompt* are such an echo."""
    words, source = _words(text), _words(prompt)
    if len(words) < 2:
        return False
    return any(source[i:i + len(words)] == words for i in range(len(source) - len(words) + 1))


def _batched[F: Callable[..., Any]](method: F) -> F:
    """Main-thread entry points: ``on_activity`` fires once, after the whole call, so
    an intermediate state (idle for a moment between recognising a request and
    delivering it) never reaches the app as a change."""

    @functools.wraps(method)
    def wrapper(self: VoiceService, *args: Any, **kwargs: Any) -> Any:
        self._activity_depth += 1
        try:
            return method(self, *args, **kwargs)
        finally:
            self._activity_depth -= 1
            if self._activity_depth == 0 and self._activity_pending:
                self._activity_pending = False
                self._notify_activity()

    return wrapper  # type: ignore[return-value]


def _pause_of(event: Any, vad: EnergyVAD) -> float:
    """Seconds of silence the detector waited for before *event* ended the phrase."""
    return vad.config.end_silence_ms / 1000 if getattr(event, "reason", "") == "silence" else 0.0


class VoiceService:
    def __init__(
        self,
        *,
        post: Callable[[Callable[[], None]], None],
        output: AudioOutput | None = None,
        input_factory: Callable[[], AudioInput] | None = None,
        worker: Any = None,
        clock: Callable[[], float] = time.monotonic,
        mic_allowed: Callable[[], bool] = lambda: False,
        on_state: Callable[[VoiceState, str], None] | None = None,
        on_result: Callable[[str, str], None] | None = None,
        on_level: Callable[[float], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_wake: Callable[[], None] | None = None,
        on_speech_start: Callable[[str, bool], None] | None = None,
        on_activity: Callable[[], None] | None = None,
    ) -> None:
        """*on_speech_start(source, for_her)*: the user started talking - push-to-talk
        or the microphone button opened the microphone (``"push"``, for her), or the
        open microphone hears speech begin (``"open_mic"``; for her in "open" mode,
        in the conversation window or right after her name, not known yet otherwise).
        *on_activity*: something :attr:`conversation_active` depends on may have
        changed (besides the state, which ``on_state`` reports)."""
        self._post = post
        self._output = output
        self._input_factory = input_factory
        self._worker = worker if worker is not None else VoiceWorker()
        self._clock = clock
        self._mic_allowed = mic_allowed
        self._on_state = on_state
        self._on_result = on_result
        self._on_level = on_level
        self._on_error = on_error
        self._on_wake = on_wake
        self._on_speech_start = on_speech_start
        self._on_activity = on_activity
        self._activity_depth = 0
        self._activity_pending = False

        self.tts: TTSEngine | None = None
        self.stt: STTEngine | None = None
        self.volume = 0.8
        #: Recognition prompt for push-to-talk (her name, spelt right).
        self.prompt = "Мія, привіт."
        #: Recognition prompt on the open microphone: nothing a user would say,
        #: so an echo of it in noise is recognised as such (see :func:`_echoes`).
        self.standby_prompt = "Розмова українською. Мія слухає."
        self.names: NameForms = MIA_FORMS
        self.max_queue = 4
        self.stale_seconds = 20.0
        self.no_speech_timeout = 6.0
        self.listen_config = VADConfig()
        self.wake_config = VADConfig(min_speech_ms=OPEN_MIC_MIN_SPEECH_MS)
        #: "name" mode: after her answer, how long anything said is still for her (0: never).
        self.follow_up_seconds = 20.0
        self.reply_wait_seconds = REPLY_WAIT_SECONDS
        self.echo_tail_seconds = ECHO_TAIL_SECONDS
        #: Open-microphone utterances dropped because one was being recognised.
        self.dropped_utterances = 0

        self._state = VoiceState.OFF
        self._listen_mode = PUSH
        self._suspended = False  # e.g. a fullscreen game, a call, Do Not Disturb
        self._queue: deque[_Utterance] = deque()
        self._current: _Utterance | None = None
        self._speak_gen = 0
        self._listen_gen = 0
        #: Bumped whenever what the open microphone heard must no longer be delivered
        #: (revoke, pause, mode change, push-to-talk, new engines).
        self._open_gen = 0
        self._input: AudioInput | None = None
        self._vad: EnergyVAD | None = None
        self._vad_lock = threading.Lock()
        #: Background level learnt by the last session (see EnergyVAD).
        self._noise_floor: float | None = None
        self._listen_started = 0.0
        #: No-speech timeout of the current listening session (longer for hold-to-talk).
        self._listen_timeout = self.no_speech_timeout
        self._timeout_posted = False
        self._last_level_post = 0.0
        self._standby_busy = False
        self._listen_source = ""
        #: Open-microphone audio before this moment is ignored (her own voice's tail).
        self._echo_until = 0.0
        #: Between the sentences of an answer still being written, stay quiet until then.
        self._hold_until = 0.0
        #: SPEAKING with nothing to say: waiting for the next sentence of an answer.
        self._waiting_for_reply = False
        # The conversation window ("name" mode): see :attr:`conversation_open`.
        self._conv_until = 0.0
        self._conv_awaiting_until = 0.0
        self._conv_speaking = False
        #: Her name alone was heard: the next phrase until then is the request.
        self._called_until = 0.0
        #: One open-microphone utterance kept while another is recognised:
        #: (audio, open gen, when it ended, end-of-phrase pause).
        self._standby_waiting: tuple[Any, int, float, float] | None = None
        #: A delivered request waits for her answer until then (whatever ``follow_up_seconds``).
        self._awaiting_until = 0.0
        #: The open microphone hears speech now (main thread; set from the audio thread's posts).
        self._heard_speech = False
        #: Audio thread: the last speech state posted (under ``_vad_lock``).
        self._vad_talking = False
        #: Timing of the last delivered request (see :class:`Delivery`).
        self.last_delivery: Delivery | None = None

    # -- configuration ---------------------------------------------------------------------

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def can_speak(self) -> bool:
        return self.tts is not None and self._output is not None

    @property
    def can_listen(self) -> bool:
        return self.stt is not None and self._input_factory is not None

    @property
    def microphone_open(self) -> bool:
        return self._input is not None

    @property
    def listen_mode(self) -> str:
        return self._listen_mode

    @property
    def conversation_open(self) -> bool:
        """A request was delivered and she is answering or has just answered:
        in "name" mode anything said now is for her, without her name."""
        if self.follow_up_seconds <= 0:
            return False
        now = self._clock()
        return self._conv_speaking or now < self._conv_until or now < self._conv_awaiting_until

    @property
    def conversation_active(self) -> bool:
        """The user is talking with her right now: push-to-talk is listening, a phrase
        is being recognised, a delivered request waits for her answer (at most
        ``reply_wait_seconds``), her answer is being spoken (also between its
        sentences), her name alone was just heard, the conversation window is open -
        or, where everything heard is for her ("open" mode), the open microphone
        hears speech or recognises a phrase.  In "name" mode a phrase without her name
        counts only once it turned out to be for her: the TV must not count."""
        if self._state in (VoiceState.LISTENING, VoiceState.TRANSCRIBING):
            return True
        now = self._clock()
        if now < self._awaiting_until or self._called() or self._answering():
            return True
        if any(u is not None and u.solicited for u in (self._current, *self._queue)):
            return True  # an answer to the user is being said (or waits to be)
        if self.conversation_open:
            return True
        return (self._state == VoiceState.STANDBY and self._listen_mode == OPEN
                and (self._heard_speech or self._standby_busy))

    def conversation_deadline(self) -> float | None:
        """The next moment (``clock`` seconds) :attr:`conversation_active` may change
        by time alone (a wait or a window running out), or None."""
        now = self._clock()
        times = [self._awaiting_until, self._called_until, self._hold_until]
        if self.follow_up_seconds > 0:
            times += [self._conv_until, self._conv_awaiting_until]
        future = [t for t in times if t > now]
        return min(future) if future else None

    @_batched
    def configure(self, *, tts: TTSEngine | None, stt: STTEngine | None) -> None:
        """Swap engines (settings changed).  Interrupts whatever is going on."""
        self.interrupt()
        self._close_input()
        self.tts, self.stt = tts, stt
        self._listen_gen += 1
        self._open_gen += 1
        self._rest("configured")

    def preload(self) -> None:
        """Load the models in the background now, so the first answer isn't
        delayed by loading (seconds for recognition, ~1 s for the voice), then warm
        recognition up (one throwaway recognition, seconds) - each a background job of
        its own, so a request or a voice sample that comes in meanwhile waits for one
        load at most, never for the warm-up."""
        for engine in (self.stt, self.tts):
            load = getattr(engine, "load", None)
            if load is not None:
                def job(load=load) -> None:
                    try:
                        load()
                    except Exception as exc:
                        message = getattr(exc, "user_message", str(exc))
                        self._post(lambda: self._error(message))
                self._worker.submit(job, VoiceWorker.BACKGROUND)
        warm_up = getattr(self.stt, "warm_up", None)
        if callable(warm_up):
            def warm(warm_up=warm_up) -> None:
                try:
                    warm_up()
                except Exception as exc:  # never fatal: the first request pays the setup
                    log.debug("speech recognition warm-up failed: %s", type(exc).__name__)
            self._worker.submit(warm, VoiceWorker.BACKGROUND)

    @_batched
    def set_listen_mode(self, mode: str) -> None:
        """"push", "name" or "open" (see the module docstring)."""
        mode = mode if mode in LISTEN_MODES else PUSH
        if mode == self._listen_mode:
            return
        self._listen_mode = mode
        self._open_gen += 1
        self._end_conversation()
        if self._state in (VoiceState.IDLE, VoiceState.STANDBY):
            self._rest("listen mode changed")

    @_batched
    def set_suspended(self, suspended: bool) -> None:
        """Pause the open microphone (push-to-talk keeps working).  What it
        heard just before is dropped, not delivered later, and the
        conversation window ends: after a resume only her name opens it."""
        if suspended == self._suspended:
            return
        self._suspended = suspended
        if suspended:
            self._open_gen += 1
            self._standby_waiting = None
            self._end_conversation()
        if self._state in (VoiceState.IDLE, VoiceState.STANDBY):
            self._rest("suspended" if suspended else "resumed")

    @_batched
    def revoke_microphone(self) -> None:
        """Permission withdrawn / kill switch: close the microphone now.  A
        transcript of speech heard before the revoke never arrives after it."""
        self._open_gen += 1
        self._end_conversation()
        if self._state in (VoiceState.LISTENING, VoiceState.STANDBY, VoiceState.TRANSCRIBING):
            self._listen_gen += 1
            self._close_input()
            self._rest("microphone revoked")
        else:
            self._close_input()

    @_batched
    def shutdown(self) -> None:
        self.interrupt()
        self._close_input()
        self._listen_gen += 1
        self._open_gen += 1
        self._set_state(VoiceState.OFF, "shutdown")
        shutdown = getattr(self._worker, "shutdown", None)
        if shutdown is not None:
            shutdown()

    # -- speaking ----------------------------------------------------------------------------

    @_batched
    def say(self, text: str, *, solicited: bool = False,
            on_start: Callable[[float], None] | None = None,
            on_done: Callable[[Outcome], None] | None = None,
            is_cancelled: Callable[[], bool] | None = None,
            prefetch: bool = False, opens_conversation: bool = False) -> bool:
        """Queue a line.  False (and no callbacks) if voice can't speak now -
        the caller then shows the text without voice.  A line whose
        *is_cancelled* returns True before its audio starts is skipped
        (``Outcome.DROPPED``).  *prefetch*: synthesise it while the line
        before it plays, so the sentences of an answer follow without a gap.
        *opens_conversation*: a question or remark of hers that invites a
        reply - once its audio has played, the conversation window opens."""
        if not self.can_speak or not text.strip() or self._state == VoiceState.OFF:
            return False
        if not solicited and (self._state in (VoiceState.LISTENING, VoiceState.TRANSCRIBING)
                              or self._user_talking() or self._called() or self._answering()):
            return False  # never talk over the user, nor into the middle of her own answer
        utterance = _Utterance(text, solicited, on_start, on_done, self._clock(), is_cancelled,
                               prefetch=prefetch, opens_conversation=opens_conversation)
        if len(self._queue) >= self.max_queue:
            victim = next((u for u in self._queue if not u.solicited), None)
            if victim is None and not solicited:
                return False
            if victim is not None:
                self._queue.remove(victim)
                victim.finish(Outcome.DROPPED)
        self._queue.append(utterance)
        if self._current is None and self._state not in (VoiceState.LISTENING, VoiceState.TRANSCRIBING):
            self._next()
        else:
            self._prefetch()
        return True

    @_batched
    def interrupt(self) -> None:
        """Stop speaking now and forget queued lines (the user cut her off)."""
        self._hold_until = 0.0
        if self._current is None and not self._queue:
            if self._waiting_for_reply:
                self._waiting_for_reply = False
                self._rest("interrupted")  # she was waiting for the rest of an answer
            return
        self._speak_gen += 1
        current, self._current = self._current, None
        if self._output is not None:
            try:
                self._output.stop()
            except Exception:
                log.exception("stopping playback failed")
        if current is not None:
            current.finish(Outcome.INTERRUPTED)
        while self._queue:
            self._queue.popleft().finish(Outcome.INTERRUPTED)
        if self._state == VoiceState.SPEAKING:
            self._rest("interrupted")
        self._activity()

    @_batched
    def hold_for_reply(self, seconds: float) -> None:
        """An answer is still being written (streamed): when a line ends and
        the next is not there yet, stay quiet with the microphone closed for
        up to *seconds* instead of reopening it between two sentences."""
        self._hold_until = self._clock() + max(0.0, float(seconds))
        self._activity()

    @_batched
    def release_reply_hold(self) -> None:
        """The answer is complete (or abandoned): no more waiting for it."""
        self._hold_until = 0.0
        self.check_hold()
        self._activity()

    @_batched
    def check_hold(self) -> None:
        """Back to rest if she is only waiting for a reply hold that is over."""
        if (self._waiting_for_reply and self._current is None and not self._queue
                and not self._held()):
            self._waiting_for_reply = False
            self._rest("done speaking")

    @_batched
    def note_reply_shown(self, seconds: float = 0.0) -> None:
        """An answer was shown without voice (*seconds* to read it): the
        conversation window runs from now, as if she had just said it."""
        self._awaiting_until = 0.0  # her answer has come
        self._activity()
        if not self.conversation_open:
            return
        self._conv_awaiting_until = 0.0
        self._conv_until = max(self._conv_until,
                               self._clock() + max(0.0, seconds) + self.follow_up_seconds)

    def _held(self) -> bool:
        return self._hold_until > self._clock()

    def _answering(self) -> bool:
        """She is voicing an answer sentence by sentence (also between two of its
        sentences): a line of her own must not come in between."""
        if self._waiting_for_reply or self._held():
            return True
        return any(u is not None and u.solicited and u.prefetch for u in (self._current, *self._queue))

    def _called(self) -> bool:
        """Her name alone was just heard: the user is about to say the request."""
        return self._clock() < self._called_until

    def _user_talking(self) -> bool:
        """The open microphone hears someone talking right now."""
        if self._state != VoiceState.STANDBY:
            return False
        with self._vad_lock:
            return self._vad is not None and self._vad.in_speech

    def _next(self) -> bool:
        """Start the next queued line; False if there was nothing (fresh) to say."""
        now = self._clock()
        while self._queue:
            utterance = self._queue.popleft()
            if utterance.cancelled() or (not utterance.solicited
                                         and now - utterance.created > self.stale_seconds):
                utterance.finish(Outcome.DROPPED)
                continue
            self._current = utterance
            self._waiting_for_reply = False
            if self.conversation_open:
                self._conv_speaking = True  # the window stays open while she talks
            if utterance.solicited:
                self._conv_awaiting_until = self._awaiting_until = 0.0  # her answer has come
            self._activity()
            self._close_input()  # never listen to herself
            self._set_state(VoiceState.SPEAKING, "speaking")
            gen = self._speak_gen
            if utterance.synth == "done":  # synthesised while the line before it played
                if utterance.clip is not None:
                    self._play(utterance, utterance.clip, gen)
                else:
                    self._synth_failed(utterance, gen, utterance.error or "")
            elif utterance.synth == "":
                self._synthesize(utterance)
            return True
        self._current = None
        return False

    def _continue(self) -> None:
        """After a line: the next one, or back to rest."""
        if self._current is not None:
            return  # an on_done callback already started the next line (re-entrant say())
        if self._next():
            return
        if self._held() and self._state == VoiceState.SPEAKING:
            self._waiting_for_reply = True  # the rest of her answer is on its way: stay quiet
            return
        self._rest("done speaking")

    def _synthesize(self, utterance: _Utterance) -> None:
        utterance.synth = "pending"
        gen = self._speak_gen
        tts = self.tts
        self._worker.submit(lambda: self._synth_job(tts, utterance, gen), VoiceWorker.SYNTHESIZE)

    def _prefetch(self) -> None:
        """Synthesise the next line now if it asked for that (see :meth:`say`)."""
        if self._current is None or not self._queue:
            return
        upcoming = self._queue[0]
        if upcoming.prefetch and upcoming.synth == "" and not upcoming.cancelled():
            self._synthesize(upcoming)

    def _synth_job(self, tts: TTSEngine | None, utterance: _Utterance, gen: int) -> None:
        if tts is None:
            self._post(lambda: self._synthesized(utterance, gen, None, "voice switched off"))
            return
        try:
            clip = tts.synthesize(utterance.text)
        except TTSError as exc:
            message = exc.user_message
            self._post(lambda: self._synthesized(utterance, gen, None, message))
            return
        except Exception as exc:  # engine bug: never let it take the app down
            log.exception("speech synthesis failed")
            message = str(exc)
            self._post(lambda: self._synthesized(utterance, gen, None, message))
            return
        self._post(lambda: self._synthesized(utterance, gen, clip, None))

    @_batched
    def _synthesized(self, utterance: _Utterance, gen: int, clip: AudioClip | None,
                     error: str | None) -> None:
        if gen != self._speak_gen or utterance.resolved:
            return
        utterance.synth = "done"
        utterance.clip, utterance.error = clip, error
        if utterance is not self._current:
            return  # synthesised ahead: it plays when its turn comes
        if clip is None:
            self._synth_failed(utterance, gen, error or "")
        else:
            self._play(utterance, clip, gen)

    def _synth_failed(self, utterance: _Utterance, gen: int, message: str) -> None:
        if gen != self._speak_gen or utterance is not self._current:
            return
        log.warning("speech synthesis failed: %s", message)
        if self._on_error is not None:
            self._on_error(message)
        self._current = None
        utterance.finish(Outcome.FAILED)
        self._continue()

    def _play(self, utterance: _Utterance, clip: AudioClip, gen: int) -> None:
        if gen != self._speak_gen or utterance is not self._current:
            return
        if utterance.cancelled():
            self._current = None
            utterance.finish(Outcome.DROPPED)
            self._continue()
            return
        if clip.empty or self._output is None:
            self._current = None
            utterance.finish(Outcome.PLAYED if clip.empty else Outcome.FAILED)
            self._continue()
            return
        try:
            self._output.play(clip, self.volume,
                              lambda completed, u=utterance, g=gen: self._post(
                                  lambda: self._played(u, g, completed)))
        except AudioUnavailable as exc:
            if self._on_error is not None:
                self._on_error(exc.user_message)
            self._current = None
            utterance.finish(Outcome.FAILED)
            self._continue()
            return
        if utterance.opens_conversation and self.follow_up_seconds > 0:
            self._conv_speaking = True  # heard: the follow-up time starts when she stops
        if utterance.on_start is not None:
            try:
                utterance.on_start(clip.duration)
            except Exception:
                log.exception("utterance start callback failed")
        if utterance is self._current:
            self._prefetch()

    @_batched
    def _played(self, utterance: _Utterance, gen: int, completed: bool) -> None:
        if gen != self._speak_gen or utterance is not self._current:
            return
        self._current = None
        utterance.finish(Outcome.PLAYED if completed else Outcome.INTERRUPTED)
        self._continue()

    # -- listening ----------------------------------------------------------------------------

    @_batched
    def toggle_listening(self, source: str = "hotkey") -> None:
        """Push-to-talk press in toggle mode / the chat microphone button."""
        if self._state == VoiceState.LISTENING:
            self.stop_listening()
        else:
            self.listen(source)

    @_batched
    def listen(self, source: str = "hotkey", *, hold: bool = False) -> bool:
        """Open the microphone for one utterance; interrupts her speech.  With
        *hold* (hold-to-talk) a pause does not end the utterance: letting go of
        the key (:meth:`stop_listening`) does, or the length limit."""
        if not self.can_listen:
            self._error("Розпізнавання мовлення не налаштоване (Налаштування → Голос).")
            return False
        if not self._mic_allowed():
            self._error("Мікрофон вимкнено в «Приватності». Дозволь його, щоб говорити зі мною.")
            return False
        if self._state in (VoiceState.LISTENING, VoiceState.TRANSCRIBING):
            return self._state == VoiceState.LISTENING
        self._waiting_for_reply = False  # no detour through rest: listening starts right here
        self.interrupt()
        self._close_input()
        self._listen_gen += 1
        self._open_gen += 1  # push-to-talk takes over from the open microphone
        gen = self._listen_gen
        config, timeout = self.listen_config, self.no_speech_timeout
        if hold:
            # Like a walkie-talkie: the user thinks aloud while holding the key.
            # Still bounded, should the release never arrive.
            longest_ms = round(config.max_utterance_seconds * 1000)
            config = replace(config, end_silence_ms=longest_ms + config.frame_ms)
            timeout = max(timeout, config.max_utterance_seconds)
        with self._vad_lock:
            self._vad = EnergyVAD(config, initial_floor=self._noise_floor)
        self._listen_started = self._clock()
        self._listen_timeout = timeout
        self._timeout_posted = False
        self._listen_source = source
        if not self._open_input(lambda chunk, g=gen: self._on_listen_audio(chunk, g)):
            self._rest("microphone failed")
            return False
        self._set_state(VoiceState.LISTENING, source)
        self._speech_started(SPEECH_PUSH, True)
        return True

    @_batched
    def stop_listening(self) -> None:
        """Push-to-talk released / pressed again: use what was said so far."""
        if self._state != VoiceState.LISTENING:
            return
        with self._vad_lock:
            vad, self._vad = self._vad, None
            if vad is not None:
                self._noise_floor = vad.noise_floor
            audio = vad.finish() if vad is not None else None
        self._close_input()
        self._listen_gen += 1
        if audio is None or len(audio) == 0:
            self._rest("nothing heard")
            self._result("", "no_speech")
            return
        self._transcribe(audio, self._clock(), 0.0)

    @_batched
    def cancel_listening(self) -> None:
        if self._state not in (VoiceState.LISTENING, VoiceState.TRANSCRIBING):
            return
        self._listen_gen += 1
        self._close_input()
        self._drop_vad()
        self._rest("cancelled")

    def _on_listen_audio(self, chunk: Any, gen: int) -> None:
        """Audio thread."""
        with self._vad_lock:
            vad = self._vad
            if vad is None or gen != self._listen_gen:
                return
            events = vad.feed(chunk)
            level = vad.level
            talking = vad.in_speech
        now = self._clock()
        ended = False
        for event in events:
            if event.kind == "end" and event.audio is not None:
                ended = True
                audio, pause = event.audio, _pause_of(event, vad)
                self._post(lambda a=audio, g=gen, t=now, p=pause: self._utterance_ready(a, g, t, p))
        if not ended and not talking and not self._timeout_posted \
                and now - self._listen_started > self._listen_timeout:
            self._timeout_posted = True
            self._post(lambda g=gen: self._no_speech(g))
        if self._on_level is not None and now - self._last_level_post >= 0.1:
            self._last_level_post = now
            on_level = self._on_level
            self._post(lambda lv=level: on_level(lv))

    @_batched
    def _utterance_ready(self, audio: Any, gen: int, ended: float | None = None, pause: float = 0.0) -> None:
        if gen != self._listen_gen or self._state != VoiceState.LISTENING:
            return
        self._drop_vad()
        self._close_input()
        self._listen_gen += 1
        self._transcribe(audio, self._clock() if ended is None else ended, pause)

    @_batched
    def _no_speech(self, gen: int) -> None:
        if gen != self._listen_gen or self._state != VoiceState.LISTENING:
            return
        self.cancel_listening()
        self._result("", "no_speech")

    def _transcribe(self, audio: Any, ended: float, pause: float) -> None:
        """Recognise a push-to-talk phrase that ended at *ended* (``clock``) after a
        *pause* of silence."""
        self._set_state(VoiceState.TRANSCRIBING, "")
        gen = self._listen_gen
        stt, prompt = self.stt, self.prompt
        self._worker.submit(lambda: self._stt_job(stt, audio, prompt, gen, ended, pause),
                            VoiceWorker.TRANSCRIBE)

    def _stt_job(self, stt: STTEngine | None, audio: Any, prompt: str, gen: int,
                 ended: float | None = None, pause: float = 0.0) -> None:
        try:
            if stt is None:
                raise STTError("speech recognition switched off")
            transcript = stt.transcribe(audio, prompt=prompt)
        except STTError as exc:
            message = exc.user_message
            self._post(lambda: self._transcribed(gen, "", message))
            return
        except Exception as exc:
            log.exception("speech recognition failed")
            message = str(exc)
            self._post(lambda: self._transcribed(gen, "", message))
            return
        text = transcript.text
        self._post(lambda: self._transcribed(gen, text, "", ended, pause))

    @_batched
    def _transcribed(self, gen: int, text: str, error: str, ended: float | None = None,
                     pause: float = 0.0) -> None:
        if gen != self._listen_gen or self._state != VoiceState.TRANSCRIBING:
            return
        self._rest("transcribed")
        if error:
            self._error(error)
            self._result("", "error")
        elif text:
            self._deliver(text, DELIVERED_PUSH, ended, pause)
        else:
            self._result("", "not_understood")

    def _deliver(self, text: str, status: str, ended: float | None = None, pause: float = 0.0) -> None:
        """A request for her: hand it over (*status*: how it reached her, see
        :data:`DELIVERED`), note when it was said (:attr:`last_delivery`) and open
        the conversation window."""
        now = self._clock()
        self._called_until = 0.0
        self._awaiting_until = now + self.reply_wait_seconds
        if self.follow_up_seconds > 0:
            self._conv_awaiting_until = now + self.reply_wait_seconds
            self._conv_until = max(self._conv_until, now + self.follow_up_seconds)
        self.last_delivery = Delivery(speech_end=now if ended is None else min(ended, now),
                                      transcribed=now, pause=max(0.0, pause))
        self._result(text, status)
        self._activity()

    def _end_conversation(self) -> None:
        self._conv_until = self._conv_awaiting_until = self._called_until = self._awaiting_until = 0.0
        self._conv_speaking = False
        self._activity()

    # -- the open microphone ("name" / "open" modes) ---------------------------------------------

    def _open_wanted(self) -> bool:
        return (self._listen_mode != PUSH and not self._suspended and self.can_listen
                and self._mic_allowed())

    def _enter_standby(self) -> None:
        self._listen_gen += 1
        gen = self._listen_gen
        with self._vad_lock:
            self._vad = EnergyVAD(self.wake_config, initial_floor=self._noise_floor)
            self._vad_talking = False
        if self._open_input(lambda chunk, g=gen: self._on_standby_audio(chunk, g)):
            self._set_state(VoiceState.STANDBY, f"open microphone ({self._listen_mode})")
        else:
            # Not retried on every pause: the user was told; the integration
            # sets the mode again when the settings change.
            self._listen_mode = PUSH
            self._open_gen += 1
            self._drop_vad()
            self._set_state(VoiceState.IDLE, "microphone failed")

    def _on_standby_audio(self, chunk: Any, gen: int) -> None:
        """Audio thread."""
        if self._clock() < self._echo_until:
            return  # the tail of her own voice
        with self._vad_lock:
            vad = self._vad
            if vad is None or gen != self._listen_gen:
                return
            events = vad.feed(chunk)
            open_gen = self._open_gen
            talking = vad.in_speech
            changed = talking != self._vad_talking
            self._vad_talking = talking
        now = self._clock()
        started = any(event.kind == "start" for event in events)
        if started or changed:
            self._post(lambda g=open_gen, t=talking, s=started: self._standby_speech(gen, g, t, s))
        for event in events:
            if event.kind == "end" and event.audio is not None:
                audio, pause = event.audio, _pause_of(event, vad)
                self._post(lambda a=audio, g=open_gen, t=now, p=pause: self._standby_utterance(a, g, t, p))

    @_batched
    def _standby_speech(self, gen: int, open_gen: int, talking: bool, started: bool) -> None:
        """The open microphone heard speech begin (*started*) or stop."""
        if (gen != self._listen_gen or not self._open_result_valid(open_gen)
                or self._state != VoiceState.STANDBY):
            return
        if talking != self._heard_speech:
            self._heard_speech = talking
            self._activity()
        if started:
            for_her = self._listen_mode == OPEN or self.conversation_open or self._called()
            self._speech_started(SPEECH_OPEN_MIC, for_her)

    def _open_result_valid(self, open_gen: int) -> bool:
        return (open_gen == self._open_gen and self._open_wanted()
                and self._state not in (VoiceState.LISTENING, VoiceState.TRANSCRIBING))

    @_batched
    def _standby_utterance(self, audio: Any, open_gen: int, ended: float | None = None,
                           pause: float = 0.0) -> None:
        if not self._open_result_valid(open_gen):
            return
        ended = self._clock() if ended is None else ended
        if self._standby_busy:
            # One recognition at a time: under constant talk (a TV) work never piles up.
            # The newest phrase waits - it is the request if the one being recognised
            # turns out to be her name alone ("Міє, ... котра година?").
            if self._standby_waiting is not None:
                self._drop_utterance()
            self._standby_waiting = (audio, open_gen, ended, pause)
            return
        self._standby_busy = True
        self._activity()
        stt, prompt = self.stt, self.standby_prompt

        def job() -> None:
            try:
                text = stt.transcribe(audio, prompt=prompt).text if stt is not None else ""
            except Exception as exc:
                log.debug("open microphone: recognition failed (%s)", type(exc).__name__)
                text = ""
            self._post(lambda: self._standby_heard(open_gen, text, ended, pause))

        self._worker.submit(job, VoiceWorker.TRANSCRIBE)

    @_batched
    def _standby_heard(self, open_gen: int, text: str, ended: float | None = None,
                       pause: float = 0.0) -> None:
        self._standby_busy = False
        self._activity()
        waiting, self._standby_waiting = self._standby_waiting, None
        if not self._open_result_valid(open_gen):
            return  # paused, revoked, mode changed or push-to-talk took over meanwhile
        found = self._request_from(text)
        if found is not None and not found[0]:
            # Only her name: the microphone stays open and the next phrase is the
            # request - also one said while her name was being recognised.
            self._called_until = self._clock() + CALLED_SECONDS
            if self._on_wake is not None:
                self._on_wake()
            if waiting is not None:
                self._standby_utterance(*waiting)
            return
        if waiting is not None:
            self._drop_utterance()
        if found is not None:
            self._deliver(*found, ended, pause)

    def _drop_utterance(self) -> None:
        self.dropped_utterances += 1
        log.debug("open microphone busy: %d utterance(s) dropped so far", self.dropped_utterances)

    def _request_from(self, text: str) -> tuple[str, str] | None:
        """What of *text* is for her: None (not for her), or (request, status) -
        the request is "" when the user only called her."""
        text = text.strip()
        if not text or _echoes(text, self.standby_prompt) or not clean_segments([Segment(text)])[0]:
            return None  # silence, a sound, a subtitle credit, the prompt echoed back
        addressed = find_address(text, self.names)
        if addressed is not None:
            return ("" if is_filler(addressed) else addressed), DELIVERED_NAMED
        if is_filler(text):
            return None
        if self._called():  # the phrase right after her name alone
            return _clean_request(text), DELIVERED_NAMED
        if self._listen_mode == OPEN or self.conversation_open:
            return text, DELIVERED_HEARD
        return None

    # -- helpers ------------------------------------------------------------------------------

    def _open_input(self, callback: Callable[[Any], None]) -> bool:
        if self._input_factory is None:
            return False
        try:
            stream = self._input_factory()
            stream.start(callback)
        except AudioUnavailable as exc:
            self._error(exc.user_message)
            return False
        except Exception as exc:
            log.exception("cannot open the microphone")
            self._error(f"Мікрофон недоступний: {exc}")
            return False
        self._input = stream
        return True

    def _drop_vad(self) -> None:
        with self._vad_lock:
            vad, self._vad = self._vad, None
            if vad is not None:
                self._noise_floor = vad.noise_floor
        if self._heard_speech:
            self._heard_speech = False
            self._activity()

    def _close_input(self) -> None:
        """Close the microphone (and keep the background level it learnt)."""
        self._drop_vad()
        stream, self._input = self._input, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                log.exception("closing the microphone failed")

    def _rest(self, reason: str) -> None:
        """Back to IDLE, or STANDBY if the open microphone is wanted."""
        self._waiting_for_reply = False
        if self.tts is None and self.stt is None:
            self._close_input()
            self._set_state(VoiceState.OFF, reason)
            return
        if self._current is not None:
            return  # still speaking
        if self._queue and self._next():
            return  # a line was waiting for the user to finish
        if self._open_wanted():
            if self._state != VoiceState.STANDBY or self._input is None:
                self._close_input()
                self._enter_standby()
            return
        self._close_input()
        self._set_state(VoiceState.IDLE, reason)

    def _set_state(self, state: VoiceState, reason: str) -> None:
        if state == self._state:
            return
        previous, self._state = self._state, state
        if previous == VoiceState.SPEAKING:
            now = self._clock()
            self._echo_until = now + self.echo_tail_seconds
            if self._conv_speaking:  # she has answered: the follow-up time starts now
                self._conv_speaking = False
                self._conv_until = max(self._conv_until, now + self.follow_up_seconds)
        if self._on_state is not None:
            self._on_state(state, reason)
        self._activity()

    def note_activity(self) -> None:
        """Something outside the service that the conversation depends on changed (an
        answer being written): ``on_activity`` fires - after the operation in
        progress, if one is."""
        self._activity()

    def _activity(self) -> None:
        """Something :attr:`conversation_active` depends on may have changed."""
        if self._activity_depth > 0:
            self._activity_pending = True
        else:
            self._notify_activity()

    def _notify_activity(self) -> None:
        if self._on_activity is None:
            return
        try:
            self._on_activity()
        except Exception:
            log.exception("voice activity callback failed")

    def _speech_started(self, source: str, for_her: bool) -> None:
        if self._on_speech_start is None:
            return
        try:
            self._on_speech_start(source, for_her)
        except Exception:
            log.exception("speech start callback failed")

    def _result(self, text: str, status: str) -> None:
        if self._on_result is not None:
            self._on_result(text, status)

    def _error(self, message: str) -> None:
        log.info("voice: %s", message)
        if self._on_error is not None:
            self._on_error(message)
