"""The voice conversation: listen -> recognise -> (brain answers) -> speak.

:class:`VoiceService` is a state machine driven from the Qt main thread.
Slow work (recognition, synthesis) runs on one background worker thread;
audio callbacks run on PortAudio's thread.  Both hand results back through
the injected ``post`` function, which must run the callable on the main
thread (the app passes a queued Qt invoker; tests run it inline).  Every
asynchronous result carries a *generation* number: when the user interrupts
or cancels, the generation changes and late results are ignored.

States::

    OFF ---(engines configured)---> IDLE <--> STANDBY (wake phrase enabled)
    IDLE/STANDBY/SPEAKING --listen()--> LISTENING --utterance--> TRANSCRIBING --> IDLE
    IDLE/STANDBY --say()--> SPEAKING --(queue empty)--> IDLE

Rules that protect the user:

* the microphone is open only in LISTENING (push-to-talk) and STANDBY (wake
  phrase, opt-in); ``mic_allowed()`` (Permission.MICROPHONE) is checked
  every time it would open, and :meth:`revoke_microphone` closes it at once;
* while she speaks, the wake-phrase microphone is closed so she can't hear
  herself; pressing push-to-talk while she speaks interrupts her;
* unsolicited lines never pile up: a full queue drops them, stale ones are
  skipped, and none are queued while the user is talking to her.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from chibi.core.worker import SerialWorker
from chibi.personality.text_uk import normalize as normalize_uk
from chibi.voice.audio_io import AudioInput, AudioOutput, AudioUnavailable
from chibi.voice.stt import STTEngine, STTError
from chibi.voice.tts import AudioClip, TTSEngine, TTSError
from chibi.voice.vad import EnergyVAD, VADConfig

log = logging.getLogger(__name__)


class VoiceState(StrEnum):
    OFF = "off"
    IDLE = "idle"
    STANDBY = "standby"  # microphone open, waiting for the wake phrase
    LISTENING = "listening"  # push-to-talk: the user is talking to her
    TRANSCRIBING = "transcribing"
    SPEAKING = "speaking"


class Outcome(StrEnum):
    PLAYED = "played"
    INTERRUPTED = "interrupted"  # the user cut her off: don't show the line either
    FAILED = "failed"  # synthesis/playback error: show the text instead
    DROPPED = "dropped"  # skipped (stale, queue full, user talking): show the text instead


@dataclass
class _Utterance:
    text: str
    solicited: bool
    on_start: Callable[[float], None] | None
    on_done: Callable[[Outcome], None] | None
    created: float
    id: int = field(default_factory=itertools.count().__next__)
    resolved: bool = False

    def finish(self, outcome: Outcome) -> None:
        if self.resolved:
            return
        self.resolved = True
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


#: How she can be called by name in the wake phrase (normalised forms).
WAKE_WORDS = ("мія", "міє", "мію", "мії", "мією", "мия", "mia", "мiя")
_WAKE_FILLERS = {"о", "ей", "гей", "слухай", "агов", "ну", "а"}


def match_wake(text: str, names: tuple[str, ...] = WAKE_WORDS) -> str | None:
    """None if *text* doesn't start with her name; otherwise what follows the
    name ("" when the user only called her).  "Мія, котра година?" ->
    "котра година"; "Гей, Міє!" -> ""."""
    words = normalize_uk(text).split()
    names_n = {normalize_uk(n) for n in names if n.strip()}
    for i, word in enumerate(words[:3]):
        if word in names_n or any(_close(word, n) for n in names_n if len(n) >= 4):
            return " ".join(words[i + 1:])
        if word not in _WAKE_FILLERS:
            return None
    return None


def _close(a: str, b: str) -> bool:
    """Edit distance <= 1 (one typo-like recognition error)."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) <= 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


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
    ) -> None:
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

        self.tts: TTSEngine | None = None
        self.stt: STTEngine | None = None
        self.volume = 0.8
        self.prompt = "Мія, привіт."
        self.wake_names: tuple[str, ...] = WAKE_WORDS
        self.max_queue = 4
        self.stale_seconds = 20.0
        self.no_speech_timeout = 6.0
        self.listen_config = VADConfig()
        self.wake_config = VADConfig(end_silence_ms=500, max_utterance_seconds=6.0)

        self._state = VoiceState.OFF
        self._wake_enabled = False
        self._suspended = False  # e.g. a fullscreen game without wake-in-games
        self._queue: deque[_Utterance] = deque()
        self._current: _Utterance | None = None
        self._speak_gen = 0
        self._listen_gen = 0
        self._input: AudioInput | None = None
        self._vad: EnergyVAD | None = None
        self._vad_lock = threading.Lock()
        #: Background level learnt by the last session (see EnergyVAD).
        self._noise_floor: float | None = None
        self._listen_started = 0.0
        self._timeout_posted = False
        self._last_level_post = 0.0
        self._wake_check_pending = False
        self._listen_source = ""

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

    def configure(self, *, tts: TTSEngine | None, stt: STTEngine | None) -> None:
        """Swap engines (settings changed).  Interrupts whatever is going on."""
        self.interrupt()
        self._close_input()
        self.tts, self.stt = tts, stt
        self._listen_gen += 1
        self._rest("configured")

    def preload(self) -> None:
        """Load the models in the background now, so the first answer isn't
        delayed by loading (seconds for recognition, ~1 s for the voice)."""
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

    def set_wake(self, enabled: bool) -> None:
        self._wake_enabled = enabled
        if self._state in (VoiceState.IDLE, VoiceState.STANDBY):
            self._rest("wake changed")

    def set_suspended(self, suspended: bool) -> None:
        """Pause the wake-phrase microphone (push-to-talk keeps working)."""
        self._suspended = suspended
        if self._state in (VoiceState.IDLE, VoiceState.STANDBY):
            self._rest("suspended" if suspended else "resumed")

    def revoke_microphone(self) -> None:
        """Permission withdrawn / kill switch: close the microphone now."""
        if self._state in (VoiceState.LISTENING, VoiceState.STANDBY, VoiceState.TRANSCRIBING):
            # TRANSCRIBING too: a transcript of speech heard before the revoke
            # must not arrive after it.
            self._listen_gen += 1
            self._close_input()
            self._rest("microphone revoked")
        else:
            self._close_input()

    def shutdown(self) -> None:
        self.interrupt()
        self._close_input()
        self._listen_gen += 1
        self._set_state(VoiceState.OFF, "shutdown")
        shutdown = getattr(self._worker, "shutdown", None)
        if shutdown is not None:
            shutdown()

    # -- speaking ----------------------------------------------------------------------------

    def say(self, text: str, *, solicited: bool = False,
            on_start: Callable[[float], None] | None = None,
            on_done: Callable[[Outcome], None] | None = None) -> bool:
        """Queue a line.  False (and no callbacks) if voice can't speak now -
        the caller then shows the text without voice."""
        if not self.can_speak or not text.strip() or self._state == VoiceState.OFF:
            return False
        if self._state in (VoiceState.LISTENING, VoiceState.TRANSCRIBING) and not solicited:
            return False  # never talk over the user
        utterance = _Utterance(text, solicited, on_start, on_done, self._clock())
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
        return True

    def interrupt(self) -> None:
        """Stop speaking now and forget queued lines (the user cut her off)."""
        if self._current is None and not self._queue:
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

    def _next(self) -> bool:
        """Start the next queued line; False if there was nothing (fresh) to say."""
        now = self._clock()
        while self._queue:
            utterance = self._queue.popleft()
            if not utterance.solicited and now - utterance.created > self.stale_seconds:
                utterance.finish(Outcome.DROPPED)
                continue
            self._current = utterance
            self._close_input()  # never listen to herself
            self._set_state(VoiceState.SPEAKING, "speaking")
            gen = self._speak_gen
            tts = self.tts
            self._worker.submit(lambda u=utterance, g=gen, t=tts: self._synth_job(t, u, g),
                                VoiceWorker.SYNTHESIZE)
            return True
        self._current = None
        return False

    def _continue(self) -> None:
        """After a line: the next one, or back to rest."""
        if self._current is not None:
            return  # an on_done callback already started the next line (re-entrant say())
        if not self._next():
            self._rest("done speaking")

    def _synth_job(self, tts: TTSEngine | None, utterance: _Utterance, gen: int) -> None:
        if tts is None:
            self._post(lambda: self._synth_failed(utterance, gen, "voice switched off"))
            return
        try:
            clip = tts.synthesize(utterance.text)
        except TTSError as exc:
            message = exc.user_message
            self._post(lambda: self._synth_failed(utterance, gen, message))
            return
        except Exception as exc:  # engine bug: never let it take the app down
            log.exception("speech synthesis failed")
            message = str(exc)
            self._post(lambda: self._synth_failed(utterance, gen, message))
            return
        self._post(lambda: self._play(utterance, clip, gen))

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
        if utterance.on_start is not None:
            try:
                utterance.on_start(clip.duration)
            except Exception:
                log.exception("utterance start callback failed")

    def _played(self, utterance: _Utterance, gen: int, completed: bool) -> None:
        if gen != self._speak_gen or utterance is not self._current:
            return
        self._current = None
        utterance.finish(Outcome.PLAYED if completed else Outcome.INTERRUPTED)
        self._continue()

    # -- listening ----------------------------------------------------------------------------

    def toggle_listening(self, source: str = "hotkey") -> None:
        """Push-to-talk press in toggle mode / the chat microphone button."""
        if self._state == VoiceState.LISTENING:
            self.stop_listening()
        else:
            self.listen(source)

    def listen(self, source: str = "hotkey") -> bool:
        """Open the microphone for one utterance; interrupts her speech."""
        if not self.can_listen:
            self._error("Розпізнавання мовлення не налаштоване (Налаштування → Голос).")
            return False
        if not self._mic_allowed():
            self._error("Мікрофон вимкнено в «Приватності». Дозволь його, щоб говорити зі мною.")
            return False
        if self._state in (VoiceState.LISTENING, VoiceState.TRANSCRIBING):
            return self._state == VoiceState.LISTENING
        self.interrupt()
        self._close_input()
        self._listen_gen += 1
        gen = self._listen_gen
        with self._vad_lock:
            self._vad = EnergyVAD(self.listen_config, initial_floor=self._noise_floor)
        self._listen_started = self._clock()
        self._timeout_posted = False
        self._listen_source = source
        if not self._open_input(lambda chunk, g=gen: self._on_listen_audio(chunk, g)):
            self._rest("microphone failed")
            return False
        self._set_state(VoiceState.LISTENING, source)
        return True

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
        self._transcribe(audio)

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
                audio = event.audio
                self._post(lambda a=audio, g=gen: self._utterance_ready(a, g))
        if not ended and not talking and not self._timeout_posted \
                and now - self._listen_started > self.no_speech_timeout:
            self._timeout_posted = True
            self._post(lambda g=gen: self._no_speech(g))
        if self._on_level is not None and now - self._last_level_post >= 0.1:
            self._last_level_post = now
            on_level = self._on_level
            self._post(lambda lv=level: on_level(lv))

    def _utterance_ready(self, audio: Any, gen: int) -> None:
        if gen != self._listen_gen or self._state != VoiceState.LISTENING:
            return
        self._drop_vad()
        self._close_input()
        self._listen_gen += 1
        self._transcribe(audio)

    def _no_speech(self, gen: int) -> None:
        if gen != self._listen_gen or self._state != VoiceState.LISTENING:
            return
        self.cancel_listening()
        self._result("", "no_speech")

    def _transcribe(self, audio: Any) -> None:
        self._set_state(VoiceState.TRANSCRIBING, "")
        gen = self._listen_gen
        stt, prompt = self.stt, self.prompt
        self._worker.submit(lambda: self._stt_job(stt, audio, prompt, gen), VoiceWorker.TRANSCRIBE)

    def _stt_job(self, stt: STTEngine | None, audio: Any, prompt: str, gen: int) -> None:
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
        self._post(lambda: self._transcribed(gen, text, ""))

    def _transcribed(self, gen: int, text: str, error: str) -> None:
        if gen != self._listen_gen or self._state != VoiceState.TRANSCRIBING:
            return
        self._rest("transcribed")
        if error:
            self._error(error)
            self._result("", "error")
        elif text:
            self._result(text, "ok")
        else:
            self._result("", "not_understood")

    # -- wake phrase ---------------------------------------------------------------------------

    def _wake_wanted(self) -> bool:
        return (self._wake_enabled and not self._suspended and self.can_listen
                and self._mic_allowed())

    def _enter_standby(self) -> None:
        self._listen_gen += 1
        gen = self._listen_gen
        with self._vad_lock:
            self._vad = EnergyVAD(self.wake_config, initial_floor=self._noise_floor)
        self._wake_check_pending = False
        if self._open_input(lambda chunk, g=gen: self._on_standby_audio(chunk, g)):
            self._set_state(VoiceState.STANDBY, "waiting for the wake phrase")
        else:
            self._wake_enabled = False
            self._set_state(VoiceState.IDLE, "microphone failed")

    def _on_standby_audio(self, chunk: Any, gen: int) -> None:
        """Audio thread."""
        with self._vad_lock:
            vad = self._vad
            if vad is None or gen != self._listen_gen:
                return
            events = vad.feed(chunk)
        for event in events:
            if event.kind == "end" and event.audio is not None:
                audio = event.audio
                self._post(lambda a=audio, g=gen: self._wake_candidate(a, g))

    def _wake_candidate(self, audio: Any, gen: int) -> None:
        if gen != self._listen_gen or self._state != VoiceState.STANDBY or self._wake_check_pending:
            return
        self._wake_check_pending = True
        stt, prompt = self.stt, self.prompt

        def job() -> None:
            try:
                text = stt.transcribe(audio, prompt=prompt).text if stt is not None else ""
            except Exception:
                log.debug("wake phrase check failed", exc_info=True)
                text = ""
            self._post(lambda: self._wake_checked(gen, text))

        self._worker.submit(job, VoiceWorker.BACKGROUND)

    def _wake_checked(self, gen: int, text: str) -> None:
        self._wake_check_pending = False
        if gen != self._listen_gen or self._state != VoiceState.STANDBY:
            return
        command = match_wake(text, self.wake_names)
        if command is None:
            return
        if command:
            # "Мія, котра година?", "Мія, помовч" - the request came with the name.
            self._result(command, "ok")
            return
        if self._on_wake is not None:
            self._on_wake()
        self.listen("wake")

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
        """Back to IDLE, or STANDBY if the wake phrase is on."""
        if self.tts is None and self.stt is None:
            self._close_input()
            self._set_state(VoiceState.OFF, reason)
            return
        if self._current is not None:
            return  # still speaking
        if self._queue and self._next():
            return  # a line was waiting for the user to finish
        if self._wake_wanted():
            if self._state != VoiceState.STANDBY or self._input is None:
                self._close_input()
                self._enter_standby()
            return
        self._close_input()
        self._set_state(VoiceState.IDLE, reason)

    def _set_state(self, state: VoiceState, reason: str) -> None:
        if state == self._state:
            return
        self._state = state
        if self._on_state is not None:
            self._on_state(state, reason)

    def _result(self, text: str, status: str) -> None:
        if self._on_result is not None:
            self._on_result(text, status)

    def _error(self, message: str) -> None:
        log.info("voice: %s", message)
        if self._on_error is not None:
            self._on_error(message)
