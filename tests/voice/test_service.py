from typing import ClassVar

import numpy as np
import pytest

from chibi.voice.audio_io import AudioUnavailable
from chibi.voice.service import Outcome, VoiceService, VoiceState, VoiceWorker, match_wake
from chibi.voice.stt import STTError, Transcript
from chibi.voice.tts import AudioClip, TTSError

SR = 16_000


class ManualWorker:
    """Runs jobs when the test says so (keeps ordering visible)."""

    def __init__(self):
        self.jobs = []

    def submit(self, job, priority):
        self.jobs.append((priority, job))

    def run_all(self):
        while self.jobs:
            self.jobs.sort(key=lambda j: j[0])
            _, job = self.jobs.pop(0)
            job()


class FakeTTS:
    name = "fake"
    is_local = True

    def __init__(self, fail=False):
        self.said = []
        self.fail = fail

    def synthesize(self, text):
        if self.fail:
            raise TTSError("boom", user_message="Голос зламався.")
        self.said.append(text)
        return AudioClip(np.zeros(22050, dtype=np.int16) + 1, 22050)  # 1 s


class FakeSTT:
    def __init__(self, text="Привіт, Міє", error=None):
        self.text = text
        self.error = error
        self.calls = []

    def transcribe(self, audio, prompt=""):
        self.calls.append((len(audio), prompt))
        if self.error:
            raise self.error
        return Transcript(self.text, len(audio) / SR)


class FakeOutput:
    def __init__(self):
        self.playing = None
        self.stopped = 0
        self.volumes = []

    def play(self, clip, volume, on_done):
        self.playing = on_done
        self.volumes.append(volume)

    def finish(self, completed=True):
        done, self.playing = self.playing, None
        done(completed)

    def stop(self):
        self.stopped += 1
        if self.playing:
            self.finish(False)


class FakeInput:
    instances: ClassVar[list] = []

    def __init__(self, fail=False):
        self.callback = None
        self.fail = fail
        FakeInput.instances.append(self)

    def start(self, on_audio):
        if self.fail:
            raise AudioUnavailable("no mic", user_message="Немає мікрофона.")
        self.callback = on_audio

    def stop(self):
        self.callback = None

    @property
    def active(self):
        return self.callback is not None


def speech(seconds, amplitude=0.2):
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def silence(seconds):
    return (np.random.default_rng(0).standard_normal(int(SR * seconds)) * 0.001).astype(np.float32)


class Harness:
    def __init__(self, *, tts=None, stt=None, mic=True, input_fail=False):
        FakeInput.instances.clear()
        self.now = 0.0
        self.worker = ManualWorker()
        self.output = FakeOutput()
        self.states = []
        self.results = []
        self.errors = []
        self.wakes = 0
        self.mic = mic
        self.service = VoiceService(
            post=lambda fn: fn(),
            output=self.output,
            input_factory=lambda: FakeInput(fail=input_fail),
            worker=self.worker,
            clock=lambda: self.now,
            mic_allowed=lambda: self.mic,
            on_state=lambda s, r: self.states.append(s),
            on_result=lambda text, status: self.results.append((text, status)),
            on_error=self.errors.append,
            on_wake=self._wake,
        )
        self.tts = tts if tts is not None else FakeTTS()
        self.stt = stt if stt is not None else FakeSTT()
        self.service.configure(tts=self.tts, stt=self.stt)

    def _wake(self):
        self.wakes += 1

    @property
    def mic_input(self):
        return next((i for i in reversed(FakeInput.instances) if i.active), None)

    def feed(self, audio, chunk=480):
        for i in range(0, audio.size, chunk):
            stream = self.mic_input
            if stream is None:
                return
            self.now += chunk / SR
            stream.callback(audio[i:i + chunk])


def test_configure_moves_from_off_to_idle_and_back():
    h = Harness()
    assert h.service.state == VoiceState.IDLE
    h.service.configure(tts=None, stt=None)
    assert h.service.state == VoiceState.OFF
    assert not h.service.say("привіт")


def test_say_synthesises_plays_and_reports():
    h = Harness()
    started, done = [], []
    assert h.service.say("Привіт!", solicited=True, on_start=started.append, on_done=done.append)
    assert h.service.state == VoiceState.SPEAKING
    h.worker.run_all()
    assert h.tts.said == ["Привіт!"]
    assert started == [pytest.approx(1.0)]
    h.output.finish()
    assert done == [Outcome.PLAYED]
    assert h.service.state == VoiceState.IDLE


def test_queue_plays_lines_in_order():
    h = Harness()
    done = []
    h.service.say("раз", on_done=lambda o: done.append(("раз", o)))
    h.service.say("два", on_done=lambda o: done.append(("два", o)))
    h.worker.run_all()
    h.output.finish()
    h.worker.run_all()
    h.output.finish()
    assert h.tts.said == ["раз", "два"]
    assert done == [("раз", Outcome.PLAYED), ("два", Outcome.PLAYED)]


def test_full_queue_drops_unsolicited_lines_first():
    h = Harness()
    h.service.max_queue = 2
    outcomes = {}
    for text in ("a", "b", "c"):
        h.service.say(text, on_done=lambda o, t=text: outcomes.setdefault(t, o))
    # "a" is being synthesised; "b" and "c" are queued.
    assert h.service.say("важливо", solicited=True, on_done=lambda o: outcomes.setdefault("важливо", o))
    assert outcomes == {"b": Outcome.DROPPED}
    # Full again ("c", "важливо"): a fresher comment replaces the older one ...
    assert h.service.say("d", on_done=lambda o: outcomes.setdefault("d", o))
    assert outcomes == {"b": Outcome.DROPPED, "c": Outcome.DROPPED}
    # ... and with only answers queued, another comment is refused.
    h.service.say("ще", solicited=True)
    assert outcomes["d"] == Outcome.DROPPED
    assert not h.service.say("e")


def test_stale_unsolicited_lines_are_skipped():
    h = Harness()
    outcomes = []
    h.service.say("перша", solicited=True)
    h.service.say("застаріла", on_done=outcomes.append)
    h.worker.run_all()
    h.now += 60
    h.output.finish()
    assert outcomes == [Outcome.DROPPED]
    assert h.tts.said == ["перша"]
    assert h.service.state == VoiceState.IDLE


def test_synthesis_failure_reports_and_continues():
    h = Harness(tts=FakeTTS(fail=True))
    outcomes = []
    h.service.say("раз", on_done=outcomes.append)
    h.worker.run_all()
    assert outcomes == [Outcome.FAILED]
    assert h.errors == ["Голос зламався."]
    assert h.service.state == VoiceState.IDLE


def test_interrupt_stops_playback_and_clears_queue():
    h = Harness()
    outcomes = []
    h.service.say("раз", on_done=outcomes.append)
    h.service.say("два", on_done=outcomes.append)
    h.worker.run_all()
    h.service.interrupt()
    assert outcomes == [Outcome.INTERRUPTED, Outcome.INTERRUPTED]
    assert h.service.state == VoiceState.IDLE
    assert h.output.stopped >= 1


def test_late_synthesis_result_after_interrupt_is_ignored():
    h = Harness()
    h.service.say("раз")
    h.service.interrupt()
    h.worker.run_all()  # synthesis finishes after the interrupt
    assert h.output.playing is None
    assert h.service.state == VoiceState.IDLE


def test_listen_requires_microphone_permission():
    h = Harness(mic=False)
    assert not h.service.listen()
    assert "Мікрофон" in h.errors[-1]
    assert h.service.state == VoiceState.IDLE
    assert h.mic_input is None


def test_listen_without_stt_explains():
    h = Harness()
    h.service.configure(tts=h.tts, stt=None)
    assert not h.service.listen()
    assert "Розпізнавання" in h.errors[-1]


def test_push_to_talk_round_trip_with_end_of_utterance_detection():
    h = Harness()
    assert h.service.listen()
    assert h.service.state == VoiceState.LISTENING
    assert h.service.microphone_open
    h.feed(np.concatenate([silence(0.5), speech(1.0), silence(1.2)]))
    assert h.service.state == VoiceState.TRANSCRIBING
    assert not h.service.microphone_open  # closed as soon as the user stopped talking
    h.worker.run_all()
    assert h.results == [("Привіт, Міє", "ok")]
    assert h.stt.calls[0][1] == "Мія, привіт."
    assert h.service.state == VoiceState.IDLE


def test_toggle_twice_uses_speech_so_far():
    h = Harness()
    h.service.toggle_listening()
    h.feed(np.concatenate([silence(0.3), speech(1.0)]))
    h.service.toggle_listening()
    assert h.service.state == VoiceState.TRANSCRIBING
    h.worker.run_all()
    assert h.results == [("Привіт, Міє", "ok")]


def test_release_without_speech_reports_no_speech():
    h = Harness()
    h.service.listen()
    h.feed(silence(0.5))
    h.service.stop_listening()
    assert h.results == [("", "no_speech")]
    assert h.service.state == VoiceState.IDLE
    assert h.stt.calls == []


def test_no_speech_timeout_closes_the_microphone():
    h = Harness()
    h.service.no_speech_timeout = 2.0
    h.service.listen()
    h.feed(silence(3.0))
    assert h.results == [("", "no_speech")]
    assert not h.service.microphone_open
    assert h.service.state == VoiceState.IDLE


def test_not_understood_and_errors():
    h = Harness(stt=FakeSTT(text=""))
    h.service.listen()
    h.feed(np.concatenate([speech(1.0), silence(1.2)]))
    h.worker.run_all()
    assert h.results == [("", "not_understood")]

    h = Harness(stt=FakeSTT(error=STTError("x", user_message="Модель зламалась.")))
    h.service.listen()
    h.feed(np.concatenate([speech(1.0), silence(1.2)]))
    h.worker.run_all()
    assert h.results == [("", "error")]
    assert h.errors == ["Модель зламалась."]


def test_listening_interrupts_her_speech():
    h = Harness()
    outcomes = []
    h.service.say("довга репліка", on_done=outcomes.append)
    h.worker.run_all()
    assert h.service.state == VoiceState.SPEAKING
    h.service.listen()
    assert outcomes == [Outcome.INTERRUPTED]
    assert h.service.state == VoiceState.LISTENING


def test_unsolicited_lines_are_refused_while_the_user_talks_but_answers_wait():
    h = Harness()
    h.service.listen()
    assert not h.service.say("коментар")
    assert h.service.say("відповідь", solicited=True)
    assert h.tts.said == []
    h.feed(np.concatenate([speech(1.0), silence(1.2)]))
    h.worker.run_all()  # transcription, then the waiting answer is synthesised
    assert h.tts.said == ["відповідь"]
    assert h.service.state == VoiceState.SPEAKING


def test_answer_waiting_when_user_says_nothing_still_plays():
    h = Harness()
    h.service.listen()
    h.service.say("відповідь", solicited=True)
    h.service.stop_listening()  # nothing was said
    assert h.service.state == VoiceState.SPEAKING
    h.worker.run_all()
    assert h.tts.said == ["відповідь"]


def test_microphone_failure_is_reported():
    h = Harness(input_fail=True)
    assert not h.service.listen()
    assert h.errors == ["Немає мікрофона."]
    assert h.service.state == VoiceState.IDLE


def test_revoking_the_microphone_closes_it_immediately():
    h = Harness()
    h.service.listen()
    h.mic = False
    h.service.revoke_microphone()
    assert not h.service.microphone_open
    assert h.service.state == VoiceState.IDLE


def test_wake_phrase_standby_and_command_in_one_breath():
    h = Harness(stt=FakeSTT(text="Мія, котра година?"))
    h.service.set_wake(True)
    assert h.service.state == VoiceState.STANDBY
    assert h.service.microphone_open
    h.feed(np.concatenate([speech(1.2), silence(1.0)]))
    h.worker.run_all()
    assert h.results == [("котра година", "ok")]
    assert h.service.state == VoiceState.STANDBY


def test_wake_phrase_alone_starts_listening():
    h = Harness(stt=FakeSTT(text="Гей, Міє!"))
    h.service.set_wake(True)
    h.feed(np.concatenate([speech(0.8), silence(1.0)]))
    h.worker.run_all()
    assert h.wakes == 1
    assert h.service.state == VoiceState.LISTENING


def test_other_speech_in_standby_is_ignored():
    h = Harness(stt=FakeSTT(text="Я сьогодні йду в магазин"))
    h.service.set_wake(True)
    h.feed(np.concatenate([speech(1.2), silence(1.0)]))
    h.worker.run_all()
    assert h.results == []
    assert h.service.state == VoiceState.STANDBY


def test_standby_microphone_is_closed_while_she_speaks():
    h = Harness()
    h.service.set_wake(True)
    h.service.say("привіт")
    assert h.service.state == VoiceState.SPEAKING
    assert not h.service.microphone_open
    h.worker.run_all()
    h.output.finish()
    assert h.service.state == VoiceState.STANDBY
    assert h.service.microphone_open


def test_wake_needs_permission_and_can_be_suspended():
    h = Harness(mic=False)
    h.service.set_wake(True)
    assert h.service.state == VoiceState.IDLE
    h = Harness()
    h.service.set_wake(True)
    h.service.set_suspended(True)
    assert h.service.state == VoiceState.IDLE and not h.service.microphone_open
    h.service.set_suspended(False)
    assert h.service.state == VoiceState.STANDBY


def test_shutdown_closes_everything():
    h = Harness()
    h.service.set_wake(True)
    h.service.shutdown()
    assert h.service.state == VoiceState.OFF
    assert not h.service.microphone_open


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Мія, котра година?", "котра година"), ("Міє!", ""), ("Гей, Міє, помовч", "помовч"),
     ("мия ти тут", "ти тут"), ("Я сьогодні втомився", None), ("", None),
     ("Слухай, Мія, а ти знаєш...", "а ти знаєш"), ("Марія прийшла", None)],
)
def test_match_wake(text, expected):
    assert match_wake(text) == expected


def test_real_worker_runs_jobs_by_priority():
    import threading

    worker = VoiceWorker()
    order = []
    gate = threading.Event()
    done = threading.Event()
    worker.submit(lambda: gate.wait(2), VoiceWorker.BACKGROUND)
    worker.submit(lambda: order.append("synth"), VoiceWorker.SYNTHESIZE)
    worker.submit(lambda: order.append("stt"), VoiceWorker.TRANSCRIBE)
    worker.submit(lambda: done.set(), VoiceWorker.BACKGROUND)
    gate.set()
    assert done.wait(2)
    assert order == ["stt", "synth"]
    worker.shutdown()


def test_preload_loads_engines_in_the_background_and_reports_failures():
    class Loadable(FakeTTS):
        loaded = False

        def load(self):
            Loadable.loaded = True

    class Broken(FakeSTT):
        def load(self):
            raise STTError("x", user_message="Модель не завантажено.")

    h = Harness(tts=Loadable(), stt=Broken())
    h.service.preload()
    assert not Loadable.loaded  # nothing on the calling thread
    h.worker.run_all()
    assert Loadable.loaded
    assert h.errors == ["Модель не завантажено."]
