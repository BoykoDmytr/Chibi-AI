from typing import ClassVar

import numpy as np
import pytest

from chibi.voice.audio_io import AudioUnavailable
from chibi.voice.service import (
    MIA_FORMS,
    NameForms,
    Outcome,
    VoiceService,
    VoiceState,
    VoiceWorker,
    find_address,
    is_filler,
)
from chibi.voice.stt import STTError, Transcript
from chibi.voice.tts import AudioClip, TTSError
from chibi.voice.vad import VADConfig

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


def test_hold_to_talk_keeps_listening_through_a_pause_until_released():
    h = Harness()
    assert h.service.listen(hold=True)
    h.feed(np.concatenate([speech(1.0), silence(1.5), speech(1.0)]))  # a pause to think
    assert h.service.state == VoiceState.LISTENING and h.service.microphone_open
    h.feed(silence(8.0))  # silent, the key still held: no "nothing heard" yet
    assert h.service.state == VoiceState.LISTENING and h.results == []
    h.service.stop_listening()  # the key let go
    assert h.service.state == VoiceState.TRANSCRIBING
    h.worker.run_all()
    assert h.stt.calls[0][0] > 3.0 * SR  # both parts, the pause between them too
    assert h.results == [("Привіт, Міє", "ok")]


def test_hold_to_talk_is_still_bounded():
    h = Harness()
    h.service.listen_config = VADConfig(max_utterance_seconds=5.0)
    h.service.listen(hold=True)
    h.feed(silence(6.0))  # the release never arrived: the microphone closes anyway
    assert h.results == [("", "no_speech")] and not h.service.microphone_open
    h.service.listen(hold=True)
    h.feed(np.concatenate([speech(0.8), silence(0.3)] * 6))  # talking on past the length limit
    assert h.service.state == VoiceState.TRANSCRIBING and not h.service.microphone_open


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
    h.service.set_listen_mode("name")
    assert h.service.state == VoiceState.STANDBY
    assert h.service.microphone_open
    h.feed(np.concatenate([speech(1.2), silence(1.0)]))
    h.worker.run_all()
    assert h.results == [("Котра година?", "named")]
    assert h.service.state == VoiceState.STANDBY


def test_wake_phrase_alone_keeps_listening_on_the_open_microphone():
    h = Harness(stt=FakeSTT(text="Гей, Міє!"))
    h.service.set_listen_mode("name")
    h.feed(np.concatenate([speech(0.8), silence(1.0)]))
    h.worker.run_all()
    assert h.wakes == 1
    assert h.service.state == VoiceState.STANDBY and h.service.microphone_open


def test_other_speech_in_standby_is_ignored():
    h = Harness(stt=FakeSTT(text="Я сьогодні йду в магазин"))
    h.service.set_listen_mode("name")
    h.feed(np.concatenate([speech(1.2), silence(1.0)]))
    h.worker.run_all()
    assert h.results == []
    assert h.service.state == VoiceState.STANDBY


def test_standby_microphone_is_closed_while_she_speaks():
    h = Harness()
    h.service.set_listen_mode("name")
    h.service.say("привіт")
    assert h.service.state == VoiceState.SPEAKING
    assert not h.service.microphone_open
    h.worker.run_all()
    h.output.finish()
    assert h.service.state == VoiceState.STANDBY
    assert h.service.microphone_open


def test_wake_needs_permission_and_can_be_suspended():
    h = Harness(mic=False)
    h.service.set_listen_mode("name")
    assert h.service.state == VoiceState.IDLE
    h = Harness()
    h.service.set_listen_mode("name")
    h.service.set_suspended(True)
    assert h.service.state == VoiceState.IDLE and not h.service.microphone_open
    h.service.set_suspended(False)
    assert h.service.state == VoiceState.STANDBY


def test_shutdown_closes_everything():
    h = Harness()
    h.service.set_listen_mode("name")
    h.service.shutdown()
    assert h.service.state == VoiceState.OFF
    assert not h.service.microphone_open


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Мія, котра година?", "Котра година?"), ("Міє!", ""), ("Гей, Міє, помовч", "Помовч"),
     ("мия ти тут", "Ти тут"), ("Я сьогодні втомився", None), ("", None),
     ("Слухай, Мія, а ти знаєш...", "А ти знаєш..."), ("Марія прийшла", None),
     # anywhere in the phrase, as a separate word
     ("Як справи, Міє?", "Як справи?"), ("А ти, Міє, що думаєш?", "А ти що думаєш?"),
     ("Скажи мені, Міє, анекдот", "Скажи мені анекдот"), ("Привіт, Мія!", "Привіт!"),
     ("А ти, Міє? Що думаєш", "А ти? Що думаєш"), ("Добре. Міє, котра година?", "Добре. Котра година?"),
     ("що думаєш міє", "Що думаєш"),
     # talking about her, not to her
     ("Подивись, що Мія робить", None), ("Я бачила Мію вчора", None), ("Віддай Мії печиво", None),
     ("Міємо", None), ("Мія-подібна штука", None)],
)
def test_find_address(text, expected):
    assert find_address(text) == expected


def test_name_forms_follow_ukrainian_declension():
    olena = NameForms.for_name("Олена")
    assert "олено" in olena.strong and "олена" in olena.weak and "олену" in olena.lead
    assert find_address("Що скажеш, Олено?", olena) == "Що скажеш?"
    assert find_address("Я бачила Олену", olena) is None
    anya = NameForms.for_name("Аня")
    assert "аню" in anya.weak and not anya.strong  # "Аню" is also the accusative
    assert find_address("Ну що, Аню, граємо?", anya) == "Ну що граємо?"
    assert find_address("я кличу аню погратися", anya) is None
    assert "тарасе" in NameForms.for_name("Тарас").strong
    assert NameForms.for_name("") == NameForms()
    assert "міє" in MIA_FORMS.strong and "mia" in MIA_FORMS.weak


@pytest.mark.parametrize(("text", "filler"), [("Угу.", True), ("Ну ну", True), ("Мм...", True), ("", True),
                                              ("так", False), ("ні", False), ("ну давай", False)])
def test_fillers(text, filler):
    assert is_filler(text) is filler


# --- the open microphone: "name" mode, conversation, "open" mode -----------------------------------


def talk(seconds):
    """Speech with the short dips real speech has between words."""
    words = max(1, round(seconds / 0.6))
    return np.concatenate([np.concatenate([speech(0.5), silence(0.1)])] * words)


def utter(h, text, seconds=1.0):
    """The user says *text* (the fake recogniser returns it) and it is recognised."""
    h.stt.text = text
    h.feed(np.concatenate([talk(seconds), silence(1.0)]))
    h.worker.run_all()


def test_name_anywhere_and_the_request_without_it():
    h = Harness()
    h.service.set_listen_mode("name")
    utter(h, "А ти, Міє, що думаєш?")
    assert h.results == [("А ти що думаєш?", "named")]
    assert h.stt.calls[-1][1] == h.service.standby_prompt


def test_open_microphone_uses_the_users_phrase_length():
    h = Harness()
    h.service.wake_config = VADConfig(end_silence_ms=800, max_utterance_seconds=30.0, min_speech_ms=350)
    h.service.set_listen_mode("name")
    utter(h, "Міє, розкажи, будь ласка, довгу історію", seconds=12.0)  # one long breath, not cut
    assert h.results == [("Розкажи, будь ласка, довгу історію", "named")]
    assert h.stt.calls[-1][0] >= 12 * SR


def test_very_short_sounds_are_not_recognised():
    h = Harness()
    h.service.set_listen_mode("open")
    h.feed(np.concatenate([speech(0.2), silence(1.0)]))  # a clack, a cough
    h.worker.run_all()
    assert h.stt.calls == [] and h.results == []


def test_conversation_follows_without_her_name_while_she_answers_and_after():
    h = Harness()
    h.service.follow_up_seconds = 20
    h.service.set_listen_mode("name")
    utter(h, "Міє, котра година?")
    assert h.service.conversation_open
    h.now += 25  # the model thinks for a while: the window waits for her answer
    assert h.service.conversation_open
    h.service.say("Пів на третю.", solicited=True)
    h.worker.run_all()
    h.now += 40  # a long answer: the window stays open while she speaks
    assert h.service.conversation_open
    h.output.finish()
    assert h.service.state == VoiceState.STANDBY
    h.now += 15
    utter(h, "А завтра о котрій вставати?")  # no name: still the conversation
    assert h.results[-1] == ("А завтра о котрій вставати?", "heard")
    h.service.say("О сьомій.", solicited=True)
    h.worker.run_all()
    h.output.finish()
    h.now += 21  # follow_up_seconds after her last answer: over
    assert not h.service.conversation_open
    utter(h, "Це я не тобі")
    assert len(h.results) == 2


def test_conversation_window_closes_when_no_answer_comes():
    h = Harness()
    h.service.follow_up_seconds = 10
    h.service.set_listen_mode("name")
    utter(h, "Міє, привіт")
    h.now += h.service.reply_wait_seconds + 1
    assert not h.service.conversation_open
    utter(h, "ну то що там")
    assert h.results == [("Привіт", "named")]


def test_an_answer_shown_without_voice_keeps_the_conversation_going():
    h = Harness()
    h.service.follow_up_seconds = 10
    h.service.set_listen_mode("name")
    utter(h, "Міє, привіт")
    h.now += 5
    h.service.note_reply_shown(4.0)  # read for 4 s, then 10 s to reply
    h.now += 13
    assert h.service.conversation_open
    h.now += 2
    assert not h.service.conversation_open


def test_follow_up_zero_means_only_by_name():
    h = Harness()
    h.service.follow_up_seconds = 0
    h.service.set_listen_mode("name")
    utter(h, "Міє, привіт")
    assert not h.service.conversation_open
    utter(h, "а як справи")
    assert h.results == [("Привіт", "named")]


def test_push_to_talk_opens_the_conversation_too():
    h = Harness()
    h.service.set_listen_mode("name")
    h.service.listen()
    assert h.service.state == VoiceState.LISTENING
    h.feed(np.concatenate([speech(1.0), silence(1.2)]))
    h.worker.run_all()
    assert h.results == [("Привіт, Міє", "ok")]
    assert h.service.state == VoiceState.STANDBY and h.service.conversation_open


def test_open_mode_hears_everything_but_fillers_and_phantoms():
    h = Harness()
    h.service.set_listen_mode("open")
    utter(h, "Яка сьогодні погода?")
    utter(h, "Угу.")
    utter(h, "Дякую за перегляд!")  # Whisper's subtitle credits
    utter(h, "Розмова українською.")  # its own prompt echoed back in noise
    utter(h, "[музика]")
    assert h.results == [("Яка сьогодні погода?", "heard")]
    utter(h, "Міє!")  # her name alone: she listens for what comes next
    assert h.wakes == 1 and h.service.state == VoiceState.STANDBY
    utter(h, "котра година?")
    assert h.results[-1] == ("Котра година?", "named")


def test_fillers_in_the_conversation_are_ignored():
    h = Harness()
    h.service.set_listen_mode("name")
    utter(h, "Міє, привіт")
    utter(h, "Ммм...")
    utter(h, "Міє, угу")  # her name and a filler: only called
    assert h.results == [("Привіт", "named")]
    assert h.wakes == 1


def test_she_does_not_hear_the_tail_of_her_own_voice():
    h = Harness()
    h.service.echo_tail_seconds = 1.0
    h.service.set_listen_mode("open")
    h.service.say("Привіт!", solicited=True)
    h.worker.run_all()
    h.output.finish()
    assert h.service.microphone_open
    h.feed(np.concatenate([speech(0.9), silence(1.0)]))  # within the tail: ignored
    h.worker.run_all()
    assert h.stt.calls == []
    utter(h, "А тепер я")
    assert h.results == [("А тепер я", "heard")]


def test_one_recognition_at_a_time_and_the_rest_is_dropped():
    h = Harness()
    h.service.set_listen_mode("open")
    h.stt.text = "раз"
    for _ in range(3):
        h.feed(np.concatenate([speech(1.0), silence(1.0)]))
    assert len(h.worker.jobs) == 1  # never piles up: one recognised, one kept
    assert h.service.dropped_utterances == 1
    h.worker.run_all()
    assert h.results == [("раз", "heard")]
    assert len(h.stt.calls) == 1 and h.service.dropped_utterances == 2  # kept only after her name
    utter(h, "два")  # free again
    assert h.results[-1] == ("два", "heard")


def test_a_request_heard_before_she_started_talking_still_arrives():
    h = Harness()
    h.service.set_listen_mode("name")
    h.stt.text = "Міє, котра година?"
    h.feed(np.concatenate([speech(1.0), silence(1.0)]))  # recognition pending ...
    h.service.say("коментар", solicited=True)  # ... and she starts a line meanwhile
    assert h.service.state == VoiceState.SPEAKING
    h.worker.run_all()
    assert h.results == [("Котра година?", "named")]


def test_her_comments_wait_while_someone_is_talking():
    h = Harness()
    h.service.set_listen_mode("name")
    h.feed(speech(0.6))  # the user is mid-sentence
    assert not h.service.say("коментар")  # shown as text instead: never talk over the user
    assert h.service.say("відповідь", solicited=True)


def test_pause_and_revoke_drop_what_was_being_recognised():
    for stop in ("suspend", "revoke", "mode", "push_to_talk"):
        h = Harness()
        h.service.set_listen_mode("name")
        h.stt.text = "Міє, котра година?"
        h.feed(np.concatenate([speech(1.0), silence(1.0)]))
        if stop == "suspend":
            h.service.set_suspended(True)
        elif stop == "revoke":
            h.mic = False
            h.service.revoke_microphone()
        elif stop == "mode":
            h.service.set_listen_mode("open")
        else:
            h.service.listen()
        h.worker.jobs = [job for job in h.worker.jobs if job[0] == VoiceWorker.TRANSCRIBE][:1]
        h.worker.run_all()
        assert h.results == [], stop


def test_revoking_ends_the_conversation():
    h = Harness()
    h.service.set_listen_mode("name")
    utter(h, "Міє, привіт")
    h.mic = False
    h.service.revoke_microphone()
    assert not h.service.conversation_open and not h.service.microphone_open
    assert h.service.state == VoiceState.IDLE


def test_switching_modes():
    h = Harness()
    h.service.set_listen_mode("name")
    h.service.set_listen_mode("open")
    assert h.service.state == VoiceState.STANDBY and h.service.listen_mode == "open"
    h.service.set_listen_mode("push")
    assert h.service.state == VoiceState.IDLE and not h.service.microphone_open
    h.service.set_listen_mode("always")  # unknown: push
    assert h.service.listen_mode == "push"


def test_open_microphone_failure_falls_back_to_push_to_talk():
    h = Harness(input_fail=True)
    h.service.set_listen_mode("open")
    assert h.errors == ["Немає мікрофона."]
    assert h.service.listen_mode == "push" and h.service.state == VoiceState.IDLE
    h.service.say("привіт")
    h.worker.run_all()
    h.output.finish()  # not retried (and reported) after every line
    assert h.errors == ["Немає мікрофона."]


# --- streamed answers: the reply hold, prefetching, cancelled lines ------------------------------------


def test_microphone_stays_closed_between_the_sentences_of_an_answer():
    h = Harness()
    h.service.set_listen_mode("name")
    h.service.hold_for_reply(8.0)
    h.service.say("Перше речення.", solicited=True, prefetch=True)
    h.worker.run_all()
    h.output.finish()
    assert h.service.state == VoiceState.SPEAKING and not h.service.microphone_open
    h.service.say("Друге речення.", solicited=True, prefetch=True)
    h.worker.run_all()
    assert h.tts.said == ["Перше речення.", "Друге речення."]
    h.output.finish()
    h.service.release_reply_hold()
    assert h.service.state == VoiceState.STANDBY and h.service.microphone_open


def test_a_stalled_answer_does_not_keep_the_microphone_closed():
    h = Harness()
    h.service.set_listen_mode("name")
    h.service.hold_for_reply(5.0)
    h.service.say("Зараз скажу.", solicited=True)
    h.worker.run_all()
    h.output.finish()
    h.service.check_hold()
    assert h.service.state == VoiceState.SPEAKING  # still within the hold
    h.now += 6
    h.service.check_hold()
    assert h.service.state == VoiceState.STANDBY


def test_interrupting_while_she_waits_for_the_rest_of_an_answer():
    h = Harness()
    h.service.set_listen_mode("name")
    h.service.hold_for_reply(8.0)
    h.service.say("Перше.", solicited=True)
    h.worker.run_all()
    h.output.finish()
    h.service.interrupt()
    assert h.service.state == VoiceState.STANDBY
    h.service.hold_for_reply(8.0)
    h.service.say("Ще одне.", solicited=True)
    h.worker.run_all()
    h.output.finish()
    assert h.service.listen()  # push-to-talk goes straight to listening
    assert h.service.state == VoiceState.LISTENING


def test_next_sentence_is_synthesised_while_the_previous_one_plays():
    h = Harness()
    h.service.say("Перше речення.", solicited=True, prefetch=True)
    h.service.say("Друге речення.", solicited=True, prefetch=True)
    h.worker.run_all()
    assert h.tts.said == ["Перше речення.", "Друге речення."]  # both, while the first plays
    h.output.finish()
    assert h.output.playing is not None and h.worker.jobs == []  # no gap: already synthesised
    h.output.finish()
    assert h.service.state == VoiceState.IDLE


def test_unprefetched_lines_wait_for_their_turn():
    h = Harness()
    h.service.say("коментар")
    h.service.say("відповідь", solicited=True)
    h.worker.run_all()
    assert h.tts.said == ["коментар"]


def test_a_prefetched_line_that_failed_reports_when_its_turn_comes():
    class Picky(FakeTTS):
        def synthesize(self, text):
            if text == "Друге.":
                raise TTSError("boom", user_message="Голос зламався.")
            return super().synthesize(text)

    h = Harness(tts=Picky())
    outcomes = []
    h.service.say("Перше.", solicited=True)
    h.service.say("Друге.", solicited=True, prefetch=True, on_done=outcomes.append)
    h.worker.run_all()
    assert outcomes == [] and h.errors == []  # failed ahead of its turn: nothing to report yet
    h.output.finish()
    assert outcomes == [Outcome.FAILED] and h.errors == ["Голос зламався."]
    assert h.service.state == VoiceState.IDLE


def test_a_prefetched_clip_is_dropped_when_she_is_interrupted():
    h = Harness()
    outcomes = []
    h.service.say("Перше.", solicited=True)
    h.service.say("Друге.", solicited=True, prefetch=True, on_done=outcomes.append)
    h.worker.run_all()
    h.service.interrupt()
    assert outcomes == [Outcome.INTERRUPTED] and h.output.playing is None
    assert h.service.state == VoiceState.IDLE


def test_a_cancelled_line_is_skipped_before_it_is_synthesised():
    h = Harness()
    cancelled = {"value": False}
    outcomes = []
    h.service.say("перша", solicited=True)
    h.service.say("скасована", is_cancelled=lambda: cancelled["value"], on_done=outcomes.append)
    h.worker.run_all()
    cancelled["value"] = True
    h.output.finish()
    h.worker.run_all()
    assert outcomes == [Outcome.DROPPED]
    assert h.tts.said == ["перша"]
    assert h.service.state == VoiceState.IDLE


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


def test_revoke_during_transcription_drops_the_transcript():
    h = Harness()
    h.service.listen()
    h.feed(np.concatenate([speech(1.0), silence(1.2)]))
    assert h.service.state == VoiceState.TRANSCRIBING
    h.mic = False
    h.service.revoke_microphone()
    h.worker.run_all()
    assert h.results == []
    assert h.service.state == VoiceState.IDLE


def test_reentrant_say_from_on_done_is_not_lost():
    h = Harness()
    said = []

    def chain(outcome):
        if outcome == Outcome.PLAYED and not said:
            said.append(h.service.say("друга", solicited=True))

    h.service.say("перша", on_done=chain)
    h.worker.run_all()
    h.output.finish()
    h.worker.run_all()
    assert said == [True]
    assert h.tts.said == ["перша", "друга"]


# --- review fixes: name matching, pauses, her own lines mid-answer, replies to her remarks --------------


@pytest.mark.parametrize(
    ("name", "text"),
    [("Віка", "Вікно зависло, мама сказала перезавантажити"), ("Ніка", "Ніби все працює"),
     ("Маша", "Мама дзвонила вчора"), ("Маша", "Мама, дай води"), ("Саша", "Сама не знаю, що робити"),
     ("Ліна", "Ліва кнопка не працює"), ("Кіра", "Кіно було класне"), ("Ліза", "Ліси горять"),
     ("Кіра", "Кіно!")],
)
def test_everyday_words_are_not_her_name(name, text):
    # A one-letter slip in a short name is too often another word: no fuzzy match.
    assert find_address(text, NameForms.for_name(name) | MIA_FORMS) is None


def test_her_exact_name_and_a_set_off_slip_of_a_long_name_still_count():
    vika = NameForms.for_name("Віка") | MIA_FORMS
    assert find_address("Віко, котра година?", vika) == "Котра година?"
    assert find_address("Віка, привіт", vika) == "Привіт"
    olena = NameForms.for_name("Олена") | MIA_FORMS
    assert find_address("Алено, котра година?", olena) == "Котра година?"  # set off: a slip
    assert find_address("Алено!", olena) == ""
    assert find_address("Алено котра година", olena) is None  # not set off: another word


def test_delivered_requests_say_how_they_reached_her():
    h = Harness()
    h.service.set_listen_mode("name")
    utter(h, "Міє, котра година?")
    utter(h, "А завтра?")  # the conversation window: no name
    assert h.results == [("Котра година?", "named"), ("А завтра?", "heard")]
    h.service.set_listen_mode("open")
    utter(h, "Яка погода?")
    assert h.results[-1] == ("Яка погода?", "heard")
    h.stt.text = "Привіт, Міє"
    h.service.listen()
    h.feed(np.concatenate([speech(1.0), silence(1.2)]))
    h.worker.run_all()
    assert h.results[-1] == ("Привіт, Міє", "ok")  # push-to-talk


def test_her_name_alone_keeps_the_open_microphone_and_takes_the_next_phrase():
    h = Harness()
    h.service.follow_up_seconds = 0  # no conversation window: only the call itself counts
    h.service.set_listen_mode("name")
    utter(h, "Міє!")
    assert h.wakes == 1
    assert h.service.state == VoiceState.STANDBY and h.service.microphone_open
    assert not h.service.say("коментар")  # she waits for the request, does not talk over it
    utter(h, "котра година?")
    assert h.results == [("Котра година?", "named")]
    utter(h, "а це вже не тобі")  # only the next phrase
    assert len(h.results) == 1


def test_after_her_name_alone_the_next_phrase_must_come_soon():
    h = Harness()
    h.service.follow_up_seconds = 0
    h.service.set_listen_mode("name")
    utter(h, "Міє!")
    h.now += 30
    utter(h, "котра година?")
    assert h.results == []


class SequenceSTT(FakeSTT):
    """Returns the given texts one after another."""

    def __init__(self, texts):
        super().__init__()
        self.texts = list(texts)

    def transcribe(self, audio, prompt=""):
        self.calls.append((len(audio), prompt))
        return Transcript(self.texts.pop(0) if self.texts else "", len(audio) / SR)


def test_a_pause_after_her_name_does_not_lose_the_request():
    h = Harness(stt=SequenceSTT(["Міє,", "котра година?"]))
    h.service.set_listen_mode("name")
    h.feed(np.concatenate([talk(0.6), silence(1.0)]))  # "Міє," - recognised ...
    h.feed(np.concatenate([talk(1.2), silence(1.0)]))  # ... while the user goes on
    h.worker.run_all()
    assert h.results == [("Котра година?", "named")]
    assert h.service.dropped_utterances == 0
    assert h.service.state == VoiceState.STANDBY


def test_the_phrase_kept_behind_a_recognition_is_dropped_unless_she_was_called():
    h = Harness(stt=SequenceSTT(["Я сьогодні йду в магазин", "Міє, котра година?"]))
    h.service.set_listen_mode("name")
    h.feed(np.concatenate([talk(1.2), silence(1.0)]))
    h.feed(np.concatenate([talk(1.2), silence(1.0)]))
    h.worker.run_all()
    assert h.results == [] and len(h.stt.calls) == 1  # the TV keeps no queue of work
    assert h.service.dropped_utterances == 1


def test_pausing_after_her_name_drops_what_comes_next():
    h = Harness()
    h.service.set_listen_mode("name")
    utter(h, "Міє!")
    h.service.set_suspended(True)
    assert not h.service.microphone_open and not h.service.conversation_open
    h.service.set_suspended(False)
    utter(h, "котра година?")
    assert h.results == []


def test_pausing_ends_the_conversation_window():
    h = Harness()
    h.service.set_listen_mode("name")
    utter(h, "Міє, привіт")
    assert h.service.conversation_open
    h.service.set_suspended(True)
    assert not h.service.conversation_open
    h.service.set_suspended(False)
    utter(h, "а як справи")
    assert h.results == [("Привіт", "named")]


def test_push_to_talk_keeps_listening_through_a_pause():
    h = Harness()
    h.service.set_listen_mode("name")
    assert h.service.listen()
    h.service.set_suspended(True)
    assert h.service.state == VoiceState.LISTENING and h.service.microphone_open
    h.feed(np.concatenate([speech(1.0), silence(1.2)]))
    h.worker.run_all()
    assert h.results == [("Привіт, Міє", "ok")]


def test_her_own_lines_never_come_between_the_sentences_of_an_answer():
    h = Harness()
    h.service.hold_for_reply(8.0)
    h.service.say("Перше довге речення відповіді.", solicited=True, prefetch=True)
    h.worker.run_all()
    assert not h.service.say("Ого, ти щойно переміг боса!")  # while sentence 1 plays
    h.output.finish()  # sentence 1 ends; sentence 2 is still being written
    assert not h.service.say("Ого, ти щойно переміг боса!")
    assert h.service.say("Друге речення відповіді тут.", solicited=True, prefetch=True)
    h.worker.run_all()
    h.output.finish()
    h.service.release_reply_hold()
    assert h.tts.said == ["Перше довге речення відповіді.", "Друге речення відповіді тут."]
    assert h.service.say("Ого!")  # the answer is over


def test_her_question_opens_the_conversation():
    h = Harness()
    h.service.follow_up_seconds = 20
    h.service.set_listen_mode("name")
    assert h.service.say("Бачу помилку. Глянемо разом?", opens_conversation=True)
    h.worker.run_all()
    h.output.finish()
    assert h.service.conversation_open
    utter(h, "Так, давай")
    assert h.results == [("Так, давай", "heard")]


def test_a_remark_that_asks_nothing_does_not_open_the_conversation():
    h = Harness()
    h.service.set_listen_mode("name")
    h.service.say("Ого, красиво!")
    h.worker.run_all()
    h.output.finish()
    utter(h, "так, гарно")
    assert h.results == []


def test_a_question_that_was_never_heard_does_not_open_the_conversation():
    h = Harness(tts=FakeTTS(fail=True))
    h.service.set_listen_mode("name")
    h.service.say("Глянемо разом?", opens_conversation=True)
    h.worker.run_all()
    assert not h.service.conversation_open
