"""VoiceIntegration: engines from settings/permissions/libraries, speech routing,
push-to-talk, privacy (revoke, kill switch), models.  Engines, audio, the
hotkey and the downloader are fakes; the VoiceService in between is real."""

from __future__ import annotations

import logging
import threading
import time
from typing import ClassVar

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication

from chibi.ai.secrets import MemorySecretStore
from chibi.core.config import Settings
from chibi.core.events import (
    ActivityCategory,
    AppInfo,
    AppShuttingDown,
    CharacterInteraction,
    DoNotDisturbChanged,
    Event,
    ForegroundAppChanged,
    FullscreenChanged,
    InteractionKind,
    Permission,
    QuietModeChanged,
    SettingsChanged,
    SpeechPlayback,
    SpeechRequested,
    VoiceStateChanged,
    VoiceTranscript,
)
from chibi.memory.repositories import PermissionRepository
from chibi.privacy import permissions as permissions_module
from chibi.privacy.permissions import PermissionManager
from chibi.voice import integration as integration_module
from chibi.voice import models
from chibi.voice.audio_io import AudioUnavailable
from chibi.voice.hotkey_win import HotkeyError
from chibi.voice.integration import (
    AZURE_KEY_NAME,
    AZURE_NEEDS_CLOUD,
    AZURE_NO_KEY,
    HOTKEY_WINDOWS_ONLY,
    LATENCY_TRACE_SECONDS,
    LISTENING_NAME,
    LISTENING_OPEN,
    MIC_NOT_ALLOWED,
    PAUSED_BY_USER,
    PAUSED_CALL,
    PAUSED_DND,
    PAUSED_FULLSCREEN,
    PAUSED_PRIVATE,
    PIPER_MODEL_MISSING,
    REPLY_HOLD_SECONDS,
    STT_MODEL_MISSING,
    STT_OFF,
    TTS_OFF,
    VOICE_MODULE_MISSING,
    SpeechStart,
    VoiceIntegration,
    VoiceLatency,
    VoiceStatus,
    is_call_app,
    recommended_threads,
)
from chibi.voice.models import CATALOG, DEFAULT_TTS_MODEL, DownloadCancelled, DownloadError
from chibi.voice.stt import STTError, Transcript
from chibi.voice.tts import AudioClip, TTSError

SR = 16_000
ALL_LIBS = frozenset({"numpy", "onnxruntime", "faster_whisper", "sounddevice"})
ALL_MODELS = frozenset({DEFAULT_TTS_MODEL, "whisper-small", "whisper-base"})
SECRET_TEXT = "мій секретний план на вечір"


# --- fakes ---------------------------------------------------------------------------------------


class ManualWorker:
    """Runs jobs when the test says so (keeps ordering visible)."""

    def __init__(self):
        self.jobs = []
        self.shut = False

    def submit(self, job, priority=1):
        self.jobs.append((priority, job))

    def run_all(self):
        while self.jobs:
            self.jobs.sort(key=lambda j: j[0])
            _, job = self.jobs.pop(0)
            job()

    def shutdown(self, timeout=None):
        self.shut = True


class FakeTTS:
    is_local = True

    def __init__(self, name="piper", speaker="", speed=1.0, get_key=None):
        self.name = name
        self.speaker = speaker
        self.speed = speed
        self.get_key = get_key
        self.said = []
        self.fail = False
        self.loads = 0

    def load(self):
        self.loads += 1

    def synthesize(self, text):
        if self.fail:
            raise TTSError("boom", user_message="Голос зламався.")
        self.said.append(text)
        if text == "🙂":
            return AudioClip(np.zeros(0, dtype=np.int16), 22050)  # nothing to say aloud
        return AudioClip(np.ones(22050, dtype=np.int16), 22050)  # 1 s


class FakeSTT:
    def __init__(self, model="whisper-small"):
        self.model = model
        self.text = SECRET_TEXT
        self.error = None
        self.loads = 0

    def load(self):
        self.loads += 1

    def transcribe(self, audio, prompt=""):
        if self.error:
            raise self.error
        return Transcript(self.text, len(audio) / SR)


class FakeEngines:
    def __init__(self):
        self.built = []

    def piper(self, voice):
        self.built.append(("piper", voice.piper_speaker, voice.speed))
        return FakeTTS("piper", voice.piper_speaker, voice.speed)

    def azure(self, voice, get_key):
        self.built.append(("azure", voice.azure_voice, voice.speed))
        return FakeTTS("azure", voice.azure_voice, voice.speed, get_key)

    def whisper(self, voice):
        self.built.append(("whisper", voice.stt_model))
        return FakeSTT(voice.stt_model)


class FakeOutput:
    def __init__(self):
        self.device = None
        self.playing = None
        self.volumes = []

    def play(self, clip, volume, on_done):
        self.playing = on_done
        self.volumes.append(volume)

    def finish(self, completed=True):
        done, self.playing = self.playing, None
        done(completed)

    def stop(self):
        if self.playing:
            self.finish(False)


class FakeInput:
    instances: ClassVar[list] = []

    def __init__(self, device, fail=False):
        self.device = device
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


class FakeHotkey:
    def __init__(self, rig, text, on_press, on_release=None):
        self.text = text
        self.on_press = on_press
        self.on_release = on_release
        self.running = False
        self.stops = 0
        self._rig = rig

    def start(self):
        if self._rig.hotkey_error is not None:
            raise self._rig.hotkey_error
        self.running = True

    def stop(self):
        self.running = False
        self.stops += 1

    def press(self):  # what the hotkey thread does
        self.on_press()

    def release(self):
        self.on_release()


class FakeComposer:
    """Like SpeechComposer + PhraseSelector: publishes a line whose source is the key."""

    def __init__(self, bus):
        self.bus = bus
        self.calls = []

    def compose(self, *keys, solicited=False, **variables):
        self.calls.append((keys, solicited))
        self.bus.publish(SpeechRequested(text=f"<{keys[0]}>", solicited=solicited, source=keys[0]))
        return True


def speech(seconds, amplitude=0.2):
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def silence(seconds):
    return (np.random.default_rng(0).standard_normal(int(SR * seconds)) * 0.001).astype(np.float32)


class Rig:
    def __init__(self, bus, permissions, clock, *, libs=ALL_LIBS, installed=ALL_MODELS,
                 hotkey_supported=True, start=True):
        FakeInput.instances.clear()
        self.bus = bus
        self.permissions = permissions
        self.clock = clock
        self.settings = Settings()
        self.settings.voice.tts_enabled = True
        self.settings.voice.stt_enabled = True
        self.libs = set(libs)
        self.installed = set(installed)
        self.secrets = MemorySecretStore()
        self.composer = FakeComposer(bus)
        self.engines = FakeEngines()
        self.output = FakeOutput()
        self.worker = ManualWorker()
        self.downloads = ManualWorker()
        self.hotkeys: list[FakeHotkey] = []
        self.hotkey_error = None
        self.hotkey_supported = hotkey_supported
        self.notes = []
        self.visible = True
        self.devices = (["USB Mic"], ["Speakers", "Headset"])
        self.download_script = None
        self.events: list[Event] = []
        bus.subscribe(Event, self.events.append)
        self.voice = VoiceIntegration(
            bus, lambda: self.settings, permissions, self.secrets, self.composer,
            notify=self.notes.append, is_visible=lambda: self.visible, clock=clock,
            engines=self.engines,
            installed=lambda spec: spec.id in self.installed,
            libraries=lambda name: name in self.libs,
            output_factory=self._output,
            input_factory=lambda device: FakeInput(device),
            list_devices=lambda: self.devices,
            hotkey_factory=self._hotkey,
            hotkey_supported=lambda: self.hotkey_supported,
            voice_worker=self.worker,
            download_worker=self.downloads,
            downloader=self._download,
            post=lambda fn: fn(),
        )
        if start:
            self.voice.start()

    def _output(self, device):
        self.output.device = device
        return self.output

    def _hotkey(self, text, on_press, on_release=None):
        hotkey = FakeHotkey(self, text, on_press, on_release)
        self.hotkeys.append(hotkey)
        return hotkey

    def _download(self, spec, root, *, on_progress, cancel):
        steps = self.download_script or [spec.size // 2, spec.size]
        for done in steps:
            if cancel.is_set():
                raise DownloadCancelled()
            on_progress(done, spec.size)
        if cancel.is_set():
            raise DownloadCancelled()
        self.installed.add(spec.id)
        return root

    # -- helpers
    @property
    def service(self):
        return self.voice.service

    @property
    def hotkey(self):
        return next((h for h in reversed(self.hotkeys) if h.running), None)

    @property
    def mic(self):
        return next((i for i in reversed(FakeInput.instances) if i.active), None)

    def dispatch(self):
        self.bus.dispatch_pending()

    def grant(self, *perms):
        for p in perms:
            self.permissions.grant(p)
        self.dispatch()

    def revoke(self, permission):
        self.permissions.revoke(permission)
        self.dispatch()

    def change(self, section="voice", **voice):
        for name, value in voice.items():
            setattr(self.settings.voice, name, value)
        self.bus.publish(SettingsChanged(section=section))
        self.dispatch()

    def feed(self, audio, chunk=480):
        for i in range(0, audio.size, chunk):
            stream = self.mic
            if stream is None:
                return
            self.clock.advance(chunk / SR)
            stream.callback(audio[i:i + chunk])

    def published(self, kind):
        self.dispatch()
        return [e for e in self.events if isinstance(e, kind)]


@pytest.fixture
def permissions(db, bus, clock, monkeypatch):
    monkeypatch.setattr(permissions_module, "AVAILABLE", permissions_module.AVAILABLE | {
        Permission.MICROPHONE, Permission.SCREEN_CAPTURE, Permission.CLOUD_VISION})
    return PermissionManager(PermissionRepository(db), bus, clock)


@pytest.fixture
def rig(bus, permissions, clock):
    return Rig(bus, permissions, clock)


@pytest.fixture
def heard(rig):
    """A rig that can listen: microphone granted, recognition ready."""
    rig.grant(Permission.MICROPHONE)
    assert rig.voice.status().stt_ready
    return rig


class Shows:
    def __init__(self):
        self.calls = []

    def __call__(self, event, seconds):
        self.calls.append((event.text, seconds))


def line(text="Привіт!", solicited=False, source="test"):
    return SpeechRequested(text=text, solicited=solicited, source=source)


# --- engines from settings, permissions and libraries ----------------------------------------


def test_starts_with_a_local_voice_and_no_recognition_without_the_microphone(rig):
    status = rig.voice.status()
    assert status.tts_ready and not status.stt_ready
    assert status.stt_reason == MIC_NOT_ALLOWED
    assert rig.engines.built == [("piper", "tetiana", 1.0)]  # recognition isn't even loaded
    assert rig.service.state == "idle"
    assert rig.hotkeys == []  # no hotkey without the microphone
    assert [priority for priority, _job in rig.worker.jobs] == [2]  # preloaded in the background
    rig.worker.run_all()
    assert rig.service.tts.loads == 1


def test_nothing_happens_before_start(bus, permissions, clock):
    rig = Rig(bus, permissions, clock, start=False)
    assert rig.service is None and rig.engines.built == []
    shows = Shows()
    rig.voice.route(line(), shows)
    assert shows.calls == [("Привіт!", None)]


def test_missing_voice_module_gives_a_clear_reason_and_text_only(bus, permissions, clock):
    rig = Rig(bus, permissions, clock, libs=())
    rig.grant(Permission.MICROPHONE)
    status = rig.voice.status()
    assert not status.tts_ready and not status.stt_ready
    assert status.tts_reason == status.stt_reason == VOICE_MODULE_MISSING
    assert "pip install .[voice]" in VOICE_MODULE_MISSING
    assert rig.service is None and status.state == "off"
    shows = Shows()
    rig.voice.route(line(), shows)
    assert shows.calls == [("Привіт!", None)]
    assert rig.voice.toggle_listening() == VOICE_MODULE_MISSING
    assert rig.voice.devices() == ([], [])


def test_each_engine_needs_its_own_library(bus, permissions, clock):
    rig = Rig(bus, permissions, clock, libs={"numpy", "sounddevice"})  # no onnxruntime, no whisper
    rig.grant(Permission.MICROPHONE)
    status = rig.voice.status()
    assert status.tts_reason == VOICE_MODULE_MISSING
    assert status.stt_reason == VOICE_MODULE_MISSING
    # Azure needs neither: with the cloud permission and a key it works.
    rig.secrets.set(AZURE_KEY_NAME, "k" * 32)
    rig.grant(Permission.CLOUD_AI)
    rig.change(tts_engine="azure")
    assert rig.voice.status().tts_ready


def test_models_not_downloaded(bus, permissions, clock):
    rig = Rig(bus, permissions, clock, installed=())
    rig.grant(Permission.MICROPHONE)
    status = rig.voice.status()
    assert status.tts_reason == PIPER_MODEL_MISSING
    assert status.stt_reason == STT_MODEL_MISSING
    assert rig.engines.built == []


def test_switched_off_in_settings(rig):
    rig.change(tts_enabled=False, stt_enabled=False)
    status = rig.voice.status()
    assert (status.tts_reason, status.stt_reason) == (TTS_OFF, STT_OFF)
    assert rig.service.state == "off"


def test_azure_needs_cloud_permission_and_a_key(rig):
    rig.change(tts_engine="azure")
    assert rig.voice.status().tts_reason == AZURE_NEEDS_CLOUD
    assert "потрібен дозвіл на хмарний AI" in AZURE_NEEDS_CLOUD
    rig.grant(Permission.CLOUD_AI)
    assert rig.voice.status().tts_reason == AZURE_NO_KEY
    rig.voice.set_azure_key("  secret-azure-key  ")
    assert rig.secrets.get(AZURE_KEY_NAME) == "secret-azure-key"
    assert rig.voice.has_azure_key()
    assert rig.voice.status().tts_ready
    engine = rig.service.tts
    assert engine.name == "azure"
    # The key is read per request and only while the permission holds.
    assert engine.get_key() == "secret-azure-key"
    rig.permissions.revoke(Permission.CLOUD_AI)  # in memory at once, before any dispatch
    assert engine.get_key() is None
    rig.dispatch()
    assert rig.voice.status().tts_reason == AZURE_NEEDS_CLOUD
    assert rig.service.tts is None


def test_settings_changes_rebuild_only_what_changed(heard):
    rig = heard
    built = len(rig.engines.built)
    rig.change(volume=40)
    rig.change(section="behavior")
    assert len(rig.engines.built) == built  # nothing the engines depend on
    assert rig.service.volume == pytest.approx(0.4)
    rig.change(piper_speaker="lada", speed=1.2)
    assert rig.engines.built[-1] == ("piper", "lada", 1.2)
    assert rig.service.tts.speaker == "lada"
    rig.change(stt_model="whisper-base")
    assert rig.engines.built[-1] == ("whisper", "whisper-base")
    assert rig.service.stt.model == "whisper-base"


def test_on_change_reports_every_reconfiguration(bus, permissions, clock):
    rig = Rig(bus, permissions, clock, start=False)
    seen = []
    rig.voice._on_change = lambda: seen.append(rig.voice.status().stt_ready)
    rig.voice.start()
    rig.grant(Permission.MICROPHONE)
    rig.revoke(Permission.MICROPHONE)
    assert seen == [False, True, False]


def test_service_options_follow_the_settings(heard):
    rig = heard
    rig.settings.character.name = "Аня"
    rig.change(section="character", end_silence_ms=1200, max_utterance_seconds=30, output_device="Headset")
    service = rig.service
    assert service.prompt == "Аня, привіт."
    assert service.standby_prompt == "Розмова українською. Аня слухає."
    assert "аня" in service.names.weak and "міє" in service.names.strong  # Mia's name still works
    assert service.listen_config.end_silence_ms == 1200
    assert service.listen_config.max_utterance_seconds == 30
    # The open microphone takes requests as long as push-to-talk does; only clacks are dropped.
    assert service.wake_config.end_silence_ms == 1200
    assert service.wake_config.max_utterance_seconds == 30
    assert service.wake_config.min_speech_ms == 350
    assert service.follow_up_seconds == 20
    rig.change(follow_up_seconds=0)
    assert service.follow_up_seconds == 0
    assert rig.output.device == "Headset"


def test_failing_engine_construction_is_reported_not_raised(rig, caplog):
    def broken(voice):
        raise RuntimeError("cannot open C:\\Users\\name\\model")

    rig.engines.piper = broken
    rig.change(piper_speaker="mykyta")
    assert rig.voice.status().tts_reason == integration_module.TTS_BUILD_FAILED
    assert "C:\\Users" not in caplog.text  # the message (paths) is never logged


# --- speech routing ----------------------------------------------------------------------------


@pytest.mark.parametrize(("speak", "solicited", "voiced"), [
    ("all", False, True), ("all", True, True),
    ("replies", False, False), ("replies", True, True),
    ("off", False, False), ("off", True, False),
])
def test_speak_modes(rig, speak, solicited, voiced):
    rig.change(speak=speak)
    shows = Shows()
    rig.voice.route(line(solicited=solicited), shows)
    rig.worker.run_all()
    if voiced:
        assert rig.service.tts.said == ["Привіт!"]
        assert shows.calls == [("Привіт!", pytest.approx(1.0))]  # the bubble lasts as long as the audio
    else:
        assert rig.service.tts.said == []
        assert shows.calls == [("Привіт!", None)]


@pytest.mark.parametrize(("speak_when_hidden", "voiced"), [(True, True), (False, False)])
def test_hidden_by_a_fullscreen_game(rig, speak_when_hidden, voiced):
    rig.change(speak_when_hidden=speak_when_hidden)
    rig.bus.publish(FullscreenChanged(active=True))
    rig.dispatch()
    rig.visible = False
    shows = Shows()
    rig.voice.route(line(), shows)
    rig.worker.run_all()
    assert bool(rig.service.tts.said) is voiced
    assert len(shows.calls) == 1


def test_fullscreen_without_hiding_her_is_normal(rig):
    rig.settings.behavior.hide_on_fullscreen = False
    rig.change(speak_when_hidden=False)
    rig.bus.publish(FullscreenChanged(active=True))
    rig.dispatch()
    rig.voice.route(line(), Shows())
    rig.worker.run_all()
    assert rig.service.tts.said == ["Привіт!"]


def test_hidden_by_the_user_voices_only_answers(rig):
    rig.visible = False
    rig.voice.route(line("сама по собі"), Shows())
    rig.voice.route(line("відповідь", solicited=True), Shows())
    rig.worker.run_all()
    assert rig.service.tts.said == ["відповідь"]


def test_voiced_line_shows_when_audio_starts_and_reports_playback(rig):
    shows = Shows()
    rig.voice.route(line(), shows)
    assert shows.calls == []  # not before the audio
    rig.worker.run_all()
    assert shows.calls == [("Привіт!", pytest.approx(1.0))]
    playback = rig.published(SpeechPlayback)
    assert [(p.playing, p.duration) for p in playback] == [(True, pytest.approx(1.0))]
    rig.output.finish()
    assert [p.playing for p in rig.published(SpeechPlayback)] == [True, False]
    assert len(shows.calls) == 1


def test_failed_synthesis_falls_back_to_text_and_notifies_once(rig, clock):
    rig.service.tts.fail = True
    shows = Shows()
    for _ in range(3):
        rig.voice.route(line(solicited=True), shows)
        rig.worker.run_all()
    assert shows.calls == [("Привіт!", None)] * 3  # never lose a line
    assert rig.notes == ["Голос зламався."]  # not spammy
    clock.advance(integration_module.NOTIFY_REPEAT_SECONDS + 1)
    rig.voice.route(line(solicited=True), shows)
    rig.worker.run_all()
    assert rig.notes == ["Голос зламався."] * 2


def test_dropped_lines_are_shown_as_text(rig):
    rig.service.max_queue = 1
    shows = Shows()
    rig.voice.route(line("перша"), shows)
    rig.voice.route(line("друга"), shows)
    rig.voice.route(line("третя"), shows)  # the queue is full: "друга" is dropped
    assert shows.calls == [("друга", None)]


def test_refused_lines_are_shown_as_text(heard):
    rig = heard
    rig.voice.toggle_listening()
    shows = Shows()
    rig.voice.route(line("коментар"), shows)  # never talk over the user
    assert shows.calls == [("коментар", None)]


def test_a_line_with_nothing_to_say_aloud_is_still_shown(rig):
    shows = Shows()
    rig.voice.route(line("🙂"), shows)
    rig.worker.run_all()
    assert shows.calls == [("🙂", None)]


def test_gesture_without_text_is_not_voiced(rig):
    shows = Shows()
    rig.voice.route(SpeechRequested(text="", state="happy"), shows)
    assert shows.calls == [("", None)]
    rig.worker.run_all()
    assert rig.service.tts.said == []


def test_listening_line_is_never_spoken(rig):
    shows = Shows()
    rig.voice.route(line("Слухаю!", solicited=True, source="voice.listening"), shows)
    rig.voice.route(line("Слухаю!", solicited=True, source="voice.listening:ai"), shows)
    assert shows.calls == [("Слухаю!", None)] * 2
    rig.worker.run_all()
    assert rig.service.tts.said == []


@pytest.mark.parametrize("kind", [InteractionKind.CLICK, InteractionKind.PET, InteractionKind.DRAG_START])
def test_clicking_her_while_she_speaks_interrupts(rig, kind):
    shows = Shows()
    rig.voice.route(line("перша"), shows)
    rig.voice.route(line("друга"), shows)
    rig.worker.run_all()
    assert shows.calls == [("перша", pytest.approx(1.0))]
    rig.bus.publish(CharacterInteraction(kind=kind))
    rig.dispatch()
    assert rig.service.state == "idle"
    assert shows.calls == [("перша", pytest.approx(1.0))]  # the user cut her off: "друга" is dropped
    assert [p.playing for p in rig.published(SpeechPlayback)] == [True, False]


def test_cutting_her_off_never_loses_an_answer_waiting_behind(heard):
    rig = heard
    comment, answer = Shows(), Shows()
    rig.voice.route(line("коментар"), comment)
    rig.voice.route(line("відповідь", solicited=True), answer)
    rig.worker.run_all()
    rig.bus.publish(CharacterInteraction(kind=InteractionKind.CLICK))
    rig.dispatch()
    assert comment.calls == [("коментар", pytest.approx(1.0))]  # her own line: cut off
    assert answer.calls == [("відповідь", None)]  # not heard yet: shown as text
    rig.voice.route(line("друга відповідь", solicited=True), answer)
    rig.worker.run_all()
    rig.voice.route(line("третя відповідь", solicited=True), answer)
    rig.hotkey.press()  # push-to-talk cuts her off too
    assert answer.calls == [("відповідь", None), ("друга відповідь", pytest.approx(1.0)),
                            ("третя відповідь", None)]
    assert rig.service.tts.said == ["коментар", "друга відповідь"]


def test_other_interactions_do_not_interrupt(rig):
    rig.voice.route(line(), Shows())
    rig.worker.run_all()
    rig.bus.publish(CharacterInteraction(kind=InteractionKind.FEED))
    rig.dispatch()
    assert rig.service.state == "speaking"


def test_settings_change_mid_line_shows_the_queued_line_as_text(rig):
    shows = Shows()
    rig.voice.route(line("перша"), shows)
    rig.voice.route(line("друга"), shows)
    rig.worker.run_all()
    rig.change(piper_speaker="lada")  # rebuilds the voice: not the user's doing
    assert shows.calls == [("перша", pytest.approx(1.0)), ("друга", None)]


def test_quiet_stops_her_own_comment_but_keeps_the_answer(rig):
    shows = Shows()
    rig.voice.route(line("коментар"), shows)
    rig.voice.route(line("Добре, мовчу.", solicited=True), shows)
    rig.worker.run_all()
    assert shows.calls == [("коментар", pytest.approx(1.0))]  # speaking her own comment
    rig.bus.publish(QuietModeChanged(enabled=True))
    rig.dispatch()
    assert rig.output.playing is None  # the comment stopped mid-sentence
    assert rig.service.state == "speaking"  # the answer is being synthesised
    rig.worker.run_all()
    assert rig.service.tts.said == ["коментар", "Добре, мовчу."]  # the answer is still said
    assert shows.calls[-1] == ("Добре, мовчу.", pytest.approx(1.0))
    rig.output.finish()
    assert rig.service.state == "idle"
    assert [p.playing for p in rig.published(SpeechPlayback)] == [True, False, True, False]


def test_quiet_drops_queued_comments_without_showing_them(rig):
    shows = Shows()
    rig.voice.route(line("відповідь", solicited=True), shows)
    rig.voice.route(line("коментар 1"), shows)
    rig.voice.route(line("друга відповідь", solicited=True), shows)
    rig.voice.route(line("коментар 2"), shows)
    rig.worker.run_all()
    rig.bus.publish(DoNotDisturbChanged(enabled=True))
    rig.dispatch()
    assert rig.output.playing is not None  # the answer playing now is not cut
    rig.output.finish()
    rig.worker.run_all()  # "коментар 1" comes up next and is skipped at once
    rig.worker.run_all()
    assert rig.service.tts.said[-1] == "друга відповідь"
    rig.output.finish()
    rig.worker.run_all()
    assert rig.service.state == "idle"
    assert [text for text, _ in shows.calls] == ["відповідь", "друга відповідь"]
    assert [p.playing for p in rig.published(SpeechPlayback)] == [True, False, True, False]


def test_a_cancelled_comment_is_skipped_before_it_is_ever_spoken(rig):
    posted = []
    rig.voice._post_override = posted.append  # like the queued Qt invoker: runs later

    def deliver():  # the event loop delivers what the worker posted
        while posted:
            posted.pop(0)()

    shows = Shows()
    rig.voice.route(line("відповідь", solicited=True), shows)
    rig.voice.route(line("коментар"), shows)
    rig.worker.run_all()
    deliver()
    rig.bus.publish(QuietModeChanged(enabled=True))
    rig.dispatch()
    rig.output.finish()
    deliver()
    rig.worker.run_all()
    deliver()
    assert rig.service.tts.said == ["відповідь"]  # not even synthesised: no blip of sound
    assert rig.output.playing is None and rig.service.state == "idle"
    assert [text for text, _ in shows.calls] == ["відповідь"]
    # A stale skip (nothing cancelled is playing any more) never cuts an answer.
    rig.voice.route(line("ще відповідь", solicited=True), shows)
    rig.worker.run_all()
    deliver()
    rig.voice._skip_cancelled()
    assert rig.output.playing is not None


def test_quiet_off_and_answers_only_change_nothing(rig):
    shows = Shows()
    rig.voice.route(line("коментар"), shows)
    rig.worker.run_all()
    rig.bus.publish(QuietModeChanged(enabled=False))
    rig.dispatch()
    assert rig.service.state == "speaking"
    rig.voice.route(line("відповідь", solicited=True), shows)
    rig.output.finish()
    rig.worker.run_all()
    rig.bus.publish(QuietModeChanged(enabled=True))  # only an answer is left: kept
    rig.dispatch()
    assert rig.service.state == "speaking"
    assert shows.calls[-1] == ("відповідь", pytest.approx(1.0))


def test_stopping_observation_drops_her_comments(heard):
    rig = heard
    rig.grant(Permission.APP_METADATA)
    shows = Shows()
    rig.voice.route(line("про гру"), shows)
    rig.voice.route(line("ще про гру"), shows)
    rig.worker.run_all()
    rig.revoke(Permission.APP_METADATA)
    assert rig.service.state == "idle"
    rig.worker.run_all()
    assert rig.service.tts.said == ["про гру"]
    assert shows.calls == [("про гру", pytest.approx(1.0))]
    rig.voice.route(line("знову"), shows)
    rig.voice.route(line("і ще"), shows)
    rig.worker.run_all()
    rig.permissions.stop_all_observation()
    rig.dispatch()
    rig.worker.run_all()
    assert rig.service.state == "idle"
    assert [text for text, _ in shows.calls] == ["про гру", "знову"]


def test_interrupt_api(rig):
    rig.voice.route(line(), Shows())
    rig.worker.run_all()
    rig.voice.interrupt()
    assert rig.service.state == "idle"


def test_route_after_shutdown_shows_text(rig):
    rig.bus.publish(AppShuttingDown())
    rig.dispatch()
    shows = Shows()
    rig.voice.route(line(), shows)
    assert shows.calls == [("Привіт!", None)]


def test_a_failing_show_callback_is_contained(rig, caplog):
    def show(event, seconds):
        raise RuntimeError(event.text)

    rig.voice.route(line(SECRET_TEXT), show)
    rig.worker.run_all()
    assert "failed: RuntimeError" in caplog.text
    assert SECRET_TEXT not in caplog.text


# --- listening, transcripts, her short lines ---------------------------------------------------------


def test_transcript_becomes_an_event_and_is_never_logged(heard, caplog):
    rig = heard
    caplog.set_level(logging.DEBUG)
    assert rig.voice.toggle_listening("button") == ""
    assert rig.mic is not None and rig.mic.device == ""
    rig.feed(np.concatenate([speech(1.0), silence(1.2)]))
    rig.worker.run_all()
    assert [e.text for e in rig.published(VoiceTranscript)] == [SECRET_TEXT]
    assert SECRET_TEXT not in caplog.text
    assert "listening" in [e.state for e in rig.published(VoiceStateChanged)]


def test_nothing_heard_says_so(heard):
    rig = heard
    rig.voice.toggle_listening()
    rig.voice.toggle_listening()  # pressed again at once: nothing was said
    assert rig.composer.calls == [(("voice.not_heard",), True)]
    assert rig.published(VoiceTranscript) == []


def test_not_understood_and_errors_say_so(heard):
    rig = heard
    rig.service.stt.text = ""
    rig.voice.toggle_listening()
    rig.feed(np.concatenate([speech(1.0), silence(1.2)]))
    rig.worker.run_all()
    rig.service.stt.error = STTError("decoder", user_message="Не вдалося розпізнати мовлення.")
    rig.voice.toggle_listening()
    rig.feed(np.concatenate([speech(1.0), silence(1.2)]))
    rig.worker.run_all()
    assert [keys for keys, _ in rig.composer.calls] == [("voice.not_understood",), ("voice.error",)]
    assert all(solicited for _, solicited in rig.composer.calls)
    assert rig.notes == ["Не вдалося розпізнати мовлення."]


def test_wake_phrase_shows_listening_as_a_bubble_only(heard):
    rig = heard
    rig.change(listen_mode="name")
    assert rig.service.state == "standby" and rig.mic is not None
    rig.service.stt.text = "Міє!"
    rig.feed(np.concatenate([speech(1.0), silence(1.0)]))
    rig.worker.run_all()
    assert rig.composer.calls == [(("voice.listening",), True)]
    assert rig.service.state == "standby" and rig.mic is not None  # the open microphone goes on
    shows = Shows()
    listening_line = rig.published(SpeechRequested)[-1]
    rig.voice.route(listening_line, shows)
    assert shows.calls == [("<voice.listening>", None)]  # shown now, never spoken


def test_toggle_listening_without_recognition_returns_the_reason(rig):
    assert rig.voice.toggle_listening() == MIC_NOT_ALLOWED
    assert rig.mic is None


def test_microphone_failure_is_notified(heard):
    rig = heard
    rig.voice._input_factory = lambda device: FakeInput(device, fail=True)
    rig.voice.toggle_listening()
    assert rig.notes == ["Немає мікрофона."]
    assert rig.service.state == "idle"


def test_input_device_comes_from_the_settings(heard):
    rig = heard
    rig.settings.voice.input_device = "USB Mic"
    rig.voice.toggle_listening()
    assert rig.mic.device == "USB Mic"


# --- privacy: revoke and kill switch ---------------------------------------------------------------


def test_revoking_the_microphone_closes_it_and_stops_the_hotkey(heard):
    rig = heard
    hotkey = rig.hotkey
    assert hotkey is not None
    rig.voice.toggle_listening()
    assert rig.mic is not None
    rig.revoke(Permission.MICROPHONE)
    assert rig.mic is None
    assert not hotkey.running and hotkey.stops == 1
    status = rig.voice.status()
    assert not status.stt_ready and status.stt_reason == MIC_NOT_ALLOWED
    assert not status.hotkey_active
    assert rig.service.stt is None  # the model's memory is freed too


def test_kill_switch_closes_the_wake_microphone_and_the_hotkey(heard):
    rig = heard
    rig.change(listen_mode="name")
    assert rig.mic is not None
    hotkey = rig.hotkey
    rig.permissions.stop_all_observation()
    rig.dispatch()
    assert rig.mic is None
    assert not hotkey.running
    assert rig.service.state == "idle"
    assert rig.voice.toggle_listening() == MIC_NOT_ALLOWED


def test_revoke_drops_a_transcription_in_flight(heard):
    rig = heard
    rig.voice.toggle_listening()
    rig.feed(np.concatenate([speech(1.0), silence(1.2)]))
    assert rig.service.state == "transcribing"  # heard while permitted, not recognised yet
    rig.revoke(Permission.MICROPHONE)
    rig.worker.run_all()
    assert rig.published(VoiceTranscript) == []


def test_audio_after_an_in_memory_revoke_is_dropped_on_the_audio_thread(heard):
    rig = heard
    rig.voice.toggle_listening()
    rig.permissions.revoke(Permission.MICROPHONE)  # the main thread hasn't dispatched it yet
    rig.feed(np.concatenate([speech(1.0), silence(1.2)]))
    assert rig.service.state == "listening"  # not a single chunk reached the detector
    assert rig.worker.jobs == [] or all(priority == 2 for priority, _ in rig.worker.jobs)
    rig.dispatch()
    rig.worker.run_all()
    assert rig.mic is None
    assert rig.published(VoiceTranscript) == []


def test_a_listen_after_an_in_memory_revoke_is_refused(heard):
    rig = heard
    rig.permissions.revoke(Permission.MICROPHONE)  # not dispatched yet
    rig.voice.toggle_listening()
    assert rig.mic is None


def test_regranting_restores_recognition_and_the_hotkey(heard):
    rig = heard
    rig.revoke(Permission.MICROPHONE)
    rig.grant(Permission.MICROPHONE)
    assert rig.voice.status().stt_ready
    assert rig.hotkey is not None


# --- push-to-talk hotkey -------------------------------------------------------------------------------


def test_hotkey_toggle_mode(heard):
    rig = heard
    hotkey = rig.hotkey
    assert hotkey.text == "ctrl+alt+space" and hotkey.on_release is None
    hotkey.press()
    assert rig.service.state == "listening"
    rig.feed(speech(0.6))
    hotkey.press()
    assert rig.service.state == "transcribing"
    rig.worker.run_all()
    assert [e.text for e in rig.published(VoiceTranscript)] == [SECRET_TEXT]


def test_hotkey_hold_mode(heard):
    rig = heard
    rig.change(ptt_mode="hold")
    hotkey = rig.hotkey
    assert hotkey.on_release is not None
    assert [h.running for h in rig.hotkeys] == [False, True]  # re-registered for the new mode
    hotkey.press()
    assert rig.service.state == "listening"
    rig.feed(speech(0.6))
    hotkey.release()
    assert rig.service.state == "transcribing"


def test_hotkey_hold_mode_listens_while_the_key_is_held(heard):
    rig = heard
    rig.change(ptt_mode="hold")
    rig.hotkey.press()
    rig.feed(np.concatenate([speech(1.0), silence(1.5), speech(1.0)]))  # a pause mid-sentence
    assert rig.service.state == "listening" and rig.mic is not None  # not cut off: still held
    rig.hotkey.release()
    assert rig.service.state == "transcribing" and rig.mic is None
    rig.worker.run_all()
    assert [e.text for e in rig.published(VoiceTranscript)] == [SECRET_TEXT]


def test_hotkey_press_interrupts_her(heard):
    rig = heard
    shows = Shows()
    rig.voice.route(line(), shows)
    rig.worker.run_all()
    rig.hotkey.press()
    assert rig.service.state == "listening"
    assert [p.playing for p in rig.published(SpeechPlayback)] == [True, False]


def test_hotkey_change_re_registers(heard):
    rig = heard
    first = rig.hotkey
    rig.change(ptt_hotkey="ctrl+shift+space")
    assert not first.running
    assert rig.hotkey.text == "ctrl+shift+space"


def test_hotkey_conflict_is_notified_and_the_button_still_works(bus, permissions, clock):
    rig = Rig(bus, permissions, clock)
    rig.hotkey_error = HotkeyError("taken", user_message="Комбінацію вже зайняла інша програма.")
    rig.grant(Permission.MICROPHONE)
    status = rig.voice.status()
    assert status.stt_ready and not status.hotkey_active
    assert status.hotkey_reason == "Комбінацію вже зайняла інша програма."
    assert rig.notes == ["Комбінацію вже зайняла інша програма."]
    rig.change(volume=50)
    assert len(rig.hotkeys) == 1  # not retried on every settings change
    assert rig.voice.toggle_listening() == ""
    assert rig.mic is not None
    rig.hotkey_error = None
    rig.change(ptt_hotkey="f13")  # another combination is tried
    assert rig.hotkey is not None and rig.voice.status().hotkey_reason == ""


def test_no_hotkey_off_windows(bus, permissions, clock):
    rig = Rig(bus, permissions, clock, hotkey_supported=False)
    rig.grant(Permission.MICROPHONE)
    status = rig.voice.status()
    assert rig.hotkeys == []
    assert status.stt_ready and status.hotkey_reason == HOTKEY_WINDOWS_ONLY
    assert status.hotkey == "Ctrl + Alt + Пробіл"


def test_late_hotkey_press_after_revoke_is_ignored(heard):
    rig = heard
    hotkey = rig.hotkey
    rig.revoke(Permission.MICROPHONE)
    hotkey.press()  # was already queued from the hotkey thread
    assert rig.mic is None


# --- fullscreen and shutdown ------------------------------------------------------------------------


def test_fullscreen_pauses_the_wake_microphone(heard):
    rig = heard
    rig.change(listen_mode="name")
    assert rig.mic is not None
    rig.bus.publish(FullscreenChanged(active=True))
    rig.dispatch()
    assert rig.mic is None and rig.service.state == "idle"
    assert rig.voice.status().pause_reason == PAUSED_FULLSCREEN
    rig.bus.publish(FullscreenChanged(active=False))
    rig.dispatch()
    assert rig.mic is not None
    rig.change(wake_in_fullscreen=True)
    rig.bus.publish(FullscreenChanged(active=True))
    rig.dispatch()
    assert rig.mic is not None


def test_shutdown_stops_everything(heard):
    rig = heard
    hotkey = rig.hotkey
    rig.voice.toggle_listening()
    rig.bus.publish(AppShuttingDown())
    rig.dispatch()
    assert not hotkey.running
    assert rig.mic is None
    assert rig.worker.shut and rig.downloads.shut
    assert rig.voice.status().state == "off"
    rig.voice.shutdown()  # idempotent


# --- "Прослухати" --------------------------------------------------------------------------------------


def test_test_voice_speaks_even_when_speech_is_off(rig):
    rig.change(speak="off")
    results = []
    rig.voice.test_voice(on_done=lambda ok, message: results.append((ok, message)))
    rig.worker.run_all()
    assert rig.service.tts.said == ["Привіт! Я Мія. Ось так звучатиме мій голос."]
    rig.output.finish()
    assert results == [(True, "")]


def test_test_voice_reports_why_not(bus, permissions, clock):
    rig = Rig(bus, permissions, clock, installed=())
    results = []
    rig.voice.test_voice("Тест", on_done=lambda ok, message: results.append((ok, message)))
    assert results == [(False, PIPER_MODEL_MISSING)]


def test_test_voice_reports_a_failure(rig):
    rig.service.tts.fail = True
    results = []
    rig.voice.test_voice(on_done=lambda ok, message: results.append((ok, message)))
    rig.worker.run_all()
    assert results == [(False, "Голос зламався.")]


# --- devices ----------------------------------------------------------------------------------------


def test_devices(rig):
    assert rig.voice.devices() == (["USB Mic"], ["Speakers", "Headset"])

    def broken():
        raise AudioUnavailable("no portaudio")

    rig.voice._list_devices = broken
    assert rig.voice.devices() == ([], [])


# --- models --------------------------------------------------------------------------------------------


def test_models_lists_the_catalog(rig):
    listed = {spec.id: installed for spec, installed in rig.voice.models()}
    assert set(listed) == set(CATALOG)
    assert listed[DEFAULT_TTS_MODEL] and not listed["whisper-medium"]


def test_download_reports_progress_then_enables_the_engine(bus, permissions, clock, monkeypatch):
    monkeypatch.setattr(integration_module, "PROGRESS_INTERVAL", 0.0)
    rig = Rig(bus, permissions, clock, installed={DEFAULT_TTS_MODEL})
    rig.grant(Permission.MICROPHONE)
    assert rig.voice.status().stt_reason == STT_MODEL_MISSING
    spec = CATALOG["whisper-small"]
    progress, done = [], []
    rig.voice.download("whisper-small", lambda d, t: progress.append((d, t)),
                       lambda ok, m: done.append((ok, m)))
    assert rig.voice.download_state().model_id == "whisper-small"
    assert rig.engines.built == [("piper", "tetiana", 1.0)]  # nothing until the worker runs
    rig.downloads.run_all()
    assert progress == [(spec.size // 2, spec.size), (spec.size, spec.size)]
    assert done == [(True, "Завантажено ✓")]
    assert rig.voice.download_state() is None
    assert rig.voice.status().stt_ready  # reconfigured with the new model


def test_progress_is_throttled(rig):
    rig.download_script = [1, 2, 3, 4, CATALOG["whisper-base"].size]
    progress = []
    rig.voice.download("whisper-base", lambda d, t: progress.append(d), lambda ok, m: None)
    rig.downloads.run_all()
    assert progress[0] == 1 and progress[-1] == CATALOG["whisper-base"].size
    assert len(progress) < 5


def test_download_can_be_cancelled(rig):
    done = []
    rig.voice.download("whisper-medium", lambda d, t: None, lambda ok, m: done.append((ok, m)))
    rig.voice.cancel_download()
    rig.downloads.run_all()
    assert done == [(False, "Завантаження скасовано.")]
    assert "whisper-medium" not in rig.installed
    assert rig.voice.download_state() is None


def test_one_download_at_a_time(rig):
    first, second = [], []
    rig.voice.download("whisper-medium", lambda d, t: None, lambda ok, m: first.append(ok))
    rig.voice.download("whisper-base", lambda d, t: None, lambda ok, m: second.append((ok, m)))
    assert second == [(False, "Зачекай, поки завантажиться інша модель.")]
    rig.downloads.run_all()
    assert first == [True]


def test_download_errors_are_reported_without_details(rig, caplog):
    def failing(spec, root, *, on_progress, cancel):
        raise DownloadError("HTTP 500 for model.bin", user_message="Сервер відповів помилкою 500.")

    rig.voice._downloader = failing
    done = []
    rig.voice.download("whisper-base", lambda d, t: None, lambda ok, m: done.append((ok, m)))
    rig.downloads.run_all()
    assert done == [(False, "Сервер відповів помилкою 500.")]

    def crashing(spec, root, *, on_progress, cancel):
        raise OSError("disk full at C:\\Users\\name")

    rig.voice._downloader = crashing
    rig.voice.download("whisper-base", lambda d, t: None, lambda ok, m: done.append((ok, m)))
    rig.downloads.run_all()
    assert done[-1] == (False, "Не вдалося завантажити модель.")
    assert "C:\\Users" not in caplog.text


def test_unknown_model_is_refused(rig):
    done = []
    rig.voice.download("../../etc", lambda d, t: None, lambda ok, m: done.append(ok))
    assert done == [False]
    assert rig.downloads.jobs == []


def test_remove_releases_the_engine_and_deletes_the_files(bus, permissions, clock, tmp_path):
    root = tmp_path / "models"
    spec = CATALOG[DEFAULT_TTS_MODEL]
    directory = models.model_dir(spec, root)
    directory.mkdir(parents=True)
    (directory / "installed.json").write_text("{}", encoding="utf-8")
    (directory / f"{spec.id}.onnx").write_bytes(b"x")
    rig = Rig(bus, permissions, clock, start=False)
    rig.voice._models_root = root
    rig.voice._installed = lambda s: (models.model_dir(s, root) / "installed.json").exists()
    rig.voice.start()
    assert rig.voice.status().tts_ready
    rig.voice.remove(DEFAULT_TTS_MODEL)
    assert not directory.exists()
    assert rig.service.tts is None
    assert rig.voice.status().tts_reason == PIPER_MODEL_MISSING


def test_remove_is_refused_while_that_model_downloads(rig):
    rig.voice.download(DEFAULT_TTS_MODEL, lambda d, t: None, lambda ok, m: None)
    removed = []
    rig.voice._remover = lambda spec, root: removed.append(spec.id)
    rig.voice.remove(DEFAULT_TTS_MODEL)
    assert removed == []


def test_real_threads_marshal_back_to_the_main_thread(bus, permissions, clock, qapp):
    """No ``post`` override: worker results come back through the queued Qt signal."""
    threads = []

    def downloader(spec, root, *, on_progress, cancel):
        on_progress(spec.size, spec.size)

    voice = VoiceIntegration(bus, Settings, permissions, MemorySecretStore(), None,
                             libraries=lambda name: False, downloader=downloader)
    voice.start()
    voice.download("whisper-base", lambda d, t: threads.append(threading.current_thread()),
                   lambda ok, m: threads.append(threading.current_thread()))
    deadline = time.monotonic() + 5
    while len(threads) < 2 and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)
    voice.shutdown()
    assert threads == [threading.main_thread()] * 2


# --- the open microphone: listen modes and pauses ----------------------------------------------------


def say_aloud(rig, text, seconds=1.0):
    """The user says *text* near the open microphone (the fake recogniser returns it)."""
    rig.service.stt.text = text
    rig.feed(np.concatenate([speech(seconds), silence(1.0)]))
    rig.worker.run_all()


def foreground(rig, process="", *, private=False):
    app = AppInfo.private_app() if private else AppInfo(process_name=process, display_name=process)
    category = ActivityCategory.PRIVATE if private else ActivityCategory.WORKING
    rig.bus.publish(ForegroundAppChanged(app=app, category=category, confidence=1.0))
    rig.dispatch()


def test_listen_mode_comes_from_the_settings(heard):
    rig = heard
    status = rig.voice.status()
    assert status.listen_mode == "push" and status.listening_text == "" and rig.mic is None
    rig.change(listen_mode="name")
    status = rig.voice.status()
    assert (status.listen_mode, status.pause_reason, status.listening_text) == ("name", "", LISTENING_NAME)
    assert rig.service.state == "standby" and rig.mic is not None
    rig.change(listen_mode="open")
    assert rig.voice.status().listening_text == LISTENING_OPEN and rig.mic is not None
    rig.change(listen_mode="push")
    assert rig.mic is None and rig.service.state == "idle"


def test_no_open_microphone_without_the_permission(rig):
    rig.change(listen_mode="open")
    assert rig.mic is None
    status = rig.voice.status()
    assert status.listen_mode == "push" and status.listening_text == "" and status.pause_reason == ""


def test_status_text_defaults():
    status = VoiceStatus(tts_ready=True, stt_ready=True, tts_reason="", stt_reason="", state="idle",
                         hotkey="", hotkey_active=False)
    assert status.listen_mode == "push" and status.pause_reason == "" and status.listening_text == ""
    paused = VoiceStatus(tts_ready=True, stt_ready=True, tts_reason="", stt_reason="", state="idle",
                         hotkey="", hotkey_active=False, listen_mode="open", pause_reason=PAUSED_DND)
    assert paused.listening_text == PAUSED_DND


def test_a_request_by_name_becomes_a_transcript_and_is_never_logged(heard, caplog):
    rig = heard
    caplog.set_level(logging.DEBUG)
    rig.change(listen_mode="name")
    say_aloud(rig, "Слухай, Міє, " + SECRET_TEXT)
    rig.clock.advance(60)  # long after: the conversation is over
    say_aloud(rig, "а це не їй, " + SECRET_TEXT)
    assert [e.text for e in rig.published(VoiceTranscript)] == [SECRET_TEXT[:1].upper() + SECRET_TEXT[1:]]
    assert "секретний" not in caplog.text


def test_her_name_alone_shows_listening_without_saying_it(heard):
    rig = heard
    rig.change(listen_mode="open")
    say_aloud(rig, "Міє!")
    assert rig.composer.calls == [(("voice.listening",), True)]
    assert rig.service.state == "standby"
    say_aloud(rig, "котра година?")
    assert [(e.text, e.origin) for e in rig.published(VoiceTranscript)] == [("Котра година?", "named")]


def test_do_not_disturb_pauses_the_open_microphone(heard):
    rig = heard
    rig.change(listen_mode="name")
    reasons = []
    rig.voice._on_change = lambda: reasons.append(rig.voice.status().pause_reason)
    rig.bus.publish(DoNotDisturbChanged(enabled=True))
    rig.dispatch()
    assert rig.mic is None and rig.service.state == "idle"
    assert rig.voice.status().listening_text == PAUSED_DND
    assert reasons == [PAUSED_DND]  # the app refreshes the tray / menus
    assert rig.voice.toggle_listening() == ""  # push-to-talk still works
    assert rig.service.state == "listening"
    rig.voice.toggle_listening()  # nothing said
    assert rig.service.state == "idle"
    rig.bus.publish(DoNotDisturbChanged(enabled=False))
    rig.dispatch()
    assert rig.mic is not None and rig.voice.status().pause_reason == ""


def test_do_not_disturb_from_the_settings(heard):
    rig = heard
    rig.settings.behavior.do_not_disturb = True
    rig.change(listen_mode="open")
    assert rig.mic is None and rig.voice.status().pause_reason == PAUSED_DND


def test_private_apps_and_calls_pause_the_open_microphone(heard):
    rig = heard
    rig.grant(Permission.APP_METADATA)
    rig.change(listen_mode="open")
    foreground(rig, private=True)  # a messenger, mail, a bank, a password manager
    assert rig.mic is None and rig.voice.status().pause_reason == PAUSED_PRIVATE
    foreground(rig, "code.exe")
    assert rig.mic is not None
    foreground(rig, "Webex.exe")
    assert rig.mic is None and rig.voice.status().pause_reason == PAUSED_CALL
    rig.change(pause_in_calls=False)
    assert rig.mic is not None
    foreground(rig, private=True)
    assert rig.mic is None  # private apps: never, whatever the settings say
    rig.revoke(Permission.APP_METADATA)
    assert rig.mic is not None  # the app in front is no longer known
    foreground(rig, private=True)  # queued before the revoke
    assert rig.mic is not None


def test_call_apps():
    for name in ("Discord.exe", "ms-teams.exe", "Zoom.exe", "ts3client_win64.exe", "slack.exe",
                 "Telegram.exe"):
        assert is_call_app(name), name
    for name in ("code.exe", "", "zoomit.exe", "chrome.exe"):
        assert not is_call_app(name), name


def test_the_user_can_pause_the_open_microphone(heard):
    rig = heard
    rig.change(listen_mode="name")
    rig.voice.set_open_mic_paused(True)
    assert rig.voice.open_mic_paused and rig.mic is None
    assert rig.voice.status().pause_reason == PAUSED_BY_USER
    assert rig.composer.calls == [(("voice.mic_paused",), True)]
    rig.voice.set_open_mic_paused(True)  # no change: nothing said
    assert len(rig.composer.calls) == 1
    assert rig.voice.toggle_listening() == "" and rig.service.state == "listening"
    rig.voice.toggle_listening()
    rig.voice.set_open_mic_paused(False)
    assert rig.mic is not None and rig.voice.status().pause_reason == ""
    assert rig.composer.calls[-1] == (("voice.mic_resumed",), True)


def test_pausing_in_push_mode_is_silent_and_remembered(heard):
    rig = heard
    rig.voice.set_open_mic_paused(True, announce=True)
    assert rig.composer.calls == [] and rig.voice.status().pause_reason == ""
    rig.change(listen_mode="name")
    assert rig.mic is None and rig.voice.status().pause_reason == PAUSED_BY_USER


def test_revoking_drops_what_the_open_microphone_was_recognising(heard):
    rig = heard
    rig.change(listen_mode="open")
    rig.feed(np.concatenate([speech(1.0), silence(1.0)]))
    assert rig.worker.jobs  # being recognised
    rig.revoke(Permission.MICROPHONE)
    rig.worker.run_all()
    assert rig.published(VoiceTranscript) == [] and rig.mic is None


def test_an_answer_shown_as_text_keeps_the_conversation_going(heard):
    rig = heard
    rig.change(listen_mode="name", speak="off")
    say_aloud(rig, "Міє, привіт")
    rig.clock.advance(25)  # the model thinks
    rig.voice.route(line("Привіт! Як ти?", solicited=True), Shows())
    rig.clock.advance(15)
    assert rig.service.conversation_open
    say_aloud(rig, "добре, а ти")
    assert [e.text for e in rig.published(VoiceTranscript)] == ["Привіт", "добре, а ти"]


# --- streamed answers -------------------------------------------------------------------------------


class Events:
    def __init__(self):
        self.events = []

    def __call__(self, event, seconds):
        self.events.append((event, seconds))

    @property
    def texts(self):
        return [event.text for event, _ in self.events]


def test_a_streamed_answer_is_spoken_sentence_by_sentence(rig):
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    assert stream.voiced and not stream.spoken
    stream.feed("Привіт! Я тут, ")
    assert rig.service.tts.said == []
    stream.feed("поруч із тобою. Зараз ")
    rig.worker.run_all()
    assert rig.service.tts.said == ["Привіт! Я тут, поруч із тобою."]  # before the model is done
    assert shown.texts == ["Привіт! Я тут, поруч із тобою."]
    stream.feed("розповім, що бачу. І ще дещо")
    rig.worker.run_all()  # synthesised while the first sentence plays
    assert rig.service.tts.said[-1] == "Зараз розповім, що бачу."
    stream.finish("Привіт! Я тут, поруч із тобою. Зараз розповім, що бачу. І ще дещо.", "happy", "smile")
    for _ in range(3):
        rig.output.finish()
        rig.worker.run_all()
    expected = ["Привіт! Я тут, поруч із тобою.", "Зараз розповім, що бачу.", "І ще дещо."]
    assert shown.texts == expected and rig.service.tts.said == expected  # each exactly once
    assert all(event.solicited and event.source == "chat" for event, _ in shown.events)
    assert (shown.events[-1][0].state, shown.events[-1][0].portrait) == ("happy", "smile")
    assert rig.service.state == "idle"


def test_an_answer_she_cannot_voice_is_shown_once_at_the_end(rig):
    rig.change(speak="off")
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    assert not stream.voiced
    stream.feed("Привіт! Я тут, поруч із тобою. ")
    assert shown.events == []
    stream.finish("Привіт! Я тут, поруч із тобою.", "happy")
    stream.finish("ще раз")  # ignored
    assert [(e.text, e.state, seconds) for e, seconds in shown.events] == [
        ("Привіт! Я тут, поруч із тобою.", "happy", None)]
    assert rig.service.tts.said == []


def test_a_short_answer_is_said_whole_at_the_end(rig):
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    stream.feed("Так!")
    stream.finish("Так!", "happy")
    rig.worker.run_all()
    assert rig.service.tts.said == ["Так!"] and shown.events[0][0].state == "happy"


def test_with_nothing_left_at_the_end_only_her_expression_is_shown(rig):
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    stream.feed("Ось і вся моя відповідь. ")
    rig.worker.run_all()
    stream.finish("Ось і вся моя відповідь.", "happy", "smile")
    assert [(e.text, e.state) for e, _ in shown.events] == [("Ось і вся моя відповідь.", "talk"),
                                                           ("", "happy")]
    assert rig.service.tts.said == ["Ось і вся моя відповідь."]


def test_a_final_text_that_differs_routes_only_what_was_not_said(rig):
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    stream.feed("Перше довге речення тут. Друге речення теж є. ")
    rig.worker.run_all()
    stream.finish("Перше довге речення тут.\nДруге речення змінилося. Третє речення.")
    for _ in range(3):
        rig.output.finish()
        rig.worker.run_all()
    assert shown.texts == ["Перше довге речення тут.", "Друге речення теж є.",
                           "Друге речення змінилося. Третє речення."]


def test_an_aborted_answer_is_never_repeated(rig):
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    stream.feed("Перше довге речення тут. Друге речення теж є. ")
    rig.worker.run_all()  # the first plays, the second is ready
    stream.abort()
    assert rig.output.playing is not None  # what she is saying finishes
    rig.output.finish()
    rig.worker.run_all()
    assert shown.texts == ["Перше довге речення тут."]  # the rest: neither said nor shown
    assert len(rig.output.volumes) == 1 and rig.service.state == "idle"
    stream.feed("Ще щось. ")
    stream.finish("Перше довге речення тут. Друге речення теж є.")
    assert shown.texts == ["Перше довге речення тут."] and stream.spoken


def test_cutting_her_off_mid_answer_shows_the_rest_once_as_text(rig):
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    stream.feed("Перше довге речення тут. Друге речення теж є. ")
    rig.worker.run_all()
    rig.bus.publish(CharacterInteraction(kind=InteractionKind.CLICK))
    rig.dispatch()
    assert rig.service.state == "idle"
    stream.feed("Третє речення вже не вголос. ")
    rig.worker.run_all()
    assert shown.texts == ["Перше довге речення тут."]  # the rest comes once the answer is complete
    stream.finish("Перше довге речення тут. Друге речення теж є. Третє речення вже не вголос. Кінець.",
                  "happy")
    assert [(e.text, e.state, seconds) for e, seconds in shown.events[1:]] == [
        ("Друге речення теж є. Третє речення вже не вголос. Кінець.", "happy", None)]
    assert len(rig.output.volumes) == 1  # only the first sentence was ever played


def test_cutting_off_a_finished_answer_shows_what_was_not_heard(rig):
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    stream.feed("Перше довге речення тут. Друге речення теж є. ")
    stream.finish("Перше довге речення тут. Друге речення теж є. Третє речення.")
    rig.worker.run_all()
    rig.voice.interrupt()
    assert shown.texts == ["Перше довге речення тут.", "Друге речення теж є. Третє речення."]


def test_push_to_talk_mid_answer_cuts_it_too(heard):
    rig = heard
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    stream.feed("Перше довге речення тут. ")
    rig.worker.run_all()
    rig.output.finish()  # said; the next sentence is not written yet
    assert rig.service.state == "speaking" and rig.mic is None
    rig.hotkey.press()
    assert rig.service.state == "listening"
    stream.feed("Друге речення теж є. ")
    stream.finish("Перше довге речення тут. Друге речення теж є.")
    assert shown.texts == ["Перше довге речення тут.", "Друге речення теж є."]
    assert shown.events[-1][1] is None and len(rig.output.volumes) == 1  # shown, not voiced


def test_the_microphone_stays_closed_between_sentences(heard):
    rig = heard
    rig.change(listen_mode="name")
    stream = rig.voice.begin_reply(show=Events())
    stream.feed("Перше довге речення тут. ")
    rig.worker.run_all()
    rig.output.finish()
    assert rig.mic is None and rig.service.state == "speaking"
    stream.feed("Друге речення теж є. ")
    rig.worker.run_all()
    rig.output.finish()
    assert rig.mic is None
    stream.finish("Перше довге речення тут. Друге речення теж є.")
    assert rig.service.state == "standby" and rig.mic is not None
    states = [e.state for e in rig.published(VoiceStateChanged)]
    assert states[-3:] == ["standby", "speaking", "standby"]  # no flicker in between


def test_a_stalled_answer_reopens_the_microphone(heard):
    rig = heard
    rig.change(listen_mode="name")
    stream = rig.voice.begin_reply(show=Events())
    stream.feed("Перше довге речення тут. ")
    rig.worker.run_all()
    rig.output.finish()
    rig.voice._check_hold()
    assert rig.mic is None
    rig.clock.advance(REPLY_HOLD_SECONDS + 1)
    rig.voice._check_hold()  # the hold timer
    assert rig.service.state == "standby" and rig.mic is not None
    stream.feed("Нарешті друге речення. ")  # late, but still said
    rig.worker.run_all()
    assert rig.service.tts.said[-1] == "Нарешті друге речення."


def test_an_answer_cut_by_settings_is_shown_not_lost(rig):
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    stream.feed("Перше довге речення тут. Друге речення теж є. ")
    rig.worker.run_all()
    rig.change(piper_speaker="lada")  # not the user's doing
    stream.feed("Третє речення вже новим голосом. ")
    rig.worker.run_all()
    stream.finish("Перше довге речення тут. Друге речення теж є. Третє речення вже новим голосом.")
    assert shown.texts[:2] == ["Перше довге речення тут.", "Друге речення теж є."]
    assert shown.texts[2] == "Третє речення вже новим голосом." and len(shown.texts) == 3


# --- review fixes -------------------------------------------------------------------------------------


def test_transcripts_say_how_they_reached_her(heard):
    rig = heard
    rig.service.stt.text = "Привіт"
    rig.voice.toggle_listening()
    rig.feed(np.concatenate([speech(1.0), silence(1.2)]))
    rig.worker.run_all()
    rig.change(listen_mode="name")
    say_aloud(rig, "Міє, котра година?")
    say_aloud(rig, "а завтра?")  # the conversation window
    assert [(e.text, e.origin) for e in rig.published(VoiceTranscript)] == [
        ("Привіт", "push"), ("Котра година?", "named"), ("а завтра?", "heard")]


def test_a_request_from_the_open_microphone_is_not_sent_once_it_pauses(heard):
    rig = heard
    rig.change(listen_mode="open")
    assert rig.voice.transcript_allowed("heard") and rig.voice.transcript_allowed("push")
    rig.voice.set_open_mic_paused(True)
    assert not rig.voice.transcript_allowed("heard") and not rig.voice.transcript_allowed("named")
    assert rig.voice.transcript_allowed("push")  # push-to-talk works while paused
    rig.voice.set_open_mic_paused(False)
    rig.change(listen_mode="push")
    assert not rig.voice.transcript_allowed("heard") and rig.voice.transcript_allowed("push")
    rig.change(stt_enabled=False)
    assert not rig.voice.transcript_allowed("push")


def test_resuming_while_something_else_keeps_the_microphone_paused_says_why(heard):
    rig = heard
    rig.change(listen_mode="name")
    rig.bus.publish(DoNotDisturbChanged(enabled=True))
    rig.dispatch()
    rig.voice.set_open_mic_paused(True)
    rig.voice.set_open_mic_paused(False)
    assert rig.mic is None and rig.voice.status().pause_reason == PAUSED_DND
    assert (("voice.mic_resumed",), True) not in rig.composer.calls  # never "I hear you again"
    assert rig.notes[-1] == PAUSED_DND


def test_her_question_is_answered_without_her_name(heard):
    rig = heard
    rig.change(listen_mode="name")
    rig.voice.route(line("Бачу помилку. Глянемо разом?", source="observer:ai"), Shows())
    rig.worker.run_all()
    rig.output.finish()
    assert rig.service.conversation_open
    say_aloud(rig, "Так, давай")
    assert [(e.text, e.origin) for e in rig.published(VoiceTranscript)] == [("Так, давай", "heard")]


def test_invites_reply():
    assert integration_module.invites_reply(line("Ого!", source="observer:ai"))
    assert integration_module.invites_reply(line("Чим займаєшся? 🙂", source="initiative.idle"))
    assert not integration_module.invites_reply(line("Ого, бос!", source="vision.boss_fight"))
    assert not integration_module.invites_reply(line("Так?", solicited=True, source="chat"))


def azure_voice(rig):
    rig.secrets.set(AZURE_KEY_NAME, "k" * 32)
    rig.grant(Permission.CLOUD_AI)
    rig.change(tts_engine="azure")
    assert rig.service.tts.name == "azure"


def test_a_cloud_voice_reads_out_what_she_saw_only_with_cloud_vision(rig):
    azure_voice(rig)
    assert not rig.voice.screen_content_may_be_voiced()
    for source in ("observer:ai", "vision.error_on_screen:ai"):
        shows = Shows()
        rig.voice.route(line("Бачу помилку в коді.", source=source), shows)
        rig.worker.run_all()
        assert shows.calls == [("Бачу помилку в коді.", None)]  # shown as text
    assert rig.service.tts.said == []
    rig.voice.route(line("Привіт!", source="initiative.greeting"), Shows())  # nothing from the screen
    rig.worker.run_all()
    assert rig.service.tts.said == ["Привіт!"]
    rig.output.finish()
    rig.grant(Permission.SCREEN_CAPTURE, Permission.CLOUD_VISION)
    assert rig.voice.screen_content_may_be_voiced()
    rig.voice.route(line("Бачу помилку в коді.", source="observer:ai"), Shows())
    rig.worker.run_all()
    assert rig.service.tts.said[-1] == "Бачу помилку в коді."
    rig.output.finish()
    rig.settings.vision.local_only = True
    assert not rig.voice.screen_content_may_be_voiced()


def test_a_local_voice_reads_out_what_she_saw(rig):
    assert rig.voice.screen_content_may_be_voiced()
    rig.voice.route(line("Бачу помилку в коді.", source="observer:ai"), Shows())
    rig.worker.run_all()
    assert rig.service.tts.said == ["Бачу помилку в коді."]


def test_her_own_line_never_comes_between_the_sentences_of_an_answer(rig):
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    stream.feed("Перше довге речення відповіді. Друге")
    rig.worker.run_all()
    rig.output.finish()  # the first sentence is over, the second is still being written
    reaction = Shows()
    rig.voice.route(line("Ого, ти щойно переміг боса!", source="vision.victory"), reaction)
    assert reaction.calls == [("Ого, ти щойно переміг боса!", None)]  # shown, not voiced
    stream.finish("Перше довге речення відповіді. Друге речення відповіді тут.")
    rig.worker.run_all()
    rig.output.finish()
    assert rig.service.tts.said == ["Перше довге речення відповіді.", "Друге речення відповіді тут."]


@pytest.mark.parametrize(
    ("deltas", "final", "spoken"),
    [(["Мія: Привіт, радий тебе чути! Сьогодні чудовий день для прогулянки. Ходімо?"],
      "Привіт, радий тебе чути! Сьогодні чудовий день для прогулянки. Ходімо?",
      ["Мія: Привіт, радий тебе чути!", "Сьогодні чудовий день для прогулянки.", "Ходімо?"]),
     (["Це <b>дуже</b> важливо для тебе. Тож ", "подумай."],
      "Це дуже важливо для тебе. Тож подумай.",
      ["Це <b>дуже</b> важливо для тебе.", "Тож подумай."])],
)
def test_artefacts_the_final_answer_drops_do_not_make_her_repeat_herself(rig, deltas, final, spoken):
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    for delta in deltas:
        stream.feed(delta)
    rig.worker.run_all()
    stream.finish(final)
    for _ in range(4):
        rig.worker.run_all()
        if rig.output.playing:
            rig.output.finish()
    assert rig.service.tts.said == spoken  # each sentence once


# --- faster first words: a long first sentence starts at its first clause ------------------------------


LONG_ANSWER = ("Ну що ж, давай подумаємо разом, що можна зробити з цим завданням. "
               "Спершу перевір налаштування гри. Потім напиши мені.")


def play_all(rig, times=6):
    for _ in range(times):
        rig.worker.run_all()
        if rig.output.playing is not None:
            rig.output.finish()
    rig.worker.run_all()


def test_she_starts_on_the_first_clause_of_a_long_first_sentence(rig):
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    stream.feed("Ну що ж, давай подумаємо разом, ")
    rig.worker.run_all()
    assert rig.service.tts.said == ["Ну що ж, давай подумаємо разом,"]  # long before the sentence ends
    stream.feed("що можна зробити з цим завданням. Спершу перевір налаштування гри. Потім")
    stream.finish(LONG_ANSWER + "", "happy")
    play_all(rig)
    expected = ["Ну що ж, давай подумаємо разом,", "що можна зробити з цим завданням.",
                "Спершу перевір налаштування гри.", "Потім напиши мені."]
    assert rig.service.tts.said == expected and shown.texts == expected  # each part exactly once


def test_an_answer_finished_right_after_its_first_clause_says_only_the_rest(rig):
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    stream.feed("Ну що ж, давай подумаємо разом, ")
    stream.finish("Ну що ж, давай подумаємо разом, що можна зробити.")
    play_all(rig)
    assert shown.texts == ["Ну що ж, давай подумаємо разом,", "що можна зробити."]


def test_cutting_her_off_after_the_first_clause_shows_the_rest_once(rig):
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    stream.feed("Ну що ж, давай подумаємо разом, ")
    rig.worker.run_all()
    rig.voice.interrupt()
    stream.feed("що можна зробити з цим завданням. Спершу перевір налаштування гри. ")
    stream.finish("Ну що ж, давай подумаємо разом, що можна зробити з цим завданням. "
                  "Спершу перевір налаштування гри.")
    assert shown.texts == ["Ну що ж, давай подумаємо разом,",
                           "що можна зробити з цим завданням. Спершу перевір налаштування гри."]
    assert len(rig.output.volumes) == 1


def test_a_different_final_text_after_a_clause_never_repeats_what_was_said(rig):
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    stream.feed("Мія: Ну що ж, давай подумаємо разом, що можна зробити з цим завданням. Спершу ")
    stream.finish("Ну що ж, давай подумаємо разом, що можна зробити з цим завданням. Спершу інше.")
    play_all(rig)
    assert shown.texts == ["Мія: Ну що ж, давай подумаємо разом,", "що можна зробити з цим завданням.",
                           "Спершу інше."]


# --- is the user talking with her (the app publishes ConversationActive) -----------------------------


class Changes:
    def __init__(self, rig):
        self.rig = rig
        self.seen = []
        rig.voice._on_change = self

    def __call__(self):
        self.seen.append(self.rig.voice.conversation_active)

    def flips(self):
        out = []
        for value in self.seen:
            if not out or out[-1] != value:
                out.append(value)
        return out


def push_to_talk(rig, text=SECRET_TEXT, recognition=0.0):
    rig.service.stt.text = text
    rig.hotkey.press()
    rig.feed(np.concatenate([speech(1.0), silence(1.2)]))
    rig.clock.advance(recognition)
    rig.worker.run_all()


def test_on_change_reports_every_flip_of_the_conversation(heard):
    rig = heard
    changes = Changes(rig)
    assert not rig.voice.conversation_active
    push_to_talk(rig)
    assert rig.voice.conversation_active  # her answer is awaited now
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)
    stream.feed("Ось і вся моя відповідь тобі. ")
    stream.finish("Ось і вся моя відповідь тобі.")
    play_all(rig)
    assert rig.service.state == "idle" and rig.voice.conversation_active  # the follow-up window
    rig.clock.advance(rig.settings.voice.follow_up_seconds + 1)
    rig.voice._check_conversation()  # the conversation timer
    assert not rig.voice.conversation_active
    assert changes.flips() == [True, False]
    assert changes.seen.count(False) == 1  # never "over" for a moment in between


def test_an_answer_being_written_is_a_conversation_even_without_a_request(rig):
    rig.change(follow_up_seconds=0)
    changes = Changes(rig)
    stream = rig.voice.begin_reply(show=Events())
    assert rig.voice.conversation_active and changes.seen[-1] is True
    stream.abort()
    assert not rig.voice.conversation_active and changes.seen[-1] is False


def test_the_conversation_timer_wakes_up_when_the_window_closes(heard, qapp):
    rig = heard
    push_to_talk(rig)
    # The follow-up window (20 s) runs out before the wait for her answer (30 s).
    assert rig.voice._conversation_timer.isActive()
    assert abs(rig.voice._conversation_timer.remainingTime() - 20_050) < 1000


def test_no_conversation_after_shutdown(heard):
    rig = heard
    push_to_talk(rig)
    rig.voice.shutdown()
    assert not rig.voice.conversation_active
    assert not rig.voice._conversation_timer.isActive()


# --- "the user started talking": the app warms the model up -------------------------------------------


def test_speech_start_listeners_hear_push_to_talk_and_the_open_microphone(heard):
    rig = heard
    starts = []
    remove = rig.voice.add_speech_start_listener(starts.append)
    rig.hotkey.press()
    assert starts == [SpeechStart("push", True)]
    rig.voice.toggle_listening()  # the microphone button: stop
    rig.worker.run_all()
    rig.change(listen_mode="name")
    rig.feed(speech(0.5))
    assert starts[-1] == SpeechStart("open_mic", False)  # not known yet whether it is for her
    rig.feed(silence(1.2))
    rig.worker.run_all()
    remove()
    rig.voice.toggle_listening()
    assert len(starts) == 2


def test_a_failing_speech_start_listener_is_contained(heard, caplog):
    rig = heard
    starts = []
    rig.voice.add_speech_start_listener(lambda event: 1 / 0)
    rig.voice.add_speech_start_listener(starts.append)
    rig.hotkey.press()
    assert starts == [SpeechStart("push", True)] and rig.service.state == "listening"
    assert "a speech start listener failed" in caplog.text


# --- how long her answer took --------------------------------------------------------------------------


def answer(rig, *, model=0.0, first_part=0.0, synthesis=0.0, text=LONG_ANSWER, source="chat"):
    """The app's part: a chat answer streamed into a ReplyStream, the voice worker
    synthesising; each stage takes the given seconds of the fake clock."""
    shown = Events()
    stream = rig.voice.begin_reply(source, show=shown, voice_request=True)
    rig.clock.advance(model)
    stream.feed(text[:10])
    rig.clock.advance(first_part)
    stream.feed(text[10:])
    rig.clock.advance(synthesis)
    rig.worker.run_all()  # synthesis of the first part; its audio starts
    stream.finish(text)
    play_all(rig)
    return stream, shown


def test_the_last_voice_answer_is_timed_step_by_step_and_logged_without_words(heard, caplog):
    rig = heard
    changes = Changes(rig)
    caplog.set_level(logging.INFO, logger="chibi.voice.integration")
    push_to_talk(rig, recognition=1.5)
    assert rig.voice.last_latency is None
    answer(rig, model=4.0, first_part=0.5, synthesis=0.25)
    latency = rig.voice.last_latency
    assert latency.recognition == pytest.approx(1.5, abs=0.05)
    assert latency.model == pytest.approx(4.0)
    assert latency.first_chunk == pytest.approx(0.5)
    assert latency.synthesis == pytest.approx(0.25)
    assert latency.total == pytest.approx(latency.recognition + 4.75)
    assert latency.pause == pytest.approx(rig.settings.voice.end_silence_ms / 1000)
    assert latency.dominant == "model"
    assert rig.voice.status().latency == latency
    assert len(changes.seen) >= 1  # the page refreshes
    lines = [r.getMessage() for r in caplog.records if "latency" in r.getMessage()]
    assert len(lines) == 1 and "4.00" in lines[0]
    assert SECRET_TEXT not in caplog.text and "подумаємо" not in caplog.text


def test_only_the_first_sentence_of_an_answer_is_timed(heard):
    rig = heard
    push_to_talk(rig)
    answer(rig, model=1.0)
    first = rig.voice.last_latency
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)  # a typed message: no voice request behind it
    stream.feed("Ще одна відповідь без запиту голосом. ")
    stream.finish("Ще одна відповідь без запиту голосом.")
    play_all(rig)
    assert rig.voice.last_latency is first


def test_a_typed_answer_never_inherits_a_voice_request_answered_without_a_stream(heard):
    """"Подивись на екран", the offline responder or "busy" answered the voice request
    without a reply stream: a message typed a minute later is not that request."""
    rig = heard
    push_to_talk(rig, recognition=1.0)
    rig.clock.advance(60.0)
    shown = Events()
    stream = rig.voice.begin_reply(show=shown)  # typed: no voice request behind it
    stream.feed("Відповідь на написане. ")
    stream.finish("Відповідь на написане.")
    play_all(rig)
    assert rig.voice.last_latency is None


def test_a_voice_request_still_waiting_keeps_its_timing_for_its_own_answer(heard):
    rig = heard
    push_to_talk(rig, recognition=1.0)
    typed = rig.voice.begin_reply(show=Events())  # an answer to typed text comes first
    typed.feed("Спершу це. ")
    typed.finish("Спершу це.")
    play_all(rig)
    assert rig.voice.last_latency is None
    answer(rig, model=2.0)  # then the answer to the voice request
    assert rig.voice.last_latency is not None and rig.voice.last_latency.recognition == pytest.approx(
        1.0, abs=0.05)


@pytest.mark.parametrize("case", ["speak_off", "aborted", "cut_off", "stale", "not_chat", "failed"])
def test_answers_that_never_reached_her_voice_are_not_timed(heard, case):
    rig = heard
    push_to_talk(rig)
    if case == "speak_off":
        rig.change(speak="off")
    if case == "stale":
        rig.clock.advance(LATENCY_TRACE_SECONDS + 1)
    if case == "failed":
        rig.service.tts.fail = True
    stream = rig.voice.begin_reply("observer" if case == "not_chat" else "chat", show=Events(),
                                   voice_request=True)
    if case == "aborted":
        stream.abort()
    stream.feed("Ну що ж, давай подумаємо разом, ")
    if case == "cut_off":
        rig.voice.interrupt()
    stream.feed("що можна зробити. ")
    stream.finish("Ну що ж, давай подумаємо разом, що можна зробити.")
    play_all(rig)
    assert rig.voice.last_latency is None


def test_a_request_from_the_open_microphone_is_timed_too(heard):
    rig = heard
    rig.change(listen_mode="name")
    rig.service.stt.text = "Міє, котра година?"
    rig.feed(np.concatenate([speech(1.0), silence(1.0)]))
    rig.clock.advance(2.0)
    rig.worker.run_all()
    assert rig.published(VoiceTranscript)[-1].origin == "named"
    answer(rig, model=1.0)
    # From the moment the pause ended the phrase (0.8 s into the 1 s of silence).
    assert 2.0 <= rig.voice.last_latency.recognition <= 2.25


def test_latency_dominant_part_and_recommended_threads():
    latency = VoiceLatency(recognition=3.0, model=1.0, first_chunk=1.5, synthesis=0.2, total=5.7)
    assert latency.dominant == "recognition"
    assert VoiceLatency(1.0, 1.0, 1.5, 0.2, 3.7).dominant == "model"  # the model's wait and first words
    assert VoiceLatency(0.5, 0.5, 0.1, 2.0, 3.1).dominant == "synthesis"
    assert [recommended_threads(n) for n in (1, 2, 4, 8, 12, 16, 32)] == [2, 2, 2, 4, 6, 6, 6]
    assert 2 <= recommended_threads() <= 6
