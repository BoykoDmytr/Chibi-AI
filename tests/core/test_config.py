from chibi.core.config import Settings


def test_defaults_roundtrip():
    s = Settings()
    assert Settings.from_dict(s.to_dict()) == s


def test_defaults_are_privacy_friendly_and_friendly():
    s = Settings()
    assert s.first_run_completed is False
    assert 30 <= s.personality.sarcasm <= 50  # "moderately sarcastic"
    assert s.personality.humor >= 60
    assert s.behavior.do_not_disturb is False


def test_unknown_keys_and_bad_types_fall_back_to_defaults():
    data = Settings().to_dict()
    data["nonsense"] = 1
    data["character"]["scale"] = "huge"
    data["character"]["always_on_top"] = 1  # int is not a bool
    data["personality"] = "garbage"
    s = Settings.from_dict(data)
    assert s.character.scale == 1.0
    assert s.character.always_on_top is True
    assert s.personality == Settings().personality


def test_numbers_are_clamped():
    data = Settings().to_dict()
    data["character"]["scale"] = 99
    data["personality"]["sarcasm"] = -5
    data["behavior"]["max_unsolicited_per_hour"] = 12.6
    s = Settings.from_dict(data)
    assert s.character.scale == 2.5
    assert s.personality.sarcasm == 0
    assert s.behavior.max_unsolicited_per_hour == 13


def test_bool_is_not_accepted_as_number():
    data = Settings().to_dict()
    data["personality"]["humor"] = True
    assert Settings.from_dict(data).personality.humor == 70


def test_pack_name_cannot_traverse_paths():
    data = Settings().to_dict()
    data["character"]["pack"] = "../../etc"
    assert Settings.from_dict(data).character.pack == "mia"


def test_name_is_trimmed_and_never_empty():
    data = Settings().to_dict()
    data["character"]["name"] = "   "
    assert Settings.from_dict(data).character.name == "Мія"
    data["character"]["name"] = "  Кіра  "
    assert Settings.from_dict(data).character.name == "Кіра"


def test_process_lists_are_normalised():
    data = Settings().to_dict()
    data["activity"]["excluded_processes"] = [" MyBank.EXE ", 5, ""]
    data["activity"]["custom_games"] = [{"process": "foo.exe", "title": "Foo"}, "bad"]
    s = Settings.from_dict(data)
    assert s.activity.excluded_processes == ["mybank.exe"]
    assert s.activity.custom_games == [{"process": "foo.exe", "title": "Foo"}]


def test_not_a_dict_gives_defaults():
    assert Settings.from_dict(None) == Settings()


def test_voice_vision_gaming_defaults_are_private_and_off():
    s = Settings()
    assert not s.voice.tts_enabled and not s.voice.stt_enabled and not s.voice.wake_phrase_enabled
    assert s.vision.mode == "off" and s.vision.allowed_games == []
    assert s.gaming.companion_mode and not s.gaming.hints_and_spoilers


def test_voice_settings_are_validated():
    s = Settings.from_dict({"voice": {"ptt_hotkey": "m", "speed": 9, "piper_speaker": "x",
                                      "azure_region": "evil.example/x", "stt_model": "huge",
                                      "speak": "always", "input_device": " Mic " + "x" * 300}})
    v = s.voice
    assert v.ptt_hotkey == "ctrl+alt+space"  # a bare letter would block typing
    assert v.speed == 1.6 and v.piper_speaker == "tetiana"
    assert v.azure_region == "westeurope" and v.stt_model == "whisper-small" and v.speak == "all"
    assert len(v.input_device) == 200 and v.input_device.startswith("Mic")
    assert Settings.from_dict({"voice": {"ptt_hotkey": "Alt+Ctrl+M"}}).voice.ptt_hotkey == "ctrl+alt+m"


def test_vision_settings_are_validated():
    s = Settings.from_dict({"vision": {"mode": "spy", "interval_seconds": 1,
                                       "allowed_games": ["stalker2", "../../etc", "stalker2", "exe:foo.exe"],
                                       "allowed_processes": ["notepad.exe", "rm -rf", "a.EXE"]}})
    assert s.vision.mode == "off"
    assert s.vision.interval_seconds == 10
    assert s.vision.allowed_games == ["stalker2", "exe:foo.exe"]
    assert s.vision.allowed_processes == ["notepad.exe", "a.exe"]


def test_new_sections_round_trip():
    s = Settings()
    s.voice.tts_enabled = True
    s.vision.mode = "game"
    s.vision.allowed_games = ["cs2"]
    s.gaming.reaction_frequency = 20
    assert Settings.from_dict(s.to_dict()) == s
