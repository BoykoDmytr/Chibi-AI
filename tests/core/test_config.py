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
