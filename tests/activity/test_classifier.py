import json

import pytest

from chibi.activity.backend import ForegroundInfo
from chibi.activity.classifier import (
    DEFAULT_DATA_PATH,
    ActivityClassifier,
    AppCatalog,
    parse_catalog,
)
from chibi.core.events import ActivityCategory as C
from chibi.gaming.games_db import GameDatabase
from chibi.privacy.filters import load_defaults


@pytest.fixture(scope="module")
def classifier():
    return ActivityClassifier(GameDatabase.load())


def fg(process, title=None, exe=None):
    return ForegroundInfo(pid=1, process_name=process, exe_path=exe, title=title)


REQUIRED_APPS = {
    C.PROGRAMMING: [
        "code.exe", "devenv.exe", "pycharm64.exe", "idea64.exe", "rider64.exe", "clion64.exe",
        "webstorm64.exe", "goland64.exe", "cursor.exe", "windsurf.exe", "zed.exe", "sublime_text.exe",
        "notepad++.exe", "studio64.exe", "unity.exe", "unrealeditor.exe", "windowsterminal.exe", "wt.exe",
    ],
    C.BROWSING: [
        "chrome.exe", "msedge.exe", "firefox.exe", "opera.exe", "brave.exe", "vivaldi.exe", "browser.exe",
    ],
    C.WATCHING_VIDEO: [
        "vlc.exe", "mpc-hc64.exe", "mpc-be64.exe", "potplayermini64.exe", "mpv.exe", "netflix.exe",
    ],
    C.WORKING: [
        "winword.exe", "excel.exe", "powerpnt.exe", "onenote.exe", "figma.exe", "photoshop.exe",
        "blender.exe", "obsidian.exe", "notion.exe", "acrord32.exe", "obs64.exe",
    ],
}


@pytest.mark.parametrize("category,process", [(c, p) for c, ps in REQUIRED_APPS.items() for p in ps])
def test_required_apps_are_known(classifier, category, process):
    known = classifier.known_app(process)
    assert known is not None, process
    assert known.category is category


def test_known_app_classification(classifier):
    result = classifier.classify(fg("code.exe"))
    assert (result.category, result.confidence, result.display_name) == (C.PROGRAMMING, 0.9, "VS Code")
    assert result.game is None
    assert classifier.classify(fg("WINWORD.EXE")).category is C.WORKING
    assert classifier.classify(fg("vlc.exe")).category is C.WATCHING_VIDEO


def test_godot_pattern(classifier):
    result = classifier.classify(fg("godot_v4.3-stable_win64.exe"))
    assert result.category is C.PROGRAMMING
    assert result.display_name == "Godot"


def test_games_come_first(classifier):
    result = classifier.classify(fg("cs2.exe"))
    assert result.category is C.GAMING
    assert result.confidence == 0.95
    assert result.display_name == "Counter-Strike 2"
    assert result.game.game_id == "cs2"


def test_a_browser_alone_is_not_video(classifier):
    result = classifier.classify(fg("chrome.exe"))
    assert (result.category, result.confidence, result.display_name) == (C.BROWSING, 0.6, "Google Chrome")


@pytest.mark.parametrize(
    "title,category,confidence",
    [
        ("Lo-fi beats - YouTube - Google Chrome", C.WATCHING_VIDEO, 0.8),
        ("Stranger Things | Netflix", C.WATCHING_VIDEO, 0.8),
        ("streamer - Twitch", C.WATCHING_VIDEO, 0.8),
        ("Серіал онлайн — MEGOGO", C.WATCHING_VIDEO, 0.8),
        ("Фільми — Sweet.tv", C.WATCHING_VIDEO, 0.8),
        ("Київстар ТБ", C.WATCHING_VIDEO, 0.8),
        ("anthropics/claude: README - GitHub", C.PROGRAMMING, 0.55),
        ("python - How to sort a dict - Stack Overflow", C.PROGRAMMING, 0.55),
        ("localhost:3000", C.PROGRAMMING, 0.55),
        ("3. Data model — Python 3.12 documentation - docs.python.org", C.PROGRAMMING, 0.55),
        ("Weather in Kyiv", C.BROWSING, 0.6),
    ],
)
def test_browser_title_upgrades(classifier, title, category, confidence):
    result = classifier.classify(fg("msedge.exe", title=title))
    assert (result.category, result.confidence) == (category, confidence)
    assert result.display_name == "Microsoft Edge"


def test_the_site_at_the_end_of_the_title_wins(classifier):
    dev = classifier.classify(fg("firefox.exe", "ytdl-org/youtube-dl - GitHub"))
    video = classifier.classify(fg("firefox.exe", "GitHub Actions tutorial - YouTube"))
    assert dev.category is C.PROGRAMMING
    assert video.category is C.WATCHING_VIDEO


def test_title_rules_need_a_title(classifier):
    # without the WINDOW_TITLES permission there is no title, so no upgrade
    assert classifier.classify(fg("chrome.exe", title=None)).category is C.BROWSING
    assert classifier.classify(fg("chrome.exe", title="")).category is C.BROWSING


def test_known_editor_ignores_the_title(classifier):
    result = classifier.classify(fg("code.exe", title="youtube.py - VS Code"))
    assert (result.category, result.confidence) == (C.PROGRAMMING, 0.9)


def test_unknown_app(classifier):
    result = classifier.classify(fg("mytool.exe", exe=r"C:\Tools\MyTool.exe"))
    assert (result.category, result.confidence, result.display_name) == (C.UNKNOWN, 0.3, "My Tool")
    assert classifier.classify(fg("weird_app.exe")).display_name == "Weird app"


def test_unknown_app_with_title(classifier):
    result = classifier.classify(fg("someplayer.exe", title="Episode 3 - YouTube"))
    assert (result.category, result.confidence) == (C.WATCHING_VIDEO, 0.8)
    assert classifier.classify(fg("someplayer.exe", title="My notes")).category is C.UNKNOWN


def test_launchers_are_known_but_uncategorised(classifier):
    for process, name in [("steam.exe", "Steam"), ("epicgameslauncher.exe", "Epic Games Launcher"),
                          ("battle.net.exe", "Battle.net"), ("riotclientservices.exe", "Riot Client")]:
        result = classifier.classify(fg(process))
        assert result.category is C.UNKNOWN, process
        assert result.display_name == name
        assert result.game is None


def test_heuristic_game(classifier):
    path = r"D:\SteamLibrary\steamapps\common\Cool Game\Cool.exe"
    result = classifier.classify(fg("mycoolgame.exe", exe=path))
    assert result.category is C.GAMING
    assert result.confidence == 0.75
    assert result.display_name == "Cool Game"
    assert result.game.game_id == "exe:mycoolgame.exe"


def test_known_app_beats_game_heuristics(classifier):
    blender = r"D:\SteamLibrary\steamapps\common\Blender\blender.exe"
    assert classifier.classify(fg("blender.exe", exe=blender)).category is C.WORKING
    assert classifier.match_game("blender.exe", blender) is None
    assert classifier.match_game("cs2.exe").game_id == "cs2"


def test_minecraft_java_by_title(classifier):
    result = classifier.classify(fg("javaw.exe", title="Minecraft* 1.21.1 - Multiplayer (3rd-party Server)"))
    assert result.category is C.GAMING and result.game.game_id == "minecraft"
    assert classifier.classify(fg("javaw.exe")).category is C.UNKNOWN


def test_custom_games(classifier):
    custom = ActivityClassifier(GameDatabase.load([{"process": "tool.exe", "title": "Моя гра"}]))
    result = custom.classify(fg("tool.exe"))
    assert result.category is C.GAMING and result.confidence == 1.0 and result.display_name == "Моя гра"


def test_never_private(classifier):
    for process in ("telegram.exe", "keepassxc.exe", "outlook.exe"):
        assert classifier.classify(fg(process, title="Банк")).category is not C.PRIVATE


# -- data hygiene -------------------------------------------------------------------------


def test_catalog_does_not_overlap_games_or_private_apps():
    catalog = AppCatalog.load()
    games = {p for info in GameDatabase.load().games() for p in info.processes}
    private = load_defaults().processes
    apps = set(catalog.apps)
    assert not apps & games
    assert not apps & private


def test_shipped_catalog_parses_strictly():
    catalog = parse_catalog(json.loads(DEFAULT_DATA_PATH.read_text(encoding="utf-8")))
    assert len(catalog.apps) > 100
    assert catalog.video_titles is not None and catalog.dev_titles is not None


def test_unusable_catalog_degrades_to_unknown(tmp_path):
    broken = tmp_path / "apps.json"
    broken.write_text(json.dumps({"categories": {"nonsense": {}}}), encoding="utf-8")
    catalog = AppCatalog.load(broken)
    result = ActivityClassifier(GameDatabase.load(), catalog).classify(fg("code.exe"))
    assert result.category is C.UNKNOWN
    # games still work without the app catalog
    assert ActivityClassifier(GameDatabase.load(), catalog).classify(fg("cs2.exe")).category is C.GAMING
