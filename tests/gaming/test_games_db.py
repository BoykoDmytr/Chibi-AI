import json

import pytest

from chibi.gaming.games_db import (
    DEFAULT_DATA_PATH,
    GENRES,
    GameDatabase,
    is_helper_process,
    library_location,
    prettify_name,
    title_from_exe,
    validate_games_data,
)

CANONICAL_IDS = [
    "stalker2", "stalker_classic", "cs2", "dota2", "minecraft", "witcher3", "cyberpunk2077", "bg3",
    "eldenring", "gta5", "valorant", "lol", "fortnite", "apex", "overwatch2", "pubg", "rust", "tarkov",
    "wow", "terraria", "stardew", "rdr2", "skyrim", "fallout4", "starfield", "hollowknight", "hades",
    "genshin", "hsr", "factorio", "civ6", "osu", "rocketleague", "amongus", "lethalcompany", "palworld",
    "helldivers2", "wukong", "metro_exodus", "dbd", "warframe", "destiny2", "valheim", "ets2",
    "darksouls3", "sekiro", "poe", "diablo4", "rimworld", "kcd2", "mhwilds", "sims4", "hearthstone",
    "phasmophobia", "deeprock", "subnautica",
]

LAUNCHERS = [
    "steam.exe", "steamwebhelper.exe", "epicgameslauncher.exe", "battle.net.exe", "riotclientservices.exe",
    "riotclientux.exe", "leagueclient.exe", "leagueclientux.exe", "galaxyclient.exe", "eadesktop.exe",
    "origin.exe", "upc.exe", "ubisoftconnect.exe", "minecraftlauncher.exe", "javaw.exe", "java.exe",
    "xboxpcapp.exe", "launcher.exe",
]


@pytest.fixture(scope="module")
def data():
    return json.loads(DEFAULT_DATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def db():
    return GameDatabase.load()


# -- the shipped data -----------------------------------------------------------------


def test_shipped_data_is_valid(data):
    assert validate_games_data(data) == []


def test_genres_are_the_fixed_set(data):
    expected = {
        "shooter", "souls", "rpg", "strategy", "sandbox", "moba", "cozy", "survival",
        "horror", "racing", "sim", "rhythm", "mmo", "action", "party", "card",
    }
    assert expected == GENRES
    assert set(data["genres"]) == GENRES
    assert {g["genre"] for g in data["games"]} <= GENRES


@pytest.mark.parametrize("game_id", CANONICAL_IDS)
def test_canonical_ids_exist_with_processes(db, game_id):
    info = db.get(game_id)
    assert info is not None, game_id
    assert info.processes
    assert info.title.strip()
    assert info.genre in GENRES


def test_about_sixty_games_or_more(db):
    assert len(db) >= 60


def test_no_process_belongs_to_two_games(data):
    seen: dict[str, str] = {}
    for game in data["games"]:
        for process in game["processes"]:
            assert process not in seen, f"{process} in {seen.get(process)} and {game['id']}"
            seen[process] = game["id"]


def test_processes_are_lower_case_exe_names(data):
    for game in data["games"]:
        for process in game["processes"]:
            assert process == process.lower().strip()
            assert process.endswith(".exe")
            assert "\\" not in process and "/" not in process


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_launchers_are_never_games(db, launcher):
    assert db.match(launcher) is None


def test_specific_exe_names(db):
    for process, game_id in [
        ("stalker2-win64-shipping.exe", "stalker2"),
        ("stalker2.exe", "stalker2"),
        ("xr_3da.exe", "stalker_classic"),
        ("xrengine.exe", "stalker_classic"),
        ("minecraft.windows.exe", "minecraft"),
        ("league of legends.exe", "lol"),
        ("bg3_dx11.exe", "bg3"),
        ("gta5_enhanced.exe", "gta5"),
        ("valorant-win64-shipping.exe", "valorant"),
    ]:
        match = db.match(process)
        assert match is not None and match.game_id == game_id, process


# -- exact matches ---------------------------------------------------------------------


def test_exact_match_is_case_insensitive(db):
    match = db.match("CS2.exe")
    assert match.game_id == "cs2"
    assert match.title == "Counter-Strike 2"
    assert match.genre == "shooter"
    assert match.confidence == 0.95
    assert match.source == "db"


def test_exact_match_ignores_exe_path(db):
    match = db.match("dota2.exe", r"D:\Somewhere\Else\dota2.exe")
    assert match.game_id == "dota2" and match.source == "db"


def test_empty_or_unknown_process(db):
    assert db.match("") is None
    assert db.match("notepad.exe") is None
    assert db.match("notepad.exe", r"C:\Windows\notepad.exe") is None


def test_minecraft_java_only_by_title(db):
    assert db.match("javaw.exe") is None
    match = db.match_title("javaw.exe", "Minecraft* 1.21.1 - Singleplayer")
    assert match.game_id == "minecraft" and match.source == "title"
    assert db.match_title("javaw.exe", "Minecraft 1.20.4") is not None
    assert db.match_title("javaw.exe", "Minecraft Launcher") is None
    assert db.match_title("javaw.exe", "IntelliJ IDEA") is None
    assert db.match_title("notepad.exe", "Minecraft 1.20.4") is None
    assert db.match_title("javaw.exe", None) is None


# -- heuristics ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,title",
    [
        (r"D:\SteamLibrary\steamapps\common\My Cool Game\Bin\MyCoolGame.exe", "My Cool Game"),
        ("D:/SteamLibrary/SteamApps/Common/My Cool Game/MyCoolGame.exe", "My Cool Game"),
        (r"C:\Program Files\Epic Games\HogwartsLike\Bin\Win64\MyCoolGame.exe", "Hogwarts Like"),
        (r"C:\Program Files (x86)\GOG Galaxy\Games\Cool Game\MyCoolGame.exe", "Cool Game"),
        (r"D:\XboxGames\Cool Game\Content\MyCoolGame.exe", "Cool Game"),
        (r"C:\Riot Games\Cool Game\live\MyCoolGame.exe", "Cool Game"),
        (r"C:\Program Files (x86)\Ubisoft\Ubisoft Game Launcher\games\Cool Game\MyCoolGame.exe", "Cool Game"),
        (r"C:\Program Files\EA Games\Cool Game\MyCoolGame.exe", "Cool Game"),
    ],
)
def test_game_library_folders(db, path, title):
    match = db.match("mycoolgame.exe", path)
    assert match is not None, path
    assert match.game_id == "exe:mycoolgame.exe"
    assert match.genre == "action"
    assert match.source == "heuristic"
    assert match.confidence == 0.75
    assert match.title == title


def test_unreal_shipping_exe(db):
    match = db.match("mygame-win64-shipping.exe")
    assert match.confidence == 0.7
    assert match.game_id == "exe:mygame-win64-shipping.exe"
    assert match.title == "Mygame"
    with_path = db.match("mygame-win64-shipping.exe", r"C:\Games\X\Binaries\Win64\MyGame-Win64-Shipping.exe")
    assert with_path.title == "My Game"  # original spelling from the path
    in_library = db.match(
        "mygame-win64-shipping.exe", r"E:\Epic Games\Dragon Saga\Binaries\Win64\MyGame-Win64-Shipping.exe"
    )
    assert in_library.confidence == 0.75 and in_library.title == "Dragon Saga"


def test_outside_libraries_is_not_a_game(db):
    assert db.match("mytool.exe", r"C:\Program Files\MyTool\mytool.exe") is None
    # an exe directly in the library root has no install folder: not a game
    assert db.match("odd.exe", r"D:\SteamLibrary\steamapps\common\odd.exe") is None


@pytest.mark.parametrize(
    "process,path",
    [
        ("unitycrashhandler64.exe", r"D:\SteamLibrary\steamapps\common\Cool Game\UnityCrashHandler64.exe"),
        ("crashreportclient.exe", r"C:\Program Files\Epic Games\Cool\Engine\CrashReportClient.exe"),
        ("easyanticheat_eos_setup.exe", r"D:\SteamLibrary\steamapps\common\Cool\EasyAntiCheat_EOS_Setup.exe"),
        ("beservice_x64.exe", r"D:\SteamLibrary\steamapps\common\Cool\BattlEye\BEService_x64.exe"),
        ("start_protected_game.exe", r"D:\SteamLibrary\steamapps\common\Cool\start_protected_game.exe"),
        ("unins000.exe", r"C:\GOG Games\Cool Game\unins000.exe"),
        ("vc_redist.x64.exe",
         r"D:\SteamLibrary\steamapps\common\Steamworks Shared\_CommonRedist\vc_redist.x64.exe"),
        ("game.exe", r"D:\SteamLibrary\steamapps\common\Cool\_CommonRedist\DirectX\game.exe"),
        ("ue4prereqsetup_x64.exe", r"D:\SteamLibrary\steamapps\common\Cool\Engine\UE4PrereqSetup_x64.exe"),
        ("coollauncher.exe", r"D:\SteamLibrary\steamapps\common\Cool\CoolLauncher.exe"),
        ("epicgameslauncher.exe",
         r"C:\Program Files (x86)\Epic Games\Launcher\Portal\Binaries\Win64\EpicGamesLauncher.exe"),
        ("epicwebhelper.exe", r"C:\Program Files (x86)\Epic Games\Launcher\Engine\EpicWebHelper.exe"),
        ("riotclientservices.exe", r"C:\Riot Games\Riot Client\RiotClientServices.exe"),
        ("leagueclient.exe", r"C:\Riot Games\League of Legends\LeagueClient.exe"),
        ("wallpaper64.exe", r"D:\SteamLibrary\steamapps\common\wallpaper_engine\wallpaper64.exe"),
        ("vrserver.exe", r"D:\SteamLibrary\steamapps\common\SteamVR\bin\win64\vrserver.exe"),
        ("crashreportclient-win64-shipping.exe", None),
    ],
)
def test_helpers_are_rejected(db, process, path):
    assert db.match(process, path) is None


def test_helper_name_patterns():
    assert is_helper_process("UnityCrashHandler32.exe")
    assert is_helper_process("BEService.exe")
    assert is_helper_process("EasyAntiCheat.exe")
    assert not is_helper_process("eldenring.exe")
    assert not is_helper_process("crashbandicoot.exe")  # a game, not a crash handler


def test_library_location():
    assert library_location(r"D:\SteamLibrary\steamapps\common\Game\bin\g.exe") == ("Game", False)
    assert library_location(r"D:\SteamLibrary\steamapps\common\Game\Redist\g.exe") == ("Game", True)
    assert library_location(r"C:\Windows\notepad.exe") is None
    # folders before the library marker do not matter
    assert library_location(r"D:\Launcher\steamapps\common\Game\g.exe") == ("Game", False)


def test_title_helpers():
    assert title_from_exe("HogwartsLegacy-Win64-Shipping.exe") == "Hogwarts Legacy"
    assert title_from_exe("Cyberpunk2077.exe") == "Cyberpunk 2077"
    assert title_from_exe("my_cool_game.exe") == "My cool game"
    assert title_from_exe("ACValhalla.exe") == "AC Valhalla"
    assert prettify_name("Black Myth Wukong") == "Black Myth Wukong"
    assert prettify_name("RocketLeague") == "Rocket League"


# -- user games ---------------------------------------------------------------------------


def test_custom_games():
    db = GameDatabase.load([{"process": "MyIndie.exe", "title": "Моя інді-гра"}, {"process": "tinygame"}])
    match = db.match("myindie.exe")
    assert match.game_id == "custom:myindie.exe"
    assert match.title == "Моя інді-гра"
    assert match.genre == "action"
    assert match.confidence == 1.0
    assert match.source == "custom"
    tiny = db.match("tinygame.exe")
    assert tiny.game_id == "custom:tinygame.exe" and tiny.title == "Tinygame"
    assert len(db) == len(GameDatabase.load()) + 2


def test_custom_games_ignore_garbage():
    base = len(GameDatabase.load())
    db = GameDatabase.load([{"process": ""}, {"title": "no process"}, "cs2.exe", {"process": "  "}])  # type: ignore[list-item]
    assert len(db) == base


def test_custom_game_for_known_exe_keeps_the_known_game():
    db = GameDatabase.load([{"process": "cs2.exe", "title": "My CS"}])
    match = db.match("cs2.exe")
    assert match.game_id == "cs2" and match.source == "db"


def test_custom_games_do_not_change_other_instances():
    GameDatabase.load([{"process": "only-here.exe", "title": "X"}])
    assert GameDatabase.load().match("only-here.exe") is None


# -- robustness ---------------------------------------------------------------------------


def test_invalid_entries_are_skipped(tmp_path):
    path = tmp_path / "games.json"
    path.write_text(json.dumps({"games": [
        {"id": "good", "title": "Good", "genre": "rpg", "processes": ["good.exe"]},
        {"id": "bad genre", "title": "Bad", "genre": "nope", "processes": ["bad.exe"]},
        {"id": "dup", "title": "Dup", "genre": "rpg", "processes": ["good.exe"]},
        {"id": "upper", "title": "Upper", "genre": "rpg", "processes": ["Upper.exe"]},
        "not an object",
    ]}), encoding="utf-8")
    db = GameDatabase.load(path=path)
    assert len(db) == 1
    assert db.match("good.exe").game_id == "good"
    assert db.match("bad.exe") is None
    problems = validate_games_data(json.loads(path.read_text(encoding="utf-8")))
    assert len(problems) == 5  # the second entry has two problems (id and genre)


def test_unreadable_database_still_allows_heuristics_and_user_games(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{oops", encoding="utf-8")
    db = GameDatabase.load([{"process": "mine.exe", "title": "Mine"}], path=path)
    assert len(db) == 1
    assert db.match("cs2.exe") is None
    assert db.match("mine.exe").source == "custom"
    assert db.match("x-win64-shipping.exe").source == "heuristic"
