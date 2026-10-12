import json

import pytest

from chibi.activity.backend import ForegroundInfo
from chibi.privacy.filters import (
    DEFAULT_DATA_PATH,
    PrivacyFilter,
    compile_keywords,
    load_defaults,
    normalize_text,
)


@pytest.fixture(scope="module")
def privacy():
    return PrivacyFilter()


def info(process="notepad.exe", title=None, exe=None):
    return ForegroundInfo(pid=42, process_name=process, exe_path=exe, title=title)


REQUIRED_PRIVATE_PROCESSES = [
    # password managers
    "keepass.exe", "keepassxc.exe", "1password.exe", "bitwarden.exe", "dashlane.exe", "enpass.exe",
    "roboform.exe", "nordpass.exe", "proton pass.exe",
    # messengers & mail
    "telegram.exe", "signal.exe", "whatsapp.exe", "viber.exe", "discord.exe", "slack.exe", "teams.exe",
    "ms-teams.exe", "skype.exe", "element.exe", "zoom.exe", "thunderbird.exe", "outlook.exe", "olk.exe",
    "mailspring.exe",
    # authenticators
    "authy desktop.exe", "winauth.exe",
]

REQUIRED_TITLE_KEYWORDS = [
    "bank", "банк", "privat24", "приват24", "monobank", "ощадбанк", "paypal", "revolut", "wise",
    "password", "пароль", "1password", "bitwarden", "keepass", "lastpass", "inprivate", "incognito",
    "private browsing", "приватний перегляд", "приватне вікно", "інкогніто", "sign in", "увійти", "вхід",
    "login", "2fa", "authenticator", "gmail", "mail", "пошта", "telegram", "whatsapp", "viber", "signal",
    "discord", "messenger",
]


@pytest.mark.parametrize("process", REQUIRED_PRIVATE_PROCESSES)
def test_default_private_processes(privacy, process):
    assert privacy.is_private_process(process)
    assert privacy.is_private(info(process))


def test_process_check_is_case_insensitive(privacy):
    assert privacy.is_private_process("KeePassXC.exe")
    assert privacy.is_private_process("  Telegram.EXE ")


@pytest.mark.parametrize("process", ["code.exe", "chrome.exe", "cs2.exe", "explorer.exe", "notepad.exe"])
def test_ordinary_apps_are_not_private(privacy, process):
    assert not privacy.is_private_process(process)
    assert not privacy.is_private(info(process))


def test_patterns_catch_variants(privacy):
    assert privacy.is_private_process("discordcanary.exe")
    assert privacy.is_private_process("keepass2.exe")


def test_unidentifiable_process_is_private(privacy):
    assert privacy.is_private_process("")
    assert privacy.is_private_process("   ")


def test_user_exclusions_are_merged():
    f = PrivacyFilter(["MyBank.exe", "secretdiary", r"C:\Tools\Vault.exe", "*broker*.exe", "", 5])
    assert f.is_private_process("mybank.exe")
    assert f.is_private_process("secretdiary.exe")  # ".exe" added to a bare name
    assert f.is_private_process("vault.exe")  # path reduced to the file name
    assert f.is_private_process("stockbroker2.exe")  # glob
    assert f.is_private_process("keepassxc.exe")  # defaults still there
    assert not f.is_private_process("code.exe")


def test_exclusions_do_not_leak_into_other_filters():
    PrivacyFilter(["mybank.exe"])
    assert not PrivacyFilter().is_private_process("mybank.exe")


@pytest.mark.parametrize("keyword", REQUIRED_TITLE_KEYWORDS)
def test_required_title_keywords(privacy, keyword):
    assert privacy.is_private_title(f"Some page - {keyword} - Browser")
    assert privacy.is_private_title(f"{keyword.upper()} — Mozilla Firefox")


@pytest.mark.parametrize(
    "title",
    [
        "Приват24 для бізнесу — Google Chrome",
        "Інтернет-банкінг ПриватБанк",
        "monobank | Web",
        "Sign in - Google Accounts",
        "Inbox (3) - me@gmail.com - Gmail",
        "[InPrivate] New tab - Microsoft Edge",
        "Mozilla Firefox Private Browsing",
        "Нова вкладка — Приватний перегляд Mozilla Firefox",
        "Змінити паролі акаунта",
        "Режим інкогніто",
        "Вхідні - Пошта",
        "Bitwarden Web vault",
        "Revolut Business",
        "Wise - Send money abroad",
        "Sign\u00a0in to your account",  # non-breaking space
    ],
)
def test_private_titles(privacy, title):
    assert privacy.is_private_title(title)


@pytest.mark.parametrize(
    "title",
    [
        "Python tutorial - YouTube - Google Chrome",
        "README.md - chibi - Visual Studio Code",
        "Cyberpunk 2077",
        "It is otherwise fine",  # "wise" only at the start of a word
        "Likewise",
        "Sweet home - Wikipedia",
    ],
)
def test_ordinary_titles(privacy, title):
    assert not privacy.is_private_title(title)


def test_missing_title_is_no_evidence(privacy):
    assert not privacy.is_private_title(None)
    assert not privacy.is_private_title("")
    assert not privacy.is_private(info("chrome.exe", title=None))


def test_is_private_combines_process_title_and_path(privacy):
    assert privacy.is_private(info("chrome.exe", title="Приват24 — Google Chrome"))
    assert not privacy.is_private(info("chrome.exe", title="Weather - Google Chrome"))
    tor = r"C:\Users\me\Desktop\Tor Browser\Browser\firefox.exe"
    assert privacy.is_private(info("firefox.exe", exe=tor))
    assert not privacy.is_private(info("firefox.exe", exe=r"C:\Program Files\Mozilla Firefox\firefox.exe"))
    # the exe file name counts even if the reported process name differs
    assert privacy.is_private(info("unknown.exe", exe=r"C:\Apps\KeePassXC\KeePassXC.exe"))


def test_fails_closed_on_errors(privacy):
    assert privacy.is_private_process(None)  # type: ignore[arg-type]
    assert privacy.is_private_title(123)  # type: ignore[arg-type]
    assert privacy.is_private(object())  # type: ignore[arg-type]


def test_fails_closed_when_data_is_unusable(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    f = PrivacyFilter(data_path=broken)
    assert f.is_private(info("notepad.exe"))
    assert f.is_private_process("code.exe")
    assert f.is_private_title("Weather")
    missing = PrivacyFilter(data_path=tmp_path / "missing.json")
    assert missing.is_private(info("code.exe"))
    wrong_shape = tmp_path / "shape.json"
    wrong_shape.write_text(json.dumps({"processes": "keepass.exe"}), encoding="utf-8")
    assert PrivacyFilter(data_path=wrong_shape).is_private(info("code.exe"))


def test_default_data_is_lower_case_and_well_formed():
    defaults = load_defaults(DEFAULT_DATA_PATH)
    assert defaults.processes
    assert all(p == p.lower().strip() for p in defaults.processes)
    assert len(defaults.title_keywords) >= len(REQUIRED_TITLE_KEYWORDS) // 2


def test_compile_keywords_matches_word_starts_only():
    pattern = compile_keywords(["bank", "sign in"])
    assert pattern.search("Online BANKING")
    assert pattern.search(normalize_text("please  sign\u00a0 in"))
    assert not pattern.search("riverbank")
    assert compile_keywords([]) is None
    assert compile_keywords(["", "  "]) is None
