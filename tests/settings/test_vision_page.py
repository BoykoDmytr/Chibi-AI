"""The "Бачення екрана" settings page: fields bound to Settings.vision (with the
dialog's baseline rule), the games checklist and app list, permission buttons
through the real PermissionManager (consent first, revoke at once), status and
"Що Мія бачила"."""

from __future__ import annotations

import re

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel, QLineEdit, QRadioButton

import chibi.privacy.permissions as permissions_module
import chibi.settings.settings_dialog as settings_dialog
from chibi.core.events import Permission
from chibi.settings.vision_page import (
    FALLBACK_CONSENT,
    MODE_CHOICES,
    VisionPage,
    consent_for,
    describe_seen,
    describe_status,
    game_title_for,
    interval_text,
    normalize_app_name,
)
from chibi.vision.integration import SeenObservation, VisionStatusView
from tests.personality.test_phrase_bank import GENDERED_USER
from tests.settings.fakes import FakeFacade

TITLE = "S.T.A.L.K.E.R. 2: Серце Чорнобиля"


class VisionFakeFacade(FakeFacade):
    def __init__(self, permissions, data_dir) -> None:
        super().__init__(permissions, data_dir)
        self.vision_view = VisionStatusView(mode="off", state="off", app_title="", blocked="", local=None)
        self.recent: list[SeenObservation] = []
        self.cleared = 0
        self.games = [("stalker2", TITLE), ("cs2", "Counter-Strike 2"), ("minecraft", "Minecraft")]

    def vision_status(self) -> VisionStatusView:
        return self.vision_view

    def vision_recent(self) -> list[SeenObservation]:
        return list(self.recent)

    def vision_clear_history(self) -> None:
        self.cleared += 1
        self.recent.clear()

    def vision_games(self) -> list[tuple[str, str]]:
        return list(self.games)


class Confirm:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.asked: list[tuple[str, str]] = []

    def __call__(self, title: str, text: str) -> bool:
        self.asked.append((title, text))
        return self.answer


@pytest.fixture(autouse=True)
def _all_permissions_available(monkeypatch):
    monkeypatch.setattr(permissions_module, "AVAILABLE", frozenset(Permission))


@pytest.fixture
def vfacade(permissions, tmp_path):
    return VisionFakeFacade(permissions, tmp_path / "data")


@pytest.fixture
def make_page(vfacade, track):
    def make(confirm: Confirm | None = None, **vision) -> VisionPage:
        if vision:
            s = vfacade.settings()
            for name, value in vision.items():
                setattr(s.vision, name, value)
            vfacade.apply_settings(s)
            vfacade.applied.clear()
        page = track(VisionPage(vfacade, confirm or Confirm(False)))
        page.show()
        return page

    return make


def collected(page: VisionPage, facade: VisionFakeFacade):
    return page.collect(facade.settings()).vision


def game_items(page: VisionPage) -> dict[str, object]:
    return {page.games_list.item(i).data(Qt.ItemDataRole.UserRole): page.games_list.item(i)
            for i in range(page.games_list.count())}


def texts(page: VisionPage) -> str:
    return "\n".join(label.text() for label in page.findChildren(QLabel))


# --- mode and plain fields ------------------------------------------------------------------------


def test_four_modes_with_explanations_in_ukrainian(make_page):
    page = make_page()
    radios = page.findChildren(QRadioButton)
    assert [r.text() for r in radios] == ["Вимкнено", "Лише за запитом", "Автоматично", "Ігровий режим"]
    assert [value for value, _t, _h in MODE_CHOICES] == ["off", "manual", "auto", "game"]
    assert page.mode() == "off" and page.mode_buttons["off"].isChecked()
    shown = texts(page)
    for _value, _text, hint in MODE_CHOICES:
        assert hint in shown and "\n" not in hint


def test_edited_fields_are_collected(make_page, vfacade):
    page = make_page()
    page.mode_buttons["game"].setChecked(True)
    page.interval_row.setValue(120)
    page.confidence_row.setValue(75)
    page.local_only_check.setChecked(True)
    page.model_edit.setText("  llava  ")
    page.adapt_check.setChecked(False)
    v = collected(page, vfacade)
    assert (v.mode, v.interval_seconds, v.min_confidence, v.local_only, v.model, v.adapt_to_load) == \
        ("game", 120, 0.75, True, "llava", False)


def test_unedited_fields_never_revert_changes_made_elsewhere(make_page, vfacade):
    page = make_page()
    s = vfacade.settings()
    s.vision.mode = "auto"
    s.vision.allowed_games = ["cs2"]
    vfacade.apply_settings(s)  # e.g. changed from the tray while the dialog is open
    page.interval_row.setValue(60)
    v = collected(page, vfacade)
    assert (v.mode, v.allowed_games, v.interval_seconds) == ("auto", ["cs2"], 60)
    assert page.has_edits()


def test_sync_respects_edits_until_applied(make_page, vfacade):
    page = make_page()
    page.model_edit.setText("llava")
    s = vfacade.settings()
    s.vision.mode = "manual"
    s.vision.model = "gemma3"
    page.sync(s, edited_too=False)
    assert page.mode() == "manual" and page.model_edit.text() == "llava"
    page.sync(s, edited_too=True)
    assert page.model_edit.text() == "gemma3" and not page.has_edits()


def test_confidence_and_interval_widgets_show_settings(make_page):
    page = make_page(min_confidence=0.85, interval_seconds=90)
    assert page.confidence_row.value() == 85 and page.interval_row.value() == 90
    assert page.interval_row.value_label.text() == "1\u00a0хв 30\u00a0с"
    page.reset_baseline()
    assert not page.has_edits()


# --- games -------------------------------------------------------------------------------------------


def test_games_checklist_from_the_database_sorted_with_unknown_ids_kept(make_page, vfacade):
    page = make_page(allowed_games=["cs2", "exe:indie-win64-shipping.exe"])
    items = game_items(page)
    titles = [page.games_list.item(i).text() for i in range(page.games_list.count())]
    assert titles[:3] == ["Counter-Strike 2", "Minecraft", TITLE]
    assert items["exe:indie-win64-shipping.exe"].text() == "Indie"
    checked = [gid for gid, item in items.items() if item.checkState() == Qt.CheckState.Checked]
    assert checked == ["cs2", "exe:indie-win64-shipping.exe"]
    assert page.games_count_label.text() == "Вибрано 2 гри."
    items["stalker2"].setCheckState(Qt.CheckState.Checked)
    assert page.games_count_label.text() == "Вибрано 3 гри."
    assert collected(page, vfacade).allowed_games == ["cs2", "stalker2", "exe:indie-win64-shipping.exe"]


def test_game_search_filters_by_title_and_id(make_page):
    page = make_page()
    page.games_search.setText("сталкер")  # nothing: the title is S.T.A.L.K.E.R.
    assert all(page.games_list.item(i).isHidden() for i in range(page.games_list.count()))
    page.games_search.setText("серце")
    visible = [page.games_list.item(i).text() for i in range(page.games_list.count())
               if not page.games_list.item(i).isHidden()]
    assert visible == [TITLE]
    page.games_search.setText("cs2")
    visible = [page.games_list.item(i).text() for i in range(page.games_list.count())
               if not page.games_list.item(i).isHidden()]
    assert visible == ["Counter-Strike 2"]
    page.games_search.clear()
    assert not any(page.games_list.item(i).isHidden() for i in range(page.games_list.count()))


def test_games_can_be_checked_with_the_keyboard(make_page, vfacade):
    page = make_page()
    page.games_list.setFocus()
    page.games_list.setCurrentRow(0)
    QTest.keyClick(page.games_list, Qt.Key.Key_Space)
    assert collected(page, vfacade).allowed_games == ["cs2"]


def test_empty_games_list_points_to_the_games_page(make_page, vfacade):
    vfacade.games = []
    page = make_page()
    assert "«Ігри»" in page.games_count_label.text()


# --- apps --------------------------------------------------------------------------------------------


def test_apps_are_added_normalised_and_removed(make_page, vfacade):
    page = make_page()
    page.app_edit.setText(r"C:\Tools\OBS64.EXE")
    page.add_app_button.click()
    page.app_edit.setText("krita")
    QTest.keyClick(page.app_edit, Qt.Key.Key_Return)  # Enter adds, never presses "Гаразд"
    page.app_edit.setText("obs64.exe")
    page.add_app_button.click()  # duplicate ignored
    assert collected(page, vfacade).allowed_processes == ["obs64.exe", "krita.exe"]
    page.apps_list.item(0).setSelected(True)
    page.remove_apps_button.click()
    assert collected(page, vfacade).allowed_processes == ["krita.exe"]


@pytest.mark.parametrize(("raw", "fragment"), [
    ("мій редактор", "не схоже"),
    ("keepass.exe", "приватна"),
    ("Telegram", "приватна"),
])
def test_bad_or_private_apps_are_refused_with_a_reason(make_page, vfacade, raw, fragment):
    page = make_page()
    page.app_edit.setText(raw)
    page.add_app_button.click()
    assert fragment in page.app_error.text() and not page.app_error.isHidden()
    assert collected(page, vfacade).allowed_processes == []


def test_the_users_own_private_apps_are_refused_too(make_page, vfacade):
    s = vfacade.settings()
    s.activity.excluded_processes = ["mybank.exe"]
    vfacade.apply_settings(s)
    page = make_page()
    page.app_edit.setText("MyBank")
    page.add_app_button.click()
    assert "приватна" in page.app_error.text()


def test_normalize_app_name():
    assert normalize_app_name('  "C:\\Program Files\\OBS\\obs64.exe" ') == "obs64.exe"
    assert normalize_app_name("Krita") == "krita.exe"
    assert normalize_app_name("my app") == "my app.exe"
    assert normalize_app_name("редактор") is None
    assert normalize_app_name("bad|name") is None
    assert normalize_app_name("   ") is None


# --- permissions -------------------------------------------------------------------------------------


def test_capture_is_granted_only_after_consent(make_page, vfacade):
    confirm = Confirm(False)
    page = make_page(confirm)
    row = page.permission_rows[Permission.SCREEN_CAPTURE]
    assert row.button.text() == "Дозволити…" and row.button.isEnabled()
    row.button.click()
    assert confirm.asked and not vfacade.permissions.is_granted(Permission.SCREEN_CAPTURE)
    confirm.answer = True
    emitted = []
    page.permissions_changed.connect(lambda: emitted.append(1))
    row.button.click()
    assert vfacade.permissions.is_granted(Permission.SCREEN_CAPTURE)
    assert row.button.text() == "Відкликати" and row.state_label.text() == "✓ дозволено"
    assert emitted == [1]


def test_consent_text_is_the_privacy_pages(make_page):
    confirm = Confirm(False)
    page = make_page(confirm)
    page.permission_rows[Permission.SCREEN_CAPTURE].button.click()
    expected = settings_dialog.PERMISSION_TEXTS[Permission.SCREEN_CAPTURE]
    assert confirm.asked == [(expected.consent_title, expected.consent_text)]


def test_fallback_consent_when_the_privacy_page_has_none(monkeypatch):
    monkeypatch.setattr(settings_dialog, "PERMISSION_TEXTS", {})
    assert consent_for(Permission.CLOUD_VISION) == FALLBACK_CONSENT[Permission.CLOUD_VISION]
    _title, text = FALLBACK_CONSENT[Permission.SCREEN_CAPTURE]
    assert "не зберігаються" in text and "ніколи" in text and not re.search(r"\bсам\b", text)


def test_cloud_analysis_needs_capture_first_and_revoking_needs_no_consent(make_page, vfacade):
    confirm = Confirm(True)
    page = make_page(confirm)
    cloud = page.permission_rows[Permission.CLOUD_VISION]
    assert not cloud.button.isEnabled() and "спершу" in cloud.state_label.text()
    page.permission_rows[Permission.SCREEN_CAPTURE].button.click()
    assert cloud.button.isEnabled()
    cloud.button.click()
    assert vfacade.permissions.is_granted(Permission.CLOUD_VISION)
    asked = len(confirm.asked)
    page.permission_rows[Permission.SCREEN_CAPTURE].button.click()  # revoke: no question
    assert len(confirm.asked) == asked
    assert not vfacade.permissions.is_granted(Permission.SCREEN_CAPTURE)
    assert not vfacade.permissions.is_granted(Permission.CLOUD_VISION)  # dependent goes too
    assert cloud.button.text() == "Дозволити…" and not cloud.button.isEnabled()


def test_permission_rows_follow_changes_made_elsewhere(make_page, vfacade):
    page = make_page()
    vfacade.permissions.grant(Permission.SCREEN_CAPTURE)
    page.refresh()
    assert page.permission_rows[Permission.SCREEN_CAPTURE].button.text() == "Відкликати"
    vfacade.permissions.stop_all_observation()
    page.refresh()
    assert page.permission_rows[Permission.SCREEN_CAPTURE].button.text() == "Дозволити…"


def test_unavailable_permission_cannot_be_granted(make_page, monkeypatch):
    monkeypatch.setattr(permissions_module, "AVAILABLE", frozenset({Permission.APP_METADATA}))
    page = make_page(Confirm(True))
    row = page.permission_rows[Permission.SCREEN_CAPTURE]
    assert not row.button.isEnabled() and row.state_label.text() == "недоступно в цій версії"


# --- status and "Що Мія бачила" ------------------------------------------------------------------------


@pytest.mark.parametrize(("view", "fragment", "role"), [
    (VisionStatusView("off", "off", "", "", None), "Вимкнено", "hint"),
    (VisionStatusView("game", "off", "", "Потрібен дозвіл", True), "Не працює: Потрібен дозвіл", "warning"),
    (VisionStatusView("game", "watching", TITLE, "", True), f"Дивиться на «{TITLE}» · аналізує локальна",
     "success"),
    (VisionStatusView("auto", "analyzing", TITLE, "", False), "Роздивляється", "success"),
    (VisionStatusView("auto", "paused", "", "", False), "На паузі", "warning"),
    (VisionStatusView("manual", "idle", "", "", True), "попросиш", "hint"),
    (VisionStatusView("auto", "idle", "", "", True), "немає дозволеної", "hint"),
])
def test_describe_status(view, fragment, role):
    text, actual_role = describe_status(view)
    assert fragment in text and actual_role == role


def test_status_card_shows_the_note(make_page, vfacade):
    vfacade.vision_view = VisionStatusView("game", "idle", "", "", True, note="Список порожній.")
    page = make_page()
    assert "Чекає" in page.status_label.text()
    assert page.note_label.text() == "Список порожній." and not page.note_label.isHidden()


def test_seen_list_and_clear(make_page, vfacade):
    vfacade.recent = [
        SeenObservation(TITLE, "gameplay", "boss_fight", 0.9, 1_700_000_000.0, True),
        SeenObservation(TITLE, "menu", "none", 0.8, 1_699_999_900.0, False),
    ]
    page = make_page()
    lines = [page.seen_list.item(i).text() for i in range(page.seen_list.count())]
    assert len(lines) == 2
    assert lines[0].endswith(f"{TITLE}: гра, бій із босом · відреагувала")
    assert lines[1].endswith(f"{TITLE}: меню")
    assert page.clear_seen_button.isEnabled()
    page.clear_seen_button.click()
    assert vfacade.cleared == 1
    assert [page.seen_list.item(i).text() for i in range(page.seen_list.count())] == ["Поки що нічого."]
    assert not page.clear_seen_button.isEnabled()


def test_describe_seen_without_title():
    item = SeenObservation("", "loading", "none", 0.5, 0.0, False)
    assert describe_seen(item).endswith("вікно: завантаження")


def test_seen_header_uses_her_name(make_page, vfacade):
    s = vfacade.settings()
    s.character.name = "Ліна"
    vfacade.apply_settings(s)
    page = make_page()
    assert page.seen_header.text() == "Що Ліна бачила"


def test_privacy_promises_are_on_the_page(make_page):
    shown = texts(make_page())
    assert "ніколи не зберігаються" in shown
    assert "не знімаються ніколи" in shown
    assert "Вікно без рамки" in shown  # exclusive fullscreen hint
    assert "Хмарний аналіз екрана" in shown  # the separate cloud permission is named
    assert "Лише в пам'яті" in shown


def test_texts_are_gender_neutral_towards_the_user(make_page):
    page = make_page()
    everything = texts(page) + "\n".join(edit.placeholderText() for edit in page.findChildren(QLineEdit))
    assert not GENDERED_USER.search(everything)
    assert not re.search(r"\bти\s+сам\b", everything)


def test_status_failure_is_shown_not_raised(make_page, vfacade):
    page = make_page()

    def broken():
        raise RuntimeError("boom")

    vfacade.vision_status = broken  # type: ignore[method-assign]
    page.refresh_live()
    assert "невідомий" in page.status_label.text()


def test_refresh_timer_runs_only_while_visible(make_page):
    page = make_page()
    assert page.refresh_timer.isActive()
    page.hide()
    assert not page.refresh_timer.isActive()


def test_helpers():
    assert interval_text(10) == "10\u00a0с"
    assert interval_text(120) == "2\u00a0хв"
    assert interval_text(150) == "2\u00a0хв 30\u00a0с"
    assert game_title_for("custom:my_game.exe") == "My game"
    assert game_title_for("weird") == "weird"
