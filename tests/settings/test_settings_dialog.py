from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QLabel, QLineEdit

from chibi.core.events import Permission
from chibi.memory.repositories import GameTotals
from chibi.settings import settings_dialog
from chibi.settings.settings_dialog import (
    PAGES,
    PERMISSION_TEXTS,
    SettingsDialog,
    normalize_process_name,
)

UNAVAILABLE = (Permission.SCREEN_CAPTURE, Permission.CLOUD_VISION, Permission.CLOUD_AI, Permission.MICROPHONE)


class Confirm:
    """Scripted stand-in for the consent / are-you-sure dialog."""

    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.asked: list[tuple[str, str]] = []

    def __call__(self, title: str, text: str) -> bool:
        self.asked.append((title, text))
        return self.answer


@pytest.fixture
def make_dialog(facade, track):
    def make(confirm: Confirm | None = None) -> SettingsDialog:
        dialog = track(SettingsDialog(facade, confirm=confirm or Confirm(False)))
        dialog.show()
        return dialog

    return make


def labels_text(widget) -> str:
    return "\n".join(label.text() for label in widget.findChildren(QLabel))


# --- construction / navigation ------------------------------------------------------------


def test_constructs_shows_every_page_and_fits_860x620(make_dialog):
    dialog = make_dialog()
    assert dialog.isVisible()
    assert [dialog.nav.item(i).text() for i in range(dialog.nav.count())] == [t for _, t in PAGES]
    for key, _ in PAGES:
        dialog.show_page(key)
        assert dialog.current_page() == key
    hint = dialog.minimumSizeHint()
    assert hint.width() <= 860 and hint.height() <= 620
    dialog.resize(860, 620)
    assert dialog.width() == 860 and dialog.height() == 620
    assert dialog.isSizeGripEnabled()
    dialog.resize(1100, 800)  # resizable both ways
    assert dialog.width() == 1100 and dialog.height() == 800


def test_buttons_are_ukrainian(make_dialog):
    dialog = make_dialog()
    assert (dialog.cancel_button.text(), dialog.apply_button.text(), dialog.ok_button.text()) == (
        "Скасувати", "Застосувати", "Гаразд")


# --- applying settings -------------------------------------------------------------------------


def test_apply_passes_edited_copy_to_facade(make_dialog, facade):
    dialog = make_dialog()
    dialog.name_edit.setText("  Ліна  ")
    dialog.nickname_edit.setText(" Сашко ")
    dialog.scale_row.setValue(150)
    dialog.renderer_combo.setCurrentIndex(dialog.renderer_combo.findData("vector"))
    dialog.always_on_top_check.setChecked(False)
    dialog.lock_position_check.setChecked(True)
    dialog.walking_check.setChecked(False)
    dialog.gravity_check.setChecked(False)
    dialog.click_through_check.setChecked(True)
    dialog.bubbles_check.setChecked(False)
    dialog.personality_rows["humor"].setValue(10)
    dialog.personality_rows["commentary_frequency"].setValue(95)
    dialog.dnd_check.setChecked(True)
    dialog.max_per_hour_spin.setValue(3)
    dialog.min_gap_spin.setValue(300)
    dialog.quiet_minutes_spin.setValue(30)
    dialog.hide_fullscreen_check.setChecked(False)
    dialog.eco_check.setChecked(False)
    dialog.sleep_after_spin.setValue(45)
    dialog.comment_checks["comment_on_browsing"].setChecked(True)
    dialog.comment_checks["comment_on_games"].setChecked(False)

    dialog.apply_button.click()

    assert len(facade.applied) == 1
    s = facade.applied[0]
    ch = s.character
    assert (ch.name, ch.user_nickname, ch.scale, ch.renderer) == ("Ліна", "Сашко", 1.5, "vector")
    assert (ch.always_on_top, ch.lock_position, ch.walking_enabled, ch.gravity, ch.click_through,
            ch.show_bubbles) == (False, True, False, False, True, False)
    assert s.personality.humor == 10 and s.personality.commentary_frequency == 95
    assert s.personality.sarcasm == 40  # untouched slider keeps its value
    b = s.behavior
    assert (b.do_not_disturb, b.max_unsolicited_per_hour, b.min_seconds_between_unsolicited,
            b.quiet_minutes, b.hide_on_fullscreen, b.auto_eco_mode, b.sleep_after_idle_minutes) == (
        True, 3, 300, 30, False, False, 45)
    assert s.activity.comment_on_browsing is True and s.activity.comment_on_games is False
    assert dialog.isVisible()  # Apply keeps the dialog open
    assert "Збережено" in dialog.status_label.text()


def test_ok_applies_and_closes(make_dialog, facade):
    dialog = make_dialog()
    dialog.name_edit.setText("Ліна")
    dialog.ok_button.click()
    assert [s.character.name for s in facade.applied] == ["Ліна"]
    assert not dialog.isVisible()
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_cancel_applies_nothing(make_dialog, facade):
    dialog = make_dialog()
    dialog.name_edit.setText("Ліна")
    dialog.walking_check.setChecked(False)
    dialog.cancel_button.click()
    assert facade.applied == []
    assert facade.settings().character.name == "Мія"
    assert dialog.result() == QDialog.DialogCode.Rejected


def test_dialog_works_on_a_copy(make_dialog, facade):
    dialog = make_dialog()
    dialog.name_edit.setText("Ліна")
    dialog.personality_rows["humor"].setValue(1)
    assert facade.settings().character.name == "Мія"
    assert facade.settings().personality.humor == 70


@pytest.mark.parametrize("bad", ["", "   ", "\t"])
def test_empty_name_is_rejected_with_a_message(make_dialog, facade, bad):
    dialog = make_dialog()
    dialog.show_page("privacy")
    dialog.name_edit.setText(bad)
    dialog.ok_button.click()
    assert facade.applied == []
    assert dialog.isVisible()
    assert not dialog.name_error.isHidden()
    assert "порожнім" in dialog.name_error.text()
    assert dialog.current_page() == "character"
    # fixing the name clears the error and applies
    dialog.name_edit.setText("Ліна")
    assert dialog.apply() is True
    assert dialog.name_error.isHidden()
    assert facade.applied[-1].character.name == "Ліна"


def test_apply_keeps_fields_the_dialog_does_not_edit(make_dialog, facade):
    dialog = make_dialog()
    # the app changes something meanwhile that this dialog has no widget for
    changed = facade.settings()
    changed.first_run_completed = True
    changed.activity.poll_interval_seconds = 7.0
    facade.apply_settings(changed)
    facade.applied.clear()
    dialog.apply_button.click()
    assert facade.applied[0].first_run_completed is True
    assert facade.applied[0].activity.poll_interval_seconds == 7.0


def test_failing_apply_reports_error_and_keeps_dialog_open(make_dialog, facade):
    def broken(_settings):
        raise RuntimeError("disk full")

    facade.apply_settings = broken
    dialog = make_dialog()
    dialog.ok_button.click()
    assert dialog.isVisible()
    assert "Не вдалося" in dialog.status_label.text()


def test_personality_page_is_honest_about_phase_1(make_dialog):
    dialog = make_dialog()
    text = labels_text(dialog.stack.widget(1))
    assert "фаза 1" in text and "фазі 2" in text
    titles = [row.title_label.text() for row in dialog.personality_rows.values()]
    assert titles == ["Гумор", "Сарказм", "Балакучість", "Цікавість", "Прив'язаність", "Незалежність",
                      "Частота коментарів"]


def test_ai_page_has_no_fake_inputs(make_dialog):
    dialog = make_dialog()
    page = dialog.stack.widget([k for k, _ in PAGES].index("ai"))
    assert page.findChildren(QLineEdit) == []
    text = labels_text(page)
    for word in ("OpenAI", "Anthropic", "Ollama", "Фаза 2", "Фаза 3", "Фаза 4", "офлайн"):
        assert word in text


# --- permissions -----------------------------------------------------------------------------


def test_granting_app_metadata_requires_consent(make_dialog, permissions):
    confirm = Confirm(True)
    dialog = make_dialog(confirm)
    box = dialog.permission_checks[Permission.APP_METADATA]
    box.click()
    assert len(confirm.asked) == 1
    title, text = confirm.asked[0]
    assert title and "назву процесу" in text
    assert permissions.is_granted(Permission.APP_METADATA)
    assert box.isChecked()


def test_declining_consent_leaves_permission_off(make_dialog, permissions):
    confirm = Confirm(False)
    dialog = make_dialog(confirm)
    box = dialog.permission_checks[Permission.APP_METADATA]
    box.click()
    assert len(confirm.asked) == 1
    assert not permissions.is_granted(Permission.APP_METADATA)
    assert not box.isChecked()


def test_revoking_is_immediate_and_asks_nothing(make_dialog, permissions):
    permissions.grant(Permission.APP_METADATA)
    confirm = Confirm(False)
    dialog = make_dialog(confirm)
    box = dialog.permission_checks[Permission.APP_METADATA]
    assert box.isChecked()
    box.click()
    assert confirm.asked == []
    assert not permissions.is_granted(Permission.APP_METADATA)


def test_window_titles_disabled_until_app_metadata_granted(make_dialog, permissions):
    dialog = make_dialog(Confirm(True))
    titles = dialog.permission_checks[Permission.WINDOW_TITLES]
    assert not titles.isEnabled()
    dialog.permission_checks[Permission.APP_METADATA].click()
    assert titles.isEnabled()
    titles.click()
    assert permissions.is_granted(Permission.WINDOW_TITLES)
    # revoking the base revokes and disables the dependent
    dialog.permission_checks[Permission.APP_METADATA].click()
    assert not permissions.is_granted(Permission.WINDOW_TITLES)
    assert not titles.isChecked() and not titles.isEnabled()


def test_unavailable_permissions_are_shown_disabled_with_their_phase(make_dialog):
    dialog = make_dialog(Confirm(True))
    for permission in UNAVAILABLE:
        box = dialog.permission_checks[permission]
        assert not box.isEnabled() and not box.isChecked()
    assert {p: PERMISSION_TEXTS[p].phase for p in UNAVAILABLE} == {
        Permission.SCREEN_CAPTURE: 3, Permission.CLOUD_VISION: 3, Permission.CLOUD_AI: 2,
        Permission.MICROPHONE: 4}
    text = labels_text(dialog)
    for phase in (2, 3, 4):
        assert f"з'явиться у фазі {phase}" in text


def test_kill_switch_revokes_everything(make_dialog, facade, permissions):
    permissions.grant(Permission.APP_METADATA)
    permissions.grant(Permission.WINDOW_TITLES)
    dialog = make_dialog()
    assert dialog.kill_switch_button.isEnabled()
    dialog.kill_switch_button.click()
    assert "stop_all_observation" in facade.calls
    assert not permissions.any_observation_active()
    assert not any(box.isChecked() for box in dialog.permission_checks.values())
    # stays clickable: a kill switch must never look broken, even with nothing to stop
    assert dialog.kill_switch_button.isEnabled()
    dialog.kill_switch_button.click()
    assert facade.calls.count("stop_all_observation") == 2
    assert "зупинено" in dialog.status_label.text()


def test_permission_history_is_listed(make_dialog, permissions):
    dialog = make_dialog(Confirm(True))
    assert dialog.history_list.item(0).text() == "Поки що жодних змін."
    dialog.permission_checks[Permission.APP_METADATA].click()
    dialog.permission_checks[Permission.APP_METADATA].click()
    texts = [dialog.history_list.item(i).text() for i in range(dialog.history_list.count())]
    assert len(texts) == 2
    assert "дозвіл відкликано" in texts[0] and "дозвіл надано" in texts[1]  # newest first
    assert all("Назви програм та ігор" in t for t in texts)


def test_unsupported_platform_warning(make_dialog, facade):
    from dataclasses import replace

    facade.status_view = replace(facade.status_view, activity_supported=False)
    dialog = make_dialog()
    assert not dialog.activity_unsupported_label.isHidden()


# --- data actions ----------------------------------------------------------------------------


def test_wipe_history_only_when_confirmed(make_dialog, facade):
    dialog = make_dialog(Confirm(False))
    dialog.wipe_history_button.click()
    assert "wipe_history" not in facade.calls
    dialog2 = make_dialog(Confirm(True))
    dialog2.wipe_history_button.click()
    assert facade.calls.count("wipe_history") == 1
    assert dialog2.isVisible()


def test_wipe_all_only_when_confirmed_then_closes_without_applying(make_dialog, facade):
    declined = make_dialog(Confirm(False))
    declined.wipe_all_button.click()
    assert "wipe_all" not in facade.calls and declined.isVisible()
    confirm = Confirm(True)
    dialog = make_dialog(confirm)
    dialog.name_edit.setText("Ліна")
    dialog.wipe_all_button.click()
    assert facade.calls.count("wipe_all") == 1
    assert "Видалити всі дані" in confirm.asked[0][0]
    assert not dialog.isVisible()
    assert facade.applied == []


def test_data_folder_is_shown_and_opened(make_dialog, facade, monkeypatch):
    opened = []
    monkeypatch.setattr(settings_dialog, "open_folder", lambda path: opened.append(path) or True)
    dialog = make_dialog()
    assert dialog.data_dir_label.text() == str(facade.data_dir())
    dialog.open_data_dir_button.click()
    assert opened == [facade.data_dir()]


# --- exclusions --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("mybank.exe", "mybank.exe"),
        ("  MyBank.EXE ", "mybank.exe"),
        ("Steam", "steam.exe"),
        ("C:\\Program Files\\Bank\\Bank App.exe", "bank app.exe"),
        ("/opt/thing/Tool", "tool.exe"),
        ('"KeePass.exe"', "keepass.exe"),
        ("", None),
        ("   ", None),
        ("a<b.exe", None),
        ("what?", None),
        ("C:\\dir\\", None),
        (".exe", None),
        ("x" * 200, None),
    ],
)
def test_normalize_process_name(raw, expected):
    assert normalize_process_name(raw) == expected


def items(list_widget) -> list[str]:
    return [list_widget.item(i).text() for i in range(list_widget.count())]


def test_exclusions_add_normalized_dedup_and_remove(make_dialog, facade):
    dialog = make_dialog()
    dialog.exclusion_edit.setText("  C:\\Apps\\MyBank.EXE ")
    dialog.add_exclusion_button.click()
    assert items(dialog.exclusions_list) == ["mybank.exe"]
    assert dialog.exclusion_edit.text() == ""
    # Enter adds too - and must not press the dialog's default "Гаразд" button
    dialog.exclusion_edit.setText("Signal")
    QTest.keyClick(dialog.exclusion_edit, Qt.Key.Key_Return)
    assert items(dialog.exclusions_list) == ["mybank.exe", "signal.exe"]
    assert dialog.isVisible() and facade.applied == []
    dialog.exclusion_edit.setText("MYBANK")
    dialog.add_exclusion_button.click()
    assert items(dialog.exclusions_list) == ["mybank.exe", "signal.exe"]
    dialog.exclusion_edit.setText("bad|name")
    dialog.add_exclusion_button.click()
    assert not dialog.exclusion_error.isHidden()
    assert items(dialog.exclusions_list) == ["mybank.exe", "signal.exe"]

    dialog.exclusions_list.item(0).setSelected(True)
    dialog.remove_exclusion_button.click()
    assert items(dialog.exclusions_list) == ["signal.exe"]
    dialog.apply_button.click()
    assert facade.applied[-1].activity.excluded_processes == ["signal.exe"]


def test_existing_exclusions_are_loaded(facade, track):
    s = facade.settings()
    s.activity.excluded_processes = ["bank.exe"]
    facade.apply_settings(s)
    dialog = track(SettingsDialog(facade, confirm=Confirm(False)))
    assert items(dialog.exclusions_list) == ["bank.exe"]


# --- games ---------------------------------------------------------------------------------------


def table_rows(table) -> list[list[str]]:
    return [[table.item(r, c).text() for c in range(table.columnCount())] for r in range(table.rowCount())]


def test_custom_games_add_update_remove(make_dialog, facade):
    dialog = make_dialog()
    dialog.custom_process_edit.setText("MyGame")
    dialog.custom_title_edit.setText("  Моя гра ")
    dialog.add_custom_game_button.click()
    dialog.custom_process_edit.setText("other.exe")
    QTest.keyClick(dialog.custom_process_edit, Qt.Key.Key_Return)
    assert table_rows(dialog.custom_games_table) == [["mygame.exe", "Моя гра"], ["other.exe", "other"]]
    # same process again updates the title instead of duplicating
    dialog.custom_process_edit.setText("mygame.exe")
    dialog.custom_title_edit.setText("Нова назва")
    dialog.add_custom_game_button.click()
    assert table_rows(dialog.custom_games_table)[0] == ["mygame.exe", "Нова назва"]
    # empty process is refused
    dialog.custom_title_edit.setText("Без процесу")
    dialog.add_custom_game_button.click()
    assert not dialog.custom_game_error.isHidden()
    assert dialog.custom_games_table.rowCount() == 2

    dialog.custom_games_table.selectRow(1)
    dialog.remove_custom_game_button.click()
    dialog.apply_button.click()
    assert facade.applied[-1].activity.custom_games == [{"process": "mygame.exe", "title": "Нова назва"}]
    assert dialog.isVisible()


def test_known_games_count_uses_ukrainian_plural(make_dialog, facade):
    facade.known_games = 152
    assert "152 гри" in make_dialog().known_games_label.text()
    facade.known_games = 21
    assert "21 гру" in make_dialog().known_games_label.text()
    facade.known_games = 11
    assert "11 ігор" in make_dialog().known_games_label.text()


def test_played_games_table(make_dialog, facade, permissions):
    permissions.grant(Permission.APP_METADATA)
    facade.totals = [
        GameTotals("stalker2", "S.T.A.L.K.E.R. 2", 3 * 3600 + 5 * 60, 4, 1_700_000_000.0),
        GameTotals("cs2", "Counter-Strike 2", 21 * 60, 1, 1_700_100_000.0),
    ]
    dialog = make_dialog()
    rows = table_rows(dialog.played_table)
    assert [r[:3] for r in rows] == [
        ["S.T.A.L.K.E.R. 2", "3 години 5 хвилин", "4"],
        ["Counter-Strike 2", "21 хвилина", "1"],
    ]
    assert rows[0][3].count(".") == 2  # dd.mm.yyyy
    assert not dialog.played_table.isHidden()
    assert dialog.played_empty_label.isHidden()


def test_played_games_empty_state_when_monitoring_off(make_dialog):
    dialog = make_dialog()
    assert dialog.played_table.isHidden()
    assert "вимкнено" in dialog.played_empty_label.text()
    assert not dialog.monitoring_note.isHidden()


def test_played_games_empty_state_when_monitoring_on(make_dialog, permissions):
    permissions.grant(Permission.APP_METADATA)
    dialog = make_dialog()
    assert "Поки що жодної гри" in dialog.played_empty_label.text()
    assert dialog.monitoring_note.isHidden()


def test_granting_from_privacy_page_updates_games_page(make_dialog):
    dialog = make_dialog(Confirm(True))
    assert not dialog.monitoring_note.isHidden()
    dialog.permission_checks[Permission.APP_METADATA].click()
    assert dialog.monitoring_note.isHidden()
