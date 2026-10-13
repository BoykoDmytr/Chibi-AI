from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QLabel

from chibi.core.events import Permission
from chibi.memory.repositories import GameTotals
from chibi.settings import settings_dialog
from chibi.settings.settings_dialog import (
    PAGES,
    PERMISSION_TEXTS,
    SettingsDialog,
    normalize_process_name,
)

UNAVAILABLE = (Permission.SCREEN_CAPTURE, Permission.CLOUD_VISION, Permission.MICROPHONE)


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
    dialog.nickname_edit.setText(" Сашку ")
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
    assert (ch.name, ch.user_nickname, ch.scale, ch.renderer) == ("Ліна", "Сашку", 1.5, "vector")
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


def change_elsewhere(facade, **changes) -> None:
    """What the tray, her context menu, Ctrl+wheel or the first-run dialog do
    while the (non-modal) settings dialog stays open."""
    s = facade.settings()
    for path, value in changes.items():
        section, name = path.split("__")
        setattr(getattr(s, section), name, value)
    facade.apply_settings(s)
    facade.applied.clear()


def test_ok_keeps_changes_made_elsewhere_while_open(make_dialog, facade):
    dialog = make_dialog()
    change_elsewhere(facade, behavior__do_not_disturb=True, character__lock_position=True,
                     character__click_through=True, character__scale=1.37, character__name="Кіра")
    dialog.nickname_edit.setText("Сашку")  # the only edit in the dialog
    dialog.ok_button.click()
    s = facade.applied[0]
    assert s.character.user_nickname == "Сашку"
    assert (s.behavior.do_not_disturb, s.character.lock_position, s.character.click_through,
            s.character.scale, s.character.name) == (True, True, True, 1.37, "Кіра")


def test_a_second_apply_does_not_revert_changes_made_elsewhere(make_dialog, facade):
    dialog = make_dialog()
    change_elsewhere(facade, behavior__do_not_disturb=True)
    dialog.apply_button.click()
    assert facade.settings().behavior.do_not_disturb is True
    assert dialog.dnd_check.isChecked()  # after Apply the widgets show what the app holds
    change_elsewhere(facade, behavior__do_not_disturb=False, character__lock_position=True)
    dialog.apply_button.click()
    assert facade.settings().behavior.do_not_disturb is False
    assert facade.settings().character.lock_position is True


def test_edits_made_in_the_dialog_still_win(make_dialog, facade):
    dialog = make_dialog()
    dialog.dnd_check.setChecked(True)
    dialog.scale_row.setValue(150)
    change_elsewhere(facade, character__scale=1.2, character__lock_position=True)
    dialog.apply_button.click()
    s = facade.applied[0]
    assert (s.behavior.do_not_disturb, s.character.scale, s.character.lock_position) == (True, 1.5, True)


def test_refresh_shows_changes_made_elsewhere_but_keeps_unsaved_edits(make_dialog, facade):
    dialog = make_dialog()
    dialog.nickname_edit.setText("Олю")  # typed, not applied yet
    change_elsewhere(facade, character__name="Кіра", character__user_nickname="Сашку",
                     behavior__do_not_disturb=True)
    dialog.hide()
    dialog.show()  # showEvent -> refresh()
    assert dialog.name_edit.text() == "Кіра" and dialog.dnd_check.isChecked()
    assert dialog.nickname_edit.text() == "Олю"
    assert "Кіра" in dialog.windowTitle()
    dialog.apply_button.click()
    s = facade.applied[0]
    assert (s.character.name, s.character.user_nickname, s.behavior.do_not_disturb) == ("Кіра", "Олю", True)


def test_click_through_needs_a_tray(make_dialog, facade):
    with_tray = make_dialog()
    assert with_tray.click_through_check.isEnabled()
    assert "треї" in labels_text(with_tray)
    facade.tray = False
    without = make_dialog()
    assert not without.click_through_check.isEnabled()
    assert "немає системного трею" in labels_text(without)


def test_nickname_examples_are_in_the_vocative(make_dialog):
    dialog = make_dialog()
    assert dialog.nickname_edit.placeholderText() == "наприклад, Сашку"
    text = labels_text(dialog)
    assert "«Сашку», «Олю», «пане капітане»" in text
    assert "Сашко" not in text


def test_failing_apply_reports_error_and_keeps_dialog_open(make_dialog, facade):
    def broken(_settings):
        raise RuntimeError("disk full")

    facade.apply_settings = broken
    dialog = make_dialog()
    dialog.ok_button.click()
    assert dialog.isVisible()
    assert "Не вдалося" in dialog.status_label.text()


def test_personality_page_explains_what_the_sliders_steer(make_dialog):
    dialog = make_dialog()
    text = labels_text(dialog.stack.widget(1))
    assert "готових реплік" in text and "сторінка «AI»" in text
    titles = [row.title_label.text() for row in dialog.personality_rows.values()]
    assert titles == ["Гумор", "Сарказм", "Балакучість", "Цікавість", "Прив'язаність", "Незалежність",
                      "Частота коментарів"]


def page_widget(dialog, key: str):
    return dialog.stack.widget([k for k, _ in PAGES].index(key))


def test_ai_page_is_the_real_one_with_later_features_labelled(make_dialog):
    dialog = make_dialog()
    assert [t for k, t in PAGES if k in ("ai", "memory")] == ["AI", "Пам'ять"]
    page = page_widget(dialog, "ai")
    assert page.isAncestorOf(dialog.ai_page)
    text = labels_text(page)
    assert "з'явиться у фазі 3" in text and "з'явиться у фазі 4" in text
    dialog.show_page("ai")
    assert dialog.current_page() == "ai"


def test_ai_settings_are_applied_with_the_dialog(make_dialog, facade):
    dialog = make_dialog()
    dialog.show_page("ai")
    ai = dialog.ai_page
    ai.provider_combo.setCurrentIndex(ai.provider_combo.findData("ollama"))
    ai.model_combo.setEditText("qwen3:8b")
    ai.per_day_spin.setValue(250)
    ai.stream_check.setChecked(False)
    dialog.name_edit.setText("Ліна")
    dialog.apply_button.click()
    s = facade.applied[-1]
    assert (s.ai.provider, s.ai.model, s.ai.max_requests_per_day, s.ai.stream) == (
        "ollama", "qwen3:8b", 250, False)
    assert s.character.name == "Ліна"
    assert not ai.has_edits()  # synced to what the app holds


def test_ai_fields_changed_elsewhere_are_not_reverted(make_dialog, facade):
    dialog = make_dialog()
    change_elsewhere(facade, ai__effort="high", ai__max_requests_per_hour=9)
    dialog.dnd_check.setChecked(True)
    dialog.apply_button.click()
    s = facade.applied[-1]
    assert (s.ai.effort, s.ai.max_requests_per_hour, s.behavior.do_not_disturb) == ("high", 9, True)


def test_refresh_shows_ai_changes_made_elsewhere(make_dialog, facade):
    dialog = make_dialog()
    dialog.ai_page.per_day_spin.setValue(77)  # unsaved edit
    change_elsewhere(facade, ai__max_requests_per_hour=11, ai__max_requests_per_day=5)
    dialog.hide()
    dialog.show()
    assert dialog.ai_page.per_hour_spin.value() == 11
    assert dialog.ai_page.per_day_spin.value() == 77


def test_invalid_ai_address_blocks_apply_and_shows_the_page(make_dialog, facade):
    dialog = make_dialog()
    ai = dialog.ai_page
    ai.provider_combo.setCurrentIndex(ai.provider_combo.findData("openai"))
    ai.openai_url_edit.setText("api.example.com")
    dialog.ok_button.click()
    assert facade.applied == []
    assert dialog.isVisible() and dialog.current_page() == "ai"
    assert "адресу" in dialog.status_label.text()


def test_cancel_discards_ai_edits(make_dialog, facade):
    dialog = make_dialog()
    dialog.ai_page.provider_combo.setCurrentIndex(dialog.ai_page.provider_combo.findData("anthropic"))
    dialog.cancel_button.click()
    assert facade.applied == []
    assert facade.settings().ai.provider == "none"


def test_ai_check_applies_only_the_ai_page(make_dialog, facade):
    dialog = make_dialog()
    dialog.name_edit.setText("Ліна")  # pending on another page
    ai = dialog.ai_page
    ai.provider_combo.setCurrentIndex(ai.provider_combo.findData("ollama"))
    ai.test_button.click()
    assert facade.settings().ai.provider == "ollama"
    assert facade.settings().character.name == "Мія"  # other pages still wait for Apply
    assert "AI збережено" in dialog.status_label.text()
    dialog.cancel_button.click()
    assert facade.settings().character.name == "Мія"


def test_granting_cloud_ai_on_the_ai_page_updates_the_privacy_page(make_dialog, permissions):
    dialog = make_dialog(Confirm(True))
    ai = dialog.ai_page
    ai.provider_combo.setCurrentIndex(ai.provider_combo.findData("anthropic"))
    ai.allow_cloud_button.click()
    assert permissions.is_granted(Permission.CLOUD_AI)
    assert dialog.permission_checks[Permission.CLOUD_AI].isChecked()
    texts = [dialog.history_list.item(i).text() for i in range(dialog.history_list.count())]
    assert "Хмарний AI: дозвіл надано" in texts[0]


def test_revoking_cloud_ai_on_the_privacy_page_updates_the_ai_page(make_dialog, permissions):
    permissions.grant(Permission.CLOUD_AI)
    dialog = make_dialog()
    ai = dialog.ai_page
    ai.provider_combo.setCurrentIndex(ai.provider_combo.findData("anthropic"))
    assert ai.cloud_banner.isHidden()
    dialog.permission_checks[Permission.CLOUD_AI].click()  # revoke: immediate, no question
    assert not permissions.is_granted(Permission.CLOUD_AI)
    assert not ai.cloud_banner.isHidden()


def test_memory_page_embeds_the_viewer(make_dialog, facade):
    facade.seed_memory("fact", "Кота звати Мурчик")
    dialog = make_dialog()
    dialog.show_page("memory")
    viewer = dialog.memory_viewer
    assert page_widget(dialog, "memory").isAncestorOf(viewer)
    assert not viewer.isWindow()
    assert viewer.table.rowCount() == 1
    facade.seed_memory("joke", "Жарт")
    dialog.show_page("ai")
    dialog.show_page("memory")  # refreshed every time the page is shown
    assert viewer.table.rowCount() == 2


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
        Permission.SCREEN_CAPTURE: 3, Permission.CLOUD_VISION: 3, Permission.MICROPHONE: 4}
    text = labels_text(page_widget(dialog, "privacy"))
    for phase in (3, 4):
        assert f"з'явиться у фазі {phase}" in text
    assert "з'явиться у фазі 2" not in text
    # cloud AI exists now: a real, enabled toggle
    assert PERMISSION_TEXTS[Permission.CLOUD_AI].phase is None
    assert dialog.permission_checks[Permission.CLOUD_AI].isEnabled()


def test_granting_cloud_ai_requires_explicit_consent(make_dialog, permissions):
    declined = Confirm(False)
    dialog = make_dialog(declined)
    box = dialog.permission_checks[Permission.CLOUD_AI]
    box.click()
    assert not permissions.is_granted(Permission.CLOUD_AI) and not box.isChecked()
    title, text = declined.asked[0]
    assert title == "Дозволити хмарний AI?"
    for part in ("пишеш у чаті", "настрій і потреби", "назву активної програми", "спогади",
                 "Знімки екрана", "політикою конфіденційності"):
        assert part in text
    dialog = make_dialog(Confirm(True))
    dialog.permission_checks[Permission.CLOUD_AI].click()
    assert permissions.is_granted(Permission.CLOUD_AI)


def test_kill_switch_leaves_cloud_ai_alone_and_says_so(make_dialog, permissions):
    permissions.grant(Permission.CLOUD_AI)
    dialog = make_dialog()
    dialog.kill_switch_button.click()
    assert permissions.is_granted(Permission.CLOUD_AI)
    assert "Хмарний AI — не спостереження" in labels_text(page_widget(dialog, "privacy"))


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


def test_wipe_texts_say_what_goes_and_what_stays(make_dialog, facade):
    confirm = Confirm(False)
    dialog = make_dialog(confirm)
    dialog.wipe_history_button.click()
    dialog.wipe_all_button.click()
    (_, history), (_, everything) = confirm.asked
    # "delete history" keeps the permission audit log; only "delete all" clears it
    assert "ігрові сесії та сказані репліки" in history
    assert "дозволи та історія їхніх змін" in history and "залишаться" in history
    for item in ("дозволи та історію їхніх змін", "журнали", "розмову в чаті"):
        assert item in everything


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


def test_ok_saves_a_typed_but_unsaved_key(make_dialog, facade):
    dialog = make_dialog()
    ai = dialog.ai_page
    ai.provider_combo.setCurrentIndex(ai.provider_combo.findData("anthropic"))
    ai.key_edit.setText("sk-ant-12345678")
    dialog.ok_button.click()
    assert facade.keys == {"anthropic": "sk-ant-12345678"}
    assert facade.applied[-1].ai.provider == "anthropic"
    assert not dialog.isVisible()


def test_bad_typed_key_blocks_ok(make_dialog, facade):
    dialog = make_dialog()
    ai = dialog.ai_page
    ai.provider_combo.setCurrentIndex(ai.provider_combo.findData("anthropic"))
    ai.key_edit.setText("two words")
    dialog.ok_button.click()
    assert facade.applied == [] and facade.keys == {}
    assert dialog.isVisible() and dialog.current_page() == "ai"
