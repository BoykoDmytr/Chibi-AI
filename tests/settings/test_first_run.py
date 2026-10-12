from __future__ import annotations

import pytest
from PySide6.QtWidgets import QDialog

from chibi.core.events import Permission
from chibi.settings.first_run import FirstRunDialog


@pytest.fixture
def make_dialog(facade, track):
    def make() -> FirstRunDialog:
        dialog = track(FirstRunDialog(facade))
        dialog.show()
        return dialog

    return make


def test_shows_with_consent_unchecked(make_dialog):
    dialog = make_dialog()
    assert dialog.isVisible()
    assert not dialog.consent_check.isChecked()
    assert dialog.consent_check.text() == "Дозволити бачити, які програми та ігри запущені"
    assert dialog.name_edit.text() == "Мія"
    assert dialog.start_button.text() == "Почати"


def test_fits_its_content_but_not_taller_than_the_screen(make_dialog):
    dialog = make_dialog()
    screen = dialog.screen().availableGeometry().height()
    assert dialog.height() <= screen
    assert dialog.height() >= dialog.minimumHeight()


def test_start_without_consent_applies_settings_and_grants_nothing(make_dialog, facade, permissions):
    dialog = make_dialog()
    dialog.name_edit.setText("  Ліна ")
    dialog.nickname_edit.setText(" Сашку ")
    dialog.start_button.click()
    assert len(facade.applied) == 1
    s = facade.applied[0]
    assert s.first_run_completed is True
    assert (s.character.name, s.character.user_nickname) == ("Ліна", "Сашку")
    assert not any(permissions.snapshot().values())
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert not dialog.isVisible()


def test_start_with_consent_grants_only_app_metadata(make_dialog, permissions):
    dialog = make_dialog()
    dialog.consent_check.setChecked(True)
    dialog.start_button.click()
    snapshot = permissions.snapshot()
    assert snapshot[Permission.APP_METADATA] is True
    assert [p for p, granted in snapshot.items() if granted] == [Permission.APP_METADATA]


def test_empty_name_is_rejected(make_dialog, facade, permissions):
    dialog = make_dialog()
    dialog.consent_check.setChecked(True)
    dialog.name_edit.setText("   ")
    assert dialog.start() is False
    assert facade.applied == []
    assert not permissions.is_granted(Permission.APP_METADATA)
    assert not dialog.name_error.isHidden()
    assert dialog.isVisible()
    dialog.name_edit.setText("Ліна")
    assert dialog.name_error.isHidden()
    assert dialog.title_label.text() == "Привіт! Я — Ліна"


def test_failed_save_grants_nothing_and_stays_open(make_dialog, facade, permissions):
    def broken(_settings):
        raise OSError("disk full")

    facade.apply_settings = broken
    dialog = make_dialog()
    dialog.consent_check.setChecked(True)
    assert dialog.start() is False
    assert not permissions.is_granted(Permission.APP_METADATA)
    assert dialog.isVisible()


def test_privacy_explanation_is_present(make_dialog):
    from PySide6.QtWidgets import QLabel

    dialog = make_dialog()
    text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert "За замовчуванням я нічого не відстежую" in text
    assert "назву програми" in text


def test_names_hint_uses_the_vocative(make_dialog):
    from PySide6.QtWidgets import QLabel

    dialog = make_dialog()
    assert dialog.nickname_edit.placeholderText() == "наприклад, Сашку"
    text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert "«Сашку», «Олю», «пане капітане»" in text
    assert "Сашко" not in text


def test_consent_says_exactly_what_is_stored(make_dialog):
    """Consent must name everything a game session row holds, not just its length."""
    from PySide6.QtWidgets import QLabel

    dialog = make_dialog()
    text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert "крім ігрових сесій: яка гра, коли почалася і скільки тривала" in text
    assert "тривалості" not in text
