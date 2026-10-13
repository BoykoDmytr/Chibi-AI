"""The "Ігровий компаньйон" settings page: Settings.gaming fields with the
dialog's baseline rule, the frequency wording, and no duplicate of the games
list that lives on the "Ігри" page."""

from __future__ import annotations

import re

from PySide6.QtWidgets import QLabel, QLineEdit, QTableWidget

from chibi.settings.gaming_page import GamingPage, frequency_text, rate_text, reactions_per_hour
from chibi.vision import integration
from tests.personality.test_phrase_bank import GENDERED_USER


def make(facade, track) -> GamingPage:
    page = track(GamingPage(facade))
    page.show()
    return page


def texts(page: GamingPage) -> str:
    return "\n".join(label.text() for label in page.findChildren(QLabel))


def test_defaults_are_shown(facade, track):
    page = make(facade, track)
    assert page.companion_check.isChecked()
    assert page.frequency_row.value() == 50 and page.frequency_row.value_label.text() == "помірно"
    assert not page.spoilers_check.isChecked()
    assert page.rate_label.text().startswith("Не більше ніж 6 реакцій на годину")
    assert not page.has_edits()


def test_edits_are_collected(facade, track):
    page = make(facade, track)
    page.companion_check.setChecked(False)
    page.frequency_row.setValue(100)
    page.spoilers_check.setChecked(True)
    g = page.collect(facade.settings()).gaming
    assert (g.companion_mode, g.reaction_frequency, g.hints_and_spoilers) == (False, 100, True)
    assert "12 реакцій" in page.rate_label.text()


def test_unedited_fields_never_revert_changes_made_elsewhere(facade, track):
    page = make(facade, track)
    s = facade.settings()
    s.gaming.hints_and_spoilers = True
    facade.apply_settings(s)
    page.frequency_row.setValue(20)
    g = page.collect(facade.settings()).gaming
    assert (g.hints_and_spoilers, g.reaction_frequency) == (True, 20)


def test_sync_and_baseline(facade, track):
    page = make(facade, track)
    page.frequency_row.setValue(80)
    s = facade.settings()
    s.gaming.companion_mode = False
    s.gaming.reaction_frequency = 10
    page.sync(s, edited_too=False)
    assert not page.companion_check.isChecked() and page.frequency_row.value() == 80
    page.sync(s, edited_too=True)
    assert page.frequency_row.value() == 10 and not page.has_edits()
    page.frequency_row.setValue(30)
    page.reset_baseline()
    assert not page.has_edits()


def test_zero_frequency_means_silence(facade, track):
    page = make(facade, track)
    page.frequency_row.setValue(0)
    assert page.frequency_row.value_label.text() == "мовчить"
    assert "мовчатиме" in page.rate_label.text()


def test_frequency_words():
    assert [frequency_text(v) for v in (0, 5, 29, 30, 59, 60, 84, 85, 100)] == [
        "мовчить", "тихо", "тихо", "помірно", "помірно", "охоче", "охоче", "балакуче", "балакуче"]
    assert "1 реакція" in rate_text(10) and "3 реакції" in rate_text(25)


def test_rate_mirrors_the_vision_integration():
    assert all(reactions_per_hour(v) == integration.reactions_per_hour(v) for v in range(-10, 111))


def test_the_games_list_is_not_duplicated_here(facade, track):
    page = make(facade, track)
    assert page.findChildren(QTableWidget) == [] and page.findChildren(QLineEdit) == []
    shown = texts(page)
    assert "«Ігри» → «Мої ігри»" in shown and "«Бачення екрана»" in shown


def test_name_and_neutral_wording(facade, track):
    s = facade.settings()
    s.character.name = "Ліна"
    facade.apply_settings(s)
    page = make(facade, track)
    shown = texts(page)
    assert "Як Ліна поводиться" in page.lead_label.text()
    assert not GENDERED_USER.search(shown) and not re.search(r"\bти\s+сам\b", shown)
    for value in (0, 50):
        assert not re.search(r"\bти\s+сам\b", rate_text(value))
