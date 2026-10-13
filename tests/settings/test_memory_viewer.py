from __future__ import annotations

import logging

import pytest
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtTest import QTest

from chibi.settings.facade import MEMORY_KINDS
from chibi.settings.memory_viewer import (
    COL_KIND,
    COL_PIN,
    COL_SOURCE,
    COL_TEXT,
    KIND_NAMES,
    KIND_TABS,
    NOT_STORED_MESSAGE,
    PRIVACY_NOTE,
    MemoryViewer,
)


class Confirm:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.asked: list[tuple[str, str]] = []

    def __call__(self, title: str, text: str) -> bool:
        self.asked.append((title, text))
        return self.answer


@pytest.fixture
def make_viewer(facade, track):
    def make(confirm: Confirm | None = None) -> MemoryViewer:
        viewer = track(MemoryViewer(facade, confirm=confirm or Confirm(False)))
        viewer.show()
        return viewer

    return make


@pytest.fixture
def seeded(facade):
    facade.seed_memory("fact", "Кота звати Мурчик")
    facade.seed_memory("preference", "Любить каву без цукру")
    facade.seed_memory("game_event", "Пройшов S.T.A.L.K.E.R. 2", source="game")
    facade.seed_memory("joke", "Жарт про борщ і дракона", source="user", pinned=True)
    return facade


def texts(viewer: MemoryViewer) -> list[str]:
    return [viewer.table.item(r, COL_TEXT).text() for r in range(viewer.table.rowCount())]


def select_text(viewer: MemoryViewer, text: str) -> None:
    viewer.table.clearSelection()
    viewer.table.selectRow(texts(viewer).index(text))


def test_tabs_and_kinds_are_in_ukrainian(make_viewer):
    viewer = make_viewer()
    assert [viewer.tabs.tabText(i) for i in range(viewer.tabs.count())] == [t for _, t in KIND_TABS]
    assert [key for key, _ in KIND_TABS[1:]] == list(MEMORY_KINDS)
    assert set(KIND_NAMES) == set(MEMORY_KINDS)
    assert viewer.current_kind() is None


def test_privacy_note_is_shown(make_viewer):
    viewer = make_viewer()
    assert viewer.note_label.text() == PRIVACY_NOTE
    assert "лише на цьому комп'ютері" in PRIVACY_NOTE and "Нічого з екрана" in PRIVACY_NOTE


def test_lists_everything_pinned_first_with_kind_source_and_date(make_viewer, seeded):
    viewer = make_viewer()
    assert texts(viewer)[0] == "Жарт про борщ і дракона"  # pinned first
    assert len(texts(viewer)) == 4
    row = texts(viewer).index("Пройшов S.T.A.L.K.E.R. 2")
    assert viewer.table.item(row, COL_KIND).text() == "Ігрова подія"
    assert viewer.table.item(row, COL_SOURCE).text() == "З гри"
    assert viewer.table.item(0, COL_PIN).text() == "📌"
    assert viewer.table.item(row, COL_PIN).text() == ""
    assert viewer.count_label.text() == "Показано 4 спогади."


def test_filter_by_kind_tab(make_viewer, seeded):
    viewer = make_viewer()
    viewer.show_kind("fact")
    assert viewer.current_kind() == "fact"
    assert texts(viewer) == ["Кота звати Мурчик"]
    assert viewer.add_kind_combo.currentData() == "fact"  # adding defaults to the open tab
    viewer.show_kind("relationship")
    assert texts(viewer) == []
    assert viewer.table.isHidden()
    assert viewer.empty_label.text() == "У цьому розділі поки порожньо."


def test_search_filters_through_the_facade(make_viewer, seeded):
    viewer = make_viewer()
    viewer.search_edit.setText("мурчик")
    assert texts(viewer) == ["Кота звати Мурчик"]
    viewer.search_edit.setText("зовсім не те")
    assert texts(viewer) == [] and viewer.empty_label.text() == "Нічого не знайдено."
    viewer.search_edit.clear()
    assert len(texts(viewer)) == 4


def test_empty_state_explains_how_memories_appear(make_viewer):
    viewer = make_viewer()
    assert viewer.table.isHidden()
    assert "Мія поки нічого про тебе не пам'ятає" in viewer.empty_label.text()
    assert not viewer.clear_button.isEnabled()


def test_remember_facts_off_is_explained(make_viewer, facade):
    s = facade.settings()
    s.ai.remember_facts = False
    facade.apply_settings(s)
    viewer = make_viewer()
    assert not viewer.remember_off_label.isHidden()


def test_add_memory(make_viewer, facade):
    viewer = make_viewer()
    viewer.add_kind_combo.setCurrentIndex(viewer.add_kind_combo.findData("preference"))
    viewer.add_edit.setText("  Любить джаз  ")
    viewer.add_button.click()
    assert facade.memory_calls == [("add", "preference", "Любить джаз")]
    assert texts(viewer) == ["Любить джаз"]
    assert viewer.add_edit.text() == ""
    assert viewer.message_label.text() == "Запам'ятала ✓"
    # Enter adds too
    viewer.add_edit.setText("Має собаку")
    QTest.keyClick(viewer.add_edit, Qt.Key.Key_Return)
    assert "Має собаку" in texts(viewer)


def test_adding_to_another_kind_switches_to_its_tab(make_viewer, facade):
    viewer = make_viewer()
    viewer.show_kind("fact")
    viewer.add_kind_combo.setCurrentIndex(viewer.add_kind_combo.findData("joke"))
    viewer.add_edit.setText("Смішний жарт")
    viewer.add_button.click()
    assert viewer.current_kind() == "joke"
    assert texts(viewer) == ["Смішний жарт"]


def test_add_refused_as_personal_data_tells_the_user(make_viewer, facade):
    viewer = make_viewer()
    viewer.add_edit.setText("мій пароль 1234")
    viewer.add_button.click()
    assert texts(viewer) == []
    assert viewer.message_label.text() == NOT_STORED_MESSAGE
    assert viewer.message_label.property("role") == "error"
    assert viewer.add_edit.text() == "мій пароль 1234"  # kept so it can be fixed


def test_blank_add_is_ignored(make_viewer, facade):
    viewer = make_viewer()
    viewer.add_edit.setText("   ")
    viewer.add_button.click()
    assert facade.memory_calls == []


def test_edit_inline(make_viewer, seeded):
    viewer = make_viewer()
    row = texts(viewer).index("Кота звати Мурчик")
    viewer.table.item(row, COL_TEXT).setText("Кота звати Мурчик, він рудий")
    QCoreApplication.processEvents()
    assert ("update", 1, "Кота звати Мурчик, він рудий") in seeded.memory_calls
    assert "Кота звати Мурчик, він рудий" in texts(viewer)
    assert viewer.message_label.text() == "Змінено ✓"
    # only the text column is editable
    flags = viewer.table.item(row, COL_KIND).flags()
    assert not flags & Qt.ItemFlag.ItemIsEditable
    assert viewer.table.item(row, COL_TEXT).flags() & Qt.ItemFlag.ItemIsEditable


def test_edit_button_opens_the_inline_editor(make_viewer, seeded):
    viewer = make_viewer()
    assert not viewer.edit_button.isEnabled()
    select_text(viewer, "Кота звати Мурчик")
    assert viewer.edit_button.isEnabled()
    viewer.edit_button.click()
    assert viewer.table.state() == viewer.table.State.EditingState


def test_refused_edit_shows_why_and_restores_the_text(make_viewer, seeded):
    viewer = make_viewer()
    row = texts(viewer).index("Кота звати Мурчик")
    viewer.table.item(row, COL_TEXT).setText("мій пароль qwerty")
    QCoreApplication.processEvents()
    assert "особисті дані" in viewer.message_label.text()
    assert "Кота звати Мурчик" in texts(viewer)


def test_emptied_edit_is_not_saved(make_viewer, seeded):
    viewer = make_viewer()
    row = texts(viewer).index("Кота звати Мурчик")
    viewer.table.item(row, COL_TEXT).setText("   ")
    QCoreApplication.processEvents()
    assert not [c for c in seeded.memory_calls if c[0] == "update"]
    assert "Порожній" in viewer.message_label.text()
    assert "Кота звати Мурчик" in texts(viewer)


def test_pin_and_unpin(make_viewer, seeded):
    viewer = make_viewer()
    select_text(viewer, "Любить каву без цукру")
    assert viewer.pin_button.text() == "Закріпити"
    viewer.pin_button.click()
    assert ("pin", 2, True) in seeded.memory_calls
    assert texts(viewer)[:2] == ["Жарт про борщ і дракона", "Любить каву без цукру"]  # pinned first
    select_text(viewer, "Любить каву без цукру")
    assert viewer.pin_button.text() == "Відкріпити"
    viewer.pin_button.click()
    assert ("pin", 2, False) in seeded.memory_calls


def test_delete_asks_first(make_viewer, seeded):
    declined = Confirm(False)
    viewer = make_viewer(declined)
    select_text(viewer, "Кота звати Мурчик")
    viewer.delete_button.click()
    assert declined.asked and "Кота звати Мурчик" in declined.asked[0][1]
    assert len(texts(viewer)) == 4
    confirm = Confirm(True)
    viewer = make_viewer(confirm)
    select_text(viewer, "Кота звати Мурчик")
    viewer.delete_button.click()
    assert "Кота звати Мурчик" not in texts(viewer)
    assert confirm.asked[0][0] == "Забути спогад?"


def test_delete_several(make_viewer, seeded):
    confirm = Confirm(True)
    viewer = make_viewer(confirm)
    viewer.table.selectAll()
    viewer.delete_button.click()
    assert confirm.asked[0][0] == "Забути 4 спогади?"
    assert texts(viewer) == []


def test_clear_this_kind_with_confirm(make_viewer, seeded):
    declined = Confirm(False)
    viewer = make_viewer(declined)
    viewer.show_kind("fact")
    assert viewer.clear_button.text() == "Забути весь розділ"
    viewer.clear_button.click()
    assert "«Факти»" in declined.asked[0][0]
    assert texts(viewer) == ["Кота звати Мурчик"]
    confirm = Confirm(True)
    viewer = make_viewer(confirm)
    viewer.show_kind("fact")
    viewer.clear_button.click()
    assert ("clear", "fact") in seeded.memory_calls
    assert texts(viewer) == []
    viewer.show_kind(None)
    assert len(texts(viewer)) == 3  # other kinds untouched


def test_clear_all_with_confirm(make_viewer, seeded):
    confirm = Confirm(True)
    viewer = make_viewer(confirm)
    assert viewer.clear_button.text() == "Забути все"
    viewer.clear_button.click()
    assert confirm.asked[0][0] == "Забути все?"
    assert "закріплені" in confirm.asked[0][1]
    assert ("clear", None) in seeded.memory_calls
    assert texts(viewer) == []


def test_refreshes_on_show(make_viewer, facade):
    viewer = make_viewer()
    assert texts(viewer) == []
    facade.seed_memory("fact", "Народився у Львові")
    viewer.hide()
    viewer.show()
    assert texts(viewer) == ["Народився у Львові"]


def test_memory_text_is_plain_and_never_logged(make_viewer, facade, caplog):
    secret_like = "<b>Таємниця</b> про <i>подарунок</i>"
    facade.seed_memory("fact", secret_like)
    with caplog.at_level(logging.DEBUG):
        viewer = make_viewer(Confirm(True))
        select_text(viewer, secret_like)
        viewer.delete_button.click()
    assert all("Таємниця" not in r.getMessage() for r in caplog.records)
    assert texts(viewer) == []


def test_read_failure_shows_a_message(make_viewer, facade):
    facade.memory_error = RuntimeError("db locked")
    viewer = make_viewer()
    assert viewer.empty_label.text() == "Не вдалося прочитати пам'ять."


def test_standalone_window_title(make_viewer):
    viewer = make_viewer()
    assert viewer.isWindow()
    assert viewer.windowTitle() == "Пам'ять — Мія"
