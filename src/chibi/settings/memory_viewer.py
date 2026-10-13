"""Memory viewer: everything she remembers about the user, to read, fix or forget.

Design notes:

* The user owns these memories: every one can be edited, pinned (pinned ones
  are never forgotten automatically) or deleted, a whole kind or everything
  can be cleared - destructive actions always ask first through the
  injectable ``confirm(title, text)``.
* Memory text is user content: it is shown as plain text only (never HTML),
  never logged (logs carry ids and kinds), and quoted in a confirmation box
  only through a plain-text message box.
* The app refuses to store anything that looks like personal data (card
  numbers, passwords, phones...).  ``facade.add_memory`` returns nothing, so
  the viewer compares the list before and after to tell the user honestly
  whether the memory was kept.
* It reads through the facade on every show and after each of its own
  actions; there is nothing to poll - memories change only while chatting.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from chibi.settings.facade import MEMORY_KINDS, AppFacade, MemoryView
from chibi.settings.widgets import (
    InfoLabel,
    SubmitLineEdit,
    ask_yes_no,
    count_uk,
    format_local_date,
    format_local_datetime,
    is_alive,
    make_button,
    make_label,
    set_style_property,
)

log = logging.getLogger(__name__)

ConfirmFn = Callable[[str, str], bool]

#: Longest memory text (the store caps longer text anyway).
MAX_MEMORY_CHARS = 300
#: The facade returns at most this many memories per list.
LIST_LIMIT = 500
#: Quoted memory text in a confirmation is cut at this length.
_QUOTE_CHARS = 120

#: Tab key ("" = every kind) and title, in display order.
KIND_TABS: tuple[tuple[str, str], ...] = (
    ("", "Усе"),
    ("fact", "Факти"),
    ("preference", "Вподобання"),
    ("relationship", "Стосунки"),
    ("game_event", "Ігри"),
    ("joke", "Жарти"),
)
#: Memory kind -> what the table and the "add" box call one memory of it.
KIND_NAMES: dict[str, str] = {
    "fact": "Факт",
    "preference": "Вподобання",
    "relationship": "Стосунки",
    "game_event": "Ігрова подія",
    "joke": "Жарт",
}
#: Memory source -> "Звідки" column.
SOURCE_NAMES: dict[str, str] = {
    "chat": "З чату",
    "system": "Програма",
    "game": "З гри",
    "user": "Вручну",
}

PRIVACY_NOTE = "Пам'ять зберігається лише на цьому комп'ютері. Нічого з екрана не запам'ятовується."
NOT_STORED_MESSAGE = (
    "Не запам'ятала: схоже на особисті дані (номер картки, пароль, телефон, пошта чи ключ) — "
    "таке не зберігається."
)

# Table columns.
COL_PIN, COL_TEXT, COL_KIND, COL_SOURCE, COL_DATE = range(5)
_HEADERS = ("📌", "Спогад", "Тип", "Звідки", "Коли")


def _quote(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _QUOTE_CHARS else text[:_QUOTE_CHARS - 1].rstrip() + "…"


def _user_message(exc: Exception, fallback: str) -> str:
    message = getattr(exc, "user_message", "")
    return message if isinstance(message, str) and message else fallback


class MemoryViewer(QWidget):
    def __init__(self, facade: AppFacade, parent: QWidget | None = None,
                 confirm: ConfirmFn | None = None) -> None:
        super().__init__(parent)
        self._facade = facade
        self._confirm: ConfirmFn = confirm or (lambda title, text: ask_yes_no(self, title, text))
        self._memories: list[MemoryView] = []
        self._texts: dict[int, str] = {}
        self._populating = False
        name = self._name()
        standalone = parent is None
        if standalone:
            self.setWindowTitle(f"Пам'ять — {name}")
            self.setMinimumSize(560, 460)
            self.resize(720, 560)

        layout = QVBoxLayout(self)
        if standalone:
            layout.setContentsMargins(16, 16, 16, 16)
        else:
            layout.setContentsMargins(4, 0, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(make_label("Пам'ять", "title", wrap=False))
        self.lead_label = InfoLabel(
            f"Що {name} пам'ятає про тебе. Закріплені спогади вона не забуде, навіть коли пам'ять "
            "переповниться. Будь-який спогад можна виправити або видалити.")
        layout.addWidget(self.lead_label)
        self.note_label = make_label(PRIVACY_NOTE, "banner")
        layout.addWidget(self.note_label)
        self.remember_off_label = make_label(
            "Запам'ятовування з чату вимкнено (сторінка «AI»): нові спогади з'являються, лише коли ти "
            "додаєш їх тут.", "warning")
        self.remember_off_label.hide()
        layout.addWidget(self.remember_off_label)

        self.tabs = QTabBar()
        self.tabs.setDrawBase(False)
        self.tabs.setExpanding(False)
        for _key, title in KIND_TABS:
            self.tabs.addTab(title)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs)

        self.search_edit = SubmitLineEdit()
        self.search_edit.setPlaceholderText("Пошук у пам'яті…")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setMaxLength(100)
        self.search_edit.textChanged.connect(lambda _text: self.reload())
        layout.addWidget(self.search_edit)

        self.table = QTableWidget(0, len(_HEADERS))
        self.table.setHorizontalHeaderLabels(list(_HEADERS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                   | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(True)
        self.table.setMinimumHeight(160)
        header = self.table.horizontalHeader()
        resize = QHeaderView.ResizeMode
        for column in range(len(_HEADERS)):
            mode = resize.Stretch if column == COL_TEXT else resize.ResizeToContents
            header.setSectionResizeMode(column, mode)
        self.table.verticalHeader().setSectionResizeMode(resize.ResizeToContents)
        self.table.horizontalHeaderItem(COL_PIN).setToolTip("Закріплені спогади не забуваються")
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self.table, 1)
        self.empty_label = InfoLabel("")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setMinimumHeight(80)
        self.empty_label.hide()
        # Takes the table's place (and its stretch) when there is nothing to list,
        # so the rest of the page doesn't spread out over the free space.
        layout.addWidget(self.empty_label, 1)
        self.count_label = make_label("", "hint")
        layout.addWidget(self.count_label)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.edit_button = make_button("Змінити")
        self.edit_button.setToolTip("Або двічі клацни на тексті спогаду")
        self.edit_button.clicked.connect(self._edit_selected)
        self.pin_button = make_button("Закріпити")
        self.pin_button.clicked.connect(self._toggle_pin)
        self.delete_button = make_button("Забути", "danger")
        self.delete_button.clicked.connect(self._delete_selected)
        self.clear_button = make_button("Забути все", "danger")
        self.clear_button.clicked.connect(self._clear)
        for button in (self.edit_button, self.pin_button, self.delete_button):
            actions.addWidget(button)
        actions.addStretch(1)
        actions.addWidget(self.clear_button)
        layout.addLayout(actions)

        add_row = QHBoxLayout()
        add_row.setSpacing(8)
        self.add_kind_combo = QComboBox()
        for kind in MEMORY_KINDS:
            self.add_kind_combo.addItem(KIND_NAMES.get(kind, kind), kind)
        self.add_edit = SubmitLineEdit()
        self.add_edit.setPlaceholderText("Наприклад: «Мого кота звати Мурчик»")
        self.add_edit.setMaxLength(MAX_MEMORY_CHARS)
        self.add_edit.submitted.connect(self._add)
        self.add_button = make_button("Запам'ятати", "accent")
        self.add_button.clicked.connect(self._add)
        add_row.addWidget(self.add_kind_combo)
        add_row.addWidget(self.add_edit, 1)
        add_row.addWidget(self.add_button)
        layout.addLayout(add_row)

        self.message_label = make_label("", "hint")
        self.message_label.hide()
        layout.addWidget(self.message_label)

        self.reload()

    # --- public API ------------------------------------------------------------------------

    def current_kind(self) -> str | None:
        """The kind of the selected tab, None for "Усе"."""
        index = self.tabs.currentIndex()
        key = KIND_TABS[index][0] if 0 <= index < len(KIND_TABS) else ""
        return key or None

    def show_kind(self, kind: str | None) -> None:
        keys = [key for key, _ in KIND_TABS]
        self.tabs.setCurrentIndex(keys.index(kind or "") if (kind or "") in keys else 0)

    def memories(self) -> list[MemoryView]:
        """What the table shows, in order."""
        return list(self._memories)

    def reload(self) -> None:
        """Re-read the memories for the current tab and search text."""
        kind = self.current_kind()
        query = self.search_edit.text().strip()
        try:
            memories = list(self._facade.list_memories(kind, query))
            failed = False
        except Exception:
            log.exception("reading memories failed (kind=%s)", kind or "all")
            memories, failed = [], True
        self._memories = memories
        self._texts = {m.id: m.text for m in memories}
        self._populate()
        self._show_empty_state(failed, query)
        self._show_remember_state()
        self._update_buttons()

    # --- table -----------------------------------------------------------------------------------

    def _populate(self) -> None:
        self._populating = True
        table = self.table
        try:
            table.clearSelection()
            table.setRowCount(len(self._memories))
            for row, memory in enumerate(self._memories):
                pin = QTableWidgetItem("📌" if memory.pinned else "")
                pin.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                text = QTableWidgetItem(memory.text)
                text.setData(Qt.ItemDataRole.UserRole, memory.id)
                text.setToolTip("Двічі клацни, щоб виправити")
                kind = QTableWidgetItem(KIND_NAMES.get(memory.kind, memory.kind))
                source = QTableWidgetItem(SOURCE_NAMES.get(memory.source, memory.source))
                date = QTableWidgetItem(format_local_date(memory.updated_at))
                date.setToolTip(f"Додано: {format_local_datetime(memory.created_at)}\n"
                                f"Змінено: {format_local_datetime(memory.updated_at)}")
                for column, item in ((COL_PIN, pin), (COL_TEXT, text), (COL_KIND, kind),
                                     (COL_SOURCE, source), (COL_DATE, date)):
                    if column != COL_TEXT:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    table.setItem(row, column, item)
        finally:
            self._populating = False

    def _show_empty_state(self, failed: bool, query: str) -> None:
        n = len(self._memories)
        if failed:
            text = "Не вдалося прочитати пам'ять."
        elif n:
            text = ""
        elif query:
            text = "Нічого не знайдено."
        elif self.current_kind() is None:
            text = (f"{self._name()} поки нічого про тебе не пам'ятає. Вона запам'ятовує те, що ти "
                    "розповідаєш у чаті, а ще сюди можна щось додати вручну.")
        else:
            text = "У цьому розділі поки порожньо."
        self.empty_label.setText(text)
        self.empty_label.setVisible(bool(text))
        self.table.setVisible(n > 0)
        if n >= LIST_LIMIT:
            count = f"Показано {LIST_LIMIT} найновіших спогадів — решту знайде пошук."
        elif n:
            count = f"Показано {count_uk(n, 'спогад', 'спогади', 'спогадів')}."
        else:
            count = ""
        self.count_label.setText(count)
        self.count_label.setVisible(bool(count))

    def _show_remember_state(self) -> None:
        try:
            remember = self._facade.settings().ai.remember_facts
        except Exception:
            log.exception("reading the settings failed")
            remember = True
        self.remember_off_label.setVisible(not remember)

    def _selected(self) -> list[MemoryView]:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        return [self._memories[r] for r in rows if 0 <= r < len(self._memories)]

    def _update_buttons(self) -> None:
        selected = self._selected()
        self.edit_button.setEnabled(len(selected) == 1)
        self.delete_button.setEnabled(bool(selected))
        self.pin_button.setEnabled(bool(selected))
        all_pinned = bool(selected) and all(m.pinned for m in selected)
        self.pin_button.setText("Відкріпити" if all_pinned else "Закріпити")
        kind = self.current_kind()
        if kind is None:
            self.clear_button.setText("Забути все")
        else:
            self.clear_button.setText("Забути весь розділ")
        # Clearing ignores the search box, so it is offered whenever the tab has anything.
        self.clear_button.setEnabled(bool(self._memories) or bool(self.search_edit.text().strip()))

    def _on_tab_changed(self, _index: int) -> None:
        kind = self.current_kind()
        if kind is not None:
            self.add_kind_combo.setCurrentIndex(max(0, self.add_kind_combo.findData(kind)))
        self.message_label.hide()
        self.reload()

    # --- actions ------------------------------------------------------------------------------------

    def _message(self, text: str, *, error: bool = False) -> None:
        self.message_label.setText(text)
        set_style_property(self.message_label, "role", "error" if error else "success")
        self.message_label.setVisible(bool(text))

    def _edit_selected(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if len(selected) != 1:
            return
        item = self.table.item(selected[0].row(), COL_TEXT)
        if item is not None:
            self.table.setCurrentItem(item)
            self.table.editItem(item)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._populating or item.column() != COL_TEXT:
            return
        memory_id = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(memory_id, int) or memory_id not in self._texts:
            return
        new = item.text().strip()
        if new == self._texts[memory_id]:
            return
        if not new:
            self._message("Порожній спогад не збережу — щоб забути його, натисни «Забути».", error=True)
        else:
            try:
                self._facade.update_memory(memory_id, new)
            except ValueError as exc:  # refused: empty or looks like personal data
                self._message(_user_message(exc, "Такий спогад не збережу."), error=True)
            except Exception as exc:
                log.error("updating memory %d failed: %s", memory_id, type(exc).__name__)
                self._message("Не вдалося змінити спогад.", error=True)
            else:
                self._message("Змінено ✓")
        # Show what was really stored (cleaned, capped or refused).  Deferred: this
        # handler may run inside the table's own editor commit, which must finish
        # before the rows it belongs to are rebuilt.
        QTimer.singleShot(0, self._reload_if_alive)

    def _reload_if_alive(self) -> None:
        if is_alive(self):
            self.reload()

    def _toggle_pin(self) -> None:
        selected = self._selected()
        if not selected:
            return
        pinned = not all(m.pinned for m in selected)
        try:
            for memory in selected:
                self._facade.set_memory_pinned(memory.id, pinned)
        except Exception as exc:
            log.error("pinning memories failed: %s", type(exc).__name__)
            self._message("Не вдалося закріпити спогад.", error=True)
        else:
            self._message("Закріплено — цього вона не забуде." if pinned else "Відкріплено.")
        self.reload()

    def _delete_selected(self) -> None:
        selected = self._selected()
        if not selected:
            return
        if len(selected) == 1:
            title = "Забути спогад?"
            text = (f"«{_quote(selected[0].text)}»\n\nВона більше не згадуватиме цього. "
                    "Цю дію не можна скасувати.")
        else:
            title = f"Забути {count_uk(len(selected), 'спогад', 'спогади', 'спогадів')}?"
            text = "Вона більше не згадуватиме вибраного. Цю дію не можна скасувати."
        if not self._confirm(title, text):
            return
        try:
            for memory in selected:
                self._facade.delete_memory(memory.id)
        except Exception as exc:
            log.error("deleting memories failed: %s", type(exc).__name__)
            self._message("Не вдалося видалити спогад.", error=True)
        else:
            self._message("Забуто.")
        self.reload()

    def _clear(self) -> None:
        kind = self.current_kind()
        if kind is None:
            title = "Забути все?"
            text = (f"{self._name()} забуде все, що пам'ятає про тебе, — і закріплені спогади теж. "
                    "Цю дію не можна скасувати.")
        else:
            section = dict(KIND_TABS)[kind]
            title = f"Забути розділ «{section}»?"
            text = (f"Буде видалено всі спогади з розділу «{section}», і закріплені теж. "
                    "Цю дію не можна скасувати.")
        if not self._confirm(title, text):
            return
        try:
            self._facade.clear_memories(kind)
        except Exception as exc:
            log.error("clearing memories failed: %s", type(exc).__name__)
            self._message("Не вдалося очистити пам'ять.", error=True)
        else:
            self._message("Пам'ять очищено." if kind is None else "Розділ очищено.")
        self.reload()

    def _add(self) -> None:
        text = self.add_edit.text().strip()[:MAX_MEMORY_CHARS]
        if not text:
            return
        kind = str(self.add_kind_combo.currentData() or MEMORY_KINDS[0])
        before = self._versions(kind)
        try:
            self._facade.add_memory(kind, text)
        except ValueError as exc:
            self._message(_user_message(exc, NOT_STORED_MESSAGE), error=True)
            return
        except Exception as exc:
            log.error("adding a memory failed: %s", type(exc).__name__)
            self._message("Не вдалося додати спогад.", error=True)
            return
        after = self._versions(kind)
        stored = before is not None and after is not None and any(
            before.get(memory_id) != version for memory_id, version in after.items())
        if before is not None and after is not None and not stored:
            self._message(NOT_STORED_MESSAGE, error=True)
            return
        self.add_edit.clear()
        self._message("Запам'ятала ✓")
        if self.current_kind() not in (None, kind):
            self.show_kind(kind)  # reloads
        else:
            self.reload()

    def _versions(self, kind: str) -> dict[int, float] | None:
        """id -> updated_at of every memory of *kind* (None if they can't be read)."""
        try:
            return {m.id: m.updated_at for m in self._facade.list_memories(kind, "")}
        except Exception:
            log.exception("reading memories failed (kind=%s)", kind)
            return None

    # --- misc ------------------------------------------------------------------------------------

    def _name(self) -> str:
        try:
            return self._facade.settings().character.name
        except Exception:
            log.exception("reading the character name failed")
            return "Мія"

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.reload()
