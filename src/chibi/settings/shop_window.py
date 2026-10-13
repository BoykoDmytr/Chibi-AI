"""Крамничка подарунків: the star balance, the gift catalog, buying and giving,
what the user holds and the keepsakes she keeps forever.

* All logic lives in :class:`~chibi.tamagotchi.gifts.GiftService` (here
  through the small :class:`ShopBackend` protocol, so tests can pass a fake
  or a real service).  The window only shows its state and its Ukrainian
  ``(ok, message)`` answers.
* "Купити" puts a gift into the inventory, "Подарувати зараз" buys and
  gives in one step, "Подарувати з інвентарю" gives one already bought.  A
  disabled button always comes with the reason in plain text next to it
  (disabled buttons cannot be focused, so a tooltip alone would be lost to
  keyboard users).  ``Alt`` + the underlined letter presses a button.
* The window follows changes made elsewhere (stars won in a mini-game while
  it is open) through ``add_listener``; while hidden it only marks itself
  stale and refreshes when shown again.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from chibi.settings.widgets import (
    Card,
    PortraitLabel,
    SectionHeader,
    is_alive,
    make_button,
    make_label,
    set_style_property,
)
from chibi.tamagotchi.gifts import DAILY_START_BONUS, GiftCatalog, GiftItem, GiftKind, Wallet, stars_uk

log = logging.getLogger(__name__)

WINDOW_TITLE = "Крамничка подарунків"
KIND_TEXT: dict[GiftKind, str] = {
    GiftKind.TREAT: "Смаколик: дарується один раз і одразу тішить.",
    GiftKind.KEEPSAKE: "Пам'ятна річ: залишиться в її колекції назавжди.",
}
EMPTY_INVENTORY = "Порожньо. Купи щось — і можна дарувати!"
EMPTY_COLLECTION = "Поки порожньо. Пам'ятні подарунки залишаться тут назавжди."
PORTRAIT_SIZE = 72


class ShopBackend(Protocol):
    """What the shop needs from :class:`~chibi.tamagotchi.gifts.GiftService`."""

    @property
    def catalog(self) -> GiftCatalog: ...

    @property
    def wallet(self) -> Wallet: ...

    def inventory(self) -> dict[str, int]: ...

    def owned(self, gift_id: str) -> int: ...

    def collection(self) -> list[GiftItem]: ...

    def has_keepsake(self, gift_id: str) -> bool: ...

    def can_buy(self, gift_id: str) -> tuple[bool, str]: ...

    def buy(self, gift_id: str) -> tuple[bool, str]: ...

    def give(self, gift_id: str) -> tuple[bool, str]: ...

    def buy_and_give(self, gift_id: str) -> tuple[bool, str]: ...

    def add_listener(self, callback: Callable[[], None]) -> Callable[[], None]: ...


def item_row_text(item: GiftItem, owned: int, has_keepsake: bool) -> str:
    """``"🍪  Печиво — 5 ⭐  · в інвентарі: 2"``."""
    text = f"{item.icon}  {item.name} — {item.price} ⭐".strip()
    if has_keepsake:
        text += "  ✓ уже в неї"
    elif owned > 0:
        text += f"  · в інвентарі: {owned}"
    return text


class ShopWindow(QWidget):
    def __init__(self, shop: ShopBackend, parent: QWidget | None = None, *,
                 portrait: Callable[[str], QPixmap | None] | None = None, name: str = "Мія") -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self._shop = shop
        self._name = name or "Мія"
        self._stale = False
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(700, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.addWidget(make_label(WINDOW_TITLE, "title", wrap=False))
        header.addStretch(1)
        self.balance_label = make_label("", "value", wrap=False)
        header.addWidget(self.balance_label, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(header)
        self.daily_label = make_label("", "hint")
        layout.addWidget(self.daily_label)

        middle = QHBoxLayout()
        middle.setSpacing(12)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setAccessibleName("Подарунки")
        self.list.setMinimumWidth(270)
        self.list.currentItemChanged.connect(lambda *_: self._on_selection())
        middle.addWidget(self.list, 1)

        details = Card(spacing=8)
        top = QHBoxLayout()
        top.setSpacing(12)
        self.portrait: PortraitLabel | None = None
        if portrait is not None:
            self.portrait = PortraitLabel(portrait, PORTRAIT_SIZE, fallback_text=self._name)
            top.addWidget(self.portrait, 0, Qt.AlignmentFlag.AlignTop)
        names = QVBoxLayout()
        names.setSpacing(2)
        self.name_label = make_label("", "section")
        self.kind_label = make_label("", "hint")
        names.addWidget(self.name_label)
        names.addWidget(self.kind_label)
        top.addLayout(names, 1)
        details.body.addLayout(top)
        self.description_label = make_label("")
        self.effects_label = make_label("", "value")
        self.price_label = make_label("", "")
        details.body.addWidget(self.description_label)
        details.body.addWidget(self.effects_label)
        details.body.addWidget(self.price_label)

        buttons = QGridLayout()
        buttons.setHorizontalSpacing(8)
        buttons.setVerticalSpacing(8)
        self.buy_button = make_button("&Купити")
        self.give_now_button = make_button("Подарувати &зараз", "primary")
        self.give_button = make_button("Подарувати з &інвентарю", "accent")
        self.buy_button.clicked.connect(lambda: self._act(self._shop.buy))
        self.give_now_button.clicked.connect(lambda: self._act(self._shop.buy_and_give))
        self.give_button.clicked.connect(lambda: self._act(self._shop.give))
        buttons.addWidget(self.buy_button, 0, 0)
        buttons.addWidget(self.give_now_button, 0, 1)
        buttons.addWidget(self.give_button, 1, 0, 1, 2)
        details.body.addLayout(buttons)
        self.reason_label = make_label("", "warning")
        details.body.addWidget(self.reason_label)
        self.status_label = make_label("", "success")
        details.body.addWidget(self.status_label)
        details.body.addStretch(1)
        middle.addWidget(details, 1)
        layout.addLayout(middle, 1)

        layout.addWidget(SectionHeader("Твій інвентар"))
        self.inventory_label = make_label("")
        layout.addWidget(self.inventory_label)
        layout.addWidget(SectionHeader("Її колекція"))
        self.collection_label = make_label("")
        layout.addWidget(self.collection_label)

        unsubscribe = shop.add_listener(self._on_changed)
        # A plain closure: it must work while the C++ side is already going away.
        self.destroyed.connect(lambda *_: unsubscribe())
        self.refresh()
        if self.list.count():
            self.list.setCurrentRow(0)
        self.list.setFocus()

    # -- public ---------------------------------------------------------------------------

    def selected_id(self) -> str | None:
        item = self.list.currentItem()
        return None if item is None else str(item.data(Qt.ItemDataRole.UserRole))

    def select(self, gift_id: str) -> bool:
        for row in range(self.list.count()):
            if self.list.item(row).data(Qt.ItemDataRole.UserRole) == gift_id:
                self.list.setCurrentRow(row)
                return True
        return False

    def refresh(self) -> None:
        """Re-read everything from the backend, keeping the selection."""
        self._stale = False
        shop = self._shop
        wallet = shop.wallet
        self.balance_label.setText(f"⭐ {wallet.stars}")
        self.balance_label.setAccessibleName(f"Баланс: {stars_uk(wallet.stars)}")
        self.balance_label.setToolTip(f"Баланс: {stars_uk(wallet.stars)}")
        if wallet.remaining_today() <= 0:
            daily = "Зірочки з міні-ігор на сьогодні вже зібрано — завтра будуть нові. Гратися можна й далі!"
        else:
            daily = (f"Зірочки з міні-ігор сьогодні: {wallet.earned_today} з {wallet.daily_cap}. "
                     f"Перший запуск щодня — ще +{DAILY_START_BONUS} ⭐.")
        self.daily_label.setText(daily)

        selected = self.selected_id()
        self.list.blockSignals(True)
        try:
            self.list.clear()
            for gift in shop.catalog:
                row = QListWidgetItem(item_row_text(gift, shop.owned(gift.id), shop.has_keepsake(gift.id)))
                row.setData(Qt.ItemDataRole.UserRole, gift.id)
                row.setToolTip(gift.description)
                self.list.addItem(row)
        finally:
            self.list.blockSignals(False)
        if (selected is None or not self.select(selected)) and self.list.count():
            self.list.setCurrentRow(0)

        held = shop.inventory()
        parts = []
        for gift_id, count in held.items():
            gift = shop.catalog.get(gift_id)
            if gift is not None:
                parts.append(f"{gift.label} ×{count}")
        self.inventory_label.setText(" · ".join(parts) if parts else EMPTY_INVENTORY)
        keepsakes = shop.collection()
        self.collection_label.setText(
            " · ".join(g.label for g in keepsakes) if keepsakes else EMPTY_COLLECTION)
        self._show_details()

    # -- internals ------------------------------------------------------------------------

    def _current(self) -> GiftItem | None:
        gift_id = self.selected_id()
        return None if gift_id is None else self._shop.catalog.get(gift_id)

    def _show_details(self) -> None:
        gift = self._current()
        enabled = gift is not None
        for button in (self.buy_button, self.give_now_button, self.give_button):
            button.setEnabled(False)
        if gift is None:
            self.name_label.setText("Крамничка порожня" if not self.list.count() else "")
            for label in (self.kind_label, self.description_label, self.effects_label, self.price_label,
                          self.reason_label):
                label.setText("")
            return
        self.name_label.setText(gift.label)
        self.kind_label.setText(KIND_TEXT[gift.kind])
        self.description_label.setText(gift.description)
        self.effects_label.setText(f"Ефект: {gift.effects_text}" if gift.effects_text else "")
        owned = self._shop.owned(gift.id)
        price = f"Ціна: {gift.price} ⭐"
        if gift.keepsake and self._shop.has_keepsake(gift.id):
            price += " · уже в її колекції ✓"
        elif owned > 0:
            price += f" · в інвентарі: {owned}"
        self.price_label.setText(price)
        if self.portrait is not None:
            self.portrait.show_portrait(gift.portrait, "smile", "calm")
        can_buy, reason = self._shop.can_buy(gift.id)
        self.buy_button.setEnabled(enabled and can_buy)
        self.give_now_button.setEnabled(enabled and can_buy)
        self.give_button.setEnabled(enabled and owned > 0)
        self.reason_label.setText("" if can_buy else reason)
        for button in (self.buy_button, self.give_now_button):
            button.setToolTip("" if can_buy else reason)

    def _on_selection(self) -> None:
        # The last action's message belongs to the gift it was about.
        self.status_label.setText("")
        self._show_details()

    def _act(self, action: Callable[[str], tuple[bool, str]]) -> None:
        gift_id = self.selected_id()
        if gift_id is None:
            return
        try:
            ok, message = action(gift_id)
        except Exception:
            log.exception("shop action failed")
            ok, message = False, "Щось пішло не так — спробуй ще раз."
        self.refresh()
        set_style_property(self.status_label, "role", "success" if ok else "error")
        self.status_label.setText(message)
        if not any(b.hasFocus() for b in (self.buy_button, self.give_now_button, self.give_button)):
            self.list.setFocus()  # the pressed button became disabled: keep the keyboard somewhere useful

    def _on_changed(self) -> None:
        if not is_alive(self):
            return
        if self.isVisible():
            self.refresh()
        else:
            self._stale = True

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self._stale:
            self.refresh()
