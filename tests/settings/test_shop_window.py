"""The gift shop window over a real GiftService (only the composer is faked)."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtTest import QTest

from chibi.settings.shop_window import EMPTY_COLLECTION, EMPTY_INVENTORY, ShopWindow, item_row_text
from chibi.tamagotchi.gifts import KEY_WALLET
from tests.games.fakes import World


@pytest.fixture
def world(db, bus, clock):
    return World(db, bus, clock)


@pytest.fixture
def gifts(world):
    return world.gifts()


@pytest.fixture
def make_shop(gifts, track):
    def make(**kwargs) -> ShopWindow:
        window = track(ShopWindow(gifts, **kwargs))
        window.show()
        return window

    return make


def give_stars(gifts, stars):
    gifts._repo.set(KEY_WALLET, {**gifts.wallet.to_dict(), "stars": stars})
    gifts.wallet.load()


def test_shows_balance_catalog_and_empty_state(make_shop, gifts):
    shop = make_shop()
    assert shop.windowTitle() == "Крамничка подарунків"
    assert shop.balance_label.text() == "⭐ 0"
    assert shop.balance_label.accessibleName() == "Баланс: 0 зірочок"
    assert "Зірочки з міні-ігор сьогодні: 0 з 60" in shop.daily_label.text()
    assert shop.list.count() == len(gifts.catalog) == 10
    assert shop.list.item(0).text() == "🍎  Яблуко — 4 ⭐"
    assert shop.selected_id() == "apple"
    assert shop.inventory_label.text() == EMPTY_INVENTORY
    assert shop.collection_label.text() == EMPTY_COLLECTION
    assert shop.list.hasFocus() or not shop.isActiveWindow()


def test_details_and_reason_when_stars_are_missing(make_shop):
    shop = make_shop()
    shop.select("cookie")
    assert shop.name_label.text() == "🍪 Печиво"
    assert shop.kind_label.text().startswith("Смаколик")
    assert shop.effects_label.text() == "Ефект: Радість +4 · Ситість +10 · Дружба +1"
    assert shop.price_label.text() == "Ціна: 5 ⭐"
    assert not shop.buy_button.isEnabled()
    assert not shop.give_now_button.isEnabled()
    assert not shop.give_button.isEnabled()
    assert shop.reason_label.text().startswith("Поки що бракує 5 ⭐")
    assert shop.buy_button.toolTip() == shop.reason_label.text()


def test_follows_stars_earned_elsewhere(make_shop, gifts):
    shop = make_shop()
    shop.select("cookie")
    gifts.wallet.earn(20)  # a mini-game while the shop is open
    assert shop.balance_label.text() == "⭐ 20"
    assert shop.buy_button.isEnabled() and shop.give_now_button.isEnabled()
    assert shop.reason_label.text() == ""
    assert shop.selected_id() == "cookie"  # the selection survives the refresh


def test_buy_then_give_from_inventory(make_shop, gifts, world):
    give_stars(gifts, 20)
    shop = make_shop()
    shop.select("cookie")
    QTest.mouseClick(shop.buy_button, Qt.MouseButton.LeftButton)
    assert shop.status_label.text() == "Куплено: 🍪 Печиво. Подарунок чекає в інвентарі."
    assert shop.status_label.property("role") == "success"
    assert shop.balance_label.text() == "⭐ 15"
    assert shop.inventory_label.text() == "🍪 Печиво ×1"
    assert shop.list.currentItem().text() == "🍪  Печиво — 5 ⭐  · в інвентарі: 1"
    assert shop.give_button.isEnabled()
    QTest.mouseClick(shop.give_button, Qt.MouseButton.LeftButton)
    assert shop.status_label.text() == "Мія отримала подарунок: 🍪 Печиво."
    assert shop.inventory_label.text() == EMPTY_INVENTORY
    assert not shop.give_button.isEnabled()
    assert world.composer.keys == [("gift.cookie", "gift.generic")]


def test_give_now(make_shop, gifts, world):
    give_stars(gifts, 30)
    shop = make_shop()
    shop.select("cake")
    QTest.mouseClick(shop.give_now_button, Qt.MouseButton.LeftButton)
    assert shop.status_label.text() == "Мія отримала подарунок: 🎂 Торт."
    assert shop.balance_label.text() == "⭐ 5"
    assert world.composer.keys == [("gift.cake", "gift.generic")]


def test_keepsake_goes_to_her_collection(make_shop, gifts):
    give_stars(gifts, 100)
    shop = make_shop()
    shop.select("plush_cat")
    assert shop.kind_label.text().startswith("Пам'ятна річ")
    QTest.mouseClick(shop.give_now_button, Qt.MouseButton.LeftButton)
    assert "скарб назавжди" in shop.status_label.text()
    assert shop.collection_label.text() == "🐱 Плюшевий котик"
    assert shop.list.currentItem().text().endswith("✓ уже в неї")
    assert shop.price_label.text() == "Ціна: 30 ⭐ · уже в її колекції ✓"
    assert not shop.buy_button.isEnabled() and not shop.give_now_button.isEnabled()
    assert shop.reason_label.text() == "🐱 Плюшевий котик уже в її колекції — назавжди."


def test_keepsake_waiting_in_the_inventory(make_shop, gifts):
    give_stars(gifts, 100)
    shop = make_shop()
    shop.select("headphones")
    QTest.mouseClick(shop.buy_button, Qt.MouseButton.LeftButton)
    assert not shop.buy_button.isEnabled()
    assert shop.give_button.isEnabled()
    assert "вже є в інвентарі" in shop.reason_label.text()


def test_failed_action_is_shown_as_an_error(make_shop, gifts, world):
    give_stars(gifts, 20)
    shop = make_shop()
    shop.select("cookie")
    world.repo.broken.add(KEY_WALLET)
    QTest.mouseClick(shop.buy_button, Qt.MouseButton.LeftButton)
    assert shop.status_label.property("role") == "error"
    assert "Не вдалося зберегти" in shop.status_label.text()
    assert shop.balance_label.text() == "⭐ 20"


def test_crashing_backend_is_reported_not_raised(make_shop, gifts, monkeypatch):
    def boom(_gift_id):
        raise RuntimeError("no")

    give_stars(gifts, 20)
    shop = make_shop()
    monkeypatch.setattr(gifts, "buy", boom)
    shop._act(gifts.buy)
    assert shop.status_label.text() == "Щось пішло не так — спробуй ще раз."


def test_selection_change_clears_the_message(make_shop, gifts):
    give_stars(gifts, 20)
    shop = make_shop()
    shop.select("cookie")
    QTest.mouseClick(shop.buy_button, Qt.MouseButton.LeftButton)
    assert shop.status_label.text()
    QTest.keyClick(shop.list, Qt.Key.Key_Down)
    assert shop.selected_id() == "cocoa"
    assert shop.status_label.text() == ""


def test_keyboard_navigation(make_shop):
    shop = make_shop()
    shop.list.setFocus()
    QTest.keyClick(shop.list, Qt.Key.Key_End)
    assert shop.selected_id() == "headphones"
    QTest.keyClick(shop.list, Qt.Key.Key_Home)
    assert shop.selected_id() == "apple"


def test_hidden_shop_refreshes_when_shown(make_shop, gifts):
    shop = make_shop()
    shop.hide()
    gifts.wallet.earn(7)
    assert shop.balance_label.text() == "⭐ 0"  # nothing done while hidden
    shop.show()
    assert shop.balance_label.text() == "⭐ 7"


def test_listener_removed_when_the_window_is_deleted(gifts, qapp):
    before = len(gifts._listeners)
    shop = ShopWindow(gifts)
    assert len(gifts._listeners) == before + 1
    shop.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert len(gifts._listeners) == before
    gifts.wallet.earn(1)  # no callback into a deleted window


def test_daily_cap_reached_text(make_shop, gifts):
    gifts.wallet.earn(60)
    shop = make_shop()
    assert "на сьогодні вже зібрано" in shop.daily_label.text()
    assert "Гратися можна й далі" in shop.daily_label.text()


def test_portrait_of_the_selected_gift(make_shop):
    pixmap = QPixmap(32, 32)
    pixmap.fill(QColor("#ff00ff"))
    shop = make_shop(portrait=lambda name: pixmap if name in ("eat", "music") else None)
    shop.select("cookie")
    assert shop.portrait.current == "eat"
    shop.select("flower")  # "shy" missing -> "smile" missing -> "calm" missing -> letter avatar
    assert shop.portrait.current == ""
    shop.select("headphones")
    assert shop.portrait.current == "music"


def test_item_row_text():
    from chibi.tamagotchi.gifts import GiftCatalog

    cookie = GiftCatalog.load().get("cookie")
    assert item_row_text(cookie, 0, False) == "🍪  Печиво — 5 ⭐"
    assert item_row_text(cookie, 3, False) == "🍪  Печиво — 5 ⭐  · в інвентарі: 3"


def test_empty_catalog(track, world):
    from chibi.tamagotchi.gifts import GiftCatalog

    gifts = world.gifts(catalog=GiftCatalog())
    shop = track(ShopWindow(gifts))
    shop.show()
    assert shop.list.count() == 0
    assert shop.name_label.text() == "Крамничка порожня"
    assert not shop.buy_button.isEnabled()
