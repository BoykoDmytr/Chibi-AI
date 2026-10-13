"""Virtual gifts: the star wallet, the gift catalog, what the user holds and
the keepsakes she keeps forever.

* :class:`Wallet` - stars ("зірочки").  Mini-games pay into it, at most
  :data:`DAILY_EARN_CAP` a day (beyond that games still cheer her up, they
  just pay nothing); the first start of a calendar day adds
  :data:`DAILY_START_BONUS`.  There is deliberately no streak: missing a day
  costs nothing, and nothing ever takes stars away except a purchase.
* :class:`GiftCatalog` - ``data/gifts.json``.  A broken entry is skipped with
  a warning (one typo must not empty the shop).
* :class:`GiftService` - buying and giving.  A *treat* is consumed when
  given; the same treat again within :data:`REPEAT_WINDOW_SECONDS` has half
  the effect and she says a "знову?" line.  A *keepsake* can be owned once and
  stays in her collection forever; the first time she gets one she remembers
  it (one ``relationship`` memory) if the user lets her remember facts.

Persistence (:class:`~chibi.memory.repositories.StateRepository`):
``wallet`` = ``{"stars", "earned_day", "earned_today", "bonus_day"}``,
``inventory`` = ``{gift_id: count}`` (what the user holds, not yet given) and
``collection`` = ``{gift_id: epoch seconds first given}``.  Loading is
tolerant: corrupt values fall back to defaults (logged), ids the catalog no
longer knows are kept on disk - an item is never lost to an update.

A purchase writes the inventory before the wallet, and rolls the inventory
back if the wallet cannot be saved: a full disk may at worst hand out an item
for free, never take stars without the item.

Main (Qt) thread only: the repository's SQLite connection, the composer and
the bus handlers live there.  Nothing here logs what she says.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from chibi.character.states import CharacterState
from chibi.core.clock import Clock
from chibi.core.config import Settings
from chibi.core.event_bus import EventBus
from chibi.core.events import AppStarted, SpeechRequested
from chibi.memory.repositories import StateRepository
from chibi.personality.text_uk import plural
from chibi.tamagotchi.needs import Needs, finite_number
from chibi.tamagotchi.service import EFFECT_FIELDS

log = logging.getLogger(__name__)

KEY_WALLET = "wallet"
KEY_INVENTORY = "inventory"
KEY_COLLECTION = "collection"

CATALOG_PATH = Path(__file__).with_name("data") / "gifts.json"
SUPPORTED_VERSION = 1

#: Stars mini-games can earn per calendar day.
DAILY_EARN_CAP = 60
#: First start of the app on a calendar day.
DAILY_START_BONUS = 5
#: A sane ceiling: stars are an int in JSON and must stay one.
MAX_STARS = 1_000_000
#: Copies of one treat the user can hold.
MAX_STACK = 99
#: The same treat again this soon has half the effect.
REPEAT_WINDOW_SECONDS = 600.0
REPEAT_FACTOR = 0.5
MAX_PRICE = 1000
#: A single gift may move a need at most this far.
MAX_EFFECT = 50.0

KEEPSAKE_MEMORY_KIND = "relationship"
KEEPSAKE_MEMORY_SOURCE = "system"
KEEPSAKE_MEMORY_IMPORTANCE = 0.6

#: Phrase keys every gift reaction falls back to.
GENERIC_KEY = "gift.generic"
AGAIN_KEY = "gift.again"
#: Said only if the phrase bank has nothing at all (broken install).
FALLBACK_THANKS = "Дякую! Це так мило."

_ID = re.compile(r"[a-z][a-z0-9_]{0,31}")
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_STATES = frozenset(s.value for s in CharacterState)

#: need -> label in the shop.  Hunger is shown the other way round, as satiety.
EFFECT_LABELS: dict[str, str] = {
    "happiness": "Радість",
    "hunger": "Ситість",
    "energy": "Енергія",
    "boredom": "Нудьга",
    "affection": "Ніжність",
    "friendship_xp": "Дружба",
}

# -- user-facing messages (the shop shows them under the buttons) -------------------------
MSG_UNKNOWN = "Такого подарунка в крамничці немає."
MSG_SAVE_FAILED = "Не вдалося зберегти — зірочки й подарунки на місці. Спробуй ще раз."
MSG_STACK_FULL = "Цього в інвентарі вже повно — спершу подаруй трохи 🙂"


def stars_uk(n: int) -> str:
    """``"1 зірочка"``, ``"3 зірочки"``, ``"5 зірочок"``."""
    return f"{n} {plural(n, ('зірочка', 'зірочки', 'зірочок'))}"


def local_day(epoch_seconds: float) -> str:
    """Local calendar date ``"YYYY-MM-DD"``: days begin at the user's midnight."""
    return datetime.fromtimestamp(epoch_seconds).date().isoformat()


def describe_effects(effects: Mapping[str, float]) -> str:
    """``"Радість +4 · Ситість +10 · Дружба +1"`` for the shop.

    Hunger is shown as satiety (a gift that lowers hunger *feeds* her), so
    every number a user sees reads "plus is nice" except boredom.
    """
    parts: list[str] = []
    for name, label in EFFECT_LABELS.items():
        value = finite_number(effects.get(name))
        if not value:
            continue
        if name == "hunger":
            value = -value
        sign = "+" if value > 0 else "−"
        parts.append(f"{label} {sign}{abs(value):g}")
    return " · ".join(parts)


# =========================================================================================
# Wallet
# =========================================================================================


class Wallet:
    """Stars, persisted under ``wallet``.  Never negative, never above :data:`MAX_STARS`."""

    def __init__(self, repo: StateRepository, clock: Clock, *, daily_cap: int = DAILY_EARN_CAP,
                 daily_bonus: int = DAILY_START_BONUS) -> None:
        self._repo = repo
        self._clock = clock
        self._daily_cap = max(0, int(daily_cap))
        self._daily_bonus = max(0, int(daily_bonus))
        self._stars = 0
        self._earned_day = ""
        self._earned_today = 0
        self._bonus_day = ""
        self._listeners: list[Callable[[], None]] = []

    # -- reading ------------------------------------------------------------------------

    @property
    def stars(self) -> int:
        return self._stars

    @property
    def daily_cap(self) -> int:
        return self._daily_cap

    @property
    def earned_today(self) -> int:
        """Stars earned from games today (0 once the day has changed)."""
        return self._earned_today if self._earned_day == self._today() else 0

    def remaining_today(self) -> int:
        return max(0, self._daily_cap - self.earned_today)

    def can_afford(self, price: int) -> bool:
        return 0 <= price <= self._stars

    def to_dict(self) -> dict[str, Any]:
        return {"stars": self._stars, "earned_day": self._earned_day,
                "earned_today": self._earned_today, "bonus_day": self._bonus_day}

    # -- persistence ----------------------------------------------------------------------

    def load(self) -> None:
        """Restore the wallet.  Anything corrupt falls back to its default (logged)."""
        try:
            data = self._repo.get(KEY_WALLET)
        except Exception:
            log.exception("could not read the wallet; starting empty")
            data = None
        self._stars, self._earned_day, self._earned_today, self._bonus_day = 0, "", 0, ""
        if data is None:
            self._notify()
            return
        if not isinstance(data, Mapping):
            log.warning("stored wallet is not an object; using defaults")
            self._notify()
            return
        stars = _count(data.get("stars"), MAX_STARS)
        if stars is None:
            log.warning("stored wallet: invalid star count; using 0")
        self._stars = stars or 0
        self._earned_day = _day(data.get("earned_day"), "earned_day")
        earned = _count(data.get("earned_today", 0), self._daily_cap)
        if earned is None:
            log.warning("stored wallet: invalid earned_today; using 0")
        self._earned_today = earned or 0
        self._bonus_day = _day(data.get("bonus_day"), "bonus_day")
        self._notify()

    def _save(self) -> bool:
        try:
            self._repo.set(KEY_WALLET, self.to_dict())
        except Exception:
            log.exception("could not save the wallet")
            return False
        return True

    # -- changing -------------------------------------------------------------------------

    def earn(self, amount: int) -> int:
        """Credit stars from a mini-game, within today's cap.  Returns how many
        were credited (0 once the cap is reached).  Credited stars stay even if
        saving fails - the next save stores them."""
        wanted = _count(amount, MAX_STARS) or 0
        self._roll_day()
        granted = min(wanted, self._daily_cap - self._earned_today, MAX_STARS - self._stars)
        if granted <= 0:
            return 0
        self._stars += granted
        self._earned_today += granted
        self._save()
        self._notify()
        return granted

    def spend(self, amount: int) -> bool:
        """Take ``amount`` stars.  False (nothing changed) if there are not enough
        or the new balance cannot be saved."""
        price = _count(amount, MAX_STARS)
        if price is None or price > self._stars:
            return False
        self._stars -= price
        if not self._save():
            self._stars += price
            return False
        self._notify()
        return True

    def daily_start(self) -> int:
        """The app started: on the first start of a calendar day, a little gift
        of stars.  No streak - yesterday does not matter.  Returns the bonus."""
        today = self._today()
        if self._bonus_day == today or self._daily_bonus <= 0:
            return 0
        self._bonus_day = today
        self._stars = min(MAX_STARS, self._stars + self._daily_bonus)
        self._save()
        self._notify()
        log.info("first start today: +%d stars", self._daily_bonus)
        return self._daily_bonus

    # -- listeners ------------------------------------------------------------------------

    def add_listener(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Called after every change (main thread).  Returns an unsubscribe function."""
        self._listeners.append(callback)

        def remove() -> None:
            if callback in self._listeners:
                self._listeners.remove(callback)

        return remove

    def _notify(self) -> None:
        for callback in list(self._listeners):
            try:
                callback()
            except Exception:
                log.exception("wallet listener failed")

    # -- internals ------------------------------------------------------------------------

    def _today(self) -> str:
        return local_day(self._clock.now())

    def _roll_day(self) -> None:
        today = self._today()
        if self._earned_day != today:
            self._earned_day = today
            self._earned_today = 0


def _count(value: Any, upper: int) -> int | None:
    """A stored non-negative whole number (clamped to ``upper``), or None if unusable."""
    number = finite_number(value)
    if number is None or number < 0 or not float(number).is_integer():
        return None
    return min(int(number), upper)


def _day(value: Any, what: str) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str) and _DAY.fullmatch(value):
        return value
    log.warning("stored wallet: invalid %s; ignoring it", what)
    return ""


# =========================================================================================
# Catalog
# =========================================================================================


class GiftKind(StrEnum):
    TREAT = "treat"  # consumed when given
    KEEPSAKE = "keepsake"  # stays in her collection forever; owned at most once


@dataclass(frozen=True)
class GiftItem:
    id: str
    name: str
    icon: str
    price: int
    kind: GiftKind
    description: str
    effects: Mapping[str, float]
    #: Phrase keys for her reaction, best first ("gift.<id>"); "gift.generic" is always appended.
    reaction_keys: tuple[str, ...]
    #: Character state / portrait shown if the phrase bank has no line at all, and in the shop.
    state: str
    portrait: str
    #: Keepsakes only: what she remembers ("Мені подарували плюшевого котика").
    memory: str = ""

    @property
    def keepsake(self) -> bool:
        return self.kind is GiftKind.KEEPSAKE

    @property
    def label(self) -> str:
        """``"🍪 Печиво"``."""
        return f"{self.icon} {self.name}".strip()

    @property
    def effects_text(self) -> str:
        return describe_effects(self.effects)


class CatalogError(ValueError):
    """One catalog entry is invalid (it is skipped)."""


class GiftCatalog:
    """The gifts in the shop, in file order (cheapest first by convention)."""

    def __init__(self, items: Sequence[GiftItem] = ()) -> None:
        self._items: dict[str, GiftItem] = {}
        for item in items:
            self._items.setdefault(item.id, item)

    @classmethod
    def load(cls, path: Path | None = None) -> GiftCatalog:
        source = path or CATALOG_PATH
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.error("gift catalog %s could not be read (%s); the shop is empty", source.name, exc)
            return cls()
        return cls.from_dict(data, source=source.name)

    @classmethod
    def from_dict(cls, data: Any, source: str = "<dict>") -> GiftCatalog:
        """Build a catalog from parsed JSON, skipping invalid entries with a warning."""
        if not isinstance(data, Mapping) or not isinstance(data.get("gifts"), list):
            log.error("gift catalog %s has no 'gifts' list; the shop is empty", source)
            return cls()
        if data.get("version") != SUPPORTED_VERSION:
            log.warning("gift catalog %s has version %r (expected %d); loading what is understood",
                        source, data.get("version"), SUPPORTED_VERSION)
        items: list[GiftItem] = []
        seen: set[str] = set()
        for index, entry in enumerate(data["gifts"]):
            try:
                item = parse_gift(entry)
            except (CatalogError, TypeError, ValueError) as exc:
                log.warning("gift catalog %s: entry %d skipped: %s", source, index, exc)
                continue
            if item.id in seen:
                log.warning("gift catalog %s: duplicate id %r skipped", source, item.id)
                continue
            seen.add(item.id)
            items.append(item)
        log.info("gift catalog %s: %d gifts", source, len(items))
        return cls(items)

    def get(self, gift_id: str) -> GiftItem | None:
        return self._items.get(gift_id)

    def items(self) -> list[GiftItem]:
        return list(self._items.values())

    def __iter__(self) -> Iterator[GiftItem]:
        return iter(list(self._items.values()))

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, gift_id: object) -> bool:
        return gift_id in self._items


def parse_gift(entry: Any) -> GiftItem:
    """Validate one catalog entry.  Raises :class:`CatalogError` with the reason."""
    if not isinstance(entry, Mapping):
        raise CatalogError("entry is not an object")
    gift_id = entry.get("id")
    if not isinstance(gift_id, str) or not _ID.fullmatch(gift_id):
        raise CatalogError(f"invalid id {gift_id!r}")
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip() or len(name) > 40:
        raise CatalogError(f"{gift_id}: invalid name")
    icon = entry.get("icon", "")
    if not isinstance(icon, str) or len(icon) > 8:
        raise CatalogError(f"{gift_id}: invalid icon")
    price = entry.get("price")
    if isinstance(price, bool) or not isinstance(price, int) or not 1 <= price <= MAX_PRICE:
        raise CatalogError(f"{gift_id}: price must be a whole number 1..{MAX_PRICE}")
    try:
        kind = GiftKind(entry.get("kind"))
    except (ValueError, TypeError):
        raise CatalogError(f"{gift_id}: unknown kind {entry.get('kind')!r}") from None
    description = entry.get("description", "")
    if not isinstance(description, str) or len(description) > 200:
        raise CatalogError(f"{gift_id}: invalid description")
    effects = _parse_effects(gift_id, entry.get("effects"))
    keys = entry.get("reaction_keys", [f"gift.{gift_id}"])
    if not isinstance(keys, list) or not keys or not all(isinstance(k, str) and k.strip() for k in keys):
        raise CatalogError(f"{gift_id}: reaction_keys must be a non-empty list of phrase keys")
    state = entry.get("state", CharacterState.HAPPY.value)
    if not isinstance(state, str) or state not in _STATES:
        raise CatalogError(f"{gift_id}: unknown state {state!r}")
    portrait = entry.get("portrait", "")
    if not isinstance(portrait, str):
        raise CatalogError(f"{gift_id}: portrait is not a string")
    memory = entry.get("memory", "")
    if not isinstance(memory, str):
        raise CatalogError(f"{gift_id}: memory is not a string")
    if kind is GiftKind.KEEPSAKE and not memory.strip():
        raise CatalogError(f"{gift_id}: a keepsake needs a memory text")
    return GiftItem(
        id=gift_id, name=name.strip(), icon=icon.strip(), price=price, kind=kind,
        description=description.strip(), effects=effects,
        reaction_keys=tuple(dict.fromkeys(k.strip() for k in keys)), state=state,
        portrait=portrait.strip(), memory=memory.strip() if kind is GiftKind.KEEPSAKE else "",
    )


def _parse_effects(gift_id: str, raw: Any) -> Mapping[str, float]:
    if not isinstance(raw, Mapping) or not raw:
        raise CatalogError(f"{gift_id}: effects must be a non-empty object")
    effects: dict[str, float] = {}
    for name, value in raw.items():
        if name not in EFFECT_FIELDS:
            raise CatalogError(f"{gift_id}: unknown effect {name!r}")
        number = finite_number(value)
        if number is None or abs(number) > MAX_EFFECT:
            raise CatalogError(f"{gift_id}: effect {name} must be a number within ±{MAX_EFFECT:g}")
        if name == "friendship_xp" and number < 0:
            raise CatalogError(f"{gift_id}: a gift never lowers friendship")
        effects[name] = number
    return MappingProxyType(effects)


# =========================================================================================
# Service
# =========================================================================================


class NeedsEffects(Protocol):
    """What gifts and games need from :class:`~chibi.tamagotchi.service.TamagotchiService`."""

    def apply_effect(self, deltas: Mapping[str, float], *, reason: str) -> Needs: ...


class LineComposer(Protocol):
    """What gifts and games need from :class:`~chibi.personality.composer.SpeechComposer`."""

    def compose(self, *keys: str, solicited: bool = False, **variables: Any) -> bool: ...


class MemorySink(Protocol):
    """What gifts need from :class:`~chibi.memory.longterm.MemoryStore`."""

    def add(self, kind: str, text: str, source: str = ..., importance: float = ...) -> Any: ...


class GiftService:
    """Buying and giving gifts.  Every action returns ``(ok, message)``: the
    message is short Ukrainian text for the shop window."""

    def __init__(self, bus: EventBus, repo: StateRepository, clock: Clock, tamagotchi: NeedsEffects,
                 composer: LineComposer, memory: MemorySink | None, settings: Callable[[], Settings], *,
                 catalog: GiftCatalog | None = None, wallet: Wallet | None = None) -> None:
        self._bus = bus
        self._repo = repo
        self._clock = clock
        self._tamagotchi = tamagotchi
        self._composer = composer
        self._memory = memory
        self._settings = settings
        self._catalog = catalog if catalog is not None else GiftCatalog.load()
        self._wallet = wallet if wallet is not None else Wallet(repo, clock)
        self._inventory: dict[str, int] = {}
        self._collection: dict[str, float] = {}
        #: treat id -> monotonic time it was last given (repeat detection; not persisted).
        self._last_given: dict[str, float] = {}
        self._listeners: list[Callable[[], None]] = []
        self._wallet.add_listener(self._notify)
        self._unsubscribers = [bus.subscribe(AppStarted, self._on_app_started)]
        self.load()

    # -- state ------------------------------------------------------------------------------

    @property
    def catalog(self) -> GiftCatalog:
        return self._catalog

    @property
    def wallet(self) -> Wallet:
        return self._wallet

    def inventory(self) -> dict[str, int]:
        """What the user holds, catalog order: ``{gift_id: count}`` (counts > 0 only)."""
        return {item.id: self._inventory[item.id] for item in self._catalog
                if self._inventory.get(item.id, 0) > 0}

    def owned(self, gift_id: str) -> int:
        return self._inventory.get(gift_id, 0)

    def collection(self) -> list[GiftItem]:
        """Her keepsakes, in the order she got them."""
        ids = sorted(self._collection, key=lambda gift_id: self._collection[gift_id])
        return [item for item in (self._catalog.get(i) for i in ids) if item is not None]

    def has_keepsake(self, gift_id: str) -> bool:
        return gift_id in self._collection

    def can_buy(self, gift_id: str) -> tuple[bool, str]:
        """Whether :meth:`buy` would succeed, and why not (without buying)."""
        item = self._catalog.get(gift_id)
        if item is None:
            return False, MSG_UNKNOWN
        if item.keepsake and self.has_keepsake(gift_id):
            return False, f"{item.label} уже в її колекції — назавжди."
        if item.keepsake and self.owned(gift_id) > 0:
            return False, f"{item.label} вже є в інвентарі — подаруй звідти!"
        if self.owned(gift_id) >= MAX_STACK:
            return False, MSG_STACK_FULL
        if not self._wallet.can_afford(item.price):
            missing = item.price - self._wallet.stars
            return False, f"Поки що бракує {missing} ⭐. Зірочки приносять міні-ігри — пограймо?"
        return True, ""

    # -- persistence --------------------------------------------------------------------------

    def load(self) -> None:
        """(Re)read wallet, inventory and collection, e.g. after "delete all data"."""
        self._wallet.load()  # notifies
        self._inventory = self._load_counts()
        self._collection = self._load_collection()
        self._last_given.clear()
        self._notify()

    def _load_counts(self) -> dict[str, int]:
        data = self._read(KEY_INVENTORY)
        if data is None:
            return {}
        if not isinstance(data, Mapping):
            log.warning("stored inventory is not an object; starting empty")
            return {}
        counts: dict[str, int] = {}
        for gift_id, value in data.items():
            count = _count(value, MAX_STACK)
            if not isinstance(gift_id, str) or not _ID.fullmatch(gift_id) or count is None:
                log.warning("stored inventory: an invalid entry was ignored")
                continue
            if count > 0:
                counts[gift_id] = count
        return counts

    def _load_collection(self) -> dict[str, float]:
        data = self._read(KEY_COLLECTION)
        if data is None:
            return {}
        if not isinstance(data, Mapping):
            log.warning("stored collection is not an object; starting empty")
            return {}
        result: dict[str, float] = {}
        for gift_id, value in data.items():
            if not isinstance(gift_id, str) or not _ID.fullmatch(gift_id):
                log.warning("stored collection: an invalid entry was ignored")
                continue
            # A broken date must not cost her a keepsake.
            at = finite_number(value)
            result[gift_id] = at if at is not None and at >= 0 else 0.0
        return result

    def _read(self, key: str) -> Any:
        try:
            return self._repo.get(key)
        except Exception:
            log.exception("could not read %s", key)
            return None

    def _write(self, key: str, value: Any) -> bool:
        try:
            self._repo.set(key, value)
        except Exception:
            log.exception("could not save %s", key)
            return False
        return True

    def close(self) -> None:
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()

    # -- actions ----------------------------------------------------------------------------

    def buy(self, gift_id: str) -> tuple[bool, str]:
        """Buy one gift into the inventory."""
        ok, reason = self.can_buy(gift_id)
        if not ok:
            return False, reason
        item = self._catalog.get(gift_id)
        if item is None:  # can_buy said yes, so this cannot happen; never trust that blindly
            return False, MSG_UNKNOWN
        before = dict(self._inventory)
        self._inventory[gift_id] = self.owned(gift_id) + 1
        # Inventory first: if the wallet cannot be saved, the item is rolled back
        # (at worst it stays free on disk) - stars are never taken without it.
        if not self._write(KEY_INVENTORY, self._inventory):
            self._inventory = before
            return False, MSG_SAVE_FAILED
        if not self._wallet.spend(item.price):
            self._inventory = before
            self._write(KEY_INVENTORY, self._inventory)
            return False, MSG_SAVE_FAILED
        log.info("bought gift %s", item.id)
        self._notify()
        return True, f"Куплено: {item.label}. Подарунок чекає в інвентарі."

    def give(self, gift_id: str) -> tuple[bool, str]:
        """Give her one gift from the inventory: effects, her reaction, a memory for a new keepsake."""
        item = self._catalog.get(gift_id)
        if item is None:
            return False, MSG_UNKNOWN
        count = self.owned(gift_id)
        if count <= 0:
            return False, f"В інвентарі немає: {item.label}. Спершу купи — і можна дарувати."
        # A new keepsake enters her collection on disk before it leaves the
        # inventory: whatever write fails, it is never lost.
        new_keepsake = item.keepsake and gift_id not in self._collection
        if new_keepsake:
            self._collection[gift_id] = self._clock.now()
            if not self._write(KEY_COLLECTION, self._collection):
                del self._collection[gift_id]
                return False, MSG_SAVE_FAILED
        before = dict(self._inventory)
        if count == 1:
            del self._inventory[gift_id]
        else:
            self._inventory[gift_id] = count - 1
        if not self._write(KEY_INVENTORY, self._inventory):
            self._inventory = before
            if new_keepsake:
                del self._collection[gift_id]
                self._write(KEY_COLLECTION, self._collection)
            return False, MSG_SAVE_FAILED

        now = self._clock.monotonic()
        last = self._last_given.get(gift_id)
        again = not item.keepsake and last is not None and now - last < REPEAT_WINDOW_SECONDS
        self._last_given[gift_id] = now
        factor = REPEAT_FACTOR if again else 1.0
        try:
            self._tamagotchi.apply_effect({k: v * factor for k, v in item.effects.items()},
                                          reason=f"gift.{item.id}")
        except Exception:  # the gift is given; a failing need update must not hide her reaction
            log.exception("gift %s: applying its effect failed", item.id)
        self._react(item, again)
        if new_keepsake:
            self._remember(item)
        log.info("gave gift %s%s", item.id, " (again, half effect)" if again else "")
        self._notify()
        return True, self._given_message(item, again, new_keepsake)

    def buy_and_give(self, gift_id: str) -> tuple[bool, str]:
        """"Подарувати зараз": buy and give in one step.  If giving fails after the
        purchase, the gift simply stays in the inventory."""
        ok, message = self.buy(gift_id)
        if not ok:
            return False, message
        return self.give(gift_id)

    # -- reactions -----------------------------------------------------------------------------

    def _react(self, item: GiftItem, again: bool) -> None:
        keys = (*((AGAIN_KEY,) if again else ()), *item.reaction_keys, GENERIC_KEY)
        try:
            # ``gift`` is context for the AI brain (an attribute of the event);
            # the phrase bank ignores it.
            spoken = self._composer.compose(*keys, solicited=True, gift=item.name)
        except Exception:
            log.exception("gift %s: composing her reaction failed", item.id)
            spoken = False
        if not spoken:
            self._bus.publish(SpeechRequested(text=FALLBACK_THANKS, state=item.state, portrait=item.portrait,
                                              solicited=True, source=f"gift.{item.id}"))

    def _remember(self, item: GiftItem) -> None:
        if self._memory is None or not item.memory:
            return
        try:
            if not self._settings().ai.remember_facts:
                return
            self._memory.add(KEEPSAKE_MEMORY_KIND, item.memory, source=KEEPSAKE_MEMORY_SOURCE,
                             importance=KEEPSAKE_MEMORY_IMPORTANCE)
        except Exception:
            log.exception("gift %s: could not remember the keepsake", item.id)

    def _given_message(self, item: GiftItem, again: bool, new_keepsake: bool) -> str:
        her = self._her_name()
        message = f"{her} отримала подарунок: {item.label}."
        if new_keepsake:
            message += " Тепер це її скарб назавжди."
        elif again:
            message += " Знову те саме так скоро — тож ефект наполовину."
        return message

    def _her_name(self) -> str:
        try:
            return self._settings().character.name or "Мія"
        except Exception:
            return "Мія"

    # -- events / listeners ------------------------------------------------------------------

    def _on_app_started(self, _event: AppStarted) -> None:
        self._wallet.daily_start()

    def add_listener(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Called after any change to stars, inventory or collection (main thread)."""
        self._listeners.append(callback)

        def remove() -> None:
            if callback in self._listeners:
                self._listeners.remove(callback)

        return remove

    def _notify(self) -> None:
        for callback in list(self._listeners):
            try:
                callback()
            except Exception:
                log.exception("gift listener failed")
