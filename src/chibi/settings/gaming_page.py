"""The "Ігровий компаньйон" page of the settings dialog: how she behaves while
the user plays (``Settings.gaming``).

The list of the user's own games is *not* here: it already lives on the
dialog's "Ігри" page ("Мої ігри"), and which games she may *see* is chosen on
the "Бачення екрана" page - this page points to both instead of duplicating
them.

Same rules as the other pages: the widgets edit a copy, ``collect`` writes
only the fields the user changed here (so a stale widget never reverts a
change made elsewhere), ``sync`` shows the app's current values.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

from chibi.core.config import Settings
from chibi.settings.widgets import Card, CheckRow, InfoLabel, SectionHeader, SliderRow, count_uk, make_label

log = logging.getLogger(__name__)

#: (upper bound inclusive, word) for the reaction-frequency slider.
FREQUENCY_WORDS: tuple[tuple[int, str], ...] = (
    (0, "мовчить"), (29, "тихо"), (59, "помірно"), (84, "охоче"), (100, "балакуче"),
)


def frequency_text(value: int) -> str:
    for bound, word in FREQUENCY_WORDS:
        if value <= bound:
            return word
    return FREQUENCY_WORDS[-1][1]


def reactions_per_hour(reaction_frequency: int) -> int:
    """0 / 6 / 12 reactions per hour for 0 / 50 / 100.  Mirrors
    ``chibi.vision.integration.reactions_per_hour`` (the settings windows
    depend only on their facade; a test keeps the two in step)."""
    return round(max(0, min(100, int(reaction_frequency))) * 12 / 100)


def rate_text(reaction_frequency: int) -> str:
    n = reactions_per_hour(reaction_frequency)
    if n <= 0:
        return "Про побачене в грі вона мовчатиме — хіба що ти спитаєш, що вона бачить."
    return (f"Не більше ніж {count_uk(n, 'реакція', 'реакції', 'реакцій')} на годину на те, що вона "
            "бачить у грі (коли ввімкнено «Бачення екрана»).")


class GamingFacade(Protocol):
    def settings(self) -> Settings: ...

    def apply_settings(self, settings: Settings) -> None: ...


@dataclass(frozen=True)
class _Field:
    """One ``Settings.gaming.<name>`` field edited by a widget."""

    name: str
    read: Callable[[], object]
    write: Callable[[object], None]


class GamingPage(QWidget):
    """Settings page for the gaming companion mode.  See the module docstring."""

    def __init__(self, facade: GamingFacade, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._facade = facade
        settings = self._read_settings() or Settings()
        g = settings.gaming

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 0, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(make_label("Ігровий компаньйон", "title", wrap=False))
        self.lead_label = InfoLabel("")
        layout.addWidget(self.lead_label)

        card = Card(spacing=12)
        card.body.addWidget(SectionHeader("Поки ти граєш"))
        companion = CheckRow(
            "Режим ігрового компаньйона",
            "Під час гри вона менше з'являється з бульбашками й говорить коротко, щоб не відволікати.")
        companion.setChecked(g.companion_mode)
        self.companion_check: QCheckBox = companion.checkbox
        card.body.addWidget(companion)

        self.frequency_row = SliderRow("Як часто реагувати на гру", 0, 100, g.reaction_frequency, step=5,
                                       formatter=frequency_text)
        frequency = QVBoxLayout()
        frequency.setSpacing(2)
        frequency.addWidget(self.frequency_row)
        scale = QHBoxLayout()
        scale.setContentsMargins(0, 0, 0, 0)
        scale.addWidget(make_label("тихо", "hint", wrap=False))
        scale.addStretch(1)
        scale.addWidget(make_label("балакуче", "hint", wrap=False))
        frequency.addLayout(scale)
        self.rate_label = InfoLabel("")
        frequency.addWidget(self.rate_label)
        card.body.addLayout(frequency)
        self.frequency_row.valueChanged.connect(self._update_rate)

        spoilers = CheckRow(
            "Підказки та сюжетні коментарі",
            "Дозволити їй підказувати й обговорювати сюжет. Поки вимкнено — жодних спойлерів: про те, "
            "що може зіпсувати сюрприз, вона змовчить.")
        spoilers.setChecked(g.hints_and_spoilers)
        self.spoilers_check: QCheckBox = spoilers.checkbox
        card.body.addWidget(spoilers)
        layout.addWidget(card)

        where = Card()
        where.body.addWidget(SectionHeader("Де ще налаштувати"))
        where.body.addWidget(InfoLabel(
            "Твої ігри, щоб вона їх упізнавала, додаються на сторінці «Ігри» → «Мої ігри». Які з них їй "
            "можна бачити — на сторінці «Бачення екрана». А чи говорити їй голосом, поки гра на весь "
            "екран, — на сторінці «Голос»."))
        layout.addWidget(where)
        layout.addStretch(1)

        self._fields: tuple[_Field, ...] = (
            _Field("companion_mode", self.companion_check.isChecked,
                   lambda v: self.companion_check.setChecked(bool(v))),
            _Field("reaction_frequency", self.frequency_row.value,
                   lambda v: self.frequency_row.setValue(int(v))),
            _Field("hints_and_spoilers", self.spoilers_check.isChecked,
                   lambda v: self.spoilers_check.setChecked(bool(v))),
        )
        self._loaded: dict[str, object] = self._read_fields()
        self._show_name(settings.character.name)
        self._update_rate(self.frequency_row.value())

    # --- public API ------------------------------------------------------------------------

    def collect(self, settings: Settings) -> Settings:
        """Write the fields edited on this page into *settings* (in place; also returned)."""
        for field in self._fields:
            value = field.read()
            if value != self._loaded[field.name]:
                setattr(settings.gaming, field.name, value)
        return settings

    def has_edits(self) -> bool:
        return any(field.read() != self._loaded[field.name] for field in self._fields)

    def sync(self, settings: Settings, *, edited_too: bool) -> None:
        """Show *settings*: every field after an apply, otherwise only unedited ones."""
        for field in self._fields:
            if edited_too or field.read() == self._loaded[field.name]:
                field.write(getattr(settings.gaming, field.name))
                self._loaded[field.name] = field.read()
        self._show_name(settings.character.name)

    def reset_baseline(self) -> None:
        self._loaded = self._read_fields()

    # --- internals -------------------------------------------------------------------------

    def _read_fields(self) -> dict[str, object]:
        return {field.name: field.read() for field in self._fields}

    def _read_settings(self) -> Settings | None:
        try:
            return self._facade.settings().copy()
        except Exception:
            log.exception("reading the current settings failed")
            return None

    def _show_name(self, name: str) -> None:
        self.lead_label.setText(
            f"Як {name} поводиться, поки ти граєш: скільки говорить і чи можна спойлерити.")

    def _update_rate(self, value: int) -> None:
        self.rate_label.setText(rate_text(value))
