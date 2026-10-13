"""The "Бачення екрана" page of the settings dialog.

Design notes - why it works this way:

* **Settings vs. permissions**, as on the AI page.  Mode, interval, the
  allowed games and apps, the confidence threshold, "Лише локальний аналіз"
  and the model override are ordinary ``Settings.vision`` fields: edited here
  and applied with the rest of the dialog through :meth:`VisionPage.collect`.
  The two permissions (window capture, cloud analysis) are not settings: their
  buttons go straight to the facade's :class:`PermissionManager` - granting
  only after the same explicit consent text the privacy page shows, revoking
  at once without a question.
* **Baseline**: the page remembers what each widget showed when it was last
  loaded from the app; ``collect`` writes only fields that differ, so a stale
  widget never reverts a change made elsewhere (the dialog's rule).
* **Honest about privacy**: frames are never saved, private apps are never
  captured even if listed, and "Що Мія бачила" lives in memory only - the page
  says so where the user makes the choice.  Adding an app the privacy filter
  treats as private is refused with an explanation instead of silently doing
  nothing.
* The page depends only on :class:`VisionFacade` (``chibi.app.ChibiApp``
  implements it); the view types come from :mod:`chibi.vision.integration`
  and are used by attribute only, so tests can pass any lookalike.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QHideEvent, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from chibi.core.config import Settings
from chibi.core.events import Permission
from chibi.gaming.games_db import title_from_exe
from chibi.privacy.filters import PrivacyFilter
from chibi.privacy.permissions import REQUIRES, PermissionManager
from chibi.settings.widgets import (
    Card,
    CheckRow,
    InfoLabel,
    SectionHeader,
    SliderRow,
    SubmitLineEdit,
    ask_yes_no,
    count_uk,
    format_local_time,
    make_button,
    make_label,
    percent,
    set_style_property,
    setting_row,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from chibi.vision.integration import SeenObservation, VisionStatusView

log = logging.getLogger(__name__)

ConfirmFn = Callable[[str, str], bool]

#: How often status and "Що Мія бачила" are re-read while the page is visible.
REFRESH_MS = 3000

#: (VisionSettings.mode, radio text, one-line explanation) in display order.
MODE_CHOICES: tuple[tuple[str, str, str], ...] = (
    ("off", "Вимкнено", "Вона зовсім не дивиться на екран."),
    ("manual", "Лише за запитом",
     "Дивиться, тільки коли ти просиш: «Подивись на екран» у її меню або «що ти бачиш?» у чаті."),
    ("auto", "Автоматично",
     "Час від часу поглядає на дозволені ігри й програми, поки вони на передньому плані."),
    ("game", "Ігровий режим",
     "Стежить лише за дозволеними іграми, а коли комп'ютер навантажений, дивиться рідше."),
)

SCENE_NAMES: dict[str, str] = {
    "gameplay": "гра", "menu": "меню", "inventory": "інвентар", "map": "мапа", "dialogue": "діалог",
    "loading": "завантаження", "death_screen": "екран смерті", "victory_screen": "екран перемоги",
    "cutscene": "заставка", "desktop": "робочий стіл", "other": "щось інше",
}
EVENT_NAMES: dict[str, str] = {
    "none": "нічого особливого", "player_death": "загибель", "victory": "перемога",
    "level_up": "новий рівень", "boss_fight": "бій із босом", "combat": "бій", "low_health": "мало здоров'я",
    "loot": "здобич", "achievement": "досягнення", "new_area": "нова локація", "puzzle": "головоломка",
    "other": "щось цікаве",
}

#: Short names used on this page; the consent texts come from the privacy page.
PERMISSION_TITLES: dict[Permission, tuple[str, str]] = {
    Permission.SCREEN_CAPTURE: (
        "Знімки дозволених вікон",
        "Без цього дозволу вона не бачить нічого. Знімається лише вікно дозволеної гри чи програми, "
        "яке зараз на передньому плані, — не весь екран.",
    ),
    Permission.CLOUD_VISION: (
        "Хмарний аналіз екрана",
        "Потрібен, лише якщо модель на сторінці «AI» працює в хмарі (Claude, OpenAI). Локальна "
        "модель — наприклад, Ollama на цьому ПК — обходиться без нього.",
    ),
}
#: Used if the privacy page offers no consent text for a permission.
FALLBACK_CONSENT: dict[Permission, tuple[str, str]] = {
    Permission.SCREEN_CAPTURE: (
        "Дозволити знімки дозволених вікон?",
        "Вона зможе час від часу знімати вікно гри чи програми, яку ти додаєш до списку на цій "
        "сторінці, і реагувати на те, що там відбувається.\n\n"
        "• Лише вікно дозволеної програми, не весь екран; приватні програми — ніколи.\n"
        "• Кадри не зберігаються: лише зменшена копія в пам'яті на час аналізу.\n"
        "• Хмарній моделі кадри надсилаються тільки з окремим дозволом.\n\n"
        "Відкликати можна будь-коли тут, у «Приватності» або кнопкою «Зупинити все спостереження».",
    ),
    Permission.CLOUD_VISION: (
        "Дозволити хмарний аналіз екрана?",
        "Зменшені кадри вікна дозволеної гри чи програми надсилатимуться твоєму хмарному "
        "AI-провайдеру для аналізу. На кадрі може бути видно все, що є в цьому вікні.\n\n"
        "Програма їх не зберігає; що з ними робить провайдер, визначають його умови.\n\n"
        "Без цього дозволу аналіз працює лише з локальною моделлю (наприклад, Ollama).",
    ),
}

FRAMES_BANNER = (
    "Кадри ніколи не зберігаються: зменшена копія вікна живе в пам'яті лише ті кілька секунд, поки "
    "модель її аналізує. Приватні програми — менеджери паролів, месенджери, банкінг і твої винятки "
    "з «Приватності» — не знімаються ніколи, навіть якщо додати їх сюди."
)
FULLSCREEN_NOTE = (
    "Ігри в ексклюзивному повноекранному режимі іноді віддають лише чорний кадр. Якщо вона каже, що "
    "нічого не бачить, вибери в налаштуваннях гри режим «Вікно без рамки» (borderless)."
)
LOCAL_ONLY_HINT = (
    "Кадри аналізує тільки модель на цьому комп'ютері (наприклад, Ollama), і вони нікуди не "
    "надсилаються. Без цієї позначки кадри може аналізувати й хмарна модель (Claude чи OpenAI) — але "
    "лише з окремим дозволом «Хмарний аналіз екрана» вище."
)
MODEL_HINT = (
    "Необов'язково. Порожньо — та сама модель, що й на сторінці «AI». Для Ollama потрібна модель, яка "
    "бачить зображення, наприклад gemma3 чи llava."
)

#: VisionSettings keeps only names that match this (see ``core.config``).
_APP_NAME_RE = re.compile(r"[a-z0-9_.\- ]{1,80}\.exe")


def normalize_app_name(raw: str) -> str | None:
    """``"  C:\\Tools\\OBS64.EXE "`` -> ``"obs64.exe"``; ``"obs64"`` -> ``"obs64.exe"``.

    None for anything ``VisionSettings`` would drop (it keeps Latin letters,
    digits, spaces, ``._-`` only).
    """
    name = raw.strip().strip("\"'").strip()
    name = re.split(r"[\\/]", name)[-1].strip().lower()
    if not name:
        return None
    if not name.endswith(".exe"):
        name += ".exe"
    return name if _APP_NAME_RE.fullmatch(name) else None


def interval_text(seconds: int) -> str:
    """``"30 с"``, ``"2 хв"``, ``"1 хв 30 с"`` (no-break spaces)."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}\u00a0с"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes}\u00a0хв" if not rest else f"{minutes}\u00a0хв {rest}\u00a0с"


def game_title_for(game_id: str) -> str:
    """A readable title for an allowed id the games database does not list."""
    kind, _, rest = game_id.partition(":")
    if kind in ("exe", "custom") and rest:
        return title_from_exe(rest)
    return game_id


def describe_status(view: VisionStatusView, name: str = "Мія") -> tuple[str, str]:
    """(status line, label role) for the "Зараз" card."""
    if view.mode == "off":
        return f"● Вимкнено — {name} не дивиться на екран", "hint"
    if view.blocked:
        return f"● Не працює: {view.blocked}", "warning"
    where = "" if view.local is None else (" · аналізує локальна модель" if view.local
                                           else " · аналізує хмарна модель")
    if view.state == "analyzing":
        title = f" «{view.app_title}»" if view.app_title else ""
        return f"● Роздивляється{title}…{where}", "success"
    if view.state == "watching":
        return f"● Дивиться на «{view.app_title}»{where}", "success"
    if view.state == "paused":
        return "● На паузі після помилки — спробує ще раз трохи згодом", "warning"
    if view.mode == "manual":
        return f"● Чекає, поки ти попросиш подивитися{where}", "hint"
    return f"● Чекає: зараз на екрані немає дозволеної гри чи програми{where}", "hint"


def describe_seen(item: SeenObservation) -> str:
    """``"14:05 — S.T.A.L.K.E.R. 2: гра, бій із босом · вона відреагувала"``."""
    scene = SCENE_NAMES.get(item.scene, item.scene)
    event = EVENT_NAMES.get(item.event, item.event)
    text = f"{format_local_time(item.at)} — {item.app_title or 'вікно'}: {scene}"
    if item.event != "none":
        text += f", {event}"
    if item.reacted:
        text += " · відреагувала"
    return text


def consent_for(permission: Permission) -> tuple[str, str]:
    """The consent (title, text) the privacy page uses, so both ask the same question."""
    try:
        # Imported lazily: the settings dialog imports this module.
        from chibi.settings.settings_dialog import PERMISSION_TEXTS

        text = PERMISSION_TEXTS.get(permission)
    except Exception:
        log.exception("reading the permission texts failed")
        text = None
    if text is not None and text.consent_title and text.consent_text:
        return text.consent_title, text.consent_text
    return FALLBACK_CONSENT[permission]


class VisionFacade(Protocol):
    """What the page needs from the app (``chibi.app.ChibiApp``)."""

    permissions: PermissionManager

    def settings(self) -> Settings: ...

    def apply_settings(self, settings: Settings) -> None: ...

    def vision_status(self) -> VisionStatusView: ...

    def vision_recent(self) -> Sequence[SeenObservation]:
        """What she saw recently, newest first (in memory only)."""
        ...

    def vision_clear_history(self) -> None: ...

    def vision_games(self) -> Sequence[tuple[str, str]]:
        """(game id, title) of every game she can recognise."""
        ...


@dataclass(frozen=True)
class _Field:
    """One ``Settings.vision.<name>`` field edited by a widget."""

    name: str
    read: Callable[[], object]
    write: Callable[[object], None]


class _PermissionRow(QWidget):
    def __init__(self, title: str, hint: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 4)
        layout.setSpacing(2)
        top = QHBoxLayout()
        self.title_label = make_label(title, "value", wrap=False)
        self.state_label = make_label("", "hint", wrap=False)
        self.button: QPushButton = make_button("Дозволити…", "accent")
        self.button.setAccessibleName(title)
        top.addWidget(self.title_label)
        top.addStretch(1)
        top.addWidget(self.state_label)
        top.addWidget(self.button)
        layout.addLayout(top)
        layout.addWidget(InfoLabel(hint))


class VisionPage(QWidget):
    """Settings page for screen vision.  See the module docstring."""

    #: A permission was granted or revoked here (the dialog refreshes its privacy page).
    permissions_changed = Signal()

    def __init__(self, facade: VisionFacade, confirm: ConfirmFn | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._facade = facade
        self._confirm: ConfirmFn = confirm or (lambda title, text: ask_yes_no(self, title, text))
        settings = self._read_settings() or Settings()
        self._settings = settings
        self._name = settings.character.name
        self._choices: list[tuple[str, str]] = self._read_choices()
        self._apps: list[str] = list(settings.vision.allowed_processes)
        self._status_failed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 0, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(make_label("Бачення екрана", "title", wrap=False))
        self.lead_label = InfoLabel("")
        layout.addWidget(self.lead_label)
        layout.addWidget(make_label(FRAMES_BANNER, "banner"))
        layout.addWidget(self._build_status_card())
        layout.addWidget(self._build_permissions_card())
        layout.addWidget(self._build_mode_card(settings))
        layout.addWidget(self._build_games_card(settings))
        layout.addWidget(self._build_apps_card())
        layout.addWidget(self._build_tuning_card(settings))
        layout.addWidget(self._build_model_card(settings))
        layout.addWidget(self._build_seen_card())
        layout.addWidget(InfoLabel(FULLSCREEN_NOTE))
        layout.addStretch(1)

        self._fields: tuple[_Field, ...] = self._bind_fields()
        self._loaded: dict[str, object] = self._read_fields()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(REFRESH_MS)
        self.refresh_timer.timeout.connect(self.refresh_live)

        self._show_name()
        self._render_apps()
        self._update_games_count()
        self.refresh()

    # --- public API ------------------------------------------------------------------------

    def collect(self, settings: Settings) -> Settings:
        """Write the fields edited on this page into *settings* (in place; also returned)."""
        for field in self._fields:
            value = field.read()
            if value != self._loaded[field.name]:
                setattr(settings.vision, field.name, value)
        return settings

    def has_edits(self) -> bool:
        return any(field.read() != self._loaded[field.name] for field in self._fields)

    def sync(self, settings: Settings, *, edited_too: bool) -> None:
        """Show *settings* (the app's current ones): every field after an apply,
        otherwise only those the user has not edited here."""
        self._settings = settings
        self._name = settings.character.name
        self._choices = self._read_choices()
        for field in self._fields:
            if edited_too or field.read() == self._loaded[field.name]:
                field.write(getattr(settings.vision, field.name))
                self._loaded[field.name] = field.read()
        self._show_name()
        self.refresh()

    def reset_baseline(self) -> None:
        """Treat what the widgets show now as unedited (after a successful apply)."""
        self._loaded = self._read_fields()

    def mode(self) -> str:
        return MODE_CHOICES[max(0, self.mode_group.checkedId())][0]

    def refresh(self) -> None:
        """Re-read what changes outside the page: permissions, status, observations."""
        self._refresh_permissions()
        self.refresh_live()

    def refresh_live(self) -> None:
        self._refresh_status()
        self._refresh_seen()

    # --- building --------------------------------------------------------------------------

    def _build_status_card(self) -> QWidget:
        card = Card(spacing=6)
        card.body.addWidget(SectionHeader("Зараз"))
        self.status_label = make_label("", "hint")
        self.status_label.setAccessibleName("Стан бачення екрана")
        self.note_label = make_label("", "warning")
        self.note_label.hide()
        card.body.addWidget(self.status_label)
        card.body.addWidget(self.note_label)
        return card

    def _build_permissions_card(self) -> QWidget:
        card = Card()
        card.body.addWidget(SectionHeader("Дозволи"))
        self.permission_rows: dict[Permission, _PermissionRow] = {}
        for permission, (title, hint) in PERMISSION_TITLES.items():
            row = _PermissionRow(title, hint)
            row.button.clicked.connect(lambda _checked=False, p=permission: self._on_permission_button(p))
            self.permission_rows[permission] = row
            card.body.addWidget(row)
        self.permission_message = make_label("", "hint")
        self.permission_message.hide()
        card.body.addWidget(self.permission_message)
        return card

    def _build_mode_card(self, settings: Settings) -> QWidget:
        card = Card()
        card.body.addWidget(SectionHeader("Коли дивитися"))
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons: dict[str, QRadioButton] = {}
        for index, (value, text, hint) in enumerate(MODE_CHOICES):
            radio = QRadioButton(text)
            self.mode_group.addButton(radio, index)
            self.mode_buttons[value] = radio
            card.body.addWidget(radio)
            explanation = InfoLabel(hint)
            explanation.setContentsMargins(28, 0, 0, 4)
            card.body.addWidget(explanation)
        self._set_mode(settings.vision.mode)
        return card

    def _build_games_card(self, settings: Settings) -> QWidget:
        card = Card()
        card.body.addWidget(SectionHeader("Ігри, які їй можна бачити"))
        card.body.addWidget(InfoLabel(
            "Познач ігри, вікна яких їй можна знімати. Якщо твоєї гри тут немає, додай її на сторінці "
            "«Ігри» → «Мої ігри», а тоді познач тут."))
        self.games_search = SubmitLineEdit()
        self.games_search.setPlaceholderText("Пошук гри…")
        self.games_search.setClearButtonEnabled(True)
        self.games_search.setAccessibleName("Пошук гри")
        self.games_search.textChanged.connect(self._filter_games)
        card.body.addWidget(self.games_search)
        self.games_list = QListWidget()
        self.games_list.setAccessibleName("Ігри, які їй можна бачити")
        self.games_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.games_list.setMinimumHeight(140)
        self.games_list.setMaximumHeight(220)
        self.games_list.itemChanged.connect(lambda _item: self._update_games_count())
        card.body.addWidget(self.games_list)
        self.games_count_label = make_label("", "hint")
        card.body.addWidget(self.games_count_label)
        self._show_games(settings.vision.allowed_games)
        return card

    def _build_apps_card(self) -> QWidget:
        card = Card()
        card.body.addWidget(SectionHeader("Інші програми"))
        card.body.addWidget(InfoLabel(
            "Не ігри, а програми, на які їй можна дивитися, — наприклад, графічний редактор чи "
            "відеоплеєр. Вкажи назву процесу, як-от obs64.exe."))
        self.apps_list = QListWidget()
        self.apps_list.setAccessibleName("Дозволені програми")
        self.apps_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.apps_list.setMinimumHeight(70)
        self.apps_list.setMaximumHeight(130)
        card.body.addWidget(self.apps_list)
        row = QHBoxLayout()
        self.app_edit = SubmitLineEdit()
        self.app_edit.setPlaceholderText("наприклад, obs64.exe")
        self.app_edit.setMaxLength(128)
        self.app_edit.setAccessibleName("Назва процесу програми")
        self.app_edit.submitted.connect(self._add_app)
        self.add_app_button = make_button("Додати")
        self.add_app_button.clicked.connect(self._add_app)
        self.remove_apps_button = make_button("Прибрати вибрані")
        self.remove_apps_button.clicked.connect(self._remove_apps)
        row.addWidget(self.app_edit, 1)
        row.addWidget(self.add_app_button)
        row.addWidget(self.remove_apps_button)
        card.body.addLayout(row)
        self.app_error = make_label("", "error")
        self.app_error.hide()
        card.body.addWidget(self.app_error)
        return card

    def _build_tuning_card(self, settings: Settings) -> QWidget:
        v = settings.vision
        card = Card(spacing=12)
        card.body.addWidget(SectionHeader("Як часто й наскільки впевнено"))
        self.interval_row = SliderRow(
            "Як часто дивитися", 10, 600, v.interval_seconds, step=5, formatter=interval_text,
            description="Пауза між кадрами в автоматичному та ігровому режимах. Рідше — менше "
                        "навантаження на комп'ютер і менше запитів до AI.")
        card.body.addWidget(self.interval_row)
        self.confidence_row = SliderRow(
            "Поріг упевненості", 30, 95, round(v.min_confidence * 100), step=5, formatter=percent,
            description="Вона реагує лише на події, у яких модель упевнена щонайменше настільки. Вища "
                        "планка — менше хибних реакцій, але щось вона може пропустити.")
        card.body.addWidget(self.confidence_row)
        adapt = CheckRow("Дивитися рідше, коли комп'ютер навантажений",
                         "В ігровому режимі: щоб гра не гальмувала через неї.")
        adapt.setChecked(v.adapt_to_load)
        self.adapt_check: QCheckBox = adapt.checkbox
        card.body.addWidget(adapt)
        return card

    def _build_model_card(self, settings: Settings) -> QWidget:
        v = settings.vision
        card = Card()
        card.body.addWidget(SectionHeader("Модель"))
        local = CheckRow("Лише локальний аналіз", LOCAL_ONLY_HINT)
        local.setChecked(v.local_only)
        self.local_only_check: QCheckBox = local.checkbox
        card.body.addWidget(local)
        self.model_edit = QLineEdit(v.model)
        self.model_edit.setMaxLength(120)
        self.model_edit.setPlaceholderText("як на сторінці «AI»")
        self.model_edit.setMinimumWidth(220)
        self.model_edit.setAccessibleName("Окрема модель для зору")
        card.body.addWidget(setting_row("Окрема модель для зору", self.model_edit, MODEL_HINT))
        return card

    def _build_seen_card(self) -> QWidget:
        card = Card()
        self.seen_header = SectionHeader("")
        card.body.addWidget(self.seen_header)
        card.body.addWidget(InfoLabel(
            "Лише в пам'яті: список зникає після перезапуску, а самі кадри не зберігаються ніколи. "
            "Описи сцен бачить тільки AI під час розмови."))
        self.seen_list = QListWidget()
        self.seen_list.setAccessibleName("Що вона бачила нещодавно")
        self.seen_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.seen_list.setMinimumHeight(90)
        self.seen_list.setMaximumHeight(170)
        card.body.addWidget(self.seen_list)
        row = QHBoxLayout()
        row.addStretch(1)
        self.clear_seen_button = make_button("Очистити", "danger")
        self.clear_seen_button.clicked.connect(self._clear_seen)
        row.addWidget(self.clear_seen_button)
        card.body.addLayout(row)
        return card

    # --- fields ------------------------------------------------------------------------------

    def _bind_fields(self) -> tuple[_Field, ...]:
        def check(name: str, box: QCheckBox) -> _Field:
            return _Field(name, box.isChecked, lambda v: box.setChecked(bool(v)))

        return (
            _Field("mode", self.mode, lambda v: self._set_mode(str(v))),
            _Field("interval_seconds", self.interval_row.value, lambda v: self.interval_row.setValue(int(v))),
            _Field("allowed_games", self._checked_games, self._show_games),
            _Field("allowed_processes", lambda: list(self._apps), self._show_apps),
            _Field("min_confidence", lambda: round(self.confidence_row.value() / 100, 2),
                   lambda v: self.confidence_row.setValue(round(float(v) * 100))),
            check("local_only", self.local_only_check),
            _Field("model", lambda: self.model_edit.text().strip()[:120],
                   lambda v: self.model_edit.setText(str(v))),
            check("adapt_to_load", self.adapt_check),
        )

    def _read_fields(self) -> dict[str, object]:
        return {field.name: field.read() for field in self._fields}

    def _read_settings(self) -> Settings | None:
        try:
            return self._facade.settings().copy()
        except Exception:
            log.exception("reading the current settings failed")
            return None

    def _read_choices(self) -> list[tuple[str, str]]:
        try:
            choices = [(str(gid), str(title)) for gid, title in self._facade.vision_games()]
        except Exception:
            log.exception("reading the games list failed")
            return []
        return sorted(dict(choices).items(), key=lambda item: item[1].casefold())

    def _show_name(self) -> None:
        self.lead_label.setText(
            f"За твоїм дозволом {self._name} може поглядати на вікна ігор і програм, які ти вибереш "
            "нижче, і реагувати на те, що там відбувається: загибель, перемогу, боса.")
        self.seen_header.setText(f"Що {self._name} бачила")

    def _set_mode(self, mode: str) -> None:
        values = [value for value, _text, _hint in MODE_CHOICES]
        index = values.index(mode) if mode in values else 0
        button = self.mode_group.button(index)
        if button is not None:
            button.setChecked(True)

    # --- games -------------------------------------------------------------------------------

    def _show_games(self, value: object) -> None:
        allowed = [str(g) for g in value] if isinstance(value, list) else []
        known = {gid for gid, _title in self._choices}
        extra = [(gid, game_title_for(gid)) for gid in allowed if gid not in known]
        self.games_list.blockSignals(True)
        try:
            self.games_list.clear()
            for gid, title in [*self._choices, *extra]:
                item = QListWidgetItem(title)
                item.setData(Qt.ItemDataRole.UserRole, gid)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if gid in allowed else Qt.CheckState.Unchecked)
                self.games_list.addItem(item)
        finally:
            self.games_list.blockSignals(False)
        self._filter_games(self.games_search.text())
        self._update_games_count()

    def _checked_games(self) -> list[str]:
        result = []
        for row in range(self.games_list.count()):
            item = self.games_list.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                result.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return result

    def _filter_games(self, text: str) -> None:
        needle = text.strip().casefold()
        for row in range(self.games_list.count()):
            item = self.games_list.item(row)
            gid = str(item.data(Qt.ItemDataRole.UserRole))
            item.setHidden(bool(needle) and needle not in item.text().casefold() and needle not in gid)

    def _update_games_count(self) -> None:
        n = len(self._checked_games())
        if not self.games_list.count():
            text = "Список ігор порожній — додай свої на сторінці «Ігри»."
        elif n:
            text = f"Вибрано {count_uk(n, 'гру', 'гри', 'ігор')}."
        else:
            text = "Поки не вибрано жодної гри."
        self.games_count_label.setText(text)

    # --- apps --------------------------------------------------------------------------------

    def _show_apps(self, value: object) -> None:
        self._apps = [str(a) for a in value] if isinstance(value, list) else []
        self._render_apps()

    def _render_apps(self) -> None:
        self.apps_list.clear()
        for name in self._apps:
            self.apps_list.addItem(name)

    def _app_error(self, text: str) -> None:
        self.app_error.setText(text)
        self.app_error.setVisible(bool(text))

    def _add_app(self) -> None:
        raw = self.app_edit.text()
        if not raw.strip():
            return
        name = normalize_app_name(raw)
        if name is None:
            self._app_error("Це не схоже на назву програми: лише латинські літери, цифри, пробіл, "
                            "крапка, дефіс чи підкреслення. Приклад: obs64.exe")
            return
        try:
            private = PrivacyFilter(self._settings.activity.excluded_processes).is_private_process(name)
        except Exception:
            log.exception("privacy check failed")
            private = True
        if private:
            self._app_error("Це приватна програма — її вона не бачитиме ніколи, тож додавати її немає сенсу.")
            return
        self._app_error("")
        if name not in self._apps:
            self._apps.append(name)
            self._render_apps()
        self.app_edit.clear()

    def _remove_apps(self) -> None:
        selected = {item.text() for item in self.apps_list.selectedItems()}
        if selected:
            self._apps = [a for a in self._apps if a not in selected]
            self._render_apps()

    # --- permissions ---------------------------------------------------------------------------

    def _refresh_permissions(self) -> None:
        manager = self._facade.permissions
        try:
            snapshot = manager.snapshot()
        except Exception:
            log.exception("reading the permissions failed")
            snapshot = {}
        for permission, row in self.permission_rows.items():
            granted = bool(snapshot.get(permission))
            base = REQUIRES.get(permission)
            base_ok = base is None or bool(snapshot.get(base))
            try:
                available = bool(manager.is_available(permission))
            except Exception:
                available = False
            if granted:
                row.state_label.setText("✓ дозволено")
                set_style_property(row.state_label, "role", "success")
                row.button.setText("Відкликати")
                set_style_property(row.button, "variant", "danger")
                row.button.setEnabled(True)  # revoking is always possible
            else:
                if not available:
                    row.state_label.setText("недоступно в цій версії")
                elif not base_ok:
                    row.state_label.setText("спершу дозволь знімки вікон")
                else:
                    row.state_label.setText("не дозволено")
                set_style_property(row.state_label, "role", "hint")
                row.button.setText("Дозволити…")
                set_style_property(row.button, "variant", "accent")
                row.button.setEnabled(available and base_ok)

    def _on_permission_button(self, permission: Permission) -> None:
        manager = self._facade.permissions
        if manager.is_granted(permission):
            manager.revoke(permission)
            self._permission_message("Дозвіл відкликано — він перестав діяти одразу.", error=False)
        else:
            title, text = consent_for(permission)
            if not self._confirm(title, text):
                return
            try:
                manager.grant(permission)
            except (PermissionError, ValueError) as exc:  # unavailable / base permission missing
                log.warning("could not grant %s: %s", permission.value, type(exc).__name__)
                self._permission_message("Цей дозвіл зараз не можна надати.", error=True)
                self.refresh()
                return
            self._permission_message("Дозвіл надано.", error=False)
        self.refresh()
        self.permissions_changed.emit()

    def _permission_message(self, text: str, *, error: bool) -> None:
        self.permission_message.setText(text)
        set_style_property(self.permission_message, "role", "error" if error else "success")
        self.permission_message.setVisible(bool(text))

    # --- live state ------------------------------------------------------------------------------

    def _refresh_status(self) -> None:
        try:
            view = self._facade.vision_status()
        except Exception:
            if not self._status_failed:  # polled: log the first failure only
                log.exception("reading the vision status failed")
            self._status_failed = True
            self.status_label.setText("● Стан бачення зараз невідомий")
            set_style_property(self.status_label, "role", "warning")
            self.note_label.hide()
            return
        self._status_failed = False
        text, role = describe_status(view, self._name)
        self.status_label.setText(text)
        set_style_property(self.status_label, "role", role)
        self.note_label.setText(view.note)
        self.note_label.setVisible(bool(view.note))

    def _refresh_seen(self) -> None:
        try:
            items = list(self._facade.vision_recent())
        except Exception:
            log.exception("reading recent observations failed")
            items = []
        self.seen_list.clear()
        if not items:
            self.seen_list.addItem("Поки що нічого.")
        for item in items:
            self.seen_list.addItem(describe_seen(item))
        self.clear_seen_button.setEnabled(bool(items))

    def _clear_seen(self) -> None:
        try:
            self._facade.vision_clear_history()
        except Exception:
            log.exception("clearing the vision history failed")
        self._refresh_seen()

    # --- visibility-driven polling -------------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.refresh()
        self.refresh_timer.start()

    def hideEvent(self, event: QHideEvent) -> None:
        self.refresh_timer.stop()
        super().hideEvent(event)
