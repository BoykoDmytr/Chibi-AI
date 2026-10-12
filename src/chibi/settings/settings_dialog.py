"""The settings dialog: character, personality, behaviour, privacy, games, AI.

Design notes:

* Everything except permissions works on a copy of the settings and reaches
  the app only through ``facade.apply_settings`` ("Застосувати" / "Гаразд");
  "Скасувати" throws the copy away.
* Permissions are *not* settings: they go straight to the PermissionManager the
  moment the user flips them, because privacy changes must never wait for an
  "Apply" click.  Granting always asks for explicit consent through
  ``confirm(title, text)``; revoking never asks.
* Features that don't exist in this build are shown (so the roadmap is honest)
  but disabled - the manager would refuse them anyway.
* ``confirm`` is injectable so tests (and future automation) don't need modal
  message boxes.
* The dialog is not modal: while it is open the tray, the character's menu and
  Ctrl+wheel keep changing settings (DND, lock, scale...).  Applying therefore
  writes only the fields the user edited *in the dialog* (widget value differs
  from what the widget was last loaded with) onto the app's current settings,
  so a stale widget never undoes a change made elsewhere.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from chibi.core.config import BehaviorSettings, CharacterSettings, Settings
from chibi.core.events import Permission
from chibi.memory.repositories import GameTotals, PermissionRecord
from chibi.privacy.permissions import AVAILABLE, REQUIRES
from chibi.settings.facade import AppFacade
from chibi.settings.widgets import (
    Card,
    CheckRow,
    InfoLabel,
    SectionHeader,
    SliderRow,
    SubmitLineEdit,
    count_uk,
    format_duration,
    format_local_date,
    format_local_datetime,
    make_button,
    make_label,
    percent,
    scroll_page,
    set_style_property,
)

log = logging.getLogger(__name__)

ConfirmFn = Callable[[str, str], bool]

#: (key, navigation title) in display order; keys are accepted by ``show_page``.
PAGES: tuple[tuple[str, str], ...] = (
    ("character", "Персонаж"),
    ("personality", "Характер"),
    ("behavior", "Поведінка"),
    ("privacy", "Приватність"),
    ("games", "Ігри"),
    ("ai", "AI та голос"),
)

#: Short permission names for the history list.
PERMISSION_NAMES: dict[Permission, str] = {
    Permission.APP_METADATA: "Назви програм та ігор",
    Permission.WINDOW_TITLES: "Заголовки вікон",
    Permission.SCREEN_CAPTURE: "Знімки екрана",
    Permission.CLOUD_VISION: "Хмарний зір",
    Permission.CLOUD_AI: "Хмарний AI",
    Permission.MICROPHONE: "Мікрофон",
}


@dataclass(frozen=True)
class PermissionText:
    title: str
    hint: str
    #: phase in which the feature arrives; None = available now
    phase: int | None = None
    consent_title: str = ""
    consent_text: str = ""


PERMISSION_TEXTS: dict[Permission, PermissionText] = {
    Permission.APP_METADATA: PermissionText(
        title="Бачити назви активних програм та ігор",
        hint=(
            "Що збирається: назва процесу активного вікна (наприклад, code.exe) і її категорія — гра, "
            "програмування, відео тощо, — а також скільки часу ти не торкаєшся миші й клавіатури "
            "(лише час, без самих натискань). Нічого з цього не зберігається, окрім ігрових сесій. "
            "Приватні програми (менеджери паролів, месенджери, банкінг і твої винятки нижче) для неї "
            "виглядають просто як «щось приватне»."
        ),
        consent_title="Дозволити бачити програми?",
        consent_text=(
            "Дозволити бачити, які програми та ігри в тебе відкриті?\n\n"
            "Вона знатиме назву процесу активного вікна та його категорію, а також скільки часу ти не "
            "користуєшся мишею й клавіатурою (без самих натискань). Ці дані потрібні лише для реакцій "
            "і не записуються, окрім ігрових сесій (яка гра, коли почалася, скільки тривала).\n\n"
            "Відкликати дозвіл можна будь-коли тут або кнопкою «Зупинити все спостереження»."
        ),
    ),
    Permission.WINDOW_TITLES: PermissionText(
        title="Читати заголовки вікон",
        hint=(
            "Лише в пам'яті й лише для того, щоб відрізнити, наприклад, YouTube від інших сайтів у "
            "браузері. Заголовки ніколи не записуються на диск і нікуди не надсилаються. "
            "Потребує дозволу «Бачити назви активних програм та ігор»."
        ),
        consent_title="Дозволити читати заголовки вікон?",
        consent_text=(
            "Заголовок активного вікна часто містить назву сайту, документа чи відео.\n\n"
            "Він використовується лише в пам'яті, щоб краще зрозуміти, чим ти займаєшся, і одразу "
            "відкидається. Заголовки ніколи не записуються на диск і нікуди не надсилаються, а вікна "
            "в режимі інкогніто та приватні програми ігноруються.\n\n"
            "Відкликати дозвіл можна будь-коли."
        ),
    ),
    Permission.SCREEN_CAPTURE: PermissionText(
        title="Бачити екран",
        hint="Щоб розуміти, що відбувається на екрані. Лише за твоїм окремим дозволом.",
        phase=3,
    ),
    Permission.CLOUD_VISION: PermissionText(
        title="Надсилати знімки екрана хмарному AI",
        hint="Для розпізнавання зображення хмарною моделлю. Потребує дозволу «Бачити екран».",
        phase=3,
    ),
    Permission.CLOUD_AI: PermissionText(
        title="Надсилати текст розмов хмарному AI",
        hint="Для розмов через OpenAI чи Anthropic. Локальна модель (Ollama) цього не потребуватиме.",
        phase=2,
    ),
    Permission.MICROPHONE: PermissionText(
        title="Слухати мікрофон",
        hint="Для голосових розмов.",
        phase=4,
    ),
}

#: Settings.personality field -> (slider title, what it changes)
PERSONALITY_SLIDERS: tuple[tuple[str, str, str], ...] = (
    ("humor", "Гумор", "Як часто вона жартує й обирає грайливі репліки."),
    ("sarcasm", "Сарказм", "Наскільки колючими бувають її підколки."),
    ("talkativeness", "Балакучість", "Наскільки охоче вона сама заводить розмову."),
    ("curiosity", "Цікавість", "Як часто розпитує, чим ти займаєшся, і роздивляється довкола."),
    ("affection", "Прив'язаність", "Наскільки вона ніжна й турботлива з тобою."),
    ("independence", "Незалежність", "Як часто займається своїми справами: гуляє, сидить, дрімає."),
    (
        "commentary_frequency",
        "Частота коментарів",
        "Як часто коментує ігри та програми, які ти відкриваєш (якщо це дозволено в «Приватності»).",
    ),
)

#: Settings.activity field -> check box text
COMMENT_TOGGLES: tuple[tuple[str, str], ...] = (
    ("comment_on_games", "Коментувати ігри"),
    ("comment_on_programming", "Коментувати програмування"),
    ("comment_on_video", "Коментувати перегляд відео"),
    ("comment_on_browsing", "Коментувати перегляд сайтів"),
)

RENDERERS: tuple[tuple[str, str], ...] = (("sprite", "Спрайти"), ("vector", "Векторний (запасний)"))

CLICK_THROUGH_HINT = (
    "Увага: поки це ввімкнено, її не можна клацнути чи перетягнути. Вимкнути можна з меню в треї "
    "(значок біля годинника)."
)
#: Without a tray her own menu is the only way back, and click-through would block it.
CLICK_THROUGH_NO_TRAY_HINT = (
    "Недоступно: на цьому робочому столі немає системного трею, а без нього цей режим потім не "
    "було б чим вимкнути."
)

WIPE_HISTORY_TEXT = (
    "Буде видалено збережену історію: ігрові сесії та сказані репліки. Налаштування, дозволи та "
    "історія їхніх змін, а також стан персонажа залишаться.\n\nЦю дію не можна скасувати."
)
WIPE_ALL_TEXT = (
    "Буде видалено все: налаштування, дозволи та історію їхніх змін, історію ігор і реплік, стан "
    "персонажа, журнали роботи програми й поточну розмову в чаті. Програма повернеться до стану "
    "першого запуску.\n\nЦю дію не можна скасувати."
)

_BAD_PROCESS_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')
MAX_PROCESS_NAME = 128
MAX_GAME_TITLE = 64


def normalize_process_name(raw: str) -> str | None:
    """``"  C:\\Games\\MyBank.EXE "`` -> ``"mybank.exe"``; ``"Steam"`` -> ``"steam.exe"``.

    Returns None for input that can't be a Windows executable name.
    """
    name = raw.strip().strip("\"'").strip()
    name = re.split(r"[\\/]", name)[-1].strip().lower()
    if not name or name in (".", "..") or _BAD_PROCESS_CHARS.search(name):
        return None
    if not name.endswith(".exe"):
        name += ".exe"
    if name == ".exe" or len(name) > MAX_PROCESS_NAME:
        return None
    return name


def open_folder(path: Path) -> bool:
    """Open *path* in the system file manager (module-level so tests can replace it)."""
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


@dataclass(frozen=True)
class _Field:
    """One settings field edited by a widget: ``Settings.<section>.<name>``."""

    section: str
    name: str
    read: Callable[[], object]  # the widget's value, as it would be stored
    write: Callable[[object], None]  # show a stored value in the widget


class SettingsDialog(QDialog):
    def __init__(self, facade: AppFacade, parent: QWidget | None = None,
                 confirm: ConfirmFn | None = None) -> None:
        super().__init__(parent)
        self._facade = facade
        self._confirm: ConfirmFn = confirm or self._ask
        self._settings: Settings = facade.settings().copy()
        self._has_tray = self._read_has_tray()
        self._exclusions: list[str] = list(self._settings.activity.excluded_processes)
        self._custom_games: list[dict[str, str]] = self._custom_games_from(
            self._settings.activity.custom_games)
        name = self._settings.character.name
        self.setWindowTitle(f"Налаштування — {name}")
        self.setModal(False)
        self.setMinimumSize(720, 520)
        self.resize(860, 620)
        self.setSizeGripEnabled(True)

        self.nav = QListWidget(self)
        self.nav.setObjectName("navList")
        self.nav.setFixedWidth(200)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.stack = QStackedWidget(self)
        builders = {
            "character": self._build_character_page,
            "personality": self._build_personality_page,
            "behavior": self._build_behavior_page,
            "privacy": self._build_privacy_page,
            "games": self._build_games_page,
            "ai": self._build_ai_page,
        }
        self._page_keys: list[str] = []
        for key, title in PAGES:
            self.nav.addItem(QListWidgetItem(title))
            self.stack.addWidget(scroll_page(builders[key]()))
            self._page_keys.append(key)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        body = QHBoxLayout()
        body.setSpacing(16)
        body.addWidget(self.nav)
        body.addWidget(self.stack, 1)

        self.status_label = make_label("", "hint", wrap=True, parent=self)
        self.cancel_button = make_button("Скасувати", parent=self)
        self.apply_button = make_button("Застосувати", "accent", parent=self)
        self.ok_button = make_button("Гаразд", "primary", parent=self)
        self.ok_button.setDefault(True)
        self.cancel_button.clicked.connect(self.reject)
        self.apply_button.clicked.connect(self.apply)
        self.ok_button.clicked.connect(self._on_ok)
        buttons = QHBoxLayout()
        buttons.addWidget(self.status_label, 1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.ok_button)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 14)
        root.setSpacing(12)
        root.addLayout(body, 1)
        root.addLayout(buttons)

        self._fields: tuple[_Field, ...] = self._bind_fields()
        #: What each widget showed when it was last loaded from the app's settings;
        #: a widget whose value differs from this was edited by the user.
        self._loaded: dict[tuple[str, str], object] = self._read_fields()
        self.refresh()

    # --- public API ------------------------------------------------------------------------

    def show_page(self, key: str) -> None:
        """Switch to a page by key (see :data:`PAGES`), e.g. ``"privacy"`` from the tray menu."""
        if key in self._page_keys:
            self.nav.setCurrentRow(self._page_keys.index(key))

    def current_page(self) -> str:
        return self._page_keys[self.stack.currentIndex()]

    def refresh(self) -> None:
        """Re-read state that can change outside the dialog: permissions, history,
        games, and the settings fields the user has not edited here."""
        self._sync_from_app(edited_too=False)
        self._refresh_permissions()
        self._refresh_history()
        self._refresh_games()

    def apply(self) -> bool:
        """Validate and hand the edited settings to the app.  Returns False if rejected."""
        name = self.name_edit.text().strip()
        if not name:
            self.name_error.setText("Ім'я не може бути порожнім.")
            self.name_error.show()
            self.show_page("character")
            self.name_edit.setFocus()
            self._set_status("Ім'я не може бути порожнім.", error=True)
            return False
        self.name_error.hide()
        settings = self._collect()
        try:
            self._facade.apply_settings(settings)
        except Exception:
            log.exception("applying settings failed")
            self._set_status("Не вдалося зберегти налаштування.", error=True)
            return False
        # From now on the widgets show what the app holds (validated, plus what
        # changed elsewhere) and count as unedited, so the next Apply can't revert it.
        if not self._sync_from_app(edited_too=True):
            self._settings = settings
            self._loaded = self._read_fields()
            self.setWindowTitle(f"Налаштування — {settings.character.name}")
        self._set_status("Збережено ✓")
        return True

    # --- building pages --------------------------------------------------------------------

    @staticmethod
    def _page(title: str, lead: str = "") -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 0, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(make_label(title, "title", wrap=False))
        if lead:
            layout.addWidget(InfoLabel(lead))
        return page, layout

    def _build_character_page(self) -> QWidget:
        ch: CharacterSettings = self._settings.character
        page, layout = self._page(
            "Персонаж", "Як її звати, якого вона розміру і як поводиться на робочому столі.")

        names = Card()
        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(6)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.name_edit = QLineEdit(ch.name)
        self.name_edit.setMaxLength(24)
        self.name_edit.setPlaceholderText("Мія")
        self.name_error = make_label("", "error")
        self.name_error.hide()
        name_box = QVBoxLayout()
        name_box.setSpacing(4)
        name_box.addWidget(self.name_edit)
        name_box.addWidget(self.name_error)
        form.addRow("Ім'я", name_box)
        self.nickname_edit = QLineEdit(ch.user_nickname)
        self.nickname_edit.setMaxLength(32)
        self.nickname_edit.setPlaceholderText("наприклад, Сашку")
        nick_box = QVBoxLayout()
        nick_box.setSpacing(4)
        nick_box.addWidget(self.nickname_edit)
        nick_box.addWidget(InfoLabel(
            "Напиши так, як хочеш, щоб вона до тебе зверталася, — у кличному відмінку: «Сашку», «Олю», "
            "«пане капітане». Якщо залишити порожнім, вона просто звертатиметься на «ти»."
        ))
        nick_label = make_label("Як вона звертається до тебе", wrap=True)
        nick_label.setMaximumWidth(170)
        form.addRow(nick_label, nick_box)
        names.body.addLayout(form)
        layout.addWidget(names)

        look = Card()
        look.body.addWidget(SectionHeader("Вигляд"))
        self.scale_row = SliderRow("Розмір", 50, 250, round(ch.scale * 100), step=5, formatter=percent,
                                   description="100\u00a0% — звичайний розмір.")
        look.body.addWidget(self.scale_row)
        renderer_row = QHBoxLayout()
        renderer_row.addWidget(make_label("Спосіб малювання", wrap=False))
        self.renderer_combo = QComboBox()
        for value, text in RENDERERS:
            self.renderer_combo.addItem(text, value)
        index = self.renderer_combo.findData(ch.renderer)
        self.renderer_combo.setCurrentIndex(max(0, index))
        renderer_row.addWidget(self.renderer_combo, 1)
        look.body.addLayout(renderer_row)
        look.body.addWidget(InfoLabel(
            "Векторний вигляд — простіша намальована версія на випадок, якщо спрайти не завантажилися."
        ))
        layout.addWidget(look)

        desk = Card()
        desk.body.addWidget(SectionHeader("На робочому столі"))
        self.always_on_top_check = self._check_row(desk, "Завжди поверх інших вікон", "", ch.always_on_top)
        self.lock_position_check = self._check_row(
            desk, "Закріпити на місці", "Її не можна буде перетягнути мишею.", ch.lock_position)
        self.walking_check = self._check_row(
            desk, "Гуляти робочим столом", "Сама ходить уздовж панелі завдань.", ch.walking_enabled)
        self.gravity_check = self._check_row(
            desk, "Падати вниз після перетягування",
            "Якщо відпустити її в повітрі, вона повернеться на панель завдань.", ch.gravity)
        self.click_through_check = self._check_row(
            desk, "Пропускати кліки крізь неї",
            CLICK_THROUGH_HINT if self._has_tray else CLICK_THROUGH_NO_TRAY_HINT, ch.click_through)
        self.click_through_check.setEnabled(self._has_tray)
        self.bubbles_check = self._check_row(
            desk, "Показувати бульбашки з репліками", "", ch.show_bubbles)
        layout.addWidget(desk)
        layout.addStretch(1)
        return page

    def _build_personality_page(self) -> QWidget:
        page, layout = self._page("Характер")
        note = make_label(
            "Поки AI не підключено (фаза 1), ці повзунки впливають на те, які з готових реплік вона "
            "обирає і як часто говорить. У фазі 2 вони визначатимуть характер відповідей AI.",
            "banner",
        )
        layout.addWidget(note)
        card = Card(spacing=14)
        self.personality_rows: dict[str, SliderRow] = {}
        for field_name, title, description in PERSONALITY_SLIDERS:
            row = SliderRow(title, 0, 100, getattr(self._settings.personality, field_name),
                            description=description)
            self.personality_rows[field_name] = row
            card.body.addWidget(row)
        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _build_behavior_page(self) -> QWidget:
        b: BehaviorSettings = self._settings.behavior
        page, layout = self._page("Поведінка", "Скільки вона говорить без запиту і коли відпочиває.")

        talk = Card()
        talk.body.addWidget(SectionHeader("Репліки"))
        self.dnd_check = self._check_row(
            talk, "Не турбувати",
            "Вона не заговорить сама — лише відповідатиме, коли ти до неї звертаєшся.", b.do_not_disturb)
        self.max_per_hour_spin = self._spin(0, 30, b.max_unsolicited_per_hour, "")
        talk.body.addWidget(self._setting_row(
            "Максимум реплік без запиту на годину", self.max_per_hour_spin,
            "0 — мовчить, доки ти до неї не звернешся."))
        self.min_gap_spin = self._spin(20, 3600, b.min_seconds_between_unsolicited, " с")
        talk.body.addWidget(self._setting_row("Мінімальна пауза між репліками", self.min_gap_spin))
        self.quiet_minutes_spin = self._spin(5, 600, b.quiet_minutes, " хв")
        talk.body.addWidget(self._setting_row(
            "Тривалість «помовч»", self.quiet_minutes_spin,
            "Скільки вона мовчатиме, якщо попросиш її помовчати."))
        layout.addWidget(talk)

        rest = Card()
        rest.body.addWidget(SectionHeader("Відпочинок і ресурси"))
        self.hide_fullscreen_check = self._check_row(
            rest, "Ховатися, коли відкрито повноекранну програму",
            "Ігри, відео, презентації. Для цього не потрібен жоден дозвіл: перевіряється лише розмір "
            "вікна.", b.hide_on_fullscreen)
        self.eco_check = self._check_row(
            rest, "Автоматичний економ-режим",
            "Менше анімації й без прогулянок, коли процесор перевантажений або ввімкнено економію заряду.",
            b.auto_eco_mode)
        self.sleep_after_spin = self._spin(2, 240, b.sleep_after_idle_minutes, " хв")
        rest.body.addWidget(self._setting_row(
            "Засинати через", self.sleep_after_spin,
            "Скільки часу без спілкування з нею, а якщо дозволено бачити програми, — то й без твоєї "
            "активності за комп'ютером."))
        layout.addWidget(rest)
        layout.addStretch(1)
        return page

    def _build_privacy_page(self) -> QWidget:
        name = self._settings.character.name
        page, layout = self._page(
            "Приватність",
            f"За замовчуванням усе вимкнено: {name} нічого не бачить і нічого не записує, поки ти цього "
            "не дозволиш. Кожен дозвіл можна відкликати будь-коли — він перестає діяти одразу.",
        )
        layout.addWidget(make_label(
            "Ніколи не зберігаються: знімки екрана, натискання клавіш, заголовки вікон. На диск "
            "записуються лише ігрові сесії: яка гра, коли почалася і скільки тривала.",
            "banner",
        ))

        perms = Card()
        perms.body.addWidget(SectionHeader("Дозволи"))
        self.activity_unsupported_label = make_label(
            "На цій системі відстеження програм не працює (лише Windows), тож ці дозволи нічого не "
            "змінять.", "warning")
        self.activity_unsupported_label.hide()
        perms.body.addWidget(self.activity_unsupported_label)
        self.permission_checks: dict[Permission, QCheckBox] = {}
        for permission in (Permission.APP_METADATA, Permission.WINDOW_TITLES):
            perms.body.addWidget(self._permission_row(permission))
        perms.body.addSpacing(4)
        perms.body.addWidget(SectionHeader("Згодом"))
        for permission in (Permission.CLOUD_AI, Permission.SCREEN_CAPTURE, Permission.CLOUD_VISION,
                           Permission.MICROPHONE):
            perms.body.addWidget(self._permission_row(permission))
        layout.addWidget(perms)

        stop = Card()
        self.kill_switch_button = make_button("Зупинити все спостереження", "danger", big=True)
        self.kill_switch_button.clicked.connect(self._stop_all)
        stop.body.addWidget(self.kill_switch_button)
        stop.body.addWidget(InfoLabel(
            "Миттєво відкликає всі дозволи на спостереження. Увімкнути їх знову можна лише вручну."))
        layout.addWidget(stop)

        excl = Card()
        excl.body.addWidget(SectionHeader("Мої приватні програми"))
        excl.body.addWidget(InfoLabel(
            "Програми, назви яких вона ніколи не бачитиме й не коментуватиме, — на додачу до вбудованого "
            "списку (менеджери паролів, месенджери, банкінг). Вкажи назву процесу, наприклад "
            "mybank.exe. Зміни набудуть чинності після «Застосувати»."))
        self.exclusions_list = QListWidget()
        self.exclusions_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.exclusions_list.setMinimumHeight(96)
        self.exclusions_list.setMaximumHeight(150)
        excl.body.addWidget(self.exclusions_list)
        add_row = QHBoxLayout()
        self.exclusion_edit = SubmitLineEdit()
        self.exclusion_edit.setPlaceholderText("наприклад, mybank.exe")
        self.exclusion_edit.setMaxLength(MAX_PROCESS_NAME)
        self.exclusion_edit.submitted.connect(self._add_exclusion)
        self.add_exclusion_button = make_button("Додати")
        self.add_exclusion_button.clicked.connect(self._add_exclusion)
        self.remove_exclusion_button = make_button("Прибрати вибрані")
        self.remove_exclusion_button.clicked.connect(self._remove_exclusions)
        add_row.addWidget(self.exclusion_edit, 1)
        add_row.addWidget(self.add_exclusion_button)
        add_row.addWidget(self.remove_exclusion_button)
        excl.body.addLayout(add_row)
        self.exclusion_error = make_label("", "error")
        self.exclusion_error.hide()
        excl.body.addWidget(self.exclusion_error)
        layout.addWidget(excl)
        self._render_exclusions()

        hist = Card()
        hist.body.addWidget(SectionHeader("Історія дозволів"))
        self.history_list = QListWidget()
        self.history_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.history_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.history_list.setMinimumHeight(90)
        self.history_list.setMaximumHeight(160)
        hist.body.addWidget(self.history_list)
        layout.addWidget(hist)

        data = Card()
        data.body.addWidget(SectionHeader("Твої дані"))
        data.body.addWidget(InfoLabel("Усе зберігається лише на цьому комп'ютері, у теці:"))
        self.data_dir_label = make_label(str(self._facade.data_dir()), "value")
        self.data_dir_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        data.body.addWidget(self.data_dir_label)
        data_buttons = QHBoxLayout()
        self.open_data_dir_button = make_button("Відкрити теку")
        self.open_data_dir_button.clicked.connect(self._open_data_dir)
        self.wipe_history_button = make_button("Видалити історію", "danger")
        self.wipe_history_button.clicked.connect(self._wipe_history)
        self.wipe_all_button = make_button("Видалити всі дані", "danger")
        self.wipe_all_button.clicked.connect(self._wipe_all)
        data_buttons.addWidget(self.open_data_dir_button)
        data_buttons.addStretch(1)
        data_buttons.addWidget(self.wipe_history_button)
        data_buttons.addWidget(self.wipe_all_button)
        data.body.addLayout(data_buttons)
        layout.addWidget(data)
        layout.addStretch(1)
        return page

    def _build_games_page(self) -> QWidget:
        a = self._settings.activity
        name = self._settings.character.name
        page, layout = self._page("Ігри", "Що вона помічає й коментує, коли ти граєш чи працюєш.")

        comments = Card()
        comments.body.addWidget(SectionHeader("Коментарі"))
        self.monitoring_note = make_label("", "warning")
        comments.body.addWidget(self.monitoring_note)
        self.comment_checks: dict[str, QCheckBox] = {}
        for field_name, text in COMMENT_TOGGLES:
            self.comment_checks[field_name] = self._check_row(comments, text, "", getattr(a, field_name))
        layout.addWidget(comments)

        custom = Card()
        custom.body.addWidget(SectionHeader("Мої ігри"))
        self.known_games_label = make_label(
            f"{name} впізнає {self._known_games_text()} за назвою процесу. Якщо твоєї гри немає серед них, "
            "додай її тут.", "hint")
        custom.body.addWidget(self.known_games_label)
        self.custom_games_table = self._table(("Процес", "Назва"), stretch=(0, 1))
        self.custom_games_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.custom_games_table.setMinimumHeight(110)
        self.custom_games_table.setMaximumHeight(170)
        custom.body.addWidget(self.custom_games_table)
        add_row = QHBoxLayout()
        self.custom_process_edit = SubmitLineEdit()
        self.custom_process_edit.setPlaceholderText("mygame.exe")
        self.custom_process_edit.setMaxLength(MAX_PROCESS_NAME)
        self.custom_title_edit = SubmitLineEdit()
        self.custom_title_edit.setPlaceholderText("Назва гри")
        self.custom_title_edit.setMaxLength(MAX_GAME_TITLE)
        for edit in (self.custom_process_edit, self.custom_title_edit):
            edit.submitted.connect(self._add_custom_game)
        self.add_custom_game_button = make_button("Додати")
        self.add_custom_game_button.clicked.connect(self._add_custom_game)
        self.remove_custom_game_button = make_button("Прибрати вибрані")
        self.remove_custom_game_button.clicked.connect(self._remove_custom_games)
        add_row.addWidget(self.custom_process_edit, 1)
        add_row.addWidget(self.custom_title_edit, 1)
        add_row.addWidget(self.add_custom_game_button)
        add_row.addWidget(self.remove_custom_game_button)
        custom.body.addLayout(add_row)
        self.custom_game_error = make_label("", "error")
        self.custom_game_error.hide()
        custom.body.addWidget(self.custom_game_error)
        layout.addWidget(custom)
        self._render_custom_games()

        played = Card()
        played.body.addWidget(SectionHeader("Зіграні ігри"))
        self.played_table = self._table(("Гра", "Загалом", "Сесій", "Востаннє"))
        self.played_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.played_table.setMinimumHeight(140)
        self.played_empty_label = InfoLabel("")
        played.body.addWidget(self.played_table)
        played.body.addWidget(self.played_empty_label)
        layout.addWidget(played)
        layout.addStretch(1)
        return page

    def _build_ai_page(self) -> QWidget:
        name = self._settings.character.name
        page, layout = self._page("AI та голос")
        now = Card()
        now.body.addWidget(SectionHeader("Зараз: офлайн-режим"))
        now.body.addWidget(make_label("AI-провайдера не підключено.", "warning"))
        now.body.addWidget(InfoLabel(
            f"{name} говорить готовими репліками, а в чаті розуміє лише прості фрази — наприклад, "
            "«привіт» чи «помовч». Жодні дані нікуди не надсилаються."))
        layout.addWidget(now)

        plan = Card()
        plan.body.addWidget(SectionHeader("Що буде далі"))
        for title, text in (
            ("Фаза 2 — розмови з AI",
             "Підтримка OpenAI, Anthropic і локальних моделей через Ollama. Хмарному провайдерові текст "
             "надсилатиметься лише з окремим дозволом. Налаштування провайдера з'являться тут разом "
             "із самою функцією."),
            ("Фаза 3 — зір",
             "За твоїм дозволом вона зможе бачити, що відбувається на екрані."),
            ("Фаза 4 — голос",
             "Розпізнавання мовлення (STT) та озвучення відповідей (TTS)."),
        ):
            plan.body.addWidget(make_label(title, "value"))
            plan.body.addWidget(InfoLabel(text))
        layout.addWidget(plan)
        layout.addStretch(1)
        return page

    # --- small builders -----------------------------------------------------------------------

    @staticmethod
    def _check_row(card: Card, text: str, hint: str, checked: bool) -> QCheckBox:
        row = CheckRow(text, hint)
        row.setChecked(checked)
        card.body.addWidget(row)
        return row.checkbox

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int, suffix: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setFixedWidth(124)
        spin.setAccelerated(True)
        return spin

    @staticmethod
    def _setting_row(text: str, field: QWidget, hint: str = "") -> QWidget:
        """Label (+ optional hint) on the left, the input on the right.

        A box layout rather than QFormLayout: the latter ignores height-for-width,
        so wrapped labels would overlap the next row.
        """
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(14)
        labels = QVBoxLayout()
        labels.setSpacing(1)
        labels.addWidget(make_label(text))
        if hint:
            labels.addWidget(InfoLabel(hint))
        layout.addLayout(labels, 1)
        layout.addWidget(field, 0, Qt.AlignmentFlag.AlignVCenter)
        return row

    @staticmethod
    def _table(headers: tuple[str, ...], stretch: tuple[int, ...] = (0,)) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(list(headers))
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        header = table.horizontalHeader()
        resize = QHeaderView.ResizeMode
        for column in range(len(headers)):
            mode = resize.Stretch if column in stretch else resize.ResizeToContents
            header.setSectionResizeMode(column, mode)
        return table

    def _permission_row(self, permission: Permission) -> QWidget:
        text = PERMISSION_TEXTS[permission]
        row = QWidget()
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 2, 0, 4)
        layout.setSpacing(2)
        top = QHBoxLayout()
        box = QCheckBox(text.title)
        top.addWidget(box)
        top.addStretch(1)
        if permission not in AVAILABLE:
            phase = text.phase if text.phase is not None else 2
            top.addWidget(make_label(f"з'явиться у фазі {phase}", "pill", wrap=False))
            box.setEnabled(False)
        else:
            box.clicked.connect(lambda checked, p=permission: self._on_permission_clicked(p, checked))
        layout.addLayout(top)
        hint = InfoLabel(text.hint)
        hint.setContentsMargins(28, 0, 0, 0)
        layout.addWidget(hint)
        self.permission_checks[permission] = box
        return row

    # --- privacy ---------------------------------------------------------------------------

    def _on_permission_clicked(self, permission: Permission, checked: bool) -> None:
        manager = self._facade.permissions
        if checked:
            text = PERMISSION_TEXTS[permission]
            if self._confirm(text.consent_title, text.consent_text):
                try:
                    manager.grant(permission)
                except (PermissionError, ValueError) as exc:  # unavailable / missing base permission
                    log.warning("could not grant %s: %s", permission.value, exc)
                    self._set_status("Цей дозвіл зараз не можна надати.", error=True)
                else:
                    self._set_status("Дозвіл надано.")
        else:
            manager.revoke(permission)
            self._set_status("Дозвіл відкликано.")
        self._refresh_permissions()
        self._refresh_history()
        self._refresh_games()

    def _stop_all(self) -> None:
        self._facade.stop_all_observation()
        self._refresh_permissions()
        self._refresh_history()
        self._refresh_games()
        self._set_status("Усе спостереження зупинено.")

    def _refresh_permissions(self) -> None:
        snapshot = self._facade.permissions.snapshot()
        for permission, box in self.permission_checks.items():
            box.blockSignals(True)
            box.setChecked(bool(snapshot.get(permission)))
            box.blockSignals(False)
            base = REQUIRES.get(permission)
            box.setEnabled(permission in AVAILABLE and (base is None or bool(snapshot.get(base))))
        try:
            supported = self._facade.status().activity_supported
        except Exception:
            log.exception("reading app status failed")
            supported = True
        self.activity_unsupported_label.setVisible(not supported)

    def _refresh_history(self) -> None:
        self.history_list.clear()
        records: list[PermissionRecord] = self._facade.permissions.history(50)
        if not records:
            self.history_list.addItem("Поки що жодних змін.")
            return
        for record in records:
            what = PERMISSION_NAMES.get(record.permission, record.permission.value)
            verdict = "дозвіл надано" if record.granted else "дозвіл відкликано"
            self.history_list.addItem(f"{format_local_datetime(record.at)} — {what}: {verdict}")

    def _render_exclusions(self) -> None:
        self.exclusions_list.clear()
        for name in self._exclusions:
            self.exclusions_list.addItem(name)

    def _add_exclusion(self) -> None:
        raw = self.exclusion_edit.text()
        if not raw.strip():
            return
        name = normalize_process_name(raw)
        if name is None:
            self.exclusion_error.setText("Це не схоже на назву програми. Приклад: mybank.exe")
            self.exclusion_error.show()
            return
        self.exclusion_error.hide()
        if name not in self._exclusions:
            self._exclusions.append(name)
            self._render_exclusions()
        self.exclusion_edit.clear()

    def _remove_exclusions(self) -> None:
        selected = {item.text() for item in self.exclusions_list.selectedItems()}
        if selected:
            self._exclusions = [n for n in self._exclusions if n not in selected]
            self._render_exclusions()

    def _open_data_dir(self) -> None:
        if not open_folder(self._facade.data_dir()):
            self._set_status("Не вдалося відкрити теку.", error=True)

    def _wipe_history(self) -> None:
        if not self._confirm("Видалити історію?", WIPE_HISTORY_TEXT):
            return
        self._facade.wipe_history()
        self.refresh()
        self._set_status("Історію видалено.")

    def _wipe_all(self) -> None:
        if not self._confirm("Видалити всі дані?", WIPE_ALL_TEXT):
            return
        self._facade.wipe_all()
        # Our working copy describes data that no longer exists: never apply it.
        self.reject()

    # --- games ------------------------------------------------------------------------------

    def _known_games_text(self) -> str:
        try:
            n = int(self._facade.known_games_count())
        except Exception:
            log.exception("reading the games database size failed")
            n = 0
        return count_uk(n, "гру", "гри", "ігор")

    def _render_custom_games(self) -> None:
        table = self.custom_games_table
        table.setRowCount(len(self._custom_games))
        for row, game in enumerate(self._custom_games):
            table.setItem(row, 0, QTableWidgetItem(game["process"]))
            table.setItem(row, 1, QTableWidgetItem(game["title"]))

    def _add_custom_game(self) -> None:
        raw = self.custom_process_edit.text()
        if not raw.strip():
            self.custom_game_error.setText("Вкажи назву процесу гри, наприклад mygame.exe.")
            self.custom_game_error.show()
            return
        process = normalize_process_name(raw)
        if process is None:
            self.custom_game_error.setText("Це не схоже на назву програми. Приклад: mygame.exe")
            self.custom_game_error.show()
            return
        self.custom_game_error.hide()
        title = self.custom_title_edit.text().strip()[:MAX_GAME_TITLE] or process.removesuffix(".exe")
        for game in self._custom_games:
            if game["process"] == process:
                game["title"] = title
                break
        else:
            self._custom_games.append({"process": process, "title": title})
        self._render_custom_games()
        self.custom_process_edit.clear()
        self.custom_title_edit.clear()

    def _remove_custom_games(self) -> None:
        rows = {index.row() for index in self.custom_games_table.selectionModel().selectedRows()}
        if rows:
            self._custom_games = [g for i, g in enumerate(self._custom_games) if i not in rows]
            self._render_custom_games()

    def _refresh_games(self) -> None:
        monitoring = self._facade.permissions.is_granted(Permission.APP_METADATA)
        self.monitoring_note.setText(
            "" if monitoring else
            "Щоб вона помічала ігри й програми, увімкни дозвіл «Бачити назви активних програм та ігор» "
            "у розділі «Приватність»."
        )
        self.monitoring_note.setVisible(not monitoring)
        try:
            totals: list[GameTotals] = list(self._facade.game_totals())
        except Exception:
            log.exception("reading game totals failed")
            totals = []
        table = self.played_table
        table.setRowCount(len(totals))
        for row, game in enumerate(totals):
            cells = (
                game.title or game.game_id,
                format_duration(game.total_seconds or 0.0),
                str(game.sessions),
                format_local_date(game.last_played),
            )
            for column, value in enumerate(cells):
                item = QTableWidgetItem(value)
                if column:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, column, item)
        table.setVisible(bool(totals))
        if totals:
            self.played_empty_label.setText(
                "" if monitoring else "Відстеження вимкнено, тому нові ігри сюди не потраплять.")
        elif monitoring:
            self.played_empty_label.setText("Поки що жодної гри. Запусти щось — і тут з'явиться статистика.")
        else:
            self.played_empty_label.setText(
                "Відстеження ігор вимкнено. Увімкни дозвіл «Бачити назви активних програм та ігор» у "
                "розділі «Приватність», і тут з'явиться статистика.")
        self.played_empty_label.setVisible(bool(self.played_empty_label.text()))

    # --- applying ---------------------------------------------------------------------------

    def _collect(self) -> Settings:
        # Start from the app's current settings so fields this dialog doesn't edit
        # (first_run_completed, pack, poll intervals...) are never clobbered, and
        # copy over only the fields edited here: the others may have been changed
        # from the tray, her menu or Ctrl+wheel since the widgets were loaded.
        s = self._facade.settings().copy()
        for field in self._fields:
            value = field.read()
            if value != self._loaded[(field.section, field.name)]:
                setattr(getattr(s, field.section), field.name, value)
        return s

    def _bind_fields(self) -> tuple[_Field, ...]:
        """Every settings field this dialog edits, with how to read and show it."""

        def text(section: str, name: str, edit: QLineEdit, limit: int) -> _Field:
            return _Field(section, name, lambda: edit.text().strip()[:limit], lambda v: edit.setText(str(v)))

        def check(section: str, name: str, box: QCheckBox) -> _Field:
            return _Field(section, name, box.isChecked, lambda v: box.setChecked(bool(v)))

        def number(section: str, name: str, widget: QSpinBox | SliderRow) -> _Field:
            return _Field(section, name, widget.value, lambda v: widget.setValue(int(v)))

        fields = [
            text("character", "name", self.name_edit, 24),
            text("character", "user_nickname", self.nickname_edit, 32),
            _Field("character", "scale", lambda: self.scale_row.value() / 100,
                   lambda v: self.scale_row.setValue(round(float(v) * 100))),
            _Field("character", "renderer", lambda: str(self.renderer_combo.currentData() or "sprite"),
                   lambda v: self.renderer_combo.setCurrentIndex(max(0, self.renderer_combo.findData(v)))),
            check("character", "always_on_top", self.always_on_top_check),
            check("character", "lock_position", self.lock_position_check),
            check("character", "walking_enabled", self.walking_check),
            check("character", "gravity", self.gravity_check),
            check("character", "click_through", self.click_through_check),
            check("character", "show_bubbles", self.bubbles_check),
            check("behavior", "do_not_disturb", self.dnd_check),
            number("behavior", "max_unsolicited_per_hour", self.max_per_hour_spin),
            number("behavior", "min_seconds_between_unsolicited", self.min_gap_spin),
            number("behavior", "quiet_minutes", self.quiet_minutes_spin),
            check("behavior", "hide_on_fullscreen", self.hide_fullscreen_check),
            check("behavior", "auto_eco_mode", self.eco_check),
            number("behavior", "sleep_after_idle_minutes", self.sleep_after_spin),
            _Field("activity", "excluded_processes", lambda: list(self._exclusions), self._show_exclusions),
            _Field("activity", "custom_games", lambda: [dict(g) for g in self._custom_games],
                   self._show_custom_games),
        ]
        fields += [number("personality", name, row) for name, row in self.personality_rows.items()]
        fields += [check("activity", name, box) for name, box in self.comment_checks.items()]
        return tuple(fields)

    def _read_fields(self) -> dict[tuple[str, str], object]:
        return {(field.section, field.name): field.read() for field in self._fields}

    def _sync_from_app(self, *, edited_too: bool) -> bool:
        """Show the app's current settings in the widgets: every field after an
        apply, otherwise only those the user has not edited here (their edits
        win until applied or cancelled).  Returns False if the settings could not be read."""
        try:
            current = self._facade.settings().copy()
        except Exception:
            log.exception("reading the current settings failed")
            return False
        for field in self._fields:
            key = (field.section, field.name)
            if edited_too or field.read() == self._loaded[key]:
                field.write(getattr(getattr(current, field.section), field.name))
                self._loaded[key] = field.read()
        self._settings = current
        self.setWindowTitle(f"Налаштування — {current.character.name}")
        return True

    def _show_exclusions(self, value: object) -> None:
        self._exclusions = [str(name) for name in value] if isinstance(value, list) else []
        self._render_exclusions()

    def _show_custom_games(self, value: object) -> None:
        self._custom_games = self._custom_games_from(value if isinstance(value, list) else [])
        self._render_custom_games()

    @staticmethod
    def _custom_games_from(games: list) -> list[dict[str, str]]:
        """The editable part (process, title) of the stored custom games."""
        return [
            {"process": g.get("process", ""), "title": g.get("title", "")}
            for g in games
            if isinstance(g, dict) and g.get("process")
        ]

    def _read_has_tray(self) -> bool:
        try:
            return bool(self._facade.has_tray())
        except Exception:
            log.exception("asking the app about the tray failed")
            return True  # the app itself never applies click-through without a tray

    def _on_ok(self) -> None:
        if self.apply():
            self.accept()

    def _set_status(self, text: str, *, error: bool = False) -> None:
        self.status_label.setText(text)
        set_style_property(self.status_label, "role", "error" if error else "success")

    def _ask(self, title: str, text: str) -> bool:
        box = QMessageBox(QMessageBox.Icon.Question, title, text, QMessageBox.StandardButton.NoButton, self)
        yes = box.addButton("Так", QMessageBox.ButtonRole.AcceptRole)
        no = box.addButton("Скасувати", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(no)
        box.setEscapeButton(no)
        box.exec()
        return box.clickedButton() is yes

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.refresh()

