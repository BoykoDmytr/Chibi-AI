"""First-run welcome: she introduces herself, asks for names and - only as an
opt-in - for permission to see which apps are running.

The permission checkbox starts UNCHECKED and its text says exactly what is
collected; ticking it is the explicit consent, so no second confirmation is
asked.  Leaving it unticked means nothing is observed at all.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLineEdit, QVBoxLayout, QWidget

from chibi.core.events import Permission
from chibi.settings.facade import AppFacade
from chibi.settings.widgets import Card, InfoLabel, PortraitLabel, make_button, make_label, scroll_page

log = logging.getLogger(__name__)

WIDTH = 580

INTRO = (
    "Я житиму на твоєму робочому столі: гулятиму, сидітиму, дріматиму й іноді щось казатиму. "
    "Мене можна погодувати, погладити або поговорити зі мною в чаті — усе це є в меню, якщо "
    "клацнути на мене правою кнопкою миші.\n\n"
    "Поки що я працюю без AI: відповідаю готовими репліками й розумію лише прості фрази."
)
PRIVACY = (
    "За замовчуванням я нічого не відстежую: не бачу, які програми відкриті, не роблю знімків "
    "екрана й не читаю, що ти друкуєш. Усе, що я пам'ятаю, зберігається лише на цьому комп'ютері."
)
CONSENT_TEXT = "Дозволити бачити, які програми та ігри запущені"
CONSENT_HINT = (
    "Тоді я зможу вболівати за тебе в іграх, підбадьорювати, коли ти програмуєш, і помічати, що ти "
    "дивишся відео. Я бачитиму лише назву програми та її тип, а ще — скільки часу ти не торкаєшся "
    "миші й клавіатури (без самих натискань). Нічого не записується, крім тривалості ігрових сесій. "
    "Передумати можна будь-коли: Налаштування → «Приватність»."
)


class FirstRunDialog(QDialog):
    def __init__(self, facade: AppFacade, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._facade = facade
        settings = facade.settings()
        name = settings.character.name
        self.setWindowTitle("Знайомство")
        self.setMinimumSize(480, 360)

        hello = QHBoxLayout()
        hello.setSpacing(16)
        self.portrait = PortraitLabel(facade.portrait, 120, fallback_text=name)
        self.portrait.show_portrait("greet", "wink", "smile", "calm")
        hello.addWidget(self.portrait, 0, Qt.AlignmentFlag.AlignTop)
        words = QVBoxLayout()
        words.setSpacing(8)
        self.title_label = make_label(f"Привіт! Я — {name}", "title")
        words.addWidget(self.title_label)
        words.addWidget(InfoLabel(INTRO))
        hello.addLayout(words, 1)

        names = Card()
        names.body.addWidget(make_label("Як мене звати?"))
        self.name_edit = QLineEdit(name)
        self.name_edit.setMaxLength(24)
        self.name_edit.setPlaceholderText("Мія")
        names.body.addWidget(self.name_edit)
        self.name_error = make_label("", "error")
        self.name_error.hide()
        names.body.addWidget(self.name_error)
        names.body.addSpacing(4)
        names.body.addWidget(make_label("Як мені до тебе звертатися?"))
        self.nickname_edit = QLineEdit(settings.character.user_nickname)
        self.nickname_edit.setMaxLength(32)
        self.nickname_edit.setPlaceholderText("наприклад, Сашко")
        names.body.addWidget(self.nickname_edit)
        names.body.addWidget(InfoLabel(
            "Так, як ти хочеш, щоб я до тебе зверталася: «Сашко», «Олю», «пане капітане». "
            "Можна залишити порожнім."))

        privacy = Card()
        privacy.body.addWidget(make_label("Приватність", "section"))
        privacy.body.addWidget(InfoLabel(PRIVACY))
        self.consent_check = QCheckBox(CONSENT_TEXT)
        self.consent_check.setChecked(False)
        privacy.body.addWidget(self.consent_check)
        consent_hint = InfoLabel(CONSENT_HINT)
        consent_hint.setContentsMargins(28, 0, 0, 0)
        privacy.body.addWidget(consent_hint)

        self.start_button = make_button("Почати", "primary", big=True)
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self.start)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(20, 0, 20, 18)
        buttons.addStretch(1)
        buttons.addWidget(self.start_button)

        self._content = QWidget()
        content = QVBoxLayout(self._content)
        content.setContentsMargins(20, 20, 20, 6)
        content.setSpacing(14)
        content.addLayout(hello)
        content.addWidget(names)
        content.addWidget(privacy)
        # Scrollable so the dialog still fits a small or heavily scaled screen.
        self._scroll = scroll_page(self._content)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        root.addWidget(self._scroll, 1)
        root.addLayout(buttons)
        self.resize(WIDTH, 640)
        self._fitted = False

        self.name_edit.textChanged.connect(self._on_name_changed)

    def start(self) -> bool:
        """Save the answers, grant the permission only if ticked, and close.  False if invalid."""
        name = self.name_edit.text().strip()
        if not name:
            self.name_error.setText("Дай мені ім'я — поле не може бути порожнім.")
            self.name_error.show()
            self.name_edit.setFocus()
            return False
        settings = self._facade.settings().copy()
        settings.first_run_completed = True
        settings.character.name = name[:24]
        settings.character.user_nickname = self.nickname_edit.text().strip()[:32]
        try:
            self._facade.apply_settings(settings)
        except Exception:
            log.exception("saving first-run settings failed")
            self.name_error.setText("Не вдалося зберегти налаштування. Спробуй ще раз.")
            self.name_error.show()
            return False
        if self.consent_check.isChecked():
            try:
                self._facade.permissions.grant(Permission.APP_METADATA)
            except (PermissionError, ValueError):
                log.exception("could not grant app metadata after first-run consent")
        self.accept()
        return True

    def showEvent(self, event: QShowEvent) -> None:
        if not self._fitted:
            self._fitted = True
            self._fit_height()
        super().showEvent(event)

    def _fit_height(self) -> None:
        """Size the window to its content (top-level windows ignore the height
        wrapped labels need), but never taller than ~90 % of the screen."""
        layout = self._content.layout()
        needed = layout.totalHeightForWidth(self.width()) if layout.hasHeightForWidth() else -1
        if needed <= 0:
            needed = self._content.sizeHint().height()
        needed += self.start_button.sizeHint().height() + 40
        screen = self.screen()
        limit = int(screen.availableGeometry().height() * 0.9) if screen is not None else needed
        self.resize(self.width(), max(self.minimumHeight(), min(needed, limit)))

    def _on_name_changed(self, text: str) -> None:
        name = text.strip()
        if name:
            self.name_error.hide()
            self.title_label.setText(f"Привіт! Я — {name}")
            self.portrait.set_fallback_text(name)
