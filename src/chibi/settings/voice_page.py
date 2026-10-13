"""The "Голос" page of the settings dialog: her voice, speech recognition,
how she listens (push-to-talk or an always-open microphone) and the voice
models.

Design notes - why it works this way:

* **Settings vs. secrets vs. permissions vs. models**, as on the AI page:
  engine, speaker, speed, hotkey... are ordinary settings, edited here and
  applied with the rest of the dialog through :meth:`VoicePage.collect`.  The
  Azure key goes straight to the OS credential store (``voice_set_azure_key``)
  and is never shown again.  ``Permission.MICROPHONE`` is granted only after
  the consent text and goes straight to the manager.  Models are downloaded
  only when the user clicks "Завантажити" - never automatically.
* **Baseline**: the page remembers what each widget showed when it was last
  loaded from the app; ``collect`` writes only fields that differ, so a stale
  widget never reverts a change made elsewhere.
* **"Прослухати" uses the page's values**: it applies this page's pending
  edits first (only the voice section), so the sample is spoken with the
  speaker and speed the page shows.
* **An always-open microphone is a choice made knowingly**: picking "name" or
  "open" in "Як слухати" asks first, with an honest account of what stays
  open, what is recorded (nothing) and what reaches the AI (and the cloud).
  Declining puts the previous choice back.  Values loaded from the settings
  never ask.
* **Never wait**: the sample and downloads are asynchronous; late answers for
  a superseded request (or a page closed meanwhile) are dropped, and a page
  reopened during a download picks it up by polling its progress.
* The page depends only on the small :class:`VoiceFacade` protocol (the app
  implements it), never on the voice engines: it works without the optional
  voice libraries and says what is missing.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QHideEvent, QShowEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QProgressBar,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from chibi.core.config import Settings, VoiceSettings
from chibi.core.events import Permission
from chibi.settings.widgets import (
    Card,
    CheckRow,
    InfoLabel,
    SectionHeader,
    SliderRow,
    SubmitLineEdit,
    ask_yes_no,
    is_alive,
    make_button,
    make_label,
    percent,
    set_style_property,
    setting_row,
)
from chibi.voice import hotkey_win
from chibi.voice.integration import STT_OFF, TTS_OFF
from chibi.voice.models import CATALOG, DEFAULT_TTS_MODEL, ModelSpec
from chibi.voice.tts import AZURE_VOICES, PIPER_SPEAKERS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from chibi.privacy.permissions import PermissionManager
    from chibi.voice.integration import DownloadProgress, VoiceStatus

log = logging.getLogger(__name__)

ConfirmFn = Callable[[str, str], bool]

#: How often the status lines are re-read while the page is visible.
STATUS_REFRESH_MS = 3000
#: Progress polling while a download runs (a page reopened mid-download).
DOWNLOAD_POLL_MS = 400
#: Give up waiting for the sample after this long (it always answers).
TEST_WATCHDOG_MS = 60_000

ENGINE_CHOICES: tuple[tuple[str, str], ...] = (
    ("piper", "Piper — на цьому ПК, без інтернету"),
    ("azure", "Azure — хмара Microsoft, найприродніше звучання"),
)
SPEAK_CHOICES: tuple[tuple[str, str], ...] = (
    ("all", "Усі репліки"),
    ("replies", "Лише відповіді тобі"),
    ("off", "Нічого — лише текст у бульбашках"),
)
PTT_MODE_CHOICES: tuple[tuple[str, str], ...] = (
    ("toggle", "Натиснути й говорити"),
    ("hold", "Тримати, поки говориш"),
)

#: How she listens (``VoiceSettings.listen_mode``): value, choice, explanation
#: (``{voc}`` is how the user calls her: "Міє").
LISTEN_CHOICES: tuple[tuple[str, str, str], ...] = (
    ("push", "Клавіша або кнопка мікрофона",
     "Мікрофон вмикається лише на час твоєї фрази."),
    ("name", "Постійно, коли звертаюся на ім'я (рекомендовано)",
     "Мікрофон відкритий, поки вона мовчить. Скажи «{voc}, котра година?» чи «Як справи, {voc}?» — "
     "і вона відповість, а далі розмова триває без імені."),
    ("open", "Постійно — усе, що чую, для неї",
     "Вона відповідає на кожну почуту фразу — зокрема на телевізор, музику чи інших людей поруч."),
)
#: Listen modes from the least to the most exposing (switching up asks first).
_EXPOSURE = {"push": 0, "name": 1, "open": 2}

PRIVACY_NOTE = (
    "Мовлення розпізнається на цьому ПК: звук не зберігається і не надсилається в інтернет. "
    "До AI потрапляє лише розпізнаний текст того, що сказано їй."
)
#: The same consent as on the privacy page (the dialog imports these).
MICROPHONE_CONSENT_TITLE = "Дозволити слухати мікрофон?"
MICROPHONE_CONSENT_TEXT = (
    "Вона зможе чути тебе так, як обрано в налаштуваннях голосу («Як слухати»):\n\n"
    "• «Клавіша або кнопка мікрофона» — мікрофон вмикається лише на час твоєї фрази.\n"
    "• «Постійно» — мікрофон лишається відкритим, поки вона мовчить; до неї доходить лише те, "
    "що звернено до неї (або все почуте — якщо так обрано).\n"
    "• Розпізнавання працює на цьому ПК: звук нікуди не надсилається й не записується. "
    "Розпізнаний текст того, що сказано їй, іде до AI, а з хмарним AI — у хмару.\n"
    "• Поки мікрофон відкритий, у треї світиться червона крапка.\n\n"
    "Відкликати можна будь-коли тут або кнопкою «Зупинити все спостереження»."
)
OPEN_MIC_CONFIRM_TITLE = "Слухати постійно?"
_OPEN_MIC_INTRO = (
    "Мікрофон лишатиметься відкритим увесь час, поки вона мовчить. На паузу він стає в режимі "
    "«Не турбувати», у грі на весь екран (це можна змінити нижче), а якщо їй дозволено бачити, "
    "яка програма відкрита, — ще й під час дзвінків і в приватних програмах.\n\n"
    "• Розпізнавання працює на цьому ПК: звук нікуди не надсилається й не записується.\n"
)
_OPEN_MIC_OUTRO = (
    "• Поки мікрофон відкритий, у треї світиться червона крапка. Поставити його на паузу можна "
    "в її меню.\n\n"
    "Увімкнути?"
)
OPEN_MIC_CONFIRM_NAME = _OPEN_MIC_INTRO + (
    "• До AI потрапляє лише текст того, що звернено до неї: фраза з її ім'ям і розмова одразу "
    "після її відповіді. Якщо AI хмарний, цей текст надсилається в хмару.\n"
) + _OPEN_MIC_OUTRO
OPEN_MIC_CONFIRM_OPEN = _OPEN_MIC_INTRO + (
    "• До AI потрапляє текст усього, що вона почує. Якщо AI хмарний, цей текст надсилається "
    "в хмару.\n"
    "• Вона відповідатиме на все: і на телевізор, музику чи розмови інших людей поруч. З платним "
    "AI кожна почута фраза — окремий запит, тож витрати можуть швидко зростати.\n"
) + _OPEN_MIC_OUTRO
DND_NOTE = (
    "Клавіша й кнопка мікрофона працюють у будь-якому режимі. У режимі «Не турбувати» постійний "
    "мікрофон стає на паузу."
)
AZURE_KEY_HINT = (
    "Ключ ресурсу Speech із порталу Azure. Він зберігається в диспетчері облікових даних Windows і "
    "більше ніде не показується — навіть тут. У журнали він не потрапляє."
)
AZURE_CLOUD_WARNING = (
    "Azure — хмарний сервіс Microsoft: тексти її реплік надсилатимуться туди, щоб їх озвучити. Для "
    "цього потрібен дозвіл «Надсилати текст розмов хмарному AI» в розділі «Приватність»; без нього "
    "вона говоритиме лише текстом."
)
AZURE_CLOUD_OK = (
    "Тексти її реплік надсилаються в Azure для озвучення. Відкликати дозвіл можна в розділі "
    "«Приватність» — він перестане діяти одразу."
)
MIC_NEEDED = "Щоб вона тебе чула, потрібен дозвіл «Слухати мікрофон»."
HOTKEY_WINDOWS_ONLY = "Гаряча клавіша працює лише у Windows — тут говори кнопкою мікрофона в чаті."
DEFAULT_DEVICE = "Типовий пристрій системи"
NOT_CONNECTED = "(не підключено)"
REGION_ERROR = "Регіон — це латинські літери й цифри, наприклад westeurope."

_REGION_RE = re.compile(r"[a-z0-9]{3,40}")
#: A key is one line of printable ASCII without spaces.
_KEY_RE = re.compile(r"[\x21-\x7e]{8,512}")
_NBSP = "\u00a0"


# --- what the page needs from the app ---------------------------------------------------------


class VoiceFacade(Protocol):
    """Implemented by the app (``chibi.app.ChibiApp``); callbacks run on the Qt main thread."""

    permissions: PermissionManager

    def settings(self) -> Settings: ...

    def apply_settings(self, settings: Settings) -> None: ...

    def voice_status(self) -> VoiceStatus: ...

    def voice_models(self) -> list[tuple[ModelSpec, bool]]:
        """Every downloadable model and whether it is installed."""
        ...

    def voice_download(self, model_id: str, on_progress: Callable[[int, int], None],
                       on_done: Callable[[bool, str], None]) -> None:
        """Start downloading; on_progress(bytes done, total); on_done(ok, Ukrainian message) once."""
        ...

    def voice_cancel_download(self) -> None: ...

    def voice_download_state(self) -> DownloadProgress | None:
        """The running download (model_id, done, total), or None."""
        ...

    def voice_remove_model(self, model_id: str) -> None: ...

    def voice_has_azure_key(self) -> bool: ...

    def voice_set_azure_key(self, key: str) -> None:
        """Store the Azure Speech key ("" deletes it).  Never logged."""
        ...

    def voice_test(self, on_done: Callable[[bool, str], None]) -> None:
        """Speak a sample line now; on_done(ok, message) once it has played or failed."""
        ...

    def voice_devices(self) -> tuple[list[str], list[str]]:
        """(microphones, speakers) by name; empty when unknown."""
        ...


# --- pure helpers --------------------------------------------------------------------------------


def format_size(size: int) -> str:
    """``483_546_902`` -> ``"484 МБ"``; ``1_527_906_378`` -> ``"1,5 ГБ"``."""
    mb = max(0, size) / 1_000_000
    if mb >= 1000:
        return f"{mb / 1000:.1f}".replace(".", ",") + f"{_NBSP}ГБ"
    return f"{max(1, round(mb)) if size > 0 else 0}{_NBSP}МБ"


def format_seconds(ms: int) -> str:
    """``800`` -> ``"0,8 с"``."""
    return f"{ms / 1000:.1f}".replace(".", ",") + f"{_NBSP}с"


def format_follow_up(seconds: int) -> str:
    """``20`` -> ``"20 с"``; ``0`` -> ``"вимкнено"``."""
    return f"{seconds}{_NBSP}с" if seconds > 0 else "вимкнено"


def vocative(name: str) -> str:
    """How the examples call her: "Мія" -> "Міє"; other names as they are."""
    name = name.strip()
    return "Міє" if name == "Мія" else (name or "Міє")


def model_choice_title(spec: ModelSpec) -> str:
    """"Розпізнавання: збалансоване (small)" -> "Збалансоване (small) — 484 МБ"."""
    title = spec.title.split(": ", 1)[-1]
    return f"{title[:1].upper()}{title[1:]} — {format_size(spec.size)}"


def hotkey_title(text: str) -> str:
    try:
        return hotkey_win.parse_hotkey(text).title
    except ValueError:
        return text


@dataclass(frozen=True)
class _Field:
    """One ``Settings.voice.<name>`` field edited by a widget."""

    name: str
    read: Callable[[], object]
    write: Callable[[object], None]


# --- one downloadable model ----------------------------------------------------------------------


class ModelRow(QWidget):
    """A model's name, size and state with Download / Cancel / Remove and a progress bar."""

    download_requested = Signal(str)
    cancel_requested = Signal()
    remove_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model_id = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)
        top = QHBoxLayout()
        self.title_label = make_label("", "value")
        self.state_label = make_label("", "pill", wrap=False)
        top.addWidget(self.title_label, 1)
        top.addWidget(self.state_label)
        layout.addLayout(top)
        self.description_label = InfoLabel("")
        layout.addWidget(self.description_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setProperty("tone", "lilac")
        self.progress.setAccessibleName("Завантаження моделі")
        self.progress.hide()
        layout.addWidget(self.progress)
        self.progress_label = make_label("", "hint", wrap=False)
        self.progress_label.hide()
        layout.addWidget(self.progress_label)
        buttons = QHBoxLayout()
        self.download_button = make_button("Завантажити", "accent")
        self.cancel_button = make_button("Скасувати")
        self.remove_button = make_button("Видалити", "danger")
        self.download_button.clicked.connect(lambda: self.download_requested.emit(self._model_id))
        self.cancel_button.clicked.connect(self._on_cancel)
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self._model_id))
        buttons.addWidget(self.download_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.message_label = make_label("", "hint")
        self.message_label.hide()
        layout.addWidget(self.message_label)

    @property
    def model_id(self) -> str:
        return self._model_id

    def set_model(self, spec: ModelSpec | None, installed: bool, *, downloading: bool = False,
                  done: int = 0, total: int = 0, other_busy: bool = False) -> None:
        """Show *spec*; *downloading* this one, or *other_busy* (one download at a time)."""
        if spec is None:
            self.hide()
            return
        buttons = (self.cancel_button, self.download_button, self.remove_button)
        had_focus = any(button.hasFocus() for button in buttons)
        self.show()
        if spec.id != self._model_id:
            self.message_label.hide()
        self._model_id = spec.id
        self.title_label.setText(f"{spec.title} — {format_size(spec.size)}")
        self.description_label.setText(f"{spec.description} Ліцензія: {spec.license}.")
        if downloading:
            state, tone = "завантажується", "alert"
        elif installed:
            state, tone = "завантажено ✓", "on"
        else:
            state, tone = "не завантажено", ""
        self.state_label.setText(state)
        set_style_property(self.state_label, "state", tone)
        self.progress.setVisible(downloading)
        self.progress_label.setVisible(downloading)
        if downloading:
            total = total or spec.size
            self.progress.setValue(round(1000 * min(1.0, done / total)) if total > 0 else 0)
            self.progress_label.setText(f"{format_size(done)} із {format_size(total)}")
        else:
            self.cancel_button.setText("Скасувати")
            self.cancel_button.setEnabled(True)
        self.download_button.setVisible(not installed and not downloading)
        self.download_button.setEnabled(not other_busy)
        self.download_button.setToolTip(
            "Зачекай, поки завантажиться інша модель." if other_busy
            else f"Завантажити {format_size(spec.size)} з huggingface.co")
        self.cancel_button.setVisible(downloading)
        self.remove_button.setVisible(installed and not downloading)
        if had_focus and not any(button.hasFocus() for button in buttons):
            # The focused button was hidden (download started / finished): keep the
            # keyboard user on this row instead of letting focus jump away.
            target = next((b for b in buttons if b.isVisibleTo(self) and b.isEnabled()), None)
            if target is not None:
                target.setFocus()

    def show_message(self, text: str, *, error: bool = False) -> None:
        self.message_label.setText(text)
        set_style_property(self.message_label, "role", "error" if error else "success")
        self.message_label.setVisible(bool(text))

    def _on_cancel(self) -> None:
        self.cancel_button.setText("Скасовую…")
        self.cancel_button.setEnabled(False)
        self.cancel_requested.emit()


# --- the page --------------------------------------------------------------------------------------


class VoicePage(QWidget):
    """Settings page for voice.  See the module docstring."""

    #: A permission was granted here (the dialog refreshes its privacy page).
    permissions_changed = Signal()
    #: The page applied its own edits before the sample (the dialog shows a note).
    settings_applied = Signal()

    def __init__(self, facade: VoiceFacade, confirm: ConfirmFn | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._facade = facade
        self._confirm: ConfirmFn = confirm or (lambda title, text: ask_yes_no(self, title, text))
        settings = self._read_settings() or Settings()
        v = settings.voice
        self._name = settings.character.name
        self._status_failed = False
        #: The listen mode accepted on this page (a declined confirmation goes back to it).
        self._listen_mode = v.listen_mode
        #: True while the page itself checks a listen-mode button (no confirmation then).
        self._setting_mode = False
        self._test_seq = 0
        self._test_busy = False
        self._download_seq = 0
        #: Model being downloaded (by this page or, after a reopen, found by polling).
        self._downloading: str | None = None
        self._progress = (0, 0)
        #: model id -> (spec, installed), re-read on refresh and after a download / removal.
        self._model_infos: dict[str, tuple[ModelSpec, bool]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 0, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(make_label("Голос", "title", wrap=False))
        self.lead_label = InfoLabel("")
        layout.addWidget(self.lead_label)
        layout.addWidget(self._build_status_card())
        layout.addWidget(self._build_voice_card(v))
        layout.addWidget(self._build_recognition_card(v))
        layout.addStretch(1)

        self._fields: tuple[_Field, ...] = self._bind_fields()
        self._loaded: dict[str, object] = self._read_fields()
        self._update_listen_controls()

        self.status_timer = QTimer(self)
        self.status_timer.setInterval(STATUS_REFRESH_MS)
        self.status_timer.timeout.connect(self.refresh_status)
        self.download_timer = QTimer(self)
        self.download_timer.setInterval(DOWNLOAD_POLL_MS)
        self.download_timer.timeout.connect(self._poll_download)
        self.test_watchdog = QTimer(self)
        self.test_watchdog.setSingleShot(True)
        self.test_watchdog.setInterval(TEST_WATCHDOG_MS)
        self.test_watchdog.timeout.connect(self._on_test_timeout)

        self._update_names()
        self._on_engine_changed()
        self._on_stt_model_changed()
        self.refresh()

    # --- public API (same shape as the AI page) -----------------------------------------------------

    def engine(self) -> str:
        return str(self.engine_combo.currentData() or "piper")

    def collect(self, settings: Settings) -> Settings:
        """Write the fields edited on this page into *settings* (in place; also returned).
        An invalid Azure region is left as it was."""
        for field in self._fields:
            value = field.read()
            if value == self._loaded[field.name]:
                continue
            if field.name == "azure_region" and not _REGION_RE.fullmatch(str(value)):
                continue
            setattr(settings.voice, field.name, value)
            if field.name == "listen_mode":
                settings.voice.wake_phrase_enabled = value != "push"  # the deprecated flag follows
        return settings

    def has_edits(self) -> bool:
        return any(field.read() != self._loaded[field.name] for field in self._fields)

    def sync(self, settings: Settings, *, edited_too: bool) -> None:
        """Show *settings* (the app's current ones): every field after an apply,
        otherwise only those the user has not edited here."""
        self._name = settings.character.name
        for field in self._fields:
            if edited_too or field.read() == self._loaded[field.name]:
                field.write(getattr(settings.voice, field.name))
                self._loaded[field.name] = field.read()
        self._update_names()
        self._on_engine_changed()
        self._on_stt_model_changed()
        self._update_listen_controls()

    def reset_baseline(self) -> None:
        """Treat what the widgets show now as unedited (after a successful apply)."""
        self._loaded = self._read_fields()

    def validate(self) -> bool:
        """The Azure region, if Azure is selected; shows the error inline."""
        bad = self.engine() == "azure" and not _REGION_RE.fullmatch(self._region_text())
        self.region_error.setText(REGION_ERROR if bad else "")
        self.region_error.setVisible(bad)
        return not bad

    def refresh(self) -> None:
        """Re-read what can change outside the page: status, key, permissions, models."""
        self.refresh_status()
        self._refresh_key_state()
        self._refresh_cloud()
        self._refresh_mic()
        self._poll_download()
        self._render_models(reload=True)

    def refresh_status(self) -> None:
        try:
            view = self._facade.voice_status()
        except Exception:
            if not self._status_failed:  # polled every few seconds: log the first failure only
                log.exception("reading the voice status failed")
            self._status_failed = True
            self.tts_status_label.setText("● Стан голосу зараз невідомий")
            set_style_property(self.tts_status_label, "role", "warning")
            self.stt_status_label.hide()
            self.hotkey_status_label.hide()
            self.listen_status_label.hide()
            return
        self._status_failed = False
        if view.tts_ready:
            text, role = f"● Голос працює — {self._voice_title()}", "success"
        else:
            text, role = f"● {view.tts_reason}", "hint" if view.tts_reason == TTS_OFF else "warning"
        self.tts_status_label.setText(text)
        set_style_property(self.tts_status_label, "role", role)
        if view.stt_ready:
            how = (f"натисни {view.hotkey} або кнопку мікрофона в чаті" if view.hotkey_active
                   else "натисни кнопку мікрофона в чаті")
            if view.listen_mode == "name":
                how = f"клич її на ім'я або {how}"
            elif view.listen_mode == "open":
                how = f"просто говори або {how}"
            text, role = f"● Розпізнавання готове — {how}.", "success"
        else:
            text, role = f"● {view.stt_reason}", "hint" if view.stt_reason == STT_OFF else "warning"
        same = not view.stt_ready and not view.tts_ready and view.stt_reason == view.tts_reason
        self.stt_status_label.setText(text)
        set_style_property(self.stt_status_label, "role", role)
        self.stt_status_label.setVisible(not same)  # "module not installed" once, not twice
        self.hotkey_status_label.setText(view.hotkey_reason)
        self.hotkey_status_label.setVisible(bool(view.hotkey_reason))
        pause = view.pause_reason if view.stt_ready else ""
        self.listen_status_label.setText(f"● {pause}" if pause else "")
        self.listen_status_label.setVisible(bool(pause))

    def save_pending_key(self) -> bool:
        """Store an Azure key typed but not saved yet ("Гаразд" should not lose it).
        Returns False if there is one and it could not be stored."""
        if not self.azure_key_edit.text().strip():
            return True
        return self._save_key()

    # --- building --------------------------------------------------------------------------------

    def _build_status_card(self) -> QWidget:
        card = Card(spacing=6)
        card.body.addWidget(SectionHeader("Зараз"))
        self.tts_status_label = make_label("", "hint")
        self.tts_status_label.setAccessibleName("Стан голосу")
        self.stt_status_label = make_label("", "hint")
        self.stt_status_label.setAccessibleName("Стан розпізнавання мовлення")
        self.hotkey_status_label = make_label("", "warning")
        self.hotkey_status_label.hide()
        self.listen_status_label = make_label("", "hint")
        self.listen_status_label.setAccessibleName("Чи слухає мікрофон")
        self.listen_status_label.hide()
        card.body.addWidget(self.tts_status_label)
        card.body.addWidget(self.stt_status_label)
        card.body.addWidget(self.listen_status_label)
        card.body.addWidget(self.hotkey_status_label)
        return card

    def _build_voice_card(self, v: VoiceSettings) -> QWidget:
        card = Card()
        self.voice_header = SectionHeader("")
        card.body.addWidget(self.voice_header)
        self.tts_enabled_check = self._check(
            card, "Говорити вголос", "Її репліки звучатимуть голосом, а не лише з'являтимуться в бульбашках.",
            v.tts_enabled)
        self.engine_combo = self._combo(ENGINE_CHOICES, v.tts_engine, "Як озвучувати")
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        card.body.addWidget(setting_row("Як озвучувати", self.engine_combo))

        # --- Piper (local)
        self.piper_box = QWidget()
        piper = QVBoxLayout(self.piper_box)
        piper.setContentsMargins(0, 0, 0, 0)
        piper.setSpacing(8)
        self.speaker_combo = self._combo(tuple(PIPER_SPEAKERS.items()), v.piper_speaker, "Голос")
        piper.addWidget(setting_row("Голос", self.speaker_combo))
        self.voice_model_row = self._model_row()
        piper.addWidget(self.voice_model_row)
        card.body.addWidget(self.piper_box)

        # --- Azure (cloud)
        self.azure_box = QWidget()
        azure = QVBoxLayout(self.azure_box)
        azure.setContentsMargins(0, 0, 0, 0)
        azure.setSpacing(8)
        self.azure_cloud_note = make_label("", "warning")
        azure.addWidget(self.azure_cloud_note)
        self.azure_voice_combo = self._combo(tuple(AZURE_VOICES.items()), v.azure_voice, "Голос Azure")
        azure.addWidget(setting_row("Голос", self.azure_voice_combo))
        self.azure_region_edit = QLineEdit(v.azure_region)
        self.azure_region_edit.setMaxLength(40)
        self.azure_region_edit.setPlaceholderText("westeurope")
        self.azure_region_edit.setMinimumWidth(200)
        self.azure_region_edit.setAccessibleName("Регіон Azure")
        self.azure_region_edit.textChanged.connect(self._on_region_changed)
        azure.addWidget(setting_row(
            "Регіон", self.azure_region_edit, "Регіон ресурсу Speech у порталі Azure, наприклад westeurope."))
        self.region_error = make_label("", "error")
        self.region_error.hide()
        azure.addWidget(self.region_error)
        azure.addWidget(self._build_key_row())
        card.body.addWidget(self.azure_box)

        self.speed_row = SliderRow("Швидкість мовлення", 60, 160, round(v.speed * 100), step=5,
                                   formatter=percent, description="100\u00a0% — звичайний темп.")
        card.body.addWidget(self.speed_row)
        self.volume_row = SliderRow("Гучність", 0, 100, v.volume, step=5, formatter=percent)
        card.body.addWidget(self.volume_row)

        test_row = QHBoxLayout()
        self.test_button = make_button("Прослухати", "accent")
        self.test_button.clicked.connect(self._test_voice)
        test_row.addWidget(self.test_button)
        self.test_result_label = make_label("", "hint")
        self.test_result_label.hide()
        test_row.addWidget(self.test_result_label, 1)
        card.body.addLayout(test_row)

        self.speak_combo = self._combo(SPEAK_CHOICES, v.speak, "Що вона говорить уголос")
        card.body.addWidget(setting_row(
            "Що вона говорить уголос", self.speak_combo,
            "«Лише відповіді» — голосом звучить те, що ти в неї питаєш; решта — у бульбашках."))
        self.speak_hidden_check = self._check(
            card, "Говорити, коли гра на весь екран ховає її",
            "Тоді вона коментує гру голосом, без бульбашок.", v.speak_when_hidden)
        return card

    def _build_key_row(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(make_label("Ключ Azure Speech", wrap=False))
        top.addStretch(1)
        self.azure_key_state_label = make_label("", "hint", wrap=False)
        top.addWidget(self.azure_key_state_label)
        layout.addLayout(top)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.azure_key_edit = SubmitLineEdit()
        self.azure_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.azure_key_edit.setMaxLength(512)
        self.azure_key_edit.setClearButtonEnabled(True)
        self.azure_key_edit.setAccessibleName("Ключ Azure Speech")
        self.azure_key_edit.textChanged.connect(
            lambda text: self.save_azure_key_button.setEnabled(bool(text.strip())))
        self.azure_key_edit.submitted.connect(self._save_key)
        self.save_azure_key_button = make_button("Зберегти ключ", "accent")
        self.save_azure_key_button.setEnabled(False)
        self.save_azure_key_button.clicked.connect(self._save_key)
        self.delete_azure_key_button = make_button("Видалити ключ", "danger")
        self.delete_azure_key_button.clicked.connect(self._delete_key)
        row.addWidget(self.azure_key_edit, 1)
        row.addWidget(self.save_azure_key_button)
        row.addWidget(self.delete_azure_key_button)
        layout.addLayout(row)
        self.azure_key_message_label = make_label("", "hint")
        self.azure_key_message_label.hide()
        layout.addWidget(self.azure_key_message_label)
        layout.addWidget(InfoLabel(AZURE_KEY_HINT))
        return box

    def _build_recognition_card(self, v: VoiceSettings) -> QWidget:
        card = Card()
        card.body.addWidget(SectionHeader("Розпізнавання мовлення"))
        card.body.addWidget(make_label(PRIVACY_NOTE, "banner"))
        self.stt_enabled_check = self._check(
            card, "Розмовляти з нею голосом",
            "Говори з нею клавішею розмови, кнопкою мікрофона в чаті або — якщо обрано нижче — просто "
            "вголос. Мовлення розпізнається на цьому ПК.", v.stt_enabled)

        self.mic_box = QWidget()
        mic = QHBoxLayout(self.mic_box)
        mic.setContentsMargins(0, 0, 0, 0)
        self.mic_permission_label = make_label(MIC_NEEDED, "warning")
        self.allow_mic_button = make_button("Дозволити…", "accent")
        self.allow_mic_button.clicked.connect(self._allow_microphone)
        mic.addWidget(self.mic_permission_label, 1)
        mic.addWidget(self.allow_mic_button)
        card.body.addWidget(self.mic_box)

        stt_specs = tuple((spec.id, model_choice_title(spec))
                          for spec in CATALOG.values() if spec.kind == "stt")
        self.stt_model_combo = self._combo(stt_specs, v.stt_model, "Модель розпізнавання")
        self.stt_model_combo.currentIndexChanged.connect(self._on_stt_model_changed)
        card.body.addWidget(setting_row("Модель розпізнавання", self.stt_model_combo))
        self.stt_model_row = self._model_row()
        card.body.addWidget(self.stt_model_row)

        self.hotkey_combo = QComboBox()
        self.hotkey_combo.setAccessibleName("Клавіша розмови")
        for preset in hotkey_win.PRESETS:
            self.hotkey_combo.addItem(hotkey_title(preset), preset)
        self._select_hotkey(v.ptt_hotkey)
        card.body.addWidget(setting_row(
            "Клавіша розмови", self.hotkey_combo,
            "Працює, навіть коли відкрито гру. Вона бачить лише цю комбінацію, а не інші натискання."))
        self.hotkey_note = make_label(HOTKEY_WINDOWS_ONLY, "warning")
        self.hotkey_note.setVisible(not hotkey_win.supported())
        card.body.addWidget(self.hotkey_note)
        self.ptt_mode_combo = self._combo(PTT_MODE_CHOICES, v.ptt_mode, "Як говорити")
        card.body.addWidget(setting_row(
            "Як говорити", self.ptt_mode_combo,
            "«Натиснути й говорити» — вона сама помітить паузу в кінці фрази. «Тримати» — як рація: "
            "говори, поки тримаєш клавішу."))
        self.pause_row = SliderRow("Пауза в кінці фрази", 400, 2500, v.end_silence_ms, step=100,
                                   formatter=format_seconds,
                                   description="Скільки тиші чекати, перш ніж вирішити, що фразу закінчено.")
        card.body.addWidget(self.pause_row)

        card.body.addWidget(self._build_listen_box(v))

        self.input_device_combo = QComboBox()
        self.input_device_combo.setAccessibleName("Мікрофон")
        self.input_device_combo.setMinimumWidth(220)
        self.output_device_combo = QComboBox()
        self.output_device_combo.setAccessibleName("Динаміки")
        self.output_device_combo.setMinimumWidth(220)
        card.body.addWidget(setting_row("Мікрофон", self.input_device_combo))
        card.body.addWidget(setting_row("Динаміки чи навушники", self.output_device_combo))
        self.devices_note = InfoLabel("")
        self.devices_note.hide()
        card.body.addWidget(self.devices_note)
        self._fill_devices(v.input_device, v.output_device)
        return card

    def _build_listen_box(self, v: VoiceSettings) -> QWidget:
        """"Як слухати": push-to-talk, by name, or everything - and the pauses."""
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(make_label("Як слухати", "value", wrap=False))
        self.listen_group = QButtonGroup(self)
        self.listen_group.setExclusive(True)
        self.listen_buttons: dict[str, QRadioButton] = {}
        self.listen_hints: dict[str, InfoLabel] = {}
        for index, (value, text, _hint) in enumerate(LISTEN_CHOICES):
            radio = QRadioButton(text)
            radio.setAccessibleName(text)
            self.listen_group.addButton(radio, index)
            self.listen_buttons[value] = radio
            layout.addWidget(radio)
            hint = InfoLabel("")
            hint.setContentsMargins(28, 0, 0, 4)
            self.listen_hints[value] = hint
            layout.addWidget(hint)
            radio.toggled.connect(lambda checked, mode=value: self._on_listen_toggled(mode, checked))
        self._check_listen_button(v.listen_mode)

        self.follow_up_row = SliderRow(
            "Розмова без імені", 0, 120, v.follow_up_seconds, step=5, formatter=format_follow_up,
            description="Скільки після її відповіді можна говорити далі, не називаючи її. "
                        "0 — щоразу на ім'я.")
        layout.addWidget(self.follow_up_row)
        self.pause_in_calls_row = CheckRow(
            "Не слухати під час дзвінків",
            "Пауза, поки на передньому плані Discord, Teams, Zoom, Skype чи інша програма для дзвінків. "
            "У месенджерах, пошті, банкінгу й менеджерах паролів вона не слухає ніколи. Програму "
            "вона розпізнає, лише якщо їй дозволено бачити, яка програма відкрита, і лише ту, що "
            "на передньому плані: дзвониш у фоні під час гри — постав мікрофон на паузу в її меню.")
        self.pause_in_calls_row.setChecked(v.pause_in_calls)
        self.pause_in_calls_check = self.pause_in_calls_row.checkbox
        layout.addWidget(self.pause_in_calls_row)
        self.fullscreen_row = CheckRow(
            "Слухати й у грі на весь екран",
            "Без цього мікрофон стає на паузу, поки гра чи відео на весь екран.")
        self.fullscreen_row.setChecked(v.wake_in_fullscreen)
        self.wake_in_fullscreen_check = self.fullscreen_row.checkbox
        layout.addWidget(self.fullscreen_row)
        layout.addWidget(InfoLabel(DND_NOTE))
        return box

    # --- small builders ---------------------------------------------------------------------------

    @staticmethod
    def _check(card: Card, text: str, hint: str, checked: bool) -> QCheckBox:
        row = CheckRow(text, hint)
        row.setChecked(checked)
        card.body.addWidget(row)
        return row.checkbox

    @staticmethod
    def _combo(choices: tuple[tuple[str, str], ...], current: str, name: str) -> QComboBox:
        combo = QComboBox()
        for value, text in choices:
            combo.addItem(text, value)
        combo.setCurrentIndex(max(0, combo.findData(current)))
        combo.setMinimumWidth(220)
        combo.setAccessibleName(name)
        return combo

    def _model_row(self) -> ModelRow:
        row = ModelRow()
        row.download_requested.connect(self._start_download)
        row.cancel_requested.connect(self._cancel_download)
        row.remove_requested.connect(self._remove_model)
        return row

    # --- fields ------------------------------------------------------------------------------------

    def _bind_fields(self) -> tuple[_Field, ...]:
        def check(name: str, box: QCheckBox) -> _Field:
            return _Field(name, box.isChecked, lambda value: box.setChecked(bool(value)))

        def combo(name: str, box: QComboBox, default: str) -> _Field:
            return _Field(name, lambda: str(box.currentData() or default),
                          lambda value: box.setCurrentIndex(max(0, box.findData(value))))

        def slider(name: str, row: SliderRow) -> _Field:
            return _Field(name, row.value, lambda value: row.setValue(int(value)))

        def device(name: str, box: QComboBox) -> _Field:
            return _Field(name, lambda: str(box.currentData() or ""),
                          lambda value: self._select_device(box, str(value or "")))

        return (
            check("tts_enabled", self.tts_enabled_check),
            combo("tts_engine", self.engine_combo, "piper"),
            combo("piper_speaker", self.speaker_combo, "tetiana"),
            combo("azure_voice", self.azure_voice_combo, "uk-UA-PolinaNeural"),
            _Field("azure_region", self._region_text,
                   lambda value: self.azure_region_edit.setText(str(value))),
            _Field("speed", lambda: round(self.speed_row.value() / 100, 2),
                   lambda value: self.speed_row.setValue(round(float(value) * 100))),
            slider("volume", self.volume_row),
            combo("speak", self.speak_combo, "all"),
            check("speak_when_hidden", self.speak_hidden_check),
            check("stt_enabled", self.stt_enabled_check),
            combo("stt_model", self.stt_model_combo, "whisper-small"),
            _Field("ptt_hotkey", lambda: str(self.hotkey_combo.currentData() or "ctrl+alt+space"),
                   lambda value: self._select_hotkey(str(value))),
            combo("ptt_mode", self.ptt_mode_combo, "toggle"),
            slider("end_silence_ms", self.pause_row),
            _Field("listen_mode", self.listen_mode, self._set_listen_mode),
            slider("follow_up_seconds", self.follow_up_row),
            check("pause_in_calls", self.pause_in_calls_check),
            check("wake_in_fullscreen", self.wake_in_fullscreen_check),
            device("input_device", self.input_device_combo),
            device("output_device", self.output_device_combo),
        )

    def _read_fields(self) -> dict[str, object]:
        return {field.name: field.read() for field in self._fields}

    def _read_settings(self) -> Settings | None:
        try:
            return self._facade.settings().copy()
        except Exception:
            log.exception("reading the current settings failed")
            return None

    def _region_text(self) -> str:
        return self.azure_region_edit.text().strip().lower()

    def _select_hotkey(self, text: str) -> None:
        index = self.hotkey_combo.findData(text)
        if index < 0:  # a combination set by hand (still valid: the settings checked it)
            self.hotkey_combo.addItem(hotkey_title(text), text)
            index = self.hotkey_combo.count() - 1
        self.hotkey_combo.setCurrentIndex(index)

    @staticmethod
    def _select_device(combo: QComboBox, name: str) -> None:
        index = combo.findData(name)
        if index < 0:
            combo.addItem(f"{name} {NOT_CONNECTED}", name)
            index = combo.count() - 1
        combo.setCurrentIndex(index)

    def _fill_devices(self, input_device: str | None = None, output_device: str | None = None) -> None:
        """(Re)list the audio devices, keeping the selection (a device may have been
        unplugged: it stays selectable, marked as not connected)."""
        try:
            inputs, outputs = self._facade.voice_devices()
            failed = False
        except Exception:
            log.exception("listing audio devices failed")
            inputs, outputs, failed = [], [], True
        for combo, names, keep in ((self.input_device_combo, inputs, input_device),
                                   (self.output_device_combo, outputs, output_device)):
            current = keep if keep is not None else str(combo.currentData() or "")
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(DEFAULT_DEVICE, "")
            for name in dict.fromkeys(str(n) for n in names if str(n).strip()):
                combo.addItem(name, name)
            if current:
                self._select_device(combo, current)
            else:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)
        if failed or not (inputs or outputs):
            self.devices_note.setText(
                "Список пристроїв зараз недоступний — використовується типовий пристрій системи.")
            self.devices_note.show()
        else:
            self.devices_note.hide()

    # --- reacting to edits ---------------------------------------------------------------------------

    def _update_names(self) -> None:
        name = self._name
        self.voice_header.setText("Голос Мії" if name == "Мія" else "Її голос")
        self.lead_label.setText(
            f"{name} може говорити вголос і чути тебе. Усе це необов'язково: без голосу вона просто "
            "пише в бульбашках і чаті.")
        voc = vocative(name)
        for value, _text, hint in LISTEN_CHOICES:
            self.listen_hints[value].setText(hint.format(voc=voc))

    # --- "Як слухати" ------------------------------------------------------------------------------

    def listen_mode(self) -> str:
        """The listen mode the page shows ("push", "name" or "open")."""
        return next((mode for mode, button in self.listen_buttons.items() if button.isChecked()), "push")

    def _set_listen_mode(self, value: object) -> None:
        """Show *value* (loaded from the settings: never asks)."""
        mode = str(value) if str(value) in _EXPOSURE else "push"
        self._listen_mode = mode
        self._check_listen_button(mode)
        self._update_listen_controls()

    def _check_listen_button(self, mode: str) -> None:
        self._setting_mode = True
        try:
            self.listen_buttons.get(mode, self.listen_buttons["push"]).setChecked(True)
        finally:
            self._setting_mode = False

    def _on_listen_toggled(self, mode: str, checked: bool) -> None:
        if not checked or self._setting_mode:
            return
        previous = self._listen_mode
        if _EXPOSURE[mode] > _EXPOSURE.get(previous, 0):
            text = OPEN_MIC_CONFIRM_OPEN if mode == "open" else OPEN_MIC_CONFIRM_NAME
            if not self._confirm(OPEN_MIC_CONFIRM_TITLE, text):
                self._check_listen_button(previous)  # declined: nothing changes
                return
        self._listen_mode = mode
        self._update_listen_controls()

    def _update_listen_controls(self) -> None:
        """The follow-up only matters by name; the pauses only with an open microphone."""
        mode = self.listen_mode()
        self.follow_up_row.setEnabled(mode == "name")
        self.pause_in_calls_row.setEnabled(mode != "push")
        self.fullscreen_row.setEnabled(mode != "push")

    def _on_engine_changed(self, _index: int = -1) -> None:
        azure = self.engine() == "azure"
        self.piper_box.setVisible(not azure)
        self.azure_box.setVisible(azure)
        if not azure:
            self.region_error.hide()
        self._refresh_cloud()
        self._refresh_key_state()

    def _on_region_changed(self, _text: str) -> None:
        if not self.region_error.isHidden() and _REGION_RE.fullmatch(self._region_text()):
            self.region_error.hide()

    def _on_stt_model_changed(self, _index: int = -1) -> None:
        self._render_models()

    def _voice_title(self) -> str:
        """What is speaking, per the applied settings: "Piper, Тетяна"."""
        settings = self._read_settings()
        if settings is None:
            return "Piper"
        v = settings.voice
        if v.tts_engine == "azure":
            return f"Azure, {AZURE_VOICES.get(v.azure_voice, v.azure_voice)}"
        return f"Piper, {PIPER_SPEAKERS.get(v.piper_speaker, v.piper_speaker)}"

    # --- permissions ---------------------------------------------------------------------------------

    def _granted(self, permission: Permission) -> bool:
        try:
            return bool(self._facade.permissions.is_granted(permission))
        except Exception:
            log.exception("reading the %s permission failed", permission.value)
            return False

    def _refresh_cloud(self) -> None:
        if self.engine() != "azure":
            return
        granted = self._granted(Permission.CLOUD_AI)
        self.azure_cloud_note.setText(AZURE_CLOUD_OK if granted else AZURE_CLOUD_WARNING)
        set_style_property(self.azure_cloud_note, "role", "hint" if granted else "warning")

    def _refresh_mic(self) -> None:
        self.mic_box.setVisible(not self._granted(Permission.MICROPHONE))

    def _allow_microphone(self) -> None:
        if not self._confirm(MICROPHONE_CONSENT_TITLE, MICROPHONE_CONSENT_TEXT):
            return
        try:
            self._facade.permissions.grant(Permission.MICROPHONE)
        except (PermissionError, ValueError) as exc:  # unavailable in this build
            log.warning("could not grant %s: %s", Permission.MICROPHONE.value, exc)
            self.mic_permission_label.setText("Цей дозвіл зараз не можна надати.")
            return
        self.mic_permission_label.setText(MIC_NEEDED)
        self.permissions_changed.emit()
        self.refresh()

    # --- Azure key ---------------------------------------------------------------------------------

    def _refresh_key_state(self) -> None:
        if self.engine() != "azure":
            return
        try:
            has_key = bool(self._facade.voice_has_azure_key())
        except Exception:
            log.exception("asking whether an Azure key is stored failed")
            has_key = False
        text, role = ("🔑 Ключ збережено", "success") if has_key else ("Ключ ще не збережено", "warning")
        self.azure_key_state_label.setText(text)
        set_style_property(self.azure_key_state_label, "role", role)
        self.delete_azure_key_button.setEnabled(has_key)
        self.azure_key_edit.setPlaceholderText("Новий ключ замінить збережений" if has_key else "Встав ключ")

    def _key_message(self, text: str, *, error: bool) -> None:
        self.azure_key_message_label.setText(text)
        set_style_property(self.azure_key_message_label, "role", "error" if error else "success")
        self.azure_key_message_label.setVisible(bool(text))

    def _save_key(self) -> bool:
        key = self.azure_key_edit.text().strip()
        if not key:
            return False
        if not _KEY_RE.fullmatch(key):
            self._key_message("Це не схоже на ключ: він має бути одним рядком без пробілів.", error=True)
            return False
        try:
            self._facade.voice_set_azure_key(key)
        except Exception as exc:
            # The type only: an exception message could quote the key.
            log.error("storing the Azure key failed: %s", type(exc).__name__)
            self._key_message("Не вдалося зберегти ключ.", error=True)
            return False
        self.azure_key_edit.clear()
        self._key_message("Ключ збережено ✓", error=False)
        self._refresh_key_state()
        self.refresh_status()
        return True

    def _delete_key(self) -> None:
        if not self._confirm(
                "Видалити ключ?",
                f"Ключ Azure Speech буде видалено з цього комп'ютера. Без нього {self._name} не "
                "зможе говорити голосом Azure, доки ти не додаси новий."):
            return
        try:
            self._facade.voice_set_azure_key("")
        except Exception as exc:
            log.error("deleting the Azure key failed: %s", type(exc).__name__)
            self._key_message("Не вдалося видалити ключ.", error=True)
            return
        self._key_message("Ключ видалено.", error=False)
        self._refresh_key_state()
        self.refresh_status()

    # --- "Прослухати" ------------------------------------------------------------------------------

    def _apply_own_edits(self) -> bool:
        """Apply this page's pending edits (and a typed key), so the sample uses what the page shows."""
        if not self.save_pending_key():
            return False
        if not self.has_edits():
            return True
        if not self.validate():
            return False
        current = self._read_settings()
        if current is None:
            return False
        try:
            self._facade.apply_settings(self.collect(current))
        except Exception:
            log.exception("applying the voice settings failed")
            return False
        applied = self._read_settings()
        if applied is not None:
            self.sync(applied, edited_too=True)
        else:
            self.reset_baseline()
        self.settings_applied.emit()
        return True

    def _test_voice(self) -> None:
        if self._test_busy:
            return
        if not self._apply_own_edits():
            self._show_test_result(False, "Спершу виправ налаштування вище — їх не вдалося зберегти.")
            return
        self._test_seq += 1
        seq = self._test_seq
        self._test_busy = True
        self.test_button.setEnabled(False)
        self.test_button.setText("Звучить…")
        self.test_result_label.hide()
        self.test_watchdog.start()

        def done(ok: bool, message: str) -> None:
            if is_alive(self) and self._test_busy and seq == self._test_seq:
                self._end_test()
                self._show_test_result(bool(ok), str(message))
                self.refresh_status()

        try:
            self._facade.voice_test(done)
        except Exception as exc:
            log.error("starting the voice sample failed: %s", type(exc).__name__)
            if self._test_busy and seq == self._test_seq:
                self._end_test()
                self._show_test_result(False, "Не вдалося запустити озвучення.")

    def _end_test(self) -> None:
        self._test_busy = False
        self.test_watchdog.stop()
        self.test_button.setEnabled(True)
        self.test_button.setText("Прослухати")

    def _on_test_timeout(self) -> None:
        if not self._test_busy:
            return
        self._test_seq += 1  # a late answer belongs to a request given up on
        self._end_test()
        self._show_test_result(False, "Звук так і не з'явився — спробуй ще раз.")

    def _show_test_result(self, ok: bool, message: str) -> None:
        text = "✓ Ось так вона звучить." if ok else f"✗ {message or 'Не вдалося озвучити.'}"
        self.test_result_label.setText(text)
        set_style_property(self.test_result_label, "role", "success" if ok else "error")
        self.test_result_label.show()

    # --- models ------------------------------------------------------------------------------------

    def _selected_stt_model(self) -> str:
        return str(self.stt_model_combo.currentData() or "whisper-small")

    def _render_models(self, *, reload: bool = False) -> None:
        """Show both model rows; *reload* re-reads which models are installed (file
        checks - not on every progress tick)."""
        if reload or not self._model_infos:
            try:
                self._model_infos = {spec.id: (spec, bool(installed))
                                     for spec, installed in self._facade.voice_models()}
            except Exception:
                log.exception("listing the voice models failed")
                self._model_infos = {}
        busy = self._downloading
        done, total = self._progress
        for row, model_id in ((self.voice_model_row, DEFAULT_TTS_MODEL),
                              (self.stt_model_row, self._selected_stt_model())):
            spec, installed = self._model_infos.get(model_id, (CATALOG.get(model_id), False))
            row.set_model(spec, installed, downloading=busy == model_id, done=done, total=total,
                          other_busy=busy is not None and busy != model_id)

    def _row_for(self, model_id: str) -> ModelRow:
        return self.voice_model_row if model_id == DEFAULT_TTS_MODEL else self.stt_model_row

    def _start_download(self, model_id: str) -> None:
        if self._downloading is not None or not model_id:
            return
        spec = CATALOG.get(model_id)
        self._download_seq += 1
        seq = self._download_seq
        self._downloading = model_id
        self._progress = (0, spec.size if spec is not None else 0)
        self._row_for(model_id).show_message("")
        self._render_models()

        def on_progress(done: int, total: int) -> None:
            if is_alive(self) and seq == self._download_seq and self._downloading == model_id:
                self._progress = (int(done), int(total))
                self._render_models()

        def on_done(ok: bool, message: str) -> None:
            if not is_alive(self) or seq != self._download_seq:
                return
            if self._downloading == model_id:
                self._end_download()
            self._row_for(model_id).show_message(str(message), error=not ok)
            self.refresh_status()

        self.download_timer.start()
        try:
            self._facade.voice_download(model_id, on_progress, on_done)
        except Exception as exc:
            log.error("starting the download failed: %s", type(exc).__name__)
            if seq == self._download_seq and self._downloading == model_id:
                self._end_download()
                self._row_for(model_id).show_message("Не вдалося почати завантаження.", error=True)

    def _end_download(self) -> None:
        self._downloading = None
        self._progress = (0, 0)
        self.download_timer.stop()
        self._render_models(reload=True)

    def _cancel_download(self) -> None:
        try:
            self._facade.voice_cancel_download()
        except Exception as exc:
            log.error("cancelling the download failed: %s", type(exc).__name__)

    def _poll_download(self) -> None:
        """Follow a running download - also one started before this page was opened."""
        try:
            state = self._facade.voice_download_state()
        except Exception:
            log.exception("reading the download state failed")
            state = None
        if state is None:
            if self._downloading is not None:
                self._end_download()
            return
        if self._downloading != state.model_id:
            self._download_seq += 1  # callbacks of an older request no longer apply
            self._downloading = state.model_id
            self.download_timer.start()
        self._progress = (int(state.done), int(state.total))
        self._render_models()

    def _remove_model(self, model_id: str) -> None:
        spec = CATALOG.get(model_id)
        if spec is None:
            return
        if not self._confirm(
                "Видалити модель?",
                f"«{spec.title}» ({format_size(spec.size)}) буде видалено з цього комп'ютера. "
                "Завантажити її знову можна будь-коли."):
            return
        try:
            self._facade.voice_remove_model(model_id)
        except Exception as exc:
            log.error("removing the model failed: %s", type(exc).__name__)
            self._row_for(model_id).show_message("Не вдалося видалити модель.", error=True)
            return
        self._render_models(reload=True)
        self._row_for(model_id).show_message("Модель видалено.")
        self.refresh_status()

    # --- visibility-driven polling ---------------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._fill_devices()
        self.refresh()
        self.status_timer.start()

    def hideEvent(self, event: QHideEvent) -> None:
        self.status_timer.stop()
        super().hideEvent(event)
