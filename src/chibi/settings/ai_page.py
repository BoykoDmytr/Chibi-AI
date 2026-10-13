"""The "AI" page of the settings dialog: which model she talks through, keys,
budgets and the live connection status.

Design notes - why it works this way:

* **Settings vs. secrets vs. permissions.**  Provider, model, budgets... are
  ordinary settings: edited here, applied with the rest of the dialog through
  :meth:`AIPage.collect`.  API keys are *not* settings - they go straight to
  the OS credential store through ``facade.ai_set_key`` and the field is
  cleared at once, so a key never sits in a widget, a Settings copy or a log.
  The stored key is never shown again; the page only says whether one exists.
  ``Permission.CLOUD_AI`` is granted only after the explicit consent text
  (the same one the privacy page uses) and goes straight to the manager.
* **Baseline, like the dialog.**  The page remembers what each widget showed
  when it was last loaded from the app; ``collect`` writes only the fields
  that differ, so a stale widget never reverts a change made elsewhere.
* **Checks use the page's own values.**  ``ai_test`` / ``ai_list_models`` run
  against the *applied* settings, so before either check the page applies its
  own pending edits (only the AI section) - otherwise "Перевірити з'єднання"
  right after switching provider would silently test the old one.
* **Never wait on the model.**  Both checks are asynchronous: the button is
  disabled while waiting, a late answer for a superseded request (or for a
  page that was closed meanwhile) is dropped, and a watchdog re-enables the
  button if an answer never comes.
* **Cloud or local** is decided here for the *selected* (maybe not yet
  applied) provider with the same rule as ``chibi.ai.providers.factory.is_cloud``
  - mirrored rather than imported because the settings windows depend only on
  the facade (a test keeps the two in step).
"""

from __future__ import annotations

import ipaddress
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QHideEvent, QShowEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from chibi.core.config import AISettings, Settings
from chibi.core.events import Permission
from chibi.settings.facade import AIStatusView, AppFacade
from chibi.settings.widgets import (
    Card,
    CheckRow,
    InfoLabel,
    SectionHeader,
    SubmitLineEdit,
    ai_provider_name,
    ai_title,
    ask_yes_no,
    count_uk,
    format_usd,
    is_alive,
    make_button,
    make_label,
    set_style_property,
    setting_row,
)

log = logging.getLogger(__name__)

ConfirmFn = Callable[[str, str], bool]

#: How often the status line is re-read while the page is visible.
STATUS_REFRESH_MS = 3000
#: Give up waiting for a connection test / model list after this long.  The
#: service always answers; this only keeps a lost answer from disabling the
#: buttons forever.
WATCHDOG_MS = 240_000

#: (AISettings.provider, combo text) in display order.
PROVIDER_CHOICES: tuple[tuple[str, str], ...] = (
    ("none", "Вимкнено"),
    ("ollama", "Ollama — локально на цьому ПК"),
    ("anthropic", "Anthropic Claude"),
    ("openai", "OpenAI-сумісний сервер"),
)
#: (AISettings.effort, combo text)
EFFORT_CHOICES: tuple[tuple[str, str], ...] = (
    ("low", "Швидко й дешево (low)"),
    ("medium", "Збалансовано (medium)"),
    ("high", "Ретельно (high)"),
)
#: Suggestions for the Claude model box (it stays editable for any other id).
CLAUDE_MODELS: tuple[str, ...] = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
#: Providers whose API key the page manages.
KEY_PROVIDERS: frozenset[str] = frozenset({"anthropic", "openai"})

MODEL_PLACEHOLDERS: dict[str, str] = {
    "ollama": "типово — перша встановлена модель",
    "anthropic": "типово — claude-opus-5-5",
    "openai": "назва моделі (обов'язково)",
}

PROVIDER_NOTES: dict[str, str] = {
    "none": (
        "AI вимкнено: {name} говорить готовими репліками, а в чаті розуміє лише прості фрази. "
        "Нічого нікуди не надсилається."
    ),
    "ollama": (
        "Ollama запускає модель просто на цьому комп'ютері: розмови нікуди не надсилаються, і це "
        "безкоштовно. Потрібно встановити Ollama (ollama.com) і завантажити хоча б одну модель. "
        "Чим більша модель, тим краще вона говорить українською, але тим повільніше відповідає."
    ),
    "anthropic": (
        "Моделі Claude працюють у хмарі Anthropic. Потрібні API-ключ (console.anthropic.com) і твій "
        "дозвіл на хмарний AI. Оплата — за використані токени, з твого рахунку в Anthropic."
    ),
    "openai": (
        "Будь-який сервер з OpenAI-сумісним API: сам OpenAI, LM Studio, llama.cpp тощо. Якщо сервер "
        "працює на цьому ПК, розмови його не залишають."
    ),
}

EFFORT_NOTE = (
    "Типова модель claude-opus-5-5 коштує $4 за 1 млн вхідних і $20 за 1 млн вихідних токенів, "
    "claude-haiku-5-5 — $0.10 і $0.50: відповідає простіше, зате в десятки разів дешевше. Вибір за тобою — "
    "денний ліміт витрат нижче не дасть вийти за бюджет. «Ретельно» означає довші роздуми моделі "
    "перед відповіддю: розумніше, але повільніше й дорожче."
)
KEY_HINT = (
    "Ключ зберігається в диспетчері облікових даних Windows і більше ніде не показується — навіть "
    "тут. У журнали він не потрапляє."
)
CHECKS_NOTE = (
    "Перевірка й пошук моделей використовують налаштування цієї сторінки: перед ними її зміни буде "
    "збережено."
)

CLOUD_AI_CONSENT_TITLE = "Дозволити хмарний AI?"
CLOUD_AI_CONSENT_TEXT = (
    "Щоб відповідати через хмарного провайдера (Anthropic Claude або OpenAI-сумісний сервер не на "
    "цьому комп'ютері), програма надсилатиме йому:\n\n"
    "• те, що ти пишеш у чаті, і кілька останніх реплік розмови;\n"
    "• її ім'я, характер і те, як вона до тебе звертається;\n"
    "• її настрій і потреби (голод, енергію, нудьгу тощо) та час доби;\n"
    "• назву активної програми чи гри — лише якщо дозволено бачити програми;\n"
    "• доречні спогади з її пам'яті.\n\n"
    "Знімки екрана, заголовки вікон і натискання клавіш не надсилаються ніколи.\n\n"
    "Надіслане обробляє провайдер за власною політикою конфіденційності. Відкликати дозвіл можна "
    "будь-коли в розділі «Приватність» — він перестає діяти одразу, і вона повертається до "
    "готових реплік."
)

_URL_RE = re.compile(r"https?://[^\s/]+(/\S*)?")
_DEFAULT_URLS: dict[str, str] = {
    "ollama": AISettings().ollama_url,
    "openai": AISettings().openai_base_url,
}
#: An API key is one line of printable ASCII without spaces.
_KEY_RE = re.compile(r"[\x21-\x7e]{8,512}")


# --- pure helpers ----------------------------------------------------------------------------


def _host(url: str) -> str:
    try:
        return (urlsplit(url.strip()).hostname or "").lower()
    except ValueError:  # e.g. an unbalanced "[" in an IPv6 literal
        return ""


def is_loopback_url(url: str) -> bool:
    """True if *url* points at this machine (``localhost``, 127.0.0.0/8, ``::1``)."""
    host = _host(url)
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return ip.is_loopback or bool(mapped is not None and mapped.is_loopback)


def is_cloud_choice(provider: str, ollama_url: str, openai_base_url: str) -> bool:
    """Whether this choice sends data off the PC (mirrors ``factory.is_cloud``)."""
    if provider == "anthropic":
        return True
    if provider == "openai":
        return not is_loopback_url(openai_base_url)
    if provider == "ollama":
        return not is_loopback_url(ollama_url)
    return False


def normalize_url(raw: str) -> str:
    return raw.strip().rstrip("/")


def is_valid_url(raw: str) -> bool:
    return bool(_URL_RE.fullmatch(normalize_url(raw)))


@dataclass(frozen=True)
class StatusText:
    status: str
    role: str  # QLabel theme role
    usage: str  # "" = nothing worth showing


def describe_status(view: AIStatusView) -> StatusText:
    """The status line and today's usage, in Ukrainian."""
    if view.online:
        where = ("хмара: розмови надсилаються провайдерові" if view.cloud
                 else "локально: розмови не залишають цей ПК")
        status, role = f"● Працює: {ai_title(view.provider, view.model)} — {where}", "success"
    elif view.provider == "none":
        status, role = "● AI вимкнено — вона говорить готовими репліками", "hint"
    else:
        status, role = f"● Офлайн: {view.reason or 'AI зараз недоступний'}", "warning"
    if view.provider == "none" and view.requests_today <= 0:
        return StatusText(status, role, "")
    requests = f"Запитів сьогодні: {max(0, view.requests_today)} із {view.requests_limit_day}"
    if view.provider == "ollama":
        cost = "без оплати"
    elif not view.cost_known:
        cost = "вартість невідома"
    else:
        cost = f"витрачено {format_usd(view.cost_today_usd)} із {format_usd(view.cost_limit_usd)}"
    return StatusText(status, role, f"{requests} · {cost}")


@dataclass(frozen=True)
class _Field:
    """One ``Settings.ai.<name>`` field edited by a widget."""

    name: str
    read: Callable[[], object]
    write: Callable[[object], None]


class _Waiter:
    """Book-keeping for one asynchronous check (test or model list).

    Each start bumps a sequence number: an answer carrying an older number
    belongs to a superseded request (or to one the watchdog gave up on) and
    is dropped.
    """

    def __init__(self, owner: QWidget, on_timeout: Callable[[], None]) -> None:
        self.seq = 0
        self.busy = False
        self.timer = QTimer(owner)
        self.timer.setSingleShot(True)
        self.timer.setInterval(WATCHDOG_MS)
        self.timer.timeout.connect(on_timeout)

    def start(self) -> int:
        self.seq += 1
        self.busy = True
        self.timer.start()
        return self.seq

    def finish(self, seq: int) -> bool:
        """True if *seq* is the request we are waiting for (and stop waiting)."""
        if not self.busy or seq != self.seq:
            return False
        self.busy = False
        self.timer.stop()
        return True

    def give_up(self) -> None:
        self.seq += 1
        self.busy = False
        self.timer.stop()


class AIPage(QWidget):
    """Settings page for the AI provider.  See the module docstring."""

    #: A permission was granted here (the dialog refreshes its privacy page).
    permissions_changed = Signal()
    #: The page applied its own edits before a check (the dialog shows a note).
    settings_applied = Signal()

    def __init__(self, facade: AppFacade, confirm: ConfirmFn | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._facade = facade
        self._confirm: ConfirmFn = confirm or (lambda title, text: ask_yes_no(self, title, text))
        settings = self._read_settings() or Settings()
        ai = settings.ai
        self._name = settings.character.name
        self._loading = False
        self._shown_provider = ai.provider
        #: Model text typed per provider in this session (switching back restores it).
        self._models: dict[str, str] = {ai.provider: ai.model}
        #: Models reported by "Оновити список", per provider.
        self._listed: dict[str, list[str]] = {}
        self._test_wait = _Waiter(self, self._on_test_timeout)
        self._models_wait = _Waiter(self, self._on_models_timeout)
        self._status_failed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 0, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(make_label("AI", "title", wrap=False))
        layout.addWidget(InfoLabel(
            f"Мовна модель, через яку {self._name} розмовляє в чаті й формулює свої репліки. Коли "
            "їй говорити, як і раніше, вирішують правила й ліміти — модель обирає лише слова. Якщо AI "
            "недоступний, вона просто говорить готовими репліками."))

        layout.addWidget(self._build_status_card())
        layout.addWidget(self._build_cloud_banner())
        layout.addWidget(self._build_provider_card(ai))
        layout.addWidget(self._build_usage_card(ai))
        layout.addWidget(self._build_limits_card(ai))
        layout.addWidget(self._build_later_card())
        layout.addStretch(1)

        self._fields: tuple[_Field, ...] = self._bind_fields()
        self._loaded: dict[str, object] = self._read_fields()

        self.status_timer = QTimer(self)
        self.status_timer.setInterval(STATUS_REFRESH_MS)
        self.status_timer.timeout.connect(self.refresh_status)

        self._on_provider_changed()
        self.refresh()

    # --- public API ------------------------------------------------------------------------

    def provider(self) -> str:
        return str(self.provider_combo.currentData() or "none")

    def model_text(self) -> str:
        return self.model_combo.currentText().strip()[:120]

    def collect(self, settings: Settings) -> Settings:
        """Write the fields edited on this page into *settings* (in place; also returned).

        Only widgets whose value differs from what they were last loaded with
        are copied, and a URL that isn't valid is left as it was.
        """
        for field in self._fields:
            value = field.read()
            if value == self._loaded[field.name]:
                continue
            if field.name in ("ollama_url", "openai_base_url") and not is_valid_url(str(value)):
                continue
            setattr(settings.ai, field.name, value)
        return settings

    def has_edits(self) -> bool:
        return any(field.read() != self._loaded[field.name] for field in self._fields)

    def sync(self, settings: Settings, *, edited_too: bool) -> None:
        """Show *settings* (the app's current ones): every field after an apply,
        otherwise only those the user has not edited here."""
        ai = settings.ai
        self._name = settings.character.name
        self._loading = True
        try:
            for field in self._fields:
                if edited_too or field.read() == self._loaded[field.name]:
                    field.write(getattr(ai, field.name))
                    self._loaded[field.name] = field.read()
        finally:
            self._loading = False
        if edited_too:
            self._models = {self.provider(): self.model_text()}
        self._on_provider_changed()

    def reset_baseline(self) -> None:
        """Treat what the widgets show now as unedited (after a successful apply)."""
        self._loaded = self._read_fields()

    def validate(self) -> bool:
        """Check the fields that matter for the selected provider; shows inline errors."""
        ok = True
        for provider, edit, error in (("ollama", self.ollama_url_edit, self.ollama_url_error),
                                      ("openai", self.openai_url_edit, self.openai_url_error)):
            bad = self.provider() == provider and not is_valid_url(edit.text())
            error.setText(f"Адреса має починатися з http:// або https://, наприклад {_DEFAULT_URLS[provider]}"
                          if bad else "")
            error.setVisible(bad)
            ok = ok and not bad
        return ok

    def refresh(self) -> None:
        """Re-read what can change outside the page: status, key, permission."""
        self.refresh_status()
        self._refresh_key_state()
        self._refresh_cloud()

    def refresh_status(self) -> None:
        try:
            view = self._facade.ai_status()
        except Exception:
            if not self._status_failed:  # polled every few seconds: log the first failure only
                log.exception("reading the AI status failed")
            self._status_failed = True
            self.status_label.setText("● Стан AI зараз невідомий")
            set_style_property(self.status_label, "role", "warning")
            self.usage_label.hide()
            return
        self._status_failed = False
        text = describe_status(view)
        self.status_label.setText(text.status)
        set_style_property(self.status_label, "role", text.role)
        self.usage_label.setText(text.usage)
        self.usage_label.setVisible(bool(text.usage))

    # --- building ------------------------------------------------------------------------------

    def _build_status_card(self) -> QWidget:
        card = Card(spacing=6)
        card.body.addWidget(SectionHeader("Зараз"))
        self.status_label = make_label("", "hint")
        self.status_label.setAccessibleName("Стан AI")
        self.usage_label = make_label("", "hint")
        card.body.addWidget(self.status_label)
        card.body.addWidget(self.usage_label)
        return card

    def _build_cloud_banner(self) -> QWidget:
        self.cloud_banner = Card(spacing=8)
        self.cloud_banner_label = make_label("", "warning")
        self.cloud_banner.body.addWidget(self.cloud_banner_label)
        row = QHBoxLayout()
        row.addStretch(1)
        self.allow_cloud_button = make_button("Дозволити…", "accent")
        self.allow_cloud_button.clicked.connect(self._allow_cloud)
        row.addWidget(self.allow_cloud_button)
        self.cloud_banner.body.addLayout(row)
        return self.cloud_banner

    def _build_provider_card(self, ai: AISettings) -> QWidget:
        card = Card()
        card.body.addWidget(SectionHeader("Провайдер"))
        self.provider_combo = QComboBox()
        for value, text in PROVIDER_CHOICES:
            self.provider_combo.addItem(text, value)
        self.provider_combo.setCurrentIndex(max(0, self.provider_combo.findData(ai.provider)))
        self.provider_combo.setMinimumWidth(260)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        card.body.addWidget(setting_row("Сервіс AI", self.provider_combo))
        self.provider_note = InfoLabel("")
        card.body.addWidget(self.provider_note)
        self.cloud_granted_note = InfoLabel(
            "Дозвіл на хмарний AI надано. Відкликати його можна в розділі «Приватність» — він "
            "перестане діяти одразу.")
        card.body.addWidget(self.cloud_granted_note)

        # --- connection details (hidden while AI is off)
        self.connection_box = QWidget()
        box = QVBoxLayout(self.connection_box)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(10)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.model_combo.setMinimumWidth(240)
        self.model_combo.lineEdit().setMaxLength(120)
        self.model_combo.setEditText(ai.model)
        self.model_combo.editTextChanged.connect(self._on_model_edited)
        self.refresh_models_button = make_button("Оновити список")
        self.refresh_models_button.clicked.connect(self._list_models)
        model_field = QWidget()
        model_layout = QHBoxLayout(model_field)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(8)
        model_layout.addWidget(self.model_combo, 1)
        model_layout.addWidget(self.refresh_models_button)
        box.addWidget(setting_row("Модель", model_field))
        self.models_status_label = make_label("", "hint")
        self.models_status_label.hide()
        box.addWidget(self.models_status_label)
        self.model_warning = make_label("", "warning")
        self.model_warning.hide()
        box.addWidget(self.model_warning)

        self.ollama_url_edit = QLineEdit(ai.ollama_url)
        self.ollama_url_edit.setMaxLength(300)
        self.ollama_url_edit.setPlaceholderText(AISettings().ollama_url)
        self.ollama_url_edit.setMinimumWidth(240)
        self.ollama_url_edit.textChanged.connect(self._on_url_changed)
        self.ollama_url_row = QWidget()
        self.ollama_url_error = make_label("", "error")
        self.ollama_url_error.hide()
        self._stack(self.ollama_url_row, setting_row(
            "Адреса Ollama", self.ollama_url_edit,
            "Зазвичай змінювати не треба. Якщо Ollama працює на іншому комп'ютері, вона вважається "
            "хмарою."), self.ollama_url_error)
        box.addWidget(self.ollama_url_row)

        self.openai_url_edit = QLineEdit(ai.openai_base_url)
        self.openai_url_edit.setMaxLength(300)
        self.openai_url_edit.setPlaceholderText(AISettings().openai_base_url)
        self.openai_url_edit.setMinimumWidth(240)
        self.openai_url_edit.textChanged.connect(self._on_url_changed)
        self.openai_url_row = QWidget()
        self.openai_url_error = make_label("", "error")
        self.openai_url_error.hide()
        self._stack(self.openai_url_row, setting_row(
            "Адреса сервера", self.openai_url_edit,
            "Наприклад, http://localhost:1234/v1 для LM Studio."), self.openai_url_error)
        box.addWidget(self.openai_url_row)

        box.addWidget(self._build_key_row())
        box.addWidget(self._build_price_row(ai))
        box.addWidget(self._build_effort_row(ai))

        test_row = QHBoxLayout()
        self.test_button = make_button("Перевірити з'єднання", "accent")
        self.test_button.clicked.connect(self._test_connection)
        test_row.addWidget(self.test_button)
        test_row.addStretch(1)
        box.addLayout(test_row)
        self.test_result_label = make_label("", "hint")
        self.test_result_label.hide()
        box.addWidget(self.test_result_label)
        box.addWidget(InfoLabel(CHECKS_NOTE))
        card.body.addWidget(self.connection_box)
        return card

    def _build_key_row(self) -> QWidget:
        self.key_row = QWidget()
        layout = QVBoxLayout(self.key_row)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(make_label("API-ключ", wrap=False))
        top.addStretch(1)
        self.key_state_label = make_label("", "hint", wrap=False)
        top.addWidget(self.key_state_label)
        layout.addLayout(top)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.key_edit = SubmitLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setMaxLength(512)
        self.key_edit.setClearButtonEnabled(True)
        self.key_edit.setAccessibleName("API-ключ")
        self.key_edit.textChanged.connect(self._on_key_typed)
        self.key_edit.submitted.connect(self._save_key)
        self.save_key_button = make_button("Зберегти ключ", "accent")
        self.save_key_button.clicked.connect(self._save_key)
        self.save_key_button.setEnabled(False)
        self.delete_key_button = make_button("Видалити ключ", "danger")
        self.delete_key_button.clicked.connect(self._delete_key)
        row.addWidget(self.key_edit, 1)
        row.addWidget(self.save_key_button)
        row.addWidget(self.delete_key_button)
        layout.addLayout(row)
        self.key_message_label = make_label("", "hint")
        self.key_message_label.hide()
        layout.addWidget(self.key_message_label)
        layout.addWidget(InfoLabel(KEY_HINT))
        return self.key_row

    def _build_price_row(self, ai: AISettings) -> QWidget:
        self.price_row = QWidget()
        layout = QVBoxLayout(self.price_row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.price_input_spin = self._money(0.0, 1000.0, ai.openai_price_input_per_mtok, decimals=3)
        self.price_output_spin = self._money(0.0, 1000.0, ai.openai_price_output_per_mtok, decimals=3)
        layout.addWidget(setting_row("Ціна вхідних токенів, за 1 млн", self.price_input_spin))
        layout.addWidget(setting_row("Ціна вихідних токенів, за 1 млн", self.price_output_spin))
        layout.addWidget(InfoLabel(
            "Потрібна, щоб рахувати витрати й дотримуватися денного ліміту. 0 — ціна невідома: тоді "
            "витрати не рахуються й діють лише ліміти кількості запитів."))
        return self.price_row

    def _build_effort_row(self, ai: AISettings) -> QWidget:
        self.effort_row = QWidget()
        layout = QVBoxLayout(self.effort_row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.effort_combo = QComboBox()
        for value, text in EFFORT_CHOICES:
            self.effort_combo.addItem(text, value)
        self.effort_combo.setCurrentIndex(max(0, self.effort_combo.findData(ai.effort)))
        layout.addWidget(setting_row("Ретельність відповідей", self.effort_combo))
        self.effort_note = InfoLabel(EFFORT_NOTE)
        layout.addWidget(self.effort_note)
        return self.effort_row

    def _build_usage_card(self, ai: AISettings) -> QWidget:
        card = Card()
        card.body.addWidget(SectionHeader("Для чого"))
        self.use_chat_check = self._check(
            card, "Відповідати в чаті через AI",
            "Інакше в чаті працюють лише прості готові відповіді.", ai.use_for_chat)
        self.use_reactions_check = self._check(
            card, "Формулювати через AI її власні репліки",
            "Реакції на ігри, привітання, турботу. Коли й як часто говорити, як і раніше, вирішують "
            "правила й ліміти зі сторінки «Поведінка».", ai.use_for_reactions)
        self.stream_check = self._check(
            card, "Показувати відповідь у чаті поступово",
            "Текст з'являється під час написання, а не весь наприкінці.", ai.stream)
        self.remember_check = self._check(
            card, "Запам'ятовувати, що ти розповідаєш у чаті",
            "Наприклад, як звати твого кота чи яку музику ти любиш. Номери карток, паролі, телефони "
            "й пошта не запам'ятовуються ніколи. Переглянути й видалити спогади можна в розділі "
            "«Пам'ять».", ai.remember_facts)
        return card

    def _build_limits_card(self, ai: AISettings) -> QWidget:
        card = Card()
        card.body.addWidget(SectionHeader("Ліміти"))
        card.body.addWidget(InfoLabel(
            "Коли ліміт вичерпано, вона до кінця години чи дня говорить готовими репліками."))
        self.per_hour_spin = self._spin(1, 2000, ai.max_requests_per_hour)
        card.body.addWidget(setting_row("Запитів на годину, не більше", self.per_hour_spin))
        self.per_day_spin = self._spin(1, 20000, ai.max_requests_per_day)
        card.body.addWidget(setting_row("Запитів на день, не більше", self.per_day_spin))
        self.cost_spin = self._money(0.01, 100.0, ai.max_cost_per_day_usd, decimals=2)
        card.body.addWidget(setting_row(
            "Витрати на день, не більше", self.cost_spin,
            "Для провайдерів із відомою ціною: Claude, а також OpenAI-сумісних, якщо ціну вказано."))
        self.timeout_spin = self._spin(5, 180, round(ai.request_timeout_seconds), " с")
        card.body.addWidget(setting_row(
            "Чекати на відповідь", self.timeout_spin,
            "Якщо модель не встигає, вона відповідає готовою реплікою."))
        self.context_spin = self._spin(2, 60, ai.context_turns)
        card.body.addWidget(setting_row(
            "Реплік розмови в запиті", self.context_spin,
            "Скільки останніх повідомлень чату модель бачить, щоб пам'ятати, про що ви говорите."))
        self.memory_items_spin = self._spin(0, 20, ai.memory_items)
        card.body.addWidget(setting_row(
            "Спогадів у запиті", self.memory_items_spin,
            "Скільки доречних спогадів із пам'яті додавати до запиту. 0 — не додавати."))
        return card

    @staticmethod
    def _build_later_card() -> QWidget:
        """Pointers to the voice and vision pages (they need the same provider)."""
        card = Card()
        card.body.addWidget(SectionHeader("Також"))
        for title, hint, where in (
            ("Голос", "Озвучення її реплік і розмова голосом (розпізнавання працює на цьому ПК).",
             "розділ «Голос»"),
            ("Бачення екрана", "За окремим дозволом вона бачить вікна дозволених ігор і реагує на події. "
             "Аналізує ця ж модель, якщо вона вміє працювати із зображеннями.", "розділ «Бачення екрана»"),
        ):
            row = QHBoxLayout()
            row.addWidget(make_label(title, "value", wrap=False))
            row.addStretch(1)
            row.addWidget(make_label(where, "pill", wrap=False))
            card.body.addLayout(row)
            card.body.addWidget(InfoLabel(hint))
        return card

    # --- small builders --------------------------------------------------------------------------

    @staticmethod
    def _stack(container: QWidget, *widgets: QWidget) -> None:
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for widget in widgets:
            layout.addWidget(widget)

    @staticmethod
    def _check(card: Card, text: str, hint: str, checked: bool) -> QCheckBox:
        row = CheckRow(text, hint)
        row.setChecked(checked)
        card.body.addWidget(row)
        return row.checkbox

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int, suffix: str = "") -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(int(value))
        spin.setSuffix(suffix)
        spin.setFixedWidth(124)
        spin.setAccelerated(True)
        return spin

    @staticmethod
    def _money(minimum: float, maximum: float, value: float, *, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(0.1 if decimals <= 2 else 0.05)
        spin.setPrefix("$")
        spin.setValue(float(value))
        spin.setFixedWidth(124)
        spin.setAccelerated(True)
        return spin

    # --- fields --------------------------------------------------------------------------------

    def _bind_fields(self) -> tuple[_Field, ...]:
        def check(name: str, box: QCheckBox) -> _Field:
            return _Field(name, box.isChecked, lambda v: box.setChecked(bool(v)))

        def number(name: str, spin: QSpinBox) -> _Field:
            return _Field(name, spin.value, lambda v: spin.setValue(int(v)))

        def money(name: str, spin: QDoubleSpinBox) -> _Field:
            return _Field(name, spin.value, lambda v: spin.setValue(float(v)))

        def url(name: str, edit: QLineEdit) -> _Field:
            return _Field(name, lambda: normalize_url(edit.text()), lambda v: edit.setText(str(v)))

        def combo(name: str, box: QComboBox, default: str) -> _Field:
            return _Field(name, lambda: str(box.currentData() or default),
                          lambda v: box.setCurrentIndex(max(0, box.findData(v))))

        return (
            combo("provider", self.provider_combo, "none"),
            _Field("model", self.model_text, lambda v: self._set_model_text(str(v))),
            url("ollama_url", self.ollama_url_edit),
            url("openai_base_url", self.openai_url_edit),
            money("openai_price_input_per_mtok", self.price_input_spin),
            money("openai_price_output_per_mtok", self.price_output_spin),
            combo("effort", self.effort_combo, "low"),
            check("use_for_chat", self.use_chat_check),
            check("use_for_reactions", self.use_reactions_check),
            check("stream", self.stream_check),
            check("remember_facts", self.remember_check),
            number("max_requests_per_hour", self.per_hour_spin),
            number("max_requests_per_day", self.per_day_spin),
            money("max_cost_per_day_usd", self.cost_spin),
            _Field("request_timeout_seconds", lambda: float(self.timeout_spin.value()),
                   lambda v: self.timeout_spin.setValue(round(float(v)))),
            number("context_turns", self.context_spin),
            number("memory_items", self.memory_items_spin),
        )

    def _read_fields(self) -> dict[str, object]:
        return {field.name: field.read() for field in self._fields}

    def _read_settings(self) -> Settings | None:
        try:
            return self._facade.settings().copy()
        except Exception:
            log.exception("reading the current settings failed")
            return None

    # --- provider and model ------------------------------------------------------------------

    def _set_model_text(self, text: str) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.setEditText(text)
        self.model_combo.blockSignals(False)
        self._update_model_warning()

    def _on_model_edited(self, _text: str) -> None:
        self._models[self.provider()] = self.model_text()
        self._update_model_warning()

    def _on_provider_changed(self, _index: int = -1) -> None:
        provider = self.provider()
        changed = provider != self._shown_provider
        if changed:
            if not self._loading:
                self._models[self._shown_provider] = self.model_text()
            self._shown_provider = provider
            # A key typed for one provider must never be saved as another's.
            self.key_edit.clear()
            self.key_message_label.hide()
            # An answer for the old provider would fill the wrong list.
            if self._models_wait.busy:
                self._models_wait.give_up()
                self._end_models_wait("")
            self.test_result_label.hide()
        self._fill_model_choices(provider)
        if changed and not self._loading:
            # Model ids don't carry over between providers: restore what was typed
            # for this provider earlier (or its default) instead.
            self._set_model_text(self._models.get(provider, ""))
        self.model_combo.lineEdit().setPlaceholderText(MODEL_PLACEHOLDERS.get(provider, ""))
        self.provider_note.setText(PROVIDER_NOTES.get(provider, "").format(name=self._name))
        self.connection_box.setVisible(provider != "none")
        self.ollama_url_row.setVisible(provider == "ollama")
        self.openai_url_row.setVisible(provider == "openai")
        self.price_row.setVisible(provider == "openai")
        self.effort_row.setVisible(provider == "anthropic")
        self.key_row.setVisible(provider in KEY_PROVIDERS)
        self._refresh_key_state()
        self._refresh_cloud()
        self._update_model_warning()

    def _fill_model_choices(self, provider: str) -> None:
        choices = list(self._listed.get(provider, []))
        if provider == "anthropic":
            choices = list(dict.fromkeys([*CLAUDE_MODELS, *choices]))
        text = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(choices)
        self.model_combo.setEditText(text)
        self.model_combo.blockSignals(False)

    def _update_model_warning(self) -> None:
        missing = self.provider() == "openai" and not self.model_text()
        self.model_warning.setText(
            "Вкажи модель: без неї OpenAI-сумісний сервер не працюватиме. Список доступних моделей "
            "покаже кнопка «Оновити список»." if missing else "")
        self.model_warning.setVisible(missing)

    def _on_url_changed(self, _text: str = "") -> None:
        # Whether the data leaves the PC (and whether a key is needed) follows the address.
        self._refresh_cloud()
        self._refresh_key_state()
        for edit, error in ((self.ollama_url_edit, self.ollama_url_error),
                            (self.openai_url_edit, self.openai_url_error)):
            if not error.isHidden() and is_valid_url(edit.text()):
                error.hide()

    # --- cloud permission ------------------------------------------------------------------------

    def _selected_is_cloud(self) -> bool:
        return is_cloud_choice(self.provider(), self.ollama_url_edit.text(), self.openai_url_edit.text())

    def _refresh_cloud(self) -> None:
        cloud = self._selected_is_cloud()
        try:
            granted = self._facade.permissions.is_granted(Permission.CLOUD_AI)
        except Exception:
            log.exception("reading the cloud AI permission failed")
            granted = False
        where = {
            "anthropic": "Anthropic Claude працює через інтернет",
            "ollama": "Ollama за цією адресою працює не на цьому комп'ютері",
        }.get(self.provider(), "Сервер за цією адресою — не на цьому комп'ютері")
        self.cloud_banner_label.setText(
            f"{where}: усе, що ти пишеш у чаті, надсилатиметься провайдерові. Поки ти цього не "
            f"дозволиш, {self._name} говоритиме лише готовими репліками.")
        self.cloud_banner.setVisible(cloud and not granted)
        self.cloud_granted_note.setVisible(cloud and granted)

    def _allow_cloud(self) -> None:
        if not self._confirm(CLOUD_AI_CONSENT_TITLE, CLOUD_AI_CONSENT_TEXT):
            return
        try:
            self._facade.permissions.grant(Permission.CLOUD_AI)
        except (PermissionError, ValueError) as exc:  # unavailable in this build
            log.warning("could not grant %s: %s", Permission.CLOUD_AI.value, exc)
            self.cloud_banner_label.setText("Цей дозвіл зараз не можна надати.")
            return
        self.permissions_changed.emit()
        self.refresh()

    # --- API key --------------------------------------------------------------------------------

    def _key_provider(self) -> str | None:
        provider = self.provider()
        return provider if provider in KEY_PROVIDERS else None

    def _refresh_key_state(self) -> None:
        provider = self._key_provider()
        if provider is None:
            return
        try:
            has_key = bool(self._facade.ai_has_key(provider))
        except Exception:
            log.exception("asking whether a %s key is stored failed", provider)
            has_key = False
        if has_key:
            text, role = "🔑 Ключ збережено", "success"
        elif provider == "openai" and is_loopback_url(self.openai_url_edit.text()):
            text, role = "Ключа немає — локальному серверу він зазвичай не потрібен", "hint"
        else:
            text, role = "Ключ ще не збережено", "warning"
        self.key_state_label.setText(text)
        set_style_property(self.key_state_label, "role", role)
        self.delete_key_button.setEnabled(has_key)
        self.key_edit.setPlaceholderText(
            "Новий ключ замінить збережений" if has_key
            else ("Встав ключ sk-ant-…" if provider == "anthropic" else "Встав ключ API"))

    def _on_key_typed(self, text: str) -> None:
        self.save_key_button.setEnabled(bool(text.strip()))

    def _key_message(self, text: str, *, error: bool) -> None:
        self.key_message_label.setText(text)
        set_style_property(self.key_message_label, "role", "error" if error else "success")
        self.key_message_label.setVisible(bool(text))

    def save_pending_key(self) -> bool:
        """Store a key typed but not saved yet ("Гаразд" should not lose it).
        Returns False if there is one and it could not be stored."""
        if self._key_provider() is None or not self.key_edit.text().strip():
            return True
        return self._save_key()

    def _save_key(self) -> bool:
        provider = self._key_provider()
        key = self.key_edit.text().strip()
        if provider is None or not key:
            return False
        if not _KEY_RE.fullmatch(key):
            self._key_message("Це не схоже на ключ API: він має бути одним рядком без пробілів.", error=True)
            return False
        try:
            self._facade.ai_set_key(provider, key)
        except Exception as exc:
            # The type only: an exception message could quote the key.
            log.error("storing the %s key failed: %s", provider, type(exc).__name__)
            self._key_message("Не вдалося зберегти ключ.", error=True)
            return False
        self.key_edit.clear()
        self._key_message("Ключ збережено ✓", error=False)
        self._refresh_key_state()
        self.refresh_status()
        return True

    def _delete_key(self) -> None:
        provider = self._key_provider()
        if provider is None:
            return
        if not self._confirm(
                "Видалити ключ?",
                f"Ключ {ai_provider_name(provider)} буде видалено з цього комп'ютера. Без нього "
                f"{self._name} говоритиме готовими репліками, доки ти не додаси новий."):
            return
        try:
            self._facade.ai_set_key(provider, "")
        except Exception as exc:
            log.error("deleting the %s key failed: %s", provider, type(exc).__name__)
            self._key_message("Не вдалося видалити ключ.", error=True)
            return
        self._key_message("Ключ видалено.", error=False)
        self._refresh_key_state()
        self.refresh_status()

    # --- asynchronous checks -----------------------------------------------------------------------

    def _apply_own_edits(self) -> bool:
        """Apply this page's pending edits (and a typed key), so a check tests what the page shows."""
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
            log.exception("applying the AI settings failed")
            return False
        applied = self._read_settings()
        if applied is not None:
            self.sync(applied, edited_too=True)
        else:
            self.reset_baseline()
        self.settings_applied.emit()
        return True

    def _test_connection(self) -> None:
        if self._test_wait.busy:
            return
        if not self._apply_own_edits():
            self._show_test_result(False, "Спершу виправ налаштування вище — їх не вдалося зберегти.")
            return
        seq = self._test_wait.start()
        self.test_button.setEnabled(False)
        self.test_button.setText("Перевіряю…")
        self.test_result_label.hide()

        def done(ok: bool, message: str) -> None:
            if is_alive(self) and self._test_wait.finish(seq):
                self._end_test_wait()
                self._show_test_result(bool(ok), str(message))
                self.refresh_status()

        try:
            self._facade.ai_test(done)
        except Exception as exc:
            log.error("starting the connection test failed: %s", type(exc).__name__)
            if self._test_wait.finish(seq):
                self._end_test_wait()
                self._show_test_result(False, "Не вдалося запустити перевірку.")

    def _end_test_wait(self) -> None:
        self.test_button.setEnabled(True)
        self.test_button.setText("Перевірити з'єднання")

    def _on_test_timeout(self) -> None:
        self._test_wait.give_up()
        self._end_test_wait()
        self._show_test_result(False, "Відповіді так і не було — спробуй ще раз.")

    def _show_test_result(self, ok: bool, message: str) -> None:
        mark = "✓" if ok else "✗"
        self.test_result_label.setText(f"{mark} {message}" if message else mark)
        set_style_property(self.test_result_label, "role", "success" if ok else "error")
        self.test_result_label.show()

    def _list_models(self) -> None:
        if self._models_wait.busy:
            return
        if not self._apply_own_edits():
            self._end_models_wait("Спершу виправ налаштування вище — їх не вдалося зберегти.", error=True)
            return
        provider = self.provider()
        seq = self._models_wait.start()
        self.refresh_models_button.setEnabled(False)
        self.refresh_models_button.setText("Шукаю…")
        self.models_status_label.hide()

        def done(models: list[str], error: str) -> None:
            if not (is_alive(self) and self._models_wait.finish(seq)):
                return
            if error:
                self._end_models_wait(str(error), error=True)
                return
            names = list(dict.fromkeys(str(m).strip() for m in models if str(m).strip()))
            self._listed[provider] = names
            if self.provider() == provider:
                self._fill_model_choices(provider)
            if names:
                self._end_models_wait(f"Знайдено {count_uk(len(names), 'модель', 'моделі', 'моделей')}. "
                                      "Вибери зі списку або впиши назву.")
            else:
                self._end_models_wait("Сервер не повідомив жодної моделі.", error=True)

        try:
            self._facade.ai_list_models(done)
        except Exception as exc:
            log.error("listing models failed: %s", type(exc).__name__)
            if self._models_wait.finish(seq):
                self._end_models_wait("Не вдалося отримати список моделей.", error=True)

    def _end_models_wait(self, message: str, *, error: bool = False) -> None:
        self.refresh_models_button.setEnabled(True)
        self.refresh_models_button.setText("Оновити список")
        self.models_status_label.setText(message)
        set_style_property(self.models_status_label, "role", "error" if error else "hint")
        self.models_status_label.setVisible(bool(message))

    def _on_models_timeout(self) -> None:
        self._models_wait.give_up()
        self._end_models_wait("Відповіді так і не було — спробуй ще раз.", error=True)

    # --- visibility-driven polling -----------------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.refresh()
        self.status_timer.start()

    def hideEvent(self, event: QHideEvent) -> None:
        self.status_timer.stop()
        super().hideEvent(event)

