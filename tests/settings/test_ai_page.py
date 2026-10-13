from __future__ import annotations

import logging
from dataclasses import replace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLineEdit

from chibi.core.config import AI_PROVIDERS, AISettings, Settings
from chibi.core.events import Permission
from chibi.settings.ai_page import (
    CLAUDE_MODELS,
    CLOUD_AI_CONSENT_TEXT,
    CLOUD_AI_CONSENT_TITLE,
    EFFORT_CHOICES,
    PROVIDER_CHOICES,
    AIPage,
    describe_status,
    is_cloud_choice,
    is_valid_url,
)
from chibi.settings.facade import AIStatusView


class Confirm:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.asked: list[tuple[str, str]] = []

    def __call__(self, title: str, text: str) -> bool:
        self.asked.append((title, text))
        return self.answer


@pytest.fixture
def make_page(facade, track):
    def make(confirm: Confirm | None = None, *, provider: str | None = None, **ai) -> AIPage:
        if provider is not None or ai:
            s = facade.settings()
            if provider is not None:
                s.ai.provider = provider
            for name, value in ai.items():
                setattr(s.ai, name, value)
            facade.apply_settings(s)
            facade.applied.clear()
        page = track(AIPage(facade, confirm or Confirm(False)))
        page.show()
        return page

    return make


def choose(page: AIPage, provider: str) -> None:
    page.provider_combo.setCurrentIndex(page.provider_combo.findData(provider))


def shown(page: AIPage, widget) -> bool:
    return widget.isVisibleTo(page)


STATUS = AIStatusView(provider="anthropic", model="claude-opus-5-5", online=True, reason="", cloud=True,
                      requests_today=12, requests_limit_day=400, cost_today_usd=0.034, cost_limit_usd=1.0,
                      cost_known=True)


# --- provider and fields ------------------------------------------------------------------------


def test_provider_choices_cover_every_provider_in_ukrainian(make_page):
    page = make_page()
    assert tuple(value for value, _ in PROVIDER_CHOICES) == AI_PROVIDERS
    texts = [page.provider_combo.itemText(i) for i in range(page.provider_combo.count())]
    assert texts == ["Вимкнено", "Ollama — локально на цьому ПК", "Anthropic Claude",
                     "OpenAI-сумісний сервер"]
    assert page.provider() == "none"
    assert [text for _, text in EFFORT_CHOICES] == [
        "Швидко й дешево (low)", "Збалансовано (medium)", "Ретельно (high)"]


def test_ai_off_hides_connection_details(make_page):
    page = make_page()
    assert not shown(page, page.connection_box)
    assert "AI вимкнено" in page.provider_note.text() and "Мія" in page.provider_note.text()
    assert not shown(page, page.cloud_banner)


@pytest.mark.parametrize(
    ("provider", "visible", "hidden"),
    [
        ("ollama", {"ollama_url_row"}, {"openai_url_row", "key_row", "price_row", "effort_row"}),
        ("anthropic", {"key_row", "effort_row"}, {"ollama_url_row", "openai_url_row", "price_row"}),
        ("openai", {"openai_url_row", "key_row", "price_row"}, {"ollama_url_row", "effort_row"}),
    ],
)
def test_switching_provider_shows_the_right_fields(make_page, provider, visible, hidden):
    page = make_page()
    choose(page, provider)
    assert shown(page, page.connection_box)
    assert shown(page, page.model_combo) and shown(page, page.refresh_models_button)
    assert shown(page, page.test_button)
    for name in visible:
        assert shown(page, getattr(page, name)), name
    for name in hidden:
        assert not shown(page, getattr(page, name)), name


def test_effort_note_states_prices_and_leaves_the_choice_to_the_user(make_page):
    page = make_page(provider="anthropic")
    note = page.effort_note.text()
    for part in ("claude-opus-5-5", "$4", "$20", "claude-haiku-5-5", "$0.10", "$0.50", "Вибір за тобою"):
        assert part in note
    assert page.effort_combo.currentData() == "low"


def test_claude_models_are_suggested_and_the_box_stays_editable(make_page):
    page = make_page(provider="anthropic")
    items = [page.model_combo.itemText(i) for i in range(page.model_combo.count())]
    assert items[:len(CLAUDE_MODELS)] == list(CLAUDE_MODELS)
    assert page.model_combo.isEditable()
    assert page.model_text() == ""  # empty = the provider default
    assert "claude-opus-5-5" in page.model_combo.lineEdit().placeholderText()
    page.model_combo.setEditText("claude-some-future-model")
    assert page.model_text() == "claude-some-future-model"


def test_typed_model_follows_its_provider(make_page):
    page = make_page(provider="anthropic", model="claude-haiku-5-5")
    assert page.model_text() == "claude-haiku-5-5"
    choose(page, "ollama")
    assert page.model_text() == ""  # a Claude id makes no sense for Ollama
    page.model_combo.setEditText("qwen3:8b")
    choose(page, "anthropic")
    assert page.model_text() == "claude-haiku-5-5"
    choose(page, "ollama")
    assert page.model_text() == "qwen3:8b"


def test_openai_without_model_warns(make_page):
    page = make_page()
    choose(page, "openai")
    assert shown(page, page.model_warning)
    assert "Оновити список" in page.model_warning.text()
    page.model_combo.setEditText("gpt-test")
    assert not shown(page, page.model_warning)


# --- collecting into settings -----------------------------------------------------------------------


def test_budgets_and_toggles_are_collected_into_settings(make_page, facade):
    page = make_page()
    choose(page, "openai")
    page.model_combo.setEditText("  my-model  ")
    page.openai_url_edit.setText(" http://localhost:1234/v1/ ")
    page.price_input_spin.setValue(0.15)
    page.price_output_spin.setValue(0.6)
    page.use_chat_check.setChecked(False)
    page.use_reactions_check.setChecked(False)
    page.stream_check.setChecked(False)
    page.remember_check.setChecked(False)
    page.per_hour_spin.setValue(30)
    page.per_day_spin.setValue(150)
    page.cost_spin.setValue(0.5)
    page.timeout_spin.setValue(45)
    page.context_spin.setValue(20)
    page.memory_items_spin.setValue(3)

    s = page.collect(facade.settings())
    ai = s.ai
    assert (ai.provider, ai.model, ai.openai_base_url) == ("openai", "my-model", "http://localhost:1234/v1")
    assert (ai.openai_price_input_per_mtok, ai.openai_price_output_per_mtok) == (0.15, 0.6)
    assert (ai.use_for_chat, ai.use_for_reactions, ai.stream, ai.remember_facts) == (False,) * 4
    assert (ai.max_requests_per_hour, ai.max_requests_per_day, ai.max_cost_per_day_usd) == (30, 150, 0.5)
    assert ai.request_timeout_seconds == 45.0 and isinstance(ai.request_timeout_seconds, float)
    assert (ai.context_turns, ai.memory_items) == (20, 3)
    # what collect produces survives the settings' own validation unchanged
    assert Settings.from_dict(s.to_dict()).ai == ai


def test_collect_writes_only_fields_edited_here(make_page, facade):
    page = make_page()
    page.per_day_spin.setValue(99)
    # changed elsewhere after the page was loaded
    s = facade.settings()
    s.ai.max_requests_per_hour = 7
    s.ai.effort = "high"
    facade.apply_settings(s)
    collected = page.collect(facade.settings()).ai
    assert collected.max_requests_per_day == 99
    assert (collected.max_requests_per_hour, collected.effort) == (7, "high")  # not reverted


def test_sync_shows_changes_made_elsewhere_but_keeps_unsaved_edits(make_page, facade):
    page = make_page()
    page.per_day_spin.setValue(99)
    s = facade.settings()
    s.ai.max_requests_per_hour = 7
    s.ai.max_requests_per_day = 10
    page.sync(s, edited_too=False)
    assert page.per_hour_spin.value() == 7
    assert page.per_day_spin.value() == 99
    page.sync(s, edited_too=True)
    assert page.per_day_spin.value() == 10
    assert not page.has_edits()


def test_invalid_url_is_reported_and_never_collected(make_page, facade):
    page = make_page()
    choose(page, "ollama")
    page.ollama_url_edit.setText("localhost:11434")  # no scheme
    assert page.validate() is False
    assert shown(page, page.ollama_url_error)
    assert "http://" in page.ollama_url_error.text()
    assert page.collect(facade.settings()).ai.ollama_url == AISettings().ollama_url
    page.ollama_url_edit.setText("http://192.168.1.5:11434")
    assert not shown(page, page.ollama_url_error)  # fixed: error goes away while typing
    assert page.validate() is True


@pytest.mark.parametrize(("url", "ok"), [
    ("http://localhost:11434", True), ("https://api.openai.com/v1/", True), ("ftp://x", False),
    ("http://", False), ("http://a b", False), ("", False),
])
def test_is_valid_url(url, ok):
    assert is_valid_url(url) is ok


# --- API key ----------------------------------------------------------------------------------------


def test_saving_a_key_stores_it_and_clears_the_field(make_page, facade):
    page = make_page(provider="anthropic")
    assert page.key_edit.echoMode() == QLineEdit.EchoMode.Password
    assert page.key_edit.text() == ""
    assert not page.save_key_button.isEnabled()
    assert page.key_state_label.text() == "Ключ ще не збережено"
    assert not page.delete_key_button.isEnabled()
    page.key_edit.setText("  sk-ant-test-1234567890  ")
    page.save_key_button.click()
    assert facade.keys == {"anthropic": "sk-ant-test-1234567890"}
    assert page.key_edit.text() == ""
    assert "Ключ збережено" in page.key_state_label.text()
    assert page.delete_key_button.isEnabled()
    assert "замінить" in page.key_edit.placeholderText()


def test_stored_key_is_never_shown(make_page, facade):
    facade.keys["anthropic"] = "sk-ant-secret-value"
    page = make_page(provider="anthropic")
    assert page.key_edit.text() == ""
    assert "Ключ збережено" in page.key_state_label.text()
    assert all("sk-ant-secret" not in edit.text() for edit in page.findChildren(QLineEdit))


def test_enter_in_the_key_field_saves(make_page, facade):
    page = make_page(provider="openai")
    page.key_edit.setText("sk-proj-abcdefgh")
    QTest.keyClick(page.key_edit, Qt.Key.Key_Return)
    assert facade.keys == {"openai": "sk-proj-abcdefgh"}


def test_key_with_spaces_is_refused(make_page, facade):
    page = make_page(provider="anthropic")
    page.key_edit.setText("sk-ant two words")
    page.save_key_button.click()
    assert facade.key_calls == []
    assert "без пробілів" in page.key_message_label.text()
    assert page.key_edit.text() == "sk-ant two words"  # kept so it can be fixed


def test_key_store_failure_is_reported_without_logging_the_key(make_page, facade, caplog):
    secret = "sk-ant-very-secret-123"
    facade.key_error = OSError(f"could not store {secret}")
    page = make_page(provider="anthropic")
    page.key_edit.setText(secret)
    with caplog.at_level(logging.DEBUG):
        page.save_key_button.click()
    assert "Не вдалося" in page.key_message_label.text()
    assert all(secret not in r.getMessage() for r in caplog.records)
    assert all(r.exc_info is None for r in caplog.records)


def test_deleting_a_key_asks_first(make_page, facade):
    facade.keys["anthropic"] = "sk-ant-x"
    declined = make_page(Confirm(False), provider="anthropic")
    declined.delete_key_button.click()
    assert facade.keys == {"anthropic": "sk-ant-x"}
    confirm = Confirm(True)
    page = make_page(confirm, provider="anthropic")
    page.delete_key_button.click()
    assert facade.keys == {}
    assert facade.key_calls[-1] == ("anthropic", 0)
    assert "Видалити ключ" in confirm.asked[0][0]
    assert page.key_state_label.text() == "Ключ ще не збережено"


def test_local_openai_server_needs_no_key(make_page):
    page = make_page(provider="openai", openai_base_url="http://localhost:1234/v1")
    assert "не потрібен" in page.key_state_label.text()


# --- list models ------------------------------------------------------------------------------------


def test_list_models_fills_the_combo_and_keeps_the_typed_model(make_page, facade):
    facade.deferred = True
    page = make_page(provider="ollama", model="llama3.2")
    page.refresh_models_button.click()
    assert not page.refresh_models_button.isEnabled()
    assert page.refresh_models_button.text() == "Шукаю…"
    page.refresh_models_button.click()  # disabled while waiting
    assert facade.models_calls == 1
    facade.resolve("models")
    assert page.refresh_models_button.isEnabled()
    items = [page.model_combo.itemText(i) for i in range(page.model_combo.count())]
    assert items == ["llama3.2", "qwen3:8b"]
    assert page.model_text() == "llama3.2"
    assert "Знайдено 2 моделі" in page.models_status_label.text()


def test_list_models_error_is_shown(make_page, facade):
    facade.models_result = ([], "Ollama не запущено на цьому ПК.")
    page = make_page(provider="ollama")
    page.refresh_models_button.click()
    assert page.models_status_label.text() == "Ollama не запущено на цьому ПК."
    assert page.models_status_label.property("role") == "error"
    assert page.refresh_models_button.isEnabled()


def test_list_models_applies_pending_page_edits_first(make_page, facade):
    applied = []
    page = make_page()
    page.settings_applied.connect(lambda: applied.append(True))
    choose(page, "ollama")
    page.ollama_url_edit.setText("http://127.0.0.1:11434")
    page.refresh_models_button.click()
    assert facade.settings_at_check[0].ai.provider == "ollama"
    assert facade.settings_at_check[0].ai.ollama_url == "http://127.0.0.1:11434"
    assert applied == [True]
    assert not page.has_edits()  # the baseline moved with the apply


def test_check_refuses_to_apply_an_invalid_url(make_page, facade):
    page = make_page()
    choose(page, "openai")
    page.openai_url_edit.setText("not a url")
    page.test_button.click()
    assert facade.test_calls == 0 and facade.applied == []
    assert "✗" in page.test_result_label.text()


def test_switching_provider_drops_a_pending_model_list(make_page, facade):
    facade.deferred = True
    page = make_page(provider="ollama")
    page.refresh_models_button.click()
    choose(page, "anthropic")
    assert page.refresh_models_button.isEnabled()
    facade.resolve("models")  # the Ollama answer arrives late
    items = [page.model_combo.itemText(i) for i in range(page.model_combo.count())]
    assert items == list(CLAUDE_MODELS)


# --- test connection -----------------------------------------------------------------------------


def test_test_connection_shows_the_result(make_page, facade):
    facade.deferred = True
    page = make_page(provider="anthropic")
    page.test_button.click()
    assert not page.test_button.isEnabled()
    assert page.test_button.text() == "Перевіряю…"
    facade.resolve("test")
    assert page.test_button.isEnabled()
    assert page.test_result_label.text().startswith("✓ Працює!")
    assert page.test_result_label.property("role") == "success"


def test_test_connection_failure(make_page, facade):
    facade.test_result = (False, "немає API-ключа")
    page = make_page(provider="anthropic")
    page.test_button.click()
    assert page.test_result_label.text() == "✗ немає API-ключа"
    assert page.test_result_label.property("role") == "error"


def test_test_connection_raising_reenables_the_button(make_page, facade):
    def broken(on_done):
        raise RuntimeError("boom")

    facade.ai_test = broken
    page = make_page(provider="ollama")
    page.test_button.click()
    assert page.test_button.isEnabled()
    assert "Не вдалося" in page.test_result_label.text()


def test_watchdog_reenables_a_check_that_never_answers(make_page, facade):
    facade.deferred = True
    page = make_page(provider="ollama")
    page.test_button.click()
    page._test_wait.timer.timeout.emit()
    assert page.test_button.isEnabled()
    assert "спробуй ще раз" in page.test_result_label.text()
    facade.resolve("test")  # too late: ignored
    assert "спробуй ще раз" in page.test_result_label.text()


def test_late_answers_after_the_page_is_deleted_do_not_crash(facade, qapp):
    facade.deferred = True
    s = facade.settings()
    s.ai.provider = "ollama"
    facade.apply_settings(s)
    page = AIPage(facade, Confirm(False))
    page.test_button.click()
    page.refresh_models_button.click()
    page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    facade.resolve("test")
    facade.resolve("models")


# --- cloud permission ----------------------------------------------------------------------------


def test_cloud_provider_without_permission_shows_a_warning(make_page, permissions):
    page = make_page()
    choose(page, "anthropic")
    assert shown(page, page.cloud_banner)
    assert "надсилатиметься" in page.cloud_banner_label.text()
    assert not permissions.is_granted(Permission.CLOUD_AI)


def test_allow_grants_cloud_ai_only_when_confirmed(make_page, permissions):
    declined = Confirm(False)
    page = make_page(declined, provider="anthropic")
    page.allow_cloud_button.click()
    assert declined.asked == [(CLOUD_AI_CONSENT_TITLE, CLOUD_AI_CONSENT_TEXT)]
    assert not permissions.is_granted(Permission.CLOUD_AI)
    assert shown(page, page.cloud_banner)

    granted = []
    confirm = Confirm(True)
    page = make_page(confirm, provider="anthropic")
    page.permissions_changed.connect(lambda: granted.append(True))
    page.allow_cloud_button.click()
    assert permissions.is_granted(Permission.CLOUD_AI)
    assert granted == [True]
    assert not shown(page, page.cloud_banner)
    assert shown(page, page.cloud_granted_note)


def test_consent_text_says_what_is_sent_and_what_never_is():
    text = CLOUD_AI_CONSENT_TEXT
    for part in ("пишеш у чаті", "настрій і потреби", "назву активної програми", "якщо дозволено бачити",
                 "спогади", "Знімки екрана", "ніколи", "політикою конфіденційності", "одразу"):
        assert part in text, part


@pytest.mark.parametrize(
    ("provider", "url_field", "url", "banner"),
    [
        ("ollama", "ollama_url_edit", "http://localhost:11434", False),
        ("ollama", "ollama_url_edit", "http://127.0.0.1:11434", False),
        ("ollama", "ollama_url_edit", "http://192.168.1.20:11434", True),
        ("openai", "openai_url_edit", "http://localhost:1234/v1", False),
        ("openai", "openai_url_edit", "https://api.openai.com/v1", True),
    ],
)
def test_cloud_warning_follows_the_address(make_page, provider, url_field, url, banner):
    page = make_page()
    choose(page, provider)
    getattr(page, url_field).setText(url)
    assert shown(page, page.cloud_banner) is banner


def test_revoking_elsewhere_brings_the_warning_back(make_page, permissions):
    permissions.grant(Permission.CLOUD_AI)
    page = make_page(provider="anthropic")
    assert not shown(page, page.cloud_banner)
    permissions.revoke(Permission.CLOUD_AI)
    page.refresh()
    assert shown(page, page.cloud_banner)


@pytest.mark.parametrize(
    ("provider", "ollama_url", "openai_url"),
    [
        ("none", "http://localhost:11434", "https://api.openai.com/v1"),
        ("anthropic", "http://localhost:11434", "https://api.openai.com/v1"),
        ("ollama", "http://localhost:11434", "https://api.openai.com/v1"),
        ("ollama", "http://[::1]:11434", "https://api.openai.com/v1"),
        ("ollama", "http://gpu-box.local:11434", "https://api.openai.com/v1"),
        ("openai", "http://localhost:11434", "http://127.0.0.2:8080/v1"),
        ("openai", "http://localhost:11434", "http://my.localhost/v1"),
        ("openai", "http://localhost:11434", "https://api.openai.com/v1"),
        ("openai", "http://localhost:11434", "http://[::ffff:127.0.0.1]:1234/v1"),
    ],
)
def test_cloud_rule_matches_the_ai_factory(provider, ollama_url, openai_url):
    from chibi.ai.providers.factory import is_cloud

    ai = AISettings(provider=provider, ollama_url=ollama_url, openai_base_url=openai_url)
    assert is_cloud_choice(provider, ollama_url, openai_url) is is_cloud(ai)


# --- status line ---------------------------------------------------------------------------------


def test_status_line_online_with_usage(make_page, facade):
    facade.ai_status_view = STATUS
    page = make_page(provider="anthropic")
    assert page.status_label.text().startswith("● Працює: Anthropic Claude · claude-opus-5-5")
    assert "хмара" in page.status_label.text()
    assert page.status_label.property("role") == "success"
    assert page.usage_label.text() == "Запитів сьогодні: 12 із 400 · витрачено $0.03 із $1.00"


def test_status_line_offline_reason(make_page, facade):
    facade.ai_status_view = replace(STATUS, online=False, reason="потрібен дозвіл на хмарний AI")
    page = make_page(provider="anthropic")
    assert page.status_label.text() == "● Офлайн: потрібен дозвіл на хмарний AI"
    assert page.status_label.property("role") == "warning"


def test_describe_status_variants():
    assert "вартість невідома" in describe_status(replace(STATUS, provider="openai", cost_known=False)).usage
    local = describe_status(replace(STATUS, provider="ollama", model="qwen3:8b", cloud=False))
    assert "локально" in local.status and "без оплати" in local.usage
    off = describe_status(replace(STATUS, provider="none", online=False, reason="AI вимкнено",
                                  requests_today=0))
    assert off.usage == "" and "вимкнено" in off.status
    unknown = describe_status(replace(STATUS, online=False, reason=""))
    assert unknown.status == "● Офлайн: AI зараз недоступний"


def test_status_is_polled_only_while_visible(make_page, facade):
    page = make_page(provider="anthropic")
    assert page.status_timer.isActive()
    facade.ai_status_view = STATUS
    page.status_timer.timeout.emit()
    assert "Працює" in page.status_label.text()
    page.hide()
    assert not page.status_timer.isActive()


def test_status_failure_is_logged_once(make_page, facade, caplog):
    facade.ai_status_error = RuntimeError("boom")
    with caplog.at_level(logging.ERROR):
        page = make_page()
        page.refresh_status()
        page.refresh_status()
    assert page.status_label.text() == "● Стан AI зараз невідомий"
    assert len([r for r in caplog.records if "AI status" in r.getMessage()]) == 1


def test_voice_and_vision_are_pointed_to(make_page):
    from PySide6.QtWidgets import QLabel

    page = make_page()
    text = "\n".join(label.text() for label in page.findChildren(QLabel))
    assert "розділ «Голос»" in text and "розділ «Бачення екрана»" in text
    assert "з'явиться у фазі" not in text


def test_switching_provider_clears_a_typed_key(make_page, facade):
    page = make_page(provider="anthropic")
    page.key_edit.setText("sk-ant-typed-but-not-saved")
    choose(page, "openai")
    assert page.key_edit.text() == ""
    page.save_key_button.click()
    assert facade.key_calls == []  # never saved as the OpenAI key


def test_save_pending_key(make_page, facade):
    page = make_page(provider="openai")
    assert page.save_pending_key() is True and facade.key_calls == []  # nothing typed
    page.key_edit.setText("sk-proj-12345678")
    assert page.save_pending_key() is True
    assert facade.keys == {"openai": "sk-proj-12345678"}
    page.key_edit.setText("not a key")
    assert page.save_pending_key() is False


def test_check_saves_a_typed_key_first(make_page, facade):
    page = make_page(provider="anthropic")
    page.key_edit.setText("sk-ant-12345678")
    page.test_button.click()
    assert facade.keys == {"anthropic": "sk-ant-12345678"}
    assert facade.test_calls == 1
    assert page.key_edit.text() == ""
