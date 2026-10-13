"""The "Голос" settings page against a recording fake of its small facade."""

from __future__ import annotations

import logging

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractButton, QApplication, QComboBox, QLabel, QLineEdit, QSlider

from chibi.core.config import Settings
from chibi.core.events import Permission
from chibi.memory.repositories import PermissionRepository
from chibi.privacy import permissions as permissions_module
from chibi.privacy.permissions import PermissionManager
from chibi.settings import voice_page
from chibi.settings.voice_page import (
    MICROPHONE_CONSENT_TEXT,
    MICROPHONE_CONSENT_TITLE,
    PRIVACY_NOTE,
    VoicePage,
    format_seconds,
    format_size,
    model_choice_title,
)
from chibi.voice.hotkey_win import PRESETS
from chibi.voice.integration import STT_OFF, VOICE_MODULE_MISSING, DownloadProgress, VoiceStatus
from chibi.voice.models import CATALOG, DEFAULT_TTS_MODEL

KEY = "azure-secret-key-0123456789"


class FakeVoiceFacade:
    def __init__(self, permissions):
        self.permissions = permissions
        self._settings = Settings()
        self.applied: list[Settings] = []
        self.status = VoiceStatus(tts_ready=False, stt_ready=False, tts_reason="Голос вимкнено.",
                                  stt_reason=STT_OFF, state="off", hotkey="Ctrl + Alt + Пробіл",
                                  hotkey_active=False)
        self.status_error: Exception | None = None
        self.installed = {DEFAULT_TTS_MODEL}
        self.downloads: list[tuple] = []
        self.state: DownloadProgress | None = None
        self.cancels = 0
        self.removed: list[str] = []
        self.key = ""
        self.key_calls: list[int] = []  # key lengths only
        self.key_error: Exception | None = None
        self.tests: list = []
        self.settings_at_test: list[Settings] = []
        self.devices = (["USB Mic", "Webcam Mic"], ["Speakers", "Headset"])
        self.devices_error: Exception | None = None

    def settings(self):
        return self._settings.copy()

    def apply_settings(self, settings):
        self.applied.append(settings.copy())
        self._settings = Settings.from_dict(settings.to_dict())

    def voice_status(self):
        if self.status_error is not None:
            raise self.status_error
        return self.status

    def voice_models(self):
        return [(spec, spec.id in self.installed) for spec in CATALOG.values()]

    def voice_download(self, model_id, on_progress, on_done):
        self.downloads.append((model_id, on_progress, on_done))
        self.state = DownloadProgress(model_id, 0, CATALOG[model_id].size)

    def finish_download(self, ok=True, message="Завантажено ✓"):
        model_id, _progress, on_done = self.downloads[-1]
        self.state = None
        if ok:
            self.installed.add(model_id)
        on_done(ok, message)

    def voice_cancel_download(self):
        self.cancels += 1

    def voice_download_state(self):
        return self.state

    def voice_remove_model(self, model_id):
        self.removed.append(model_id)
        self.installed.discard(model_id)

    def voice_has_azure_key(self):
        return bool(self.key)

    def voice_set_azure_key(self, key):
        if self.key_error is not None:
            raise self.key_error
        self.key_calls.append(len(key))
        self.key = key

    def voice_test(self, on_done):
        self.settings_at_test.append(self._settings.copy())
        self.tests.append(on_done)

    def voice_devices(self):
        if self.devices_error is not None:
            raise self.devices_error
        return self.devices


@pytest.fixture
def permissions(db, bus, clock, monkeypatch):
    monkeypatch.setattr(permissions_module, "AVAILABLE", permissions_module.AVAILABLE | {
        Permission.MICROPHONE, Permission.SCREEN_CAPTURE, Permission.CLOUD_VISION})
    return PermissionManager(PermissionRepository(db), bus, clock)


@pytest.fixture
def vf(permissions):
    return FakeVoiceFacade(permissions)


class Confirm:
    def __init__(self, answer=True):
        self.answer = answer
        self.asked = []

    def __call__(self, title, text):
        self.asked.append((title, text))
        return self.answer


@pytest.fixture
def confirm():
    return Confirm()


@pytest.fixture
def make_page(vf, confirm, track):
    def make() -> VoicePage:
        page = track(VoicePage(vf, confirm))
        page.resize(760, 1600)
        page.show()
        return page

    return make


def labels(page) -> list[str]:
    return [label.text() for label in page.findChildren(QLabel) if label.isVisibleTo(page)]


# --- helpers ----------------------------------------------------------------------------------------


def test_formatting_helpers():
    assert format_size(483_546_902) == "484\u00a0МБ"
    assert format_size(1_527_906_378) == "1,5\u00a0ГБ"
    assert format_size(2_002) == "1\u00a0МБ"
    assert format_seconds(800) == "0,8\u00a0с"
    spec = CATALOG["whisper-small"]
    assert model_choice_title(spec) == f"Збалансоване (small) — {format_size(spec.size)}"


# --- bound to the settings --------------------------------------------------------------------------


def test_widgets_show_the_settings(vf, make_page):
    s = vf._settings
    s.voice.tts_enabled = True
    s.voice.piper_speaker = "lada"
    s.voice.speed = 1.2
    s.voice.volume = 55
    s.voice.speak = "replies"
    s.voice.stt_model = "whisper-base"
    s.voice.ptt_hotkey = "f13"
    s.voice.ptt_mode = "hold"
    s.voice.end_silence_ms = 1200
    page = make_page()
    assert page.tts_enabled_check.isChecked()
    assert page.speaker_combo.currentData() == "lada"
    assert page.speed_row.value() == 120
    assert page.volume_row.value() == 55
    assert page.speak_combo.currentData() == "replies"
    assert page.stt_model_combo.currentData() == "whisper-base"
    assert page.hotkey_combo.currentData() == "f13"
    assert page.ptt_mode_combo.currentData() == "hold"
    assert page.pause_row.value() == 1200
    assert not page.has_edits()


def test_collect_writes_only_edited_fields(vf, make_page):
    page = make_page()
    page.speaker_combo.setCurrentIndex(page.speaker_combo.findData("mykyta"))
    page.speed_row.setValue(90)
    vf._settings.voice.volume = 30  # changed elsewhere meanwhile
    collected = page.collect(vf.settings())
    assert collected.voice.piper_speaker == "mykyta"
    assert collected.voice.speed == pytest.approx(0.9)
    assert collected.voice.volume == 30  # not reverted by the stale slider


def test_sync_keeps_unapplied_edits_and_reset_baseline(vf, make_page):
    page = make_page()
    page.speak_combo.setCurrentIndex(page.speak_combo.findData("off"))
    other = vf.settings()
    other.voice.volume = 20
    other.voice.speak = "all"
    page.sync(other, edited_too=False)
    assert page.speak_combo.currentData() == "off"  # the user's edit wins until applied
    assert page.volume_row.value() == 20
    page.reset_baseline()
    assert not page.has_edits()


def test_hotkey_presets_with_titles_and_a_custom_one(vf, make_page):
    vf._settings.voice.ptt_hotkey = "ctrl+alt+shift+f11"
    page = make_page()
    data = [page.hotkey_combo.itemData(i) for i in range(page.hotkey_combo.count())]
    assert data[:len(PRESETS)] == list(PRESETS)
    assert page.hotkey_combo.itemText(0) == "Ctrl + Alt + Пробіл"
    assert page.hotkey_combo.currentData() == "ctrl+alt+shift+f11"
    assert not page.has_edits()


def test_stt_model_choices_come_from_the_catalog(make_page):
    page = make_page()
    titles = [page.stt_model_combo.itemText(i) for i in range(page.stt_model_combo.count())]
    expected = ("whisper-base", "whisper-small", "whisper-medium")
    assert titles == [model_choice_title(CATALOG[m]) for m in expected]
    page.stt_model_combo.setCurrentIndex(page.stt_model_combo.findData("whisper-medium"))
    assert page.stt_model_row.model_id == "whisper-medium"
    assert CATALOG["whisper-medium"].description in page.stt_model_row.description_label.text()


def test_privacy_note_and_experimental_wake_phrase(make_page):
    page = make_page()
    shown = labels(page)
    assert PRIVACY_NOTE in shown
    assert PRIVACY_NOTE == ("Мікрофон вмикається лише коли ти натискаєш клавішу або кнопку. "
                            "Звук не зберігається і не надсилається в інтернет.")
    assert "експериментально" in shown
    assert voice_page.WAKE_WARNING in shown
    assert "Голос Мії" in shown


def test_header_follows_her_name(vf, make_page):
    vf._settings.character.name = "Аня"
    page = make_page()
    assert "Її голос" in labels(page)
    assert any(text.startswith("Аня може говорити") for text in labels(page))


# --- engines and the Azure key ----------------------------------------------------------------------


def test_engine_switch_shows_the_right_options(make_page):
    page = make_page()
    assert page.piper_box.isVisibleTo(page) and not page.azure_box.isVisibleTo(page)
    page.engine_combo.setCurrentIndex(page.engine_combo.findData("azure"))
    assert page.azure_box.isVisibleTo(page) and not page.piper_box.isVisibleTo(page)
    assert page.azure_cloud_note.text() == voice_page.AZURE_CLOUD_WARNING


def test_azure_cloud_note_when_permitted(vf, make_page):
    vf.permissions.grant(Permission.CLOUD_AI)
    vf._settings.voice.tts_engine = "azure"
    page = make_page()
    assert page.azure_cloud_note.text() == voice_page.AZURE_CLOUD_OK


def test_azure_key_is_saved_and_never_shown_back(vf, make_page, caplog):
    caplog.set_level(logging.DEBUG)
    vf._settings.voice.tts_engine = "azure"
    page = make_page()
    assert page.azure_key_edit.echoMode() == QLineEdit.EchoMode.Password
    assert page.azure_key_state_label.text() == "Ключ ще не збережено"
    page.azure_key_edit.setText(KEY)
    page.save_azure_key_button.click()
    assert vf.key == KEY
    assert page.azure_key_edit.text() == ""
    assert page.azure_key_state_label.text() == "🔑 Ключ збережено"
    assert KEY not in caplog.text
    assert all(KEY not in text for text in labels(page))
    assert page.delete_azure_key_button.isEnabled()


def test_bad_or_failing_azure_key(vf, make_page, caplog):
    vf._settings.voice.tts_engine = "azure"
    page = make_page()
    page.azure_key_edit.setText("has spaces in it")
    page.save_azure_key_button.click()
    assert vf.key_calls == []
    assert "одним рядком" in page.azure_key_message_label.text()
    vf.key_error = RuntimeError(KEY)
    page.azure_key_edit.setText(KEY)
    page.save_azure_key_button.click()
    assert page.azure_key_message_label.text() == "Не вдалося зберегти ключ."
    assert KEY not in caplog.text


def test_delete_azure_key_asks_first(vf, make_page, confirm):
    vf.key = KEY
    vf._settings.voice.tts_engine = "azure"
    page = make_page()
    confirm.answer = False
    page.delete_azure_key_button.click()
    assert vf.key == KEY
    confirm.answer = True
    page.delete_azure_key_button.click()
    assert vf.key == "" and vf.key_calls == [0]


def test_pending_key_is_saved_by_save_pending_key(vf, make_page):
    page = make_page()
    page.azure_key_edit.setText(KEY)
    assert page.save_pending_key()
    assert vf.key == KEY


def test_bad_region_is_flagged_and_not_collected(vf, make_page):
    page = make_page()
    page.engine_combo.setCurrentIndex(page.engine_combo.findData("azure"))
    page.azure_region_edit.setText("west europe!")
    assert not page.validate()
    assert page.region_error.isVisibleTo(page)
    assert page.collect(vf.settings()).voice.azure_region == "westeurope"
    page.azure_region_edit.setText("NorthEurope")
    assert not page.region_error.isVisibleTo(page)
    assert page.validate()
    assert page.collect(vf.settings()).voice.azure_region == "northeurope"


# --- "Прослухати" -------------------------------------------------------------------------------------


def test_listen_applies_the_page_first_then_reports(vf, make_page):
    page = make_page()
    page.speaker_combo.setCurrentIndex(page.speaker_combo.findData("lada"))
    page.test_button.click()
    assert vf.settings_at_test[-1].voice.piper_speaker == "lada"  # tested with what the page shows
    assert not page.has_edits()
    assert not page.test_button.isEnabled() and page.test_button.text() == "Звучить…"
    page.test_button.click()  # ignored while busy
    assert len(vf.tests) == 1
    vf.tests[-1](True, "")
    assert page.test_button.isEnabled()
    assert page.test_result_label.text() == "✓ Ось так вона звучить."


def test_listen_failure_and_late_answers(vf, make_page):
    page = make_page()
    page.test_button.click()
    vf.tests[-1](False, "Голос ще не завантажено (Налаштування → Голос).")
    assert page.test_result_label.text() == "✗ Голос ще не завантажено (Налаштування → Голос)."
    page.test_button.click()
    page.test_watchdog.timeout.emit()
    assert page.test_button.isEnabled()
    assert "так і не з'явився" in page.test_result_label.text()
    vf.tests[-1](True, "")  # the late answer is dropped
    assert "так і не з'явився" in page.test_result_label.text()


def test_listen_refuses_a_bad_region(vf, make_page):
    page = make_page()
    page.engine_combo.setCurrentIndex(page.engine_combo.findData("azure"))
    page.azure_region_edit.setText("??")
    page.test_button.click()
    assert vf.tests == []
    assert page.test_result_label.text().startswith("✗ Спершу виправ")


# --- models ---------------------------------------------------------------------------------------------


def test_model_rows_show_state(vf, make_page):
    page = make_page()
    voice_row, stt_row = page.voice_model_row, page.stt_model_row
    assert voice_row.state_label.text() == "завантажено ✓"
    assert voice_row.remove_button.isVisibleTo(page) and not voice_row.download_button.isVisibleTo(page)
    assert stt_row.model_id == "whisper-small"
    assert stt_row.state_label.text() == "не завантажено"
    assert stt_row.download_button.isVisibleTo(page) and not stt_row.remove_button.isVisibleTo(page)
    assert format_size(CATALOG["whisper-small"].size) in stt_row.title_label.text()


def test_download_progress_cancel_and_done(vf, make_page):
    page = make_page()
    row = page.stt_model_row
    row.download_button.click()
    model_id, on_progress, _on_done = vf.downloads[-1]
    assert model_id == "whisper-small"
    assert row.progress.isVisibleTo(page) and row.cancel_button.isVisibleTo(page)
    assert not row.download_button.isVisibleTo(page)
    other = page.voice_model_row.download_button
    assert not other.isEnabled() or other.isHidden()
    size = CATALOG["whisper-small"].size
    on_progress(size // 2, size)
    assert row.progress.value() == 500
    assert "із" in row.progress_label.text()
    row.cancel_button.click()
    assert vf.cancels == 1
    assert not row.cancel_button.isEnabled() and row.cancel_button.text() == "Скасовую…"
    vf.finish_download(False, "Завантаження скасовано.")
    assert not row.progress.isVisibleTo(page)
    assert row.message_label.text() == "Завантаження скасовано."
    assert row.message_label.property("role") == "error"
    row.download_button.click()
    vf.finish_download(True)
    assert row.state_label.text() == "завантажено ✓"
    assert row.message_label.text() == "Завантажено ✓"


def test_only_one_download_at_a_time(vf, make_page):
    vf.installed = set()
    page = make_page()
    page.stt_model_row.download_button.click()
    assert not page.voice_model_row.download_button.isEnabled()
    page.voice_model_row.download_requested.emit(DEFAULT_TTS_MODEL)
    assert len(vf.downloads) == 1


def test_reopened_page_follows_a_running_download(vf, make_page):
    size = CATALOG["whisper-medium"].size
    vf._settings.voice.stt_model = "whisper-medium"
    vf.state = DownloadProgress("whisper-medium", size // 4, size)
    page = make_page()
    row = page.stt_model_row
    assert row.progress.isVisibleTo(page) and row.progress.value() == 250
    assert page.download_timer.isActive()
    vf.state = DownloadProgress("whisper-medium", size, size)
    page._poll_download()
    assert row.progress.value() == 1000
    vf.state = None
    vf.installed.add("whisper-medium")
    page._poll_download()
    assert not row.progress.isVisibleTo(page) and not page.download_timer.isActive()
    assert row.state_label.text() == "завантажено ✓"


def test_remove_asks_first(vf, make_page, confirm):
    page = make_page()
    confirm.answer = False
    page.voice_model_row.remove_button.click()
    assert vf.removed == []
    confirm.answer = True
    page.voice_model_row.remove_button.click()
    assert vf.removed == [DEFAULT_TTS_MODEL]
    assert "буде видалено" in confirm.asked[-1][1]
    assert page.voice_model_row.state_label.text() == "не завантажено"
    assert page.voice_model_row.message_label.text() == "Модель видалено."


def test_download_is_never_started_without_a_click(vf, make_page):
    vf.installed = set()
    page = make_page()
    page.refresh()
    page.stt_model_combo.setCurrentIndex(page.stt_model_combo.findData("whisper-base"))
    assert vf.downloads == []


# --- microphone permission --------------------------------------------------------------------------------


def test_microphone_permission_needs_consent(vf, make_page, confirm):
    page = make_page()
    assert page.mic_box.isVisibleTo(page)
    granted = []
    page.permissions_changed.connect(lambda: granted.append(True))
    confirm.answer = False
    page.allow_mic_button.click()
    assert not vf.permissions.is_granted(Permission.MICROPHONE)
    confirm.answer = True
    page.allow_mic_button.click()
    assert confirm.asked[-1] == (MICROPHONE_CONSENT_TITLE, MICROPHONE_CONSENT_TEXT)
    assert vf.permissions.is_granted(Permission.MICROPHONE)
    assert granted == [True]
    assert not page.mic_box.isVisibleTo(page)


def test_microphone_unavailable_in_this_build(vf, make_page, monkeypatch):
    monkeypatch.setattr(permissions_module, "AVAILABLE", frozenset({Permission.APP_METADATA}))
    page = make_page()
    page.allow_mic_button.click()
    assert page.mic_permission_label.text() == "Цей дозвіл зараз не можна надати."


def test_revoked_elsewhere_shows_the_warning_again(vf, make_page):
    vf.permissions.grant(Permission.MICROPHONE)
    page = make_page()
    assert not page.mic_box.isVisibleTo(page)
    vf.permissions.revoke(Permission.MICROPHONE)
    page.refresh()
    assert page.mic_box.isVisibleTo(page)


# --- devices ---------------------------------------------------------------------------------------------


def test_devices_are_listed_with_the_system_default(vf, make_page):
    page = make_page()
    items = [page.input_device_combo.itemText(i) for i in range(page.input_device_combo.count())]
    assert items == [voice_page.DEFAULT_DEVICE, "USB Mic", "Webcam Mic"]
    assert page.output_device_combo.currentData() == ""
    page.output_device_combo.setCurrentIndex(page.output_device_combo.findData("Headset"))
    assert page.collect(vf.settings()).voice.output_device == "Headset"


def test_unplugged_device_stays_selected(vf, make_page):
    vf._settings.voice.input_device = "Old Mic"
    page = make_page()
    assert page.input_device_combo.currentData() == "Old Mic"
    assert page.input_device_combo.currentText() == "Old Mic (не підключено)"
    assert not page.has_edits()


def test_device_listing_failure_is_guarded(vf, make_page):
    vf.devices_error = RuntimeError("PortAudio")
    page = make_page()
    assert page.input_device_combo.count() == 1
    assert page.devices_note.isVisibleTo(page)


# --- status --------------------------------------------------------------------------------------------


def test_status_lines(vf, make_page):
    vf.status = VoiceStatus(tts_ready=True, stt_ready=True, tts_reason="", stt_reason="", state="idle",
                            hotkey="Ctrl + Alt + Пробіл", hotkey_active=True)
    page = make_page()
    assert page.tts_status_label.text() == "● Голос працює — Piper, Тетяна"
    assert page.tts_status_label.property("role") == "success"
    assert "Ctrl + Alt + Пробіл" in page.stt_status_label.text()
    assert not page.hotkey_status_label.isVisibleTo(page)


def test_status_reasons(vf, make_page):
    vf.status = VoiceStatus(tts_ready=False, stt_ready=False, tts_reason=VOICE_MODULE_MISSING,
                            stt_reason=VOICE_MODULE_MISSING, state="off", hotkey="", hotkey_active=False)
    page = make_page()
    assert page.tts_status_label.text() == f"● {VOICE_MODULE_MISSING}"
    assert page.tts_status_label.property("role") == "warning"
    assert not page.stt_status_label.isVisibleTo(page)  # said once, not twice
    vf.status = VoiceStatus(tts_ready=True, stt_ready=True, tts_reason="", stt_reason="", state="idle",
                            hotkey="F13", hotkey_active=False, hotkey_reason="Комбінацію вже зайнято.")
    page.refresh_status()
    assert page.hotkey_status_label.text() == "Комбінацію вже зайнято."
    assert "кнопку мікрофона" in page.stt_status_label.text()


def test_status_failure_is_logged_once(vf, make_page, caplog):
    vf.status_error = RuntimeError("boom")
    page = make_page()
    page.refresh_status()
    assert page.tts_status_label.text() == "● Стан голосу зараз невідомий"
    assert caplog.text.count("reading the voice status failed") == 1


# --- keyboard accessibility ------------------------------------------------------------------------------


def test_every_control_is_keyboard_reachable_and_named(make_page):
    page = make_page()
    controls = [w for w in page.findChildren(QAbstractButton) + page.findChildren(QComboBox)
                + page.findChildren(QSlider) + page.findChildren(QLineEdit)
                if w.isVisibleTo(page) and w.parent() is not None]
    assert controls
    for widget in controls:
        assert widget.focusPolicy() & Qt.FocusPolicy.TabFocus, widget
        if isinstance(widget, (QComboBox, QSlider, QLineEdit)) and not isinstance(widget.parent(), QComboBox):
            assert widget.accessibleName() or isinstance(widget.parent(), QComboBox), widget


def test_keyboard_user_keeps_focus_on_the_model_row(vf, make_page):
    page = make_page()
    page.activateWindow()
    assert QTest.qWaitForWindowActive(page, 1000)
    row = page.stt_model_row
    row.download_button.setFocus()
    QTest.keyClick(row.download_button, Qt.Key.Key_Space)
    assert vf.downloads and QApplication.focusWidget() is row.cancel_button
    assert row.state_label.property("role") == "pill" and row.state_label.property("state") == "alert"
    vf.finish_download(True)
    assert QApplication.focusWidget() is row.remove_button
    assert row.state_label.property("state") == "on"
