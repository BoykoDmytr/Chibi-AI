"""How fast she answers, as the settings show it: the AI page's "Тримати модель у
пам'яті" (Ollama's keep-alive) and the model's side of her last answer in the voice
page's "Швидкість відповіді" card."""

from __future__ import annotations

import pytest

from chibi.ai.service import RequestTiming
from chibi.core.config import Settings
from chibi.settings import voice_page
from chibi.settings.ai_page import KEEP_LOADED_HINT, AIPage
from chibi.settings.voice_page import MODEL_LOAD_ADVICE, VoicePage, model_timing_text
from chibi.voice.integration import STT_OFF, VoiceStatus


@pytest.fixture
def ai_page(facade, track):
    def make(provider: str = "ollama", **ai) -> AIPage:
        s = facade.settings()
        s.ai.provider = provider
        for name, value in ai.items():
            setattr(s.ai, name, value)
        facade.apply_settings(s)
        facade.applied.clear()
        page = track(AIPage(facade, lambda _title, _text: False))
        page.show()
        return page

    return make


def test_keep_loaded_minutes_are_on_the_ai_page_for_ollama_only(ai_page):
    page = ai_page(keep_model_loaded_minutes=45)
    assert page.keep_loaded_row.isVisibleTo(page)
    assert page.keep_loaded_spin.value() == 45 and page.keep_loaded_spin.suffix() == " хв"
    assert (page.keep_loaded_spin.minimum(), page.keep_loaded_spin.maximum()) == (5, 240)
    assert "відеопам'ять" in KEEP_LOADED_HINT and "5–20 с" in KEEP_LOADED_HINT
    for provider in ("anthropic", "openai", "none"):
        page.provider_combo.setCurrentIndex(page.provider_combo.findData(provider))
        assert not page.keep_loaded_row.isVisibleTo(page), provider


def test_keep_loaded_minutes_are_collected_like_any_other_ai_setting(ai_page, facade):
    page = ai_page()
    assert not page.has_edits()
    page.keep_loaded_spin.setValue(120)
    assert page.has_edits()
    assert page.collect(facade.settings()).ai.keep_model_loaded_minutes == 120
    page.keep_loaded_spin.setValue(1000)
    assert page.collect(facade.settings()).ai.keep_model_loaded_minutes == 240


def plain(text: str) -> str:
    return text.replace("\N{NO-BREAK SPACE}", " ")  # "1,2 с" is kept on one line


def test_model_timing_text():
    quick = RequestTiming(queued=0.0, first_text=1.24, total=3.0)
    assert plain(model_timing_text(quick)) == (
        "Остання відповідь моделі (голосом чи в чаті): перші слова за 1,2 с.")
    loading = RequestTiming(queued=0.0, first_text=12.0, total=14.0, model_load=9.5, prompt_read=0.8)
    text = plain(model_timing_text(loading))
    assert "перші слова за 12,0 с, з них завантаження моделі — 9,5 с." in text
    assert text.endswith(MODEL_LOAD_ADVICE) and "Тримати модель у пам'яті" in text
    queued = RequestTiming(queued=4.0, first_text=None, total=6.0, model_load=0.02)
    text = plain(model_timing_text(queued))
    assert "перші слова за 6,0 с, з них черга — 4,0 с." in text  # not streamed: the whole answer
    assert MODEL_LOAD_ADVICE not in text


class SpeedFacade:
    """Just what the voice page reads, with the AI timing the app adds."""

    def __init__(self, permissions, timing=None):
        self.permissions = permissions
        self._settings = Settings()
        self.timing = timing
        self.timing_error: Exception | None = None

    def settings(self):
        return self._settings.copy()

    def apply_settings(self, settings):
        self._settings = settings.copy()

    def voice_status(self):
        return VoiceStatus(tts_ready=False, stt_ready=False, tts_reason="Голос вимкнено.", stt_reason=STT_OFF,
                           state="off", hotkey="Ctrl + Alt + Пробіл", hotkey_active=False)

    def voice_models(self):
        return []

    def voice_download_state(self):
        return None

    def voice_has_azure_key(self):
        return False

    def voice_devices(self):
        return [], []

    def ai_last_timing(self):
        if self.timing_error is not None:
            raise self.timing_error
        return self.timing


def test_the_speed_card_shows_the_models_side_of_the_last_answer(permissions, track):
    facade = SpeedFacade(permissions)
    page = track(VoicePage(facade, lambda _title, _text: True))
    page.show()
    assert not page.model_timing_label.isVisibleTo(page)  # no answer yet
    facade.timing = RequestTiming(queued=0.0, first_text=11.0, total=13.0, model_load=8.0)
    page.refresh_status()
    assert page.model_timing_label.isVisibleTo(page)
    assert "завантаження моделі — 8,0 с" in plain(page.model_timing_label.text())
    facade.timing_error = RuntimeError("gone")
    page.refresh_status()  # a failing read hides the line, the rest of the page stays
    assert not page.model_timing_label.isVisibleTo(page)


def test_the_speed_card_works_without_the_ai_timing(permissions, track, monkeypatch):
    facade = SpeedFacade(permissions)
    monkeypatch.delattr(SpeedFacade, "ai_last_timing")
    page = track(VoicePage(facade, lambda _title, _text: True))
    page.show()
    page.refresh_status()
    assert not page.model_timing_label.isVisibleTo(page)
    assert page.latency_total_label.text() == voice_page.LATENCY_NONE
