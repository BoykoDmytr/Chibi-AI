from __future__ import annotations

import logging
from dataclasses import replace

import pytest
from PySide6.QtGui import QColor, QPixmap

from chibi.character.states import CharacterState
from chibi.core.events import Permission
from chibi.settings.facade import AIStatusView
from chibi.settings.status_window import EMOTION_BARS, REFRESH_MS, StatusWindow, ai_line, portrait_for
from chibi.settings.widgets import STATE_NAMES_UK, STATE_PORTRAITS

NOW = 1_700_000_000.0


@pytest.fixture
def make_window(facade, track):
    def make(**kwargs) -> StatusWindow:
        window = track(StatusWindow(facade, now=lambda: NOW, **kwargs))
        window.show()
        return window

    return make


def pixmap(color: str = "#ff00ff") -> QPixmap:
    p = QPixmap(64, 64)
    p.fill(QColor(color))
    return p


def test_shows_name_mood_and_state(make_window, facade):
    facade.status_view = replace(facade.status_view, state="sleep")
    window = make_window()
    assert window.isVisible()
    assert window.name_label.text() == "Мія"
    assert window.mood_label.text() == "Настрій: Весела"
    assert window.state_label.text() == "Зараз: спить"
    assert "Мія" in window.windowTitle()


def test_bars_reflect_needs(make_window, facade):
    facade.needs_view = replace(facade.needs_view, hunger=30, energy=80, happiness=64.6, boredom=12,
                                affection=55)
    window = make_window()
    assert window.satiety_bar.value() == 70  # Ситість = 100 - hunger
    assert window.satiety_bar.title_label.text() == "Ситість"
    assert window.energy_bar.value() == 80
    assert window.happiness_bar.value() == 65
    assert window.boredom_bar.value() == 12
    assert window.affection_bar.value() == 55
    assert window.satiety_bar.value_label.text() == "70 %"


def test_bar_colours_warn_when_needs_are_bad(make_window, facade):
    facade.needs_view = replace(facade.needs_view, hunger=95, energy=30, boredom=90, happiness=70)
    window = make_window()
    assert window.satiety_bar.tone() == "danger"
    assert window.energy_bar.tone() == "warning"
    assert window.boredom_bar.tone() == "danger"  # inverted: a lot of boredom is bad
    assert window.happiness_bar.tone() == "pink"
    facade.needs_view = replace(facade.needs_view, hunger=10, energy=90, boredom=5)
    window.refresh()
    assert window.satiety_bar.tone() == "success"
    assert window.boredom_bar.tone() == "lilac"


def test_friendship_level_and_progress(make_window, facade):
    window = make_window()
    assert window.friendship_label.text() == "Рівень дружби 3"
    assert window.friendship_bar.value() == 45
    assert "45" in window.friendship_hint.text()
    facade.needs_view = replace(facade.needs_view, friendship_level=10, friendship_progress=0.3)
    window.refresh()
    assert window.friendship_label.text() == "Рівень дружби 10"
    assert window.friendship_bar.value() == 100
    assert "найвищий" in window.friendship_hint.text()


def test_observation_indicator(make_window, permissions):
    window = make_window()
    assert window.observation_label.text().startswith("👁")
    assert "вимкнено" in window.observation_label.text()
    permissions.grant(Permission.APP_METADATA)
    window.refresh()
    assert "програми" in window.observation_label.text()
    assert window.observation_label.property("state") == "on"


def test_observation_unsupported_platform(make_window, facade):
    facade.status_view = replace(facade.status_view, activity_supported=False)
    window = make_window()
    assert "недоступне" in window.observation_label.text()


def test_quiet_dnd_and_ai_indicators(make_window, facade):
    window = make_window()
    assert window.quiet_label.isHidden()
    assert window.dnd_label.isHidden()
    assert window.ai_label.text() == "🤖 AI вимкнено: говорить готовими репліками"
    facade.status_view = replace(facade.status_view, quiet_until=NOW + 1800, do_not_disturb=True)
    window.refresh()
    assert not window.quiet_label.isHidden()
    assert window.quiet_label.text().startswith("🔇 Мовчить до ")
    assert not window.dnd_label.isHidden()
    # a value that can't be a near-future epoch time is still shown, just without a clock time
    facade.status_view = replace(facade.status_view, quiet_until=12345.0)
    window.refresh()
    assert window.quiet_label.text() == "🔇 Мовчить"


def test_buttons_call_the_facade(make_window, facade):
    window = make_window()
    window.feed_button.click()
    window.pet_button.click()
    window.rest_button.click()
    assert facade.calls == ["feed", "pet", "rest"]
    assert [b.text() for b in (window.feed_button, window.pet_button, window.rest_button)] == [
        "Погодувати", "Погладити", "Відпочити"]


def test_polls_every_2s_only_while_visible(make_window, facade):
    window = make_window()
    assert window.refresh_timer.interval() == REFRESH_MS == 2000
    assert window.refresh_timer.isActive()
    facade.needs_view = replace(facade.needs_view, energy=11)
    window.refresh_timer.timeout.emit()
    assert window.energy_bar.value() == 11
    window.hide()
    assert not window.refresh_timer.isActive()
    window.show()
    assert window.refresh_timer.isActive()


def test_timer_not_running_before_first_show(facade, track):
    window = track(StatusWindow(facade))
    assert not window.refresh_timer.isActive()


def test_portrait_falls_back_to_letter_avatar(make_window, facade):
    window = make_window()
    assert window.portrait.current == ""
    assert not window.portrait.pixmap().isNull()


def test_portrait_follows_state_and_needs(make_window, facade):
    facade.portraits = {"tired": pixmap(), "calm": pixmap(), "laugh": pixmap()}
    facade.needs_view = replace(facade.needs_view, energy=10)
    window = make_window()
    assert window.portrait.current == "tired"
    facade.status_view = replace(facade.status_view, state="laugh")
    window.refresh()
    assert window.portrait.current == "laugh"
    requests = len(facade.portrait_requests)
    window.refresh()
    assert len(facade.portrait_requests) == requests  # cached, no reload every 2 s


def test_portrait_for_prefers_feelings_while_idle():
    from chibi.settings.facade import NeedsView

    base = NeedsView(hunger=20, energy=80, happiness=50, boredom=20, affection=50,
                     friendship_level=1, friendship_progress=0.0, mood="")
    assert portrait_for("idle", base) == ["calm"]
    assert portrait_for("idle", replace(base, energy=10))[0] == "tired"
    assert portrait_for("walk", replace(base, hunger=90))[0] == "offended"
    assert portrait_for("sit", replace(base, happiness=90))[0] == "joy"
    assert portrait_for("angry", replace(base, energy=10)) == ["angry", "calm"]
    assert portrait_for("unknown-state", base) == ["calm"]


def test_refresh_failure_is_logged_once(make_window, facade, caplog):
    window = make_window()

    def broken():
        raise RuntimeError("boom")

    facade.needs = broken
    with caplog.at_level(logging.ERROR, logger="chibi.settings.status_window"):
        window.refresh()
        window.refresh()
    assert len([r for r in caplog.records if "refresh failed" in r.getMessage()]) == 1


def test_every_character_state_has_a_ukrainian_name_and_portrait():
    states = {s.value for s in CharacterState}
    assert states <= set(STATE_NAMES_UK)
    assert states <= set(STATE_PORTRAITS)


# --- Phase 2: feelings and AI ----------------------------------------------------------------------


def test_emotion_bars_show_curiosity_excitement_and_trust(make_window, facade):
    window = make_window()
    assert [key for key, *_ in EMOTION_BARS] == ["curiosity", "excitement", "trust"]
    assert window.curiosity_bar.title_label.text() == "Цікавість"
    assert window.excitement_bar.title_label.text() == "Азарт"
    assert window.trust_bar.title_label.text() == "Довіра"
    assert (window.curiosity_bar.value(), window.excitement_bar.value(), window.trust_bar.value()) == (
        62, 35, 48)
    facade.emotion_values.update(curiosity=5.4, excitement=99.6, trust=0.0)
    window.refresh()
    assert (window.curiosity_bar.value(), window.excitement_bar.value(), window.trust_bar.value()) == (
        5, 100, 0)
    # feelings are not needs: a low value is a calmer mood, not a warning
    assert window.curiosity_bar.tone() == "lilac"
    assert window.trust_bar.tone() == "success"


def test_emotion_failure_keeps_the_rest_and_the_last_values(make_window, facade, caplog):
    window = make_window()
    facade.emotion_error = RuntimeError("engine not ready")
    facade.needs_view = replace(facade.needs_view, energy=12)
    with caplog.at_level(logging.ERROR, logger="chibi.settings.status_window"):
        window.refresh()
        window.refresh()
    assert window.energy_bar.value() == 12  # needs still refreshed
    assert window.curiosity_bar.value() == 62  # last known value kept
    assert len([r for r in caplog.records if "emotions" in r.getMessage()]) == 1


def test_missing_or_bad_emotion_values_are_skipped(make_window, facade):
    window = make_window()
    facade.emotion_values = {"curiosity": "a lot", "trust": float("nan")}
    window.refresh()
    assert window.curiosity_bar.value() == 62
    assert window.trust_bar.value() == 0  # NaN shows as 0, never crashes
    assert window.excitement_bar.value() == 35


AI_ONLINE = AIStatusView(provider="anthropic", model="claude-opus-5-5", online=True, reason="", cloud=True,
                         requests_today=3, requests_limit_day=400, cost_today_usd=0.01, cost_limit_usd=1.0,
                         cost_known=True)


def test_ai_line_online_offline_and_off(make_window, facade):
    facade.ai_status_view = AI_ONLINE
    window = make_window()
    assert window.ai_label.text() == "🤖 AI: Anthropic Claude · claude-opus-5-5 (хмара)"
    assert window.ai_label.property("state") == "on"
    facade.ai_status_view = replace(AI_ONLINE, online=False, reason="немає API-ключа")
    window.refresh()
    assert window.ai_label.text() == "🤖 AI офлайн: немає API-ключа"
    assert window.ai_label.property("state") == "alert"
    local = replace(AI_ONLINE, provider="ollama", model="qwen3:8b", cloud=False)
    assert ai_line(local) == ("🤖 AI: Ollama · qwen3:8b (локально)", "on")


def test_ai_line_falls_back_to_status_when_ai_status_fails(make_window, facade):
    facade.ai_status_error = RuntimeError("boom")
    facade.status_view = replace(facade.status_view, ai_connected=True)
    window = make_window()
    assert window.ai_label.text() == "🤖 AI: підключено"
    assert window.name_label.text() == "Мія"  # the rest still refreshed
