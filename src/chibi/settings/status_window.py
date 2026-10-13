"""Status window: how she is doing right now (needs, feelings, mood, friendship,
what is active, whether she talks through AI).

It polls the facade every 2 s, but only while it is visible - a hidden or
minimised status window costs nothing.  Everything shown is computed by the
app; this window only formats it.  Each part is read separately, so a failing
part (e.g. the AI status) never blanks the rest.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QHideEvent, QShowEvent
from PySide6.QtWidgets import QHBoxLayout, QProgressBar, QVBoxLayout, QWidget

from chibi.settings.facade import AIStatusView, AppFacade, NeedsView, StatusView
from chibi.settings.widgets import (
    STATE_PORTRAITS,
    Card,
    NeedsBar,
    PortraitLabel,
    SectionHeader,
    ai_title,
    format_local_time,
    make_button,
    make_label,
    percent,
    set_style_property,
    state_name_uk,
)

log = logging.getLogger(__name__)

REFRESH_MS = 2000
MAX_FRIENDSHIP_LEVEL = 10
#: How far ahead a "quiet until" timestamp may be and still be shown as a clock time.
_MAX_QUIET_AHEAD_SECONDS = 2 * 24 * 3600

#: Postures in which the portrait follows her needs rather than the state itself.
_CALM_STATES = frozenset({"idle", "walk", "sit"})

#: facade.emotion() key -> (bar title, tone, tooltip).  The other emotions
#: (happiness, energy, boredom, affection, hunger) are the needs bars above.
EMOTION_BARS: tuple[tuple[str, str, str, str], ...] = (
    ("curiosity", "Цікавість", "lilac", "Наскільки їй цікаво, що відбувається довкола."),
    ("excitement", "Азарт", "pink", "Росте, коли ти граєш чи з нею бавишся, і поволі вщухає."),
    ("trust", "Довіра", "success", "Росте від спілкування й турботи — повільно, як і справжня."),
)


def ai_line(view: AIStatusView) -> tuple[str, str]:
    """(pill text, pill state) for the AI indicator."""
    if view.online:
        where = "хмара" if view.cloud else "локально"
        return f"🤖 AI: {ai_title(view.provider, view.model)} ({where})", "on"
    if view.provider == "none":
        return "🤖 AI вимкнено: говорить готовими репліками", ""
    reason = view.reason.strip()
    return (f"🤖 AI офлайн: {reason}" if reason else "🤖 AI: офлайн-режим"), "alert"


def portrait_for(state: str, needs: NeedsView) -> list[str]:
    """Portrait names to try, best first, for *state* and *needs*.

    Actions (laughing, eating, angry...) show their own face; while she is just
    standing, walking or sitting, the face reflects how she feels.
    """
    names: list[str] = []
    if state in _CALM_STATES:
        if needs.energy < 25:
            names.append("tired")
        elif needs.hunger > 75 or needs.happiness < 30:
            names.append("offended")
        elif needs.boredom > 75:
            names.append("phone")
        elif needs.happiness > 80:
            names.append("joy")
        elif needs.happiness > 60:
            names.append("smile")
    names.append(STATE_PORTRAITS.get(state, "calm"))
    names.append("calm")
    return list(dict.fromkeys(names))


class StatusWindow(QWidget):
    def __init__(self, facade: AppFacade, parent: QWidget | None = None, *,
                 now: Callable[[], float] = time.time) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self._facade = facade
        self._now = now
        self._failed = False
        self._emotion_failed = False
        self._ai_failed = False
        self.setWindowTitle("Як вона?")
        self.setMinimumWidth(380)

        # --- header: portrait, name, mood, state
        header = Card()
        top = QHBoxLayout()
        top.setSpacing(16)
        self.portrait = PortraitLabel(facade.portrait, 112)
        top.addWidget(self.portrait, 0, Qt.AlignmentFlag.AlignTop)
        info = QVBoxLayout()
        info.setSpacing(4)
        self.name_label = make_label("", "title", wrap=False)
        self.mood_label = make_label("", "subtitle")
        self.state_label = make_label("", "hint")
        info.addWidget(self.name_label)
        info.addWidget(self.mood_label)
        info.addWidget(self.state_label)
        info.addStretch(1)
        top.addLayout(info, 1)
        header.body.addLayout(top)

        # --- needs
        needs_card = Card(spacing=10)
        needs_card.body.addWidget(SectionHeader("Потреби"))
        self.satiety_bar = NeedsBar("Ситість", "success", tooltip="Чим вище, тим менше вона голодна.")
        self.energy_bar = NeedsBar("Енергія", "lilac")
        self.happiness_bar = NeedsBar("Настрій", "pink")
        self.boredom_bar = NeedsBar("Нудьга", "lilac", inverted=True,
                                    tooltip="Чим вище, тим більше їй нудно — пограйся з нею.")
        self.affection_bar = NeedsBar("Прив'язаність", "pink")
        for bar in (self.satiety_bar, self.energy_bar, self.happiness_bar, self.boredom_bar,
                    self.affection_bar):
            needs_card.body.addWidget(bar)

        # --- feelings (the emotion engine); no "low = bad" colours here
        self.emotion_card = Card(spacing=10)
        self.emotion_card.body.addWidget(SectionHeader("Почуття"))
        self.emotion_bars: dict[str, NeedsBar] = {}
        for key, title, tone, tooltip in EMOTION_BARS:
            bar = NeedsBar(title, tone, alerts=False, tooltip=tooltip)
            self.emotion_bars[key] = bar
            self.emotion_card.body.addWidget(bar)
        self.curiosity_bar = self.emotion_bars["curiosity"]
        self.excitement_bar = self.emotion_bars["excitement"]
        self.trust_bar = self.emotion_bars["trust"]

        # --- friendship
        friend_card = Card(spacing=6)
        row = QHBoxLayout()
        self.friendship_label = make_label("", "section", wrap=False)
        self.friendship_hint = make_label("", "hint", wrap=False)
        row.addWidget(self.friendship_label)
        row.addStretch(1)
        row.addWidget(self.friendship_hint)
        friend_card.body.addLayout(row)
        self.friendship_bar = QProgressBar()
        self.friendship_bar.setRange(0, 100)
        self.friendship_bar.setTextVisible(False)
        self.friendship_bar.setProperty("tone", "lilac")
        self.friendship_bar.setAccessibleName("Рівень дружби")
        friend_card.body.addWidget(self.friendship_bar)

        # --- indicators
        flags = Card(spacing=6)
        self.observation_label = make_label("", "pill")
        self.quiet_label = make_label("", "pill")
        self.dnd_label = make_label("🌙 Режим «Не турбувати»", "pill")
        self.ai_label = make_label("", "pill")
        for label in (self.observation_label, self.quiet_label, self.dnd_label, self.ai_label):
            flags.body.addWidget(label)

        # --- actions
        actions = QHBoxLayout()
        self.feed_button = make_button("Погодувати", "primary")
        self.pet_button = make_button("Погладити", "accent")
        self.rest_button = make_button("Відпочити")
        self.feed_button.clicked.connect(lambda: self._act(self._facade.feed))
        self.pet_button.clicked.connect(lambda: self._act(self._facade.pet))
        self.rest_button.clicked.connect(lambda: self._act(self._facade.rest))
        for button in (self.feed_button, self.pet_button, self.rest_button):
            actions.addWidget(button)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        for widget in (header, needs_card, self.emotion_card, friend_card, flags):
            root.addWidget(widget)
        root.addLayout(actions)
        root.addStretch(1)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(REFRESH_MS)
        self.refresh_timer.timeout.connect(self.refresh)
        self.refresh()

    # --- refreshing -----------------------------------------------------------------------

    def refresh(self) -> None:
        try:
            status = self._facade.status()
            needs = self._facade.needs()
        except Exception:
            # Log the first failure only: this runs every 2 s.
            if not self._failed:
                log.exception("status window refresh failed")
            self._failed = True
            return
        self._failed = False
        self._show_identity(status, needs)
        self._show_needs(needs)
        self._show_flags(status)
        self._show_emotions()
        self._show_ai(status)

    def _show_emotions(self) -> None:
        try:
            values = self._facade.emotion()
        except Exception:
            if not self._emotion_failed:
                log.exception("reading her emotions failed")
            self._emotion_failed = True
            return  # keep the last values rather than flash zeros
        self._emotion_failed = False
        for key, bar in self.emotion_bars.items():
            value = values.get(key) if isinstance(values, Mapping) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                bar.set_value(float(value))

    def _show_ai(self, status: StatusView) -> None:
        try:
            view = self._facade.ai_status()
        except Exception:
            if not self._ai_failed:
                log.exception("reading the AI status failed")
            self._ai_failed = True
            text, state = ("🤖 AI: підключено", "on") if status.ai_connected else ("🤖 AI: офлайн-режим", "")
        else:
            self._ai_failed = False
            text, state = ai_line(view)
        self.ai_label.setText(text)
        set_style_property(self.ai_label, "state", state)

    def _show_identity(self, status: StatusView, needs: NeedsView) -> None:
        name = status.character_name
        self.setWindowTitle(f"{name} — як справи")
        self.name_label.setText(name)
        self.portrait.set_fallback_text(name)
        self.portrait.show_portrait(*portrait_for(status.state, needs))
        self.mood_label.setText(f"Настрій: {needs.mood}" if needs.mood else "")
        self.state_label.setText(f"Зараз: {state_name_uk(status.state).lower()}")

    def _show_needs(self, needs: NeedsView) -> None:
        self.satiety_bar.set_value(100 - needs.hunger)
        self.energy_bar.set_value(needs.energy)
        self.happiness_bar.set_value(needs.happiness)
        self.boredom_bar.set_value(needs.boredom)
        self.affection_bar.set_value(needs.affection)
        level = max(1, int(needs.friendship_level))
        self.friendship_label.setText(f"Рівень дружби {level}")
        if level >= MAX_FRIENDSHIP_LEVEL:
            self.friendship_bar.setValue(100)
            self.friendship_hint.setText("найвищий рівень")
        else:
            progress = min(max(float(needs.friendship_progress), 0.0), 1.0) * 100
            self.friendship_bar.setValue(round(progress))
            self.friendship_hint.setText(f"до наступного рівня: {percent(progress)}")

    def _show_flags(self, status: StatusView) -> None:
        if not status.activity_supported:
            text, state = "👁 Спостереження за програмами недоступне на цій системі", ""
        elif status.observing:
            text, state = "👁 Бачить, які програми й ігри відкриті", "on"
        else:
            text, state = "👁 Спостереження вимкнено", ""
        self.observation_label.setText(text)
        set_style_property(self.observation_label, "state", state)

        self.quiet_label.setVisible(status.quiet_until is not None)
        if status.quiet_until is not None:
            now = self._now()
            until = status.quiet_until
            # quiet_until is epoch seconds; show a clock time only when it is plausible.
            if now < until < now + _MAX_QUIET_AHEAD_SECONDS:
                self.quiet_label.setText(f"🔇 Мовчить до {format_local_time(until)}")
            else:
                self.quiet_label.setText("🔇 Мовчить")
            set_style_property(self.quiet_label, "state", "alert")

        self.dnd_label.setVisible(status.do_not_disturb)
        set_style_property(self.dnd_label, "state", "alert")

    def _act(self, action: Callable[[], None]) -> None:
        try:
            action()
        except Exception:
            log.exception("status window action failed")
        self.refresh()

    # --- visibility-driven polling ---------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.refresh()
        self.refresh_timer.start()

    def hideEvent(self, event: QHideEvent) -> None:
        # Also called when minimised: no polling while nobody can see it.
        self.refresh_timer.stop()
        super().hideEvent(event)
