"""SpeechComposer: AI line or phrase-bank fallback, exactly one SpeechRequested
per compose, timeouts, late results, quiet mode - and the ReactionEngine /
DecisionEngine wiring around it."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime

import pytest

from chibi.ai.types import Purpose
from chibi.core.config import Settings
from chibi.core.events import (
    AppShuttingDown,
    AppStarted,
    DoNotDisturbChanged,
    GameStarted,
    InteractionKind,
    InteractionOutcome,
    QuietModeChanged,
    SpeechRequested,
)
from chibi.memory.repositories import PhraseHistoryRepository
from chibi.personality.composer import (
    DEFAULT_AI_TIMEOUT_SECONDS,
    MAX_PENDING_AI,
    SOLICITED_TIMEOUT_SECONDS,
    SpeechComposer,
)
from chibi.personality.phrases import PhraseBank, PhraseSelector
from chibi.personality.reactions import ReactionEngine


def entry(id_: str, text: str, state: str = "talk") -> dict:
    return {"id": id_, "text": text, "tone": "warm", "state": state, "portrait": "smile"}


BANK = PhraseBank.from_dict({"version": 1, "phrases": {
    "game_start.cs2": [entry("cs2.1", "О, {game}!"), entry("cs2.2", "{game}? Удачі!")],
    "game_start.generic": [entry("gen.1", "{game} — гарний вибір!")],
    "idle.joke": [entry("joke.1", "Жарт один."), entry("joke.2", "Жарт два.")],
    "interaction.pet.happy": [entry("pet.1", "Мур!", state="happy")],
    "late_night": [entry("night.1", "Вже {time}, час спати.")],
    "greet.morning": [entry("gm.1", "Доброго ранку!")], "greet.day": [entry("gd.1", "Привіт!")],
    "greet.evening": [entry("ge.1", "Добрий вечір!")], "greet.night": [entry("gn.1", "Доброї ночі!")],
    "voice.listening": [entry("listen.1", "Слухаю!")],
}})


class FakeBrain:
    """Records phrase() calls; tests complete them by hand (or synchronously)."""

    def __init__(self, available: bool = True) -> None:
        self.is_available = available
        self.purposes: list[Purpose] = []
        self.calls: list[tuple[tuple[str, ...], dict[str, str], bool, Callable]] = []
        self.sync_result: SpeechRequested | str | None = "pending"  # "pending": complete later
        self.raise_on_phrase = False

    def available(self, purpose: Purpose) -> bool:
        self.purposes.append(purpose)
        return self.is_available

    def phrase(self, keys: Sequence[str], variables: Mapping[str, str], solicited: bool,
               on_done: Callable[[SpeechRequested | None], None]) -> None:
        if self.raise_on_phrase:
            raise RuntimeError("boom")
        self.calls.append((tuple(keys), dict(variables), solicited, on_done))
        if self.sync_result != "pending":
            on_done(self.sync_result)  # type: ignore[arg-type]

    def complete(self, index: int, text: str | None, state: str = "happy") -> None:
        keys, _vars, solicited, on_done = self.calls[index]
        on_done(None if text is None else SpeechRequested(text=text, state=state, portrait="joy",
                                                          solicited=solicited, source=f"{keys[0]}:ai"))


class ManualSchedule:
    def __init__(self) -> None:
        self.timers: list[tuple[int, Callable[[], None]]] = []

    def __call__(self, msec: int, callback: Callable[[], None]) -> None:
        self.timers.append((msec, callback))

    def fire_all(self) -> None:
        timers, self.timers = self.timers, []
        for _msec, callback in timers:
            callback()


class Harness:
    def __init__(self, bus, db, clock, brain: FakeBrain | None = None, settings: Settings | None = None,
                 seed: int = 5) -> None:
        self.bus = bus
        self.db = db
        self.settings = settings or Settings()
        self.selector = PhraseSelector(BANK, PhraseHistoryRepository(db), lambda: self.settings, clock,
                                       random.Random(seed))
        self.brain = brain
        self.schedule = ManualSchedule()
        self.composer = SpeechComposer(bus, self.selector, brain, lambda: self.settings, clock,
                                       schedule=self.schedule)
        self.speech: list[SpeechRequested] = []
        bus.subscribe(SpeechRequested, self.speech.append)

    def take(self) -> list[SpeechRequested]:
        self.bus.dispatch_pending()
        out, self.speech[:] = list(self.speech), []
        return out


@pytest.fixture
def make(bus, db, clock):
    def build(brain: FakeBrain | None = None, **kwargs) -> Harness:
        return Harness(bus, db, clock, brain, **kwargs)
    return build


# -- without AI -------------------------------------------------------------------------------


def test_offline_compose_is_the_selector_line(make, bus, db, clock):
    h = make()
    assert h.composer.compose("game_start.cs2", "game_start.generic", game="CS2")
    [speech] = h.take()
    # The same seed through the selector directly gives the very same line.
    reference = PhraseSelector(BANK, PhraseHistoryRepository(db), lambda: h.settings, clock,
                               random.Random(5))
    reference.reload_history()
    db.conn.execute("DELETE FROM phrase_history")
    reference.reload_history()
    expected = reference.say("game_start.cs2", "game_start.generic", game="CS2")
    assert (speech.text, speech.state, speech.portrait, speech.source, speech.solicited) == (
        expected.text, expected.state, expected.portrait, expected.source, expected.solicited)
    assert h.schedule.timers == []


def test_offline_compose_without_a_phrase_publishes_nothing(make):
    h = make()
    assert not h.composer.compose("no.such.key")
    assert not h.composer.compose()
    assert h.take() == []


def test_offline_records_history(make, db):
    h = make()
    h.composer.compose("idle.joke")
    assert len(PhraseHistoryRepository(db).recent()) == 1


# -- with AI ---------------------------------------------------------------------------------------


def test_ai_success_publishes_exactly_the_ai_line(make, db):
    brain = FakeBrain()
    h = make(brain)
    assert h.composer.compose("game_start.cs2", game="CS2")
    assert h.take() == []  # waiting for the model
    assert brain.calls[0][:3] == (("game_start.cs2",), {"game": "CS2"}, False)
    assert brain.purposes == [Purpose.REACTION]
    brain.complete(0, "Ого, CS2! Покажи клас!")
    [speech] = h.take()
    assert speech.text == "Ого, CS2! Покажи клас!" and speech.source == "game_start.cs2:ai"
    h.schedule.fire_all()  # the timeout comes later: nothing more
    assert h.take() == []
    assert PhraseHistoryRepository(db).recent() == []  # no phrase-bank line was used
    assert h.composer.pending == 0


def test_ai_failure_falls_back_to_the_phrase_bank(make):
    brain = FakeBrain()
    h = make(brain)
    h.composer.compose("game_start.cs2", game="CS2")
    brain.complete(0, None)
    [speech] = h.take()
    assert speech.source == "game_start.cs2"
    assert "CS2" in speech.text
    h.schedule.fire_all()
    assert h.take() == []


def test_timeout_falls_back_and_the_late_ai_line_is_dropped(make):
    brain = FakeBrain()
    h = make(brain)
    h.composer.compose("idle.joke")
    assert h.schedule.timers[0][0] == int(DEFAULT_AI_TIMEOUT_SECONDS * 1000)
    h.schedule.fire_all()
    [fallback] = h.take()
    assert fallback.source == "idle.joke"
    brain.complete(0, "Запізнілий жарт.")
    assert h.take() == []


def test_solicited_lines_wait_less_and_stay_solicited(make):
    brain = FakeBrain()
    h = make(brain)
    h.composer.compose("interaction.pet.happy", solicited=True)
    assert h.schedule.timers[0][0] == int(SOLICITED_TIMEOUT_SECONDS * 1000)
    assert brain.calls[0][2] is True
    _keys, _vars, _sol, on_done = brain.calls[0]
    on_done(SpeechRequested(text="Мррр!", state="happy", solicited=False, source="x:ai"))
    [speech] = h.take()
    assert speech.solicited is True and speech.text == "Мррр!"


def test_initiative_keys_ask_for_the_initiative_purpose(make):
    brain = FakeBrain()
    h = make(brain)
    h.composer.compose("idle.joke")
    h.composer.compose("initiative.ask_game", game="CS2")
    assert brain.purposes == [Purpose.INITIATIVE, Purpose.INITIATIVE]


@pytest.mark.parametrize("setup", ["unavailable", "reactions_off"])
def test_no_ai_when_unavailable_or_switched_off(make, setup):
    brain = FakeBrain(available=setup != "unavailable")
    h = make(brain)
    if setup == "reactions_off":
        h.settings.ai.use_for_reactions = False
    assert h.composer.compose("idle.joke")
    [speech] = h.take()
    assert speech.source == "idle.joke"
    assert brain.calls == [] and h.schedule.timers == []


def test_interface_feedback_never_waits_for_the_model(make):
    # "Слухаю!" must come out the moment the microphone opens, not seconds later.
    brain = FakeBrain()
    h = make(brain)
    assert h.composer.compose("voice.listening", solicited=True)
    assert [s.text for s in h.take()] == ["Слухаю!"]
    assert brain.calls == [] and brain.purposes == [] and h.schedule.timers == []


def test_synchronous_none_from_the_brain_falls_back_at_once(make):
    brain = FakeBrain()
    brain.sync_result = None
    h = make(brain)
    assert h.composer.compose("idle.joke")
    [speech] = h.take()
    assert speech.source == "idle.joke"
    h.schedule.fire_all()
    assert h.take() == []


def test_brain_exception_falls_back(make):
    brain = FakeBrain()
    brain.raise_on_phrase = True
    h = make(brain)
    assert h.composer.compose("idle.joke")
    assert [s.source for s in h.take()] == ["idle.joke"]


def test_ai_path_without_any_phrase_bank_line(make):
    brain = FakeBrain()
    h = make(brain)
    assert h.composer.compose("initiative.share_memory")  # asked the AI
    brain.complete(0, None)
    assert h.take() == []  # nothing to fall back to: silence, not an error


def test_variables_are_passed_as_text(make):
    brain = FakeBrain()
    h = make(brain)
    h.composer.compose("game_start.cs2", game="CS2", duration=None, app="  ", count=3)
    assert brain.calls[0][1] == {"game": "CS2", "count": "3"}


def test_at_most_two_lines_wait_for_the_model(make):
    brain = FakeBrain()
    h = make(brain)
    for _ in range(MAX_PENDING_AI):
        h.composer.compose("idle.joke")
    assert h.composer.pending == MAX_PENDING_AI
    assert h.composer.compose("game_start.cs2", game="CS2")
    [immediate] = h.take()
    assert immediate.source == "game_start.cs2"
    assert len(brain.calls) == MAX_PENDING_AI


@pytest.mark.parametrize("event", [QuietModeChanged(enabled=True), DoNotDisturbChanged(enabled=True),
                                   AppShuttingDown()])
def test_quiet_dnd_or_shutdown_abandon_pending_lines(make, bus, event):
    brain = FakeBrain()
    h = make(brain)
    h.composer.compose("idle.joke")
    h.composer.compose("game_start.cs2", game="CS2")
    bus.publish(event)
    assert h.take() == []
    brain.complete(0, "Жарт після «помовч».")
    h.schedule.fire_all()
    brain.complete(1, None)
    assert h.take() == []
    assert h.composer.pending == 0


def test_quiet_off_does_not_abandon(make, bus):
    brain = FakeBrain()
    h = make(brain)
    h.composer.compose("idle.joke")
    bus.publish(QuietModeChanged(enabled=False))
    h.take()
    brain.complete(0, "Жарт.")
    assert [s.text for s in h.take()] == ["Жарт."]


def test_timer_failure_falls_back(make):
    brain = FakeBrain()
    h = make(brain)

    def broken(_msec, _callback):
        raise RuntimeError("no event loop")

    h.composer._schedule = broken
    assert h.composer.compose("idle.joke")
    assert [s.source for s in h.take()] == ["idle.joke"]
    assert brain.calls == []


def test_default_scheduler_runs_on_the_qt_loop(qapp, bus, db, clock):
    import time

    settings = Settings()
    selector = PhraseSelector(BANK, PhraseHistoryRepository(db), lambda: settings, clock, random.Random(1))
    brain = FakeBrain()
    composer = SpeechComposer(bus, selector, brain, lambda: settings, clock, ai_timeout=0.05)
    speech: list[SpeechRequested] = []
    bus.subscribe(SpeechRequested, speech.append)
    composer.compose("idle.joke")
    deadline = time.monotonic() + 2.0
    while not speech and time.monotonic() < deadline:
        qapp.processEvents()
        bus.dispatch_pending()
        time.sleep(0.005)
    assert [s.source for s in speech] == ["idle.joke"]
    composer.close()


# -- ReactionEngine wiring ---------------------------------------------------------------------------


class ReactionHarness(Harness):
    def __init__(self, bus, db, clock, brain=None, *, idle_chatter=True, rng_value=0.0):
        super().__init__(bus, db, clock, brain)

        class Fixed(random.Random):
            def random(self) -> float:
                return rng_value

        self.engine = ReactionEngine(bus, self.selector, lambda: self.settings, clock, Fixed(0),
                                     composer=self.composer, idle_chatter=idle_chatter)
        dnd = self.settings.behavior.do_not_disturb
        self.settings.behavior.do_not_disturb = True
        bus.publish(AppStarted())
        bus.dispatch_pending()
        self.take()
        self.settings.behavior.do_not_disturb = dnd


def test_reaction_lines_go_through_the_composer(bus, db, clock):
    brain = FakeBrain()
    h = ReactionHarness(bus, db, clock, brain)
    bus.publish(GameStarted(game_id="cs2", title="CS2", process_name="cs2.exe"))
    assert h.take() == []
    assert brain.calls[0][0][0] in ("game_start.cs2", "game_start.generic")
    assert brain.calls[0][1] == {"game": "CS2"}
    # Counted at compose time: the budget / gap already see it.
    assert not h.engine.can_speak_unsolicited()
    brain.complete(0, "CS2! Удачі в каткі!")
    assert [s.text for s in h.take()] == ["CS2! Удачі в каткі!"]


def test_interactions_go_through_the_composer_and_quiet_mode_stays_silent(bus, db, clock):
    brain = FakeBrain()
    h = ReactionHarness(bus, db, clock, brain)
    bus.publish(InteractionOutcome(kind=InteractionKind.PET, outcome="happy"))
    h.take()
    assert brain.calls[0][0] == ("interaction.pet.happy",) and brain.calls[0][2] is True
    brain.complete(0, "Мур-мур!")
    [speech] = h.take()
    assert speech.solicited and speech.text == "Мур-мур!"
    h.engine.set_quiet(30)
    h.take()
    bus.publish(InteractionOutcome(kind=InteractionKind.PET, outcome="happy"))
    [gesture] = h.take()
    assert gesture.text == "" and gesture.state == "happy"
    assert len(brain.calls) == 1  # the AI is never asked for a state-only gesture


def test_idle_chatter_can_be_left_to_the_decision_engine(bus, db, clock):
    on = ReactionHarness(bus, db, clock, idle_chatter=True, rng_value=0.0)
    for _ in range(60):
        clock.advance(10)
        on.engine.tick()
    said_on = [s.source for s in on.take() if s.source.startswith("idle.")]
    on.engine.close()
    assert said_on
    off = ReactionHarness(bus, db, clock, idle_chatter=False, rng_value=0.0)
    for _ in range(60):
        clock.advance(10)
        off.engine.tick()
    assert [s for s in off.take() if s.source.startswith("idle.")] == []


def test_note_unsolicited_shares_the_budget(bus, db, clock):
    h = ReactionHarness(bus, db, clock)
    assert h.engine.can_speak_unsolicited()
    h.engine.note_unsolicited()
    assert not h.engine.can_speak_unsolicited()  # min gap
    assert h.engine.can_speak_unsolicited(ignore_gap=True)


def test_ai_phrased_late_night_line_is_not_repeated(bus, db, clock):
    brain = FakeBrain()
    h = ReactionHarness(bus, db, clock, brain, idle_chatter=False)
    clock.set_now(datetime(2026, 3, 10, 2, 0).timestamp())
    h.engine.tick()
    assert [c[0] for c in brain.calls] == [("late_night",)]
    brain.complete(0, "Друга ночі! Може, спати?")
    h.take()
    for _ in range(30):  # 10 more minutes of ticks, well past the minimum gap
        clock.advance(20)
        h.engine.tick()
    assert len(brain.calls) == 1
