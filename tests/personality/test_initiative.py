"""DecisionEngine: scoring, gates, cooldowns, actions and seeded rates over
simulated hours."""

from __future__ import annotations

import dataclasses
import random
from collections import Counter
from itertools import pairwise
from typing import Any

import pytest

from chibi.core.config import PersonalitySettings, Settings
from chibi.core.events import (
    AppStarted,
    DoNotDisturbChanged,
    InitiativeAction,
    QuietModeChanged,
    SettingsChanged,
    SpeechRequested,
)
from chibi.memory.repositories import PhraseHistoryRepository
from chibi.personality.composer import SpeechComposer
from chibi.personality.initiative import (
    GLOBAL_GAP_SECONDS,
    NEUTRAL_EMOTIONS,
    SPEECH_INTERRUPTION,
    THRESHOLD,
    Candidate,
    DecisionEngine,
    InitiativeContext,
    action_interruption,
    default_candidates,
    initiative_rate,
    score,
    speech_interruption,
)
from chibi.personality.phrases import PhraseBank, PhraseSelector
from chibi.personality.reactions import ReactionEngine, budget_limit

MIN = 60.0
HOUR = 3600.0


class FakeComposer:
    def __init__(self, result: bool = True) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.result = result

    def compose(self, *keys: str, solicited: bool = False, **variables: Any) -> bool:
        assert solicited is False
        self.calls.append((keys, variables))
        return self.result

    @property
    def names(self) -> list[str]:
        return [keys[0] for keys, _vars in self.calls]


class FakeEmotion:
    def __init__(self, **values: float) -> None:
        self.data = {**NEUTRAL_EMOTIONS, **values}

    def values(self) -> dict[str, float]:
        return dict(self.data)


class Harness:
    def __init__(self, bus, clock, *, seed: int = 7, emotion: FakeEmotion | None = None,
                 settings: Settings | None = None, composer=None, start: bool = True, **ctx: Any) -> None:
        self.bus = bus
        self.clock = clock
        self.settings = settings or Settings()
        self.composer = composer or FakeComposer()
        self.emotion = emotion or FakeEmotion()
        self.ctx = InitiativeContext(now=clock.monotonic(), **ctx)
        self.speak_allowed = True
        self.spoken = 0
        self.actions: list[str] = []
        bus.subscribe(InitiativeAction, lambda e: self.actions.append(e.action))
        self.engine = DecisionEngine(bus, self.composer, lambda: self.settings, clock, random.Random(seed),
                                     self.emotion, self.context, lambda: self.speak_allowed,
                                     on_spoken=self._spoken)
        if start:
            self.start()

    def _spoken(self) -> None:
        self.spoken += 1

    def context(self) -> InitiativeContext:
        return dataclasses.replace(self.ctx, now=self.clock.monotonic())

    def set(self, **changes: Any) -> None:
        self.ctx = dataclasses.replace(self.ctx, **changes)

    def start(self) -> None:
        self.bus.publish(AppStarted())
        self.bus.dispatch_pending()

    def run(self, seconds: float, step: float = 10.0) -> list[tuple[float, str]]:
        """Tick for *seconds*; returns (time, what) for every initiative."""
        out: list[tuple[float, str]] = []
        elapsed = 0.0
        while elapsed < seconds:
            self.clock.advance(step)
            elapsed += step
            before_calls, before_actions = len(self.composer.calls), len(self.actions)
            self.engine.tick()
            self.bus.dispatch_pending()
            for keys, _vars in self.composer.calls[before_calls:]:
                out.append((self.clock.monotonic(), keys[0]))
            for action in self.actions[before_actions:]:
                out.append((self.clock.monotonic(), f"action:{action}"))
        return out


@pytest.fixture
def make(bus, clock):
    engines: list[Harness] = []

    def build(**kwargs: Any) -> Harness:
        h = Harness(bus, clock, **kwargs)
        engines.append(h)
        return h

    yield build
    for h in engines:
        h.engine.close()


# -- scoring --------------------------------------------------------------------------------------


def ctx(**kwargs: Any) -> InitiativeContext:
    return InitiativeContext(now=0.0, **kwargs)


def by_name(name: str, personality: PersonalitySettings | None = None) -> Candidate:
    return next(c for c in default_candidates(personality or PersonalitySettings()) if c.name == name)


def test_interruption_costs():
    assert SPEECH_INTERRUPTION == {"gaming": 0.8, "watching_video": 0.7, "programming": 0.5,
                                   "working": 0.5, "browsing": 0.3, "idle": 0.1}
    assert speech_interruption(ctx(activity_category="idle")) == 0.1
    assert speech_interruption(ctx(activity_category="browsing", game_running=True)) == 0.8
    assert speech_interruption(ctx()) == 0.3  # unknown
    assert speech_interruption(ctx(fullscreen=True)) == float("inf")
    assert action_interruption(ctx(activity_category="gaming")) == pytest.approx(0.4)


def test_score_is_priority_times_fit_times_relevance_minus_cost():
    joke = by_name("tell_joke")
    values = dict(NEUTRAL_EMOTIONS)
    idle = score(joke, ctx(activity_category="idle"), values)
    expected = joke.priority * joke.emotion_fit(values) * 1.0 - 0.1
    assert idle == pytest.approx(expected)
    programming = score(joke, ctx(activity_category="programming"), values)
    assert programming == pytest.approx(expected + 0.1 - 0.5)
    night = score(joke, ctx(activity_category="idle", time_of_day="night"), values)
    assert night < idle


def test_fullscreen_and_absorbing_activities_rule_out_chatter():
    values = dict(NEUTRAL_EMOTIONS)
    for name in ("tell_joke", "share_thought", "ask_question", "share_memory", "propose_talk"):
        candidate = by_name(name)
        assert score(candidate, ctx(fullscreen=True, ai_available=True), values) is None
        assert score(candidate, ctx(activity_category="watching_video", ai_available=True), values) is None
        assert score(candidate, ctx(game_running=True, ai_available=True), values) is None


def test_personality_moves_priorities():
    funny = by_name("tell_joke", PersonalitySettings(humor=100))
    dry = by_name("tell_joke", PersonalitySettings(humor=0))
    assert funny.priority > dry.priority
    curious = by_name("ask_question", PersonalitySettings(curiosity=100))
    incurious = by_name("ask_question", PersonalitySettings(curiosity=0))
    assert curious.priority > incurious.priority
    assert initiative_rate(PersonalitySettings(talkativeness=100)) > initiative_rate(
        PersonalitySettings(talkativeness=0))


def test_emotion_fit_shapes_choices():
    happy, sad = dict(NEUTRAL_EMOTIONS, happiness=95), dict(NEUTRAL_EMOTIONS, happiness=10)
    joke = by_name("tell_joke")
    assert score(joke, ctx(), happy) > score(joke, ctx(), sad)
    nap = by_name("sleep_from_boredom")
    assert score(nap, ctx(), dict(NEUTRAL_EMOTIONS, boredom=90, energy=30)) > THRESHOLD
    assert score(nap, ctx(), dict(NEUTRAL_EMOTIONS, boredom=90, energy=60)) < THRESHOLD
    jump = by_name("jump")
    assert score(jump, ctx(), dict(NEUTRAL_EMOTIONS, excitement=95)) > THRESHOLD
    assert score(jump, ctx(), dict(NEUTRAL_EMOTIONS, excitement=50)) < THRESHOLD
    talk = by_name("propose_talk")
    lonely = ctx(minutes_since_last_chat=180)
    assert score(talk, lonely, dict(NEUTRAL_EMOTIONS, affection=90)) > THRESHOLD
    assert score(talk, lonely, dict(NEUTRAL_EMOTIONS, affection=40)) < THRESHOLD


def test_game_and_break_requirements():
    values = dict(NEUTRAL_EMOTIONS, excitement=45)
    ask = by_name("ask_about_game")
    playing = ctx(game_running=True, game_title="CS2", game_minutes=30)
    assert score(ask, playing, values) > THRESHOLD
    for minutes in (5, 200, None):
        assert score(ask, dataclasses.replace(playing, game_minutes=minutes), values) is None
    assert score(ask, dataclasses.replace(playing, game_title=None), values) is None
    assert score(ask, dataclasses.replace(playing, fullscreen=True), values) is None
    rest = by_name("suggest_break")
    assert score(rest, ctx(game_running=True, game_minutes=60), values) is None
    assert score(rest, ctx(game_running=True, game_minutes=130), values) > THRESHOLD
    assert score(rest, ctx(activity_category="programming", activity_minutes=100), values) > THRESHOLD
    assert score(rest, ctx(activity_category="programming", activity_minutes=30), values) is None
    assert score(rest, ctx(activity_category="watching_video", activity_minutes=300), values) is None


def test_wave_needs_a_recent_return_and_share_memory_needs_ai():
    values = dict(NEUTRAL_EMOTIONS)
    wave = by_name("wave")
    assert wave.action == "wave" and not wave.spontaneous
    assert score(wave, ctx(seconds_since_return=20), values) > THRESHOLD
    assert score(wave, ctx(seconds_since_return=600), values) is None
    assert score(wave, ctx(), values) is None
    memory = by_name("share_memory")
    assert score(memory, ctx(ai_available=False), values) is None
    assert score(memory, ctx(ai_available=True), values) > THRESHOLD
    assert memory.keys == ("initiative.share_memory", "idle.thought")


# -- gates ------------------------------------------------------------------------------------------


def test_nothing_before_app_started(make):
    h = make(start=False, activity_category="idle")
    assert h.run(2 * HOUR) == []


def test_first_initiative_waits_a_full_gap_after_start(make):
    h = make(activity_category="idle")
    events = h.run(3 * HOUR)
    assert events
    assert events[0][0] - h.engine._started_at >= GLOBAL_GAP_SECONDS


@pytest.mark.parametrize("blocker", ["fullscreen", "away", "asleep", "quiet", "dnd_event", "dnd_setting"])
def test_never_acts_while_blocked(make, bus, blocker):
    h = make(activity_category="idle", emotion=FakeEmotion(excitement=99, boredom=95, energy=10))
    if blocker == "fullscreen":
        h.set(fullscreen=True)
    elif blocker == "away":
        h.set(user_away=True)
    elif blocker == "asleep":
        h.set(posture="sleep")
    elif blocker == "quiet":
        bus.publish(QuietModeChanged(enabled=True, until=None))
    elif blocker == "dnd_event":
        bus.publish(DoNotDisturbChanged(enabled=True))
    else:
        h.settings.behavior.do_not_disturb = True
    bus.dispatch_pending()
    assert h.run(6 * HOUR) == []
    assert h.spoken == 0


def test_quiet_mode_with_an_end_expires(make, bus, clock):
    h = make(activity_category="idle")
    bus.publish(QuietModeChanged(enabled=True, until=clock.now() + 30 * MIN))
    bus.dispatch_pending()
    assert h.run(30 * MIN) == []
    assert h.run(3 * HOUR) != []


def test_dnd_override_is_reset_by_behavior_settings(make, bus):
    h = make(activity_category="idle")
    bus.publish(DoNotDisturbChanged(enabled=True))
    bus.dispatch_pending()
    assert h.run(HOUR) == []
    bus.publish(SettingsChanged(section="behavior"))
    bus.dispatch_pending()
    assert h.run(3 * HOUR) != []


def test_can_speak_gates_speech_but_not_actions(make):
    h = make(activity_category="idle", emotion=FakeEmotion(excitement=99))
    h.speak_allowed = False
    events = h.run(4 * HOUR)
    assert h.composer.calls == []
    assert {what for _t, what in events} == {"action:jump"}


def test_global_gap_and_candidate_cooldowns(make):
    h = make(activity_category="idle", ai_available=True, emotion=FakeEmotion(excitement=90))
    events = h.run(12 * HOUR)
    assert len(events) > 10
    times = [t for t, _what in events]
    assert all(b - a >= GLOBAL_GAP_SECONDS for a, b in pairwise(times))
    cooldowns = {c.name: c.cooldown for c in default_candidates(PersonalitySettings())}
    keyed = {"idle.joke": "tell_joke", "idle.thought": "share_thought", "idle.question": "ask_question",
             "initiative.share_memory": "share_memory", "action:jump": "jump"}
    for what, name in keyed.items():
        when = [t for t, w in events if w == what]
        assert all(b - a >= cooldowns[name] for a, b in pairwise(when)), what


def test_spoken_lines_are_reported_and_actions_are_not(make):
    h = make(activity_category="idle", emotion=FakeEmotion(excitement=99))
    events = h.run(6 * HOUR)
    lines = [w for _t, w in events if not w.startswith("action:")]
    assert lines and h.spoken == len(lines)
    assert any(w.startswith("action:") for _t, w in events)


def test_a_line_that_could_not_be_said_does_not_hold_the_gap(make):
    h = make(activity_category="idle", composer=FakeComposer(result=False))
    h.run(2 * HOUR)
    assert h.spoken == 0
    assert h.engine._last_initiative == h.engine._started_at


# -- situational candidates --------------------------------------------------------------------------


def test_asks_about_a_windowed_game(make):
    h = make(emotion=FakeEmotion(excitement=45), game_running=True, game_title="osu!", game_minutes=20,
             activity_category="gaming")
    events = h.run(2 * HOUR)
    assert [w for _t, w in events] and all(w == "initiative.ask_game" for _t, w in events)
    keys, variables = h.composer.calls[0]
    assert keys == ("initiative.ask_game",) and variables == {"game": "osu"}


def test_suggests_a_break_with_the_session_length(make):
    h = make(activity_category="programming", activity_minutes=100)
    h.run(2 * HOUR)
    assert ("initiative.suggest_break",) in [k for k, _v in h.composer.calls]
    variables = next(v for k, v in h.composer.calls if k == ("initiative.suggest_break",))
    assert variables == {"duration": "1 годину 40 хвилин"}


def test_proposes_a_talk_only_with_high_affection_after_a_long_silence(make):
    fond = make(activity_category="idle", minutes_since_last_chat=200, emotion=FakeEmotion(affection=95))
    assert "initiative.propose_talk" in [w for _t, w in fond.run(4 * HOUR)]
    recent = make(activity_category="idle", minutes_since_last_chat=30, emotion=FakeEmotion(affection=95))
    assert "initiative.propose_talk" not in [w for _t, w in recent.run(HOUR)]
    cool = make(activity_category="idle", minutes_since_last_chat=500, emotion=FakeEmotion(affection=40))
    assert "initiative.propose_talk" not in [w for _t, w in cool.run(4 * HOUR)]


def test_no_chat_yet_counts_from_the_session_start(make):
    h = make(activity_category="idle", emotion=FakeEmotion(affection=95))
    early = [w for _t, w in h.run(90 * MIN)]
    assert "initiative.propose_talk" not in early
    later = [w for _t, w in h.run(4 * HOUR)]
    assert "initiative.propose_talk" in later


def test_shares_a_memory_only_when_ai_is_available(make):
    offline = make(activity_category="idle", ai_available=False, emotion=FakeEmotion(affection=90, trust=90))
    assert "initiative.share_memory" not in [w for _t, w in offline.run(8 * HOUR)]
    online = make(activity_category="idle", ai_available=True, emotion=FakeEmotion(affection=90, trust=90))
    assert "initiative.share_memory" in [w for _t, w in online.run(8 * HOUR)]


def test_waves_when_the_user_comes_back(make):
    h = make(activity_category="idle")
    h.run(10 * MIN)  # past the start gap; whatever happens, happens
    h.set(user_away=True)
    assert h.run(30 * MIN) == []
    h.set(user_away=False)
    events = h.run(60)
    assert [w for _t, w in events] == ["action:wave"]
    # Only once: the window closes and the cooldown holds.
    h.set(user_away=True)
    h.run(10 * MIN)
    h.set(user_away=False)
    assert "action:wave" not in [w for _t, w in h.run(60)]


def test_naps_from_boredom(make):
    h = make(activity_category="idle", emotion=FakeEmotion(boredom=92, energy=25))
    events = h.run(5 * MIN)
    assert "action:sleep" in [w for _t, w in events]
    h.set(posture="sleep")
    assert h.run(2 * HOUR) == []


def test_jumps_when_excited_but_not_over_a_game(make):
    h = make(activity_category="idle", emotion=FakeEmotion(excitement=97))
    assert "action:jump" in [w for _t, w in h.run(2 * HOUR)]
    g = make(game_running=True, game_title="CS2", game_minutes=200, emotion=FakeEmotion(excitement=97))
    assert "action:jump" not in [w for _t, w in g.run(2 * HOUR)]


# -- rates -------------------------------------------------------------------------------------------


def simulate(bus, clock, hours: float, seed: int, **settings_changes: int) -> list[tuple[float, str]]:
    settings = Settings()
    for name, value in settings_changes.items():
        setattr(settings.personality, name, value)
    h = Harness(bus, clock, seed=seed, settings=settings)
    try:
        return h.run(hours * HOUR)
    finally:
        h.engine.close()


def test_seeded_runs_are_reproducible(bus, clock):
    first = [w for _t, w in simulate(bus, clock, 6, seed=11)]
    second = [w for _t, w in simulate(bus, clock, 6, seed=11)]
    assert first == second
    assert first != [w for _t, w in simulate(bus, clock, 6, seed=12)]


def test_default_rate_over_simulated_hours(bus, clock):
    counts = [len(simulate(bus, clock, 10, seed=seed)) for seed in range(5)]
    per_hour = sum(counts) / (5 * 10)
    # ~2.4 opportunities an hour, thinned by cooldowns: a companion, not a chatterbox.
    assert 1.0 <= per_hour <= 3.5, counts


def test_talkativeness_changes_the_rate(bus, clock):
    quiet = sum(len(simulate(bus, clock, 10, seed=s, talkativeness=0, commentary_frequency=0))
                for s in range(4))
    chatty = sum(len(simulate(bus, clock, 10, seed=s, talkativeness=100, commentary_frequency=100))
                 for s in range(4))
    assert chatty > quiet * 1.5


def test_variety_of_idle_lines(bus, clock):
    lines = Counter(w for s in range(4) for _t, w in simulate(bus, clock, 10, seed=s))
    assert {"idle.joke", "idle.thought", "idle.question"} <= set(lines)
    # No single kind of line crowds out the others.
    assert max(lines.values()) < 0.6 * sum(lines.values())


def test_personality_steers_the_mix(bus, clock):
    funny = Counter(w for s in range(4) for _t, w in simulate(bus, clock, 10, seed=s, humor=100,
                                                               curiosity=10))
    curious = Counter(w for s in range(4) for _t, w in simulate(bus, clock, 10, seed=s, humor=10,
                                                                 curiosity=100))
    assert funny["idle.joke"] > funny["idle.question"]
    assert curious["idle.question"] > curious["idle.joke"]


# -- with the real reaction engine and composer ----------------------------------------------------------


def test_shared_budget_with_the_reaction_engine(bus, db, clock):
    bank = PhraseBank.from_dict({"version": 1, "phrases": {
        key: [{"id": f"{key}.{i}", "text": f"{key} {i}.", "tone": "warm", "state": "talk", "portrait": ""}
              for i in range(4)]
        for key in ("idle.joke", "idle.thought", "idle.question", "greet.morning", "greet.day",
                    "greet.evening", "greet.night")
    }})
    settings = Settings()
    settings.personality.talkativeness = 100
    settings.personality.commentary_frequency = 0
    selector = PhraseSelector(bank, PhraseHistoryRepository(db), lambda: settings, clock, random.Random(1))
    composer = SpeechComposer(bus, selector, None, lambda: settings, clock)
    reactions = ReactionEngine(bus, selector, lambda: settings, clock, random.Random(2), composer=composer,
                               idle_chatter=False)
    context = InitiativeContext(now=0.0, activity_category="idle")
    engine = DecisionEngine(bus, composer, lambda: settings, clock, random.Random(3), FakeEmotion(),
                            lambda: context, reactions.can_speak_unsolicited,
                            on_spoken=reactions.note_unsolicited)
    spoken: list[float] = []
    bus.subscribe(SpeechRequested, lambda e: spoken.append(clock.monotonic()) if not e.solicited else None)
    bus.publish(AppStarted())
    bus.dispatch_pending()
    spoken.clear()  # the greeting
    for _ in range(int(8 * HOUR / 10)):
        clock.advance(10)
        reactions.tick()
        engine.tick()
        bus.dispatch_pending()
    limit = budget_limit(settings)
    assert spoken
    for i, start in enumerate(spoken):
        assert sum(1 for t in spoken[i:] if t - start < HOUR) <= limit
    engine.close()
    reactions.close()
    composer.close()
