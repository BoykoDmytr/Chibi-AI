import logging
import math
import random
from collections import Counter
from datetime import datetime, timedelta

import pytest

from chibi.core.config import PersonalitySettings, Settings
from chibi.core.events import (
    ActivityCategory,
    ActivityMilestone,
    AppInfo,
    AppStarted,
    DoNotDisturbChanged,
    ForegroundAppChanged,
    FullscreenChanged,
    GameEnded,
    GameStarted,
    InteractionKind,
    InteractionOutcome,
    NeedsChanged,
    ObservationStopped,
    Permission,
    PermissionChanged,
    QuietModeChanged,
    SettingsChanged,
    SpeechRequested,
    UserIdle,
    UserReturned,
)
from chibi.memory.repositories import PhraseHistoryRepository
from chibi.personality.phrases import PhraseBank, PhraseSelector
from chibi.personality.reactions import (
    ReactionEngine,
    budget_limit,
    idle_chatter_rates,
    min_gap_seconds,
)

MIN = 60.0
HOUR = 3600.0

#: Every key the engine uses -> placeholder suffix, so tests can see which
#: variables a rule passes.  Each key gets three phrases.
TEMPLATES = {
    "greet.first_run": "", "greet.morning": "", "greet.day": "", "greet.evening": "", "greet.night": "",
    "greet.return_long": " {duration}",
    "user_returned.short": " {duration}", "user_returned.long": " {duration}",
    "user_returned.very_long": " {duration}",
    "game_start.generic": " {game}", "game_start.cs2": " {game}", "game_start.genre.shooter": " {game}",
    "game_end.short": " {game} {duration}", "game_end.generic": " {game} {duration}",
    "game_end.long": " {game} {duration}",
    "programming.start": " {app}", "video.start": " {app}", "browsing.start": "",
    "programming.long": " {duration}", "gaming.long": " {game} {duration}", "browsing.long": "",
    "video.long": "", "working.long": " {duration}",
    "late_night": " {time}",
    "interaction.click.poke": "", "interaction.click.annoyed": "", "interaction.pet.happy": "",
    "interaction.feed.fed": "",
    "needs.hungry": "", "needs.sleepy": "", "needs.bored": "",
    "idle.joke": "", "idle.thought": "", "idle.question": "",
}
STATES = {"interaction.pet.happy": "happy", "interaction.click.poke": "surprised"}


def build_bank():
    phrases = {
        key: [
            {"id": f"{key}.{i}", "text": f"<{key}>{suffix}", "tone": "warm",
             "state": STATES.get(key, "talk"), "portrait": ""}
            for i in range(3)
        ]
        for key, suffix in TEMPLATES.items()
    }
    return PhraseBank.from_dict({"version": 1, "phrases": phrases})


BANK = build_bank()


class FixedRandom(random.Random):
    """random() always returns ``value``: makes probabilistic branches deterministic."""

    def __init__(self, value: float) -> None:
        super().__init__(0)
        self.value = value

    def random(self) -> float:
        return self.value


def local(hour, minute=0, day=10):
    return datetime(2026, 3, day, hour, minute).timestamp()


class Harness:
    def __init__(self, bus, db, clock, rng=None, genre=None, bank=BANK, settings=None, sleeping=None):
        self.bus, self.db, self.clock = bus, db, clock
        self.settings = settings or Settings()
        self.selector = PhraseSelector(bank, PhraseHistoryRepository(db), lambda: self.settings, clock,
                                       random.Random(3))
        self.engine = ReactionEngine(bus, self.selector, lambda: self.settings, clock,
                                     rng if rng is not None else FixedRandom(0.99), game_genre=genre,
                                     sleeping=sleeping)
        self.speech: list[SpeechRequested] = []
        self.quiet: list[QuietModeChanged] = []
        bus.subscribe(SpeechRequested, self.speech.append)
        bus.subscribe(QuietModeChanged, self.quiet.append)

    def fire(self, *events):
        for event in events:
            self.bus.publish(event)
        self.bus.dispatch_pending()
        return self.take()

    def tick(self):
        self.engine.tick()
        self.bus.dispatch_pending()
        return self.take()

    def take(self):
        out = list(self.speech)
        self.speech.clear()
        return out


def sources(speech):
    return [s.source for s in speech]


@pytest.fixture
def h(bus, db, clock):
    clock.set_now(local(14))
    return Harness(bus, db, clock)


def game(game_id="cs2", title="Counter-Strike 2"):
    return GameStarted(game_id=game_id, title=title, process_name=f"{game_id}.exe")


def app(category, confidence=0.9, name="VS Code", process="code.exe"):
    return ForegroundAppChanged(app=AppInfo(process, name), category=category, confidence=confidence)


def needs(hunger=30.0, energy=80.0, boredom=20.0):
    return NeedsChanged(hunger=hunger, energy=energy, happiness=70.0, boredom=boredom, affection=50.0,
                        friendship_level=1)


# -- formulas ----------------------------------------------------------------------------------


def test_budget_and_gap_formulas():
    s = Settings()
    assert budget_limit(s) == 5  # round(6 * (0.4 + 1.2 * 0.4)) = round(5.28)
    s.personality.commentary_frequency = 0
    assert budget_limit(s) == 2  # round(2.4)
    s.personality.commentary_frequency = 100
    assert budget_limit(s) == 10  # round(9.6)
    s.behavior.max_unsolicited_per_hour = 0
    assert budget_limit(s) == 0
    s = Settings()
    assert min_gap_seconds(s) == pytest.approx(90)
    s.personality.talkativeness = 100
    assert min_gap_seconds(s) == pytest.approx(45)
    s.personality.talkativeness = 0
    assert min_gap_seconds(s) == pytest.approx(135)


def test_idle_rates_at_defaults_are_two_to_four_per_hour():
    total = sum(idle_chatter_rates(PersonalitySettings()).values())
    assert 2.0 <= total <= 4.0
    quiet = sum(idle_chatter_rates(PersonalitySettings(talkativeness=0)).values())
    chatty = sum(idle_chatter_rates(PersonalitySettings(talkativeness=100)).values())
    assert quiet < total < chatty


# -- app start ---------------------------------------------------------------------------------


def test_first_run_greeting(h):
    assert sources(h.fire(AppStarted(first_run=True))) == ["greet.first_run"]


def test_return_after_a_long_absence(h):
    (speech,) = h.fire(AppStarted(away_seconds=9 * HOUR))
    assert speech.source == "greet.return_long"
    assert speech.text == "<greet.return_long> 9 годин"
    assert not speech.solicited


@pytest.mark.parametrize(
    ("hour", "key"),
    [(8, "greet.morning"), (13, "greet.day"), (19, "greet.evening"), (23, "greet.night"), (3, "greet.night")],
)
def test_time_of_day_greeting(h, clock, hour, key):
    clock.set_now(local(hour))
    assert sources(h.fire(AppStarted(away_seconds=2 * HOUR))) == [key]


def test_greeting_ignores_min_gap_but_not_dnd_or_quiet(h, clock):
    h.fire(game())  # an unsolicited line just now
    assert sources(h.fire(AppStarted(away_seconds=HOUR))) == ["greet.day"]
    h.settings.behavior.do_not_disturb = True
    assert h.fire(AppStarted(away_seconds=HOUR)) == []
    h.settings.behavior.do_not_disturb = False
    h.engine.set_quiet(10)
    assert h.fire(AppStarted(away_seconds=HOUR)) == []


# -- gating ------------------------------------------------------------------------------------


def test_do_not_disturb_from_settings(h):
    h.settings.behavior.do_not_disturb = True
    assert not h.engine.can_speak_unsolicited()
    assert h.fire(game()) == []


def test_do_not_disturb_event_and_settings_reset(h):
    h.fire(DoNotDisturbChanged(enabled=True))
    assert not h.engine.can_speak_unsolicited()
    h.fire(DoNotDisturbChanged(enabled=False))
    assert h.engine.can_speak_unsolicited()
    h.fire(DoNotDisturbChanged(enabled=True))
    h.fire(SettingsChanged(section="behavior"))  # persisted settings are authoritative again
    assert h.engine.can_speak_unsolicited()


def test_fullscreen_blocks_unsolicited(h, clock):
    h.fire(FullscreenChanged(active=True))
    assert h.fire(game()) == []
    h.fire(FullscreenChanged(active=False))
    clock.advance(20 * MIN)
    assert sources(h.fire(game())) == ["game_start.cs2"]


def test_quiet_mode_lifecycle(h, clock):
    start = clock.now()
    h.engine.set_quiet(30)
    h.bus.dispatch_pending()
    assert (h.quiet[-1].enabled, h.quiet[-1].until) == (True, start + 30 * MIN)
    assert h.engine.quiet_active and h.engine.quiet_until == start + 30 * MIN
    assert h.fire(game()) == []
    clock.advance(29 * MIN)
    assert h.engine.quiet_active
    clock.advance(2 * MIN)
    assert not h.engine.quiet_active and h.engine.quiet_until is None
    h.tick()
    assert h.quiet[-1].enabled is False  # expiry is announced
    assert sources(h.fire(game("dota2", "Dota 2"))) == ["game_start.generic"]


def test_quiet_until_lifted(h, clock):
    h.engine.set_quiet(None)
    clock.advance(24 * HOUR)
    assert h.engine.quiet_active and h.engine.quiet_until is None
    assert not h.engine.can_speak_unsolicited()
    h.engine.clear_quiet()
    h.bus.dispatch_pending()
    assert not h.engine.quiet_active
    assert h.quiet[-1].enabled is False


def test_set_quiet_zero_lifts_it(h):
    h.engine.set_quiet(10)
    h.engine.set_quiet(0)
    h.bus.dispatch_pending()
    assert not h.engine.quiet_active
    with pytest.raises(ValueError):
        h.engine.set_quiet(math.nan)


def test_quiet_mode_from_another_source(h, clock):
    h.fire(QuietModeChanged(enabled=True, until=clock.now() + 5 * MIN))
    assert h.engine.quiet_active
    assert h.fire(game()) == []
    clock.advance(6 * MIN)
    assert h.engine.can_speak_unsolicited()


def test_hourly_budget(h, clock):
    assert budget_limit(h.settings) == 5
    for i in range(5):
        assert len(h.fire(game(f"g{i}", f"Гра {i}"))) == 1
        clock.advance(3 * MIN)
    assert h.fire(game("g5", "Гра 5")) == []
    clock.advance(60 * MIN - 15 * MIN)  # the first line is now an hour old
    assert len(h.fire(game("g6", "Гра 6"))) == 1


def test_min_gap(h, clock):
    h.settings.behavior.min_seconds_between_unsolicited = 600  # x1.0 at talkativeness 50
    h.settings.activity.comment_on_browsing = True
    assert len(h.fire(app(ActivityCategory.PROGRAMMING))) == 1
    clock.advance(5 * MIN)
    assert h.fire(app(ActivityCategory.WATCHING_VIDEO, name="VLC", process="vlc.exe")) == []
    clock.advance(5 * MIN)
    assert len(h.fire(app(ActivityCategory.BROWSING, name="Firefox", process="firefox.exe"))) == 1


def test_game_start_skips_min_gap_but_not_budget(h, clock):
    # A greeting or comment a minute ago must not swallow "you started a game".
    h.settings.behavior.min_seconds_between_unsolicited = 600
    assert len(h.fire(app(ActivityCategory.PROGRAMMING))) == 1
    clock.advance(1 * MIN)
    assert len(h.fire(game())) == 1
    # ...but the hourly budget still caps it.
    h.settings.behavior.max_unsolicited_per_hour = 1
    clock.advance(30 * MIN)
    assert h.fire(game("dota2", "Dota 2")) == []


def test_unsolicited_waits_after_an_answer(h, clock):
    h.fire(InteractionOutcome(kind=InteractionKind.CLICK, outcome="poke"))
    clock.advance(5)
    assert h.fire(app(ActivityCategory.PROGRAMMING)) == []
    clock.advance(15)
    assert len(h.fire(app(ActivityCategory.PROGRAMMING))) == 1


def test_game_cooldowns(h, clock):
    h.settings.behavior.min_seconds_between_unsolicited = 20
    h.settings.personality.talkativeness = 100
    assert len(h.fire(game("cs2"))) == 1
    clock.advance(1 * MIN)
    assert h.fire(game("dota2", "Dota 2")) == []  # global 3-minute cooldown
    clock.advance(3 * MIN)
    assert len(h.fire(game("dota2", "Dota 2"))) == 1
    clock.advance(10 * MIN)
    assert h.fire(game("cs2")) == []  # same game within 20 minutes
    clock.advance(10 * MIN)
    assert len(h.fire(game("cs2"))) == 1


# -- solicited ---------------------------------------------------------------------------------


def test_interactions_are_solicited_and_bypass_the_gate(h, clock):
    h.settings.behavior.do_not_disturb = True
    h.fire(FullscreenChanged(active=True))
    h.settings.behavior.max_unsolicited_per_hour = 0
    (speech,) = h.fire(InteractionOutcome(kind=InteractionKind.PET, outcome="happy"))
    assert speech.solicited and speech.source == "interaction.pet.happy"
    assert speech.text == "<interaction.pet.happy>" and speech.state == "happy"
    clock.advance(1)
    assert len(h.fire(InteractionOutcome(kind=InteractionKind.CLICK, outcome="annoyed"))) == 1


def test_quiet_mode_turns_interactions_into_gestures(h, db):
    h.engine.set_quiet(30)
    (speech,) = h.fire(InteractionOutcome(kind=InteractionKind.PET, outcome="happy"))
    assert speech.text == "" and speech.state == "happy" and speech.solicited
    assert speech.source == "interaction.pet.happy"
    assert PhraseHistoryRepository(db).recent() == []  # a gesture is not "said"


def test_unknown_interaction_outcome_is_silent(h):
    assert h.fire(InteractionOutcome(kind=InteractionKind.CALL, outcome="ok")) == []


# -- games -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("game_id", "variety", "genre", "key"),
    [
        ("cs2", 0.99, "shooter", "game_start.cs2"),
        ("cs2", 0.10, "shooter", "game_start.genre.shooter"),  # occasional variety
        ("apex", 0.99, "shooter", "game_start.genre.shooter"),  # no game-specific lines
        ("apex", 0.99, None, "game_start.generic"),
        ("custom:foo.exe", 0.99, "action", "game_start.generic"),  # genre without lines
    ],
)
def test_game_start_key_choice(bus, db, clock, game_id, variety, genre, key):
    clock.set_now(local(14))
    h = Harness(bus, db, clock, rng=FixedRandom(variety), genre=lambda _gid: genre)
    (speech,) = h.fire(game(game_id, "Моя гра"))
    assert speech.source == key
    assert speech.text == f"<{key}> Моя гра"


def test_game_genre_lookup_failure_falls_back(bus, db, clock):
    def broken(_gid):
        raise RuntimeError("db gone")

    h = Harness(bus, db, clock, genre=broken)
    assert sources(h.fire(game("apex", "Apex"))) == ["game_start.generic"]


def test_no_game_comments_when_disabled(h):
    h.settings.activity.comment_on_games = False
    assert h.fire(game()) == []
    assert h.fire(GameEnded(game_id="cs2", title="CS2", duration_seconds=HOUR)) == []


@pytest.mark.parametrize(
    ("duration", "key", "text"),
    [
        (2 * MIN, None, None),
        (10 * MIN, "game_end.short", "<game_end.short> CS2 10 хвилин"),
        (HOUR, "game_end.generic", "<game_end.generic> CS2 1 годину"),
        (3 * HOUR, "game_end.long", "<game_end.long> CS2 3 години"),
    ],
)
def test_game_end(h, duration, key, text):
    speech = h.fire(GameEnded(game_id="cs2", title="CS2", duration_seconds=duration))
    if key is None:
        assert speech == []
    else:
        assert [(s.source, s.text) for s in speech] == [(key, text)]


# -- foreground apps ---------------------------------------------------------------------------


def test_programming_start_and_cooldown(h, clock):
    (speech,) = h.fire(app(ActivityCategory.PROGRAMMING))
    assert (speech.source, speech.text) == ("programming.start", "<programming.start> VS Code")
    clock.advance(30 * MIN)
    assert h.fire(app(ActivityCategory.PROGRAMMING)) == []
    clock.advance(16 * MIN)
    assert len(h.fire(app(ActivityCategory.PROGRAMMING))) == 1


def test_video_and_browsing(h, clock):
    assert sources(h.fire(app(ActivityCategory.WATCHING_VIDEO, name="YouTube"))) == ["video.start"]
    clock.advance(5 * MIN)
    assert h.fire(app(ActivityCategory.BROWSING)) == []  # comment_on_browsing is off by default
    h.settings.activity.comment_on_browsing = True
    assert sources(h.fire(app(ActivityCategory.BROWSING))) == ["browsing.start"]


@pytest.mark.parametrize(
    "event",
    [
        app(ActivityCategory.PROGRAMMING, confidence=0.5),
        ForegroundAppChanged(app=AppInfo.private_app(), category=ActivityCategory.PRIVATE, confidence=1),
        ForegroundAppChanged(app=AppInfo.private_app(), category=ActivityCategory.PROGRAMMING, confidence=1),
        app(ActivityCategory.UNKNOWN, confidence=1.0),
        app(ActivityCategory.WORKING),
        app(ActivityCategory.GAMING),
    ],
)
def test_no_comment_for_uncertain_private_or_other_apps(h, event):
    assert h.fire(event) == []


def test_programming_toggle(h):
    h.settings.activity.comment_on_programming = False
    assert h.fire(app(ActivityCategory.PROGRAMMING)) == []


# -- milestones --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("category", "key"),
    [
        (ActivityCategory.PROGRAMMING, "programming.long"),
        (ActivityCategory.GAMING, "gaming.long"),
        (ActivityCategory.WATCHING_VIDEO, "video.long"),
        (ActivityCategory.WORKING, "working.long"),
    ],
)
def test_two_hour_milestones(h, category, key):
    event = ActivityMilestone(category=category, app=AppInfo("x.exe", "Штука"), seconds=2 * HOUR)
    assert sources(h.fire(event)) == [key]


def milestone(category=ActivityCategory.PROGRAMMING, seconds=2 * HOUR, app_info=None):
    return ActivityMilestone(category=category, app=app_info or AppInfo("c.exe", "C"), seconds=seconds)


def test_milestone_passes_duration(h):
    event = milestone(ActivityCategory.GAMING, 3 * HOUR, AppInfo("cs2.exe", "CS2"))
    assert h.fire(event)[0].text == "<gaming.long> CS2 3 години"


def test_browsing_milestone_respects_toggle(h):
    event = ActivityMilestone(category=ActivityCategory.BROWSING, app=AppInfo("x.exe", "X"), seconds=2 * HOUR)
    assert h.fire(event) == []
    h.settings.activity.comment_on_browsing = True
    assert sources(h.fire(event)) == ["browsing.long"]


@pytest.mark.parametrize(("talkativeness", "draw", "speaks"), [(0, 0.0, False), (100, 0.99, True),
                                                                (50, 0.49, True), (50, 0.5, False)])
def test_one_hour_milestone_depends_on_talkativeness(bus, db, clock, talkativeness, draw, speaks):
    clock.set_now(local(14))
    h = Harness(bus, db, clock, rng=FixedRandom(draw))
    h.settings.personality.talkativeness = talkativeness
    event = ActivityMilestone(category=ActivityCategory.PROGRAMMING, app=AppInfo("c.exe", "C"), seconds=HOUR)
    assert bool(h.fire(event)) is speaks


def test_ignored_milestones(h, clock):
    short = milestone(seconds=30 * MIN)
    private = milestone(app_info=AppInfo.private_app())
    idle = milestone(ActivityCategory.IDLE)
    assert h.fire(short, private, idle) == []


def test_milestone_cooldown(h, clock):
    event = milestone()
    assert len(h.fire(event)) == 1
    clock.advance(10 * MIN)
    assert h.fire(event) == []


# -- user returned -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("away", "key"),
    [
        (5 * MIN, None),  # below return_greeting_minutes (10)
        (20 * MIN, "user_returned.short"),
        (2 * HOUR, "user_returned.long"),
        (5 * HOUR, "user_returned.very_long"),
    ],
)
def test_user_returned(h, away, key):
    speech = h.fire(UserReturned(away_seconds=away))
    assert sources(speech) == ([key] if key else [])


def test_user_returned_skips_gap_but_is_budgeted(h, clock):
    h.fire(game())  # just spoke
    (speech,) = h.fire(UserReturned(away_seconds=20 * MIN))
    assert speech.text == "<user_returned.short> 20 хвилин"
    h.settings.behavior.max_unsolicited_per_hour = 0
    clock.advance(30 * MIN)
    assert h.fire(UserReturned(away_seconds=20 * MIN)) == []


def test_return_threshold_follows_settings(h):
    h.settings.activity.return_greeting_minutes = 30
    assert h.fire(UserReturned(away_seconds=20 * MIN)) == []


# -- needs -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("event", "key"),
    [
        (needs(hunger=90), "needs.hungry"),
        (needs(energy=10), "needs.sleepy"),
        (needs(boredom=90), "needs.bored"),
        (needs(), None),
        (needs(hunger=80, energy=15, boredom=85), None),  # thresholds are strict
    ],
)
def test_needs_hints(h, event, key):
    assert sources(h.fire(event)) == ([key] if key else [])


def test_needs_cooldown_and_retry_on_tick(h, clock):
    h.fire(FullscreenChanged(active=True))
    assert h.fire(needs(hunger=90)) == []  # gated
    h.fire(FullscreenChanged(active=False))
    assert sources(h.tick()) == ["needs.hungry"]  # retried from the last snapshot
    clock.advance(10 * MIN)
    assert h.fire(needs(hunger=95)) == []  # 45-minute cooldown
    assert h.tick() == []
    clock.advance(40 * MIN)
    assert sources(h.tick()) == ["needs.hungry"]


# -- late night --------------------------------------------------------------------------------


def test_late_night_once_per_night(h, clock):
    clock.set_now(local(2, 5))
    (speech,) = h.tick()
    assert (speech.source, speech.text) == ("late_night", "<late_night> 02:05")
    for _ in range(12):
        clock.advance(10 * MIN)
        assert "late_night" not in sources(h.tick())
    clock.set_now(local(2, 5, day=11))
    assert sources(h.tick()) == ["late_night"]


def test_late_night_survives_restart(bus, db, clock):
    clock.set_now(local(1, 30))
    first = Harness(bus, db, clock)
    assert sources(first.tick()) == ["late_night"]
    first.engine.close()
    clock.advance(30 * MIN)
    second = Harness(bus, db, clock)  # new selector + engine over the same database
    assert second.tick() == []


@pytest.mark.parametrize("hour", [0, 5, 12, 23])
def test_no_late_night_outside_1_to_5(h, clock, hour):
    clock.set_now(local(hour, 30))
    assert "late_night" not in sources(h.tick())


def test_late_night_waits_for_the_gate(h, clock):
    clock.set_now(local(3))
    h.fire(FullscreenChanged(active=True))
    assert h.tick() == []
    h.fire(FullscreenChanged(active=False))
    clock.advance(10)
    assert sources(h.tick()) == ["late_night"]


# -- idle chatter ------------------------------------------------------------------------------


def simulate(h, hours, step=10.0):
    said = []
    for _ in range(int(hours * HOUR / step)):
        h.clock.advance(step)
        said.extend(h.tick())
    return said


def idle_count(speech):
    return sum(1 for s in speech if s.source.startswith("idle."))


def test_idle_chatter_rate_at_defaults(bus, db, clock):
    clock.set_now(local(6))  # 06:00 -> 24:00, no late-night care in between
    h = Harness(bus, db, clock, rng=random.Random(1))
    said = simulate(h, 18)
    assert all(s.source.startswith("idle.") and not s.solicited for s in said)
    assert 2.0 <= idle_count(said) / 18 <= 4.0  # roughly 2-4 lines per hour
    assert set(sources(said)) == {"idle.joke", "idle.thought", "idle.question"}


def test_idle_chatter_follows_personality(bus, db, clock):
    def run(**personality):
        clock.set_now(local(8))
        for name, value in personality.items():
            setattr(harness.settings.personality, name, value)
        return simulate(harness, 10)

    harness = Harness(bus, db, clock, rng=random.Random(7))
    quiet = run(talkativeness=0)
    harness.engine.close()
    harness = Harness(bus, db, clock, rng=random.Random(7))
    funny = run(talkativeness=100, humor=100, curiosity=0)
    assert idle_count(quiet) < 15
    assert idle_count(funny) > idle_count(quiet)
    assert idle_count(funny) <= 50  # the hourly budget (5) caps it
    counts = Counter(sources(funny))
    assert counts["idle.joke"] > counts["idle.question"]


def test_idle_chatter_respects_the_budget(bus, db, clock):
    clock.set_now(local(8))
    h = Harness(bus, db, clock, rng=random.Random(1))
    h.settings.personality.talkativeness = 100
    h.settings.behavior.max_unsolicited_per_hour = 1
    said = simulate(h, 5)
    assert idle_count(said) <= 5


def test_no_idle_chatter_while_fullscreen_or_gaming(bus, db, clock):
    clock.set_now(local(8))
    h = Harness(bus, db, clock, rng=random.Random(11))
    h.fire(FullscreenChanged(active=True))
    assert idle_count(simulate(h, 3)) == 0
    h.fire(FullscreenChanged(active=False))
    h.settings.activity.comment_on_games = False
    h.fire(game())
    assert idle_count(simulate(h, 3)) == 0
    h.fire(GameEnded(game_id="cs2", title="CS2", duration_seconds=3 * HOUR))
    assert idle_count(simulate(h, 3)) > 0


def test_no_idle_chatter_while_a_game_is_in_front(bus, db, clock):
    clock.set_now(local(8))
    h = Harness(bus, db, clock, rng=random.Random(11))
    h.fire(app(ActivityCategory.GAMING))
    assert idle_count(simulate(h, 3)) == 0


@pytest.mark.parametrize(
    "stop",
    [ObservationStopped(), PermissionChanged(permission=Permission.APP_METADATA, granted=False)],
)
def test_stopping_observation_forgets_running_games(bus, db, clock, stop):
    clock.set_now(local(8))
    h = Harness(bus, db, clock, rng=random.Random(11))
    h.settings.activity.comment_on_games = False
    h.fire(game(), app(ActivityCategory.GAMING))
    h.fire(stop)  # no GameEnded will ever come for it
    assert idle_count(simulate(h, 3)) > 0


def test_no_autonomous_chatter_while_she_sleeps(bus, db, clock):
    asleep = [True]
    clock.set_now(local(8))
    h = Harness(bus, db, clock, rng=random.Random(11), sleeping=lambda: asleep[0])
    assert h.fire(needs(hunger=95)) == []
    assert idle_count(simulate(h, 3)) == 0
    clock.set_now(local(2, day=11))
    assert h.tick() == []  # no late-night care either
    assert h.fire(InteractionOutcome(kind=InteractionKind.CLICK, outcome="poke"))  # solicited still works
    asleep[0] = False
    clock.set_now(local(8, day=11))
    said = simulate(h, 3)
    assert "needs.hungry" in sources(said)
    assert idle_count(said) > 0


def test_sleeping_callback_failure_counts_as_awake(bus, db, clock, caplog):
    def broken():
        raise RuntimeError("no tamagotchi")

    clock.set_now(local(8))
    h = Harness(bus, db, clock, sleeping=broken)
    assert sources(h.fire(needs(hunger=95))) == ["needs.hungry"]
    assert "sleeping() failed" in caplog.text


def test_no_autonomous_chatter_while_the_user_is_away(bus, db, clock):
    clock.set_now(local(8))
    h = Harness(bus, db, clock, rng=random.Random(11))
    h.fire(UserIdle(idle_seconds=10 * MIN))
    assert h.fire(needs(hunger=95)) == []
    assert idle_count(simulate(h, 3)) == 0
    assert sources(h.fire(UserReturned(away_seconds=3 * HOUR))) == ["user_returned.long"]
    assert idle_count(simulate(h, 3)) > 0


@pytest.mark.parametrize(
    "sign_of_life",
    [
        app(ActivityCategory.WORKING),
        InteractionOutcome(kind=InteractionKind.PET, outcome="happy"),
        ObservationStopped(),
        PermissionChanged(permission=Permission.APP_METADATA, granted=False),
    ],
)
def test_user_away_ends_with_any_sign_of_life(bus, db, clock, sign_of_life):
    clock.set_now(local(8))
    h = Harness(bus, db, clock)
    h.fire(UserIdle(idle_seconds=10 * MIN), sign_of_life)
    clock.advance(MIN)
    assert sources(h.fire(needs(hunger=95))) == ["needs.hungry"]


def test_logs_never_contain_titles_or_app_names(h, clock, caplog):
    with caplog.at_level(logging.DEBUG, logger="chibi"):
        h.fire(game("cs2", "Таємна Гра"))
        h.fire(app(ActivityCategory.PROGRAMMING, name="СекретнийРедактор"))
        clock.advance(10 * MIN)
        h.fire(GameEnded(game_id="cs2", title="Таємна Гра", duration_seconds=HOUR))
        h.fire(milestone(app_info=AppInfo("x.exe", "СекретнийРедактор")))
    assert "Таємна" not in caplog.text and "Секретний" not in caplog.text
    assert any("game_start" in r.getMessage() for r in caplog.records)  # keys are logged


def test_close_unsubscribes(h):
    h.engine.close()
    assert h.fire(game(), AppStarted(first_run=True)) == []


# -- with the real phrase bank -----------------------------------------------------------------


def test_real_bank_lines_are_fully_rendered(bus, db, clock):
    clock.set_now(local(1, 30))
    h = Harness(bus, db, clock, bank=PhraseBank.load(), genre=lambda _gid: "shooter")
    h.settings.behavior.max_unsolicited_per_hour = 30
    h.settings.behavior.min_seconds_between_unsolicited = 20
    h.settings.personality.commentary_frequency = 100
    events = [
        AppStarted(away_seconds=10 * HOUR),
        game("stalker2", "S.T.A.L.K.E.R. 2"),
        GameEnded(game_id="stalker2", title="S.T.A.L.K.E.R. 2", duration_seconds=3 * HOUR),
        app(ActivityCategory.PROGRAMMING),
        milestone(app_info=AppInfo("code.exe", "VS Code")),
        UserReturned(away_seconds=5 * HOUR),
        needs(hunger=95),
        InteractionOutcome(kind=InteractionKind.FEED, outcome="starving_fed"),
        InteractionOutcome(kind=InteractionKind.DRAG_START, outcome="ok"),
    ]
    said = []
    for event in events:
        said.extend(h.fire(event))
        clock.advance(4 * MIN)
    clock.set_now(local(2, 30))
    said.extend(h.tick())
    assert len(said) == len(events) + 1
    for speech in said:
        assert speech.text and "{" not in speech.text and "}" not in speech.text, speech.source
    assert "late_night" in sources(said)


def test_tick_tolerates_long_gaps(h, clock):
    h.tick()
    clock.advance(timedelta(days=2).total_seconds())
    h.tick()  # no exception, dt is clamped
