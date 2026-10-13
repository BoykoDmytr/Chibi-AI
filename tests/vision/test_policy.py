from dataclasses import replace

import pytest

from chibi.vision.analysis import Observation
from chibi.vision.policy import PolicyConfig, VisionReactionPolicy


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def obs(event="player_death", confidence=0.9, scene="death_screen", spoiler=False, interesting=True):
    return Observation(scene=scene, event=event, confidence=confidence, interesting=interesting,
                       suggested_emotion="laugh", description="", spoiler=spoiler)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def policy(clock):
    return VisionReactionPolicy(PolicyConfig(), clock)


def test_important_event_needs_a_second_frame(policy, clock):
    first = policy.consider(obs())
    assert first.action == "confirm"
    clock.t += 5
    second = policy.consider(obs())
    assert second.action == "react"
    assert second.observation.confirmed


def test_confirmation_must_come_soon_and_match(policy, clock):
    assert policy.consider(obs()).action == "confirm"
    clock.t += 60
    assert policy.consider(obs()).action == "confirm"  # too late: starts over
    clock.t += 3
    assert policy.consider(obs("victory", scene="victory_screen")).action == "confirm"  # other event
    clock.t += 3
    assert policy.consider(obs()).action == "confirm"


@pytest.mark.parametrize(("o", "reason"), [
    (obs("none", scene="gameplay"), "not notable"),
    (obs("combat", scene="gameplay"), "not notable"),
    (obs(confidence=0.4), "low confidence"),
    (obs(spoiler=True), "spoiler"),
])
def test_ignored_observations(policy, o, reason):
    decision = policy.consider(o)
    assert (decision.action, decision.reason) == ("ignore", reason)


def test_spoilers_allowed_when_the_user_says_so(clock):
    p = VisionReactionPolicy(PolicyConfig(allow_spoilers=True), clock)
    assert p.consider(obs("new_area", scene="gameplay", spoiler=True)).action == "react"


def test_minor_events_react_without_confirmation(policy):
    assert policy.consider(obs("new_area", scene="gameplay")).action == "react"


def confirmed_reaction(policy, clock, event="player_death", scene="death_screen", **kw):
    policy.consider(obs(event, scene=scene, **kw))
    clock.t += 2
    return policy.consider(obs(event, scene=scene, **kw))


def test_event_and_global_cooldowns(policy, clock):
    assert confirmed_reaction(policy, clock).action == "react"
    clock.t += 30
    assert confirmed_reaction(policy, clock).reason == "event cooldown"
    clock.t += 10
    assert policy.consider(obs("new_area", scene="gameplay")).reason == "global cooldown"
    clock.t += 50
    assert policy.consider(obs("new_area", scene="gameplay")).action == "react"


def test_hourly_budget(clock):
    p = VisionReactionPolicy(PolicyConfig(reactions_per_hour=2, global_cooldown=0,
                                          event_cooldowns={}), clock)
    events = ["new_area", "loot", "low_health"]
    results = []
    for e in events:
        clock.t += 1
        results.append(p.consider(obs(e, scene="gameplay", confidence=1.0)).reason)
    assert results == ["ok", "ok", "hourly budget"]
    clock.t += 3601
    assert p.consider(obs("low_health", scene="gameplay", confidence=1.0)).action == "react"


def test_quiet_mode_blocks_reactions(policy, clock):
    policy.consider(obs())
    clock.t += 2
    assert policy.consider(obs(), quiet=True).reason == "quiet"


def test_repeated_deaths_get_less_interesting(clock):
    p = VisionReactionPolicy(PolicyConfig(global_cooldown=0, event_cooldowns={"player_death": 0},
                                          reactions_per_hour=100), clock)
    scores = []
    for _ in range(8):
        clock.t += 10
        decision = confirmed_reaction(p, clock, confidence=0.75, interesting=False)
        scores.append((decision.action, round(decision.score, 2)))
    assert scores[0][0] == "react"
    assert scores[-1][0] == "ignore"  # the 8th death in a row is not news
    assert [s for _, s in scores] == sorted((s for _, s in scores), reverse=True)


def test_new_session_resets_novelty(clock):
    p = VisionReactionPolicy(PolicyConfig(global_cooldown=0, event_cooldowns={"player_death": 0},
                                          reactions_per_hour=100), clock)
    for _ in range(6):
        clock.t += 10
        confirmed_reaction(p, clock, confidence=0.75, interesting=False)
    p.reset_session()
    clock.t += 10
    assert confirmed_reaction(p, clock, confidence=0.75, interesting=False).action == "react"


def test_already_confirmed_observation_skips_confirmation(policy):
    assert policy.consider(replace(obs(), confirmed=True)).action == "react"
