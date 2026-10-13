"""When does something she *saw* deserve a comment?

The vision model's word is never enough on its own (its confidence is a
guess, not a probability).  A comment needs, in order:

1. a notable event (death, victory, boss, achievement...), confidence above
   ``min_confidence``, and no spoiler risk unless the user allows spoilers;
2. for important events, the same event in a second frame shortly after
   (``confirm_within``) - the pipeline captures that frame right away;
3. no reaction to the same kind of event within its cooldown, none at all
   within the global cooldown, and budget left for this hour;
4. no quiet mode / do-not-disturb;
5. a relevance score above the threshold: event weight x confidence x
   novelty (the 10th death in a session is less worth a joke than the 1st).

The policy is pure logic with an injected clock; :class:`VisionService`
asks it about every observation and does what it says.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from chibi.vision.analysis import Observation

EVENT_WEIGHTS: dict[str, float] = {
    "player_death": 1.0, "victory": 1.0, "boss_fight": 0.9, "achievement": 0.85,
    "level_up": 0.75, "new_area": 0.6, "low_health": 0.55, "loot": 0.45,
}
EVENT_COOLDOWNS: dict[str, float] = {
    "player_death": 90, "victory": 120, "boss_fight": 300, "achievement": 120,
    "level_up": 120, "new_area": 240, "low_health": 180, "loot": 300,
}


@dataclass
class PolicyConfig:
    min_confidence: float = 0.6
    confirm_within: float = 20.0
    global_cooldown: float = 45.0
    reactions_per_hour: int = 6
    relevance_threshold: float = 0.4
    allow_spoilers: bool = False
    event_cooldowns: dict[str, float] = field(default_factory=lambda: dict(EVENT_COOLDOWNS))


@dataclass(frozen=True, slots=True)
class Decision:
    action: str  # "react" | "confirm" | "ignore"
    reason: str
    score: float = 0.0
    observation: Observation | None = None


class VisionReactionPolicy:
    def __init__(self, config: PolicyConfig | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config or PolicyConfig()
        self._clock = clock
        self._pending: tuple[str, float] | None = None  # (event, first seen)
        self._last_by_event: dict[str, float] = {}
        self._last_reaction = float("-inf")
        self._reactions: deque[float] = deque()
        self._seen_counts: dict[str, int] = {}

    def reset_session(self) -> None:
        """A new game session: novelty starts over, pending confirmation dropped."""
        self._pending = None
        self._seen_counts.clear()

    def consider(self, obs: Observation, *, quiet: bool = False) -> Decision:
        c = self.config
        now = self._clock()
        if not obs.notable:
            return Decision("ignore", "not notable", observation=obs)
        if obs.confidence < c.min_confidence:
            return Decision("ignore", "low confidence", observation=obs)
        if obs.spoiler and not c.allow_spoilers:
            return Decision("ignore", "spoiler", observation=obs)

        if obs.needs_confirmation and not obs.confirmed:
            pending = self._pending
            if pending is not None and pending[0] == obs.event and now - pending[1] <= c.confirm_within:
                self._pending = None
                obs = _confirmed(obs)
            else:
                self._pending = (obs.event, now)
                return Decision("confirm", "needs a second frame", observation=obs)

        # Count every confirmed sighting, reacted to or not: novelty is about
        # what happened in the game, not about what she said.
        count = self._seen_counts.get(obs.event, 0)
        self._seen_counts[obs.event] = count + 1

        last = self._last_by_event.get(obs.event)
        if last is not None and now - last < c.event_cooldowns.get(obs.event, 120.0):
            return Decision("ignore", "event cooldown", observation=obs)
        if now - self._last_reaction < c.global_cooldown:
            return Decision("ignore", "global cooldown", observation=obs)
        while self._reactions and now - self._reactions[0] > 3600:
            self._reactions.popleft()
        if len(self._reactions) >= c.reactions_per_hour:
            return Decision("ignore", "hourly budget", observation=obs)
        if quiet:
            return Decision("ignore", "quiet", observation=obs)

        novelty = 1.0 / (1.0 + 0.35 * count)
        score = EVENT_WEIGHTS.get(obs.event, 0.3) * obs.confidence * novelty
        if obs.interesting:
            score *= 1.15
        if score < c.relevance_threshold:
            return Decision("ignore", "low relevance", score, observation=obs)

        self._last_by_event[obs.event] = now
        self._last_reaction = now
        self._reactions.append(now)
        return Decision("react", "ok", score, observation=obs)


def _confirmed(obs: Observation) -> Observation:
    from dataclasses import replace

    return replace(obs, confirmed=True)
