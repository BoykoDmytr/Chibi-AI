"""«Лови печиво»: cookies fall for :data:`GAME_SECONDS`; click them to catch
them, but leave the burnt ones alone.

The physics runs in a fixed logical field (:data:`FIELD_WIDTH` x
:data:`FIELD_HEIGHT`) at a fixed step of 1/60 s: :meth:`CookieCatch.step`
accumulates real time and runs whole steps, so the same seed and the same
sequence of ``step`` / ``click`` calls always give the same game, and
neither a fast nor a slow frame rate changes how the game plays.  Cookies
fall faster and come more often as the game goes on.

Keyboard players get a basket at the bottom (``←`` / ``→``): it appears with
the first arrow key, so mouse players never get points (or burnt cookies)
they did not aim for.

Game time only advances in ``step``: a hidden or paused window simply stops
calling it, and a long hiccup is cut to :data:`MAX_FRAME_SECONDS` so cookies
never teleport past the player.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum

from chibi.games import GameResult, frozen_effects

FIELD_WIDTH = 400.0
FIELD_HEIGHT = 300.0
GAME_SECONDS = 30.0
STEP_SECONDS = 1.0 / 60.0
MAX_FRAME_SECONDS = 0.25

COOKIE_RADIUS = 18.0
#: Clicks this far outside a cookie still catch it (small targets, moving).
CLICK_SLOP = 6.0
FIRST_SPAWN_SECONDS = 0.4
SPAWN_INTERVAL_START = 0.9
SPAWN_INTERVAL_END = 0.4
SPEED_START = 70.0
SPEED_END = 210.0
SPEED_JITTER = 0.15
BURNT_CHANCE = 0.18
BURNT_PENALTY = 2

BASKET_WIDTH = 70.0
BASKET_HEIGHT = 14.0
BASKET_SPEED = 280.0

POPUP_SECONDS = 0.6

#: Verdict thresholds (score).  ~45 cookies fall in a game, ~37 of them good.
GREAT_SCORE = 24
OK_SCORE = 12
MAX_STARS = 15


class Catch(StrEnum):
    CAUGHT = "caught"
    BURNT = "burnt"
    MISS = "miss"
    IGNORED = "ignored"  # the game is over


@dataclass
class Cookie:
    id: int
    x: float
    y: float
    speed: float
    burnt: bool
    #: Rotation in degrees, only for drawing.
    spin: float


@dataclass
class Popup:
    """A short "+1" / "−2" floating where a cookie was caught."""

    x: float
    y: float
    text: str
    good: bool
    age: float = 0.0


class CookieCatch:
    def __init__(self, rng: random.Random, *, duration: float = GAME_SECONDS) -> None:
        if duration <= 0:
            raise ValueError("duration must be positive")
        self._rng = rng
        self.duration = float(duration)
        self.elapsed = 0.0
        self.cookies: list[Cookie] = []
        self.popups: list[Popup] = []
        self.score = 0
        self.caught = 0
        self.burnt_hits = 0
        self.missed = 0
        self.spawned = 0
        self.basket_x = FIELD_WIDTH / 2
        self.basket_active = False
        self._basket_dir = 0
        self._next_id = 1
        self._spawn_timer = FIRST_SPAWN_SECONDS
        self._acc = 0.0

    # -- state ----------------------------------------------------------------------------

    @property
    def finished(self) -> bool:
        return self.elapsed >= self.duration

    @property
    def time_left(self) -> float:
        return max(0.0, self.duration - self.elapsed)

    @property
    def progress(self) -> float:
        """0..1 through the game: drives speed and spawn rate."""
        return min(1.0, self.elapsed / self.duration)

    def fall_speed(self) -> float:
        return SPEED_START + (SPEED_END - SPEED_START) * self.progress

    def spawn_interval(self) -> float:
        return SPAWN_INTERVAL_START + (SPAWN_INTERVAL_END - SPAWN_INTERVAL_START) * self.progress

    # -- input ----------------------------------------------------------------------------

    def click(self, x: float, y: float) -> Catch:
        """A click at field coordinates.  The cookie drawn on top (newest) wins."""
        if self.finished:
            return Catch.IGNORED
        reach = (COOKIE_RADIUS + CLICK_SLOP) ** 2
        for cookie in reversed(self.cookies):
            if (cookie.x - x) ** 2 + (cookie.y - y) ** 2 <= reach:
                self.cookies.remove(cookie)
                return self._take(cookie)
        return Catch.MISS

    def set_basket_direction(self, direction: int) -> None:
        """-1 left, 0 stop, +1 right (arrow keys).  The first move shows the basket."""
        self._basket_dir = (direction > 0) - (direction < 0)
        if self._basket_dir and not self.finished:
            self.basket_active = True

    # -- time -----------------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance by ``dt`` seconds of play (in fixed 1/60 s steps)."""
        if self.finished or not dt > 0:
            return
        self._acc += min(float(dt), MAX_FRAME_SECONDS)
        # The epsilon absorbs float drift: 60 calls of step(1/60) are 60 steps.
        while self._acc >= STEP_SECONDS - 1e-9 and not self.finished:
            self._acc = max(0.0, self._acc - STEP_SECONDS)
            self._tick(STEP_SECONDS)

    def _tick(self, h: float) -> None:
        self.elapsed = min(self.duration, self.elapsed + h)
        self._spawn_timer -= h
        if self._spawn_timer <= 0:
            self._spawn()
            self._spawn_timer += self.spawn_interval()
        if self._basket_dir:
            half = BASKET_WIDTH / 2
            moved = self.basket_x + self._basket_dir * BASKET_SPEED * h
            self.basket_x = min(FIELD_WIDTH - half, max(half, moved))
        basket_top = FIELD_HEIGHT - BASKET_HEIGHT
        for cookie in list(self.cookies):
            previous_bottom = cookie.y + COOKIE_RADIUS
            cookie.y += cookie.speed * h
            bottom = cookie.y + COOKIE_RADIUS
            if (self.basket_active and previous_bottom < basket_top <= bottom
                    and abs(cookie.x - self.basket_x) <= BASKET_WIDTH / 2 + COOKIE_RADIUS / 2):
                self.cookies.remove(cookie)
                self._take(cookie)
            elif cookie.y - COOKIE_RADIUS > FIELD_HEIGHT:
                self.cookies.remove(cookie)
                if not cookie.burnt:
                    self.missed += 1
        for popup in self.popups:
            popup.age += h
        self.popups = [p for p in self.popups if p.age < POPUP_SECONDS]
        if self.finished:  # a clean final frame for the "time is up" screen
            self.cookies.clear()
            self.popups.clear()
            self._basket_dir = 0

    def _spawn(self) -> None:
        rng = self._rng
        x = rng.uniform(COOKIE_RADIUS, FIELD_WIDTH - COOKIE_RADIUS)
        speed = self.fall_speed() * rng.uniform(1.0 - SPEED_JITTER, 1.0 + SPEED_JITTER)
        burnt = rng.random() < BURNT_CHANCE
        spin = rng.uniform(0.0, 360.0)
        self.cookies.append(Cookie(self._next_id, x, -COOKIE_RADIUS, speed, burnt, spin))
        self._next_id += 1
        self.spawned += 1

    def _take(self, cookie: Cookie) -> Catch:
        if cookie.burnt:
            self.burnt_hits += 1
            self.score = max(0, self.score - BURNT_PENALTY)
            self.popups.append(Popup(cookie.x, cookie.y, f"−{BURNT_PENALTY}", good=False))
            return Catch.BURNT
        self.caught += 1
        self.score += 1
        self.popups.append(Popup(cookie.x, cookie.y, "+1", good=True))
        return Catch.CAUGHT

    # -- result ---------------------------------------------------------------------------

    def verdict(self) -> str:
        if self.score >= GREAT_SCORE:
            return "great"
        return "ok" if self.score >= OK_SCORE else "poor"

    def result(self) -> GameResult:
        if not self.finished:
            raise RuntimeError("the game is not over yet")
        score = self.score
        effects = frozen_effects(
            boredom=-(15 + min(15, score // 2)),
            happiness=5 + min(7, score // 4),
            friendship_xp=2 + min(3, score // 10),
            # She eats what was caught.
            hunger=-(2 + min(10, self.caught // 3)),
        )
        return GameResult(game="cookies", score=score, won=score >= OK_SCORE,
                          stars=min(MAX_STARS, 2 + score // 3), verdict=self.verdict(), effects=effects)
