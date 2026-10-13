"""Reaction engine: turns what happens (games, apps, the user coming back,
interactions, needs, the clock) into :class:`~chibi.core.events.SpeechRequested`.

The hard part of a desktop companion is not *what* to say but *when to shut
up*, so every unsolicited line passes one gate (:meth:`ReactionEngine.
can_speak_unsolicited`):

* do-not-disturb is off (settings flag, or the latest ``DoNotDisturbChanged``);
* quiet mode ("помовч") is not active - it expires on its own;
* no app is fullscreen (games, films, presentations);
* the rolling one-hour budget is not used up:
  ``round(max_unsolicited_per_hour * (0.4 + 1.2 * commentary_frequency / 100))``;
* the minimum gap since the previous unsolicited line has passed:
  ``min_seconds_between_unsolicited * (1.5 - talkativeness / 100)``, plus a
  short grace after a solicited line so a comment never replaces the answer
  the user is reading.

On top of that every rule has its own cooldown.  Greetings (app start, user
returned) are "high priority": they skip the minimum gap but still respect
do-not-disturb, quiet mode, fullscreen and the budget.  A game start prefers
``game_start.<id>``, then ``game_start.genre.<genre>``, then
``game_start.generic``; one start in four opens with the genre / generic line
so a game played every day does not cycle through its four lines only.

Autonomous chatter (idle jokes, needs hints, late-night care) additionally
waits while she sleeps (if the app tells her via ``sleeping``) and while the
user is away from the keyboard (``UserIdle`` .. ``UserReturned``, only known
when activity monitoring is permitted): nobody would hear it, and it would
waste the budget.

Solicited lines (``InteractionOutcome`` - the user clicked, petted, fed her)
bypass the gate entirely.  In quiet mode they become *state-only* speech
(empty text): she still reacts with a gesture, silently.

Every spoken line goes through a :class:`~chibi.personality.composer.SpeechComposer`:
with an AI brain wired it may be phrased by the model (with a phrase-bank
fallback), without one it is the Phase 1 phrase-bank line, published at
once.  The budget and the cooldowns count a line when it is composed, not
when it is shown, so a line still waiting for the model cannot be doubled.
When the :class:`~chibi.personality.initiative.DecisionEngine` is wired it
takes over idle chatter (``idle_chatter=False``) and reports its own lines
through :meth:`ReactionEngine.note_unsolicited`, so both share one budget.

Nothing self-initiated (needs hints, late-night care, idle chatter) is said
before ``AppStarted``: the needs restored at launch are published before it,
and a "я голодна" must not open the session ahead of the greeting.

Privacy: rules only ever see what the monitor already filtered - private apps
arrive as ``AppInfo.private_app()`` / category ``PRIVATE`` and are ignored.
Given a permission checker, activity events (games, apps, milestones, idle /
return) are only acted on while ``APP_METADATA`` is granted *when they are
dispatched*: the monitor may enqueue one just before the kill switch and the
bus delivers it after.  A ``GameEnded(forgotten=True)`` (the game was made
private while running) is forgotten silently.  Nothing here logs rendered
text, app names or game titles; debug logs name the rule key only.

Main (Qt) thread only: handlers run from the bus dispatch, ``tick`` from a
main-thread timer.
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol

from chibi.core.clock import Clock
from chibi.core.config import PersonalitySettings, Settings
from chibi.core.event_bus import EventBus
from chibi.core.events import (
    ActivityCategory,
    ActivityMilestone,
    AppStarted,
    DoNotDisturbChanged,
    ForegroundAppChanged,
    FullscreenChanged,
    GameEnded,
    GameStarted,
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
from chibi.personality.composer import SpeechComposer
from chibi.personality.phrases import PhraseSelector
from chibi.personality.text_uk import format_clock, format_duration, time_of_day

log = logging.getLogger(__name__)

HOUR = 3600.0
MINUTE = 60.0

#: Rolling window of the unsolicited-speech budget.
BUDGET_WINDOW_SECONDS = HOUR
#: After a solicited line (click, pet...) unsolicited ones wait this long.
SOLICITED_GRACE_SECONDS = 15.0

#: App start: "welcome back" instead of a time-of-day greeting after this long.
#: Longer than a night (23:00 -> 08:00 is a normal morning, not a comeback) and
#: than a working day away from the PC.
RETURN_LONG_AWAY_SECONDS = 20 * HOUR

GAME_COOLDOWN_SECONDS = 20 * MINUTE  # per game id
GAME_GLOBAL_COOLDOWN_SECONDS = 3 * MINUTE  # any game start
#: Probability of opening with the genre / generic line instead of the
#: game-specific one, so a daily game does not cycle through four lines only.
GAME_VARIETY_CHANCE = 0.25
GAME_END_MIN_SECONDS = 3 * MINUTE
GAME_END_SHORT_BELOW = 15 * MINUTE
GAME_END_LONG_ABOVE = 2 * HOUR
GAME_END_COOLDOWN_SECONDS = 3 * MINUTE

MIN_APP_CONFIDENCE = 0.6
#: category -> (phrase key, ActivitySettings toggle, cooldown seconds)
APP_RULES: dict[ActivityCategory, tuple[str, str, float]] = {
    ActivityCategory.PROGRAMMING: ("programming.start", "comment_on_programming", 45 * MINUTE),
    ActivityCategory.WATCHING_VIDEO: ("video.start", "comment_on_video", 60 * MINUTE),
    ActivityCategory.BROWSING: ("browsing.start", "comment_on_browsing", 90 * MINUTE),
}

#: category -> (phrase key, ActivitySettings toggle or None)
MILESTONE_RULES: dict[ActivityCategory, tuple[str, str | None]] = {
    ActivityCategory.PROGRAMMING: ("programming.long", "comment_on_programming"),
    ActivityCategory.GAMING: ("gaming.long", "comment_on_games"),
    ActivityCategory.BROWSING: ("browsing.long", "comment_on_browsing"),
    ActivityCategory.WATCHING_VIDEO: ("video.long", "comment_on_video"),
    ActivityCategory.WORKING: ("working.long", None),
}
MILESTONE_MIN_SECONDS = 60 * MINUTE  # 60-min milestones speak with probability talkativeness/100
MILESTONE_LONG_SECONDS = 120 * MINUTE  # from here on always (budget permitting)
MILESTONE_COOLDOWN_SECONDS = 50 * MINUTE

RETURN_SHORT_BELOW = HOUR
RETURN_LONG_BELOW = 4 * HOUR
RETURN_COOLDOWN_SECONDS = 10 * MINUTE

NEEDS_COOLDOWN_SECONDS = 45 * MINUTE
HUNGRY_ABOVE = 80.0
SLEEPY_BELOW = 15.0
BORED_ABOVE = 85.0

LATE_NIGHT_START_HOUR = 1
LATE_NIGHT_END_HOUR = 5
#: Longer than the 01:00-05:00 window and much shorter than a day: once per night.
LATE_NIGHT_REPEAT_SECONDS = 6 * HOUR

IDLE_COOLDOWN_SECONDS = 5 * MINUTE
#: ``tick`` is expected every ~10 s; the first tick and long stalls are clamped.
DEFAULT_TICK_SECONDS = 10.0
MAX_TICK_SECONDS = 60.0

IDLE_KEYS: tuple[str, ...] = ("idle.joke", "idle.thought", "idle.question")


class PermissionChecker(Protocol):
    def is_granted(self, permission: Permission) -> bool: ...


def speakable_title(title: str) -> str:
    """A game title as it reads inside a sentence.

    Lines put punctuation right after ``{game}`` ("{game}! Не збийся з ритму"),
    so a title that already ends in "!" or "?" ("osu!") would render "osu!!".
    Only trailing marks go, and never the whole title.
    """
    stripped = title.rstrip().rstrip("!?").rstrip()
    return stripped or title


def budget_limit(settings: Settings) -> int:
    """Unsolicited lines allowed per rolling hour."""
    scale = 0.4 + 1.2 * settings.personality.commentary_frequency / 100
    return max(0, round(settings.behavior.max_unsolicited_per_hour * scale))


def min_gap_seconds(settings: Settings) -> float:
    """Minimum seconds between two unsolicited lines."""
    scale = 1.5 - settings.personality.talkativeness / 100
    return settings.behavior.min_seconds_between_unsolicited * scale


def idle_chatter_rates(personality: PersonalitySettings) -> dict[str, float]:
    """Expected idle lines per hour (before the budget), per phrase key.

    Talkativeness scales the overall rate; humor favours jokes, curiosity
    questions.  Defaults (50/70/60) give about three lines an hour.
    """
    base = 0.3 + 2.5 * personality.talkativeness / 100
    return {
        "idle.joke": base * (0.2 + 0.8 * personality.humor / 100),
        "idle.thought": base * 0.5,
        "idle.question": base * (0.2 + 0.8 * personality.curiosity / 100),
    }


class ReactionEngine:
    def __init__(
        self,
        bus: EventBus,
        selector: PhraseSelector,
        settings: Callable[[], Settings],
        clock: Clock,
        rng: random.Random,
        game_genre: Callable[[str], str | None] | None = None,
        sleeping: Callable[[], bool] | None = None,
        *,
        permissions: PermissionChecker | None = None,
        composer: SpeechComposer | None = None,
        idle_chatter: bool = True,
    ) -> None:
        """``game_genre`` maps a game id to its genre (``"shooter"``) for
        ``game_start.genre.<genre>`` lines; without it only game-specific and
        generic lines are used.  ``sleeping`` tells whether she is asleep, so
        she does not wake herself up to tell a joke.  ``permissions`` is
        checked for ``APP_METADATA`` whenever an activity event is dispatched
        (None: trust the events, as in tests).  ``composer`` says the lines
        (None: a phrase-bank-only composer, the Phase 1 behaviour);
        ``idle_chatter=False`` leaves idle jokes / thoughts / questions to the
        decision engine."""
        self._bus = bus
        self._selector = selector
        self._owns_composer = composer is None
        self._composer = composer if composer is not None else SpeechComposer(
            bus, selector, None, settings, clock)
        self._idle_chatter_enabled = idle_chatter
        #: Epoch seconds of the last late-night line composed this session: an AI-phrased
        #: line leaves no phrase-history entry for ``last_used_at`` to find.
        self._late_night_said: float | None = None
        self._settings = settings
        self._clock = clock
        self._rng = rng
        self._game_genre = game_genre
        self._sleeping = sleeping
        self._permissions = permissions
        self._started = False  # AppStarted seen: autonomous lines may follow the greeting

        self._quiet = False
        self._quiet_until: float | None = None  # epoch seconds, for display
        self._quiet_deadline: float | None = None  # monotonic, for expiry
        self._dnd_override: bool | None = None
        self._fullscreen = False
        self._games: dict[str, str] = {}  # running game id -> title
        self._foreground: ActivityCategory | None = None
        self._needs: NeedsChanged | None = None
        self._user_away = False  # between UserIdle and the user's next sign of life

        self._spoken: deque[float] = deque()  # monotonic times of unsolicited lines
        self._last_unsolicited: float | None = None
        self._last_solicited: float | None = None
        self._cooldowns: dict[str, float] = {}  # rule -> monotonic time it last fired
        self._last_tick: float | None = None

        self._unsubscribers: list[Callable[[], None]] = [
            bus.subscribe(AppStarted, self._on_app_started),
            bus.subscribe(GameStarted, self._on_game_started),
            bus.subscribe(GameEnded, self._on_game_ended),
            bus.subscribe(ForegroundAppChanged, self._on_foreground),
            bus.subscribe(ActivityMilestone, self._on_milestone),
            bus.subscribe(UserReturned, self._on_user_returned),
            bus.subscribe(UserIdle, self._on_user_idle),
            bus.subscribe(InteractionOutcome, self._on_interaction),
            bus.subscribe(NeedsChanged, self._on_needs),
            bus.subscribe(FullscreenChanged, self._on_fullscreen),
            bus.subscribe(DoNotDisturbChanged, self._on_dnd),
            bus.subscribe(QuietModeChanged, self._on_quiet_changed),
            bus.subscribe(SettingsChanged, self._on_settings_changed),
            bus.subscribe(PermissionChanged, self._on_permission_changed),
            bus.subscribe(ObservationStopped, self._on_observation_stopped),
        ]

    def close(self) -> None:
        """Unsubscribe from the bus (app shutdown, tests)."""
        if self._owns_composer:
            self._composer.close()
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()

    # -- quiet mode ---------------------------------------------------------------------

    @property
    def quiet_active(self) -> bool:
        """Quiet mode is on (and has not expired yet)."""
        if not self._quiet:
            return False
        return self._quiet_deadline is None or self._clock.monotonic() < self._quiet_deadline

    @property
    def quiet_until(self) -> float | None:
        """Epoch seconds when quiet mode ends; None when it is off *or* on until lifted
        (see :attr:`quiet_active`)."""
        return self._quiet_until if self.quiet_active else None

    def set_quiet(self, minutes: float | None) -> None:
        """Be quiet for ``minutes`` (None or infinity: until :meth:`clear_quiet`).

        Non-positive minutes lift quiet mode instead.  Publishes ``QuietModeChanged``.
        """
        if minutes is not None:
            minutes = float(minutes)
            if math.isnan(minutes):
                raise ValueError("quiet minutes must be a number")
            if minutes <= 0:
                self.clear_quiet()
                return
        if minutes is None or math.isinf(minutes):
            until = deadline = None
        else:
            until = self._clock.now() + minutes * MINUTE
            deadline = self._clock.monotonic() + minutes * MINUTE
        self._quiet, self._quiet_until, self._quiet_deadline = True, until, deadline
        log.info("quiet mode on (%s)", "until lifted" if minutes is None or until is None
                 else f"{minutes:g} min")
        self._bus.publish(QuietModeChanged(enabled=True, until=until))

    def clear_quiet(self) -> None:
        """Lift quiet mode.  Publishes ``QuietModeChanged(False)`` (also when it was off,
        so every view resynchronises)."""
        if self._quiet:
            log.info("quiet mode off")
        self._reset_quiet()
        self._bus.publish(QuietModeChanged(enabled=False))

    def _reset_quiet(self) -> None:
        self._quiet, self._quiet_until, self._quiet_deadline = False, None, None

    def _expire_quiet(self) -> None:
        if self._quiet and not self.quiet_active:
            log.info("quiet mode expired")
            self._reset_quiet()
            self._bus.publish(QuietModeChanged(enabled=False))

    # -- gating ---------------------------------------------------------------------------

    def can_speak_unsolicited(self, *, ignore_gap: bool = False, ignore_fullscreen: bool = False) -> bool:
        """Would an unsolicited line be allowed right now (ignoring per-rule cooldowns)?
        ``ignore_fullscreen`` for screen vision, which exists to comment on (often
        fullscreen) games - voiced only while she is hidden."""
        return self._blocked_reason(self._settings(), ignore_gap=ignore_gap,
                                    ignore_fullscreen=ignore_fullscreen) is None

    def note_unsolicited(self) -> None:
        """Count an unsolicited line said by another component (the decision engine)
        against the shared hourly budget and minimum gap."""
        now = self._clock.monotonic()
        self._spoken.append(now)
        self._last_unsolicited = now

    @property
    def composer(self) -> SpeechComposer:
        """The composer every line goes through (shared with the decision engine)."""
        return self._composer

    def _blocked_reason(self, settings: Settings, *, ignore_gap: bool = False,
                        ignore_fullscreen: bool = False) -> str | None:
        self._expire_quiet()
        if settings.behavior.do_not_disturb or self._dnd_override:
            return "do not disturb"
        if self._quiet:
            return "quiet mode"
        if self._fullscreen and not ignore_fullscreen:
            return "fullscreen"
        now = self._clock.monotonic()
        while self._spoken and now - self._spoken[0] >= BUDGET_WINDOW_SECONDS:
            self._spoken.popleft()
        if len(self._spoken) >= budget_limit(settings):
            return "hourly budget"
        if not ignore_gap:
            last = self._last_unsolicited
            if last is not None and now - last < min_gap_seconds(settings):
                return "min gap"
            if self._last_solicited is not None and now - self._last_solicited < SOLICITED_GRACE_SECONDS:
                return "just answered"
        return None

    def _ready(self, rule: str, cooldown: float) -> bool:
        last = self._cooldowns.get(rule)
        return last is None or self._clock.monotonic() - last >= cooldown

    def _start_cooldown(self, *rules: str) -> None:
        now = self._clock.monotonic()
        for rule in rules:
            self._cooldowns[rule] = now

    def _speak(self, settings: Settings, keys: list[str], *, ignore_gap: bool = False,
               **variables: Any) -> bool:
        """Say an unsolicited line for the first usable key, if the gate allows it."""
        reason = self._blocked_reason(settings, ignore_gap=ignore_gap)
        if reason is not None:
            log.debug("not saying %s: %s", keys[0], reason)
            return False
        if not self._composer.compose(*keys, solicited=False, **variables):
            log.debug("no usable phrase for %s", keys)
            return False
        self.note_unsolicited()
        log.debug("unsolicited line for %s", keys[0])
        return True

    # -- environment ----------------------------------------------------------------------

    def _on_fullscreen(self, event: FullscreenChanged) -> None:
        self._fullscreen = event.active

    def _on_dnd(self, event: DoNotDisturbChanged) -> None:
        self._dnd_override = event.enabled

    def _on_settings_changed(self, event: SettingsChanged) -> None:
        # Persisted settings are authoritative again once behavior settings change.
        if event.section in ("", "behavior"):
            self._dnd_override = None

    def _on_quiet_changed(self, event: QuietModeChanged) -> None:
        # Also receives our own announcements; applying them again is harmless.
        if not event.enabled:
            self._reset_quiet()
            return
        self._quiet = True
        self._quiet_until = event.until
        if event.until is None:
            self._quiet_deadline = None
        else:
            remaining = max(0.0, event.until - self._clock.now())
            self._quiet_deadline = self._clock.monotonic() + remaining

    def _on_permission_changed(self, event: PermissionChanged) -> None:
        if event.permission is Permission.APP_METADATA and not event.granted:
            self._forget_activity()

    def _on_observation_stopped(self, _event: ObservationStopped) -> None:
        self._forget_activity()

    def _forget_activity(self) -> None:
        """Observation stopped: no GameEnded / UserReturned will come for what we know,
        so forget it (otherwise idle chatter would stay suppressed forever)."""
        self._games.clear()
        self._foreground = None
        self._user_away = False

    def _observing(self) -> bool:
        """May activity events be acted on now?  Checked at dispatch time, because an
        event enqueued just before the kill switch is delivered after it, and acting
        on it would keep a game title in memory and idle chatter off for good."""
        if self._permissions is None:
            return True
        try:
            return bool(self._permissions.is_granted(Permission.APP_METADATA))
        except Exception:
            log.exception("permission check failed; ignoring the activity event")
            return False

    def _on_user_idle(self, _event: UserIdle) -> None:
        if not self._observing():
            return
        self._user_away = True

    def _autonomy_paused(self) -> bool:
        """She is asleep, nobody is at the computer, or the app has not greeted yet:
        no self-initiated chatter."""
        if not self._started or self._user_away:
            return True
        if self._sleeping is None:
            return False
        try:
            return bool(self._sleeping())
        except Exception:
            log.exception("sleeping() failed")
            return False

    # -- rules ------------------------------------------------------------------------------

    def _local_now(self) -> datetime:
        return datetime.fromtimestamp(self._clock.now())

    def _on_app_started(self, event: AppStarted) -> None:
        self._started = True
        settings = self._settings()
        greeting = f"greet.{time_of_day(self._local_now())}"
        away = event.away_seconds
        variables: dict[str, Any] = {}
        if away is not None and math.isfinite(away) and away >= 0:
            variables["duration"] = format_duration(away)
        if event.first_run:
            keys = ["greet.first_run", greeting]
        elif away is not None and away > RETURN_LONG_AWAY_SECONDS:
            keys = ["greet.return_long", greeting]
        else:
            keys = [greeting]
        self._speak(settings, keys, ignore_gap=True, **variables)

    def _on_game_started(self, event: GameStarted) -> None:
        if not self._observing():
            return
        self._games[event.game_id] = event.title
        settings = self._settings()
        if not settings.activity.comment_on_games:
            return
        per_game = f"game:{event.game_id}"
        if not self._ready(per_game, GAME_COOLDOWN_SECONDS):
            return
        if not self._ready("game", GAME_GLOBAL_COOLDOWN_SECONDS):
            return
        # A game launch is rare and salient and has its own cooldowns, so it
        # skips the minimum gap (a greeting a minute ago must not swallow it);
        # the hourly budget, DND, quiet mode and fullscreen still apply.
        if self._speak(settings, self._game_start_keys(event.game_id), ignore_gap=True,
                       game=speakable_title(event.title)):
            self._start_cooldown(per_game, "game")

    def _game_start_keys(self, game_id: str) -> list[str]:
        specific = f"game_start.{game_id}"
        genre: str | None = None
        if self._game_genre is not None:
            try:
                genre = self._game_genre(game_id)
            except Exception:
                log.exception("game genre lookup failed")
        fallbacks = [f"game_start.genre.{genre}"] if genre else []
        fallbacks.append("game_start.generic")
        if self._rng.random() < GAME_VARIETY_CHANCE:
            return [*fallbacks, specific]
        return [specific, *fallbacks]

    def _on_game_ended(self, event: GameEnded) -> None:
        # Always forget the game (idle chatter waits while one runs), but say
        # nothing about one the user made private or that ended after revocation.
        self._games.pop(event.game_id, None)
        if event.forgotten or not self._observing():
            return
        duration = event.duration_seconds
        if not math.isfinite(duration) or duration < GAME_END_MIN_SECONDS:
            return
        settings = self._settings()
        if not settings.activity.comment_on_games or not self._ready("game_end", GAME_END_COOLDOWN_SECONDS):
            return
        if duration < GAME_END_SHORT_BELOW:
            key = "game_end.short"
        elif duration > GAME_END_LONG_ABOVE:
            key = "game_end.long"
        else:
            key = "game_end.generic"
        if self._speak(settings, [key, "game_end.generic"], game=speakable_title(event.title),
                       duration=format_duration(duration)):
            self._start_cooldown("game_end")

    def _on_foreground(self, event: ForegroundAppChanged) -> None:
        if not self._observing():
            return
        self._foreground = event.category
        self._user_away = False  # switching apps is a sign of life
        if event.app.private or event.category in (ActivityCategory.PRIVATE, ActivityCategory.UNKNOWN):
            return
        if event.confidence < MIN_APP_CONFIDENCE:
            return
        rule = APP_RULES.get(event.category)
        if rule is None:
            return
        key, toggle, cooldown = rule
        settings = self._settings()
        if not getattr(settings.activity, toggle) or not self._ready(key, cooldown):
            return
        if self._speak(settings, [key], app=event.app.display_name):
            self._start_cooldown(key)

    def _on_milestone(self, event: ActivityMilestone) -> None:
        if not self._observing():
            return
        rule = MILESTONE_RULES.get(event.category)
        if rule is None or event.app.private:
            return
        seconds = event.seconds
        if not math.isfinite(seconds) or seconds < MILESTONE_MIN_SECONDS:
            return
        key, toggle = rule
        settings = self._settings()
        if toggle is not None and not getattr(settings.activity, toggle):
            return
        cooldown = f"milestone:{event.category.value}"
        if not self._ready(cooldown, MILESTONE_COOLDOWN_SECONDS):
            return
        chance = settings.personality.talkativeness / 100
        if seconds < MILESTONE_LONG_SECONDS and self._rng.random() >= chance:
            return
        variables: dict[str, Any] = {"duration": format_duration(seconds), "app": event.app.display_name}
        if event.category is ActivityCategory.GAMING:
            title = event.app.display_name or next(reversed(self._games.values()), None)
            variables["game"] = speakable_title(title) if title else None
        if self._speak(settings, [key], **variables):
            self._start_cooldown(cooldown)

    def _on_user_returned(self, event: UserReturned) -> None:
        if not self._observing():
            return
        self._user_away = False
        away = event.away_seconds
        settings = self._settings()
        if not math.isfinite(away) or away < settings.activity.return_greeting_minutes * MINUTE:
            return
        if not self._ready("user_returned", RETURN_COOLDOWN_SECONDS):
            return
        if away < RETURN_SHORT_BELOW:
            key = "user_returned.short"
        elif away < RETURN_LONG_BELOW:
            key = "user_returned.long"
        else:
            key = "user_returned.very_long"
        if self._speak(settings, [key], ignore_gap=True, duration=format_duration(away)):
            self._start_cooldown("user_returned")

    def _on_interaction(self, event: InteractionOutcome) -> None:
        self._user_away = False  # they just clicked / petted / fed her
        kind = getattr(event.kind, "value", event.kind)
        key = f"interaction.{kind}.{event.outcome}"
        self._expire_quiet()
        if self._quiet:
            # Quiet mode: react with the gesture only, never with words.
            picked = self._selector.pick(key)
            if picked is not None:
                gesture = SpeechRequested(text="", state=picked.phrase.state, solicited=True, source=key)
                self._bus.publish(gesture)
            return
        if not self._composer.compose(key, solicited=True):
            log.debug("no phrase for %s", key)
            return
        self._last_solicited = self._clock.monotonic()

    def _on_needs(self, event: NeedsChanged) -> None:
        self._needs = event
        self._check_needs(self._settings())

    def _check_needs(self, settings: Settings) -> bool:
        needs = self._needs
        if needs is None or self._autonomy_paused():
            return False
        wanted = []
        if needs.hunger > HUNGRY_ABOVE:
            wanted.append("needs.hungry")
        if needs.energy < SLEEPY_BELOW:
            wanted.append("needs.sleepy")
        if needs.boredom > BORED_ABOVE:
            wanted.append("needs.bored")
        for key in wanted:
            if not self._ready(key, NEEDS_COOLDOWN_SECONDS):
                continue
            if self._speak(settings, [key]):
                self._start_cooldown(key)
                return True
            return False  # gated: the others would be gated too
        return False

    # -- periodic -----------------------------------------------------------------------------

    def tick(self) -> None:
        """Called by the app every ~10 s: quiet expiry, late-night care, needs hints, idle chatter."""
        now = self._clock.monotonic()
        dt = DEFAULT_TICK_SECONDS if self._last_tick is None else now - self._last_tick
        self._last_tick = now
        dt = min(max(dt, 0.0), MAX_TICK_SECONDS)
        settings = self._settings()
        self._expire_quiet()
        if self._autonomy_paused():
            return
        if self._late_night(settings):
            return
        if self._check_needs(settings):
            return
        self._idle_chatter(settings, dt)

    def _late_night(self, settings: Settings) -> bool:
        local = self._local_now()
        if not LATE_NIGHT_START_HOUR <= local.hour < LATE_NIGHT_END_HOUR:
            return False
        now = self._clock.now()
        for last in (self._selector.last_used_at("late_night"), self._late_night_said):
            if last is not None and 0 <= now - last < LATE_NIGHT_REPEAT_SECONDS:
                return False
        if not self._speak(settings, ["late_night"], time=format_clock(local)):
            return False
        self._late_night_said = now
        return True

    def _idle_chatter(self, settings: Settings, dt: float) -> bool:
        if not self._idle_chatter_enabled:
            return False  # the decision engine owns autonomous chatter
        if self._fullscreen or self._games or self._foreground is ActivityCategory.GAMING:
            return False
        rates = idle_chatter_rates(settings.personality)
        total = sum(rates.values())
        if total <= 0 or dt <= 0:
            return False
        probability = 1.0 - math.exp(-total * dt / HOUR)
        if self._rng.random() >= probability:
            return False
        if not self._ready("idle", IDLE_COOLDOWN_SECONDS):
            return False
        threshold = self._rng.random() * total
        key = IDLE_KEYS[-1]
        running = 0.0
        for candidate, rate in rates.items():
            running += rate
            if threshold < running:
                key = candidate
                break
        if self._speak(settings, [key]):
            self._start_cooldown("idle")
            return True
        return False
