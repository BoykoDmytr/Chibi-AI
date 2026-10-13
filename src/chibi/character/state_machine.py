"""Which state the character is in, and who may change it.

Everything that wants her to *do* something (autonomous behavior, reactions
to apps, user clicks, the chat, dragging) goes through
:meth:`CharacterStateMachine.request` with a :class:`StatePriority`.  One
arbiter keeps those sources from fighting: a fidget cannot cut off a line
she is saying to the user, and nothing interrupts a drag.

Two kinds of states (see :data:`chibi.character.states.POSTURES`):

* **postures** (idle, walk, run, sit, sleep, gaming) persist until replaced;
* **actions** (talk, laugh, wave, ...) last a duration and then fall back to
  the posture underneath.

Rules:

* a request is accepted if its priority is ``>=`` the active priority, or the
  active action has already expired (``tick`` may simply not have run yet);
* a posture request sets both ``posture`` and ``current`` and takes over the
  request's priority; when an action ends the active priority drops back to
  ``AMBIENT`` so autonomous behavior can move her again.  A posture request
  may pass a ``duration``: the posture still persists, but its priority is
  only held that long (a REACTION wake-up must not freeze autonomous
  behavior until some unrelated action happens to end);
* ``DRAGGED`` is entered only with ``StatePriority.DRAG`` and held until
  :meth:`~CharacterStateMachine.release_drag` (posture becomes ``IDLE``);
* while asleep, requests below ``REACTION`` are rejected; an accepted action
  wakes her (posture -> ``IDLE``);
* an action while walking or running ends the walk (posture -> ``IDLE``): the
  controller stops the movement when it sees the posture change;
* requesting the current state again only refreshes its duration/priority and
  does not notify listeners.

Pure Python and single-threaded (Qt main thread); time comes from
``clock.monotonic()`` so tests drive it with a fake clock.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from chibi.character.states import POSTURES, CharacterState, StatePriority
from chibi.core.clock import Clock

log = logging.getLogger(__name__)

S = CharacterState

#: How long each action lasts when the requester gives no duration.
DEFAULT_ACTION_SECONDS: Mapping[CharacterState, float] = MappingProxyType(
    {
        S.TALK: 3.0,
        S.LAUGH: 2.5,
        S.HAPPY: 3.0,
        S.ANGRY: 3.0,
        S.SAD: 4.0,
        S.SURPRISED: 1.8,
        S.CURIOUS: 3.0,
        S.THINKING: 3.0,
        S.WAVE: 2.4,
        S.EAT: 3.0,
        S.SHY: 3.0,
        S.YAWN: 1.8,
        S.BYE: 2.2,
        S.POINT: 1.6,
        # Phase 5.  Lengths are whole effect cycles of their clips (see the
        # renderer's fx timing), so they end on the ground and upright:
        S.JUMP: 2.5,  # two jumps of 1.25 s
        S.DANCE: 4.4,  # four sway cycles of 1.11 s
        S.EXCITED: 2.5,  # about eight little hops of 0.32 s
        S.PROUD: 3.0,
        S.STRETCH: 3.2,  # one stretch: up 1 s, hold 0.8 s, down 1 s, rest
    }
)

#: Used for an action missing from the table (a state added later).
FALLBACK_ACTION_SECONDS = 3.0
#: Shorter actions would not even show one animation frame.
MIN_ACTION_SECONDS = 0.1

_MOVING = frozenset({S.WALK, S.RUN})


@dataclass(frozen=True)
class StateChange:
    previous: CharacterState
    current: CharacterState
    posture: CharacterState


Listener = Callable[[StateChange], None]


class CharacterStateMachine:
    def __init__(self, clock: Clock, initial: CharacterState = CharacterState.IDLE) -> None:
        if initial not in POSTURES:
            raise ValueError(f"initial state must be a posture, got {initial!r}")
        self._clock = clock
        self._current: CharacterState = initial
        self._posture: CharacterState = initial
        self._priority: StatePriority = StatePriority.AMBIENT
        #: monotonic end of the running action; None for postures and DRAGGED
        self._action_until: float | None = None
        #: monotonic end of a posture's priority hold (posture requests with a duration)
        self._priority_until: float | None = None
        self._listeners: list[Listener] = []
        self._pending: deque[StateChange] = deque()
        self._emitting = False

    # -- state ----------------------------------------------------------------------

    @property
    def current(self) -> CharacterState:
        return self._current

    @property
    def posture(self) -> CharacterState:
        return self._posture

    @property
    def priority(self) -> StatePriority:
        return self._priority

    @property
    def dragged(self) -> bool:
        return self._current is S.DRAGGED

    @property
    def asleep(self) -> bool:
        return self._posture is S.SLEEP

    @property
    def action_active(self) -> bool:
        """An action (not a posture, not a drag) is showing and has not expired."""
        return self._action_until is not None and self._clock.monotonic() < self._action_until

    def action_remaining(self) -> float | None:
        """Seconds until the running action ends, or None when there is none."""
        if self._action_until is None:
            return None
        return max(0.0, self._action_until - self._clock.monotonic())

    # -- listeners ------------------------------------------------------------------

    def add_listener(self, callback: Listener) -> None:
        self._listeners.append(callback)

    def remove_listener(self, callback: Listener) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    # -- transitions ----------------------------------------------------------------

    def request(self, state: CharacterState | str, priority: StatePriority = StatePriority.AMBIENT,
                duration: float | None = None) -> bool:
        """Ask for *state*.  Returns True if accepted (also when it was already current).

        For actions *duration* is how long they last (None means
        :data:`DEFAULT_ACTION_SECONDS`).  Postures persist regardless; for
        them a *duration* only limits how long *priority* is held before it
        drops to ``AMBIENT``.  *state* may be a value string, as carried by
        ``SpeechRequested.state``.
        """
        try:
            state = CharacterState(state)
        except ValueError:
            log.warning("ignoring request for unknown state %r", state)
            return False
        priority = StatePriority(priority)

        if self._current is S.DRAGGED:
            # Nothing interrupts a drag; it ends only through release_drag().
            return state is S.DRAGGED and priority >= StatePriority.DRAG
        if state is S.DRAGGED:
            if priority < StatePriority.DRAG:
                log.debug("rejected dragged with priority %s", priority.name)
                return False
        elif priority > StatePriority.USER:
            # DRAG priority is reserved for the drag: a posture held at DRAG could
            # never be replaced again (nothing else may use DRAG).
            priority = StatePriority.USER

        now = self._clock.monotonic()
        if self._posture is S.SLEEP and priority < StatePriority.REACTION:
            return False
        if priority < self._active_priority(now):
            return False

        previous_current, previous_posture = self._current, self._posture
        self._priority_until = None
        if state in POSTURES:
            self._posture = state
            self._current = state
            self._action_until = None
            if duration is not None and math.isfinite(duration) and priority > StatePriority.AMBIENT:
                self._priority_until = now + max(MIN_ACTION_SECONDS, float(duration))
        else:
            if self._posture is S.SLEEP or self._posture in _MOVING:
                self._posture = S.IDLE
            self._current = state
            self._action_until = None if state is S.DRAGGED else now + self._duration_for(state, duration)
        self._priority = priority

        if (self._current, self._posture) != (previous_current, previous_posture):
            self._emit(StateChange(previous_current, self._current, self._posture))
        return True

    def release_drag(self) -> None:
        """The user let go: she lands in the IDLE posture (the controller handles the fall)."""
        if self._current is not S.DRAGGED:
            return
        self._current = S.IDLE
        self._posture = S.IDLE
        self._priority = StatePriority.AMBIENT
        self._action_until = None
        self._priority_until = None
        self._emit(StateChange(S.DRAGGED, S.IDLE, S.IDLE))

    def tick(self) -> StateChange | None:
        """Expire a finished action (or posture priority hold).  Returns the
        state change it caused, if any."""
        now = self._clock.monotonic()
        if self._priority_until is not None and now >= self._priority_until:
            self._priority_until = None
            self._priority = StatePriority.AMBIENT  # posture stays: no state change
        if self._action_until is None or now < self._action_until:
            return None
        previous = self._current
        self._current = self._posture
        self._priority = StatePriority.AMBIENT
        self._action_until = None
        change = StateChange(previous, self._current, self._posture)
        self._emit(change)
        return change

    # -- internals ------------------------------------------------------------------

    def _active_priority(self, now: float) -> StatePriority:
        """Priority a request has to match.  An expired action or priority hold
        counts as AMBIENT even before ``tick`` has run."""
        if self._action_until is not None and now >= self._action_until:
            return StatePriority.AMBIENT
        if self._priority_until is not None and now >= self._priority_until:
            return StatePriority.AMBIENT
        return self._priority

    @staticmethod
    def _duration_for(state: CharacterState, duration: float | None) -> float:
        if duration is None or not math.isfinite(duration):
            return DEFAULT_ACTION_SECONDS.get(state, FALLBACK_ACTION_SECONDS)
        return max(MIN_ACTION_SECONDS, float(duration))

    def _emit(self, change: StateChange) -> None:
        """Deliver *change* to every listener.

        Listeners may call :meth:`request` themselves; the queue makes such
        nested changes arrive after the current one, in order, at every
        listener.  A failing listener is logged and skipped so one bad
        consumer cannot wedge her animation.
        """
        self._pending.append(change)
        if self._emitting:
            return
        self._emitting = True
        try:
            while self._pending:
                item = self._pending.popleft()
                for listener in list(self._listeners):
                    try:
                        listener(item)
                    except Exception:
                        log.exception("state listener %r failed for %s -> %s",
                                      listener, item.previous.value, item.current.value)
        finally:
            self._emitting = False
