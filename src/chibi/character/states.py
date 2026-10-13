"""Character states shared by the state machine, renderers and reactions.

The AI / reaction layers only ever speak in these states; renderers map them
to whatever they draw (sprite clips today, Live2D motions later).  That
mapping is the only thing a new renderer has to provide.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class CharacterState(StrEnum):
    # postures: long-running, the state machine returns to them after actions
    IDLE = "idle"
    WALK = "walk"
    RUN = "run"
    SIT = "sit"
    SLEEP = "sleep"
    GAMING = "gaming"
    # actions / emotions: transient, end after a duration
    TALK = "talk"
    LAUGH = "laugh"
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    SURPRISED = "surprised"
    CURIOUS = "curious"
    THINKING = "thinking"
    WAVE = "wave"
    EAT = "eat"
    SHY = "shy"
    YAWN = "yawn"
    BYE = "bye"
    POINT = "point"
    # Phase 5: livelier actions and emotions (existing frames + procedural effects)
    JUMP = "jump"  # one big hop after another, with squash and stretch
    DANCE = "dance"  # swaying to a beat only she hears
    EXCITED = "excited"  # quick little hops, can barely stand still
    PROUD = "proud"  # content smile, chin up
    STRETCH = "stretch"  # a slow stretch upwards (after a nap, or just because)
    # held while the user drags her with the mouse
    DRAGGED = "dragged"


POSTURES: frozenset[CharacterState] = frozenset(
    {
        CharacterState.IDLE,
        CharacterState.WALK,
        CharacterState.RUN,
        CharacterState.SIT,
        CharacterState.SLEEP,
        CharacterState.GAMING,
    }
)


class Facing(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"


class StatePriority(IntEnum):
    """Who asked for a state.  A request wins if its priority is >= the
    current state's priority (or the current state has ended)."""

    AMBIENT = 0  # autonomous idle fidgets, wandering
    BEHAVIOR = 10  # autonomous decisions with intent (sleep when tired)
    REACTION = 50  # reactions to events (game started, user returned)
    USER = 80  # direct user interaction (click, pet, feed, chat)
    DRAG = 100  # being dragged; nothing interrupts it
