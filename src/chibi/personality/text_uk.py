"""Small Ukrainian text helpers: plural forms, durations, time of day,
phrase rendering and normalisation of chat input.

Durations are produced in the *accusative* ("1 годину", "2 години
15 хвилин"), because every phrase that uses ``{duration}`` reads it after
"вже", "цілих", "тебе не було ..." - places where Ukrainian wants the
accusative of time.  For "2..4" and "5+" the accusative equals the
nominative, so the same strings also work as a bare exclamation.

Nothing here touches settings, the clock or the event bus, so it is trivially
testable and safe to call from anywhere.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

#: (one, few, many) - accusative, as used after "вже" / "цілих".
MINUTE_FORMS: tuple[str, str, str] = ("хвилину", "хвилини", "хвилин")
HOUR_FORMS: tuple[str, str, str] = ("годину", "години", "годин")
DAY_FORMS: tuple[str, str, str] = ("день", "дні", "днів")

LESS_THAN_A_MINUTE = "менше хвилини"

#: Local hour boundaries: morning 5-11, day 11-17, evening 17-22, night 22-5.
MORNING_START = 5
DAY_START = 11
EVENING_START = 17
NIGHT_START = 22

TIMES_OF_DAY: tuple[str, ...] = ("morning", "day", "evening", "night")

# Every character people (and keyboard layouts) use as an apostrophe in
# Ukrainian words like "пам'ять": right/left single quote, modifier letter
# apostrophe, backtick, acute accent, prime, fullwidth apostrophe.
_APOSTROPHES = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "ʼ": "'",
        "`": "'",
        "´": "'",
        "′": "'",
        "＇": "'",
    }
)
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
# Latin look-alikes that slip into Ukrainian words when the layout has no
# "і"/"ї" key (common on Russian layouts): "привiт" must still match "привіт".
_LATIN_LOOKALIKES = str.maketrans({"i": "і", "ï": "ї"})
_SPACES = re.compile(r"\s+")
# An apostrophe that is not between two word characters is just a quote mark.
_STRAY_APOSTROPHE = re.compile(r"(?<!\w)'|'(?!\w)")


def plural(n: float, forms3: Sequence[str]) -> str:
    """Ukrainian plural form of a noun for the number ``n``.

    ``forms3`` is (one, few, many), e.g. ("годину", "години", "годин"):
    1, 21, 101 -> one; 2-4, 22-24 -> few; 0, 5-20, 25-30, 111-114 -> many.
    Fractions take the "few" form ("2,5 години"), as in standard Ukrainian.
    Only the noun is returned, so the caller decides how to print the number.
    """
    if len(forms3) != 3:
        raise ValueError("forms3 must have exactly three forms: (one, few, many)")
    if isinstance(n, float):
        if not math.isfinite(n):
            return forms3[2]
        if not n.is_integer():
            return forms3[1]
    count = abs(int(n))
    last_two = count % 100
    last = count % 10
    if last == 1 and last_two != 11:
        return forms3[0]
    if 2 <= last <= 4 and not 12 <= last_two <= 14:
        return forms3[1]
    return forms3[2]


def _count(n: int, forms3: Sequence[str]) -> str:
    return f"{n} {plural(n, forms3)}"


def format_duration(seconds: float) -> str:
    """Human Ukrainian duration that reads well after "вже" / "цілих".

    ``45 -> "менше хвилини"``, ``60 -> "1 хвилину"``, ``2700 -> "45 хвилин"``,
    ``3600 -> "1 годину"``, ``8100 -> "2 години 15 хвилин"``,
    ``97200 -> "1 день 3 години"``.  Minutes are dropped once days appear:
    nobody cares about the minutes of a three-day absence.  Negative, NaN and
    other unusable values read as "менше хвилини" rather than raising, because
    durations come from clocks that can jump.
    """
    try:
        value = float(seconds)
    except (TypeError, ValueError, OverflowError):
        return LESS_THAN_A_MINUTE
    if not math.isfinite(value) or value < 60.0:
        return LESS_THAN_A_MINUTE
    total_minutes = int(value // 60)
    days, rest = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(rest, 60)
    parts: list[str] = []
    if days:
        parts.append(_count(days, DAY_FORMS))
        if hours:
            parts.append(_count(hours, HOUR_FORMS))
    elif hours:
        parts.append(_count(hours, HOUR_FORMS))
        if minutes:
            parts.append(_count(minutes, MINUTE_FORMS))
    else:
        parts.append(_count(minutes, MINUTE_FORMS))
    return " ".join(parts)


def time_of_day(moment: datetime) -> str:
    """"morning" (05-11), "day" (11-17), "evening" (17-22) or "night" (22-05) for a local time."""
    hour = moment.hour
    if MORNING_START <= hour < DAY_START:
        return "morning"
    if DAY_START <= hour < EVENING_START:
        return "day"
    if EVENING_START <= hour < NIGHT_START:
        return "evening"
    return "night"


def format_clock(moment: datetime) -> str:
    """"HH:MM" (24-hour), the way Ukrainians write the time."""
    return moment.strftime("%H:%M")


def render(template: str, variables: Mapping[str, Any]) -> str:
    """Fill ``{placeholders}`` in a phrase template.

    A missing variable raises :class:`KeyError` on purpose: the phrase
    selector catches it and skips that phrase, so a line about ``{game}`` is
    never shown with a hole in it.
    """
    return template.format_map(variables)


def normalize(text: str) -> str:
    """Canonical form of chat input for keyword matching.

    Lower case; every apostrophe variant becomes ``'``; Latin "i"/"ï" inside
    Cyrillic words become "і"/"ї"; punctuation (including hyphens: "па-па" ->
    "па па") becomes spaces; runs of whitespace collapse to one space.
    Apostrophes inside words survive ("пам'ять").
    """
    lowered = text.lower().translate(_APOSTROPHES)
    cleaned = "".join(ch if ch.isalnum() or ch == "'" else " " for ch in lowered)
    cleaned = _STRAY_APOSTROPHE.sub(" ", cleaned)
    words = []
    for word in cleaned.split():
        if _CYRILLIC.search(word):
            word = word.translate(_LATIN_LOOKALIKES)
        words.append(word)
    return _SPACES.sub(" ", " ".join(words)).strip()
