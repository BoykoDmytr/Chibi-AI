"""Long-term memory (facts she remembers about the user) and the short-term
chat history.

Long-term memories live in the SQLite table ``memories`` (migration v2).  They
are small sentences ("Кота звати Мурчик") of a few *kinds* - fact, preference,
relationship, game_event, joke - written by the chat's memory extraction or by
the user in the memory viewer, and retrieved for every AI request by
:meth:`MemoryStore.search`.

Design notes - why it looks like this:

* **Privacy first.**  :func:`looks_sensitive` rejects card and IBAN numbers,
  passwords / PINs / CVV / one-time codes, e-mail addresses, phone numbers and
  API-key-like tokens *before* anything is written: such text is never stored,
  whoever asks (the model's extraction or the user).  The check errs on the
  side of caution - a false positive only means one memory is not kept.
  Memory text is never logged; logs carry ids and kinds only.
* **No embedding model.**  Retrieval is BM25 over light Ukrainian-friendly
  tokens, which runs instantly on the main thread, needs no download and works
  offline.  Tokens are lower-cased (NFKC + casefold), every apostrophe variant
  is unified and then dropped ("м'ясо" == "мʼясо" == "мясо"), dotted acronyms
  are joined ("S.T.A.L.K.E.R." -> "stalker"), a small Ukrainian + English
  stopword list is dropped (negations are *kept*: "не люблю" must not look like
  "люблю"), and words are stemmed by keeping their first 5 characters.
  Cyrillic stems additionally lose trailing vowels / soft sign down to 3
  letters, so short inflected words meet ("кава" / "каву" / "кавою" -> "кав",
  "Мурчик" / "Мурчика" -> "мурч").
* **Known limits** (documented, not bugs): vowel alternation in roots ("кіт" /
  "кота", "ніч" / "ночі") and consonant mutation ("собака" / "собаці") are not
  matched, and there is no cross-script matching - "сталкер" (Cyrillic) does
  not find "S.T.A.L.K.E.R." (Latin).  A stored memory is still found by any of
  its other words.
* **Bounded.**  At most :data:`MAX_MEMORIES` rows: beyond that the least
  important, least recently used unpinned memories are forgotten.  Near
  duplicates of the same kind (token Jaccard >= 0.75) update the existing row
  instead of piling up copies.

Threading: :class:`MemoryStore` is main (Qt) thread only - the database
connection lives there.  :class:`ShortTermMemory` is a plain in-memory object,
also used on the main thread; it is never persisted.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import unicodedata
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from chibi.ai.types import ChatTurn, Role
from chibi.core.clock import Clock
from chibi.core.event_bus import EventBus
from chibi.core.events import MemoryChanged
from chibi.memory.database import Database

log = logging.getLogger(__name__)

#: Memory kinds (tabs of the memory viewer).  Mirrors
#: ``chibi.settings.facade.MEMORY_KINDS`` - defined here as well so this
#: headless module does not import the Qt facade (a test keeps them equal).
MEMORY_KINDS: tuple[str, ...] = ("fact", "preference", "relationship", "game_event", "joke")
#: Who created a memory.
MEMORY_SOURCES: tuple[str, ...] = ("chat", "system", "game", "user")

#: Longest stored text; longer text is cut at a word boundary with an ellipsis.
MAX_TEXT_CHARS = 300
ELLIPSIS = "…"
#: Capacity of the memories table.
MAX_MEMORIES = 2000
#: Same-kind memories at least this similar (token Jaccard) are one memory.
DUPLICATE_JACCARD = 0.75
DEFAULT_IMPORTANCE = 0.5

# -- BM25 / ranking ------------------------------------------------------------------------
BM25_K1 = 1.2
BM25_B = 0.75
PINNED_BOOST = 1.5
#: Recency boost: a memory updated just now scores up to (1 + RECENCY_BOOST)
#: times higher than an ancient one; the extra halves every RECENCY_HALF_LIFE.
RECENCY_BOOST = 0.5
RECENCY_HALF_LIFE_SECONDS = 30 * 24 * 3600.0
#: Importance (0..1) scales a score between these factors.
IMPORTANCE_MIN_FACTOR = 0.75
IMPORTANCE_MAX_FACTOR = 1.25
#: Long chat messages are used as queries; their tail adds little.
MAX_QUERY_TERMS = 64
STEM_PREFIX = 5
MIN_STEM = 3

#: Short-term memory: one chat turn is capped at this many characters.
MAX_TURN_CHARS = 4000
#: add() / update() look at no more input than this: stored text is at most
#: MAX_TEXT_CHARS anyway, and it bounds the work of the sensitive-data checks.
MAX_INPUT_CHARS = 4000


@dataclass(frozen=True)
class Memory:
    id: int
    kind: str
    text: str
    source: str
    importance: float
    pinned: bool
    created_at: float
    updated_at: float
    last_used_at: float | None


class MemoryRejectedError(ValueError):
    """An edit was refused (empty text or personal data).  ``user_message`` is
    a short Ukrainian explanation for the memory viewer."""

    def __init__(self, message: str, *, user_message: str) -> None:
        super().__init__(message)
        self.user_message = user_message


# =========================================================================================
# Text sanitation
# =========================================================================================

#: Zero-width joiner is a format character but part of emoji sequences (👩‍💻).
_KEEP_FORMAT_CHARS = frozenset({"‍"})
_WHITESPACE = re.compile(r"\s+")
_TRAILING_JUNK = " ,.;:!?-–—"


def _clean(text: str) -> str:
    """NFC, control / invisible format characters removed, whitespace collapsed.

    Removing format characters (zero-width spaces, bidi overrides) also stops
    them from hiding a card number from :func:`looks_sensitive`.
    """
    text = unicodedata.normalize("NFC", text)
    chars: list[str] = []
    for ch in text:
        if ch.isspace():
            chars.append(" ")
            continue
        category = unicodedata.category(ch)
        if category in ("Cc", "Cs") or (category == "Cf" and ch not in _KEEP_FORMAT_CHARS):
            continue
        chars.append(ch)
    return _WHITESPACE.sub(" ", "".join(chars)).strip()


def _cap(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    """Cut *text* to at most *limit* characters at a word boundary, ending with "…"."""
    if len(text) <= limit:
        return text
    room = limit - len(ELLIPSIS)
    cut = text[:room]
    if not text[room].isspace():
        space = cut.rfind(" ")
        # Prefer a word boundary, but never throw away more than ~40 % of the room
        # (a 250-character "word" is a URL or garbage: cut it hard).
        if space >= room * 0.6:
            cut = cut[:space]
    return cut.rstrip(_TRAILING_JUNK) + ELLIPSIS


def _clean_input(text: object) -> str:
    """Cleaned, bounded user / model input for add() and update()."""
    if not isinstance(text, str):
        return ""
    return _clean(text[:MAX_INPUT_CHARS * 2])[:MAX_INPUT_CHARS]


def sanitize_text(text: str) -> str:
    """The form a memory is stored in: cleaned (see :func:`_clean`) and capped
    at :data:`MAX_TEXT_CHARS` characters."""
    if not isinstance(text, str):
        return ""
    return _cap(_clean(text))


# =========================================================================================
# Sensitive data detection
# =========================================================================================

# -- payment cards: 13-19 digits, optionally grouped by spaces / dashes, Luhn-valid ------
_DIGIT_RUN = re.compile(r"\d+(?:[ \-]\d+)*")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, ch in enumerate(reversed(digits)):
        value = int(ch)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _has_card(text: str) -> bool:
    for run in _DIGIT_RUN.finditer(text):
        groups = re.split(r"[ \-]", run.group(0))
        # Any contiguous window of groups may be the card ("my card 4111 1111 1111 1111 22").
        for start in range(len(groups)):
            total = 0
            for end in range(start, len(groups)):
                total += len(groups[end])
                if total > 19:
                    break
                if total < 13:
                    continue
                window = groups[start:end + 1]
                # Cards are printed in groups of 3+ digits (4-4-4-4, 4-6-5); a run
                # of short groups is a list of small numbers, not a card.
                if len(window) > 1 and any(len(g) < 3 for g in window):
                    continue
                if _luhn_ok("".join(window)):
                    return True
    return False


# -- IBAN ---------------------------------------------------------------------------------
#: IBAN length per country (ISO 13616 registry).
_IBAN_LENGTHS: dict[str, int] = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16, "BG": 22, "BH": 22,
    "BR": 29, "BY": 28, "CH": 21, "CR": 22, "CY": 28, "CZ": 24, "DE": 22, "DK": 18, "DO": 28,
    "EE": 20, "EG": 29, "ES": 24, "FI": 18, "FO": 18, "FR": 27, "GB": 22, "GE": 22, "GI": 23,
    "GL": 18, "GR": 27, "GT": 28, "HR": 21, "HU": 28, "IE": 22, "IL": 23, "IQ": 23, "IS": 26,
    "IT": 27, "JO": 30, "KW": 30, "KZ": 20, "LB": 28, "LC": 32, "LI": 21, "LT": 20, "LU": 20,
    "LV": 21, "MC": 27, "MD": 24, "ME": 22, "MK": 19, "MR": 27, "MT": 31, "MU": 30, "NL": 18,
    "NO": 15, "PK": 24, "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "SA": 24,
    "SC": 31, "SE": 24, "SI": 19, "SK": 24, "SM": 27, "ST": 25, "SV": 28, "TL": 23, "TN": 24,
    "TR": 26, "UA": 29, "VA": 22, "VG": 24, "XK": 20,
}
#: Ukrainian IBAN: "UA" + 27 digits - flagged even with a typo in the checksum.
_IBAN_UA = re.compile(r"(?<![A-Za-z0-9])UA\s?\d{2}(?:\s?\d){25}(?!\d)", re.IGNORECASE)
_IBAN_CANDIDATE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2}\d{2}(?:\s?[A-Za-z0-9]){11,32}")


def _iban_checksum_ok(iban: str) -> bool:
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def _has_iban(text: str) -> bool:
    if _IBAN_UA.search(text):
        return True
    for match in _IBAN_CANDIDATE.finditer(text):
        compact = match.group(0).replace(" ", "").upper()
        length = _IBAN_LENGTHS.get(compact[:2])
        if length is None or len(compact) < length:
            continue
        # The candidate may have swallowed a following word; the IBAN is its prefix.
        iban = compact[:length]
        if iban.isascii() and iban.isalnum() and _iban_checksum_ok(iban):
            return True
    return False


# -- e-mail -------------------------------------------------------------------------------
# Quantifiers are bounded (RFC limits: local part 64, label 63): an unbounded
# "[\w.]+@" backtracks quadratically over a long run of letters without "@".
_EMAIL = re.compile(r"[\w.+\-]{1,64}@[\w\-]{1,63}(?:\.[\w\-]{1,63}){0,8}\.[^\W\d_]{2,24}")

# -- phone numbers ------------------------------------------------------------------------
# Ukrainian: +380 67 123 45 67, 380671234567, (067) 123-45-67, 067 123 45 67, 0671234567.
_PHONE_UA = re.compile(
    r"(?<![\d+])(?:\+\s?)?(?:38\s?)?\(?0\s?\d{2}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)"
)
# International with a plus: +1 (555) 123-4567, +44 20 7946 0958 (8..15 digits).
_PHONE_INTL = re.compile(r"(?<![\w+])\+\s?\d(?:[\s\-()]{0,2}\d){7,14}(?!\d)")


def _has_phone(text: str) -> bool:
    return bool(_PHONE_UA.search(text) or _PHONE_INTL.search(text))


# -- passwords, PINs, CVV, one-time codes -------------------------------------------------
_PASSWORD_WORDS = r"парол\w{1,3}|password|passwd|passcode|pwd|пасворд\w{0,2}"
_CODE_WORDS = (
    r"пін[\s\-]?код\w{0,2}|pin[\s\-]?code|pin|пін|cvv2?|cvc2?|otp|код(?:у|ом|и|ів)?|code"
)
_PASSWORD_KEYWORD = re.compile(rf"(?:{_PASSWORD_WORDS})", re.IGNORECASE)
# Words that may sit between the keyword and the value: "пін-код від картки 1234",
# "код з смс 4821", "my new password: ...".
_SECRET_FILLER = (
    r"мій|моя|моє|мого|моєї|мої|твій|твоя|my|your|the|a|new|old|новий|нова|старий|"
    r"ось|тепер|now|з|із|зі|смс|sms|підтвердження|доступу|безпеки|верифікації|картки|карти|"
    r"card|security|access|verification|login|логіну|входу"
)
_SECRET = re.compile(
    rf"(?<!\w)(?P<kw>{_PASSWORD_WORDS}|{_CODE_WORDS})(?!\w)"
    rf"(?:\s+(?:{_SECRET_FILLER})(?!\w)"
    rf"|\s+(?:від|до|для|for|to|of|on|in|у|в|на)\s+[^\s:=]+){{0,3}}"
    rf"(?P<sep>\s*[:=]\s*|\s+[-—–]\s+|\s+(?:is|це|є|was|був|була|такий|така)\s+|\s+)"
    rf"(?P<value>\S+)",
    re.IGNORECASE,
)
# "Пароль користувача: qwerty123", "пін картки дружини = 4821": any few words
# between the keyword and an explicit ":" / "=" still introduce a value.
_SECRET_EXPLICIT = re.compile(
    rf"(?<!\w)(?P<kw>{_PASSWORD_WORDS}|{_CODE_WORDS})(?!\w)(?P<between>(?:\s+[^\s:=]{{1,40}}){{0,6}})"
    rf"\s*[:=]\s*(?P<value>\S+)",
    re.IGNORECASE,
)
# "Пароль користувача qwerty123": a password-looking value (letters and digits
# or symbols) a couple of words after the keyword, without any separator.
_SECRET_NEARBY = re.compile(
    rf"(?<!\w)(?:{_PASSWORD_WORDS})(?!\w)(?:\s+[^\s:=]{{1,40}}){{1,3}}\s+(?P<value>\S{{6,64}})",
    re.IGNORECASE,
)
#: "код помилки 404", "error code: 500" talk about codes, not secrets.
_NOT_SECRET_CODE = re.compile(r"помил|error|статус|status|http|країн|міст|товар|продукт|поштов|zip",
                              re.IGNORECASE)
_CODE_VALUE = re.compile(r"[^\W_][\w\-]{2,15}")
_VALUE_PUNCTUATION = ".,!?;:()[]{}«»\"'…"


def _has_secret(text: str) -> bool:
    for match in _SECRET_EXPLICIT.finditer(text):
        value = match.group("value").strip(_VALUE_PUNCTUATION)
        if _PASSWORD_KEYWORD.fullmatch(match.group("kw")):
            if len(value) >= 3:
                return True
        elif (sum(ch.isdigit() for ch in value) >= 3 and _CODE_VALUE.fullmatch(value)
              and not _NOT_SECRET_CODE.search(text[max(0, match.start() - 24):match.start()]
                                              + match.group("between"))):
            return True
    for match in _SECRET_NEARBY.finditer(text):
        value = match.group("value").strip(_VALUE_PUNCTUATION)
        has_letter = any(ch.isalpha() for ch in value)
        has_other = any(ch.isdigit() or not ch.isalnum() for ch in value)
        if len(value) >= 6 and has_letter and has_other:
            return True
    for match in _SECRET.finditer(text):
        value = match.group("value").strip(_VALUE_PUNCTUATION)
        if not value:
            continue
        digits = sum(ch.isdigit() for ch in value)
        if not _PASSWORD_KEYWORD.fullmatch(match.group("kw")):
            # PIN / CVV / code: a short digit (or digit-and-letter) code.  "3.12"
            # after "код на Python" is a version, not a code; "error code 500" is
            # a status, not a secret.
            context = text[max(0, match.start() - 24):match.end()]
            if digits >= 3 and _CODE_VALUE.fullmatch(value) and not _NOT_SECRET_CODE.search(context):
                return True
            continue
        if match.group("sep").strip():
            # "пароль: котик", "password is swordfish": an explicit value.
            if len(value) >= 3:
                return True
            continue
        # "пароль kotik2020": only a password-looking word (digits or symbols);
        # "пароль надто простий" is talk about a password, not the password.
        if len(value) >= 4 and (digits or any(not ch.isalnum() for ch in value)):
            return True
    return False


# -- API keys and other tokens ------------------------------------------------------------
_API_KEY_PREFIXED = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"sk-[A-Za-z0-9_\-]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|xox[abposr]-[A-Za-z0-9\-]{10,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z_\-]{30,}"
    r"|glpat-[A-Za-z0-9_\-]{20,}"
    r"|hf_[A-Za-z0-9]{30,}"
    r")"
)
_LONG_TOKEN = re.compile(r"[A-Za-z0-9_+\-]{32,}")
_HEX_TOKEN = re.compile(r"[0-9a-fA-F\-]+")


def _has_api_key(text: str) -> bool:
    if _API_KEY_PREFIXED.search(text):
        return True
    for match in _LONG_TOKEN.finditer(text):
        token = match.group(0)
        has_digit = any(ch.isdigit() for ch in token)
        has_upper = any(ch.isupper() for ch in token)
        has_lower = any(ch.islower() for ch in token)
        if (_HEX_TOKEN.fullmatch(token) and has_digit and (has_upper or has_lower)
                and sum(ch != "-" for ch in token) >= 32):
            return True  # hex secrets, hashes, UUID-like keys (not "AAAA..." shouting)
        if has_digit and has_upper and has_lower:
            return True  # base64 / random mixed-case tokens
        letters = sum(ch.isalpha() for ch in token)
        digits = sum(ch.isdigit() for ch in token)
        if digits >= 6 and letters >= 6:
            return True  # lower-case alphanumeric tokens
    return False


def _has_email(text: str) -> bool:
    return "@" in text and bool(_EMAIL.search(text))


_SENSITIVE_CHECKS = (_has_card, _has_iban, _has_email, _has_phone, _has_secret, _has_api_key)


def looks_sensitive(text: str) -> bool:
    """True if *text* contains something that must never be stored: a payment
    card number (Luhn-valid), an IBAN, a password / PIN / CVV / one-time code
    with its value, an e-mail address, a phone number or an API-key-like token.

    The text is NFKC-normalised first, so fullwidth digits or zero-width
    characters cannot smuggle a number past the checks.
    """
    if not isinstance(text, str) or not text:
        return False
    normalized = unicodedata.normalize("NFKC", _clean(text))
    return any(check(normalized) for check in _SENSITIVE_CHECKS)


# =========================================================================================
# Tokens
# =========================================================================================

_APOSTROPHES = str.maketrans({ch: "'" for ch in "’‘ʼ`´′＇ʹ"})
_ACRONYM = re.compile(r"\b(?:[^\W\d_]\.){2,}")
_WORD = re.compile(r"[^\W_]+")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
# Latin look-alikes typed on layouts without "і"/"ї" ("кiт" must be "кіт").
_LATIN_LOOKALIKES = str.maketrans({"i": "і", "ï": "ї"})
_UK_SOFT_ENDINGS = frozenset("аеєиіїоуюяь")

#: Function words that carry no meaning for retrieval.  Negations ("не", "ні",
#: "без", "not", "no") are deliberately absent: they flip a fact's meaning.
_STOPWORDS_TEXT = """
    і й та а але або чи що як це цей ця ці цього цієї той ті то так же ж би б у в на з із зі
    до від по за про для під над при через після перед між біля мене мені мій моя моє мої
    мого моєї моїх моєму моїм ти тебе тобі твій твоя твоє твої твого він вона воно вони
    його її їх їм їй йому ним нею ми нас нам наш наша наше наші ви вас вам ваш ваша я є
    був була було були буде будуть дуже ще вже теж також тут там коли який яка яке які
    якого яких де бо щоб якщо хто свій своя своє свої себе собі саме лише тільки просто
    ось оце цим цьому тому всі все весь вся
    a an the and or but of to in on at for with from by is are was were be been being am
    i me my mine you your yours he him his she her it its we us our they them their this
    that these those do does did have has had so very just about as into than then there
    here what which who whom how when where will would can could shall should also too
"""
STOPWORDS: frozenset[str] = frozenset(_STOPWORDS_TEXT.split())


def _stem(token: str) -> str:
    if token.isdigit():
        return token
    stem = token[:STEM_PREFIX]
    if _CYRILLIC.search(stem):
        while len(stem) > MIN_STEM and stem[-1] in _UK_SOFT_ENDINGS:
            stem = stem[:-1]
    elif len(stem) > MIN_STEM and stem.endswith("s") and not stem.endswith("ss"):
        stem = stem[:-1]  # games -> game, cats -> cat
    return stem


def tokenize(text: str) -> list[str]:
    """Search tokens of *text* (see the module docstring for the rules)."""
    if not text:
        return []
    lowered = unicodedata.normalize("NFKC", text).casefold().translate(_APOSTROPHES)
    lowered = _ACRONYM.sub(lambda m: m.group(0).replace(".", ""), lowered)
    lowered = lowered.replace("'", "")
    tokens: list[str] = []
    for word in _WORD.findall(lowered):
        if _CYRILLIC.search(word):
            word = word.translate(_LATIN_LOOKALIKES)
        if word in STOPWORDS:
            continue
        if len(word) < 2 and not word.isdigit():
            continue
        tokens.append(_stem(word))
    return tokens


_NEGATIONS: frozenset[str] = frozenset(
    _stem(word)
    for word in ("не", "ні", "без", "ніколи", "нема", "немає", "жоден", "жодна", "нічого",
                 "not", "no", "never", "dont", "doesnt", "didnt", "isnt", "cant", "wont")
)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _fold(text: str) -> str:
    """Case- and apostrophe-insensitive form for substring filtering."""
    folded = unicodedata.normalize("NFKC", text).casefold().translate(_APOSTROPHES)
    return _WHITESPACE.sub(" ", folded).strip()


# =========================================================================================
# Long-term store
# =========================================================================================


def _check_kind(kind: str) -> str:
    if kind not in MEMORY_KINDS:
        raise ValueError(f"unknown memory kind {kind!r}; expected one of {MEMORY_KINDS}")
    return kind


def _check_source(source: str) -> str:
    if source not in MEMORY_SOURCES:
        raise ValueError(f"unknown memory source {source!r}; expected one of {MEMORY_SOURCES}")
    return source


def _clamp_importance(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return DEFAULT_IMPORTANCE
    if math.isnan(number):
        return DEFAULT_IMPORTANCE
    return min(max(number, 0.0), 1.0)


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=int(row["id"]),
        kind=str(row["kind"]),
        text=str(row["text"]),
        source=str(row["source"]),
        importance=float(row["importance"]),
        pinned=bool(row["pinned"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        last_used_at=None if row["last_used_at"] is None else float(row["last_used_at"]),
    )


_COLUMNS = "id, kind, text, source, importance, pinned, created_at, updated_at, last_used_at"

#: Ukrainian explanations for the memory viewer (add() returns None for the same reasons).
EMPTY_MEMORY_MESSAGE = "Порожній спогад не збережу."
SENSITIVE_MEMORY_MESSAGE = (
    "Схоже на особисті дані (номер картки, пароль, телефон, пошта чи ключ) - "
    "таке я не запам'ятовую."
)


@dataclass(frozen=True, slots=True)
class _Doc:
    """A memory's text, its search tokens and their counts (cached per memory id)."""

    text: str
    tokens: tuple[str, ...]
    counts: dict[str, int]

    @classmethod
    def of(cls, text: str) -> _Doc:
        tokens = tuple(tokenize(text))
        return cls(text, tokens, dict(Counter(tokens)))


class MemoryStore:
    """Long-term memories in SQLite.  Main thread only; publishes
    :class:`~chibi.core.events.MemoryChanged` after every change (never after
    a search)."""

    def __init__(self, db: Database, bus: EventBus | None, clock: Clock) -> None:
        self._db = db
        self._bus = bus
        self._clock = clock
        #: id -> tokenised text: memories are re-tokenised only when their text changes.
        self._token_cache: dict[int, _Doc] = {}

    # -- writing ----------------------------------------------------------------------------

    def add(self, kind: str, text: str, source: str = "chat",
            importance: float = DEFAULT_IMPORTANCE) -> Memory | None:
        """Remember *text*.  Returns the stored memory, or None if the text is
        empty or looks like personal data.

        A near-duplicate of an existing memory of the same kind updates that
        memory instead (the longer text survives; when one of the two is
        negated - "любить каву" vs "не любить каву" - the newer one does,
        because it corrects the old fact).  Importance only ever rises.  A
        memory the user wrote (source "user") or pinned keeps its text unless
        the user restates it.
        """
        _check_kind(kind)
        _check_source(source)
        importance = _clamp_importance(importance)
        cleaned = _clean_input(text)
        if not cleaned:
            return None
        if looks_sensitive(cleaned):
            log.info("memory not stored: looks like personal data (kind=%s, source=%s)", kind, source)
            return None
        stored_text = _cap(cleaned)
        tokens = tuple(tokenize(stored_text))
        now = self._clock.now()
        evicted_other_kinds = False
        with self._db.transaction() as conn:
            duplicate = self._find_duplicate(conn, kind, stored_text, frozenset(tokens))
            if duplicate is not None:
                # Text the user wrote or pinned is theirs: an automatic restatement
                # (chat extraction, game events) refreshes it but never rewrites it.
                protected = (duplicate.pinned or duplicate.source == "user") and source != "user"
                new_text = duplicate.text if protected else self._merged_text(duplicate.text, stored_text)
                new_importance = max(duplicate.importance, importance)
                conn.execute(
                    "UPDATE memories SET text = ?, importance = ?, updated_at = ? WHERE id = ?",
                    (new_text, new_importance, now, duplicate.id),
                )
                memory = replace(duplicate, text=new_text, importance=new_importance, updated_at=now)
                log.debug("memory %d (%s) refreshed by a near-duplicate", memory.id, kind)
            else:
                cursor = conn.execute(
                    "INSERT INTO memories(kind, text, source, importance, pinned, created_at, "
                    "updated_at, last_used_at) VALUES(?, ?, ?, ?, 0, ?, ?, NULL)",
                    (kind, stored_text, source, importance, now, now),
                )
                memory_id = int(cursor.lastrowid or 0)
                memory = Memory(memory_id, kind, stored_text, source, importance, False, now, now, None)
                evicted_kinds = self._enforce_capacity(conn, keep_id=memory_id)
                evicted_other_kinds = any(k != kind for k in evicted_kinds)
                log.debug("memory %d (%s) added", memory_id, kind)
        self._token_cache[memory.id] = _Doc.of(memory.text)
        self._publish("" if evicted_other_kinds else kind)
        return memory

    def update(self, memory_id: int, text: str) -> None:
        """Replace a memory's text (memory viewer edit).

        Raises :class:`MemoryRejectedError` (a ``ValueError``) for empty text
        or text that looks like personal data - the user should see why the
        edit did not stick.  An unknown id is ignored (the viewer was stale).
        """
        cleaned = _clean_input(text)
        if not cleaned:
            raise MemoryRejectedError("empty memory text", user_message=EMPTY_MEMORY_MESSAGE)
        if looks_sensitive(cleaned):
            log.info("memory %d edit refused: looks like personal data", memory_id)
            raise MemoryRejectedError("memory text looks sensitive", user_message=SENSITIVE_MEMORY_MESSAGE)
        stored_text = _cap(cleaned)
        current = self.get(memory_id)
        if current is None:
            log.debug("update of unknown memory %d ignored", memory_id)
            return
        if current.text == stored_text:
            return
        with self._db.transaction() as conn:
            conn.execute("UPDATE memories SET text = ?, updated_at = ? WHERE id = ?",
                         (stored_text, self._clock.now(), memory_id))
        self._token_cache.pop(memory_id, None)
        self._publish(current.kind)

    def set_pinned(self, memory_id: int, pinned: bool) -> None:
        """Pinned memories are never evicted and rank higher in search."""
        current = self.get(memory_id)
        if current is None or current.pinned == bool(pinned):
            return
        with self._db.transaction() as conn:
            conn.execute("UPDATE memories SET pinned = ? WHERE id = ?", (int(bool(pinned)), memory_id))
        self._publish(current.kind)

    def delete(self, memory_id: int) -> None:
        current = self.get(memory_id)
        if current is None:
            return
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._token_cache.pop(memory_id, None)
        log.info("memory %d (%s) deleted", memory_id, current.kind)
        self._publish(current.kind)

    def clear(self, kind: str | None = None) -> None:
        """Forget every memory (or every memory of one kind)."""
        if kind is not None:
            _check_kind(kind)
        with self._db.transaction() as conn:
            if kind is None:
                conn.execute("DELETE FROM memories")
            else:
                conn.execute("DELETE FROM memories WHERE kind = ?", (kind,))
        if kind is None:
            self._token_cache.clear()
        else:
            self._prune_cache()
        log.info("memories cleared (%s)", kind or "all kinds")
        self._publish(kind or "")

    # -- reading ----------------------------------------------------------------------------

    def get(self, memory_id: int) -> Memory | None:
        row = self._db.conn.execute(
            f"SELECT {_COLUMNS} FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return None if row is None else _row_to_memory(row)

    def count(self, kind: str | None = None) -> int:
        if kind is None:
            row = self._db.conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        else:
            row = self._db.conn.execute("SELECT COUNT(*) FROM memories WHERE kind = ?",
                                        (_check_kind(kind),)).fetchone()
        return int(row[0])

    def list(self, kind: str | None = None, query: str = "", limit: int = 500) -> list[Memory]:
        """Memories for the viewer: pinned first, then most recently updated.

        *query* is a plain case-insensitive substring filter (what a search box
        in a list is expected to do); use :meth:`search` for relevance.
        """
        if limit <= 0:
            return []
        if kind is None:
            rows = self._db.conn.execute(
                f"SELECT {_COLUMNS} FROM memories ORDER BY pinned DESC, updated_at DESC, id DESC"
            ).fetchall()
        else:
            rows = self._db.conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE kind = ? "
                "ORDER BY pinned DESC, updated_at DESC, id DESC",
                (_check_kind(kind),),
            ).fetchall()
        needle = _fold(query) if isinstance(query, str) else ""
        result: list[Memory] = []
        for row in rows:
            if needle and needle not in _fold(row["text"]):
                continue
            result.append(_row_to_memory(row))
            if len(result) >= limit:
                break
        return result

    def search(self, query: str, k: int = 6, kinds: Iterable[str] | None = None) -> list[Memory]:
        """The *k* memories most relevant to *query* (BM25 with pinned,
        recency and importance boosts), best first; only memories sharing at
        least one search token with the query are returned.  Marks the
        returned memories as used (``last_used_at``), which protects them
        from eviction.  Never raises on a database error: retrieval is
        optional context and must not break a chat reply.
        """
        if k <= 0 or not isinstance(query, str):
            return []
        kind_filter = None if kinds is None else tuple(dict.fromkeys(_check_kind(x) for x in kinds))
        if kind_filter is not None and not kind_filter:
            return []
        terms = list(dict.fromkeys(tokenize(query)))[:MAX_QUERY_TERMS]
        if not terms:
            return []
        try:
            rows = self._select(kind_filter)
        except sqlite3.Error:
            log.exception("memory search failed")
            return []
        if not rows:
            return []
        docs = [self._doc_for(int(row["id"]), str(row["text"])) for row in rows]
        if kind_filter is None:
            self._prune_cache({int(row["id"]) for row in rows})
        scores = _bm25_counts(terms, [doc.counts for doc in docs], [len(doc.tokens) for doc in docs])
        now = self._clock.now()
        ranked: list[tuple[float, Memory]] = []
        for row, score in zip(rows, scores, strict=True):
            if score <= 0.0:
                continue
            memory = _row_to_memory(row)  # only for matches: most rows never score
            ranked.append((score * self._boost(memory, now), memory))
        ranked.sort(key=lambda item: (-item[0], -item[1].updated_at, -item[1].id))
        top = [memory for _score, memory in ranked[:k]]
        if not top:
            return []
        try:
            with self._db.transaction() as conn:
                conn.executemany("UPDATE memories SET last_used_at = ? WHERE id = ?",
                                 [(now, memory.id) for memory in top])
        except sqlite3.Error:
            log.exception("could not mark memories as used")
            return top
        return [replace(memory, last_used_at=now) for memory in top]

    # -- internals --------------------------------------------------------------------------

    def _select(self, kinds: Sequence[str] | None) -> list[sqlite3.Row]:
        if kinds is None:
            return self._db.conn.execute(f"SELECT {_COLUMNS} FROM memories").fetchall()
        placeholders = ", ".join("?" for _ in kinds)
        return self._db.conn.execute(
            f"SELECT {_COLUMNS} FROM memories WHERE kind IN ({placeholders})", tuple(kinds)
        ).fetchall()

    def _doc_for(self, memory_id: int, text: str) -> _Doc:
        cached = self._token_cache.get(memory_id)
        if cached is not None and cached.text == text:
            return cached
        doc = _Doc.of(text)
        self._token_cache[memory_id] = doc
        return doc

    def _prune_cache(self, live_ids: set[int] | None = None) -> None:
        """Drop cached tokens of memories that no longer exist (they hold user text)."""
        if live_ids is None:
            live_ids = {int(row[0]) for row in self._db.conn.execute("SELECT id FROM memories")}
        for memory_id in [i for i in self._token_cache if i not in live_ids]:
            del self._token_cache[memory_id]

    def _find_duplicate(self, conn: sqlite3.Connection, kind: str, text: str,
                        tokens: frozenset[str]) -> Memory | None:
        folded = _fold(text)
        best: sqlite3.Row | None = None
        best_similarity = 0.0
        for row in conn.execute(f"SELECT {_COLUMNS} FROM memories WHERE kind = ?", (kind,)):
            row_text = str(row["text"])
            if _fold(row_text) == folded:
                return _row_to_memory(row)
            doc = self._doc_for(int(row["id"]), row_text)
            similarity = _jaccard(tokens, frozenset(doc.tokens))
            if similarity >= DUPLICATE_JACCARD and similarity > best_similarity:
                best, best_similarity = row, similarity
        return None if best is None else _row_to_memory(best)

    @staticmethod
    def _merged_text(old: str, new: str) -> str:
        old_negated = bool(_NEGATIONS & set(tokenize(old)))
        new_negated = bool(_NEGATIONS & set(tokenize(new)))
        if old_negated != new_negated:
            return new  # a correction: the newer statement wins
        return new if len(new) >= len(old) else old

    def _enforce_capacity(self, conn: sqlite3.Connection, keep_id: int) -> set[str]:
        """Evict the least important, least recently used unpinned memories
        above :data:`MAX_MEMORIES`.  Returns the kinds that lost memories."""
        total = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        excess = total - MAX_MEMORIES
        if excess <= 0:
            return set()
        victims = conn.execute(
            "SELECT id, kind FROM memories WHERE pinned = 0 AND id != ? "
            "ORDER BY importance ASC, COALESCE(last_used_at, updated_at) ASC, id ASC LIMIT ?",
            (keep_id, excess),
        ).fetchall()
        if len(victims) < excess:
            log.warning("memory capacity exceeded by pinned memories (%d rows)", total)
        conn.executemany("DELETE FROM memories WHERE id = ?", [(row["id"],) for row in victims])
        for row in victims:
            self._token_cache.pop(int(row["id"]), None)
        if victims:
            log.info("forgot %d least important memories (capacity %d)", len(victims), MAX_MEMORIES)
        return {str(row["kind"]) for row in victims}

    @staticmethod
    def _boost(memory: Memory, now: float) -> float:
        factor = PINNED_BOOST if memory.pinned else 1.0
        age = max(0.0, now - memory.updated_at)
        factor *= 1.0 + RECENCY_BOOST * 0.5 ** (age / RECENCY_HALF_LIFE_SECONDS)
        span = IMPORTANCE_MAX_FACTOR - IMPORTANCE_MIN_FACTOR
        factor *= IMPORTANCE_MIN_FACTOR + span * _clamp_importance(memory.importance)
        return factor

    def _publish(self, kind: str) -> None:
        if self._bus is not None:
            self._bus.publish(MemoryChanged(kind=kind))


def _bm25(terms: Sequence[str], documents: Sequence[Sequence[str]]) -> list[float]:
    """Okapi BM25 of each token list in *documents* for the query *terms*."""
    return _bm25_counts(terms, [Counter(doc) for doc in documents], [len(doc) for doc in documents])


def _bm25_counts(terms: Sequence[str], counts: Sequence[Mapping[str, int]],
                 lengths: Sequence[int]) -> list[float]:
    """Okapi BM25 (k1 = 1.2, b = 0.75) over pre-counted documents.

    IDF uses the non-negative ``ln(1 + (N - n + 0.5) / (n + 0.5))`` form, so a
    term present in every document still counts a little instead of going
    negative on a tiny corpus.
    """
    count = len(counts)
    if count == 0:
        return []
    average_length = sum(lengths) / count or 1.0
    idf: dict[str, float] = {}
    for term in terms:
        frequency = sum(1 for doc in counts if term in doc)
        if frequency:
            idf[term] = math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
    if not idf:
        return [0.0] * count
    scores: list[float] = []
    for doc, length in zip(counts, lengths, strict=True):
        score = 0.0
        if length:
            norm = BM25_K1 * (1.0 - BM25_B + BM25_B * length / average_length)
            for term, weight in idf.items():
                tf = doc.get(term, 0)
                if tf:
                    score += weight * tf * (BM25_K1 + 1.0) / (tf + norm)
        scores.append(score)
    return scores


# =========================================================================================
# Short-term memory
# =========================================================================================


class ShortTermMemory:
    """The current conversation (chat turns), newest last.  Never persisted:
    it is gone when the app closes or the chat is cleared."""

    def __init__(self, max_turns: int = 60) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self._turns: deque[ChatTurn] = deque(maxlen=max_turns)

    def add(self, role: Role, text: str) -> None:
        """Append one turn.  Blank text is ignored (a failed reply adds nothing);
        very long text is capped so one paste cannot flood every later prompt."""
        role = Role(role)
        if not isinstance(text, str):
            return
        text = text.strip()
        if not text:
            return
        if len(text) > MAX_TURN_CHARS:
            text = text[:MAX_TURN_CHARS - len(ELLIPSIS)].rstrip() + ELLIPSIS
        self._turns.append(ChatTurn(role=role, text=text))

    def turns(self, limit: int) -> tuple[ChatTurn, ...]:
        """The last *limit* turns, starting with a user turn: provider APIs
        reject (or mis-handle) a conversation that opens with the assistant,
        so a leading assistant turn is dropped."""
        if limit <= 0:
            return ()
        recent = list(self._turns)[-limit:]
        while recent and recent[0].role is not Role.USER:
            recent.pop(0)
        return tuple(recent)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
