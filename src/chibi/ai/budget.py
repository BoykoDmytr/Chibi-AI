"""AI usage accounting and spending limits.

Every completed model request is written to the ``ai_usage`` table (one row per
local day / provider / model / purpose, UPSERTed), so the status window can
show today's requests and cost and :class:`BudgetGuard` can stop her before a
limit is crossed.  Main thread only: the SQLite connection refuses other
threads.

Cost is an *estimate* from public list prices.  Whether a day's cost is known
is derived from the rows themselves instead of a separate column: a priced
model always yields a cost above zero once tokens were spent, so a row with
tokens but a zero cost belongs to a provider we can't price (an
OpenAI-compatible model without a price set).  Ollama runs locally and is free.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from chibi.ai.types import BudgetExceededError, Purpose, Usage
from chibi.core.clock import Clock
from chibi.core.config import AISettings
from chibi.memory.database import Database

log = logging.getLogger(__name__)

#: Claude model id -> (input, output) list price in USD per 1M tokens.
PRICES_PER_MTOK: Mapping[str, tuple[float, float]] = MappingProxyType({
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
})

#: Cache-read prices that differ from the usual 0.1 x input price.
CACHE_READ_PER_MTOK: Mapping[str, float] = MappingProxyType({
    "claude-opus-5-5": 0.20,
    "claude-sonnet-5-5": 0.20,
    "claude-fable-5-1": 0.25,
})
CACHE_READ_FACTOR = 0.1
#: 5-minute cache writes cost 1.25 x the input price.
CACHE_WRITE_FACTOR = 1.25

#: Providers that run on this machine and cost nothing.
FREE_PROVIDERS = frozenset({"ollama", "fake"})

#: CHAT may go this far over the hourly request cap: the user is waiting for her.
CHAT_HOURLY_OVERDRAFT = 0.2
#: Screen vision may use at most this share of the hourly request cap (counting its own
#: requests): a window that changes all the time (a game, a video) would otherwise use
#: up the whole hour in half an hour and leave nothing for the chat and her own lines.
VISION_HOURLY_SHARE = 0.5

_HOUR = 3600.0


@dataclass(frozen=True)
class UsageSummary:
    requests: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    #: False when some of today's requests went to a model we can't price.
    cost_known: bool


def estimate_cost(provider: str, model: str, usage: Usage, ai: AISettings) -> float | None:
    """Estimated USD cost of one request, or None when the price is unknown."""
    if provider in FREE_PROVIDERS:
        return 0.0
    if provider == "anthropic":
        price = PRICES_PER_MTOK.get(model)
        if price is None:
            return None
        p_in, p_out = price
        p_read = CACHE_READ_PER_MTOK.get(model, p_in * CACHE_READ_FACTOR)
        total = (usage.input_tokens * p_in + usage.output_tokens * p_out
                 + usage.cache_read_tokens * p_read + usage.cache_write_tokens * p_in * CACHE_WRITE_FACTOR)
        return total / 1_000_000
    if provider == "openai":
        p_in = ai.openai_price_input_per_mtok
        p_out = ai.openai_price_output_per_mtok
        if p_in <= 0 and p_out <= 0:
            return None
        tokens_in = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
        return (tokens_in * max(p_in, 0.0) + usage.output_tokens * max(p_out, 0.0)) / 1_000_000
    return None


class UsageLedger:
    """Daily usage per provider/model/purpose in SQLite.  Main thread only."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock
        #: (monotonic time, purpose) of requests recorded in the last hour (this session).
        self._recent: deque[tuple[float, Purpose]] = deque()

    def _day(self) -> str:
        # Local calendar day: "today" resets at the user's midnight, not UTC's.
        return _dt.datetime.fromtimestamp(self._clock.now()).date().isoformat()

    def record(self, provider: str, model: str, purpose: Purpose, usage: Usage,
               cost_usd: float | None) -> None:
        tokens_in = max(0, usage.input_tokens) + max(0, usage.cache_read_tokens) \
            + max(0, usage.cache_write_tokens)
        tokens_out = max(0, usage.output_tokens)
        cost = cost_usd if cost_usd is not None and math.isfinite(cost_usd) and cost_usd > 0 else 0.0
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO ai_usage(day, provider, model, purpose, requests, input_tokens, "
                "output_tokens, cost_usd) VALUES(?, ?, ?, ?, 1, ?, ?, ?) "
                "ON CONFLICT(day, provider, model, purpose) DO UPDATE SET "
                "requests = requests + 1, "
                "input_tokens = input_tokens + excluded.input_tokens, "
                "output_tokens = output_tokens + excluded.output_tokens, "
                "cost_usd = cost_usd + excluded.cost_usd",
                (self._day(), provider, model[:120], Purpose(purpose).value, tokens_in, tokens_out, cost),
            )
        self._recent.append((self._clock.monotonic(), Purpose(purpose)))
        self._prune()

    def today(self) -> UsageSummary:
        rows = self._db.conn.execute(
            "SELECT provider, requests, input_tokens, output_tokens, cost_usd "
            "FROM ai_usage WHERE day = ?",
            (self._day(),),
        ).fetchall()
        requests = input_tokens = output_tokens = 0
        cost = 0.0
        known = True
        for row in rows:
            requests += int(row["requests"])
            input_tokens += int(row["input_tokens"])
            output_tokens += int(row["output_tokens"])
            cost += float(row["cost_usd"])
            spent_tokens = int(row["input_tokens"]) + int(row["output_tokens"]) > 0
            if row["provider"] not in FREE_PROVIDERS and spent_tokens and float(row["cost_usd"]) <= 0:
                known = False
        return UsageSummary(requests, input_tokens, output_tokens, round(cost, 6), known)

    def requests_last_hour(self, purpose: Purpose | None = None) -> int:
        """Requests recorded during the last 60 minutes of this session (of *purpose* only,
        when given)."""
        self._prune()
        if purpose is None:
            return len(self._recent)
        return sum(1 for _at, kind in self._recent if kind == purpose)

    def _prune(self) -> None:
        cutoff = self._clock.monotonic() - _HOUR
        while self._recent and self._recent[0][0] <= cutoff:
            self._recent.popleft()


class BudgetGuard:
    """Refuses a request before it would cross a configured limit."""

    def __init__(self, ledger: UsageLedger, settings: Callable[[], AISettings]) -> None:
        self._ledger = ledger
        self._settings = settings

    def check(self, purpose: Purpose) -> None:
        """Raise :class:`BudgetExceededError` (with a Ukrainian ``user_message``) if over a limit."""
        ai = self._settings()
        today = self._ledger.today()
        if today.cost_usd >= ai.max_cost_per_day_usd:
            raise BudgetExceededError(
                "daily cost limit reached",
                user_message="Денний ліміт витрат на AI вичерпано.",
            )
        if today.requests >= ai.max_requests_per_day:
            raise BudgetExceededError(
                "daily request limit reached",
                user_message="Ліміт запитів до AI на сьогодні вичерпано.",
            )
        limit = self.hourly_limit(purpose, ai)
        if self._ledger.requests_last_hour() >= limit:
            raise BudgetExceededError(
                "hourly request limit reached",
                user_message="Забагато запитів до AI за останню годину — трохи перепочину.",
                hourly=True,
            )
        if purpose == Purpose.VISION and \
                self._ledger.requests_last_hour(Purpose.VISION) >= self.vision_hourly_limit(ai):
            raise BudgetExceededError(
                "hourly vision share reached",
                user_message="Бачення екрана вже використало свою частку запитів до AI на цю годину.",
                hourly=True,
            )

    def daily_exhausted(self) -> bool:
        """True when today's request or cost cap is reached (nothing can run until tomorrow)."""
        ai = self._settings()
        today = self._ledger.today()
        return today.requests >= ai.max_requests_per_day or today.cost_usd >= ai.max_cost_per_day_usd

    @staticmethod
    def hourly_limit(purpose: Purpose, ai: AISettings) -> int:
        limit = ai.max_requests_per_hour
        if purpose == Purpose.CHAT:
            limit += max(1, round(limit * CHAT_HOURLY_OVERDRAFT))
        return limit

    @staticmethod
    def vision_hourly_limit(ai: AISettings) -> int:
        """Screen-vision requests allowed per rolling hour (its share of the hourly cap)."""
        return max(1, int(ai.max_requests_per_hour * VISION_HOURLY_SHARE))
