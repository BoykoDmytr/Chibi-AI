from __future__ import annotations

import datetime as dt

import pytest

from chibi.ai.budget import (
    PRICES_PER_MTOK,
    BudgetGuard,
    UsageLedger,
    UsageSummary,
    estimate_cost,
)
from chibi.ai.types import BudgetExceededError, Purpose, Usage
from chibi.core.config import AISettings


def local_epoch(year, month, day, hour, minute=0) -> float:
    return dt.datetime(year, month, day, hour, minute).timestamp()


# -- prices -----------------------------------------------------------------------------------


def test_price_table_matches_published_prices():
    assert PRICES_PER_MTOK["claude-opus-5-5"] == (4.0, 20.0)
    assert PRICES_PER_MTOK["claude-sonnet-5-5"] == (2.0, 10.0)
    assert PRICES_PER_MTOK["claude-haiku-5-5"] == (0.10, 0.50)
    assert PRICES_PER_MTOK["claude-fable-5-1"] == (10.0, 50.0)
    assert PRICES_PER_MTOK["claude-opus-5"] == (5.0, 25.0)
    assert PRICES_PER_MTOK["claude-sonnet-5"] == (2.0, 10.0)
    for model in ("claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6"):
        assert PRICES_PER_MTOK[model] == (5.0, 25.0)
    assert PRICES_PER_MTOK["claude-sonnet-4-6"] == (3.0, 15.0)
    assert PRICES_PER_MTOK["claude-haiku-4-5"] == (1.0, 5.0)
    assert all("2026" not in model for model in PRICES_PER_MTOK)  # never date-suffixed ids
    with pytest.raises(TypeError):
        PRICES_PER_MTOK["x"] = (1, 1)  # type: ignore[index]


def test_estimate_cost_claude():
    ai = AISettings()
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000)
    assert estimate_cost("anthropic", "claude-opus-5-5", usage, ai) == pytest.approx(4.0 + 2.0)
    assert estimate_cost("anthropic", "claude-haiku-4-5", Usage(input_tokens=2000, output_tokens=1000), ai) \
        == pytest.approx((2000 * 1 + 1000 * 5) / 1e6)


def test_estimate_cost_counts_cache_reads_and_writes():
    ai = AISettings()
    cached = Usage(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
    # opus 5.5: reads $0.20, writes 1.25 x $4
    assert estimate_cost("anthropic", "claude-opus-5-5", cached, ai) == pytest.approx(0.20 + 5.0)
    # sonnet 4.6: reads 0.1 x $3
    assert estimate_cost("anthropic", "claude-sonnet-4-6", cached, ai) == pytest.approx(0.3 + 3.75)


def test_estimate_cost_unknown_and_free():
    ai = AISettings()
    usage = Usage(input_tokens=500, output_tokens=500)
    assert estimate_cost("anthropic", "claude-unknown-9", usage, ai) is None
    assert estimate_cost("openai", "gpt-x", usage, ai) is None  # no price configured
    assert estimate_cost("ollama", "gemma3", usage, ai) == 0.0
    assert estimate_cost("fake", "fake-model", usage, ai) == 0.0
    assert estimate_cost("mystery", "m", usage, ai) is None


def test_estimate_cost_openai_with_user_prices():
    ai = AISettings(openai_price_input_per_mtok=2.5, openai_price_output_per_mtok=10.0)
    usage = Usage(input_tokens=800, output_tokens=100, cache_read_tokens=200)
    assert estimate_cost("openai", "gpt-x", usage, ai) == pytest.approx((1000 * 2.5 + 100 * 10) / 1e6)


# -- ledger -----------------------------------------------------------------------------------


def test_ledger_upserts_per_day_provider_model_purpose(db, clock):
    ledger = UsageLedger(db, clock)
    ledger.record("anthropic", "claude-opus-5-5", Purpose.CHAT, Usage(100, 20, 50, 10), 0.01)
    ledger.record("anthropic", "claude-opus-5-5", Purpose.CHAT, Usage(100, 20), 0.02)
    ledger.record("anthropic", "claude-opus-5-5", Purpose.REACTION, Usage(10, 5), 0.001)
    rows = db.conn.execute("SELECT * FROM ai_usage ORDER BY purpose").fetchall()
    assert [(r["purpose"], r["requests"]) for r in rows] == [("chat", 2), ("reaction", 1)]
    chat = rows[0]
    assert chat["input_tokens"] == 100 + 50 + 10 + 100
    assert chat["output_tokens"] == 40
    assert chat["cost_usd"] == pytest.approx(0.03)
    today = ledger.today()
    assert today == UsageSummary(requests=3, input_tokens=270, output_tokens=45,
                                 cost_usd=pytest.approx(0.031), cost_known=True)


def test_ledger_day_is_local_and_resets_at_midnight(db, clock):
    ledger = UsageLedger(db, clock)
    clock.set_now(local_epoch(2026, 5, 4, 23, 50))
    ledger.record("ollama", "gemma3", Purpose.CHAT, Usage(10, 10), 0.0)
    assert db.conn.execute("SELECT day FROM ai_usage").fetchone()["day"] == "2026-05-04"
    clock.set_now(local_epoch(2026, 5, 5, 0, 10))
    assert ledger.today().requests == 0
    ledger.record("ollama", "gemma3", Purpose.CHAT, Usage(10, 10), 0.0)
    assert ledger.today().requests == 1


def test_ledger_cost_known_flags_unpriced_rows(db, clock):
    ledger = UsageLedger(db, clock)
    ledger.record("ollama", "gemma3", Purpose.CHAT, Usage(100, 100), 0.0)
    assert ledger.today().cost_known is True
    ledger.record("openai", "lmstudio-model", Purpose.CHAT, Usage(0, 0), None)  # nothing spent
    assert ledger.today().cost_known is True
    ledger.record("openai", "lmstudio-model", Purpose.CHAT, Usage(100, 100), None)
    assert ledger.today().cost_known is False


def test_ledger_ignores_bogus_costs(db, clock):
    ledger = UsageLedger(db, clock)
    ledger.record("anthropic", "claude-opus-5-5", Purpose.CHAT, Usage(1, 1), float("nan"))
    ledger.record("anthropic", "claude-opus-5-5", Purpose.CHAT, Usage(1, 1), -5.0)
    assert ledger.today().cost_usd == 0.0


def test_requests_last_hour_is_rolling(db, clock):
    clock.set_now(local_epoch(2026, 5, 4, 10))
    ledger = UsageLedger(db, clock)
    for _ in range(3):
        ledger.record("ollama", "m", Purpose.REACTION, Usage(), 0.0)
        clock.advance(1200)
    assert ledger.requests_last_hour() == 2  # the first one is 60 min old now
    clock.advance(2400)
    assert ledger.requests_last_hour() == 0
    assert ledger.today().requests == 3  # the daily count is persistent, the hourly one rolls


# -- guard ------------------------------------------------------------------------------------


def guard_for(db, clock, **ai_kwargs):
    ai = AISettings(**ai_kwargs)
    ledger = UsageLedger(db, clock)
    return BudgetGuard(ledger, lambda: ai), ledger


def test_guard_daily_request_cap(db, clock):
    guard, ledger = guard_for(db, clock, max_requests_per_day=2, max_requests_per_hour=100)
    guard.check(Purpose.REACTION)
    ledger.record("ollama", "m", Purpose.REACTION, Usage(), 0.0)
    ledger.record("ollama", "m", Purpose.REACTION, Usage(), 0.0)
    for purpose in Purpose:
        with pytest.raises(BudgetExceededError) as err:
            guard.check(purpose)
        assert "сьогодні" in err.value.user_message
    assert guard.daily_exhausted()


def test_guard_hourly_cap_with_chat_overdraft(db, clock):
    guard, ledger = guard_for(db, clock, max_requests_per_hour=10, max_requests_per_day=1000)
    for _ in range(10):
        ledger.record("ollama", "m", Purpose.REACTION, Usage(), 0.0)
    with pytest.raises(BudgetExceededError) as err:
        guard.check(Purpose.REACTION)
    assert "годину" in err.value.user_message
    with pytest.raises(BudgetExceededError):
        guard.check(Purpose.INITIATIVE)
    guard.check(Purpose.CHAT)  # the user is waiting: 20 % extra
    ledger.record("ollama", "m", Purpose.CHAT, Usage(), 0.0)
    guard.check(Purpose.CHAT)
    ledger.record("ollama", "m", Purpose.CHAT, Usage(), 0.0)
    with pytest.raises(BudgetExceededError):
        guard.check(Purpose.CHAT)  # 12 = 10 + 20 %
    assert not guard.daily_exhausted()
    clock.advance(3601)
    guard.check(Purpose.REACTION)


def test_guard_small_hourly_cap_still_lets_one_chat_through(db, clock):
    guard, ledger = guard_for(db, clock, max_requests_per_hour=1)
    ledger.record("ollama", "m", Purpose.CHAT, Usage(), 0.0)
    guard.check(Purpose.CHAT)
    with pytest.raises(BudgetExceededError):
        guard.check(Purpose.REACTION)


def test_guard_cost_cap_applies_to_chat_too(db, clock):
    guard, ledger = guard_for(db, clock, max_cost_per_day_usd=0.05)
    ledger.record("anthropic", "claude-opus-5-5", Purpose.MEMORY, Usage(1000, 1000), 0.049)
    guard.check(Purpose.CHAT)
    ledger.record("anthropic", "claude-opus-5-5", Purpose.MEMORY, Usage(10, 10), 0.002)
    with pytest.raises(BudgetExceededError) as err:
        guard.check(Purpose.CHAT)
    assert "витрат" in err.value.user_message
    assert guard.daily_exhausted()


def test_guard_reads_settings_live(db, clock):
    ai = AISettings(max_requests_per_day=1)
    ledger = UsageLedger(db, clock)
    guard = BudgetGuard(ledger, lambda: ai)
    ledger.record("ollama", "m", Purpose.CHAT, Usage(), 0.0)
    with pytest.raises(BudgetExceededError):
        guard.check(Purpose.CHAT)
    ai.max_requests_per_day = 5
    guard.check(Purpose.CHAT)
