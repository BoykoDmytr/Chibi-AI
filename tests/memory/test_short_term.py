"""ShortTermMemory: the current conversation, never persisted."""

import pytest

from chibi.ai.types import ChatTurn, Role
from chibi.memory.longterm import MAX_TURN_CHARS, ShortTermMemory


def test_turns_returns_the_last_n_in_order():
    memory = ShortTermMemory()
    memory.add(Role.USER, "привіт")
    memory.add(Role.ASSISTANT, "Привіт-привіт!")
    memory.add(Role.USER, "як справи?")
    assert memory.turns(10) == (
        ChatTurn(Role.USER, "привіт"),
        ChatTurn(Role.ASSISTANT, "Привіт-привіт!"),
        ChatTurn(Role.USER, "як справи?"),
    )
    assert memory.turns(1) == (ChatTurn(Role.USER, "як справи?"),)


def test_a_leading_assistant_turn_is_dropped():
    memory = ShortTermMemory()
    memory.add(Role.USER, "раз")
    memory.add(Role.ASSISTANT, "два")
    memory.add(Role.USER, "три")
    memory.add(Role.ASSISTANT, "чотири")
    # the last 3 turns would start with "два" (assistant)
    assert [t.text for t in memory.turns(3)] == ["три", "чотири"]
    assert memory.turns(4)[0].role is Role.USER


def test_only_assistant_turns_give_nothing():
    memory = ShortTermMemory()
    memory.add(Role.ASSISTANT, "Я тут!")
    assert memory.turns(5) == ()


def test_non_positive_limit():
    memory = ShortTermMemory()
    memory.add(Role.USER, "привіт")
    assert memory.turns(0) == ()
    assert memory.turns(-3) == ()


def test_capacity_drops_the_oldest():
    memory = ShortTermMemory(max_turns=4)
    for i in range(10):
        memory.add(Role.USER if i % 2 == 0 else Role.ASSISTANT, f"репліка {i}")
    assert len(memory) == 4
    assert [t.text for t in memory.turns(100)] == ["репліка 6", "репліка 7", "репліка 8", "репліка 9"]


def test_blank_text_is_ignored_and_text_is_stripped():
    memory = ShortTermMemory()
    memory.add(Role.USER, "   ")
    memory.add(Role.ASSISTANT, "")
    memory.add(Role.USER, None)  # type: ignore[arg-type]
    memory.add(Role.USER, "  привіт  ")
    assert memory.turns(5) == (ChatTurn(Role.USER, "привіт"),)


def test_very_long_turns_are_capped():
    memory = ShortTermMemory()
    memory.add(Role.USER, "а" * (MAX_TURN_CHARS * 3))
    text = memory.turns(1)[0].text
    assert len(text) <= MAX_TURN_CHARS
    assert text.endswith("…")


def test_role_strings_are_accepted_and_unknown_roles_rejected():
    memory = ShortTermMemory()
    memory.add("user", "привіт")  # type: ignore[arg-type]
    assert memory.turns(1)[0].role is Role.USER
    with pytest.raises(ValueError):
        memory.add("system", "ти - бот")  # type: ignore[arg-type]


def test_clear():
    memory = ShortTermMemory()
    memory.add(Role.USER, "привіт")
    memory.clear()
    assert len(memory) == 0
    assert memory.turns(5) == ()


def test_invalid_capacity():
    with pytest.raises(ValueError):
        ShortTermMemory(max_turns=0)
