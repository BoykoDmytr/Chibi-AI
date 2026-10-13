"""Reasoning blocks (<think>...</think>) of local models never reach her words."""

import random

import pytest

from chibi.ai.providers.reasoning import ReasoningFilter, strip_reasoning

ANSWER = "<<state=happy;action=none>> Привіт! Як справи?"


@pytest.mark.parametrize("raw,expected", [
    (f"<think>The user greets me. I should answer in Ukrainian.</think>\n\n{ANSWER}", ANSWER),
    (f"<THINK>\nx\n</THINK>{ANSWER}", ANSWER),
    (f"<thinking>a</thinking>{ANSWER}", ANSWER),
    (f"<reasoning>a</reasoning>\n{ANSWER}", ANSWER),
    (ANSWER, ANSWER),
    ("a < b, а 3<4 і <3 сердечко", "a < b, а 3<4 і <3 сердечко"),
    # The template opened the block in the prompt: only the closing tag is in the answer.
    (f"Okay, let me think.\n</think>\n\n{ANSWER}", ANSWER),
    # Cut off by the token limit while still reasoning: nothing usable.
    ("<think>The user wants a joke. Let me", ""),
    (f"{ANSWER}<think>afterthought", ANSWER),
    ('<think>look at the frame</think>\n```json\n{"scene": "combat"}\n```',
     '```json\n{"scene": "combat"}\n```'),
])
def test_strip_reasoning(raw, expected):
    assert strip_reasoning(raw) == expected


def stream(raw: str, seed: int) -> str:
    rng = random.Random(seed)
    f = ReasoningFilter()
    out, i = [], 0
    while i < len(raw):
        n = rng.randint(1, 7)
        out.append(f.feed(raw[i:i + n]))
        i += n
    out.append(f.finish())
    return "".join(out)


@pytest.mark.parametrize("raw,expected", [
    (f"<think>The user greets me.</think>\n\n{ANSWER}", ANSWER),
    (f"<Think>a < b</Think>{ANSWER}", ANSWER),
    (ANSWER, ANSWER),
    ("a < b, а 3<4 і <3 сердечко", "a < b, а 3<4 і <3 сердечко"),
    ("<think>unterminated reasoning", ""),
    ("Текст і <thi", "Текст і <thi"),  # a held-back partial tag is text after all
    (f"{ANSWER}<think>afterthought</think>", ANSWER),
])
def test_stream_filter_matches_the_complete_answer_for_any_chunking(raw, expected):
    for seed in range(40):
        assert stream(raw, seed) == expected, seed


def test_stream_never_shows_a_tag_or_reasoning_text():
    raw = f"<think>SECRET reasoning with <b>markup</b></think>\n{ANSWER}"
    for seed in range(40):
        rng = random.Random(seed)
        f = ReasoningFilter()
        seen, i = [], 0
        while i < len(raw):
            n = rng.randint(1, 4)
            seen.append(f.feed(raw[i:i + n]))
            i += n
        seen.append(f.finish())
        assert all("SECRET" not in s and "think" not in s.lower() for s in seen)


def test_lone_closing_tag_in_a_stream_drops_the_reasoning_still_held():
    f = ReasoningFilter()
    shown = f.feed("let me think</thi") + f.feed("nk>\n\n") + f.feed(ANSWER) + f.finish()
    # "let me think" was passed on before the tag could be recognised; the final
    # answer (strip_reasoning) replaces it.  The tag itself never shows.
    assert "</think>" not in shown and shown.endswith(ANSWER)
