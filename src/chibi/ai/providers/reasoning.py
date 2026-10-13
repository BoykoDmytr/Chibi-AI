"""Reasoning ("thinking") blocks from local models, removed before anyone sees them.

Many models people run locally - Qwen3, DeepSeek-R1 distills, QwQ - write
their reasoning into the answer itself as ``<think>...</think>``, and
OpenAI-compatible servers (llama.cpp, LM Studio) and Ollama often pass it on in
``content``.  Shown in her bubble, voiced, or parsed for the control line, it
would be English chain-of-thought in place of her words.

* :func:`strip_reasoning` cleans a complete answer.  It is the source of
  truth: the final text of every OpenAI-compatible / Ollama result goes
  through it.
* :class:`ReasoningFilter` does the same for a stream, chunk by chunk, so the
  chat never shows a reasoning block while it is being written.  A tag split
  across chunks is held back until it is complete.

Some chat templates put the opening tag into the prompt, so the model's own
text starts inside the block and only the closing tag appears.  The complete
answer handles that (everything up to the last closing tag goes); a stream
cannot take back text it already passed on, so live text may briefly show such
reasoning before the final answer replaces it.
"""

from __future__ import annotations

import re

#: Tag names models use for reasoning, matched case-insensitively.
_TAGS = ("think", "thinking", "reasoning")
_NAMES = "|".join(_TAGS)
_OPEN_RE = re.compile(rf"<(?:{_NAMES})>", re.IGNORECASE)
_CLOSE_RE = re.compile(rf"</(?:{_NAMES})>", re.IGNORECASE)
_BLOCK_RE = re.compile(rf"<({_NAMES})>.*?</\1>", re.IGNORECASE | re.DOTALL)
_OPEN_TAGS = tuple(f"<{t}>" for t in _TAGS)
_CLOSE_TAGS = tuple(f"</{t}>" for t in _TAGS)


def strip_reasoning(text: str) -> str:
    """*text* without reasoning blocks (also an unterminated one, cut off by the
    token limit, and a lone closing tag whose opening tag was in the prompt)."""
    if "<" not in text:
        return text
    had_block = bool(_OPEN_RE.search(text) or _CLOSE_RE.search(text))
    if not had_block:
        return text
    out = _BLOCK_RE.sub("", text)
    closes = list(_CLOSE_RE.finditer(out))
    if closes:  # the opening tag was in the prompt: everything before the last close is reasoning
        out = out[closes[-1].end():]
    opened = _OPEN_RE.search(out)
    if opened:  # cut off mid-reasoning: the rest is not an answer
        out = out[:opened.start()]
    return out.strip()


def _partial_suffix(text: str, tags: tuple[str, ...]) -> int:
    """Length of the longest end of *text* that could be the start of one of *tags*."""
    lowered = text[-12:].lower()
    for size in range(min(len(lowered), max(len(t) for t in tags) - 1), 0, -1):
        tail = lowered[-size:]
        if any(t.startswith(tail) for t in tags):
            return size
    return 0


class ReasoningFilter:
    """Streaming counterpart of :func:`strip_reasoning`: ``feed()`` each chunk and
    pass on what it returns; ``finish()`` returns what was held back at the end."""

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False
        self._started = False  # some visible text has been passed on

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        visible: list[str] = []
        while self._buffer:
            if self._inside:
                close = _CLOSE_RE.search(self._buffer)
                if close is None:
                    keep = _partial_suffix(self._buffer, _CLOSE_TAGS)
                    self._buffer = self._buffer[len(self._buffer) - keep:] if keep else ""
                    break
                self._buffer = self._buffer[close.end():]
                self._inside = False
                continue
            opened = _OPEN_RE.search(self._buffer)
            closed = _CLOSE_RE.search(self._buffer)
            if closed is not None and (opened is None or closed.start() < opened.start()):
                # A lone closing tag: the opening one was in the prompt, so what came
                # before it was reasoning.  The part already passed on can't be taken
                # back (the final answer replaces it); the rest of it goes now.
                self._buffer = self._buffer[closed.end():]
                visible.clear()
                self._started = False
                continue
            if opened is not None:
                visible.append(self._buffer[:opened.start()])
                self._buffer = self._buffer[opened.end():]
                self._inside = True
                continue
            keep = _partial_suffix(self._buffer, _OPEN_TAGS + _CLOSE_TAGS)
            cut = len(self._buffer) - keep
            visible.append(self._buffer[:cut])
            self._buffer = self._buffer[cut:]
            break
        return self._emit("".join(visible))

    def finish(self) -> str:
        """End of the stream: a held-back partial tag was text after all; an
        unterminated block is dropped."""
        rest = "" if self._inside else self._buffer
        self._buffer = ""
        return self._emit(rest)

    def _emit(self, text: str) -> str:
        if not self._started:
            text = text.lstrip()  # answers usually start with blank lines after the block
            if not text:
                return ""
            self._started = True
        return text
