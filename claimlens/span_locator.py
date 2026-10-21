"""Deterministic quote-to-span locator.

Python — not the model — owns character offsets. Given a model-proposed exact
quote, we locate it in the ORIGINAL decoded answer text using zero-based
Unicode code-point positions. Repeated occurrences are resolved by source
order (a monotonic cursor), never by gold spans. Ambiguous / not-found /
boundary-invalid quotes are rejected rather than mis-assigned.

No fuzzy matching, no normalization, no edited source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SENTENCE_ENDINGS = (".", "!", "?")

# An internal sentence terminator (optionally followed by a closing quote/
# bracket) followed by whitespace indicates the quote spans MORE than one
# sentence, i.e. a merged quote. Checked against quote[:-1] so the quote's own
# final punctuation is excluded.
_INTERNAL_BOUNDARY = re.compile(r"[.!?][\"')\]]?\s")


@dataclass(frozen=True)
class LocatedSpan:
    start_char: int
    end_char: int


class SpanError(Exception):
    """Raised when a quote cannot be unambiguously and validly located."""


def _sentence_start_ok(text: str, start: int) -> bool:
    """True if `start` is a real sentence boundary.

    Whitespace immediately before `start` is necessary but NOT sufficient: a
    fragment like "is red." inside "It is red." is also whitespace-preceded.
    We require the first non-space character before the leading whitespace run
    to be a sentence terminator (.!?) or start-of-text.
    """
    i = start - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    return i < 0 or text[i] in ".!?"


def _boundary_ok(text: str, start: int, end: int) -> bool:
    """A located span must be a complete sentence, not a fragment.

    `start` must be a real sentence boundary (see `_sentence_start_ok`) and the
    char after `end` must be whitespace or end-of-text. This rejects
    mid-sentence fragments even when they sit at a word boundary.
    """
    before = text[start - 1] if start > 0 else None
    after = text[end] if end < len(text) else None
    left_ok = (before is None or before.isspace()) and _sentence_start_ok(text, start)
    right_ok = after is None or after.isspace()
    return left_ok and right_ok


def _is_single_sentence(quote: str) -> bool:
    """False if the quote contains an internal sentence boundary (merged)."""
    return _INTERNAL_BOUNDARY.search(quote[:-1]) is None


def locate(text: str, quote: str, cursor: int = 0) -> LocatedSpan:
    """Locate `quote` in `text` at or after `cursor`.

    Raises SpanError if the quote is empty, has surrounding whitespace, does
    not end in sentence punctuation, cannot be found at/after the cursor, or
    fails the sentence-boundary check.
    """
    if not quote:
        raise SpanError("empty quote")
    if quote != quote.strip():
        raise SpanError("quote has leading/trailing whitespace")
    if not quote.endswith(SENTENCE_ENDINGS):
        raise SpanError("quote does not end with sentence punctuation")
    if not _is_single_sentence(quote):
        raise SpanError("quote spans more than one sentence (merged)")

    search_from = cursor
    while True:
        idx = text.find(quote, search_from)
        if idx == -1:
            raise SpanError(
                f"quote not found at/after position {cursor}: {quote[:40]!r}"
            )
        end = idx + len(quote)
        if _boundary_ok(text, idx, end):
            return LocatedSpan(idx, end)
        # Matched substring sits mid-word/mid-sentence; keep searching.
        search_from = idx + 1


def _all_occurrences(text: str, quote: str) -> list[int]:
    """All sentence-boundary-valid start positions of `quote` in `text`."""
    starts: list[int] = []
    frm = 0
    while True:
        idx = text.find(quote, frm)
        if idx == -1:
            break
        if _boundary_ok(text, idx, idx + len(quote)):
            starts.append(idx)
        frm = idx + 1
    return starts


def locate_all(text: str, quotes: list[str]) -> list[LocatedSpan]:
    """Locate each quote at its true position, independent of model ordering.

    Each quote is validated (non-empty, trimmed, sentence-ending) and located
    by its actual occurrence(s) in the source. Unique quotes (the common case —
    gold never repeats a quote within an answer) are placed at their only
    position even if the model returns claims out of order. Genuinely repeated
    quotes are disambiguated by source order (prefer the next unused occurrence
    at/after the previous pick). A quote with no valid occurrence, or whose
    occurrences are all already consumed, raises SpanError (=> answer-level
    error). Ordering/overlap is enforced downstream by the validator.
    """
    spans: list[LocatedSpan] = []
    used: set[int] = set()
    approx_cursor = 0
    for q in quotes:
        if not q:
            raise SpanError("empty quote")
        if q != q.strip():
            raise SpanError("quote has leading/trailing whitespace")
        if not q.endswith(SENTENCE_ENDINGS):
            raise SpanError("quote does not end with sentence punctuation")
        if not _is_single_sentence(q):
            raise SpanError("quote spans more than one sentence (merged)")

        occs = _all_occurrences(text, q)
        if not occs:
            raise SpanError(f"quote not found: {q[:40]!r}")

        choice = next((s for s in occs if s >= approx_cursor and s not in used), None)
        if choice is None:
            choice = next((s for s in occs if s not in used), None)
        if choice is None:
            raise SpanError(f"all occurrences already consumed: {q[:40]!r}")

        used.add(choice)
        approx_cursor = choice + len(q)
        spans.append(LocatedSpan(choice, choice + len(q)))
    return spans
