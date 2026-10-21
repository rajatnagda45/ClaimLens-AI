"""Deterministic source spans with conservative English sentence boundaries.

No gold, normalization, fuzzy matching, or model-supplied offsets are used.
Abbreviations are heuristic; unresolved repeated quotations fail explicitly.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_CLOSERS = "\"'”’)]}"
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "e.g", "i.e"}
_TOKEN = re.compile(r"([A-Za-z.]+)\.$")


@dataclass(frozen=True)
class LocatedSpan:
    start_char: int
    end_char: int


class SpanError(Exception):
    """A quotation is absent, incomplete, merged, or ambiguous."""


def sentence_spans(text: str) -> list[LocatedSpan]:
    """Return original offsets, including a final unfinished clause if present.

    Decimal points and common abbreviations do not terminate sentences. Closing
    quotes/brackets belong to the preceding sentence. This is deliberately a
    small English heuristic, not a claim of universal linguistic segmentation.
    """
    spans = []
    start = 0
    while start < len(text) and text[start].isspace():
        start += 1
    i = start
    while i < len(text):
        char = text[i]
        if char not in ".!?":
            i += 1
            continue
        end = i + 1
        while end < len(text) and text[end] in ".!?" + _CLOSERS:
            end += 1
        if end < len(text) and not text[end].isspace():
            i += 1
            continue
        following = end
        while following < len(text) and text[following].isspace():
            following += 1
        if char == "." and following < len(text):
            token_match = _TOKEN.search(text[start:i + 1])
            token = token_match.group(1) if token_match else ""
            next_word = re.match(r"[A-Z][a-z]+", text[following:])
            if token.lower() in _ABBREVIATIONS or (len(token) == 1 and token.isupper() and next_word):
                i += 1
                continue
        # Quoted speech or a parenthetical sentence followed by a continuation.
        if end > i + 1 and following < len(text) and text[following].islower():
            i += 1
            continue
        spans.append(LocatedSpan(start, end))
        start = following
        i = following
    if start < len(text):
        end = len(text.rstrip())
        if end > start:
            spans.append(LocatedSpan(start, end))
    return spans


def _validate_quote(quote: str) -> None:
    if not quote or quote != quote.strip():
        raise SpanError("quote is empty or has surrounding whitespace")
    if not quote.rstrip(_CLOSERS).endswith((".", "!", "?")):
        raise SpanError("quote does not end with sentence punctuation")


def _occurrences(text: str, quote: str) -> list[LocatedSpan]:
    _validate_quote(quote)
    return [s for s in sentence_spans(text) if text[s.start_char:s.end_char] == quote]


def locate(text: str, quote: str, cursor: int = 0) -> LocatedSpan:
    candidates = [s for s in _occurrences(text, quote) if s.start_char >= cursor]
    if not candidates:
        raise SpanError("quote is not a complete source sentence")
    if len(candidates) != 1:
        raise SpanError("ambiguous repeated quote; supply every occurrence in source order")
    return candidates[0]


def locate_all(text: str, quotes: list[str]) -> list[LocatedSpan]:
    counts = Counter(quotes)
    occurrences = {q: _occurrences(text, q) for q in counts}
    for q, count in counts.items():
        if len(occurrences[q]) != count:
            raise SpanError("quote missing, duplicated, incomplete, merged, or ambiguous")
    # All repeated occurrences must be represented. Assign identical quotations
    # in source order, while allowing distinct quotations to arrive out of order.
    used = Counter()
    result = []
    for q in quotes:
        result.append(occurrences[q][used[q]])
        used[q] += 1
    return result
