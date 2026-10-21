"""Independent validation of model output and source spans.

This module turns model-proposed claims into fully validated `Claim` objects,
or raises. It enforces every invariant from the assessment:

- claim_id == answer_id:start_char:end_char
- quote == ai_answer[start:end], no surrounding whitespace, ends a sentence
- offsets are zero-based Unicode code points, start inclusive / end exclusive
- claims ordered by ascending start_char, non-overlapping
- no duplicate IDs / spans
- labels in {supported, contradicted, unknown}
- every fact_id exists in the catalogue (empty list allowed)
- reason is at most two sentences

A failure here is an answer-level error — never silently downgraded to
`unknown`, never a partially-accepted prediction set.
"""
from __future__ import annotations

import re

from .schemas import Claim, Label, ModelClaim
from .span_locator import SpanError, locate_all

VALID_LABELS = {"supported", "contradicted", "unknown"}


class ValidationError(Exception):
    """Answer-level validation failure."""


def _count_sentences(text: str) -> int:
    parts = [p for p in re.split(r"[.!?]+", text.strip()) if p.strip()]
    return len(parts)


def build_claims(
    answer_id: str,
    ai_answer: str,
    model_claims: list[ModelClaim],
    valid_fact_ids: set[str],
) -> list[Claim]:
    """Validate model claims into ordered, non-overlapping `Claim`s or raise."""
    # 1. Locate each quote deterministically (order-resolved by cursor).
    try:
        spans = locate_all(ai_answer, [mc.quote for mc in model_claims])
    except SpanError as e:
        raise ValidationError(f"span location failed: {e}") from e

    claims: list[Claim] = []
    for mc, span in zip(model_claims, spans):
        start, end = span.start_char, span.end_char
        slice_ = ai_answer[start:end]
        if slice_ != mc.quote:
            raise ValidationError(
                f"quote != slice for {answer_id} at {start}:{end}"
            )
        if mc.label not in VALID_LABELS:
            raise ValidationError(f"invalid label: {mc.label!r}")
        for fid in mc.fact_ids:
            if fid not in valid_fact_ids:
                raise ValidationError(f"invented fact_id: {fid!r}")
        if _count_sentences(mc.reason) > 2:
            raise ValidationError("reason longer than two sentences")

        claims.append(Claim(
            claim_id=f"{answer_id}:{start}:{end}",
            start_char=start, end_char=end, quote=slice_,
            label=mc.label,  # type: ignore[arg-type]
            fact_ids=list(mc.fact_ids), reason=mc.reason.strip(),
        ))

    # The model's ordering is not authoritative; offsets are. Emit claims in
    # ascending start_char order, then enforce no duplicates / no overlaps.
    claims.sort(key=lambda c: c.start_char)
    seen_ids: set[str] = set()
    seen_spans: set[tuple[int, int]] = set()
    prev_end = -1
    for c in claims:
        if c.claim_id in seen_ids:
            raise ValidationError(f"duplicate claim_id: {c.claim_id}")
        if (c.start_char, c.end_char) in seen_spans:
            raise ValidationError(f"duplicate span: {c.start_char}:{c.end_char}")
        if c.start_char < prev_end:
            raise ValidationError("overlapping claims")
        seen_ids.add(c.claim_id)
        seen_spans.add((c.start_char, c.end_char))
        prev_end = c.end_char

    return claims
