"""Deterministic exact-span evaluator (sentence-spans-and-labels-v4).

No LLM judge, no fuzzy matching. A prediction matches a gold claim only when
(answer_id, start_char, end_char) are identical AND the prediction's quote
equals the gold quote (quote equality validated first). Each gold span matches
at most one prediction (one-to-one).

Metrics across the whole labeled split:
  G = gold claims, P = accepted predictions,
  M = exact span matches (any label), C = span matches with correct label.
  extraction: P=M/P, R=M/G, F1=2M/(P+G)
  joint:      P=C/P, R=C/G, F1=2C/(P+G)
Zero denominator -> 0.0 when the split has gold claims.
"""
from __future__ import annotations

from .data_loader import GoldClaim
from .schemas import AnswerResult, AnswerStatus, Metrics


def _safe_div(num: int, den: int) -> float:
    return num / den if den else 0.0


def score(
    answer_results: list[AnswerResult],
    gold: list[GoldClaim],
) -> Metrics:
    """Score accepted predictions against gold for a labeled (development) split.

    Only SUCCESS/EMPTY answers contribute accepted predictions; ERROR answers
    contribute none, yet their gold claims remain in G (counted as missed).
    """
    gold_by_key: dict[tuple[str, int, int], GoldClaim] = {
        (g.answer_id, g.start_char, g.end_char): g for g in gold
    }
    G = len(gold)

    accepted: list = []  # (answer_id, Claim)
    for ar in answer_results:
        if ar.status in (AnswerStatus.SUCCESS, AnswerStatus.EMPTY):
            for c in ar.claims:
                accepted.append((ar.answer_id, c))
    P = len(accepted)

    matched_gold_keys: set[tuple[str, int, int]] = set()
    matched_pred_ids: set[str] = set()
    wrong_label_ids: list[str] = []
    M = 0
    C = 0

    for answer_id, c in accepted:
        key = (answer_id, c.start_char, c.end_char)
        g = gold_by_key.get(key)
        if g is None or key in matched_gold_keys:
            continue  # unmatched prediction or gold already taken (1-to-1)
        # Validate quote equality before accepting the match.
        if c.quote != g.quote:
            continue
        matched_gold_keys.add(key)
        matched_pred_ids.add(c.claim_id)
        M += 1
        if c.label == g.expected_label:
            C += 1
        else:
            wrong_label_ids.append(c.claim_id)

    missed_gold_ids = [
        g.claim_id for g in gold
        if (g.answer_id, g.start_char, g.end_char) not in matched_gold_keys
    ]
    unmatched_pred_ids = [
        c.claim_id for (_, c) in accepted if c.claim_id not in matched_pred_ids
    ]

    success = sum(1 for a in answer_results if a.status == AnswerStatus.SUCCESS)
    empty = sum(1 for a in answer_results if a.status == AnswerStatus.EMPTY)
    error = sum(1 for a in answer_results if a.status == AnswerStatus.ERROR)

    return Metrics(
        gold_claims=G,
        accepted_predictions=P,
        span_matches=M,
        correct_label_matches=C,
        extraction_precision=_safe_div(M, P),
        extraction_recall=_safe_div(M, G),
        extraction_f1=_safe_div(2 * M, P + G),
        joint_precision=_safe_div(C, P),
        joint_recall=_safe_div(C, G),
        joint_f1=_safe_div(2 * C, P + G),
        gold_coverage=_safe_div(C, G),
        missed_gold_ids=missed_gold_ids,
        unmatched_prediction_ids=unmatched_pred_ids,
        wrong_label_ids=wrong_label_ids,
        answers_success=success,
        answers_empty=empty,
        answers_error=error,
    )


def score_counts(G: int, P: int, M: int, C: int) -> Metrics:
    """Pure-arithmetic scorer used by the mandatory scorer test."""
    return Metrics(
        gold_claims=G, accepted_predictions=P, span_matches=M,
        correct_label_matches=C,
        extraction_precision=_safe_div(M, P),
        extraction_recall=_safe_div(M, G),
        extraction_f1=_safe_div(2 * M, P + G),
        joint_precision=_safe_div(C, P),
        joint_recall=_safe_div(C, G),
        joint_f1=_safe_div(2 * C, P + G),
        gold_coverage=_safe_div(C, G),
    )


def check_split_metrics(answer_results: list[AnswerResult]) -> Metrics:
    """Metrics for the check split: all gold-dependent fields stay null."""
    success = sum(1 for a in answer_results if a.status == AnswerStatus.SUCCESS)
    empty = sum(1 for a in answer_results if a.status == AnswerStatus.EMPTY)
    error = sum(1 for a in answer_results if a.status == AnswerStatus.ERROR)
    return Metrics(
        answers_success=success, answers_empty=empty, answers_error=error,
    )
