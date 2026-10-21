"""Evaluator / scorer tests, including the mandatory arithmetic example."""
import pytest

from claimlens.data_loader import GoldClaim
from claimlens.evaluator import score, score_counts
from claimlens.schemas import AnswerResult, AnswerStatus, Claim, ErrorType


def gold(answer_id, start, end, quote, label):
    return GoldClaim(answer_id, f"{answer_id}:{start}:{end}", start, end, quote,
                     label, [], "", [])


def claim(answer_id, start, end, quote, label):
    return Claim(claim_id=f"{answer_id}:{start}:{end}", start_char=start,
                 end_char=end, quote=quote, label=label, fact_ids=[], reason="")


def ok(answer_id, claims):
    status = AnswerStatus.EMPTY if not claims else AnswerStatus.SUCCESS
    return AnswerResult(answer_id=answer_id, status=status, claims=claims)


def test_mandatory_arithmetic():
    m = score_counts(G=10, P=10, M=8, C=7)
    assert m.extraction_precision == 0.80
    assert m.extraction_recall == 0.80
    assert m.extraction_f1 == 0.80
    assert m.joint_precision == 0.70
    assert m.joint_recall == 0.70
    assert m.joint_f1 == 0.70


def test_perfect_predictions():
    g = [gold("A01", 0, 10, "abcdefghi.", "supported")]
    r = [ok("A01", [claim("A01", 0, 10, "abcdefghi.", "supported")])]
    m = score(r, g)
    assert (m.span_matches, m.correct_label_matches) == (1, 1)
    assert m.joint_f1 == 1.0


def test_missed_claim_counts_against_recall():
    g = [gold("A01", 0, 10, "abcdefghi.", "supported"),
         gold("A01", 11, 20, "jklmnopq.", "unknown")]
    r = [ok("A01", [claim("A01", 0, 10, "abcdefghi.", "supported")])]
    m = score(r, g)
    assert m.span_matches == 1 and m.gold_claims == 2
    assert m.extraction_recall == 0.5
    assert m.missed_gold_ids == ["A01:11:20"]


def test_extra_advice_hurts_precision():
    g = [gold("A01", 0, 10, "abcdefghi.", "supported")]
    r = [ok("A01", [claim("A01", 0, 10, "abcdefghi.", "supported"),
                    claim("A01", 20, 30, "extrააdvic.", "unknown")])]
    m = score(r, g)
    assert m.accepted_predictions == 2 and m.span_matches == 1
    assert m.extraction_precision == 0.5
    assert m.unmatched_prediction_ids == ["A01:20:30"]


def test_correct_span_wrong_label():
    g = [gold("A01", 0, 10, "abcdefghi.", "supported")]
    r = [ok("A01", [claim("A01", 0, 10, "abcdefghi.", "contradicted")])]
    m = score(r, g)
    assert m.span_matches == 1 and m.correct_label_matches == 0
    assert m.wrong_label_ids == ["A01:0:10"]
    assert m.extraction_f1 == 1.0 and m.joint_f1 == 0.0


def test_provider_error_answer_keeps_gold_as_missed():
    g = [gold("A01", 0, 10, "abcdefghi.", "supported")]
    r = [AnswerResult(answer_id="A01", status=AnswerStatus.ERROR,
                      error_type=ErrorType.PROVIDER_ERROR, error_message="x")]
    m = score(r, g)
    assert m.accepted_predictions == 0 and m.gold_claims == 1
    assert m.extraction_recall == 0.0 and m.answers_error == 1


def test_valid_empty_prediction():
    g = [gold("A01", 0, 10, "abcdefghi.", "supported")]
    r = [ok("A01", [])]
    m = score(r, g)
    assert m.accepted_predictions == 0 and m.answers_empty == 1
    assert m.extraction_f1 == 0.0


def test_empty_split_has_null_quality_metrics():
    # An empty scheduled split (no gold) -> null quality metrics, not 0.0.
    m = score([], [])
    assert m.gold_claims is None
    assert m.joint_f1 is None
    assert m.extraction_f1 is None
    assert m.span_matches is None


def test_zero_predictions_with_gold_present_is_zero_not_null():
    # Distinct from an empty split: gold exists but nothing was predicted.
    g = [gold("A01", 0, 10, "abcdefghi.", "supported")]
    m = score([ok("A01", [])], g)
    assert m.gold_claims == 1
    assert m.joint_f1 == 0.0  # zero, not null
    assert m.extraction_f1 == 0.0


def test_duplicate_answer_outcomes_do_not_double_count_gold():
    # Two outcome rows for the same answer must not inflate matches: one-to-one
    # matching keeps a gold span matched at most once.
    g = [gold("A01", 0, 10, "abcdefghi.", "supported")]
    dup = ok("A01", [claim("A01", 0, 10, "abcdefghi.", "supported")])
    m = score([dup, dup], g)
    assert m.span_matches == 1
    assert m.accepted_predictions == 2  # duplicate prediction counted in P
    assert m.extraction_precision == 0.5


def test_missing_answer_outcome_counts_gold_as_missed():
    # Gold for A02 with no outcome row for A02 -> its claims remain missed.
    g = [gold("A01", 0, 10, "abcdefghi.", "supported"),
         gold("A02", 0, 8, "jklmnop.", "unknown")]
    m = score([ok("A01", [claim("A01", 0, 10, "abcdefghi.", "supported")])], g)
    assert m.gold_claims == 2
    assert m.span_matches == 1
    assert "A02:0:8" in m.missed_gold_ids


def test_one_to_one_matching():
    # Two predictions cannot both match a single gold span.
    g = [gold("A01", 0, 10, "abcdefghi.", "supported")]
    r = [ok("A01", [claim("A01", 0, 10, "abcdefghi.", "supported")]),
         ok("A01", [claim("A01", 0, 10, "abcdefghi.", "supported")])]
    # duplicate answer rows would normally be rejected upstream; here we only
    # assert the scorer never double-counts a gold span.
    m = score(r, g)
    assert m.span_matches == 1
