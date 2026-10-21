"""Claim-validation tests."""
import pytest

from claimlens.schemas import ModelClaim
from claimlens.validator import ValidationError, build_claims

TEXT = "The Mini is cheap. The Mini is waterproof. It weighs 300 g."
FACTS = {"cleartrail.mini.price_2026", "cleartrail.mini.water_rating"}


def mc(quote, label="unknown", fact_ids=None, reason="r."):
    return ModelClaim(quote=quote, label=label, fact_ids=fact_ids or [], reason=reason)


def test_valid_claims_build():
    claims = build_claims("A01", TEXT, [
        mc("The Mini is cheap.", "supported", ["cleartrail.mini.price_2026"]),
        mc("The Mini is waterproof.", "contradicted", ["cleartrail.mini.water_rating"]),
    ], FACTS)
    assert [c.claim_id for c in claims] == ["A01:0:18", "A01:19:42"]
    assert all(TEXT[c.start_char:c.end_char] == c.quote for c in claims)


def test_empty_claim_list_valid():
    assert build_claims("A01", TEXT, [], FACTS) == []


def test_invalid_label_rejected():
    # Strict schema rejects an unsupported label at construction time; if a
    # label somehow bypassed that, build_claims re-checks VALID_LABELS.
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        mc("The Mini is cheap.", "maybe")
    bad = mc("The Mini is cheap.")
    object.__setattr__(bad, "label", "maybe")
    with pytest.raises(ValidationError):
        build_claims("A01", TEXT, [bad], FACTS)


def test_invented_fact_id_rejected():
    with pytest.raises(ValidationError):
        build_claims("A01", TEXT, [mc("The Mini is cheap.", "supported", ["nope.fact"])], FACTS)


def test_duplicate_span_rejected():
    with pytest.raises(ValidationError):
        build_claims("A01", TEXT, [
            mc("The Mini is cheap."), mc("The Mini is cheap."),
        ], FACTS)


def test_out_of_order_model_output_is_sorted():
    # Model returns claims out of source order; locator places them correctly
    # and the validator emits them sorted by start_char.
    claims = build_claims("A01", TEXT, [
        mc("The Mini is waterproof."), mc("The Mini is cheap."),
    ], FACTS)
    assert [c.start_char for c in claims] == sorted(c.start_char for c in claims)
    assert claims[0].quote == "The Mini is cheap."


def test_overlapping_spans_rejected():
    text = "A. B."
    with pytest.raises(ValidationError):
        build_claims("A01", text, [mc("A."), mc("A. B.")], FACTS)


def test_reason_too_long_rejected():
    with pytest.raises(ValidationError):
        build_claims("A01", TEXT, [
            mc("The Mini is cheap.", reason="One. Two. Three."),
        ], FACTS)


def test_quote_not_in_text_rejected():
    with pytest.raises(ValidationError):
        build_claims("A01", TEXT, [mc("The Max is loud.")], FACTS)


def test_unicode_offsets_validated():
    text = "日本語 is fun. The Mini is cheap."
    claims = build_claims("A02", text, [mc("The Mini is cheap.", "unknown")], FACTS)
    assert text[claims[0].start_char:claims[0].end_char] == "The Mini is cheap."
