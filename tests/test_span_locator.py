"""Quote-to-span locator tests."""
import pytest

from claimlens.span_locator import SpanError, locate, locate_all


TEXT = "The Mini is cheap. The Mini is waterproof. Buy it now."


def test_basic_offsets_and_slice():
    span = locate(TEXT, "The Mini is cheap.")
    assert TEXT[span.start_char:span.end_char] == "The Mini is cheap."
    assert span.start_char == 0


def test_unicode_code_point_offsets():
    text = "café is nice. 日本語 rocks."
    span = locate(text, "日本語 rocks.")
    assert text[span.start_char:span.end_char] == "日本語 rocks."
    # offset is code points, not bytes
    assert span.start_char == text.index("日本語")


def test_repeated_quote_resolved_by_order():
    text = "It is red. It is red. Done."
    spans = locate_all(text, ["It is red.", "It is red."])
    assert spans[0].start_char == 0
    assert spans[1].start_char == 11
    assert spans[0].end_char <= spans[1].start_char


def test_leading_trailing_whitespace_rejected():
    with pytest.raises(SpanError):
        locate(TEXT, " The Mini is cheap.")


def test_missing_punctuation_rejected():
    with pytest.raises(SpanError):
        locate(TEXT, "The Mini is cheap")


def test_not_found_rejected():
    with pytest.raises(SpanError):
        locate(TEXT, "The Max is loud.")


def test_boundary_rejects_midsentence_fragment():
    text = "Supercalifragilistic is a word."
    # "califragilistic" is a substring but not sentence-bounded -> not found
    with pytest.raises(SpanError):
        locate(text, "califragilistic.")


def test_fragment_at_word_boundary_rejected():
    # "is red." sits at a word boundary (space before) but is NOT a sentence
    # start -> must be rejected, not silently located.
    text = "It is red. It is blue."
    with pytest.raises(SpanError):
        locate(text, "is red.")


def test_merged_two_sentences_rejected():
    text = "It is red. It is blue."
    with pytest.raises(SpanError):
        locate(text, "It is red. It is blue.")


def test_merged_rejected_in_locate_all():
    text = "It is red. It is blue."
    with pytest.raises(SpanError):
        locate_all(text, ["It is red. It is blue."])


def test_decimal_and_single_sentence_not_flagged_as_merged():
    # A legitimate single sentence containing a decimal must still locate.
    text = "The weight is 1.5 kilograms."
    span = locate(text, "The weight is 1.5 kilograms.")
    assert text[span.start_char:span.end_char] == "The weight is 1.5 kilograms."
