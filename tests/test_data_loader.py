"""CSV and source-integrity tests."""
import csv
import io

from claimlens import data_loader as dl
from claimlens.data_loader import (
    BrandFact, interpret_value, load_answers, load_brand_facts, load_gold,
    source_hashes,
)


def test_quoted_commas_and_newlines_preserved(tmp_path):
    p = tmp_path / "answers.csv"
    text = (
        'answer_id,split,brand_id,question,ai_answer\n'
        'A01,development,cleartrail,"q, with comma","line one\nline two, ok."\n'
    )
    p.write_text(text, encoding="utf-8")
    answers = load_answers(path=p)
    assert len(answers) == 1
    assert answers[0].ai_answer == "line one\nline two, ok."
    assert answers[0].question == "q, with comma"


def test_ai_answer_preserved_exactly():
    answers = load_answers()
    # Round-trip against a raw csv read to confirm no normalization.
    with open("answers.csv", newline="", encoding="utf-8") as fh:
        raw = {r["answer_id"]: r["ai_answer"] for r in csv.DictReader(fh)}
    for a in answers:
        assert a.ai_answer == raw[a.answer_id]


def test_utf8_text(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text(
        'answer_id,split,brand_id,question,ai_answer\n'
        'A1,development,b,q,"café — 日本語 ✓."\n', encoding="utf-8")
    a = load_answers(path=p)[0]
    assert a.ai_answer == "café — 日本語 ✓."


def test_source_hashes_present():
    h = source_hashes()
    assert set(h) == {"answers.csv", "brand_facts.csv", "development_gold.csv"}
    assert all(len(v) == 64 for v in h.values())


def test_split_selection():
    dev = [a for a in load_answers() if a.split == "development"]
    chk = [a for a in load_answers() if a.split == "check"]
    assert len(dev) == 40 and len(chk) == 10


def test_boolean_value_type_not_python_bool():
    f = BrandFact("x", "b", "m", "wifi", "false", "boolean", "", "", "", "")
    assert interpret_value(f) is False  # never bool("false") == True
    t = BrandFact("x", "b", "m", "wifi", "true", "boolean", "", "", "", "")
    assert interpret_value(t) is True


def test_gold_quotes_match_source():
    answers = {a.answer_id: a.ai_answer for a in load_answers()}
    for g in load_gold():
        assert answers[g.answer_id][g.start_char:g.end_char] == g.quote
