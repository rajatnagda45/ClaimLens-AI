"""Fact-tool tests."""
import pytest

from claimlens import FACT_VERSION
from claimlens.fact_tool import FactTool, ToolError


def test_known_brand_returns_facts_at_v4():
    t = FactTool()
    res = t.get_brand_facts("cleartrail")
    assert res["fact_version"] == FACT_VERSION
    assert res["brand_id"] == "cleartrail"
    assert len(res["facts"]) > 0
    assert all(f["brand_id"] == "cleartrail" for f in res["facts"])


def test_comparison_brand_lookup_allowed():
    t = FactTool()
    # Any known brand is permitted (not limited to a primary brand).
    assert t.get_brand_facts("stonewave")["facts"]
    assert t.get_brand_facts("cloudnote")["facts"]


def test_unknown_brand_rejected():
    t = FactTool()
    with pytest.raises(ToolError):
        t.get_brand_facts("sony")


def test_malformed_brand_arg_rejected():
    t = FactTool()
    with pytest.raises(ToolError):
        t.get_brand_facts("")
