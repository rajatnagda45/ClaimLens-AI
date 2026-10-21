"""Saved artifacts are verified, never regenerated or used in inference."""
import json

import pytest

from claimlens.analysis import compare, verify_report
from claimlens.config import REPORTS_DIR


@pytest.mark.parametrize("name", ["v1_development", "v2_development", "v1_check", "v2_check"])
def test_frozen_report_metrics_reproduce(name):
    result = verify_report(REPORTS_DIR / f"{name}.json")
    if name.endswith("check"):
        assert result["metrics"]["joint_f1"] is None
        assert result["metrics"]["accepted_predictions"] > 0


def test_comparison_has_regressions_and_negative_delta():
    result = compare(REPORTS_DIR / "v1_development.json", REPORTS_DIR / "v2_development.json")
    assert result["delta_joint_f1"] == pytest.approx(-0.03912572876436)
    assert len(result["lost_correct_ids"]) - len(result["newly_correct_ids"]) == 20


def test_tampered_metric_rejected(tmp_path):
    report = json.loads((REPORTS_DIR / "v1_development.json").read_text())
    report["metrics"]["joint_f1"] = 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="metric mismatch"):
        verify_report(path)


def test_changed_settings_rejected(tmp_path):
    report = json.loads((REPORTS_DIR / "v2_development.json").read_text())
    report["metadata"]["settings"]["temperature"] = 1
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="uncontrolled comparison"):
        compare(REPORTS_DIR / "v1_development.json", path)
