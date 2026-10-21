"""Execute the four live runs in sequence and emit the comparison.

Requires OPENAI_API_KEY in the environment / .env. Produces:
  reports/v1_development.{json,md}  reports/v2_development.{json,md}
  reports/v1_check.{json,md}        reports/v2_check.{json,md}
  reports/comparison.json           (v1-vs-v2 dev metrics + failure summary)

Usage:  python -m scripts.run_all          (all four)
        python -m scripts.run_all v1 development   (one run)
"""
from __future__ import annotations

import json
import sys

from claimlens.analysis import compare
from claimlens.config import REPORTS_DIR, get_settings
from claimlens.provider import OpenAIProvider
from claimlens.runner import Runner

RUNS = [("v1", "development"), ("v2", "development"),
        ("v1", "check"), ("v2", "check")]


def main(argv: list[str]) -> int:
    s = get_settings()
    if not s.openai_api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env first.", file=sys.stderr)
        return 2

    runs = RUNS
    if len(argv) == 2:
        runs = [(argv[0], argv[1])]

    provider = OpenAIProvider(s)
    runner = Runner(provider, s)
    for version, split in runs:
        print(f"=== running {version} / {split} (model={s.model}) ===", flush=True)
        report = runner.run_split(version, split, persist=True, write_report=True)
        m = report["metrics"]
        if m.get("joint_f1") is not None:
            print(f"  joint_f1={m['joint_f1']:.3f} extraction_f1={m['extraction_f1']:.3f} "
                  f"G={m['gold_claims']} P={m['accepted_predictions']} "
                  f"M={m['span_matches']} C={m['correct_label_matches']}")
        print(f"  success={m['answers_success']} empty={m['answers_empty']} "
              f"error={m['answers_error']}")

    # Comparison (only if both dev reports exist)
    v1d = REPORTS_DIR / "v1_development.json"
    v2d = REPORTS_DIR / "v2_development.json"
    if v1d.exists() and v2d.exists():
        cmp = compare(v1d, v2d)
        (REPORTS_DIR / "comparison.json").write_text(
            json.dumps(cmp, indent=2), encoding="utf-8")
        print("=== v1 vs v2 (development) ===")
        print(json.dumps(cmp, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
