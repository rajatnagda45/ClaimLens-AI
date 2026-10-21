"""Read-only verification of frozen inputs, prompts, and historical reports.

Run: python -m scripts.verify_evidence. No provider requests or database writes.
"""
import hashlib
import json

from claimlens.analysis import compare, verify_report
from claimlens.config import PROJECT_ROOT, REPORTS_DIR


def verify_frozen_files():
    manifest = json.loads((PROJECT_ROOT / "evidence/frozen_sha256.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen file changed: {name}")
    return len(manifest)


def main():
    result = {"frozen_files_verified": verify_frozen_files(), "runs": {}}
    for name in ("v1_development", "v2_development", "v1_check", "v2_check"):
        verified = verify_report(REPORTS_DIR / f"{name}.json")
        result["runs"][name] = {k: v for k, v in verified["metrics"].items()
                                if not k.endswith("_ids")}
    comparison = compare(REPORTS_DIR / "v1_development.json", REPORTS_DIR / "v2_development.json")
    result["comparison"] = {
        "delta_joint_f1": comparison["delta_joint_f1"],
        "delta_extraction_f1": comparison["delta_extraction_f1"],
        "newly_correct": len(comparison["newly_correct_ids"]),
        "lost_correct": len(comparison["lost_correct_ids"]),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
