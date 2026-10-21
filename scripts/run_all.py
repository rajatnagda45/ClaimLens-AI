"""Run explicitly authorized new experiments without replacing frozen evidence."""
from __future__ import annotations

import argparse
import json
import sys

from claimlens.analysis import compare
from claimlens.config import REPORTS_DIR, get_settings
from claimlens.provider import OpenAIProvider, ProviderError
from claimlens.runner import Runner, RunFailure

RUNS = [("v1", "development"), ("v2", "development"),
        ("v1", "check"), ("v2", "check")]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?", choices=["v1", "v2"])
    parser.add_argument("split", nargs="?", choices=["development", "check"])
    parser.add_argument("--confirm-live", action="store_true",
                        help="Authorize paid API requests for a new experiment")
    args = parser.parse_args(argv)
    if not args.confirm_live:
        parser.error("--confirm-live is required for paid API requests")
    if bool(args.version) != bool(args.split):
        parser.error("version and split must be supplied together")
    settings = get_settings()
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2
    runs = [(args.version, args.split)] if args.version else RUNS
    try:
        runner = Runner(OpenAIProvider(settings), settings)
        dev_reports = {}
        for version, split in runs:
            report = runner.run_split(version, split)
            path = REPORTS_DIR / "runs" / (report["metadata"]["run_id"] + ".json")
            print(json.dumps({"report": str(path), "metrics": report["metrics"]}))
            if split == "development":
                dev_reports[version] = path
        if len(dev_reports) == 2:
            print(json.dumps(compare(dev_reports["v1"], dev_reports["v2"]), indent=2))
    except (ProviderError, RunFailure) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
