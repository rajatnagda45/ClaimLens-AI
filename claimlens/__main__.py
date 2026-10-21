"""CLI: classify one answer, run a split, or init the database.

Examples:
  python -m claimlens init-db
  python -m claimlens run --version v1 --split development
  python -m claimlens classify --answer-id A01 --version v2
"""
from __future__ import annotations

import argparse
import json
import sys

from .data_loader import load_answers
from .database import init_db
from .provider import OpenAIProvider
from .runner import Runner


def _run(args):
    runner = Runner(OpenAIProvider())
    report = runner.run_split(args.version, args.split)
    print(json.dumps(report["metrics"], indent=2))


def _classify(args):
    answers = {a.answer_id: a for a in load_answers()}
    a = answers.get(args.answer_id)
    if a is None:
        print(f"unknown answer_id {args.answer_id}", file=sys.stderr)
        sys.exit(1)
    from .agent import Agent
    agent = Agent(OpenAIProvider())
    res = agent.run(a.answer_id, a.brand_id, a.question, a.ai_answer, args.version)
    print(res.model_dump_json(indent=2))


def main():
    p = argparse.ArgumentParser(prog="claimlens")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db")

    r = sub.add_parser("run")
    r.add_argument("--version", choices=["v1", "v2"], required=True)
    r.add_argument("--split", choices=["development", "check"], required=True)

    c = sub.add_parser("classify")
    c.add_argument("--answer-id", required=True)
    c.add_argument("--version", choices=["v1", "v2"], default="v2")

    args = p.parse_args()
    if args.cmd == "init-db":
        init_db()
        print("database initialized")
    elif args.cmd == "run":
        _run(args)
    elif args.cmd == "classify":
        _classify(args)


if __name__ == "__main__":
    main()
