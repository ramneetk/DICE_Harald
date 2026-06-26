#!/usr/bin/env python3
"""Summarize parse failures from llm_exp JSONL logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def analyze(log_dir: Path) -> None:
    files = sorted(log_dir.glob("*.jsonl"))
    if not files:
        print(f"No logs in {log_dir}")
        return

    errors: Counter[str] = Counter()
    ok = fail = 0
    no_json = 0
    thinking_like = 0

    for fp in files:
        with fp.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("parse_success"):
                    ok += 1
                else:
                    fail += 1
                    err = rec.get("parse_error") or "unknown"
                    errors[err] += 1
                    preview = (rec.get("raw_preview") or "").strip()
                    if not preview:
                        no_json += 1
                    elif preview.lower().startswith("thinking process"):
                        thinking_like += 1

    total = ok + fail
    print(f"Logs: {log_dir}")
    print(f"Records: {total}  success: {ok} ({100*ok/max(total,1):.1f}%)  fail: {fail}")
    if fail:
        print(f"  missing raw_preview: {no_json}/{fail}")
        print(f"  thinking-prefixed previews: {thinking_like}/{fail}")
    print("\nTop parse errors:")
    for err, count in errors.most_common(12):
        print(f"  {count:5d}  {err[:100]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze LLM JSONL logs")
    parser.add_argument(
        "--logs",
        type=Path,
        default=Path("llm_exp/results/logs"),
        help="Directory containing *.jsonl",
    )
    args = parser.parse_args()
    analyze(args.logs)


if __name__ == "__main__":
    main()
