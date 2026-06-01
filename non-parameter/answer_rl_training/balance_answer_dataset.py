"""
Balance answer-agent SFT data.

The generated dataset can contain many empty Neo4j results because synthetic
question/entity combinations do not always exist in the graph.  Training on too
many empty answers biases the answer model toward saying "no matching results".

This script keeps all non-empty examples and samples empty-result examples up
to a target ratio.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_success(record: Dict[str, Any]) -> bool:
    return bool((record.get("db_result") or {}).get("success"))


def is_empty(record: Dict[str, Any]) -> bool:
    result = record.get("db_result") or {}
    return bool(result.get("success")) and int(result.get("count", 0)) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance answer-agent SFT data by downsampling empty results.")
    parser.add_argument("--input", default="answer_rl_training/train.answer_sft.jsonl")
    parser.add_argument("--output", default="answer_rl_training/train.answer_sft.balanced.jsonl")
    parser.add_argument("--empty_ratio", type=float, default=0.5, help="Maximum empty examples / non-empty examples")
    parser.add_argument("--include_failed", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = list(iter_jsonl(Path(args.input)))
    non_empty = [record for record in records if is_success(record) and not is_empty(record)]
    empty = [record for record in records if is_empty(record)]
    failed = [record for record in records if not is_success(record)]

    rng = random.Random(args.seed)
    rng.shuffle(non_empty)
    rng.shuffle(empty)
    rng.shuffle(failed)

    max_empty = int(len(non_empty) * args.empty_ratio)
    selected = non_empty + empty[:max_empty]
    if args.include_failed:
        selected += failed[: max(1, int(0.02 * len(selected)))]
    rng.shuffle(selected)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in selected:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Input records: {len(records)}")
    print(f"Non-empty success records kept: {len(non_empty)}")
    print(f"Empty success records available: {len(empty)}")
    print(f"Empty success records kept: {min(len(empty), max_empty)}")
    print(f"Failed records available: {len(failed)}")
    print(f"Output records: {len(selected)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
