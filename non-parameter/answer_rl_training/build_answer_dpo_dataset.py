"""
Build DPO preference data for the answer agent.

The chosen answer is the clean/SFT gold answer from the SFT dataset.  The
rejected answer is a deterministic corrupted variant that mimics the failure
modes observed in the original Llama answer generator:

- gene mutation, e.g. AIFM2 -> AIFM
- title/MeSH paraphrasing or truncation
- count mutation
- unsupported interpretation of CO_OCCURS

This gives DPO a clear preference signal before attempting PPO/GRPO.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def prompt_from_messages(record: Dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages") or []
    return [m for m in messages if m.get("role") != "assistant"]


def mutate_gene(value: str) -> str:
    if len(value) > 4 and any(ch.isdigit() for ch in value):
        return re.sub(r"\d+$", "", value)
    if len(value) > 3:
        return value[:-1]
    return value + "1"


def mutate_title(value: str) -> str:
    value = value.replace("Suppressive", "Suppressing")
    value = value.replace("metastasizing", "metastasising")
    value = value.replace("ferroptosis", "ferrotosis")
    return value


def corrupt_answer(record: Dict[str, Any], rng: random.Random) -> str:
    answer = record.get("answer", "")
    db_result = record.get("db_result") or {}
    data = db_result.get("data") or []
    corrupted = answer

    # Mutate one concrete field when possible.
    field_values: list[tuple[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if value is not None:
                field_values.append((key, value))

    rng.shuffle(field_values)
    for key, value in field_values:
        text = str(value)
        if key == "gene_entity" and text in corrupted:
            corrupted = corrupted.replace(text, mutate_gene(text), 1)
            break
        if key == "title" and text in corrupted:
            corrupted = corrupted.replace(text, mutate_title(text), 1)
            break
        if key == "mesh_entity" and text in corrupted and "/" in text:
            corrupted = corrupted.replace(text, text.replace("/", " "), 1)
            break
        if key.endswith("_count") and str(value) in corrupted:
            try:
                corrupted = corrupted.replace(str(value), str(int(value) + rng.choice([1, 2, 5])), 1)
                break
            except ValueError:
                pass

    if corrupted == answer:
        count = db_result.get("count")
        if isinstance(count, int) and str(count) in corrupted:
            corrupted = corrupted.replace(str(count), str(count + 1), 1)

    cypher = record.get("cypher", "")
    if "CO_OCCURS" in cypher and "co-occurs" in corrupted:
        corrupted = corrupted.replace("co-occurs", "is mechanistically associated", 1)

    if corrupted == answer:
        corrupted = answer.rstrip(".") + " and may indicate a broader biological mechanism."
    return corrupted


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO chosen/rejected pairs for answer generation.")
    parser.add_argument("--input", default="answer_rl_training/train.answer_sft.real_nonempty.jsonl")
    parser.add_argument("--output", default="answer_rl_training/train.answer_dpo.synthetic.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for record in iter_jsonl(Path(args.input)):
            if args.limit and written >= args.limit:
                break
            chosen = record.get("answer", "").strip()
            if not chosen:
                continue
            rejected = corrupt_answer(record, rng).strip()
            if rejected == chosen:
                continue
            dpo_record = {
                "prompt_messages": prompt_from_messages(record),
                "chosen": chosen,
                "rejected": rejected,
                "question": record.get("question"),
                "cypher": record.get("cypher"),
                "metadata": {
                    **(record.get("metadata") or {}),
                    "dpo_source": "synthetic_corruption",
                },
            }
            out.write(json.dumps(dpo_record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} DPO records to {output_path}")


if __name__ == "__main__":
    main()
