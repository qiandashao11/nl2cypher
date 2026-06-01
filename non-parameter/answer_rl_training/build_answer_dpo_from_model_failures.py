"""
Build DPO preference data from real answer-model failures.

Unlike build_answer_dpo_dataset.py, this script does not synthesize bad answers.
It runs the current SFT answer model with reward guard disabled, scores that raw
answer, and keeps only cases where:

    chosen gold answer is high reward
    raw model answer is lower reward
    chosen - rejected margin is large enough

The resulting records are better DPO data because rejected answers are actual
model failure modes instead of artificial corruptions.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from answer_generator_clean import CleanAnswerGenerator
from answer_generator_sft import SFTAnswerGenerator
from answer_rl_training.answer_reward import score_answer


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def prompt_from_messages(record: Dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages") or []
    return [m for m in messages if m.get("role") != "assistant"]


def gold_answer(record: Dict[str, Any]) -> str:
    if record.get("answer"):
        return str(record["answer"]).strip()
    messages = record.get("messages") or []
    assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    if assistant:
        return str(assistant.get("content", "")).strip()
    return ""


def fallback_answer(question: str, db_result: Dict[str, Any], max_display: int) -> str | None:
    renderer = CleanAnswerGenerator.__new__(CleanAnswerGenerator)
    renderer.max_display = max_display
    return CleanAnswerGenerator._fallback_answer(renderer, question, db_result)


def choose_answer(
    record: Dict[str, Any],
    min_chosen_reward: float,
    allow_fallback_chosen: bool,
    max_display: int,
) -> tuple[str, Dict[str, Any]]:
    question = record.get("question", "")
    cypher = record.get("cypher", "")
    db_result = record.get("db_result") or {}

    chosen = gold_answer(record)
    chosen_score = score_answer(question, cypher, db_result, chosen) if chosen else {"reward": 0.0}
    if chosen_score["reward"] >= min_chosen_reward:
        return chosen, chosen_score

    if allow_fallback_chosen:
        fallback = fallback_answer(question, db_result, max_display)
        if fallback:
            fallback_score = score_answer(question, cypher, db_result, fallback)
            if fallback_score["reward"] >= min_chosen_reward:
                return fallback, fallback_score

    return "", chosen_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO data from real SFT answer failures.")
    parser.add_argument("--input", default="answer_rl_training/train.answer_sft.real_nonempty.jsonl")
    parser.add_argument("--output", default="answer_rl_training/train.answer_dpo.real_failures.jsonl")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_dir", default="answer_rl_training/lora_out_llama3_answer_sft_real_nonempty")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--limit", type=int, default=200, help="Maximum records to scan/generate.")
    parser.add_argument("--max_pairs", type=int, default=100, help="Maximum DPO pairs to write.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--min_chosen_reward", type=float, default=0.88)
    parser.add_argument("--max_rejected_reward", type=float, default=0.84)
    parser.add_argument("--min_margin", type=float, default=0.08)
    parser.add_argument("--max_display", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--no_fallback_chosen", action="store_true")
    parser.add_argument("--progress_every", type=int, default=10)
    args = parser.parse_args()

    records = list(iter_jsonl(Path(args.input)))
    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(records)
    if args.start:
        records = records[args.start :]
    if args.limit:
        records = records[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = SFTAnswerGenerator(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        hf_token=args.hf_token,
        max_display=args.max_display,
        use_4bit=not args.no_4bit,
        use_reward_guard=False,
    )

    scanned = 0
    written = 0
    skipped_low_chosen = 0
    skipped_high_rejected = 0
    skipped_small_margin = 0
    pending: list[dict[str, Any]] = []

    def generate_rejected_batch(batch: list[dict[str, Any]]) -> list[str]:
        if len(batch) == 1:
            item = batch[0]
            return [generator.generate(item["question"], item["cypher"], item["db_result"]).strip()]

        prompts = [
            generator._build_prompt(item["question"], item["cypher"], item["db_result"])
            for item in batch
        ]
        original_padding_side = generator.tokenizer.padding_side
        generator.tokenizer.padding_side = "left"
        try:
            inputs = generator.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(generator.model.device)
        finally:
            generator.tokenizer.padding_side = original_padding_side

        with torch.no_grad():
            outputs = generator.model.generate(
                **inputs,
                max_new_tokens=90,
                do_sample=False,
                repetition_penalty=1.18,
                no_repeat_ngram_size=6,
                pad_token_id=generator.tokenizer.eos_token_id,
                eos_token_id=generator.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        answers = []
        for output_ids in outputs:
            answer = generator.tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
            answers.append(generator._clean_answer(answer).strip())
        return answers

    def flush_pending(out) -> None:
        nonlocal written, skipped_high_rejected, skipped_small_margin
        if not pending:
            return
        batch = pending[:]
        pending.clear()
        rejected_answers = generate_rejected_batch(batch)
        for item, rejected in zip(batch, rejected_answers):
            if written >= args.max_pairs:
                return
            rejected_score = score_answer(item["question"], item["cypher"], item["db_result"], rejected)

            if rejected_score["reward"] > args.max_rejected_reward:
                skipped_high_rejected += 1
                continue

            margin = item["chosen_score"]["reward"] - rejected_score["reward"]
            if margin < args.min_margin or rejected == item["chosen"]:
                skipped_small_margin += 1
                continue

            dpo_record = {
                "prompt_messages": prompt_from_messages(item["record"]),
                "chosen": item["chosen"],
                "rejected": rejected,
                "question": item["question"],
                "cypher": item["cypher"],
                "metadata": {
                    **(item["record"].get("metadata") or {}),
                    "dpo_source": "real_sft_failure",
                    "chosen_reward": item["chosen_score"]["reward"],
                    "rejected_reward": rejected_score["reward"],
                    "reward_margin": round(margin, 4),
                },
                "chosen_reward_score": item["chosen_score"],
                "rejected_reward_score": rejected_score,
            }
            out.write(json.dumps(dpo_record, ensure_ascii=False) + "\n")
            out.flush()
            written += 1

    with output_path.open("w", encoding="utf-8") as out:
        for record in records:
            if written >= args.max_pairs:
                break

            scanned += 1
            question = record.get("question", "")
            cypher = record.get("cypher", "")
            db_result = record.get("db_result") or {}

            chosen, chosen_score = choose_answer(
                record,
                min_chosen_reward=args.min_chosen_reward,
                allow_fallback_chosen=not args.no_fallback_chosen,
                max_display=args.max_display,
            )
            if not chosen:
                skipped_low_chosen += 1
                continue

            pending.append(
                {
                    "record": record,
                    "question": question,
                    "cypher": cypher,
                    "db_result": db_result,
                    "chosen": chosen,
                    "chosen_score": chosen_score,
                }
            )

            if len(pending) >= max(1, args.batch_size):
                flush_pending(out)

            if args.progress_every and scanned % args.progress_every == 0:
                print(
                    f"scanned={scanned} written={written} "
                    f"skipped_low_chosen={skipped_low_chosen} "
                    f"skipped_high_rejected={skipped_high_rejected} "
                    f"skipped_small_margin={skipped_small_margin}"
                )

        flush_pending(out)

    print(
        "Done. "
        f"scanned={scanned}, written={written}, "
        f"skipped_low_chosen={skipped_low_chosen}, "
        f"skipped_high_rejected={skipped_high_rejected}, "
        f"skipped_small_margin={skipped_small_margin}, "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
