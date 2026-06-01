"""
Build a supervised dataset for the second-stage answer agent.

Source data:
    train.chat.phase3_multihop.jsonl

Each source record already contains a user question and a gold Cypher query.
This script executes the Cypher against Neo4j, writes a compact database-result
payload, and uses the deterministic clean-answer renderer as the gold answer.
The resulting JSONL can be used for SFT and later as the "chosen" side of DPO.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from answer_generator_clean import CleanAnswerGenerator
from neo4j_executor import Neo4jExecutor

from answer_reward import score_answer


ANSWER_SYSTEM_PROMPT = """You are a biomedical knowledge graph answer writer.

Answer the user's question using only the provided Cypher result.

Rules:
- Output only the final answer text.
- Preserve every gene symbol, MeSH term, PMID, title, year, journal, and count exactly as provided.
- Do not invent entities, mechanisms, papers, URLs, or external facts.
- If the result is empty, say that no matching results were found.
- If many rows are returned, mention the total count and give representative examples from the result.
- For CO_OCCURS relationships, say "co-occurs", "is linked", or "is associated in the graph"; do not say cause, regulate, interact, or co-express unless those words appear in the result.
- Keep the answer concise and natural."""


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_question_cypher(record: Dict[str, Any]) -> Tuple[str, str]:
    messages = record.get("messages") or []
    question = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    cypher = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
    return question.strip(), cypher.strip().rstrip(";")


def sampled_query(cypher: str, max_rows: int) -> Tuple[str, str]:
    inner = cypher.strip().rstrip(";")
    count_cypher = f"CALL () {{ {inner} }} RETURN count(*) AS __row_count"
    data_cypher = f"CALL () {{ {inner} }} RETURN * LIMIT {max_rows}"
    return count_cypher, data_cypher


def execute_with_sampling(
    executor: Neo4jExecutor,
    cypher: str,
    database: str,
    max_rows: int,
) -> Dict[str, Any]:
    count_cypher, data_cypher = sampled_query(cypher, max_rows)
    count_result = executor.execute(count_cypher, database=database)
    data_result = executor.execute(data_cypher, database=database)

    if count_result.get("success") and data_result.get("success"):
        row_count = 0
        if count_result.get("data"):
            row_count = int(count_result["data"][0].get("__row_count", 0))
        return {
            "success": True,
            "data": data_result.get("data", []),
            "count": row_count,
            "truncated": row_count > len(data_result.get("data", [])),
            "max_rows": max_rows,
        }

    # Some Cypher variants may not be valid inside CALL. Fall back to direct
    # execution so the dataset builder is still useful.
    direct = executor.execute(cypher, database=database)
    if direct.get("success"):
        direct["data"] = (direct.get("data") or [])[:max_rows]
        direct["truncated"] = direct.get("count", 0) > len(direct["data"])
        direct["max_rows"] = max_rows
    return direct


def make_renderer(max_display: int) -> CleanAnswerGenerator:
    # Avoid loading an LLM: the deterministic fallback methods do not require
    # tokenizer/model state.
    renderer = CleanAnswerGenerator.__new__(CleanAnswerGenerator)
    renderer.max_display = max_display
    return renderer


def render_clean_answer(
    renderer: CleanAnswerGenerator,
    question: str,
    cypher: str,
    db_result: Dict[str, Any],
) -> str:
    answer = CleanAnswerGenerator._fallback_answer(renderer, question, db_result)
    if answer:
        return answer

    if not db_result.get("success"):
        return "I could not answer the question because the generated Cypher query failed to execute."

    data = db_result.get("data") or []
    if not data:
        return "No matching results were found in the knowledge graph."

    total = db_result.get("count", len(data))
    examples = []
    for row in data[:5]:
        pieces = [f"{key}={value}" for key, value in row.items()]
        examples.append("; ".join(pieces))
    return f"Found {total} matching results. Examples include {' | '.join(examples)}."


def format_db_result_for_prompt(db_result: Dict[str, Any]) -> str:
    payload = {
        "success": db_result.get("success", False),
        "count": db_result.get("count", 0),
        "data": db_result.get("data", []),
    }
    if db_result.get("truncated"):
        payload["truncated"] = True
        payload["note"] = f"Only the first {len(db_result.get('data', []))} rows are shown."
    if not db_result.get("success"):
        payload["error"] = db_result.get("error", "Unknown error")
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_user_prompt(question: str, cypher: str, db_result: Dict[str, Any]) -> str:
    return f"""Question:
{question}

Cypher:
{cypher}

Cypher result:
{format_db_result_for_prompt(db_result)}

Write the final answer."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build answer-agent SFT data from gold Cypher and Neo4j results.")
    parser.add_argument("--source", default="phase3_multihop_training/train.chat.phase3_multihop.jsonl")
    parser.add_argument("--output", default="answer_rl_training/train.answer_sft.jsonl")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--neo4j_uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", default="neo4j")
    parser.add_argument("--neo4j_password", default="neo4j")
    parser.add_argument("--limit", type=int, default=0, help="0 means all records")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=20)
    parser.add_argument("--skip_failed", action="store_true")
    parser.add_argument("--dedupe_question", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = make_renderer(max_display=args.max_rows)
    seen_questions = set()
    written = 0
    skipped = 0

    with Neo4jExecutor(args.neo4j_uri, args.neo4j_user, args.neo4j_password) as executor:
        with output_path.open("w", encoding="utf-8") as out:
            for index, record in enumerate(iter_jsonl(source_path)):
                if index < args.offset:
                    continue
                if args.limit and written >= args.limit:
                    break

                question, cypher = extract_question_cypher(record)
                if not question or not cypher:
                    skipped += 1
                    continue
                if args.dedupe_question and question in seen_questions:
                    skipped += 1
                    continue
                seen_questions.add(question)

                db_result = execute_with_sampling(executor, cypher, args.database, args.max_rows)
                if args.skip_failed and not db_result.get("success"):
                    skipped += 1
                    continue

                answer = render_clean_answer(renderer, question, cypher, db_result)
                reward = score_answer(question, cypher, db_result, answer)
                user_prompt = build_user_prompt(question, cypher, db_result)

                output_record = {
                    "messages": [
                        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": answer},
                    ],
                    "question": question,
                    "cypher": cypher,
                    "db_result": db_result,
                    "answer": answer,
                    "metadata": {
                        **(record.get("metadata") or {}),
                        "source_index": index,
                        "answer_reward": reward["reward"],
                    },
                    "reward_score": reward,
                }
                out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                written += 1

                if written % 50 == 0:
                    print(f"Written {written} records...")

    print(f"Done. Wrote {written} records to {output_path}. Skipped {skipped}.")


if __name__ == "__main__":
    main()
