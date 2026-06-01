"""
Run a 10-question end-to-end QA test with the SFT answer agent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qa_system_sft import SFTNeo4jQASystem


QUESTIONS = [
    "Find MeSH entitys co-occurring with ACSL3.",
    "Show literature linked to ACSL3.",
    "Show literature for MeSH t-lymphocytes cytotoxic/immunology.",
    "Papers mentioning both ACSL3 and phospholipid hydroperoxide glutathione peroxidase/metabolism.",
    "Count MeSH entitys shared by ACSL4 and AIFM2.",
    "Find gene and MeSH pairs reported in the paper with PMID 34093872.",
    "Find World Neurosurg papers that report SLC40A1.",
    "SLC40A1 papers after 2020.",
    "Find papers where MeSH contains 'melanoma' and genes are reported after 2019.",
    "Find genes related to both MeSH oxidative stress/drug effects and MeSH phospholipid hydroperoxide glutathione peroxidase/metabolism.",
]


def md_block(text: str, lang: str = "text") -> str:
    return f"```{lang}\n{text}\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 10 QA tests with SFT answer agent.")
    parser.add_argument("--output", default="phase3_multihop_training/qa_test_results_sft_10.md")
    parser.add_argument("--neo4j_uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", default="neo4j")
    parser.add_argument("--neo4j_password", default="neo4j")
    parser.add_argument("--cypher_lora_dir", default="phase3_multihop_training/lora_out_llama3_8b_multihop")
    parser.add_argument("--answer_lora_dir", default="answer_rl_training/lora_out_llama3_answer_sft_real_nonempty")
    parser.add_argument("--disable_reward_guard", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# SFT Answer Generator Test Results",
        "",
        "Pipeline:",
        "",
        f"- Cypher generator: `{args.cypher_lora_dir}`",
        f"- Answer generator: `{args.answer_lora_dir}`",
        f"- Reward guard: `{not args.disable_reward_guard}`",
        "- QA system: `qa_system_sft.py`",
        "- Database: local Neo4j",
        "",
    ]

    with SFTNeo4jQASystem(
        cypher_lora_dir=args.cypher_lora_dir,
        answer_lora_dir=args.answer_lora_dir,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        use_reward_guard=not args.disable_reward_guard,
    ) as qa:
        for index, question in enumerate(QUESTIONS, 1):
            result = qa.answer(question, verbose=True)
            rows = result["results"].get("data") or []
            sample_rows = rows[:8]
            execution = {
                "success": result["results"].get("success"),
                "count": result["results"].get("count"),
                "sample_data": sample_rows,
            }
            if not result["results"].get("success"):
                execution["error"] = result["results"].get("error")

            lines.extend(
                [
                    f"## Result {index}",
                    "",
                    "Question:",
                    "",
                    md_block(question),
                    "",
                    "Cypher:",
                    "",
                    md_block(result["cypher"], "cypher"),
                    "",
                    "Execution:",
                    "",
                    md_block(json.dumps(execution, ensure_ascii=False, indent=2)),
                    "",
                    "Answer:",
                    "",
                    md_block(result["answer"]),
                    "",
                    "Answer reward:",
                    "",
                    md_block(json.dumps(result.get("answer_reward"), ensure_ascii=False, indent=2)),
                    "",
                ]
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
