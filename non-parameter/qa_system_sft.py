"""
Neo4j QA pipeline using the SFT-trained answer generator.

The first agent remains the existing NL2Cypher LoRA.  The second answer agent
uses the answer SFT LoRA trained on real non-empty Neo4j results.
"""
from __future__ import annotations

from typing import Any, Dict

from answer_generator_sft import SFTAnswerGenerator
from cypher_generator import CypherGenerator
from neo4j_executor import Neo4jExecutor


class SFTNeo4jQASystem:
    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        cypher_lora_dir: str = "phase3_multihop_training/lora_out_llama3_8b_multihop",
        answer_lora_dir: str = "answer_rl_training/lora_out_llama3_answer_sft_real_nonempty",
        hf_token: str | None = None,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "neo4j",
        answer_use_4bit: bool = True,
        use_reward_guard: bool = True,
    ):
        print("=" * 60)
        print("Initializing SFT Neo4j QA System...")
        print("=" * 60)

        self.cypher_gen = CypherGenerator(base_model, cypher_lora_dir, hf_token)
        self.neo4j_exec = Neo4jExecutor(neo4j_uri, neo4j_user, neo4j_password)
        self.answer_gen = SFTAnswerGenerator(
            base_model=base_model,
            adapter_dir=answer_lora_dir,
            hf_token=hf_token,
            use_4bit=answer_use_4bit,
            use_reward_guard=use_reward_guard,
        )

        print("=" * 60)
        print("SFT QA system initialized")
        print("=" * 60)

    def answer(
        self,
        question: str,
        database: str = "neo4j",
        params: dict | None = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if verbose:
            print("\n" + "=" * 60)
            print(f"Question: {question}")

        cypher = self.cypher_gen.generate(question, params)
        results = self.neo4j_exec.execute(cypher, database)
        answer = self.answer_gen.generate(question, cypher, results)

        if verbose:
            print(f"Cypher: {cypher}")
            print(f"Success: {results.get('success')} | Count: {results.get('count')}")
            if not results.get("success"):
                print(f"Error: {results.get('error')}")
            print(f"Answer: {answer}")

        return {
            "question": question,
            "cypher": cypher,
            "results": results,
            "answer": answer,
            "answer_reward": self.answer_gen.last_reward_score,
        }

    def close(self):
        self.neo4j_exec.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="SFT-answer Neo4j QA System")
    ap.add_argument("--question", required=True)
    ap.add_argument("--database", default="neo4j")
    ap.add_argument("--cypher_lora_dir", default="phase3_multihop_training/lora_out_llama3_8b_multihop")
    ap.add_argument("--answer_lora_dir", default="answer_rl_training/lora_out_llama3_answer_sft_real_nonempty")
    ap.add_argument("--neo4j_uri", default="bolt://localhost:7687")
    ap.add_argument("--neo4j_user", default="neo4j")
    ap.add_argument("--neo4j_password", default="neo4j")
    ap.add_argument("--answer_no_4bit", action="store_true")
    ap.add_argument("--disable_reward_guard", action="store_true")
    args = ap.parse_args()

    with SFTNeo4jQASystem(
        cypher_lora_dir=args.cypher_lora_dir,
        answer_lora_dir=args.answer_lora_dir,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        answer_use_4bit=not args.answer_no_4bit,
        use_reward_guard=not args.disable_reward_guard,
    ) as qa:
        result = qa.answer(args.question, database=args.database, verbose=True)
        print("\nFINAL ANSWER:")
        print(result["answer"])
