# ==================== qa_system_clean.py ====================
"""
Clean Neo4j QA pipeline.

This keeps qa_system.py unchanged and uses CleanAnswerGenerator for cleaner
natural-language final answers.
"""
from typing import Any, Dict

from answer_generator_clean import CleanAnswerGenerator
from cypher_generator import CypherGenerator
from neo4j_executor import Neo4jExecutor


class CleanNeo4jQASystem:
    """Generate Cypher, execute it in Neo4j, and return a clean answer."""

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        lora_dir: str = "phase3_multihop_training/lora_out_llama3_8b_multihop",
        hf_token: str | None = None,
        neo4j_uri: str = "neo4j://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "neo4j",
    ):
        print("=" * 60)
        print("Initializing Clean Neo4j QA System...")
        print("=" * 60)

        self.cypher_gen = CypherGenerator(base_model, lora_dir, hf_token)
        self.neo4j_exec = Neo4jExecutor(neo4j_uri, neo4j_user, neo4j_password)
        self.answer_gen = CleanAnswerGenerator(base_model=base_model, hf_token=hf_token)

        print("=" * 60)
        print("Clean QA system initialized")
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
        }

    def close(self):
        self.neo4j_exec.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Clean Neo4j QA System")
    ap.add_argument("--question", required=True)
    ap.add_argument("--database", default="neo4j")
    ap.add_argument("--lora_dir", default="phase3_multihop_training/lora_out_llama3_8b_multihop")
    ap.add_argument("--neo4j_uri", default="neo4j://localhost:7687")
    ap.add_argument("--neo4j_user", default="neo4j")
    ap.add_argument("--neo4j_password", default="neo4j")
    args = ap.parse_args()

    with CleanNeo4jQASystem(
        lora_dir=args.lora_dir,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
    ) as qa:
        result = qa.answer(args.question, database=args.database, verbose=True)
        print("\nFINAL ANSWER:")
        print(result["answer"])
