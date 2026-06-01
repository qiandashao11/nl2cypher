# ==================== qa_system.py ====================
"""
Complete Neo4j QA system
Integrate Cypher generation, query execution, and answer generation
"""
from cypher_generator import CypherGenerator
from neo4j_executor import Neo4jExecutor
from answer_generator import AnswerGenerator
from typing import Dict, Any


class Neo4jQASystem:
    """Complete QA system"""
    
    def __init__(self,
                 # Cypher generator parameters
                 base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
                 lora_dir: str = "./lora_out_llama3_8b",
                 hf_token: str = None,
                 # Neo4j parameters
                 neo4j_uri: str = "neo4j://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "neo4j"):
        """
        Initialize QA system
        
        Args:
            base_model: Llama base model
            lora_dir: LoRA path
            hf_token: HF token
            neo4j_uri: Neo4j URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        print("=" * 60)
        print("Initializing Neo4j QA System...")
        print("=" * 60)
        
        self.cypher_gen = CypherGenerator(base_model, lora_dir, hf_token)
        self.neo4j_exec = Neo4jExecutor(neo4j_uri, neo4j_user, neo4j_password)
        self.answer_gen = AnswerGenerator(base_model=base_model, hf_token=hf_token)
        
        print("=" * 60)
        print("✅ System initialized")
        print("=" * 60)
    
    def answer(self,
              question: str,
              database: str = "neo4j",
              params: dict = None,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Complete QA workflow
        
        Args:
            question: user question
            database: Neo4j database name
            params: optional parameters
            verbose: whether to print progress/details
            
        Returns:
            dictionary containing full results:
            {
                "question": str,
                "cypher": str,
                "results": dict,
                "answer": str
            }
        """
        if verbose:
            print("\n" + "=" * 60)
            print("Starting QA Pipeline")
            print("=" * 60)
            print(f"\n[Question] {question}\n")
        
        # Step 1: Generate Cypher
        if verbose:
            print("[Step 1] Generating Cypher...")
        cypher = self.cypher_gen.generate(question, params)
        if verbose:
            print(f"[Cypher]\n{cypher}\n")
        
        # Step 2: Execute query
        if verbose:
            print("[Step 2] Executing query...")
        results = self.neo4j_exec.execute(cypher, database)
        if verbose:
            if results["success"]:
                print(f"[Results] Retrieved {results['count']} records\n")
            else:
                print(f"[Error] {results.get('error')}\n")
        
        # Step 3: Generate answer
        if verbose:
            print("[Step 3] Generating answer...")
        answer = self.answer_gen.generate(question, cypher, results)
        
        if verbose:
            print("\n" + "=" * 60)
            print("Pipeline Complete")
            print("=" * 60 + "\n")
        
        return {
            "question": question,
            "cypher": cypher,
            "results": results,
            "answer": answer
        }
    
    def close(self):
        """Close connection"""
        self.neo4j_exec.close()
    
    def __enter__(self):
        """Support with-statements"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support with-statements"""
        self.close()


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Neo4j QA System")
    ap.add_argument("--question", required=True, help="Your question")
    ap.add_argument("--database", default="neo4j", help="Neo4j database name")
    ap.add_argument("--language", default="English", choices=["English"])
    ap.add_argument("--lora_dir", default="./lora_out_llama3_8b")
    ap.add_argument("--neo4j_uri", default="neo4j://localhost:7687")
    ap.add_argument("--neo4j_user", default="neo4j")
    ap.add_argument("--neo4j_password", default="neo4j")
    args = ap.parse_args()
    
    with Neo4jQASystem(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        lora_dir=args.lora_dir,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password
    ) as qa_system:
        result = qa_system.answer(
            question=args.question,
            database=args.database,
            verbose=True
        )
        
        print("\n" + "🎯" * 30)
        print("FINAL ANSWER:")
        print("🎯" * 30)
        print(result["answer"])
        print("🎯" * 30 + "\n")
