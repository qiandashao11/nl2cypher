# ==================== qa_system.py ====================
"""
完整的Neo4j问答系统
整合Cypher生成、查询执行和回答生成
"""
from cypher_generator import CypherGenerator
from neo4j_executor import Neo4jExecutor
from answer_generator import AnswerGenerator
from typing import Dict, Any


class Neo4jQASystem:
    """完整的问答系统"""
    
    def __init__(self,
                 # Cypher生成器参数
                 base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
                 lora_dir: str = "./lora_out_llama3_8b3",
                 hf_token: str = None,
                 # Neo4j参数
                 neo4j_uri: str = "neo4j://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "neo4j"):
        """
        初始化问答系统
        
        Args:
            base_model: Llama基础模型
            lora_dir: LoRA路径
            hf_token: HF token
            neo4j_uri: Neo4j地址
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
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
        完整的问答流程
        
        Args:
            question: 用户问题
            database: Neo4j数据库名
            params: 可选参数
            verbose: 是否打印过程信息
            
        Returns:
            包含完整结果的字典:
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
        
        # 步骤1: 生成Cypher
        if verbose:
            print("[Step 1] Generating Cypher...")
        cypher = self.cypher_gen.generate(question, params)
        if verbose:
            print(f"[Cypher]\n{cypher}\n")
        
        # 步骤2: 执行查询
        if verbose:
            print("[Step 2] Executing query...")
        results = self.neo4j_exec.execute(cypher, database)
        if verbose:
            if results["success"]:
                print(f"[Results] Retrieved {results['count']} records\n")
            else:
                print(f"[Error] {results.get('error')}\n")
        
        # 步骤3: 生成回答
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
        """关闭连接"""
        self.neo4j_exec.close()
    
    def __enter__(self):
        """支持with语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持with语句"""
        self.close()


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Neo4j QA System")
    ap.add_argument("--question", required=True, help="Your question")
    ap.add_argument("--database", default="neo4j", help="Neo4j database name")
    ap.add_argument("--language", default="English", choices=["English"])
    ap.add_argument("--lora_dir", default="./lora_out_llama3_8b4")
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
