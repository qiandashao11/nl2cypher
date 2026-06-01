# ==================== answer_generator.py ====================
"""
自然语言回答生成器
将查询结果转换为自然语言回答
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, Any


class AnswerGenerator:
    """自然语言回答生成器"""

    def __init__(self,
                 base_model="meta-llama/Llama-3.1-8B-Instruct",
                 hf_token=None):
        """
        初始化生成器

        Args:
            base_model: 使用的模型路径（Llama 3.1 模型）
            hf_token: Hugging Face token
        """
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=hf_token,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
        )
        self.model.eval()

    def format_results(self, query_results: Dict[str, Any]) -> str:
        """
        格式化查询结果

        Args:
            query_results: execute()返回的结果字典

        Returns:
            格式化的文本
        """
        if not query_results["success"]:
            return f"Query execution failed: {query_results.get('error', 'Unknown error')}"

        if not query_results["data"]:
            return "No results returned from the query."

        # 限制显示数量
        max_display = 20
        records = query_results["data"][:max_display]
        count = query_results["count"]

        formatted = f"Query returned {count} records"
        if count > max_display:
            formatted += f" (showing the first {max_display} records)"
        formatted += ":\n\n"

        for i, record in enumerate(records, 1):
            formatted += f"Record {i}:\n"
            for key, value in record.items():
                formatted += f"  - {key}: {value}\n"
            formatted += "\n"

        return formatted

    def generate(self,
                 question: str,
                 cypher: str,
                 query_results: Dict[str, Any]) -> str:
        """
        生成自然语言回答

        Args:
            question: 用户原始问题
            cypher: 执行的Cypher查询
            query_results: 查询结果

        Returns:
            自然语言回答
        """
        formatted_results = self.format_results(query_results)

        prompt = f"""You are a biomedical knowledge graph Q&A assistant.

User's original question:
{question}

Executed Cypher query:
{cypher}

Query results:
{formatted_results}

Please provide a natural, clear answer based on the query results.
- If results are empty or query failed, politely inform the user.
- If there are results, summarize key information directly.
- Keep the answer concise and accurate.
- Don't repeat raw data, summarize in natural language."""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text


if __name__ == "__main__":
    # 测试
    gen = AnswerGenerator()

    mock_results = {
        "success": True,
        "data": [
            {"gene": "BRCA1", "count": 150},
            {"gene": "TP53", "count": 120}
        ],
        "count": 2
    }

    answer = gen.generate(
        question="Which genes are related to breast cancer?",
        cypher="MATCH (g:Gene)-[:CO_OCCURS]-(m:MeSH {entity: 'Breast Cancer'}) RETURN g.entity AS gene, count(*) AS count",
        query_results=mock_results
    )

    print("=== Generated Answer ===")
    print(answer)
