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

Your task:
Given the user's question, the Cypher query, and the query results,
produce ONE short natural language answer (2–4 sentences).

Rules:
- Output ONLY the final answer. 
- Do NOT include the question.
- Do NOT include the Cypher query.
- Do NOT mention “query results”, “records”, or list raw data.
- Do NOT explain your reasoning.
- Do NOT evaluate the answer.
- Do NOT ask follow-up questions.
- Do NOT generate multiple versions.
- Do NOT add disclaimers unless absolutely necessary.
- Output only the answer text and nothing else.

User question:
{question}

Cypher query executed:
{cypher}

Query results:
{formatted_results}

Write the final answer now:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.7,
                repetition_penalty=1.18,   
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

            # 连 prompt + answer 都在 sequences 里
            full_sequence = outputs.sequences[0]

            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = full_sequence[prompt_len:]

            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            print("Full sequence len:", len(full_sequence))
            print("Prompt len:", prompt_len)
            print("Generated:", text)
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
