# ==================== answer_generator_llm.py ====================
"""
LLM-based natural-language answer generator.

This version is based on the original answer_generator.py idea: pass the
question, Cypher, and Neo4j results to the base LLM and let it write the final
answer. It avoids deterministic result-type templates, but uses a stricter
prompt and light output cleanup to reduce obvious meta-text.
"""
import re
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMAnswerGenerator:
    """Generate grounded natural-language answers from executed query results."""

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        hf_token: str | None = None,
        max_display: int = 20,
    ):
        self.max_display = max_display
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=hf_token,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )
        self.model.eval()

    def format_results(self, query_results: Dict[str, Any]) -> str:
        if not query_results.get("success"):
            return f"The query failed with this error: {query_results.get('error', 'Unknown error')}"

        data = query_results.get("data") or []
        count = query_results.get("count", len(data))
        if not data:
            return "The query executed successfully but returned no rows."

        lines = [f"Total rows: {count}"]
        shown = data[: self.max_display]
        for i, record in enumerate(shown, 1):
            parts = [f"{key}: {value}" for key, value in record.items()]
            lines.append(f"{i}. " + "; ".join(parts))
        if count > len(shown):
            lines.append(f"Only the first {len(shown)} rows are shown above.")
        return "\n".join(lines)

    def _build_prompt(self, question: str, cypher: str, formatted_results: str) -> str:
        return f"""You are a biomedical knowledge graph question-answering assistant.

Use only the provided Neo4j query results to answer the user's question.

Important constraints:
- Do not invent entities, papers, URLs, mechanisms, or external facts.
- Do not mention Cypher, Neo4j, rows, records, or query execution.
- If there are no rows, say that no matching results were found.
- If the result is a count, state the count and briefly explain what it counts.
- If the result lists entities or papers, summarize the result in natural language and include representative examples.
- Keep the answer concise: 2-4 sentences.
- Output only the final answer text.

User question:
{question}

Executed Cypher:
{cypher}

Result data:
{formatted_results}

Final answer:
"""

    def _clean_answer(self, text: str) -> str:
        text = text.strip()
        text = re.split(r"\b(?:Cypher|Neo4j|Query results?|Records?|Rows?):", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.split(r"\bBookmark(?:let)?\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.split(r"\n?\s*(?:Final answer|Answer|Text):\s*", text, maxsplit=1, flags=re.IGNORECASE)[-1]
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\(?\s*Note:\s*.*", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"\bThank you!?\s*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def generate(self, question: str, cypher: str, query_results: Dict[str, Any]) -> str:
        formatted_results = self.format_results(query_results)
        prompt = self._build_prompt(question, cypher, formatted_results)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
                repetition_penalty=1.18,
                no_repeat_ngram_size=5,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._clean_answer(answer)


if __name__ == "__main__":
    mock_results = {
        "success": True,
        "data": [
            {"mesh_entity": "oxidative stress/drug effects"},
            {"mesh_entity": "oleic acid/metabolism"},
            {"mesh_entity": "melanoma/blood"},
        ],
        "count": 3,
    }
    gen = LLMAnswerGenerator()
    print(
        gen.generate(
            "Find MeSH entitys co-occurring with ACSL3.",
            "MATCH (:Gene {ENTITY:'ACSL3'})-[:CO_OCCURS]-(m:MeSH) RETURN DISTINCT m.ENTITY AS mesh_entity",
            mock_results,
        )
    )
