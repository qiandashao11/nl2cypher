# ==================== answer_generator_biomistral.py ====================
"""
BioMistral-based natural-language answer generator.

Only the answer-generation stage uses BioMistral. Cypher generation can keep the
existing Llama 3.1 LoRA model.
"""
import re
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class BioMistralAnswerGenerator:
    """Generate biomedical natural-language answers from executed query results."""

    def __init__(
        self,
        base_model: str = "BioMistral/BioMistral-7B",
        hf_token: str | None = None,
        max_display: int = 20,
    ):
        self.max_display = max_display
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            token=hf_token,
            use_fast=True,
        )
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
            return f"QUERY_FAILED: {query_results.get('error', 'Unknown error')}"

        data = query_results.get("data") or []
        count = query_results.get("count", len(data))
        if not data:
            return "NO_ROWS"

        lines = [f"TOTAL_ROWS: {count}"]
        for i, record in enumerate(data[: self.max_display], 1):
            fields = [f"{key}={value}" for key, value in record.items()]
            lines.append(f"ROW_{i}: " + "; ".join(fields))
        if count > self.max_display:
            lines.append(f"TRUNCATED: showing first {self.max_display} rows only")
        return "\n".join(lines)

    def _build_prompt(self, question: str, cypher: str, formatted_results: str) -> str:
        # BioMistral's tokenizer uses the Mistral instruction template:
        # <s>[INST] user content [/INST]
        content = f"""You are a biomedical knowledge graph answer writer.

Answer the user using ONLY the provided database results.

Strict rules:
- Preserve gene symbols, MeSH terms, PMIDs, journal names, and titles exactly as written.
- Do not invent mechanisms, URLs, papers, counts, or entity names.
- Do not reinterpret CO_OCCURS as co-expression or causality.
- Do not mention Cypher, Neo4j, rows, records, or the database query.
- If there are no rows, say no matching results were found.
- Keep the answer to 2-4 concise sentences.
- Output only the final answer.

User question:
{question}

Cypher used internally:
{cypher}

Database results:
{formatted_results}"""
        messages = [{"role": "user", "content": content}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _clean_answer(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^</?s>\s*", "", text)
        text = re.split(r"\b(?:Cypher|Neo4j|Rows?|Records?|Query):", text, maxsplit=1, flags=re.IGNORECASE)[0]
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

        if formatted_results == "NO_ROWS":
            return "No matching results were found in the knowledge graph."

        prompt = self._build_prompt(question, cypher, formatted_results)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
                repetition_penalty=1.15,
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
    gen = BioMistralAnswerGenerator()
    print(
        gen.generate(
            "Find MeSH entitys co-occurring with ACSL3.",
            "MATCH (:Gene {ENTITY:'ACSL3'})-[:CO_OCCURS]-(m:MeSH) RETURN DISTINCT m.ENTITY AS mesh_entity",
            mock_results,
        )
    )
