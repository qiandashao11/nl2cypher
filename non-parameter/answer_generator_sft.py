"""
SFT LoRA-based answer generator.

This module loads the answer-agent LoRA trained in:

    answer_rl_training/lora_out_llama3_answer_sft_real_nonempty

It keeps the original Cypher generator unchanged.  Only the second-stage
answer generation model is replaced.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from answer_generator_clean import CleanAnswerGenerator
from answer_rl_training.answer_reward import score_answer


ANSWER_SYSTEM_PROMPT = """You are a biomedical knowledge graph answer writer.

Answer the user's question using only the provided Cypher result.

Rules:
- Output only the final answer text.
- Preserve every gene symbol, MeSH term, PMID, title, year, journal, and count exactly as provided.
- Do not invent entities, mechanisms, papers, URLs, or external facts.
- If the result is empty, say that no matching results were found.
- If many rows are returned, mention the total count and give representative examples from the result.
- For CO_OCCURS relationships, say "co-occurs", "is linked", or "is associated in the graph"; do not say cause, regulate, interact, or co-express unless those words appear in the result.
- If the result only contains gene_entity values, only list the returned genes and the count. Do not add themes, PMIDs, mechanisms, or representative terms.
- If the result only contains mesh_entity values, only list the returned MeSH terms and the count. Do not add genes, PMIDs, or mechanisms.
- Do not restate a MeSH term unless you copy it exactly from the question or result.
- Keep the answer concise and natural."""


class SFTAnswerGenerator:
    """Generate grounded natural-language answers with the trained answer LoRA."""

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        adapter_dir: str = "answer_rl_training/lora_out_llama3_answer_sft_real_nonempty",
        hf_token: str | None = None,
        max_display: int = 20,
        use_4bit: bool = True,
        use_reward_guard: bool = True,
        fallback_reward_threshold: float = 0.88,
    ):
        self.max_display = max_display
        self.adapter_dir = adapter_dir
        self.use_reward_guard = use_reward_guard
        self.fallback_reward_threshold = fallback_reward_threshold
        self.clean_renderer = CleanAnswerGenerator.__new__(CleanAnswerGenerator)
        self.clean_renderer.max_display = max_display
        self.last_reward_score: Dict[str, Any] | None = None

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            )

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=hf_token,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            quantization_config=quantization_config,
        )
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.eval()

    def _compact_result(self, query_results: Dict[str, Any]) -> Dict[str, Any]:
        data = query_results.get("data") or []
        compact: Dict[str, Any] = {
            "success": query_results.get("success", False),
            "count": query_results.get("count", len(data)),
            "data": data[: self.max_display],
        }
        if not query_results.get("success"):
            compact["error"] = query_results.get("error", "Unknown error")
        if len(data) > self.max_display or query_results.get("count", len(data)) > len(data[: self.max_display]):
            compact["truncated"] = True
            compact["note"] = f"Only the first {len(data[: self.max_display])} rows are shown."
        return compact

    def _build_user_prompt(self, question: str, cypher: str, query_results: Dict[str, Any]) -> str:
        result_text = json.dumps(self._compact_result(query_results), ensure_ascii=False, indent=2)
        return f"""Question:
{question}

Cypher:
{cypher}

Cypher result:
{result_text}

Write the final answer."""

    def _build_prompt(self, question: str, cypher: str, query_results: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(question, cypher, query_results)},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _clean_answer(self, text: str) -> str:
        text = text.strip()
        text = re.split(r"\n?\s*(?:Final answer|Answer|Text):\s*", text, maxsplit=1, flags=re.IGNORECASE)[-1]
        text = re.split(r"\b(?:Question|Cypher result|Cypher):", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\b(?:Best regards|Let me know|As an AI)\b.*", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(
            r"(The returned items provide direct graph evidence for the requested relationship\.).*",
            r"\1",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _contains_exact(self, answer: str, value: Any) -> bool:
        return re.sub(r"\s+", " ", str(value).strip().lower()) in re.sub(r"\s+", " ", answer.strip().lower())

    def _quoted_cypher_values(self, cypher: str) -> list[str]:
        values = []
        for single, double in re.findall(r"'([^']+)'|\"([^\"]+)\"", cypher):
            value = single or double
            if value:
                values.append(value)
        return values

    def _looks_like_entity_value(self, value: str) -> bool:
        if "/" in value or value.isdigit():
            return True
        if re.fullmatch(r"[A-Z0-9][A-Za-z0-9_.-]+", value):
            return True
        return False

    def _needs_fallback(
        self,
        question: str,
        cypher: str,
        query_results: Dict[str, Any],
        answer: str,
        reward: Dict[str, Any],
    ) -> bool:
        if reward["reward"] < self.fallback_reward_threshold:
            return True

        data = query_results.get("data") or []
        result_count = int(query_results.get("count", len(data)) or 0)
        if query_results.get("success") and result_count <= 5:
            required_keys = {"gene_entity", "mesh_entity", "pmid", "title", "journal", "journal_name", "year"}
            for row in data:
                if not isinstance(row, dict):
                    continue
                for key, value in row.items():
                    if key in required_keys and value is not None and not self._contains_exact(answer, value):
                        return True

        for value in self._quoted_cypher_values(cypher):
            if self._looks_like_entity_value(value) and not self._contains_exact(answer, value):
                # The model may omit filter values for large summarized results,
                # but if it mentions a distorted variant the reward usually drops.
                # For count/single-row answers, exact filter values should appear.
                if result_count <= 5 or any(word in question.lower() for word in ["count", "both", "shared"]):
                    return True

        details = reward.get("details") or {}
        entity_detail = details.get("entity") or {}
        if entity_detail.get("unsupported_pmids") or entity_detail.get("unsupported_gene_like_tokens"):
            return True
        return False

    def generate(self, question: str, cypher: str, query_results: Dict[str, Any]) -> str:
        prompt = self._build_prompt(question, cypher, query_results)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=90,
                do_sample=False,
                repetition_penalty=1.18,
                no_repeat_ngram_size=6,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        answer = self._clean_answer(answer)
        reward = score_answer(question, cypher, query_results, answer)
        self.last_reward_score = reward
        if self.use_reward_guard and self._needs_fallback(question, cypher, query_results, answer, reward):
            fallback = CleanAnswerGenerator._fallback_answer(self.clean_renderer, question, query_results)
            if fallback:
                self.last_reward_score = score_answer(question, cypher, query_results, fallback)
                return fallback
        return answer


if __name__ == "__main__":
    mock_results = {
        "success": True,
        "data": [
            {"gene_entity": "S100A4"},
            {"gene_entity": "ATL1"},
            {"gene_entity": "AIFM2"},
            {"gene_entity": "ACSL4"},
        ],
        "count": 4,
    }
    generator = SFTAnswerGenerator()
    print(
        generator.generate(
            "Find genes related to both MeSH antineoplastic agents/pharmacology and MeSH neoplasms/drug therapy.",
            "MATCH (:MeSH {ENTITY:'antineoplastic agents/pharmacology'})-[:CO_OCCURS]-(g:Gene)-[:CO_OCCURS]-(:MeSH {ENTITY:'neoplasms/drug therapy'}) RETURN DISTINCT g.ENTITY AS gene_entity",
            mock_results,
        )
    )
