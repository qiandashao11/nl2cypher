# ==================== answer_generator.py ====================
"""
Natural-language answer generator (final stable version)
- Completely forbid code block output
- Prevent repeated ```python output
- Use two-stage generation (draft + refinement)
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, Any


class AnswerGenerator:
    """Natural-language answer generator"""

    def __init__(self,
                 base_model="meta-llama/Llama-3.1-8B-Instruct",
                 hf_token=None,
                 enable_refine=True,
                 max_new_tokens=1024):
        """
        Initialize the generator
        """
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=hf_token,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
        )
        self.model.eval()

        self.enable_refine = enable_refine
        self.max_new_tokens = max_new_tokens

        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.eos_token_id

    # ----------------------------------------------------------------------
    # Format Neo4j query results
    # ----------------------------------------------------------------------
    def format_results(self, query_results: Dict[str, Any]) -> str:

        if not query_results["success"]:
            return f"Query execution failed: {query_results.get('error','Unknown error')}"

        if not query_results["data"]:
            return "No results returned."

        max_display = 20
        data = query_results["data"][:max_display]
        count = query_results["count"]

        text = f"Query returned {count} records"
        if count > max_display:
            text += f" (showing first {max_display})"
        text += ":\n\n"

        for i, row in enumerate(data, 1):
            text += f"Record {i}:\n"
            for k, v in row.items():
                text += f"  - {k}: {v}\n"
            text += "\n"

        return text

    # ----------------------------------------------------------------------
    # Shared wrapper used for the first and second passes
    # ----------------------------------------------------------------------
    def _generate_text(self, prompt: str, tag: str = "") -> str:

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.18,     
                return_dict_in_generate=True,
            )

        seq = outputs.sequences[0]
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = seq[prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        text = text.replace("```", "").replace("`", "")

        print(f"[{tag}] Generated:", text)
        return text

    # ----------------------------------------------------------------------
    # Stage 1: draft generation
    # ----------------------------------------------------------------------
    def _first_pass(self, question: str, cypher: str, query_results: Dict[str, Any]) -> str:

        formatted = self.format_results(query_results)

        prompt = f"""
            You are a biomedical knowledge graph Q&A assistant.

            Your task:
            Given the user's question, the Cypher query, and the query results,
            produce ONE short natural-language answer (2–4 sentences).

            STRICT RULES:
            - Output ONLY plain text.
            - DO NOT output code blocks.
            - DO NOT output backticks (`).
            - DO NOT output anything resembling ```python or ```text.
            - DO NOT include the question.
            - DO NOT include the Cypher query.
            - DO NOT mention “query results”, “records”, graphs, or databases.
            - DO NOT list items or use bullet points.
            - DO NOT repeat sentences.
            - DO NOT ask questions.
            - DO NOT explain reasoning.

            User question:
            {question}

            Cypher query:
            {cypher}

            Query results (for reference only, do NOT mention them explicitly):
            {formatted}

            Write the answer now (plain text only):
            """

        return self._generate_text(prompt, tag="FIRST PASS")

    # ----------------------------------------------------------------------
    # Stage 2: refined generation (more natural)
    # ----------------------------------------------------------------------
    def _refine_pass(self, question: str, cypher: str, draft: str) -> str:

        prompt = f"""
            You are a biomedical knowledge graph Q&A assistant.

            Your job:
            Review and slightly refine the draft answer below. 
            The meaning MUST remain the same.
            The output MUST be 2–4 fluent sentences of plain English.

            STRICT RULES:
            - Output ONLY plain text.
            - DO NOT output backticks.
            - DO NOT output code blocks.
            - DO NOT use ``` or anything similar.
            - DO NOT add new facts.
            - DO NOT mention queries, Cypher, records, or databases.
            - DO NOT repeat ideas or sentences.

            Draft answer:
            {draft}

            Now output the improved final answer (plain text only):
            """

        return self._generate_text(prompt, tag="REFINE PASS")

    # ----------------------------------------------------------------------
    # Public interface
    # ----------------------------------------------------------------------
    def generate(self, question: str, cypher: str, query_results: Dict[str, Any]) -> str:

        draft = self._first_pass(question, cypher, query_results)

        if not self.enable_refine:
            return draft

        final = self._refine_pass(question, cypher, draft)
        return final


# ==================== Self-test ====================
if __name__ == "__main__":
    gen = AnswerGenerator(enable_refine=False)

    mock_results = {
        "success": True,
        "data": [
            {"entity": "antineoplastic agents"},
            {"entity": "autophagy"},
        ],
        "count": 2
    }

    ans = gen.generate(
        "What MeSH entities are mentioned in this paper?",
        "MATCH (m:MeSH) RETURN m.entity",
        mock_results
    )

    print("\n=== Final Answer ===\n", ans)
