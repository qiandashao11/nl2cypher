# ==================== answer_generator_clean.py ====================
"""
Clean natural-language answer generator.

This module keeps the same high-level interface as answer_generator.py, but it
adds stricter output cleanup and deterministic handling for empty/error results.
"""
import re
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


THEME_RULES = [
    ("oxidative stress", ["oxidative stress", "glutathione", "peroxidase", "ferroptosis"]),
    ("lipid metabolism", ["lipid", "phospholipid", "oleic acid", "fatty acid", "metabolism"]),
    ("cancer and tumor biology", ["neoplasm", "tumor", "melanoma", "glioblastoma", "lymphoma", "metastasis", "neoplastic"]),
    ("immune processes", ["immun", "t-lymphocyte", "cytokine", "interleukin", "killer cell"]),
    ("genetics and gene regulation", ["genetic", "genetics", "gene expression", "variation"]),
    ("drug response", ["drug effects", "drug therapy", "pharmacology", "antineoplastic"]),
    ("cell survival and cell death", ["cell survival", "autophagy", "apoptosis", "cell death"]),
    ("iron-related biology", ["iron", "heme", "ferritin"]),
]


class CleanAnswerGenerator:
    """Convert Neo4j query results into concise natural-language answers."""

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
            return f"ERROR: {query_results.get('error', 'Unknown error')}"

        data = query_results.get("data") or []
        if not data:
            return "EMPTY_RESULT"

        records = data[: self.max_display]
        lines = [f"TOTAL_RECORDS: {query_results.get('count', len(data))}"]
        for i, record in enumerate(records, 1):
            values = []
            for key, value in record.items():
                values.append(f"{key}={value}")
            lines.append(f"RECORD_{i}: " + "; ".join(values))
        return "\n".join(lines)

    def _fallback_answer(self, question: str, query_results: Dict[str, Any]) -> str | None:
        if not query_results.get("success"):
            return "I could not answer the question because the generated Cypher query failed to execute."

        data = query_results.get("data") or []
        if not data:
            return "No matching results were found in the knowledge graph."

        if query_results.get("count") == 1 and len(data[0]) == 1:
            key, value = next(iter(data[0].items()))
            if isinstance(value, bool):
                return self._boolean_answer(question, key, value)
            if key.endswith("_count") or key in {"count", "paper_count", "gene_count", "mesh_count"}:
                return self._count_answer(question, key, value)

        keys = set().union(*(row.keys() for row in data))
        total = query_results.get("count", len(data))

        if keys == {"mesh_entity"}:
            values = [row["mesh_entity"] for row in data[:8]]
            return self._mesh_answer(question, values, total)

        if keys == {"gene_entity"}:
            values = [row["gene_entity"] for row in data[:8]]
            focus = self._focus_from_question(question)
            prefix = f"For {focus}, " if focus else ""
            return prefix + self._list_answer("genes", values, total)

        if keys == {"pmid"}:
            values = [row["pmid"] for row in data[:8]]
            focus = self._focus_from_question(question)
            prefix = f"For {focus}, " if focus else ""
            return prefix + self._list_answer("papers", values, total)

        if {"pmid", "title", "year", "journal"}.issubset(keys):
            return self._paper_answer(question, data, total)

        if {"gene_entity", "mesh_entity"}.issubset(keys) and "pmid" not in keys:
            examples = [
                f"{row.get('gene_entity')} with {row.get('mesh_entity')}"
                for row in data[:5]
            ]
            themes = self._infer_themes([row.get("mesh_entity", "") for row in data])
            theme_sentence = self._theme_sentence(themes)
            return f"Found {total} matching gene-MeSH pairs. {theme_sentence} Examples include {', '.join(examples)}."

        if {"pmid", "gene_entity", "mesh_entity"}.issubset(keys):
            examples = [
                f"{row.get('gene_entity')} / {row.get('mesh_entity')} in PMID {row.get('pmid')}"
                for row in data[:5]
            ]
            themes = self._infer_themes([row.get("mesh_entity", "") for row in data])
            theme_sentence = self._theme_sentence(themes)
            return f"Found {total} paper-linked gene-MeSH associations. {theme_sentence} Examples include {', '.join(examples)}."

        return None

    def _count_answer(self, question: str, key: str, value: Any) -> str:
        focus = self._focus_from_question(question)
        if value == 0:
            if "shared" in question.lower():
                return f"No shared MeSH entities were found for {focus}." if focus else "No shared MeSH entities were found."
            return "The count is 0, so no matching items were found."
        if "shared" in question.lower() and "mesh" in question.lower():
            label = "MeSH entity" if value == 1 else "MeSH entities"
            return f"{focus} share {value} {label} in the knowledge graph." if focus else f"There are {value} shared {label}."
        label = key.replace("_", " ")
        return f"The {label} is {value}."

    def _boolean_answer(self, question: str, key: str, value: bool) -> str:
        gene_mesh = re.search(r"Does\s+(.+?)\s+co-occur with\s+MeSH\s+(.+?)\?", question, flags=re.IGNORECASE)
        if gene_mesh:
            gene = gene_mesh.group(1).strip()
            mesh = gene_mesh.group(2).strip()
            if value:
                return f"Yes, {gene} co-occurs with MeSH {mesh} in the knowledge graph."
            return f"No, {gene} does not co-occur with MeSH {mesh} in the knowledge graph."
        if value:
            return "Yes, the requested relationship exists in the knowledge graph."
        return "No, the requested relationship was not found in the knowledge graph."

    def _list_answer(self, label: str, values: list[Any], total: int) -> str:
        rendered = [str(v) for v in values]
        if not rendered:
            return "No matching results were found in the knowledge graph."
        if total <= len(rendered):
            return f"Found {total} matching {label}: {', '.join(rendered)}."
        return f"Found {total} matching {label}. Examples include {', '.join(rendered)}."

    def _mesh_answer(self, question: str, values: list[Any], total: int) -> str:
        focus = self._focus_from_question(question)
        themes = self._infer_themes(values)
        theme_sentence = self._theme_sentence(themes)
        examples = ", ".join(str(v) for v in values[:6])
        subject = focus or "the query"
        label = "MeSH entity" if total == 1 else "MeSH entities"
        if focus and " and " in focus:
            return (
                f"{subject} share {total} {label} in the knowledge graph. "
                f"{theme_sentence} Representative terms include {examples}."
            )
        return (
            f"{subject} is linked to {total} {label} in the knowledge graph. "
            f"{theme_sentence} Representative terms include {examples}."
        )

    def _paper_answer(self, question: str, data: list[dict[str, Any]], total: int) -> str:
        first = data[0]
        title = first.get("title") or "an untitled paper"
        pmid = first.get("pmid")
        year = first.get("year")
        journal = first.get("journal")
        focus = self._focus_from_question(question)
        focus_text = f" for {focus}" if focus else ""
        if total == 1:
            return (
                f"One matching paper was found{focus_text}: \"{title}\" "
                f"({journal}, {year}; PMID: {pmid})."
            )
        examples = [
            f"\"{row.get('title') or 'untitled'}\" ({row.get('journal')}, {row.get('year')}; PMID: {row.get('pmid')})"
            for row in data[:3]
        ]
        return f"{total} matching papers were found{focus_text}. Examples include {'; '.join(examples)}."

    def _infer_themes(self, values: list[Any]) -> list[str]:
        text = " ".join(str(v).lower() for v in values)
        themes = []
        for theme, keywords in THEME_RULES:
            if any(keyword in text for keyword in keywords):
                themes.append(theme)
        return themes[:4]

    def _theme_sentence(self, themes: list[str]) -> str:
        if not themes:
            return "The returned items provide direct graph evidence for the requested relationship."
        if len(themes) == 1:
            return f"The returned items mainly point to {themes[0]}."
        return f"The returned items mainly point to {', '.join(themes[:-1])}, and {themes[-1]}."

    def _focus_from_question(self, question: str) -> str | None:
        shared = re.search(r"shared by ([^.?!]+)", question, flags=re.IGNORECASE)
        if shared:
            return shared.group(1).strip()
        pmid = re.search(r"PMID\s+([A-Za-z0-9_-]+)", question, flags=re.IGNORECASE)
        if pmid:
            return f"PMID {pmid.group(1)}"
        mesh = re.search(r"MeSH\s+(.+?)(?:[.?]|$)", question, flags=re.IGNORECASE)
        if mesh:
            return f"MeSH {mesh.group(1).strip()}"
        gene = re.search(r"\b(?:with|to|linked to|report)\s+([A-Z0-9][A-Za-z0-9_.-]+)\b", question)
        if gene:
            return gene.group(1).strip()
        return None

    def _clean_answer(self, text: str) -> str:
        text = text.strip()

        # Drop common model meta-comments and politeness tails.
        text = re.split(r"\bBookmark\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.split(r"\bBookmarklet\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.split(r"\n?\s*Answer:\s*", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.split(r"\n?\s*Text:\s*", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\(?\s*Note:\s*I have followed.*", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"\bThank you!?\s*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bHere is the final answer:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bFinal answer:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+\Z", "", text)

        # Remove accidental raw-result framing.
        text = re.sub(r"\b(query results?|records?)\b\s*:?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def generate(self, question: str, cypher: str, query_results: Dict[str, Any]) -> str:
        fallback = self._fallback_answer(question, query_results)
        if fallback:
            return fallback

        formatted_results = self.format_results(query_results)
        prompt = f"""You are a biomedical knowledge graph answer writer.

Write a concise final answer to the user's question using only the provided results.

Rules:
- Output only the answer text.
- Use 1-3 short sentences.
- Do not mention Cypher, records, query results, or database internals.
- Do not add notes, thanks, disclaimers, or explanations of your instructions.
- If the result contains IDs or entity names, summarize the most relevant ones.
- If many results are present, mention the total count and list a few examples.

Question:
{question}

Cypher:
{cypher}

Results:
{formatted_results}

Answer:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._clean_answer(text)


if __name__ == "__main__":
    mock = {
        "success": True,
        "data": [{"mesh_entity": "tumor suppressor protein P53 genetics"}],
        "count": 1,
    }
    gen = CleanAnswerGenerator()
    print(
        gen.generate(
            "Find MeSH entitys co-occurring with TP53.",
            "MATCH (:Gene {ENTITY:'TP53'})-[:CO_OCCURS]-(m:MeSH) RETURN DISTINCT m.ENTITY AS mesh_entity",
            mock,
        )
    )
