"""
Faithfulness reward for graph-grounded answer generation.

The reward is designed for the second-stage answer agent:

    question + generated Cypher + Neo4j result -> final natural-language answer

It emphasizes exact copying of database facts over free-form biomedical
interpretation.  The score is intentionally rule based so that obvious KGQA
errors such as changed PMIDs, changed gene symbols, wrong counts, or relation
over-interpretation are caught deterministically.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


IMPORTANT_FIELDS = {
    "gene_entity",
    "mesh_entity",
    "pmid",
    "title",
    "year",
    "journal",
    "journal_name",
    "paper_count",
    "gene_count",
    "mesh_count",
    "shared_mesh_count",
    "count",
}

BAD_META_PHRASES = [
    "step 1",
    "step 2",
    "step 3",
    "best regards",
    "let me know",
    "as an ai",
    "i can rephrase",
    "please let me know",
    "knowledge graph assistant",
    "final answer:",
    "answer:",
]

BAD_COOCCUR_WORDS = [
    "cause",
    "causes",
    "caused",
    "regulate",
    "regulates",
    "regulated",
    "interaction",
    "interacts",
    "interact",
    "co-expressed",
    "coexpressed",
    "expression",
    "mechanism",
    "pathway",
    "therapeutic target",
]

GOOD_COOCCUR_WORDS = [
    "co-occur",
    "co-occurs",
    "co-occurring",
    "linked",
    "associated",
    "reported together",
    "appears together",
]

NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}

GENE_STOPWORDS = {
    "PMID",
    "MESH",
    "KGQA",
    "DNA",
    "RNA",
    "URL",
    "HTTP",
    "HTTPS",
    "THE",
    "AND",
    "NOT",
    "ONLY",
    "TOTAL",
}


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def contains_exact(answer: str, value: Any) -> bool:
    text = normalize_text(answer)
    value_text = normalize_text(value)
    if not value_text:
        return False
    return value_text in text


def row_values(db_result: Dict[str, Any]) -> List[Tuple[str, Any]]:
    values: List[Tuple[str, Any]] = []
    for row in db_result.get("data") or []:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if key in IMPORTANT_FIELDS or key.endswith("_count"):
                values.append((key, value))
    return values


def exact_allowed_strings(question: str, cypher: str, db_result: Dict[str, Any]) -> Set[str]:
    allowed: Set[str] = set()
    for text in (question, cypher):
        for quoted in re.findall(r"'([^']+)'|\"([^\"]+)\"", text):
            allowed.add(next(part for part in quoted if part))
        for pmid in extract_pmids(text):
            allowed.add(pmid)
    for _, value in row_values(db_result):
        allowed.add(str(value))
    return {v for v in allowed if v}


def extract_pmids(text: str) -> Set[str]:
    return set(re.findall(r"\b\d{6,9}\b", text))


def extract_standalone_numbers(text: str) -> Set[int]:
    values = set()
    for raw in re.findall(r"(?<![A-Za-z0-9])\d{1,4}(?![A-Za-z0-9])", text):
        num = int(raw)
        if 1800 <= num <= 2100:
            continue
        values.add(num)
    lowered = text.lower()
    for number, word in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            values.add(number)
    return values


def extract_gene_like_tokens(text: str) -> Set[str]:
    tokens = set()
    for token in re.findall(r"\b[A-Z][A-Z0-9-]{1,}\b", text):
        if token in GENE_STOPWORDS:
            continue
        if token.isdigit():
            continue
        # Keep gene-like symbols while avoiding normal title-case words.
        if any(ch.isdigit() for ch in token) or token.isupper():
            tokens.add(token)
    return tokens


def expected_count(db_result: Dict[str, Any]) -> int:
    data = db_result.get("data") or []
    if len(data) == 1 and isinstance(data[0], dict) and len(data[0]) == 1:
        key, value = next(iter(data[0].items()))
        if key.endswith("_count") or key in {"count", "paper_count", "gene_count", "mesh_count"}:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return int(db_result.get("count", len(data)))


def entity_faithfulness(question: str, cypher: str, db_result: Dict[str, Any], answer: str) -> Tuple[float, Dict[str, Any]]:
    allowed_strings = exact_allowed_strings(question, cypher, db_result)
    allowed_text_blob = " ".join(allowed_strings)

    allowed_pmids = set()
    allowed_gene_tokens = set()
    for value in allowed_strings:
        allowed_pmids |= extract_pmids(value)
        allowed_gene_tokens |= extract_gene_like_tokens(value)

    answer_pmids = extract_pmids(answer)
    answer_gene_tokens = extract_gene_like_tokens(answer)

    unsupported_pmids = sorted(answer_pmids - allowed_pmids)
    unsupported_genes = []
    for token in sorted(answer_gene_tokens - allowed_gene_tokens):
        unsupported_genes.append(token)

    required_values = [
        value
        for key, value in row_values(db_result)
        if key in {"gene_entity", "mesh_entity", "pmid"} and str(value)
    ]
    if expected_count(db_result) <= 5:
        required_values += [
            value
            for key, value in row_values(db_result)
            if key in {"title", "journal", "journal_name", "year"} and str(value)
        ]

    missing = [str(value) for value in required_values if not contains_exact(answer, value)]
    unsupported_count = len(unsupported_pmids) + len(unsupported_genes)

    score = 1.0
    score -= min(1.0, unsupported_count * 0.35)
    if required_values:
        score -= 0.35 * (len(missing) / len(required_values))
    return max(0.0, score), {
        "unsupported_pmids": unsupported_pmids,
        "unsupported_gene_like_tokens": unsupported_genes,
        "missing_required_values": missing[:20],
    }


def count_accuracy(db_result: Dict[str, Any], answer: str) -> Tuple[float, Dict[str, Any]]:
    count = expected_count(db_result)
    answer_numbers = extract_standalone_numbers(answer)
    pmids = extract_pmids(answer)
    allowed_years = {
        int(value)
        for key, value in row_values(db_result)
        if key == "year" and str(value).isdigit()
    }
    filtered_numbers = {
        number for number in answer_numbers
        if str(number) not in pmids and number not in allowed_years
    }

    mentions_expected = count in filtered_numbers
    if count <= 10 and NUMBER_WORDS.get(count):
        mentions_expected = mentions_expected or re.search(
            rf"\b{NUMBER_WORDS[count]}\b", answer.lower()
        ) is not None

    wrong_numbers = sorted(number for number in filtered_numbers if number != count)
    if mentions_expected and not wrong_numbers:
        return 1.0, {"expected_count": count, "wrong_count_numbers": []}
    if mentions_expected and wrong_numbers:
        return 0.6, {"expected_count": count, "wrong_count_numbers": wrong_numbers}
    if wrong_numbers:
        return 0.0, {"expected_count": count, "wrong_count_numbers": wrong_numbers}
    return 0.5, {"expected_count": count, "wrong_count_numbers": []}


def result_coverage(db_result: Dict[str, Any], answer: str) -> Tuple[float, Dict[str, Any]]:
    data = db_result.get("data") or []
    count = expected_count(db_result)
    if not db_result.get("success"):
        return (1.0 if "failed" in answer.lower() or "could not" in answer.lower() else 0.2), {}
    if count == 0 or not data:
        ok = any(phrase in answer.lower() for phrase in ["no matching", "no results", "not found"])
        return (1.0 if ok else 0.0), {"expected_empty": True}

    values = []
    for key, value in row_values(db_result):
        if key in {"gene_entity", "mesh_entity", "pmid", "title", "journal", "journal_name", "year"}:
            values.append(value)

    if not values:
        return 0.7, {"covered_values": 0, "total_values": 0}

    if count <= 5:
        denominator = len(values)
        covered = sum(1 for value in values if contains_exact(answer, value))
        return covered / denominator, {"covered_values": covered, "total_values": denominator}

    sample_values = values[: min(8, len(values))]
    covered = sum(1 for value in sample_values if contains_exact(answer, value))
    count_score, _ = count_accuracy(db_result, answer)
    value_score = covered / max(1, len(sample_values))
    return 0.5 * count_score + 0.5 * value_score, {
        "covered_sample_values": covered,
        "total_sample_values": len(sample_values),
    }


def relation_semantics(cypher: str, answer: str) -> Tuple[float, Dict[str, Any]]:
    if "CO_OCCURS" not in cypher:
        return 1.0, {"relation": "not_co_occurs"}

    lowered = answer.lower()
    bad_hits = [word for word in BAD_COOCCUR_WORDS if word in lowered]
    good_hit = any(word in lowered for word in GOOD_COOCCUR_WORDS)

    if bad_hits:
        return 0.0, {"relation": "co_occurs", "bad_terms": bad_hits}
    if good_hit:
        return 1.0, {"relation": "co_occurs", "bad_terms": []}
    return 0.75, {"relation": "co_occurs", "bad_terms": []}


def format_cleanliness(answer: str) -> Tuple[float, Dict[str, Any]]:
    lowered = answer.lower()
    bad_hits = [phrase for phrase in BAD_META_PHRASES if phrase in lowered]
    repeated_markers = len(re.findall(r"#{2,}|\]\]>|]]0", answer))
    score = 1.0 - min(1.0, 0.25 * len(bad_hits) + 0.2 * repeated_markers)
    return max(0.0, score), {"bad_meta_phrases": bad_hits, "artifact_count": repeated_markers}


def fluency(answer: str) -> Tuple[float, Dict[str, Any]]:
    stripped = answer.strip()
    if not stripped:
        return 0.0, {"reason": "empty"}
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", stripped)) or 1
    token_count = len(stripped.split())
    score = 1.0
    if sentence_count > 5:
        score -= 0.25
    if token_count < 5:
        score -= 0.25
    if token_count > 180:
        score -= 0.25
    if stripped.endswith(("of", "with", "and", "between", "to")):
        score -= 0.3
    return max(0.0, score), {"sentence_count": sentence_count, "token_count": token_count}


def score_answer(question: str, cypher: str, db_result: Dict[str, Any], answer: str) -> Dict[str, Any]:
    entity_score, entity_detail = entity_faithfulness(question, cypher, db_result, answer)
    count_score, count_detail = count_accuracy(db_result, answer)
    coverage_score, coverage_detail = result_coverage(db_result, answer)
    relation_score, relation_detail = relation_semantics(cypher, answer)
    clean_score, clean_detail = format_cleanliness(answer)
    fluency_score, fluency_detail = fluency(answer)

    components = {
        "entity_faithfulness": entity_score,
        "count_accuracy": count_score,
        "result_coverage": coverage_score,
        "relation_semantics": relation_score,
        "format_cleanliness": clean_score,
        "fluency": fluency_score,
    }
    weighted = (
        0.30 * entity_score
        + 0.20 * count_score
        + 0.20 * coverage_score
        + 0.15 * relation_score
        + 0.10 * clean_score
        + 0.05 * fluency_score
    )

    unsupported = len(entity_detail["unsupported_pmids"]) + len(entity_detail["unsupported_gene_like_tokens"])
    hallucination_penalty = min(0.8, unsupported * 0.15)
    final_score = max(0.0, min(1.0, weighted - hallucination_penalty))

    return {
        "reward": round(final_score, 4),
        "weighted_reward_before_penalty": round(weighted, 4),
        "hallucination_penalty": round(hallucination_penalty, 4),
        "components": {key: round(value, 4) for key, value in components.items()},
        "details": {
            "entity": entity_detail,
            "count": count_detail,
            "coverage": coverage_detail,
            "relation": relation_detail,
            "format": clean_detail,
            "fluency": fluency_detail,
        },
    }


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score answer-agent outputs with the KG faithfulness reward.")
    parser.add_argument("--input", required=True, help="JSONL with question, cypher, db_result, and answer fields")
    parser.add_argument("--output", default=None, help="Optional JSONL path for scored records")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    scored = []
    for record in iter_jsonl(input_path):
        question = record.get("question", "")
        cypher = record.get("cypher", "")
        db_result = record.get("db_result") or record.get("query_result") or {}
        answer = record.get("answer", "")
        score = score_answer(question, cypher, db_result, answer)
        scored_record = {**record, "reward_score": score}
        scored.append(scored_record)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for record in scored:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    if scored:
        avg = sum(item["reward_score"]["reward"] for item in scored) / len(scored)
        print(f"Scored {len(scored)} records. Average reward: {avg:.4f}")
    else:
        print("No records scored.")


if __name__ == "__main__":
    main()
