"""
Generate a larger non-empty SFT dataset directly from real Neo4j facts.

Unlike build_answer_sft_dataset.py, this script does not start from synthetic
question/Cypher pairs that may or may not match the current database.  It first
samples entities, papers, journals, and relationship combinations that actually
exist in Neo4j, then creates question/Cypher pairs from those facts.

The output is intended as the main SFT dataset for the second-stage answer
agent.  Empty results are skipped by default.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neo4j_executor import Neo4jExecutor

from answer_reward import score_answer
from build_answer_sft_dataset import (
    ANSWER_SYSTEM_PROMPT,
    build_user_prompt,
    execute_with_sampling,
    make_renderer,
    render_clean_answer,
)


def cypher_str(value: Any) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def query_rows(executor: Neo4jExecutor, cypher: str, database: str) -> List[Dict[str, Any]]:
    result = executor.execute(cypher, database=database)
    if not result.get("success"):
        raise RuntimeError(f"Candidate query failed: {result.get('error')}\n{cypher}")
    return result.get("data") or []


def has_positive_result(db_result: Dict[str, Any]) -> bool:
    if not db_result.get("success"):
        return False
    data = db_result.get("data") or []
    if not data:
        return False

    if len(data) == 1 and isinstance(data[0], dict) and len(data[0]) == 1:
        key, value = next(iter(data[0].items()))
        if isinstance(value, bool):
            return value
        if key.endswith("_count") or key in {"count", "paper_count", "gene_count", "mesh_count"}:
            try:
                return int(value) > 0
            except (TypeError, ValueError):
                return False
    return int(db_result.get("count", len(data))) > 0


def choose(rng: random.Random, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return rows[rng.randrange(len(rows))]


def build_candidates(executor: Neo4jExecutor, database: str) -> Dict[str, List[Dict[str, Any]]]:
    candidates = {
        "genes": query_rows(
            executor,
            "MATCH (g:Gene) RETURN DISTINCT g.ENTITY AS gene_entity",
            database,
        ),
        "mesh": query_rows(
            executor,
            "MATCH (m:MeSH) RETURN DISTINCT m.ENTITY AS mesh_entity",
            database,
        ),
        "papers": query_rows(
            executor,
            "MATCH (l:Literature) RETURN DISTINCT l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal",
            database,
        ),
        "gene_mesh": query_rows(
            executor,
            "MATCH (g:Gene)-[:CO_OCCURS]-(m:MeSH) RETURN DISTINCT g.ENTITY AS gene_entity, m.ENTITY AS mesh_entity",
            database,
        ),
        "gene_lit": query_rows(
            executor,
            "MATCH (g:Gene)-[:HAS_SOURCE]->(l:Literature) RETURN DISTINCT g.ENTITY AS gene_entity, l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal",
            database,
        ),
        "mesh_lit": query_rows(
            executor,
            "MATCH (m:MeSH)-[:HAS_SOURCE]->(l:Literature) RETURN DISTINCT m.ENTITY AS mesh_entity, l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal",
            database,
        ),
        "paper_gene_mesh": query_rows(
            executor,
            "MATCH (g:Gene)-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(m:MeSH) "
            "RETURN DISTINCT g.ENTITY AS gene_entity, m.ENTITY AS mesh_entity, l.PMID AS pmid, "
            "l.Title AS title, l.Year AS year, l.Journal AS journal",
            database,
        ),
        "gene_gene_paper": query_rows(
            executor,
            "MATCH (g1:Gene)-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(g2:Gene) "
            "WHERE g1.ENTITY < g2.ENTITY "
            "RETURN DISTINCT g1.ENTITY AS gene1, g2.ENTITY AS gene2, l.PMID AS pmid, "
            "l.Title AS title, l.Year AS year, l.Journal AS journal",
            database,
        ),
        "shared_mesh_genes": query_rows(
            executor,
            "MATCH (g1:Gene)-[:CO_OCCURS]-(m:MeSH)-[:CO_OCCURS]-(g2:Gene) "
            "WHERE g1.ENTITY < g2.ENTITY "
            "RETURN DISTINCT g1.ENTITY AS gene1, g2.ENTITY AS gene2",
            database,
        ),
        "two_mesh_genes": query_rows(
            executor,
            "MATCH (m1:MeSH)-[:CO_OCCURS]-(g:Gene)-[:CO_OCCURS]-(m2:MeSH) "
            "WHERE m1.ENTITY < m2.ENTITY "
            "RETURN DISTINCT m1.ENTITY AS mesh1, m2.ENTITY AS mesh2, g.ENTITY AS gene_entity",
            database,
        ),
        "journals": query_rows(
            executor,
            "MATCH (l:Literature) WHERE l.Journal IS NOT NULL RETURN DISTINCT l.Journal AS journal",
            database,
        ),
    }
    return candidates


ExampleFactory = Callable[[random.Random, Dict[str, List[Dict[str, Any]]]], Optional[Tuple[str, str, str]]]


def make_factories() -> List[ExampleFactory]:
    def gene_to_mesh(rng, c):
        row = choose(rng, c["gene_mesh"])
        gene = row["gene_entity"]
        question = rng.choice([
            f"Find MeSH entities co-occurring with {gene}.",
            f"Which MeSH terms are linked to {gene} by CO_OCCURS?",
            f"List MeSH entities associated in the graph with {gene}.",
        ])
        cypher = f"MATCH (:Gene {{ENTITY:{cypher_str(gene)}}})-[:CO_OCCURS]-(m:MeSH) RETURN DISTINCT m.ENTITY AS mesh_entity"
        return "real_gene_to_mesh", question, cypher

    def mesh_to_gene(rng, c):
        row = choose(rng, c["gene_mesh"])
        mesh = row["mesh_entity"]
        question = rng.choice([
            f"Which genes co-occur with the MeSH entity {mesh}?",
            f"Find genes linked to MeSH {mesh}.",
            f"List genes associated in the graph with MeSH {mesh}.",
        ])
        cypher = f"MATCH (g:Gene)-[:CO_OCCURS]-(:MeSH {{ENTITY:{cypher_str(mesh)}}}) RETURN DISTINCT g.ENTITY AS gene_entity"
        return "real_mesh_to_gene", question, cypher

    def gene_literature(rng, c):
        row = choose(rng, c["gene_lit"])
        gene = row["gene_entity"]
        question = rng.choice([
            f"Show literature linked to {gene}.",
            f"Find papers that report {gene}.",
            f"Which papers are connected to gene {gene}?",
        ])
        cypher = (
            f"MATCH (:Gene {{ENTITY:{cypher_str(gene)}}})-[:HAS_SOURCE]->(l:Literature) "
            "RETURN DISTINCT l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal"
        )
        return "real_gene_literature", question, cypher

    def mesh_literature(rng, c):
        row = choose(rng, c["mesh_lit"])
        mesh = row["mesh_entity"]
        question = rng.choice([
            f"Show literature for MeSH {mesh}.",
            f"Find papers linked to the MeSH entity {mesh}.",
            f"Which papers are connected to MeSH {mesh}?",
        ])
        cypher = (
            f"MATCH (:MeSH {{ENTITY:{cypher_str(mesh)}}})-[:HAS_SOURCE]->(l:Literature) "
            "RETURN DISTINCT l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal"
        )
        return "real_mesh_literature", question, cypher

    def papers_gene_mesh(rng, c):
        row = choose(rng, c["paper_gene_mesh"])
        gene, mesh = row["gene_entity"], row["mesh_entity"]
        question = rng.choice([
            f"Find papers mentioning both {gene} and MeSH {mesh}.",
            f"Which papers contain both gene {gene} and MeSH {mesh}?",
            f"Show literature shared by {gene} and MeSH {mesh}.",
        ])
        cypher = (
            f"MATCH (:Gene {{ENTITY:{cypher_str(gene)}}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-"
            f"(:MeSH {{ENTITY:{cypher_str(mesh)}}}) "
            "RETURN DISTINCT l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal"
        )
        return "real_papers_gene_mesh", question, cypher

    def count_papers_gene_mesh(rng, c):
        row = choose(rng, c["paper_gene_mesh"])
        gene, mesh = row["gene_entity"], row["mesh_entity"]
        question = f"Count papers that mention both {gene} and MeSH {mesh}."
        cypher = (
            f"MATCH (:Gene {{ENTITY:{cypher_str(gene)}}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-"
            f"(:MeSH {{ENTITY:{cypher_str(mesh)}}}) RETURN count(DISTINCT l) AS paper_count"
        )
        return "real_count_papers_gene_mesh", question, cypher

    def papers_two_genes(rng, c):
        row = choose(rng, c["gene_gene_paper"])
        gene1, gene2 = row["gene1"], row["gene2"]
        question = rng.choice([
            f"Find papers that mention both {gene1} and {gene2}.",
            f"Which papers report both genes {gene1} and {gene2}?",
        ])
        cypher = (
            f"MATCH (:Gene {{ENTITY:{cypher_str(gene1)}}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-"
            f"(:Gene {{ENTITY:{cypher_str(gene2)}}}) "
            "RETURN DISTINCT l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal"
        )
        return "real_papers_two_genes", question, cypher

    def shared_mesh_two_genes(rng, c):
        row = choose(rng, c["shared_mesh_genes"])
        gene1, gene2 = row["gene1"], row["gene2"]
        question = rng.choice([
            f"Find MeSH entities shared by {gene1} and {gene2}.",
            f"Which MeSH terms co-occur with both {gene1} and {gene2}?",
        ])
        cypher = (
            f"MATCH (:Gene {{ENTITY:{cypher_str(gene1)}}})-[:CO_OCCURS]-(m:MeSH)-[:CO_OCCURS]-"
            f"(:Gene {{ENTITY:{cypher_str(gene2)}}}) RETURN DISTINCT m.ENTITY AS mesh_entity"
        )
        return "real_shared_mesh_two_genes", question, cypher

    def count_shared_mesh_two_genes(rng, c):
        row = choose(rng, c["shared_mesh_genes"])
        gene1, gene2 = row["gene1"], row["gene2"]
        question = f"Count MeSH entities shared by {gene1} and {gene2}."
        cypher = (
            f"MATCH (:Gene {{ENTITY:{cypher_str(gene1)}}})-[:CO_OCCURS]-(m:MeSH)-[:CO_OCCURS]-"
            f"(:Gene {{ENTITY:{cypher_str(gene2)}}}) RETURN count(DISTINCT m) AS shared_mesh_count"
        )
        return "real_count_shared_mesh_two_genes", question, cypher

    def genes_two_mesh(rng, c):
        row = choose(rng, c["two_mesh_genes"])
        mesh1, mesh2 = row["mesh1"], row["mesh2"]
        question = rng.choice([
            f"Find genes related to both MeSH {mesh1} and MeSH {mesh2}.",
            f"Which genes co-occur with both MeSH {mesh1} and MeSH {mesh2}?",
        ])
        cypher = (
            f"MATCH (:MeSH {{ENTITY:{cypher_str(mesh1)}}})-[:CO_OCCURS]-(g:Gene)-[:CO_OCCURS]-"
            f"(:MeSH {{ENTITY:{cypher_str(mesh2)}}}) RETURN DISTINCT g.ENTITY AS gene_entity"
        )
        return "real_genes_two_mesh", question, cypher

    def gene_after_year(rng, c):
        row = choose(rng, [r for r in c["gene_lit"] if r.get("year")])
        gene = row["gene_entity"]
        year = max(2000, int(row["year"]) - rng.choice([1, 2, 3]))
        question = rng.choice([
            f"{gene} papers after {year}.",
            f"Find literature for {gene} after {year}.",
        ])
        cypher = (
            f"MATCH (:Gene {{ENTITY:{cypher_str(gene)}}})-[:HAS_SOURCE]->(l:Literature) WHERE l.Year > {year} "
            "RETURN DISTINCT l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal"
        )
        return "real_gene_after_year", question, cypher

    def mesh_after_year(rng, c):
        row = choose(rng, [r for r in c["mesh_lit"] if r.get("year")])
        mesh = row["mesh_entity"]
        year = max(2000, int(row["year"]) - rng.choice([1, 2, 3]))
        question = rng.choice([
            f"Find literature for MeSH {mesh} after {year}.",
            f"Show papers linked to MeSH {mesh} after {year}.",
        ])
        cypher = (
            f"MATCH (:MeSH {{ENTITY:{cypher_str(mesh)}}})-[:HAS_SOURCE]->(l:Literature) WHERE l.Year > {year} "
            "RETURN DISTINCT l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal"
        )
        return "real_mesh_after_year", question, cypher

    def gene_in_journal(rng, c):
        row = choose(rng, [r for r in c["gene_lit"] if r.get("journal")])
        gene, journal = row["gene_entity"], row["journal"]
        question = f"Find {journal} papers that report {gene}."
        cypher = (
            f"MATCH (:Gene {{ENTITY:{cypher_str(gene)}}})-[:HAS_SOURCE]->(l:Literature) "
            f"WHERE l.Journal = {cypher_str(journal)} "
            "RETURN DISTINCT l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal"
        )
        return "real_gene_in_journal", question, cypher

    def mesh_in_journal(rng, c):
        row = choose(rng, [r for r in c["mesh_lit"] if r.get("journal")])
        mesh, journal = row["mesh_entity"], row["journal"]
        question = f"Find {journal} papers linked to MeSH {mesh}."
        cypher = (
            f"MATCH (:MeSH {{ENTITY:{cypher_str(mesh)}}})-[:HAS_SOURCE]->(l:Literature) "
            f"WHERE l.Journal = {cypher_str(journal)} "
            "RETURN DISTINCT l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal"
        )
        return "real_mesh_in_journal", question, cypher

    def genes_by_pmid(rng, c):
        row = choose(rng, c["gene_lit"])
        pmid = row["pmid"]
        question = f"Find genes reported in the paper with PMID {pmid}."
        cypher = f"MATCH (g:Gene)-[:HAS_SOURCE]->(:Literature {{PMID:{cypher_str(pmid)}}}) RETURN DISTINCT g.ENTITY AS gene_entity"
        return "real_genes_by_pmid", question, cypher

    def mesh_by_pmid(rng, c):
        row = choose(rng, c["mesh_lit"])
        pmid = row["pmid"]
        question = f"Find MeSH entities reported in the paper with PMID {pmid}."
        cypher = f"MATCH (m:MeSH)-[:HAS_SOURCE]->(:Literature {{PMID:{cypher_str(pmid)}}}) RETURN DISTINCT m.ENTITY AS mesh_entity"
        return "real_mesh_by_pmid", question, cypher

    def gene_mesh_pairs_by_pmid(rng, c):
        row = choose(rng, c["paper_gene_mesh"])
        pmid = row["pmid"]
        question = f"Find gene and MeSH pairs reported in the paper with PMID {pmid}."
        cypher = (
            f"MATCH (g:Gene)-[:HAS_SOURCE]->(l:Literature {{PMID:{cypher_str(pmid)}}})<-[:HAS_SOURCE]-(m:MeSH) "
            "RETURN DISTINCT g.ENTITY AS gene_entity, m.ENTITY AS mesh_entity"
        )
        return "real_gene_mesh_pairs_by_pmid", question, cypher

    def papers_for_genes_related_to_mesh(rng, c):
        row = choose(rng, c["gene_mesh"])
        mesh = row["mesh_entity"]
        question = f"Find papers for genes related to MeSH {mesh}."
        cypher = (
            f"MATCH (:MeSH {{ENTITY:{cypher_str(mesh)}}})-[:CO_OCCURS]-(g:Gene)-[:HAS_SOURCE]->(l:Literature) "
            "RETURN DISTINCT g.ENTITY AS gene_entity, l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal"
        )
        return "real_papers_for_genes_related_to_mesh", question, cypher

    def exists_gene_mesh(rng, c):
        row = choose(rng, c["gene_mesh"])
        gene, mesh = row["gene_entity"], row["mesh_entity"]
        question = f"Does {gene} co-occur with MeSH {mesh}?"
        cypher = (
            f"MATCH (:Gene {{ENTITY:{cypher_str(gene)}}})-[:CO_OCCURS]-(:MeSH {{ENTITY:{cypher_str(mesh)}}}) "
            "RETURN count(*) > 0 AS exists"
        )
        return "real_exists_gene_mesh", question, cypher

    def keyword_gene_papers(rng, c):
        row = choose(rng, c["paper_gene_mesh"])
        mesh = row["mesh_entity"]
        keyword = rng.choice([part for part in mesh.replace("/", " ").replace("-", " ").split() if len(part) >= 5])
        year = max(1990, int(row.get("year") or 2020) - rng.choice([1, 2, 3, 5]))
        question = f"Find papers where MeSH contains '{keyword}' and genes are reported after {year}."
        cypher = (
            "MATCH (m:MeSH)-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(g:Gene) "
            f"WHERE toLower(m.ENTITY) CONTAINS toLower({cypher_str(keyword)}) AND l.Year > {year} "
            "RETURN DISTINCT l.PMID AS pmid, g.ENTITY AS gene_entity, m.ENTITY AS mesh_entity"
        )
        return "real_keyword_gene_papers", question, cypher

    return [
        gene_to_mesh,
        mesh_to_gene,
        gene_literature,
        mesh_literature,
        papers_gene_mesh,
        count_papers_gene_mesh,
        papers_two_genes,
        shared_mesh_two_genes,
        count_shared_mesh_two_genes,
        genes_two_mesh,
        gene_after_year,
        mesh_after_year,
        gene_in_journal,
        mesh_in_journal,
        genes_by_pmid,
        mesh_by_pmid,
        gene_mesh_pairs_by_pmid,
        papers_for_genes_related_to_mesh,
        exists_gene_mesh,
        keyword_gene_papers,
    ]


def write_record(
    out,
    renderer,
    question: str,
    cypher: str,
    db_result: Dict[str, Any],
    example_type: str,
    source: str,
    index: int,
) -> None:
    answer = render_clean_answer(renderer, question, cypher, db_result)
    reward = score_answer(question, cypher, db_result, answer)
    user_prompt = build_user_prompt(question, cypher, db_result)
    record = {
        "messages": [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer},
        ],
        "question": question,
        "cypher": cypher,
        "db_result": db_result,
        "answer": answer,
        "metadata": {
            "type": example_type,
            "source": source,
            "source_index": index,
            "answer_reward": reward["reward"],
        },
        "reward_score": reward,
    }
    out.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real non-empty answer-agent SFT data from Neo4j.")
    parser.add_argument("--output", default="answer_rl_training/train.answer_sft.real_nonempty.jsonl")
    parser.add_argument("--target", type=int, default=3000)
    parser.add_argument("--max_rows", type=int, default=20)
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--neo4j_uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", default="neo4j")
    parser.add_argument("--neo4j_password", default="neo4j")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_attempts", type=int, default=80000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    renderer = make_renderer(max_display=args.max_rows)
    factories = make_factories()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_empty = 0
    skipped_duplicate = 0
    skipped_failed = 0
    seen = set()

    with Neo4jExecutor(args.neo4j_uri, args.neo4j_user, args.neo4j_password) as executor:
        candidates = build_candidates(executor, args.database)
        print("Candidate sizes:")
        for name, rows in candidates.items():
            print(f"- {name}: {len(rows)}")

        with output_path.open("w", encoding="utf-8") as out:
            for attempt in range(1, args.max_attempts + 1):
                if written >= args.target:
                    break
                factory = rng.choice(factories)
                try:
                    made = factory(rng, candidates)
                except (IndexError, ValueError):
                    skipped_empty += 1
                    continue
                if not made:
                    skipped_empty += 1
                    continue

                example_type, question, cypher = made
                key = (question, cypher)
                if key in seen:
                    skipped_duplicate += 1
                    continue
                seen.add(key)

                db_result = execute_with_sampling(executor, cypher, args.database, args.max_rows)
                if not db_result.get("success"):
                    skipped_failed += 1
                    continue
                if not has_positive_result(db_result):
                    skipped_empty += 1
                    continue

                write_record(out, renderer, question, cypher, db_result, example_type, "real_neo4j_sampling", attempt)
                written += 1
                if written % 100 == 0:
                    print(f"Written {written} records...")

    print(f"Done. Wrote {written} records to {output_path}.")
    print(f"Skipped empty/non-positive: {skipped_empty}")
    print(f"Skipped duplicates: {skipped_duplicate}")
    print(f"Skipped failed: {skipped_failed}")


if __name__ == "__main__":
    main()
