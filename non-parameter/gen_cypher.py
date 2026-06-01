#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full NL→Cypher dataset generator for the Gene/MeSH/Literature graph.

Schema:
  Gene(entity, `Closeness.centrality`)
  MeSH(entity)
  Literature(PMID, Title, Year, Journal)

Relations:
  HAS_SOURCE: (Gene|MeSH) -> (Literature)     # directed to Literature ONLY
  CO_OCCURS:  Gene — Gene  or  Gene — MeSH    # treated as undirected in queries

Generates chat-style JSONL training data including:
  - 28 base templates
  - 12 extra templates (PMID, multi-IN, co-entity)
  - 1 thematic-year filter template (e.g., immunology + Year>2020)

Usage:
  python gen_nl2cypher_full.py \
      --data /path/to/gene_mesh_dataset.csv \
      --n 2000 \
      --out /path/to/train.chat.jsonl
  # If auto-detection of columns fails, pass: --gene-col entity --mesh-col entity
"""

import argparse
import csv
import json
import pathlib
import random
from typing import List, Dict, Optional, Tuple

# -------------------- helpers --------------------
JOURNALS = ["Nature","Science","Cell","PNAS","Lancet","NEJM","BMJ","JAMA"]
TITLE_KEYWORDS = ["cancer","genome","mutation","immune","signal","pathway","tumor","metabolism"]
THEMATIC_KEYWORDS = ["immunology","oncology","metabolism","apoptosis","neurology"]

def cypher_prop(p: str) -> str:
    return f"`{p}`" if "." in p else p

def choose(lst: List[str]) -> str:
    return random.choice(lst)

def dedup(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out

def q(nl: str, cy: str, qtype: str) -> Dict:
    # 保证 assistant 的 Cypher 是单行、无多余空白
    cy_single_line = cy.replace("\n", " ").strip()
    return {
        "messages": [
            {"role": "user", "content": nl},
            {"role": "assistant", "content": cy_single_line},
        ],
        "metadata": {"type": qtype}
    }

# -------------------- load lists --------------------
def load_lists(
    data_path: Optional[str] = None,
    gene_col: Optional[str] = None,
    mesh_col: Optional[str] = None,
    genes_path: Optional[str] = None,
    meshs_path: Optional[str] = None,
    max_items: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    genes, meshs = [], []
    if data_path:
        with open(data_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise SystemExit("CSV missing header.")
            def detect(explicit: Optional[str], keys: List[str]) -> Optional[str]:
                if explicit:
                    return explicit
                for c in reader.fieldnames:
                    if any(k in c.lower() for k in keys):
                        return c
                return None
            gc = detect(gene_col, ["gene","entity","gene_entity"])
            mc = detect(mesh_col, ["mesh","entity","mesh_entity"])
            if not gc or not mc:
                raise SystemExit(
                    f"Auto-detect failed. Detected columns: {reader.fieldnames}. "
                    f"Pass --gene-col and --mesh-col."
                )
            for r in reader:
                g = (r.get(gc) or "").strip()
                m = (r.get(mc) or "").strip()
                if g: genes.append(g)
                if m: meshs.append(m)

    if genes_path:
        with open(genes_path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                genes.append(s.split(",")[0])

    if meshs_path:
        with open(meshs_path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                meshs.append(s.split(",")[0])

    genes, meshs = dedup(genes), dedup(meshs)

    if max_items:
        random.shuffle(genes)
        random.shuffle(meshs)
        genes = genes[:max_items]
        meshs = meshs[:max_items]

    if not genes or not meshs:
        raise SystemExit("Empty gene or MeSH list.")
    return genes, meshs

# -------------------- augmentation (safe no-op for directions) --------------------
def maybe_reverse(cypher: str) -> str:
    """
    Safe augmentation placeholder:
    We DO NOT flip HAS_SOURCE direction.
    For CO_OCCURS we already use undirected '-[:CO_OCCURS]-',
    so there is nothing to reverse. Return cypher unchanged.
    """
    return cypher

# -------------------- 28 base templates --------------------
def make_basic_28(g: str, m: str) -> List[Dict]:
    y = random.randint(2000, 2024)
    y2 = random.randint(2018, 2024)
    kw = choose(TITLE_KEYWORDS)
    journal = choose(JOURNALS)
    c = cypher_prop("Closeness.centrality")

    out: List[Dict] = []
    # Lists (return node objects only)
    out.append(q("List all genes.", "MATCH (g:Gene) RETURN g", "list_genes"))
    out.append(q("List all MeSH entitys.", "MATCH (m:MeSH) RETURN m", "list_mesh"))
    out.append(q("List literature nodes.", "MATCH (l:Literature) RETURN l", "list_lit"))

    out.append(q(
        f"Find MeSH entitys co-occurring with {g}.",
        f"MATCH (:Gene {{entity:'{g}'}})-[:CO_OCCURS]-(m:MeSH) RETURN DISTINCT m",
        "gene_to_mesh"
    ))
    out.append(q(
        f"Which genes co-occur with the MeSH entity {m}?",
        f"MATCH (g:Gene)-[:CO_OCCURS]-(:MeSH {{entity:'{m}'}}) RETURN DISTINCT g",
        "mesh_to_gene"
    ))

    out.append(q(
        f"Show literature linked to {g}.",
        f"MATCH (:Gene {{entity:'{g}'}})-[:HAS_SOURCE]->(l:Literature) RETURN DISTINCT l",
        "gene_to_lit"
    ))
    out.append(q(
        f"Show literature for MeSH {m}.",
        f"MATCH (:MeSH {{entity:'{m}'}})-[:HAS_SOURCE]->(l:Literature) RETURN DISTINCT l",
        "mesh_to_lit"
    ))

    out.append(q(
        f"Papers whose title contains '{kw}'.",
        f"MATCH (l:Literature) WHERE toLower(l.Title) CONTAINS toLower('{kw}') RETURN l",
        "title_kw"
    ))
    out.append(q(
        f"{g} papers after {y2}.",
        f"MATCH (:Gene {{entity:'{g}'}})-[:HAS_SOURCE]->(l:Literature) WHERE l.Year > {y2} RETURN DISTINCT l",
        "after_year"
    ))
    out.append(q(
        f"{g} papers in {journal}.",
        f"MATCH (:Gene {{entity:'{g}'}})-[:HAS_SOURCE]->(l:Literature) WHERE l.Journal = '{journal}' RETURN DISTINCT l",
        "in_journal"
    ))

    # Aggregations (keep useful scalars)
    out.append(q(
        "Count MeSH entitys per gene.",
        "MATCH (g:Gene)-[:CO_OCCURS]-(m:MeSH) RETURN g.entity AS gene, count(DISTINCT m) AS mesh_count",
        "count_mesh_per_gene"
    ))
    out.append(q(
        "Count genes per MeSH entity.",
        "MATCH (m:MeSH)<-[:CO_OCCURS]-(g:Gene) RETURN m.entity AS mesh, count(DISTINCT g) AS gene_count",
        "count_gene_per_mesh"
    ))

    # 原来的 “Top 10 MeSH for {g} by Strength.” 模板已删除

    smin = round(random.uniform(0.2, 0.9), 2)
    out.append(q(
        f"MeSH linked to {g} with Strength ≥ {smin}.",
        f"MATCH (:Gene {{entity:'{g}'}})-[r:CO_OCCURS]-(m:MeSH) WHERE r.Strength >= {smin} RETURN m, r.Strength AS strength",
        "threshold_strength"
    ))

    out.append(q(
        f"How many papers mention {g}?",
        f"MATCH (:Gene {{entity:'{g}'}})-[:HAS_SOURCE]->(l:Literature) RETURN count(DISTINCT l) AS paper_count",
        "paper_count_gene"
    ))

    out.append(q(
        f"Genes co-occurring with {m} after {y}.",
        f"MATCH (:MeSH {{entity:'{m}'}})<-[:CO_OCCURS]-(g:Gene)-[:HAS_SOURCE]->(l:Literature) WHERE l.Year > {y} RETURN DISTINCT g",
        "mesh_year_filter"
    ))

    out.append(q(
        f"Genes that share at least one MeSH with {g}.",
        f"MATCH (g1:Gene {{entity:'{g}'}})-[:CO_OCCURS]-(m:MeSH)-[:CO_OCCURS]-(g2:Gene) WHERE g1 <> g2 RETURN DISTINCT g2",
        "gene_shared_mesh"
    ))

    out.append(q(
        f"Common MeSH of {g} and TP53.",
        f"MATCH (:Gene {{entity:'{g}'}})-[:CO_OCCURS]-(m:MeSH)-[:CO_OCCURS]-(:Gene {{entity:'TP53'}}) RETURN DISTINCT m",
        "common_mesh_two_genes"
    ))

    out.append(q(
        f"Papers mentioning both {g} and {m}.",
        f"MATCH (:Gene {{entity:'{g}'}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(:MeSH {{entity:'{m}'}}) RETURN DISTINCT l",
        "papers_gene_mesh"
    ))

    out.append(q(
        f"Find a shortest path between {g} and {m}.",
        f"MATCH p = shortestPath( (:Gene {{entity:'{g}'}})-[*..4]-(:MeSH {{entity:'{m}'}}) ) RETURN p",
        "shortest_path_gene_mesh"
    ))

    out.append(q(
        f"Does {g} co-occur with {m}?",
        f"MATCH (:Gene {{entity:'{g}'}})-[:CO_OCCURS]-(:MeSH {{entity:'{m}'}}) RETURN count(*) > 0 AS exists",
        "exists_gene_mesh"
    ))

    out.append(q(
        f"List MeSH for {g} (page 2, size 20).",
        f"MATCH (:Gene {{entity:'{g}'}})-[:CO_OCCURS]-(m:MeSH) RETURN DISTINCT m",
        "pagination_mesh_gene"
    ))

    out.append(q(
        f"Show closeness centrality of {g}.",
        f"MATCH (g:Gene {{entity:'{g}'}}) RETURN g.{c} AS closeness",
        "closeness_gene"
    ))

    out.append(q(
        f"{g}–{m} raw occurrence and Wscore.",
        f"MATCH (:Gene {{entity:'{g}'}})-[r:CO_OCCURS]-(:MeSH {{entity:'{m}'}}) RETURN r.RawOccurrence AS raw, r.Wscore AS wscore",
        "edge_props"
    ))

    out.append(q(
        "Papers whose title starts with 'Gene'.",
        "MATCH (l:Literature) WHERE l.Title =~ '(?i)^gene.*' RETURN l",
        "regex_title"
    ))

    out.append(q(
        "Papers that have a PMID.",
        "MATCH (l:Literature) WHERE l.PMID IS NOT NULL RETURN l",
        "exists_pmid"
    ))

    out.append(q(
        f"Average Strength of {g} MeSH links.",
        f"MATCH (:Gene {{entity:'{g}'}})-[r:CO_OCCURS]-(:MeSH) RETURN avg(r.Strength) AS avg_strength",
        "avg_strength_gene"
    ))

    out.append(q(
        f"{g} papers in Nature or Science after {y2}.",
        f"MATCH (:Gene {{entity:'{g}'}})-[:HAS_SOURCE]->(l:Literature) WHERE l.Year > {y2} AND (l.Journal = 'Nature' OR l.Journal = 'Science') RETURN DISTINCT l",
        "or_filter_year"
    ))

    out.append(q(
        f"Distinct journals for {m}-related genes.",
        f"MATCH (:MeSH {{entity:'{m}'}})<-[:CO_OCCURS]-(g:Gene)-[:HAS_SOURCE]->(l:Literature) RETURN DISTINCT l.Journal AS journal",
        "distinct_journals_mesh"
    ))

    return out

# -------------------- 12 extra + thematic-year filter --------------------
def make_extra(g: str, m: str, genes: List[str], meshs: List[str]) -> List[Dict]:
    out: List[Dict] = []

    genes3 = random.sample(genes, 3) if len(genes) >= 3 else genes[:]
    meshs3 = random.sample(meshs, 3) if len(meshs) >= 3 else meshs[:]
    word = choose(["immun", "metab", "oncolog"])

    out.append(q(
        f"Find all papers mentioning the following genes: {', '.join(genes3)}.",
        f"MATCH (g:Gene)-[:HAS_SOURCE]->(l:Literature) WHERE g.entity IN {json.dumps(genes3)} RETURN DISTINCT l",
        "papers_by_genes_in"
    ))

    out.append(q(
        f"Find all papers mentioning the following MeSH entitys: {', '.join(meshs3)}.",
        f"MATCH (m:MeSH)-[:HAS_SOURCE]->(l:Literature) WHERE m.entity IN {json.dumps(meshs3)} RETURN DISTINCT l",
        "papers_by_meshes_in"
    ))

    out.append(q(
        "Find all genes and MeSH reported in the paper with a given PMID.",
        "MATCH (l:Literature {PMID: $pmid})<-[:HAS_SOURCE]-(e) WHERE e:Gene OR e:MeSH RETURN DISTINCT e",
        "entities_by_pmid"
    ))

    out.append(q(
        "Find all genes reported in the paper with a given PMID.",
        "MATCH (l:Literature {PMID: $pmid})<-[:HAS_SOURCE]-(g:Gene) RETURN DISTINCT g",
        "genes_by_pmid"
    ))

    out.append(q(
        "Find all MeSH entitys reported in the paper with a given PMID.",
        "MATCH (l:Literature {PMID: $pmid})<-[:HAS_SOURCE]-(m:MeSH) RETURN DISTINCT m",
        "mesh_by_pmid"
    ))

    out.append(q(
        f"Find all papers that report MeSH containing the word '{word}'.",
        "MATCH (m:MeSH)-[:HAS_SOURCE]->(l:Literature) WHERE toLower(m.entity) CONTAINS toLower($word) RETURN DISTINCT l",
        "papers_by_mesh_like"
    ))

    out.append(q(
        f"Find all papers that report MeSH containing the word '{word}', and return the MeSH entitys too.",
        "MATCH (m:MeSH)-[:HAS_SOURCE]->(l:Literature) WHERE toLower(m.entity) CONTAINS toLower($word) RETURN DISTINCT l, m",
        "papers_and_mesh_by_mesh_like"
    ))

    out.append(q(
        f"Find all papers that report {g}.",
        f"MATCH (:Gene {{entity:'{g}'}})-[:HAS_SOURCE]->(l:Literature) RETURN DISTINCT l",
        "papers_by_gene"
    ))

    out.append(q(
        f"Find all papers that report {g} and return those papers, {g}, and other co-occurring genes.",
        f"MATCH (g:Gene {{entity:'{g}'}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(g2:Gene) RETURN DISTINCT l, g, g2",
        "papers_gene_and_cogenes"
    ))

    out.append(q(
        f"Find all papers that report {g} and return those papers, {g}, and other co-occurring MeSH entitys.",
        f"MATCH (g:Gene {{entity:'{g}'}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(m:MeSH) RETURN DISTINCT l, g, m",
        "papers_gene_and_mesh"
    ))

    out.append(q(
        f"Find all papers that report {g} and return those papers, {g}, and other co-occurring entities.",
        f"MATCH (g:Gene {{entity:'{g}'}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(e) WHERE e:Gene OR e:MeSH RETURN DISTINCT l, g, e",
        "papers_gene_and_coentities"
    ))

    # thematic-year filter (e.g., immunology + Year > 2020)
    keyword = choose(THEMATIC_KEYWORDS)
    cy = (
        "MATCH (p:Literature)<-[:HAS_SOURCE]-(g:Gene)-[:CO_OCCURS]-(m:MeSH) "
        "WHERE p.Year > 2020 AND toLower(m.entity) CONTAINS toLower($keyword) "
        "RETURN DISTINCT p, g, m"
    )
    nl = f"Find all publications after 2020 where genes co-occur with MeSH entitys containing '{keyword}'."
    out.append(q(nl, cy, "thematic_year_filter"))

    return out

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="/mnt/data/gene_mesh_dataset.csv", help="CSV with gene+mesh columns")
    ap.add_argument("--genes", type=str, default=None, help="Optional plain list of genes (one per line)")
    ap.add_argument("--meshs", type=str, default=None, help="Optional plain list of MeSH entitys (one per line)")
    ap.add_argument("--gene-col", type=str, default=None, help="Column name for gene in --data")
    ap.add_argument("--mesh-col", type=str, default=None, help="Column name for MeSH in --data")
    ap.add_argument("--n", type=int, default=2000, help="Number of QA pairs to generate")
    ap.add_argument("--out", type=str, default="/mnt/data/train.chat.jsonl", help="Output JSONL path")
    ap.add_argument("--max-pool", type=int, default=800, help="Cap on unique names sampled from each list")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    # keep augment flag for compatibility; it's a no-op for direction safety
    ap.add_argument("--augment-reverse", action="store_true", help="(No-op) Kept for compatibility")
    args = ap.parse_args()

    random.seed(args.seed)

    genes, meshs = load_lists(
        data_path=args.data,
        gene_col=args.gene_col,
        mesh_col=args.mesh_col,
        genes_path=args.genes,
        meshs_path=args.meshs,
        max_items=args.max_pool
    )

    per_draw = 28 + 12 + 1  # 28 basic + 12 extra + 1 thematic-year = 41
    draws_needed = max(1, (args.n + per_draw - 1) // per_draw)

    items: List[Dict] = []
    for _ in range(draws_needed):
        g = choose(genes)
        m = choose(meshs)
        batch = make_basic_28(g, m) + make_extra(g, m, genes, meshs)

        if args.augment_reverse:
            # Safe no-op: we explicitly avoid flipping directions to keep HAS_SOURCE correct
            batch = [{**b} for b in batch]  # shallow copy

        items.extend(batch)

    items = items[:args.n]

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items)} examples to {out_path}")

if __name__ == "__main__":
    main()
