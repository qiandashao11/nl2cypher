
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic NL→Cypher generator (English-only) for your Gene/MeSH/Literature schema.

- Works with a single CSV that has both a gene column and a MeSH column
  (default path: /mnt/data/gene_mesh_dataset.csv), OR with two separate lists.
- Auto-detects column names (you can also override with --gene-col/--mesh-col).
- Produces JSONL with chat-style messages: [{"role": "user"}, {"role": "assistant"}].
- Covers 30 core question types (the "basic set") using your schema:
    Node labels:   Gene, MeSH, Literature
    Relations:     CO_OCCURS, HAS_SOURCE
    Common props:  Title, Year, Journal, PMID, Strength, RawOccurrence, Wscore,
                   `Closeness.centrality` (quoted when needed)

Example:
  python gen_nl2cypher_basic.py \
      --data /mnt/data/gene_mesh_dataset.csv \
      --n 1500 \
      --out /mnt/data/train.chat.jsonl

If auto-detection fails, add:
  --gene-col symbol --mesh-col term

Author: Qian - work 1
"""
import argparse, csv, json, pathlib, random, re, sys
from typing import List, Dict, Tuple, Optional

# ----------------------- Helpers -----------------------

JOURNALS = [
    "Nature", "Science", "Cell", "PNAS", "Lancet", "NEJM", "BMJ", "JAMA"
]

TITLE_KEYWORDS = [
    "breast", "cancer", "genome", "mutation", "pathway", "protein",
    "tumor", "immune", "signal", "marker"
]

def load_lists(
    data_path: Optional[str],
    gene_col: Optional[str],
    mesh_col: Optional[str],
    genes_path: Optional[str],
    meshs_path: Optional[str],
    max_items: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    genes, meshs = [], []

    if data_path:
        with open(data_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            cols = [c.lower() for c in reader.fieldnames or []]
            # auto-detect columns when not provided
            def detect(colname: Optional[str], keywords: List[str]) -> Optional[str]:
                if colname:
                    return colname
                for c in reader.fieldnames or []:
                    cl = c.lower()
                    if any(k in cl for k in keywords):
                        return c
                return None

            gene_col = detect(gene_col, ["gene", "symbol", "gene_symbol"])
            mesh_col = detect(mesh_col, ["mesh", "term", "mesh_term"])

            if not gene_col or not mesh_col:
                raise SystemExit(
                    f"Could not auto-detect columns. "
                    f"Detected columns: {reader.fieldnames}. "
                    f"Pass --gene-col and --mesh-col explicitly."
                )

            for row in reader:
                g = (row.get(gene_col) or "").strip()
                m = (row.get(mesh_col) or "").strip()
                if g: genes.append(g)
                if m: meshs.append(m)

    if genes_path:
        with open(genes_path, encoding="utf-8") as f:
            for line in f:
                g = line.strip().split(",")[0]
                if g: genes.append(g)

    if meshs_path:
        with open(meshs_path, encoding="utf-8") as f:
            for line in f:
                m = line.strip().split(",")[0]
                if m: meshs.append(m)

    # de-duplicate while preserving order
    def dedup(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    genes = dedup(genes)
    meshs = dedup(meshs)

    if max_items:
        random.shuffle(genes); random.shuffle(meshs)
        genes = genes[:max_items]
        meshs = meshs[:max_items]

    if not genes or not meshs:
        raise SystemExit("Empty gene or MeSH list after loading. Check inputs.")

    return genes, meshs

# ------------------- Template Builders -------------------

def cypher_prop(prop: str) -> str:
    # Quote props with dots for Neo4j
    return f"`{prop}`" if "." in prop else prop

def choose(lst: List[str]) -> str:
    return random.choice(lst)

def q(nl: str, cypher: str, qtype: str) -> Dict:
    return {
        "messages": [
            {"role": "user", "content": nl},
            {"role": "assistant", "content": cypher.strip()},
        ],
        "metadata": {"type": qtype}
    }

def make_templates(gene: str, mesh: str) -> List[Dict]:
    """Return 30 basic QA pairs for one (gene, mesh) draw, with small NL variation."""
    year = random.randint(2000, 2024)
    year2 = random.randint(2018, 2024)
    kw = choose(TITLE_KEYWORDS)
    journal = choose(JOURNALS)
    closeness = cypher_prop("Closeness.centrality")

    out = []

    # 1 List all genes
    out.append(q(
        "List all genes.",
        "MATCH (g:Gene) RETURN g.symbol LIMIT 100",
        "list_genes"
    ))

    # 2 List all MeSH
    out.append(q(
        "List all MeSH terms.",
        "MATCH (m:MeSH) RETURN m.term LIMIT 100",
        "list_mesh"
    ))

    # 3 List literature
    out.append(q(
        "List literature titles and years.",
        "MATCH (l:Literature) RETURN l.Title, l.Year LIMIT 100",
        "list_literature"
    ))

    # 4 Gene -> MeSH
    out.append(q(
        f"Find MeSH terms co-occurring with {gene}.",
        f"MATCH (g:Gene {{symbol:'{gene}'}})-[:CO_OCCURS]->(m:MeSH) "
        "RETURN DISTINCT m.term",
        "gene_to_mesh"
    ))

    # 5 MeSH -> Gene
    out.append(q(
        f"Which genes co-occur with the MeSH term {mesh}?",
        f"MATCH (m:MeSH {{term:'{mesh}'}})<-[:CO_OCCURS]-(g:Gene) "
        "RETURN DISTINCT g.symbol",
        "mesh_to_gene"
    ))

    # 6 Gene -> Literature
    out.append(q(
        f"Show literature linked to {gene}.",
        f"MATCH (g:Gene {{symbol:'{gene}'}})-[:HAS_SOURCE]->(l:Literature) "
        "RETURN l.Title, l.Year, l.Journal",
        "gene_to_lit"
    ))

    # 7 MeSH -> Literature
    out.append(q(
        f"Show literature for MeSH {mesh}.",
        f"MATCH (m:MeSH {{term:'{mesh}'}})-[:HAS_SOURCE]->(l:Literature) "
        "RETURN l.Title, l.Year",
        "mesh_to_lit"
    ))

    # 8 Title contains keyword
    out.append(q(
        f"Papers whose title contains '{kw}'.",
        "MATCH (l:Literature) "
        f"WHERE toLower(l.Title) CONTAINS toLower('{kw}') "
        "RETURN l.Title, l.Year",
        "title_contains"
    ))

    # 9 After year
    out.append(q(
        f"{gene} papers after {year2}.",
        f"MATCH (g:Gene {{symbol:'{gene}'}})-[:HAS_SOURCE]->(l:Literature) "
        f"WHERE l.Year > {year2} RETURN l.Title, l.Year",
        "after_year"
    ))

    # 10 In journal
    out.append(q(
        f"{gene} papers in {journal}.",
        f"MATCH (g:Gene {{symbol:'{gene}'}})-[:HAS_SOURCE]->(l:Literature) "
        f"WHERE l.Journal = '{journal}' RETURN l.Title, l.Year",
        "in_journal"
    ))

    # 11 Count MeSH per gene
    out.append(q(
        "Count MeSH terms per gene.",
        "MATCH (g:Gene)-[:CO_OCCURS]->(m:MeSH) "
        "RETURN g.symbol, count(DISTINCT m) AS mesh_count "
        "ORDER BY mesh_count DESC",
        "count_mesh_per_gene"
    ))

    # 12 Count genes per MeSH
    out.append(q(
        "Count genes per MeSH term.",
        "MATCH (m:MeSH)<-[:CO_OCCURS]-(g:Gene) "
        "RETURN m.term, count(DISTINCT g) AS gene_count "
        "ORDER BY gene_count DESC",
        "count_gene_per_mesh"
    ))

    # 13 Top-N by Strength
    out.append(q(
        f"Top 10 MeSH for {gene} by Strength.",
        f"MATCH (g:Gene {{symbol:'{gene}'}})-[r:CO_OCCURS]->(m:MeSH) "
        "RETURN m.term, r.Strength ORDER BY r.Strength DESC LIMIT 10",
        "top_by_strength"
    ))

    # 14 Threshold Strength
    minS = round(random.uniform(0.2, 0.9), 2)
    out.append(q(
        f"MeSH linked to {gene} with Strength ≥ {minS}.",
        f"MATCH (g:Gene {{symbol:'{gene}'}})-[r:CO_OCCURS]->(m:MeSH) "
        f"WHERE r.Strength >= {minS} RETURN m.term, r.Strength",
        "threshold_strength"
    ))

    # 15 Count papers for gene
    out.append(q(
        f"How many papers mention {gene}?",
        f"MATCH (g:Gene {{symbol:'{gene}'}})-[:HAS_SOURCE]->(l:Literature) "
        "RETURN count(DISTINCT l) AS paper_count",
        "paper_count_gene"
    ))

    # 16 Mesh + Year filter
    out.append(q(
        f"Genes co-occurring with {mesh} after {year}.",
        f"MATCH (m:MeSH {{term:'{mesh}'}})<-[:CO_OCCURS]-(g:Gene)-[:HAS_SOURCE]->(l:Literature) "
        f"WHERE l.Year > {year} RETURN DISTINCT g.symbol",
        "mesh_year_filter"
    ))

    # 17 Genes sharing MeSH with gene
    out.append(q(
        f"Genes that share at least one MeSH with {gene}.",
        f"MATCH (g1:Gene {{symbol:'{gene}'}})-[:CO_OCCURS]->(m:MeSH)<-[:CO_OCCURS]-(g2:Gene) "
        "WHERE g1 <> g2 RETURN DISTINCT g2.symbol",
        "gene_shared_mesh"
    ))

    # 18 Common MeSH of two genes
    gB = gene if random.random() < 0.2 else mesh  # just to vary; corrected below
    # Ensure a different gene name for display; fallback to real string
    out.append(q(
        f"Common MeSH of {gene} and TP53.",
        f"MATCH (g1:Gene {{symbol:'{gene}'}})-[:CO_OCCURS]->(m:MeSH)<-[:CO_OCCURS]-(g2:Gene {{symbol:'TP53'}}) "
        "RETURN DISTINCT m.term",
        "common_mesh_two_genes"
    ))

    # 19 Papers mentioning both gene and mesh
    out.append(q(
        f"Papers mentioning both {gene} and {mesh}.",
        f"MATCH (g:Gene {{symbol:'{gene}'}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(m:MeSH {{term:'{mesh}'}}) "
        "RETURN DISTINCT l.Title, l.Year",
        "papers_gene_mesh"
    ))

    # 20 Shortest path gene <-> mesh
    out.append(q(
        f"Find a shortest path between {gene} and {mesh}.",
        f"MATCH p = shortestPath( (g:Gene {{symbol:'{gene}'}})-[*..4]-(m:MeSH {{term:'{mesh}'}}) ) RETURN p",
        "shortest_path_gene_mesh"
    ))

    # 21 Existence boolean
    out.append(q(
        f"Does {gene} co-occur with {mesh}?",
        f"MATCH (g:Gene {{symbol:'{gene}'}})-[:CO_OCCURS]->(m:MeSH {{term:'{mesh}'}}) "
        "RETURN count(*) > 0 AS exists",
        "exists_gene_mesh"
    ))

    # 22 Pagination
    out.append(q(
        f"List MeSH for {gene} (page 2, size 20).",
        f"MATCH (g:Gene {{symbol:'{gene}'}})-[:CO_OCCURS]->(m:MeSH) "
        "RETURN DISTINCT m.term SKIP 20 LIMIT 20",
        "pagination_mesh_gene"
    ))

    # 23 Graph metric
    out.append(q(
        f"Show closeness centrality of {gene}.",
        f"MATCH (g:Gene {{symbol:'{gene}'}}) RETURN g.{closeness}",
        "closeness_gene"
    ))

    # 24 Relationship properties
    out.append(q(
        f"{gene}–{mesh} raw occurrence and Wscore.",
        f"MATCH (g:Gene {{symbol:'{gene}'}})-[r:CO_OCCURS]->(m:MeSH {{term:'{mesh}'}}) "
        "RETURN r.RawOccurrence, r.Wscore",
        "edge_props"
    ))

    # 25 Multi-sort
    out.append(q(
        f"Order {gene}-linked MeSH by Strength then paper count.",
        f"MATCH (g:Gene {{symbol:'{gene}'}})-[r:CO_OCCURS]->(m:MeSH) "
        "OPTIONAL MATCH (m)<-[:HAS_SOURCE]-(l:Literature) "
        "WITH m, r, count(DISTINCT l) AS nPapers "
        "RETURN m.term, r.Strength, nPapers "
        "ORDER BY r.Strength DESC, nPapers DESC",
        "multi_sort"
    ))

    # 26 Regex title
    out.append(q(
        "Papers whose title starts with 'Gene'.",
        "MATCH (l:Literature) WHERE l.Title =~ '(?i)^gene.*' RETURN l.Title",
        "regex_title"
    ))

    # 27 Has PMID
    out.append(q(
        "Papers that have a PMID.",
        "MATCH (l:Literature) WHERE exists(l.PMID) RETURN l.Title, l.PMID",
        "exists_pmid"
    ))

    # 28 Avg Strength for gene
    out.append(q(
        f"Average Strength of {gene} MeSH links.",
        f"MATCH (:Gene {{symbol:'{gene}'}})-[r:CO_OCCURS]->(:MeSH) "
        "RETURN avg(r.Strength) AS avg_strength",
        "avg_strength_gene"
    ))

    # 29 OR filter + year
    out.append(q(
        f"{gene} papers in Nature or Science after {year2}.",
        f"MATCH (:Gene {{symbol:'{gene}'}})-[:HAS_SOURCE]->(l:Literature) "
        f"WHERE l.Year > {year2} AND (l.Journal = 'Nature' OR l.Journal = 'Science') "
        "RETURN l.Title, l.Journal, l.Year",
        "or_filter_year"
    ))

    # 30 Distinct journals for mesh-related genes
    out.append(q(
        f"Distinct journals for {mesh}-related genes.",
        f"MATCH (:MeSH {{term:'{mesh}'}})<-[:CO_OCCURS]-(g:Gene)-[:HAS_SOURCE]->(l:Literature) "
        "RETURN DISTINCT l.Journal",
        "distinct_journals_mesh"
    ))

    return out

# ----------------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="/mnt/data/gene_mesh_dataset.csv",
                    help="CSV containing both gene and mesh columns")
    ap.add_argument("--genes", type=str, default=None, help="Optional plain list of genes (one per line)")
    ap.add_argument("--meshs", type=str, default=None, help="Optional plain list of MeSH terms (one per line)")
    ap.add_argument("--gene-col", type=str, default=None, help="Column name for gene in --data")
    ap.add_argument("--mesh-col", type=str, default=None, help="Column name for MeSH in --data")
    ap.add_argument("--n", type=int, default=1500, help="Number of QA pairs to generate")
    ap.add_argument("--out", type=str, default="/mnt/data/train.chat.jsonl", help="Output JSONL path")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--max-pool", type=int, default=500, help="Optional cap on unique names sampled from each list")
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

    # Make sure we have enough combos to reach n using 30-per-draw batching
    # We'll loop draws of (gene, mesh) and add 30 templates each time.
    per_draw = 30
    draws_needed = max(1, (args.n + per_draw - 1) // per_draw)

    items: List[Dict] = []
    for _ in range(draws_needed):
        g = choose(genes)
        m = choose(meshs)
        items.extend(make_templates(g, m))

    # Trim to exactly N
    items = items[:args.n]

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items)} examples to {out_path}")

if __name__ == "__main__":
    main()