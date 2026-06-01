#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 NL→Cypher dataset generator for the Gene/MeSH/Literature graph.

Schema (Phase 1 uses primary keys only):
  Gene(ENTITY)
  MeSH(ENTITY)
  Literature(PMID, Title, Year, Journal)

Relations:
  HAS_SOURCE: (Gene|MeSH) -> (Literature)     # directed to Literature ONLY
  CO_OCCURS:  Gene — Gene  or  Gene — MeSH    # treated as undirected in queries

Generates chat-style JSONL training data including:
  - 20 base templates  (Phase 1)
  - 12 extra templates (PMID, multi-IN, co-entity)
  - 1 thematic-year filter template (e.g., immunology + Year>2020)
  - 30 multi-hop/compositional templates

All Cypher:
  - Use exact property-name casing: ENTITY, PMID, Title, Year, Journal
  - RETURN only primary-key fields or simple scalars (count, journal, etc.)
  - Do not use $parameter; write all placeholder values directly into natural language and Cypher
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

def choose(lst: List[str]) -> str:
    return random.choice(lst)

def dedup(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out

def cypher_str(value: str) -> str:
    """Return a single-quoted Cypher string literal."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"

def cypher_list(values: List[str]) -> str:
    return "[" + ", ".join(cypher_str(v) for v in values) + "]"

def choose_other(lst: List[str], current: str) -> str:
    candidates = [x for x in lst if x != current]
    return choose(candidates or lst)

def q(nl: str, cy: str, qtype: str) -> Dict:
    """Package NL + Cypher as one training sample; keep Cypher single-line with no extra whitespace."""
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

# -------------------- augmentation (no-op) --------------------
def maybe_reverse(cypher: str) -> str:
    """
    Safe augmentation placeholder:
    We DO NOT flip HAS_SOURCE direction.
    For CO_OCCURS we already use undirected '-[:CO_OCCURS]-',
    so there is nothing to reverse. Return cypher unchanged.
    """
    return cypher

# -------------------- 20 base templates (Phase 1) --------------------
def make_basic_20(g: str, m: str) -> List[Dict]:
    """
    Phase 1:
    - Use only primary-key properties: Gene.ENTITY, MeSH.ENTITY, Literature.PMID (+Title/Year/Journal in WHERE)
    - RETURN only primary keys or simple scalars
    """
    y = random.randint(2000, 2024)
    y2 = random.randint(2018, 2024)
    kw = choose(TITLE_KEYWORDS)
    journal = choose(JOURNALS)

    out: List[Dict] = []

    # ---- 1) List nodes (primary keys) ----
    out.append(q(
        "List all genes.",
        "MATCH (g:Gene) RETURN g.ENTITY AS gene_entity",
        "list_genes"
    ))
    out.append(q(
        "List all MeSH entitys.",
        "MATCH (m:MeSH) RETURN m.ENTITY AS mesh_entity",
        "list_mesh"
    ))
    out.append(q(
        "List literature nodes.",
        "MATCH (l:Literature) RETURN l.PMID AS pmid",
        "list_lit"
    ))

    # ---- 2) Gene <-> MeSH / Literature mapping ----
    out.append(q(
        f"Find MeSH entitys co-occurring with {g}.",
        f"MATCH (:Gene {{ENTITY:'{g}'}})-[:CO_OCCURS]-(m:MeSH) "
        f"RETURN DISTINCT m.ENTITY AS mesh_entity",
        "gene_to_mesh"
    ))
    out.append(q(
        f"Which genes co-occur with the MeSH entity {m}?",
        f"MATCH (g:Gene)-[:CO_OCCURS]-(:MeSH {{ENTITY:'{m}'}}) "
        f"RETURN DISTINCT g.ENTITY AS gene_entity",
        "mesh_to_gene"
    ))
    out.append(q(
        f"Show literature linked to {g}.",
        f"MATCH (:Gene {{ENTITY:'{g}'}})-[:HAS_SOURCE]->(l:Literature) "
        f"RETURN DISTINCT l.PMID AS pmid",
        "gene_to_lit"
    ))
    out.append(q(
        f"Show literature for MeSH {m}.",
        f"MATCH (:MeSH {{ENTITY:'{m}'}})-[:HAS_SOURCE]->(l:Literature) "
        f"RETURN DISTINCT l.PMID AS pmid",
        "mesh_to_lit"
    ))

    # ---- 3) Simple literature filters (by Title / Year / Journal) ----
    out.append(q(
        f"Papers whose title contains '{kw}'.",
        f"MATCH (l:Literature) "
        f"WHERE toLower(l.Title) CONTAINS toLower('{kw}') "
        f"RETURN l.PMID AS pmid",
        "title_kw"
    ))
    out.append(q(
        f"{g} papers after {y2}.",
        f"MATCH (:Gene {{ENTITY:'{g}'}})-[:HAS_SOURCE]->(l:Literature) "
        f"WHERE l.Year > {y2} "
        f"RETURN DISTINCT l.PMID AS pmid",
        "after_year"
    ))
    out.append(q(
        f"{g} papers in {journal}.",
        f"MATCH (:Gene {{ENTITY:'{g}'}})-[:HAS_SOURCE]->(l:Literature) "
        f"WHERE l.Journal = '{journal}' "
        f"RETURN DISTINCT l.PMID AS pmid",
        "in_journal"
    ))

    # ---- 4) Basic count statistics ----
    out.append(q(
        "Count MeSH entitys per gene.",
        "MATCH (g:Gene)-[:CO_OCCURS]-(m:MeSH) "
        "RETURN g.ENTITY AS gene_entity, count(DISTINCT m) AS mesh_count",
        "count_mesh_per_gene"
    ))
    out.append(q(
        "Count genes per MeSH entity.",
        "MATCH (m:MeSH)<-[:CO_OCCURS]-(g:Gene) "
        "RETURN m.ENTITY AS mesh_entity, count(DISTINCT g) AS gene_count",
        "count_gene_per_mesh"
    ))
    out.append(q(
        f"How many papers mention {g}?",
        f"MATCH (:Gene {{ENTITY:'{g}'}})-[:HAS_SOURCE]->(l:Literature) "
        f"RETURN count(DISTINCT l) AS paper_count",
        "paper_count_gene"
    ))

    # ---- 5) Cross-relationship combinations and co-occurrence structures ----
    out.append(q(
        f"Genes co-occurring with {m} after {y}.",
        f"MATCH (:MeSH {{ENTITY:'{m}'}})<-[:CO_OCCURS]-(g:Gene)-[:HAS_SOURCE]->(l:Literature) "
        f"WHERE l.Year > {y} "
        f"RETURN DISTINCT g.ENTITY AS gene_entity",
        "mesh_year_filter"
    ))
    out.append(q(
        f"Genes that share at least one MeSH with {g}.",
        f"MATCH (g1:Gene {{ENTITY:'{g}'}})-[:CO_OCCURS]-(m:MeSH)-[:CO_OCCURS]-(g2:Gene) "
        f"WHERE g1 <> g2 "
        f"RETURN DISTINCT g2.ENTITY AS gene_entity",
        "gene_shared_mesh"
    ))
    out.append(q(
        f"Common MeSH of {g} and TP53.",
        f"MATCH (:Gene {{ENTITY:'{g}'}})-[:CO_OCCURS]-(m:MeSH)-[:CO_OCCURS]-(:Gene {{ENTITY:'TP53'}}) "
        f"RETURN DISTINCT m.ENTITY AS mesh_entity",
        "common_mesh_two_genes"
    ))
    out.append(q(
        f"Papers mentioning both {g} and {m}.",
        f"MATCH (:Gene {{ENTITY:'{g}'}})-[:HAS_SOURCE]->(l:Literature)"
        f"<-[:HAS_SOURCE]-(:MeSH {{ENTITY:'{m}'}}) "
        f"RETURN DISTINCT l.PMID AS pmid",
        "papers_gene_mesh"
    ))
    out.append(q(
        f"Does {g} co-occur with {m}?",
        f"MATCH (:Gene {{ENTITY:'{g}'}})-[:CO_OCCURS]-(:MeSH {{ENTITY:'{m}'}}) "
        f"RETURN count(*) > 0 AS exists",
        "exists_gene_mesh"
    ))

    # ---- 6) Existence checks and DISTINCT properties ----
    out.append(q(
        "Papers that have a PMID.",
        "MATCH (l:Literature) WHERE l.PMID IS NOT NULL RETURN l.PMID AS pmid",
        "exists_pmid"
    ))
    out.append(q(
        f"Distinct journals for {m}-related genes.",
        f"MATCH (:MeSH {{ENTITY:'{m}'}})<-[:CO_OCCURS]-(g:Gene)-[:HAS_SOURCE]->(l:Literature) "
        f"RETURN DISTINCT l.Journal AS journal",
        "distinct_journals_mesh"
    ))

    return out

# -------------------- 12 extra + thematic-year filter --------------------
def make_extra(g: str, m: str, genes: List[str], meshs: List[str]) -> List[Dict]:
    out: List[Dict] = []

    genes3 = random.sample(genes, 3) if len(genes) >= 3 else genes[:]
    meshs3 = random.sample(meshs, 3) if len(meshs) >= 3 else meshs[:]
    word = choose(["immun", "metab", "oncolog"])

    # Randomly create an example PMID for NL and Cypher (no $pmid)
    pmid_example = str(random.randint(10_000_000, 99_999_999))

    # genes IN list -> papers
    out.append(q(
        f"Find all papers mentioning the following genes: {', '.join(genes3)}.",
        f"MATCH (g:Gene)-[:HAS_SOURCE]->(l:Literature) "
        f"WHERE g.ENTITY IN {json.dumps(genes3)} "
        f"RETURN DISTINCT l.PMID AS pmid",
        "papers_by_genes_in"
    ))

    # mesh IN list -> papers
    out.append(q(
        f"Find all papers mentioning the following MeSH entitys: {', '.join(meshs3)}.",
        f"MATCH (m:MeSH)-[:HAS_SOURCE]->(l:Literature) "
        f"WHERE m.ENTITY IN {json.dumps(meshs3)} "
        f"RETURN DISTINCT l.PMID AS pmid",
        "papers_by_meshes_in"
    ))

    # PMID -> entities (use a concrete PMID literal instead of $pmid)
    out.append(q(
        f"Find all genes and MeSH reported in the paper with PMID {pmid_example}.",
        f"MATCH (l:Literature {{PMID: '{pmid_example}'}})<-[:HAS_SOURCE]-(e) "
        f"WHERE e:Gene OR e:MeSH "
        f"RETURN DISTINCT e.ENTITY AS entity",
        "entities_by_pmid"
    ))
    out.append(q(
        f"Find all genes reported in the paper with PMID {pmid_example}.",
        f"MATCH (l:Literature {{PMID: '{pmid_example}'}})<-[:HAS_SOURCE]-(g:Gene) "
        f"RETURN DISTINCT g.ENTITY AS gene_entity",
        "genes_by_pmid"
    ))
    out.append(q(
        f"Find all MeSH entitys reported in the paper with PMID {pmid_example}.",
        f"MATCH (l:Literature {{PMID: '{pmid_example}'}})<-[:HAS_SOURCE]-(m:MeSH) "
        f"RETURN DISTINCT m.ENTITY AS mesh_entity",
        "mesh_by_pmid"
    ))

    # MeSH CONTAINS word -> papers / papers+MeSH (use a literal such as 'metab' instead of $word)
    out.append(q(
        f"Find all papers that report MeSH containing the word '{word}'.",
        f"MATCH (m:MeSH)-[:HAS_SOURCE]->(l:Literature) "
        f"WHERE toLower(m.ENTITY) CONTAINS toLower('{word}') "
        f"RETURN DISTINCT l.PMID AS pmid",
        "papers_by_mesh_like"
    ))
    out.append(q(
        f"Find all papers that report MeSH containing the word '{word}', and return the MeSH entitys too.",
        f"MATCH (m:MeSH)-[:HAS_SOURCE]->(l:Literature) "
        f"WHERE toLower(m.ENTITY) CONTAINS toLower('{word}') "
        f"RETURN DISTINCT l.PMID AS pmid, m.ENTITY AS mesh_entity",
        "papers_and_mesh_by_mesh_like"
    ))

    # Single gene -> papers / papers+co-entities
    out.append(q(
        f"Find all papers that report {g}.",
        f"MATCH (:Gene {{ENTITY:'{g}'}})-[:HAS_SOURCE]->(l:Literature) "
        f"RETURN DISTINCT l.PMID AS pmid",
        "papers_by_gene"
    ))
    out.append(q(
        f"Find all papers that report {g} and return those papers, {g}, and other co-occurring genes.",
        f"MATCH (g:Gene {{ENTITY:'{g}'}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(g2:Gene) "
        f"RETURN DISTINCT l.PMID AS pmid, g.ENTITY AS gene_entity, g2.ENTITY AS co_gene_entity",
        "papers_gene_and_cogenes"
    ))
    out.append(q(
        f"Find all papers that report {g} and return those papers, {g}, and other co-occurring MeSH entitys.",
        f"MATCH (g:Gene {{ENTITY:'{g}'}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(m:MeSH) "
        f"RETURN DISTINCT l.PMID AS pmid, g.ENTITY AS gene_entity, m.ENTITY AS mesh_entity",
        "papers_gene_and_mesh"
    ))
    out.append(q(
        f"Find all papers that report {g} and return those papers, {g}, and other co-occurring entities.",
        f"MATCH (g:Gene {{ENTITY:'{g}'}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(e) "
        f"WHERE e:Gene OR e:MeSH "
        f"RETURN DISTINCT l.PMID AS pmid, g.ENTITY AS gene_entity, e.ENTITY AS co_entity",
        "papers_gene_and_coentities"
    ))

    # thematic-year filter (use a literal keyword instead of $keyword)
    keyword = choose(THEMATIC_KEYWORDS)
    cy = (
        f"MATCH (p:Literature)<-[:HAS_SOURCE]-(g:Gene)-[:CO_OCCURS]-(m:MeSH) "
        f"WHERE p.Year > 2020 AND toLower(m.ENTITY) CONTAINS toLower('{keyword}') "
        f"RETURN DISTINCT p.PMID AS pmid, g.ENTITY AS gene_entity, m.ENTITY AS mesh_entity"
    )
    nl = f"Find all publications after 2020 where genes co-occur with MeSH entitys containing '{keyword}'."
    out.append(q(nl, cy, "thematic_year_filter"))

    return out


# -------------------- multi-hop / compositional templates --------------------
def make_multihop(g: str, m: str, genes: List[str], meshs: List[str]) -> List[Dict]:
    """
    Multi-hop extension for non-parameter NL2Cypher.

    Coverage goals:
    - Gene-Gene direct co-occurrence
    - Gene-MeSH-Gene and Gene-MeSH-Gene-Literature paths
    - Shared Literature paths: Gene-Literature-MeSH/Gene
    - Multi-condition filters: Year, Journal, Title keyword
    - Aggregations and boolean existence checks
    - IN-list composition without $parameters
    """
    out: List[Dict] = []

    g2 = choose_other(genes, g)
    g3 = choose_other(genes, g2)
    m2 = choose_other(meshs, m)
    m3 = choose_other(meshs, m2)
    genes3 = random.sample(genes, 3) if len(genes) >= 3 else genes[:]
    meshs3 = random.sample(meshs, 3) if len(meshs) >= 3 else meshs[:]
    y = random.randint(2000, 2024)
    y2 = random.randint(2015, 2024)
    journal = choose(JOURNALS)
    kw = choose(TITLE_KEYWORDS)
    word = choose(["immun", "metab", "oncolog", "neuro", "cell"])
    pmid_example = str(random.randint(10_000_000, 99_999_999))

    gq, g2q, g3q = cypher_str(g), cypher_str(g2), cypher_str(g3)
    mq, m2q, m3q = cypher_str(m), cypher_str(m2), cypher_str(m3)
    journal_q, kw_q, word_q = cypher_str(journal), cypher_str(kw), cypher_str(word)
    pmid_q = cypher_str(pmid_example)

    out.append(q(
        f"Does {g} directly co-occur with {g2}?",
        f"MATCH (:Gene {{ENTITY:{gq}}})-[:CO_OCCURS]-(:Gene {{ENTITY:{g2q}}}) "
        f"RETURN count(*) > 0 AS exists",
        "exists_gene_gene"
    ))
    out.append(q(
        f"Find papers that mention both {g} and {g2}.",
        f"MATCH (:Gene {{ENTITY:{gq}}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(:Gene {{ENTITY:{g2q}}}) "
        f"RETURN DISTINCT l.PMID AS pmid",
        "papers_two_genes"
    ))
    out.append(q(
        f"Find MeSH entitys shared by {g} and {g2}.",
        f"MATCH (:Gene {{ENTITY:{gq}}})-[:CO_OCCURS]-(m:MeSH)-[:CO_OCCURS]-(:Gene {{ENTITY:{g2q}}}) "
        f"RETURN DISTINCT m.ENTITY AS mesh_entity",
        "shared_mesh_two_genes"
    ))
    out.append(q(
        f"Count MeSH entitys shared by {g} and {g2}.",
        f"MATCH (:Gene {{ENTITY:{gq}}})-[:CO_OCCURS]-(m:MeSH)-[:CO_OCCURS]-(:Gene {{ENTITY:{g2q}}}) "
        f"RETURN count(DISTINCT m) AS shared_mesh_count",
        "count_shared_mesh_two_genes"
    ))
    out.append(q(
        f"Find genes that share MeSH entitys with {g}.",
        f"MATCH (g1:Gene {{ENTITY:{gq}}})-[:CO_OCCURS]-(m:MeSH)-[:CO_OCCURS]-(g2:Gene) "
        f"WHERE g1 <> g2 RETURN DISTINCT g2.ENTITY AS gene_entity",
        "genes_sharing_mesh_with_gene"
    ))
    out.append(q(
        f"Find papers of genes that share MeSH entitys with {g}.",
        f"MATCH (g1:Gene {{ENTITY:{gq}}})-[:CO_OCCURS]-(m:MeSH)-[:CO_OCCURS]-(g2:Gene)-[:HAS_SOURCE]->(l:Literature) "
        f"WHERE g1 <> g2 RETURN DISTINCT g2.ENTITY AS gene_entity, l.PMID AS pmid",
        "papers_of_genes_sharing_mesh"
    ))
    out.append(q(
        f"Find papers after {y2} of genes that share MeSH entitys with {g}.",
        f"MATCH (g1:Gene {{ENTITY:{gq}}})-[:CO_OCCURS]-(m:MeSH)-[:CO_OCCURS]-(g2:Gene)-[:HAS_SOURCE]->(l:Literature) "
        f"WHERE g1 <> g2 AND l.Year > {y2} RETURN DISTINCT g2.ENTITY AS gene_entity, l.PMID AS pmid",
        "papers_of_shared_mesh_genes_after_year"
    ))
    out.append(q(
        f"Find journals for papers of genes sharing MeSH entitys with {g}.",
        f"MATCH (g1:Gene {{ENTITY:{gq}}})-[:CO_OCCURS]-(m:MeSH)-[:CO_OCCURS]-(g2:Gene)-[:HAS_SOURCE]->(l:Literature) "
        f"WHERE g1 <> g2 RETURN DISTINCT l.Journal AS journal",
        "journals_of_shared_mesh_genes"
    ))
    out.append(q(
        f"Count genes that share MeSH entitys with {g}.",
        f"MATCH (g1:Gene {{ENTITY:{gq}}})-[:CO_OCCURS]-(m:MeSH)-[:CO_OCCURS]-(g2:Gene) "
        f"WHERE g1 <> g2 RETURN count(DISTINCT g2) AS gene_count",
        "count_genes_sharing_mesh"
    ))
    out.append(q(
        f"Find MeSH entitys connected to genes that share MeSH entitys with {g}.",
        f"MATCH (g1:Gene {{ENTITY:{gq}}})-[:CO_OCCURS]-(:MeSH)-[:CO_OCCURS]-(g2:Gene)-[:CO_OCCURS]-(m2:MeSH) "
        f"WHERE g1 <> g2 RETURN DISTINCT m2.ENTITY AS mesh_entity",
        "second_order_mesh_from_gene"
    ))

    out.append(q(
        f"Find papers about genes related to MeSH {m}.",
        f"MATCH (:MeSH {{ENTITY:{mq}}})-[:CO_OCCURS]-(g:Gene)-[:HAS_SOURCE]->(l:Literature) "
        f"RETURN DISTINCT g.ENTITY AS gene_entity, l.PMID AS pmid",
        "papers_genes_related_to_mesh"
    ))
    out.append(q(
        f"Find {journal} papers about genes related to MeSH {m}.",
        f"MATCH (:MeSH {{ENTITY:{mq}}})-[:CO_OCCURS]-(g:Gene)-[:HAS_SOURCE]->(l:Literature) "
        f"WHERE l.Journal = {journal_q} RETURN DISTINCT g.ENTITY AS gene_entity, l.PMID AS pmid",
        "journal_papers_genes_related_to_mesh"
    ))
    out.append(q(
        f"Find papers whose title contains '{kw}' about genes related to MeSH {m}.",
        f"MATCH (:MeSH {{ENTITY:{mq}}})-[:CO_OCCURS]-(g:Gene)-[:HAS_SOURCE]->(l:Literature) "
        f"WHERE toLower(l.Title) CONTAINS toLower({kw_q}) RETURN DISTINCT g.ENTITY AS gene_entity, l.PMID AS pmid",
        "title_papers_genes_related_to_mesh"
    ))
    out.append(q(
        f"Count genes related to MeSH {m}.",
        f"MATCH (:MeSH {{ENTITY:{mq}}})-[:CO_OCCURS]-(g:Gene) RETURN count(DISTINCT g) AS gene_count",
        "count_genes_related_to_mesh"
    ))
    out.append(q(
        f"Find genes related to both MeSH {m} and MeSH {m2}.",
        f"MATCH (:MeSH {{ENTITY:{mq}}})-[:CO_OCCURS]-(g:Gene)-[:CO_OCCURS]-(:MeSH {{ENTITY:{m2q}}}) "
        f"RETURN DISTINCT g.ENTITY AS gene_entity",
        "genes_related_to_two_mesh"
    ))
    out.append(q(
        f"Find papers for genes related to both MeSH {m} and MeSH {m2}.",
        f"MATCH (:MeSH {{ENTITY:{mq}}})-[:CO_OCCURS]-(g:Gene)-[:CO_OCCURS]-(:MeSH {{ENTITY:{m2q}}}) "
        f"MATCH (g)-[:HAS_SOURCE]->(l:Literature) RETURN DISTINCT g.ENTITY AS gene_entity, l.PMID AS pmid",
        "papers_genes_related_to_two_mesh"
    ))

    out.append(q(
        f"Find papers after {y} that mention both {g} and MeSH {m}.",
        f"MATCH (:Gene {{ENTITY:{gq}}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(:MeSH {{ENTITY:{mq}}}) "
        f"WHERE l.Year > {y} RETURN DISTINCT l.PMID AS pmid",
        "papers_gene_mesh_after_year"
    ))
    out.append(q(
        f"Find {journal} papers that mention both {g} and MeSH {m}.",
        f"MATCH (:Gene {{ENTITY:{gq}}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(:MeSH {{ENTITY:{mq}}}) "
        f"WHERE l.Journal = {journal_q} RETURN DISTINCT l.PMID AS pmid",
        "papers_gene_mesh_in_journal"
    ))
    out.append(q(
        f"Find papers whose title contains '{kw}' and mention both {g} and MeSH {m}.",
        f"MATCH (:Gene {{ENTITY:{gq}}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(:MeSH {{ENTITY:{mq}}}) "
        f"WHERE toLower(l.Title) CONTAINS toLower({kw_q}) RETURN DISTINCT l.PMID AS pmid",
        "papers_gene_mesh_title_keyword"
    ))
    out.append(q(
        f"Count papers that mention both {g} and MeSH {m}.",
        f"MATCH (:Gene {{ENTITY:{gq}}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(:MeSH {{ENTITY:{mq}}}) "
        f"RETURN count(DISTINCT l) AS paper_count",
        "count_papers_gene_mesh"
    ))
    out.append(q(
        f"Does any paper mention both {g} and MeSH {m}?",
        f"MATCH (:Gene {{ENTITY:{gq}}})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(:MeSH {{ENTITY:{mq}}}) "
        f"RETURN count(DISTINCT l) > 0 AS exists",
        "exists_paper_gene_mesh"
    ))

    out.append(q(
        f"Find genes in the paper with PMID {pmid_example} that also co-occur with MeSH {m}.",
        f"MATCH (l:Literature {{PMID:{pmid_q}}})<-[:HAS_SOURCE]-(g:Gene)-[:CO_OCCURS]-(:MeSH {{ENTITY:{mq}}}) "
        f"RETURN DISTINCT g.ENTITY AS gene_entity",
        "pmid_genes_related_to_mesh"
    ))
    out.append(q(
        f"Find MeSH entitys in the paper with PMID {pmid_example} that also co-occur with {g}.",
        f"MATCH (l:Literature {{PMID:{pmid_q}}})<-[:HAS_SOURCE]-(m:MeSH)-[:CO_OCCURS]-(:Gene {{ENTITY:{gq}}}) "
        f"RETURN DISTINCT m.ENTITY AS mesh_entity",
        "pmid_mesh_related_to_gene"
    ))
    out.append(q(
        f"Find gene and MeSH pairs reported in the paper with PMID {pmid_example}.",
        f"MATCH (g:Gene)-[:HAS_SOURCE]->(l:Literature {{PMID:{pmid_q}}})<-[:HAS_SOURCE]-(m:MeSH) "
        f"RETURN DISTINCT g.ENTITY AS gene_entity, m.ENTITY AS mesh_entity",
        "pmid_gene_mesh_pairs"
    ))

    out.append(q(
        f"Find papers mentioning any of these genes and MeSH {m}: {', '.join(genes3)}.",
        f"MATCH (g:Gene)-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(:MeSH {{ENTITY:{mq}}}) "
        f"WHERE g.ENTITY IN {cypher_list(genes3)} RETURN DISTINCT g.ENTITY AS gene_entity, l.PMID AS pmid",
        "papers_gene_list_and_mesh"
    ))
    out.append(q(
        f"Find genes co-occurring with any of these MeSH entitys: {', '.join(meshs3)}.",
        f"MATCH (g:Gene)-[:CO_OCCURS]-(m:MeSH) WHERE m.ENTITY IN {cypher_list(meshs3)} "
        f"RETURN DISTINCT g.ENTITY AS gene_entity",
        "genes_related_to_mesh_list"
    ))
    out.append(q(
        f"Find papers for genes co-occurring with any of these MeSH entitys: {', '.join(meshs3)}.",
        f"MATCH (g:Gene)-[:CO_OCCURS]-(m:MeSH) WHERE m.ENTITY IN {cypher_list(meshs3)} "
        f"MATCH (g)-[:HAS_SOURCE]->(l:Literature) RETURN DISTINCT g.ENTITY AS gene_entity, l.PMID AS pmid",
        "papers_for_genes_related_to_mesh_list"
    ))
    out.append(q(
        f"Find papers where MeSH contains '{word}' and genes are reported after {y2}.",
        f"MATCH (m:MeSH)-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(g:Gene) "
        f"WHERE toLower(m.ENTITY) CONTAINS toLower({word_q}) AND l.Year > {y2} "
        f"RETURN DISTINCT l.PMID AS pmid, g.ENTITY AS gene_entity, m.ENTITY AS mesh_entity",
        "keyword_mesh_gene_papers_after_year"
    ))
    out.append(q(
        f"Find MeSH pairs connected through gene {g}.",
        f"MATCH (:MeSH)-[:CO_OCCURS]-(g:Gene {{ENTITY:{gq}}})-[:CO_OCCURS]-(m2:MeSH) "
        f"RETURN DISTINCT m2.ENTITY AS mesh_entity",
        "mesh_pairs_through_gene"
    ))
    out.append(q(
        f"Find genes connected to {g} through MeSH {m}.",
        f"MATCH (:Gene {{ENTITY:{gq}}})-[:CO_OCCURS]-(:MeSH {{ENTITY:{mq}}})-[:CO_OCCURS]-(g2:Gene) "
        f"RETURN DISTINCT g2.ENTITY AS gene_entity",
        "genes_connected_through_specific_mesh"
    ))
    out.append(q(
        f"Find papers of genes connected to {g} through MeSH {m}.",
        f"MATCH (:Gene {{ENTITY:{gq}}})-[:CO_OCCURS]-(:MeSH {{ENTITY:{mq}}})-[:CO_OCCURS]-(g2:Gene)-[:HAS_SOURCE]->(l:Literature) "
        f"RETURN DISTINCT g2.ENTITY AS gene_entity, l.PMID AS pmid",
        "papers_genes_connected_through_specific_mesh"
    ))

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
    ap.add_argument("--out", type=str, default="/mnt/data/train.chat.phase1.jsonl", help="Output JSONL path")
    ap.add_argument("--max-pool", type=int, default=800, help="Cap on unique names sampled from each list")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
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

    per_draw = 20 + 12 + 1 + 30  # basic + extra/thematic + multihop
    draws_needed = max(1, (args.n + per_draw - 1) // per_draw)

    items: List[Dict] = []
    for _ in range(draws_needed):
        g = choose(genes)
        m = choose(meshs)
        batch = make_basic_20(g, m) + make_extra(g, m, genes, meshs) + make_multihop(g, m, genes, meshs)

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
