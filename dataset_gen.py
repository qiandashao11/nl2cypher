#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_cypher_contextual.py
English-only natural instruction → Cypher dataset with a wording map that hides graph jargon.
Schema assumed: Gene/Literature/MeSH; CO_OCCURS/HAS_SOURCE (+ dotted props quoted).
Usage:
  python make_cypher_contextual.py --out dataset_en.jsonl --max_per_template 30 --seed 42
"""

import json, random, argparse

# -----------------------
# Vocabulary (replace with samples from your DB if desired)
# -----------------------
VOCAB = {
    "GENE": ["TP53","EGFR","BRCA1","KRAS","PIK3CA","MYC","ACSL3"],
    "JOURNAL": ["Nature","Science","Cell","NEJM","Lancet Oncology"],
    "YEAR": [str(y) for y in range(2018, 2025)],
    "KW": ["apoptosis","metastasis","angiogenesis","biomarker","immunotherapy"],
    "MESH": ["Apoptosis","Signal Transduction","DNA Repair","Cell Cycle"]
}
LIMITS = [5,10,20,50]

def qprop(p: str) -> str:
    return f"`{p}`" if "." in p else p

def ch(v): return random.choice(v)
def samp(v,k): return random.sample(v,k)

def dedup(rows):
    seen=set(); out=[]
    for r in rows:
        k=(r["instruction"], r["output"])
        if k in seen: continue
        seen.add(k); out.append(r)
    return out

# -----------------------
# Tech → Context wording map
# -----------------------
WORDING = {
    "radius_small": [
        "immediate network", "local neighborhood", "surroundings",
        "nearby area of the co-occurrence graph", "close vicinity"
    ],
    "shortest_path": [
        "most direct connection", "briefest chain of connections",
        "minimal route that links them", "shortest connection"
    ],
    "direct_neighbors": [
        "directly connected genes", "immediate neighbors",
        "genes linked straight to", "first-degree connections"
    ],
    "significance": [
        "statistically significant links", "connections that look significant",
        "associations passing a significance check"
    ],
    "has_source": [
        "linked publications", "associated PMIDs", "attached literature"
    ],
    "strength": [
        "how strong the link is", "connection strength",
        "edge strength in this network"
    ]
}

def choose_phrase(key): return ch(WORDING[key])

# -----------------------
# Natural instruction builders (contextual phrasings)
# -----------------------
def say_direct_neighbors(gene, limit):
    p = choose_phrase("direct_neighbors")
    return [
        f"Which genes are {p} to {gene}?",
        f"Show {gene}'s {p}, up to {limit}.",
        f"List the genes that sit right next to {gene} in the co-occurrence network.",
        f"Give me the genes linked straight to {gene}.",
        f"Return distinct genes directly connected to {gene}."
    ]

def say_small_neighborhood(gene, limit):
    ctx = choose_phrase("radius_small")
    return [
        f"Which genes are in {gene}'s {ctx}?",
        f"Show genes around {gene} in its {ctx}, up to {limit}.",
        f"List genes closely connected to {gene} within its {ctx}.",
        f"Give me genes that stay very close to {gene} in the co-occurrence network.",
        f"Return nearby genes for {gene}."
    ]

def say_shortest(a,b):
    sp = choose_phrase("shortest_path")
    return [
        f"Find the {sp} between {a} and {b}.",
        f"What is the {sp} linking {a} to {b}?",
        f"Show the {sp} from {a} to {b} in the co-occurrence graph.",
        f"Return the minimal route that ties {a} to {b}."
    ]

def say_lit_since(journal, year, limit):
    return [
        f"List papers published in {journal} since {year}.",
        f"Show publications that appeared in {journal} from {year} onward.",
        f"Return recent {journal} articles not earlier than {year}, up to {limit}.",
        f"Pull {journal} papers starting in {year}."
    ]

def say_lit_kw(kw, limit):
    return [
        f"Find papers with “{kw}” mentioned in the title.",
        f"Show articles whose titles include “{kw}”.",
        f"Which publications have “{kw}” in their titles?",
        f"Return literature that references “{kw}” in the title, up to {limit}."
    ]

def say_optional_pmids(gene, limit):
    phr = choose_phrase("has_source")
    return [
        f"Show {phr} for {gene}, if any.",
        f"Return {phr} for {gene}, up to {limit}.",
        f"Which publications are attached to {gene}?",
        f"List PMIDs associated with {gene}."
    ]

def say_edge_strength(gene, limit):
    s = choose_phrase("strength")
    return [
        f"List genes closely tied to {gene} and include {s}.",
        f"Show {gene}'s connected genes ordered by {s}.",
        f"Return genes linked to {gene} with their {s}, up to {limit}.",
        f"Give me {gene}'s neighbors together with {s}."
    ]

def say_significance(gene, limit):
    sg = choose_phrase("significance")
    return [
        f"Filter {gene}'s associated genes by {sg}.",
        f"Show {gene}'s related genes that pass a significance check.",
        f"Return {gene}'s connections limited to significant associations, up to {limit}.",
        f"Give me {gene}'s neighbors where the link is statistically meaningful."
    ]

def say_mesh(mesh, limit):
    return [
        f"Find MeSH terms containing “{mesh}”.",
        f"Show MeSH entries that mention “{mesh}”.",
        f"Return MeSH terms that include “{mesh}”, up to {limit}.",
        f"List MeSH concepts matching “{mesh}”."
    ]

# -----------------------
# Cypher templates (schema-accurate)
# -----------------------
def t_direct_neighbors():
    gene = ch(VOCAB["GENE"]); lim = ch(LIMITS)
    cy = f'''MATCH (g:Gene {{ENTITY:"{gene}"}})-[:CO_OCCURS]-(n:Gene)
RETURN DISTINCT n.ENTITY AS gene
ORDER BY gene
LIMIT {lim}'''
    return [{"instruction": i, "output": cy, "category":"read/direct-neighbors"} for i in say_direct_neighbors(gene, lim)]

def t_near_local():
    # Internally uses *1..2 but wording is contextual (no numbers).
    gene = ch(VOCAB["GENE"]); lim = ch(LIMITS)
    cy = f'''MATCH p=(g:Gene {{ENTITY:"{gene}"}})-[:CO_OCCURS*1..2]-(n:Gene)
RETURN DISTINCT n.ENTITY AS gene, length(p) AS steps
ORDER BY steps, gene
LIMIT {lim}'''
    return [{"instruction": i, "output": cy, "category":"read/local-neighborhood"} for i in say_small_neighborhood(gene, lim)]

def t_shortest_route():
    a,b = samp(VOCAB["GENE"], 2)
    cy = f'''MATCH (a:Gene {{ENTITY:"{a}"}}), (b:Gene {{ENTITY:"{b}"}})
MATCH p=shortestPath((a)-[:CO_OCCURS*]-(b))
RETURN [n IN nodes(p) | n.ENTITY] AS path, length(p) AS steps'''
    return [{"instruction": i, "output": cy, "category":"read/shortest-route"} for i in say_shortest(a, b)]

def t_literature_since():
    j = ch(VOCAB["JOURNAL"]); y = ch(VOCAB["YEAR"]); lim = ch(LIMITS)
    cy = f'''MATCH (p:Literature)
WHERE p.Journal = "{j}" AND toInteger(p.Year) >= {y}
RETURN p.Title AS title, p.Year AS year, p.Journal AS journal
ORDER BY year DESC, title
LIMIT {lim}'''
    return [{"instruction": i, "output": cy, "category":"read/lit-since"} for i in say_lit_since(j, y, lim)]

def t_literature_kw():
    kw = ch(VOCAB["KW"]); lim = ch(LIMITS)
    cy = f'''MATCH (p:Literature)
WHERE toLower(p.Title) CONTAINS "{kw}"
RETURN p.Title AS title, p.Journal AS journal, p.Year AS year
ORDER BY year DESC
LIMIT {lim}'''
    return [{"instruction": i, "output": cy, "category":"read/lit-title"} for i in say_lit_kw(kw, lim)]

def t_optional_pmids():
    gene = ch(VOCAB["GENE"]); lim = ch(LIMITS)
    cy = f'''MATCH (g:Gene {{ENTITY:"{gene}"}})
OPTIONAL MATCH (g)-[:HAS_SOURCE]-(p:Literature)
RETURN g.ENTITY AS gene, coalesce(collect(p.PMID), []) AS pmids
LIMIT {lim}'''
    return [{"instruction": i, "output": cy, "category":"read/optional-pmids"} for i in say_optional_pmids(gene, lim)]

def t_edge_strength():
    gene = ch(VOCAB["GENE"]); lim = ch(LIMITS)
    cy = f'''MATCH (g:Gene {{ENTITY:"{gene}"}})-[r:CO_OCCURS]-(n:Gene)
RETURN n.ENTITY AS gene, r.Strength AS strength, r.{qprop("Raw.CO.Occurrence")} AS raw_co, r.{qprop("Expansion")} AS expansion
ORDER BY strength DESC NULLS LAST, gene
LIMIT {lim}'''
    return [{"instruction": i, "output": cy, "category":"read/edge-strength"} for i in say_edge_strength(gene, lim)]

def t_significant_links():
    gene = ch(VOCAB["GENE"]); lim = ch(LIMITS)
    cy = f'''MATCH (g:Gene {{ENTITY:"{gene}"}})-[r:CO_OCCURS]-(n:Gene)
WHERE coalesce(toFloat(r.{qprop("P.value")}),1.0) < 0.05
   OR coalesce(toFloat(r.{qprop("FisherP.value")}),1.0) < 0.05
RETURN n.ENTITY AS gene, r.{qprop("P.value")} AS pval, r.{qprop("FisherP.value")} AS fpval
ORDER BY pval ASC NULLS LAST, gene
LIMIT {lim}'''
    return [{"instruction": i, "output": cy, "category":"read/significance"} for i in say_significance(gene, lim)]

def t_mesh_contains():
    mesh = ch(VOCAB["MESH"]); lim = ch(LIMITS)
    cy = f'''MATCH (m:MeSH)
WHERE toLower(m.ENTITY) CONTAINS "{mesh.lower()}"
RETURN m.ENTITY AS mesh
ORDER BY mesh
LIMIT {lim}'''
    return [{"instruction": i, "output": cy, "category":"read/mesh"} for i in say_mesh(mesh, lim)]

TEMPLATES = [
    t_direct_neighbors,
    t_near_local,
    t_shortest_route,
    t_literature_since,
    t_literature_kw,
    t_optional_pmids,
    t_edge_strength,
    t_significant_links,
    t_mesh_contains,
]

# -----------------------
# Generate & save
# -----------------------
def generate(max_per_template=30):
    rows=[]
    for t in TEMPLATES:
        for _ in range(max_per_template):
            rows.extend(t())
    return dedup(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="dataset_en.jsonl")
    ap.add_argument("--max_per_template", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    data = generate(args.max_per_template)
    with open(args.out, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"✅ Wrote {len(data)} samples to {args.out}")
