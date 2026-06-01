#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nl2cypher_fixed_schema_entity.py
Generate read-only Cypher from natural language for a FIXED schema:

Identifiers:
- Gene.ENTITY       -> gene symbol (e.g., TP53)
- MeSH.ENTITY       -> mesh term
- Literature.PMIDList -> literature identifier string/list

Relationships:
- (Gene)-[:CO_OCCURS {Expansion, adj.P.value, P.value, FisherP.value, Raw.CO.Occurrence, Strength, Wscore, Representativeness, ztPois.cdf}]->(Gene)
- (Gene)-[:CO_OCCURS]->(MeSH)
- (MeSH)-[:CO_OCCURS]->(MeSH)
- (Gene)-[:HAS_SOURCE]->(Literature)
- (MeSH)-[:HAS_SOURCE]->(Literature)

This script does NOT connect to Neo4j. It only prints the generated Cypher.
Model: microsoft/phi-3-mini-4k-instruct
Usage:
  micromamba activate slm_nl2cypher
  python nl2cypher_fixed_schema_entity.py "return genes co-occurring with TP53 (Expansion=1)"
"""

import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = "microsoft/phi-3-mini-4k-instruct"

SYSTEM_RULES = """You are a Cypher query generator for Neo4j.
Translate the user question into a valid READ-ONLY Cypher query for the fixed schema below.

Schema:
- Node labels:
  * Gene {ENTITY, Knode, `Knode.rank`, Betweenness.centrality, Closeness.centrality, CommunityMembership, Strength, Wscore, Representativeness, `Raw.Occurrence`}
  * MeSH {ENTITY, QuerySet, Link, `Link.explicit`}
  * Literature {PMIDList, PMID, Title, Year, Journal, Authors, Issue, `DOI.link`, Affiliations, `Title.Abstract.OT`}

- Relationships:
  * (Gene)-[:CO_OCCURS {...}]->(Gene)
  * (Gene)-[:CO_OCCURS]->(MeSH)
  * (MeSH)-[:CO_OCCURS]->(MeSH)
  * (Gene)-[:HAS_SOURCE]->(Literature)
  * (MeSH)-[:HAS_SOURCE]->(Literature)

Hard rules:
- Use ENTITY for Gene/MeSH identifiers; use PMIDList for Literature.
- CO_OCCURS is NEVER used with Literature.
- Expansion and p-values belong on CO_OCCURS relationships only.
- Allowed: MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT.
- Forbidden: CREATE, MERGE, DELETE, DROP, SET, REMOVE, LOAD CSV, dbms.*, apoc.*
- Prefer undirected CO_OCCURS ( -[:CO_OCCURS {..}]- ) unless direction is specified.
- Output ONLY a single fenced block: ```cypher ... ```

Examples:
User: return genes co-occurring with TP53 (Expansion=1)
Cypher:
```cypher
MATCH (g:Gene {ENTITY:"TP53"})-[:CO_OCCURS {Expansion:1}]-(p:Gene)
RETURN DISTINCT p.ENTITY AS gene
ORDER BY gene
```
User: list MeSH terms co-occurring with TP53 (Expansion=1), limit 50
Cypher:
```cypher
MATCH (g:Gene {ENTITY:"TP53"})-[:CO_OCCURS {Expansion:1}]-(m:MeSH)
RETURN DISTINCT m.ENTITY AS mesh
ORDER BY mesh
LIMIT 50
```
User: return genes from literature with PMIDList contains 34093872
Cypher:
```cypher
MATCH (l:Literature {PMIDList:"[34093872]"})<-[:HAS_SOURCE]-(g:Gene)
RETURN DISTINCT g.ENTITY AS gene, l.PMIDList AS literature
ORDER BY gene
```
"""

PROMPT_TEMPLATE = """{rules}

User question (do not translate names/strings):
{question}

Output only the Cypher in a fenced code block:
```cypher
"""

def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(rules=SYSTEM_RULES, question=question)

def extract_cypher(text: str) -> str:
    m = re.search(r"```cypher\s*(.*?)```", text, flags=re.S|re.I)
    if m:
        return m.group(1).strip()
    m2 = re.findall(r"```[a-zA-Z]*\n(.*?)```", text, flags=re.S)
    return m2[-1].strip() if m2 else text.strip()

def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "return genes co-occurring with TP53 (Expansion=1)"
    print(f"Loading model: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    print("Model loaded on:", next(model.parameters()).device)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    prompt = build_prompt(question)
    out = generator(prompt, max_new_tokens=320, do_sample=False, temperature=0.1)[0]["generated_text"]
    cypher = extract_cypher(out)

    print("\n=== Raw model output ===\n", out)
    print("\n=== Extracted Cypher ===\n", cypher)

if __name__ == "__main__":
    main()