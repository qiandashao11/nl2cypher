# Phase 3 Multihop Non-Parameter Training

This folder contains the expanded non-parameter NL2Cypher training set for the
Gene-MeSH-Literature graph.

## Schema

- `Gene(ENTITY)`
- `MeSH(ENTITY)`
- `Literature(PMID, Title, Year, Journal)`

Relationships:

- `(Gene)-[:HAS_SOURCE]->(Literature)`
- `(MeSH)-[:HAS_SOURCE]->(Literature)`
- `(Gene)-[:CO_OCCURS]-(Gene)`
- `(Gene)-[:CO_OCCURS]-(MeSH)`

## Dataset

- `train.chat.phase3_multihop.jsonl`
- 5000 examples
- 63 query types
- No `$parameter` placeholders
- No lowercase `.entity`
- No `LIMIT` or `SKIP`

The dataset extends the previous phase with multi-hop and compositional query
types, including shared MeSH, shared papers, Gene-MeSH-Gene-Literature paths,
PMID-based lookup, filters, counts, and existence checks.

## Train

Run from `nl2cypher/non-parameter`:

```bash
DATA_PATH=phase3_multihop_training/train.chat.phase3_multihop.jsonl \
OUTPUT_DIR=phase3_multihop_training/lora_out_llama3_8b_multihop \
python3 llama31.py
```

