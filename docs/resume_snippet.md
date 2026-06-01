# Resume Snippet

## English

Built a biomedical knowledge-graph QA system that converts natural-language
questions into Neo4j Cypher using LoRA fine-tuned Llama models, executes queries
over a Gene-MeSH-Literature graph, and generates grounded final answers with an
SFT-trained answer agent plus a rule-based reward guard. Created a 5,000-example
multi-hop NL2Cypher dataset covering 63 query types, implemented training and
evaluation workflows for SFT/DPO experiments, and improved answer faithfulness
by detecting entity, PMID, count, and relationship hallucinations.

## Short Version

Developed a LoRA-based biomedical NL2Cypher QA pipeline for a Neo4j
Gene-MeSH-Literature graph, including multi-hop data generation, model
fine-tuning, grounded answer generation, and reward-based hallucination checks.

## Project Bullets

- Fine-tuned Llama-family models with LoRA to translate biomedical natural
  language questions into executable Neo4j Cypher queries.
- Built a two-agent QA pipeline that executes generated Cypher against Neo4j and
  writes database-grounded answers from structured query results.
- Generated and evaluated a 5,000-example multi-hop training set spanning 63
  Gene, MeSH, Literature, co-occurrence, count, and shared-evidence query types.
- Designed a rule-based answer reward/guard to penalize entity distortion,
  unsupported PMIDs, incorrect counts, and misleading relationship semantics.
