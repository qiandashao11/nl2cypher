# ENQUIRE BioKG NL2Cypher QA

Natural-language question answering over an
[ENQUIRE](https://github.com/Muszeb/ENQUIRE)-style biomedical knowledge graph.
This repository adds an NL2Cypher and answer-generation layer on top of a
Gene-MeSH-Literature Neo4j graph: it trains and evaluates LoRA adapters that
translate English biomedical questions into Cypher, executes the query, and
writes grounded answers from the database result.

## Relationship To ENQUIRE

[ENQUIRE](https://github.com/Muszeb/ENQUIRE) (Expanding Networks by Querying
Unexpectedly Inter-Related Entities) is the upstream biomedical text-mining and
network-expansion project. It reconstructs and expands co-occurrence networks
of genes and biomedical ontologies from user-selected PubMed literature corpora
and network-inferred PubMed queries.

This repository is a downstream QA extension for ENQUIRE/ENQUIRE2KG-style
outputs. It assumes the literature-mined entities and relationships have been
loaded into Neo4j, then focuses on:

- translating natural-language biomedical questions into executable Cypher;
- querying the ENQUIRE-derived Gene/MeSH/Literature graph;
- generating concise, faithful answers from the Cypher result;
- evaluating answer faithfulness with reward/guard checks.

It is not a replacement for ENQUIRE itself and does not vendor the original
ENQUIRE pipeline. Use the upstream ENQUIRE project to produce or inspect the
source literature-mining workflow.

## Highlights

- Downstream QA layer for ENQUIRE/ENQUIRE2KG biomedical co-occurrence graphs.
- Two-stage QA pipeline: question -> Cypher -> Neo4j result -> final answer.
- Gene-MeSH-Literature schema with `HAS_SOURCE` and `CO_OCCURS` relationships.
- Phase 3 multi-hop NL2Cypher training set with 5,000 examples and 63 query
  types.
- Answer-agent SFT/DPO experiments with a rule-based faithfulness reward.
- Current best answer setting: SFT answer generator plus reward-guard fallback.
- CLI and Tkinter GUI entry points for local querying.

## Repository Layout

```text
.
|-- non-parameter/
|   |-- cypher_generator.py              # LoRA-backed NL -> Cypher generator
|   |-- neo4j_executor.py                # Neo4j execution wrapper
|   |-- qa_system*.py                    # End-to-end QA variants
|   |-- cli_app.py / gui_app.py          # Local interfaces
|   |-- enquire2kg-KGtest/               # ENQUIRE2KG input/config example
|   |-- phase3_multihop_training/        # Multi-hop NL2Cypher data and evals
|   `-- answer_rl_training/              # Answer SFT/DPO data, reward, training
|-- pipline/                             # Earlier pipeline prototype
|-- *.jsonl / *.csv / *.txt              # Small datasets and graph resources
|-- requirements.txt
`-- .env.example
```

Large model directories and checkpoints are intentionally ignored by git. Keep
LoRA adapters, optimizer states, and local base-model downloads outside version
control, or publish them separately through Hugging Face / Git LFS.

## Setup

```bash
cd nl2cypher
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with conda/mamba:

```bash
conda env create -f environment.yml
conda activate nl2cypher
```

For gated Hugging Face models, export a token locally:

```bash
export HF_TOKEN=your_huggingface_token
```

Neo4j must be running with the expected Gene, MeSH, and Literature graph loaded.
The default connection is `bolt://localhost:7687`.

## Expected Graph

The QA layer expects an ENQUIRE-derived Neo4j graph with the following core
schema:

- `Gene` nodes, keyed by `ENTITY`
- `MeSH` nodes, keyed by `ENTITY`
- `Literature` nodes with fields such as `PMID`, `Title`, `Year`, and `Journal`
- `HAS_SOURCE` edges from `Gene`/`MeSH` entities to supporting `Literature`
- undirected `CO_OCCURS` semantics between `Gene-Gene` and `Gene-MeSH`

## Run Inference

Run from `nl2cypher/non-parameter` after placing or training the required LoRA
adapters:

```bash
python qa_system_sft.py \
  --question "Which genes co-occur with MeSH term Breast Neoplasms?" \
  --cypher_lora_dir phase3_multihop_training/lora_out_llama3_8b_multihop \
  --answer_lora_dir answer_rl_training/lora_out_llama3_answer_sft_real_nonempty \
  --neo4j_uri bolt://localhost:7687 \
  --neo4j_user neo4j \
  --neo4j_password "$NEO4J_PASSWORD"
```

Interactive CLI:

```bash
python cli_app.py --interactive \
  --lora_dir phase3_multihop_training/lora_out_llama3_8b_multihop
```

GUI:

```bash
python gui_app.py
```

## Train NL2Cypher LoRA

```bash
cd nl2cypher/non-parameter
DATA_PATH=phase3_multihop_training/train.chat.phase3_multihop.jsonl \
OUTPUT_DIR=phase3_multihop_training/lora_out_llama3_8b_multihop \
python llama31.py
```

## Train Answer Agent

Build real non-empty answer examples:

```bash
cd nl2cypher/non-parameter
python answer_rl_training/generate_real_answer_sft_dataset.py \
  --target 3000 \
  --max_rows 20 \
  --output answer_rl_training/train.answer_sft.real_nonempty.jsonl
```

Train the answer LoRA:

```bash
DATA_PATH=answer_rl_training/train.answer_sft.real_nonempty.jsonl \
OUTPUT_DIR=answer_rl_training/lora_out_llama3_answer_sft_real_nonempty \
python answer_rl_training/train_answer_sft.py
```

## Evaluation Notes

The answer-generation experiments are summarized in:

- `non-parameter/answer_rl_training/answer_experiment_summary.md`
- `non-parameter/phase3_multihop_training/qa_test_results_*.md`

The current practical pipeline is:

```text
Cypher LoRA -> SFT answer generator -> reward guard fallback
```

DPO experiments are included for research traceability, but the raw DPO models
did not outperform SFT plus reward guard on the held-out QA checks.

## Acknowledgements

This project builds on the biomedical network-mining direction of
[Muszeb/ENQUIRE](https://github.com/Muszeb/ENQUIRE). If you use ENQUIRE-derived
data or workflows, cite the upstream ENQUIRE project as requested in their
repository.

## GitHub Publishing Notes

- Do not commit `.env`, Hugging Face tokens, Neo4j passwords, or private data.
- Do not commit `lora_out*`, `checkpoint-*`, `.safetensors`, `.pt`, `.pth`, or
  local base-model directories.
- If a token was ever committed or shared, revoke it and create a new one.
- Keep large artifacts in an external model registry and document how to fetch
  them.
