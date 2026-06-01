# Answer-Agent SFT/RL Training

This folder is for training the second-stage answer generator only.

The existing NL2Cypher model stays unchanged:

```text
question -> Cypher -> Neo4j result -> answer agent
```

The new training target is:

```text
question + gold/executed Cypher + Neo4j result -> faithful final answer
```

## Files

- `build_answer_sft_dataset.py`
  - Executes gold Cypher from `phase3_multihop_training/train.chat.phase3_multihop.jsonl`
  - Builds answer-agent SFT records
  - Uses deterministic clean answers as gold answers
  - Adds a rule-based faithfulness reward score for each example

- `generate_real_answer_sft_dataset.py`
  - Recommended main data generator
  - Samples real entities, papers, journals, and relationship combinations from Neo4j
  - Generates only positive/non-empty answer-agent SFT examples by default
  - Avoids the empty-result bias caused by random synthetic combinations

- `answer_reward.py`
  - Rule-based reward function for future RL/DPO
  - Checks entity preservation, count accuracy, result coverage, CO_OCCURS semantics, format cleanliness, and fluency

- `train_answer_sft.py`
  - LoRA SFT warm-up for the answer agent
  - Should be run before DPO/RL

- `balance_answer_dataset.py`
  - Downsamples empty-result examples so the answer model does not learn to
    over-answer with "no matching results"

## Step 1: Build a small verification dataset

Recommended real non-empty generator:

```bash
micromamba run -n nl2 python answer_rl_training/generate_real_answer_sft_dataset.py \
  --target 50 \
  --max_rows 20 \
  --output answer_rl_training/train.answer_sft.real_nonempty.sample50.jsonl
```

Older generator from the NL2Cypher training JSONL:

```bash
micromamba run -n nl2 python answer_rl_training/build_answer_sft_dataset.py \
  --limit 50 \
  --max_rows 20 \
  --neo4j_uri bolt://localhost:7687 \
  --output answer_rl_training/train.answer_sft.sample50.jsonl
```

## Step 2: Build the full SFT dataset

Recommended real non-empty SFT dataset:

```bash
micromamba run -n nl2 python answer_rl_training/generate_real_answer_sft_dataset.py \
  --target 3000 \
  --max_rows 20 \
  --output answer_rl_training/train.answer_sft.real_nonempty.jsonl
```

The older generator below is useful for robustness data, but it may contain many
empty results because some synthetic entity combinations do not exist in Neo4j:

```bash
micromamba run -n nl2 python answer_rl_training/build_answer_sft_dataset.py \
  --max_rows 20 \
  --dedupe_question \
  --neo4j_uri bolt://localhost:7687 \
  --output answer_rl_training/train.answer_sft.jsonl
```

## Step 3: SFT warm-up

If you use the older generator, balance the generated data first:

```bash
micromamba run -n nl2 python answer_rl_training/balance_answer_dataset.py \
  --input answer_rl_training/train.answer_sft.jsonl \
  --output answer_rl_training/train.answer_sft.balanced.jsonl \
  --empty_ratio 0.5
```

Then train:

```bash
DATA_PATH=answer_rl_training/train.answer_sft.real_nonempty.jsonl \
OUTPUT_DIR=answer_rl_training/lora_out_llama3_answer_sft_real_nonempty \
micromamba run -n nl2 python answer_rl_training/train_answer_sft.py
```

Useful environment variables:

```bash
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct
DATA_PATH=answer_rl_training/train.answer_sft.real_nonempty.jsonl
OUTPUT_DIR=answer_rl_training/lora_out_llama3_answer_sft_real_nonempty
MAX_LEN=2048
USE_4BIT=1
NUM_EPOCHS=2
LORA_R=16
```

## Step 4: RL/DPO direction

The current `nl2` environment does not have `trl` installed. After installing
`trl`, the reward in `answer_reward.py` can be used in PPO/GRPO-style training.

Before RL, prefer this sequence:

```text
SFT -> DPO with chosen/rejected answers -> RL reward optimization
```

Recommended reward components:

```text
0.30 entity_faithfulness
0.20 count_accuracy
0.20 result_coverage
0.15 relation_semantics
0.10 format_cleanliness
0.05 fluency
- hallucination_penalty
```

The reward is intentionally strict about exact strings. It should punish errors
like:

```text
AIFM2 -> AIFM1
GPX4 -> GXP4
SLC40A1 -> SLC40A11
38301989 -> 323001989
count 32 -> 40
CO_OCCURS -> "causes" / "regulates" / "co-expressed"
```
