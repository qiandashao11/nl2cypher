# Answer Generation Experiment Summary

Date: 2026-05-20

## Goal

Compare four second-stage answer generation settings:

1. Baseline: original Llama answer generator
2. SFT: `answer_sft_real_nonempty`
3. SFT + reward guard
4. SFT + DPO/RL

The Cypher generator is unchanged in these tests:

`phase3_multihop_training/lora_out_llama3_8b_multihop`

## Test Files

| Setting | Output file | Notes |
| --- | --- | --- |
| Baseline original Llama | `phase3_multihop_training/qa_test_results_llm_10.md` | Older result file, no reward blocks in this markdown. Shows obvious hallucination/editing issues. |
| SFT raw | `phase3_multihop_training/qa_test_results_sft_raw_10.md` | Uses `answer_rl_training/lora_out_llama3_answer_sft_real_nonempty`, reward guard disabled. |
| SFT + reward guard | `phase3_multihop_training/qa_test_results_sft_guard_10.md` | Same SFT adapter, reward guard enabled. |
| DPO raw, full epoch | `phase3_multihop_training/qa_test_results_dpo_10.md` | Uses `answer_rl_training/lora_out_llama3_answer_dpo`, reward guard disabled. |
| DPO + reward guard | `phase3_multihop_training/qa_test_results_dpo_guard_10.md` | Same DPO adapter, reward guard enabled. |
| DPO raw, SFT-ref 50-step | `phase3_multihop_training/qa_test_results_dpo_ref50_10.md` | Uses `answer_rl_training/lora_out_llama3_answer_dpo_ref50`, reward guard disabled. |
| DPO raw, real failures 30-step | `phase3_multihop_training/qa_test_results_dpo_real_failures_30_10.md` | Uses 99 real SFT-failure preference pairs. |
| DPO raw, real failures 250-pair 50-step | `phase3_multihop_training/qa_test_results_dpo_real_failures_250_50_10.md` | Uses 250 real SFT-failure preference pairs. |

## Reward Summary

| Setting | Avg reward | Min | Max | Verdict |
| --- | ---: | ---: | ---: | --- |
| SFT raw | 0.8105 | 0.5780 | 0.9238 | Better structure, but still edits entities/titles. |
| SFT + reward guard | 0.9598 | 0.8059 | 1.0000 | Current best. |
| DPO raw, full epoch | 0.6237 | 0.0000 | 0.9238 | Not usable: over-optimized and damages text generation. |
| DPO + reward guard | 0.9598 | 0.8059 | 1.0000 | Guard catches failures, but DPO itself did not help. |
| DPO raw, SFT-ref 50-step | 0.7373 | 0.0000 | 0.9238 | Softer than full DPO, but still worse than SFT raw. |
| DPO raw, real failures 30-step | 0.7120 | 0.0000 | 0.9238 | Not enough improvement; same entity/title corruption remains. |
| DPO raw, real failures 250-pair 50-step | 0.7120 | 0.0000 | 0.9238 | More real failures did not improve the fixed held-out test. |

## Main Finding

The best current pipeline is:

`Cypher LoRA -> SFT answer generator -> reward guard fallback`

DPO/RL is not ready as a replacement. The first full DPO run learned the synthetic preference task too aggressively and produced distorted outputs, including changed entity names, changed titles, and strange character artifacts. The softer 50-step DPO run with a frozen SFT reference avoided the worst collapse, but it still did not improve over SFT raw. A follow-up real-failure DPO experiment collected actual low-reward SFT outputs, but the 30-step and 250-pair 50-step runs still did not improve the held-out 10-question test.

## Why DPO Did Not Help Yet

- The rejected answers are synthetic corruptions, not real model failures sampled from the current answer model.
- The full DPO run used too much optimization for an easy synthetic task.
- The first DPO script used an insufficiently stable reference setup; this was changed to support a frozen SFT reference.
- DPO optimizes preference separation, not direct factual correctness. If the rejected samples are too artificial, the model learns odd surface-level avoidance.
- The reward guard still performs better because it checks the actual final answer against the Cypher result.
- Real-failure DPO did collect better rejected samples, but the current LoRA/DPO setup still fails to fix exact-copy behavior on held-out questions. The main remaining failure is token-level copying of MeSH terms, titles, and gene symbols.

## Files Created Or Updated

- `answer_rl_training/build_answer_dpo_dataset.py`
- `answer_rl_training/train_answer_dpo.py`
- `answer_rl_training/train.answer_dpo.synthetic.jsonl`
- `answer_rl_training/lora_out_llama3_answer_dpo`
- `answer_rl_training/lora_out_llama3_answer_dpo_ref50`
- `phase3_multihop_training/qa_test_results_sft_raw_10.md`
- `phase3_multihop_training/qa_test_results_sft_guard_10.md`
- `phase3_multihop_training/qa_test_results_dpo_10.md`
- `phase3_multihop_training/qa_test_results_dpo_guard_10.md`
- `phase3_multihop_training/qa_test_results_dpo_ref50_10.md`
- `answer_rl_training/build_answer_dpo_from_model_failures.py`
- `answer_rl_training/train.answer_dpo.real_failures.jsonl`
- `answer_rl_training/train.answer_dpo.real_failures.batch250.jsonl`
- `answer_rl_training/lora_out_llama3_answer_dpo_real_failures_30`
- `answer_rl_training/lora_out_llama3_answer_dpo_real_failures_250_50`
- `phase3_multihop_training/qa_test_results_dpo_real_failures_30_10.md`
- `phase3_multihop_training/qa_test_results_dpo_real_failures_250_50_10.md`

## Recommended Next Step

Keep `SFT + reward guard` as the current usable version.

For the next DPO/RL iteration, build preference data from real failures:

1. Run SFT raw on many real KG questions.
2. Score each answer with `answer_reward.py`.
3. Use high-reward guarded/fallback answers as `chosen`.
4. Use the model's low-reward raw answers as `rejected`.
5. Train with frozen SFT reference, low learning rate, and small step budget.
6. Accept DPO only if raw DPO improves over SFT raw on held-out questions.

The real-failure iteration has now been tried at small scale. The next best technical direction is not more DPO steps, but stronger constrained generation:

1. Force exact-value rendering for small results before the LLM writes any explanation.
2. Add a copy-first answer schema such as `facts -> answer`, where facts are immutable strings copied from the DB.
3. Use DPO only after the rejected/chosen pairs target this copy-first format.
4. Keep reward guard as the production safety layer.
