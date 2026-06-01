"""
DPO training for the answer agent.

Run after SFT.  This script expects a TRL version compatible with the current
Torch/Transformers stack:

    micromamba run -n nl2 pip install trl==0.9.6

Input format is produced by build_answer_dpo_dataset.py:

    {"prompt_messages": [...], "chosen": "...", "rejected": "..."}
"""
from __future__ import annotations

import json
import os

import torch
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from trl import DPOConfig, DPOTrainer
except ImportError as exc:
    raise SystemExit(
        "TRL is not installed. Install it first with:\n"
        "  micromamba run -n nl2 pip install trl\n"
        "Then rerun train_answer_dpo.py."
    ) from exc


BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
SFT_ADAPTER = os.environ.get("SFT_ADAPTER", "answer_rl_training/lora_out_llama3_answer_sft_real_nonempty")
DATA_PATH = os.environ.get("DATA_PATH", "answer_rl_training/train.answer_dpo.synthetic.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "answer_rl_training/lora_out_llama3_answer_dpo")
MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))
USE_4BIT = os.environ.get("USE_4BIT", "1") == "1"
NUM_EPOCHS = float(os.environ.get("NUM_EPOCHS", "1"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-6"))
BETA = float(os.environ.get("BETA", "0.1"))
DO_EVAL = os.environ.get("DO_EVAL", "0") == "1"
USE_SEPARATE_REF = os.environ.get("USE_SEPARATE_REF", "1") == "1"
HF_TOKEN = os.environ.get("HF_TOKEN")


def load_base_model() -> AutoModelForCausalLM:
    quantization_config = None
    if USE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        quantization_config=quantization_config,
    )
    base.config.use_cache = False
    if USE_4BIT:
        base = prepare_model_for_kbit_training(base)
    return base


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = PeftModel.from_pretrained(load_base_model(), SFT_ADAPTER, is_trainable=True)
    model.print_trainable_parameters()

    ref_model = None
    if USE_SEPARATE_REF:
        ref_model = PeftModel.from_pretrained(load_base_model(), SFT_ADAPTER, is_trainable=False)
        ref_model.eval()

    raw = load_dataset("json", data_files=DATA_PATH, split="train")

    def format_record(example):
        prompt_messages = example["prompt_messages"]
        if isinstance(prompt_messages, str):
            prompt_messages = json.loads(prompt_messages)
        return {
            "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True),
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    if DO_EVAL:
        split = raw.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"].map(format_record, remove_columns=split["train"].column_names)
        eval_ds = split["test"].map(format_record, remove_columns=split["test"].column_names)
    else:
        train_ds = raw.map(format_record, remove_columns=raw.column_names)
        eval_ds = None

    args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        evaluation_strategy="steps" if DO_EVAL else "no",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        max_length=MAX_LEN,
        max_prompt_length=MAX_LEN - 256,
        beta=BETA,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"DPO answer LoRA saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
