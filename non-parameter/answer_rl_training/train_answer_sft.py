"""
SFT warm-up for the second-stage answer agent.

This trains a LoRA adapter to map:

    question + Cypher + compact Neo4j result -> faithful natural-language answer

Use this before DPO/RL.  A model that has not first learned the answer format is
usually too unstable for direct reward optimization.
"""
from __future__ import annotations

import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DATA_PATH = os.environ.get("DATA_PATH", "answer_rl_training/train.answer_sft.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "answer_rl_training/lora_out_llama3_answer_sft")
MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))
USE_4BIT = os.environ.get("USE_4BIT", "1") == "1"
LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"
HF_TOKEN = os.environ.get("HF_TOKEN")
EVAL_RATIO = float(os.environ.get("EVAL_RATIO", "0.1"))
NUM_EPOCHS = float(os.environ.get("NUM_EPOCHS", "2"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))

IS_LOCAL_PATH = os.path.isdir(MODEL_NAME)

print(f"Base model : {MODEL_NAME} ({'local dir' if IS_LOCAL_PATH else 'remote id'})")
print(f"Data file  : {DATA_PATH}")
print(f"Output dir : {OUTPUT_DIR}")
print(f"4-bit quant: {USE_4BIT}")
print(f"Local-only : {LOCAL_FILES_ONLY}")
print(f"Max length : {MAX_LEN}")


def load_messages(example):
    messages = example.get("messages", [])
    if isinstance(messages, str):
        messages = json.loads(messages)
    assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    if assistant is None:
        raise ValueError("Each record must contain an assistant message.")
    prompt_messages = messages[: messages.index(assistant)]
    return {
        "prompt_messages": json.dumps(prompt_messages, ensure_ascii=False),
        "target": assistant.get("content", "").strip(),
    }


def pick_target_modules(model):
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.add(name.split(".")[-1])
    llama_like = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if llama_like & names:
        return sorted(llama_like & names)
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")
splits = raw_ds.train_test_split(test_size=EVAL_RATIO, seed=42)
train_ds = splits["train"].map(load_messages)
eval_ds = splits["test"].map(load_messages)

bnb_config = None
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

common_kwargs = {"local_files_only": LOCAL_FILES_ONLY or IS_LOCAL_PATH}
if HF_TOKEN and not (LOCAL_FILES_ONLY or IS_LOCAL_PATH):
    common_kwargs["token"] = HF_TOKEN

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    trust_remote_code=False,
    **common_kwargs,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

config = AutoConfig.from_pretrained(MODEL_NAME, **common_kwargs)
config.use_cache = False

model_kwargs = {
    "config": config,
    "device_map": "auto",
    "trust_remote_code": False,
    "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    "quantization_config": bnb_config if USE_4BIT else None,
    **common_kwargs,
}

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
targets = pick_target_modules(model)
print("LoRA target_modules ->", targets)

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    target_modules=targets,
    bias="none",
    task_type="CAUSAL_LM",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()


def tokenize_function(batch):
    prompt_messages = [json.loads(item) for item in batch["prompt_messages"]]
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in prompt_messages
    ]
    targets = batch["target"]
    merged = [prompt + target for prompt, target in zip(prompts, targets)]

    tokenized = tokenizer(merged, truncation=True, padding="max_length", max_length=MAX_LEN)
    prompt_tokenized = tokenizer(prompts, truncation=True, padding="max_length", max_length=MAX_LEN)

    labels = []
    for ids, attention_mask, prompt_mask in zip(
        tokenized["input_ids"],
        tokenized["attention_mask"],
        prompt_tokenized["attention_mask"],
    ):
        prompt_len = sum(prompt_mask)
        label = [-100] * len(ids)
        for index in range(prompt_len, len(ids)):
            if attention_mask[index] == 1:
                label[index] = ids[index]
        labels.append(label)
    tokenized["labels"] = labels
    return tokenized


train_tok = train_ds.map(tokenize_function, batched=True, remove_columns=train_ds.column_names)
eval_tok = eval_ds.map(tokenize_function, batched=True, remove_columns=eval_ds.column_names)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1 if USE_4BIT else 2,
    gradient_accumulation_steps=8 if USE_4BIT else 4,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.05,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Answer SFT LoRA saved to: {OUTPUT_DIR}")
