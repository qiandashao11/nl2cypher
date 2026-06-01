import os
import json
import torch
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# =============== 基础配置 ===============
MODEL_NAME = os.environ.get("BASE_MODEL", "microsoft/phi-3-mini-4k-instruct")
DATA_PATH  = os.environ.get("DATA_PATH", "dataset_en.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./lora_out")
MAX_LEN    = int(os.environ.get("MAX_LEN", "512"))
USE_4BIT   = os.environ.get("USE_4BIT", "1") == "1"   # 也可设 USE_8BIT=1

print(f"Base model: {MODEL_NAME}")
print(f"Data file : {DATA_PATH}")
print(f"4-bit     : {USE_4BIT}")

# =============== 加载数据集 ===============
# JSONL 的字段：{"instruction": "...", "output": "CYTHER ..."}
raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")

def build_sample(example):
    # chat 风格 prompt。你也可以自己改模板。
    prompt = (
        "You are a Cypher generator.\n"
        f"User: {example['instruction']}\n"
        "Cypher:\n"
        f"{example['output']}"
    )
    return {"text": prompt}

ds = raw_ds.map(build_sample)

# =============== Tokenizer & Model ===============
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True if USE_4BIT else False,
    load_in_8bit=False if USE_4BIT else True,  # 二选一：默认用 4bit
    device_map="auto",
    trust_remote_code=True,
)

# =============== 自动选择 LoRA 注入层 ===============
def pick_target_modules(m):
    names = set()
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            names.add(n.split(".")[-1])
    names = sorted(names)

    llama_like = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
    phi_like   = {"qkv_proj","o_proj","gate_up_proj","down_proj","fc1","fc2"}

    if llama_like & set(names):
        return sorted(list(llama_like & set(names)))
    if phi_like & set(names):
        return sorted(list(phi_like & set(names)))

    return "all-linear"  # 兜底：给所有 Linear 注入（显存稍多）

targets = pick_target_modules(model)
print("LoRA target_modules ->", targets)

# =============== LoRA 配置 ===============
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=targets,
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# =============== Tokenize ===============
def tokenize(batch):
    out = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )
    out["labels"] = out["input_ids"].copy()
    return out

tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

# =============== 训练参数 ===============
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    bf16=False,
    report_to="none",
)

# =============== Trainer ===============
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ LoRA finished. Saved to: {OUTPUT_DIR}")
