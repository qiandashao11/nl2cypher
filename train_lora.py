# nl2cypher/train_lora.py
import os
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

# =============== Basic configuration ===============
MODEL_NAME = os.environ.get("BASE_MODEL", "microsoft/phi-3-mini-4k-instruct")
DATA_PATH  = os.environ.get("DATA_PATH", "dataset_en.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./lora_out")
MAX_LEN    = int(os.environ.get("MAX_LEN", "512"))
USE_4BIT   = os.environ.get("USE_4BIT", "1") == "1"

print(f"Base model : {MODEL_NAME}")
print(f"Data file  : {DATA_PATH}")
print(f"4-bit quant: {USE_4BIT}")

# =============== Load dataset ===============
# Format: {"instruction": "...", "output": "..."}
raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")

def build_prompt(example):
    prompt = (
        "You are a Cypher generator.\n"
        f"User: {example['instruction']}\n"
        "Cypher:\n"
        f"{example['output']}"
    )
    return {"text": prompt}

ds = raw_ds.map(build_prompt)

# =============== Tokenizer & Base Model ===============
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True if USE_4BIT else False,
    load_in_8bit=False if USE_4BIT else True,
    device_map="auto",
    trust_remote_code=True,
)

# =============== Automatically select LoRA target layers ===============
def pick_target_modules(m):
    names = set()
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            names.add(n.split(".")[-1])
    llama_like = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
    phi_like   = {"qkv_proj","o_proj","gate_up_proj","down_proj","fc1","fc2"}
    if llama_like & names:
        return sorted(list(llama_like & names))
    if phi_like & names:
        return sorted(list(phi_like & names))
    return "all-linear"

targets = pick_target_modules(model)
print("LoRA target_modules ->", targets)

# =============== LoRA configuration ===============
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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

# =============== Training arguments ===============
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

# =============== Save model ===============
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ LoRA finished. Saved to: {OUTPUT_DIR}")
