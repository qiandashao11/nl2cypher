# ==================== train_lora_llama3_text2cypher2025_schema_localmodel.py ====================
import os
import json
import torch
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -------------------- ✅ No login or token required --------------------
# Disable automatic Hugging Face Hub login
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)

# -------------------- Model and data settings --------------------
MODEL_NAME = os.environ.get("BASE_MODEL", os.path.expanduser("~/models/llama3_8b_instruct"))
DATASET_ID = os.environ.get("DATASET_ID", "neo4j/text2cypher-2025v1")
LORA_R     = int(os.environ.get("LORA_R", "32"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"./lora_out_llama3_local_r{LORA_R}_t2c2025_schema")

MAX_LEN    = int(os.environ.get("MAX_LEN", "2048"))
USE_4BIT   = os.environ.get("USE_4BIT", "1") == "1"

# Model is a local path; dataset loading may use the network
LOCAL_FILES_ONLY = False
IS_LOCAL_PATH    = os.path.isdir(MODEL_NAME)
if not IS_LOCAL_PATH:
    raise FileNotFoundError(f"❌ Model directory does not exist: {MODEL_NAME}")

print(f"Base model : {MODEL_NAME} (local dir)")
print(f"Dataset    : {DATASET_ID}")
print(f"4-bit quant: {USE_4BIT}")
print(f"Local-only : {LOCAL_FILES_ONLY}")
print(f"Max length : {MAX_LEN}")
print(f"LoRA rank  : {LORA_R}")
print(f"Output dir : {OUTPUT_DIR}")

# -------------------- Dataset: load online --------------------
def _load_splits():
    try:
        train_raw = load_dataset(DATASET_ID, split="train")
        eval_raw  = load_dataset(DATASET_ID, split="test")
        print("✅ Loaded official splits: train/test")
        return train_raw, eval_raw
    except Exception as e:
        print("ℹ️ Fallback to default split:", repr(e))
        raw = load_dataset(DATASET_ID, split="default")
        splits = raw.train_test_split(test_size=0.1, seed=42)
        return splits["train"], splits["test"]

train_raw, eval_raw = _load_splits()

# -------------------- Build prompt --------------------
def build_prompt(example):
    q = example.get("question", "") or example.get("question_en", "")
    c = example.get("cypher", "") or ""
    schema_text = (example.get("schema") or "").strip()
    prompt = "You are a Cypher generator.\n"
    prompt += "Schema:\n" + schema_text + "\n"
    prompt += f"User: {q}\nCypher:\n"
    target = c
    return {"prompt": prompt, "target": target}

train_ds = train_raw.map(build_prompt)
eval_ds  = eval_raw.map(build_prompt)

# -------------------- 4-bit quantization --------------------
bnb_config = None
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

# -------------------- Load local tokenizer --------------------
tok_kwargs = {
    "use_fast": True,
    "trust_remote_code": False,
    "local_files_only": True,   # ✅ local loading only
}
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **tok_kwargs)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -------------------- Load local config --------------------
config = AutoConfig.from_pretrained(MODEL_NAME, local_files_only=True)
config.use_cache = False

ATTN_IMPL = os.environ.get("ATTN_IMPL")
if ATTN_IMPL:
    config.attn_implementation = ATTN_IMPL
ROPE_SCALE = os.environ.get("ROPE_SCALE")
if ROPE_SCALE:
    config.rope_scaling = {"type": "dynamic", "factor": float(ROPE_SCALE)}

# -------------------- Load local model --------------------
model_kwargs = {
    "config": config,
    "device_map": "auto",
    "trust_remote_code": False,
    "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    "quantization_config": bnb_config if USE_4BIT else None,
    "local_files_only": True,   # ✅ force local loading
}
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

# -------------------- LoRA adaptation --------------------
def pick_target_modules(m):
    names = set()
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            names.add(n.split(".")[-1])
    llama_like = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
    if llama_like & names:
        return sorted(list(llama_like & names))
    return ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

targets = pick_target_modules(model)
print("LoRA target_modules ->", targets)

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=32,
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

# -------------------- Tokenize --------------------
def tokenize_function(batch):
    inputs = [p + t for p, t in zip(batch["prompt"], batch["target"])]
    out = tokenizer(inputs, truncation=True, padding="max_length", max_length=MAX_LEN)
    prompt_tok = tokenizer(batch["prompt"], truncation=True, padding="max_length", max_length=MAX_LEN)
    labels = []
    for i in range(len(out["input_ids"])):
        ids   = out["input_ids"][i]
        amask = out["attention_mask"][i]
        plen  = sum(prompt_tok["attention_mask"][i])
        lab = [-100] * len(ids)
        for j in range(plen, len(ids)):
            if amask[j] == 1:
                lab[j] = ids[j]
        labels.append(lab)
    out["labels"] = labels
    return out

train_tok = train_ds.map(tokenize_function, batched=True, remove_columns=train_ds.column_names)
eval_tok  = eval_ds.map(tokenize_function,  batched=True, remove_columns=eval_ds.column_names)

# -------------------- Training arguments --------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8 if USE_4BIT else 16,
    num_train_epochs=3,
    learning_rate=2e-4,
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
    args=args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

# -------------------- Save --------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ LoRA training finished. Model saved to: {OUTPUT_DIR}")
# ==================== end ====================
