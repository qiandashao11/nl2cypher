# ==================== train_lora_llama3.py ====================
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
from huggingface_hub import login

# -------------------- 登录 Hugging Face --------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("✅ Successfully logged in to Hugging Face")
    except Exception as e:
        print("⚠️ Login failed:", e)
else:
    print("⚠️ No HF_TOKEN provided")

# -------------------- 常量配置 --------------------
MODEL_NAME = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DATA_PATH  = os.environ.get("DATA_PATH", "nl2cypher_train_en_1000.jsonl")
LORA_R     = int(os.environ.get("LORA_R", "32"))   # LoRA rank（默认32）
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"./lora_out_llama3_8b_r{LORA_R}")
MAX_LEN    = int(os.environ.get("MAX_LEN", "2048"))
USE_4BIT   = os.environ.get("USE_4BIT", "1") == "1"

LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"
IS_LOCAL_PATH    = os.path.isdir(MODEL_NAME)

print(f"Base model : {MODEL_NAME} ({'local dir' if IS_LOCAL_PATH else 'remote id'})")
print(f"Data file  : {DATA_PATH}")
print(f"4-bit quant: {USE_4BIT}")
print(f"Local-only : {LOCAL_FILES_ONLY}")
print(f"Max length : {MAX_LEN}")
print(f"LoRA rank  : {LORA_R}")
print(f"Output dir : {OUTPUT_DIR}")

# -------------------- 数据集 --------------------
raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")

# 划分验证集（10%）
splits = raw_ds.train_test_split(test_size=0.1, seed=42)
train_raw = splits["train"]
eval_raw  = splits["test"]

def build_prompt(example):
    q = example.get("question_en", "")
    c = example.get("cypher", "")
    p = example.get("params", None)

    prompt = "You are a Cypher generator.\n"
    prompt += f"User: {q}\nCypher:\n"
    target = c if c is not None else ""
    if p:
        target += "\nParams:\n" + json.dumps(p, ensure_ascii=False, separators=(",", ":"))
    return {"prompt": prompt, "target": target}

train_ds = train_raw.map(build_prompt)
eval_ds  = eval_raw.map(build_prompt)

# -------------------- 4-bit 量化 --------------------
bnb_config = None
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

# -------------------- Tokenizer --------------------
tok_kwargs = {
    "use_fast": True,
    "trust_remote_code": False,
    "local_files_only": LOCAL_FILES_ONLY or IS_LOCAL_PATH,
}
if not (LOCAL_FILES_ONLY or IS_LOCAL_PATH) and HF_TOKEN:
    tok_kwargs["token"] = HF_TOKEN

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **tok_kwargs)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -------------------- Config --------------------
cfg_kwargs = {
    "local_files_only": LOCAL_FILES_ONLY or IS_LOCAL_PATH,
}
if not (LOCAL_FILES_ONLY or IS_LOCAL_PATH) and HF_TOKEN:
    cfg_kwargs["token"] = HF_TOKEN

config = AutoConfig.from_pretrained(MODEL_NAME, **cfg_kwargs)
config.use_cache = False  

ATTN_IMPL = os.environ.get("ATTN_IMPL")  
if ATTN_IMPL:
    config.attn_implementation = ATTN_IMPL
ROPE_SCALE = os.environ.get("ROPE_SCALE")
if ROPE_SCALE:
    config.rope_scaling = {"type": "dynamic", "factor": float(ROPE_SCALE)}

model_kwargs = {
    "config": config,
    "device_map": "auto",
    "trust_remote_code": False,
    "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    "quantization_config": bnb_config if USE_4BIT else None,
    "local_files_only": LOCAL_FILES_ONLY or IS_LOCAL_PATH,
}
if not (LOCAL_FILES_ONLY or IS_LOCAL_PATH) and HF_TOKEN:
    model_kwargs["token"] = HF_TOKEN

def _try_load():
    try:
        return AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    except Exception as e:
        if not (LOCAL_FILES_ONLY or IS_LOCAL_PATH):
            print("⚠️ Loading fallback:", repr(e))
            model_kwargs["local_files_only"] = True
            model_kwargs.pop("token", None)
            return AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
        raise

model = _try_load()

# -------------------- LoRA 目标模块 --------------------
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
    r=LORA_R,              # 使用变量控制 rank
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

# -------------------- Tokenization --------------------
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

# -------------------- 训练参数 --------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1 if USE_4BIT else 2,
    gradient_accumulation_steps=8 if USE_4BIT else 4,
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

# -------------------- 保存 --------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ LoRA training finished. Model saved to: {OUTPUT_DIR}")
# ==================== end ====================
