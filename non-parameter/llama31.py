# ==================== train_lora_llama3.py (chat-template version) ====================
import os, json, torch
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    Trainer, TrainingArguments, BitsAndBytesConfig, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login

# -------- 登录（建议改为环境变量）--------
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("✅ 已登录 Hugging Face")
    except Exception as e:
        print("⚠️ 登录失败：", e)
else:
    print("ℹ️ 未设置 HF_TOKEN，将依赖本地缓存/已登录状态。")

# -------- 基础配置 --------
MODEL_NAME = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DATA_PATH  = os.environ.get("DATA_PATH", "phase3_multihop_training/train.chat.phase3_multihop.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "phase3_multihop_training/lora_out_llama3_8b_multihop")
MAX_LEN    = int(os.environ.get("MAX_LEN", "2048"))
USE_4BIT   = os.environ.get("USE_4BIT", "1") == "1"
LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"
IS_LOCAL_PATH    = os.path.isdir(MODEL_NAME)

print(f"Base model : {MODEL_NAME} ({'local dir' if IS_LOCAL_PATH else 'remote id'})")
print(f"Data file  : {DATA_PATH}")
print(f"4-bit quant: {USE_4BIT}")
print(f"Local-only : {LOCAL_FILES_ONLY}")
print(f"Max length : {MAX_LEN}")

# -------- 数据集加载 --------
raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")
splits  = raw_ds.train_test_split(test_size=0.1, seed=42)
train_raw, eval_raw = splits["train"], splits["test"]

# -------- Chat prompt 定义（与 infer 一致）--------
SCHEMA_SYSTEM_PROMPT = """You are a Cypher generator for a Neo4j graph.

Your job:
- Take a natural language question about the graph.
- Return ONLY ONE Cypher query that answers the question.

Output rules:
- The FIRST token MUST be one of: MATCH, CREATE, MERGE, RETURN.
- Return ONLY the Cypher query. No explanations, no prose, no markdown fences, no comments.
- Do NOT print parameters or JSON. Output pure Cypher only.
- Prefer a single MATCH … RETURN pattern when possible.

Graph schema (Phase 1):

Node labels and properties:
- Gene
  - ENTITY: main identifier of the gene (e.g. gene symbol)
- MeSH
  - ENTITY: main identifier of the MeSH term
- Literature
  - PMID: primary identifier of the paper
  - Title: title of the paper
  - Year: publication year (integer)
  - Journal: journal name

Relationship types:
- (Gene)-[:HAS_SOURCE]->(Literature)
- (MeSH)-[:HAS_SOURCE]->(Literature)
- (Gene)-[:CO_OCCURS]-(Gene)
- (Gene)-[:CO_OCCURS]-(MeSH)

Conventions:
- Use correct HAS_SOURCE direction:
  (g:Gene)-[:HAS_SOURCE]->(l:Literature)
  (m:MeSH)-[:HAS_SOURCE]->(l:Literature)
- For co-occurrence, always use an undirected pattern:
  (a)-[:CO_OCCURS]-(b)
- When looking up specific genes or MeSH terms, match by ENTITY, for example:
  (g:Gene {ENTITY: "TP53"})
  (m:MeSH {ENTITY: "Breast Neoplasms"})
- When returning nodes, RETURN their key properties, not the whole node:
  Gene  -> g.ENTITY AS gene_entity
  MeSH  -> m.ENTITY AS mesh_entity
  Paper -> l.PMID  AS pmid
- It is allowed to use Title, Year, Journal and PMID in WHERE filters,
  but the RETURN clause should only contain ENTITY, PMID or simple scalars
  such as counts or journal names.

Hard constraints:
- NEVER generate UNION or UNION ALL.
- NEVER generate SKIP or LIMIT unless the user explicitly asks for pagination or a specific number of results.
- NEVER use subqueries, CALL { ... } or APOC procedures.
- Do NOT use regular expressions (=~) unless the user explicitly asks for regex matching.
"""
def _extract_qa(example):
    if "messages" in example and isinstance(example["messages"], list):
        user_msg = next((m.get("content","") for m in example["messages"] if m.get("role")=="user"), "")
        asst_msg = next((m.get("content","") for m in example["messages"] if m.get("role")=="assistant"), "")
        return user_msg, asst_msg, None
    return example.get("question_en",""), example.get("cypher",""), example.get("params", None)

def build_prompt(example):
    q, c, p = _extract_qa(example)
    target = (c or "").strip()
    user_text = q or ""
    messages = [
        {"role": "system", "content": SCHEMA_SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
        {"role": "assistant", "content": target},
    ]
    return {"messages": json.dumps(messages, ensure_ascii=False), "target": target}

train_ds = train_raw.map(build_prompt)
eval_ds  = eval_raw.map(build_prompt)

# -------- 4-bit 量化（可选）--------
bnb_config = None
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

# -------- Tokenizer --------
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

# -------- Config & Model --------
cfg_kwargs = {"local_files_only": LOCAL_FILES_ONLY or IS_LOCAL_PATH}
if not (LOCAL_FILES_ONLY or IS_LOCAL_PATH) and HF_TOKEN:
    cfg_kwargs["token"] = HF_TOKEN
config = AutoConfig.from_pretrained(MODEL_NAME, **cfg_kwargs)
config.use_cache = False

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
            print("⚠️ 远程加载失败，回退 local_files_only=True。错误：", repr(e))
            model_kwargs["local_files_only"] = True
            model_kwargs.pop("token", None)
            return AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
        raise

model = _try_load()

# -------- LoRA 设置 --------
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
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=targets, bias="none", task_type="CAUSAL_LM",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# -------- Tokenize（chat模板版本）--------
def tokenize_function(batch):
    msgs_list = [json.loads(x) for x in batch["messages"]]
    prompts = [
        tokenizer.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
        for m in msgs_list
    ]
    targets = batch["target"]
    merged_texts = [p + (t or "") for p, t in zip(prompts, targets)]
    out = tokenizer(merged_texts, truncation=True, padding="max_length", max_length=MAX_LEN)
    prompt_tok = tokenizer(prompts, truncation=True, padding="max_length", max_length=MAX_LEN)
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

# -------- 训练参数 --------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1 if USE_4BIT else 2,
    gradient_accumulation_steps=8 if USE_4BIT else 4,
    num_train_epochs=2,
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
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ LoRA finished. Saved to: {OUTPUT_DIR}")
# ==================== end ====================
