# nl2cypher/infer_lora_advanced.py
import os, sys, json, pathlib, argparse
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DEF_BASE   = "microsoft/phi-3-mini-4k-instruct"
DEF_ADAPT  = "./lora_out"          # LoRA 适配器目录
DEF_MERGED = None                  # 若已merge，可传该目录
STOP_HINTS = ["\n\n", "\nUser:", "\nAssistant:", "```"]

def load_text(path: Optional[str]) -> str:
    if not path: return ""
    p = pathlib.Path(path)
    if not p.exists(): raise FileNotFoundError(f"Schema file not found: {p}")
    return p.read_text(encoding="utf-8").strip()

def build_prompt(query: str, style: str, schema_text: str = "") -> str:
    """style: 'train' | 'plain' | 'minimal'"""
    sys_prefix = "You are a Cypher generator."
    schema_block = f"\nSchema/Notes:\n{schema_text}\n" if schema_text else "\n"
    if style == "train":
        instruction = ("Translate the following natural language query into a Cypher statement.\n"
                       f"{query}")
        return f"{sys_prefix}{schema_block}User: {instruction}\nCypher:\n"
    elif style == "plain":
        return f"{sys_prefix}{schema_block}User: {query}\nCypher:\n"
    else:  # minimal
        return f"User: {query}\nCypher:\n"

def postprocess(text: str) -> str:
    """抓取 Cypher 段，尽量截断到合适位置。"""
    # 取 'Cypher:\n' 之后
    if "Cypher:\n" in text:
        text = text.split("Cypher:\n", 1)[-1]
    # 首选；如果出现分号，取第一条语句
    if ";" in text:
        first = text.split(";", 1)[0] + ";"
        return first.strip()
    # 次选：遇到提示语或空行截断
    for s in STOP_HINTS:
        if s in text:
            text = text.split(s, 1)[0]
    return text.strip()

def generate_one(tok, model, prompt: str, max_new_tokens=200,
                 temperature=0.0, top_p=1.0, repetition_penalty=1.0) -> str:
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    return postprocess(text)

def load_model(base: Optional[str], adapter: Optional[str], merged: Optional[str],
               device: str = "auto"):
    """
    三种加载方式：
    1) merged!=None  -> 直接加载合并后的完整模型
    2) adapter!=None -> 加载 base + LoRA 适配器
    3) 仅 base       -> 直接用底模
    """
    target = merged or base or DEF_BASE
    print(f"[Load] tokenizer/model from: {target}")
    tok = AutoTokenizer.from_pretrained(target, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        target,
        device_map=device,
        trust_remote_code=True,
    )

    if merged:
        print("[Load] merged full model ✔")
        return tok, model

    if adapter:
        print(f"[Load] attach LoRA adapter from: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    else:
        print("[Load] base model only (no adapter)")

    return tok, model

def run_single(args, tok, model, schema_text):
    prompt = build_prompt(args.query, args.style, schema_text=schema_text)
    cypher = generate_one(
        tok, model, prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(cypher + "\n")
        print(f"[Saved] {args.out}")
    print("\n=== Query ===\n" + args.query)
    print("=== Generated Cypher ===\n" + cypher)

def run_batch(args, tok, model, schema_text):
    src = pathlib.Path(args.infile)
    if not src.exists():
        raise FileNotFoundError(src)
    out_path = pathlib.Path(args.out or "infer_outputs.jsonl")
    total = 0
    with open(src, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            q = line.strip()
            if not q:
                continue
            prompt = build_prompt(q, args.style, schema_text=schema_text)
            cypher = generate_one(
                tok, model, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            fout.write(json.dumps({"question": q, "cypher": cypher}, ensure_ascii=False) + "\n")
            total += 1
    print(f"[Batch] wrote {total} lines to {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Advanced NL→Cypher inference with LoRA.")
    ap.add_argument("--base", default=os.environ.get("BASE_MODEL", DEF_BASE),
                    help="Base HF model id or path.")
    ap.add_argument("--adapter", default=os.environ.get("ADAPTER", DEF_ADAPT),
                    help="PEFT LoRA adapter directory. Leave empty if using merged.")
    ap.add_argument("--merged", default=os.environ.get("MERGED_MODEL", DEF_MERGED),
                    help="Merged full model directory (takes precedence over adapter).")
    ap.add_argument("--device", default="auto", help="'auto' | 'cpu' | 'cuda'")
    ap.add_argument("--style", choices=["train", "plain", "minimal"], default="train",
                    help="Prompt style (train=与训练一致，推荐).")
    ap.add_argument("--schema-file", default=None, help="Optional schema/notes file to prepend.")
    ap.add_argument("--q", "--query", dest="query", default=None, help="Single query text.")
    ap.add_argument("--infile", default=None, help="Batch file: one question per line.")
    ap.add_argument("--out", default=None, help="Output path (for single .txt or batch .jsonl).")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    args = ap.parse_args()

    schema_text = load_text(args.schema_file)
    tok, model = load_model(args.base, args.adapter if not args.merged else None,
                            args.merged, device=args.device)

    if args.query and args.infile:
        print("Use either --q or --infile, not both.", file=sys.stderr)
        sys.exit(2)

    if args.query:
        run_single(args, tok, model, schema_text)
    elif args.infile:
        run_batch(args, tok, model, schema_text)
    else:
        # demo
        demo_qs = [
            "Show all literature (PMID) nodes co-occurring with any node that co-occurs with the gene 'MIR31HG'.",
            "Which genes are associated with the MeSH term 'Apoptosis'?",
        ]
        for q in demo_qs:
            prompt = build_prompt(q, args.style, schema_text=schema_text)
            cypher = generate_one(tok, model, prompt,
                                  max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature,
                                  top_p=args.top_p,
                                  repetition_penalty=args.repetition_penalty)
            print("\n=== Query ===\n" + q)
            print("=== Generated Cypher ===\n" + cypher)

if __name__ == "__main__":
    main()
