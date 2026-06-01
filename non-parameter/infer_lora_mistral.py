#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, re, torch, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login

HF_TOKEN = os.environ.get("HF_TOKEN")

# ---------- Stronger prompt construction (consistent with the Llama version) ----------
def build_prompt_with_chat_template(tok, question: str, params: dict | None = None) -> str:
    system = (
        "You are a Cypher generator for a Neo4j graph.\n"
        "Return ONLY the Cypher query. No explanations, no prose, no markdown.\n"
        "The first token MUST be one of: MATCH, CREATE, MERGE, RETURN.\n"
        "Schema:\n"
        "- Node labels: Gene, MeSH, Literature\n"
        "- Relationships: CO_OCCURS, HAS_SOURCE\n"
        "- Common properties:\n"
        "  Gene: entity, `Closeness.centrality`\n"
        "  MeSH: entity\n"
        "  Literature: Title, Year, Journal, PMID, DOIlink\n"
        "  Edge(CO_OCCURS): Strength, RawOccurrence, Wscore\n"
        "Quote dotted property names with backticks.\n"
    )
    user = f"{question}"
    if params:
        user += "\nParameters: " + json.dumps(params, ensure_ascii=False, separators=(',', ':'))
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    # Mistral models also support chat_template
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ---------- Main function ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--lora_dir",   default="./lora_out_mistral_7b")
    ap.add_argument("--question",   required=True, help="English natural-language question")
    ap.add_argument("--params_json", default=None, help='optional parameters JSON, e.g. \'{"gene":"BRCA1"}\'')
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling (off by default)")
    args = ap.parse_args()

    # ---------- Hugging Face login (from environment variable) ----------
    if HF_TOKEN:
        try:
            login(token=HF_TOKEN, add_to_git_credential=False)
            print("✅ Logged in to Hugging Face")
        except Exception as e:
            print("⚠️ Login failed:", e)

    # ---------- Load tokenizer ----------
    tok = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---------- Load base model + LoRA ----------
    print(f"🔹 Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    # Try SDPA acceleration (ignore failures if any)
    try:
        base.config.attn_implementation = "sdpa"
    except Exception:
        pass

    print(f"🔹 Loading LoRA adapter from: {args.lora_dir}")
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.eval()

    # ---------- Build prompt ----------
    hint_params = json.loads(args.params_json) if args.params_json else None
    prompt = build_prompt_with_chat_template(tok, args.question, hint_params)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    # ---------- Inference ----------
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            num_beams=1 if args.do_sample else 1,
            repetition_penalty=1.05,
        )

    # ---------- Decode + post-process (same style as the Llama version) ----------
    text = tok.decode(out[0], skip_special_tokens=True)

    # Keep only the part after "Cypher:" if present
    if "Cypher:" in text:
        text = text.split("Cypher:", 1)[-1]

    # Capture the first line that starts with MATCH/CREATE/MERGE/RETURN
    m = re.search(r'(?m)^(MATCH|CREATE|MERGE|RETURN)\b.*', text)
    cypher = m.group(0).strip() if m else text.strip().splitlines()[0].strip() if text.strip() else "RETURN 1"

    print("\n=== Cypher ===\n" + cypher)


if __name__ == "__main__":
    main()
