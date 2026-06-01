# ==================== infer_lora.py ====================
import os, json, re, torch, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login

# ---------- Hugging Face Token ----------
HF_TOKEN = os.environ.get("HF_TOKEN")


# ---------- Prompt construction ----------
def build_prompt_with_chat_template(tok, question: str, params: dict | None = None) -> str:
    system = (
        "You are a Cypher generator for a Neo4j graph.\n"
        "Return ONLY the Cypher query. No explanations, no prose, no markdown, no code fences, no leading labels.\n"
        "The first token MUST be one of: MATCH, CREATE, MERGE, RETURN.\n"
        "Hard constraints:"
        "- NEVER generate UNION or UNION ALL."
        "- NEVER generate SKIP unless the user explicitly asks."
        "- NEVER generate subqueries, CALL {...}, WITH {...}, or SELECT."
        "- NEVER wrap queries in parentheses."
        "- NEVER repeat the query."
        "- NEVER add any additional statements after RETURN."
        "- NEVER guess or invent structure."
        "\n"
        "Schema:\n"
        "- Node labels: Gene, MeSH, Literature\n"
        "- Relationships:\n"
        "  * HAS_SOURCE: (Gene|MeSH) -> (Literature)   // directed to Literature\n"
        "  * CO_OCCURS:  Gene–Gene and Gene–MeSH       // undirected semantics; match with -[:CO_OCCURS]-\n"
        "- Common properties:\n"
        "  * Gene: ENTITY, `Closeness.centrality`\n"
        "  * MeSH: entity\n"
        "  * Literature: Title, Year, Journal, PMID, DOIlink\n"
        "\n"
        "Conventions:\n"
        "- Use correct HAS_SOURCE direction: (g:Gene)-[:HAS_SOURCE]->(l:Literature) and (m:MeSH)-[:HAS_SOURCE]->(l).\n"
        "- Do NOT add ORDER BY unless the natural language explicitly requests sorting.\n"
        "- Do not add comments or parameter descriptions; output a single Cypher query only.\n"
    )

    user = f"{question}"
    if params:
        user += "\nParameters: " + json.dumps(params, ensure_ascii=False, separators=(',', ':'))
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ---------- Main function ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--lora_dir", default="./lora_out_llama3_8b3")
    ap.add_argument("--question", required=True, help="English natural-language question")
    ap.add_argument("--params_json", default=None, help='optional parameters JSON, e.g. \'{"gene":"BRCA1"}\'')
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling (off by default)")
    args = ap.parse_args()

    # ---------- Hugging Face login ----------
    if HF_TOKEN:
        try:
            login(token=HF_TOKEN, add_to_git_credential=False)
            print("✅ Logged in to Hugging Face")
        except Exception as e:
            print("⚠️ Login failed:", e)

    # ---------- Load tokenizer ----------
    tok = AutoTokenizer.from_pretrained(
        args.base_model,
        token=HF_TOKEN if HF_TOKEN else None,
        use_fast=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---------- Load model ----------
    print(f"🔹 Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=HF_TOKEN if HF_TOKEN else None,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

    print(f"🔹 Loading LoRA adapter from: {args.lora_dir}")
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.eval()

    # ---------- Build prompt ----------
    hint_params = json.loads(args.params_json) if args.params_json else None
    prompt = build_prompt_with_chat_template(tok, args.question, hint_params)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    # ---------- Inference ----------
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if args.do_sample:
        gen_kwargs.update(temperature=max(args.temperature, 1e-6), top_p=args.top_p)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    # ---------- Decode + post-process ----------
    text = tok.decode(out[0], skip_special_tokens=True)

    # Print raw output for debugging (can be commented out)
    # print("=== RAW OUTPUT ===")
    # print(repr(text))

    # Slice from the first MATCH/CREATE/MERGE/RETURN to the end
    start = re.search(r'(?m)^(MATCH|CREATE|MERGE|RETURN)\b', text)
    cypher = text[start.start():].strip() if start else text.strip()

    # Remove possible markdown code fences
    cypher = re.sub(r'^\s*```(?:cypher)?\s*', '', cypher)
    cypher = re.sub(r'\s*```.*$', '', cypher)

    print("\n=== Cypher ===\n" + cypher)


if __name__ == "__main__":
    main()
# ==================== end ====================
