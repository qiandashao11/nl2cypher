# ==================== infer_llama31_base.py ====================
import os, json, re, torch, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

def build_prompt_with_chat_template(tok, question: str, params: dict | None = None) -> str:
    system = (
        "You are a Cypher generator for a Neo4j graph.\n"
        "Return ONLY the Cypher query. No explanations, no prose, no markdown.\n"

        "Schema:\n"
        "- Node labels: Gene, MeSH, Literature\n"
        "- Relationships: CO_OCCURS, HAS_SOURCE\n"
        "- Common properties:\n"
        "  Gene: symbol, `Closeness.centrality`\n"
        "  MeSH: term\n"
        "  Literature: Title, Year, Journal, PMID, PMIDs, DOIlink\n"
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
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--question",   required=True, help="英文自然语言问题")
    ap.add_argument("--params_json", default=None, help='可选 JSON，比如 {\"gene\":\"BRCA1\"}')
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--do_sample", action="store_true", help="是否启用采样（默认关闭）")
    args = ap.parse_args()

    # ---- HF 登录（如不需要可置空或用环境变量）----
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        try:
            login(token=HF_TOKEN, add_to_git_credential=False)
            print("✅ 已登录 Hugging Face")
        except Exception as e:
            print("⚠️ 登录失败：", e)

    # ---- 加载 tokenizer 与基座模型（无 LoRA）----
    tok = AutoTokenizer.from_pretrained(
        args.base_model,
        token=HF_TOKEN if HF_TOKEN else None,
        use_fast=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"🔹 Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=HF_TOKEN if HF_TOKEN else None,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    model.eval()

    # ---- 构造 prompt ----
    hint_params = json.loads(args.params_json) if args.params_json else None
    prompt = build_prompt_with_chat_template(tok, args.question, hint_params)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    # ---- 生成 ----
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tok.eos_token_id,
        )

    # ---- 解码：保留完整“新生成”文本，不做删除 ----
    generated_ids = out[0]
    # 只解码新生成的 tokens，避免把 prompt 也一起打印出来
    new_token_ids = generated_ids[inputs["input_ids"].shape[1]:]
    text = tok.decode(new_token_ids, skip_special_tokens=True)

    print("\n=== Raw Model Output ===\n" + text)

    # ---- 额外：尝试检测并打印第一条 Cypher（不删除其它内容）----
    m = re.search(r"(?mi)^(MATCH|CREATE|MERGE|RETURN)\\b.*", text)
    if m:
        cypher_line = m.group(0).strip()
        print("\n=== Detected Cypher (first line) ===\n" + cypher_line)

if __name__ == "__main__":
    main()
# ==================== end ====================
