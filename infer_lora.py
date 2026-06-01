#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer Cypher (Neo4j) with a base model + LoRA adapter.

Usage:
  ADAPTER_DIR=./lora_out python infer_lora.py -q "list genes co-occurring with <GENE>"
"""

import os
import re
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ----------------------------
# 1) Basic config (edit if needed)
# ----------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "microsoft/phi-3-mini-4k-instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "./lora_out")  # LoRA adapter folder (should contain adapter_config.json)


def ensure_adapter(path: str):
    """Ensure the LoRA adapter folder exists and contains adapter_config.json."""
    if not os.path.isdir(path):
        print(f"[ERR] LoRA folder not found: {path}")
        print("      Please set ADAPTER_DIR to the directory that contains 'adapter_config.json'.")
        sys.exit(1)
    cfg = os.path.join(path, "adapter_config.json")
    if not os.path.exists(cfg):
        print(f"[ERR] 'adapter_config.json' not found in: {path}")
        sys.exit(1)


# ----------------------------
# 2) System prompt (schema + rules + examples) — customize here
# ----------------------------
PROMPT_SYSTEM = """You are a Cypher generator for Neo4j.
Only output a single valid READ-ONLY Cypher query.
Never output SQL or explanations.
"""


# Additional strict policy to avoid SQL and non-Cypher output
PROMPT_POLICY = """
Output requirements:
- Only produce a single fenced code block starting with ```cypher and nothing else.
- Never output SQL or any language other than Cypher.
- Do not add explanations or comments outside the fenced block.
"""


def build_prompt(user_question: str, strong: bool = False) -> str:
    """Compose the final prompt. Toggle 'strong' for stricter constraints (used on retry)."""
    system = PROMPT_SYSTEM + PROMPT_POLICY
    if strong:
        system += "\nNEVER output SQL. If you output SQL, that is a hard error. Only Cypher is allowed.\n"
    prompt = f"""{system}

User question:
{user_question}

Return only the Cypher in a single fenced block:
```cypher
"""
    return prompt


SQL_PATTERN = re.compile(r"\\bSELECT\\b|\\bFROM\\b|\\bWHERE\\b", re.IGNORECASE)


def extract_cypher(text: str) -> str:
    """Extract ```cypher ... ``` contents; otherwise return stripped text."""
    m = re.search(r"```cypher\\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


@torch.inference_mode()
def generate(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    """Generate Cypher with one retry if SQL is detected."""
    # First try
    prompt1 = build_prompt(question, strong=False)
    inputs = tokenizer(prompt1, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,  # compatible with DynamicCache in newer transformers
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    cypher1 = extract_cypher(text)

    if SQL_PATTERN.search(cypher1):
        # Retry with stronger constraint
        prompt2 = build_prompt(question, strong=True)
        inputs2 = tokenizer(prompt2, return_tensors="pt").to(model.device)
        out2 = model.generate(
            **inputs2,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
        text2 = tokenizer.decode(out2[0], skip_special_tokens=True)
        cypher2 = extract_cypher(text2)
        return cypher2

    return cypher1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q", "--question",
        type=str,
        default="List MeSH terms containing 'immune' (up to 10).",
        help="User natural language question",
    )
    args = parser.parse_args()

    ensure_adapter(ADAPTER_DIR)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    cypher = generate(model, tokenizer, args.question, max_new_tokens=256)
    print("========== Cypher ==========")
    print("```cypher")
    print(cypher)
    print("```")


if __name__ == "__main__":
    main()