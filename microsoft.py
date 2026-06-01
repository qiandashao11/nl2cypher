# nl2cypher_fixed_schema.py
"""
自然语言 -> Cypher 查询生成器 (固定 Schema 版本)
使用 HuggingFace 模型: microsoft/phi-3-mini-4k-instruct
"""

import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = "microsoft/phi-3-mini-4k-instruct"

print(f"Loading model: {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("Model loaded on:", next(model.parameters()).device)

# 固定的 Schema 说明
SCHEMA = """
Schema:
- Node labels: Literature, Gene, MeSH
- Relationship types:
  * (Literature)<-[:HAS_SOURCE]-(Gene)
  * (MeSH)<-[:HAS_SOURCE]-(Literature)
  * (Gene)-[:CO_OCCURS {Expansion:int}]->(Gene)
  * (Gene)-[:CO_OCCURS {Expansion:int}]->(MeSH)
  * (MeSH)-[:CO_OCCURS {Expansion:int}]->(Gene)
  * (MeSH)-[:CO_OCCURS {Expansion:int}]->(MeSH)
- Rules:
  * Only use these nodes and relationships.
  * Expansion belongs only to CO_OCCURS.
  * Queries must be read-only (MATCH / WHERE / RETURN / ORDER BY / LIMIT).
  * Output must be inside ```cypher ... ``` fenced code block.
"""

def build_prompt(question: str) -> str:
    return f"""
You are a Cypher expert. Convert the following user question into a read-only Cypher query.

{SCHEMA}

User question:
{question}

Output:
"""

def extract_cypher(text: str) -> str:
    match = re.search(r"```cypher(.*?)```", text, re.S)
    if match:
        return match.group(1).strip()
    return text.strip()

def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "Return up to 200 pairs of Literature and Gene where Gene also has CO_OCCURS Expansion=1"

    prompt = build_prompt(question)
    outputs = generator(prompt, max_new_tokens=256, do_sample=False)
    raw_output = outputs[0]["generated_text"]

    print("=== Raw model output ===")
    print(raw_output)
    print("\n=== Extracted Cypher query ===")
    print(extract_cypher(raw_output))

if __name__ == "__main__":
    main()
