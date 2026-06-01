# infer.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

# ====== 路径配置 ======
BASE_MODEL = "microsoft/phi-3-mini-4k-instruct"      # 与训练时相同
LORA_PATH  = "lora_out"        # 你训练输出的目录

# ====== 加载模型 ======
print("🚀 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

# ====== 推理函数 ======
def generate_cypher(question: str, schema: str = None, max_new_tokens: int = 512):
    """
    输入英文自然语言问题，返回模型生成的 Cypher 和 Params。
    """
    prompt = "You are a Cypher generator."
    if schema:
        prompt += f"\nSchema:\n{schema.strip()}"
    prompt += f"\nUser:\n{question.strip()}\nCypher:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Cypher:" in result:
        result = result.split("Cypher:", 1)[1].strip()
    return result

# ====== 示例 ======
if __name__ == "__main__":
    q = "Which genes are associated with the MeSH term 'cold'?"
    output = generate_cypher(q)
    print("\nModel output:")
    print(output)
