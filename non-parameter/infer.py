# infer.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

# ====== Path configuration ======
BASE_MODEL = "microsoft/phi-3-mini-4k-instruct"      # same as during training
LORA_PATH  = "lora_out"        # your training output directory

# ====== Load model ======
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

# ====== Inference function ======
def generate_cypher(question: str, schema: str = None, max_new_tokens: int = 512):
    """
    Input an English natural-language question and return the model-generated Cypher and Params.
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

# ====== Example ======
if __name__ == "__main__":
    q = "Which genes are associated with the MeSH term 'cold'?"
    output = generate_cypher(q)
    print("\nModel output:")
    print(output)
