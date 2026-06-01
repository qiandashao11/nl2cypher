# ==================== cypher_generator.py ====================
"""
Cypher query generator
Use Llama 3 to convert natural language into Neo4j Cypher queries
"""
import os, json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login

HF_TOKEN = os.environ.get("HF_TOKEN")


class CypherGenerator:
    """Cypher query generator"""
    
    def __init__(self, 
                 base_model="meta-llama/Llama-3.1-8B-Instruct",
                 lora_dir="/home/qianzi/nl2cypher/non-parameter/lora_out_llama3_8b2",
                 hf_token=None):
        """
        Initialize the generator
        
        Args:
            base_model: base model path
            lora_dir: LoRA adapter path
            hf_token: Hugging Face token
        """
        self.base_model = base_model
        self.lora_dir = lora_dir
        self.hf_token = hf_token or HF_TOKEN
        
        if self.hf_token:
            try:
                login(token=self.hf_token, add_to_git_credential=False)
                print("✅ Logged in to Hugging Face")
            except Exception as e:
                print(f"⚠️ Login failed: {e}")
        
        self._load_model()
    
    def _load_model(self):
        """Load model"""
        print(f"🔹 Loading tokenizer: {self.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            token=self.hf_token,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"🔹 Loading base model: {self.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            token=self.hf_token,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )
        
        print(f"🔹 Loading LoRA: {self.lora_dir}")
        self.model = PeftModel.from_pretrained(base_model, self.lora_dir)
        self.model.eval()
        print("✅ Model loaded")
    
    def _build_prompt(self, question: str, params: dict = None) -> str:
        """Build prompt"""
        system = (
            "You are a Cypher generator for a Neo4j graph.\n"
            "Return ONLY the Cypher query. No explanations, no prose, no markdown, no code fences.\n"
            "The first token MUST be one of: MATCH, CREATE, MERGE, RETURN.\n"
            "Hard constraints:\n"
            "- NEVER generate UNION or UNION ALL\n"
            "- NEVER generate SKIP unless explicitly asked\n"
            "- NEVER generate subqueries or CALL\n"
            "- NEVER wrap queries in parentheses\n"
            "- NEVER repeat the query\n"
            "\nSchema:\n"
            "- Node labels: Gene, MeSH, Literature\n"
            "- Relationships:\n"
            "  * HAS_SOURCE: (Gene|MeSH) -> (Literature)\n"
            "  * CO_OCCURS: Gene–Gene and Gene–MeSH (undirected)\n"
            "- Properties:\n"
            "  * Gene: entity, Closeness.centrality\n"
            "  * MeSH: entity\n"
            "  * Literature: Title, Year, Journal, PMID, DOIlink\n"
        )
        
        user = question
        if params:
            user += "\nParameters: " + json.dumps(params, ensure_ascii=False)
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def _clean_output(self, raw_text: str) -> str:
        """Clean generated output"""
        start = re.search(r'(?m)^(MATCH|CREATE|MERGE|RETURN)\b', raw_text)
        cypher = raw_text[start.start():].strip() if start else raw_text.strip()
        cypher = re.sub(r'^\s*```(?:cypher)?\s*', '', cypher)
        cypher = re.sub(r'\s*```.*$', '', cypher)
        return cypher
    
    def generate(self, 
                question: str, 
                params: dict = None,
                max_new_tokens: int = 256,
                temperature: float = 0.7,
                top_p: float = 1.0,
                do_sample: bool = False) -> str:
        """
        Generate a Cypher query
        
        Args:
            question: natural-language question
            params: optional parameter dictionary
            max_new_tokens: maximum number of generated tokens
            temperature: sampling temperature
            top_p: nucleus sampling parameter
            do_sample: whether to sample
            
        Returns:
            Cypher query string
        """
        prompt = self._build_prompt(question, params)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if do_sample:
            gen_kwargs.update({
                "temperature": max(temperature, 1e-6),
                "top_p": top_p
            })
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        raw_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._clean_output(raw_text)
    
    def __call__(self, question: str, **kwargs) -> str:
        """Make the object callable"""
        return self.generate(question, **kwargs)


def generate_cypher(question: str, **kwargs) -> str:
    """Convenience function: generate Cypher in one call"""
    gen = CypherGenerator(**kwargs)
    return gen.generate(question)


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--lora_dir", default="/home/qianzi/nl2cypher/non-parameter/lora_out_llama3_8b2")
    ap.add_argument("--question", required=True)
    ap.add_argument("--params_json", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--do_sample", action="store_true")
    args = ap.parse_args()
    
    params = json.loads(args.params_json) if args.params_json else None
    gen = CypherGenerator(args.base_model, args.lora_dir)
    cypher = gen.generate(args.question, params, args.max_new_tokens, args.temperature, do_sample=args.do_sample)
    
    print("\n=== Cypher Query ===")
    print(cypher)