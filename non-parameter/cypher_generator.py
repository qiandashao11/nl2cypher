#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================== cypher_generator.py ====================
"""
Cypher查询生成器
使用微调的Llama 3.1将自然语言转换为Neo4j Cypher查询
"""
import os, json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login

HF_TOKEN = os.environ.get("HF_TOKEN")


class CypherGenerator:
    """Cypher查询生成器"""
    
    def __init__(self, 
                 base_model="meta-llama/Llama-3.1-8B-Instruct",
                 lora_dir="./lora_out_llama3_8b",
                 hf_token=None):
        """
        初始化生成器
        
        Args:
            base_model: 基础模型路径
            lora_dir: LoRA适配器路径
            hf_token: Hugging Face token
        """
        self.base_model = base_model
        self.lora_dir = lora_dir
        self.hf_token = hf_token or HF_TOKEN
        
        if self.hf_token:
            try:
                login(token=self.hf_token, add_to_git_credential=False)
                print("✅ 已登录 Hugging Face")
            except Exception as e:
                print(f"⚠️ 登录失败: {e}")
        
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
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
        print("✅ 模型加载完成")
    
    def _build_prompt(self, question: str, params: dict = None) -> str:
        """构造prompt"""
        system = (
            """You are a Cypher generator for a Neo4j graph.

            Your job:
            - Take a natural language question about the graph.
            - Return ONLY ONE Cypher query that answers the question.

            Output rules:
            - The FIRST token MUST be one of: MATCH, CREATE, MERGE, RETURN.
            - Return ONLY the Cypher query. No explanations, no prose, no markdown fences, no comments.
            - Do NOT print parameters or JSON. Output pure Cypher only.
            - Prefer a single MATCH … RETURN pattern when possible.

            Graph schema :

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
            """)
        
        user = f"{question}"
        if params:
            user += "\nParameters: " + json.dumps(params, ensure_ascii=False, separators=(',', ':'))
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    


    def _clean_output(self, raw_text: str) -> str:
        """
        清理输出（参考你的代码逻辑）
        1. 提取第一个MATCH/CREATE/MERGE/RETURN开始的部分
        2. 去掉markdown代码围栏
        3. 只保留第一个RETURN
        4. 去重RETURN后的列
        5. 禁止输出 SKIP / LIMIT（全部删除）
        """
        # 从第一处 MATCH/CREATE/MERGE/RETURN 开始截取
        start = re.search(r'(?m)^(MATCH|CREATE|MERGE|RETURN)\b', raw_text)
        cypher = raw_text[start.start():].strip() if start else raw_text.strip()
        
        # 去掉markdown代码围栏
        cypher = re.sub(r'^\s*```(?:cypher)?\s*', '', cypher)
        cypher = re.sub(r'\s*```.*$', '', cypher)
        
        # 只保留第一个RETURN
        m = re.search(r'(?i)\bRETURN\b', cypher)
        if m:
            head = cypher[:m.start()]
            after = cypher[m.start():]
            after_line = after.split("\n", 1)[0]  # 第一行
            
            # "RETURN a RETURN b" → "RETURN a"
            parts = re.split(r'(?i)\bRETURN\b', after_line)
            if len(parts) > 1:
                cleaned_return = "RETURN " + parts[1].strip()
                cypher = (head + cleaned_return).strip()
        
        # 去重RETURN后的列
        m_ret = re.search(r'(?i)\bRETURN\b', cypher)
        if m_ret:
            head = cypher[:m_ret.start()]
            tail = cypher[m_ret.start():].strip()
            
            m_clause = re.match(r'(?i)RETURN\s+(DISTINCT\s+)?(.*)', tail)
            if m_clause:
                distinct_part = m_clause.group(1) or ""
                cols_part = m_clause.group(2).strip()
                
                # 按逗号拆列
                raw_cols = [c.strip() for c in cols_part.split(",") if c.strip()]
                
                seen = set()
                kept = []
                for col in raw_cols:
                    # 提取AS前面的表达式
                    expr = re.split(r'(?i)\s+AS\s+', col, maxsplit=1)[0].strip()
                    key = expr.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    kept.append(col)
                
                new_tail = "RETURN " + distinct_part + ", ".join(kept)
                cypher = (head + new_tail).strip()

        # ⭐ 禁止输出 SKIP / LIMIT：把任何 " SKIP xxx" / " LIMIT xxx" 直接删除
        #   用 \S+ 而不是 \d+，以防模型输出 SKIP $n 这类参数形式
        cypher = re.sub(r'\s+SKIP\s+\S+', '', cypher, flags=re.IGNORECASE)
        cypher = re.sub(r'\s+LIMIT\s+\S+', '', cypher, flags=re.IGNORECASE)
        
        # 清理空行
        lines = [ln.rstrip() for ln in cypher.splitlines() if ln.strip()]
        cypher = "\n".join(lines)
        
        return cypher

        
    def generate(self, 
                question: str, 
                params: dict = None,
                max_new_tokens: int = 256,
                temperature: float = 0.7,
                top_p: float = 1.0,
                do_sample: bool = False) -> str:
        """
        生成Cypher查询
        
        Args:
            question: 自然语言问题
            params: 可选参数字典
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
            do_sample: 是否采样
            
        Returns:
            Cypher查询字符串
        """
        prompt = self._build_prompt(question, params)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # 只有在do_sample=True时才用temperature/top_p
        if do_sample:
            safe_temp = max(temperature, 1e-4)
            gen_kwargs.update(temperature=safe_temp, top_p=top_p)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        raw_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._clean_output(raw_text)
    
    def __call__(self, question: str, **kwargs) -> str:
        """让对象可调用"""
        return self.generate(question, **kwargs)


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--lora_dir", default="./lora_out_llama3_8b")
    ap.add_argument("--question", required=True)
    ap.add_argument("--params_json", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--do_sample", action="store_true")
    args = ap.parse_args()
    
    params = json.loads(args.params_json) if args.params_json else None
    gen = CypherGenerator(args.base_model, args.lora_dir)
    cypher = gen.generate(
        args.question, params, args.max_new_tokens, 
        args.temperature, args.top_p, args.do_sample
    )
    
    print("\n=== Cypher ===")
    print(cypher)