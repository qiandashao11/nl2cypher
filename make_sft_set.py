# nl2cypher/make_sft_set.py
import json, jsonlines, pathlib

SRC = pathlib.Path("nl2cypher_train_en_1000.jsonl")   # 原始生成文件
DST = pathlib.Path("dataset_en.jsonl")                # 训练集输出

count = 0
with jsonlines.open(SRC) as reader, open(DST, "w", encoding="utf-8") as out:
    for obj in reader:
        q = obj.get("question_en", "").strip()
        cy = obj.get("alt_cypher") or obj.get("cypher")
        if not q or not cy:
            continue
        inst = (
            "Translate the following natural language query into a Cypher statement.\n"
            + q
        )
        out.write(json.dumps({"instruction": inst, "output": cy}, ensure_ascii=False) + "\n")
        count += 1

print(f"✅ Converted {count} samples to {DST}")
