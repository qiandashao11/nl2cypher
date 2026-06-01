import random
import pandas as pd

# 文件路径（换成你自己的本地路径）
headings_file = "MeshHeadings2026.txt"
qualifiers_file = "MeshQualifiers2026.txt"

# 读取 Heading 和 Qualifier 列表
with open(headings_file, "r", encoding="utf-8") as f:
    headings = [line.strip() for line in f if line.strip()]

with open(qualifiers_file, "r", encoding="utf-8") as f:
    qualifiers = [line.strip() for line in f if line.strip()]

def random_mesh_term():
    """随机生成一个 MeSH term"""
    h = random.choice(headings)
    if random.random() < 0.5:  # 50% 概率只用 Heading
        return h
    else:  # 50% 概率加 Qualifier
        q = random.choice(qualifiers)
        return f"{h}/{q}"

# 生成数据集
num_samples = 1000  # 想要生成多少个 term
terms = [random_mesh_term() for _ in range(num_samples)]

# 保存为 CSV
df = pd.DataFrame(terms, columns=["MeSH_term"])
df.to_csv("random_mesh_dataset.csv", index=False, encoding="utf-8")

print("✅ 已生成 random_mesh_dataset.csv，示例前 10 行：")
print(df.head(10))
