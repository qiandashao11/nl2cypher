import random
import pandas as pd

# File paths (replace with your local paths)
headings_file = "MeshHeadings2026.txt"
qualifiers_file = "MeshQualifiers2026.txt"

# Read Heading and Qualifier lists
with open(headings_file, "r", encoding="utf-8") as f:
    headings = [line.strip() for line in f if line.strip()]

with open(qualifiers_file, "r", encoding="utf-8") as f:
    qualifiers = [line.strip() for line in f if line.strip()]

def random_mesh_term():
    """Randomly generate one MeSH term"""
    h = random.choice(headings)
    if random.random() < 0.5:  # 50% chance to use only the Heading
        return h
    else:  # 50% chance to add a Qualifier
        q = random.choice(qualifiers)
        return f"{h}/{q}"

# Generate dataset
num_samples = 1000  # number of terms to generate
terms = [random_mesh_term() for _ in range(num_samples)]

# Save as CSV
df = pd.DataFrame(terms, columns=["MeSH_term"])
df.to_csv("random_mesh_dataset.csv", index=False, encoding="utf-8")

print("✅ Generated random_mesh_dataset.csv; first 10 rows:")
print(df.head(10))
