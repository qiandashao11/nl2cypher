import pandas as pd
import random

gene_file = "Gene_Symbol_to_Alias_lookup_table.tsv"
mesh_file = "random_mesh_dataset.csv"

# Use only the first column
df_gene = pd.read_csv(gene_file, sep="\t", usecols=[0], names=["gene"], header=0)
df_mesh = pd.read_csv(mesh_file)

genes = df_gene["gene"].dropna().unique().tolist()
meshes = df_mesh.iloc[:, 0].dropna().unique().tolist()

print(f"Loaded {len(genes)} genes and {len(meshes)} mesh terms.")

records = []
for gene in random.sample(genes, min(len(genes), 500)):
    num_mesh = random.randint(1, 3)
    selected_meshes = random.sample(meshes, num_mesh)
    for m in selected_meshes:
        records.append({"gene": gene, "mesh": m})

df_pairs = pd.DataFrame(records)
df_pairs.to_csv("gene_mesh_dataset.csv", index=False)
