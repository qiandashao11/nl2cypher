# BioMistral Answer Generator Test Results

Pipeline:

- Cypher generator: `phase3_multihop_training/lora_out_llama3_8b_multihop`
- Answer generator: `answer_generator_biomistral.py` using `BioMistral/BioMistral-7B`
- QA system: `qa_system_biomistral.py`
- Database: local Neo4j

## Result 1

Question:

```text
Find MeSH entitys co-occurring with ACSL3.
```

Cypher:

```cypher
MATCH (:Gene {ENTITY:'ACSL3'})-[:CO_OCCURS]-(m:MeSH) RETURN DISTINCT m.ENTITY AS mesh_entity
```

Execution:

```text
success=True
count=10
sample_data=[{'mesh_entity': 'phospholipid hydroperoxide glutathione peroxidase/metabolism'}, {'mesh_entity': 'oxidative stress/drug effects'}, {'mesh_entity': 'oleic acid/metabolism'}, {'mesh_entity': 'neoplasm metastasis/drug therapy'}, {'mesh_entity': 'melanoma/blood'}, {'mesh_entity': 'lymph/metabolism'}, {'mesh_entity': 'iron/metabolism'}, {'mesh_entity': 'glutathione/metabolism'}]
```

Answer:

```text
The following entities co-occurred with ACSL in PubMed articles: phospholipid hydoperoxide glutatione peroxidase / metabolism; oxidative stress / drug effects; oleic acid / metabolism, neoplasm metastasies / drug therapy; melanomas / blood; lymph / metabolism iron / metabolism glutathione / metabolism coenzyme a ligase / metabolisms cell survival
```

## Result 2

Question:

```text
Show literature linked to ACSL3.
```

Cypher:

```cypher
MATCH (:Gene {ENTITY:'ACSL3'})-[:HAS_SOURCE]->(l:Literature) RETURN DISTINCT l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal
```

Execution:

```text
success=True
count=1
sample_data=[{'pmid': '32814895', 'title': 'Lymph protects metastasizing melanoma cells from ferroptosis.', 'year': 2020, 'journal': 'Nature'}]
```

Answer:

```text
Lymph protects protects metastatic melanoma cells against ferroptotic cell death by providing iron sequestration .
```

## Result 3

Question:

```text
Show literature for MeSH t-lymphocytes cytotoxic/immunology.
```

Cypher:

```cypher
MATCH (:MeSH {ENTITY:'t-lymphocytes cytotoxic/immunology'})-[:HAS_SOURCE]->(l:Literature) RETURN DISTINCT l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal
```

Execution:

```text
success=True
count=1
sample_data=[{'pmid': '34062295', 'title': 'Ferroptosis Suppressive Genes Correlate with Immunosuppression in Glioblastoma.', 'year': 2021, 'journal': 'World Neurosurg'}]
```

Answer:

```text
Ferroptosis suppressive genes correlate with immunosuppression in glioblastoma.
```

## Result 4

Question:

```text
Papers mentioning both ACSL3 and phospholipid hydroperoxide glutathione peroxidase/metabolism.
```

Cypher:

```cypher
MATCH (:Gene {ENTITY:'ACSL3'})-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(:MeSH {ENTITY:'phospholipid hydroperoxide glutathione peroxidase/metabolism'}) RETURN DISTINCT l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal
```

Execution:

```text
success=True
count=1
sample_data=[{'pmid': '32814895', 'title': 'Lymph protects metastasizing melanoma cells from ferroptosis.', 'year': 2020, 'journal': 'Nature'}]
```

Answer:

```text
The paper with PubMed ID 32817695 mentions both ACSL4 and phosphol lipid hydroperoxides glutathione S-transferases.
```

## Result 5

Question:

```text
Count MeSH entitys shared by ACSL4 and AIFM2.
```

Cypher:

```cypher
MATCH (:Gene {ENTITY:'ACSL4'})-[:CO_OCCURS]-(m:MeSH)-[:CO_OCCURS]-(:Gene {ENTITY:'AIFM2'}) RETURN count(DISTINCT m) AS shared_mesh_count
```

Execution:

```text
success=True
count=1
sample_data=[{'shared_mesh_count': 5}]
```

Answer:

```text
Five MeSH entities are shared by ACS L4 and AIF M2.
```

## Result 6

Question:

```text
Find gene and MeSH pairs reported in the paper with PMID 34093872.
```

Cypher:

```cypher
MATCH (g:Gene)-[:HAS_SOURCE]->(l:Literature {PMID:'34093872'})<-[:HAS_SOURCE]-(m:MeSH) RETURN DISTINCT g.ENTITY AS gene_entity, m.ENTITY AS mesh_entity
```

Execution:

```text
success=True
count=165
sample_data=[{'gene_entity': 'RPL17-C18orf32', 'mesh_entity': 'mitochondria/drug effects'}, {'gene_entity': 'RPL17-C18orf32', 'mesh_entity': 'cell survival/drug effects'}, {'gene_entity': 'RPL17-C18orf32', 'mesh_entity': 'cd8-positive t-lymphocytes/immunology'}, {'gene_entity': 'RPL17-C18orf32', 'mesh_entity': 'tumor microenvironment/drug effects'}, {'gene_entity': 'RPL17-C18orf32', 'mesh_entity': 'nf-e2-related factor 2/genetics'}, {'gene_entity': 'RPL17-C18orf32', 'mesh_entity': 'amp-activated protein kinase kinases'}, {'gene_entity': 'RPL17-C18orf32', 'mesh_entity': 'glycogen synthase kinase 3/metabolism'}, {'gene_entity': 'RPL17-C18orf32', 'mesh_entity': 'lung neoplasms/drug therapy'}]
```

Answer:

```text
The following genes have been reported to interact with mitochondria: RPL17-CO18ORF32, RPL17, and NFE2L2.
```

## Result 7

Question:

```text
Find World Neurosurg papers that report SLC40A1.
```

Cypher:

```cypher
MATCH (:Gene {ENTITY:'SLC40A1'})-[:HAS_SOURCE]->(l:Literature) WHERE l.Journal = 'World Neurosurg' RETURN DISTINCT l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal
```

Execution:

```text
success=True
count=1
sample_data=[{'pmid': '34062295', 'title': 'Ferroptosis Suppressive Genes Correlate with Immunosuppression in Glioblastoma.', 'year': 2021, 'journal': 'World Neurosurg'}]
```

Answer:

```text
The paper with PMID 34068295 reports on SLC40AI1.
```

## Result 8

Question:

```text
SLC40A1 papers after 2020.
```

Cypher:

```cypher
MATCH (:Gene {ENTITY:'SLC40A1'})-[:HAS_SOURCE]->(l:Literature) WHERE l.Year > 2020 RETURN DISTINCT l.PMID AS pmid, l.Title AS title, l.Year AS year, l.Journal AS journal
```

Execution:

```text
success=True
count=1
sample_data=[{'pmid': '34062295', 'title': 'Ferroptosis Suppressive Genes Correlate with Immunosuppression in Glioblastoma.', 'year': 2021, 'journal': 'World Neurosurg'}]
```

Answer:

```text
Ferroptosis Suppression Genes Correlates with Immunosupression in Gliomas.
```

## Result 9

Question:

```text
Find papers where MeSH contains 'melanoma' and genes are reported after 2019.
```

Cypher:

```cypher
MATCH (m:MeSH)-[:HAS_SOURCE]->(l:Literature)<-[:HAS_SOURCE]-(g:Gene) WHERE toLower(m.ENTITY) CONTAINS toLower('melanoma') AND l.Year > 2019 RETURN DISTINCT l.PMID AS pmid, g.ENTITY AS gene_entity, m.ENTITY AS mesh_entity
```

Execution:

```text
success=True
count=6
sample_data=[{'pmid': '32814895', 'gene_entity': 'GPX4', 'mesh_entity': 'melanoma/blood'}, {'pmid': '32814895', 'gene_entity': 'ACSL3', 'mesh_entity': 'melanoma/blood'}, {'pmid': '31043744', 'gene_entity': 'CD8A', 'mesh_entity': 'melanoma/genetics'}, {'pmid': '31043744', 'gene_entity': 'SLC9A1', 'mesh_entity': 'melanoma/genetics'}, {'pmid': '31043744', 'gene_entity': 'SLC7A11', 'mesh_entity': 'melanoma/genetics'}, {'pmid': '31043744', 'gene_entity': 'SLC3A2', 'mesh_entity': 'melanoma/genetics'}]
```

Answer:

```text
The following papers were returned from PubMed with publication dates after 2020:
```

## Result 10

Question:

```text
Find genes related to both MeSH oxidative stress/drug effects and MeSH phospholipid hydroperoxide glutathione peroxidase/metabolism.
```

Cypher:

```cypher
MATCH (:MeSH {ENTITY:'oxidative stress/drug effects'})-[:CO_OCCURS]-(g:Gene)-[:CO_OCCURS]-(:MeSH {ENTITY:'phospholipid hydroperoxide glutathione peroxidase/metabolism'}) RETURN DISTINCT g.ENTITY AS gene_entity
```

Execution:

```text
success=True
count=1
sample_data=[{'gene_entity': 'ACSL3'}]
```

Answer:

```text
ACSL3 is a gene related to both MeSh oxidative stress / drug effects and MeSh phospholipids hydroperoxide GPx metabolism.
```
