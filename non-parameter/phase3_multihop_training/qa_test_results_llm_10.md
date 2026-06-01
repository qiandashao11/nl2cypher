# LLM Answer Generator Test Results

Pipeline:

- Cypher generator: `phase3_multihop_training/lora_out_llama3_8b_multihop`
- Answer generator: `answer_generator_llm.py`
- QA system: `qa_system_llm.py`
- Environment: `micromamba run -n nl2`
- Database: local Neo4j

These 10 questions were built from entities and combinations that exist in the database.

## Summary

All 10 generated Cypher queries executed successfully. However, the LLM-only answer generator was unstable: it sometimes changed entity names, added unsupported explanations, produced formatting artifacts, or generated step-by-step text. This makes it less reliable than `answer_generator_clean.py` for final factual answers.

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
sample_data=[
  {'mesh_entity': 'phospholipid hydroperoxide glutathione peroxidase/metabolism'},
  {'mesh_entity': 'oxidative stress/drug effects'},
  {'mesh_entity': 'oleic acid/metabolism'},
  {'mesh_entity': 'neoplasm metastasis/drug therapy'},
  {'mesh_entity': 'melanoma/blood'}
]
```

Answer:

```text
The following MeSH entities have been identified as co-occurring (co-expressed) with ACSL3: Phospholipid Hydroperoxide Glutathione Peroxidase Metabolism, Oxidative Stress Drug Effects, Oleic Acid Metabolism, Neoplasm Metastasis Drug Therapy, Melanoma Blood, Lymph Metabolism, Iron Metabolism, Glutathione Metabolism, Coenzyme A Ligases Metabolism, Cell Survival. These represent biological processes related to ACSL3 function. Representative examples include metabolism of specific compounds like oleic acid and phospholipids, drug-induced changes such as oxidative stress, and cellular responses including cell survival and metastasis. No further information about these relationships can be inferred from this analysis. 10 distinct MeSH entities were found.
```

Note:

```text
Problem: it changes co-occurring to co-expressed, which is not guaranteed by the graph schema.
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
sample_data=[
  {'pmid': '32814895', 'title': 'Lymph protects metastasizing melanoma cells from ferroptosis.', 'year': 2020, 'journal': 'Nature'}
]
```

Answer:

```text
There was one paper linking ACSL3 to lymph protecting metastasizing melanomacells from ferroptosiscalled "Lymph protects metastasisingmelanoma cellsfromferroptosis" publishedinNaturein2020. The PMID of thispaperis32814895.
```

Note:

```text
Problem: semantically close, but words are glued together and the title is distorted.
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
sample_data=[
  {'pmid': '34062295', 'title': 'Ferroptosis Suppressive Genes Correlate with Immunosuppression in Glioblastoma.', 'year': 2021, 'journal': 'World Neurosurg'}
]
```

Answer:

```text
There was one study related to 't-lymphocyte cytotoxicity' published in 2021 in the Journal of World Neurosurg. The study titled "Ferroptosis Suppressive...". 
(no additional information about this specific paper as per constraint.) 

However, I can rephrase the response based on your feedback:

"There is one relevant study from 2021 in World Neurosurg." 
This revised version meets all the requirements you specified. Let me know if further adjustments are needed! 

Please let me know how I should proceed next. Should we continue testing my responses? Would you like me to generate more questions or revise anything else before moving forward? 

Best regards,
[Your Name] 
Knowledge Graph Assistant 
Biomedical Domain Expertise 
AI-Powered Response Generation 
High-Quality Information Retrieval 
Domain-Specific Knowledge Base 
Entity Disambiguation & Contextualization 
Concise & Accurate Answering 
Continuous Improvement through Feedback 
Adaptability to New Domains & Topics 
Scalable Architecture for Large Datasets 
Real-Time
```

Note:

```text
Problem: answer starts reasonably but then generates meta text, sign-off text, and unrelated promotional phrases.
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
sample_data=[
  {'pmid': '32814895', 'title': 'Lymph protects metastasizing melanoma cells from ferroptosis.', 'year': 2020, 'journal': 'Nature'}
]
```

Answer:

```text
One paper mentions both ACSL3 (acyl-CoA synthetase long-chain family member 3) and phospholipide hydroperoxide glutathioperoxidase metabolism. This single study was published in Nature in 2020 with PMID 32814895. The paper discusses how lymph protects metastasizing cancer cells from cell death by ferroptosis. No other studies match this search criteria.
```

Note:

```text
Mostly useful, but it adds an expanded gene name and changes the MeSH wording. These were not directly returned by the database.
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
sample_data=[
  {'shared_mesh_count': 5}
]
```

Answer:

```text
There are 5 MeSH terms shared between ACSL4 and AIFFM2. These MeSH terms represent common biological themes associated with both genes.
```

Note:

```text
Problem: count is correct, but AIFM2 is misspelled as AIFFM2.
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
sample_data=[
  {'gene_entity': 'RPL17-C18orf32', 'mesh_entity': 'mitochondria/drug effects'},
  {'gene_entity': 'RPL17-C18orf32', 'mesh_entity': 'cell survival/drug effects'},
  {'gene_entity': 'RPL17-C18orf32', 'mesh_entity': 'cd8-positive t-lymphocytes/immunology'},
  {'gene_entity': 'RPL17-C18orf32', 'mesh_entity': 'tumor microenvironment/drug effects'},
  {'gene_entity': 'RPL17-C18orf32', 'mesh_entity': 'nf-e2-related factor 2/genetics'}
]
```

Answer:

```text
The paper with PMID 341093872 reports associations between genes RPL17 and C18orf32 and various MeSH terms including drug effects on mitochondria, cell survival, CD8+ T lymphocytes, and NF-E2 related factor 2 genetics. Representative examples of these associations include RPL17 being linked to mitochondrial drug effects and CD8 positive T lymphocyte immunology. No other unique gene-MeSH pairs were identified beyond those involving RPL17 and its isoform C18orf32. The number of distinct gene-MeSH pairings reported in this study is 40.
```

Note:

```text
Serious problem: PMID changed from 34093872 to 341093872, gene name was split, and count changed from 165 to 40.
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
sample_data=[
  {'pmid': '34062295', 'title': 'Ferroptosis Suppressive Genes Correlate with Immunosuppression in Glioblastoma.', 'year': 2021, 'journal': 'World Neurosurg'}
]
```

Answer:

```text
There was one paper published in World Neurosurg that reports on SLC40A11 correlates with immunosuppression in glioblastoma. The paper has PMID 34062295. No other details about this paper could be determined from the available information. ]]0|_text=]]0|_text=Ferroptosis Suppressive Gene]]0|_correlation with Immun]]0|_suppression in Glia]]0|_blastoma. (no further info). ]]0|_PMID 34062]]0|_295. ]]>0|_The paper titled "Ferroptosis Supressive Genes Correlates with Immun suppression in Gliobastoma" with PMID 3406229]5. ]]>0|. ]]>0|No additional informaion is avaiable for thiis paer. ]]>0|(no furtuer infomation availble). ]]>0|This papwer haas been publishehd in Worlld Neurorogiruyr. ]]>0|[
```

Note:

```text
Serious problem: SLC40A1 changed to SLC40A11 and the answer contains corrupted text.
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
sample_data=[
  {'pmid': '34062295', 'title': 'Ferroptosis Suppressive Genes Correlate with Immunosuppression in Glioblastoma.', 'year': 2021, 'journal': 'World Neurosurg'}
]
```

Answer:

```text
There was one paper related to SLC40A1 published after 2020. The paper "Ferroptosis Suppressive Gene..." from 2021 appeared in the Journal of World Neurosurgery. It has PMID number 34062295.
```

Note:

```text
This answer is mostly acceptable, although it changes "World Neurosurg" to "World Neurosurgery".
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
sample_data=[
  {'pmid': '32814895', 'gene_entity': 'GPX4', 'mesh_entity': 'melanoma/blood'},
  {'pmid': '32814895', 'gene_entity': 'ACSL3', 'mesh_entity': 'melanoma/blood'},
  {'pmid': '31043744', 'gene_entity': 'CD8A', 'mesh_entity': 'melanoma/genetics'},
  {'pmid': '31043744', 'gene_entity': 'SLC9A1', 'mesh_entity': 'melanoma/genetics'},
  {'pmid': '31043744', 'gene_entity': 'SLC7A11', 'mesh_entity': 'melanoma/genetics'}
]
```

Answer:

```text
There are 6 papers related to melanoma with genetic information published after 2019. Representative papers include one on "Melanoma/Blood" discussing GPX4 and ACSL3, and another on "Melanomas/Genetics" examining multiple genes including CD8A, SLC9A, SLC7A, and SLC3A. The first paper has PMID 32814895 while the second has PMID 31043744. These studies provide insights into the genetics of melanoma. Further details can be found by accessing these specific research articles directly.
```

Note:

```text
Partly acceptable, but it changes gene names such as SLC9A1 to SLC9A and adds extra external-reading language.
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
sample_data=[
  {'gene_entity': 'ACSL3'}
]
```

Answer:

```text
The gene ACSL3 is associated with both MeSH terms 'oxidative stress/drugs' and 'phospholipide hydroperoxide glutathioperoxidase metabolism'. No other genes match this relationship. 

### Step 1: Analyze the problem statement
We need to find genes related to two specific Medical Subject Headings (MeSH) terms: "oxidative stress/drugal drug effects" and "phospholipids hydroperoxide glutamine peroxidase/metabolic process".

### Step 2: Understand the given query results
There is one row of result data showing that the gene ACSL3 co-occurs with both specified MeSH terms.

### Step 3: Determine how many genes meet the criteria
Only one gene meets the specified conditions based on the query results.

### Step 4: Summarize the findings for the user
Based on the analysis, we can conclude that the gene ACSl3 is linked to both mentioned MeSH terms.

The final answer is: The gene ACSL3 has been identified as being connected to both
```

Note:

```text
Serious problem: answer includes chain-of-thought style steps and changes the MeSH text.
```

## Overall Assessment

The LLM-based answer generator is more natural in some cases, but it is not reliable enough for factual KGQA output. Its main failure modes are:

- Entity mutation: `AIFM2` -> `AIFFM2`, `SLC40A1` -> `SLC40A11`
- PMID mutation: `34093872` -> `341093872`
- Count mutation: `165` -> `40`
- Unsupported semantic shift: `co-occurring` -> `co-expressed`
- Corrupted text artifacts
- Step-by-step reasoning output
- External or unsupported language

Recommendation:

Use `answer_generator_clean.py` as the default factual answer module. If a more natural style is needed, use an LLM only as a final polishing step over the already grounded clean answer, not directly over raw query results.

