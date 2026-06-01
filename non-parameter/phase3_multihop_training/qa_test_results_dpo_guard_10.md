# SFT Answer Generator Test Results

Pipeline:

- Cypher generator: `phase3_multihop_training/lora_out_llama3_8b_multihop`
- Answer generator: `answer_rl_training/lora_out_llama3_answer_dpo`
- Reward guard: `True`
- QA system: `qa_system_sft.py`
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
{
  "success": true,
  "count": 10,
  "sample_data": [
    {
      "mesh_entity": "phospholipid hydroperoxide glutathione peroxidase/metabolism"
    },
    {
      "mesh_entity": "oxidative stress/drug effects"
    },
    {
      "mesh_entity": "oleic acid/metabolism"
    },
    {
      "mesh_entity": "neoplasm metastasis/drug therapy"
    },
    {
      "mesh_entity": "melanoma/blood"
    },
    {
      "mesh_entity": "lymph/metabolism"
    },
    {
      "mesh_entity": "iron/metabolism"
    },
    {
      "mesh_entity": "glutathione/metabolism"
    }
  ]
}
```

Answer:

```text
MeSH entitys co-occurring with ACSL3 is linked to 10 MeSH entities in the knowledge graph. The returned items mainly point to oxidative stress, lipid metabolism, cancer and tumor biology, and drug response. Representative terms include phospholipid hydroperoxide glutathione peroxidase/metabolism, oxidative stress/drug effects, oleic acid/metabolism, neoplasm metastasis/drug therapy, melanoma/blood, lymph/metabolism.
```

Answer reward:

```text
{
  "reward": 0.933,
  "weighted_reward_before_penalty": 0.933,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 0.86,
    "count_accuracy": 1.0,
    "result_coverage": 0.875,
    "relation_semantics": 1.0,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [],
      "missing_required_values": [
        "iron/metabolism",
        "glutathione/metabolism",
        "coenzyme a ligases/metabolism",
        "cell survival"
      ]
    },
    "count": {
      "expected_count": 10,
      "wrong_count_numbers": []
    },
    "coverage": {
      "covered_sample_values": 6,
      "total_sample_values": 8
    },
    "relation": {
      "relation": "co_occurs",
      "bad_terms": []
    },
    "format": {
      "bad_meta_phrases": [],
      "artifact_count": 0
    },
    "fluency": {
      "sentence_count": 3,
      "token_count": 49
    }
  }
}
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
{
  "success": true,
  "count": 1,
  "sample_data": [
    {
      "pmid": "32814895",
      "title": "Lymph protects metastasizing melanoma cells from ferroptosis.",
      "year": 2020,
      "journal": "Nature"
    }
  ]
}
```

Answer:

```text
One matching paper was found for ACSL3: "Lymph protects metastasizing melanoma cells from ferroptosis." (Nature, 2020; PMID: 32814895).
```

Answer reward:

```text
{
  "reward": 1.0,
  "weighted_reward_before_penalty": 1.0,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 1.0,
    "count_accuracy": 1.0,
    "result_coverage": 1.0,
    "relation_semantics": 1.0,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [],
      "missing_required_values": []
    },
    "count": {
      "expected_count": 1,
      "wrong_count_numbers": []
    },
    "coverage": {
      "covered_values": 4,
      "total_values": 4
    },
    "relation": {
      "relation": "not_co_occurs"
    },
    "format": {
      "bad_meta_phrases": [],
      "artifact_count": 0
    },
    "fluency": {
      "sentence_count": 1,
      "token_count": 18
    }
  }
}
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
{
  "success": true,
  "count": 1,
  "sample_data": [
    {
      "pmid": "34062295",
      "title": "Ferroptosis Suppressive Genes Correlate with Immunosuppression in Glioblastoma.",
      "year": 2021,
      "journal": "World Neurosurg"
    }
  ]
}
```

Answer:

```text
One matching paper was found for MeSH t-lymphocytes cytotoxic/immunology: "Ferroptosis Suppressive Genes Correlate with Immunosuppression in Glioblastoma." (World Neurosurg, 2021; PMID: 34062295).
```

Answer reward:

```text
{
  "reward": 1.0,
  "weighted_reward_before_penalty": 1.0,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 1.0,
    "count_accuracy": 1.0,
    "result_coverage": 1.0,
    "relation_semantics": 1.0,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [],
      "missing_required_values": []
    },
    "count": {
      "expected_count": 1,
      "wrong_count_numbers": []
    },
    "coverage": {
      "covered_values": 4,
      "total_values": 4
    },
    "relation": {
      "relation": "not_co_occurs"
    },
    "format": {
      "bad_meta_phrases": [],
      "artifact_count": 0
    },
    "fluency": {
      "sentence_count": 1,
      "token_count": 22
    }
  }
}
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
{
  "success": true,
  "count": 1,
  "sample_data": [
    {
      "pmid": "32814895",
      "title": "Lymph protects metastasizing melanoma cells from ferroptosis.",
      "year": 2020,
      "journal": "Nature"
    }
  ]
}
```

Answer:

```text
One matching paper was found: "Lymph protects metastasizing melanoma cells from ferroptosis." (Nature, 2020; PMID: 32814895).
```

Answer reward:

```text
{
  "reward": 1.0,
  "weighted_reward_before_penalty": 1.0,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 1.0,
    "count_accuracy": 1.0,
    "result_coverage": 1.0,
    "relation_semantics": 1.0,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [],
      "missing_required_values": []
    },
    "count": {
      "expected_count": 1,
      "wrong_count_numbers": []
    },
    "coverage": {
      "covered_values": 4,
      "total_values": 4
    },
    "relation": {
      "relation": "not_co_occurs"
    },
    "format": {
      "bad_meta_phrases": [],
      "artifact_count": 0
    },
    "fluency": {
      "sentence_count": 1,
      "token_count": 16
    }
  }
}
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
{
  "success": true,
  "count": 1,
  "sample_data": [
    {
      "shared_mesh_count": 5
    }
  ]
}
```

Answer:

```text
ACSL4 and AIFM2 share 5 MeSH entities in the knowledge graph.
```

Answer reward:

```text
{
  "reward": 0.9025,
  "weighted_reward_before_penalty": 0.9025,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 1.0,
    "count_accuracy": 1.0,
    "result_coverage": 0.7,
    "relation_semantics": 0.75,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [],
      "missing_required_values": []
    },
    "count": {
      "expected_count": 5,
      "wrong_count_numbers": []
    },
    "coverage": {
      "covered_values": 0,
      "total_values": 0
    },
    "relation": {
      "relation": "co_occurs",
      "bad_terms": []
    },
    "format": {
      "bad_meta_phrases": [],
      "artifact_count": 0
    },
    "fluency": {
      "sentence_count": 1,
      "token_count": 11
    }
  }
}
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
{
  "success": true,
  "count": 165,
  "sample_data": [
    {
      "gene_entity": "RPL17-C18orf32",
      "mesh_entity": "mitochondria/drug effects"
    },
    {
      "gene_entity": "RPL17-C18orf32",
      "mesh_entity": "cell survival/drug effects"
    },
    {
      "gene_entity": "RPL17-C18orf32",
      "mesh_entity": "cd8-positive t-lymphocytes/immunology"
    },
    {
      "gene_entity": "RPL17-C18orf32",
      "mesh_entity": "tumor microenvironment/drug effects"
    },
    {
      "gene_entity": "RPL17-C18orf32",
      "mesh_entity": "nf-e2-related factor 2/genetics"
    },
    {
      "gene_entity": "RPL17-C18orf32",
      "mesh_entity": "amp-activated protein kinase kinases"
    },
    {
      "gene_entity": "RPL17-C18orf32",
      "mesh_entity": "glycogen synthase kinase 3/metabolism"
    },
    {
      "gene_entity": "RPL17-C18orf32",
      "mesh_entity": "lung neoplasms/drug therapy"
    }
  ]
}
```

Answer:

```text
Found 165 matching gene-MeSH pairs. The returned items mainly point to oxidative stress, lipid metabolism, cancer and tumor biology, and immune processes. Examples include RPL17-C18orf32 with mitochondria/drug effects, RPL17-C18orf32 with cell survival/drug effects, RPL17-C18orf32 with cd8-positive t-lymphocytes/immunology, RPL17-C18orf32 with tumor microenvironment/drug effects, RPL17-C18orf32 with nf-e2-related factor 2/genetics.
```

Answer reward:

```text
{
  "reward": 0.8059,
  "weighted_reward_before_penalty": 0.8059,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 0.7529,
    "count_accuracy": 0.6,
    "result_coverage": 0.8,
    "relation_semantics": 1.0,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [],
      "missing_required_values": [
        "amp-activated protein kinase kinases",
        "glycogen synthase kinase 3/metabolism",
        "lung neoplasms/drug therapy",
        "iron/chemistry",
        "tor serine-threonine kinases/metabolism",
        "oxidative stress/drug effects",
        "amp-activated protein kinase kinases",
        "glycogen synthase kinase 3/metabolism",
        "lung neoplasms/drug therapy",
        "iron/chemistry",
        "tor serine-threonine kinases/metabolism",
        "oxidative stress/drug effects",
        "PDCD1",
        "PDCD1",
        "PDCD1",
        "PDCD1",
        "PDCD1",
        "PDCD1",
        "amp-activated protein kinase kinases",
        "PDCD1"
      ]
    },
    "count": {
      "expected_count": 165,
      "wrong_count_numbers": [
        2
      ]
    },
    "coverage": {
      "covered_sample_values": 8,
      "total_sample_values": 8
    },
    "relation": {
      "relation": "not_co_occurs"
    },
    "format": {
      "bad_meta_phrases": [],
      "artifact_count": 0
    },
    "fluency": {
      "sentence_count": 3,
      "token_count": 47
    }
  }
}
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
{
  "success": true,
  "count": 1,
  "sample_data": [
    {
      "pmid": "34062295",
      "title": "Ferroptosis Suppressive Genes Correlate with Immunosuppression in Glioblastoma.",
      "year": 2021,
      "journal": "World Neurosurg"
    }
  ]
}
```

Answer:

```text
One matching paper was found for SLC40A1: "Ferroptosis Suppressive Genes Correlate with Immunosuppression in Glioblastoma." (World Neurosurg, 2021; PMID: 34062295).
```

Answer reward:

```text
{
  "reward": 1.0,
  "weighted_reward_before_penalty": 1.0,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 1.0,
    "count_accuracy": 1.0,
    "result_coverage": 1.0,
    "relation_semantics": 1.0,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [],
      "missing_required_values": []
    },
    "count": {
      "expected_count": 1,
      "wrong_count_numbers": []
    },
    "coverage": {
      "covered_values": 4,
      "total_values": 4
    },
    "relation": {
      "relation": "not_co_occurs"
    },
    "format": {
      "bad_meta_phrases": [],
      "artifact_count": 0
    },
    "fluency": {
      "sentence_count": 1,
      "token_count": 20
    }
  }
}
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
{
  "success": true,
  "count": 1,
  "sample_data": [
    {
      "pmid": "34062295",
      "title": "Ferroptosis Suppressive Genes Correlate with Immunosuppression in Glioblastoma.",
      "year": 2021,
      "journal": "World Neurosurg"
    }
  ]
}
```

Answer:

```text
One matching paper was found: "Ferroptosis Suppressive Genes Correlate with Immunosuppression in Glioblastoma." (World Neurosurg, 2021; PMID: 34062295).
```

Answer reward:

```text
{
  "reward": 1.0,
  "weighted_reward_before_penalty": 1.0,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 1.0,
    "count_accuracy": 1.0,
    "result_coverage": 1.0,
    "relation_semantics": 1.0,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [],
      "missing_required_values": []
    },
    "count": {
      "expected_count": 1,
      "wrong_count_numbers": []
    },
    "coverage": {
      "covered_values": 4,
      "total_values": 4
    },
    "relation": {
      "relation": "not_co_occurs"
    },
    "format": {
      "bad_meta_phrases": [],
      "artifact_count": 0
    },
    "fluency": {
      "sentence_count": 1,
      "token_count": 18
    }
  }
}
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
{
  "success": true,
  "count": 6,
  "sample_data": [
    {
      "pmid": "32814895",
      "gene_entity": "GPX4",
      "mesh_entity": "melanoma/blood"
    },
    {
      "pmid": "32814895",
      "gene_entity": "ACSL3",
      "mesh_entity": "melanoma/blood"
    },
    {
      "pmid": "31043744",
      "gene_entity": "CD8A",
      "mesh_entity": "melanoma/genetics"
    },
    {
      "pmid": "31043744",
      "gene_entity": "SLC9A1",
      "mesh_entity": "melanoma/genetics"
    },
    {
      "pmid": "31043744",
      "gene_entity": "SLC7A11",
      "mesh_entity": "melanoma/genetics"
    },
    {
      "pmid": "31043744",
      "gene_entity": "SLC3A2",
      "mesh_entity": "melanoma/genetics"
    }
  ]
}
```

Answer:

```text
Found 6 paper-linked gene-MeSH associations. The returned items mainly point to cancer and tumor biology, and genetics and gene regulation. Examples include GPX4 / melanoma/blood in PMID 32814895, ACSL3 / melanoma/blood in PMID 32814895, CD8A / melanoma/genetics in PMID 31043744, SLC9A1 / melanoma/genetics in PMID 31043744, SLC7A11 / melanoma/genetics in PMID 31043744.
```

Answer reward:

```text
{
  "reward": 0.9942,
  "weighted_reward_before_penalty": 0.9942,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 0.9806,
    "count_accuracy": 1.0,
    "result_coverage": 1.0,
    "relation_semantics": 1.0,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [],
      "missing_required_values": [
        "SLC3A2"
      ]
    },
    "count": {
      "expected_count": 6,
      "wrong_count_numbers": []
    },
    "coverage": {
      "covered_sample_values": 8,
      "total_sample_values": 8
    },
    "relation": {
      "relation": "not_co_occurs"
    },
    "format": {
      "bad_meta_phrases": [],
      "artifact_count": 0
    },
    "fluency": {
      "sentence_count": 3,
      "token_count": 52
    }
  }
}
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
{
  "success": true,
  "count": 1,
  "sample_data": [
    {
      "gene_entity": "ACSL3"
    }
  ]
}
```

Answer:

```text
For MeSH oxidative stress/drug effects and MeSH phospholipid hydroperoxide glutathione peroxidase/metabolism, Found 1 matching genes: ACSL3.
```

Answer reward:

```text
{
  "reward": 0.9625,
  "weighted_reward_before_penalty": 0.9625,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 1.0,
    "count_accuracy": 1.0,
    "result_coverage": 1.0,
    "relation_semantics": 0.75,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [],
      "missing_required_values": []
    },
    "count": {
      "expected_count": 1,
      "wrong_count_numbers": []
    },
    "coverage": {
      "covered_values": 1,
      "total_values": 1
    },
    "relation": {
      "relation": "co_occurs",
      "bad_terms": []
    },
    "format": {
      "bad_meta_phrases": [],
      "artifact_count": 0
    },
    "fluency": {
      "sentence_count": 1,
      "token_count": 16
    }
  }
}
```
