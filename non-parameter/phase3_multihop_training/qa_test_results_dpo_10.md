# SFT Answer Generator Test Results

Pipeline:

- Cypher generator: `phase3_multihop_training/lora_out_llama3_8b_multihop`
- Answer generator: `answer_rl_training/lora_out_llama3_answer_dpo`
- Reward guard: `False`
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
MeSH entitys co-occuring with ACSL3 is linked to 10 MeSH entitys in the knowledge graph. The returned items mainly point to oxidative stress, lipid metabolism, cancer and tumor biology, drug response, and iron-related biology. Representative terms include phospholipid hydroper oxide glutathione per oxidase/metabolism, oxidative stress/drug effects, oleic acid/metabolism, neoplasm metastasis/dr
```

Answer reward:

```text
{
  "reward": 0.841,
  "weighted_reward_before_penalty": 0.841,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 0.72,
    "count_accuracy": 1.0,
    "result_coverage": 0.625,
    "relation_semantics": 1.0,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [],
      "missing_required_values": [
        "phospholipid hydroperoxide glutathione peroxidase/metabolism",
        "neoplasm metastasis/drug therapy",
        "melanoma/blood",
        "lymph/metabolism",
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
      "covered_sample_values": 2,
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
      "sentence_count": 2,
      "token_count": 50
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
One matching paper was found for ACSL3: "Lymph protects metastasising melanoma cells from fer roptosis." Representative terms include ferroptosis. Representative genes include ACSL3. Representative MeSH terms include cancer and tumor biology, iron-related biology. Representative CO_OCCURS relationships include co-occurs with HIF1A, SLC7A11. The paper count is 1. Representative papers include "L
```

Answer reward:

```text
{
  "reward": 0.2,
  "weighted_reward_before_penalty": 0.5,
  "hallucination_penalty": 0.3,
  "components": {
    "entity_faithfulness": 0.0,
    "count_accuracy": 1.0,
    "result_coverage": 0.0,
    "relation_semantics": 1.0,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [
        "HIF1A",
        "SLC7A11"
      ],
      "missing_required_values": [
        "32814895",
        "Lymph protects metastasizing melanoma cells from ferroptosis.",
        "2020",
        "Nature"
      ]
    },
    "count": {
      "expected_count": 1,
      "wrong_count_numbers": []
    },
    "coverage": {
      "covered_values": 0,
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
      "sentence_count": 5,
      "token_count": 50
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
One matching paper was found for MeSH t-lysmpc0: "FerroptosisSuppressiveGenesCorrelateWithImmunosuppressionInGlioblastoma." Representative terms include immunology. Representative genes include CD247, FAS, HAVCR2, LAG3, PDCD1LG2. Representative PMIDs include 34062295. Representative mechanisms include immune processes. Representative MeSH terms
```

Answer reward:

```text
{
  "reward": 0.0,
  "weighted_reward_before_penalty": 0.55,
  "hallucination_penalty": 0.75,
  "components": {
    "entity_faithfulness": 0.0,
    "count_accuracy": 1.0,
    "result_coverage": 0.25,
    "relation_semantics": 1.0,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [
        "CD247",
        "FAS",
        "HAVCR2",
        "LAG3",
        "PDCD1LG2"
      ],
      "missing_required_values": [
        "Ferroptosis Suppressive Genes Correlate with Immunosuppression in Glioblastoma.",
        "2021",
        "World Neurosurg"
      ]
    },
    "count": {
      "expected_count": 1,
      "wrong_count_numbers": []
    },
    "coverage": {
      "covered_values": 1,
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
      "sentence_count": 4,
      "token_count": 33
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
One matching paper was found for ACSL3 and phosphолipid hydroperoxide	glutathione peroxidase.metabolism: "Lymph protects metastасizing melanoma cells from фerroptosis." (Nature, 2020; PMID: 32814895). Representative terms include phosphолipid hydroпероксиде glu	tathione пероксидасе метаболизм. Representative genes include ACSL3.
```

Answer reward:

```text
{
  "reward": 0.9238,
  "weighted_reward_before_penalty": 0.9238,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 0.9125,
    "count_accuracy": 1.0,
    "result_coverage": 0.75,
    "relation_semantics": 1.0,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [],
      "missing_required_values": [
        "Lymph protects metastasizing melanoma cells from ferroptosis."
      ]
    },
    "count": {
      "expected_count": 1,
      "wrong_count_numbers": []
    },
    "coverage": {
      "covered_values": 3,
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
      "sentence_count": 3,
      "token_count": 36
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
ACSL4 and AIFM۲ share 5 MeSH entitys in the knowledge graph. Representative terms include the requested items directly from the response. Representative terms are not available for this relationship. Representative PMID(s) are not available for thisrelationship. Representative genes are not available for thisRelationship. Representative mechanisms are notavailable. Representative journals are notavailable. Representative titles are notavailable. Representative years are notavailable. Representative sentences are notavailable.
```

Answer reward:

```text
{
  "reward": 0.7775,
  "weighted_reward_before_penalty": 0.7775,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 1.0,
    "count_accuracy": 1.0,
    "result_coverage": 0.7,
    "relation_semantics": 0.0,
    "format_cleanliness": 1.0,
    "fluency": 0.75
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
      "bad_terms": [
        "mechanism"
      ]
    },
    "format": {
      "bad_meta_phrases": [],
      "artifact_count": 0
    },
    "fluency": {
      "sentence_count": 10,
      "token_count": 63
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
Found 165 matching gene-MeSH pairs. The returned items mainly point to oxidative stress, lipid metabolism, cancer and tumor biology, immune processes, genetics and gene regulation, and drug response. Representative terms include mitochondria/drug effects, cell survival/drug effects, cd8-positive t-lymphtocytes/immunology, tumor microenvironment/drug effects, nf-e2-related factor 1/genetics, amp-activated protein kinase
```

Answer reward:

```text
{
  "reward": 0.7268,
  "weighted_reward_before_penalty": 0.7268,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 0.6977,
    "count_accuracy": 0.6,
    "result_coverage": 0.4875,
    "relation_semantics": 1.0,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [],
      "missing_required_values": [
        "RPL17-C18orf32",
        "RPL17-C18orf32",
        "RPL17-C18orf32",
        "cd8-positive t-lymphocytes/immunology",
        "RPL17-C18orf32",
        "RPL17-C18orf32",
        "nf-e2-related factor 2/genetics",
        "RPL17-C18orf32",
        "amp-activated protein kinase kinases",
        "RPL17-C18orf32",
        "glycogen synthase kinase 3/metabolism",
        "RPL17-C18orf32",
        "lung neoplasms/drug therapy",
        "RPL17-C18orf32",
        "iron/chemistry",
        "RPL17-C18orf32",
        "tor serine-threonine kinases/metabolism",
        "RPL17-C18orf32",
        "oxidative stress/drug effects",
        "RPL17"
      ]
    },
    "count": {
      "expected_count": 165,
      "wrong_count_numbers": [
        1
      ]
    },
    "coverage": {
      "covered_sample_values": 3,
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
      "sentence_count": 2,
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
One matching paper was found for SLC40A1: "FerroptosisSuppressive Genes CorrelATE with Immunosuppression In GlIoblastoma." Representative terms include cancer and tumor biology, immune processes. Representative genes include SLC40A1. Representative diseases include glioblastoma. The paper count is 1. Representative terms mainly point to lipid metabolism, and iron-related biology. Representative genes provide direct graph evidence
```

Answer reward:

```text
{
  "reward": 0.695,
  "weighted_reward_before_penalty": 0.695,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 0.65,
    "count_accuracy": 1.0,
    "result_coverage": 0.0,
    "relation_semantics": 1.0,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [],
      "missing_required_values": [
        "34062295",
        "Ferroptosis Suppressive Genes Correlate with Immunosuppression in Glioblastoma.",
        "2021",
        "World Neurosurg"
      ]
    },
    "count": {
      "expected_count": 1,
      "wrong_count_numbers": []
    },
    "coverage": {
      "covered_values": 0,
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
      "sentence_count": 5,
      "token_count": 52
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
One matching paper was found: "FerroptosisSuppressive Genes CorrelATE with Immunosuppressionin Glioblastoma." is not in the response. Examples include "Ferroptosis Supervised Genes Correlate With Immunosuppression In Glioblastoma Is Not in the response. The returned items provide direct graph evidence for the requested relationship.
```

Answer reward:

```text
{
  "reward": 0.695,
  "weighted_reward_before_penalty": 0.695,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 0.65,
    "count_accuracy": 1.0,
    "result_coverage": 0.0,
    "relation_semantics": 1.0,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [],
      "missing_required_values": [
        "34062295",
        "Ferroptosis Suppressive Genes Correlate with Immunosuppression in Glioblastoma.",
        "2021",
        "World Neurosurg"
      ]
    },
    "count": {
      "expected_count": 1,
      "wrong_count_numbers": []
    },
    "coverage": {
      "covered_values": 0,
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
      "sentence_count": 3,
      "token_count": 42
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
Found 6 paper-linked gene-MeSH associations. The returned items provide direct graph evidence for the requested relationship.
```

Answer reward:

```text
{
  "reward": 0.795,
  "weighted_reward_before_penalty": 0.795,
  "hallucination_penalty": 0.0,
  "components": {
    "entity_faithfulness": 0.65,
    "count_accuracy": 1.0,
    "result_coverage": 0.5,
    "relation_semantics": 1.0,
    "format_cleanliness": 1.0,
    "fluency": 1.0
  },
  "details": {
    "entity": {
      "unsupported_pmids": [],
      "unsupported_gene_like_tokens": [],
      "missing_required_values": [
        "32814895",
        "GPX4",
        "melanoma/blood",
        "32814895",
        "ACSL3",
        "melanoma/blood",
        "31043744",
        "CD8A",
        "melanoma/genetics",
        "31043744",
        "SLC9A1",
        "melanoma/genetics",
        "31043744",
        "SLC7A11",
        "melanoma/genetics",
        "31043744",
        "SLC3A2",
        "melanoma/genetics"
      ]
    },
    "count": {
      "expected_count": 6,
      "wrong_count_numbers": []
    },
    "coverage": {
      "covered_sample_values": 0,
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
      "sentence_count": 2,
      "token_count": 16
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
For MeSH oxidative stress/drog effects and MeSH phosphoslipid hydroperoxide glutatheion peroxidase/metabolizm, Found 1 matching genes: ACSL3. Representative terms include oxidative stress, lipid metabolism. Representative PMIDs include 22109337. Representative mechanisms include antioxidant activity. Representative genes include ACSL3. Representative MeSH terms include oxidative stress, drug response. Representative journals include Antioxid Redox Signal. Representative
```

Answer reward:

```text
{
  "reward": 0.5825,
  "weighted_reward_before_penalty": 0.7325,
  "hallucination_penalty": 0.15,
  "components": {
    "entity_faithfulness": 0.65,
    "count_accuracy": 1.0,
    "result_coverage": 1.0,
    "relation_semantics": 0.0,
    "format_cleanliness": 1.0,
    "fluency": 0.75
  },
  "details": {
    "entity": {
      "unsupported_pmids": [
        "22109337"
      ],
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
      "bad_terms": [
        "mechanism"
      ]
    },
    "format": {
      "bad_meta_phrases": [],
      "artifact_count": 0
    },
    "fluency": {
      "sentence_count": 7,
      "token_count": 51
    }
  }
}
```
