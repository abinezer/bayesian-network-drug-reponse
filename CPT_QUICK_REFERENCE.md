# CPT Quick Reference Guide

## What is a CPT?

A **Conditional Probability Table (CPT)** stores `P(node | parents)` for each node in the Bayesian network.

---

## CPT Structure

### Format:
```json
"NodeName": {
  "(parent1_val, parent2_val, ...)": {
    "0": 0.75,  // P(node = 0 | parents)
    "1": 0.25   // P(node = 1 | parents)
  }
}
```

### Key Points:
- **Parent Configuration**: Tuple of parent node values (order matters!)
- **"0" column**: Probability node = 0 (inactive/absent)
- **"1" column**: Probability node = 1 (active/present)
- **Sum Rule**: For each row, `P(0) + P(1) = 1.0`

---

## How CPTs are Calculated

### Maximum Likelihood (ML)
```
For each parent configuration:
  P(node = value | config) = count(node=value, config) / count(config)
```

**Example**: 
- 100 cell lines with `RB1_mut=1, CCND1_amp=0`
- 60 have `CDK_Pathway=1`, 40 have `CDK_Pathway=0`
- → `P(CDK_Pathway=1 | RB1=1, CCND1=0) = 60/100 = 0.6`

**Limitation**: Requires observed data. Cannot learn hidden nodes.

### Expectation-Maximization (EM)
```
Iterate until convergence:
  E-step: Estimate hidden node values using parents + children
  M-step: Update CPTs using expected counts
```

**Example**:
- Hidden node `CDK_Pathway` not directly observed
- Infer from: (1) parent genes, (2) child DrugResponse
- Update probabilities iteratively

**Advantage**: Can learn hidden nodes and handle missing data.

---

## What Each Column Means

### Root Node (No Parents)
```json
"RB1_mut": {
  "()": {"0": 0.8867, "1": 0.1133}
}
```
- `"()"` = empty tuple = no parents
- `"0"` = P(RB1 mutation absent) = 88.67%
- `"1"` = P(RB1 mutation present) = 11.33%

### Node with Parents
```json
"CDK_Pathway": {
  "(1, 0, 0, 0, 1, 1, 1)": {"0": 0.4444, "1": 0.5556}
}
```
- `(1, 0, 0, 0, 1, 1, 1)` = parent configuration
  - Order: `[RB1_mut, CCND1_amp, CDK4_mut, CDK6_mut, CDKN2A_del, CDKN2B_del, TissueType]`
  - Values: `[1, 0, 0, 0, 1, 1, 1]` = RB1 mutated, CDKN2A deleted, CDKN2B deleted, tissue=1
- `"0"` = P(CDK_Pathway inactive | this config) = 44.44%
- `"1"` = P(CDK_Pathway active | this config) = 55.56%

### Target Node (DrugResponse)
```json
"DrugResponse": {
  "(0, 0, 0, 0)": {"0": 0.5, "1": 0.5}
}
```
- `(0, 0, 0, 0)` = all pathways inactive
  - Order: `[CDK_Pathway, Histone_Transcription, DNA_Damage_Response, GrowthFactor_Signaling]`
- `"0"` = P(Low drug response | all pathways inactive) = 50%
- `"1"` = P(High drug response | all pathways inactive) = 50%

---

## Questions You Can Answer

### 1. Prediction
**Q**: Given gene alterations, what's the probability of high drug response?
```
1. Set evidence: genes = observed values
2. Infer pathways: P(pathways | genes)
3. Predict: P(DrugResponse = 1 | pathways)
```

### 2. Diagnosis
**Q**: Patient has high response. Which genes are most likely mutated?
```
Compute: P(gene = 1 | DrugResponse = 1)
```

### 3. Causality
**Q**: How does RB1 mutation affect drug response?
```
Compare: P(DrugResponse = 1 | RB1_mut = 1) vs P(DrugResponse = 1 | RB1_mut = 0)
```

### 4. Pathway Importance
**Q**: Which pathway is most important for drug response?
```
For each pathway:
  Compare: P(DrugResponse = 1 | pathway = 1) vs P(DrugResponse = 1 | pathway = 0)
```

### 5. Sensitivity Analysis
**Q**: Which genes have strongest influence?
```
For each gene:
  sensitivity = |P(DrugResponse=1 | gene=1) - P(DrugResponse=1 | gene=0)|
```

### 6. Combination Effects
**Q**: Do RB1 and TP53 mutations work together?
```
Compare: P(DrugResponse=1 | RB1=1, TP53=1) vs P(DrugResponse=1 | RB1=1) × P(DrugResponse=1 | TP53=1)
```

---

## ML vs EM Summary

| Node Type | ML | EM |
|-----------|----|----|
| **Observed root** (genes) | ✅ Same as EM | ✅ Direct counting |
| **Observed with observed parents** | ✅ Direct counting | ✅ Same as ML |
| **Hidden nodes** (pathways) | ❌ Cannot learn | ✅ Infers from parents + children |
| **Nodes with hidden parents** (DrugResponse) | ❌ Cannot learn | ✅ Uses inferred hidden parents |

---

## Example Interpretations

### Example 1: RB1_mut (Root Node)
```
P(RB1_mut = 1) = 0.1133
```
→ 11.33% of cell lines have RB1 mutation

### Example 2: CDK_Pathway (Hidden Node)
```
P(CDK_Pathway = 1 | RB1=1, CCND1=0, CDK4=0, CDK6=0, CDKN2A=1, CDKN2B=1, Tissue=1) = 0.5556
```
→ When RB1 is mutated, CDKN2A and CDKN2B are deleted, and tissue type is 1:
   - 55.56% chance CDK pathway is active

### Example 3: DrugResponse (Target Node)
```
P(DrugResponse = 1 | all pathways inactive) = 0.5
```
→ When all pathways are inactive:
   - 50% chance of high drug response (baseline)

---

## Files

- **Full Explanation**: `cpt_explanation.md`
- **Interactive Examples**: Run `python3 explain_cpts_with_examples.py`
- **ML CPTs**: `cpts_ml_hidden_pathways.json`
- **EM CPTs**: `cpts_em_hidden_pathways.json`


