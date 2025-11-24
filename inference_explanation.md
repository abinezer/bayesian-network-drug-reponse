# How Inference Works in the Bayesian Network

## Overview

Inference in a Bayesian Network means computing the probability of unobserved (or query) variables given observed evidence. In our case, we want to predict `DrugResponse` given observed gene alterations.

## The Inference Problem

**Input**: Observed gene alterations (e.g., `RB1_mut=1`, `TP53_mut=0`, `EGFR_alt=1`, ...)

**Output**: Probability of high drug response `P(DrugResponse = 1 | observed_genes)`

**Challenge**: `DrugResponse` depends on pathway nodes, which are **hidden** (not directly observed). We must infer pathway states first.

---

## Step-by-Step Inference Process

### Step 1: Get Observed Evidence

For each test sample (cell line), we have:
- **Observed nodes**: Gene alterations (mutations, amplifications, deletions)
- **Query node**: `DrugResponse` (we want to predict this)
- **Hidden nodes**: Pathway nodes (`CDK_Pathway`, `Histone_Transcription`, `DNA_Damage_Response`, `GrowthFactor_Signaling`)

**Example**:
```
Observed: RB1_mut=1, CCND1_amp=0, CDK4_mut=0, TP53_mut=1, EGFR_alt=1, ...
Query: DrugResponse = ?
Hidden: CDK_Pathway = ?, Histone_Transcription = ?, ...
```

### Step 2: Infer Hidden Pathway Nodes

For each pathway node, we need to compute `P(pathway | observed_parents)`.

#### 2.1 Get Parent Configuration

Each pathway has gene parents. For example:
- `CDK_Pathway` parents: `[RB1_mut, CCND1_amp, CDK4_mut, CDK6_mut, CDKN2A_del, CDKN2B_del, TissueType]`

**Algorithm**:
```python
for pathway in [CDK_Pathway, Histone_Transcription, ...]:
    parent_values = []
    for parent in pathway.parents:
        if parent is observed:
            parent_values.append(observed_value[parent])
        elif parent is hidden:
            # Use most likely value from its CPT (simplified)
            parent_values.append(infer_hidden_parent(parent))
    
    parent_config = tuple(parent_values)
```

**Example**:
```
CDK_Pathway parents: [RB1_mut=1, CCND1_amp=0, CDK4_mut=0, CDK6_mut=0, 
                      CDKN2A_del=1, CDKN2B_del=0, TissueType=1]
→ parent_config = (1, 0, 0, 0, 1, 0, 1)
```

#### 2.2 Look Up Pathway CPT

Use the parent configuration to get `P(pathway = 1 | parent_config)` from the learned CPT:

```python
pathway_cpt = cpts[pathway]  # Loaded from JSON
pathway_probs = pathway_cpt[parent_config]
P(pathway = 1 | parent_config) = pathway_probs['1']
```

**Example**:
```json
"CDK_Pathway": {
  "(1, 0, 0, 0, 1, 0, 1)": {
    "0": 0.4444,  // P(CDK_Pathway = 0 | config)
    "1": 0.5556  // P(CDK_Pathway = 1 | config)
  }
}
```

So `P(CDK_Pathway = 1 | observed_genes) = 0.5556`

#### 2.3 Convert to Binary Value (for CPT lookup)

For simplicity, we use the most likely value:
```python
pathway_value = 1 if P(pathway = 1) > 0.5 else 0
```

**Note**: This is a **simplification**. Proper inference would:
- Use the full probability distribution (not just binary)
- Use belief propagation to handle dependencies between hidden nodes
- Consider all possible pathway configurations (not just the most likely)

### Step 3: Predict DrugResponse

Now that we have pathway values (or probabilities), we can predict `DrugResponse`.

#### 3.1 Get Pathway Configuration

```python
pathway_config = (CDK_Pathway_value, Histone_Transcription_value, 
                  DNA_Damage_Response_value, GrowthFactor_Signaling_value)
```

**Example**:
```
pathway_config = (1, 0, 1, 1)  # CDK active, Histone inactive, DNA active, Growth active
```

#### 3.2 Look Up DrugResponse CPT

```python
drug_cpt = cpts['DrugResponse']
drug_probs = drug_cpt[pathway_config]
P(DrugResponse = 1 | pathway_config) = drug_probs['1']
```

**Example**:
```json
"DrugResponse": {
  "(1, 0, 1, 1)": {
    "0": 0.3,  // P(Low response | pathways)
    "1": 0.7   // P(High response | pathways)
  }
}
```

So `P(DrugResponse = 1 | pathways) = 0.7` → **70% chance of high response**

#### 3.3 Make Prediction

```python
if P(DrugResponse = 1) > 0.5:
    prediction = 1  # High response
else:
    prediction = 0  # Low response
```

---

## Complete Example

Let's trace through a complete example:

### Input (Observed Genes)
```
RB1_mut = 1
CCND1_amp = 0
CDK4_mut = 0
CDK6_mut = 0
CDKN2A_del = 1
CDKN2B_del = 0
TP53_mut = 1
BRCA1_mut = 0
EGFR_alt = 1
TissueType = 1
... (other genes)
```

### Step 1: Infer CDK_Pathway

**Parent config**: `(RB1=1, CCND1=0, CDK4=0, CDK6=0, CDKN2A=1, CDKN2B=0, Tissue=1)`

**Look up CPT**:
```json
"(1, 0, 0, 0, 1, 0, 1)": {"0": 0.4444, "1": 0.5556}
```

**Result**: `P(CDK_Pathway = 1) = 0.5556` → Use value `1` (active)

### Step 2: Infer Histone_Transcription

**Parent config**: `(KAT6A=0, TBL1XR1=0, ..., TP53=1, ...)`

**Look up CPT**: Find matching configuration → `P(Histone_Transcription = 1) = 0.42`

**Result**: Use value `0` (inactive, since 0.42 < 0.5)

### Step 3: Infer DNA_Damage_Response

**Parent config**: `(TP53=1, BRCA1=0, BRCA2=0, ...)`

**Look up CPT**: `P(DNA_Damage_Response = 1) = 0.78`

**Result**: Use value `1` (active)

### Step 4: Infer GrowthFactor_Signaling

**Parent config**: `(EGFR=1, ERBB2=0, ...)`

**Look up CPT**: `P(GrowthFactor_Signaling = 1) = 0.65`

**Result**: Use value `1` (active)

### Step 5: Predict DrugResponse

**Pathway config**: `(CDK=1, Histone=0, DNA=1, Growth=1)` = `(1, 0, 1, 1)`

**Look up DrugResponse CPT**:
```json
"(1, 0, 1, 1)": {"0": 0.3, "1": 0.7}
```

**Result**: `P(DrugResponse = 1) = 0.7` → **Prediction: High response (1)**

---

## Implementation Details

### Current Implementation (Simplified)

The current code uses a **greedy inference** approach:

1. **For each pathway**:
   - Get observed parent values
   - Look up pathway CPT
   - Use most likely value (binary: 0 or 1)

2. **For DrugResponse**:
   - Use binary pathway values to form configuration
   - Look up DrugResponse CPT
   - Return probability

### Limitations

1. **Binary pathway values**: We use `argmax` (most likely value) instead of full probability distribution
2. **No belief propagation**: Hidden nodes are inferred independently
3. **Greedy approach**: Doesn't consider all possible pathway configurations

### More Sophisticated Inference (Future Work)

**Variable Elimination**:
- Sum over all possible pathway configurations
- Compute exact `P(DrugResponse | observed_genes)`

**Belief Propagation**:
- Pass messages between nodes
- Handle dependencies between hidden nodes

**Sampling (MCMC)**:
- Sample from joint distribution
- Estimate probabilities from samples

---

## Code Location

The inference is implemented in:
- **File**: `evaluate_bn.py`
- **Function**: `predict_drug_response()`
- **Lines**: 100-220

### Key Code Snippets

#### Inferring Pathway Values
```python
for pathway in drug_parents:
    if pathway in hidden_nodes:
        # Get parent configuration
        pathway_parent_vals = []
        for p in pathway_parents:
            if p in node_to_col:
                p_idx = node_to_col[p]
                pathway_parent_vals.append(int(data[i, p_idx]))
        
        # Look up CPT
        pathway_config = tuple(pathway_parent_vals)
        pathway_cpt = cpts.get(pathway, {})
        pathway_probs = pathway_cpt.get(pathway_config, {0: 0.5, 1: 0.5})
        
        # Use most likely value
        pathway_val = 1 if float(pathway_probs.get('1', 0.5)) > 0.5 else 0
        pathway_values.append(pathway_val)
```

#### Predicting DrugResponse
```python
# Form pathway configuration
pathway_config = tuple(pathway_values)

# Look up DrugResponse CPT
drug_cpt = cpts.get('DrugResponse', {})
drug_probs = drug_cpt.get(pathway_config, None)

if drug_probs:
    prob_drug_1 = float(drug_probs.get('1', 0.5))
else:
    # Fallback: use average pathway activation
    prob_drug_1 = np.mean(pathway_values)

# Make prediction
prediction = 1 if prob_drug_1 > 0.5 else 0
```

---

## Summary

**Inference Flow**:
```
Observed Genes → Infer Pathways → Predict DrugResponse
     ↓              ↓                    ↓
  Evidence      Hidden Nodes         Query Node
```

**Key Steps**:
1. Get observed gene values
2. For each pathway: look up CPT using parent gene configuration → get `P(pathway = 1)`
3. Convert pathway probabilities to binary values (most likely)
4. Form pathway configuration
5. Look up DrugResponse CPT → get `P(DrugResponse = 1)`
6. Make binary prediction (threshold at 0.5)

**Current Method**: Greedy, binary inference (simplified but fast)

**Future Improvement**: Full probabilistic inference with variable elimination or belief propagation

