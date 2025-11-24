# Bayesian Network CPTs: How They Work and What They Mean

## Table of Contents
1. [What is a CPT?](#what-is-a-cpt)
2. [How CPTs are Calculated](#how-cpts-are-calculated)
3. [Understanding CPT Structure](#understanding-cpt-structure)
4. [Maximum Likelihood (ML) Method](#maximum-likelihood-ml-method)
5. [Expectation-Maximization (EM) Method](#expectation-maximization-em-method)
6. [What Questions Can Be Answered?](#what-questions-can-be-answered)

---

## What is a CPT?

A **Conditional Probability Table (CPT)** defines the probability distribution of a node (variable) given the states of its parent nodes in the Bayesian network.

**Key Concept**: For each unique combination of parent values, the CPT specifies:
- `P(node = 0 | parent_configuration)` 
- `P(node = 1 | parent_configuration)`

These probabilities must sum to 1.0 for each parent configuration.

---

## Understanding CPT Structure

### Example 1: Root Node (No Parents)

For a root node like `RB1_mut` (no parents), the CPT has only one row:

```json
"RB1_mut": {
  "()": {
    "0": 0.8867,  // P(RB1_mut = 0) = 88.67%
    "1": 0.1133   // P(RB1_mut = 1) = 11.33%
  }
}
```

**What this means:**
- `"()"` = empty tuple = no parent configuration (root node)
- `"0"` = probability that RB1 mutation is absent (0)
- `"1"` = probability that RB1 mutation is present (1)
- **Interpretation**: In the training data, 88.67% of cell lines had no RB1 mutation, 11.33% had RB1 mutation

### Example 2: Node with Parents

For a pathway node like `CDK_Pathway` with 7 parents, the CPT has many rows:

```json
"CDK_Pathway": {
  "(0, 0, 0, 0, 0, 0, 0)": {
    "0": 0.6667,  // P(CDK_Pathway = 0 | all parents = 0)
    "1": 0.3333   // P(CDK_Pathway = 1 | all parents = 0)
  },
  "(1, 0, 0, 0, 1, 1, 1)": {
    "0": 0.4444,  // P(CDK_Pathway = 0 | specific parent config)
    "1": 0.5556   // P(CDK_Pathway = 1 | specific parent config)
  },
  ...
}
```

**What this means:**
- Each tuple like `(0, 0, 0, 0, 0, 0, 0)` represents a **parent configuration**
- The order matches the parent nodes in the network structure
- For `CDK_Pathway`, parents are: `[RB1_mut, CCND1_amp, CDK4_mut, CDK6_mut, CDKN2A_del, CDKN2B_del, TissueType]`
- `(0, 0, 0, 0, 0, 0, 0)` means: all gene alterations absent, tissue type = 0
- `"0"` = probability that CDK pathway is inactive
- `"1"` = probability that CDK pathway is active

### Example 3: DrugResponse with 4 Pathway Parents

```json
"DrugResponse": {
  "(0, 0, 0, 0)": {
    "0": 0.5,  // P(DrugResponse = Low | all pathways inactive)
    "1": 0.5   // P(DrugResponse = High | all pathways inactive)
  },
  "(1, 1, 1, 1)": {
    "0": 0.2,  // P(DrugResponse = Low | all pathways active)
    "1": 0.8   // P(DrugResponse = High | all pathways active)
  }
}
```

**What this means:**
- Parents are: `[CDK_Pathway, Histone_Transcription, DNA_Damage_Response, GrowthFactor_Signaling]`
- `(0, 0, 0, 0)` = all pathways inactive → 50/50 chance of high/low response
- `(1, 1, 1, 1)` = all pathways active → 80% chance of high response, 20% low

---

## How CPTs are Calculated

### Maximum Likelihood (ML) Method

**Principle**: Count how often each node value occurs for each parent configuration, then normalize.

#### Algorithm:

1. **For each data sample (cell line)**:
   - Get the node's value (0 or 1)
   - Get the parent configuration (tuple of parent values)
   - Increment count: `counts[parent_config][node_value] += 1`

2. **For each parent configuration**:
   - Calculate probabilities:
     ```
     P(node = 0 | parent_config) = count(node=0, parent_config) / total(parent_config)
     P(node = 1 | parent_config) = count(node=1, parent_config) / total(parent_config)
     ```

3. **Example Calculation**:

   Suppose we have 1000 cell lines for `RB1_mut`:
   - 887 have `RB1_mut = 0` (no mutation)
   - 113 have `RB1_mut = 1` (mutation)
   
   Then:
   ```
   P(RB1_mut = 0) = 887/1000 = 0.8867
   P(RB1_mut = 1) = 113/1000 = 0.1133
   ```

#### Limitations:
- **Requires observed data**: Can only learn CPTs for nodes that are directly observed in the data
- **Cannot handle hidden nodes**: If a node is not in the data (like pathway nodes), ML cannot learn its CPT
- **Cannot handle missing parents**: If a node's parents are hidden, ML cannot learn that node's CPT

#### Code Implementation:

```python
def learn_ml_cpt(self, node, data, node_idx_map, parent_idx_map):
    node_idx = node_idx_map[node]
    parent_nodes = self.structure[node]
    parent_indices = [node_idx_map[p] for p in parent_nodes]
    
    # Count occurrences
    counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(int)
    
    for i in range(len(data)):
        # Get parent configuration
        parent_vals = [int(data[i, idx]) for idx in parent_indices]
        parent_config = tuple(parent_vals)
        
        # Get node value
        node_val = int(data[i, node_idx])
        
        # Count
        counts[parent_config][node_val] += 1
        total_counts[parent_config] += 1
    
    # Build CPT (normalize counts to probabilities)
    cpt = {}
    for parent_config in counts:
        total = total_counts[parent_config]
        cpt[parent_config] = {
            0: counts[parent_config].get(0, 0) / total,
            1: counts[parent_config].get(1, 0) / total
        }
    
    return cpt
```

---

### Expectation-Maximization (EM) Method

**Principle**: Iteratively estimate probabilities when some nodes are hidden or have missing values.

#### Algorithm:

**E-Step (Expectation)**: For each data sample, estimate the expected value of hidden nodes:
- If node is **observed**: use the actual value
- If node is **hidden**: infer its value using:
  1. Parent information: `P(hidden | parents)` from current CPT
  2. Child information: `P(hidden | child)` from child's CPT
  3. Combine: `P(hidden | parents, child) ∝ P(hidden | parents) × P(child | hidden, other_parents)`

**M-Step (Maximization)**: Update CPTs using expected counts:
- For each parent configuration, compute expected counts:
  ```
  expected_count[config][value] = sum over all samples of P(node=value | config, data)
  ```
- Normalize to get probabilities:
  ```
  P(node = value | config) = expected_count[config][value] / total[config]
  ```

#### Multi-Pass Strategy:

1. **Pass 1**: Learn observed root nodes (genes, tissue) - these have no hidden parents
2. **Pass 2**: Learn hidden nodes (pathways) using only parent information
3. **Pass 3**: Learn nodes with hidden parents (DrugResponse) using learned pathway CPTs
4. **Pass 4+**: Refine hidden nodes using child information (DrugResponse outcomes)

#### Example: Learning CDK_Pathway (Hidden Node)

**Initialization**: Use data-driven prior based on correlation with DrugResponse:
- If parent genes suggest pathway should be active, initialize `P(active) = 0.7`
- Otherwise, `P(active) = 0.3`

**E-Step** (for each cell line):
```
1. Get parent gene values: (RB1_mut=1, CCND1_amp=0, CDK4_mut=0, ...)
2. Get child value: DrugResponse = 1 (high response)
3. Compute:
   P(CDK_Pathway=1 | parents, child) ∝ 
     P(CDK_Pathway=1 | parents) × P(DrugResponse=1 | CDK_Pathway=1, other_pathways)
4. Use this probability as "soft count"
```

**M-Step**:
```
For each parent configuration:
  expected_count[config][1] = sum of P(CDK_Pathway=1 | config, data) over all samples
  P(CDK_Pathway=1 | config) = expected_count[config][1] / total_samples_with_config
```

**Iteration**: Repeat E-step and M-step until convergence (probabilities stop changing).

#### Code Implementation (Simplified):

```python
def learn_em_cpt_wrapper(node, data, node_idx_map, hidden_nodes, all_cpts, max_iter=50):
    # Initialize CPT
    cpt = defaultdict(lambda: {0: 0.5, 1: 0.5})
    
    for iteration in range(max_iter):
        # E-Step: Compute expected counts
        expected_counts = defaultdict(lambda: defaultdict(float))
        expected_totals = defaultdict(float)
        
        for i in range(len(data)):
            # Get parent configuration (observed + inferred hidden)
            parent_config = get_parent_config(data[i], parents, hidden_nodes, all_cpts)
            
            # If node is hidden, infer its value using parent + child info
            if node in hidden_nodes:
                prob_1 = infer_hidden_probability(node, data[i], parent_config, all_cpts)
                prob_0 = 1 - prob_1
                
                expected_counts[parent_config][1] += prob_1
                expected_counts[parent_config][0] += prob_0
            else:
                # Observed node: use actual value
                node_val = int(data[i, node_idx])
                expected_counts[parent_config][node_val] += 1.0
            
            expected_totals[parent_config] += 1.0
        
        # M-Step: Update CPT
        for config in expected_totals:
            total = expected_totals[config]
            cpt[config][0] = expected_counts[config].get(0, 0) / total
            cpt[config][1] = expected_counts[config].get(1, 0) / total
        
        # Check convergence
        if probabilities_stopped_changing(cpt, old_cpt):
            break
    
    return cpt
```

#### Advantages:
- **Can learn hidden nodes**: Infers pathway states from gene alterations and drug response
- **Handles missing data**: Uses expected values instead of discarding samples
- **Uses child information**: For hidden nodes, leverages downstream effects (e.g., DrugResponse)

---

## What Questions Can Be Answered?

### 1. **Prediction Questions**

**Q: Given a cell line's gene alterations, what is the probability of high drug response?**

**Answer**: Use inference (variable elimination or sampling):
```
1. Set evidence: gene nodes = observed values
2. Infer hidden pathway states: P(pathways | genes)
3. Predict: P(DrugResponse = 1 | pathways, genes)
```

**Example**:
- Input: `RB1_mut=1, TP53_mut=1, EGFR_alt=1, ...`
- Infer: `P(CDK_Pathway=1 | ...) = 0.75`, `P(DNA_Damage_Response=1 | ...) = 0.82`
- Predict: `P(DrugResponse=1 | pathways) = 0.68` → **68% chance of high response**

### 2. **Diagnostic Questions**

**Q: A patient has high drug response. Which gene alterations are most likely?**

**Answer**: Use backward inference (diagnostic reasoning):
```
P(gene = 1 | DrugResponse = 1) ∝ P(DrugResponse = 1 | gene, pathways) × P(gene = 1)
```

**Example**:
- Given: `DrugResponse = 1` (high response)
- Compute: `P(RB1_mut=1 | DrugResponse=1)`, `P(TP53_mut=1 | DrugResponse=1)`, etc.
- Result: Most likely genes are those with high posterior probability

### 3. **Causal Questions**

**Q: How does RB1 mutation affect drug response?**

**Answer**: Compare probabilities:
```
P(DrugResponse = 1 | RB1_mut = 1) vs P(DrugResponse = 1 | RB1_mut = 0)
```

**Example**:
- `P(DrugResponse=1 | RB1_mut=1) = 0.65`
- `P(DrugResponse=1 | RB1_mut=0) = 0.45`
- **Conclusion**: RB1 mutation increases probability of high response by 20%

### 4. **Pathway Analysis Questions**

**Q: Which pathway is most important for drug response?**

**Answer**: Compare pathway activations:
```
For each pathway:
  P(DrugResponse = 1 | pathway = 1) vs P(DrugResponse = 1 | pathway = 0)
```

**Example**:
- `P(DrugResponse=1 | CDK_Pathway=1) = 0.72`
- `P(DrugResponse=1 | CDK_Pathway=0) = 0.38`
- **Conclusion**: CDK pathway activation strongly predicts high response

### 5. **Intervention Questions**

**Q: If we could activate CDK_Pathway, what would be the effect on drug response?**

**Answer**: Use do-calculus (intervention):
```
P(DrugResponse = 1 | do(CDK_Pathway = 1))
```

This is different from conditioning because we're forcing the pathway to be active, not just observing it.

### 6. **Sensitivity Analysis**

**Q: Which genes have the strongest influence on drug response?**

**Answer**: Compute sensitivity:
```
For each gene:
  sensitivity = |P(DrugResponse=1 | gene=1) - P(DrugResponse=1 | gene=0)|
```

**Example**:
- `RB1_mut`: sensitivity = |0.65 - 0.45| = 0.20
- `TP53_mut`: sensitivity = |0.58 - 0.52| = 0.06
- **Conclusion**: RB1 mutation has stronger influence

### 7. **Combination Effects**

**Q: Do RB1 and TP53 mutations have synergistic effects?**

**Answer**: Compare joint vs. individual effects:
```
P(DrugResponse=1 | RB1=1, TP53=1) vs 
P(DrugResponse=1 | RB1=1) × P(DrugResponse=1 | TP53=1)
```

If joint probability is much higher than product, there's synergy.

### 8. **Missing Data Imputation**

**Q: A cell line has missing drug response. What is the most likely value?**

**Answer**: Use inference:
```
P(DrugResponse = 1 | observed_genes, pathways) = 0.62
→ Most likely: DrugResponse = 1 (high response)
```

---

## Summary

### CPT Columns Explained:

1. **Parent Configuration (tuple)**: The values of all parent nodes
   - `()` = no parents (root node)
   - `(0, 1, 0)` = parent1=0, parent2=1, parent3=0

2. **"0" column**: Probability that the node = 0 (inactive/absent)
   - Always between 0 and 1
   - Sums to 1 with "1" column for each parent configuration

3. **"1" column**: Probability that the node = 1 (active/present)
   - Always between 0 and 1
   - Sums to 1 with "0" column for each parent configuration

### Key Differences: ML vs EM

| Aspect | ML | EM |
|--------|----|----|
| **Observed nodes** | ✅ Direct counting | ✅ Direct counting (same as ML) |
| **Hidden nodes** | ❌ Cannot learn | ✅ Infers from parents + children |
| **Missing parents** | ❌ Cannot learn child | ✅ Infers hidden parents |
| **Complexity** | O(n) - single pass | O(n × iterations) - iterative |
| **Accuracy** | Exact for observed | Approximate for hidden |

### What Makes This BN Powerful:

1. **Captures biological hierarchy**: Genes → Pathways → Drug Response
2. **Handles hidden variables**: Pathway states are inferred, not directly measured
3. **Probabilistic reasoning**: Provides uncertainty estimates, not just yes/no
4. **Causal structure**: Can answer "what if" questions about interventions
5. **Data integration**: Combines gene alterations, tissue type, and drug response

---

## References

- **Network Structure**: `final_project250A.dot`
- **ML CPTs**: `cpts_ml_hidden_pathways.json`
- **EM CPTs**: `cpts_em_hidden_pathways.json`
- **Learning Code**: `learn_bn_cpts_hidden_pathways.py`
- **Evaluation**: `evaluate_bn.py`


