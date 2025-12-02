# Bayesian Network for Palbociclib Drug Response Prediction

This project implements a Bayesian Network to predict drug response (sensitivity/resistance) to Palbociclib based on gene alterations and pathway activations.

## Overview

The model uses a hierarchical Bayesian Network structure:
- **Input Layer**: 20 gene alterations (mutations, amplifications, deletions)
- **Hidden Layer**: 4 pathway nodes (CDK Pathway, Histone Transcription, DNA Damage Response, Growth Factor Signaling)
- **Output Layer**: Drug response (sensitive/resistant to Palbociclib)

The key innovation is using **Expectation-Maximization (EM)** to learn hidden pathway nodes that are not directly observed in the data, enabling the model to capture biological mechanisms that connect gene alterations to drug response.

## Network Structure

The Bayesian Network has **26 nodes**:

- **20 Gene Nodes**: Root nodes representing gene alterations
  - Mutations: `RB1_mut`, `TP53_mut`, `CDK4_mut`, `CDK6_mut`, `BRCA1_mut`, `BRCA2_mut`, `CHEK1_mut`, `PIK3CA_mut`
  - Amplifications: `CCND1_amp`, `KAT6A_amp`, `TBL1XR1_amp`, `RUNX1_amp`, `MYC_amp`
  - Deletions: `CDKN2A_del`, `CDKN2B_del`
  - Alterations (mutation OR amplification): `CREBBP_alt`, `EP300_alt`, `HDAC1_alt`, `EGFR_alt`, `ERBB2_alt`

- **4 Pathway Nodes (Hidden)**: Intermediate nodes representing biological pathways
  - `CDK_Pathway`: Cell cycle regulation
  - `Histone_Transcription`: Histone modification and transcription
  - `DNA_Damage_Response`: DNA repair mechanisms
  - `GrowthFactor_Signaling`: Growth factor signaling

- **1 Tissue Node**: `TissueType` (binary tissue classification)

- **1 Target Node**: `DrugResponse` (binary: high/low response to Palbociclib)

## How the Model Works

### 1. Learning Conditional Probability Tables (CPTs)

Each node in the network has a **Conditional Probability Table (CPT)** that defines:
- `P(node = 1 | parent_configuration)` - probability the node is active given its parents
- `P(node = 0 | parent_configuration)` - probability the node is inactive given its parents

#### Maximum Likelihood (ML) Method

**Principle**: Direct counting from observed data.

**Algorithm**:
1. For each training sample, get the node value and its parent values
2. Count: `counts[parent_config][node_value] += 1`
3. Normalize: `P(node = value | parent_config) = counts[parent_config][value] / total[parent_config]`

**Limitation**: Can only learn CPTs for nodes that are **directly observed** in the data. Cannot learn hidden pathway nodes or nodes with hidden parents (like DrugResponse).

#### Expectation-Maximization (EM) Method

**Principle**: Iteratively infer hidden node values and update CPTs.

**Algorithm** (Multi-pass EM):
1. **Pass 1**: Learn observed root nodes (genes, tissue) using ML
2. **Pass 2**: Learn hidden pathway nodes using parent gene information
3. **Pass 3**: Learn DrugResponse using inferred pathway states
4. **Pass 4+**: Refine hidden nodes using child (DrugResponse) information

**E-Step (Expectation)**: For each training sample:
- Infer hidden pathway states from parent genes: `P(pathway | genes)`
- Use these probabilities to compute expected counts

**M-Step (Maximization)**: Update CPTs using expected counts:
- `P(node = value | config) = expected_counts[config][value] / expected_totals[config]`

**Key Innovation for DrugResponse**: Instead of hard thresholding pathway values, we use **probabilistic soft counting**:
- For each training sample, compute `P(pathway=1)` for all 4 pathways
- Iterate over all 16 possible pathway configurations
- For each configuration, compute its probability: `P(config) = P(pathway1=val1) × P(pathway2=val2) × P(pathway3=val3) × P(pathway4=val4)`
- Accumulate expected counts weighted by these probabilities
- This ensures all 16 configurations receive expected counts and can be learned

### 2. Inference (Prediction)

To predict drug response for a new cell line:

**Step 1**: Get observed gene alterations (input)

**Step 2**: Infer pathway states
- For each pathway, look up its CPT using parent gene configuration
- Get `P(pathway = 1 | genes)`

**Step 3**: Predict DrugResponse
- Build pathway configuration: `(CDK, Histone, DNA, Growth)`
- Look up DrugResponse CPT: `P(DrugResponse = 1 | pathway_config)`
- Return probability of high drug response

**Example**:
```
Input: RB1_mut=1, TP53_mut=1, EGFR_alt=1, ...
→ Infer: P(CDK_Pathway=1) = 0.6, P(Histone=1) = 0.4, P(DNA=1) = 0.7, P(Growth=1) = 0.8
→ Pathway config: (1, 0, 1, 1) (using most likely values)
→ Lookup: P(DrugResponse=1 | (1,0,1,1)) = 0.65
→ Prediction: 65% chance of high drug response
```

## Project Structure

### Core Code Files

- **`learn_bn_cpts.py`** - Base Bayesian Network class and utility functions
- **`learn_bn_cpts_hidden_pathways.py`** - Main script to learn CPTs using ML and EM
- **`evaluate_bn.py`** - Evaluate learned models on test set
- **`plot_evaluation_results.py`** - Generate evaluation plots (ROC curves, confusion matrices)
- **`create_bn_hidden_cpts_tables.py`** - Generate network visualizations with CPTs
- **`visualize_correlations.py`** - Generate correlation analysis plots

### Data Files

- **`final_project250A.dot`** - Network structure definition (Graphviz DOT format)
- **`Palbociclib/Palbociclib_train/`** - Training data
- **`Palbociclib/Palbociclib_test/`** - Test data

### Learned Models

- **`cpts_ml_hidden_pathways.json`** - CPTs learned using Maximum Likelihood
- **`cpts_em_hidden_pathways.json`** - CPTs learned using Expectation-Maximization

## Quick Start

### 1. Learn CPTs

```bash
conda activate cuda11_env
python3 learn_bn_cpts_hidden_pathways.py
```

This generates:
- `cpts_ml_hidden_pathways.json` (ML CPTs - only observed nodes)
- `cpts_em_hidden_pathways.json` (EM CPTs - observed + hidden nodes)

### 2. Visualize Network with CPTs

```bash
python3 create_bn_hidden_cpts_tables.py
dot -Tpng palbo_bn_hidden_em.dot -o palbo_bn_hidden_em.png
```

### 3. Evaluate Models

```bash
python3 evaluate_bn.py
```

Evaluates both ML and EM models on the test set and saves results to `evaluation_results.json`.

### 4. Generate Evaluation Plots

```bash
python3 plot_evaluation_results.py
```

Generates ROC curves, confusion matrices, and comparison plots.

## Results

### Model Performance (Test Set)

- **EM Model**: 
  - Accuracy: 47.6%
  - AUC-ROC: **0.5925** (better than random)
  - Can learn all 16 DrugResponse configurations

- **ML Model**: 
  - Accuracy: 47.6%
  - AUC-ROC: 0.5000 (random performance)
  - Cannot learn hidden pathway nodes or DrugResponse

**Key Finding**: EM outperforms ML because it can learn hidden pathway nodes that capture biological mechanisms connecting gene alterations to drug response.

### DrugResponse CPT

The EM model learns **all 16 possible pathway configurations** with diverse probabilities:

- `(0,0,0,0)` (all pathways inactive): P(high response) = 0.403
- `(1,1,1,1)` (all pathways active): P(high response) = 0.604
- Other configurations have intermediate probabilities

This enables nuanced predictions based on different pathway activation patterns.

## Understanding CPTs

### CPT Structure

A CPT maps each parent configuration to a probability distribution:

```json
"CDK_Pathway": {
  "(0, 0, 0, 0, 0, 0, 0)": {
    "0": 0.67,  // P(CDK_Pathway inactive | all genes normal)
    "1": 0.33   // P(CDK_Pathway active | all genes normal)
  },
  "(1, 0, 0, 0, 1, 1, 1)": {
    "0": 0.44,  // P(CDK_Pathway inactive | RB1 mutated, CDKN2A/B deleted)
    "1": 0.56   // P(CDK_Pathway active | RB1 mutated, CDKN2A/B deleted)
  }
}
```

**Interpretation**:
- Parent configuration `(0,0,0,0,0,0,0)` means: all parent genes are normal
- `"0"` = probability pathway is inactive
- `"1"` = probability pathway is active
- Probabilities sum to 1.0 for each configuration

### What Questions Can Be Answered?

1. **Prediction**: Given gene alterations, what's the probability of high drug response?
2. **Diagnosis**: Patient has high response. Which genes are most likely mutated?
3. **Causality**: How does RB1 mutation affect drug response?
4. **Sensitivity Analysis**: Which genes have the strongest influence?
5. **Combination Effects**: Do RB1 and TP53 mutations work together?

## Dependencies

- Python 3.x
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Graphviz (for network visualization)

## Key Concepts

### Why Hidden Nodes?

Pathway nodes are **not directly observed** in the data - we only have:
- Gene alterations (mutations, amplifications, deletions)
- Drug response (sensitive/resistant)

Pathway nodes represent **biological mechanisms** that connect genes to drug response. EM infers these hidden mechanisms from the data.

### Why EM Works Better Than ML

- **ML**: Can only learn from directly observed data → cannot learn hidden pathway nodes
- **EM**: Can infer hidden nodes from parent genes and child (DrugResponse) outcomes → learns biological mechanisms

### How DrugResponse Learning Works

The key innovation is **probabilistic soft counting**:
- Instead of hard thresholding pathway values (0 or 1), we use probabilities
- For each training sample, we compute probabilities for all 16 pathway configurations
- We accumulate expected counts weighted by these probabilities
- This ensures all configurations are learned, not just the most common one

## Citation

If you use this code, please cite the related work on CDK4/6 inhibitor response prediction and Bayesian Networks for drug response modeling.
