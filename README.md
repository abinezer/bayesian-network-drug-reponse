# Bayesian Network for Palbociclib Drug Response Prediction

This project implements a Bayesian Network to predict drug response (sensitivity/resistance) to Palbociclib based on gene alterations and pathway activations.

## Project Structure

### Core Code Files

- **`learn_bn_cpts.py`** - Base Bayesian Network class and utility functions (imported by other scripts)
- **`learn_bn_cpts_hidden_pathways.py`** - Main script to learn CPTs using ML and EM with truly hidden pathway nodes
- **`evaluate_bn.py`** - Evaluate learned models on test set (computes accuracy and AUC-ROC)
- **`plot_evaluation_results.py`** - Generate evaluation plots (ROC curves, confusion matrices, etc.)
- **`create_bn_hidden_cpts_tables.py`** - Generate DOT files with CPTs visualized as HTML tables
- **`explain_cpts_with_examples.py`** - Interactive script to explain CPTs with concrete examples
- **`visualize_correlations.py`** - Generate correlation analysis plots (Pearson correlations, pathway analysis)
- **`diagnose_correlation.py`** - Diagnostic script to analyze prediction quality

### Data Files

- **`final_project250A.dot`** - Network structure definition (Graphviz DOT format)
- **`Palbociclib/Palbociclib_train/`** - Training data (mutations, amplifications, deletions, drug response)
- **`Palbociclib/Palbociclib_test/`** - Test data for evaluation

### Learned Models

- **`cpts_ml_hidden_pathways.json`** - CPTs learned using Maximum Likelihood (ML)
- **`cpts_em_hidden_pathways.json`** - CPTs learned using Expectation-Maximization (EM)

### Visualizations

- **`palbo_bn.png`** - Original network structure visualization
- **`palbo_bn_hidden_ml.dot`** / **`palbo_bn_hidden_ml.png`** - Network with ML CPTs
- **`palbo_bn_hidden_em.dot`** / **`palbo_bn_hidden_em.png`** - Network with EM CPTs
- **`evaluation_comparison.png`** - Bar charts comparing ML vs EM performance
- **`evaluation_roc_curves.png`** - ROC curves for both models
- **`evaluation_confusion_matrices.png`** - Confusion matrices
- **`evaluation_prediction_distributions.png`** - Probability distributions
- **`evaluation_comprehensive.png`** - All evaluation plots in one figure
- **`correlation_analysis_ml.png`** - Correlation analysis for ML model
- **`correlation_analysis_em.png`** - Correlation analysis for EM model
- **`correlation_comparison_ml_vs_em.png`** - Side-by-side correlation comparison

### Documentation

- **`cpt_explanation.md`** - Comprehensive explanation of CPTs, ML/EM algorithms, and what questions can be answered
- **`CPT_QUICK_REFERENCE.md`** - Quick reference guide for CPT structure and interpretation
- **`inference_explanation.md`** - Detailed explanation of how inference works in the Bayesian Network (step-by-step process, examples, implementation details)
- **`evaluation_results.json`** - Evaluation metrics (accuracy, AUC-ROC) for ML and EM models

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
```

This generates:
- `palbo_bn_hidden_ml.dot` / `palbo_bn_hidden_ml.png`
- `palbo_bn_hidden_em.dot` / `palbo_bn_hidden_em.png`

### 3. Evaluate Models

```bash
python3 evaluate_bn.py
```

This evaluates both ML and EM models on the test set and saves results to `evaluation_results.json`.

### 4. Generate Evaluation Plots

```bash
python3 plot_evaluation_results.py
```

This generates all evaluation visualization files.

### 5. Understand CPTs

```bash
python3 explain_cpts_with_examples.py
```

This prints detailed explanations of CPTs with concrete examples from your data.

### 6. Visualize Correlations

```bash
python3 visualize_correlations.py
```

This generates correlation analysis plots showing how predictions correlate with true labels and pathway activations.

### 7. Understand Inference

Read `inference_explanation.md` for a detailed explanation of:
- How inference works step-by-step
- How hidden pathway nodes are inferred from gene alterations
- How DrugResponse is predicted from pathway states
- Complete worked examples
- Implementation details and limitations

## Network Structure

The Bayesian Network has 26 nodes:

- **20 Gene Nodes**: Mutations, amplifications, deletions (e.g., `RB1_mut`, `TP53_mut`, `EGFR_alt`)
- **4 Pathway Nodes** (Hidden): `CDK_Pathway`, `Histone_Transcription`, `DNA_Damage_Response`, `GrowthFactor_Signaling`
- **1 Tissue Node**: `TissueType`
- **1 Target Node**: `DrugResponse` (sensitive/resistant to Palbociclib)

## Key Results

- **EM Model Performance**: 59.0% accuracy, 0.610 AUC-ROC
- **ML Model Performance**: 47.6% accuracy, 0.500 AUC-ROC
- **Improvement**: EM outperforms ML by 11.4% accuracy because it can learn hidden pathway nodes

## Dependencies

- Python 3.x
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Graphviz (for network visualization)

## Documentation

### Understanding CPTs
- **`cpt_explanation.md`** - Comprehensive guide to Conditional Probability Tables
- **`CPT_QUICK_REFERENCE.md`** - Quick reference for CPT structure

### Understanding Inference
- **`inference_explanation.md`** - Step-by-step explanation of how the Bayesian Network performs inference to predict drug response

### Key Concepts
- How CPTs are calculated (ML vs EM)
- What each column means
- How inference works (genes → pathways → drug response)
- What questions can be answered from the BN

