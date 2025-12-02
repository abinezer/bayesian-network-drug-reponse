#!/usr/bin/env python3
"""
Visualize Pearson correlations on test set predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, roc_auc_score
import json
import sys
import os

# Import from evaluate_bn
sys.path.insert(0, os.path.dirname(__file__))
from evaluate_bn import (
    load_cpts, create_network_structure, map_genes_to_indices,
    create_test_data_matrix, predict_drug_response
)

def load_test_data(data_dir):
    """Load test data"""
    # Load gene map
    gene_map = {}
    with open(f"{data_dir}/gene2ind.txt", 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                idx, gene = parts[0], parts[1]
                gene_map[gene] = int(idx)
    
    # Load cell map
    cell_map = {}
    with open(f"{data_dir}/cell2ind.txt", 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                idx, cell = parts[0], parts[1]
                cell_map[int(idx)] = cell
    
    # Load mutations
    mutations = []
    with open(f"{data_dir}/cell2mutation.txt", 'r') as f:
        for line in f:
            mutations.append([int(x) for x in line.strip().split(',')])
    mutations = np.array(mutations)
    
    # Load amplifications
    amplifications = []
    with open(f"{data_dir}/cell2cnamplification.txt", 'r') as f:
        for line in f:
            amplifications.append([int(x) for x in line.strip().split(',')])
    amplifications = np.array(amplifications)
    
    # Load deletions
    deletions = []
    with open(f"{data_dir}/cell2cndeletion.txt", 'r') as f:
        for line in f:
            deletions.append([int(x) for x in line.strip().split(',')])
    deletions = np.array(deletions)
    
    # Load drug response
    n_cells = len(cell_map)
    drug_response_dict = {}
    with open(f"{data_dir}/test_data.txt", 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                cell_name = parts[0]
                try:
                    response = float(parts[2])
                    drug_response_dict[cell_name] = response
                except ValueError:
                    pass
    
    # Align drug response
    drug_response = np.full(n_cells, np.nan)
    for cell_idx, cell_name in cell_map.items():
        if cell_name in drug_response_dict:
            drug_response[cell_idx] = drug_response_dict[cell_name]
    
    return gene_map, cell_map, mutations, amplifications, deletions, drug_response

def get_predictions_with_pathways(cpts_file, test_data_dir):
    """Get predictions and inferred pathway values"""
    # Load CPTs
    cpts = load_cpts(cpts_file)
    
    # Load test data
    gene_map, cell_map, mutations, amplifications, deletions, drug_response = load_test_data(test_data_dir)
    
    # Create network structure
    structure, gene_nodes = create_network_structure()
    
    # Map genes to indices
    node_idx_map, gene_alteration_map = map_genes_to_indices(gene_map, gene_nodes)
    
    # Create test data matrix
    test_data, node_to_col = create_test_data_matrix(
        mutations, amplifications, deletions, drug_response,
        gene_alteration_map, cell_map, structure, {}
    )
    
    hidden_nodes = {'CDK_Pathway', 'Histone_Transcription', 'DNA_Damage_Response', 'GrowthFactor_Signaling'}
    
    # Get predictions and pathway values
    drug_response_idx = node_to_col.get('DrugResponse', None)
    predictions = []
    probabilities = []
    true_labels = []
    pathway_values_all = {pathway: [] for pathway in hidden_nodes}
    
    for i in range(len(test_data)):
        true_val = test_data[i, drug_response_idx]
        if np.isnan(true_val):
            continue
        true_labels.append(int(true_val))
        
        # Infer pathway values
        drug_parents = structure['DrugResponse']
        pathway_values = []
        pathway_dict = {}
        
        for pathway in drug_parents:
            if pathway in hidden_nodes:
                pathway_parents = structure[pathway]
                pathway_parent_vals = []
                
                for p in pathway_parents:
                    if p in node_to_col:
                        p_idx = node_to_col[p]
                        if not np.isnan(test_data[i, p_idx]):
                            pathway_parent_vals.append(int(test_data[i, p_idx]))
                        else:
                            pathway_parent_vals = None
                            break
                    elif p in hidden_nodes:
                        pathway_parent_cpt = cpts.get(p, {})
                        if pathway_parent_cpt:
                            marginal = pathway_parent_cpt.get((), {0: 0.5, 1: 0.5})
                            pathway_parent_vals.append(1 if marginal.get('1', 0.5) > 0.5 else 0)
                        else:
                            pathway_parent_vals = None
                            break
                
                if pathway_parent_vals is not None:
                    pathway_config = tuple(pathway_parent_vals)
                    pathway_cpt = cpts.get(pathway, {})
                    if pathway_cpt:
                        pathway_probs = pathway_cpt.get(pathway_config, {0: 0.5, 1: 0.5})
                        pathway_val = 1 if float(pathway_probs.get('1', 0.5)) > 0.5 else 0
                        pathway_prob = float(pathway_probs.get('1', 0.5))
                    else:
                        pathway_val = 0
                        pathway_prob = 0.5
                    
                    pathway_values.append(pathway_val)
                    pathway_dict[pathway] = pathway_prob
                else:
                    pathway_values.append(0)
                    pathway_dict[pathway] = 0.5
            else:
                # Observed pathway (shouldn't happen in our case)
                if pathway in node_to_col:
                    p_idx = node_to_col[pathway]
                    pathway_values.append(int(test_data[i, p_idx]))
                    pathway_dict[pathway] = float(test_data[i, p_idx])
        
        # Store pathway values
        for pathway in hidden_nodes:
            pathway_values_all[pathway].append(pathway_dict.get(pathway, 0.5))
        
        # Predict DrugResponse
        pathway_config = tuple(pathway_values)
        drug_cpt = cpts.get('DrugResponse', {})
        
        if not drug_cpt or len(drug_cpt) == 0:
            prob_drug_1 = np.mean(pathway_values)
        else:
            drug_probs = drug_cpt.get(pathway_config, None)
            if drug_probs is None:
                prob_drug_1 = np.mean(pathway_values)
            else:
                prob_drug_1 = float(drug_probs.get('1', 0.5))
        
        if abs(prob_drug_1 - 0.5) < 0.01:
            prob_drug_1 = np.mean(pathway_values)
        
        predictions.append(1 if prob_drug_1 > 0.5 else 0)
        probabilities.append(prob_drug_1)
    
    return (np.array(predictions), np.array(probabilities), np.array(true_labels),
            pathway_values_all)

def plot_correlations(probabilities, true_labels, pathway_values, method_name):
    """Plot correlation analysis"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Predicted vs True (scatter)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(true_labels, probabilities, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    corr, p_value = pearsonr(true_labels, probabilities)
    ax1.set_xlabel('True Drug Response (0=Low, 1=High)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted Probability (P(High Response))', fontsize=11, fontweight='bold')
    ax1.set_title(f'{method_name}: Predicted vs True\nPearson r = {corr:.3f} (p = {p_value:.2e})', 
                 fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Low', 'High'])
    
    # 2. Correlation heatmap: Pathways vs Drug Response
    ax2 = plt.subplot(2, 3, 2)
    pathway_names = list(pathway_values.keys())
    correlation_data = []
    for pathway in pathway_names:
        pathway_vals = np.array(pathway_values[pathway])
        corr, _ = pearsonr(true_labels, pathway_vals)
        correlation_data.append(corr)
    
    correlation_matrix = np.array([[corr] for corr in correlation_data])
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                xticklabels=['Drug Response'], yticklabels=pathway_names,
                cbar_kws={'label': 'Pearson r'}, ax=ax2, vmin=-1, vmax=1)
    ax2.set_title('Pathway-Drug Response Correlations', fontsize=12, fontweight='bold')
    
    # 3. Predicted probabilities distribution by true label
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(probabilities[true_labels == 0], bins=20, alpha=0.6, label='Low Response (True)', 
            color='#e74c3c', edgecolor='black', linewidth=1)
    ax3.hist(probabilities[true_labels == 1], bins=20, alpha=0.6, label='High Response (True)', 
            color='#2ecc71', edgecolor='black', linewidth=1)
    ax3.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, label='Decision Threshold')
    ax3.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Probability Distribution by True Label', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3, linestyle='--', axis='y')
    
    # 4. Pathway activation vs Drug Response
    ax4 = plt.subplot(2, 3, 4)
    pathway_activations = []
    pathway_labels = []
    for pathway in pathway_names:
        pathway_vals = np.array(pathway_values[pathway])
        pathway_activations.extend(pathway_vals)
        pathway_labels.extend([pathway] * len(pathway_vals))
    
    pathway_df = {'Pathway': pathway_labels, 'Activation': pathway_activations, 
                  'Drug Response': np.tile(true_labels, len(pathway_names))}
    
    # Box plot
    pathway_data_list = []
    pathway_name_list = []
    response_list = []
    for pathway in pathway_names:
        pathway_vals = np.array(pathway_values[pathway])
        for label in [0, 1]:
            pathway_data_list.extend(pathway_vals[true_labels == label])
            pathway_name_list.extend([pathway] * np.sum(true_labels == label))
            response_list.extend([label] * np.sum(true_labels == label))
    
    # Create grouped box plot
    positions = []
    data_for_box = []
    labels_for_box = []
    x_pos = 0
    for pathway in pathway_names:
        for label in [0, 1]:
            pathway_vals = np.array(pathway_values[pathway])
            subset = pathway_vals[true_labels == label]
            if len(subset) > 0:
                data_for_box.append(subset)
                positions.append(x_pos)
                labels_for_box.append(f'{pathway}\n{"Low" if label == 0 else "High"}')
                x_pos += 1
        x_pos += 0.5
    
    bp = ax4.boxplot(data_for_box, positions=positions, widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.7)
    ax4.set_xticks(positions[::2])
    ax4.set_xticklabels(pathway_names, rotation=45, ha='right')
    ax4.set_ylabel('Pathway Activation Probability', fontsize=11, fontweight='bold')
    ax4.set_title('Pathway Activation by Drug Response', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3, linestyle='--', axis='y')
    
    # 5. Correlation matrix: All pathways
    ax5 = plt.subplot(2, 3, 5)
    pathway_matrix = np.array([pathway_values[p] for p in pathway_names]).T
    corr_matrix = np.corrcoef(pathway_matrix.T)
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                xticklabels=pathway_names, yticklabels=pathway_names,
                cbar_kws={'label': 'Pearson r'}, ax=ax5, vmin=-1, vmax=1,
                square=True, linewidths=1)
    ax5.set_title('Inter-Pathway Correlations', fontsize=12, fontweight='bold')
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    
    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate correlations
    pred_corr, pred_p = pearsonr(true_labels, probabilities)
    pathway_corrs = []
    for pathway in pathway_names:
        corr, p = pearsonr(true_labels, np.array(pathway_values[pathway]))
        pathway_corrs.append((pathway, corr, p))
    
    summary_text = f"""
CORRELATION SUMMARY
{method_name}

Predicted vs True Response:
  Pearson r = {pred_corr:.4f}
  p-value = {pred_p:.2e}

Pathway Correlations with Drug Response:
"""
    for pathway, corr, p in sorted(pathway_corrs, key=lambda x: abs(x[1]), reverse=True):
        summary_text += f"  {pathway:25s} r = {corr:6.3f} (p = {p:.2e})\n"
    
    summary_text += f"\nAccuracy: {accuracy_score(true_labels, probabilities > 0.5):.4f}\n"
    try:
        summary_text += f"AUC-ROC: {roc_auc_score(true_labels, probabilities):.4f}"
    except:
        summary_text += "AUC-ROC: N/A"
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', fontweight='bold')
    
    plt.suptitle(f'Correlation Analysis: {method_name} on Test Set', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig

def main():
    test_data_dir = "/cellar/users/abishai/ClassProjects/cse250A/Palbociclib/Palbociclib_test"
    
    print("="*80)
    print("CORRELATION ANALYSIS ON TEST SET")
    print("="*80)
    
    # ML model
    print("\nAnalyzing ML model...")
    ml_pred, ml_prob, ml_true, ml_pathways = get_predictions_with_pathways(
        "cpts_ml_hidden_pathways.json", test_data_dir
    )
    
    print(f"  ML: {len(ml_true)} samples")
    fig_ml = plot_correlations(ml_prob, ml_true, ml_pathways, "ML (Maximum Likelihood)")
    fig_ml.savefig('correlation_analysis_ml.png', dpi=300, bbox_inches='tight')
    print("  Saved: correlation_analysis_ml.png")
    
    # EM model
    print("\nAnalyzing EM model...")
    em_pred, em_prob, em_true, em_pathways = get_predictions_with_pathways(
        "cpts_em_hidden_pathways.json", test_data_dir
    )
    
    print(f"  EM: {len(em_true)} samples")
    fig_em = plot_correlations(em_prob, em_true, em_pathways, "EM (Expectation Maximization)")
    fig_em.savefig('correlation_analysis_em.png', dpi=300, bbox_inches='tight')
    print("  Saved: correlation_analysis_em.png")
    
    # Combined comparison
    print("\nCreating combined comparison...")
    fig_combined = plt.figure(figsize=(16, 8))
    
    # ML side
    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(ml_true, ml_prob, alpha=0.6, s=50, edgecolors='black', linewidth=0.5, color='#e74c3c')
    corr_ml, p_ml = pearsonr(ml_true, ml_prob)
    ax1.set_xlabel('True Drug Response', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted Probability', fontsize=11, fontweight='bold')
    ax1.set_title(f'ML: r = {corr_ml:.3f} (p = {p_ml:.2e})', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Low', 'High'])
    
    # EM side
    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(em_true, em_prob, alpha=0.6, s=50, edgecolors='black', linewidth=0.5, color='#2ecc71')
    corr_em, p_em = pearsonr(em_true, em_prob)
    ax2.set_xlabel('True Drug Response', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Predicted Probability', fontsize=11, fontweight='bold')
    ax2.set_title(f'EM: r = {corr_em:.3f} (p = {p_em:.2e})', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Low', 'High'])
    
    plt.suptitle('ML vs EM: Correlation Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_combined.savefig('correlation_comparison_ml_vs_em.png', dpi=300, bbox_inches='tight')
    print("  Saved: correlation_comparison_ml_vs_em.png")
    
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - correlation_analysis_ml.png")
    print("  - correlation_analysis_em.png")
    print("  - correlation_comparison_ml_vs_em.png")
    
    plt.close('all')

if __name__ == "__main__":
    main()




