#!/usr/bin/env python3
"""
Evaluate learned Bayesian Network on test set
Predicts DrugResponse given gene mutations and compares ML vs EM
"""

import numpy as np
import json
import sys
import os
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Import functions from learning script
sys.path.insert(0, os.path.dirname(__file__))
from learn_cpts import (
    read_files, create_network_structure, map_gene_to_indices,
    BayesianNetwork
)

def load_cpts(filename):
    """Load CPTs from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def create_test_data_matrix(mutations, amplifications, deletions, drug_response, 
                            gene_alteration_map, cell_map, structure, node_to_col_train):
    """
    Create data matrix for test set using the same node mapping as training
    """
    n_cells = len(mutations)
    n_genes = len(gene_alteration_map)
    
    # Create gene data matrix
    gene_data = np.zeros((n_cells, n_genes), dtype=float)
    node_to_col = {}
    col_idx = 0
    
    # Add gene nodes (same order as training)
    for node, (gene_idx, alt_type) in gene_alteration_map.items():
        node_to_col[node] = col_idx
        
        if alt_type == 'mut':
            gene_data[:, col_idx] = mutations[:, gene_idx]
        elif alt_type == 'amp':
            gene_data[:, col_idx] = amplifications[:, gene_idx]
        elif alt_type == 'del':
            gene_data[:, col_idx] = deletions[:, gene_idx]
        elif alt_type == 'alt':
            gene_data[:, col_idx] = np.maximum(mutations[:, gene_idx], 
                                             amplifications[:, gene_idx])
        col_idx += 1
    
    # # Add TissueType (same logic as training)
    # tissue_types = set()
    # for cell_idx, cell_name in cell_map.items():
    #     if '_' in cell_name:
    #         tissue = cell_name.split('_', 1)[1]
    #         tissue_types.add(tissue)
    
    # tissue_list = sorted(list(tissue_types))
    # node_to_col['TissueType'] = col_idx
    # tissue_data = np.zeros(n_cells)
    # for i, cell_idx in enumerate(range(len(cell_map))):
    #     if cell_idx in cell_map:
    #         cell_name = cell_map[cell_idx]
    #         if '_' in cell_name:
    #             tissue = cell_name.split('_', 1)[1]
    #             tissue_data[i] = 1 if tissue in tissue_list[:len(tissue_list)//2] else 0
    # gene_data = np.column_stack([gene_data, tissue_data])
    # col_idx += 1
    
    # Add DrugResponse (for evaluation - we'll predict this)
    # Use training median for binarization to match training data
    # Load training median from training data
    train_data_dir = "/Users/GayathriRajesh/Desktop/UCSD 27/Fall 2025/250A/project/Palbociclib_train"
    train_drug_response_dict = {}
    with open(f"{train_data_dir}/train_data.txt", 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                try:
                    train_drug_response_dict[parts[0]] = float(parts[2])
                except ValueError:
                    pass
    
    train_responses = np.array(list(train_drug_response_dict.values()))
    train_median = np.median(train_responses)
    
    valid_responses = drug_response[~np.isnan(drug_response)]
    if len(valid_responses) > 0:
        node_to_col['DrugResponse'] = col_idx
        drug_binary = np.where(np.isnan(drug_response), np.nan, 
                              (drug_response > train_median).astype(float))
        gene_data = np.column_stack([gene_data, drug_binary])
    
    return gene_data, node_to_col

def predict_drug_response(bn, cpts, data, node_to_col, structure, hidden_nodes):
    """
    Predict DrugResponse for each test sample using the learned CPTs
    """
    predictions = []
    probabilities = []
    true_labels = []
    
    drug_response_idx = node_to_col.get('DrugResponse', None)
    if drug_response_idx is None:
        print("ERROR: DrugResponse not found in test data")
        return None, None, None
    
    for i in range(len(data)):
        # Get true label
        true_val = data[i, drug_response_idx]
        if np.isnan(true_val):
            continue
        true_labels.append(int(true_val))
        
        # Predict DrugResponse using inference
        # DrugResponse depends on pathway nodes, which depend on genes
        # We need to infer pathway nodes first, then DrugResponse
        
        # Get parent configuration for DrugResponse
        # DrugResponse parents: "CDK_Overdrive","Chromatin_Remodeling_State","DNA_Repair_Capacity","RTK_PI3K_Signaling","RB_Pathway_Activity"
        drug_parents = structure['DrugResponse']
        
        # For each pathway parent, infer its value from its gene parents
        pathway_values = []
        for pathway in drug_parents:
            if pathway in hidden_nodes:
                # Infer pathway value from its parents
                pathway_parents = structure[pathway]
                pathway_parent_vals = []
                
                for p in pathway_parents:
                    if p in node_to_col:
                        p_idx = node_to_col[p]
                        if not np.isnan(data[i, p_idx]):
                            pathway_parent_vals.append(int(data[i, p_idx]))
                        else:
                            # Missing parent - skip this pathway
                            pathway_parent_vals = None
                            break
                    elif p in hidden_nodes:
                        # Hidden parent of hidden node - use most likely value from its CPT
                        # This is a simplification - proper inference would use belief propagation
                        pathway_parent_cpt = cpts.get(p, {})
                        if pathway_parent_cpt:
                            # Use marginal if available, otherwise uniform
                            marginal = pathway_parent_cpt.get((), {0: 0.5, 1: 0.5})
                            pathway_parent_vals.append(1 if marginal.get('1', 0.5) > 0.5 else 0)
                        else:
                            pathway_parent_vals = None
                            break
                
                if pathway_parent_vals is None:
                    # Can't infer - use uniform
                    pathway_values.append(0.5)  # Probability
                else:
                    # Get pathway CPT for this parent configuration
                    pathway_cpt = cpts.get(pathway, {})
                    parent_config = tuple(pathway_parent_vals)
                    
                    # Try to find matching config (handle string keys from JSON)
                    pathway_probs = None
                    if parent_config in pathway_cpt:
                        pathway_probs = pathway_cpt[parent_config]
                    else:
                        # Try string representation
                        config_str = str(parent_config)
                        if config_str in pathway_cpt:
                            pathway_probs = pathway_cpt[config_str]
                        else:
                            # Find closest match or use default
                            pathway_probs = {0: 0.5, 1: 0.5}
                    
                    # Use probability of pathway=1
                    pathway_values.append(float(pathway_probs.get('1', pathway_probs.get(1, 0.5))))
            else:
                # Pathway is observed (shouldn't happen in hidden case, but handle it)
                if pathway in node_to_col:
                    p_idx = node_to_col[pathway]
                    pathway_values.append(float(data[i, p_idx]))
                else:
                    pathway_values.append(0.5)
        
        # Now predict DrugResponse given pathway values
        # Convert pathway probabilities to most likely values for CPT lookup
        pathway_config = tuple(1 if p > 0.5 else 0 for p in pathway_values)
        
        drug_cpt = cpts.get('DrugResponse', {})
        
        # If DrugResponse CPT is empty (ML case) or only has one config, use pathway-based prediction
        if not drug_cpt or len(drug_cpt) == 0:
            # ML case: DrugResponse not learned (has hidden parents)
            # Use average pathway activation as proxy
            prob_drug_1 = np.mean(pathway_values)
        else:
            # Try exact match (handle both tuple and string keys)
            drug_probs = drug_cpt.get(pathway_config, None)
            if drug_probs is None:
                # Try string representation
                config_str = str(pathway_config)
                drug_probs = drug_cpt.get(config_str, None)
            
            if drug_probs is not None:
                # Exact match found
                prob_drug_1 = float(drug_probs.get('1', 0.5))
            else:
                # No exact match - use weighted interpolation
                if len(drug_cpt) > 0:
                    # Find closest matching configs by Hamming distance
                    weighted_sum = 0.0
                    total_weight = 0.0
                    
                    for cfg_key, probs in drug_cpt.items():
                        # Convert string keys back to tuples
                        if isinstance(cfg_key, str):
                            try:
                                cfg = eval(cfg_key)
                            except:
                                continue
                        else:
                            cfg = cfg_key
                        
                        if isinstance(cfg, tuple) and len(cfg) == len(pathway_config):
                            # Compute Hamming distance
                            distance = sum(abs(a - b) for a, b in zip(cfg, pathway_config))
                            # Weight by inverse distance (closer = higher weight)
                            weight = 1.0 / (distance + 1)
                            weighted_sum += float(probs.get('1', 0.5)) * weight
                            total_weight += weight
                    
                    if total_weight > 0:
                        prob_drug_1 = weighted_sum / total_weight
                    else:
                        # Fallback: use average pathway activation
                        prob_drug_1 = np.mean(pathway_values)
                else:
                    # No learned configs - use pathway average
                    prob_drug_1 = np.mean(pathway_values)
        
        # If still uniform, use pathway-based prediction
        if abs(prob_drug_1 - 0.5) < 0.01:
            prob_drug_1 = np.mean(pathway_values)
        
        predictions.append(1 if prob_drug_1 > 0.5 else 0)
        probabilities.append(prob_drug_1)
    
    return np.array(predictions), np.array(probabilities), np.array(true_labels)

def evaluate_model(cpts_file, test_data_dir, method_name, debug=False):
    """
    Evaluate a model on test set
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {method_name} model")
    print(f"{'='*80}")
    
    # Load CPTs
    cpts = load_cpts(cpts_file)
    print(f"Loaded CPTs for {len(cpts)} nodes")
    
    if debug:
        print(f"\nDebug: DrugResponse CPT has {len(cpts.get('DrugResponse', {}))} configurations")
        for pathway in ['CellCycleControl',
                                    'CDK_Overdrive',
                                    'RB_Pathway_Activity',
                                    'Chromatin_Remodeling_State',
                                    'Histone_Transcription',
                                    'Epigenetic_Dysregulation',
                                    'DNA_Repair_Capacity',
                                    'DNA_Damage_Response',
                                    'Genomic_Instability',
                                    'RTK_PI3K_Signaling',
                                    'GrowthFactor_Signaling',
                                    'Proliferative_Phenotype',
                                    'DrugResponse_Modulator']:
            print(f"  {pathway}: {len(cpts.get(pathway, {}))} configs")
    
    # Load test data (need to modify load_data or create wrapper)
    # The load_data function expects train_data.txt, so we'll load manually
    def load_test_data(data_dir):
        """Load test data (similar to load_data but uses test_data.txt)"""
        # Load gene map (format: idx\tgene)
        gene_map = {}
        with open(f"{data_dir}/gene2ind.txt", 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    idx, gene = parts[0], parts[1]
                    gene_map[gene] = int(idx)
        
        # Load cell map (format: idx\tcell)
        cell_map = {}
        with open(f"{data_dir}/cell2ind.txt", 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    idx, cell = parts[0], parts[1]
                    cell_map[int(idx)] = cell
        
        # Load mutations (format: comma-separated list per line)
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
        
        # Load drug response (test_data.txt format: cell_name\tSMILES\tresponse\t...)
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
        
        # Align drug response with cell indices
        drug_response = np.full(n_cells, np.nan)
        for cell_idx, cell_name in cell_map.items():
            if cell_name in drug_response_dict:
                drug_response[cell_idx] = drug_response_dict[cell_name]
        
        return gene_map, cell_map, mutations, amplifications, deletions, drug_response
    
    gene_map, cell_map, mutations, amplifications, deletions, drug_response = load_test_data(test_data_dir)
    print(f"Test data: {len(mutations)} cell lines")
    
    # Create network structure
    structure, gene_nodes = create_network_structure()
    
    # Map genes to indices (same as training)
    node_idx_map, gene_alteration_map = map_gene_to_indices(gene_map, gene_nodes)
    
    # Create test data matrix
    # We need the training node_to_col to match the order
    # For now, we'll create it the same way
    test_data, node_to_col = create_test_data_matrix(
        mutations, amplifications, deletions, drug_response,
        gene_alteration_map, cell_map, structure, {}
    )
    
    print(f"Test data matrix shape: {test_data.shape}")
    
    # Define hidden nodes
    hidden_nodes = {'CellCycleControl',
                                    'CDK_Overdrive',
                                    'RB_Pathway_Activity',
                                    'Chromatin_Remodeling_State',
                                    'Histone_Transcription',
                                    'Epigenetic_Dysregulation',
                                    'DNA_Repair_Capacity',
                                    'DNA_Damage_Response',
                                    'Genomic_Instability',
                                    'RTK_PI3K_Signaling',
                                    'GrowthFactor_Signaling',
                                    'Proliferative_Phenotype',
                                    'DrugResponse_Modulator'}
    
    # Predict
    predictions, probabilities, true_labels = predict_drug_response(
        None, cpts, test_data, node_to_col, structure, hidden_nodes
    )
    
    if predictions is None or len(predictions) == 0:
        print("ERROR: No predictions generated")
        return None
    
    # Evaluate
    accuracy = accuracy_score(true_labels, predictions)
    
    # AUC (handle case where all predictions are same class)
    try:
        auc = roc_auc_score(true_labels, probabilities)
    except ValueError:
        auc = 0.5  # Can't compute AUC if only one class
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Predictions: {len(predictions)} samples")
    print(f"  Predicted class distribution: {np.bincount(predictions)}")
    print(f"  True class distribution: {np.bincount(true_labels)}")
    
    return {
        'method': method_name,
        'accuracy': accuracy,
        'auc': auc,
        'predictions': predictions,
        'probabilities': probabilities,
        'true_labels': true_labels
    }

def main():
    test_data_dir = "/Users/GayathriRajesh/Desktop/UCSD 27/Fall 2025/250A/project/Palbociclib_test"
    
    # Evaluate ML model
    ml_results = evaluate_model(
        "cpts_ml_hidden_pathways.json",
        test_data_dir,
        "ML (Maximum Likelihood)",
        debug=True
    )
    
    # Evaluate EM model
    em_results = evaluate_model(
        "cpts_em_hidden_pathways.json",
        test_data_dir,
        "EM (Expectation Maximization)",
        debug=True
    )
    
    # Compare results
    if ml_results and em_results:
        print(f"\n{'='*80}")
        print("COMPARISON: ML vs EM")
        print(f"{'='*80}")
        print(f"{'Metric':<20} {'ML':<15} {'EM':<15} {'Difference':<15}")
        print("-" * 65)
        print(f"{'Accuracy':<20} {ml_results['accuracy']:<15.4f} {em_results['accuracy']:<15.4f} {em_results['accuracy'] - ml_results['accuracy']:<15.4f}")
        print(f"{'AUC-ROC':<20} {ml_results['auc']:<15.4f} {em_results['auc']:<15.4f} {em_results['auc'] - ml_results['auc']:<15.4f}")
        
        # Save results
        results = {
            'ml': {
                'accuracy': float(ml_results['accuracy']),
                'auc': float(ml_results['auc'])
            },
            'em': {
                'accuracy': float(em_results['accuracy']),
                'auc': float(em_results['auc'])
            }
        }
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to evaluation_results.json")

if __name__ == "__main__":
    main()
