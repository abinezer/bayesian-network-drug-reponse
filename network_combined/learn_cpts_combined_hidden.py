#!/usr/bin/env python3
"""
Bayesian Network CPT Learning with TRULY HIDDEN pathway nodes
Pathway nodes are NOT computed - they are learned as latent variables using EM only
"""

import numpy as np
import json
from collections import defaultdict
import sys
import os

# Import functions from original script
sys.path.insert(0, os.path.dirname(__file__))
from learn_cpts_combined import (
    read_files, create_network_structure, map_gene_to_indices,
    BayesianNetwork
)

def create_data_matrix_hidden(mutations, amplifications, deletions, drug_response, 
                              gene_alteration_map, cell_map, structure):
    """
    Create data matrix for OBSERVED nodes only (genes, tissue, drug response)
    Pathway nodes are NOT computed - they are truly hidden
    """
    n_cells = len(mutations)
    n_genes = len(gene_alteration_map)
    
    # Create gene data matrix (ONLY observed nodes)
    gene_data = np.zeros((n_cells, n_genes), dtype=float)
    node_to_col = {}
    col_idx = 0
    
    # Add gene nodes (observed)
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
    
    # Add DrugResponse (observed, but has missing values)
    valid_responses = drug_response[~np.isnan(drug_response)]
    if len(valid_responses) > 0:
        median_response = np.median(valid_responses)
        node_to_col['DrugResponse'] = col_idx
        drug_binary = np.where(np.isnan(drug_response), np.nan, 
                              (drug_response > median_response).astype(float))
        gene_data = np.column_stack([gene_data, drug_binary])
    
    # NOTE: Pathway nodes are NOT added to node_to_col or gene_data
    # They are truly hidden and will only be learned via EM
    
    return gene_data, node_to_col

def main():
    data_dir = "/Users/GayathriRajesh/Desktop/UCSD 27/Fall 2025/250A/project/Palbociclib_train"
    
    # Load data
    amplifications, deletions, mutations, cell_map, gene_map, drug_response = read_files(data_dir)
    # gene_map, cell_map, mutations, amplifications, deletions, drug_response = load_data(data_dir)
    
    # Create network structure
    structure, gene_nodes = create_network_structure()
    print(f"\nNetwork structure created with {len(structure)} nodes")
    print(f"Gene nodes: {len(gene_nodes)}")

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
    print(f"Hidden nodes (not in data): {hidden_nodes}")
    
    # Map genes to indices
    node_idx_map, gene_alteration_map = map_gene_to_indices(gene_map, gene_nodes)
    print(f"\nMapped {len(node_idx_map)} genes to indices")
    
    # Create data matrix for OBSERVED nodes only (no pathway nodes)
    all_data, node_to_col = create_data_matrix_hidden(
        mutations, amplifications, deletions, drug_response,
        gene_alteration_map, cell_map, structure
    )
    
    print(f"\nData matrix shape: {all_data.shape}")
    print(f"Observed nodes in data: {list(node_to_col.keys())}")
    print(f"Hidden nodes (not in data): {list(hidden_nodes)}")
    
    # Create Bayesian Network
    bn_ml = BayesianNetwork(structure)
    bn_em = BayesianNetwork(structure)
    
    # Learn CPTs using Maximum Likelihood (only for observed nodes)
    print("\n" + "="*80)
    print("LEARNING CPTs USING MAXIMUM LIKELIHOOD")
    print("="*80)
    print("NOTE: ML can only learn CPTs for OBSERVED nodes")
    print("      Hidden pathway nodes will be skipped or fail")
    
    # Modified learn_all_cpts_ml to handle hidden nodes
    print("\nLearning CPTs using Maximum Likelihood...")
    for node in bn_ml.nodes:
        if node in hidden_nodes:
            print(f"  SKIPPING {node} (hidden node - no observations, cannot use ML)")
            # ML cannot learn hidden nodes - skip or use uniform
            bn_ml.cpts[node] = {}  # Empty CPT for hidden nodes
        elif node in node_to_col:
            # Check if all parents are observed (not hidden)
            parent_nodes = structure[node]
            has_hidden_parents = any(p in hidden_nodes for p in parent_nodes)
            
            if has_hidden_parents:
                print(f"  SKIPPING {node} (has hidden parents, cannot use ML)")
                bn_ml.cpts[node] = {}  # Cannot learn if parents are hidden
            else:
                print(f"  Learning CPT for {node} (observed, all parents observed)...")
                bn_ml.cpts[node] = bn_ml.learn_ml_cpt(node, all_data, node_to_col, {})
        else:
            print(f"  WARNING: {node} not in data and not marked as hidden")
            bn_ml.cpts[node] = {}
    print("Done!")
    
    # Learn CPTs using EM (works for both observed and hidden nodes)
    print("\n" + "="*80)
    print("LEARNING CPTs USING EXPECTATION MAXIMIZATION")
    print("="*80)
    print("NOTE: EM can learn CPTs for BOTH observed and hidden nodes")
    print("      Hidden nodes are inferred from parent genes and child DrugResponse")
    
    # Modified learn_all_cpts_em to handle hidden nodes
    # We need to create a wrapper that handles hidden nodes properly
    print("\nLearning CPTs using Expectation Maximization...")
    
    def get_children(node, structure):
        """Get all children of a node"""
        children = []
        for child, parents in structure.items():
            if node in parents:
                children.append(child)
        return children

        
    
    def learn_em_cpt_wrapper(node, data, node_idx_map, parent_idx_map, hidden_nodes, all_cpts, max_iter=50):
        """Wrapper for EM learning that handles hidden nodes using child information"""
        node_idx = node_idx_map.get(node, None)
        parent_nodes = structure[node]
        
        # Separate parents into observed and hidden
        parent_indices_observed = [node_idx_map.get(p, None) for p in parent_nodes if p in node_idx_map]
        parent_nodes_hidden = [p for p in parent_nodes if p in hidden_nodes]
        
        # If node is hidden, node_idx will be None
        is_hidden = (node_idx is None) or (node in hidden_nodes)
        
        # Find children of this node (for hidden nodes, use child info to infer)
        child_nodes = get_children(node, structure)
        child_indices = {}
        for child in child_nodes:
            if child in node_idx_map:
                child_indices[child] = node_idx_map[child]
        
        # Initialize CPT - use OR logic for hidden pathway nodes as prior
        cpt = defaultdict(lambda: {0: 0.5, 1: 0.5})
        
        # Get all unique parent configurations from observed data
        # For nodes with hidden parents, we'll infer them during EM iterations
        if parent_indices_observed and all(idx is not None for idx in parent_indices_observed):
            # Start with observed parent configs, will add hidden parent values during EM
            unique_configs = set()
            for i in range(len(data)):
                if all(not np.isnan(data[i, idx]) for idx in parent_indices_observed):
                    # For now, just track observed parent configs
                    # Hidden parents will be inferred and added during EM
                    config_obs = tuple(int(data[i, idx]) for idx in parent_indices_observed)
                    if parent_nodes_hidden:
                        # Will expand with hidden parent values during EM
                        unique_configs.add(config_obs)  # Placeholder
                    else:
                        unique_configs.add(config_obs)
        else:
            if parent_nodes_hidden:
                # Only hidden parents - will infer during EM
                unique_configs = set()
            else:
                unique_configs = {()}
        
        # Initialize with data-driven prior for hidden pathway nodes
        # Learn from correlation between parent configs and child (DrugResponse) outcomes
        for config in unique_configs:
            if is_hidden and node in ['CellCycleControl',
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
                # For pathway nodes, learn initial probabilities from data
                # Count how often this parent config appears with different child outcomes
                child_name = 'DrugResponse'  # Main child
                child_idx = node_idx_map.get(child_name, None)
                
                if child_idx is not None and config != ():
                    # Count occurrences of this parent config with child=0 vs child=1
                    count_child_0 = 0
                    count_child_1 = 0
                    total_count = 0
                    
                    for i in range(len(data)):
                        # Check if this data point matches the parent config
                        if parent_indices_observed and all(idx is not None for idx in parent_indices_observed):
                            parent_vals = []
                            skip = False
                            for idx in parent_indices_observed:
                                if np.isnan(data[i, idx]):
                                    skip = True
                                    break
                                parent_vals.append(int(data[i, idx]))
                            if skip:
                                continue
                            
                            if tuple(parent_vals) == config:
                                child_val = data[i, child_idx]
                                if not np.isnan(child_val):
                                    child_val = int(child_val)
                                    if child_val == 0:
                                        count_child_0 += 1
                                    else:
                                        count_child_1 += 1
                                    total_count += 1
                    
                    # Use data-driven initialization
                    if total_count > 0:
                        # If child=1 is more common, favor hidden=1; if child=0, favor hidden=0
                        # But use a soft prior (not too extreme)
                        p1 = (count_child_1 + 1) / (total_count + 2)  # Laplace smoothing
                        p0 = 1 - p1
                        # Constrain to reasonable range (0.1 to 0.9)
                        p1 = max(0.1, min(0.9, p1))
                        p0 = 1 - p1
                        cpt[config] = {0: p0, 1: p1}
                    else:
                        # No data for this config - use OR logic as fallback
                        if any(x == 1 for x in config):
                            cpt[config] = {0: 0.3, 1: 0.7}  # Moderate prior
                        else:
                            cpt[config] = {0: 0.6, 1: 0.4}  # Moderate prior
                else:
                    # No child data or no parents - use moderate OR logic
                    if config == ():
                        cpt[config] = {0: 0.5, 1: 0.5}
                    else:
                        if any(x == 1 for x in config):
                            cpt[config] = {0: 0.3, 1: 0.7}
                        else:
                            cpt[config] = {0: 0.6, 1: 0.4}
            else:
                # For other nodes, use uniform
                cpt[config] = {0: 0.5, 1: 0.5}
        
        # EM iterations
        for iteration in range(max_iter):
            old_cpt = {k: v.copy() for k, v in cpt.items()}
            
            # E-step: Compute expected counts
            expected_counts = defaultdict(lambda: defaultdict(float))
            expected_totals = defaultdict(float)

                            
            # Special handling for Proliferative Phenotype, Drug Response, DrugResponse Modulator: use probabilistic sampling over all pathway configurations
            if (node == 'Proliferative_Phenotype' and len(parent_nodes_hidden) == 5) or (node == 'DrugResponse_Modulator' and len(parent_nodes_hidden) == 3) or (node == 'DrugResponse' and len(parent_nodes_hidden) == 13):
                # DrugResponse has 4 hidden pathway parents - use probabilistic sampling
                # Initialize all 16 possible pathway configurations
                count = len(parent_nodes_hidden)
                count_raised_2 = 2**count
                for pathway_config_idx in range(count_raised_2):
                    pathway_config = tuple((pathway_config_idx >> j) & 1 for j in range(count))
                    unique_configs.add(pathway_config)
                    if pathway_config not in cpt:
                        cpt[pathway_config] = {0: 0.5, 1: 0.5}
                    # if any(pathway_config):
                    #     cpt[pathway_config] = {0: 0.4, 1: 0.6}
                    # else:
                    #     cpt[pathway_config] = {0: 0.6, 1: 0.4}
                
                for i in range(len(data)):
                    # Get observed parent values (none for DrugResponse, but check anyway)
                    skip = False
                    if parent_indices_observed and all(idx is not None for idx in parent_indices_observed):
                        for idx in parent_indices_observed:
                            if np.isnan(data[i, idx]):
                                skip = True
                                break
                    if skip:
                        continue
                    
                    # For each pathway, compute P(pathway=1 | its parents)
                    pathway_probs = []
                    for hidden_parent in parent_nodes_hidden:
                        hidden_parent_cpt = all_cpts.get(hidden_parent, {})
                        if hidden_parent_cpt:
                            # Get hidden parent's parents to determine its config
                            hidden_parent_parents = structure[hidden_parent]
                            hidden_parent_parent_indices = [node_idx_map.get(p, None) for p in hidden_parent_parents if p in node_idx_map]
                            
                            if hidden_parent_parent_indices and all(idx is not None for idx in hidden_parent_parent_indices):
                                hidden_parent_config_vals = []
                                for idx in hidden_parent_parent_indices:
                                    if np.isnan(data[i, idx]):
                                        skip = True
                                        break
                                    hidden_parent_config_vals.append(int(data[i, idx]))
                                if skip:
                                    break
                                
                                hidden_parent_config = tuple(hidden_parent_config_vals)
                                # Get probability of pathway=1
                                hidden_parent_probs = hidden_parent_cpt.get(hidden_parent_config, {0: 0.5, 1: 0.5})
                                prob_1 = float(hidden_parent_probs.get('1', hidden_parent_probs.get(1, 0.5)))
                                pathway_probs.append(prob_1)
                            else:
                                # No parents or missing - use marginal
                                hidden_parent_probs = hidden_parent_cpt.get((), {0: 0.5, 1: 0.5})
                                prob_1 = float(hidden_parent_probs.get('1', hidden_parent_probs.get(1, 0.5)))
                                pathway_probs.append(prob_1)
                        else:
                            # Hidden parent not learned yet - use uniform
                            pathway_probs.append(0.5)
                    
                    if skip or len(pathway_probs) != count:
                        continue
                    
                    # Now accumulate expected counts over ALL 16 possible pathway configurations
                    # Weight each configuration by its probability
                    node_val = data[i, node_idx] if node_idx is not None and not np.isnan(data[i, node_idx]) else None
                    
                    for pathway_config_idx in range(count_raised_2): 
                        pathway_config = tuple((pathway_config_idx >> j) & 1 for j in range(count))
                        
                        # Compute probability of this pathway configuration
                        config_prob = 1.0
                        for j, prob_1 in enumerate(pathway_probs):
                            if pathway_config[j] == 1:
                                config_prob *= prob_1
                            else:
                                config_prob *= (1.0 - prob_1)
                        
                        # Accumulate expected counts weighted by configuration probability
                        if node_val is not None:
                            # Observed value
                            expected_counts[pathway_config][int(node_val)] += config_prob
                            expected_totals[pathway_config] += config_prob
                        else:
                            # Missing value: use current CPT estimate
                            if pathway_config in cpt:
                                prob_1_current = float(cpt[pathway_config].get('1', cpt[pathway_config].get(1, 0.5)))
                                expected_counts[pathway_config][1] += config_prob * prob_1_current
                                expected_counts[pathway_config][0] += config_prob * (1.0 - prob_1_current)
                            else:
                                expected_counts[pathway_config][1] += config_prob * 0.5
                                expected_counts[pathway_config][0] += config_prob * 0.5
                            expected_totals[pathway_config] += config_prob
            else:
                # Standard handling for other nodes
                for i in range(len(data)):
                    # Get parent configuration (observed + inferred hidden)
                    parent_vals = []
                    skip = False
                    
                    # Get observed parent values
                    if parent_indices_observed and all(idx is not None for idx in parent_indices_observed):
                        for idx in parent_indices_observed:
                            if np.isnan(data[i, idx]):
                                skip = True
                                break
                            parent_vals.append(int(data[i, idx]))
                        if skip:
                            continue
                    
                    # Get hidden parent values (infer from their CPTs)
                    hidden_parent_vals = []
                    for hidden_parent in parent_nodes_hidden:
                        hidden_parent_cpt = all_cpts.get(hidden_parent, {})
                        if hidden_parent_cpt:
                            # Get hidden parent's parents to determine its config
                            hidden_parent_parents = structure[hidden_parent]
                            hidden_parent_parent_indices = [node_idx_map.get(p, None) for p in hidden_parent_parents if p in node_idx_map]
                            
                            if hidden_parent_parent_indices and all(idx is not None for idx in hidden_parent_parent_indices):
                                hidden_parent_config_vals = []
                                for idx in hidden_parent_parent_indices:
                                    if np.isnan(data[i, idx]):
                                        skip = True
                                        break
                                    hidden_parent_config_vals.append(int(data[i, idx]))
                                if skip:
                                    break
                                
                                hidden_parent_config = tuple(hidden_parent_config_vals)
                                # Use probabilistic sampling instead of hard threshold
                                hidden_parent_probs = hidden_parent_cpt.get(hidden_parent_config, {0: 0.5, 1: 0.5})
                                prob_1 = float(hidden_parent_probs.get('1', hidden_parent_probs.get(1, 0.5)))
                                # Sample probabilistically
                                hidden_parent_val = 1 if np.random.random() < prob_1 else 0
                                hidden_parent_vals.append(hidden_parent_val)
                            else:
                                # No parents or missing - use marginal
                                hidden_parent_probs = hidden_parent_cpt.get((), {0: 0.5, 1: 0.5})
                                prob_1 = float(hidden_parent_probs.get('1', hidden_parent_probs.get(1, 0.5)))
                                hidden_parent_val = 1 if np.random.random() < prob_1 else 0
                                hidden_parent_vals.append(hidden_parent_val)
                        else:
                            # Hidden parent not learned yet - use uniform
                            hidden_parent_vals.append(0)  # Default
                    
                    if skip:
                        continue
                    
                    # Combine observed and hidden parent values
                    # Order: observed parents first (in parent_nodes order), then hidden
                    all_parent_vals = []
                    for p in parent_nodes:
                        if p in node_idx_map:
                            p_idx = node_idx_map[p]
                            all_parent_vals.append(int(data[i, p_idx]))
                        else:
                            # Hidden parent - use inferred value
                            hidden_idx = parent_nodes_hidden.index(p)
                            all_parent_vals.append(hidden_parent_vals[hidden_idx])
                    
                    parent_config = tuple(all_parent_vals) if all_parent_vals else ()
                    
                    # Add to unique configs if new
                    if parent_config not in unique_configs:
                        unique_configs.add(parent_config)
                        if parent_config not in cpt:
                            cpt[parent_config] = {0: 0.5, 1: 0.5}
                    
                    # Get node value (if observed) or estimate (if hidden)
                    if is_hidden:
                        # Hidden node: use BOTH parent and child information
                        # P(hidden | parents, child) ∝ P(hidden | parents) * P(child | hidden, other_parents)
                        
                        # Start with parent-based probability
                        prob_1_parent = cpt[parent_config][1]
                        prob_0_parent = cpt[parent_config][0]
                    
                        # Update using child information if available
                        prob_1 = prob_1_parent
                        prob_0 = prob_0_parent
                        
                        # Use child information to refine estimate
                        # P(H | parents, child) ∝ P(H | parents) * P(child | H, other_parents_of_child)
                        if child_indices:
                            for child_name, child_idx in child_indices.items():
                                child_val = data[i, child_idx]
                                if not np.isnan(child_val):
                                    child_val = int(child_val)
                                    # Get child's CPT (if learned)
                                    child_cpt = all_cpts.get(child_name, {})
                                    if child_cpt:
                                        # Get child's parents (including this hidden node)
                                        child_parents = structure[child_name]
                                        
                                        # Build child's parent configuration (excluding this hidden node)
                                        child_parent_vals_other = []
                                        child_parent_vals_all = []
                                        for p in child_parents:
                                            if p == node:
                                                # This is the hidden node - we'll try both values
                                                child_parent_vals_all.append(None)  # Placeholder
                                            else:
                                                p_idx = node_idx_map.get(p, None)
                                                if p_idx is not None and not np.isnan(data[i, p_idx]):
                                                    val = int(data[i, p_idx])
                                                    child_parent_vals_other.append(val)
                                                    child_parent_vals_all.append(val)
                                                else:
                                                    # Missing parent - skip this child
                                                    child_parent_vals_all = None
                                                    break
                                        
                                        if child_parent_vals_all is not None:
                                            # Get child CPT probabilities for H=0 and H=1
                                            # Build configs: (other_parents, H=0) and (other_parents, H=1)
                                            # Note: order matters - need to match child_parents order
                                            config_h0 = []
                                            config_h1 = []
                                            for p in child_parents:
                                                if p == node:
                                                    config_h0.append(0)
                                                    config_h1.append(1)
                                                else:
                                                    p_idx = node_idx_map.get(p, None)
                                                    if p_idx is not None and not np.isnan(data[i, p_idx]):
                                                        config_h0.append(int(data[i, p_idx]))
                                                        config_h1.append(int(data[i, p_idx]))
                                            
                                            config_h0_tuple = tuple(config_h0) if config_h0 else ()
                                            config_h1_tuple = tuple(config_h1) if config_h1 else ()
                                            
                                            # Get P(child | H=0, other_parents) and P(child | H=1, other_parents)
                                            p_child_given_h0 = child_cpt.get(config_h0_tuple, {}).get(str(child_val), 0.5)
                                            p_child_given_h1 = child_cpt.get(config_h1_tuple, {}).get(str(child_val), 0.5)
                                            
                                            # Weight parent probabilities by child likelihood
                                            prob_1 = prob_1_parent * p_child_given_h1
                                            prob_0 = prob_0_parent * p_child_given_h0
                        
                        # Normalize
                        total = prob_0 + prob_1
                        if total > 0:
                            prob_0 /= total
                            prob_1 /= total
                        else:
                            prob_0 = prob_0_parent
                            prob_1 = prob_1_parent
                        
                        expected_counts[parent_config][1] += prob_1
                        expected_counts[parent_config][0] += prob_0
                        expected_totals[parent_config] += 1.0
                    else:
                        # Observed node
                        node_val = data[i, node_idx]
                        
                        if np.isnan(node_val):
                            # Missing value: use current CPT to estimate
                            prob_1 = cpt[parent_config][1]
                            prob_0 = cpt[parent_config][0]
                            
                            expected_counts[parent_config][1] += prob_1
                            expected_counts[parent_config][0] += prob_0
                            expected_totals[parent_config] += 1.0
                        else:
                            # Observed value
                            node_val = int(node_val)
                            expected_counts[parent_config][node_val] += 1.0
                            expected_totals[parent_config] += 1.0
            
            # M-step: Update CPT using expected counts
            for config in expected_totals:
                total = expected_totals[config]
                if total > 0:
                    cpt[config][0] = expected_counts[config].get(0, 0) / total
                    cpt[config][1] = expected_counts[config].get(1, 0) / total
                else:
                    cpt[config] = {0: 0.5, 1: 0.5}
            
            # Check convergence
            max_diff = 0.0
            for config in cpt:
                for val in [0, 1]:
                    diff = abs(cpt[config][val] - old_cpt.get(config, {0: 0.5, 1: 0.5})[val])
                    max_diff = max(max_diff, diff)
            
            if max_diff < 1e-6:
                break
        
        return dict(cpt)
    
    # Multi-pass EM: Learn in multiple iterations
    # Pass 1: Learn observed nodes (genes, tissue)
    # Pass 2: Learn hidden nodes (pathways) using parent info
    # Pass 3: Learn nodes with hidden parents (DrugResponse) using learned hidden node CPTs
    # Pass 4+: Iteratively refine hidden nodes using child info
    
    print("\n  Pass 1: Learning observed root nodes (genes, tissue)...")
    observed_root_nodes = [node for node in bn_em.nodes 
                          if node in node_to_col and 
                          not any(p in hidden_nodes for p in structure[node])]
    for node in observed_root_nodes:
        print(f"    {node}...")
        bn_em.cpts[node] = learn_em_cpt_wrapper(node, all_data, node_to_col, {}, hidden_nodes, bn_em.cpts, max_iter=50)
    
    print("\n  Pass 2: Learning hidden nodes (pathways) using parent info only...")
    for node in hidden_nodes:
        print(f"    {node}...")
        bn_em.cpts[node] = learn_em_cpt_wrapper(node, all_data, node_to_col, {}, hidden_nodes, bn_em.cpts, max_iter=50)
    
    print("\n  Pass 3: Learning nodes with hidden parents (DrugResponse)...")
    nodes_with_hidden_parents = [node for node in bn_em.nodes 
                                 if node in node_to_col and 
                                 any(p in hidden_nodes for p in structure[node])]
    
    # for node in bn_em.nodes:
    #     for p in structure[node]:
    #         print(node)
    #         if p in hidden_nodes:
    #             print("NOOOODE", node)
    for node in nodes_with_hidden_parents:
        print(f"    {node}...")
        bn_em.cpts[node] = learn_em_cpt_wrapper(node, all_data, node_to_col, {}, hidden_nodes, bn_em.cpts, max_iter=50)
    
    print("\n  Pass 4: Refining hidden nodes using child info...")
    for iteration in range(3):  # 3 refinement iterations
        print(f"    Refinement iteration {iteration + 1}...")
        for node in hidden_nodes:
            bn_em.cpts[node] = learn_em_cpt_wrapper(node, all_data, node_to_col, {}, hidden_nodes, bn_em.cpts, max_iter=10)
    
    print("Done!")
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON: ML vs EM")
    print("="*80)
    
    print("\nNodes learned by ML:")
    ml_learned = [node for node in bn_ml.nodes if bn_ml.cpts.get(node)]
    print(f"  Total: {len(ml_learned)}")
    for node in ml_learned:
        num_configs = len(bn_ml.cpts[node])
        print(f"    {node}: {num_configs} configurations")
    
    print("\nNodes learned by EM:")
    em_learned = [node for node in bn_em.nodes if bn_em.cpts.get(node)]
    print(f"  Total: {len(em_learned)}")
    for node in em_learned:
        num_configs = len(bn_em.cpts[node])
        print(f"    {node}: {num_configs} configurations")
    
    print("\nHidden nodes (pathway nodes):")
    for node in hidden_nodes:
        ml_has = len(bn_ml.cpts.get(node, {})) > 0
        em_has = len(bn_em.cpts.get(node, {})) > 0
        print(f"  {node}:")
        print(f"    ML learned: {ml_has}")
        print(f"    EM learned: {em_has}")
        if em_has:
            cpt = bn_em.cpts[node]
            print(f"    EM CPT has {len(cpt)} configurations")
            # Show first config
            if cpt:
                first_config = list(cpt.items())[0]
                print(f"    Sample: {first_config[0]} -> P(0)={first_config[1].get('0', 0):.4f}, P(1)={first_config[1].get('1', 0):.4f}")
    
    # Compare observed nodes
    print("\n" + "="*80)
    print("COMPARING OBSERVED NODES (ML vs EM)")
    print("="*80)
    
    observed_nodes = [node for node in bn_ml.nodes if node not in hidden_nodes]
    differences = []
    
    for node in observed_nodes:
        if node in node_to_col:
            cpt_ml = bn_ml.cpts.get(node, {})
            cpt_em = bn_em.cpts.get(node, {})
            
            if cpt_ml and cpt_em:
                # Compare CPTs
                max_diff = 0.0
                for config in set(list(cpt_ml.keys()) + list(cpt_em.keys())):
                    p0_ml = float(cpt_ml.get(config, {}).get('0', 0.5))
                    p0_em = float(cpt_em.get(config, {}).get('0', 0.5))
                    p1_ml = float(cpt_ml.get(config, {}).get('1', 0.5))
                    p1_em = float(cpt_em.get(config, {}).get('1', 0.5))
                    
                    diff = max(abs(p0_ml - p0_em), abs(p1_ml - p1_em))
                    max_diff = max(max_diff, diff)
                
                differences.append((node, max_diff))
    
    differences.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nObserved nodes with differences:")
    print(f"{'Node':<30} {'Max Difference':<15}")
    print("-" * 45)
    for node, diff in differences:
        print(f"{node:<30} {diff:<15.6f}")
    
    # Save results
    bn_ml.save_cpts("cpts_ml_hidden_pathways.json")
    bn_em.save_cpts("cpts_em_hidden_pathways.json")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print(f"  1. ML learned CPTs for {len(ml_learned)} nodes (observed only)")
    print(f"  2. EM learned CPTs for {len(em_learned)} nodes (observed + hidden)")
    print(f"  3. Hidden pathway nodes: {len([n for n in hidden_nodes if len(bn_em.cpts.get(n, {})) > 0])} learned by EM, 0 by ML")
    print("\nFiles saved:")
    print("  - cpts_ml_hidden_pathways.json (ML - only observed nodes)")
    print("  - cpts_em_hidden_pathways.json (EM - observed + hidden nodes)")

if __name__ == "__main__":
    main()
