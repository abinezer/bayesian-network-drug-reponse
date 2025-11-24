#!/usr/bin/env python3
"""
Bayesian Network CPT Learning for Palbociclib Drug Response
Uses Maximum Likelihood and EM algorithms to learn CPTs
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import json

class BayesianNetwork:
    def __init__(self, structure):
        """
        Initialize Bayesian Network with structure
        structure: dict mapping node -> list of parents
        """
        self.structure = structure
        self.nodes = list(structure.keys())
        self.cpts = {}
        
    def learn_ml_cpt(self, node, data, node_idx_map, parent_idx_map):
        """
        Learn CPT using Maximum Likelihood estimation
        """
        node_idx = node_idx_map[node]
        parent_nodes = self.structure[node]
        parent_indices = [node_idx_map[p] for p in parent_nodes] if parent_nodes else []
        
        # Get node values from data
        node_values = data[:, node_idx]
        
        # Count occurrences for each parent configuration
        counts = defaultdict(lambda: defaultdict(int))
        total_counts = defaultdict(int)
        
        for i in range(len(data)):
            # Skip if node value is missing
            if np.isnan(node_values[i]):
                continue
            
            # Get parent configuration (skip if any parent is missing)
            if parent_indices:
                parent_vals = []
                skip = False
                for idx in parent_indices:
                    if np.isnan(data[i, idx]):
                        skip = True
                        break
                    parent_vals.append(int(data[i, idx]))
                if skip:
                    continue
                parent_config = tuple(parent_vals)
            else:
                parent_config = ()  # No parents
            
            node_val = int(node_values[i])
            counts[parent_config][node_val] += 1
            total_counts[parent_config] += 1
        
        # Build CPT
        cpt = {}
        for parent_config in counts:
            total = total_counts[parent_config]
            if total > 0:
                cpt[parent_config] = {
                    0: counts[parent_config].get(0, 0) / total,
                    1: counts[parent_config].get(1, 0) / total
                }
            else:
                # No data for this configuration - use uniform prior
                cpt[parent_config] = {0: 0.5, 1: 0.5}
        
        return cpt
    
    def learn_em_cpt(self, node, data, node_idx_map, parent_idx_map, max_iter=50, tol=1e-6):
        """
        Learn CPT using Expectation Maximization
        Handles missing data by iteratively estimating parameters
        """
        node_idx = node_idx_map[node]
        parent_nodes = self.structure[node]
        parent_indices = [node_idx_map[p] for p in parent_nodes] if parent_nodes else []
        
        # Initialize CPT with uniform probabilities
        cpt = defaultdict(lambda: {0: 0.5, 1: 0.5})
        
        # Get all unique parent configurations
        if parent_indices:
            unique_configs = set()
            for i in range(len(data)):
                if not np.isnan(data[i, node_idx]):  # Only use complete cases for configs
                    config = tuple(int(data[i, idx]) if not np.isnan(data[i, idx]) else -1 
                                  for idx in parent_indices)
                    unique_configs.add(config)
        else:
            unique_configs = {()}
        
        # Initialize with uniform
        for config in unique_configs:
            cpt[config] = {0: 0.5, 1: 0.5}
        
        # EM iterations
        for iteration in range(max_iter):
            old_cpt = {k: v.copy() for k, v in cpt.items()}
            
            # E-step: Compute expected counts
            expected_counts = defaultdict(lambda: defaultdict(float))
            expected_totals = defaultdict(float)
            
            for i in range(len(data)):
                node_val = data[i, node_idx]
                
                if parent_indices:
                    parent_config = tuple(int(data[i, idx]) if not np.isnan(data[i, idx]) else -1 
                                        for idx in parent_indices)
                else:
                    parent_config = ()
                
                # If node value is missing, use current CPT to estimate
                if np.isnan(node_val):
                    prob_1 = cpt[parent_config][1]
                    prob_0 = cpt[parent_config][0]
                    
                    expected_counts[parent_config][1] += prob_1
                    expected_counts[parent_config][0] += prob_0
                    expected_totals[parent_config] += 1.0
                else:
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
            
            if max_diff < tol:
                break
        
        return dict(cpt)
    
    def learn_all_cpts_ml(self, data, node_idx_map):
        """Learn all CPTs using Maximum Likelihood"""
        print("Learning CPTs using Maximum Likelihood...")
        for node in self.nodes:
            print(f"  Learning CPT for {node}...")
            self.cpts[node] = self.learn_ml_cpt(node, data, node_idx_map, {})
        print("Done!")
    
    def learn_all_cpts_em(self, data, node_idx_map, max_iter=50):
        """Learn all CPTs using EM"""
        print("Learning CPTs using Expectation Maximization...")
        for node in self.nodes:
            print(f"  Learning CPT for {node}...")
            self.cpts[node] = self.learn_em_cpt(node, data, node_idx_map, {}, max_iter=max_iter)
        print("Done!")
    
    def save_cpts(self, filename):
        """Save CPTs to file"""
        # Convert to JSON-serializable format
        cpts_serializable = {}
        for node, cpt in self.cpts.items():
            cpts_serializable[node] = {}
            for config, probs in cpt.items():
                if isinstance(config, tuple):
                    key = str(config)
                else:
                    key = str(config)
                cpts_serializable[node][key] = probs
        
        with open(filename, 'w') as f:
            json.dump(cpts_serializable, f, indent=2)
        print(f"CPTs saved to {filename}")
    
    def print_cpt_summary(self, node, max_configs=5):
        """Print summary of a CPT"""
        if node not in self.cpts:
            print(f"No CPT for {node}")
            return
        
        cpt = self.cpts[node]
        print(f"\nCPT for {node}:")
        print(f"  Number of parent configurations: {len(cpt)}")
        
        # Show first few configurations
        for i, (config, probs) in enumerate(list(cpt.items())[:max_configs]):
            print(f"  Config {config}: P(0)={probs[0]:.4f}, P(1)={probs[1]:.4f}")


def load_data(data_dir):
    """Load all data files"""
    print("Loading data...")
    
    # Load gene mapping
    gene_map = {}
    with open(f"{data_dir}/gene2ind.txt", 'r') as f:
        for line in f:
            idx, gene = line.strip().split('\t')
            gene_map[gene] = int(idx)
    
    # Load cell line mapping
    cell_map = {}
    with open(f"{data_dir}/cell2ind.txt", 'r') as f:
        for line in f:
            idx, cell = line.strip().split('\t')
            cell_map[int(idx)] = cell
    
    # Load mutations
    mutations = []
    with open(f"{data_dir}/cell2mutation.txt", 'r') as f:
        for line in f:
            mutations.append([int(x) for x in line.strip().split(',')])
    mutations = np.array(mutations)
    
    # Load CN amplifications
    amplifications = []
    with open(f"{data_dir}/cell2cnamplification.txt", 'r') as f:
        for line in f:
            amplifications.append([int(x) for x in line.strip().split(',')])
    amplifications = np.array(amplifications)
    
    # Load CN deletions
    deletions = []
    with open(f"{data_dir}/cell2cndeletion.txt", 'r') as f:
        for line in f:
            deletions.append([int(x) for x in line.strip().split(',')])
    deletions = np.array(deletions)
    
    # Load drug response - align with cell line indices
    drug_response_dict = {}
    with open(f"{data_dir}/train_data.txt", 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                cell_name = parts[0]
                try:
                    response_val = float(parts[2])
                    drug_response_dict[cell_name] = response_val
                except ValueError:
                    pass  # Skip invalid values
    
    # Align drug response with mutation data using cell_map
    # Create reverse mapping: cell_name -> cell_idx
    cell_name_to_idx = {name: idx for idx, name in cell_map.items()}
    
    drug_response = []
    for cell_idx in range(len(mutations)):
        if cell_idx in cell_map:
            cell_name = cell_map[cell_idx]
            if cell_name in drug_response_dict:
                drug_response.append(drug_response_dict[cell_name])
            else:
                drug_response.append(np.nan)  # Missing value
        else:
            drug_response.append(np.nan)
    
    drug_response = np.array(drug_response)
    
    print(f"Loaded {len(mutations)} cell lines, {mutations.shape[1]} genes")
    print(f"Drug response: {np.sum(~np.isnan(drug_response))} values, {np.sum(np.isnan(drug_response))} missing")
    
    return gene_map, cell_map, mutations, amplifications, deletions, drug_response


def create_network_structure():
    """
    Create Bayesian network structure based on final_project250A.dot
    Focus on 20 key genes
    """
    # Select 20 key genes (prioritizing those in the network)
    # CDK Pathway: RB1, CCND1, CDK4, CDK6, CDKN2A, CDKN2B (6 genes)
    # Histone/Transcription: KAT6A, TBL1XR1, RUNX1, TERT, MYC, CREBBP, EP300, HDAC1, HDAC2, TP53 (10 genes)
    # DNA Damage: BRCA1, BRCA2, RAD51C, CHEK1 (4 genes) - TP53 already counted
    # Growth Factor: EGFR, ERBB2, ERBB3, FGFR1, FGFR2, PIK3CA (6 genes)
    # Total: 6 + 9 + 3 + 6 = 24, but we'll select 20 most important
    
    # 20 selected genes (removing some less critical ones)
    gene_nodes = [
        'RB1_mut', 'CCND1_amp', 'CDK4_mut', 'CDK6_mut', 'CDKN2A_del', 'CDKN2B_del',  # CDK (6)
        'KAT6A_amp', 'TBL1XR1_amp', 'RUNX1_amp', 'TERT_amp', 'MYC_amp',  # Histone (5)
        'CREBBP_alt', 'EP300_alt', 'HDAC1_alt', 'HDAC2_alt', 'TP53_mut',  # Histone/DNA (5)
        'BRCA1_mut', 'BRCA2_mut', 'RAD51C_mut', 'CHEK1_mut',  # DNA (4)
        'EGFR_alt', 'ERBB2_alt', 'ERBB3_alt', 'FGFR1_alt', 'FGFR2_alt', 'PIK3CA_mut'  # Growth (6)
    ]
    
    # Actually, let's use exactly 20 by selecting the most important
    gene_nodes = [
        'RB1_mut', 'CCND1_amp', 'CDK4_mut', 'CDK6_mut', 'CDKN2A_del', 'CDKN2B_del',  # CDK (6)
        'KAT6A_amp', 'TBL1XR1_amp', 'RUNX1_amp', 'MYC_amp',  # Histone (4)
        'CREBBP_alt', 'EP300_alt', 'TP53_mut',  # Histone/DNA (3)
        'BRCA1_mut', 'BRCA2_mut', 'CHEK1_mut',  # DNA (3)
        'EGFR_alt', 'ERBB2_alt', 'PIK3CA_mut'  # Growth (3)
    ]  # Total: 6+4+3+3+3 = 19, add one more
    gene_nodes.append('HDAC1_alt')  # 20 genes
    
    # Pathway nodes
    pathway_nodes = [
        'CDK_Pathway',
        'Histone_Transcription',
        'DNA_Damage_Response',
        'GrowthFactor_Signaling'
    ]
    
    # Other nodes
    other_nodes = ['TissueType', 'DrugResponse']
    
    all_nodes = gene_nodes + pathway_nodes + other_nodes
    
    # Define structure: node -> list of parents
    structure = {}
    
    # Gene nodes have no parents (root nodes)
    for gene in gene_nodes:
        structure[gene] = []
    
    # CDK Pathway parents
    structure['CDK_Pathway'] = [
        'RB1_mut', 'CCND1_amp', 'CDK4_mut', 'CDK6_mut', 'CDKN2A_del', 'CDKN2B_del',
        'GrowthFactor_Signaling', 'Histone_Transcription', 'DNA_Damage_Response', 'TissueType'
    ]
    
    # Histone Transcription parents
    structure['Histone_Transcription'] = [
        'KAT6A_amp', 'TBL1XR1_amp', 'RUNX1_amp', 'MYC_amp',
        'CREBBP_alt', 'EP300_alt', 'HDAC1_alt', 'TP53_mut',
        'DNA_Damage_Response', 'TissueType'
    ]
    
    # DNA Damage Response parents
    structure['DNA_Damage_Response'] = [
        'TP53_mut', 'BRCA1_mut', 'BRCA2_mut', 'CHEK1_mut', 'TissueType'
    ]
    
    # Growth Factor Signaling parents
    structure['GrowthFactor_Signaling'] = [
        'EGFR_alt', 'ERBB2_alt', 'PIK3CA_mut', 'TissueType'
    ]
    
    # TissueType has no parents
    structure['TissueType'] = []
    
    # DrugResponse parents
    structure['DrugResponse'] = [
        'CDK_Pathway', 'Histone_Transcription', 'DNA_Damage_Response', 'GrowthFactor_Signaling'
    ]
    
    return structure, gene_nodes


def map_genes_to_indices(gene_map, gene_nodes):
    """Map gene node names to data indices"""
    # Map gene names to their alteration types
    gene_name_map = {
        'RB1_mut': 'RB1',
        'CCND1_amp': 'CCND1',
        'CDK4_mut': 'CDK4',
        'CDK6_mut': 'CDK6',
        'CDKN2A_del': 'CDKN2A',
        'CDKN2B_del': 'CDKN2B',
        'KAT6A_amp': 'KAT6A',
        'TBL1XR1_amp': 'TBL1XR1',
        'RUNX1_amp': 'RUNX1',
        'MYC_amp': 'MYC',
        'CREBBP_alt': 'CREBBP',
        'EP300_alt': 'EP300',
        'HDAC1_alt': 'HDAC1',
        'TP53_mut': 'TP53',
        'BRCA1_mut': 'BRCA1',
        'BRCA2_mut': 'BRCA2',
        'CHEK1_mut': 'CHEK1',
        'EGFR_alt': 'EGFR',
        'ERBB2_alt': 'ERBB2',
        'PIK3CA_mut': 'PIK3CA'
    }
    
    node_idx_map = {}
    gene_alteration_map = {}  # Maps node -> (gene_idx, alteration_type)
    
    for node in gene_nodes:
        gene_name = gene_name_map.get(node, node.split('_')[0])
        if gene_name in gene_map:
            gene_idx = gene_map[gene_name]
            alt_type = node.split('_')[1]  # mut, amp, del, alt
            gene_alteration_map[node] = (gene_idx, alt_type)
            node_idx_map[node] = gene_idx  # Use gene index as temporary
        else:
            print(f"Warning: Gene {gene_name} not found in gene map")
    
    return node_idx_map, gene_alteration_map


def create_data_matrix(mutations, amplifications, deletions, drug_response, 
                      gene_alteration_map, cell_map, structure):
    """Create data matrix for all nodes"""
    n_cells = len(mutations)
    n_genes = len(gene_alteration_map)
    
    # Create gene data matrix
    gene_data = np.zeros((n_cells, n_genes), dtype=float)
    node_to_col = {}
    col_idx = 0
    
    for node, (gene_idx, alt_type) in gene_alteration_map.items():
        node_to_col[node] = col_idx
        
        if alt_type == 'mut':
            gene_data[:, col_idx] = mutations[:, gene_idx]
        elif alt_type == 'amp':
            gene_data[:, col_idx] = amplifications[:, gene_idx]
        elif alt_type == 'del':
            gene_data[:, col_idx] = deletions[:, gene_idx]
        elif alt_type == 'alt':
            # For 'alt', use mutation OR amplification
            gene_data[:, col_idx] = np.maximum(mutations[:, gene_idx], 
                                             amplifications[:, gene_idx])
        
        col_idx += 1
    
    # Compute pathway nodes from gene data (initial values from direct gene parents)
    # These will be refined when we consider pathway-to-pathway dependencies
    
    # DNA Damage Response: OR of DNA damage genes (no pathway dependencies)
    dna_genes = ['TP53_mut', 'BRCA1_mut', 'BRCA2_mut', 'CHEK1_mut']
    dna_cols = [node_to_col[g] for g in dna_genes if g in node_to_col]
    if dna_cols:
        node_to_col['DNA_Damage_Response'] = col_idx
        gene_data = np.column_stack([gene_data, np.max(gene_data[:, dna_cols], axis=1)])
        col_idx += 1
    
    # Growth Factor Signaling: OR of growth factor genes (no pathway dependencies)
    growth_genes = ['EGFR_alt', 'ERBB2_alt', 'PIK3CA_mut']
    growth_cols = [node_to_col[g] for g in growth_genes if g in node_to_col]
    if growth_cols:
        node_to_col['GrowthFactor_Signaling'] = col_idx
        gene_data = np.column_stack([gene_data, np.max(gene_data[:, growth_cols], axis=1)])
        col_idx += 1
    
    # Histone Transcription: OR of histone genes + DNA_Damage_Response
    histone_genes = ['KAT6A_amp', 'TBL1XR1_amp', 'RUNX1_amp', 'MYC_amp', 
                     'CREBBP_alt', 'EP300_alt', 'HDAC1_alt', 'TP53_mut']
    histone_cols = [node_to_col[g] for g in histone_genes if g in node_to_col]
    if histone_cols and 'DNA_Damage_Response' in node_to_col:
        # Combine gene contributions with DNA_Damage_Response
        gene_contrib = np.max(gene_data[:, histone_cols], axis=1)
        dna_contrib = gene_data[:, node_to_col['DNA_Damage_Response']]
        node_to_col['Histone_Transcription'] = col_idx
        gene_data = np.column_stack([gene_data, np.maximum(gene_contrib, dna_contrib)])
        col_idx += 1
    
    # CDK Pathway: OR of CDK genes + pathway dependencies
    cdk_genes = ['RB1_mut', 'CCND1_amp', 'CDK4_mut', 'CDK6_mut', 'CDKN2A_del', 'CDKN2B_del']
    cdk_cols = [node_to_col[g] for g in cdk_genes if g in node_to_col]
    if cdk_cols:
        # Combine gene contributions with pathway dependencies
        gene_contrib = np.max(gene_data[:, cdk_cols], axis=1)
        pathway_contribs = []
        if 'GrowthFactor_Signaling' in node_to_col:
            pathway_contribs.append(gene_data[:, node_to_col['GrowthFactor_Signaling']])
        if 'Histone_Transcription' in node_to_col:
            pathway_contribs.append(gene_data[:, node_to_col['Histone_Transcription']])
        if 'DNA_Damage_Response' in node_to_col:
            pathway_contribs.append(gene_data[:, node_to_col['DNA_Damage_Response']])
        
        if pathway_contribs:
            pathway_contrib = np.max(np.column_stack(pathway_contribs), axis=1)
            node_to_col['CDK_Pathway'] = col_idx
            gene_data = np.column_stack([gene_data, np.maximum(gene_contrib, pathway_contrib)])
        else:
            node_to_col['CDK_Pathway'] = col_idx
            gene_data = np.column_stack([gene_data, gene_contrib])
        col_idx += 1
    
    # TissueType: Extract from cell line names (simplified - just use a hash)
    # For simplicity, we'll use a binary encoding based on tissue type
    tissue_types = set()
    for cell_idx, cell_name in cell_map.items():
        if '_' in cell_name:
            tissue = cell_name.split('_', 1)[1]
            tissue_types.add(tissue)
    
    # Create binary tissue type (simplified: 0 or 1 based on most common)
    tissue_list = sorted(list(tissue_types))
    node_to_col['TissueType'] = col_idx
    tissue_data = np.zeros(n_cells)
    for i, cell_idx in enumerate(range(len(cell_map))):
        if cell_idx in cell_map:
            cell_name = cell_map[cell_idx]
            if '_' in cell_name:
                tissue = cell_name.split('_', 1)[1]
                # Binary: 1 if tissue is in first half, 0 otherwise
                tissue_data[i] = 1 if tissue in tissue_list[:len(tissue_list)//2] else 0
    gene_data = np.column_stack([gene_data, tissue_data])
    col_idx += 1
    
    # DrugResponse: Binarize based on median (handle missing values)
    valid_responses = drug_response[~np.isnan(drug_response)]
    if len(valid_responses) > 0:
        median_response = np.median(valid_responses)
        node_to_col['DrugResponse'] = col_idx
        drug_binary = np.where(np.isnan(drug_response), np.nan, 
                              (drug_response > median_response).astype(float))
        gene_data = np.column_stack([gene_data, drug_binary])
    else:
        print("Warning: No valid drug response data")
    
    return gene_data, node_to_col
    
    return gene_data, node_to_col


def main():
    data_dir = "/cellar/users/abishai/ClassProjects/cse250A/Palbociclib/Palbociclib_train"
    
    # Load data
    gene_map, cell_map, mutations, amplifications, deletions, drug_response = load_data(data_dir)
    
    # Create network structure
    structure, gene_nodes = create_network_structure()
    print(f"\nNetwork structure created with {len(structure)} nodes")
    print(f"Gene nodes: {len(gene_nodes)}")
    
    # Map genes to indices
    node_idx_map, gene_alteration_map = map_genes_to_indices(gene_map, gene_nodes)
    print(f"\nMapped {len(node_idx_map)} genes to indices")
    
    # Create data matrix for all nodes
    all_data, node_to_col = create_data_matrix(
        mutations, amplifications, deletions, drug_response,
        gene_alteration_map, cell_map, structure
    )
    
    print(f"\nData matrix shape: {all_data.shape}")
    print(f"Data - min: {all_data.min()}, max: {all_data.max()}, "
          f"mean: {all_data.mean():.3f}")
    
    # Create Bayesian Network for all nodes
    bn_all = BayesianNetwork(structure)
    
    # Learn CPTs using Maximum Likelihood for ALL nodes
    print("\n" + "="*60)
    print("LEARNING CPTs FOR ALL NODES USING MAXIMUM LIKELIHOOD")
    print("="*60)
    bn_all.learn_all_cpts_ml(all_data, node_to_col)
    
    # Print CPT summaries for different node types
    print("\nCPT Summaries (ML) - Gene Nodes:")
    for i, node in enumerate(gene_nodes[:5]):  # Show first 5 genes
        bn_all.print_cpt_summary(node)
    
    print("\nCPT Summaries (ML) - Pathway Nodes:")
    pathway_nodes = ['CDK_Pathway', 'Histone_Transcription', 'DNA_Damage_Response', 'GrowthFactor_Signaling']
    for node in pathway_nodes:
        if node in bn_all.cpts:
            bn_all.print_cpt_summary(node)
    
    print("\nCPT Summaries (ML) - DrugResponse:")
    if 'DrugResponse' in bn_all.cpts:
        bn_all.print_cpt_summary('DrugResponse')
    
    # Save ML CPTs
    bn_all.save_cpts("cpts_ml_all.json")
    
    # Learn CPTs using EM for ALL nodes
    print("\n" + "="*60)
    print("LEARNING CPTs FOR ALL NODES USING EM")
    print("="*60)
    bn_all_em = BayesianNetwork(structure)
    bn_all_em.learn_all_cpts_em(all_data, node_to_col, max_iter=50)
    
    # Print CPT summaries
    print("\nCPT Summaries (EM) - Gene Nodes:")
    for i, node in enumerate(gene_nodes[:5]):  # Show first 5 genes
        bn_all_em.print_cpt_summary(node)
    
    print("\nCPT Summaries (EM) - Pathway Nodes:")
    for node in pathway_nodes:
        if node in bn_all_em.cpts:
            bn_all_em.print_cpt_summary(node)
    
    print("\nCPT Summaries (EM) - DrugResponse:")
    if 'DrugResponse' in bn_all_em.cpts:
        bn_all_em.print_cpt_summary('DrugResponse')
    
    # Save EM CPTs
    bn_all_em.save_cpts("cpts_em_all.json")
    
    print("\n" + "="*60)
    print("CPT LEARNING COMPLETE")
    print("="*60)
    print(f"\nLearned CPTs for {len(bn_all.cpts)} nodes")
    print(f"  - {len(gene_nodes)} gene nodes")
    print(f"  - {len(pathway_nodes)} pathway nodes")
    print(f"  - 1 DrugResponse node")
    print(f"  - 1 TissueType node")
    print("\nSaved to:")
    print("  - cpts_ml_all.json (Maximum Likelihood)")
    print("  - cpts_em_all.json (Expectation Maximization)")


if __name__ == "__main__":
    main()

