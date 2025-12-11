import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import json
import os

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


def read_files(dir):
    """Load Data from directory"""
    print("Loading Data....")
    #Load gene mapping
    gene2idx = os.path.join(dir,'gene2ind.txt')
    gene2idx_map = {}
    
    with open(gene2idx, 'r') as f:
        values = f.readlines()
        values = [(v.strip().split("\t")) for v in values]
        
    for v in values:
        gene2idx_map[v[1]] = int(v[0])
        
    #Load cell line mapping
    cell2idx = os.path.join(dir,'cell2ind.txt')
    cell2idx_map = {}
    
    with open(cell2idx, 'r') as f:
        values = f.readlines()
        values = [(v.strip().split("\t")) for v in values]
        
    for v in values:
        cell2idx_map[v[1]] = int(v[0])
        
    #Load mutations
    cell2mutation_path = os.path.join(dir, 'cell2mutation.txt')
    
    with open(cell2mutation_path, 'r') as f:
        values = f.readlines()
        values = [list(map(int, v.strip().split(',')))for v in values]
        
    cell2mutate = np.array(values)
        
    #Load Amplifications
    cell2amp_path = os.path.join(dir, 'cell2cnamplification.txt')
    
    with open(cell2amp_path, 'r') as f:
        values = f.readlines()
        values = [list(map(int, v.strip().split(','))) for v in values]
        
    cell2amp = np.array(values)
    
    #Load Deletions
    cell2delete_path = os.path.join(dir, 'cell2cndeletion.txt')
    
    with open(cell2delete_path, 'r') as f:
        values = f.readlines()
        values = [list(map(int, v.strip().split(',')))for v in values]
        
    cell2delete = np.array(values)
    
    #Load drug response
        
    drug_response_path = os.path.join(dir,'train_data.txt')
    
    with open(drug_response_path, 'r') as f:
        values = f.readlines()
        values = [(v.strip().split("\t")) for v in values]
        
    drug_response_dict = {}
    
    for v in values:
        if len(v)>=3:
            try:
                drug_response_dict[v[0]] = (float(v[2]))
            except ValueError:
                pass
            
    #Reverse mapping from idx->cell
    idx2cell_map = {idx: cell for (cell, idx) in cell2idx_map.items()}
    
    drug_response = []
    
    for cellidx in range(len(cell2mutate)):
        if cellidx in idx2cell_map:
            cell_name = idx2cell_map[cellidx]
            if cell_name in drug_response_dict:
                drug_response.append(drug_response_dict[cell_name])
                
            else:
                drug_response.append(np.nan)
                
        else:
            drug_response.append(np.nan)
            
    drug_response = np.array(drug_response)
    
    print(f"Loaded {len(cell2mutate)} cell lines, {cell2mutate.shape[1]} genes")
    print(f"Drug response: {np.sum(~np.isnan(drug_response))} values, {np.sum(np.isnan(drug_response))} missing")
    
        
    return cell2amp, cell2delete, cell2mutate, cell2idx_map, gene2idx_map, drug_response

        
def create_network_structure():
    """Create Bayesian Network focussing on 20 genes"""
    gene_nodes = [
        "CCND1_amp",
        "CDK4_mut",
        "CDK6_mut",
        "CREBBP_alt",
        "EP300_alt",
        "HDAC1_alt",#
        "HDAC2_alt",#
        "KAT6A_amp",
        "TBL1XR1_amp",
        "BRCA1_mut",
        "BRCA2_mut",
        "CHEK1_mut",
        "RAD51C_mut",#
        "TP53_mut",
        "EGFR_alt",
        "ERBB2_alt",
        "ERBB3_alt",
        "FGFR1_alt",
        "FGFR2_alt",
        "PIK3CA_mut",
        "RB1_mut",
        "CDKN2A_del",
        "CDKN2B_del"
        
    ]
    
    pathway_nodes = ["CDK_Overdrive",
                     "Chromatin_Remodeling_State",
                     "DNA_Repair_Capacity",
                     "RTK_PI3K_Signaling",
                     "RB_Pathway_Activity"]
    
    combine_node = ["Proliferative_Phenotype"]
    
    output_node = ["DrugResponse"]
    #Define the structure node->parents
    structure = {}
    
    for gene in gene_nodes:
        structure[gene] = []
        
    structure["CDK_Overdrive"] = ["CCND1_amp",
                                "CDK4_mut",
                                "CDK6_mut",]
    
    structure["Chromatin_Remodeling_State"] = ["CREBBP_alt",
                                                "EP300_alt",
                                                "HDAC1_alt",#
                                                "HDAC2_alt",#
                                                "KAT6A_amp",
                                                "TBL1XR1_amp"]
    
    structure["DNA_Repair_Capacity"] = ["BRCA1_mut",
                                        "BRCA2_mut",
                                        "CHEK1_mut",
                                        "RAD51C_mut",#
                                        "TP53_mut",]
    
    structure["RTK_PI3K_Signaling"] = ["EGFR_alt",
                                        "ERBB2_alt",
                                        "ERBB3_alt",
                                        "FGFR1_alt",
                                        "FGFR2_alt",
                                        "PIK3CA_mut",]
    
    structure["RB_Pathway_Activity"] = ["RB1_mut",
                                        "CDKN2A_del",
                                        "CDKN2B_del"]
    
    # structure["Proliferative_Phenotype"] = pathway_nodes+["TP53_mut"]
    
    # structure["DrugResponse"] = ["Proliferative_Phenotype"]
    
    structure["DrugResponse"] = pathway_nodes
    
    return structure, gene_nodes

def map_gene_to_indices(gene2idx_map, gene_nodes):
    gene_name_map = {"CCND1_amp": "CCND1" ,
                "CDK4_mut": "CDK4",
                "CDK6_mut": "CDK6",
                "CREBBP_alt": "CREBBP",
                "EP300_alt": "EP300",
                "HDAC1_alt": "HDAC1",#
                "HDAC2_alt": "HDAC2",#
                "KAT6A_amp": "KAT6A",
                "TBL1XR1_amp": "TBL1XR1",
                "BRCA1_mut": "BRCA1",
                "BRCA2_mut": "BRCA2",
                "CHEK1_mut": "CHEK1",
                "RAD51C_mut": "RAD51C",#
                "TP53_mut": "TP53",
                "EGFR_alt": "EGFR",
                "ERBB2_alt": "ERBB2",
                "ERBB3_alt": "ERBB3",
                "FGFR1_alt": "FGFR1",
                "FGFR2_alt": "FGFR2",
                "PIK3CA_mut": "PIK3CA",
                "RB1_mut": "RB1",
                "CDKN2A_del": "CDKN2A",
                "CDKN2B_del": "CDKN2B"}
    node_idx_map = {}
    gene_alteration_map = {}  # Maps node -> (gene_idx, alteration_type)
    
    for node in gene_nodes:
        gene_name = gene_name_map.get(node, node.split('_')[0])
        if gene_name in gene2idx_map:
            gene_idx = gene2idx_map[gene_name]
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
    
    pathway_nodes = ["CDK_Overdrive",
                     "Chromatin_Remodeling_State",
                     "DNA_Repair_Capacity",
                     "RTK_PI3K_Signaling",
                     "RB_Pathway_Activity"]
    
    # CDK Overdrive Response: OR of CDK overdrive genes (no pathway dependencies)
    cdk_genes = ["CCND1_amp", "CDK4_mut", "CDK6_mut",]
    cdk_cols = [node_to_col[g] for g in cdk_genes if g in node_to_col]
    if cdk_cols:
        node_to_col['CDK_Overdrive'] = col_idx
        gene_data = np.column_stack([gene_data, np.max(gene_data[:, cdk_cols], axis=1)])
        col_idx += 1
        
    # Chromatin Remodeling State: OR of chromatin remodeling genes (no pathway dependencies)
    chromatin_genes = ["CREBBP_alt", "EP300_alt", "HDAC1_alt", "HDAC2_alt", "KAT6A_amp", "TBL1XR1_amp"]
    chromatin_cols = [node_to_col[g] for g in chromatin_genes if g in node_to_col]
    if chromatin_cols:
        node_to_col['Chromatin_Remodeling_State'] = col_idx
        gene_data = np.column_stack([gene_data, np.max(gene_data[:, chromatin_cols], axis=1)])
        col_idx += 1
        
    # DNA Repair Capacity: OR of DNA repairing genes (no pathway dependencies)
    dna_genes = ["BRCA1_mut","BRCA2_mut","CHEK1_mut","RAD51C_mut","TP53_mut",]
    dna_cols = [node_to_col[g] for g in dna_genes if g in node_to_col]
    if dna_cols:
        node_to_col['DNA_Repair_Capacity'] = col_idx
        gene_data = np.column_stack([gene_data, np.max(gene_data[:, dna_cols], axis=1)])
        col_idx += 1
        
    # RTK PI3K Signaling: OR of RTK PI3K Signaling genes (no pathway dependencies)
    rtk_genes = ["EGFR_alt", "ERBB2_alt", "ERBB3_alt", "FGFR1_alt", "FGFR2_alt", "PIK3CA_mut",]
    rtk_cols = [node_to_col[g] for g in rtk_genes if g in node_to_col]
    if rtk_cols:
        node_to_col['RTK_PI3K_Signaling'] = col_idx
        gene_data = np.column_stack([gene_data, np.max(gene_data[:, rtk_cols], axis=1)])
        col_idx += 1
        
    # RB_Pathway_Activity: OR of RB_Pathway_Activity (no pathway dependencies)
    rb_genes = ["RB1_mut", "CDKN2A_del", "CDKN2B_del"]
    rb_cols = [node_to_col[g] for g in rb_genes if g in node_to_col]
    if rb_cols:
        node_to_col['RB_Pathway_Activity'] = col_idx
        gene_data = np.column_stack([gene_data, np.max(gene_data[:, rb_cols], axis=1)])
        col_idx += 1
        
    #Phenotype: OR of CDK Overdrive Response + Chromatin Remodeling State + DNA Repair Capacity + RTK PI3K Signaling + RB_Pathway_Activity + Phenotype genes
    # phenotype_genes = ["TP53_mut"]
    # phenotype_cols = [node_to_col[g] for g in phenotype_genes if g in node_to_col]
    # if phenotype_cols:
    #     # Combine gene contributions with pathway dependencies
    #     gene_contrib = np.max(gene_data[:, phenotype_cols], axis=1)
    #     pathway_contribs = []
        
    #     if "CDK_Overdrive" in node_to_col:
    #         pathway_contribs.append(gene_data[:, node_to_col['CDK_Overdrive']])
            
    #     if "Chromatin_Remodeling_State" in node_to_col:
    #         pathway_contribs.append(gene_data[:, node_to_col['Chromatin_Remodeling_State']])
            
    #     if "DNA_Repair_Capacity" in node_to_col:
    #         pathway_contribs.append(gene_data[:, node_to_col['DNA_Repair_Capacity']])
            
    #     if "RTK_PI3K_Signaling" in node_to_col:
    #         pathway_contribs.append(gene_data[:, node_to_col['RTK_PI3K_Signaling']])
            
    #     if "RB_Pathway_Activity" in node_to_col:
    #         pathway_contribs.append(gene_data[:, node_to_col['RB_Pathway_Activity']])
            
    #     if pathway_contribs:
    #         pathway_contrib = np.max(np.column_stack(pathway_contribs), axis=1)
    #         node_to_col['Proliferative_Phenotype'] = col_idx
    #         gene_data = np.column_stack([gene_data, np.maximum(gene_contrib, pathway_contrib)])
    #     else:
    #         node_to_col['Proliferative_Phenotype'] = col_idx
    #         gene_data = np.column_stack([gene_data, gene_contrib])
    #     col_idx += 1
    
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
    
def main():
    #Read data
    data_dir = "/Users/GayathriRajesh/Desktop/UCSD 27/Fall 2025/250A/project/Palbociclib_train"
    cell2amp, cell2delete, cell2mutate, cell2idx_map, gene2idx_map, drug_response = read_files(data_dir)
    
    # Create network structure
    structure, gene_nodes = create_network_structure()
    print(f"\nNetwork structure created with {len(structure)} nodes")
    print(f"Gene nodes: {len(gene_nodes)}")
    
    node_idx_map, gene_alteration_map = map_gene_to_indices(gene2idx_map, gene_nodes)
    
    #Create Data Matrix
    
    all_data, node_to_col = create_data_matrix(
        cell2mutate, cell2amp, cell2delete, drug_response,
        gene_alteration_map, cell2idx_map, structure
    )
    
    print(f"\nData matrix shape: {all_data.shape}")
    print(f"Data - min: {all_data.min()}, max: {all_data.max()}, "
          f"mean: {all_data.mean():.3f}")
    
    
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