#!/usr/bin/env python3
"""
Visual examples of CPTs with concrete interpretations
"""

import json
import numpy as np

def load_cpts(filename):
    """Load CPTs from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def explain_root_node_cpt(node_name, cpt):
    """Explain a root node CPT (no parents)"""
    print(f"\n{'='*80}")
    print(f"EXAMPLE 1: Root Node - {node_name}")
    print(f"{'='*80}")
    print(f"\nNode: {node_name}")
    print(f"Parents: None (root node)")
    print(f"\nCPT:")
    print(f"  Parent Config: ()  (empty - no parents)")
    
    config = cpt.get("()", {})
    p0 = float(config.get("0", 0))
    p1 = float(config.get("1", 0))
    
    print(f"    P({node_name} = 0) = {p0:.4f} ({p0*100:.2f}%)")
    print(f"    P({node_name} = 1) = {p1:.4f} ({p1*100:.2f}%)")
    
    print(f"\nInterpretation:")
    print(f"  • In the training data:")
    print(f"    - {p0*100:.1f}% of cell lines have {node_name} = 0 (absent/inactive)")
    print(f"    - {p1*100:.1f}% of cell lines have {node_name} = 1 (present/active)")
    
    if 'mut' in node_name:
        print(f"  • This is a mutation node: 0 = no mutation, 1 = mutation present")
    elif 'amp' in node_name:
        print(f"  • This is an amplification node: 0 = no amplification, 1 = amplification present")
    elif 'del' in node_name:
        print(f"  • This is a deletion node: 0 = no deletion, 1 = deletion present")

def explain_pathway_cpt(node_name, cpt, structure_info):
    """Explain a pathway node CPT with parents"""
    print(f"\n{'='*80}")
    print(f"EXAMPLE 2: Pathway Node - {node_name}")
    print(f"{'='*80}")
    print(f"\nNode: {node_name}")
    
    # Get parent names from structure
    parents = structure_info.get(node_name, [])
    print(f"Parents: {', '.join(parents) if parents else 'None'}")
    print(f"Number of parent configurations: {len(cpt)}")
    
    # Show a few example configurations
    print(f"\nSample CPT Entries:")
    
    configs = list(cpt.items())[:5]  # Show first 5
    for i, (config_str, probs) in enumerate(configs):
        config = eval(config_str) if config_str != "()" else ()
        p0 = float(probs.get("0", 0))
        p1 = float(probs.get("1", 0))
        
        print(f"\n  Configuration {i+1}: {config_str}")
        if parents:
            print(f"    Parent values: ", end="")
            for j, (parent, val) in enumerate(zip(parents, config)):
                print(f"{parent}={val}", end="")
                if j < len(config) - 1:
                    print(", ", end="")
            print()
        print(f"    P({node_name} = 0 | config) = {p0:.4f} ({p0*100:.2f}%)")
        print(f"    P({node_name} = 1 | config) = {p1:.4f} ({p1*100:.2f}%)")
        
        # Interpretation
        if all(x == 0 for x in config):
            print(f"    → When all parents are inactive: {p1*100:.1f}% chance pathway is active")
        elif all(x == 1 for x in config):
            print(f"    → When all parents are active: {p1*100:.1f}% chance pathway is active")
        else:
            active_count = sum(config)
            print(f"    → When {active_count}/{len(config)} parents are active: {p1*100:.1f}% chance pathway is active")
    
    if len(cpt) > 5:
        print(f"\n  ... and {len(cpt) - 5} more configurations")
    
    print(f"\nInterpretation:")
    print(f"  • This CPT shows how parent gene alterations affect pathway activation")
    print(f"  • Each row represents a unique combination of parent values")
    print(f"  • The probabilities show how likely the pathway is active (1) vs inactive (0)")
    print(f"  • Generally: more active parents → higher probability of active pathway")

def explain_drug_response_cpt(node_name, cpt, structure_info):
    """Explain DrugResponse CPT"""
    print(f"\n{'='*80}")
    print(f"EXAMPLE 3: Target Node - {node_name}")
    print(f"{'='*80}")
    print(f"\nNode: {node_name}")
    
    parents = structure_info.get(node_name, [])
    print(f"Parents: {', '.join(parents) if parents else 'None'}")
    print(f"Number of parent configurations: {len(cpt)}")
    
    # Show key configurations
    print(f"\nKey CPT Entries:")
    
    # Find all-zero and all-one configurations
    all_zero_config = None
    all_one_config = None
    
    for config_str in cpt:
        config = eval(config_str) if config_str != "()" else ()
        if all(x == 0 for x in config):
            all_zero_config = config_str
        if all(x == 1 for x in config):
            all_one_config = config_str
    
    configs_to_show = []
    if all_zero_config:
        configs_to_show.append(all_zero_config)
    if all_one_config:
        configs_to_show.append(all_one_config)
    
    # Add a few more random ones
    other_configs = [c for c in list(cpt.keys())[:3] if c not in configs_to_show]
    configs_to_show.extend(other_configs)
    
    for config_str in configs_to_show[:5]:
        probs = cpt[config_str]
        config = eval(config_str) if config_str != "()" else ()
        p0 = float(probs.get("0", 0))
        p1 = float(probs.get("1", 0))
        
        print(f"\n  Configuration: {config_str}")
        if parents:
            print(f"    Pathway states: ", end="")
            for j, (parent, val) in enumerate(zip(parents, config)):
                status = "ACTIVE" if val == 1 else "inactive"
                print(f"{parent}={status}", end="")
                if j < len(config) - 1:
                    print(", ", end="")
            print()
        print(f"    P({node_name} = Low Response | config) = {p0:.4f} ({p0*100:.2f}%)")
        print(f"    P({node_name} = High Response | config) = {p1:.4f} ({p1*100:.2f}%)")
        
        # Interpretation
        active_pathways = sum(config)
        if active_pathways == 0:
            print(f"    → All pathways inactive: {p1*100:.1f}% chance of HIGH drug response")
        elif active_pathways == len(config):
            print(f"    → All pathways active: {p1*100:.1f}% chance of HIGH drug response")
        else:
            print(f"    → {active_pathways}/{len(config)} pathways active: {p1*100:.1f}% chance of HIGH drug response")
    
    print(f"\nInterpretation:")
    print(f"  • This CPT predicts drug response based on pathway activation states")
    print(f"  • High response (1) = sensitive to palbociclib")
    print(f"  • Low response (0) = resistant to palbociclib")
    print(f"  • Generally: more active pathways → higher probability of high response")

def show_ml_vs_em_comparison(ml_cpts, em_cpts, node_name):
    """Compare ML and EM CPTs for a node"""
    print(f"\n{'='*80}")
    print(f"COMPARISON: ML vs EM for {node_name}")
    print(f"{'='*80}")
    
    ml_cpt = ml_cpts.get(node_name, {})
    em_cpt = em_cpts.get(node_name, {})
    
    print(f"\nML CPT:")
    if ml_cpt:
        print(f"  Number of configurations: {len(ml_cpt)}")
        if ml_cpt:
            first_config = list(ml_cpt.items())[0]
            print(f"  Sample: {first_config[0]} → P(0)={first_config[1].get('0', 0):.4f}, P(1)={first_config[1].get('1', 0):.4f}")
    else:
        print(f"  ❌ NOT LEARNED (node is hidden or has hidden parents)")
    
    print(f"\nEM CPT:")
    if em_cpt:
        print(f"  Number of configurations: {len(em_cpt)}")
        if em_cpt:
            first_config = list(em_cpt.items())[0]
            print(f"  Sample: {first_config[0]} → P(0)={first_config[1].get('0', 0):.4f}, P(1)={first_config[1].get('1', 0):.4f}")
    else:
        print(f"  ❌ NOT LEARNED")
    
    if ml_cpt and em_cpt:
        # Compare a common configuration
        common_configs = set(ml_cpt.keys()) & set(em_cpt.keys())
        if common_configs:
            config = list(common_configs)[0]
            ml_p1 = float(ml_cpt[config].get("1", 0))
            em_p1 = float(em_cpt[config].get("1", 0))
            diff = abs(ml_p1 - em_p1)
            print(f"\n  Comparison for config {config}:")
            print(f"    ML: P(1) = {ml_p1:.4f}")
            print(f"    EM: P(1) = {em_p1:.4f}")
            print(f"    Difference: {diff:.4f}")

def main():
    print("="*80)
    print("CPT EXPLANATION WITH CONCRETE EXAMPLES")
    print("="*80)
    
    # Load CPTs
    print("\nLoading CPTs...")
    ml_cpts = load_cpts("cpts_ml_hidden_pathways.json")
    em_cpts = load_cpts("cpts_em_hidden_pathways.json")
    
    # Network structure (simplified - just parent relationships)
    structure = {
        "RB1_mut": [],
        "CCND1_amp": [],
        "CDK4_mut": [],
        "CDK6_mut": [],
        "CDKN2A_del": [],
        "CDKN2B_del": [],
        "CDK_Pathway": ["RB1_mut", "CCND1_amp", "CDK4_mut", "CDK6_mut", "CDKN2A_del", "CDKN2B_del", "TissueType"],
        "Histone_Transcription": ["KAT6A_amp", "TBL1XR1_amp", "RUNX1_amp", "TERT_amp", "MYC_amp", 
                                  "CREBBP_alt", "EP300_alt", "HDAC1_alt", "HDAC2_alt", "TP53_mut"],
        "DNA_Damage_Response": ["TP53_mut", "BRCA1_mut", "BRCA2_mut", "RAD51C_mut", "CHEK1_mut"],
        "GrowthFactor_Signaling": ["EGFR_alt", "ERBB2_alt", "ERBB3_alt", "FGFR1_alt", "FGFR2_alt", "PIK3CA_mut"],
        "DrugResponse": ["CDK_Pathway", "Histone_Transcription", "DNA_Damage_Response", "GrowthFactor_Signaling"]
    }
    
    # Example 1: Root node
    explain_root_node_cpt("RB1_mut", em_cpts.get("RB1_mut", {}))
    explain_root_node_cpt("TP53_mut", em_cpts.get("TP53_mut", {}))
    
    # Example 2: Pathway node
    explain_pathway_cpt("CDK_Pathway", em_cpts.get("CDK_Pathway", {}), structure)
    
    # Example 3: Drug response
    explain_drug_response_cpt("DrugResponse", em_cpts.get("DrugResponse", {}), structure)
    
    # Comparison: ML vs EM
    print(f"\n{'='*80}")
    print("ML vs EM COMPARISON")
    print(f"{'='*80}")
    
    print("\n1. Observed Node (RB1_mut):")
    show_ml_vs_em_comparison(ml_cpts, em_cpts, "RB1_mut")
    
    print("\n2. Hidden Node (CDK_Pathway):")
    show_ml_vs_em_comparison(ml_cpts, em_cpts, "CDK_Pathway")
    
    print("\n3. Node with Hidden Parents (DrugResponse):")
    show_ml_vs_em_comparison(ml_cpts, em_cpts, "DrugResponse")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("\nKey Takeaways:")
    print("1. Root nodes (genes): Both ML and EM learn the same CPTs (direct counting)")
    print("2. Hidden nodes (pathways): Only EM can learn (ML has no observations)")
    print("3. Nodes with hidden parents (DrugResponse): Only EM can learn (ML needs all parents observed)")
    print("\nCPT Structure:")
    print("  • Each row = one parent configuration (tuple of parent values)")
    print("  • '0' column = P(node = 0 | parent_config)")
    print("  • '1' column = P(node = 1 | parent_config)")
    print("  • For each row: P(0) + P(1) = 1.0")
    print("\nInterpretation:")
    print("  • Root nodes: Marginal probabilities (base rates in data)")
    print("  • Pathway nodes: Conditional probabilities given gene alterations")
    print("  • DrugResponse: Conditional probabilities given pathway activations")

if __name__ == "__main__":
    main()





