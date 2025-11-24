#!/usr/bin/env python3
"""
Create DOT files with CPTs as HTML tables for hidden pathways analysis
Shows ML (only observed) vs EM (observed + hidden)
"""

import json
import ast

def format_cpt_as_html_table(cpt, node_name, max_configs=3):
    """
    Format CPT as HTML table for Graphviz
    """
    if not cpt or len(cpt) == 0:
        return f"""<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
<TR><TD BORDER="1" COLSPAN="2"><B>{node_name}</B></TD></TR>
<TR><TD BORDER="1">No CPT learned</TD><TD BORDER="1">(hidden/missing)</TD></TR>
</TABLE>"""
    
    # For nodes with no parents (root nodes)
    if "()" in cpt and len(cpt) == 1:
        probs = cpt["()"]
        p0 = float(probs.get('0', 0.0))
        p1 = float(probs.get('1', 0.0))
        html = f"""<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
<TR><TD BORDER="1" COLSPAN="2"><B>{node_name}</B></TD></TR>
<TR><TD BORDER="1"><B>Value</B></TD><TD BORDER="1"><B>Probability</B></TD></TR>
<TR><TD BORDER="1">0</TD><TD BORDER="1">{p0:.3f}</TD></TR>
<TR><TD BORDER="1">1</TD><TD BORDER="1">{p1:.3f}</TD></TR>
</TABLE>"""
        return html
    
    # For nodes with parents, show conditional probability table
    num_configs = len(cpt)
    configs_to_show = min(num_configs, max_configs)
    
    # Get first config to determine number of parents
    first_config = list(cpt.keys())[0]
    try:
        if first_config == "()":
            num_parents = 0
        else:
            config = ast.literal_eval(first_config)
            if isinstance(config, tuple):
                num_parents = len(config)
            else:
                num_parents = 1
    except:
        num_parents = 0
    
    # Build table header
    html = f"""<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">
<TR><TD BORDER="1" COLSPAN="{num_parents + 3}"><B>{node_name}</B></TD></TR>
<TR>"""
    
    # Add parent column headers
    for i in range(num_parents):
        html += f'<TD BORDER="1"><B>P{i+1}</B></TD>'
    
    # Add value and probability columns
    html += f'<TD BORDER="1"><B>Value</B></TD>'
    html += f'<TD BORDER="1"><B>P(0)</B></TD>'
    html += f'<TD BORDER="1"><B>P(1)</B></TD>'
    html += '</TR>'
    
    # Add rows for each configuration
    shown = 0
    for config_str, probs in list(cpt.items())[:configs_to_show]:
        html += '<TR>'
        
        # Parse and display parent configuration
        try:
            if config_str == "()":
                for i in range(num_parents):
                    html += '<TD BORDER="1">-</TD>'
            else:
                config = ast.literal_eval(config_str)
                if isinstance(config, tuple):
                    for i, val in enumerate(config[:8]):  # Limit display
                        html += f'<TD BORDER="1">{int(val)}</TD>'
                    for i in range(len(config), num_parents):
                        html += '<TD BORDER="1">-</TD>'
                else:
                    html += f'<TD BORDER="1">{int(config)}</TD>'
                    for i in range(1, num_parents):
                        html += '<TD BORDER="1">-</TD>'
        except:
            for i in range(num_parents):
                html += '<TD BORDER="1">?</TD>'
        
        # Add probabilities
        p0 = float(probs.get('0', 0.0))
        p1 = float(probs.get('1', 0.0))
        html += f'<TD BORDER="1">0/1</TD>'
        html += f'<TD BORDER="1">{p0:.3f}</TD>'
        html += f'<TD BORDER="1">{p1:.3f}</TD>'
        html += '</TR>'
        shown += 1
    
    # Add summary row if there are more configurations
    if num_configs > shown:
        html += f'<TR><TD BORDER="1" COLSPAN="{num_parents + 3}" ALIGN="CENTER">... +{num_configs - shown} more</TD></TR>'
    
    html += '</TABLE>'
    return html

def format_cpt_as_simple_table(cpt, node_name):
    """Format CPT as simpler table for nodes with many parents"""
    if not cpt or len(cpt) == 0:
        return f"""<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
<TR><TD BORDER="1" COLSPAN="2"><B>{node_name}</B></TD></TR>
<TR><TD BORDER="1">No CPT learned</TD><TD BORDER="1">(hidden/missing)</TD></TR>
</TABLE>"""
    
    if "()" in cpt and len(cpt) == 1:
        probs = cpt["()"]
        p0 = float(probs.get('0', 0.0))
        p1 = float(probs.get('1', 0.0))
        html = f"""<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
<TR><TD BORDER="1" COLSPAN="2"><B>{node_name}</B></TD></TR>
<TR><TD BORDER="1"><B>Value</B></TD><TD BORDER="1"><B>Probability</B></TD></TR>
<TR><TD BORDER="1">0</TD><TD BORDER="1">{p0:.3f}</TD></TR>
<TR><TD BORDER="1">1</TD><TD BORDER="1">{p1:.3f}</TD></TR>
</TABLE>"""
        return html
    
    # For conditional nodes, show summary table
    num_configs = len(cpt)
    sample_configs = list(cpt.items())[:3]
    
    html = f"""<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">
<TR><TD BORDER="1" COLSPAN="3"><B>{node_name}</B></TD></TR>
<TR><TD BORDER="1"><B>Config</B></TD><TD BORDER="1"><B>P(0)</B></TD><TD BORDER="1"><B>P(1)</B></TD></TR>"""
    
    for config_str, probs in sample_configs:
        try:
            if config_str == "()":
                config_display = "no parents"
            else:
                config = ast.literal_eval(config_str)
                if isinstance(config, tuple):
                    config_display = ",".join(str(int(x)) for x in config[:5])
                    if len(config) > 5:
                        config_display += "..."
                else:
                    config_display = str(int(config))
        except:
            config_display = str(config_str)[:10]
        
        p0 = float(probs.get('0', 0.0))
        p1 = float(probs.get('1', 0.0))
        html += f'<TR><TD BORDER="1">{config_display}</TD><TD BORDER="1">{p0:.3f}</TD><TD BORDER="1">{p1:.3f}</TD></TR>'
    
    if num_configs > len(sample_configs):
        html += f'<TR><TD BORDER="1" COLSPAN="3" ALIGN="CENTER">... +{num_configs - len(sample_configs)} more</TD></TR>'
    
    html += '</TABLE>'
    return html

def load_cpts(filename):
    """Load CPTs from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def create_dot_with_cpts(cpts_file, output_file, method_name):
    """
    Create DOT file with CPTs as HTML tables
    """
    # Load CPTs
    cpts = load_cpts(cpts_file)
    print(f"Loaded CPTs for {len(cpts)} nodes from {cpts_file}")
    
    # Define network structure
    gene_nodes_cdk = ['RB1_mut', 'CCND1_amp', 'CDK4_mut', 'CDK6_mut', 'CDKN2A_del', 'CDKN2B_del']
    gene_nodes_histone = ['KAT6A_amp', 'TBL1XR1_amp', 'RUNX1_amp', 'MYC_amp', 
                          'CREBBP_alt', 'EP300_alt', 'HDAC1_alt', 'TP53_mut']
    gene_nodes_dna = ['BRCA1_mut', 'BRCA2_mut', 'CHEK1_mut']
    gene_nodes_growth = ['EGFR_alt', 'ERBB2_alt', 'PIK3CA_mut']
    
    all_genes = gene_nodes_cdk + gene_nodes_histone + gene_nodes_dna + gene_nodes_growth
    
    # Build DOT file
    lines = []
    lines.append("digraph PalboBN {")
    lines.append("    rankdir=LR;")
    lines.append("    fontsize=10;")
    lines.append(f"    label=\"Bayesian Network with CPTs ({method_name})\";")
    lines.append("    node [fontsize=8];")
    lines.append("")
    lines.append("    // Gene-level alterations")
    lines.append("    subgraph cluster_genes {")
    lines.append("        label=\"Gene-level alterations\";")
    lines.append("        style=dashed;")
    lines.append("        color=gray;")
    lines.append("")
    
    # Add gene nodes with CPT tables
    for gene in all_genes:
        if gene in cpts and len(cpts[gene]) > 0:
            cpt_table = format_cpt_as_html_table(cpts[gene], gene, max_configs=1)
            lines.append(f"        {gene} [label=<{cpt_table}>];")
        else:
            lines.append(f"        {gene} [label=\"{gene}\\n(no CPT)\"];")
    
    lines.append("    }")
    lines.append("")
    lines.append("    // Pathway nodes (HIDDEN/LATENT)")
    lines.append("    node [shape=ellipse, style=filled, fillcolor=lightblue];")
    lines.append("")
    
    # Add pathway nodes with CPT tables
    pathway_nodes = ['CDK_Pathway', 'Histone_Transcription', 'DNA_Damage_Response', 'GrowthFactor_Signaling']
    
    for pathway in pathway_nodes:
        if pathway in cpts and len(cpts[pathway]) > 0:
            # Use simpler table format for pathway nodes
            cpt_table = format_cpt_as_simple_table(cpts[pathway], pathway)
            lines.append(f"    {pathway} [label=<{cpt_table}>];")
        else:
            lines.append(f"    {pathway} [label=\"{pathway}\\n(HIDDEN - no CPT learned)\"];")
    
    lines.append("")
    lines.append("    // Target node")
    lines.append("    node [shape=rectangle, style=\"rounded,filled\", fillcolor=lightgray];")
    if 'DrugResponse' in cpts and len(cpts['DrugResponse']) > 0:
        cpt_table = format_cpt_as_simple_table(cpts['DrugResponse'], 'DrugResponse')
        lines.append(f"    DrugResponse [label=<{cpt_table}>];")
    else:
        lines.append("    DrugResponse [label=\"DrugResponse\\n(no CPT - hidden parents)\"];")
    
    lines.append("")
    lines.append("    // Context node")
    lines.append("    node [shape=rectangle, style=rounded, fillcolor=white];")
    if 'TissueType' in cpts and len(cpts['TissueType']) > 0:
        cpt_table = format_cpt_as_html_table(cpts['TissueType'], 'TissueType', max_configs=1)
        lines.append(f"    TissueType [label=<{cpt_table}>];")
    else:
        lines.append("    TissueType [label=\"TissueType\\n(no CPT)\"];")
    
    lines.append("")
    lines.append("    // Edges")
    lines.append("")
    
    # CDK pathway edges
    lines.append("    // CDK pathway")
    for gene in gene_nodes_cdk:
        lines.append(f"    {gene} -> CDK_Pathway;")
    
    # Histone transcription edges
    lines.append("")
    lines.append("    // Histone / transcription")
    for gene in gene_nodes_histone:
        lines.append(f"    {gene} -> Histone_Transcription;")
    
    # DNA damage response edges
    lines.append("")
    lines.append("    // DNA damage response")
    lines.append("    TP53_mut -> DNA_Damage_Response;")
    for gene in gene_nodes_dna:
        lines.append(f"    {gene} -> DNA_Damage_Response;")
    
    # Growth factor signaling edges
    lines.append("")
    lines.append("    // Growth factor signaling")
    for gene in gene_nodes_growth:
        lines.append(f"    {gene} -> GrowthFactor_Signaling;")
    
    # Assembly-level dependencies
    lines.append("")
    lines.append("    // Assembly-level dependencies")
    lines.append("    GrowthFactor_Signaling -> CDK_Pathway;")
    lines.append("    Histone_Transcription -> CDK_Pathway;")
    lines.append("    DNA_Damage_Response -> CDK_Pathway;")
    lines.append("    DNA_Damage_Response -> Histone_Transcription;")
    
    # Tissue type edges
    lines.append("")
    lines.append("    // Tissue type dependencies")
    lines.append("    TissueType -> CDK_Pathway;")
    lines.append("    TissueType -> Histone_Transcription;")
    lines.append("    TissueType -> GrowthFactor_Signaling;")
    lines.append("    TissueType -> DNA_Damage_Response;")
    
    # Drug response edges
    lines.append("")
    lines.append("    // Assemblies -> Drug response")
    for pathway in pathway_nodes:
        lines.append(f"    {pathway} -> DrugResponse;")
    
    lines.append("}")
    
    # Write output
    with open(output_file, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"Created DOT file: {output_file}")

def main():
    cpts_ml_file = "cpts_ml_hidden_pathways.json"
    cpts_em_file = "cpts_em_hidden_pathways.json"
    
    # Create DOT with ML CPTs
    print("Creating DOT file with ML CPTs (hidden pathways)...")
    create_dot_with_cpts(cpts_ml_file, "palbo_bn_hidden_ml.dot", "ML - Observed Only")
    
    # Create DOT with EM CPTs
    print("\nCreating DOT file with EM CPTs (hidden pathways)...")
    create_dot_with_cpts(cpts_em_file, "palbo_bn_hidden_em.dot", "EM - Observed + Hidden")
    
    print("\nDone! Generated files:")
    print("  - palbo_bn_hidden_ml.dot (ML - only observed nodes have CPTs)")
    print("  - palbo_bn_hidden_em.dot (EM - all nodes including hidden)")
    print("\nTo visualize, run:")
    print("  dot -Tpng palbo_bn_hidden_ml.dot -o palbo_bn_hidden_ml.png")
    print("  dot -Tpng palbo_bn_hidden_em.dot -o palbo_bn_hidden_em.png")

if __name__ == "__main__":
    main()


