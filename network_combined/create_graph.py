from graphviz import Digraph

from learn_cpts_combined import (
    read_files, create_network_structure, map_gene_to_indices,
    BayesianNetwork
)

def plot_bn_png(structure, filename='bayesian_network.png'):
    """
    Generate a PNG visualization of a Bayesian Network.
    
    structure: dict mapping node -> list of parents
    filename: output PNG file
    """
    dot = Digraph(comment='Bayesian Network', format='png')
    
    # Optional: color coding
    gene_nodes = [n for n, parents in structure.items() if len(parents) == 0]
    pathway_nodes = [n for n, parents in structure.items() if n not in gene_nodes and n != 'DrugResponse']
    
    # Add nodes
    for n in structure.keys():
        if n in gene_nodes:
            dot.node(n, n, shape='ellipse', style='filled', color='lightblue')
        elif n in pathway_nodes:
            dot.node(n, n, shape='box', style='filled', color='lightgreen')
        else:  # DrugResponse
            dot.node(n, n, shape='diamond', style='filled', color='lightpink')
    
    # Add edges (parents → child)
    for child, parents in structure.items():
        for parent in parents:
            dot.edge(parent, child)
    
    # Render PNG
    dot.render(filename, cleanup=True)
    print(f"Graph saved as {filename}")

# Example usage:
if __name__ == "__main__":
    structure, gene_nodes = create_network_structure()
    plot_bn_png(structure, filename='bn_visualization')
