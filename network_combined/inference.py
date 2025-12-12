import itertools
from itertools import product
import numpy as np

from learn_cpts import (
    read_files, create_network_structure, map_gene_to_indices,
    BayesianNetwork
)

import json
from tqdm import tqdm


def joint_prob(assignment, structure, CPTs):
    """
    Compute joint probability P(X1=x1, X2=x2, ..., Xn=xn)
    assignment: dict {node: value}
    structure: dict node -> list of parents
    CPTs: dict node -> dict of parent_tuple_str -> {0: p0, 1: p1}
    """
    prob = 1.0

    for X in structure:
        parents = structure[X]
        parent_values = tuple(assignment[p] for p in parents) if parents else ()
        # Convert tuple to string as used in JSON
        parent_key = str(parent_values)

        # If the key is missing (shouldn’t happen if CPTs are fully expanded)
        if parent_key not in CPTs[X]:
            # Use uniform default if missing
            p = 0.5
        else:
            p = CPTs[X][parent_key][str(assignment[X])]

        prob *= p

    return prob

def compute_p_drugresponse_given_CDK4_mut(structure, CPTs):
    """
    Compute P(DrugResponse=1 | CDK4_mut=1) with a progress bar.
    """
    nodes = list(structure.keys())
    total_assignments = 2 ** len(nodes)  # total combinations

    numerator = 0.0
    denominator = 0.0

    for values in tqdm(product([0, 1], repeat=len(nodes)), total=total_assignments, desc="Computing joint probabilities"):
        assign = dict(zip(nodes, values))
        if assign['CDK4_mut'] != 1:
            continue
        joint = joint_prob(assign, structure, CPTs)
        if assign['DrugResponse'] == 1:
            numerator += joint
        denominator += joint

    return numerator / denominator if denominator > 0 else 0.0

if __name__ == "__main__":
    path = "/Users/GayathriRajesh/Desktop/UCSD 27/Fall 2025/250A/project/network_combined/cpts_em_hidden_pathways.json"
    
    with open(path, 'r') as f:
        CPTs = json.load(f)
        
    print("Hii")
    
    structure, gene_nodes = create_network_structure()
    
    val = compute_p_drugresponse_given_CDK4_mut(structure, CPTs)
        
   