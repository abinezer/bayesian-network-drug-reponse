from project_2 import read_files
from typing import Dict, Tuple, Optional
import numpy as np

GENES = ['RB1',
         'CCND1',
         'CDK4',
         'CDK6',
         'CDKN2A',
         'CDKN2B',
         'RUNX1',
         'TERT',
         'MYC',
         'CREBBP',
         'EP300',
         'HDAC1',
         'HDAC2',
         'KAT6A',
         'TP53',
         'TBL1XR1',
         'RAD51C',
         'CHEK1',
         'BRCA1',
         'BRCA2',
         'EGFR',
         'ERBB2',
         'ERBB3',
         'FGFR1',
         'FGFR2',
         'PIK3CA'
         ]

print(len(GENES))

def estimate_gene_mutation_mle_named(cell2mutate, gene2idx_map):
    probs = cell2mutate.mean(axis=0)
    
    gene_probs = {}
    for idx, name in gene2idx_map.items():
        if name in GENES:
            gene_probs[name] = probs[idx]
    
    return gene_probs

def estimate_drug_response_given_gene_mutation(drug_response, cell2idx_map, cell2mutate, gene2idx_map):
    probs = {}

    # Reverse 
    name_to_gene_idx = {v: k for k, v in gene2idx_map.items()}

    for gene in GENES:
        if gene not in name_to_gene_idx:
            probs[gene] = None
            continue

        g_idx = name_to_gene_idx[gene]

        mutated_cells = []
        response_values = []

        for cell_idx, cell_name in cell2idx_map.items():
            if cell2mutate[cell_idx][g_idx] == 1 and cell_name in drug_response:
                mutated_cells.append(cell_name)
                response_values.append(drug_response[cell_name])

        if len(mutated_cells) == 0:
            probs[gene] = None  
            continue

        # MLE: fraction of mutated cells that have response = 1
        p = sum(response_values) / len(response_values)
        probs[gene] = p

    return probs

from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

def build_y(cell2idx_map, drug_response):
    """Build y only for cells that have a recorded drug response."""
    y = []
    valid_indices = []

    for idx in range(len(cell2idx_map)):
        cell_name = cell2idx_map[idx]
        if cell_name in drug_response:
            y.append(drug_response[cell_name])
            valid_indices.append(idx)

    return np.array(y), np.array(valid_indices)


def predict_scores(cell2mutate, gene2idx_map, response_given_mutation):
    name_to_idx = {v: k for k, v in gene2idx_map.items()}

    coeffs = []
    gene_indices = []

    for gene in GENES:
        if gene in name_to_idx and response_given_mutation.get(gene) is not None:
            gene_indices.append(name_to_idx[gene])
            coeffs.append(response_given_mutation[gene])

    coeffs = np.array(coeffs)
    X = cell2mutate[:, gene_indices] 

    scores = X @ coeffs              
    return scores

def plot_auc(cell2mutate_test, gene2idx_map_test, response_given_mutation,
             cell2idx_map_test, drug_response_test):

    scores_all = predict_scores(
        cell2mutate_test, gene2idx_map_test, response_given_mutation
    )
    y, valid_indices = build_y(cell2idx_map_test, drug_response_test)
    scores = scores_all[valid_indices]

    # Now compute ROC
    fpr, tpr, _ = roc_curve(y, scores)
    auc_val = auc(fpr, tpr)

    # plot
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    return auc_val



dir = "Palbociclib_train"
cell2amp, cell2delete, cell2mutate, cell2idx_map, gene2idx_map, drug_response = read_files(dir)

test_dir = "Palbociclib_test"
cell2amp_test, cell2delete_test, cell2mutate_test, cell2idx_map_test, gene2idx_map_test, drug_response_test = read_files(test_dir)

gene_cpt = estimate_gene_mutation_mle_named(cell2mutate, gene2idx_map)

print(gene_cpt)

response_given_mutation = estimate_drug_response_given_gene_mutation(drug_response, cell2idx_map, cell2mutate, gene2idx_map)

print(response_given_mutation)

auc_value = plot_auc(
    cell2mutate_test,
    gene2idx_map_test,
    response_given_mutation,
    cell2idx_map_test,
    drug_response_test
)

print("AUC =", auc_value)



# def build_response_vector(cell2idx_map: Dict[int, str],
#                           drug_response: Dict[str, float],
#                           n_cells: int,
#                           missing_strategy: str = "nan"
#                          ) -> np.ndarray:
#     """
#     Create an array r of length n_cells such that r[i] is drug response prob for the cell
#     whose index is i (according to cell2idx_map).

#     Args:
#         cell2idx_map: mapping index -> cell_name (as in your read_files).
#         drug_response: mapping cell_name -> response probability (float in [0,1]).
#         n_cells: number of rows in cell2mutate (expected max index+1).
#         missing_strategy: "nan" (default) to set missing responses to np.nan,
#                           "zero" to set missing responses to 0.0,
#                           "drop" to raise later errors if any missing entries exist.

#     Returns:
#         np.ndarray shape (n_cells,) with response probabilities aligned to row indices.
#     """
#     r = np.full(n_cells, np.nan, dtype=float)
#     # cell2idx_map maps int index -> cell_name (string)
#     for idx in range(n_cells):
#         if idx not in cell2idx_map:
#             # index missing from mapping -> leave as nan
#             continue
#         cell_name = cell2idx_map[idx]
#         # r[idx] = float(drug_response[cell_name])
        
#         if cell_name in drug_response:
#             r[idx] = float(drug_response[cell_name])
#         else:
#             if missing_strategy == "zero":
#                 r[idx] = 0.0
#             elif missing_strategy == "drop":
#                 raise KeyError(f"Missing drug response for cell {cell_name} (index {idx}).")
#             # else leave np.nan
#     return r

# def conditional_response_given_mutation(
#     cell2mutate: np.ndarray,
#     cell2idx_map: Dict[int, str],
#     drug_response: Dict[str, float],
#     laplace_alpha: float = 0.0,
#     missing_strategy: str = "nan",
# ) -> np.ndarray:
#     """
#     Estimate P(response | gene mutated) for each gene.

#     Args:
#         cell2mutate: (n_cells, n_genes) binary matrix.
#         cell2idx_map: mapping index -> cell_name.
#         gene2idx_map: mapping index -> gene_name.
#         drug_response: mapping cell_name -> response probability (float).
#         laplace_alpha: pseudo-count for Laplace smoothing (default 0 -> no smoothing).
#         missing_strategy: behavior for missing cell responses: "nan", "zero", or "drop".
#         return_df: if True return a pandas DataFrame, else return arrays only.

#     Returns:
#         p_cond (np.ndarray): shape (n_genes,) P(response | gene mutated) (np.nan if no mutated cells and no smoothing)
#         df (pd.DataFrame or None): if return_df True, DataFrame with columns:
#             ['gene_index', 'gene_name', 'mutated_count', 'sum_response', 'p_given_mut', 'se', 'ci95_low', 'ci95_high']
#     """
#     n_cells, n_genes = cell2mutate.shape
#     # Build response vector aligned to rows:
#     r = build_response_vector(cell2idx_map, drug_response, n_cells, missing_strategy=missing_strategy)

#     # We will ignore rows where r is nan when computing averages (equivalent to using only cells with known responses)
#     valid_mask = ~np.isnan(r)
#     if not np.any(valid_mask):
#         raise ValueError("No valid cell response values found. Check drug_response and cell2idx_map.")

#     # counts and weighted sums using only valid rows
#     mutated_counts_all = cell2mutate.sum(axis=0)  # counts across all cells (including those with nan responses)
#     # To compute sums only over cells with known r, mask rows with nan
#     masked_cell2mutate = cell2mutate.copy()
#     masked_cell2mutate[~valid_mask, :] = 0  # zero out rows with missing response
#     mutated_counts = masked_cell2mutate.sum(axis=0)  # counts among cells with known r
#     weighted_sums = (masked_cell2mutate * r[:, None]).sum(axis=0)

    
#     p_cond = np.full(n_genes, np.nan, dtype=float)
#     nonzero = mutated_counts > 0
#     p_cond[nonzero] = weighted_sums[nonzero] / mutated_counts[nonzero]
    
#     return p_cond

# def conditional_response_given_no_mutation(
#     cell2mutate: np.ndarray,
#     cell2idx_map: Dict[int, str],
#     drug_response: Dict[str, float],
#     missing_strategy: str = "nan",
# ) -> np.ndarray:
#     """
#     Estimate P(response | gene NOT mutated) for each gene.

#     Args:
#         cell2mutate: (n_cells, n_genes) binary mutation matrix.
#         cell2idx_map: mapping index -> cell_name.
#         drug_response: mapping cell_name -> response (float).
#         laplace_alpha: optional Laplace smoothing.
#         missing_strategy: "nan", "zero", or "drop" for unknown responses.

#     Returns:
#         p_cond_no_mut (np.ndarray): shape (n_genes,)
#             P(response | gene NOT mutated)
#             (nan if no non-mutated cells and no smoothing)
#     """

#     n_cells, n_genes = cell2mutate.shape

#     # Build response vector aligned to cell order
#     r = build_response_vector(
#         cell2idx_map, drug_response, n_cells, missing_strategy=missing_strategy
#     )

#     # Ignore cells with missing response
#     valid_mask = ~np.isnan(r)
#     if not np.any(valid_mask):
#         raise ValueError("No valid cell response values found.")

#     # Mask missing-response rows in mutation matrix
#     masked = cell2mutate.copy()
#     masked[~valid_mask, :] = np.nan  # temporarily use nan

#     # Boolean mask for NON-mutation among valid cells
#     nonmut_mat = (masked == 0)

#     # Count non-mutated cells per gene
#     nonmut_counts = np.nansum(nonmut_mat, axis=0)

#     # Sum response among non-mutated cells
#     weighted_sums = np.nansum(nonmut_mat * r[:, None], axis=0)

#     # Compute conditional response
#     p_cond_no_mut = np.full(n_genes, np.nan, dtype=float)
#     nonzero_mask = nonmut_counts > 0

   
#     p_cond_no_mut[nonzero_mask] = (
#             weighted_sums[nonzero_mask] / nonmut_counts[nonzero_mask]
#         )

#     return p_cond_no_mut