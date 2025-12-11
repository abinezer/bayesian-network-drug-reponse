#!/usr/bin/env python3
"""
analyze_cancerxgene_palbociclib.py

Usage:
    python analyze_cancerxgene_palbociclib.py --data-dir /path/to/dataset_folder

Assumptions about file names in the folder:
    - cell2mutation.txt
    - cell2cnamplification.txt
    - cell2cndelection.txt
    - cell2ind.txt
    - gene2ind.txt
    - labels_train.txt or train_labels.txt (optional - script will attempt to infer)
    - labels_test.txt or test_labels.txt (optional)

If the dataset already comes split into train/test, place both in the data dir and pass the path.
If labels files are named differently, update the filenames below or pass them via code.
"""

import os
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    mean_squared_error, r2_score
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------------
# Helpers to load files
# ---------------------
def load_index_map(path):
    """
    Load a two-column map file (name index) into dict name->index.
    Accepts files where lines are "name<TAB>index" or "name index".
    """
    mapping = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name, idx = parts[1], int(parts[0])
            mapping[name] = idx
    return mapping

def parse_cell2feature(path, cell2ind, gene2ind, n_cells=None, n_genes=None):
    """
    Parse files like "cell2mutation" that usually contain rows:
      cell_id <tab> gene_name <tab> value
    or sometimes indices. We will try to be flexible.

    Returns a CSR sparse matrix of shape (n_cells, n_genes).
    If n_cells or n_genes not provided, infer from mappings.
    """
    rows = []
    cols = []
    vals = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Allow "cell gene value" or "cell gene" (binary assumed 1)
            if len(parts) == 2:
                cell_name, gene_name = parts
                value = 1
            elif len(parts) >= 3:
                cell_name, gene_name, value = parts[0], parts[1], parts[2]
                try:
                    value = float(value)
                except:
                    value = 1.0
            else:
                continue

            # map to indices
            if cell_name not in cell2ind or gene_name not in gene2ind:
                # skip unknowns (robustness)
                continue
            r = cell2ind[cell_name]
            c = gene2ind[gene_name]
            rows.append(r)
            cols.append(c)
            vals.append(value)

    if n_cells is None:
        n_cells = max(cell2ind.values()) + 1
    if n_genes is None:
        n_genes = max(gene2ind.values()) + 1

    mat = sp.coo_matrix((vals, (rows, cols)), shape=(n_cells, n_genes)).tocsr()
    return mat

def load_labels(path, cell2ind):
    """
    expected: lines like "cell_id <tab> label" or "cell_id label"
    Returns a pandas Series indexed by cell index.
    """
    labels = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 2:
                cell_name, val = parts[0], parts[-1]
                try:
                    val = float(val)
                except:
                    try:
                        val = float(val.replace(',', '.'))
                    except:
                        val = val
                if cell_name in cell2ind:
                    labels[cell2ind[cell_name]] = val
    return pd.Series(labels)

# -------------------------
# EDA and plotting helpers
# -------------------------
def plot_missingness(mutation, ampl, deletion, out_dir):
    """
    Create a simple missingness visualization: fraction of non-zero features per cell and per gene
    """
    os.makedirs(out_dir, exist_ok=True)
    n_cells, n_genes = mutation.shape
    # For missingness: we'll treat a zero across all three channels as "missing measurement" only if the dataset uses sparse reporting.
    combined_nonzero = ((mutation != 0).astype(int) + (ampl != 0).astype(int) + (deletion != 0).astype(int))
    per_cell_nonzero = np.array(combined_nonzero.sum(axis=1)).ravel()
    per_gene_nonzero = np.array(combined_nonzero.sum(axis=0)).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    sns.histplot(per_cell_nonzero, bins=50, ax=axes[0])
    axes[0].set_title("Per-cell number of measured gene-entries (non-zero across any channel)")
    axes[0].set_xlabel("Non-zero feature count")

    sns.histplot(per_gene_nonzero, bins=50, ax=axes[1])
    axes[1].set_title("Per-gene number of cell-lines with non-zero measurement")
    axes[1].set_xlabel("Number of cells")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nonzero_counts.png"), dpi=150)
    plt.close(fig)

    # proportion of measured features per cell as heatmap for first N genes
    Ngenes = min(200, n_genes)
    subset = combined_nonzero[:min(200, n_cells), :Ngenes].toarray()
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    sns.heatmap((subset>0).astype(int), cmap="viridis", cbar=False)
    ax.set_title(f"Measured features heatmap (cells x first {Ngenes} genes)")
    plt.savefig(os.path.join(out_dir, "measured_heatmap_small.png"), dpi=150)
    plt.close(fig)

def print_basic_stats(mutation, ampl, deletion, labels_series=None):
    print("---- Basic dataset stats ----")
    print("Shape (cells x genes):", mutation.shape)
    def nz(mat):
        return mat.nnz
    print("Mutation nonzeros:", nz(mutation))
    print("Amplification nonzeros:", nz(ampl))
    print("Deletion nonzeros:", nz(deletion))
    if labels_series is not None:
        print("Number of labeled cells:", len(labels_series))
        val_counts = pd.Series(labels_series.values).value_counts(dropna=False)
        print("Label value counts (top):")
        print(val_counts.head(10))

# -------------------------
# Modeling helpers
# -------------------------
def build_feature_matrix(mutation, ampl, deletion, max_features=20000, var_threshold=1e-5):
    """
    Stack feature channels horizontally (mut, ampl, del). Returns dense array.
    Optionally reduce features by low-variance threshold.
    """
    from scipy.sparse import hstack
    X_sparse = hstack([mutation, ampl, deletion], format='csr')
    print("Combined sparse shape:", X_sparse.shape)
    # Optionally remove zero-variance features
    vt = VarianceThreshold(threshold=var_threshold)
    # VarianceThreshold expects dense or sparse; it supports sparse in sci-kit 0.24+
    X_reduced = vt.fit_transform(X_sparse)
    print("After variance threshold shape:", X_reduced.shape)
    # Optionally limit to top max_features by variance
    if X_reduced.shape[1] > max_features:
        variances = np.array(X_reduced.power(2).mean(axis=0)).ravel() if sp.issparse(X_reduced) else np.var(X_reduced, axis=0)
        top_idx = np.argsort(variances)[-max_features:]
        if sp.issparse(X_reduced):
            X_reduced = X_reduced[:, top_idx].toarray()
        else:
            X_reduced = X_reduced[:, top_idx]
    elif sp.issparse(X_reduced):
        X_reduced = X_reduced.toarray()
    return X_reduced

def train_evaluate(X, y, problem_type='classification', out_dir='./outputs', random_state=42):
    os.makedirs(out_dir, exist_ok=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.pkl'))

    if problem_type == 'classification':
        clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        roc_scores = cross_val_score(clf, Xs, y, cv=cv, scoring='roc_auc')
        acc_scores = cross_val_score(clf, Xs, y, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(clf, Xs, y, cv=cv, scoring='f1')
        print("CV ROC AUC: {:.4f} ± {:.4f}".format(np.mean(roc_scores), np.std(roc_scores)))
        print("CV Accuracy: {:.4f} ± {:.4f}".format(np.mean(acc_scores), np.std(acc_scores)))
        print("CV F1: {:.4f} ± {:.4f}".format(np.mean(f1_scores), np.std(f1_scores)))

        # Fit final model and save
        clf.fit(Xs, y)
        joblib.dump(clf, os.path.join(out_dir, 'rf_classifier.pkl'))
        # feature importances
        importances = clf.feature_importances_
        topk = min(30, len(importances))
        idx = np.argsort(importances)[::-1][:topk]
        plt.figure(figsize=(8,6))
        sns.barplot(y=np.arange(topk).astype(str), x=importances[idx])
        plt.xlabel("Feature importance")
        plt.title("Top feature importances (by index)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "feature_importances.png"), dpi=200)
        plt.close()

    else:
        # regression
        reg = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        mse = -cross_val_score(reg, Xs, y, cv=cv, scoring='neg_mean_squared_error')
        r2 = cross_val_score(reg, Xs, y, cv=cv, scoring='r2')
        print("CV MSE: {:.4f} ± {:.4f}".format(np.mean(mse), np.std(mse)))
        print("CV R2: {:.4f} ± {:.4f}".format(np.mean(r2), np.std(r2)))
        reg.fit(Xs, y)
        joblib.dump(reg, os.path.join(out_dir, 'rf_regressor.pkl'))

# -------------------------
# Main routine
# -------------------------
def main(args):
    data_dir = args.data_dir
    out_dir = args.out_dir

    # expected files
    cell2ind_path = os.path.join(data_dir, 'cell2ind.txt')
    gene2ind_path = os.path.join(data_dir, 'gene2ind.txt')
    mutation_path = os.path.join(data_dir, 'cell2mutation.txt')
    ampl_path = os.path.join(data_dir, 'cell2cnamplification.txt')
    del_path = os.path.join(data_dir, 'cell2cndelection.txt')

    # if some files missing, raise informative errors
    for p in [cell2ind_path, gene2ind_path, mutation_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    print("Loading index maps...")
    cell2ind = load_index_map(cell2ind_path)
    gene2ind = load_index_map(gene2ind_path)
    n_cells = max(cell2ind.values()) + 1
    n_genes = max(gene2ind.values()) + 1
    print(f"Detected {n_cells} cells and {n_genes} genes from index files.")

    print("Parsing mutation matrix...")
    mutation = parse_cell2feature(mutation_path, cell2ind, gene2ind, n_cells=n_cells, n_genes=n_genes)
    print("Parsed mutation shape:", mutation.shape)

    # optional channels
    ampl = sp.csr_matrix((n_cells, n_genes))
    deletion = sp.csr_matrix((n_cells, n_genes))
    if os.path.exists(ampl_path):
        print("Parsing amplification matrix...")
        ampl = parse_cell2feature(ampl_path, cell2ind, gene2ind, n_cells=n_cells, n_genes=n_genes)
    else:
        print("Amplification file not found; continuing with zeros for amplification.")

    if os.path.exists(del_path):
        print("Parsing deletion matrix...")
        deletion = parse_cell2feature(del_path, cell2ind, gene2ind, n_cells=n_cells, n_genes=n_genes)
    else:
        print("Deletion file not found; continuing with zeros for deletion.")

    # Attempt to load label files: check train/test or labels in folder
    label_files = [f for f in os.listdir(data_dir) if 'label' in f.lower() or 'response' in f.lower() or 'drug' in f.lower()]
    labels = None
    if label_files:
        # pick first reasonable file
        chosen = label_files[0]
        print("Found possible label file(s):", label_files)
        print("Loading labels from:", chosen)
        labels = load_labels(os.path.join(data_dir, chosen), cell2ind)
    else:
        print("No label files automatically detected. If you have labels (train/test), put them in the data dir with 'label' in the filename.")
    # EDA
    os.makedirs(out_dir, exist_ok=True)
    plot_missingness(mutation, ampl, deletion, out_dir)
    print_basic_stats(mutation, ampl, deletion, labels)

    # If labels exist, build X and run baseline model
    if labels is not None and len(labels) > 5:
        # Align X and y
        X_all = build_feature_matrix(mutation, ampl, deletion, max_features=args.max_features)
        # pick indices that have labels
        labeled_idx = labels.index.values
        y = labels.loc[labeled_idx].values
        X = X_all[labeled_idx, :]
        print("X shape for labeled subset:", X.shape)

        # detect problem type: if y takes two unique values -> classification; if continuous -> regression
        unique_vals = np.unique(y)
        if len(unique_vals) <= 2:
            problem_type = 'classification'
            # convert to int labels
            y_proc = np.array([int(v) for v in y])
        else:
            problem_type = 'regression'
            y_proc = y.astype(float)

        print("Detected problem type:", problem_type)
        # Simple imputation - although features are binary/sparse, we convert to numeric; use mean for regression/classification
        imp = SimpleImputer(strategy='constant', fill_value=0)
        X_imp = imp.fit_transform(X)
        joblib.dump(imp, os.path.join(out_dir, 'imputer.pkl'))

        train_evaluate(X_imp, y_proc, problem_type=problem_type, out_dir=out_dir)

        # Save basic EDA CSVs
        pd.DataFrame({
            'cell_index': np.arange(mutation.shape[0]),
            'measured_count': np.array(((mutation!=0).astype(int) + (ampl!=0).astype(int) + (deletion!=0).astype(int)).sum(axis=1)).ravel()
        }).to_csv(os.path.join(out_dir, 'cell_measured_counts.csv'), index=False)
    else:
        print("No labels available or insufficient labels. Only EDA and missingness analysis were produced.")

    print("Analysis completed. Outputs saved to:", out_dir)
    print("If you want me to run model training / cross-validation on a train/test split provided in separate files, re-run after providing label files named e.g. 'train_labels.txt' and 'test_labels.txt'.")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/Users/GayathriRajesh/Desktop/UCSD 27/Fall 2025/250A/Palbociclib_train',
                        help='Path to folder containing the dataset files (download Drive folder here).')
    parser.add_argument('--out-dir', type=str, default='./outputs',
                        help='Folder to write outputs (figures, models, CSVs).')
    parser.add_argument('--max-features', type=int, default=20000,
                        help='Max number of features to keep (by variance) to prevent extreme memory usage.')
    args = parser.parse_args()
    main(args)
