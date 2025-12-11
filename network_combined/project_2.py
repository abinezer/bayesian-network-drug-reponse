import numpy as np
import os
import matplotlib.pyplot as plt

def read_files(dir):
    median = 0.89
    cell2amp_path = os.path.join(dir, 'cell2cnamplification.txt')
    
    with open(cell2amp_path, 'r') as f:
        values = f.readlines()
        values = [list(map(int, v.strip().split(','))) for v in values]
        
    cell2amp = np.array(values)
    
    cell2delete_path = os.path.join(dir, 'cell2cndeletion.txt')
    
    with open(cell2delete_path, 'r') as f:
        values = f.readlines()
        values = [list(map(int, v.strip().split(',')))for v in values]
        
    cell2delete = np.array(values)
    
    cell2mutation_path = os.path.join(dir, 'cell2mutation.txt')
    
    with open(cell2mutation_path, 'r') as f:
        values = f.readlines()
        values = [list(map(int, v.strip().split(',')))for v in values]
        
    cell2mutate = np.array(values)
    
    gene2idx = os.path.join(dir,'gene2ind.txt')
    gene2idx_map = {}
    
    with open(gene2idx, 'r') as f:
        values = f.readlines()
        values = [(v.strip().split("\t")) for v in values]
        
    for v in values:
        gene2idx_map[int(v[0])] = v[1]
        
    cell2idx = os.path.join(dir,'cell2ind.txt')
    cell2idx_map = {}
    
    with open(cell2idx, 'r') as f:
        values = f.readlines()
        values = [(v.strip().split("\t")) for v in values]
        
    for v in values:
        cell2idx_map[int(v[0])] = v[1]
        
    drug_response_path = os.path.join(dir,'train_data.txt')
    
    with open(drug_response_path, 'r') as f:
        values = f.readlines()
        values = [(v.strip().split("\t")) for v in values]
        
    drug_response = {}
    
    for v in values:
        drug_response[v[0]] = int(float(v[2])<median)
    
        
    return cell2amp, cell2delete, cell2mutate, cell2idx_map, gene2idx_map, drug_response
        
        
def count_gene_events(cell2amp, cell2delete, cell2mutate, cell2idx_map):
    """
    Returns a dict:
        cell_line -> {
            'amplified': count,
            'deleted': count,
            'mutated': count
        }
    """
    summary = {}

    # Iterate over rows (each row = cell line)
    for idx in range(len(cell2amp)):
        cell_name = cell2idx_map[idx]

        amp_count = np.sum(cell2amp[idx] == 1)
        delete_count = np.sum(cell2delete[idx] == 1)
        mutate_count = np.sum(cell2mutate[idx] == 1)

        summary[cell_name] = {
            "amplified": int(amp_count),
            "deleted": int(delete_count),
            "mutated": int(mutate_count)
        }

    return summary

def plot_alteration_burden(cell2amp, cell2delete, cell2mutate, cell2idx_map):
    cells = list(cell2idx_map.values())
    amp_counts = np.sum(cell2amp, axis=1)
    del_counts = np.sum(cell2delete, axis=1)
    mut_counts = np.sum(cell2mutate, axis=1)

    plt.figure(figsize=(10,5))
    plt.plot(amp_counts, label="Amplifications")
    plt.plot(del_counts, label="Deletions")
    plt.plot(mut_counts, label="Mutations")
    plt.legend()
    plt.title("Alteration Burden Across Cell Lines")
    plt.xlabel("Cell Line Index")
    plt.ylabel("Number of Altered Genes")
    plt.show()

    return amp_counts, del_counts, mut_counts

def top_genes_by_frequency(matrix, gene2idx_map, top_k=20):
    freq = np.sum(matrix, axis=0)
    gene_names = np.array([gene2idx_map[i] for i in range(len(freq))])
    
    sorted_idx = np.argsort(freq)[::-1]   # descending order
    top_genes = gene_names[sorted_idx][:top_k]
    top_freq = freq[sorted_idx][:top_k]

    plt.figure(figsize=(10,6))
    plt.barh(top_genes[::-1], top_freq[::-1])
    plt.title("Top Genes by Deletion Frequency")
    plt.xlabel("Number of Cell Lines Altered")
    plt.show()

    return list(zip(top_genes, top_freq))

def plot_correlation_between_event_types(cell2amp, cell2delete, cell2mutate):
    amp_counts = np.sum(cell2amp, axis=1)
    del_counts = np.sum(cell2delete, axis=1)
    mut_counts = np.sum(cell2mutate, axis=1)

    plt.figure(figsize=(6, 6))
    plt.scatter(mut_counts, amp_counts)
    plt.xlabel("Mutation Burden")
    plt.ylabel("Amplification Burden")
    plt.title("Correlation: Mutations vs Amplifications")
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(mut_counts, del_counts)
    plt.xlabel("Mutation Burden")
    plt.ylabel("Deletion Burden")
    plt.title("Correlation: Mutations vs Deletions")
    plt.show()
    
def find_outlier_cell_lines(cell2amp, cell2delete, cell2mutate, cell2idx_map, z_threshold=2.5):
    amp_counts = np.sum(cell2amp, axis=1)
    del_counts = np.sum(cell2delete, axis=1)
    mut_counts = np.sum(cell2mutate, axis=1)

    total = amp_counts + del_counts + mut_counts
    mean = np.mean(total)
    std = np.std(total)

    outliers = []
    for i, val in enumerate(total):
        z = (val - mean) / std
        if abs(z) > z_threshold:
            outliers.append((cell2idx_map[i], val, z))

    return outliers

import seaborn as sns

def top_k_genes(matrix, gene2idx_map, k=20):
    freq = np.sum(matrix, axis=0)
    gene_names = np.array([gene2idx_map[i] for i in range(len(freq))])
    sorted_idx = np.argsort(freq)[::-1]
    return sorted_idx[:k], gene_names[sorted_idx][:k], freq[sorted_idx][:k]

def correlate_amplification_mutation(cell2amp, cell2mutate, gene2idx_map, top_k=20):
    top_amp_idx, top_amp_names, top_amp_freq = top_k_genes(cell2amp, gene2idx_map, top_k)
    top_mut_idx, top_mut_names, top_mut_freq = top_k_genes(cell2mutate, gene2idx_map, top_k)
    combined_gene_idx = list(set(top_amp_idx) | set(top_mut_idx))
    combined_gene_names = [gene2idx_map[i] for i in combined_gene_idx]
    amp_vals = np.sum(cell2amp[:, combined_gene_idx], axis=0)
    mut_vals = np.sum(cell2mutate[:, combined_gene_idx], axis=0)
    correlation = np.corrcoef(amp_vals, mut_vals)[0, 1]
    print(f"\nCorrelation between amplification & mutation (top {top_k} genes each): {correlation:.4f}")
    plt.figure(figsize=(10, 6))
    data = np.vstack([amp_vals, mut_vals])

    sns.heatmap(data, annot=False, fmt="d", cmap="viridis",
                xticklabels=combined_gene_names, yticklabels=["Amplified", "Mutated"])

    plt.title(f"Amplification vs Mutation Frequencies\nTop {top_k} Genes (Union)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    return correlation, combined_gene_names, amp_vals, mut_vals


if __name__ == "__main__":   
    dir = "Palbociclib_train"
    cell2amp, cell2delete, cell2mutate, cell2idx_map, gene2idx_map = read_files(dir)

    summary = count_gene_events(cell2amp, cell2delete, cell2mutate, cell2idx_map)

    plot_alteration_burden(cell2amp, cell2delete, cell2mutate, cell2idx_map)

    top_mut = top_genes_by_frequency(cell2mutate, gene2idx_map)
    top_amp = top_genes_by_frequency(cell2amp, gene2idx_map)
    top_del = top_genes_by_frequency(cell2delete, gene2idx_map)

    # plot_correlation_between_event_types(cell2amp, cell2delete, cell2mutate)

    correlate_amplification_mutation(cell2amp, cell2mutate, gene2idx_map, top_k=20)