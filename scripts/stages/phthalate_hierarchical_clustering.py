"""
Perform hierarchical clustering on phthalate activity matrix
based on similar assay activity values.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

cachedir = Path("cache")
datadir = cachedir / "entity_similarity2"
# Load the phthalates data
activity_matrix_filled = pd.read_parquet(datadir / "activity_matrix_filled.parquet")

def example_cluster():
    # Perform hierarchical clustering
    Z = linkage(activity_matrix_filled.T, method='ward')

    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(
        Z,
        # labels=activity_matrix_filled.index,
        labels=None,
        leaf_rotation=90, leaf_font_size=10
    )
    plt.title('Hierarchical Clustering Dendrogram of Phthalates')
    plt.xlabel('Phthalates')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(datadir / "phthalate_dendrogram.png")

# --- Sweep over multiple linkage methods and distance metrics
linkage_methods = ['ward', 'average', 'complete']
distance_metrics = ['euclidean', 'cityblock', 'cosine']

def parameter_sweep():
    """
    Perform a parameter sweep over different linkage methods and distance metrics.
    """
    for method in linkage_methods:
        for metric in distance_metrics:
            # Ward linkage is only defined for Euclidean distances
            if method == 'ward' and metric != 'euclidean':
                continue

            # Perform hierarchical clustering with the chosen parameters
            Z = linkage(
                activity_matrix_filled.T,
                method=method,
                metric=metric
            )

            # Plot and save the dendrogram
            plt.figure(figsize=(10, 7))
            dendrogram(
                Z,
                labels=None,
                leaf_rotation=90,
                leaf_font_size=10
            )
            plt.title(f'Hierarchical Clustering (method={method}, metric={metric})')
            plt.xlabel('Assays')
            plt.ylabel('Distance')
            plt.tight_layout()
            out_file = datadir / f'phthalate_dendrogram_{method}_{metric}.png'
            plt.savefig(out_file)
            plt.close()


def evaluate_cluster_quality():
    """
    Evaluate clustering quality using cophenetic correlation coefficient and silhouette score.
    """
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist
    from sklearn.metrics import silhouette_score
    from scipy.cluster.hierarchy import fcluster
    from tabulate import tabulate

    def cluster_quality(X, Z):
        # Cophenetic correlation coefficient (CCC)
        c, _ = cophenet(Z, pdist(X))
        return c

    best_model = None
    scores = []
    for method in linkage_methods:
        for metric in distance_metrics:
            if method == 'ward' and metric != 'euclidean':
                continue
            Z = linkage(activity_matrix_filled.T, method=method, metric=metric)
            ccc = cluster_quality(activity_matrix_filled.T, Z)
            # silhouettes need a flat clustering; cut at the first big jump (heuristic)
            cut_height = np.percentile(Z[:, 2], 90)  # top-10 % jump
            
            labels = fcluster(Z, t=cut_height, criterion='distance')
            sil = silhouette_score(activity_matrix_filled.T, labels, metric=metric)
            scores.append((method, metric, ccc, sil))
            if best_model is None or sil > best_model[-1]:
                best_model = (method, metric, ccc, sil)

    # Print results as a formatted table
    headers = ["Method", "Metric", "CCC", "Silhouette"]
    table = [[row[0], row[1], f"{row[2]:.3f}", f"{row[3]:.3f}"] for row in scores]
    print(tabulate(table, headers=headers, tablefmt="github"))
    print(f"\nBEST ⇒ Method: {best_model[0]}, Metric: {best_model[1]}, CCC: {best_model[2]:.3f}, Silhouette: {best_model[3]:.3f}")

    return best_model

best_model = evaluate_cluster_quality()
