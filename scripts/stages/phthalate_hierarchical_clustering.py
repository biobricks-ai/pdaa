"""
Perform hierarchical clustering on phthalate activity matrix
based on similar assay activity values.
"""

import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

cachedir = Path("cache")
datadir = cachedir / "entity_similarity2"
# Load the phthalates data
activity_matrix_filled = pd.read_parquet(datadir / "activity_matrix_filled.parquet")

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
