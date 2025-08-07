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
plt.show()