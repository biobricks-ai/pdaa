#!/usr/bin/env python3
"""
Sparse Dictionary Learning on Phthalate Assay Matrix
===================================================

This script replaces hierarchical clustering with Sparse Dictionary Learning
(SDL) to uncover groups of assays that jointly explain chemical activity
patterns. It outputs:

1. `component_loadings.csv`: top-k assays (features) for each learned
   dictionary component with their absolute loadings.
2. `chemical_codes.parquet`: sparse code matrix giving the activation of
   each component for every chemical (sample).
3. `component_heatmap_<i>.png`: heatmaps visualising the loadings of every
   component across all assays for manual inspection.

SDL Objectives
--------------
* **Interpretability**: Each component is sparse so its biological meaning
  can be interpreted by looking at a handful of high-loading assays.
* **Redundancy Handling**: Components capture shared variance, reducing the
  dimensionality without averaging away distinct biological signals.

Run
---
```bash
python sparse_dictionary_learning_phthalates.py
```
Assumes `cache/entity_similarity2/activity_matrix_filled.parquet` exists, as
in the original hierarchical clustering script.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------

N_COMPONENTS = 40            # Number of dictionary atoms to learn
ALPHA = 1.0                  # Sparsity regularisation strength (higher => sparser)
TOP_K = 10                   # Number of top assays to keep per component
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

CACHEDIR = Path("cache")
DATADIR = CACHEDIR / "entity_similarity2"
DATADIR.mkdir(parents=True, exist_ok=True)

activity_matrix_fp = DATADIR / "activity_matrix_filled.parquet"
if not activity_matrix_fp.exists():
    raise FileNotFoundError(
        f"Expected {activity_matrix_fp} to exist. Verify your data path.")

activity_matrix = pd.read_parquet(activity_matrix_fp)

# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

# Rows = chemicals, Columns = assays
scaler = StandardScaler()
X = scaler.fit_transform(activity_matrix.values)  # shape: (n_chemicals, n_assays)

# ---------------------------------------------------------------------------
# Sparse Dictionary Learning
# ---------------------------------------------------------------------------

print("Fitting DictionaryLearning (n_components =", N_COMPONENTS, ") ...")

with warnings.catch_warnings():
    # Ignore convergence warnings if any
    warnings.simplefilter("ignore")
    dict_learner = DictionaryLearning(
        n_components=N_COMPONENTS,
        alpha=ALPHA,
        max_iter=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        fit_algorithm="lars",
        transform_algorithm="lasso_lars",
    ).fit(X)

components = dict_learner.components_  # shape: (n_components, n_assays)
print("Done. Components shape:", components.shape)

# ---------------------------------------------------------------------------
# Save chemical codes (activations)
# ---------------------------------------------------------------------------

codes = dict_learner.transform(X)  # shape: (n_chemicals, n_components)
code_df = pd.DataFrame(
    codes,
    index=activity_matrix.index,  # chemical identifiers
    columns=[f"Comp_{i:02d}" for i in range(N_COMPONENTS)],
)
code_df.to_parquet(DATADIR / "chemical_codes.parquet")
print("Saved sparse codes to chemical_codes.parquet")

# ---------------------------------------------------------------------------
# Save top-k assays per component
# ---------------------------------------------------------------------------

assay_names = activity_matrix.columns.tolist()
rows = []
for i, comp in enumerate(components):
    abs_comp = np.abs(comp)
    top_idx = np.argsort(abs_comp)[-TOP_K:][::-1]
    for rank, idx in enumerate(top_idx, start=1):
        rows.append(
            {
                "Component": f"Comp_{i:02d}",
                "Rank": rank,
                "Assay": assay_names[idx],
                "Loading": comp[idx],
                "AbsLoading": abs_comp[idx],
            }
        )

top_loadings_df = pd.DataFrame(rows)
loadings_fp = DATADIR / "component_loadings.csv"
top_loadings_df.to_csv(loadings_fp, index=False)
print(f"Saved top loadings per component to {loadings_fp}")

# ---------------------------------------------------------------------------
# Visualise each component as a heatmap for manual inspection
# ---------------------------------------------------------------------------

print("Generating component heatmaps ...")

for i, comp in enumerate(components):
    plt.figure(figsize=(12, 1.8))
    plt.imshow(comp[np.newaxis, :], aspect="auto", cmap="coolwarm", vmin=-np.max(np.abs(comp)), vmax=np.max(np.abs(comp)))
    plt.colorbar(label="Loading")
    plt.yticks([])
    plt.xticks(np.arange(len(assay_names)), assay_names, rotation=90, fontsize=6)
    plt.title(f"Dictionary Component {i:02d} Loadings")
    plt.tight_layout()
    heatmap_fp = DATADIR / f"component_heatmap_{i:02d}.png"
    plt.savefig(heatmap_fp, dpi=200)
    plt.close()

print("All heatmaps saved in", DATADIR)

# ---------------------------------------------------------------------------
# Quick interpretability report (console)
# ---------------------------------------------------------------------------

def print_component_summary(comp_idx: int):
    """Print top assays for a single component."""
    subset = top_loadings_df[top_loadings_df["Component"] == f"Comp_{comp_idx:02d}"]
    print("\nComponent", comp_idx)
    print("--------------")
    for _, row in subset.iterrows():
        print(f"{row['Rank']:2d}. {row['Assay']} (loading = {row['Loading']:+.3f})")

print("\nTop assays per component:")
for i in range(N_COMPONENTS):
    print_component_summary(i)

print("\nFinished Sparse Dictionary Learning workflow.")
