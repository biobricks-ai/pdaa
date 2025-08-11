#!/usr/bin/env python3
"""
Compute a simple, defensible toxicity index (TI) from an assay activity matrix.

Pipeline:
1) Z-score assays (columns).
2) Cluster assays using hierarchical clustering on d = 1 - |Pearson r|.
3) For each cluster, compute PC1 (principal component 1), align sign, and score each chemical.
4) TI_equal = mean_c |score_ic|, TI_weighted = sum_c w_c * |score_ic| with w_c ∝ var_explained_pc1.
5) Save cluster assignments, PC1 loadings, per-chemical cluster scores, TI tables, and basic plots.

CLI:
    --input (default: cache/entity_similarity2/activity_matrix_filled.parquet)
    --outdir (default: parent of input)
    --n-clusters (int) manual override
    --corr-threshold (float in (0,1)) optional dendrogram cut at d = 1 - τ
    --silhouette-range (two ints, default 5 60)
    --random-state (int, default 42)

Notes:
- If both --n-clusters and --corr-threshold are given, --n-clusters takes precedence (warning logged).
- Silhouette optimization uses precomputed distances and ignores singleton clusters for the score.
- Warnings:
  * >20% clusters are single-assay (kept, marked).
  * Any cluster with PC1 variance explained < 0.3.
- Single-assay clusters: PC1 reduces to the standardized assay.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem

# ----------------------------- I/O and preprocessing ----------------------------- #

def load_data(input_path: Path) -> pd.DataFrame:
    """
    Load activity matrix with rows=chemicals and columns=assays.

    Returns
    -------
    df : DataFrame [n_chemicals x n_assays]
    """
    df = pd.read_parquet(input_path)
    if df.shape[0] < 2 or df.shape[1] < 2:
        raise ValueError("Input matrix must have at least 2 chemicals and 2 assays.")
    if not np.issubdtype(df.dtypes.values[0], np.number):
        # Heuristic check; more thorough checks happen later.
        logging.warning("Non-numeric dtypes detected; attempting to coerce to numeric.")
        df = df.apply(pd.to_numeric, errors="coerce")
    return df


def zscore_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Z-score each assay (column) to zero mean, unit variance.

    Drops assays with zero post-standardization variance (i.e., constant columns).

    Returns
    -------
    Xz : DataFrame (standardized)
    dropped : list of assay names dropped due to zero variance or all-NaN
    """
    # Compute column-wise mean and std (population std, ddof=0)
    mu = df.mean(axis=0)
    sigma = df.std(axis=0, ddof=0)

    # Identify columns with zero or NaN std (constant or invalid)
    zero_var = sigma.isna() | (sigma == 0)
    dropped = df.columns[zero_var].tolist()
    if dropped:
        logging.warning("Dropping %d assays with zero variance or invalid std.", len(dropped))

    keep_cols = df.columns[~zero_var]
    Xz = (df[keep_cols] - mu[keep_cols]) / sigma[keep_cols]

    # Check for NaNs after standardization; these indicate problematic input
    if Xz.isna().any().any():
        n_bad_cols = Xz.columns[Xz.isna().any(axis=0)].size
        n_bad_cells = int(Xz.isna().sum().sum())
        raise ValueError(
            f"NaNs present after standardization: {n_bad_cells} NaNs across {n_bad_cols} assays. "
            "Please provide a filled matrix."
        )

    return Xz, dropped


# ----------------------------- Clustering ----------------------------- #

def assay_distance_matrix(Xz: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute Pearson correlation across assays (columns), take absolute value,
    and convert to distance d = 1 - |r|.

    Returns
    -------
    D : ndarray [n_assays x n_assays] symmetric distance matrix
    D_condensed : ndarray condensed upper triangle for scipy.linkage
    assays : list of assay names (column order)
    """
    assays = list(Xz.columns)
    # Correlation across assays; Xz is [chemicals x assays]
    R = np.corrcoef(Xz.values, rowvar=False)
    # Numerical guard for tiny floating errors
    R = np.clip(R, -1.0, 1.0)
    D = 1.0 - np.abs(R)
    np.fill_diagonal(D, 0.0)
    # Ensure symmetry
    D = (D + D.T) / 2.0
    D_condensed = squareform(D, checks=False)
    return D, D_condensed, assays


def _labels_for_k(Z: np.ndarray, k: int) -> np.ndarray:
    """Helper to cut dendrogram into exactly k clusters."""
    return fcluster(Z, t=k, criterion="maxclust")


def _labels_for_threshold(Z: np.ndarray, t: float) -> np.ndarray:
    """Helper to cut dendrogram at distance threshold t."""
    return fcluster(Z, t=t, criterion="distance")


def _silhouette_for_labels(D: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """
    Average silhouette using precomputed distances, excluding singleton clusters.

    Returns
    -------
    score : float in [-1, 1] or None if undefined (e.g., all singletons or 1 cluster)
    """
    unique, counts = np.unique(labels, return_counts=True)
    mask_non_single = np.isin(labels, unique[counts >= 2])
    if mask_non_single.sum() < 2:
        return None
    # Subset distances and labels to non-singletons
    D_sub = D[np.ix_(mask_non_single, mask_non_single)]
    labels_sub = labels[mask_non_single]
    try:
        return silhouette_score(D_sub, labels_sub, metric="precomputed")
    except Exception:
        return None


def _threshold_for_k(Z: np.ndarray, n_leaves: int, k: int) -> Optional[float]:
    """
    Compute a distance threshold that yields exactly k clusters for a given linkage Z.

    Picks a midpoint between the (n-k)th and (n-k+1)th merge distances.

    Returns
    -------
    t_line : float or None if undefined (e.g., k out of range)
    """
    if not (2 <= k <= n_leaves - 1):
        return None
    heights = Z[:, 2]
    m = n_leaves - k  # merges needed
    # When m == 0 (k == n), threshold below first merge; not useful to draw
    if m == 0:
        return None
    if m >= len(heights):
        # k == 1 (single cluster); draw above last merge
        return heights[-1] + 1e-6
    low = heights[m - 1]
    high = heights[m] if m < len(heights) else low + 1.0
    return (low + high) / 2.0


def cluster_assays(
    Xz: pd.DataFrame,
    n_clusters: Optional[int],
    corr_threshold: Optional[float],
    silhouette_range: Tuple[int, int],
    random_state: int,
    linkage_method: str = "average",
) -> Tuple[np.ndarray, np.ndarray, List[str], int, Optional[float], Optional[float], Dict[int, List[int]]]:
    """
    Perform hierarchical clustering on assays and select cluster labels.

    Returns
    -------
    Z : linkage matrix
    D : square distance matrix
    assays : list of assay names
    C : chosen number of clusters
    avg_sil : average silhouette for chosen labels (excluding singletons), or None
    t_line : distance threshold drawn on dendrogram (None if not applicable)
    cluster_members : dict cluster_id -> list of assay indices (0-based)
    """
    D, D_condensed, assays = assay_distance_matrix(Xz)

    # Average linkage; optimal_ordering improves dendrogram readability
    Z = linkage(D_condensed, method=linkage_method, optimal_ordering=True)

    if n_clusters is not None and corr_threshold is not None:
        logging.warning("--n-clusters provided; ignoring --corr-threshold.")

    if n_clusters is not None:
        labels = _labels_for_k(Z, n_clusters)
        C = int(len(np.unique(labels)))
        t_line = _threshold_for_k(Z, n_leaves=len(assays), k=C)
        avg_sil = _silhouette_for_labels(D, labels)
    elif corr_threshold is not None:
        t = 1.0 - float(corr_threshold)
        labels = _labels_for_threshold(Z, t)
        C = int(len(np.unique(labels)))
        t_line = t
        avg_sil = _silhouette_for_labels(D, labels)
    else:
        lo, hi = silhouette_range
        lo = max(2, lo)
        hi = min(hi, len(assays) - 1)
        if lo > hi:
            lo, hi = 2, max(2, min(10, len(assays) - 1))
            logging.warning("Adjusted silhouette range to [%d, %d] due to assay count.", lo, hi)

        best_k, best_score = None, -np.inf
        for k in range(lo, hi + 1):
            cand_labels = _labels_for_k(Z, k)
            score = _silhouette_for_labels(D, cand_labels)
            if score is None:
                continue
            if score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9 and (best_k is None or k < best_k)):
                best_k, best_score = k, score
            # print(f"Silhouette for k={k}: {score:.3f}")
        if best_k is None:
            # Fallback: choose k=lo even if silhouette undefined
            logging.warning("Silhouette undefined for all k; falling back to k=%d.", lo)
            best_k = lo
            best_score = None
        labels = _labels_for_k(Z, best_k)
        C = int(len(np.unique(labels)))
        avg_sil = best_score
        t_line = _threshold_for_k(Z, n_leaves=len(assays), k=C)

    # Relabel clusters to 1..C in a stable order (sorted by min assay index)
    unique_labels = np.unique(labels)
    order = sorted(unique_labels, key=lambda lab: np.where(labels == lab)[0].min())
    relabel = {lab: i + 1 for i, lab in enumerate(order)}
    labels = np.array([relabel[lab] for lab in labels], dtype=int)

    # Build cluster membership index mapping
    cluster_members: Dict[int, List[int]] = {cid: [] for cid in range(1, C + 1)}
    for idx, cid in enumerate(labels):
        cluster_members[cid].append(idx)

    # Logging stats
    sizes = np.array([len(cluster_members[cid]) for cid in range(1, C + 1)])
    pct_single = 100.0 * (sizes == 1).sum() / C
    if pct_single > 20.0:
        logging.warning("High singleton rate: %.1f%% of clusters have a single assay.", pct_single)
    logging.info("Chosen clusters: C=%d; average silhouette (excl. singletons)=%s",
                 C, f"{avg_sil:.3f}" if avg_sil is not None else "NA")
    logging.info("Cluster size summary (min/median/max) = %d/%d/%d",
                 sizes.min(), int(np.median(sizes)), sizes.max())

    return Z, D, assays, C, avg_sil, t_line, cluster_members


# ----------------------------- PC1 per cluster and scores ----------------------------- #

def pc1_per_cluster(
    Xz: pd.DataFrame,
    assays: List[str],
    cluster_members: Dict[int, List[int]],
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, float]]:
    """
    For each cluster, compute PC1 loadings and per-chemical scores with sign alignment.

    Returns
    -------
    loadings_df : DataFrame with columns [cluster_id, assay, loading, abs_loading, var_explained_pc1]
    scores_df : DataFrame [n_chemicals x C] with signed PC1 scores per cluster
    var_explained : dict cluster_id -> variance explained by PC1
    """
    chemicals = Xz.index
    C = len(cluster_members)
    scores = np.zeros((len(chemicals), C), dtype=float)
    loadings_rows: List[Dict] = []
    var_explained: Dict[int, float] = {}

    for cid in range(1, C + 1):
        cols_idx = cluster_members[cid]
        cols = [assays[i] for i in cols_idx]
        Xc = Xz[cols].values  # [n_samples x n_features]

        if Xc.shape[1] == 1:
            # Single-assay cluster: PC1 is the assay itself
            loading = np.array([1.0], dtype=float)
            score = Xc[:, 0].copy()
            ve = 1.0
            # Alignment rule: sum(loadings) >= 0 already satisfied
        else:
            pca = PCA(n_components=1, svd_solver="full", random_state=random_state)
            comp = pca.fit(Xc)
            loading = comp.components_[0].copy()  # [n_features]
            score = comp.transform(Xc)[:, 0].copy()  # [n_samples]
            ve = float(comp.explained_variance_ratio_[0])

            # Sign alignment: sum(loadings) >= 0; if near zero, make mean(score) >= 0
            s = loading.sum()
            if s < 0:
                loading *= -1.0
                score *= -1.0
            elif abs(s) < 1e-10 and score.mean() < 0:
                loading *= -1.0
                score *= -1.0

        var_explained[cid] = ve
        scores[:, cid - 1] = score

        for assay_name, w in zip(cols, loading):
            loadings_rows.append({
                "cluster_id": cid,
                "assay": assay_name,
                "loading": float(w),
                "abs_loading": float(abs(w)),
                "var_explained_pc1": ve
            })

    loadings_df = pd.DataFrame(loadings_rows).sort_values(["cluster_id", "abs_loading"], ascending=[True, False])
    # Scores DataFrame with named columns Cluster_001..Cluster_C
    score_cols = [f"Cluster_{cid:03d}" for cid in range(1, C + 1)]
    scores_df = pd.DataFrame(scores, index=chemicals, columns=score_cols)

    # Variance explained diagnostics
    ve_values = np.array(list(var_explained.values()))
    if (ve_values < 0.3).any():
        bad = np.where(ve_values < 0.3)[0] + 1
        logging.warning("Clusters with low PC1 variance explained (<0.3): %s", ",".join(map(str, bad)))
    logging.info("PC1 variance explained summary (min/median/max) = %.3f/%.3f/%.3f",
                 ve_values.min(), np.median(ve_values), ve_values.max())

    return loadings_df, scores_df, var_explained

def compute_cluster_mean_scores(
    Xz: pd.DataFrame,
    assays: List[str],
    cluster_members: Dict[int, List[int]]
) -> pd.DataFrame:
    """
    Compute per-chemical mean standardized activity within each cluster.

    Returns
    -------
    scores_df : DataFrame [n_chemicals x C] with signed mean scores per cluster
    """
    chemicals = Xz.index
    C = len(cluster_members)
    scores = np.zeros((len(chemicals), C), dtype=float)
    for cid in range(1, C + 1):
        cols_idx = cluster_members[cid]
        cols = [assays[i] for i in cols_idx]
        Xc = Xz[cols].values
        scores[:, cid - 1] = Xc.mean(axis=1)
    score_cols = [f"Cluster_{cid:03d}" for cid in range(1, C + 1)]
    return pd.DataFrame(scores, index=chemicals, columns=score_cols)

# ----------------------------- PCA diagnostics ----------------------------- #

def pca_broken_stick_diagnostic(Xz: pd.DataFrame) -> None:
    """
    Log a broken-stick diagnostic on the assay correlation spectrum.

    For p assays, eigenvalue proportions (PCA on assay correlation matrix) are
    compared to the broken-stick expectation. We log how many components exceed
    the null and show the top few proportions vs the null.
    """
    p = Xz.shape[1]
    if p < 2:
        logging.info("Broken-stick diagnostic skipped (p<2).")
        return

    R = np.corrcoef(Xz.values, rowvar=False)
    # Eigenvalues of a correlation matrix sum to p
    evals = np.linalg.eigvalsh(R)  # ascending
    props = evals[::-1] / float(p)  # descending proportions

    # Broken-stick expected proportions b_k
    # b_k = (1/p) * sum_{i=k}^p (1/i)
    inv = 1.0 / np.arange(1, p + 1, dtype=float)
    csum = np.cumsum(inv[::-1])[::-1]  # sums from k..p
    broken = csum / float(p)

    k_keep = int(np.sum(props > broken))
    top = min(5, p)
    pairs = " ; ".join([f"{i+1}:{props[i]:.3f}>{broken[i]:.3f}" if props[i] > broken[i]
                        else f"{i+1}:{props[i]:.3f}≤{broken[i]:.3f}" for i in range(top)])
    logging.info("Broken-stick diagnostic: components above null = %d (of %d); top comps (prop vs null): %s",
                 k_keep, p, pairs)

# ----------------------------- Toxicity Index ----------------------------- #

def compute_ti_from_scores(scores_df: pd.DataFrame, weights: Dict[int, float]) -> pd.DataFrame:
    """
    Compute TI_equal and TI_weighted from arbitrary per-chemical cluster scores.

    Parameters
    ----------
    scores_df : DataFrame [n_chemicals x C], signed cluster scores per chemical
    weights   : dict cluster_id -> nonnegative weight (e.g., cluster size)
    """
    abs_scores = scores_df.abs().values
    C = abs_scores.shape[1]
    ti_equal = abs_scores.mean(axis=1)

    w = np.array([weights[cid] for cid in range(1, C + 1)], dtype=float)
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        w = np.ones_like(w) / C
    else:
        w /= w_sum
    ti_weighted = abs_scores.dot(w)

    return pd.DataFrame({"TI_equal": ti_equal, "TI_weighted": ti_weighted}, index=scores_df.index)

def compute_ti(scores_df: pd.DataFrame, var_explained: Dict[int, float]) -> pd.DataFrame:
    """
    Compute TI_equal and TI_weighted from per-chemical PC1 scores.

    Returns
    -------
    ti_df : DataFrame with columns [TI_equal, TI_weighted]
    """
    abs_scores = scores_df.abs().values  # [n_samples x C]
    n_clusters = abs_scores.shape[1]
    ti_equal = abs_scores.mean(axis=1)

    # Weights proportional to PC1 variance explained; normalize to sum 1
    C = n_clusters
    w = np.array([var_explained[cid] for cid in range(1, C + 1)], dtype=float)
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        logging.warning("Non-positive or invalid sum of variance explained; falling back to equal weights.")
        w = np.ones_like(w) / len(w)
    else:
        w /= w_sum
    ti_weighted = abs_scores.dot(w)

    ti_df = pd.DataFrame({
        "TI_equal": ti_equal,
        "TI_weighted": ti_weighted
    }, index=scores_df.index)
    return ti_df


# ----------------------------- Example phthalates I/O ----------------------------- #

def load_example_phthalates(path: Optional[Path]) -> Dict[str, str]:
    """
    Load example phthalates and produce a mapping from InChI -> short name.

    Expected CSV columns:
      - Required: 'name'
      - One of:   'inchi' OR 'smiles'
    If only 'smiles' is provided, RDKit is used to convert to InChI.

    Names are shortened by removing a leading 'Dimethyl ' to match other scripts.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        logging.warning("Examples CSV not found at %s; skipping annotation.", path)
        return {}

    df = pd.read_csv(path)
    if "name" not in df.columns:
        raise ValueError("Examples CSV must include a 'name' column.")
    names = df["name"].astype(str).str.replace("Dimethyl ", "", regex=False)

    if "inchi" in df.columns:
        inchis = df["inchi"].astype(str)
    elif "smiles" in df.columns:
        if Chem is None:
            raise ImportError("RDKit is required to convert SMILES to InChI but RDKit is not available.")
        mols = df["smiles"].astype(str).apply(Chem.MolFromSmiles)
        if mols.isna().any():
            n_bad = int(mols.isna().sum())
            raise ValueError(f"{n_bad} SMILES failed RDKit parsing.")
        inchis = mols.apply(Chem.MolToInchi)
    else:
        raise ValueError("Examples CSV must include either 'inchi' or 'smiles'.")

    mapping = {i: n for i, n in zip(inchis, names)}
    return mapping

# ----------------------------- Saving outputs ----------------------------- #

def save_outputs(
    outdir: Path,
    assays: List[str],
    labels: np.ndarray,
    cluster_members: Dict[int, List[int]],
    loadings_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    ti_df: pd.DataFrame,
    mean_scores_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Save cluster assignments, loadings, scores, and TI tables to disk.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # assay_clusters.csv
    cluster_df = pd.DataFrame({
        "assay": assays,
        "cluster_id": labels.astype(int)
    }).sort_values(["cluster_id", "assay"])
    # Mark singletons
    counts = cluster_df["cluster_id"].value_counts()
    cluster_df["is_singleton"] = cluster_df["cluster_id"].map(lambda c: counts.get(c, 0) == 1)
    cluster_df.to_csv(outdir / "assay_clusters.csv", index=False)

    # cluster_pc1_loadings.csv
    loadings_df.to_csv(outdir / "cluster_pc1_loadings.csv", index=False)

    # cluster_summary.csv
    summary_rows = []
    for cid, idxs in cluster_members.items():
        n_assays = len(idxs)
        ve = float(loadings_df.loc[loadings_df["cluster_id"] == cid, "var_explained_pc1"].iloc[0])
        summary_rows.append({"cluster_id": cid, "n_assays": n_assays, "var_explained_pc1": ve,
                             "is_singleton": n_assays == 1})
    pd.DataFrame(summary_rows).sort_values("cluster_id").to_csv(outdir / "cluster_summary.csv", index=False)

    # chemical_cluster_scores.parquet (PC1-based)
    scores_df.to_parquet(outdir / "chemical_cluster_scores.parquet", index=True)

    # Optional: mean-based cluster scores
    if mean_scores_df is not None:
        mean_scores_df.to_parquet(outdir / "chemical_cluster_scores_mean.parquet", index=True)

    # toxicity_index.parquet (based on selected TI mode)
    ti_df.to_parquet(outdir / "toxicity_index.parquet", index=True)


# ----------------------------- Plots ----------------------------- #

def make_plots(
    outdir: Path,
    Z: np.ndarray,
    t_line: Optional[float],
    ti_df: pd.DataFrame,
    var_explained: Dict[int, float],
    example_labels: Optional[Dict[str, str]] = None,
) -> None:
    """
    Create dendrogram with cut annotation, TI histogram, and variance explained bars.
    """
    # Dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(Z, no_labels=True, count_sort=True)
    if t_line is not None and np.isfinite(t_line):
        plt.axhline(y=t_line, linestyle="--", linewidth=1.5)
        plt.text(0.02, 0.98, f"Cut @ d={t_line:.3f}", transform=plt.gca().transAxes,
                 va="top", ha="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    plt.title("Assay dendrogram (average linkage)")
    plt.ylabel("Distance (1 - |r|)")
    plt.tight_layout()
    plt.savefig(outdir / "assay_dendrogram.png", dpi=200)
    plt.close()

    # TI histogram (TI_equal)
    plt.figure(figsize=(8, 5))
    plt.hist(ti_df["TI_equal"].values, bins=50)
    plt.xlabel("TI_equal")
    plt.ylabel("Count")
    plt.title("Distribution of TI_equal")

    # Optional annotations for example compounds
    if example_labels:
        ax = plt.gca()
        ymax = ax.get_ylim()[1]
        # Keep only examples that are in the TI index
        matches = [(chem, example_labels[chem]) for chem in ti_df.index if chem in example_labels]
        for j, (chem, label) in enumerate(matches):
            x = float(ti_df.loc[chem, "TI_equal"])
            ax.axvline(x, linestyle="--", linewidth=1.0)
            # Stagger label heights to reduce overlap
            y = ymax * (0.85 - 0.05 * (j % 6))
            ax.text(
                x, y, label, rotation=90, va="top", ha="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7)
            )
    
    plt.tight_layout()
    plt.savefig(outdir / "ti_histogram.png", dpi=200)
    plt.close()

    # Variance explained bar plot
    C = len(var_explained)
    xs = np.arange(1, C + 1)
    ves = np.array([var_explained[cid] for cid in xs], dtype=float)
    plt.figure(figsize=(10, 5))
    plt.bar(xs, ves)
    plt.xlabel("Cluster ID")
    plt.ylabel("PC1 variance explained")
    plt.title("PC1 variance explained per cluster")
    plt.xticks(xs if C <= 40 else xs[::max(1, C // 40)])
    plt.tight_layout()
    plt.savefig(outdir / "cluster_var_explained.png", dpi=200)
    plt.close()


# ----------------------------- Main ----------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Compute toxicity index via assay clustering and cluster PC1.")
    parser.add_argument("--input", type=str,
                        default="cache/entity_similarity2/activity_matrix_filled.parquet",
                        help="Path to activity matrix Parquet file (rows=chemicals, cols=assays).")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory (default: parent of input).")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="Manual number of clusters. If provided with --corr-threshold, this takes precedence.")
    parser.add_argument("--corr-threshold", type=float, default=None,
                        help="Correlation threshold τ in (0,1). Dendrogram cut at distance d = 1 - τ.")
    parser.add_argument("--silhouette-range", type=int, nargs=2, default=[5, 60],
                        help="Range [lo hi] of candidate cluster counts for silhouette-based selection.")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for deterministic behavior where applicable.")
    parser.add_argument("--linkage", type=str, default="average", choices=["complete", "average"],
                        help="Linkage method for hierarchical clustering (default: 'average').")
    parser.add_argument("--examples-csv", type=str, default=None,
                        help="Optional CSV with example phthalates to annotate on the TI histogram; "
                             "expects columns ['name', 'inchi'] or ['name', 'smiles'].")
    parser.add_argument("--use-pc1", action="store_true",
                    help="Use principal component 1 (PC1)-based cluster scores for TI. "
                            "Default is mean standardized activity within each cluster.")
    args = parser.parse_args()

    # Logging config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    input_path = Path(args.input)
    if args.outdir is None:
        outdir = input_path.parent
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.corr_threshold is not None:
        if not (0.0 < args.corr_threshold < 1.0):
            raise ValueError("--corr-threshold must be in (0,1).")

    # Load and standardize
    logging.info("Loading data from %s", input_path)
    df = load_data(input_path)
    logging.info("Matrix shape: %d chemicals x %d assays", df.shape[0], df.shape[1])

    Xz, dropped = zscore_columns(df)
    if dropped:
        (outdir / "dropped_assays.txt").write_text("\n".join(dropped))
        logging.info("Dropped %d assays due to zero variance or invalid std (written to dropped_assays.txt).", len(dropped))
    logging.info("After standardization: %d chemicals x %d assays", Xz.shape[0], Xz.shape[1])

    # Cluster assays
    Z, D, assays, C, avg_sil, t_line, cluster_members = cluster_assays(
        Xz=Xz,
        n_clusters=args.n_clusters,
        corr_threshold=args.corr_threshold,
        silhouette_range=(args.silhouette_range[0], args.silhouette_range[1]),
        random_state=args.random_state,
        linkage_method=args.linkage,
    )

    # PCA broken-stick diagnostic (assay correlation spectrum)
    pca_broken_stick_diagnostic(Xz)

    # Prepare labels aligned to assays order
    labels = np.zeros(len(assays), dtype=int)
    for cid, idxs in cluster_members.items():
        for i in idxs:
            labels[i] = cid

    # PC1 per cluster and scores
    loadings_df, scores_df, var_explained = pc1_per_cluster(
        Xz=Xz, assays=assays, cluster_members=cluster_members, random_state=args.random_state
    )

    # Per-cluster mean scores (default TI mode)
    mean_scores_df = compute_cluster_mean_scores(
        Xz=Xz, assays=assays, cluster_members=cluster_members
    )

    # TI computation: default uses mean scores; --use-pc1 switches to PC1 scores
    if args.use_pc1:
        ti_df = compute_ti(scores_df, var_explained)
        logging.info("TI mode: PC1-based cluster scores.")
    else:
        size_weights = {cid: len(cluster_members[cid]) for cid in cluster_members}
        ti_df = compute_ti_from_scores(mean_scores_df, size_weights)
        logging.info("TI mode: mean activity within each cluster (weights ∝ cluster size).")

    # Load example phthalates and prepare labels (InChI -> short name)
    example_labels: Dict[str, str] = {}
    if args.examples_csv:
        try:
            example_labels = load_example_phthalates(Path(args.examples_csv))
            if example_labels:
                # Save a small table with their TI values (only those present)
                rows = [
                    (chem, example_labels[chem],
                     float(ti_df.loc[chem, "TI_equal"]),
                     float(ti_df.loc[chem, "TI_weighted"]))
                    for chem in ti_df.index if chem in example_labels
                ]
                if rows:
                    ex_df = pd.DataFrame(rows, columns=["chemical_id", "name", "TI_equal", "TI_weighted"])
                    ex_df.to_csv(outdir / "example_phthalates_ti.csv", index=False)
        except Exception as e:
            logging.warning("Failed to process examples CSV: %s", e)

    # Save outputs
    save_outputs(
        outdir=outdir,
        assays=assays,
        labels=labels,
        cluster_members=cluster_members,
        loadings_df=loadings_df,
        scores_df=scores_df,            # PC1-based cluster scores
        ti_df=ti_df,                    # TI from selected mode
        mean_scores_df=mean_scores_df   # mean-based cluster scores
    )

    # Plots
    make_plots(
        outdir=outdir,
        Z=Z,
        t_line=t_line,
        ti_df=ti_df,
        var_explained=var_explained,
        example_labels=example_labels,
    )

    # Final log summary
    n_singletons = sum(1 for cid in range(1, C + 1) if len(cluster_members[cid]) == 1)
    pct_single = 100.0 * n_singletons / C
    ve_values = np.array(list(var_explained.values()))
    logging.info("Summary:")
    logging.info("  C = %d clusters; avg silhouette (excl. singletons) = %s",
                 C, f"{avg_sil:.3f}" if avg_sil is not None else "NA")
    logging.info("  Singleton clusters: %d (%.1f%%)", n_singletons, pct_single)
    logging.info("  PC1 var explained (min/median/max): %.3f / %.3f / %.3f",
                 ve_values.min(), np.median(ve_values), ve_values.max())
    logging.info("Outputs written to: %s", outdir.resolve())


if __name__ == "__main__":
    main()
