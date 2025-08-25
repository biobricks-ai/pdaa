#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chemical_cluster_label.py

Hierarchical clustering of chemicals (rows) based on an assay activity matrix.

- Input matrix: cache/entity_similarity/activity_matrix_filled.parquet
  (rows = chemicals, columns = assays; values in [0,1] or z-scores)
- Per-assay (column) z-scoring with ddof=0; drop zero-variance assays.
- Similarity: Pearson correlation across assays between chemicals.
- Distance: d = 1 - |r|  (absolute-correlation distance).
- Linkage: average (default) or complete.
- Cluster count selection:
    * --n_clusters K: cut to exactly K (log silhouette).
    * --corr_threshold τ in (0,1): cut at distance 1-τ.
    * Otherwise: automatic K by maximizing average silhouette over a range,
      excluding singleton members from the average; ties break toward smaller K.

Outputs (to the input's parent directory by default):
- chemical_clusters.csv
- cluster_summary.csv
- dropped_assays.txt

Acronyms:
- PCA: principal component analysis (not used here).
- TI: toxicity index (not used here).

Potential shared helpers with assay_cluster_toxicity_index.py (to reduce duplication):
- zscore_columns
- chemical_distance_matrix (absolute-correlation distance)
- stable_relabel_clusters
- compute_silhouette_mean

Author: insilica.co tooling
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples

import sys
sys.path.append('./')  # so utility scripts can be found
from scripts.utils.helpers import zscore_columns


# ----------------------------- I/O and preprocessing -----------------------------


def load_data(path: Path) -> pd.DataFrame:
    """
    Load the activity matrix (rows=chemicals, columns=assays).

    Parameters
    ----------
    path : Path
        Parquet file path.

    Returns
    -------
    pd.DataFrame
        DataFrame with chemical IDs as index. Values as floats.
    """
    df = pd.read_parquet(path)
    if not df.index.is_unique:
        dup = df.index[df.index.duplicated()].unique().tolist()
        raise ValueError(f"Chemical index contains duplicates (n={len(dup)}). Examples: {dup[:5]}")
    if df.shape[0] < 2 or df.shape[1] < 2:
        raise ValueError(f"Matrix too small for clustering: {df.shape}")
    df = df.astype(float)
    logging.info("Loaded matrix: %s (chemicals=%d, assays=%d)", path, df.shape[0], df.shape[1])
    return df


# ----------------------------- distances and clustering -----------------------------


def chemical_distance_matrix(zcols: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise absolute-correlation distance between chemicals.

    Pearson correlation r across assays (columns) is computed between each pair
    of chemicals (rows) AFTER column z-scoring. Distance is d = 1 - |r|.

    Implementation detail:
    - We first form the standard correlation distance (d_corr = 1 - r) using a
      vectorized approach on the full row-vs-row correlation matrix. We then
      convert it to absolute-correlation distance via:
         d_abs = 1 - |r| = 1 - |1 - d_corr|.
    - Rows that are constant across assays (rare after column z-scoring) lead to
      undefined r; we conservatively set distances involving such rows to 1.0.

    Parameters
    ----------
    zcols : pd.DataFrame
        Column-zscored activity matrix (rows=chemicals).

    Returns
    -------
    d_condensed : np.ndarray
        Condensed distance vector (length n*(n-1)/2) with values in [0,1].
    d_square : np.ndarray
        Square distance matrix (n x n) with zeros on the diagonal.
    """
    # Compute row-by-row Pearson correlation matrix.
    # Using np.corrcoef is memory-friendly relative to per-pair loops and provides an n x n result.
    X = zcols.to_numpy(dtype=float, copy=False)
    with np.errstate(invalid="ignore"):
        R = np.corrcoef(X)  # rows=observations
    n = R.shape[0]
    if R.shape != (n, n):
        raise RuntimeError("Unexpected correlation matrix shape.")

    # Handle NaNs from constant rows: set off-diagonal NaNs to 0 (=> distance 1), diagonal to 1.
    nan_mask = np.isnan(R)
    if nan_mask.any():
        logging.warning("Detected rows with zero variance across assays; treating their correlations as 0.")
        np.fill_diagonal(R, 1.0)  # ensure diagonal is 1
        R[nan_mask] = 0.0
        np.fill_diagonal(R, 1.0)

    # Absolute-correlation distance: d = 1 - |r|
    D = 1.0 - np.abs(R)
    # Numerical cleanup
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 1.0)

    d_condensed = squareform(D, checks=False)
    return d_condensed, D


def stable_relabel_clusters(labels: np.ndarray, order_keys: Sequence[int]) -> np.ndarray:
    """
    Relabel arbitrary cluster labels to 1..K deterministically using the minimum
    positional index per cluster as the ordering key.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels as returned by fcluster (1-based arbitrary integers).
    order_keys : Sequence[int]
        Positional indices 0..n-1 corresponding to the row order.

    Returns
    -------
    np.ndarray
        Relabeled clusters in 1..K with stable ordering.
    """
    mapping: Dict[int, int] = {}
    for lab in np.unique(labels):
        idx_positions = np.where(labels == lab)[0]
        anchor = int(np.min([order_keys[i] for i in idx_positions]))
        mapping[lab] = anchor
    # Order by anchor and assign 1..K
    ordered = sorted(mapping.items(), key=lambda kv: kv[1])
    lab2new = {old: i + 1 for i, (old, _) in enumerate(ordered)}
    return np.array([lab2new[l] for l in labels], dtype=int)


def compute_silhouette_mean(D_square: np.ndarray, labels: np.ndarray, exclude_singletons: bool = True) -> Tuple[float, int]:
    """
    Compute mean silhouette using a precomputed distance matrix.

    Parameters
    ----------
    D_square : np.ndarray
        Square distance matrix (n x n).
    labels : np.ndarray
        Cluster labels (1..K).
    exclude_singletons : bool
        If True, exclude samples from singleton clusters when averaging.

    Returns
    -------
    mean_sil : float
        Mean silhouette; np.nan if not defined (e.g., insufficient clusters).
    n_eval : int
        Number of samples included in the average.
    """
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan"), 0

    # scikit-learn requires distances with zeros on diagonal and non-negative.
    s = silhouette_samples(D_square, labels, metric="precomputed")

    if exclude_singletons:
        # Exclude members of clusters of size 1.
        _, counts = np.unique(labels, return_counts=True)
        singletons = set(unique[np.where(counts == 1)[0]])
        mask = np.array([lab not in singletons for lab in labels], dtype=bool)
    else:
        mask = np.ones_like(labels, dtype=bool)

    if mask.sum() == 0:
        return float("nan"), 0

    return float(np.nanmean(s[mask])), int(mask.sum())


def _choose_k_by_silhouette(
    Z: np.ndarray,
    D_square: np.ndarray,
    k_min: int,
    k_max: int,
    order_keys: Sequence[int],
) -> Tuple[np.ndarray, int, float, int]:
    """
    Grid-search K in [k_min,k_max] to maximize mean silhouette (excluding singletons).
    Ties are broken toward smaller K.

    Returns
    -------
    labels_best : np.ndarray
        Best labels (stable-relabelled).
    k_best : int
        Best K.
    sil_best : float
        Best mean silhouette (excluding singletons).
    n_eval : int
        Number of samples included in the best silhouette average.
    """
    best = {"k": None, "sil": -np.inf, "labels": None, "n_eval": 0}

    for k in range(k_min, k_max + 1):
        labs = fcluster(Z, t=k, criterion="maxclust")
        sil, n_eval = compute_silhouette_mean(D_square, labs, exclude_singletons=True)
        # If silhouette is nan (e.g., degenerate), treat as -inf
        score = -np.inf if np.isnan(sil) else sil
        # print(f"k = {k:2d} | silhouette = {score:.4f}")

        if (score > best["sil"]) or (np.isclose(score, best["sil"]) and (best["k"] is None or k < best["k"])):
            best.update({"k": k, "sil": score, "labels": labs, "n_eval": n_eval})

    if best["labels"] is None or best["k"] is None:
        # Fallback: 2 clusters
        logging.warning("Silhouette selection failed; defaulting to K=2.")
        labs = fcluster(Z, t=2, criterion="maxclust")
        sil, n_eval = compute_silhouette_mean(D_square, labs, exclude_singletons=True)
        labels_best = stable_relabel_clusters(labs, order_keys)
        return labels_best, int(np.unique(labels_best).size), float(sil), int(n_eval)

    labels_best = stable_relabel_clusters(best["labels"], order_keys)
    return labels_best, int(best["k"]), float(best["sil"]), int(best["n_eval"])


def cluster_chemicals(
    d_condensed: np.ndarray,
    D_square: np.ndarray,
    index: pd.Index,
    linkage_method: str = "average",
    n_clusters: Optional[int] = None,
    corr_threshold: Optional[float] = None,
    silhouette_range: Tuple[int, int] = (5, 60),
) -> Tuple[np.ndarray, int, float, Tuple[int, int, int], float]:
    """
    Perform hierarchical clustering and select the number of clusters.

    Parameters
    ----------
    d_condensed : np.ndarray
        Condensed distance vector for hierarchical clustering.
    D_square : np.ndarray
        Square distance matrix for silhouette (precomputed).
    index : pd.Index
        Row index (chemical IDs) to define deterministic relabeling.
    linkage_method : str
        'average' or 'complete'.
    n_clusters : int, optional
        If provided, force cut to exactly K.
    corr_threshold : float, optional
        If provided, cut at distance t = 1 - corr_threshold.
    silhouette_range : (int, int)
        Range [min, max] for automatic K selection.

    Returns
    -------
    labels_final : np.ndarray
        Stable-relabelled cluster IDs (1..K) for each chemical in index order.
    K : int
        Final number of clusters.
    sil_mean : float
        Mean silhouette excluding singletons (nan if undefined).
    size_stats : (int, int, int)
        (min, median, max) cluster sizes.
    singleton_frac : float
        Fraction of clusters that are singletons.
    """
    if linkage_method not in ("average", "complete"):
        raise ValueError("linkage_method must be 'average' or 'complete'.")

    Z = linkage(d_condensed, method=linkage_method, optimal_ordering=False)

    n = len(index)
    order_keys = np.arange(n)

    # Determine labels
    chosen_mode = None
    if n_clusters is not None:
        chosen_mode = "n_clusters"
        labs = fcluster(Z, t=int(n_clusters), criterion="maxclust")
        if np.unique(labs).size != int(n_clusters):
            logging.warning("Requested K=%d but obtained %d clusters after cut.", int(n_clusters), np.unique(labs).size)
        sil, _ = compute_silhouette_mean(D_square, labs, exclude_singletons=True)
        labels_final = stable_relabel_clusters(labs, order_keys)
        K = int(np.unique(labels_final).size)

    elif corr_threshold is not None:
        chosen_mode = "corr_threshold"
        if not (0.0 < float(corr_threshold) < 1.0):
            raise ValueError("--corr_threshold must be in (0,1).")
        t = 1.0 - float(corr_threshold)
        labs = fcluster(Z, t=t, criterion="distance")
        labels_final = stable_relabel_clusters(labs, order_keys)
        K = int(np.unique(labels_final).size)
        sil, _ = compute_silhouette_mean(D_square, labels_final, exclude_singletons=True)

    else:
        chosen_mode = "silhouette"
        k_lo, k_hi = silhouette_range
        k_lo = max(2, int(k_lo))
        k_hi = min(max(2, int(k_hi)), n - 1)
        if k_lo > k_hi:
            k_lo, k_hi = 2, max(2, min(10, n - 1))  # safe fallback window
            logging.warning("Adjusted silhouette range to [%d, %d].", k_lo, k_hi)
        labels_final, K, sil, _ = _choose_k_by_silhouette(Z, D_square, k_lo, k_hi, order_keys)

    # Cluster size stats
    _, counts = np.unique(labels_final, return_counts=True)
    size_min = int(counts.min())
    size_med = int(np.median(counts))
    size_max = int(counts.max())
    singleton_frac = float((counts == 1).sum() / counts.size)

    if singleton_frac > 0.20:
        logging.warning(
            "High singleton rate: %.1f%% of clusters are singletons (%d/%d).",
            100.0 * singleton_frac,
            int((counts == 1).sum()),
            counts.size,
        )

    logging.info(
        "Cluster selection mode: %s | K=%d | silhouette(excl singletons)=%.4f | size(min/med/max)=(%d/%d/%d)",
        chosen_mode,
        K,
        sil if not np.isnan(sil) else float("nan"),
        size_min,
        size_med,
        size_max,
    )

    return labels_final, K, float(sil), (size_min, size_med, size_max), singleton_frac


# ----------------------------- examples CSV integration -----------------------------


def _shorten_name(name: str) -> str:
    """Strip an exact leading 'Dimethyl ' if present."""
    prefix = "Dimethyl "
    return name[len(prefix) :] if isinstance(name, str) and name.startswith(prefix) else (name or "")


def load_example_phthalates(examples_csv: Path) -> pd.DataFrame:
    """
    Load examples CSV with columns:
      - required: name
      - one of: inchi OR smiles
    If only 'smiles' is provided, convert to InChI using RDKit (optional dependency).

    Returns
    -------
    pd.DataFrame
        Columns: inchi, short_name
    """
    df = pd.read_csv(examples_csv)
    required = {"name"}
    if not required.issubset(df.columns):
        raise ValueError(f"Examples CSV missing required columns: {required - set(df.columns)}")

    has_inchi = "inchi" in df.columns
    has_smiles = "smiles" in df.columns

    if not (has_inchi or has_smiles):
        raise ValueError("Examples CSV must contain either 'inchi' or 'smiles'.")

    if has_inchi:
        out = pd.DataFrame(
            {
                "inchi": df["inchi"].astype(str),
                "short_name": df["name"].apply(_shorten_name).astype(str),
            }
        )
        out = out.dropna(subset=["inchi"]).drop_duplicates(subset=["inchi"])
        return out

    # Convert SMILES -> InChI via RDKit if only smiles provided.
    try:
        from rdkit import Chem
        from rdkit.Chem import inchi as rdInchi
    except Exception as e:
        raise ImportError(
            "RDKit is required to convert SMILES to InChI for --examples_csv. "
            "Install RDKit or provide 'inchi' directly."
        ) from e

    inchis: List[Optional[str]] = []
    for s in df["smiles"].astype(str).tolist():
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            inchis.append(None)
            continue
        try:
            inchi_str = rdInchi.MolToInchi(mol)
        except Exception:
            inchi_str = None
        inchis.append(inchi_str)

    out = pd.DataFrame(
        {
            "inchi": pd.Series(inchis, dtype="object"),
            "short_name": df["name"].apply(_shorten_name).astype(str),
        }
    )
    n_bad = out["inchi"].isna().sum()
    if n_bad:
        logging.warning("Failed to convert %d SMILES to InChI; those examples will be skipped.", n_bad)

    out = out.dropna(subset=["inchi"]).drop_duplicates(subset=["inchi"])
    return out


# ----------------------------- outputs -----------------------------


def save_outputs(
    outdir: Path,
    index: pd.Index,
    labels: np.ndarray,
    dropped_assays: Iterable[str],
    examples_df: Optional[pd.DataFrame] = None,
) -> Tuple[Path, Path, Path]:
    """
    Save clustering outputs and dropped assay list.

    Parameters
    ----------
    outdir : Path
        Output directory.
    index : pd.Index
        Chemical IDs (DataFrame index order is preserved).
    labels : np.ndarray
        Cluster IDs 1..K for each chemical.
    dropped_assays : Iterable[str]
        Names of assays dropped during column z-scoring.
    examples_df : pd.DataFrame, optional
        Columns: inchi, short_name

    Returns
    -------
    paths : (Path, Path, Path)
        Paths to chemical_clusters.csv, cluster_summary.csv, dropped_assays.txt
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Map examples by InChI to short_name
    example_map: Dict[str, str] = {}
    if examples_df is not None and not examples_df.empty:
        example_map = dict(zip(examples_df["inchi"].astype(str), examples_df["short_name"].astype(str)))

    # Build per-chemical table
    df_clusters = pd.DataFrame(
        {
            "chemical_id": index.astype(str).to_list(),
            "cluster_id": labels.astype(int).tolist(),
        }
    )
    # Singleton flag per cluster
    _, counts = np.unique(labels, return_counts=True)
    singletons = set(np.unique(labels)[np.where(counts == 1)[0]])
    df_clusters["is_singleton"] = df_clusters["cluster_id"].isin(singletons)

    # Example flags and names
    df_clusters["is_example"] = df_clusters["chemical_id"].astype(str).isin(example_map.keys())
    df_clusters["name"] = df_clusters["chemical_id"].astype(str).map(example_map).fillna("")

    # Cluster summary
    cluster_summary = (
        df_clusters.groupby("cluster_id", as_index=False)["chemical_id"]
        .count()
        .rename(columns={"chemical_id": "n_chemicals"})
    )
    cluster_summary["is_singleton"] = cluster_summary["n_chemicals"] == 1

    # Write files
    p_clusters = outdir / "chemical_clusters.csv"
    p_summary = outdir / "cluster_summary.csv"
    p_dropped = outdir / "dropped_assays.txt"

    df_clusters.to_csv(p_clusters, index=False)
    cluster_summary.to_csv(p_summary, index=False)

    with p_dropped.open("w", encoding="utf-8") as f:
        for a in dropped_assays:
            f.write(f"{a}\n")

    logging.info("Wrote: %s (%d rows)", p_clusters, len(df_clusters))
    logging.info("Wrote: %s (%d clusters)", p_summary, cluster_summary.shape[0])
    logging.info("Wrote: %s (dropped assays: %d)", p_dropped, len(list(dropped_assays)))
    return p_clusters, p_summary, p_dropped


# ----------------------------- CLI and main -----------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hierarchical clustering over chemicals using absolute-correlation distance."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("cache/entity_similarity/activity_matrix_filled.parquet"),
        help="Input parquet path [default: cache/entity_similarity/activity_matrix_filled.parquet]",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory [default: parent of --input]",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Cut dendrogram to exactly K clusters.",
    )
    parser.add_argument(
        "--corr_threshold",
        type=float,
        default=None,
        help="Correlation threshold τ in (0,1); cut at distance d=1-τ.",
    )
    parser.add_argument(
        "--silhouette_range",
        type=int,
        nargs=2,
        default=(2, 60),
        metavar=("K_MIN", "K_MAX"),
        help="Range [K_MIN K_MAX] for automatic silhouette selection [default: 2 60].",
    )
    parser.add_argument(
        "--linkage",
        type=str,
        choices=("average", "complete"),
        default="average",
        help="Linkage method [default: average].",
    )
    parser.add_argument(
        "--examples_csv",
        type=Path,
        default=None,
        help="Optional CSV with example chemicals (columns: name and inchi OR smiles).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for determinism where applicable (hierarchical clustering is deterministic).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Prefer --n_clusters over --corr_threshold if both provided.
    if args.n_clusters is not None and args.corr_threshold is not None:
        logging.warning("--n_clusters provided together with --corr_threshold; proceeding with --n_clusters only.")

    input_path: Path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    outdir: Path = args.outdir if args.outdir is not None else input_path.resolve().parent
    logging.info("Output directory: %s", outdir)

    # Load and standardize
    df = load_data(input_path)
    zcols, dropped = zscore_columns(df)

    # Distances
    d_condensed, D_square = chemical_distance_matrix(zcols)

    # Clustering and K selection
    labels, K, sil_mean, size_stats, singleton_frac = cluster_chemicals(
        d_condensed=d_condensed,
        D_square=D_square,
        index=zcols.index,
        linkage_method=args.linkage,
        n_clusters=args.n_clusters,
        corr_threshold=(None if args.n_clusters is not None else args.corr_threshold),
        silhouette_range=tuple(args.silhouette_range),
    )

    # Examples integration
    examples_df = None
    if args.examples_csv is not None:
        examples_df = load_example_phthalates(args.examples_csv)
        # Warn for examples not present in matrix
        if not examples_df.empty:
            present = set(zcols.index.astype(str))
            missing = [i for i in examples_df["inchi"].astype(str) if i not in present]
            if missing:
                logging.warning(
                    "Examples not found in matrix (n=%d). Showing first 5: %s",
                    len(missing),
                    missing[:5],
                )

    # Save outputs
    save_outputs(
        outdir=outdir,
        index=zcols.index,
        labels=labels,
        dropped_assays=dropped,
        examples_df=examples_df,
    )

    # Final log summary
    logging.info(
        "Done. K=%d clusters | silhouette(excl singletons)=%.4f | size(min/med/max)=%s | singleton clusters=%.1f%%",
        K,
        sil_mean if not np.isnan(sil_mean) else float("nan"),
        size_stats,
        100.0 * singleton_frac,
    )


if __name__ == "__main__":
    main()
