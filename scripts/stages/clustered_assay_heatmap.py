#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate assays into K clusters and plot a heatmap (chemicals x clusters).

Inputs
------
- --matrix: Parquet file of activity matrix with rows=chemicals, cols=assays.
- Either:
    * --chosen_k_json: path to chosen_k.json (expects {"chosen_k": <int>, ...})
      produced by choose_k_assay_clusters.py, or
    * --k: explicit integer number of clusters.
- Optional format and behavior flags (see argparse help).

Outputs
-------
- Heatmap PNG (default: <outdir>/heatmap_clusters.png)
- Aggregated matrix parquet (default: <outdir>/activity_by_cluster.parquet)
- CSV mapping assays -> cluster (default: <outdir>/assay_to_cluster.csv)

This script is intentionally modular so common pieces can be moved
to a shared helper module later.

Conventions borrowed/adapted from:
- Pearson |r| distance + hierarchical linkage flow used during K selection.  # :contentReference[oaicite:2]{index=2}
- Heatmap visual styling generalized from existing seaborn figure helper.     # :contentReference[oaicite:3]{index=3}
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, NullFormatter
import seaborn as sns
from rdkit import Chem


# ------------------------ General utilities ------------------------

def zscore_by_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Z-score each column; drop zero-variance columns.
    Returns (zscored_df, dropped_columns).
    """
    X = df.copy()
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    bad = sd <= 0
    dropped = list(X.columns[bad])
    X = X.loc[:, ~bad]
    mu = mu[~bad]
    sd = sd[~bad]
    Xz = (X - mu) / sd
    Xz = Xz.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return Xz, dropped


def assay_distance_matrix_pearson_abs(Xz: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Pearson |r| distance between assay columns.

    Returns:
        D_full: square distance matrix (assay x assay)
        D_condensed: condensed vector suitable for scipy.linkage
        assays: list of assay column names in the given order
    """
    assays = list(Xz.columns)
    R = np.corrcoef(Xz.values, rowvar=False)
    R = np.clip(R, -1.0, 1.0)
    D_full = 1.0 - np.abs(R)
    np.fill_diagonal(D_full, 0.0)
    D_full = (D_full + D_full.T) / 2.0
    D_condensed = squareform(D_full, checks=False)
    return D_full, D_condensed, assays


def cluster_assays(
    Xz: pd.DataFrame,
    k: int,
    *,
    linkage_method: str = "average",
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Cluster assays with hierarchical linkage on Pearson-|r| distance.

    Returns:
        labels: array of shape [n_assays] with labels in {1..k}
        col_order: leaf order (list of column indices) from dendrogram
        assays: list of assay names aligned to labels
    """
    _, D_condensed, assays = assay_distance_matrix_pearson_abs(Xz)
    if D_condensed.size == 0:
        raise ValueError("Distance matrix is empty; not enough assays to cluster.")

    Z = linkage(D_condensed, method=linkage_method, optimal_ordering=True)

    # Stable 1..k labels
    labels = fcluster(Z, t=k, criterion="maxclust").astype(int)

    # Column order to keep visual consistency (optional for aggregated view)
    col_order = list(leaves_list(Z))
    return labels, col_order, assays


def aggregate_activity_by_cluster(
    activity: pd.DataFrame,
    assays: List[str],
    labels: np.ndarray,
    *,
    agg: str = "mean",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate per-chemical activity over assays within each cluster.

    Args:
        activity: rows=chemicals, cols=assays
        assays: list of assay column names aligned with 'labels'
        labels: cluster labels for each assay (len == len(assays))
        agg: aggregation function name supported by pandas ("mean", "median", etc.)

    Returns:
        M: aggregated matrix (rows=chemicals, cols=cluster_1..cluster_K)
        mapping_df: DataFrame with columns [assay, cluster]
    """
    if activity.shape[1] != len(assays):
        # realign activity columns to 'assays'
        activity = activity.loc[:, assays]

    df_cols = pd.DataFrame({"assay": assays, "cluster": labels})
    k = int(df_cols["cluster"].max())

    # group columns by cluster id and aggregate across assays within each cluster
    grouped: Dict[int, pd.DataFrame] = {}
    for cid in range(1, k + 1):
        cols_c = df_cols.loc[df_cols["cluster"] == cid, "assay"].tolist()
        if not cols_c:
            # still emit a column to keep shape consistent (all-NaN -> fill later)
            grouped[cid] = activity.iloc[:, :0].copy()
        else:
            if agg == "mean":
                grouped[cid] = activity[cols_c].mean(axis=1, skipna=True).to_frame(f"cluster_{cid}")
            else:
                grouped[cid] = getattr(activity[cols_c], agg)(axis=1, skipna=True).to_frame(f"cluster_{cid}")

    M = pd.concat([grouped[cid] for cid in range(1, k + 1)], axis=1)
    return M, df_cols


def order_rows_by_mean(M: pd.DataFrame, *, descending: bool = True) -> List[int]:
    """
    Return row order indices by row-wise mean activity.
    """
    m = M.mean(axis=1)
    order = np.argsort(m.values)
    if descending:
        order = order[::-1]
    return order.tolist()


def styled_heatmap(
    matrix: pd.DataFrame,
    row_colors: Optional[Iterable] = None,
    *,
    dpi: int = 600,
    fontcolor: str = "black",
    linecolor: str = "black",
    z_scale: bool = False,
    xlabel: str = "Assay Clusters",
    figsize: Tuple[float, float] = (18.0, 9.0),
    cmap_diverging: str = "vlag",
):
    """
    Generalized heatmap styling similar to prior figures.
    """
    if z_scale:
        vscale = 3.0
        vmin, vmax = -vscale, +vscale
        cmap = cmap_diverging
    else:
        vmin, vmax = 0.0, 1.0
        cmap = "viridis"

    g = sns.clustermap(
        matrix,
        cbar_kws={"drawedges": False},
        cmap=cmap,
        row_cluster=False,
        col_cluster=False,
        row_colors=row_colors,
        # xticklabels=True,
        xticklabels=False,
        yticklabels=False,
        linecolor=linecolor,
        figsize=figsize,
        cbar_pos=(0.95, 0.3, 0.02, 0.4),
        dendrogram_ratio=(0.10, 0.05),
        tree_kws={"linewidths": 0.5},
        vmin=vmin,
        vmax=vmax,
    )

    # Remove seaborn’s default colorbar and re-add a custom one
    if hasattr(g, "cax") and g.cax:
        g.cax.remove()

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(g.ax_heatmap)
    cax = divider.append_axes("right", size="2%", pad=0.6)
    g.divider = divider  # expose for callers if needed

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax, clip=True))
    sm.set_array([])
    cb = g.figure.colorbar(sm, cax=cax)
    cb.set_label("Activity Score", fontsize=18, labelpad=10)
    cb.ax.tick_params(labelsize=14)

    # Optionally show a placeholder dendrogram axis for consistent margins
    g.ax_col_dendrogram.set_visible(False)

    g.ax_heatmap.set_xlabel(xlabel, color=fontcolor, fontsize=20, labelpad=15)

    # Also label the colorbar on the main heatmap collections
    cbar = g.ax_heatmap.collections[0].colorbar
    cbar.set_label("Activity Score", fontsize=20, labelpad=10)

    g.figure.subplots_adjust(left=0.05, right=0.90, top=0.95, bottom=0.05)
    return g


# ------------------------ Main CLI flow ------------------------

def main():
    ap = argparse.ArgumentParser(description="Aggregate assays into K clusters and render a heatmap (chemicals x clusters).")
    io = ap.add_argument_group("I/O")
    io.add_argument("--matrix", required=True, help="Parquet: rows=chemicals, cols=assays.")
    io.add_argument("--outdir", default="cache/clustered_heatmap", help="Where to write outputs.")
    io.add_argument("--agg", default="mean", choices=["mean", "median"], help="Aggregation within clusters.")
    io.add_argument("--example_csv", default="resources/example_phthalates.csv",
                    help="CSV with columns [name, smiles] for labeling example phthalates on the side bar.")

    kcfg = ap.add_argument_group("Cluster config")
    kcfg.add_argument("--chosen_k_json", default=None, help="Path to chosen_k.json containing {'chosen_k': K}.")
    kcfg.add_argument("--k", type=int, default=None, help="Number of assay clusters (overrides --chosen_k_json if both set).")
    kcfg.add_argument("--linkage", default="average", choices=["average", "complete"], help="Hierarchical linkage method.")

    viz = ap.add_argument_group("Visualization")
    viz.add_argument("--z_scale", action="store_true", help="Z-scale values for display.")
    viz.add_argument("--xlabel", default="Assay Clusters", help="X-axis label.")
    viz.add_argument("--figwidth", type=float, default=18.0, help="Figure width in inches.")
    viz.add_argument("--figheight", type=float, default=9.0, help="Figure height in inches.")
    viz.add_argument("--png_name", default="heatmap_clusters.png", help="Output PNG filename.")
    viz.add_argument("--no_row_sort", action="store_true", help="Do not reorder rows by mean activity.")

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load matrix
    activity = pd.read_parquet(args.matrix)
    if activity.index.duplicated().any():
        activity = activity[~activity.index.duplicated(keep="first")]
    logging.info("Loaded activity matrix: %s rows (chemicals) x %s cols (assays).", activity.shape[0], activity.shape[1])

    # 2) Z-score columns for distance computation; keep original for aggregation
    Xz, dropped = zscore_by_column(activity)
    if dropped:
        logging.warning("Dropped %d zero-variance assays for clustering: %s", len(dropped), ", ".join(dropped))

    # 3) Pick K
    K = args.k
    if K is None and args.chosen_k_json:
        with open(args.chosen_k_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        K = int(cfg.get("chosen_k", 0))
    if not K or K < 1:
        raise SystemExit("Provide --k or a --chosen_k_json with a valid 'chosen_k' integer.")

    # 4) Cluster assays at K and build mapping
    labels, col_order, assays = cluster_assays(Xz, K, linkage_method=args.linkage)
    mapping_df = pd.DataFrame({"assay": assays, "cluster": labels}).sort_values(["cluster", "assay"])
    mapping_df.to_csv(outdir / "assay_to_cluster.csv", index=False)
    logging.info("Clustered %d assays into K=%d clusters.", len(assays), K)

    # 5) Aggregate activity by cluster
    # M, mapping_df = aggregate_activity_by_cluster(activity, assays, labels, agg=args.agg)
    M, mapping_df = aggregate_activity_by_cluster(Xz, assays, labels, agg=args.agg)
    M.to_parquet(outdir / "activity_by_cluster.parquet")
    logging.info("Aggregated matrix shape: %s x %s (chemicals x clusters).", M.shape[0], M.shape[1])

    # 6) Row ordering
    if not args.no_row_sort:
        row_order = order_rows_by_mean(M, descending=True)
        M = M.iloc[row_order, :]

    # 7) Optional z-scale for visualization only
    M_plot = M.copy()
    if args.z_scale:
        # z-scale per column for display
        M_plot = (M_plot - M_plot.mean(axis=0)) / M_plot.std(axis=0)

    # Use integer cluster labels on the x-axis (no "cluster_*" prefix)
    M_plot_display = M_plot.copy()
    M_plot_display.columns = list(range(1, M_plot_display.shape[1] + 1))

    # 8) Draw heatmap
    sns.set_theme(context="notebook", style="white")
    g = styled_heatmap(
        M_plot_display,
        row_colors=None,
        z_scale=args.z_scale,
        xlabel=args.xlabel,  # "Assay Clusters"
        figsize=(args.figwidth, args.figheight),
    )

    # X tick labels: integers, no rotation
    g.ax_heatmap.tick_params(axis="x", labelrotation=0)

    # Y label and margins
    g.ax_heatmap.yaxis.set_label_position("left")
    g.ax_heatmap.set_ylabel("Diester Phthalates", color="black", fontsize=20, labelpad=15)
    g.ax_heatmap.yaxis.tick_left()
    g.figure.subplots_adjust(left=0.03, right=0.90, top=1.00, bottom=0.06)

    # ---- Side bar with MAV and labeled example phthalates (copied from entity_similarity.py) ----
    try:
        # load examples
        ex_df = pd.read_csv(args.example_csv)
        ex_df["name"] = ex_df["name"].str.replace("Dimethyl ", "", regex=False)
        ex_mols = [Chem.MolFromSmiles(s) for s in ex_df["smiles"]]
        ex_inchi = [Chem.MolToInchi(m) for m in ex_mols]
        ex_inchi2name = dict(zip(ex_inchi, ex_df["name"]))

        # MAV per chemical = mean across the already-aggregated cluster means
        # Robust to NaN/inf: skip NaNs in row-mean, coerce non-finite to 0 just for bars.
        mean_activity = (
            M.replace([np.inf, -np.inf], np.nan)
            .mean(axis=1, skipna=True)
            .reindex(M_plot_display.index)
        )

        # right-side bar axis (reuse divider created in styled_heatmap)
        ax_bar = g.divider.append_axes("right", size="15%", pad=1.0)

        # Build clean arrays for plotting (drop NaN/inf so matplotlib actually draws bars)
        x = pd.to_numeric(mean_activity, errors="coerce").to_numpy(dtype=float)
        y = np.arange(len(x), dtype=float)
        valid = np.isfinite(x)

        # draw one stripe per row; height=1 fills a row and avoids sub-pixel rasterization
        bars = ax_bar.barh(y, x, color="#ACACAD", height=1.0, linewidth=0, snap=True, antialiased=False, zorder=3)
        # align to the heatmap rows after plotting
        ax_bar.set_ylim(-0.5, len(x) - 0.5)
        # plt.show()

        # # single neutral color (mirrors 'color_by=None' path)
        # bar_color = "#ACACAD"
        # ax_bar.barh(y[valid], x[valid], color=bar_color)

        # Include negatives (common with z-scored/centered data); autoscale when possible
        if np.any(valid):
            xmin = float(np.nanmin(x[valid]))
            xmax = float(np.nanmax(x[valid]))
            # Symmetric bounds if any negative values or when z-scaling is on
            if xmin < 0.0 or args.z_scale:
                bound = 1.05 * max(abs(xmin), abs(xmax), 1e-9)
                ax_bar.set_xlim(-bound, bound)
                print(f"Setting symmetric xlim to ±{bound:.3f}")
            else:
                ax_bar.set_xlim(0.0, 1.05 * xmax if xmax > 0.0 else 1.0)
        else:
            # Nothing finite; give a tiny symmetric window
            ax_bar.set_xlim(-1.0, 1.0)



        # map chemical id (index) -> displayed row
        reordered_indices = {idx: pos for pos, idx in enumerate(M_plot_display.index)}

        # examples present in the matrix, sorted by row
        examples = sorted(
            [(reordered_indices[i], i, ex_inchi2name[i]) for i in ex_inchi if i in reordered_indices],
            key=lambda t: t[0]
        )
        base_rows = np.array([p for p, _, _ in examples], dtype=float)

        # iterative collision resolution (identical to entity_similarity.py)
        min_sep = 33.0
        shifts = np.zeros_like(base_rows)
        max_iter = 600
        for _ in range(max_iter):
            moved = False
            for j in range(1, len(base_rows)):
                y_prev = base_rows[j-1] + shifts[j-1]
                y_curr = base_rows[j]   + shifts[j]
                gap = y_curr - y_prev
                if gap < min_sep:
                    delta = 0.5 * (min_sep - gap)
                    shifts[j-1] -= delta
                    shifts[j]   += delta
                    moved = True
            if not moved:
                break

        # ensure bar axis has final limits before placing labels
        ax_bar.set_ylim(g.ax_heatmap.get_ylim())

        # remove ALL y ticks/labels on the bar axis so nothing overlaps the colorbar label
        ax_bar.set_yticks([])  # belt
        ax_bar.tick_params(axis="y", which="both", left=False, right=False,
                        labelleft=False, labelright=False)  # suspenders
        ax_bar.yaxis.set_major_locator(NullLocator())
        ax_bar.yaxis.set_minor_locator(NullLocator())
        ax_bar.yaxis.set_major_formatter(NullFormatter())

        # use sorted limits to handle inverted y-axes; clamp into [ylow+pad, yhigh-pad]
        y0, y1 = ax_bar.get_ylim()
        ylow, yhigh = (min(y0, y1), max(y0, y1))
        pad_rows = 8.0

        # also clamp x so labels stay inside the right bound
        xmin, xmax = ax_bar.get_xlim()
        xpad = 0.02 * (xmax - xmin)  # 2% gutter

        bar_tip_fraction = 0.25 if args.z_scale else 0.10
        x_offset = float(mean_activity.max()) * bar_tip_fraction

        for (pos, inchi, name), y_shift in zip(examples, shifts):
            xw = float(mean_activity.iloc[pos])
            x_adj = max(xw, 0)
            if not np.isfinite(xw):
                continue

            # y: clamp within visible band
            y_target = pos + y_shift
            if y_target < ylow + pad_rows:
                y_target = ylow + pad_rows
            elif y_target > yhigh - pad_rows:
                y_target = yhigh - pad_rows

            # x: keep label inside the axis while still to the right of the bar tip
            # x_text = min(max(xw + x_offset, xmin + xpad), xmax - xpad)
            x_text = min(max(x_adj + x_offset, xmin + xpad), xmax - xpad)

            ax_bar.annotate(
                name,
                xy=(xw, pos),
                xytext=(x_text, y_target),
                ha="left", va="center",
                fontsize=11, color="black",
                arrowprops=dict(
                    arrowstyle="-",
                    lw=0.6,
                    color="black",
                ),
                clip_on=True,              # safe since we now keep text inside
                annotation_clip=True,
            )


    except Exception as e:
        logging.warning("Skipping example labels side bar: %s", e)

    # 9) Save figure
    out_png = outdir / args.png_name
    g.figure.savefig(out_png, dpi=600)
    plt.close(g.figure)
    logging.info("Saved heatmap to %s", out_png)




if __name__ == "__main__":
    main()
