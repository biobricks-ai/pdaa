#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Choose the number of assay clusters (K) using ED category guidance.
- Sweeps K, bootstraps for stability, and reports multiple criteria:
  * Reconstruction error via KL: RE(K) = mean_c KL(p_c || hat p_c), where hat p_c = sum_k P(k|c) q_k
  * Label sharpness: S_c = max_k P(k|c)
  * Category fragmentation: NE_k = 1 / sum_c m_{k,c}^2 with m_{k,c} ∝ n_c P(k|c)
  * Stability across bootstraps: cosine of matched P(k|c), ARI/NMI of labels

Inputs
------
- Activity matrix (rows=chemicals, cols=assays) [Parquet]
- ED artifacts produced by cluster_category_scoring.py build_posteriors:
  * word_category_posteriors.csv (rows=words, cols=categories) = P(k|w)
  * category_prototypes_q.csv    (rows=categories, cols=words) = P(w|k)

By default this script imports functions from the clustering script to ensure
consistent tokenization and word clouds for p_c(w):
  - compute_cluster_word_matrix (WordCloud-based)
If import fails, it falls back to an internal implementation.

CLI examples
------------
# Simple sweep with existing artifacts, 5 bootstraps, report to outdir/
python choose_k_assay_clusters.py \
  --matrix cache/entity_similarity/activity_matrix_filled.parquet \
  --ed_artifacts cache/ed_artifacts \
  --k_range 8 24 4 --repeats 5 --bootstrap_frac 0.8 \
  --outdir cache/choose_k_report

# Build artifacts first (pass-through to cluster_category_scoring.py build_posteriors)
python choose_k_assay_clusters.py \
  --matrix cache/entity_similarity/activity_matrix_filled.parquet \
  --build_posteriors \
  --bp_script scripts/stages/cluster_category_scoring.py \
  --bp_corpus resources/assay_names.csv --bp_text_col text \
  --bp_vocab_size 5000 --bp_bg_mode corpus --bp_priors uniform \
  --bp_outdir cache/ed_artifacts --bp_stopwords resources/stopwords.pkl \
  --bp_collocations --bp_collocation_threshold 30 \
  --k_range 8 24 4 --repeats 5 --bootstrap_frac 0.8 \
  --outdir cache/choose_k_report

Outputs
-------
- choose_k_metrics.csv        : per-K means and standard errors for all metrics
- choose_k_perrun.csv         : per-run metrics (diagnostic)
- chosen_k.json               : chosen K and rationale
- per-K directory with Pkc matrices and summaries (optional via --emit_perk)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

from tqdm import tqdm

# Optional imports from your repo
sys.path.append("./")
from scripts.utils.helpers import zscore_columns

# Try to import the clustering helpers to reuse WordCloud behavior for p_c(w)
def _import_cluster_module(path_hint: Optional[str] = None):
    candidates = []
    if path_hint:
        candidates.append(Path(path_hint))
    candidates.append(Path("scripts/stages/assay_cluster_toxicity_index.py"))
    candidates.append(Path(__file__).parent / "assay_cluster_toxicity_index.py")
    for p in candidates:
        if p and p.exists():
            sys.path.append(str(p.parent))
            mod = __import__(p.stem)
            return mod
    return None

# --- Fallbacks if repo helpers are unavailable ---

def _zscore_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Fallback z-score by column with guard for zero-variance columns."""
    X = df.copy()
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    bad = sd <= 0
    dropped = list(X.columns[bad])
    X = X.loc[:, ~bad]
    mu = mu[~bad]; sd = sd[~bad]
    Xz = (X - mu) / sd
    Xz = Xz.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return Xz, dropped

def _assay_distance_matrix(Xz: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute Pearson |r| distance between assays."""
    assays = list(Xz.columns)

    R = np.corrcoef(Xz.values, rowvar=False)
    R = np.clip(R, -1.0, 1.0)
    D = 1.0 - np.abs(R)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    D_condensed = squareform(D, checks=False)
    return D, D_condensed, assays

def _labels_for_k(Z: np.ndarray, k: int) -> np.ndarray:
    return fcluster(Z, t=k, criterion="maxclust")

def _relabel_stable(labels: np.ndarray) -> np.ndarray:
    """Relabel to 1..C sorted by min assay index (stable)."""
    unique = np.unique(labels)
    order = sorted(unique, key=lambda lab: np.where(labels == lab)[0].min())
    relabel = {lab: i + 1 for i, lab in enumerate(order)}
    return np.array([relabel[lab] for lab in labels], dtype=int)

def _cluster_members(labels: np.ndarray) -> Dict[int, List[int]]:
    C = int(labels.max())
    mem: Dict[int, List[int]] = {cid: [] for cid in range(1, C + 1)}
    for i, lab in enumerate(labels):
        mem[int(lab)].append(i)
    return mem

# --- Metrics utilities ---

def _threshold_for_k(Z: np.ndarray, n_leaves: int, k: int) -> float:
    """
    Return the dendrogram cut distance t_K that yields exactly k clusters under
    fcluster(..., criterion='maxclust'). Uses the height of the (n_leaves - k)-th merge.
    """
    if k >= n_leaves:
        return 0.0
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    heights = Z[:, 2].astype(float)
    idx = n_leaves - k - 1
    if idx < 0 or idx >= len(heights):
        return 0.0
    return float(heights[idx])

def _cosine_sim(A: np.ndarray, B: np.ndarray) -> float:
    """Mean cosine similarity between matched rows of A and B (A,B shape: [C,K])."""
    # Rows are cluster-level P(k|c)
    # Normalize row-wise to unit norm to compute cosine
    A2 = normalize(A, norm="l2", axis=1, copy=True)
    B2 = normalize(B, norm="l2", axis=1, copy=True)
    # Cost = 1 - cosine; we want to maximize cosine via Hungarian on cost
    Cmat = 1.0 - (A2 @ B2.T)
    row_ind, col_ind = linear_sum_assignment(Cmat)
    sims = 1.0 - Cmat[row_ind, col_ind]
    return float(np.mean(sims))

def _kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))

# --- Core pipeline ---

@dataclass
class RunMetrics:
    k: int
    repeat: int
    re_mean: float
    sharp_mean: float
    sharp_frac: float
    frag_max: float
    stab_cos: Optional[float]  # filled when comparing across repeats
    ari: Optional[float]
    nmi: Optional[float]

def compute_metrics_for_labels(
    assays: List[str],
    labels: np.ndarray,
    P_kw: pd.DataFrame,
    Q_kw: pd.DataFrame,
    top_n_words: int,
    cluster_mod,
) -> Tuple[RunMetrics, np.ndarray, Dict[int, List[int]]]:
    """
    Compute metrics for a single (k, repeat) labeling.
    Returns (RunMetrics-without-stability, Pkc matrix [C x K], cluster_members).
    """
    C = int(labels.max())
    cluster_members = _cluster_members(labels)
    cluster_sizes = np.array([len(cluster_members[cid]) for cid in range(1, C + 1)], dtype=float)

    # Word distributions per cluster (global top words for consistency)
    words_order, word_mat = cluster_mod.compute_cluster_word_matrix(
        assays=assays,
        cluster_members=cluster_members,
        top_n=top_n_words,
        per_cluster_top=False,  # global top-N
    )
    # Align vocab with P(k|w) and Q(w|k)
    words = list(words_order)
    common = [w for w in words if w in P_kw.index]
    if not common:
        raise RuntimeError("No overlap between cluster words and ED vocabulary.")
    P_kw_sub = P_kw.loc[common]                     # [W x K]
    Q_kw_sub = Q_kw.loc[:, common]                  # [K x W]
    P_wc = word_mat[common].to_numpy(dtype=float)   # [C x W]
    # Normalize rows to sum 1 (guard)
    P_wc = P_wc / np.clip(P_wc.sum(axis=1, keepdims=True), 1e-12, None)

    # P(k|c) = sum_w p_c(w) P(k|w)
    P_kc = (P_wc @ P_kw_sub.to_numpy(dtype=float))  # [C x K]

    # Reconstruction error: KL(p_c || hat_p_c)
    # hat_p_c(w) = sum_k P(k|c) q_k(w)
    Q_kw_arr = Q_kw_sub.to_numpy(dtype=float)       # [K x W]
    hat_P_wc = (P_kc @ Q_kw_arr)                    # [C x W]
    re_c = np.array([_kl_div(P_wc[i, :], hat_P_wc[i, :]) for i in range(C)], dtype=float)
    re_mean = float(np.average(re_c, weights=cluster_sizes))

    # Sharpness
    sharp = P_kc.max(axis=1)                        # [C]
    sharp_mean = float(np.average(sharp, weights=cluster_sizes))
    sharp_frac = float(np.mean(sharp >= 0.6))

    # Fragmentation per category
    m_kc = (cluster_sizes[:, None] * P_kc)          # [C x K]
    m_kc = m_kc / np.clip(m_kc.sum(axis=0, keepdims=True), 1e-12, None)  # col-normalize
    hhi_k = (m_kc ** 2).sum(axis=0)                 # [K]
    ne_k = 1.0 / np.clip(hhi_k, 1e-12, None)
    frag_max = float(np.max(ne_k))

    # Return: metrics so far (stability filled later when comparing runs), P_kc, and members
    run_metrics = RunMetrics(
        k=C, repeat=-1, re_mean=re_mean,
        sharp_mean=sharp_mean, sharp_frac=sharp_frac,
        frag_max=frag_max, stab_cos=None, ari=None, nmi=None
    )
    return run_metrics, P_kc, cluster_members

def load_ed_artifacts(ed_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    P_kw = pd.read_csv(Path(ed_dir) / "word_category_posteriors.csv", index_col=0)
    Q = pd.read_csv(Path(ed_dir) / "category_prototypes_q.csv", index_col=0)
    # Ensure consistent orientation: P_kw rows=words, cols=categories; Q rows=categories, cols=words
    if set(Q.index) <= set(P_kw.columns) and set(P_kw.columns) <= set(Q.index):
        pass
    elif set(Q.columns) <= set(P_kw.index):
        Q = Q.T
    return P_kw, Q

def load_zscored_matrix(matrix_path: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = pd.read_parquet(matrix_path)
    Xz, dropped = zscore_columns(df)
    if dropped:
        logging.info("Dropped %d zero-variance assays.", len(dropped))
    assays = list(Xz.columns)
    return Xz, assays, dropped

def import_cluster_module_or_die(script_path: str):
    mod = _import_cluster_module(script_path)
    if mod is None:
        raise SystemExit("Could not import assay_cluster_toxicity_index.py. Provide --cluster_script with a valid path.")
    return mod

def build_k_list(args) -> List[int]:
    if args.k_grid:
        return list(sorted(set(int(k) for k in args.k_grid)))
    if args.k_range:
        start, stop, step = args.k_range
        if step <= 0:
            raise SystemExit("--k_range step must be positive.")
        return list(range(int(start), int(stop) + 1, int(step)))
    raise SystemExit("Provide either --k_grid or --k_range.")

def preflight_corr_threshold_warning(Xz: pd.DataFrame, args, cluster_mod) -> None:
    if args.corr_threshold is None:
        return
    tau = float(args.corr_threshold)
    t_allowed = 1.0 - tau
    if hasattr(cluster_mod, "assay_distance_matrix"):
        _, Dcond_full, _ = cluster_mod.assay_distance_matrix(Xz)
    else:
        _, Dcond_full, _ = _assay_distance_matrix(Xz)
    min_d = float(np.min(Dcond_full)) if Dcond_full.size else float("inf")
    if not np.isfinite(min_d) or min_d > t_allowed + 1e-12:
        logging.warning(
            "corr_threshold τ=%.3f is globally unattainable for this matrix: "
            "min pairwise distance d_min=%.3f > 1-τ=%.3f (i.e., max |r| < τ). "
            "All K < n_assays will fail unless τ is lowered.",
            tau, min_d, t_allowed
        )

def collect_metrics_over_repeats(
    Xz: pd.DataFrame,
    P_kw: pd.DataFrame,
    Q: pd.DataFrame,
    args,
    cluster_mod,
    K_list: List[int],
) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.RandomState(args.random_seed)
    per_run_rows: List[Dict] = []
    per_k_rows: List[Dict] = []
    per_k_data = {K: {"Pkc_runs": [], "label_runs": [], "metrics_runs": [], "success_count": 0} for K in K_list}

    for r in tqdm(range(args.repeats), desc="Repeats", unit="r"):
        frac = max(min(args.bootstrap_frac, 1.0), 0.05)
        n = Xz.shape[0]
        idx = rng.randint(0, n, size=max(2, int(round(frac * n))))
        Xz_boot = Xz.iloc[idx, :]

        # Distances for this repeat
        if hasattr(cluster_mod, "assay_distance_matrix"):
            _, D_condensed, assays_boot = cluster_mod.assay_distance_matrix(Xz_boot)
        else:
            _, D_condensed, assays_boot = _assay_distance_matrix(Xz_boot)

        # If tau is impossible for this repeat, skip all K for this repeat
        if args.corr_threshold is not None:
            tau = float(args.corr_threshold)
            t_allowed = 1.0 - tau
            d_min = float(np.min(D_condensed)) if D_condensed.size else float("inf")
            if (not np.isfinite(d_min)) or d_min > t_allowed + 1e-12:
                logging.info(
                    "Skipping repeat r=%d: unattainable corr_threshold τ=%.3f "
                    "(d_min=%.3f > 1-τ=%.3f; max|r|<τ)", r, tau, d_min, t_allowed
                )
                continue

        # Linkage for this repeat
        Z = linkage(D_condensed, method=args.linkage, optimal_ordering=True)

        # Evaluate all Ks on the same dendrogram
        for K in K_list:
            # Enforce corr_threshold per K on this repeat (reject this (r,K) only)
            if args.corr_threshold is not None:
                t_k = _threshold_for_k(Z, n_leaves=len(assays_boot), k=K)
                t_allowed = 1.0 - float(args.corr_threshold)
                if t_k > t_allowed + 1e-12:
                    logging.debug(
                        "repeat=%d: K=%d violates τ=%.3f (t_K=%.3f > 1-τ=%.3f); skipping (r,K).",
                        r, K, args.corr_threshold, t_k, t_allowed
                    )
                    continue

            # Labels and metrics for this (r,K)
            labels = _labels_for_k(Z, K)
            labels = _relabel_stable(labels)
            run_metrics, Pkc, _members = compute_metrics_for_labels(
                assays=assays_boot, labels=labels, P_kw=P_kw, Q_kw=Q,
                top_n_words=args.top_n_words, cluster_mod=cluster_mod,
            )
            run_metrics.k = K
            run_metrics.repeat = r

            # Per-run success: meets per-run criteria (stability is per-K, so not checked here)
            pass_run = (
                (run_metrics.sharp_mean >= args.sharpness_threshold) and
                (run_metrics.sharp_frac >= args.sharpness_frac) and
                (run_metrics.frag_max  <= args.frag_threshold)
            )
            if pass_run:
                per_k_data[K]["success_count"] += 1

            per_k_data[K]["metrics_runs"].append(run_metrics)
            per_k_data[K]["Pkc_runs"].append(Pkc)
            per_k_data[K]["label_runs"].append(labels)

            per_run_rows.append({
                "K": K, "repeat": r,
                "RE": run_metrics.re_mean, "Sharp": run_metrics.sharp_mean,
                "Sharp_frac": run_metrics.sharp_frac, "Frag_max": run_metrics.frag_max,
                "Pass": int(pass_run)
            })

    # Summarize per-K after all repeats
    def agg(vals: List[float]) -> Tuple[float, float]:
        arr = np.array(vals, dtype=float)
        mean = float(np.mean(arr)) if arr.size else float("nan")
        se = float(np.std(arr, ddof=1) / math.sqrt(max(len(arr), 1))) if len(arr) > 1 else 0.0
        return mean, se

    for K in K_list:
        Kdata = per_k_data[K]
        metrics_runs = Kdata["metrics_runs"]
        if not metrics_runs:
            logging.warning("Skipping K=%d: no repeats passed --corr_threshold.", K)
            continue

        # Compute success stats (denominator = configured repeats)
        successes = int(Kdata["success_count"])
        success_rate = successes / float(max(args.repeats, 1))
        if success_rate < args.min_success_rate:
            logging.warning("Skipping K=%d: success_rate=%.3f < min_success_rate=%.3f (successes=%d of %d).",
                            K, success_rate, args.min_success_rate, successes, args.repeats)
            continue

        # Stability across repeats (pairwise averaged) for this K
        Pkc_runs = Kdata["Pkc_runs"]
        label_runs = Kdata["label_runs"]
        cos_sims, aris, nmis = [], [], []
        for i in range(len(Pkc_runs)):
            for j in range(i + 1, len(Pkc_runs)):
                cos_sims.append(_cosine_sim(Pkc_runs[i], Pkc_runs[j]))
                aris.append(adjusted_rand_score(label_runs[i], label_runs[j]))
                nmis.append(normalized_mutual_info_score(label_runs[i], label_runs[j]))
        stab_cos = float(np.mean(cos_sims)) if cos_sims else None
        ari = float(np.mean(aris)) if aris else None
        nmi = float(np.mean(nmis)) if nmis else None

        re_mean, re_se = agg([m.re_mean for m in metrics_runs])
        sharp_mean, sharp_se = agg([m.sharp_mean for m in metrics_runs])
        sharp_frac, sharp_frac_se = agg([m.sharp_frac for m in metrics_runs])
        frag_max, frag_se = agg([m.frag_max for m in metrics_runs])

        per_k_rows.append({
            "K": K,
            "RE_mean": re_mean, "RE_se": re_se,
            "Sharp_mean": sharp_mean, "Sharp_se": sharp_se,
            "Sharp_frac": sharp_frac, "Sharp_frac_se": sharp_frac_se,
            "Frag_max": frag_max, "Frag_se": frag_se,
            "Stab_cos": stab_cos, "ARI": ari, "NMI": nmi,
            "Successes": successes, "Success_rate": success_rate,
        })

        # Optional representative P(k|c) emission remains the caller's choice

    return per_k_rows, per_run_rows

def choose_k_from_metrics(per_k_df: pd.DataFrame, args) -> Tuple[int, str]:
    # 1) Find K* minimizing RE_mean
    idx_min = per_k_df["RE_mean"].idxmin()
    re_star = float(per_k_df.loc[idx_min, "RE_mean"])
    K_min = int(per_k_df.loc[idx_min, "K"])
    # 2) If one-SE rule, allow K with RE <= re_star + se_star and pick smallest
    if args.one_se:
        se_star = float(per_k_df.loc[idx_min, "RE_se"])
        cand = per_k_df[per_k_df["RE_mean"] <= re_star + se_star].copy()
        candidates = cand.sort_values("K")
    else:
        candidates = per_k_df[per_k_df["K"] == K_min]
    # 3) Apply thresholds to candidates; pick smallest acceptable K; else fallback
    def ok(row) -> bool:
        return (
            row["Sharp_mean"] >= args.sharpness_threshold and
            row["Sharp_frac"] >= args.sharpness_frac and
            row["Frag_max"] <= args.frag_threshold and
            (np.isnan(row["Stab_cos"]) or row["Stab_cos"] >= args.stab_threshold)
        )
    cands_ok = [int(r["K"]) for _, r in candidates.iterrows() if ok(r)]
    if cands_ok:
        return min(cands_ok), ("one-SE" if args.one_se else "min-RE")
    return K_min, "min-RE-fallback"

def main():
    ap = argparse.ArgumentParser(description="Label-guided selection of assay cluster count (K).")
    io = ap.add_argument_group("Core I/O")
    io.add_argument("--matrix", required=True, help="Parquet matrix: rows=chemicals, cols=assays.")
    io.add_argument("--ed_artifacts", default=None,
                    help="Directory containing word_category_posteriors.csv and category_prototypes_q.csv.")
    io.add_argument("--outdir", required=True, help="Directory to write reports.")

    sweep = ap.add_argument_group("K sweep & stability")
    sweep.add_argument("--k_grid", nargs="+", type=int, default=None, help="Explicit list of K values (e.g., 8 12 16).")
    sweep.add_argument("--corr_threshold", type=float, default=None,
                       help="Correlation threshold tau in (0,1). For each K, compute its implied dendrogram cut distance "
                       "t_K and skip K if t_K > 1 - tau.")
    sweep.add_argument("--k_range", nargs=3, type=int, default=None,
                       help="Start Stop Step for K (inclusive start, inclusive stop).")
    sweep.add_argument("--repeats", type=int, default=5, help="Number of bootstrap repeats per K (default: 5).")
    sweep.add_argument("--bootstrap_frac", type=float, default=0.8,
                       help="Fraction of chemicals to sample with replacement per bootstrap (default: 0.8).")
    sweep.add_argument("--linkage", type=str, default="average", choices=["complete", "average"],
                       help="Hierarchical linkage (default: average).")
    sweep.add_argument("--top_n_words", type=int, default=30, help="Global top-N words per cluster for metrics.")
    sweep.add_argument("--emit_perk", action="store_true", help="Write per-K P(k|c) matrices to disk.")

    crit = ap.add_argument_group("Selection criteria (defaults are conservative)")
    crit.add_argument("--sharpness_threshold", type=float, default=0.8, help="Weighted mean sharpness threshold.")
    crit.add_argument("--sharpness_frac", type=float, default=0.9, help="Min fraction of clusters with max P(k|c) >= 0.6.")
    crit.add_argument("--frag_threshold", type=float, default=2.5, help="Max allowed NE_k (inverse HHI).")
    crit.add_argument("--stab_threshold", type=float, default=0.9, help="Min cosine stability across repeats.")
    crit.add_argument("--one_se", action="store_true", help="Apply one-standard-error rule on RE(K).")
    crit.add_argument("--min_success_rate", type=float, default=0.05,
                  help="Exclude K with fewer than this fraction of repeats passing per-run criteria (default: 0.05).")

    bp = ap.add_argument_group("build_posteriors passthrough (optional)")
    bp.add_argument("--build_posteriors", action="store_true",
                    help="If set, call cluster_category_scoring.py build_posteriors before sweep.")
    bp.add_argument("--bp_script", default="scripts/stages/cluster_category_scoring.py",
                    help="Path to cluster_category_scoring.py.")
    bp.add_argument("--bp_corpus", default=None, help="--corpus for build_posteriors.")
    bp.add_argument("--bp_text_col", default="title", help="--text_col for build_posteriors.")
    bp.add_argument("--bp_vocab", default=None, help="--vocab for build_posteriors.")
    bp.add_argument("--bp_vocab_size", type=int, default=5000, help="--vocab_size for build_posteriors.")
    bp.add_argument("--bp_bg_mode", choices=["uniform", "corpus"], default="corpus", help="--bg_mode.")
    bp.add_argument("--bp_priors", choices=["uniform", "from-seeds", "from-file"], default="uniform", help="--priors.")
    bp.add_argument("--bp_priors_file", default=None, help="--priors_file.")
    bp.add_argument("--bp_stopwords", default=None, help="--stopwords.")
    bp.add_argument("--bp_collocations", action="store_true", help="--collocations for build_posteriors.")
    bp.add_argument("--bp_collocation_threshold", type=int, default=30, help="--collocation_threshold for build_posteriors.")
    bp.add_argument("--bp_outdir", default=None, help="Where to write ED artifacts (defaults to --ed_artifacts).")

    ap.add_argument("--cluster_script", default="scripts/stages/assay_cluster_toxicity_index.py",
                    help="Path to assay_cluster_toxicity_index.py to import WordCloud logic for p_c(w).")
    ap.add_argument("--random_seed", type=int, default=42, help="Base random seed.")
    args = ap.parse_args()

    # Logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Optionally build ED artifacts
    if args.build_posteriors:
        ed_dir = Path(args.bp_outdir or (args.ed_artifacts or (outdir / "ed_artifacts")))
        ed_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, args.bp_script, "build_posteriors",
            "--outdir", str(ed_dir),
            "--bg_mode", args.bp_bg_mode,
            "--priors", args.bp_priors,
            "--vocab_size", str(args.bp_vocab_size),
            "--text_col", args.bp_text_col,
        ]
        if args.bp_corpus: cmd += ["--corpus", args.bp_corpus]
        if args.bp_vocab: cmd += ["--vocab", args.bp_vocab]
        if args.bp_priors_file: cmd += ["--priors_file", args.bp_priors_file]
        if args.bp_stopwords: cmd += ["--stopwords", args.bp_stopwords]
        if args.bp_collocations: cmd += ["--collocations"]
        if args.bp_collocation_threshold is not None:
            cmd += ["--collocation_threshold", str(args.bp_collocation_threshold)]
        logging.info("Building ED artifacts via: %s", " ".join(cmd))
        rc = os.spawnvp(os.P_WAIT, cmd[0], cmd)
        if rc != 0:
            raise SystemExit(f"build_posteriors failed with exit code {rc}")
        # Point ed-artifacts to built directory if not provided
        if args.ed_artifacts is None:
            args.ed_artifacts = str(ed_dir)

    if args.ed_artifacts is None:
        raise SystemExit("--ed_artifacts is required (or provide --build_posteriors to create them).")

    # Load ED artifacts
    P_kw, Q = load_ed_artifacts(args.ed_artifacts)

    # Load and z-score the activity matrix
    Xz, assays, dropped = load_zscored_matrix(args.matrix)

    # Try to import cluster module for WordCloud word matrix
    cluster_mod = import_cluster_module_or_die(args.cluster_script)

    # --- Preflight: if tau is given, warn if globally unattainable (min distance > 1 - tau) ---
    preflight_corr_threshold_warning(Xz, args, cluster_mod)

    # Prepare K grid
    K_list = build_k_list(args)

    per_k_rows, per_run_rows = collect_metrics_over_repeats(
        Xz=Xz, P_kw=P_kw, Q=Q, args=args, cluster_mod=cluster_mod, K_list=K_list
    )

    # If all K were skipped by corr_threshold, stop early with guidance
    if not per_k_rows:
        raise SystemExit("No K values satisfied --corr_threshold; relax the threshold or adjust --k_grid/--k_range.")

    # Write reports
    per_k_df = pd.DataFrame(per_k_rows).sort_values("K")
    per_run_df = pd.DataFrame(per_run_rows).sort_values(["K", "repeat"])
    per_k_df.to_csv(Path(args.outdir) / "choose_k_metrics.csv", index=False)
    per_run_df.to_csv(Path(args.outdir) / "choose_k_perrun.csv", index=False)

    # Choose K using rules
    chosen_k, reason = choose_k_from_metrics(per_k_df, args)

    # Count successes/failures for the chosen K (treat skipped repeats as failures)
    runs_ck = per_run_df[per_run_df["K"] == chosen_k].copy()
    if "Pass" in runs_ck.columns:
        successes = int(runs_ck["Pass"].sum())
    else:
        # Derive per-run pass from thresholds if Pass column isn't present
        successes = int(((runs_ck["Sharp"] >= args.sharpness_threshold) &
                        (runs_ck["Sharp_frac"] >= args.sharpness_frac) &
                        (runs_ck["Frag_max"] <= args.frag_threshold)).sum())
    failures = int(max(args.repeats, 0) - successes)
    success_rate = (successes / float(max(args.repeats, 1)))

    with open(Path(args.outdir) / "chosen_k.json", "w", encoding="utf-8") as f:
        json.dump({
            "chosen_k": chosen_k,
            "reason": reason,
            "successes": successes,
            "failures": failures,
            "success_rate": success_rate,
            "criteria": {
                "sharpness_threshold": args.sharpness_threshold,
                "sharpness_frac": args.sharpness_frac,
                "frag_threshold": args.frag_threshold,
                "stab_threshold": args.stab_threshold,
                "one_se": args.one_se,
            }
        }, f, indent=2, sort_keys=True)


    print(f"Chosen K = {chosen_k} ({reason}); metrics written to {Path(args.outdir) / 'choose_k_metrics.csv'}")

if __name__ == "__main__":
    main()
