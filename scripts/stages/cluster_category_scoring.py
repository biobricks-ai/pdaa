
#!/usr/bin/env python3
"""
Compute continuous category scores for clusters from word-frequency histograms.

Inputs
------
- cluster_word_matrix.csv: rows=cluster_id, columns=words, values=relative frequencies
  (as exported from your histogram code; each row should sum to ~1.0).

Method
------
For each category k, build a smoothed prototype word distribution q_k(w)
from seed words and a background distribution p_bg(w). Score each cluster's
histogram p_c(w) with expected log-likelihood s_{c,k} = sum_w p_c(w) * log q_k(w),
then convert to probabilities via a temperature-softmax.

This yields stable, continuous scores that change smoothly with the input
histograms and remain interpretable (per-word contributions are additive).

Usage
-----
python cluster_category_scoring.py \
  -i /path/to/cluster_histograms/cluster_word_matrix.csv \
  -o /path/to/cluster_histograms/cluster_category_scores.csv \
  --alpha 50 --tau 2.0

Optional: provide your own category seeds as a JSON file:
{
  "ER (estrogen receptor)": ["er", "estrogen", "estradiol", "e2", "esr1", "esr2"],
  "AR (androgen receptor)": ["ar", "androgen", "dht", "dihydrotestosterone"]
  ...
}
python cluster_category_scoring.py -i cluster_word_matrix.csv -s seeds.json

Notes
-----
- The default seed set below is a pragmatic starting point. Customize it for your domain.
- All matching is case-insensitive and exact on tokens as they appear in the CSV columns.
- q_k(w) uses Dirichlet smoothing towards p_bg(w) to avoid overconfident spikes.
- The output is a tidy CSV of cluster-by-category probabilities.
"""

from __future__ import annotations
import argparse
import json
import hashlib
import re
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, Tuple, Optional, Sequence

import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

from pathlib import Path

import sys
sys.path.append('./')  # so utility scripts can be found
from scripts.utils.helpers import clean_title


# Default seed lexicons (customize for your domain).
# Acronyms are expanded in labels for clarity.
# DEFAULT_CATEGORY_SEEDS = {
#     "ER (estrogen receptor)": [
#         "er", "estrogen", "estradiol", "e2", "esr1", "esr2"
#     ],
#     "AR (androgen receptor)": [
#         "ar", "androgen", "dht", "dihydrotestosterone"
#     ],
#     "PR (progesterone receptor)": [
#         "pr", "progesterone", "pgr"
#     ],
#     "GR (glucocorticoid receptor)": [
#         "gr", "glucocorticoid", "nr3c1", "cortisol"
#     ],
#     "THR (thyroid hormone receptor)": [
#         "thr", "thyroid", "tr", "tralpha", "trbeta", "t3", "t4"
#     ],
#     "VDR (vitamin D receptor)": [
#         "vdr", "vitamin d", "calcitriol"
#     ],
#     "FXR (farnesoid X receptor)": [
#         "fxr", "nr1h4"
#     ],
#     "RAR (retinoic acid receptor)": [
#         "rar", "retinoic", "retinoid", "ralpha", "rbeta", "rgamma"
#     ],
#     "RXR (retinoid X receptor)": [
#         "rxr"
#     ],
#     "PXR (pregnane X receptor)": [
#         "pxr", "nr1i2"
#     ],
#     "CAR (constitutive androstane receptor)": [
#         "car", "nr1i3"
#     ],
#     "PPARG (peroxisome proliferator-activated receptor gamma)": [
#         "pparg", "ppar gamma", "pparγ"
#     ],
#     "AHR (aryl hydrocarbon receptor)": [
#         "ahr", "aryl hydrocarbon"
#     ],

#     # Optional endocrine-relevant targets often seen in assay text
#     "Aromatase (CYP19A1)": [
#         "aromatase", "cyp19", "cyp19a1"
#     ],
#     "RORA/RORC (retinoic acid receptor-related orphan)": [
#         "ror", "rora", "rorc"
#     ],
#     "PPAR (broad)": [
#         "ppar", "ppara", "ppard", "pparg"
#     ],
#     "HDAC (histone deacetylase)": [
#         "histone deacetylase", "hdac"
#     ],
#     "Somatostatin": [
#         "somatostatin"
#     ],
#     "Melanocortin": [
#         "melanocortin"
#     ],
#     "Adrenergic": [
#         "adrenergic", "adrenoceptor"
#     ],
#     "Angiotensin/TGF": [
#         "angiotensin", "tgf"
#     ],
#     "Prostaglandin": [
#         "prostaglandin"
#     ],
# }
DEFAULT_CATEGORY_SEEDS = {
    # Steroid hormone receptors
    "ER (estrogen receptor)": ["er", "estrogen", "estradiol", "e2", "esr1", "esr2"],
    "AR (androgen receptor)": ["ar", "androgen", "dht", "dihydrotestosterone"],
    "PR (progesterone receptor)": ["pr", "progesterone", "pgr"],
    "GR (glucocorticoid receptor)": ["gr", "glucocorticoid", "nr3c1", "cortisol"],
    "MR (mineralocorticoid receptor)": ["mr", "nr3c2", "aldosterone"],

    # Thyroid axis (receptor + key mechanisms)
    "TR (thyroid hormone receptor)": ["tr", "tralpha", "trbeta", "thyroid hormone receptor", "t3", "t4"],
    "Thyroid synthesis/metabolism (TPO/NIS/DIO)": [
        "tpo", "thyroid peroxidase", "nis", "sodium iodide symporter", "slc5a5",
        "deiodinase", "dio1", "dio2", "dio3"
    ],
    "Thyroid transport (TTR/TBG)": [
        "transthyretin", "ttr", "thyroxine binding globulin", "tbg", "thyroxine transport"
    ],

    # Steroid biosynthesis
    "Steroidogenesis (H295R/enzyme panel)": [
        "steroidogenesis", "h295r", "star", "cyp11a1", "cyp17a1", "cyp21a2",
        "hsd3b1", "hsd3b2", "hsd17b1", "srd5a1", "srd5a2",
        "progesterone", "testosterone", "androstenedione", "11-deoxycortisol"
    ],
    "Aromatase (CYP19A1)": ["aromatase", "cyp19", "cyp19a1"],

    # Metabolic nuclear receptors
    "PPARα": ["ppara", "ppar alpha"],
    "PPARδ": ["ppard", "ppar delta"],
    "PPARγ": ["pparg", "ppar gamma", "ppargamma"],

    # Retinoid / vitamin D
    "RAR (retinoic acid receptor)": ["rar", "retinoic", "retinoid", "rara", "rarb", "rarg"],
    "RXR (retinoid X receptor)": ["rxr", "rxra", "rxrb", "rxrg"],
    "VDR (vitamin D receptor)": ["vdr", "vitamin d", "calcitriol"],

    # Xenobiotic / cross-talk NRs
    "AhR (aryl hydrocarbon receptor)": ["ahr", "aryl hydrocarbon"],
    "PXR (pregnane X receptor)": ["pxr", "nr1i2", "pregnane x receptor"],
    "CAR (constitutive androstane receptor)": ["car", "nr1i3"],
    # Optional (enable only if screened)
    # "FXR (farnesoid X receptor)": ["fxr", "nr1h4", "bile acid receptor"],
    # "ROR (retinoic acid receptor-related orphan)": ["ror", "rora", "rorc"],
}


def load_seeds(path: str | None) -> Mapping[str, List[str]]:
    if not path:
        return DEFAULT_CATEGORY_SEEDS
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith(".json"):
            return json.load(f)
        try:
            import yaml  # type: ignore
            return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(
                "Only .json or .yaml/.yml seeds are supported; install PyYAML for YAML."
            ) from e

def load_external_background(path: str, columns: Iterable[str]) -> pd.Series:
    """
    Load a word->weight mapping. Accepts:
      (a) CSV with two columns [word, weight], or
      (b) single-row CSV whose columns match the input matrix columns.
    Returns a Series indexed by 'columns'.
    """
    df = pd.read_csv(path)
    cols = list(columns)

    # Case (b): single-row with same columns
    if set(df.columns) >= set(cols):
        row = df.iloc[0][cols].astype(float)
        return pd.Series(row.values, index=cols, dtype=float)

    # Case (a): two-column mapping
    if df.shape[1] >= 2:
        wcol, vcol = df.columns[:2]
        s = pd.Series(df[vcol].values, index=df[wcol].astype(str).values, dtype=float)
        return s.reindex(cols)

    raise ValueError("Unrecognized external background format")

def build_background(
    df: pd.DataFrame,
    *,
    mode: str = "cluster-mean",
    external: pd.Series | None = None,
) -> np.ndarray:
    """
    Background distribution p_bg(w) over words.

    mode:
      - "cluster-mean": column sums of the cluster matrix (current behavior; depends on how you cluster)
      - "uniform": uniform over vocabulary (invariant to clustering)
      - "external": use an externally supplied vector aligned to df.columns (e.g., corpus-level background)

    external: when mode="external", a pd.Series indexed by df.columns (or coercible to that).
    """
    V = df.shape[1]
    if mode == "uniform":
        return np.full(V, 1.0 / max(V, 1), dtype=float)

    if mode == "external":
        if external is None:
            raise ValueError("external background requested but none provided")
        s = external.reindex(df.columns).astype(float).to_numpy()
        s = np.clip(s, 0.0, None)
        total = s.sum()
        if not np.isfinite(s).all() or total <= 0:
            raise ValueError("Invalid external background; must be nonnegative and nonzero")
        return s / total

    # default: cluster-mean (equal weight per cluster)
    col_sum = df.sum(axis=0).to_numpy(dtype=float)
    col_sum = np.clip(col_sum, 0.0, None)
    if not np.isfinite(col_sum).all() or col_sum.sum() <= 0:
        raise ValueError("Invalid input matrix for background distribution")
    return col_sum / col_sum.sum()

def build_background_from_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.clip(counts.astype(float), 0.0, None)
    total = counts.sum()
    if not np.isfinite(counts).all() or total <= 0:
        raise ValueError("Invalid counts for background: non-finite or zero total.")
    return counts / total

def build_prototypes(
    vocab: List[str],
    seeds: Mapping[str, Iterable[str]],
    p_bg: np.ndarray,
    alpha: float,
    bump: float = 1.0,
) -> Tuple[List[str], np.ndarray]:
    """
    Create category prototypes q_k(w) with Dirichlet smoothing.
    q_k(w) ∝ bump*1[w in seeds_k] + alpha*p_bg(w).

    Returns
    -------
    categories: list[str] of category names (order preserved from input mapping)
    Q: np.ndarray of shape (K, V) with rows summing to 1
    """
    V = len(vocab)
    vocab_lc = [w.lower() for w in vocab]

    # Pre-index vocabulary for O(1) lookup
    index = {w: i for i, w in enumerate(vocab_lc)}

    categories = list(seeds.keys())
    K = len(categories)
    Q = np.empty((K, V), dtype=float)

    for k, cat in enumerate(categories):
        inds = []
        for token in seeds[cat]:
            t0 = clean_title(token)
            for cand in (t0, t0.replace("_", " "), t0.replace(" ", "_")):
                if cand in index:
                    inds.append(index[cand])

        base = np.zeros(V, dtype=float)
        if inds:
            base[inds] = bump
        # Dirichlet smoothing towards background
        q = base + alpha * p_bg
        q_sum = q.sum()
        if q_sum <= 0:
            raise ValueError(f"Prototype for category '{cat}' sums to zero")
        Q[k, :] = q / q_sum
    return categories, Q


def softmax(x: np.ndarray, tau: float = 1.0, axis: int = -1) -> np.ndarray:
    z = x / max(tau, 1e-8)
    z = z - np.max(z, axis=axis, keepdims=True)
    np.exp(z, out=z)
    denom = np.sum(z, axis=axis, keepdims=True)
    return z / np.clip(denom, 1e-12, None)


def score_clusters(
    df: pd.DataFrame,
    seeds: Mapping[str, Iterable[str]],
    alpha: float,
    tau: float,
    bump: float = 1.0,
    *,
    bg_mode: str = "cluster-mean",
    external_bg: pd.Series | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute category probabilities per cluster.

    Returns
    -------
    probs_df : DataFrame [clusters x categories] of probabilities
    q_df     : DataFrame [categories x words] of prototype distributions
    """
    # Ensure non-negative and row-normalized
    M = df.to_numpy(dtype=float)
    M = np.clip(M, 0.0, None)
    row_sum = M.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    P = M / row_sum  # p_c(w)

    p_bg = build_background(df, mode=bg_mode, external=external_bg)  # p_bg(w)
    categories, Q = build_prototypes(
        vocab=list(df.columns), seeds=seeds, p_bg=p_bg, alpha=float(alpha), bump=float(bump)
    )

    # Expected log-likelihood per cluster and category: s_{c,k} = E_{p_c}[log q_k]
    logQ = np.log(np.clip(Q, 1e-12, None))  # (K, V)
    S = P @ logQ.T  # (C, K)

    # Calibrated probabilities via temperature softmax
    Probs = softmax(S, tau=float(tau), axis=1)  # (C, K)

    probs_df = pd.DataFrame(Probs, index=df.index, columns=categories)
    q_df = pd.DataFrame(Q, index=categories, columns=df.columns)
    return probs_df, q_df


# ---------------- Additional helpers for building P(k|w) ----------------

def load_phrases(path: Optional[str]) -> List[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [clean_title(line) for line in f if line.strip()]

def apply_phrases(s: str, phrases: List[str]) -> str:
    phrases_sorted = sorted(set(phrases), key=len, reverse=True)
    for p in phrases_sorted:
        s = re.sub(rf"\b{re.escape(p)}\b", p.replace(" ", "_"), s)
    return s

def tokenize_text(s: str, phrases: Optional[List[str]] = None) -> List[str]:
    # Not used for corpus counting when --collocations is enabled; kept for compatibility/debug.
    s = clean_title(s)
    return s.split()

def load_stopwords(path: Optional[str]) -> set:
    if not path:
        return set()
    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            stopwords = {clean_title(w) for w in f if w}
            # remove empty lines and normalize
            stopwords.discard("")
            return stopwords
    elif path.endswith(".pkl"):
        import pickle
        return pickle.load(open(path, "rb"))

def read_corpus(path: str, text_col: str = 'title') -> List[str]:
    p = Path(path)
    if p.suffix.lower() in {".txt"}:
        return [line.rstrip("\n") for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    sep = "\t" if p.suffix.lower() in {".tsv"} else ","
    df = pd.read_csv(p, sep=sep)
    col = text_col
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {path}")
    return df[col].astype(str).tolist()

# def build_vocab_from_corpus(texts: Sequence[str],
#                             vocab_size: int,
#                             seeds: Mapping[str, Iterable[str]],
#                             stopwords: set,
#                             phrases: List[str]) -> List[str]:
#     counts: Dict[str, int] = {}
#     for s in texts:
#         toks = tokenize_text(s, phrases)
#         for t in toks:
#             if t in stopwords:
#                 continue
#             counts[t] = counts.get(t, 0) + 1
#     # ensure seed tokens exist
#     for ts in seeds.values():
#         for t in ts:
#             tt = str(t).lower().strip()
#             counts.setdefault(tt, 0)
#     sorted_tokens = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
#     vocab = [w for w, _ in sorted_tokens[:vocab_size]]
#     seed_set = {str(t).lower().strip() for ts in seeds.values() for t in ts}
#     for t in seed_set:
#         if t not in vocab:
#             vocab.append(t)
#     return vocab
# def build_vocab_from_corpus(texts: Sequence[str],
#                             vocab_size: int,
#                             seeds: Mapping[str, Iterable[str]],
#                             stopwords: set,
#                             phrases: List[str]) -> Tuple[List[str], Dict[str, int]]:
#     """
#     Build counts using WordCloud.process_text to filter tokens:
#     - include_numbers=False (default) drops pure numbers like '1', '2024'
#     - honors combined stopwords (WordCloud STOPWORDS ∪ custom stopwords)
#     - respects phrase joining (e.g., 'vitamin d' -> 'vitamin_d') before processing
#     """
#     # Combine stopwords with WordCloud's defaults (should already be done, but as a safeguard)
#     combined_stop = set(STOPWORDS) | set(stopwords)
#     wc = WordCloud(
#         stopwords=combined_stop,
#         collocations=True,  # keep phrases intact
#         # include_numbers defaults to False in WordCloud, which drops pure-number tokens
#         background_color="white",  # irrelevant for process_text but required by some versions
#         regexp=r"[a-z0-9_]+",
#     )

#     counts: Dict[str, int] = {}
#     for s in texts:
#         # normalize + join phrases first so WordCloud sees 'vitamin_d' as one token
#         s = apply_phrases(clean_title(s), phrases)
#         for w, c in wc.process_text(s).items():
#             counts[w] = counts.get(w, 0) + int(c)

#     # ensure seed tokens exist (even if absent from corpus counts)
#     for cat, toks in seeds.items():
#         for t in toks:
#             tt = str(t).lower().strip()
#             counts.setdefault(tt, 0)

#     # take top-N + ensure seeds included
#     sorted_tokens = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
#     vocab = [w for w, _ in sorted_tokens[:vocab_size]]
#     seed_set = {str(t).lower().strip() for ts in seeds.values() for t in ts}
#     for t in seed_set:
#         if t not in vocab:
#             vocab.append(t)

#     aligned_counts = {w: counts.get(w, 0) for w in vocab}
#     return vocab, aligned_counts
def build_vocab_from_corpus(texts: Sequence[str],
                            vocab_size: int,
                            seeds: Mapping[str, Iterable[str]],
                            stopwords: set,
                            *,
                            collocations: bool,
                            collocation_threshold: int) -> List[str]:
    wc = _make_wc(stopwords, collocations=collocations, collocation_threshold=collocation_threshold)

    counts: Dict[str, int] = {}
    for s in texts:
        s = clean_title(s)  # project-standard normalization
        for w, c in wc.process_text(s).items():
            counts[w] = counts.get(w, 0) + int(c)

    # Ensure seed tokens exist (in either space or underscore form)
    for ts in seeds.values():
        for t in ts:
            t0 = clean_title(t)
            for cand in (t0, t0.replace("_", " "), t0.replace(" ", "_")):
                counts.setdefault(cand, 0)

    sorted_tokens = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    vocab = [w for w, _ in sorted_tokens[:vocab_size]]

    # Force-include seeds if any are missing after top-N cut
    seed_cands = {clean_title(t) for ts in seeds.values() for t in ts}
    seed_cands |= {s.replace("_", " ") for s in list(seed_cands)}
    for t in sorted(seed_cands):
        if t not in vocab:
            vocab.append(t)

    return vocab


# def background_from_corpus(texts: Sequence[str],
#                            vocab: List[str],
#                            stopwords: set,
#                            phrases: List[str]) -> np.ndarray:
#     index = {w: i for i, w in enumerate(vocab)}
#     counts = np.zeros(len(vocab), dtype=float)
#     for s in texts:
#         toks = tokenize_text(s, phrases)
#         for t in toks:
#             if t in stopwords:
#                 continue
#             if t in index:
#                 counts[index[t]] += 1.0
#     return (counts / max(counts.sum(), 1e-12)).astype(float)
# def background_from_corpus(texts: Sequence[str],
#                            vocab: List[str],
#                            stopwords: set,
#                            phrases: List[str]) -> np.ndarray:
#     """
#     Use WordCloud.process_text so background inherits the same filtering behavior
#     (pure numbers dropped, stopwords applied, phrase joining respected).
#     """
#     combined_stop = set(STOPWORDS) | set(stopwords)
#     wc = WordCloud(
#         stopwords=combined_stop,
#         collocations=True,
#         background_color="white",
#         regexp=r"[a-z0-9_]+"
#     )

#     index = {w: i for i, w in enumerate(vocab)}
#     counts = np.zeros(len(vocab), dtype=float)
#     for s in texts:
#         s = apply_phrases(clean_title(s), phrases)
#         for w, c in wc.process_text(s).items():
#             if w in index:
#                 counts[index[w]] += float(c)
#     return build_background_from_counts(counts)
def background_from_corpus(texts: Sequence[str],
                           vocab: List[str],
                           stopwords: set,
                           *,
                           collocations: bool,
                           collocation_threshold: int) -> np.ndarray:
    wc = _make_wc(stopwords, collocations=collocations, collocation_threshold=collocation_threshold)
    index = {w: i for i, w in enumerate(vocab)}
    counts = np.zeros(len(vocab), dtype=float)

    for s in texts:
        s = clean_title(s)
        for w, c in wc.process_text(s).items():
            if w in index:
                counts[index[w]] += float(c)

    total = counts.sum()
    return (counts / max(total, 1e-12)).astype(float)


def choose_priors(categories: List[str],
                  mode: str,
                  seeds: Mapping[str, Iterable[str]],
                  vocab: List[str],
                  priors_file: Optional[str]) -> np.ndarray:
    K = len(categories)
    if mode == "uniform":
        return np.full(K, 1.0 / max(K, 1), dtype=float)
    if mode == "from-seeds":
        vocab_set = set(vocab)
        counts = np.array([sum(1 for t in seeds[c] if str(t).lower().strip() in vocab_set) or 1 for c in categories], dtype=float)
        return counts / counts.sum()
    if mode == "from-file":
        if not priors_file:
            raise ValueError("--priors from-file requires --priors-file")
        import json as _json
        import pandas as _pd
        if priors_file.lower().endswith(".json"):
            with open(priors_file, "r", encoding="utf-8") as f:
                m = _json.load(f)
            arr = np.array([float(m.get(c, 0.0)) for c in categories], dtype=float)
        else:
            dfp = _pd.read_csv(priors_file)
            if set(["category", "prior"]).issubset(dfp.columns):
                mapping = {row["category"]: float(row["prior"]) for _, row in dfp.iterrows()}
                arr = np.array([mapping.get(c, 0.0) for c in categories], dtype=float)
            else:
                arr = dfp.iloc[0][categories].to_numpy(dtype=float)
        if arr.sum() <= 0:
            raise ValueError("Invalid priors from file: must be positive and nonzero.")
        return arr / arr.sum()
    raise ValueError(f"Unknown priors mode: {mode}")

def compute_word_posteriors(Q: np.ndarray,
                            categories: List[str],
                            vocab: List[str],
                            priors: np.ndarray) -> pd.DataFrame:
    priors = np.asarray(priors, dtype=float)
    priors = priors / max(priors.sum(), 1e-12)
    num = (priors[:, None] * Q)
    denom = np.clip(num.sum(axis=0, keepdims=True), 1e-12, None)
    R = num / denom
    R_df = pd.DataFrame(R.T, index=vocab, columns=categories)
    R_df.index.name = "word"
    return R_df

# ---------------- Subcommand implementations ----------------

def cmd_score(args):
    df = pd.read_csv(args.input, index_col=0)
    seeds = load_seeds(args.seeds)
    V = df.shape[1]
    alpha = args.alpha if args.alpha is not None else (50.0 / max(V, 1))
        # Score with selected background mode (avoid code duplication & the uniform bug)
    external_bg = None
    if args.bg_mode == "external":
        if not args.bg_file:
            raise SystemExit("--bg-mode=external requires --bg-file")
        ext = pd.read_csv(args.bg_file)
        if set(ext.columns) >= set(df.columns):
            s = ext.iloc[0][df.columns].astype(float)
        elif ext.shape[1] >= 2:
            wcol, vcol = ext.columns[:2]
            s = ext.set_index(wcol)[vcol].reindex(df.columns).astype(float).fillna(0.0)
        else:
            raise SystemExit("Unrecognized external background format")
        external_bg = s

    probs_df, q_df = score_clusters(
        df, seeds=seeds, alpha=alpha, tau=args.tau, bump=args.bump,
        bg_mode=args.bg_mode, external_bg=external_bg
    )


    out_path = args.output or str(Path(args.input).with_name("cluster_category_scores.csv"))
    probs_df.to_csv(out_path, index=True)
    proto_path = str(Path(out_path).with_name("category_prototypes_q.csv"))
    q_df.to_csv(proto_path, index=True)

    if args.emit_word_posteriors:
        categories = list(q_df.index)
        if args.class_priors == "uniform":
            priors = np.full(len(categories), 1.0 / max(len(categories), 1), dtype=float)
        else:
            vocab_set = set(w.lower() for w in df.columns)
            counts = np.array([sum(1 for t in seeds[cat] if str(t).lower().strip() in vocab_set) or 1 for cat in categories], dtype=float)
            priors = counts / counts.sum()
        Q = q_df.to_numpy(dtype=float)
        num = (priors[:, None] * Q)
        denom = np.clip(num.sum(axis=0, keepdims=True), 1e-12, None)
        R = num / denom
        R_df = pd.DataFrame(R.T, index=df.columns, columns=categories)
        R_df.index.name = "word"
        (Path(out_path).with_name("word_category_posteriors.csv")).write_text(R_df.to_csv(index=True))

    print(f"Wrote: {out_path}")
    print(f"Wrote: {proto_path}")
    if args.emit_word_posteriors:
        print("Wrote:", Path(out_path).with_name("word_category_posteriors.csv"))

def _make_wc(custom_stop: set, *, collocations: bool, collocation_threshold: int) -> WordCloud:
    """
    Unified WordCloud config for both vocab and background counting.
    - collocations=True enables bigrams and adds them as 'new york' (with a space).
    - include_numbers=False (default) drops pure numbers.
    - regexp keeps only [a-z0-9] tokens; phrases are emitted by collocation logic, not underscores.
    """
    combined_stop = set(STOPWORDS) | set(custom_stop)
    return WordCloud(
        stopwords=combined_stop,
        collocations=collocations,
        collocation_threshold=collocation_threshold,
        background_color="white",
        regexp=r"[a-z0-9]+",
    )

def cmd_build_posteriors(args):
    # seeds, phrases, stopwords
    seeds = load_seeds(args.seeds)
    phrases = load_phrases(args.phrases)
    stopwords = load_stopwords(args.stopwords)

    # vocab
    if args.vocab:
        vocab = [clean_title(line) for line in Path(args.vocab).read_text(encoding="utf-8").splitlines() if line.strip()]
        vocab_path = Path(args.vocab)
    else:
        if not args.corpus:
            raise SystemExit("When --vocab is not provided, --corpus is required to build the vocabulary.")
        texts = read_corpus(args.corpus, args.text_col)
        vocab = build_vocab_from_corpus(
            texts, args.vocab_size, seeds, stopwords,
            collocations=args.collocations, collocation_threshold=args.collocation_threshold
        )
        vocab_path = Path(args.outdir) / "vocab.txt"
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        (vocab_path).write_text("\n".join(vocab) + "\n", encoding="utf-8")

    V = len(vocab)
    alpha = args.alpha if args.alpha is not None else (50.0 / max(V, 1))

    # background
    if args.bg_mode == "uniform":
        p_bg = np.full(V, 1.0 / max(V, 1), dtype=float)
        pd.DataFrame({"word": vocab, "weight": np.ones(V, dtype=float)}).to_csv(Path(args.outdir) / "background.csv", index=False)
    else:
        if not args.corpus:
            raise SystemExit("--bg-mode=corpus requires --corpus")
        texts = read_corpus(args.corpus, args.text_col)
        p_bg = background_from_corpus(
            texts, vocab, stopwords,
            collocations=args.collocations, collocation_threshold=args.collocation_threshold
        )
        pd.DataFrame({"word": vocab, "weight": p_bg}).to_csv(Path(args.outdir) / "background.csv", index=False)

    categories, Q = build_prototypes(vocab=vocab, seeds=seeds, p_bg=p_bg, alpha=alpha, bump=args.bump)
    q_df = pd.DataFrame(Q, index=categories, columns=vocab)
    q_df.to_csv(Path(args.outdir) / "category_prototypes_q.csv", index=True)

    priors = choose_priors(categories, args.priors, seeds, vocab, args.priors_file)
    pd.DataFrame({"category": categories, "prior": priors}).to_csv(Path(args.outdir) / "priors.csv", index=False)

    post_df = compute_word_posteriors(Q, categories, vocab, priors)
    post_df.to_csv(Path(args.outdir) / "word_category_posteriors.csv", index=True)

    manifest = {
        "pipeline": "cluster_category_scoring.build-posteriors",
        "vocab_size": V,
        "alpha": float(alpha),
        "bump": float(args.bump),
        "bg_mode": args.bg_mode,
        "priors_mode": args.priors,
        "seeds_file": args.seeds,
        "vocab_file": str(vocab_path) if args.vocab else str(vocab_path),
        "background_file": str(Path(args.outdir) / "background.csv"),
        "priors_file": str(Path(args.outdir) / "priors.csv"),
        "phrases_file": args.phrases,
        "stopwords_file": args.stopwords,
    }
    (Path(args.outdir) / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Wrote:", Path(args.outdir) / ("vocab.txt" if not args.vocab else Path(args.vocab).name))
    print("Wrote:", Path(args.outdir) / "background.csv")
    print("Wrote:", Path(args.outdir) / "category_prototypes_q.csv")
    print("Wrote:", Path(args.outdir) / "priors.csv")
    print("Wrote:", Path(args.outdir) / "word_category_posteriors.csv")
    print("Wrote:", Path(args.outdir) / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description="Endocrine category tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # score subcommand (previous default behavior)
    ps = sub.add_parser("score", help="Score clusters using log-likelihood with Dirichlet smoothing")
    ps.add_argument("-i", "--input", required=True, help="Path to cluster_word_matrix.csv")
    ps.add_argument("-o", "--output", default=None, help="Path to write cluster_category_scores.csv")
    ps.add_argument("-s", "--seeds", default=None, help="Optional JSON or YAML mapping of category -> [seeds]")
    ps.add_argument("--alpha", type=float, default=None, help="Dirichlet smoothing weight towards background")
    ps.add_argument("--tau", type=float, default=2.0, help="Temperature for softmax calibration")
    ps.add_argument("--bump", type=float, default=1.0, help="Prototype bump for seed tokens")
    ps.add_argument("--bg-mode", choices=["cluster-mean", "uniform", "external"], default="cluster-mean",
                    help="Background mode for P(w). 'uniform'/'external' are invariant to number of clusters.")
    ps.add_argument("--bg-file", default=None, help="CSV for --bg-mode=external (two columns word,weight or single-row).")
    ps.add_argument("--emit-word-posteriors", action="store_true", help="Also write word_category_posteriors.csv")
    ps.add_argument("--class-priors", choices=["uniform", "from-seeds"], default="uniform",
                    help="Priors used when emitting P(category|word).")

    # build-posteriors subcommand
    pb = sub.add_parser("build-posteriors", help="Build P(k|w) and prototypes from corpus or vocab")
    pb.add_argument("--corpus", help="Text corpus path (.txt or .csv/.tsv). If provided, can build vocab/background from it.")
    pb.add_argument("--text-col", default="title", help="Column for text in --corpus (default: title)")
    pb.add_argument("--vocab", help="Optional fixed vocab.txt (one token per line). If omitted, build from corpus.")
    pb.add_argument("--vocab-size", type=int, default=5000, help="Target vocab size when building from corpus (default: 5000).")
    pb.add_argument("--seeds", help="JSON or YAML mapping category -> [seed tokens]. If omitted, uses defaults.")
    pb.add_argument("--stopwords", help="Optional stopwords.txt (one word per line) or stopwords.pkl (set).")
    pb.add_argument("--phrases", help="(Deprecated when --collocations is set) Optional phrases.txt; kept for backward compatibility.")
    pb.add_argument("--collocations", action="store_true", help="Enable WordCloud bigram extraction (collocations=True).")
    pb.add_argument("--collocation-threshold", type=int, default=30, help="WordCloud collocation cutoff (default: 30).")
    pb.add_argument("--alpha", type=float, default=None, help="Dirichlet smoothing weight (default: 50/V).")
    pb.add_argument("--bump", type=float, default=1.0, help="Prototype bump for seed tokens (default: 1.0).")
    pb.add_argument("--bg-mode", choices=["uniform", "corpus"], default="corpus", help="Background mode for prototypes.")
    pb.add_argument("--priors", choices=["uniform", "from-seeds", "from-file"], default="uniform", help="Category priors mode.")
    pb.add_argument("--priors-file", help="If --priors=from-file, provide JSON mapping or CSV with [category, prior].")
    pb.add_argument("--outdir", required=True, help="Output directory for artifacts")

    args = parser.parse_args()
    if args.cmd == "score":
        return cmd_score(args)
    if args.cmd == "build-posteriors":
        return cmd_build_posteriors(args)


if __name__ == "__main__":
    main()
