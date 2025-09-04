#!/usr/bin/env python3
"""
Explain an already-trained XGBClassifier pipeline for log(RBA) high/low.

This script assumes your training pipeline has:
  - a column selector named "filter_mi" that exposes either
    .get_feature_names_out(original_columns) or .get_support(),
  - a classifier named "clf" that is an xgboost.XGBClassifier.

Typical usage:
  python explain_rba_model.py \
      --model path/to/pipeline.pkl \
      --data path/to/features.csv \
      --target_col y_binary \
      --all

Optional: run selected analyses, e.g.:
  --pi --shap_global --pdp TPSA cLogP Fsp3 --split_stats --threshold youden
"""

import argparse
import json
import os
import sys
import pickle
from typing import Dict, Iterable, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

import sys
sys.path.append("./")
from scripts.utils.helpers import process_dataset_endpoint

# --------------------------- Utilities ---------------------------

def _load_pipeline(path: str):
    """Load sklearn/xgboost pipeline via joblib or pickle."""
    import joblib  # noqa: F401
    return joblib.load(path)  # type: ignore[attr-defined]
    # try:
    
    # except Exception:
    #     with open(path, "rb") as f:
    #         return pickle.load(f)

def _ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

def _infer_feature_frame(df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) after dropping target and any extra columns."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    y = df[target_col]
    drop_set = set(drop_cols + [target_col])
    keep = [c for c in df.columns if c not in drop_set]
    if len(keep) == 0:
        raise ValueError("No feature columns remain after dropping target and extra columns.")
    return df[keep], y

def _get_kept_feature_names(selector, orig_cols: Iterable[str]) -> np.ndarray:
    """
    Robustly recover kept feature names from a selector.
    - Prefer get_feature_names_out.
    - Fallback to get_support on orig_cols.
    """
    if hasattr(selector, "get_feature_names_out"):
        return selector.get_feature_names_out(np.array(list(orig_cols)))
    if hasattr(selector, "get_support"):
        mask = selector.get_support()
        cols = np.array(list(orig_cols))
        if mask is None or len(mask) != len(cols):
            raise RuntimeError("Selector.get_support returned invalid mask.")
        return cols[mask]
    raise RuntimeError("Selector does not expose get_feature_names_out or get_support.")

# --------------------------- Interpretability helpers ---------------------------

def _transform_X_via_pipe(pipe, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X_trans, feature_names) after the 'filter_mi' step in your pipeline."""
    if "filter_mi" not in pipe.named_steps:
        raise KeyError("Pipeline is missing 'filter_mi' step.")
    selector = pipe.named_steps["filter_mi"]
    X_trans = selector.transform(X)
    feat_names = _get_kept_feature_names(selector, X.columns)
    return X_trans, feat_names

def permutation_importance_report(best_pipe, X: pd.DataFrame, y: pd.Series, n_repeats=20, random_state=0) -> pd.Series:
    """Permutation importances for the full pipeline (scoring=roc_auc)."""
    from sklearn.inspection import permutation_importance
    r = permutation_importance(
        best_pipe, X, y, scoring="roc_auc",
        n_repeats=n_repeats, random_state=random_state, n_jobs=-1,
    )
    selector = best_pipe.named_steps["filter_mi"]
    feat_names = _get_kept_feature_names(selector, X.columns)
    s = pd.Series(r.importances_mean, index=feat_names).sort_values(ascending=False)
    return s

def shap_global_local(best_pipe, X: pd.DataFrame, max_samples=3000):
    """Compute SHAP values for kept features using TreeExplainer on XGBClassifier."""
    try:
        import shap  # noqa: F401
    except Exception as e:
        raise RuntimeError("Missing dependency: shap") from e
    from xgboost import XGBClassifier  # type: ignore

    X_trans, feat_names = _transform_X_via_pipe(best_pipe, X)
    clf = best_pipe.named_steps["clf"]
    if not isinstance(clf, XGBClassifier):
        raise TypeError("Pipeline 'clf' is not XGBClassifier.")

    # Subsample for speed
    if X_trans.shape[0] > max_samples:
        rng = np.random.RandomState(0)
        idx = rng.choice(X_trans.shape[0], size=max_samples, replace=False)
        X_sub = X_trans[idx]
    else:
        X_sub = X_trans

    import shap
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sub)
    return shap_values, X_trans, feat_names, X_sub

def save_shap_summary_plots(shap_values, X_mat: np.ndarray, feat_names: np.ndarray, outdir: str, prefix: str = "shap"):
    """Save SHAP summary plots (bar and violin)."""
    import shap
    _ensure_outdir(outdir)
    # Bar
    plt.figure()
    shap.summary_plot(shap_values, X_mat, feature_names=feat_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_summary_bar.png"), dpi=200)
    plt.close()
    # Violin
    plt.figure()
    shap.summary_plot(shap_values, X_mat, feature_names=feat_names, plot_type="violin", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_summary_violin.png"), dpi=200)
    plt.close()

def partial_dependence_plots(best_pipe, X: pd.DataFrame, features: List[str], outdir: str):
    """Compute and save PDP line plots for selected features."""
    from sklearn.inspection import partial_dependence
    _ensure_outdir(outdir)
    X_trans, feat_names = _transform_X_via_pipe(best_pipe, X)
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    for f in features:
        if f not in name_to_idx:
            print(f"[WARN] Feature '{f}' not in kept feature set; skipping.", file=sys.stderr)
            continue
        idx = name_to_idx[f]
        pd_res = partial_dependence(best_pipe.named_steps["clf"], X_trans, features=[idx], kind="average")
        grid = pd_res["grid_values"][0]
        pdv = pd_res["average"][0]
        plt.figure()
        plt.plot(grid, pdv)
        plt.xlabel(f)
        plt.ylabel("Partial dependence (model output)")
        plt.title(f"PDP: {f}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"pdp_{f}.png"), dpi=200)
        plt.close()

def summarize_split_thresholds(best_pipe, X: pd.DataFrame = None) -> pd.DataFrame:
    """Aggregate split thresholds across all boosted trees."""
    booster = best_pipe.named_steps["clf"].get_booster()
    df = booster.trees_to_dataframe()  # columns: Feature, Split, Gain, Cover, Tree, Node, Yes, No, Missing, etc.
    df = df.dropna(subset=["Feature", "Split"])

    # Get kept feature names from pipeline
    if X is not None:
        selector = best_pipe.named_steps["filter_mi"]
        kept_feature_names = _get_kept_feature_names(selector, X.columns)
        feature_map = {f"f{i}": name for i, name in enumerate(kept_feature_names)}
        df["Feature"] = df["Feature"].map(feature_map).fillna(df["Feature"])

    summary = (
        df.groupby("Feature")["Split"]
        .agg(["count", "median", "mean"])
        .sort_values("count", ascending=False)
    )
    modes = (
        df.groupby("Feature")["Split"]
        .apply(lambda s: s.round(3).mode().iloc[0] if not s.empty else np.nan)
        .rename("mode_split")
    )
    out = summary.join(modes)
    return out

def shap_pairwise_interactions(best_pipe, X: pd.DataFrame, feat_pair: Tuple[str, str], max_samples=3000) -> np.ndarray:
    """Compute SHAP interaction values for a pair of kept features."""
    try:
        import shap  # noqa: F401
    except Exception as e:
        raise RuntimeError("Missing dependency: shap") from e
    X_trans, feat_names = _transform_X_via_pipe(best_pipe, X)
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    if feat_pair[0] not in name_to_idx or feat_pair[1] not in name_to_idx:
        raise ValueError("Requested features not in kept set.")
    i, j = name_to_idx[feat_pair[0]], name_to_idx[feat_pair[1]]

    if X_trans.shape[0] > max_samples:
        rng = np.random.RandomState(0)
        idx = rng.choice(X_trans.shape[0], size=max_samples, replace=False)
        X_sub = X_trans[idx]
    else:
        X_sub = X_trans

    import shap
    explainer = shap.TreeExplainer(best_pipe.named_steps["clf"])
    inter = explainer.shap_interaction_values(X_sub)  # shape: [n, d, d]
    return inter[:, i, j]

def optimize_probability_threshold(best_pipe, X: pd.DataFrame, y: pd.Series, criterion="youden") -> Tuple[float, Dict[str, int]]:
    """Pick a probability threshold by Youden's J or F1."""
    from sklearn.metrics import roc_curve, f1_score, confusion_matrix
    proba = best_pipe.predict_proba(X)[:, 1]
    if criterion == "youden":
        fpr, tpr, thr = roc_curve(y, proba)
        j = tpr - fpr
        k = int(np.argmax(j))
        threshold = float(thr[k])
    elif criterion == "f1":
        grid = np.linspace(0.05, 0.95, 181)
        scores = [f1_score(y, (proba >= t).astype(int)) for t in grid]
        threshold = float(grid[int(np.argmax(scores))])
    else:
        raise ValueError("criterion must be 'youden' or 'f1'.")

    yhat = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    report = dict(threshold=threshold, TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))
    return threshold, report

def visualize_one_tree(best_pipe, tree_index=0, outdir: str = ".", dpi=900, figsize=(12, 8)):
    """Save a matplotlib rendering of one boosted tree."""
    from xgboost import plot_tree  # type: ignore
    _ensure_outdir(outdir)
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(best_pipe.named_steps["clf"], tree_idx=tree_index, ax=ax)
    fig.tight_layout()
    path = os.path.join(outdir, f"tree_{tree_index}.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path

def networkx_graph_for_tree(best_pipe, tree_index=0):
    """Optional: build a NetworkX DiGraph of one tree for custom analytics."""
    try:
        import networkx as nx  # noqa: F401
    except Exception as e:
        raise RuntimeError("Missing dependency: networkx") from e
    booster = best_pipe.named_steps["clf"].get_booster()
    df = booster.trees_to_dataframe()
    df_tree = df[df["Tree"] == tree_index].copy()
    G = nx.DiGraph()
    for _, row in df_tree.iterrows():
        node_id = int(row["Node"])
        label = row["Feature"] if isinstance(row["Feature"], str) else "Leaf"
        split = None if pd.isna(row["Split"]) else float(row["Split"])
        G.add_node(node_id, label=label, split=split, gain=row.get("Gain", None))
    for _, row in df_tree.dropna(subset=["Yes", "No"]).iterrows():
        parent = int(row["Node"])
        G.add_edge(parent, int(row["Yes"]), decision="yes")
        G.add_edge(parent, int(row["No"]), decision="no")
    return G

def train_surrogate_rules(best_pipe, X: pd.DataFrame, max_depth=3):
    """Fit a shallow decision tree on XGB probabilities and return (estimator, text rules)."""
    from sklearn.tree import DecisionTreeClassifier, export_text
    proba = best_pipe.predict_proba(X)[:, 1]
    y_star = (proba >= 0.5).astype(int)
    X_trans, feat_names = _transform_X_via_pipe(best_pipe, X)
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    dt.fit(X_trans, y_star)
    rules = export_text(dt, feature_names=list(feat_names))
    return dt, rules

# --------------------------- CLI ---------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    cachedir = Path("cache/combined")
    p = argparse.ArgumentParser(description="Interpret a trained XGBClassifier for log(RBA) high/low.")
    p.add_argument("--model", default=cachedir / "xgb_classifier_logRBA_model_descriptors.joblib", help="Path to pipeline.")
    p.add_argument("--data", default=cachedir / "descriptor_matrix.parquet", type=Path, help="Path to CSV/Parquet with features and target.")
    # p.add_argument("--target_col", default="y", help="Target column name in --data.")
    # p.add_argument("--drop_cols", nargs="*", default=[], help="Extra columns to drop (IDs, metadata).")
    p.add_argument("--outdir", default=cachedir / "explain_rba_out", help="Output directory for plots and tables.")

    # Toggles
    p.add_argument("--all", action="store_true", help="Run all analyses that require available inputs.")
    p.add_argument("--pi", action="store_true", help="Permutation importance.")
    p.add_argument("--shap_global", action="store_true", help="SHAP global summary plots.")
    p.add_argument("--pdp", nargs="*", default=None, metavar="FEAT", help="Features for PDP plots (e.g., TPSA cLogP Fsp3).")
    p.add_argument("--split_stats", action="store_true", help="Aggregate split thresholds across trees.")
    p.add_argument("--interactions", nargs=2, default=None, metavar=("FEAT_A", "FEAT_B"), help="SHAP interaction for a pair.")
    p.add_argument("--threshold", choices=["youden", "f1"], default=None, help="Optimize probability threshold by given criterion.")
    p.add_argument("--tree_plot", nargs="*", type=int, default=None, metavar="IDX", help="Save plot(s) for given tree indices.")
    p.add_argument("--surrogate", action="store_true", help="Train shallow surrogate tree and export rules.")

    # Params
    p.add_argument("--n_repeats", type=int, default=20, help="Permutation importance repeats.")
    p.add_argument("--max_samples", type=int, default=3000, help="Max samples for SHAP computations.")
    p.add_argument("--topk", type=int, default=20, help="Top-k to print for some summaries.")
    p.add_argument("--surrogate_depth", type=int, default=3, help="Depth for surrogate rules tree.")

    return p

def main():
    args = build_arg_parser().parse_args()
    _ensure_outdir(args.outdir)

    pipe = _load_pipeline(args.model)
    # # Create a new XGBClassifier instance
    # pipe = XGBClassifier()
    # # Load the model from the JSON file
    # pipe.load_model(args.model)

    # Determine which tasks need data:
    needs_Xy = (
        args.all or args.pi or args.shap_global or args.pdp is not None
        or args.interactions is not None or args.threshold is not None
        or args.surrogate
    )

    # Load data if needed
    if needs_Xy:
        if args.data is None:
            print("[ERROR] --data is required for the selected analyses.", file=sys.stderr)
            sys.exit(2)
        # if args.data.lower().endswith(".csv"):
        if args.data.suffix == (".csv"):
            df = pd.read_csv(args.data)
        # elif args.data.lower().endswith(".parquet"):
        elif args.data.suffix == (".parquet"):
            df = pd.read_parquet(args.data)
        else:
            print("[ERROR] --data must be .csv or .parquet.", file=sys.stderr)
            sys.exit(2)
        # X, y = _infer_feature_frame(df, args.target_col, args.drop_cols)
        dataset_parquet = "resources/combined_full.parquet"
        dataset = pd.read_parquet(dataset_parquet)
        
        vals, _, threshold = process_dataset_endpoint(dataset, "logRBA")
        
        index_intersection = df.index.intersection(vals.index)
        X = df.loc[index_intersection]
        y = (vals.loc[index_intersection] > threshold).squeeze() # binary target based on threshold
    else:
        X = None
        y = None


    # ----- Run tasks -----

    # Split stats do not require data matrix
    if args.all or args.split_stats:
        try:
            split_df = summarize_split_thresholds(pipe, X)
            split_path = os.path.join(args.outdir, "split_thresholds.csv")
            split_df.to_csv(split_path)
            print(f"[OK] Split thresholds saved -> {split_path}")
            # Print top-k by count
            print(split_df.head(args.topk))
        except Exception as e:
            print(f"[WARN] split-stats failed: {e}", file=sys.stderr)

    if needs_Xy:
        # Permutation importance
        if args.all or args.pi:
            try:
                pi = permutation_importance_report(pipe, X, y, n_repeats=args.n_repeats)
                pi_path = os.path.join(args.outdir, "permutation_importance.csv")
                pi.to_csv(pi_path, header=["importance_mean"])
                print(f"[OK] Permutation importance saved -> {pi_path}")
                print(pi.head(args.topk))
            except Exception as e:
                print(f"[WARN] permutation importance failed: {e}", file=sys.stderr)

        # SHAP global
        if args.all or args.shap_global:
            try:
                shap_values, X_trans, feat_names, X_sub = shap_global_local(pipe, X, max_samples=args.max_samples)
                save_shap_summary_plots(shap_values, X_sub, feat_names, args.outdir, prefix="shap")
                print(f"[OK] SHAP global plots saved under {args.outdir}")
            except Exception as e:
                print(f"[WARN] SHAP global failed: {e}", file=sys.stderr)

        # PDP
        if args.all or args.pdp is not None:
            feats = args.pdp if args.pdp is not None else ["TPSA", "cLogP", "Fsp3", "BranchingRatio"]
            try:
                partial_dependence_plots(pipe, X, feats, args.outdir)
                print(f"[OK] PDP plots saved for: {', '.join(feats)}")
            except Exception as e:
                print(f"[WARN] PDP failed: {e}", file=sys.stderr)

        # Interactions
        if args.all or args.interactions is not None:
            pair = args.interactions if args.interactions is not None else ("TPSA", "cLogP")
            try:
                inter_vals = shap_pairwise_interactions(pipe, X, tuple(pair), max_samples=args.max_samples)
                # Save basic stats and a histogram
                stats = {
                    "feature_a": pair[0],
                    "feature_b": pair[1],
                    "mean": float(np.mean(inter_vals)),
                    "std": float(np.std(inter_vals)),
                    "p05": float(np.percentile(inter_vals, 5)),
                    "p50": float(np.percentile(inter_vals, 50)),
                    "p95": float(np.percentile(inter_vals, 95)),
                    "n": int(inter_vals.shape[0]),
                }
                with open(os.path.join(args.outdir, "interaction_stats.json"), "w") as f:
                    json.dump(stats, f, indent=2)
                plt.figure()
                plt.hist(inter_vals, bins=40)
                plt.xlabel(f"SHAP interaction: {pair[0]} x {pair[1]}")
                plt.ylabel("Count")
                plt.title("Interaction distribution")
                plt.tight_layout()
                plt.savefig(os.path.join(args.outdir, "interaction_hist.png"), dpi=200)
                plt.close()
                print(f"[OK] Interaction stats and histogram saved for {pair[0]} x {pair[1]}")
                print(stats)
            except Exception as e:
                print(f"[WARN] interactions failed: {e}", file=sys.stderr)

        # Threshold optimization
        if args.all or args.threshold is not None:
            criterion = args.threshold if args.threshold is not None else "youden"
            try:
                thr, rep = optimize_probability_threshold(pipe, X, y, criterion=criterion)
                with open(os.path.join(args.outdir, "threshold_report.json"), "w") as f:
                    json.dump(rep, f, indent=2)
                print(f"[OK] Threshold ({criterion}) = {thr:.4f}")
                print(rep)
            except Exception as e:
                print(f"[WARN] threshold optimization failed: {e}", file=sys.stderr)

        # Surrogate rules
        if args.all or args.surrogate:
            try:
                dt, rules = train_surrogate_rules(pipe, X, max_depth=args.surrogate_depth)
                txt_path = os.path.join(args.outdir, "surrogate_rules.txt")
                with open(txt_path, "w") as f:
                    f.write(rules)
                print(f"[OK] Surrogate rules saved -> {txt_path}")
                print(rules)
            except Exception as e:
                print(f"[WARN] surrogate rules failed: {e}", file=sys.stderr)

    # Tree plots (no data required)
    if args.all or (args.tree_plot is not None and len(args.tree_plot) > 0):
        indices = args.tree_plot if args.tree_plot is not None else [0]
        for idx in indices:
            try:
                path = visualize_one_tree(pipe, tree_index=idx, outdir=args.outdir)
                print(f"[OK] Tree {idx} plot saved -> {path}")
            except Exception as e:
                print(f"[WARN] tree-plot {idx} failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
