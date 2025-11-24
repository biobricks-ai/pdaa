#!/usr/bin/env python3
# xgb_feature_selection_fast.py
# High-impact runtime improvements for XGBoost training with CV/HPO.

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple
import json

import numpy as np
import pandas as pd

from joblib import Memory

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

try:
    # Prefer the official wrapper; will be used inside our wrapper class.
    from xgboost import XGBClassifier
except Exception as e:
    raise RuntimeError("xgboost must be installed to run this script.") from e

# Optional imports from your existing codebase. If present, we will use them.
# This allows the script to slot into your current project layout without duplicating logic.
try:
    import model_dataset as md  # expects /mnt/data/model_dataset.py or module on PYTHONPATH
    HAS_MODEL_DATASET = True
except Exception:
    HAS_MODEL_DATASET = False


class EarlyStoppingXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    Thin sklearn-compatible wrapper that injects a small validation split on each .fit()
    so that XGBoost's early stopping can terminate boosting rounds early.

    Design goals:
      - Works inside sklearn Pipeline and search CV.
      - Exposes XGB params with get_params/set_params for HPO.
      - Keeps n_estimators high; relies on early_stopping_rounds to stop earlier.
      - Single-threaded by default to avoid nested parallelism; let the outer CV parallelize.

    Notes:
      - We intentionally do a lightweight internal StratifiedKFold split of the incoming training fold.
      - Validation size is controlled by 'validation_size' (default 0.15).
      - If 'early_stopping_rounds' is None, acts like a normal XGBClassifier.
    """

    def __init__(
        self,
        *,
        early_stopping_rounds: int = 50,
        validation_size: float = 0.15,
        random_state: int = 42,
        n_jobs: int = 1,
        tree_method: str = "hist",
        **xgb_kwargs,
    ):
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_size = validation_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tree_method = tree_method
        self.xgb_kwargs = xgb_kwargs

        # Construct the underlying classifier lazily in fit() to survive cloning during HPO.
        self._clf = None

    def get_params(self, deep: bool = True):
        params = {
            "early_stopping_rounds": self.early_stopping_rounds,
            "validation_size": self.validation_size,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "tree_method": self.tree_method,
        }
        # Expose nested XGB params at top level for sklearn search.
        for k, v in self.xgb_kwargs.items():
            params[k] = v
        return params

    def set_params(self, **params):
        known = {"early_stopping_rounds", "validation_size", "random_state", "n_jobs", "tree_method"}
        xgb_updates = {}
        for k, v in params.items():
            if k in known:
                setattr(self, k, v)
            else:
                xgb_updates[k] = v
        # mutate existing dict
        self.xgb_kwargs.update(xgb_updates)
        return self

    def fit(
        self,
        X,
        y,
        sample_weight=None,
    ):
        # Build the inner XGB with single-threading to prevent nested oversubscription.
        # eval_metric aligned with ROC-AUC scoring, but XGB expects a metric; we choose 'auc'.
        base = XGBClassifier(
            tree_method=self.tree_method,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            eval_metric="auc",
            early_stopping_rounds=self.early_stopping_rounds,  # set on estimator instead of fit
            **self.xgb_kwargs,
        )

        if self.early_stopping_rounds is None or self.early_stopping_rounds <= 0:
            self._clf = base.fit(
                X,
                y,
                sample_weight=sample_weight,
            )
            return self

        # Create a small validation split deterministically for each fit call.
        # We use one split for speed; early stopping uses this as the evaluation set.
        skf = StratifiedKFold(
            n_splits=max(5, int(1.0 / max(1e-6, self.validation_size))),  # guard rails
            shuffle=True,
            random_state=self.random_state,
        )
        try:
            # Take the first split only (one eval split per fit).
            train_idx, valid_idx = next(iter(skf.split(X, y)))
            X_train, y_train = X[train_idx], y[train_idx]
            X_valid, y_valid = X[valid_idx], y[valid_idx]

            self._clf = base.fit(
                X_train,
                y_train,
                sample_weight=None if sample_weight is None else np.asarray(sample_weight)[train_idx],
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )
        except (ValueError, StopIteration):
            # Too few samples per class even for 2-fold split; disable early stopping for this fit.
            base.set_params(early_stopping_rounds=None)
            self._clf = base.fit(
                X,
                y,
                sample_weight=None if sample_weight is None else np.asarray(sample_weight),
                verbose=False,
            )

        # Forward learned attributes required by sklearn’s scoring & introspection.
        # These are present on the inner estimator after fit().
        if hasattr(self._clf, "classes_"):
            self.classes_ = self._clf.classes_
        if hasattr(self._clf, "n_features_in_"):
            self.n_features_in_ = self._clf.n_features_in_
        if hasattr(self._clf, "feature_names_in_"):
            self.feature_names_in_ = self._clf.feature_names_in_

        return self


    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

    def decision_function(self, X):
        # XGBClassifier supports predict_proba; let sklearn consume that for scoring.
        proba = self.predict_proba(X)
        # Return the positive-class score for compatibility with some scorers.
        return proba[:, 1]


def build_pipeline(
    *,
    k: Optional[int | str],
    # use_gpu: bool,
    random_state: int,
    cache_dir: Path,
) -> Tuple[Pipeline, dict]:
    """
    Build the sklearn Pipeline with optional MI feature selection and an XGB classifier
    wrapped for early stopping. Returns the pipeline and a lean HPO search space.
    """
    memory = Memory(location=str(cache_dir), verbose=0)

    steps = []
    if k != "all":
        # Only compute MI when requested; computing MI for k='all' is a waste.
        steps.append(
            (
                "filter_mi",
                SelectKBest(
                    score_func=mutual_info_classif,
                    k=k,
                ),
            )
        )

    # tree_method = "gpu_hist" if use_gpu else "hist"
    # Underlying XGB runs single-threaded; outer CV controls parallelism.
    clf = EarlyStoppingXGBClassifier(
        # tree_method=tree_method,
        tree_method="hist",
        n_jobs=1,
        random_state=random_state,
        # Defaults suitable for early stopping; n_estimators is searched via successive halving.
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        early_stopping_rounds=50,
        validation_size=0.15,
    )

    steps.append(("clf", clf))
    pipe = Pipeline(
        steps,
        memory=memory,
    )

    # Lean, high-yield search space. Early stopping + SH will prune quickly.
    param_dist = {
        # "clf__n_estimators": [300, 600, 1000],
        "clf__max_depth": [3, 4, 5],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
        # Regularization kept minimal unless needed; can widen later.
        "clf__reg_lambda": [1.0, 2.0],
        "clf__reg_alpha": [0.0, 0.5],
    }
    # # If user pins max_depth via CLI, restrict search to that value.
    # if args.max_depth is not None:
    #     param_dist["clf__max_depth"] = [args.max_depth]

    return pipe, param_dist


def run_one_endpoint(
    *,
    endpoint: str,
    matrix_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    k: int | str,
    n_jobs_outer: int,
    n_iter: int,
    # use_gpu: bool,
    random_state: int,
    outdir: Path,
    use_descriptors: bool = False,
    build_matrix: bool = False,
    transform: Optional[str] = None,
    max_depth: Optional[int] = None,
) -> None:
    """
    Train and evaluate models for a single endpoint using nested CV with
    HalvingRandomSearchCV for HPO and early stopping inside the estimator.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = outdir / "sk_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Match old script: load matrix parquet from cache, or build and save.
    matrix_name = "descriptor matrix" if use_descriptors else "activity matrix"
    suffix = "" if use_descriptors else "_filled"
    matrix_parquet = outdir / f'{matrix_name.replace(" ", "_")}{suffix}.parquet'

    if matrix_parquet.exists() and not build_matrix:
        print(f"Loading cached {matrix_name} from {matrix_parquet}...")
        matrix_df = pd.read_parquet(matrix_parquet)
    else:
        print(f"Building {matrix_name} from scratch...")
        if use_descriptors:
            if not (HAS_MODEL_DATASET and hasattr(md, "get_descriptor_df")):
                raise RuntimeError("get_descriptor_df not found in model_dataset.py")
            inchis = dataset_df.inchi.dropna().unique()
            matrix_df = md.get_descriptor_df(
                inchis,
                use_phthalate_set=False,
                vif_threshold=10.0,
            )

        else:
            if not (HAS_MODEL_DATASET and hasattr(md, "build_activity_matrix_filled_from_inchis")):
                raise RuntimeError("build_activity_matrix_filled_from_inchis not found (old script expects it).")
            inchis = dataset_df.inchi.dropna().unique()
            matrix_df = md.build_activity_matrix_filled_from_inchis(inchis)

        matrix_df.to_parquet(matrix_parquet)

    if "inchi" in getattr(matrix_df, "columns", []):
        matrix_df = matrix_df.set_index("inchi")


    # Optional transform to match prior CLI
    if HAS_MODEL_DATASET and hasattr(md, "transform_X") and transform:
        matrix_df = md.transform_X(matrix_df, transform)

    vals, adj_endpoint, threshold = md.process_dataset_endpoint(
        dataset_df,
        endpoint,
    )

    # Match original behavior: use the cleaned index from process_dataset_endpoint.
    if isinstance(vals, pd.Series):
        vals_s = vals
    elif isinstance(vals, pd.DataFrame):
        # Single-column DataFrame -> Series
        if vals.shape[1] != 1:
            raise ValueError("Expected a single-column DataFrame from process_dataset_endpoint.")
        vals_s = vals.iloc[:, 0]
    else:
        raise ValueError("process_dataset_endpoint must return a Series or single-column DataFrame.")


    # Final alignment exactly like the old script: index intersection
    index_intersection = matrix_df.index.intersection(vals_s.index)

    X_trans = matrix_df.loc[index_intersection].to_numpy()                              # 2-D
    y_binary = (vals_s.loc[index_intersection].to_numpy().ravel() > float(threshold)).astype(int)  # 1-D





    # Build pipeline and search space.
    pipe, param_dist = build_pipeline(
        k=k,
        # use_gpu=use_gpu,
        random_state=random_state,
        cache_dir=cache_dir,
    )

    if max_depth is not None:
        param_dist["clf__max_depth"] = [max_depth]

    # Respect --max_depth if provided (old CLI exposed this)
    if hasattr(argparse, "_") or True:  # sentinel to allow patch-in without refactor
        # We’re inside run_one_endpoint, so we don’t have args; pass max_depth down from main if you like.
        # Minimal change: infer from existing 'param_dist' if a single value was intended externally.
        pass

    # Inner HPO: HalvingRandomSearchCV over n_estimators as the resource.
    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    search = HalvingRandomSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        factor=3,
        resource="clf__n_estimators",
        max_resources=1000,
        min_resources=100,
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=1,  # keep inner single-threaded; avoid nested parallelism
        random_state=random_state,
        verbose=1,
        refit=True,
    )

    # Outer CV: parallelize only here.
    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=7 * random_state + 1,
    )
    try:
        cv_results = cross_validate(
            estimator=search,
            X=X_trans,
            y=y_binary,
            cv=outer_cv,
            scoring=["accuracy", "roc_auc"],
            return_estimator=True,
            n_jobs=n_jobs_outer,  # single parallel layer
            verbose=1,
        )
    except ValueError as e:
        print(f"Skipping endpoint '{endpoint}' due to CV error: {e}")
        return  # bypass training/reporting for this endpoint

    # Persist a concise report.
    report = {
        "endpoint": endpoint,
        "k": k,
        # "use_gpu": use_gpu,
        "scores": {
            "accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
            "accuracy_std": float(np.std(cv_results["test_accuracy"])),
            "roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
            "roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
        },
        "best_params_per_fold": [],
    }

    for est in cv_results["estimator"]:
        # Each outer fold refit HalvingRandomSearchCV; capture its best params.
        if hasattr(est, "best_params_"):
            report["best_params_per_fold"].append(est.best_params_)


    out_path = outdir / f"xgb_classifier_feature_selection_fast_{endpoint}.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report) + "\n")


    # Also write a simple text summary for human inspection.
    txt_path = outdir / f"xgb_classifier_feature_selection_fast_{endpoint}.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Endpoint: {endpoint}\n")
        f.write(f"k: {k}\n")
        # f.write(f"use_gpu: {use_gpu}\n")
        f.write(
            "Accuracy (mean ± std): "
            f"{report['scores']['accuracy_mean']:.4f} ± {report['scores']['accuracy_std']:.4f}\n"
        )
        f.write(
            "ROC AUC (mean ± std): "
            f"{report['scores']['roc_auc_mean']:.4f} ± {report['scores']['roc_auc_std']:.4f}\n"
        )
        f.write("Best params per fold:\n")
        for i, bp in enumerate(report["best_params_per_fold"], 1):
            f.write(f"  Fold {i}: {bp}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run dataset characterization."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["eadb", "ekdb", "combined"],
        help="Name of the database.",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="",
        choices=["", "z_scale", "binary"],
        help="Type of transformation to apply to the feature matrix.",
    )
    parser.add_argument(
        "--k",
        type=str,
        default=None,
        help="Number of top features to keep for feature selection (or 'all').",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Maximum tree depth to force during HPO; if provided, search is restricted to this value.",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=30,
        help="Number of iterations for hyperparameter search (upper bound; SH may evaluate fewer).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Specific EADB endpoint to process (if provided).",
    )
    parser.add_argument(
        "--use_descriptors",
        action="store_true",
        help="Use chemical descriptors instead of activity matrix.",
    )
    parser.add_argument(
        "--build_matrix",
        action="store_true",
        help="Build the activity or descriptor matrix from scratch instead of using cached version.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for the OUTER CV (−1 uses all cores).",
    )
    args = parser.parse_args()


    # outdir = args.outdir

    cachedir = Path('cache')
    outdir = cachedir / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)

    # Load dataset parquet based on --dataset; matrix is built inside run_one_endpoint.
    resourcedir = Path("resources")
    parquet_map = {
        "eadb":     resourcedir / "eadb_full.parquet",
        "ekdb":     resourcedir / "ekdb_full.parquet",
        "combined": resourcedir / "combined_full.bak.0.parquet",
        # "combined": resourcedir / "combined_full.parquet",
    }
    dataset_parquet = parquet_map[args.dataset]
    dataset_df = pd.read_parquet(dataset_parquet)
    endpoints = list(pd.unique(dataset_df["EndpointName"]))

    # Placeholder matrix_df; actual matrix construction happens per-endpoint.
    matrix_df = pd.DataFrame(index=dataset_df.index)


    if args.endpoint is not None:
        endpoints = [args.endpoint]

    # Preserve prior script behavior for k
    if (args.k is None) or (args.k == "None") or (args.k == "all"):
        k_value = "all"
    else:
        k_value = int(args.k)


    # Run endpoints sequentially to avoid overloading the machine; outer CV already parallelizes.
    for ep in endpoints:
        run_one_endpoint(
            endpoint=ep,
            matrix_df=matrix_df,
            dataset_df=dataset_df,
            k=k_value,
            n_jobs_outer=args.n_jobs if args.n_jobs != 0 else 1,
            n_iter=args.n_iter,
            # use_gpu=args.use_gpu,
            random_state=args.random_state,
            outdir=outdir,
            use_descriptors=args.use_descriptors,
            build_matrix=args.build_matrix,
            transform=args.transform if args.transform != "" else None,
            max_depth=args.max_depth,
        )


if __name__ == "__main__":
    main()
