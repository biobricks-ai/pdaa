#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrich assay_fuzzy_map.csv with additional similarity metrics and flags.

Input columns required:
- rhs_title
- lhs_best_match
- score   # from your initial fuzzy pass

Output:
- assay_fuzzy_map_enriched.csv  # same dir as input
- Optional: per-flag CSVs if you uncomment the saves at bottom
"""

import argparse
import pandas as pd
import re
from pathlib import Path

# Prefer RapidFuzz; fallback to difflib
try:
    from rapidfuzz import fuzz as rf_fuzz
    HAVE_RF = True
except Exception:
    import difflib
    HAVE_RF = False

# Domain stop-words that inflate scores but carry little discrimination
DOMAIN_STOP = {
    "assay","screen","screening","activity","binding","viability","toxicity","cell","cells",
    "inhibition","inhibitor","agonist","antagonist","receptor","human","mouse","rat","reporter",
    "activation","response","signal","pathway","transcription","luciferase","bla","hla","beta",
    "alpha","gamma","kappa","delta","sigma","mu","upregulation","downregulation","induction",
    "factor","nuclear","hormone","dependent","independent","modulation","evaluation","measurement",
    "test","analysis","detection","profiling","target","targets","gene","protein","kinase","channel"
}

WORD_RE = re.compile(r"[a-z0-9]+")

def tokenize(s: str) -> list[str]:
    return WORD_RE.findall(str(s).lower())

def tokens_wo_stop(s: str) -> list[str]:
    return [t for t in tokenize(s) if t not in DOMAIN_STOP]

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def coverage(a: set, b: set) -> float:
    # fraction of a covered by b
    return (len(a & b) / len(a)) if a else 0.0

def compute_row_metrics(rhs: str, lhs: str) -> dict:
    ta, tb = set(tokens_wo_stop(lhs)), set(tokens_wo_stop(rhs))
    jac = jaccard(ta, tb)
    cov_rhs = coverage(tb, ta)
    cov_lhs = coverage(ta, tb)

    if HAVE_RF:
        tsr = int(rf_fuzz.token_set_ratio(rhs, lhs))
        tor = int(rf_fuzz.token_sort_ratio(rhs, lhs))
        wr  = int(rf_fuzz.WRatio(rhs, lhs))
        pr  = int(rf_fuzz.partial_ratio(rhs, lhs))
        base = int(rf_fuzz.ratio(rhs, lhs))
    else:
        # difflib approximations if RapidFuzz missing
        base_ratio = difflib.SequenceMatcher(None, str(rhs), str(lhs)).ratio()
        base = int(round(base_ratio * 100))
        tsr = base
        tor = base
        wr  = base
        pr  = base

    return {
        "token_set_ratio": tsr,
        "token_sort_ratio": tor,
        "wratio": wr,
        "partial_ratio": pr,
        "char_ratio": base,
        "jaccard_wo_stop": jac,
        "cov_rhs_wo_stop": cov_rhs,
        "cov_lhs_wo_stop": cov_lhs,
        "n_tokens_rhs_wo_stop": len(set(tokens_wo_stop(rhs))),
        "n_tokens_lhs_wo_stop": len(set(tokens_wo_stop(lhs))),
    }

def print_indented(df: pd.DataFrame, indent_level: int = 1):
    """
    Print DataFrame with indentation for better readability.
    """
    indent = '\t' * indent_level
    print_str = df.to_string(index=False, header=True)
    print_lines = print_str.split('\n')
    for line in print_lines:
        print(indent + line)
    # for _, row in df.iterrows():
    #     print(indent + str(row.to_dict()))

def enrich_csv(in_path: Path) -> Path:
    df = pd.read_csv(in_path)
    required = {"rhs_title", "lhs_best_match", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)} in {in_path}")

    metrics = df.apply(
        lambda r: compute_row_metrics(r["rhs_title"], r["lhs_best_match"]),
        axis=1, result_type="expand"
    )

    out = pd.concat([df, metrics], axis=1)

    # Flags. Adjust thresholds as needed.
    out["flag_strong_candidate"] = (
        (out["token_set_ratio"] >= 90)
        & (out["jaccard_wo_stop"] >= 0.60)
        & (out["cov_rhs_wo_stop"] >= 0.70)
    )
    out["flag_reordering"] = (
        (out["token_set_ratio"] >= 85)
        & (out["jaccard_wo_stop"] >= 0.60)
        & ((out["token_set_ratio"] - out["token_sort_ratio"]) >= 20)
    )
    out["flag_overinflated"] = (
        (out["token_set_ratio"] >= 90)
        & (out["jaccard_wo_stop"] < 0.45)
        & (out["cov_rhs_wo_stop"] < 0.60)
    )

    out_path = in_path.with_name("assay_fuzzy_map_enriched.csv")
    out.to_csv(out_path, index=False)

    # Optional: write subsets for quick triage. Uncomment if desired.
    # out[out["flag_strong_candidate"]].to_csv(in_path.with_name("assay_fuzzy_strong.csv"), index=False)
    # out[out["flag_reordering"]].to_csv(in_path.with_name("assay_fuzzy_reordering.csv"), index=False)
    # out[out["flag_overinflated"]].to_csv(in_path.with_name("assay_fuzzy_overinflated.csv"), index=False)

    # Quick summary print
    print({
        "rows": len(out),
        "strong_candidates": int(out["flag_strong_candidate"].sum()),
        "reordering_candidates": int(out["flag_reordering"].sum()),
        "overinflated_suspects": int(out["flag_overinflated"].sum()),
        "score>=90": int((out["token_set_ratio"] >= 90).sum()),
        "score70_89": int(((out["token_set_ratio"] >= 70) & (out["token_set_ratio"] < 90)).sum()),
        "score<70": int((out["token_set_ratio"] < 70).sum()),
    })
    print(f"Wrote: {out_path}")

    print("Strong candidates:")
    print_indented(out[out["flag_strong_candidate"]][["rhs_title", "lhs_best_match"]])

    print("\n\nOverinflated suspects:")
    print_indented(out[out["flag_overinflated"]][["rhs_title", "lhs_best_match"]])
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("assay_fuzzy_map.csv"),
        help="Path to assay_fuzzy_map.csv",
    )
    args = ap.parse_args()
    enrich_csv(args.input)

if __name__ == "__main__":
    main()
