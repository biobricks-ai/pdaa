"""
Two-panel line plot showing Adjusted MAV 1 at different DART/ED strength
cutoffs (>= 6, 7, 8, 9).

Phthalates ordered by descending z-scored Current MAV.

Usage:
    python scripts/plot_adj1_cutoff_sweep.py
"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
ACTIVITY_PATH = PROJECT_ROOT / "cache" / "entity_similarity" / "activity_matrix_filled.parquet"
PREDICT_JSON = PROJECT_ROOT / "resources" / "predict_all.json"
ASSAY_STRENGTH = PROJECT_ROOT / "resources" / "assay_strength.csv"
COMMERCIAL_CSV = PROJECT_ROOT / "resources" / "phthalate_lists" / "commercial_top20.csv"
OUTPUT_DIR = PROJECT_ROOT / "cache" / "adjusted_mav"

CUTOFFS = [6, 7, 8, 9]


def build_dart_strength_map(am_columns):
    with open(PREDICT_JSON) as f:
        data = json.load(f)

    dart_cats = {'developmental toxicity', 'reproductive toxicity', 'endocrine disruption'}
    token_strength = {}
    for item in data:
        tok = item['property_token']
        for c in item['property']['categories']:
            if c['category'] in dart_cats:
                s = c['strength']
                if tok not in token_strength or s > token_strength[tok]:
                    token_strength[tok] = s

    astr = pd.read_csv(ASSAY_STRENGTH)
    title_to_token = dict(zip(astr['title'].str.strip(), astr['token']))

    col_strength = {}
    for col in am_columns:
        tok = title_to_token.get(col.strip())
        if tok is not None and tok in token_strength:
            col_strength[col] = token_strength[tok]
    return col_strength


def compute_mav(activity, columns=None, zscore=False):
    sub = activity[columns] if columns is not None else activity
    if zscore:
        sub = (sub - sub.mean(axis=0)) / sub.std(axis=0)
    return sub.mean(axis=1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    activity = pd.read_parquet(ACTIVITY_PATH)
    col_strength = build_dart_strength_map(list(activity.columns))

    top20 = pd.read_csv(COMMERCIAL_CSV)
    phthalate_inchis = {}
    for _, row in top20.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None:
            continue
        inchi = Chem.MolToInchi(mol)
        if inchi in activity.index:
            phthalate_inchis[row['abbreviation']] = inchi

    # Current MAV (z-scored) for ordering
    current_z = compute_mav(activity, zscore=True)
    current_raw = compute_mav(activity, zscore=False)

    abbrevs = list(phthalate_inchis.keys())
    inchis = [phthalate_inchis[a] for a in abbrevs]
    order = sorted(abbrevs, key=lambda a: current_z.loc[phthalate_inchis[a]], reverse=True)

    # Build tables for each cutoff
    results_raw = {}
    results_zsc = {}
    assay_counts = {}

    results_raw['Current (975)'] = [current_raw.loc[phthalate_inchis[a]] for a in order]
    results_zsc['Current (975)'] = [current_z.loc[phthalate_inchis[a]] for a in order]

    for cutoff in CUTOFFS:
        cols = [c for c, s in col_strength.items() if s >= cutoff]
        assay_counts[cutoff] = len(cols)
        logger.info("Cutoff >= %d: %d assays", cutoff, len(cols))

        mav_raw = compute_mav(activity, cols, zscore=False)
        mav_zsc = compute_mav(activity, cols, zscore=True)

        label = f"≥ {cutoff} ({len(cols)})"
        results_raw[label] = [mav_raw.loc[phthalate_inchis[a]] for a in order]
        results_zsc[label] = [mav_zsc.loc[phthalate_inchis[a]] for a in order]

    colors = {
        'Current (975)': '#999999',
    }
    cutoff_colors = ['#e07b54', '#2c5f7c', '#5ea86e', '#9b6fb0']
    for cutoff, color in zip(CUTOFFS, cutoff_colors):
        label = f"≥ {cutoff} ({assay_counts[cutoff]})"
        colors[label] = color

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)

    for label in results_raw:
        lw = 1.0 if label.startswith('Current') else 1.5
        ls = '--' if label.startswith('Current') else '-'
        alpha = 0.5 if label.startswith('Current') else 0.85
        ax1.plot(range(len(order)), results_raw[label], marker='o', ms=5,
                 color=colors[label], label=label, linewidth=lw, linestyle=ls, alpha=alpha)
        ax2.plot(range(len(order)), results_zsc[label], marker='o', ms=5,
                 color=colors[label], label=label, linewidth=lw, linestyle=ls, alpha=alpha)

    for ax, ylabel, title in [
        (ax1, "MAV (raw)", "Raw MAV — DART/ED strength cutoff sweep"),
        (ax2, "MAV (z-scored)", "Z-scored MAV — DART/ED strength cutoff sweep"),
    ]:
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="medium")
        ax.grid(True, ls="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9, title="DART/ED cutoff (n assays)")

    plt.tight_layout()
    outpath = OUTPUT_DIR / "adj1_cutoff_sweep.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved %s", outpath)


if __name__ == '__main__':
    main()
