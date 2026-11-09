"""
Generate Figure 2 panels: MAV vs. four QSAR descriptors.

Panel A shows raw MAV vs. MolWt (all compounds).
Panels B–D show MW-adjusted MAV residuals vs. BranchingRatio, RotB, and
cLogP for compounds with MW > 250 Da.

Usage:
    python scripts/figure2_descriptor_panels.py
    python scripts/figure2_descriptor_panels.py --mw_cutoff 300
    python scripts/figure2_descriptor_panels.py --figsize 16 12
"""
import argparse
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

try:
    from skmisc.loess import loess
except ImportError:  # fall back to statsmodels LOWESS (no confidence bands)
    loess = None

import sys

from pathlib import Path as _P
import os as _os
_MATRIX_OVERRIDE = _os.environ.get("PDAA_ACTIVITY_MATRIX")

sys.path.append('./')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CACHEDIR = pathlib.Path('cache/entity_similarity')
DESCDIR = pathlib.Path('cache/descriptors')
COMMERCIAL_CSV = pathlib.Path('resources/phthalate_lists/commercial_top20.csv')

POINT_COLOR = '#2c5f7c'
LABEL_COLOR = '#333333'

PANELS = [
    {'key': 'MolWt',         'label': 'Molecular Weight (Da)',    'mw_adjust': False},
    {'key': 'BranchingRatio', 'label': 'Branching Ratio',         'mw_adjust': True},
    {'key': 'RotB',           'label': 'Number of Rotatable Bonds', 'mw_adjust': True},
    {'key': 'cLogP',          'label': 'cLogP',                   'mw_adjust': True},
]


def loess_curve(x, y, span=0.5, n_grid=200):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    xg = np.linspace(xs.min(), xs.max(), n_grid)
    if loess is not None:
        try:
            model = loess(xs, ys, span=span, degree=1)
            model.fit()
            pred = model.predict(xg, stderror=True)
            conf = pred.confidence(alpha=0.05)
            return xg, pred.values, conf.lower, conf.upper
        except Exception:
            pass
    result = sm.nonparametric.lowess(ys, xs, frac=span)
    return result[:, 0], result[:, 1], None, None


def loess_residuals(x, y, span=0.5):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    fitted = None
    if loess is not None:
        try:
            model = loess(xs, ys, span=span, degree=1)
            model.fit()
            fitted = model.predict(xs).values
        except Exception:
            fitted = None
    if fitted is None:
        result = sm.nonparametric.lowess(ys, xs, frac=span)
        fitted = result[:, 1]
    resid = ys - fitted
    return resid[np.argsort(order)]


def load_data():
    activity = pd.read_parquet(_P(_MATRIX_OVERRIDE) if _MATRIX_OVERRIDE else CACHEDIR / 'activity_matrix_filled.parquet')
    Z = (activity - activity.mean(axis=0)) / activity.std(axis=0)
    mav = Z.mean(axis=1)
    mav.name = 'MAV'

    descriptors = pd.read_parquet(DESCDIR / 'descriptors.parquet')
    logger.info("Loaded descriptors: %d compounds", len(descriptors))

    common = mav.index.intersection(descriptors.index)
    return mav.loc[common], descriptors.loc[common]


def load_commercial_phthalates(descriptors):
    top20 = pd.read_csv(COMMERCIAL_CSV)
    rows = []
    for _, row in top20.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None:
            continue
        inchi = Chem.MolToInchi(mol)
        if inchi not in descriptors.index:
            logger.warning("%s not in descriptor data, skipping", row['abbreviation'])
            continue
        rows.append({'abbrev': row['abbreviation'], 'inchi': inchi})
    return pd.DataFrame(rows)


def annotate_commercial(ax, commercial_df, descriptors, y_series, key):
    texts = []
    xpts, ypts = [], []
    for _, row in commercial_df.iterrows():
        inchi = row['inchi']
        if inchi not in y_series.index:
            continue
        xval = descriptors.at[inchi, key]
        yval = y_series.at[inchi]
        if pd.isna(xval) or pd.isna(yval):
            continue
        ax.scatter(xval, yval, color=POINT_COLOR, s=45, zorder=5,
                   alpha=0.85, edgecolors='white', linewidth=0.5)
        xpts.append(xval)
        ypts.append(yval)
        texts.append(ax.annotate(
            row['abbrev'], (xval, yval),
            fontsize=7, alpha=0.9, color=LABEL_COLOR,
        ))
    if texts:
        adjust_text(
            texts, x=xpts, y=ypts, ax=ax,
            arrowprops=dict(arrowstyle='-', color='#888888', lw=0.5),
            expand_points=(1.5, 1.5),
            expand_text=(1.2, 1.2),
            force_text=(0.5, 0.5),
            force_points=(0.5, 0.5),
            lim=200,
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mw_cutoff', type=float, default=250,
                        help='MW threshold for adjusted panels (default: 250)')
    parser.add_argument('--span', type=float, default=0.5,
                        help='LOESS span for descriptor curves (default: 0.5)')
    parser.add_argument('--mw_span', type=float, default=0.5,
                        help='LOESS span for MW adjustment (default: 0.5)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[14, 10],
                        metavar=('W', 'H'))
    args = parser.parse_args()

    outdir = DESCDIR / 'mw_adjusted_descriptor_panels'
    outdir.mkdir(parents=True, exist_ok=True)

    mav, descriptors = load_data()
    commercial_df = load_commercial_phthalates(descriptors)
    logger.info("Loaded %d commercial phthalates with data", len(commercial_df))

    mw_mask = descriptors['MolWt'] >= args.mw_cutoff
    mw_above = descriptors.loc[mw_mask, 'MolWt'].values
    mav_above = mav.loc[mw_mask].values
    mav_resid = pd.Series(
        loess_residuals(mw_above, mav_above, span=args.mw_span),
        index=mav.loc[mw_mask].index,
    )
    logger.info("MW >= %.0f Da: %d compounds", args.mw_cutoff, mw_mask.sum())

    panel_labels = ['A', 'B', 'C', 'D']
    fig, axes = plt.subplots(2, 2, figsize=args.figsize, dpi=300)
    axes = axes.ravel()
    scatter_color = '#61b2d8'

    for i, panel in enumerate(PANELS):
        ax = axes[i]
        key = panel['key']

        if panel['mw_adjust']:
            idx = mav_resid.index.intersection(descriptors[key].dropna().index)
            x = descriptors.loc[idx, key].values
            y = mav_resid.loc[idx].values
            ylabel = 'MAV residual (MW-adjusted)'
            note = f'MW ≥ {args.mw_cutoff:.0f} Da, MW-adjusted'
            y_series = mav_resid
        else:
            valid = descriptors[key].notna() & mav.notna()
            idx = descriptors.index[valid]
            x = descriptors.loc[idx, key].values
            y = mav.loc[idx].values
            ylabel = 'MAV'
            note = 'all compounds'
            y_series = mav

        r, p_r = pearsonr(x, y)
        rho, p_s = spearmanr(x, y)

        ax.scatter(x, y, color=scatter_color, alpha=0.25, s=25,
                   edgecolors='none', zorder=1)

        xg, curve, lo, hi = loess_curve(x, y, span=args.span)
        ax.plot(xg, curve, color='black', lw=2.5, zorder=4)
        if lo is not None:
            ax.fill_between(xg, lo, hi, color='grey', alpha=0.15, zorder=2)

        if panel['mw_adjust']:
            # Residual panels are centered on zero by construction; the reference
            # line and note prevent the residuals being misread as raw activity.
            ax.axhline(0.0, color='#b23b3b', ls='--', lw=1.0, zorder=3)
            ax.text(0.02, 0.03,
                    'Residuals: activity with MW effect removed\n(0 = no signal beyond MW)',
                    transform=ax.transAxes, fontsize=7, color='#b23b3b',
                    va='bottom', ha='left')

        annotate_commercial(ax, commercial_df, descriptors, y_series, key)

        ax.set_title(
            f'{note}\n'
            f'n={len(x)}, r={r:.3f} (p={p_r:.2g}), '
            f'ρ={rho:.3f} (p={p_s:.2g})',
            fontsize=10,
        )
        ax.set_xlabel(panel['label'], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, ls='--', alpha=0.4)

        ax.text(-0.08, 1.12, f'{panel_labels[i]})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')

        logger.info("Panel %s (%s): n=%d, r=%.3f (p=%.2g), rho=%.3f (p=%.2g)",
                    panel_labels[i], key, len(x), r, p_r, rho, p_s)

    plt.tight_layout()
    outpath = outdir / 'combined_panels.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("Saved %s", outpath)

    for i, panel in enumerate(PANELS):
        fig_single, ax = plt.subplots(figsize=(7, 5), dpi=300)
        key = panel['key']

        if panel['mw_adjust']:
            idx = mav_resid.index.intersection(descriptors[key].dropna().index)
            x = descriptors.loc[idx, key].values
            y = mav_resid.loc[idx].values
            ylabel = 'MAV residual (MW-adjusted)'
            y_series = mav_resid
        else:
            valid = descriptors[key].notna() & mav.notna()
            idx = descriptors.index[valid]
            x = descriptors.loc[idx, key].values
            y = mav.loc[idx].values
            ylabel = 'MAV'
            y_series = mav

        r, p_r = pearsonr(x, y)
        rho, p_s = spearmanr(x, y)

        ax.scatter(x, y, color=scatter_color, alpha=0.25, s=25,
                   edgecolors='none', zorder=1)
        xg, curve, lo, hi = loess_curve(x, y, span=args.span)
        ax.plot(xg, curve, color='black', lw=2.5, zorder=4)
        if lo is not None:
            ax.fill_between(xg, lo, hi, color='grey', alpha=0.15, zorder=2)

        annotate_commercial(ax, commercial_df, descriptors, y_series, key)

        ax.set_xlabel(panel['label'], fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(True, ls='--', alpha=0.4)

        plt.tight_layout()
        single_path = outdir / f'panel_{panel_labels[i]}_{key}.png'
        plt.savefig(single_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info("Saved %s", single_path)


if __name__ == '__main__':
    main()
