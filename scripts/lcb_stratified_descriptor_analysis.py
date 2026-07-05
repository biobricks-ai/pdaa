"""
MAV vs. Branching Ratio within LCB groups, controlling for MW.

For each LongestCarbonBackbone group (1–3, 4–6, 7–8, 9+), plots MAV vs.
BranchingRatio with MW-adjusted partial residuals.

By default, produces a single MW-adjusted curve per panel.  With --sweep,
overlays curves at progressively higher within-group MW cutoffs.

Usage:
    python scripts/lcb_stratified_br_analysis.py
    python scripts/lcb_stratified_br_analysis.py --sweep --n_thresholds 5
    python scripts/lcb_stratified_br_analysis.py --figsize 14 10
"""
import argparse
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from skmisc.loess import loess
import statsmodels.api as sm

import sys
sys.path.append('./')
from scripts.utils.helpers import get_example_phthalates_df

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CACHEDIR = pathlib.Path('cache/entity_similarity')
OUTDIR = pathlib.Path('cache/descriptors')

LCB_GROUPS = [
    ('1–3', lambda x: x <= 3),
    ('4–6', lambda x: (x >= 4) & (x <= 6)),
    ('7–8', lambda x: (x >= 7) & (x <= 8)),
    ('9+',  lambda x: x >= 9),
]

DESCRIPTOR_LABELS = {
    'BranchingRatio': 'Branching Ratio',
    'RotB': 'Number of Rotatable Bonds',
    'cLogP': 'cLogP',
}


def loess_curve(x, y, span=0.5, n_grid=200):
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    x_grid = np.linspace(x_sorted.min(), x_sorted.max(), n_grid)
    try:
        model = loess(x_sorted, y_sorted, span=span, degree=1)
        model.fit()
        pred = model.predict(x_grid, stderror=True)
        conf = pred.confidence(alpha=0.05)
        return x_grid, pred.values, conf.lower, conf.upper
    except Exception:
        result = sm.nonparametric.lowess(y_sorted, x_sorted, frac=span)
        return result[:, 0], result[:, 1], None, None


def loess_residuals(x, y, span=0.5):
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    try:
        model = loess(x_sorted, y_sorted, span=span, degree=1)
        model.fit()
        fitted = model.predict(x_sorted).values
    except Exception:
        result = sm.nonparametric.lowess(y_sorted, x_sorted, frac=span)
        fitted = result[:, 1]
    resid_sorted = y_sorted - fitted
    inverse = np.argsort(order)
    return resid_sorted[inverse]


def load_data():
    activity_parquet = CACHEDIR / 'activity_matrix_filled.parquet'
    activity_matrix = pd.read_parquet(activity_parquet)
    Z_activity = (activity_matrix - activity_matrix.mean(axis=0)) / activity_matrix.std(axis=0)
    mav = Z_activity.mean(axis=1)
    mav.name = 'MAV'

    descriptor_parquet = OUTDIR / 'descriptors.parquet'
    descriptors = pd.read_parquet(descriptor_parquet)
    logger.info("Loaded descriptors: %d compounds", len(descriptors))

    common = mav.index.intersection(descriptors.index)
    return mav.loc[common], descriptors.loc[common]


def compute_mw_thresholds(mw_series, n_thresholds):
    quantiles = np.linspace(0, 0.75, n_thresholds)
    thresholds = [0] + [int(round(mw_series.quantile(q))) for q in quantiles[1:]]
    return sorted(set(thresholds))


def _get_example_lookup(descriptors):
    example_df = get_example_phthalates_df()
    return {
        row['inchi']: row['name']
        for _, row in example_df.iterrows()
        if row['inchi'] in descriptors.index
    }


def _group_data(mav, descriptors, group_filter, descriptor_key):
    lcb = descriptors['LongestCarbonBackbone']
    mask = (group_filter(lcb)
            & mav.notna()
            & descriptors[descriptor_key].notna()
            & descriptors['MolWt'].notna())
    idx = descriptors.index[mask]
    return idx, mav.loc[idx], descriptors.loc[idx, descriptor_key].values, descriptors.loc[idx, 'MolWt']


def _annotate_examples(ax, example_lookup, group_idx, descriptors, mav, descriptor_key):
    for inchi, name in example_lookup.items():
        if inchi not in group_idx:
            continue
        xval = descriptors.at[inchi, descriptor_key]
        yval = mav.at[inchi]
        if pd.isna(xval) or pd.isna(yval):
            continue
        ax.scatter(xval, yval, marker='D', s=60, color='red',
                   edgecolor='black', linewidth=0.6, zorder=10)
        ax.annotate(name, (xval, yval), fontsize=6.5, ha='left',
                    va='bottom', xytext=(3, 3),
                    textcoords='offset points')


def plot_lcb_stratified(mav, descriptors, args, descriptor_key, outdir):
    example_lookup = _get_example_lookup(descriptors)
    axis_label = DESCRIPTOR_LABELS.get(descriptor_key, descriptor_key)
    default_color = '#61b2d8'

    fig, axes = plt.subplots(2, 2, figsize=args.figsize, dpi=300)
    axes = axes.ravel()

    for panel_idx, (group_label, group_filter) in enumerate(LCB_GROUPS):
        ax = axes[panel_idx]
        group_idx, mav_group, desc_group, mw_group = _group_data(
            mav, descriptors, group_filter, descriptor_key)

        mav_resid = loess_residuals(mw_group.values, mav_group.values,
                                    span=args.mw_span)
        r, p = pearsonr(desc_group, mav_resid)
        rho, p_s = spearmanr(desc_group, mav_resid)
        logger.info("%s — LCB %s (n=%d): r=%.3f (p=%.2g), rho=%.3f (p=%.2g)",
                    descriptor_key, group_label, len(group_idx), r, p, rho, p_s)

        ax.scatter(desc_group, mav_resid, color=default_color, alpha=0.25,
                   s=30, edgecolors='none', zorder=1)

        xg, curve, lo, hi = loess_curve(desc_group, mav_resid, span=args.span)
        ax.plot(xg, curve, color='black', lw=2.5, zorder=4)
        if lo is not None:
            ax.fill_between(xg, lo, hi, color='grey', alpha=0.15, zorder=2)

        _annotate_examples(ax, example_lookup, group_idx, descriptors,
                           pd.Series(mav_resid, index=group_idx, name='MAV'),
                           descriptor_key)

        ax.set_title(
            f'LCB {group_label}  (n={len(group_idx)}, r={r:+.2f}, '
            f'ρ={rho:+.2f})',
            fontsize=12, fontweight='bold',
        )
        ax.set_xlabel(axis_label, fontsize=11)
        ax.set_ylabel('MAV residual (MW-adjusted)', fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, ls='--', alpha=0.4)

    fig.suptitle(
        f'MAV vs. {axis_label} by LCB Group (MW-adjusted)',
        fontsize=15, fontweight='bold', y=1.02,
    )
    outpath = outdir / f'lcb_stratified_{descriptor_key}.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("Saved %s", outpath)
    return outpath


def plot_lcb_stratified_sweep(mav, descriptors, args, descriptor_key, outdir):
    example_lookup = _get_example_lookup(descriptors)
    axis_label = DESCRIPTOR_LABELS.get(descriptor_key, descriptor_key)

    fig, axes = plt.subplots(2, 2, figsize=args.figsize, dpi=300)
    axes = axes.ravel()

    cmap = plt.cm.viridis

    for panel_idx, (group_label, group_filter) in enumerate(LCB_GROUPS):
        ax = axes[panel_idx]
        group_idx, mav_group, desc_group, mw_group = _group_data(
            mav, descriptors, group_filter, descriptor_key)

        thresholds = compute_mw_thresholds(mw_group, args.n_thresholds)
        logger.info("%s — LCB %s (n=%d): MW thresholds = %s",
                     descriptor_key, group_label, len(group_idx), thresholds)

        norm = plt.Normalize(vmin=0, vmax=len(thresholds) - 1)
        colors = [cmap(norm(i)) for i in range(len(thresholds))]

        for t_idx, mw_thresh in enumerate(thresholds):
            mask = mw_group.values >= mw_thresh
            n_pts = mask.sum()

            if n_pts < 15:
                logger.warning("  MW >= %d: only %d points, skipping", mw_thresh, n_pts)
                continue

            mw_sub = mw_group.values[mask]
            mav_sub = mav_group.values[mask]
            desc_sub = desc_group[mask]

            if mw_thresh == 0:
                y_plot = mav_sub
            else:
                y_plot = loess_residuals(mw_sub, mav_sub, span=args.mw_span)

            r, p = pearsonr(desc_sub, y_plot)

            if t_idx == 0:
                ax.scatter(
                    desc_sub, y_plot,
                    color=colors[t_idx], alpha=0.20, s=30,
                    edgecolors='none', zorder=1,
                )

            if n_pts >= 30:
                xg, curve, lo, hi = loess_curve(desc_sub, y_plot, span=args.span)
                ax.plot(xg, curve, color=colors[t_idx], lw=2.5, zorder=4,
                        label=f'MW≥{mw_thresh} (n={n_pts}, r={r:+.2f})')
                if lo is not None:
                    ax.fill_between(xg, lo, hi, color=colors[t_idx],
                                    alpha=0.08, zorder=2)
            else:
                coeffs = np.polyfit(desc_sub, y_plot, deg=1)
                xg = np.linspace(desc_sub.min(), desc_sub.max(), 100)
                ax.plot(xg, np.polyval(coeffs, xg), color=colors[t_idx],
                        lw=2.5, ls='--', zorder=4,
                        label=f'MW≥{mw_thresh} (n={n_pts}, r={r:+.2f})')

        _annotate_examples(ax, example_lookup, group_idx, descriptors, mav,
                           descriptor_key)

        ax.set_title(f'LCB {group_label}', fontsize=13, fontweight='bold')
        ax.set_xlabel(axis_label, fontsize=11)
        ax.set_ylabel('MAV / MAV residual', fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=7.5, loc='best')

    fig.suptitle(
        f'MAV vs. {axis_label} by LCB Group — MW Threshold Sweep',
        fontsize=15, fontweight='bold', y=1.02,
    )
    outpath = outdir / f'lcb_stratified_{descriptor_key}_mw_sweep.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("Saved %s", outpath)
    return outpath


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--descriptor', type=str, nargs='*',
                        default=list(DESCRIPTOR_LABELS.keys()),
                        choices=list(DESCRIPTOR_LABELS.keys()),
                        help='Descriptor(s) to plot (default: all three)')
    parser.add_argument('--sweep', action='store_true',
                        help='Overlay curves at multiple MW thresholds per panel')
    parser.add_argument('--n_thresholds', type=int, default=4,
                        help='MW thresholds per LCB group in sweep mode (default: 4)')
    parser.add_argument('--span', type=float, default=0.5,
                        help='LOESS span for descriptor curves (default: 0.5)')
    parser.add_argument('--mw_span', type=float, default=0.5,
                        help='LOESS span for MW adjustment (default: 0.5)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[14, 10],
                        metavar=('W', 'H'),
                        help='Figure size in inches (default: 14 10)')
    args = parser.parse_args()

    outdir = OUTDIR / 'lcb_stratified_mw_adjusted'
    outdir.mkdir(parents=True, exist_ok=True)

    mav, descriptors = load_data()
    plot_fn = plot_lcb_stratified_sweep if args.sweep else plot_lcb_stratified
    for key in args.descriptor:
        plot_fn(mav, descriptors, args, key, outdir)
    logger.info("Done. Output directory: %s", outdir)


if __name__ == '__main__':
    main()
