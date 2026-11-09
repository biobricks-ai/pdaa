"""
Scatter plots of MAV vs. molecular descriptors, controlled for molecular weight.

Three methods are available (--method):

  bins              Stratify by MW quantile bins; plot per-bin LOESS curves.
  partial_residuals Regress MAV on MW (LOESS), plot residuals vs. descriptor
                    with a single master LOESS curve.
  sensitivity       Sweep mw_span over a range and plot Pearson r / Spearman rho
                    of each descriptor's residual curve as a function of mw_span.
                    Answers: "does my conclusion depend on the span choice?"

Example commercial phthalates are overlaid with distinct markers.

Usage:
    python scripts/mw_controlled_descriptor_plots.py --method partial_residuals
    python scripts/mw_controlled_descriptor_plots.py --method bins --n_bins 4
    python scripts/mw_controlled_descriptor_plots.py --method sensitivity
"""
import argparse
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import pearsonr, spearmanr
from skmisc.loess import loess
import statsmodels.api as sm

import sys
sys.path.append('./')
from scripts.utils.helpers import (
    get_descriptors,
    get_example_phthalates_df,
)

from pathlib import Path
import os as _os
_MATRIX_OVERRIDE = _os.environ.get("PDAA_ACTIVITY_MATRIX")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CACHEDIR = pathlib.Path('cache/entity_similarity')
OUTDIR = pathlib.Path('cache/descriptors')

METRICS = {
    'BranchingRatio': 'Branching Ratio',
    'cLogP': 'cLogP',
    'RotB': 'Number of Rotatable Bonds',
    'Rgyr': 'Radius of Gyration (Å)',
    'Kappa1': 'Kappa Shape Index 1',
    'TPSA': 'TPSA (Å²)',
    'LongestCarbonBackbone': 'Longest Carbon Backbone',
    'MeanSideChainLength': 'Mean Ester Side-Chain Length',
}


def load_data():
    activity_parquet = Path(_MATRIX_OVERRIDE) if _MATRIX_OVERRIDE else CACHEDIR / 'activity_matrix_filled.parquet'
    activity_matrix = pd.read_parquet(activity_parquet)
    Z_activity = (activity_matrix - activity_matrix.mean(axis=0)) / activity_matrix.std(axis=0)
    mav = Z_activity.mean(axis=1)
    mav.name = 'MAV'

    descriptor_parquet = OUTDIR / 'descriptors.parquet'
    if descriptor_parquet.exists():
        descriptors = pd.read_parquet(descriptor_parquet)
        logger.info("Loaded cached descriptors: %d compounds", len(descriptors))
    else:
        logger.info("Computing descriptors from scratch...")
        rows = []
        for inchi in activity_matrix.index:
            mol = AllChem.MolFromInchi(inchi)
            if mol is None:
                continue
            mol_H = AllChem.AddHs(mol)
            rows.append(get_descriptors(mol_H))
        descriptors = pd.DataFrame(rows, index=activity_matrix.index)
        descriptors.to_parquet(descriptor_parquet)

    common = mav.index.intersection(descriptors.index)
    return mav.loc[common], descriptors.loc[common]


def loess_curve(x, y, span=0.5, n_grid=200):
    """LOESS smoothing with confidence bands.

    Falls back to statsmodels lowess (no CI) when skmisc.loess fails,
    which happens on discrete variables with few unique x-values.
    """
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
    """Fit LOESS of y on x and return residuals (same length/order as input)."""
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


def make_mw_bins(mw_series: pd.Series, n_bins: int) -> pd.Series:
    bins = pd.qcut(mw_series, q=n_bins, duplicates='drop')
    label_map = {}
    for interval in sorted(bins.cat.categories):
        lo = int(round(interval.left))
        hi = int(round(interval.right))
        label_map[interval] = f"{lo}–{hi} Da"
    return bins.map(label_map)


def _build_example_markers(example_lookup):
    marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'h', 'p', '8']
    palette = sns.color_palette('Set2', n_colors=len(example_lookup))
    return {
        name: (marker_styles[i % len(marker_styles)], palette[i])
        for i, name in enumerate(sorted(example_lookup.values()))
    }


def _scatter_examples(ax, example_lookup, example_markers, descriptors, y_series, metric):
    for inchi, name in example_lookup.items():
        if inchi not in descriptors.index:
            continue
        xval = descriptors.at[inchi, metric]
        yval = y_series.get(inchi, np.nan)
        if pd.isna(xval) or pd.isna(yval):
            continue
        marker, color = example_markers[name]
        ax.scatter(
            xval, yval,
            marker=marker, s=300,
            color=color, edgecolor='black', linewidth=0.8,
            zorder=10, label=name,
        )


def _dedup_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), **kwargs)


# ── Method: MW bins ──────────────────────────────────────────────────────────

def plot_bins(mav, descriptors, example_lookup, example_markers, args):
    mw = descriptors['MolWt']
    mw_labels = make_mw_bins(mw, args.n_bins)

    palette = sns.color_palette('viridis', n_colors=args.n_bins)
    bin_order = sorted(mw_labels.unique(), key=lambda s: int(s.split('–')[0]))
    bin_colors = {label: palette[i] for i, label in enumerate(bin_order)}

    for metric, axis_label in METRICS.items():
        if metric not in descriptors.columns:
            logger.warning("Descriptor '%s' not found; skipping.", metric)
            continue

        x_all = descriptors[metric]
        valid = x_all.notna() & mav.notna()
        x_clean = x_all[valid].values
        y_clean = mav[valid].values
        bins_clean = mw_labels[valid]

        fig, ax = plt.subplots(figsize=args.figsize or (24, 6), dpi=600)

        for bin_label in bin_order:
            mask = (bins_clean == bin_label).values
            n_pts = mask.sum()
            bx, by = x_clean[mask], y_clean[mask]

            ax.scatter(
                bx, by,
                color=bin_colors[bin_label],
                alpha=0.25, s=100,
                edgecolors='white', linewidths=0.3,
                label=f'{bin_label} (n={n_pts})',
                zorder=2,
            )

            if n_pts >= args.min_bin_size:
                xg, curve, lo, hi = loess_curve(bx, by, span=args.span)
                ax.plot(xg, curve, color=bin_colors[bin_label], lw=2.5, zorder=4)
                if lo is not None:
                    ax.fill_between(xg, lo, hi, color=bin_colors[bin_label],
                                    alpha=0.12, zorder=3)

        _scatter_examples(ax, example_lookup, example_markers, descriptors, mav, metric)
        _dedup_legend(ax, title='MW Bin / Examples', fontsize=12,
                      title_fontsize=14, loc='best', ncol=2)

        corr = pd.Series(x_clean).corr(pd.Series(y_clean))
        ax.set_title(f'MAV vs. {axis_label} by MW Bin (Pearson r = {corr:.3f})',
                     fontsize=24, pad=20)
        ax.set_xlabel(axis_label, fontsize=20, fontweight='bold')
        ax.set_ylabel('MAV', fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)

        outpath = OUTDIR / f'mw_controlled_{metric}.png'
        plt.tight_layout()
        plt.savefig(outpath, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info("Saved %s", outpath)


# ── Method: partial residuals ────────────────────────────────────────────────

def plot_partial_residuals(mav, descriptors, example_lookup, example_markers, args):
    control = args.control
    control_label = METRICS.get(control, control)
    short_control = control_label.split('(')[0].strip()

    if control not in descriptors.columns:
        logger.error("Control variable '%s' not found in descriptors.", control)
        return

    ctrl = descriptors[control]
    valid = ctrl.notna() & mav.notna()
    ctrl_clean = ctrl[valid].values
    mav_clean = mav[valid].values

    mav_resid_all = pd.Series(
        loess_residuals(ctrl_clean, mav_clean, span=args.mw_span),
        index=mav[valid].index,
        name='MAV_resid',
    )
    logger.info(
        "%s adjustment: LOESS span=%.2f, residual std=%.4f (original std=%.4f)",
        control, args.mw_span, mav_resid_all.std(), mav[valid].std(),
    )

    default_color = '#61b2d8'

    for metric, axis_label in METRICS.items():
        if metric not in descriptors.columns:
            logger.warning("Descriptor '%s' not found; skipping.", metric)
            continue
        if metric == control:
            continue

        x_all = descriptors[metric]
        shared = mav_resid_all.index.intersection(x_all.dropna().index)
        x_clean = x_all.loc[shared].values
        y_clean = mav_resid_all.loc[shared].values

        fig, ax = plt.subplots(figsize=args.figsize or (24, 6), dpi=600)

        ax.scatter(
            x_clean, y_clean,
            color=default_color, alpha=0.3, s=100,
            edgecolors='white', linewidths=0.3,
            zorder=2,
        )

        xg, curve, lo, hi = loess_curve(x_clean, y_clean, span=args.span)
        ax.plot(xg, curve, color='black', lw=2.5, zorder=4)
        if lo is not None:
            ax.fill_between(xg, lo, hi, color='grey', alpha=0.20, zorder=3)

        _scatter_examples(ax, example_lookup, example_markers,
                          descriptors, mav_resid_all, metric)
        _dedup_legend(ax, title='Examples', fontsize=12,
                      title_fontsize=14, loc='best')

        r, p = pearsonr(x_clean, y_clean)
        rho, p_s = spearmanr(x_clean, y_clean)
        ax.set_title(
            f'MAV ({short_control}-adjusted) vs. {axis_label}'
            f'  (Pearson r = {r:.3f}, p = {p:.2g}'
            f' | Spearman ρ = {rho:.3f}, p = {p_s:.2g})',
            fontsize=22, pad=20,
        )
        ax.set_xlabel(axis_label, fontsize=20, fontweight='bold')
        ax.set_ylabel(f'MAV residual ({short_control} removed)',
                      fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)

        outpath = OUTDIR / f'{control}_partial_{metric}.png'
        plt.tight_layout()
        plt.savefig(outpath, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info("Saved %s", outpath)


# ── Method: sensitivity sweep ────────────────────────────────────────────────

def plot_sensitivity(mav, descriptors, args):
    mw = descriptors['MolWt']
    valid_mw = mw.notna() & mav.notna()
    mw_clean = mw[valid_mw].values
    mav_clean = mav[valid_mw].values

    spans = np.linspace(0.15, 0.95, 17)

    available = [m for m in METRICS if m in descriptors.columns]

    records = []
    for s in spans:
        resid = loess_residuals(mw_clean, mav_clean, span=s)
        resid_series = pd.Series(resid, index=mav[valid_mw].index)
        residual_std = resid_series.std()
        variance_removed = 1.0 - (residual_std ** 2) / (mav[valid_mw].std() ** 2)

        for metric in available:
            x = descriptors.loc[resid_series.index, metric]
            shared = x.dropna().index.intersection(resid_series.index)
            xv = x.loc[shared].values
            yv = resid_series.loc[shared].values
            r, p_r = pearsonr(xv, yv)
            rho, p_s = spearmanr(xv, yv)
            records.append({
                'mw_span': s,
                'metric': metric,
                'label': METRICS[metric],
                'pearson_r': r,
                'pearson_p': p_r,
                'spearman_rho': rho,
                'spearman_p': p_s,
                'variance_removed': variance_removed,
            })

    df = pd.DataFrame(records)

    palette = sns.color_palette('tab10', n_colors=len(available))
    color_map = {m: palette[i] for i, m in enumerate(available)}

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), dpi=300, sharex=True)

    # Panel A: Pearson r vs mw_span
    ax = axes[0]
    for metric in available:
        sub = df[df['metric'] == metric]
        ax.plot(sub['mw_span'], sub['pearson_r'],
                marker='o', ms=4, lw=1.8,
                color=color_map[metric], label=METRICS[metric])
        sig = sub['pearson_p'] < 0.05
        if sig.any():
            ax.scatter(sub.loc[sig, 'mw_span'], sub.loc[sig, 'pearson_r'],
                       color=color_map[metric], s=30, zorder=5,
                       edgecolors='black', linewidths=0.6)
    ax.axhline(0, color='grey', ls='--', lw=0.8)
    ax.set_ylabel('Pearson r', fontsize=14)
    ax.set_title('Sensitivity of descriptor–MAV correlation to MW adjustment span',
                 fontsize=16, pad=12)
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.text(0.01, 0.97, 'filled = p < 0.05', transform=ax.transAxes,
            fontsize=9, va='top', ha='left', color='grey')

    # Panel B: Spearman rho vs mw_span
    ax = axes[1]
    for metric in available:
        sub = df[df['metric'] == metric]
        ax.plot(sub['mw_span'], sub['spearman_rho'],
                marker='o', ms=4, lw=1.8,
                color=color_map[metric], label=METRICS[metric])
        sig = sub['spearman_p'] < 0.05
        if sig.any():
            ax.scatter(sub.loc[sig, 'mw_span'], sub.loc[sig, 'spearman_rho'],
                       color=color_map[metric], s=30, zorder=5,
                       edgecolors='black', linewidths=0.6)
    ax.axhline(0, color='grey', ls='--', lw=0.8)
    ax.set_ylabel('Spearman ρ', fontsize=14)
    ax.text(0.01, 0.97, 'filled = p < 0.05', transform=ax.transAxes,
            fontsize=9, va='top', ha='left', color='grey')

    # Panel C: fraction of MAV variance removed
    ax = axes[2]
    var_removed = df.drop_duplicates('mw_span')[['mw_span', 'variance_removed']]
    ax.plot(var_removed['mw_span'], var_removed['variance_removed'],
            marker='s', ms=5, lw=2, color='black')
    ax.set_ylabel('Fraction of MAV\nvariance removed', fontsize=14)
    ax.set_xlabel('MW adjustment LOESS span', fontsize=14)
    ax.set_ylim(bottom=0)

    for a in axes:
        a.tick_params(labelsize=12)
        a.grid(True, ls='--', alpha=0.5)

    panel_labels = ['A)', 'B)', 'C)']
    for a, lab in zip(axes, panel_labels):
        a.text(-0.06, 1.05, lab, transform=a.transAxes,
               fontsize=14, fontweight='bold', va='top', ha='right')

    outpath = OUTDIR / 'mw_span_sensitivity.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("Saved %s", outpath)

    df.to_csv(OUTDIR / 'mw_span_sensitivity.csv', index=False)
    logger.info("Saved tabular results to %s", OUTDIR / 'mw_span_sensitivity.csv')


# ── Method: curve overlay for a single descriptor ───────────────────────────

def _fit_display_curve(xv, yv, span, min_loess=20):
    """Fit a smoothing curve appropriate for the sample size.

    For n >= min_loess, uses LOESS.  For smaller n, falls back to
    numpy polyfit (degree 1) so the trend direction is still visible.
    Returns (x_grid, y_curve).
    """
    order = np.argsort(xv)
    xs, ys = xv[order], yv[order]
    if len(xs) >= min_loess:
        xg, curve, _, _ = loess_curve(xs, ys, span=span)
        return xg, curve
    coeffs = np.polyfit(xs, ys, deg=min(2, len(xs) - 1))
    xg = np.linspace(xs.min(), xs.max(), 200)
    return xg, np.polyval(coeffs, xg)


def plot_curve_overlay(mav, descriptors, example_lookup, example_markers, args):
    metric = args.overlay_metric
    if metric not in descriptors.columns:
        logger.error("Descriptor '%s' not found in data.", metric)
        return

    axis_label = METRICS.get(metric, metric)
    mw = descriptors['MolWt']
    valid_mw = mw.notna() & mav.notna()
    mw_clean = mw[valid_mw].values
    mav_clean = mav[valid_mw].values

    example_inchis = set(example_lookup.keys())
    use_examples_only = args.examples_only and example_inchis

    spans = np.linspace(0.15, 0.95, 17)
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=spans.min(), vmax=spans.max())

    fig, ax = plt.subplots(figsize=args.figsize or (24, 8), dpi=600)

    # Background scatter at default mw_span
    default_resid = loess_residuals(mw_clean, mav_clean, span=args.mw_span)
    default_resid_s = pd.Series(default_resid, index=mav[valid_mw].index)

    if use_examples_only:
        bg_idx = default_resid_s.index.intersection(example_inchis)
    else:
        bg_idx = default_resid_s.index

    x_desc = descriptors.loc[bg_idx, metric]
    shared = x_desc.dropna().index.intersection(default_resid_s.index)
    ax.scatter(
        x_desc.loc[shared].values, default_resid_s.loc[shared].values,
        color='#d0d0d0', alpha=0.35 if use_examples_only else 0.20,
        s=120 if use_examples_only else 60,
        edgecolors='none', zorder=1,
    )

    n_curve_pts = None
    for s in spans:
        resid = loess_residuals(mw_clean, mav_clean, span=s)
        resid_s = pd.Series(resid, index=mav[valid_mw].index)

        if use_examples_only:
            curve_idx = resid_s.index.intersection(example_inchis)
        else:
            curve_idx = resid_s.index

        x_vals = descriptors.loc[curve_idx, metric]
        shared_idx = x_vals.dropna().index.intersection(resid_s.index)
        xv = x_vals.loc[shared_idx].values
        yv = resid_s.loc[shared_idx].values
        n_curve_pts = len(xv)

        xg, curve = _fit_display_curve(xv, yv, span=args.span)
        r, _ = pearsonr(xv, yv)

        color = cmap(norm(s))
        ax.plot(xg, curve, color=color, lw=2.0, alpha=0.85, zorder=3,
                label=f'span={s:.2f}  (r={r:+.3f})')

    _scatter_examples(ax, example_lookup, example_markers,
                      descriptors, default_resid_s, metric)

    sm_scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_scalar.set_array([])
    cbar = fig.colorbar(sm_scalar, ax=ax, pad=0.01, aspect=30)
    cbar.set_label('MW adjustment LOESS span', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    example_handles = {k: v for k, v in by_label.items() if not k.startswith('span=')}

    if example_handles:
        legend_ex = ax.legend(
            example_handles.values(), example_handles.keys(),
            title='Examples', fontsize=10, title_fontsize=12,
            loc='upper left', bbox_to_anchor=(0.0, 1.0),
        )
        ax.add_artist(legend_ex)

    subset_note = f' (n={n_curve_pts} named phthalates)' if use_examples_only else ''
    fit_note = 'polynomial' if (use_examples_only and n_curve_pts < 20) else 'LOESS'
    ax.set_title(
        f'MAV (MW-adjusted) vs. {axis_label} — {fit_note} curves across MW span values'
        f'{subset_note}',
        fontsize=22, pad=20,
    )
    ax.set_xlabel(axis_label, fontsize=20, fontweight='bold')
    ax.set_ylabel('MAV residual (MW removed)', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=1.0)

    suffix = '_examples' if use_examples_only else ''
    outpath = OUTDIR / f'mw_curve_overlay_{metric}{suffix}.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("Saved %s", outpath)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method',
                        choices=['bins', 'partial_residuals', 'sensitivity', 'curve_overlay'],
                        default='partial_residuals',
                        help='MW-control method (default: partial_residuals)')
    parser.add_argument('--overlay_metric', type=str, default='BranchingRatio',
                        help='Descriptor to plot for curve_overlay method (default: BranchingRatio)')
    parser.add_argument('--n_bins', type=int, default=4,
                        help='Number of MW quantile bins for "bins" method (default: 4)')
    parser.add_argument('--span', type=float, default=0.5,
                        help='LOESS span for descriptor curves (default: 0.5)')
    parser.add_argument('--control', type=str, default='MolWt',
                        help='Descriptor to control for in partial_residuals (default: MolWt)')
    parser.add_argument('--mw_span', type=float, default=0.5,
                        help='LOESS span for control variable adjustment (default: 0.5)')
    parser.add_argument('--min_bin_size', type=int, default=30,
                        help='Skip LOESS for bins with fewer points (default: 30)')
    parser.add_argument('--examples_only', action='store_true',
                        help='Restrict scatter and trend curves to named example phthalates only. '
                             'MW adjustment is still fit on all data.')
    parser.add_argument('--figsize', type=float, nargs=2, default=None, metavar=('W', 'H'),
                        help='Figure size in inches, e.g. --figsize 12 6 (default: method-dependent)')
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    mav, descriptors = load_data()

    example_df = get_example_phthalates_df()
    example_lookup = {
        row['inchi']: row['name']
        for _, row in example_df.iterrows()
        if row['inchi'] in descriptors.index
    }
    example_markers = _build_example_markers(example_lookup)

    if args.method == 'bins':
        plot_bins(mav, descriptors, example_lookup, example_markers, args)
    elif args.method == 'sensitivity':
        plot_sensitivity(mav, descriptors, args)
    elif args.method == 'curve_overlay':
        plot_curve_overlay(mav, descriptors, example_lookup, example_markers, args)
    else:
        plot_partial_residuals(mav, descriptors, example_lookup, example_markers, args)


if __name__ == '__main__':
    main()
