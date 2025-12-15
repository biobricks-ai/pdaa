import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('./')  # so utility scripts can be found
from scripts.utils.helpers import get_activity_df

from rdkit.Chem import AllChem


# Default savepath for figures (matches the style in descriptor_activity_relationship.py)
fig_path = Path('cache/compare_phthalate_activities')


@dataclass(frozen=True)
class BootstrapConfig:
    n_boot: int
    n_draw: int
    seed: int


def _read_inchis_from_file(path: Path) -> List[str]:
    """
    Read InChIs from a file.

    Supported formats:
      - .txt: one InChI per line
      - .csv: expects an 'inchi' column (case-insensitive)
    """
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".txt":
        inchis: List[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                inchis.append(s)
        return inchis

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        col_map = {c.lower(): c for c in df.columns}
        if "inchi" not in col_map:
            raise ValueError(f"CSV {path} must have an 'inchi' column.")
        inchi_col = col_map["inchi"]
        inchis = (
            df[inchi_col]
            .astype(str)
            .map(lambda s: s.strip())
            .loc[lambda s: s != ""]
            .tolist()
        )
        return inchis

    raise ValueError(f"Unsupported input file type: {path.suffix}")


def _load_example_name_map(csv_path: Path) -> Dict[str, str]:
    """
    Build a mapping: InChI -> name from resources/example_phthalates.csv.

    Requires columns: 'name', 'inchi' (case-insensitive).
    """
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    col_map = {c.lower(): c for c in df.columns}
    required = {"name", "inchi"}
    missing = required - set(col_map)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    name_col = col_map["name"]
    inchi_col = col_map["inchi"]

    name_map: Dict[str, str] = {}
    for _, row in df.iterrows():
        inchi = str(row[inchi_col]).strip()
        name = str(row[name_col]).strip()
        if inchi and name:
            name_map[inchi] = name

    return name_map


def _canonical_smiles_from_inchi(inchi: str) -> Optional[str]:
    mol = AllChem.MolFromInchi(inchi)
    if mol is None:
        return None
    try:
        smiles = AllChem.MolToSmiles(
            mol,
            canonical=True,
        )
        return smiles
    except Exception:
        return None


def _label_for_inchi(
        inchi: str,
        name_map: Dict[str, str],
    ) -> str:
    if inchi in name_map:
        return name_map[inchi]

    smiles = _canonical_smiles_from_inchi(inchi)
    if smiles is not None:
        return smiles

    # Last resort: the InChI itself
    return inchi


def _compute_bootstrap_means(
        activity_df: pd.DataFrame,
        inchis: List[str],
        config: BootstrapConfig,
    ) -> Dict[str, np.ndarray]:
    """
    For each bootstrap iteration:
      - sample assay columns WITH replacement (same selection for all phthalates)
      - compute row-wise mean for each requested InChI over the sampled columns (skip NaNs)

    Returns:
      dict mapping inchi -> array of bootstrap mean values (NaNs preserved; filter later as needed)
    """
    n_cols = int(activity_df.shape[1])
    if n_cols == 0:
        raise ValueError("Activity matrix has 0 columns; cannot bootstrap.")

    rng = np.random.default_rng(config.seed)

    # Pre-validate inchis present in activity_df
    present_inchis = [i for i in inchis if i in activity_df.index]
    if not present_inchis:
        raise ValueError("None of the provided InChIs were found in the activity matrix index.")

    # Convert each row to a NumPy array once to speed up bootstrapping
    row_arrays: Dict[str, np.ndarray] = {
        inchi: activity_df.loc[inchi].to_numpy(dtype=float, copy=False)
        for inchi in present_inchis
    }

    dists: Dict[str, np.ndarray] = {
        inchi: np.full(config.n_boot, np.nan, dtype=float)
        for inchi in present_inchis
    }

    for b in range(config.n_boot):
        col_idx = rng.integers(
            low=0,
            high=n_cols,
            size=config.n_draw,
            dtype=np.int64,
        )

        for inchi, row in row_arrays.items():
            sampled = row[col_idx]
            # Skip NaNs in the mean; if all sampled values are NaN, mean becomes NaN
            m = np.nanmean(sampled)
            dists[inchi][b] = m

    return dists


def _plot_bootstrap_boxplots(
        dists: Dict[str, np.ndarray],
        labels: Dict[str, str],
        mavs: Dict[str, float],
        outpath: Path,
        *,
        title: str = "Bootstrap distributions of mean activity",
    ) -> None:
    inchis = list(dists.keys())
    n = len(inchis)

    # Filter to finite values per phthalate for plotting
    plot_data: List[np.ndarray] = []
    plot_titles: List[str] = []
    plot_mavs: List[float] = []

    for inchi in inchis:
        vals = dists[inchi]
        finite = vals[np.isfinite(vals)]
        plot_data.append(finite)
        plot_titles.append(labels[inchi])
        plot_mavs.append(float(mavs.get(inchi, np.nan)))

    fig_w = max(
        4.0,
        3.0 * n,
    )
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n,
        figsize=(fig_w, 6.0),
        dpi=120,
        sharey=True,
    )

    if n == 1:
        axes = [axes]

    for ax, data, t, mav in zip(axes, plot_data, plot_titles, plot_mavs):
        ax.boxplot(
            data,
            showfliers=False,
        )

        if np.isfinite(mav):
            mav_str = f"{mav:.4g}"
        else:
            mav_str = "NA"

        ax.set_title(
            f"{t}\n(MAV={mav_str})",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel("Mean activity (bootstrapped)")
    fig.suptitle(title)

    plt.tight_layout()
    outpath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    plt.savefig(outpath)
    print(f"Saved figure to {outpath}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap mean-activity distributions for a list of phthalate InChIs, "
            "using a shared column-resample per bootstrap iteration."
        ),
    )
    parser.add_argument(
        '--cachedir',
        type=str,
        default='cache/entity_similarity',
        help='Directory to cache/load the activity matrix.',
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default=str(fig_path),
        help='Directory to save the boxplot figure and optional outputs.',
    )
    parser.add_argument(
        '--inchi',
        nargs='+',
        default=[],
        help='One or more phthalate InChIs (space-separated).',
    )
    parser.add_argument(
        '--inchi_file',
        type=Path,
        default=None,
        help="Optional file containing InChIs (.txt: one per line; .csv: 'inchi' column).",
    )
    parser.add_argument(
        '--phthalates_csv',
        type=Path,
        default=Path('resources/example_phthalates.csv'),
        help='CSV listing example phthalates (must contain name, inchi).',
    )
    parser.add_argument(
        '--n_boot',
        type=int,
        default=2000,
        help='Number of bootstrap iterations.',
    )
    parser.add_argument(
        '--n_draw',
        type=int,
        default=None,
        help='Number of assay columns to draw (with replacement) per bootstrap iteration. Default: all columns.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='RNG seed for reproducibility.',
    )
    parser.add_argument(
        '--save_long_csv',
        action='store_true',
        help='Also save a long-form CSV of bootstrap means (inchi, iteration, mean_activity).',
    )
    args = parser.parse_args()

    cachedir = Path(args.cachedir)
    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Collect InChIs from CLI and/or file
    inchis: List[str] = [s.strip() for s in args.inchi if str(s).strip()]
    if args.inchi_file is not None:
        inchis.extend(
            _read_inchis_from_file(
                args.inchi_file,
            ),
        )

    # Deduplicate while preserving order
    inchis = list(dict.fromkeys(inchis))
    if not inchis:
        raise ValueError("No InChIs provided. Use --inchi and/or --inchi_file.")

    activity_df = get_activity_df(
        cachedir,
    )

    n_cols = int(activity_df.shape[1])
    n_draw = int(args.n_draw) if args.n_draw is not None else n_cols
    if n_draw <= 0:
        raise ValueError("--n_draw must be > 0.")
    if n_cols == 0:
        raise ValueError("Activity matrix has 0 columns; cannot proceed.")

    config = BootstrapConfig(
        n_boot=int(args.n_boot),
        n_draw=n_draw,
        seed=int(args.seed),
    )

    name_map = _load_example_name_map(
        args.phthalates_csv,
    )

    dists = _compute_bootstrap_means(
        activity_df=activity_df,
        inchis=inchis,
        config=config,
    )

    mavs: Dict[str, float] = {}
    for inchi in dists.keys():
        row = activity_df.loc[inchi].to_numpy(dtype=float, copy=False)
        mavs[inchi] = float(np.nanmean(row))


    # Build labels (name if present in example_phthalates.csv; else canonical SMILES)
    labels: Dict[str, str] = {}
    for inchi in dists.keys():
        labels[inchi] = _label_for_inchi(
            inchi,
            name_map,
        )

    # Optional: save long-form CSV of bootstrap values
    if args.save_long_csv:
        rows: List[Tuple[str, int, float]] = []
        for inchi, vals in dists.items():
            for i, v in enumerate(vals):
                rows.append(
                    (
                        inchi,
                        int(i),
                        float(v),
                    ),
                )
        long_df = pd.DataFrame(
            rows,
            columns=[
                "inchi",
                "iteration",
                "mean_activity",
            ],
        )
        csv_path = outdir / "phthalate_bootstrap_means_long.csv"
        long_df.to_csv(
            csv_path,
            index=False,
        )
        print(f"Saved bootstrap samples to {csv_path}")

    fig_outpath = outdir / "phthalate_activity_bootstrap_boxplots.png"
    _plot_bootstrap_boxplots(
        dists=dists,
        labels=labels,
        mavs=mavs,
        outpath=fig_outpath,
        title="Bootstrap distributions of mean activity (shared assay resample per iteration)",
    )


if __name__ == "__main__":
    main()
