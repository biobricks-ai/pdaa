#!/usr/bin/env python3
"""
Render a grid image of phthalate structures from a CSV with columns:
name, smiles, inchi

Output: phthalates_grid.png (configurable via --out_path)

Notes:
- Prefers SMILES; falls back to InChI if SMILES is missing/invalid.
- Places compound names as legends under each structure.
- Computes 2D coords for consistent depiction.
"""

import argparse
import math
import sys
from typing import List, Tuple
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.inchi import MolFromInchi
from rdkit import RDLogger


# Detects a mono-ester phthalate core: one ring carboxyl is –COOH.
# We key on the substructure c1ccccc1C(=O)OC(=O)O (aromatic ring with one ester and one acid).
_PH_MONO = Chem.MolFromSmarts("c1ccccc1C(=O)OC(=O)O")
def is_mono_phthalate(mol: Chem.Mol) -> bool:
    return mol.HasSubstructMatch(_PH_MONO)

def load_molecules(csv_path: str | Path) -> Tuple[List[Chem.Mol], List[str]]:
    """
    Load molecules from CSV. Returns (mols, names).
    Skips rows that fail both SMILES and InChI parsing.
    """
    df = pd.read_csv(csv_path)

    required_cols = {"name", "smiles", "inchi"}
    missing = required_cols - set(c.lower() for c in df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # Normalize column names just in case of capitalization differences.
    col_map = {c.lower(): c for c in df.columns}
    name_col = col_map["name"]
    smi_col = col_map["smiles"]
    inchi_col = col_map["inchi"]

    mols: List[Chem.Mol] = []
    names: List[str] = []

    for i, row in df.iterrows():
        name = str(row[name_col]).strip()

        mol = None
        smi = str(row[smi_col]).strip() if pd.notna(row[smi_col]) else ""
        if smi:
            mol = Chem.MolFromSmiles(smi, sanitize=True)

        if mol is None:
            inchi = str(row[inchi_col]).strip() if pd.notna(row[inchi_col]) else ""
            if inchi:
                # Requires RDKit built with InChI support.
                mol = MolFromInchi(inchi, treatWarningAsError=False)

        if mol is None:
            print(f"Warning: could not parse row {i} ({name}); skipping.", file=sys.stderr)
            continue

        # Generate 2D coordinates for consistent depiction.
        # We explicitly compute even if RDKit might auto-generate, to keep control.
        rdDepictor.Compute2DCoords(mol)
        mols.append(mol)
        names.append(name)

    if not mols:
        raise ValueError("No valid molecules parsed from CSV.")
    return mols, names


def choose_grid(n: int, max_per_row: int) -> int:
    """
    Heuristic: square-ish grid, capped by max_per_row.
    """
    if n <= 0:
        return 1
    per_row = math.ceil(math.sqrt(n))
    return min(max(1, per_row), max_per_row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a grid of phthalate structures with labels.")
    resourcedir = Path("resources")
    outdir = Path("cache/render_phthalates")
    outdir.mkdir(parents=True, exist_ok=True)
    parser.add_argument("--csv_path", type=Path, default=resourcedir / "example_phthalates.csv",
                        help="Path to input CSV with columns: name, smiles, inchi.")
    parser.add_argument("--out_path", type=Path, default=outdir / "phthalates_grid.png",
                        help="Path for the output image file.")
    parser.add_argument("--mol_img_size", type=int, default=500,
                        help="Size of each cell in pixels (width == height).")
    parser.add_argument("--max_per_row", type=int, default=5,
                        help="Maximum number of molecules per row.")
    parser.add_argument("--legend_font_size", type=int, default=18,
                        help="Legend font size (pixels).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress RDKit warnings.")
    args = parser.parse_args()

    if args.quiet:
        RDLogger.DisableLog("rdApp.*")

    mols, names = load_molecules(args.csv_path)
    mols_per_row = choose_grid(len(mols), args.max_per_row)
    # Add a clear “(mono)” vs “(di)” tag to each legend based on the phthalate core.
    legends = [f"{n} (mono)" if is_mono_phthalate(m) else f"{n} (di)" for m, n in zip(mols, names)]

    # Configure drawing options. We avoid kekulization to reduce failures on tricky aromatics.
    # Legends are drawn under each molecule.
    draw_settings = {
        "molsPerRow": mols_per_row,
        "subImgSize": (args.mol_img_size, args.mol_img_size),
        # "legends": names,
        "legends": legends,
        "useSVG": False,
    }
    img = Draw.MolsToGridImage(
        mols,
        **draw_settings,
        # Note: RDKit >= 2022.03 supports drawOptions; older versions may ignore some fields.
        # We set them defensively via a callback setter.
    )

    # If you need to enforce legend font size on older RDKit, use the MolDraw2D path instead.
    # Here we attempt to set global defaults where available.
    try:
        opts = Draw.rdMolDraw2D.MolDrawOptions()
        opts.legendFontSize = args.legend_font_size
        Draw.SetMolsToGridImageOptions(opts)
    except Exception:
        # Safe fallback if RDKit version does not expose this.
        pass

    # Re-render with options applied where supported.
    img = Draw.MolsToGridImage(
        mols,
        **draw_settings,
    )

    img.save(args.out_path)
    print(f"Wrote {args.out_path} with {len(mols)} structures in a {mols_per_row}-per-row grid.")


if __name__ == "__main__":
    main()
