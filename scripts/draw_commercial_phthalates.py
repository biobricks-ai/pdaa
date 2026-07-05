"""
Draw structures of the Top 20 commercial phthalates and verify MAV lookup.

For each phthalate in commercial_top20.csv, converts the SMILES to an RDKit
molecule, draws the 2D structure, converts to InChI, and looks up the MAV in
the activity matrix.  Produces a grid image with abbreviation, LCB, and MAV
annotated under each structure.

Usage:
    python scripts/draw_commercial_phthalates.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image, ImageDraw, ImageFont

sys.path.append('./')
from scripts.utils.helpers import longest_carbon_backbone, largest_carbon_backbone

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
COMMERCIAL_CSV = PROJECT_ROOT / "resources" / "phthalate_lists" / "commercial_top20.csv"
ACTIVITY_PATH = PROJECT_ROOT / "cache" / "entity_similarity" / "activity_matrix_filled.parquet"
OUTPUT_DIR = PROJECT_ROOT / "cache" / "wide_groupings"


def load_mav():
    activity = pd.read_parquet(ACTIVITY_PATH)
    Z = (activity - activity.mean(axis=0)) / activity.std(axis=0)
    mav = Z.mean(axis=1)
    mav.name = "MAV"
    return activity, mav


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(COMMERCIAL_CSV)
    activity_raw, mav_z = load_mav()
    logger.info("Loaded %d phthalates from CSV, %d compounds in activity matrix",
                len(df), len(activity_raw))

    cell_w, cell_h = 400, 480
    label_h = 80
    img_h = cell_h - label_h
    ncols = 5
    nrows = (len(df) + ncols - 1) // ncols

    mols = []
    labels = []
    lookup_report = []

    for _, row in df.iterrows():
        abbrev = row["abbreviation"]
        smiles = row["smiles"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Invalid SMILES for %s: %s", abbrev, smiles)
            mols.append(None)
            labels.append((abbrev, "invalid SMILES", ""))
            lookup_report.append(f"  {abbrev}: INVALID SMILES")
            continue

        AllChem.Compute2DCoords(mol)

        inchi = Chem.MolToInchi(mol)
        csv_inchi = row.get("inchi", "")

        if pd.notna(csv_inchi) and csv_inchi and inchi != csv_inchi:
            logger.warning("%s: SMILES-derived InChI differs from CSV inchi column", abbrev)
            lookup_report.append(f"  {abbrev}: InChI MISMATCH between SMILES and CSV column")

        lcb = longest_carbon_backbone(mol)
        lrg = largest_carbon_backbone(mol)

        in_matrix = inchi in activity_raw.index
        raw_mav = activity_raw.loc[inchi].mean() if in_matrix else np.nan
        z_mav = mav_z.loc[inchi] if in_matrix else np.nan

        status = "OK" if in_matrix else "NOT FOUND"
        lookup_report.append(
            f"  {abbrev:<6} InChI {'FOUND' if in_matrix else 'MISSING':>7}  "
            f"LCB={lcb:>2}  Largest={lrg:>2}  "
            f"rawMAV={raw_mav:>7.4f}  zMAV={z_mav:>7.4f}" if in_matrix else
            f"  {abbrev:<6} InChI MISSING  LCB={lcb:>2}  Largest={lrg:>2}"
        )

        mols.append(mol)
        line2 = f"LCB={lcb}  Largest={lrg}"
        if in_matrix:
            line3 = f"rawMAV={raw_mav:.3f}  zMAV={z_mav:+.3f}"
        else:
            line3 = "not in activity matrix"
        labels.append((abbrev, line2, line3))

    logger.info("Lookup report:\n%s", "\n".join(lookup_report))

    grid = Image.new("RGB", (ncols * cell_w, nrows * cell_h), "white")
    draw = ImageDraw.Draw(grid)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font_title = ImageFont.load_default()
        font_body = font_title

    for i, (mol, (abbrev, line2, line3)) in enumerate(zip(mols, labels)):
        col = i % ncols
        row_idx = i // ncols
        x0 = col * cell_w
        y0 = row_idx * cell_h

        if mol is not None:
            mol_img = Draw.MolToImage(mol, size=(cell_w, img_h))
            grid.paste(mol_img, (x0, y0))

        label_y = y0 + img_h
        draw.text((x0 + cell_w // 2, label_y + 2), abbrev,
                  fill="black", font=font_title, anchor="mt")
        draw.text((x0 + cell_w // 2, label_y + 26), line2,
                  fill="#444444", font=font_body, anchor="mt")
        draw.text((x0 + cell_w // 2, label_y + 48), line3,
                  fill="#666666", font=font_body, anchor="mt")

        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1],
                       outline="#cccccc", width=1)

    outpath = OUTPUT_DIR / "commercial_top20_structures.png"
    grid.save(outpath, dpi=(150, 150))
    logger.info("Saved %s", outpath)


if __name__ == "__main__":
    main()
