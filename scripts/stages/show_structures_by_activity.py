from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image, ImageDraw, ImageFont  # pillow

import sys
sys.path.append('./')
from scripts.utils.helpers import zscore_columns

def _compute_row_means(df: pd.DataFrame) -> pd.Series:
    # Mean across numeric assay columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric assay columns found.")
    return df[numeric_cols].mean(axis=1)

def _pick_groups_by_activity(df_with_mean: pd.DataFrame, N: int) -> Tuple[List[str], List[str], List[str], Dict[str, float]]:
    # df_with_mean is df with a column "_mean_activity"
    s = df_with_mean["_mean_activity"]
    ordered = s.sort_values(ascending=True)
    n_total = len(ordered)
    if N <= 0:
        raise ValueError("N must be >= 1.")
    if 3 * N > n_total:
        N = max(1, n_total // 3)  # tighten to fit

    # Least and most
    least_idx = list(ordered.head(N).index)
    most_idx = list(ordered.tail(N).index)

    # Middle: choose N closest to the median activity
    median_val = float(s.median())
    middle_idx = list((s - median_val).abs().sort_values().head(N).index)
    # Sort each group by mean for a consistent visual order
    least_idx  = list(ordered.loc[least_idx].sort_values().index)
    middle_idx = list(ordered.loc[middle_idx].sort_values().index)
    most_idx   = list(ordered.loc[most_idx].sort_values().index)

    # Map inchi -> mean for labeling
    means = {inchi: float(s.loc[inchi]) for inchi in ordered.index}
    return least_idx, middle_idx, most_idx, means

def make_activity_panel(df: pd.DataFrame, N: int = 5, out_path: str = "activity_panel.png"):
    # 1) compute means and groups
    means = _compute_row_means(df)
    df2 = df.copy()
    df2["_mean_activity"] = means

    least, middle, most, mean_map = _pick_groups_by_activity(df2, N)

    # 2) draw each column via RDKit
    def _to_mol(inchi: str):
        # MolFromInchi returns None if parse fails or InChI support is missing
        mol = Chem.MolFromInchi(inchi, sanitize=True)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        return mol

    def _legend(inchi: str) -> str:
        # Keep the legend short and deterministic: mean and optional InChIKey
        val = mean_map[inchi]
        return f"MAV = {val:.3f}"
        # key = None
        # try:
        #     m = Chem.MolFromInchi(inchi, sanitize=False)
        #     if m is not None:
        #         key = Chem.inchi.MolToInchiKey(m)
        # except Exception:
        #     key = None
        # if key:
        #     return f"{val:.3f} | {key}"
        # # Fallback: shortened InChI head
        # head = inchi.split("/", 1)[0]  # e.g., "InChI=1S"
        # return f"{val:.3f} | {head}"

    def _column_image(inchis: List[str], title: str):
        mols = []
        legends = []
        for inc in inchis:
            mol = _to_mol(inc)
            mols.append(mol)  # RDKit accepts None placeholders; they'll render blank
            legends.append(_legend(inc))
        # One molecule per row, uniform tile size
        col_img = Draw.MolsToGridImage(
            mols,
            molsPerRow=1,
            subImgSize=(320, 260),
            legends=legends,
            useSVG=False  # PIL image out
        )
        # # Add a simple header above the column
        # # Convert to PIL Image if rdkit returns a PIL Image already this is a no-op.
        # if not isinstance(col_img, Image.Image):
        #     col_img = col_img  # RDKit on recent versions returns PIL Image
        # # Create header + column stack
        # header_h = 40
        # stacked = Image.new("RGB", (col_img.width, col_img.height + header_h), "white")
        # draw = ImageDraw.Draw(stacked)
        # draw.text((10, 10), title, fill="black")
        # stacked.paste(col_img, (0, header_h))
        # return stacked

        # Add a centered, bold header above the column
        if not isinstance(col_img, Image.Image):
            col_img = col_img  # RDKit on recent versions returns PIL Image

        # Try to use a truetype bold font; fall back to default and simulate bold
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=28)
            draw_bold = None  # real bold available
        except Exception:
            font = ImageFont.load_default()
            def draw_bold(d, xy, text, fill, font):
                # Simulate bold with small offsets
                x, y = xy
                for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    d.text((x + dx, y + dy), text, fill=fill, font=font)

        # Measure text and compute header height with padding
        tmp = Image.new("RGB", (col_img.width, 60), "white")
        dtmp = ImageDraw.Draw(tmp)
        bbox = dtmp.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad_y = 8
        line_pad = 10  # reserve space for a global underline
        header_h = text_h + 2 * pad_y + line_pad

        stacked = Image.new("RGB", (col_img.width, col_img.height + header_h), "white")
        draw = ImageDraw.Draw(stacked)
        tx = (col_img.width - text_w) // 2  # centered
        ty = pad_y
        if 'draw_bold' in locals() and callable(draw_bold):
            draw_bold(draw, (tx, ty), title, fill="black", font=font)
        else:
            draw.text((tx, ty), title, fill="black", font=font)

        # Do not draw the underline here; we will draw one continuous line on the final panel.
        stacked.paste(col_img, (0, header_h))
        return stacked, header_h - (line_pad // 2)



    col_least, y_line = _column_image(least,  "Least Active")
    col_middle, _     = _column_image(middle, "Middle Active")
    col_most, _       = _column_image(most,   "Most Active")

    # 3) assemble final 3-column panel
    gutter = 20
    w = col_least.width + col_middle.width + col_most.width + 2 * gutter
    h = max(col_least.height, col_middle.height, col_most.height)
    panel = Image.new("RGB", (w, h), "white")
    x = 0
    for col in (col_least, col_middle, col_most):
        panel.paste(col, (x, 0))
        x += col.width + gutter

    draw = ImageDraw.Draw(panel)

    # One continuous horizontal line under all headers
    draw.line((0, y_line, w, y_line), fill=(180, 180, 180), width=2)

    # Vertical separator lines centered in the gutters
    x1 = col_least.width + gutter // 2
    x2 = col_least.width + gutter + col_middle.width + gutter // 2
    draw.line((x1, 0, x1, h), fill=(180, 180, 180), width=2)
    draw.line((x2, 0, x2, h), fill=(180, 180, 180), width=2)

    panel.save(out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create an activity panel from a DataFrame.")
    parser.add_argument("--input_data", default="cache/entity_similarity/activity_matrix_filled.parquet", help="Path to input parquet file with InChI and assay data.")
    parser.add_argument("--N", type=int, default=5, help="Number of compounds per group (default: 5).")
    parser.add_argument("--outdir", default="cache/entity_similarity", help="Output directory.")
    parser.add_argument("--out_file", default="activity_panel.png", help="Output image path (default: activity_panel.png).")
    
    args = parser.parse_args()
    
    df = pd.read_parquet(args.input_data)
    Z, _ = zscore_columns(df)
    out_path = Path(args.outdir) / args.out_file
    img_path = make_activity_panel(Z, N=args.N, out_path=out_path)
    print("Saved:", out_path)