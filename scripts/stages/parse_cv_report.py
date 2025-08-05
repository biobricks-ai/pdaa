#!/usr/bin/env python3
"""
Parse XGB-classifier cross-validation report and print summary table.

Expected file format (excerpt):
    Processed LogRP with 1234 values, adjusted endpoint: LogRP
    ...
    Mean CV accuracy: 0.887 ± 0.010 (SE)
    Mean CV ROC-AUC: 0.849 ± 0.008 (SE)

Table columns:
    Endpoints   N   Accuracy    ROC-AUC
The number in parentheses is the SE (standard error) scaled ×1000, e.g.
    0.887 ± 0.010  →  0.887(10)
"""
import math
import re
import argparse
import pandas as pd
from pathlib import Path


_METRICS_RE = {
    "accuracy": re.compile(r"Mean CV accuracy:\s+([\d.]+)\s+±\s+([\d.]+)"),
    "roc_auc":  re.compile(r"Mean CV ROC-AUC:\s+([\d.]+)\s+±\s+([\d.]+)")
}
_PROCESSED_RE = re.compile(
    r"Processed\s+\S+\s+with\s+(\d+)\s+values,\s+adjusted endpoint:\s+(\S+)",
    re.IGNORECASE
)


# def print_mu_with_err(mu: float, err: float) -> str:
#     """
#     Return Mean (mu) with Error (err) in parentheses.
#     The last digit of the error aligns with the last digit of the mean.

#     e.g., 0.887 ± 0.010 → 0.887(10)
#     """
#     get_sci_str = lambda n: f"{n:e}"
#     get_exp = lambda s: int(s.split('e')[1]) if 'e' in s else 0

#     mu_str = get_sci_str(mu)
#     err_str = get_sci_str(err)
#     mu_exp = get_exp(mu_str)
#     err_exp = get_exp(err_str)

def print_mu_with_err(mu: float, err: float) -> str:
    """
    Return Mean (mu) with Error (err) in parentheses, following the
    “last-digit alignment” convention, e.g. 0.887 ± 0.010 → 0.887(10).

    Algorithm:
      - Round the error to one significant digit, unless its leading
        digit is 1 (then keep two).
      - Round the mean to the same decimal place.
      - Convert the rounded error to an integer whose digits align with
        the last digits of the mean.
    """
    
    if err == 0 or math.isnan(err):
        return f"{mu}"

    err_abs = abs(err)
    err_exp = math.floor(math.log10(err_abs))
    leading = err_abs / 10 ** err_exp

    sig_digits = 2 if leading < 2 else 1          # CODATA rule
    dec_places = -err_exp + (sig_digits - 1)      # decimal places to show

    err_round = round(err_abs, dec_places)
    mu_round = round(mu, dec_places)

    scale = 10 ** max(dec_places, 0)
    err_digits = int(round(err_round * scale))

    if dec_places > 0:
        mu_str = f"{mu_round:.{dec_places}f}"
    else:                                         # dec_places ≤ 0 → tens, hundreds…
        mu_str = f"{int(mu_round):d}"

    return f"{mu_str}({err_digits})"



def parse_report(path: Path) -> pd.DataFrame:
    rows = []
    current = {}

    for line in path.read_text().splitlines():
        if m := _PROCESSED_RE.match(line):
            # start of a new block
            current = {"N": int(m.group(1)), "Endpoints": m.group(2)}
        elif m := _METRICS_RE["accuracy"].match(line):
            # current["Accuracy"] = f"{float(m.group(1)):.3f}({_scale_se(float(m.group(2)))})"
            current["Accuracy"] = print_mu_with_err(float(m.group(1)), float(m.group(2)))
        elif m := _METRICS_RE["roc_auc"].match(line):
            current["ROC-AUC"] = print_mu_with_err(float(m.group(1)), float(m.group(2)))
            rows.append(current)          # end of block – store it

    df = pd.DataFrame(
        rows, columns=["Endpoints", "N", "Accuracy", "ROC-AUC"]
    ).sort_values(by=["N"], ascending=[False])

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse XGB-classifier cross-validation report and print summary table."
    )
    parser.add_argument(
        "--infile", "-i",
        type=Path,
        required=True,
        help="Path to the XGB-classifier cross-validation report file."
    )
    args = parser.parse_args()
    # infile = Path("cache/edkb/xgb_classifier_feature_selection.txt")  # rename if needed
    df = parse_report(args.infile)
    # Pretty print: no index, left-justified columns
    print(df.to_string(index=False, justify="left"))
