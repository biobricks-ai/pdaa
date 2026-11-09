"""
Isomer-envelope (robustness) analysis for the commercial UVCB phthalates.

Each of DEHP/DINP/DIDP/DIUP/DTDP is scored in the manuscript by a single
activity-matrix isomer. This script asks how sensitive the MAV is to that choice
by examining the full set of activity-matrix isomers of each molecular formula
(all scored by the same ChemProp predict_all endpoint -- no cross-method artifact).

Two systematic effects are quantified:

  1. Stereochemistry artifact. Within every formula, isomers carrying a defined
     InChI stereo layer score systematically higher than achiral isomers, even
     though remote alkyl stereocenters have no plausible bioactivity effect. This
     is a data/model artifact. Commercial UVCBs are stereochemically undefined
     mixtures, so the faithful comparison uses achiral (flat) isomers only.

  2. Isomer sensitivity of the point estimates. The single representative used per
     UVCB is compared against the achiral-isomer distribution of its formula.

The separability of the two longest-chain UVCBs (DIUP vs DTDP) is tested on the
achiral envelopes.

Outputs:
    cache/isomer_envelope/isomer_envelope.png        - achiral-isomer distribution figure
    cache/isomer_envelope/isomer_envelope_stats.json - quartiles, stereo effect, overlap
"""
import json
import logging
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import mannwhitneyu

from pathlib import Path
import os as _os
_MATRIX_OVERRIDE = _os.environ.get("PDAA_ACTIVITY_MATRIX")


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CACHEDIR = pathlib.Path("cache/entity_similarity")
OUTDIR = pathlib.Path("cache/isomer_envelope")

# Representative structures are read from the canonical commercial list rather than
# hard-coded, so they cannot drift out of step with it. A hard-coded DTDP previously
# lagged the 2026-07-17 revision to the internally di-branched convention, which put
# this figure on a different structure from the rest of the analysis.
_COMMERCIAL = pathlib.Path("resources/phthalate_lists/commercial_top20.csv")
_NOMINAL_C = {"DEHP": 8, "DINP": 9, "DIDP": 10, "DIUP": 11, "DTDP": 13}


def _load_uvcbs():
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    df = pd.read_csv(_COMMERCIAL).set_index("abbreviation")
    out = []
    for ab, nominal in _NOMINAL_C.items():
        smi = df.loc[ab, "smiles"]
        mol = Chem.MolFromSmiles(smi)
        out.append({"abbrev": ab, "formula": rdMolDescriptors.CalcMolFormula(mol),
                    "nominal_c": nominal, "smiles": smi})
    return out


UVCBS = _load_uvcbs()
PLOT_ORDER = ["DEHP", "DINP", "DIDP", "DIUP", "DTDP"]
BOX_COLOR = "#7eb8da"
REP_COLOR = "#e07b54"


def formula_from_inchi(inchi: str) -> str | None:
    m = re.match(r"InChI=1S?/([A-Za-z0-9]+)/", inchi)
    return m.group(1) if m else None


def has_defined_stereo(inchi: str) -> bool:
    """True if the InChI carries any defined stereochemistry.

    The earlier test required /t together with /m or /s. An InChI with relative
    tetrahedral stereochemistry carries /t alone (e.g. .../t23-,24+ with no /m or /s),
    so that test let stereo-defined isomers into the achiral set. Because such isomers
    score systematically higher, a single leaked structure could dominate a formula's
    envelope: the C28H46O4 (DIDP) envelope maximum of +2.38 was one, and it set the
    y-axis of the UVCB figure. Any /t or /b layer now counts as defined stereochemistry.
    """
    return "/t" in inchi or "/b" in inchi


def load_mav() -> pd.Series:
    activity = pd.read_parquet(Path(_MATRIX_OVERRIDE) if _MATRIX_OVERRIDE else CACHEDIR / "activity_matrix_filled.parquet")
    z = (activity - activity.mean(axis=0)) / activity.std(axis=0)
    mav = z.mean(axis=1)
    mav.name = "MAV"
    logger.info("Computed MAV for %d compounds across %d assays",
                activity.shape[0], activity.shape[1])
    return mav


def representative_mav(smiles: str, mav: pd.Series) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchi = Chem.MolToInchi(mol)
    return float(mav.loc[inchi]) if inchi in mav.index else None


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta; positive means b tends to exceed a."""
    gt = sum((bi > a).sum() for bi in b)
    lt = sum((bi < a).sum() for bi in b)
    return (gt - lt) / (len(a) * len(b))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    mav = load_mav()
    formula = mav.index.to_series().map(formula_from_inchi)
    stereo = mav.index.to_series().map(has_defined_stereo)

    envelopes = {}
    stats = {}
    for u in UVCBS:
        f_mask = formula.values == u["formula"]
        flat_mask = f_mask & (~stereo.values)
        flat_vals = mav[flat_mask].values
        stereo_vals = mav[f_mask & stereo.values].values
        rep = representative_mav(u["smiles"], mav)
        envelopes[u["abbrev"]] = {"flat": flat_vals, "rep": rep, **u}

        q = np.percentile(flat_vals, [0, 25, 50, 75, 100]) if len(flat_vals) else [np.nan] * 5
        # rank of representative within the achiral envelope (1 = lowest)
        rep_rank = (int((flat_vals < rep).sum()) + 1) if rep is not None and len(flat_vals) else None
        stats[u["abbrev"]] = {
            "formula": u["formula"],
            "nominal_carbon": u["nominal_c"],
            "n_isomers_total": int(f_mask.sum()),
            "n_isomers_achiral": int(len(flat_vals)),
            "n_isomers_stereo": int(len(stereo_vals)),
            "achiral_min": float(q[0]), "achiral_q1": float(q[1]),
            "achiral_median": float(q[2]), "achiral_q3": float(q[3]), "achiral_max": float(q[4]),
            "achiral_mean": float(np.mean(flat_vals)) if len(flat_vals) else None,
            "stereo_median": float(np.median(stereo_vals)) if len(stereo_vals) else None,
            "representative_mav": rep,
            "representative_rank_in_achiral": rep_rank,
        }
        logger.info("%s (%s, C%d): achiral n=%d median=%.3f IQR=[%.3f,%.3f] | stereo n=%d median=%s | rep=%.3f rank=%s/%d",
                    u["abbrev"], u["formula"], u["nominal_c"], len(flat_vals), q[2], q[1], q[3],
                    len(stereo_vals), f"{np.median(stereo_vals):.3f}" if len(stereo_vals) else "NA",
                    rep if rep is not None else float("nan"),
                    rep_rank, len(flat_vals))

    # --- DIUP vs DTDP separability on achiral envelopes ---
    diup = envelopes["DIUP"]["flat"]
    dtdp = envelopes["DTDP"]["flat"]
    delta = cliffs_delta(diup, dtdp)
    p_sup = sum((di < dtdp).sum() for di in diup) / (len(diup) * len(dtdp))
    u_stat, u_p = mannwhitneyu(dtdp, diup, alternative="two-sided")
    stats["DIUP_vs_DTDP_achiral"] = {
        "n_diup": int(len(diup)), "n_dtdp": int(len(dtdp)),
        "cliffs_delta_DTDP_vs_DIUP": float(delta),
        "prob_DTDP_gt_DIUP": float(p_sup),
        "mannwhitney_U": float(u_stat), "mannwhitney_p": float(u_p),
    }
    logger.info("DIUP vs DTDP (achiral): Cliff's d=%.3f  P(DTDP>DIUP)=%.2f  MWU p=%.3f",
                delta, p_sup, u_p)

    # Achiral median ordering (the robust, faithful ranking)
    order = sorted(PLOT_ORDER, key=lambda a: stats[a]["achiral_median"], reverse=True)
    stats["achiral_median_ordering"] = " > ".join(
        f"{a} ({stats[a]['achiral_median']:.2f})" for a in order)
    logger.info("Achiral median ordering: %s", stats["achiral_median_ordering"])

    with open(OUTDIR / "isomer_envelope_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Wrote %s", OUTDIR / "isomer_envelope_stats.json")

    # --- Figure: achiral-isomer envelopes with representative overlaid ---
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    data = [envelopes[a]["flat"] for a in PLOT_ORDER]
    positions = range(1, len(PLOT_ORDER) + 1)
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color="#22384a", lw=1.6),
                    whiskerprops=dict(color="#5a5a5a"), capprops=dict(color="#5a5a5a"))
    for patch in bp["boxes"]:
        patch.set_facecolor(BOX_COLOR)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#4a4a4a")

    rng = np.random.default_rng(42)
    for i, a in enumerate(PLOT_ORDER, start=1):
        vals = envelopes[a]["flat"]
        jitter = rng.uniform(-0.16, 0.16, len(vals))
        ax.scatter(i + jitter, vals, s=14, color="#2c5f7c", alpha=0.4,
                   edgecolors="none", zorder=3)
        rep = envelopes[a]["rep"]
        if rep is not None:
            ax.scatter(i, rep, marker="D", s=75, color=REP_COLOR, zorder=6,
                       edgecolors="white", linewidth=0.7,
                       label="Manuscript representative" if i == 1 else "")

    ax.axhline(0.0, color="#999999", ls="--", lw=0.8, zorder=1)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(
        [f"{a}\n({envelopes[a]['formula']}, n={len(envelopes[a]['flat'])})" for a in PLOT_ORDER],
        fontsize=9)
    ax.set_ylabel("MAV (z-scored across assays)", fontsize=11)
    ax.set_title("MAV across achiral activity-matrix isomers of each UVCB formula",
                 fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", ls="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    outpath = OUTDIR / "isomer_envelope.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved %s", outpath)


if __name__ == "__main__":
    main()
