"""
Generate tables of Current MAV vs three adjusted MAV variants for the
top 20 commercial phthalates.

Adjusted MAV 1 — AI-classified DART/ED relevance (strength > 8 on 1–10 scale
    from ToxTransformer categories in predict_all.json).
Adjusted MAV 2 — Manual keyword-based DART/ED classification using a
    consistent rule set applied to all 975 assay titles.
Adjusted MAV 3 — Intersection of assays in the activity matrix with the
    CMRE assay list (fuzzy-matched by stripping non-alphanumeric characters).

Each table is produced in two versions: raw and z-scored.

Usage:
    python scripts/adjusted_mav_tables.py
"""
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.append('./')

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
COMMERCIAL_CSV = PROJECT_ROOT / "resources" / "phthalate_lists" / "commercial_top20.csv"
ACTIVITY_PATH = PROJECT_ROOT / "cache" / "entity_similarity" / "activity_matrix_filled.parquet"
PREDICT_JSON = PROJECT_ROOT / "resources" / "predict_all.json"
ASSAY_STRENGTH = PROJECT_ROOT / "resources" / "assay_strength.csv"
CMRE_ASSAYS = Path("/home/john/Documents/Insilica/astra/cache_analogs/resources/cmre_assays.txt")
OUTPUT_DIR = PROJECT_ROOT / "cache" / "adjusted_mav"


def _normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())


def build_dart_ed_strength_map(am_columns: list[str]) -> dict[str, float]:
    """Map each AM column to its max DART/ED strength from predict_all.json."""
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


def classify_dart_ed_manual(am_columns: list[str]) -> set[str]:
    """Keyword-based DART/ED classification of assay titles.

    An assay is classified as DART/ED-relevant if its title matches any of
    the positive patterns AND does not match the exclusion patterns.

    Positive patterns target:
        - Estrogen, androgen, progesterone, thyroid, glucocorticoid,
          mineralocorticoid receptor assays
        - Aromatase / steroidogenesis assays
        - Steroid biosynthesis enzymes (HSD, CYP11, CYP17, CYP19)
        - H295R steroidogenesis assays
        - Developmental / reproductive toxicity endpoints
        - Endocrine disruption assays
        - PPAR-gamma (metabolic endocrine link)
        - Retinoic acid receptor (developmental role)
        - AhR / aryl hydrocarbon receptor (ED and developmental pathway)
        - Vitamin D receptor (calcium/developmental signaling)
        - Gonadotropin / GnRH receptor assays
        - Oxytocin receptor assays
        - Prolactin / growth hormone related
        - Zebrafish developmental / teratogenicity assays
        - CAR/PXR nuclear receptors (xenobiotic metabolism, ED-adjacent)
        - FXR, LXR, RXR nuclear receptors (cross-talk with steroid signaling)
        - ERR (estrogen-related receptor)
        - NR0B1/DAX-1, SF-1/NR5A1 (reproductive development)

    Excluded pathways that are general-toxicity rather than DART/ED-specific:
        p53 (genotoxicity), ARE/Nrf2 (oxidative stress), HSE/heat shock
        (proteotoxicity), XBP1/UPR (ER stress), AP-1 (general transcription),
        TGF-beta (growth/cancer), bile acid receptor, insulin/IGF receptors.
    """
    positive_patterns = [
        # --- Steroid hormone receptors ---
        r'estrogen\s*receptor',
        r'\ber[\s_-]?(alpha|beta|α|β)\b',
        r'\bera\b',
        r'\berb\b',
        r'\ber[\s_-]luc\b',
        r'\ber[\s_-]bla\b',
        r'eralpha|erbeta|erα|erβ',
        r'\bere[\s_-]',
        r'androgen\s*receptor',
        r'\bar[\s_-]?(agonist|antagonist|bla|luc|mda|tif)',
        r'\bar[\s_-]bla\b',
        r'progesterone\s*receptor',
        r'\bpr[\s_-]bla\b',
        r'glucocorticoid\s*receptor',
        r'\bgr[\s_-]bla\b',
        r'mineralocorticoid',
        # --- Thyroid ---
        r'thyroid',
        r'\btsh\b',
        r'\btshr\b',
        r'\btr[\s_-]luc\b',
        # --- Steroidogenesis enzymes ---
        r'aromatase',
        r'steroidogen',
        r'h295r',
        r'hydroxysteroid\s*dehydrogenase',
        r'\bhsd\b',
        r'cyp11b|cyp17|cyp19',
        r'steroid.*alpha.*hydroxylase',
        r'steroid.*dehydrogenase',
        r'5.*alpha.*reductase',
        r'steryl.sulfatase',
        # --- Developmental / reproductive endpoints ---
        r'developmental\s*tox',
        r'reproductive\s*tox',
        r'endocrine\s*disrupt',
        r'teratogen',
        r'developmental\s*loael',
        r'developmental\s*noael',
        r'reproductive.*loael',
        r'reproductive.*noael',
        r'oral\s*reproductive',
        r'neonatal',
        r'pregnancy',
        r'congenital.*familial.*genetic',
        r'zebrafish.*(embryo|develop|teratogen|jaw|mortality)',
        # --- PPAR-gamma (metabolic ED link) ---
        r'\bppar[\s_-]?g',
        r'\bpparg\b',
        r'ppargamma|pparγ',
        r'peroxisome\s*proliferator.*gamma',
        # --- RAR / RXR (developmental signaling) ---
        r'retinoic\s*acid\s*receptor',
        r'\brar\b',
        r'\brxr\b',
        r'retinoid\s*x\s*receptor',
        # --- AhR (established ED and developmental pathway) ---
        r'aryl\s*hydrocarbon\s*receptor',
        r'\bahr\b',
        # --- VDR (developmental / reproductive signaling) ---
        r'vitamin\s*d.*receptor',
        r'\bvdr\b',
        # --- Reproductive endocrine receptors ---
        r'gonadotropin',
        r'\bgnrh\b',
        r'oxytocin\s*receptor',
        r'hormone.*receptor',
        r'hormonal\s*disrupt',
        r'prolactin',
        r'cortisol',
        r'testosterone',
        r'androgen',
        r'progesterone',
        r'estradiol',
        # --- ED-relevant nuclear receptors ---
        r'\bcar[\s_-](agonist|antagonist)\b',
        r'constitutive\s*androstane',
        r'pregnane\s*x\s*receptor',
        r'\bpxr\b',
        r'\berr\b',
        r'estrogen.*related.*receptor',
        r'nr0b1|dax[\s_-]?1',
        r'sf[\s_-]?1\b',
        r'nr5a',
        r'\bror[\s_-]?(gamma|g)\b',
        r'rorg\b',
        r'nuclear\s*receptor\s*ror',
        r'er[\s_-]?(gfp|egfp)',
        r'cell\s*viability.*hormone',
    ]

    exclusion_patterns = [
        r'nci[\s_-]60',
        r'nci[\s_-]yeast',
        r'nci.*human\s*tumor\s*cell',
        r'nci.*in\s*vivo\s*anticancer',
        r'pediatric\s*cancer',
        r'drug\s*repurpos',
        r'ewing',
        r'leukemia\s*cell.*growth\s*inhibit',
        r'melanoma.*growth\s*inhibit',
        r'colon.*cell.*growth\s*inhibit',
        r'breast.*cell.*growth\s*inhibit',
        r'renal.*cell.*growth\s*inhibit',
        r'prostate.*cell.*growth\s*inhibit',
        r'ovarian.*cell.*growth\s*inhibit',
        r'lung.*growth\s*inhibit',
        r'glioma.*cell',
        r'glioblastoma',
        r'pontine\s*glioma',
        r'sarcoma',
        r'carcinoma',
        r'merkel\s*cell',
        r'cell\s*proliferation.*molecular\s*libraries',
        r'mutagenicity\s*assay',
        r'salmonella\s*mutagenicity',
        r'carcinogenic.*potency.*database',
        r'dsstox.*carcinogen',
        r'dsstox.*mutagen',
        r'ccris',
        r'anticancer',
    ]

    dart_ed_positive_re = re.compile('|'.join(positive_patterns), re.IGNORECASE)
    exclusion_re = re.compile('|'.join(exclusion_patterns), re.IGNORECASE)

    relevant = set()
    for col in am_columns:
        title = col.strip()
        if dart_ed_positive_re.search(title) and not exclusion_re.search(title):
            relevant.add(col)
    return relevant


def build_cmre_mask(am_columns: list[str]) -> set[str]:
    """Return AM columns that match CMRE assay names."""
    with open(CMRE_ASSAYS) as f:
        cmre_raw = [line.strip().strip('"').strip('[]') for line in f if line.strip()]

    cmre_norm = {_normalize(a) for a in cmre_raw}

    matched = set()
    for col in am_columns:
        if _normalize(col.strip()) in cmre_norm:
            matched.add(col)
    return matched


def compute_mav(activity: pd.DataFrame, columns: list[str] | None = None,
                zscore: bool = False) -> pd.Series:
    """Compute MAV over the specified columns."""
    if columns is not None:
        sub = activity[columns]
    else:
        sub = activity

    if zscore:
        sub = (sub - sub.mean(axis=0)) / sub.std(axis=0)

    return sub.mean(axis=1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    activity = pd.read_parquet(ACTIVITY_PATH)
    am_cols = list(activity.columns)
    logger.info("Activity matrix: %d compounds x %d assays", *activity.shape)

    # --- Adjusted MAV 1: ToxTransformer DART/ED strength > 8 ---
    col_strength = build_dart_ed_strength_map(am_cols)
    adj1_cols = [c for c, s in col_strength.items() if s >= 8.0]
    logger.info("Adjusted MAV 1: %d assays with DART/ED strength >= 8", len(adj1_cols))

    # --- Adjusted MAV 2: Manual keyword classification ---
    adj2_cols = list(classify_dart_ed_manual(am_cols))
    logger.info("Adjusted MAV 2: %d assays classified as DART/ED-relevant", len(adj2_cols))

    # --- Adjusted MAV 3: CMRE intersection ---
    cmre_set = build_cmre_mask(am_cols)
    adj3_cols = list(cmre_set)
    logger.info("Adjusted MAV 3: %d assays in AM ∩ CMRE", len(adj3_cols))

    # --- Load commercial phthalates ---
    top20 = pd.read_csv(COMMERCIAL_CSV)
    phthalate_inchis = {}
    for _, row in top20.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None:
            continue
        inchi = Chem.MolToInchi(mol)
        if inchi in activity.index:
            phthalate_inchis[row['abbreviation']] = inchi
        else:
            logger.warning("%s (InChI from SMILES) not in activity matrix", row['abbreviation'])

    logger.info("Found %d / %d phthalates in activity matrix", len(phthalate_inchis), len(top20))

    # --- Build both tables ---
    for zscore, suffix in [(False, 'raw'), (True, 'zscore')]:
        current = compute_mav(activity, zscore=zscore)
        mav1 = compute_mav(activity, adj1_cols, zscore=zscore) if adj1_cols else None
        mav2 = compute_mav(activity, adj2_cols, zscore=zscore) if adj2_cols else None
        mav3 = compute_mav(activity, adj3_cols, zscore=zscore) if adj3_cols else None

        rows = []
        for abbrev, inchi in phthalate_inchis.items():
            rows.append({
                'Phthalate': abbrev,
                'Current MAV': current.loc[inchi],
                'Adjusted MAV 1': mav1.loc[inchi] if mav1 is not None else np.nan,
                'Adjusted MAV 2': mav2.loc[inchi] if mav2 is not None else np.nan,
                'Adjusted MAV 3': mav3.loc[inchi] if mav3 is not None else np.nan,
            })

        table = pd.DataFrame(rows)
        csv_path = OUTPUT_DIR / f"adjusted_mav_{suffix}.csv"
        table.to_csv(csv_path, index=False, float_format='%.6f')
        logger.info("Saved %s", csv_path)

        print(f"\n{'=' * 70}")
        print(f"  MAV Table ({suffix.upper()})")
        print(f"  Assay counts: Current={len(am_cols)}, "
              f"Adj1={len(adj1_cols)}, Adj2={len(adj2_cols)}, Adj3={len(adj3_cols)}")
        print(f"{'=' * 70}")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Save the classification lists for inspection ---
    adj1_titles = sorted([c.strip() for c in adj1_cols])
    adj2_titles = sorted([c.strip() for c in adj2_cols])
    adj3_titles = sorted([c.strip() for c in adj3_cols])

    with open(OUTPUT_DIR / "adj1_assays_toxtransformer.txt", 'w') as f:
        f.write(f"# Assays with ToxTransformer DART/ED strength > 8 ({len(adj1_titles)} assays)\n")
        for t in adj1_titles:
            f.write(t + '\n')

    with open(OUTPUT_DIR / "adj2_assays_manual_darted.txt", 'w') as f:
        f.write(f"# Assays classified as DART/ED-relevant by keyword rules ({len(adj2_titles)} assays)\n")
        for t in adj2_titles:
            f.write(t + '\n')

    with open(OUTPUT_DIR / "adj3_assays_cmre_intersection.txt", 'w') as f:
        f.write(f"# Assays in activity matrix ∩ CMRE list ({len(adj3_titles)} assays)\n")
        for t in adj3_titles:
            f.write(t + '\n')

    logger.info("Saved assay lists to %s", OUTPUT_DIR)


if __name__ == '__main__':
    main()
