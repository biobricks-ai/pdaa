import sys
import pandas as pd
# from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

sys.path.append('./')
from scripts.utils.helpers import (
    smiles_to_inchi,
    inchi_to_smiles,
    is_diester_phthalate,
    classify_isomer,
    # SMARTS_PATTERNS,
)
from stages.utils.pdaa import predict_all_properties_with_sqlite_cache

# Replacement templates
# ORTHO_REPLACE = Chem.MolFromSmarts("c1cc(C(=O)O[*:1])c(C(=O)O[*:2])cc1")
# PARA_REPLACE  = Chem.MolFromSmarts("c1c(C(=O)O[*:1])ccc(C(=O)O[*:2])c1")
ORTHO_TO_PARA_RXN = AllChem.ReactionFromSmarts(
    "c1cc(C(=O)O[*:1])c(C(=O)O[*:2])cc1>>"
    "c1c(C(=O)O[*:1])ccc(C(=O)O[*:2])c1"
)

PARA_TO_ORTHO_RXN = AllChem.ReactionFromSmarts(
    "c1c(C(=O)O[*:1])ccc(C(=O)O[*:2])c1>>"
    "c1cc(C(=O)O[*:1])c(C(=O)O[*:2])cc1"
)

# def general_convert_isomerism(smiles: str) -> str:
#     """
#     Convert an ortho-phthalate SMILES to the corresponding tere-phthalate,
#     and vice versa, by replacing the aromatic diester core while preserving
#     side chains.
#     """
#     if ortho_core in smiles:
#         return smiles.replace(ortho_core, tere_core, 1)
#     elif tere_core in smiles:
#         return smiles.replace(tere_core, ortho_core, 1)
#     else:
#         return smiles

# def convert_phthalate_isomer(mol, target="para"):
#     if target == "para":
#         patt, repl = ORTHO_REPLACE, PARA_REPLACE
#     else:
#         patt, repl = PARA_REPLACE, ORTHO_REPLACE

#     if not mol.HasSubstructMatch(patt):
#         return mol

#     products = Chem.ReplaceSubstructs(
#         mol,
#         patt,
#         repl,
#         replaceAll=False,
#     )
#     if not products:
#         return mol
#     out = products[0]
#     Chem.SanitizeMol(out)
#     return out

# def convert_phthalate_isomer(mol, target="para"):
#     if target == "para":
#         rxn = ORTHO_TO_PARA_RXN
#     else:
#         rxn = PARA_TO_ORTHO_RXN

#     products = rxn.RunReactants((mol,))
#     if not products:
#         return mol

#     out = products[0][0]
#     Chem.SanitizeMol(out)
#     return out

def convert_phthalate_isomer(mol, target="para"):
    """
    Apply the ortho <-> para phthalate core swap using RDKit reactions.

    This version considers all reaction products and returns the first one
    that the classifier recognizes as the desired opposite isomer. If none
    are suitable, the original molecule is returned unchanged.
    """
    if target == "para":
        rxn = ORTHO_TO_PARA_RXN
        desired_iso = 2
    else:
        rxn = PARA_TO_ORTHO_RXN
        desired_iso = 0

    products = rxn.RunReactants((mol,))
    if not products:
        return mol

    # Flatten product tuples: products is a tuple of tuples
    for prod_tuple in products:
        if not prod_tuple:
            continue
        cand = prod_tuple[0]
        try:
            Chem.SanitizeMol(cand)
        except Exception:
            continue
        try:
            iso = classify_isomer(cand)
        except ValueError:
            continue
        if iso == desired_iso:
            return cand

    # If none of the products classify as the desired isomer, fall back
    return mol


def general_convert_isomerism(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not is_diester_phthalate(mol):
        return smiles

    try:
        iso = classify_isomer(mol)
    except ValueError:
        breakpoint()
        return smiles
    if iso == 0:
        target = "para"
    elif iso == 2:
        target = "ortho"
    else:
        return smiles

    new_mol = convert_phthalate_isomer(mol, target)
    try:
        new_iso = classify_isomer(new_mol)
    except ValueError:
        breakpoint()
    if (iso, new_iso) not in [(0,2), (2,0)]:
        return smiles

    return Chem.MolToSmiles(new_mol)


df = pd.read_parquet("cache/entity_similarity/activity_matrix_filled.parquet")

# create SMILES column
df["smiles"] = df.index.map(
    lambda inchi: inchi_to_smiles(inchi)
)

new_inchis = []
conversion_dict = {}
for smiles in df["smiles"]:
    converted_smiles = general_convert_isomerism(smiles)
    converted_inchi = smiles_to_inchi(converted_smiles)
    if converted_inchi in new_inchis:
        raise ValueError(
            f"Duplicate converted InChI detected: {converted_inchi}"
            f"\nfrom SMILES (new): {smiles}"
            f"\nfrom SMILES (old): {conversion_dict[converted_inchi]}"
        )
    if converted_inchi not in df.index:
        new_inchis.append(converted_inchi)
        conversion_dict[converted_inchi] = smiles

new_inchis = list(dict.fromkeys(new_inchis))  # deduplicate while preserving order
_ = predict_all_properties_with_sqlite_cache(new_inchis)

# # optional verification
# all_inchis = list(df.index) + new_inchis
# all_smiles = [inchi_to_smiles(inchi) for inchi in all_inchis]
# # count number of ortho and tere phthalates
# num_ortho = 0
# num_tere  = 0

# for smi in all_smiles:
#     mol = Chem.MolFromSmiles(smi)
#     if mol is None or not is_diester_phthalate(mol):
#         continue
#     iso = classify_isomer(mol)
#     if iso == 0:
#         num_ortho += 1
#     elif iso == 2:
#         num_tere += 1

# print(f"num_ortho = {num_ortho}, num_tere = {num_tere}")

# detailed diagnostics on conversion behavior

stats = {
    "ortho_to_tere": {
        "total": 0,
        "rxn_no_products": 0,
        "rxn_products_but_bad_iso": 0,
        "rxn_products_good_iso_dup": 0,
        "rxn_products_good_iso_new": 0,
        "classify_error_before": 0,
        "classify_error_after": 0,
    },
    "tere_to_ortho": {
        "total": 0,
        "rxn_no_products": 0,
        "rxn_products_but_bad_iso": 0,
        "rxn_products_good_iso_dup": 0,
        "rxn_products_good_iso_new": 0,
        "classify_error_before": 0,
        "classify_error_after": 0,
    },
    "other": {
        "not_diester_phthalate": 0,
        "meta_phthalate": 0,
        "smiles_parse_fail": 0,
    },
}

original_inchis = list(df.index)
original_smiles = [inchi_to_smiles(inchi) for inchi in original_inchis]
original_inchi_set = set(original_inchis)

for inchi, smi in zip(original_inchis, original_smiles):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        stats["other"]["smiles_parse_fail"] += 1
        continue

    if not is_diester_phthalate(mol):
        stats["other"]["not_diester_phthalate"] += 1
        continue

    try:
        iso = classify_isomer(mol)
    except ValueError:
        # could log or store these separately if useful
        # for now just count
        if 0:  # placeholder branch to make explicit that we could inspect this
            pass
        # treat as "other" for now
        stats["other"]["meta_phthalate"] += 1
        continue

    if iso not in (0, 2):
        # meta or other phthalate-like but not ortho/tere
        stats["other"]["meta_phthalate"] += 1
        continue

    # from here we know it's a diester phthalate and iso in {0, 2}
    if iso == 0:
        bucket = stats["ortho_to_tere"]
        target = "para"
    else:
        bucket = stats["tere_to_ortho"]
        target = "ortho"

    bucket["total"] += 1

    # run the same conversion logic
    try:
        new_mol = convert_phthalate_isomer(mol, target)
    except Exception:
        # if the reaction itself raises, count as "no products" for now
        bucket["rxn_no_products"] += 1
        continue

    try:
        new_iso = classify_isomer(new_mol)
    except ValueError:
        bucket["classify_error_after"] += 1
        continue

    desired_iso = 2 if iso == 0 else 0

    if new_iso != desired_iso:
        # Inspect the molecule if needed
        print("Conversion resulted in unexpected isomer:")
        print(f"  original SMILES: {smi}")
        print(f"  original isomer: {iso}")
        print(f"  original InChI: {inchi}")
        bucket["rxn_products_but_bad_iso"] += 1
        continue

    # at this point, we have a chemically good conversion (by your classifier)
    new_smi = Chem.MolToSmiles(new_mol)
    new_inchi = smiles_to_inchi(new_smi)

    if new_inchi in original_inchi_set:
        bucket["rxn_products_good_iso_dup"] += 1
    else:
        bucket["rxn_products_good_iso_new"] += 1

print("Conversion diagnostics:")
for direction, vals in stats.items():
    print(direction)
    for k, v in vals.items():
        print(f"  {k}: {v}")

# final counts
all_inchis = set(df.index).union(new_inchis)  # dedupe
num_ortho = 0
num_tere = 0

for inchi in all_inchis:
    smi = inchi_to_smiles(inchi)
    mol = Chem.MolFromSmiles(smi)
    if mol is None or not is_diester_phthalate(mol):
        continue
    iso = classify_isomer(mol)
    if iso == 0:
        num_ortho += 1
    elif iso == 2:
        num_tere += 1
print(f"Final counts after conversion:")
print(f"  num_ortho = {num_ortho}, num_tere = {num_tere}")