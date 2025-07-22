# ── loader.py ─────────────────────────────────────────────────────────
from rdkit import Chem
import pandas as pd

def load_edkb(path, activity_field):
    # rename *.txt to *.sdf* if you prefer
    suppl = Chem.SDMolSupplier(path, sanitize=False, removeHs=False)
    records = []
    for mol in suppl:
        if mol is None:              # skip bad parses
            continue
        log_rba = float(mol.GetProp(activity_field))
        if log_rba <= -9_000:        # sentinel for “inactive”
            continue                 # or keep and label 0 for classification
        records.append({
            "smiles": Chem.MolToSmiles(mol),
            "log_rba": log_rba
        })
    return pd.DataFrame(records)
