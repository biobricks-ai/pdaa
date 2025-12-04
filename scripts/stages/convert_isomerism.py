import sys
import pandas as pd
from pathlib import Path

sys.path.append('./')
from scripts.utils.helpers import smiles_to_inchi

input_csv = Path("resources") / "long_chain_phthalates.csv"   # change to your actual filename
# output_csv = "phthalates_tere.csv"
output_csv = input_csv

ortho_core = "OC(=O)c1ccccc1C(=O)O"
tere_core = "OC(=O)c1ccc(cc1)C(=O)O"

def convert_to_tere(smiles: str) -> str:
    """
    Convert an ortho-phthalate SMILES to the corresponding terephthalate
    by replacing the aromatic diester core while preserving side chains.
    """
    count = smiles.count(ortho_core)
    if count == 0:
        raise ValueError(
            f"Expected ortho phthalate core not found in SMILES: {smiles}"
        )
    if count > 1:
        raise ValueError(
            f"Ortho phthalate core appears {count} times in SMILES: {smiles}"
        )
    return smiles.replace(ortho_core, tere_core, 1)

df = pd.read_csv(input_csv)

if "smiles" not in df.columns:
    raise KeyError("Input CSV must contain a 'smiles' column.")

# create a new dataframe for tere-phthalates
df_tere = df.copy()

# add converted tere SMILES and corresponding InChI
df_tere["smiles"] = df_tere["smiles"].apply(convert_to_tere)
df_tere["inchi"] = df_tere["smiles"].apply(smiles_to_inchi)

df = pd.concat([df, df_tere], ignore_index=True)
df = df.drop_duplicates(subset=["inchi"])
df = df.reset_index(drop=True)

df.to_csv(output_csv, index=False)
print(f"Wrote tere-phthalates to {output_csv}")
