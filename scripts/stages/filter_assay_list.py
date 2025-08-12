import pandas as pd
from pathlib import Path

def get_assay_df(fname: str, keep_positive_only: bool = True) -> pd.DataFrame:
    """
    Load assay flags from a file and return as a DataFrame.
    """
    df = pd.read_csv(
        fname,
        sep="\t",
        header=None,
        names=["title", "flag"]
    ).assign(
        title=lambda d: d["title"].str.lower().str.strip()
    )

    if keep_positive_only:
        df = df.loc[df["flag"].astype(bool)]

    return df


resourcedir = Path("resources")
fnames = [
    "assay_flags2_dart.txt",
    "assay_flags2_ed.txt",
]
# Load the full assay list    
full_assay_df = get_assay_df(resourcedir / fnames[0], keep_positive_only=False)
dfs = [
    get_assay_df(resourcedir / fname) 
    for fname in fnames
]
combined_df = pd.concat(dfs, ignore_index=True)

subtraction = set(full_assay_df["title"]) - set(combined_df["title"])

if subtraction:
    print(f"Assays in full list but not in combined: {len(subtraction)}")

    # Save the subtraction to a file
    with open(resourcedir / "assays_not_in_combined.txt", "w") as f:
        for assay in subtraction:
            f.write(f"{assay}\n")