import time
import json
import math
import re
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as Descr
from rdkit.Chem.MolStandardize import rdMolStandardize as Std

# Add near the top with other imports:
from pathlib import Path
import argparse


# ---------- Utilities ----------

CAS_RE = re.compile(r'^(\d{2,7})-(\d{2})-(\d)$')

def normalize_cas(cas_raw: str) -> Optional[str]:
    if not isinstance(cas_raw, str):
        return None
    s = re.sub(r'[^0-9-]', '', cas_raw.strip())
    m = CAS_RE.match(s)
    if not m:
        return None
    body = ''.join(m.groups()[:-1])
    check = int(m.group(3))
    # CAS check digit = sum of digits*weight from right to left starting at 1
    digits = list(map(int, body))
    calc = sum(d * (i + 1) for i, d in enumerate(reversed(digits))) % 10
    return s if calc == check else None

def clean_name(name: str) -> Optional[str]:
    if not isinstance(name, str):
        return None
    s = ' '.join(name.strip().split())
    if not s:
        return None
    return s

session = requests.Session()
session.headers.update({"User-Agent": "insilica-structure-resolver/1.0"})

def _get(url: str, timeout: float = 10.0) -> Optional[requests.Response]:
    for attempt in range(3):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200 and r.text:
                return r
            # backoff on non-200 as well
        except requests.RequestException:
            pass
        time.sleep(0.5 * (2 ** attempt))
    return None

# ---------- Standardization ----------

_largest = Std.LargestFragmentChooser()
_unc = Std.Uncharger()
_taut = Std.TautomerEnumerator()

def standardize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    try:
        mol = _largest.choose(mol)
        mol = _unc.uncharge(mol)
        mol = _taut.Canonicalize(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None

def _clean_str(x: object) -> Optional[str]:
    # Return None for None, NaN (including numpy.nan), and empty/placeholder strings.
    if x is None:
        return None
    # Covers float('nan') and numpy.nan without depending on numpy explicitly
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.lower() in {"nan", "none"}:
        return None
    return s

def mol_from_any(smiles: Optional[str], inchi: Optional[str]) -> Optional[Chem.Mol]:
    mol = None
    inchi_s = _clean_str(inchi)
    smiles_s = _clean_str(smiles)

    if inchi_s:
        try:
            mol = Chem.MolFromInchi(inchi_s, treatWarningAsError=False)
        except Exception:
            mol = None  # keep going and try SMILES

    if mol is None and smiles_s:
        try:
            mol = Chem.MolFromSmiles(smiles_s)
        except Exception:
            mol = None

    return standardize_mol(mol) if mol is not None else None

def mol_props(mol: Chem.Mol) -> Dict[str, str]:
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    inchi = Chem.MolToInchi(mol)
    inchikey = Chem.InchiToInchiKey(inchi)
    formula = Descr.CalcMolFormula(mol)
    mass = Descr.CalcExactMolWt(mol)
    return {
        "smiles": smiles,
        "inchi": inchi,
        "inchikey": inchikey,
        "formula": formula,
        "exact_mass": f"{mass:.6f}",
    }

# ---------- Resolvers ----------

def pubchem_by_ident(identifier: str) -> Optional[Dict[str, str]]:
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    # Try name or CAS; PubChem treats both under 'compound/name'
    url = f"{base}/compound/name/{requests.utils.quote(identifier)}/property/CanonicalSMILES,InChI,InChIKey,MolecularFormula,ExactMass/JSON"
    r = _get(url)
    if not r:
        return None
    try:
        props = r.json()["PropertyTable"]["Properties"][0]
        return {
            "source": "pubchem",
            "cid": str(props.get("CID", "")),
            "smiles": props.get("CanonicalSMILES"),
            "inchi": props.get("InChI"),
            "inchikey": props.get("InChIKey"),
            "formula": props.get("MolecularFormula"),
            "exact_mass": str(props.get("ExactMass")),
        }
    except Exception:
        return None

def opsin_by_name(name: str) -> Optional[Dict[str, str]]:
    # OPSIN returns JSON with SMILES; we convert to InChI via RDKit
    url = f"https://opsin.ch.cam.ac.uk/opsin/{requests.utils.quote(name)}.json"
    r = _get(url)
    if not r:
        return None
    try:
        j = r.json()
        if j.get("status") != "OK":
            return None
        smiles = j.get("smiles")
        mol = mol_from_any(smiles, None)
        if not mol:
            return None
        props = mol_props(mol)
        props["source"] = "opsin"
        return props
    except Exception:
        return None

def cactus_by_ident(identifier: str) -> Optional[Dict[str, str]]:
    base = "https://cactus.nci.nih.gov/chemical/structure"
    # Prefer InChI; fall back to SMILES
    r_inchi = _get(f"{base}/{requests.utils.quote(identifier)}/stdinchi")
    r_smiles = r_inchi or _get(f"{base}/{requests.utils.quote(identifier)}/smiles")
    if not r_smiles:
        return None
    val = r_smiles.text.strip()
    mol = mol_from_any(val if "InChI=" not in val else None,
                       val if val.startswith("InChI=") else None)
    if not mol:
        return None
    props = mol_props(mol)
    props["source"] = "cactus"
    return props

def resolve_structure(name: Optional[str], cas: Optional[str]) -> Optional[Dict[str, str]]:
    # Try highest-confidence routes first
    for ident in [cas, name]:
        if ident:
            p = pubchem_by_ident(ident)
            if p:
                return p
    # Name-only routes next
    if name:
        o = opsin_by_name(name)
        if o:
            return o
        c = cactus_by_ident(name)
        if c:
            return c
    # CAS-only fallback
    if cas:
        c = cactus_by_ident(cas)
        if c:
            return c
    return None

# ---------- Validation ----------

def validate_record(row: pd.Series, candidate: Dict[str, str]) -> Tuple[float, Dict[str, str]]:
    notes = []
    score = 0.0

    # Build Mol from candidate and standardize
    mol = mol_from_any(candidate.get("smiles"), candidate.get("inchi"))
    if not mol:
        return 0.0, {"status": "unparsable", "notes": "RDKit could not parse candidate"}

    props = mol_props(mol)

    # Formula check
    fmla_src = str(row.get("Fmla") or "").strip()
    if fmla_src and props["formula"] == fmla_src:
        score += 0.3
    elif fmla_src:
        notes.append(f"Formula mismatch: src={fmla_src}, got={props['formula']}")

    # Mass check (ppm tolerance)
    mw_src = row.get("Molecular Weight")
    try:
        mw_src = float(mw_src)
        mass = float(props["exact_mass"])
        ppm = abs(mw_src - mass) / mass * 1e6
        if ppm < 10:
            score += 0.2
        else:
            notes.append(f"Mass delta {ppm:.1f} ppm (src={mw_src}, exact={mass})")
    except Exception:
        pass

    # CAS or name provenance via PubChem when available
    if candidate.get("source") == "pubchem" and candidate.get("cid"):
        score += 0.2  # CID provides stronger provenance
    # Stereo/isomer specificity — assume high if full InChIKey present
    if candidate.get("inchikey"):
        score += 0.2

    # Parsability + standardization success
    score += 0.1

    out = {
        **candidate,
        **props,
        "status": "ok" if score >= 0.7 else "review",
        "confidence": f"{score:.2f}",
        "notes": "; ".join(notes) if notes else "",
    }
    return score, out

# ---------- Main pipeline ----------

def enrich_df(df: pd.DataFrame, name_col="Name", cas_col="Cas Decimal",
              smiles_col="smiles", inchi_col="inchi") -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        name = clean_name(row.get(name_col))
        cas = normalize_cas(str(row.get(cas_col))) if pd.notna(row.get(cas_col)) else None

        # If present, try to standardize existing structure first
        given_mol = mol_from_any(row.get(smiles_col), row.get(inchi_col))
        if given_mol:
            props = mol_props(given_mol)
            candidate = {"source": "given", **props}
        else:
            candidate = resolve_structure(name, cas) or {}

        if candidate:
            _, audited = validate_record(row, candidate)
        else:
            audited = {"status": "unresolved", "notes": "No resolver returned a structure"}

        # Keep context
        audited.update({
            "orig_name": row.get(name_col),
            "orig_cas": row.get(cas_col),
            "orig_smiles": row.get(smiles_col),
            "orig_inchi": row.get(inchi_col),
            "id": row.get("ID"),
        })
        rows.append(audited)

        # polite pause for API
        time.sleep(0.12)

    return pd.DataFrame(rows)

# Add at the very end of the file:
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve and audit chemical structures (SMILES/InChI) with validation."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input CSV (e.g., bad_inchis.csv).")
    parser.add_argument("-o", "--output", required=True, help="Path to output CSV (resolution_report.csv).")
    parser.add_argument("--name_col", default="Name", help="Column containing chemical names.")
    parser.add_argument("--cas_col", default="Cas Decimal", help="Column containing CAS RN.")
    parser.add_argument("--smiles_col", default="smiles", help="Column containing existing SMILES, if any.")
    parser.add_argument("--inchi_col", default="inchi", help="Column containing existing InChI, if any.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    for col in [args.inchi_col, args.smiles_col, args.name_col, args.cas_col]:
        if col in df.columns:
            df[col] = df[col].astype("object")  # avoid pandas NA semantics breaking truthiness

    # Ensure expected columns exist (no-op if already present)
    for col in [args.name_col, args.cas_col, args.smiles_col, args.inchi_col, "Fmla", "Molecular Weight", "ID"]:
        if col not in df.columns:
            df[col] = None

    # Run resolver + auditor
    out = enrich_df(
        df,
        name_col=args.name_col,
        cas_col=args.cas_col,
        smiles_col=args.smiles_col,
        inchi_col=args.inchi_col,
    )

    # Nice column order for triage
    preferred = [
        "id",
        "orig_name", "orig_cas", "orig_smiles", "orig_inchi",
        "smiles", "inchi", "inchikey", "formula", "exact_mass",
        "cid", "source", "status", "confidence", "notes",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[cols]

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    # Brief summary to stdout
    status_counts = out["status"].value_counts(dropna=False).to_dict()
    print(json.dumps({"n_rows": len(out), "status_counts": status_counts}, indent=2))


if __name__ == "__main__":
    main()
