import time
import json
import math
import os
import re
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as Descr
from rdkit.Chem.MolStandardize import rdMolStandardize as Std
from tqdm.auto import tqdm

from pathlib import Path
import argparse

# Formula normalization helpers
FORMULA_TOKEN = re.compile(r'([A-Z][a-z]?)(\d*)')

def _parse_formula(formula: str) -> Dict[str, int]:
    if not isinstance(formula, str):
        return {}
    counts: Dict[str, int] = {}
    for part in formula.replace(" ", "").split("·"):  # dot-separated solvates/counterions
        for elem, num in FORMULA_TOKEN.findall(part):
            n = int(num) if num else 1
            counts[elem] = counts.get(elem, 0) + n
    return counts

def _hill_rebuild(counts: Dict[str, int]) -> str:
    if not counts:
        return ""
    out = []
    if "C" in counts:
        c = counts["C"]
        out.append(f"C{'' if c == 1 else c}")
        h = counts.get("H", 0)
        if h:
            out.append(f"H{'' if h == 1 else h}")
    # Remaining elements in alphabetical order, excluding C and H
    others = sorted(k for k in counts.keys() if k not in {"C", "H"})
    for e in others:
        n = counts[e]
        out.append(f"{e}{'' if n == 1 else n}")
    return "".join(out)

def normalize_dataset_formula(formula: str) -> str:
    # Keep the largest dot fragment by heavy-atom count; rebuild to Hill order.
    parts = formula.replace(" ", "").split("·")
    def heavy_count(s: str) -> int:
        c = _parse_formula(s)
        return sum(v for k, v in c.items() if k != "H")
    best = max(parts, key=heavy_count) if parts else formula
    return _hill_rebuild(_parse_formula(best))

# Stereo heuristics
_STEREO_HINTS = (" cis", " trans", "(r)", "(s)", " r-", " s-", "(e)", "(z)", " e-", " z-", "alpha", "beta", "racemate", "dl-", "d,l-")
def name_has_stereo_hint(name: Optional[str]) -> bool:
    if not isinstance(name, str):
        return False
    s = " " + name.lower().strip()  # pad to catch word-boundary hints like " cis"
    return any(h in s for h in _STEREO_HINTS)

def inchi_has_stereo_layers(inchi: Optional[str]) -> bool:
    if not isinstance(inchi, str):
        return False
    # InChI stereo info typically present in /t (tetrahedral), /b (double bond), +/- with /m,/s layers
    return any(tag in inchi for tag in ("/t", "/b", "/m", "/s"))


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

    # Formula: require match when dataset formula present (after normalization)
    fmla_src_raw = str(row.get("Fmla") or "").strip()
    fmla_required_ok = False
    if fmla_src_raw:
        fmla_required_ok = True
        fmla_src = normalize_dataset_formula(fmla_src_raw)
        if props["formula"] == fmla_src:
            score += 0.35
        else:
            notes.append(f"Formula mismatch (required): src={fmla_src_raw} -> {fmla_src}, got={props['formula']}")

        # Element-set screen: strong signal of misassignment
        src_elems = set(k for k in _parse_formula(fmla_src).keys())
        got_elems = set(k for k in _parse_formula(props["formula"]).keys())
        if src_elems != got_elems:
            notes.append(f"Element-set mismatch: src={sorted(src_elems)}, got={sorted(got_elems)}")
            # Do not add score when element sets differ.

    # Mass check (ppm tolerance)
    mw_src = row.get("Molecular Weight")
    try:
        mw_src_f = float(mw_src)
        mass = float(props["exact_mass"])
        ppm = abs(mw_src_f - mass) / mass * 1e6
        if ppm < 10:
            score += 0.2
        else:
            notes.append(f"Mass delta {ppm:.1f} ppm (src={mw_src_f}, exact={mass})")
    except Exception:
        pass

    # Stereo heuristic
    nm = row.get("Name")
    if name_has_stereo_hint(nm) and not inchi_has_stereo_layers(props.get("inchi")):
        notes.append("Stereo hint in name but InChI lacks stereo layers")

    # PubChem provenance bonus
    if candidate.get("source") == "pubchem" and candidate.get("cid"):
        score += 0.2

    # InChIKey present (structure specificity)
    if candidate.get("inchikey"):
        score += 0.15

    # Parsability + standardization success
    score += 0.1

    # Final status: never 'ok' if formula required and mismatched
    status = "ok" if (score >= 0.7 and (not fmla_required_ok or normalize_dataset_formula(fmla_src_raw) == props["formula"])) else "review"

    out = {
        **candidate,
        **props,
        "status": status,
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

        # Prefer existing structure; validate; if not ok, try resolver and keep better
        cand_best = None
        score_best = -1.0
        audited = None

        given_mol = mol_from_any(row.get(smiles_col), row.get(inchi_col))
        if given_mol:
            cand_given = {"source": "given", **mol_props(given_mol)}
            score_g, audit_g = validate_record(row, cand_given)
            cand_best, score_best, audited = cand_given, score_g, audit_g

        # Decide whether we need resolution
        need_resolve = (audited is None) or (audited.get("status") != "ok")

        if need_resolve:
            cand_res = resolve_structure(name, cas)
            if cand_res:
                score_r, audit_r = validate_record(row, cand_res)
                if score_r > score_best:
                    cand_best, score_best, audited = cand_res, score_r, audit_r

        if audited is None:
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

def _row_key(idx: int, row: pd.Series) -> str:
    """Stable key per row for checkpointing. Prefer the dataset ID when present."""
    rid = row.get("ID")
    if pd.notna(rid):
        return f"id:{rid}"
    return f"idx:{idx}"

def iter_enriched(
    df: pd.DataFrame,
    name_col: str = "Name",
    cas_col: str = "Cas Decimal",
    smiles_col: str = "smiles",
    inchi_col: str = "inchi",
):
    """
    Stream enriched rows one at a time. This enables progress bars and checkpointing.
    """
    for idx, row in df.iterrows():
        name = clean_name(row.get(name_col))
        cas = normalize_cas(str(row.get(cas_col))) if pd.notna(row.get(cas_col)) else None

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

        audited.update({
            "orig_name": row.get(name_col),
            "orig_cas": row.get(cas_col),
            "orig_smiles": row.get(smiles_col),
            "orig_inchi": row.get(inchi_col),
            "id": row.get("ID"),
            "row_index": idx,
            "row_key": _row_key(idx, row),
        })

        yield audited
        time.sleep(0.12)

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
    parser.add_argument("--checkpoint_path", default=None, help="Optional path to a checkpoint CSV.")
    parser.add_argument("--checkpoint_every", type=int, default=100, help="Write checkpoint every N rows.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if present.")
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype="object")  # keep string-like columns as object
    print(f"Loaded {len(df)} rows from {args.input}")

    # Ensure expected columns exist
    for col in [args.name_col, args.cas_col, args.smiles_col, args.inchi_col, "Fmla", "Molecular Weight", "ID"]:
        if col not in df.columns:
            df[col] = None

    # Minimize redundant checks
    before = len(df)
    df.drop_duplicates(subset=["ID", "Name", "Cas Decimal"], inplace=True, ignore_index=True)
    after = len(df)
    if after != before:
        print(f"De-duplicated rows: {before - after} removed; {after} remain")

    # Determine checkpoint path
    checkpoint_path = args.checkpoint_path or (str(Path(args.output)) + ".checkpoint.csv")

    # Load prior progress if resuming
    existing = None
    processed_keys = set()
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint {checkpoint_path}")
        try:
            existing = pd.read_csv(checkpoint_path, dtype="object")
            if "row_key" in existing.columns:
                processed_keys = set(existing["row_key"].dropna().astype(str).tolist())
            elif "id" in existing.columns:
                processed_keys = set("id:" + existing["id"].dropna().astype(str))
        except Exception:
            existing = None
            processed_keys = set()

    # Progress bar setup
    n_total = len(df)
    # Estimate remaining; exact remaining is computed on the fly
    pbar = tqdm(total=n_total, desc="Resolving structures", unit="row")

    # If resuming, advance the bar to the number of already processed rows we can identify
    if processed_keys:
        # Best-effort approximation; safe even if keys are for a different input file
        pbar.update(min(len(processed_keys), n_total))

    status_counts: Dict[str, int] = {"ok": 0, "review": 0, "unresolved": 0}
    new_rows: list = []

    # Append mode writer for checkpoint
    def _flush_checkpoint(rows_chunk: list) -> None:
        if not rows_chunk:
            return
        df_chunk = pd.DataFrame(rows_chunk)
        header = not os.path.exists(checkpoint_path)
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        df_chunk.to_csv(checkpoint_path, mode="a", header=header, index=False)

    # Stream processing with resume
    print("Starting database standardization and structure resolution...")
    processed_since_flush = 0
    for idx, row in df.iterrows():
        key = _row_key(idx, row)
        if key in processed_keys:
            pbar.update(1)
            continue

        audited = next(iter_enriched(
            pd.DataFrame([row]),
            name_col=args.name_col,
            cas_col=args.cas_col,
            smiles_col=args.smiles_col,
            inchi_col=args.inchi_col,
        ))

        status = str(audited.get("status", "unresolved"))
        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts[status] = 1

        new_rows.append(audited)
        processed_since_flush += 1

        # Periodic checkpoint
        if processed_since_flush >= args.checkpoint_every:
            _flush_checkpoint(new_rows)
            new_rows.clear()
            processed_since_flush = 0

        # Progress bar update
        pbar.update(1)
        pbar.set_postfix(status=status_counts, refresh=False)

    # Final checkpoint flush
    _flush_checkpoint(new_rows)
    new_rows.clear()
    pbar.close()

    # Consolidate final output: combine existing checkpoint (if any) with new rows and write output
    parts = []
    if existing is not None:
        parts.append(existing)
    if os.path.exists(checkpoint_path):
        # Reload to ensure we include final flushed rows
        parts.append(pd.read_csv(checkpoint_path, dtype="object"))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[
        "id","orig_name","orig_cas","orig_smiles","orig_inchi","smiles","inchi","inchikey",
        "formula","exact_mass","cid","source","status","confidence","notes","row_index","row_key"
    ])

    # Column order for triage
    preferred = [
        "id",
        "orig_name", "orig_cas", "orig_smiles", "orig_inchi",
        "smiles", "inchi", "inchikey", "formula", "exact_mass",
        "cid", "source", "status", "confidence", "notes",
        "row_index", "row_key",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[cols]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")

    # Summary to stdout
    final_counts = out["status"].value_counts(dropna=False).to_dict() if "status" in out.columns else {}
    print(json.dumps({
        "n_rows": len(out),
        "status_counts": final_counts,
        "output": args.output,
        "checkpoint": checkpoint_path,
    }, indent=2))

if __name__ == "__main__":
    main()