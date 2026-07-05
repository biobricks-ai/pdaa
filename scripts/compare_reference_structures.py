"""
Compare a set of candidate representative SMILES for the commercial UVCB phthalates
against (a) the activity-matrix current isomer, (b) the achiral isomer distribution.

For each candidate SMILES this reports:
  - molecular formula, longest carbon backbone (LCB), symmetry of the two ester arms
  - whether the InChI is present in the activity matrix; its MAV if so
  - the percentile of that MAV within the compound's achiral-isomer envelope and
    within the full dataset

Edit CANDIDATES below (or import and call evaluate()) as agent results arrive.
"""
import json
import logging
import pathlib
import re

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import percentileofscore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CACHEDIR = pathlib.Path("cache/entity_similarity")
DESCDIR = pathlib.Path("cache/descriptors")

# Activity-matrix representatives that reproduce the MANUSCRIPT numbers
# (DINP +0.03, DIDP +0.04, DIUP -0.73, DTDP -0.38).
AM_CURRENT = {
    "DINP": "CCCC(C)C(C)CCOC(=O)c1ccccc1C(=O)OCCC(C)C(C)CCC",   # bis(3,4-dimethylheptyl)
    "DIDP": "CCCC(C)CC(C)CCOC(=O)c1ccccc1C(=O)OCCC(C)CC(C)CCC", # bis(3,5-dimethyloctyl)
    "DIUP": "CC(C)CCCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCCC(C)C",   # bis(9-methyldecyl)
    "DTDP": "CC(C)CCCCCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCCCCC(C)C", # bis(11-methyldodecyl)
}

# Candidate representative SMILES to test, keyed by UVCB then a label.
# Pre-seeded with the alternate "boxplot-script" isomers actually present in some
# project scripts (terminal-isopropyl and linear), to expose intra-project variance.
# PubChem/literature structures are appended as the lookups return.
CANDIDATES = {
    # PubChem CID 590836 (bis(7-methyloctyl)) is the canonical surrogate for both
    # DINP CAS 68515-48-0 and 28553-12-0; identical to the project's boxplot-script isomer.
    "DINP": {"PubChem canonical CID 590836 (7-methyloctyl)":
             "CC(C)CCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCC(C)C"},
    # PubChem CID 33599 (bis(8-methylnonyl)) is canonical for DIDP CAS 68515-49-1 and 26761-40-0.
    "DIDP": {"PubChem canonical CID 33599 (8-methylnonyl)":
             "CC(C)CCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCC(C)C"},
    # PubChem CID 175686 (bis(9-methyldecyl)) — coincides with the AM/manuscript rep.
    "DIUP": {"PubChem canonical CID 175686 (9-methyldecyl)":
             "CC(C)CCCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCCC(C)C"},
    # PubChem branched CID 161559 (bis(11-methyldodecyl)) coincides with AM rep;
    # PubChem also maps commercial DTDP trade names to the LINEAR CID 8379.
    "DTDP": {"PubChem branched CID 161559 (11-methyldodecyl)":
             "CC(C)CCCCCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCCCCC(C)C",
             "PubChem linear CID 8379 (di-n-tridecyl, trade-name-mapped)":
             "CCCCCCCCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCCCCCCC"},
}


def formula_from_inchi(inchi):
    m = re.match(r"InChI=1S?/([A-Za-z0-9]+)/", inchi)
    return m.group(1) if m else None


def has_defined_stereo(inchi):
    return "/t" in inchi and ("/m" in inchi or "/s" in inchi)


def longest_carbon_backbone(mol):
    """Longest path of carbon atoms among the ester side chains (DFS), matching the
    descriptor definition used elsewhere in the project."""
    # carbon-only subgraph, exclude aromatic ring carbons
    idxs = [a.GetIdx() for a in mol.GetAtoms()
            if a.GetSymbol() == "C" and not a.GetIsAromatic()]
    idxset = set(idxs)
    adj = {i: [] for i in idxs}
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 in idxset and a2 in idxset:
            adj[a1].append(a2)
            adj[a2].append(a1)
    best = 0

    def dfs(node, visited):
        nonlocal best
        visited.add(node)
        longest_here = 1
        for nb in adj[node]:
            if nb not in visited:
                longest_here = max(longest_here, 1 + dfs(nb, visited))
        visited.discard(node)
        return longest_here

    for start in idxs:
        best = max(best, dfs(start, set()))
    return best


def ester_arms_symmetric(mol):
    patt = Chem.MolFromSmarts("[CX3](=O)[OX2][CX4]")
    matches = mol.GetSubstructMatches(patt)
    if len(matches) != 2:
        return None
    bonds = [mol.GetBondBetweenAtoms(m[2], m[3]).GetIdx() for m in matches]
    frag = Chem.FragmentOnBonds(mol, bonds, addDummies=True)
    pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)
    arms = []
    for p in pieces:
        if any(a.GetIsAromatic() for a in p.GetAtoms()):
            continue
        for a in p.GetAtoms():
            if a.GetAtomicNum() == 0:
                a.SetIsotope(0)
        arms.append(Chem.MolToSmiles(p))
    return (len(arms) == 2 and arms[0] == arms[1]) if len(arms) == 2 else None


def load():
    activity = pd.read_parquet(CACHEDIR / "activity_matrix_filled.parquet")
    z = (activity - activity.mean(0)) / activity.std(0)
    mav = z.mean(1)
    formula = mav.index.to_series().map(formula_from_inchi)
    stereo = mav.index.to_series().map(has_defined_stereo)
    return mav, formula, stereo


def evaluate(label, smiles, mav, formula, stereo):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        log.warning("%s: invalid SMILES %s", label, smiles)
        return None
    inchi = Chem.MolToInchi(mol)
    f = formula_from_inchi(inchi)
    lcb = longest_carbon_backbone(mol)
    sym = ester_arms_symmetric(mol)
    in_am = inchi in mav.index
    row = {"label": label, "smiles": smiles, "formula": f, "lcb": lcb,
           "symmetric": sym, "in_am": in_am}
    if in_am:
        val = float(mav.loc[inchi])
        env = mav[(formula.values == f) & (~stereo.values)].values
        row["mav"] = round(val, 3)
        row["dataset_pct"] = round(percentileofscore(mav.values, val), 1)
        if len(env):
            row["envelope_pct"] = round(percentileofscore(env, val), 1)
            row["envelope_median"] = round(float(np.median(env)), 3)
            row["envelope_n"] = int(len(env))
    return row


def main():
    mav, formula, stereo = load()
    results = []
    for uvcb, current in AM_CURRENT.items():
        results.append(evaluate(f"{uvcb} [AM current]", current, mav, formula, stereo))
        for lbl, smi in CANDIDATES.get(uvcb, {}).items():
            results.append(evaluate(f"{uvcb} [{lbl}]", smi, mav, formula, stereo))
    results = [r for r in results if r]
    df = pd.DataFrame(results)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(df.to_string(index=False))
    out = pathlib.Path("cache/isomer_envelope/reference_structure_comparison.json")
    out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
