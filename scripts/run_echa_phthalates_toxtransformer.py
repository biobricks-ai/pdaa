"""
Submit ECHA-derived phthalate SMILES to ToxTransformer via ToxIndex gateway.

These are the representative isomers for DINP, DIDP, DIUP, and DTDP derived
from ECHA Markush structures for ExxonMobil's UVCB phthalate registrations.

Usage:
    python scripts/run_echa_phthalates_toxtransformer.py
"""

import json
import time
import logging
import requests
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

BASE = "https://gateway.toxindex.com/v1"
TOKEN_FILE = Path("/home/john/Documents/Insilica/ToxTransformer_manuscript/preservatives_run/.token")

OUTDIR = Path(__file__).parent.parent / "cache" / "echa_toxtransformer"

PHTHALATES = {
    "DINP_2methyloctyl":   "CCCCCCC(C)COC(=O)c1ccccc1C(=O)OCC(C)CCCCCC",
    "DIDP_2methylnonyl":   "CCCCCCCC(C)COC(=O)c1ccccc1C(=O)OCC(C)CCCCCCC",
    "DIUP_2methyldecyl":   "CCCCCCCCC(C)COC(=O)c1ccccc1C(=O)OCC(C)CCCCCCCC",
    "DTDP_2methyldodecyl": "CCCCCCCCCCC(C)COC(=O)c1ccccc1C(=O)OCC(C)CCCCCCCCCC",
    "DIUP_tris_dimethylnonyl": "CCCC(C)CC(C)CCCOC(=O)c1ccccc1C(=O)OCCC(C)CC(C)CCCC",
    "DIDP_2ethyloctyl":    "CCCCCCC(CC)COC(=O)c1ccccc1C(=O)OCC(CC)CCCCCC",
}

RUNS_FILE = OUTDIR / "run_ids.json"
RESULTS_FILE = OUTDIR / "results.json"


def load_token():
    if not TOKEN_FILE.exists():
        log.error("No token found at %s", TOKEN_FILE)
        return None
    return TOKEN_FILE.read_text().strip()


def make_session(token):
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def submit_all(session):
    if RUNS_FILE.exists():
        runs = json.loads(RUNS_FILE.read_text())
        log.info("Loaded %d existing run IDs", len(runs))
    else:
        runs = {}

    for name, smiles in PHTHALATES.items():
        if name in runs:
            log.info("Already submitted: %s (%s)", name, runs[name])
            continue
        try:
            r = session.post(
                f"{BASE}/runs/toxtransformer",
                json={"smiles": smiles},
                timeout=30,
            )
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                log.warning("Rate-limited, waiting %ds", wait)
                time.sleep(wait)
                r = session.post(
                    f"{BASE}/runs/toxtransformer",
                    json={"smiles": smiles},
                    timeout=30,
                )
            r.raise_for_status()
            data = r.json()
            run_id = data.get("run_id") or data.get("id")
            runs[name] = run_id
            log.info("Submitted %-30s -> %s", name, run_id)
            RUNS_FILE.write_text(json.dumps(runs, indent=2))
            time.sleep(1)
        except Exception:
            log.exception("Failed to submit %s", name)

    RUNS_FILE.write_text(json.dumps(runs, indent=2))
    return runs


def poll_all(session, runs, max_wait=1800):
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())
    else:
        results = {}

    pending = {
        n: rid for n, rid in runs.items()
        if n not in results or results[n].get("status") not in ("completed", "failed")
    }
    start = time.time()
    poll_interval = 5

    while pending and (time.time() - start) < max_wait:
        for name, run_id in list(pending.items()):
            try:
                r = session.get(f"{BASE}/runs/{run_id}", timeout=30)
                r.raise_for_status()
                data = r.json()
                status = data.get("status", "unknown")
                if status in ("completed", "failed"):
                    results[name] = data
                    pending.pop(name)
                    n_props = len(data.get("result", {}).get("properties", [])) if status == "completed" else 0
                    log.info("%-30s %s (%d properties)", name, status, n_props)
                    RESULTS_FILE.write_text(json.dumps(results, indent=2))
            except Exception:
                log.exception("Error polling %s", name)
            time.sleep(0.5)

        if pending:
            remaining = len(pending)
            elapsed = int(time.time() - start)
            log.info("%d runs still pending after %ds, waiting %ds...",
                     remaining, elapsed, poll_interval)
            time.sleep(poll_interval)
            poll_interval = min(poll_interval + 5, 30)

    if pending:
        log.warning("Timed out with %d runs still pending: %s",
                     len(pending), list(pending.keys()))
    else:
        log.info("All %d runs completed", len(results))

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    return results


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    token = load_token()
    if not token:
        return

    session = make_session(token)

    log.info("=== ToxTransformer ECHA Phthalates Run ===")
    log.info("Submitting %d compounds", len(PHTHALATES))
    runs = submit_all(session)
    log.info("Polling for results...")
    results = poll_all(session, runs)

    completed = sum(1 for r in results.values() if r.get("status") == "completed")
    failed = sum(1 for r in results.values() if r.get("status") == "failed")
    log.info("Done: %d completed, %d failed, %d total", completed, failed, len(results))


if __name__ == "__main__":
    main()
