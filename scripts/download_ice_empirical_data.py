#!/usr/bin/env python3
"""
Download empirical training data from ICE (Integrated Chemical Environment) API.

This script fetches dose-response curve data from NIEHS ICE to identify which
chemical-assay pairs have actual experimental observations (vs. predictions).

Outputs:
- cache/ice_empirical_data.parquet: Raw empirical observations
- cache/ice_empirical_coverage.csv: Binary matrix of empirical coverage
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"
CACHE_DIR = PROJECT_ROOT / "cache"


def download_ice_curves(
    assay_url: str,
    max_retries: int = 3,
    timeout: int = 30
) -> List[Dict]:
    """
    Download dose-response curves from ICE API for a specific assay.

    Parameters
    ----------
    assay_url : str
        Full ICE API URL (e.g., https://ice.ntp.niehs.nih.gov/api/v1/curves?assay=...)
    max_retries : int
        Number of retry attempts for failed requests
    timeout : int
        Request timeout in seconds

    Returns
    -------
    list of dict
        List of curve records with chemical and response data
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(assay_url, timeout=timeout)
            response.raise_for_status()

            data = response.json()

            # ICE API returns a list of curve records
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'curves' in data:
                return data['curves']
            else:
                logger.warning(f"Unexpected response format from {assay_url}")
                return []

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries} for {assay_url}")
            time.sleep(2 ** attempt)  # Exponential backoff

        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed on attempt {attempt + 1}/{max_retries}: {e}")
            time.sleep(2 ** attempt)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from {assay_url}: {e}")
            return []

    logger.error(f"Failed to download data from {assay_url} after {max_retries} attempts")
    return []


def parse_ice_record(record: Dict, assay_token: str, assay_title: str) -> Dict:
    """
    Parse an ICE curve record to extract relevant fields.

    Parameters
    ----------
    record : dict
        Raw ICE curve record
    assay_token : str
        Property token for this assay
    assay_title : str
        Human-readable assay title

    Returns
    -------
    dict
        Parsed record with standardized fields
    """
    # Extract chemical identifier (DTXSID or CAS)
    dtxsid = record.get('dtxsid', record.get('DTXSID', None))
    casrn = record.get('casrn', record.get('CASRN', None))

    # Extract response data
    hit_call = record.get('hitcall', record.get('hitCall', None))
    ac50 = record.get('ac50', record.get('AC50', None))

    # Some assays use different field names
    if hit_call is None:
        hit_call = record.get('call', None)

    return {
        'dtxsid': dtxsid,
        'casrn': casrn,
        'assay_token': assay_token,
        'assay_title': assay_title,
        'hit_call': hit_call,
        'ac50': ac50,
        'has_data': hit_call is not None or ac50 is not None
    }


def main():
    """Download ICE empirical data for all DART/ED assays."""
    # Load assay list
    assays_path = RESOURCES_DIR / "dart_ed_assays.csv"
    if not assays_path.exists():
        logger.error(f"Assay file not found: {assays_path}")
        return

    assays_df = pd.read_csv(assays_path)
    logger.info(f"Loaded {len(assays_df)} assays from {assays_path}")

    # Download data for each assay
    all_records = []

    for _, row in tqdm(assays_df.iterrows(), total=len(assays_df), desc="Downloading ICE data"):
        url = row['uri']
        token = str(row['token'])
        title = row['title']

        # Download curves for this assay
        curves = download_ice_curves(url)

        # Parse each curve record
        for record in curves:
            parsed = parse_ice_record(record, token, title)
            all_records.append(parsed)

        # Be nice to the API
        time.sleep(0.5)

    # Convert to DataFrame
    empirical_df = pd.DataFrame(all_records)
    logger.info(f"Downloaded {len(empirical_df)} total records")

    # Filter to records with actual data
    has_data = empirical_df[empirical_df['has_data']]
    logger.info(f"Found {len(has_data)} records with empirical data")

    # Save raw empirical data
    output_path = CACHE_DIR / "ice_empirical_data.parquet"
    empirical_df.to_parquet(output_path)
    logger.info(f"Saved raw empirical data to {output_path}")

    # Calculate coverage statistics
    unique_chemicals = empirical_df['dtxsid'].nunique()
    unique_assays = empirical_df['assay_token'].nunique()

    logger.info(f"\nEmpirical data coverage:")
    logger.info(f"  Unique chemicals: {unique_chemicals}")
    logger.info(f"  Unique assays: {unique_assays}")
    logger.info(f"  Chemical-assay pairs with data: {len(has_data)}")

    # Create binary coverage matrix (chemical × assay)
    # For now, just save the pairs - we'll build the full matrix in the plotting script
    coverage_pairs = has_data[['dtxsid', 'assay_token']].drop_duplicates()
    coverage_path = CACHE_DIR / "ice_empirical_pairs.csv"
    coverage_pairs.to_csv(coverage_path, index=False)
    logger.info(f"Saved {len(coverage_pairs)} unique chemical-assay pairs to {coverage_path}")


if __name__ == "__main__":
    main()
