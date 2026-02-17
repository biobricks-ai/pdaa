# ICE Empirical Data Coverage Analysis

This directory contains scripts for downloading and visualizing empirical training data from the NIEHS ICE (Integrated Chemical Environment) database.

## Purpose

The ToxTransformer model was trained on over 21 million bioactivity datapoints from ICE/ToxCast/Tox21. These scripts:

1. Download the empirical observations from ICE API
2. Map chemical identifiers (DTXSID → InChI)
3. Create visualization showing which chemical-assay pairs have empirical data vs. model predictions

## Workflow

### Step 1: Download ICE Empirical Data

```bash
python scripts/download_ice_empirical_data.py
```

This script:
- Reads the 154 DART/ED assays from `resources/dart_ed_assays.csv`
- Fetches dose-response curves from ICE API for each assay
- Extracts chemical identifiers (DTXSID) and hit calls
- Saves empirical observations to `cache/ice_empirical_data.parquet`
- Creates list of chemical-assay pairs with data in `cache/ice_empirical_pairs.csv`

**Expected runtime:** 2-5 minutes (depends on API response time)

**Output files:**
- `cache/ice_empirical_data.parquet` - Raw empirical observations
- `cache/ice_empirical_pairs.csv` - Unique chemical-assay pairs with data

### Step 2: Create Coverage Heatmaps

```bash
python scripts/plot_empirical_coverage.py
```

This script:
- Loads empirical pairs from Step 1
- Maps DTXSID to InChI using CompTox Dashboard API
- Creates binary coverage matrix (chemicals × assays)
- Generates visualization comparing empirical vs prediction coverage

**Expected runtime:** 5-15 minutes (depends on number of DTXSIDs to map)

**Output files:**
- `cache/empirical_coverage/empirical_coverage_heatmap.png` - Dot matrix showing empirical data
- `cache/empirical_coverage/empirical_vs_predictions.png` - Side-by-side comparison
- `cache/empirical_coverage/empirical_coverage_matrix.parquet` - Binary coverage matrix
- `cache/empirical_coverage/coverage_statistics.txt` - Coverage summary report
- `cache/dtxsid_to_inchi_cache.csv` - Cached DTXSID→InChI mappings (reusable)

## Dependencies

Required Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- requests
- tqdm
- rdkit

All should already be installed in the pdaa conda environment.

## API Endpoints

### ICE API
- **Base URL:** `https://ice.ntp.niehs.nih.gov/api/v1/`
- **Curves endpoint:** `/curves?assay={assay_name}`
- **Documentation:** https://ice.ntp.niehs.nih.gov/

### CompTox Dashboard API
- **Base URL:** `https://comptox.epa.gov/dashboard-api/`
- **Chemical detail:** `/ccdapp2/chemical-detail/search/by-dtxsid/{dtxsid}`
- **Documentation:** https://comptox.epa.gov/dashboard/api

## Data Sources

1. **ICE/ToxCast/Tox21** - High-throughput screening assays from EPA and NIH
2. **CompTox Dashboard** - EPA's chemistry dashboard for structure mapping
3. **DART/ED Assays** - 154 assays relevant to developmental and reproductive toxicity

## Output Interpretation

### Empirical Coverage Heatmap
- **Black dots:** Chemical-assay pairs with empirical observations from ICE
- **White space:** Pairs without empirical data (predictions only)
- **X-axis:** Assays (975 total in activity matrix)
- **Y-axis:** Chemicals (956 total in activity matrix)

### Coverage Statistics
- **Coverage percentage:** Proportion of possible pairs with empirical data
- **Chemicals with data:** How many chemicals have ≥1 empirical observation
- **Assays with data:** How many assays have ≥1 empirical observation

## Troubleshooting

**API timeout errors:**
- The scripts include retry logic with exponential backoff
- If persistent, increase timeout in `download_ice_empirical_data.py`

**DTXSID mapping failures:**
- Cached in `cache/dtxsid_to_inchi_cache.csv` for reuse
- CompTox API may rate limit - script includes 0.1s delay between requests

**Missing token mapping:**
- If `cache/entity_similarity/token_title_mapping.csv` doesn't exist
- Run `python scripts/stages/entity_similarity.py` first to generate it

## Notes

- The ICE API is public but may have rate limits
- DTXSID→InChI mapping is cached to avoid redundant API calls
- Coverage patterns may reveal biases in training data
- Not all chemicals in activity matrix will have DTXSID mappings
