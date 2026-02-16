# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDAA (Phthalates Data Aggregation and Analysis) is a Python-based scientific data pipeline for aggregating, analyzing, and modeling chemical property data for phthalates. Developed by Insilica LLC in collaboration with EMBSI.

## Commands

### Environment Setup
```bash
conda env create -f environment.yml
conda activate pdaa
```

### Running the Pipeline
```bash
# View pipeline DAG
dvc dag

# Run full pipeline
dvc repro

# Run individual stage
python stages/01_process_embsi_files.py
```

### Web Applications
```bash
# Streamlit UI
streamlit run streamlit/streamlit-app.py

# Flask dashboard
python webdashboard/app.py
```

### Interactive Analysis
```bash
jupyter notebook pdaa.ipynb
```

## Architecture

### DVC Pipeline Stages (`stages/`)
Sequential numbered scripts that process data through the pipeline:
- `01_process_embsi_files.py` - Parse EMBSI spreadsheets into parquet
- `02_build_phthalates.py` - Filter ZINC database for phthalates via substructure matching
- `03_chemharmony_properties.py` - Integrate chemical properties from ChemHarmony
- `04_pubtator.py` - Extract phthalate mentions from PubMed via PubTator
- `05_visualize_phthalates.py` - Generate molecular visualizations and cluster heatmaps
- `08_model_phthalates.py` - Train and run predictive models

### Utility Modules (`stages/utils/`)
- `simple_cache.py` - Function caching decorators (`@simple_cache`, `@simple_cache_df`)
- `chemprop.py` - ChemProp Transformer API client (async predictions with retry)
- `openai.py` - OpenAI embeddings and structured outputs
- `pubchem.py` - PubChem REST API with rate limiting
- `sparql.py` - Blazegraph SPARQL queries
- `sqlite_rdf.py` - SQLite-backed RDF store

### Data Flow
- **Input**: `resources/` (EMBSI files, assay mappings)
- **Intermediate**: `cache/<stage_name>/` (parquet files, embeddings, models)
- **Output**: `brick/` (SQLite database with model predictions)

### External Data Sources (via BioBricks)
- ZINC database (purchasable compounds)
- ChemHarmony (harmonized substance-property-value relationships)
- PubTator (chemical mentions in PubMed)
- PubChem (chemical annotations)

## Key Patterns

### Caching
Use `@simple_cache` or `@simple_cache_df` decorators for expensive computations:
```python
from stages.utils.simple_cache import simple_cache, simple_cache_df

@simple_cache(cache_dir="cache/my_function")
def expensive_computation(arg):
    ...
```

### Chemistry Operations
- RDKit for SMILES/InChI handling, fingerprinting, substructure matching
- Morgan fingerprints (radius=2, 2048 bits) for similarity
- Tanimoto similarity for compound comparison
- InChI preferred over SMILES for canonical representation

### API Clients
- ChemProp Transformer: Async HTTP with tenacity retry
- OpenAI: Embeddings (text-embedding-3-small) and structured outputs (GPT-4o)
- PubChem: Rate-limited REST API
- Blazegraph: SPARQL endpoint at `http://localhost:9999/bigdata/namespace/pdaa/sparql`

### SQLite Predictions Database
```python
# Schema: predictions(inchi, property_token, positive_prediction)
# Uses WAL mode and NORMAL synchronous for concurrent access
```

## Environment Variables
Required in `.env`:
- `OPENAI_API_KEY`
- `BIOBRICKS_PUBLIC_TOKEN` (for BioBricks data assets)
