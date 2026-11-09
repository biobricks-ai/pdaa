# **Phthalates Data Aggregation and Analysis (PDAA)**

## **Resources**

- [**Google Drive Folder**](https://drive.google.com/drive/folders/1bj4s3G3d6bLRhCHrCUK-s9ka8AvT2wxq?usp=drive_link)

---

## **Overview**

This repository contains the code, data structures, and documentation for the **Phthalates Data Aggregation and Analysis (PDAA)** project, developed by **Insilica LLC** in collaboration with **EMBSI**. The project aims to aggregate, organize, and analyze public data on phthalates, with a focus on their developmental and reproductive toxicity.

- **Project Duration**: July 1, 2024 – December 31, 2024

> **Note**: Many deliverables for this project rely on repositories developed externally to this repository. See **Supporting Resources** below for additional details.

---

## **Objectives**

1. **Data Aggregation**: Create and maintain a comprehensive dataset of chemical-property values for phthalates.
2. **Data Analysis**: Provide statistical insights and prioritize key chemical sources through detailed reports.
3. **Document Extraction**: Collect and process phthalate-related open-access publications, extracting relevant compound mentions.
4. **Predictive Modeling**: Apply and refine predictive models for phthalates, incorporating feedback from EMBSI.
5. **Stakeholder Communication**: Maintain consistent collaboration via biweekly meetings.

---

## **Deliverables**

### **D1 - Data Compilation**

- **Task 1.1**: Aggregate phthalate data into a structured, accessible data store.
- **Task 1.2**: Analyze the compiled data and produce reports highlighting chemical-property distributions and key statistics.
- **Milestone 1**: Deliver a final data export and report, usable by EMBSI without additional support.

### **D2 - Data Extraction from Documents**

- **Task 2.1**: Systematically collect open-access documents related to phthalates.
- **Task 2.2**: Leverage Named Entity Recognition (NER) models to identify and extract phthalate-related compound mentions.
- **Milestone 2**: Deliver a structured corpus containing extracted compound mentions.

### **D3 - Predictive Modeling**

- **Task 3.1**: Run ToxIndex models to predict properties of phthalate compounds.
- **Task 3.2**: Generate reports to guide model updates and application of results.
- **Milestone 3**: Deliver modeling results and associated analysis.

### **D4 - Stakeholder Communication**

- Regular biweekly meetings to ensure alignment, progress tracking, and project updates.

---

## **Supporting Resources**

This project is supported by several external repositories that provide critical datasets and tools for phthalates research:

1. [**PubTator Brick**](https://github.com/biobricks-ai/pubtator): Extracts chemical mentions from PubMed documents.
2. [**Zinc Brick**](https://github.com/biobricks-ai/zinc): A database of 230 million purchasable compounds, including real phthalates.
3. [**PubChem Annotations Brick**](https://github.com/biobricks-ai/pubchem-annotations): Contains millions of chemical annotations organized from PubChem.
4. [**ChemHarmony Brick**](https://github.com/biobricks-ai/chemharmony): A harmonized database of substance-property-value relationships from 15 source databases.
5. [**OpenStemDocs Brick**](https://github.com/biobricks-ai/OpenStemDocs): A collection of open-access peer-reviewed publications for scientific exploration.

## Manuscript analysis code (v12.8)

The latest manuscript, `Phthalate_Frontiers_v12.8.md`, uses the archived 3,797-endpoint model (D1). Its phthalate analyses use the frozen 975-assay panel for developmental and reproductive toxicity (DART) and endocrine disruption (ED). External validation uses a separate 154-assay ICE feature set. The retrained gateway model produces a different activity matrix; its results must be identified separately when comparing model generations.

### Inputs and activity normalization

Run the analysis scripts from the repository root. The primary inputs are:

| Input | Role |
| --- | --- |
| `cache/entity_similarity/activity_matrix_filled.parquet` | D1 predictions, with compound InChIs as rows and assay titles as columns. |
| `cache/descriptors/descriptors.parquet` | Molecular descriptors indexed by the same compound InChIs. |
| `resources/phthalate_lists/commercial_top20.csv` | Commercial compound names, abbreviations, representative SMILES, and chain-length groups. Use the resolved v12.6 structure set retained in v12.8. |
| `cache/adjusted_mav/adj*_assays_*.txt` | Saved assay subsets used for panel-composition sensitivity analyses. |
| `resources/eadb_full.parquet`, `resources/edkb_full.parquet` | Endpoint measurements for external validation. |
| `cache/eadb/activity_matrix_filled.parquet`, `cache/edkb/activity_matrix_filled.parquet` | The 154-assay validation feature matrices. |
| `cache/combined_full/descriptor_matrix.parquet` | Descriptor comparison features for the paired validation benchmark. |

Mean activity value (MAV) is the mean of a compound's assay-wise standardized predictions: each assay column is centered and scaled across compounds before averaging across assays. The reusable plotting scripts use pandas' sample standard deviation (`ddof=1`); the final v12.6 structure-audit scripts use population standard deviation (`ddof=0`). Preserve the relevant generator when reproducing numerical results.

The four scripts `figure2_descriptor_panels.py`, `mw_controlled_descriptor_plots.py`, `phthalates_boxplot_flexible.py`, and `isomer_envelope.py` accept `PDAA_ACTIVITY_MATRIX` as an environment-variable override. The default is the D1 matrix above. Set the override before launching Python; the scripts read it at import time. An override changes the matrix only: descriptor and representative-structure files retain their configured paths.

### Core scripts and functions

| Code | Important functions and manuscript use |
| --- | --- |
| [scripts/utils/helpers.py](scripts/utils/helpers.py) | `get_descriptors()` calculates RDKit and phthalate-specific descriptors. `get_branching_ratio()` measures carbon branching; `longest_carbon_backbone()` measures the longest carbon path in an ester side chain; `largest_carbon_backbone()` counts total carbons in the larger ester side chain. The latter defines the carbon-count metric used in Figure 3. |
| [scripts/figure2_descriptor_panels.py](scripts/figure2_descriptor_panels.py) | Figure 2. `load_data()` aligns MAV and descriptors by InChI. `loess_residuals()` removes the molecular-weight trend from MAV; `loess_curve()` fits descriptor trends. Panel A uses all compounds; panels B–D use molecular weight ≥250 Da by default. Output: `cache/descriptors/mw_adjusted_descriptor_panels/combined_panels.png` and individual panels. |
| [scripts/mw_controlled_descriptor_plots.py](scripts/mw_controlled_descriptor_plots.py) | Supporting molecular-weight analyses. `load_data()` loads or calculates descriptors; `loess_residuals()` supplies the residual analysis. The command-line methods are `partial_residuals`, `bins`, and `sensitivity`, with outputs under `cache/descriptors/`. |
| [scripts/phthalates_boxplot_flexible.py](scripts/phthalates_boxplot_flexible.py) | Reusable chain-group analysis. `load_activity_data()` optionally standardizes assays; `get_phthalate_activities()` matches representative structures to predictions; `create_boxplot()` draws the distributions and optional Conover–Iman and Cliff's delta panels. Outputs are written under `cache/wide_groupings/`. The final Figure 3 uses the companion generator below to apply its compound exclusions. |
| [scripts/isomer_envelope.py](scripts/isomer_envelope.py) | Supplementary Figure S4 and representative-structure sensitivity. `_load_uvcbs()` reads representatives from the commercial CSV; `has_defined_stereo()` excludes InChIs containing `/t` or `/b`; `load_mav()` calculates MAV; `representative_mav()` locates each selected structure. Outputs: `cache/isomer_envelope/isomer_envelope.png` and `isomer_envelope_stats.json`. The envelopes describe formula-matched structures in the prediction set. |
| [scripts/adjusted_mav_tables.py](scripts/adjusted_mav_tables.py) | `build_dart_ed_strength_map()`, `classify_dart_ed_manual()`, and `build_cmre_mask()` define assay subsets; `compute_mav()` calculates subset-specific activity. Outputs under `cache/adjusted_mav/` support the panel sensitivity analyses. |
| [scripts/stages/explain_dataset_model.py](scripts/stages/explain_dataset_model.py) | Model interpretation: `permutation_importance_report()`, `shap_global_local()`, `save_shap_summary_plots()`, and `partial_dependence_plots()` support supplementary Figures S1–S3. `optimize_probability_threshold()` and `train_surrogate_rules()` provide threshold selection and a shallow surrogate tree. |

The residual analyses adjust MAV for molecular weight and compare the residuals with each descriptor. They do not residualize both variables. LOESS defaults to span 0.5; the Figure 2 script falls back to statsmodels LOWESS without confidence bands if scikit-misc is unavailable or the fit fails.

### Final manuscript generators in the companion workspace

Several final reruns are stored separately in `PDAA_manuscript/analysis/`. The following paths are relative to that companion workspace and are **not included in this repository**. Their version suffixes identify the analyses retained in v12.8.

| Manuscript result | Companion code and role |
| --- | --- |
| Figure 1 | `v10_reruns/make_figure1_v122.py`: `main()` standardizes the D1 matrix, clusters rows and columns by correlation distance with average linkage, and writes the heatmap and MAV row order. |
| Figure 3 and commercial MAV values | `structure_audit/make_figure3_v126.py` and `recompute_v126.py`: use the resolved commercial CSV, exclude BBP and DCHP, and retain scoreable representatives, including DTDP. Calculate Kruskal–Wallis, Holm-adjusted Conover–Iman contrasts, Cliff's delta, and DIPP-exclusion sensitivity. |
| Table 2 | `v10_reruns/rerun_table2_v122.py`: `endpoint_series()` prepares endpoint labels; `nested_cv()` compares activity and descriptor features on identical compound intersections, using five outer folds and a three-fold inner randomized search. Requires `v10_reruns/checkpoint.py` for resumable runs. |
| Table S4 | `structure_audit/recompute_tableS4_v128.py`: `mav()` recomputes activity over assay subsets; `order()` groups adjacent commercial-series estimates separated by at most 0.05 MAV units. Includes a random 50% assay-subset control. |
| Table S5 | `structure_audit/recompute_tableS5_v126.py`: `series()` compares the frozen panel with the deterministic panel using the resolved structures. Requires `v10_reruns/results/deterministic_panel.csv`, produced by `v10_reruns/build_deterministic_panel.py`. |
| Tier and training-overlap audits | `structure_audit/gap_rule_check.py` compares representative ordering with isomer-envelope medians; `tier_consistency.py` checks tier separation; `audit_leakage.py` examines validation-compound overlap with the training corpus. |

For exact v12.8 reproduction, retain the resolved input files together with these companion scripts. Some companion scripts contain local absolute paths, and the Figure 1 generator imports an external `mpl_house_style` module. A fresh clone of this repository alone is insufficient to reproduce the complete manuscript.

### Execution and dependencies

The plotting analyses require NumPy, pandas, PyArrow, RDKit, SciPy, matplotlib, seaborn, statsmodels, adjustText, scikit-misc, scikit-posthocs, and cliffs-delta. Model validation and interpretation additionally require scikit-learn, XGBoost, and SHAP. The existing `requirements.txt` lists only part of this analysis environment; `environment.yml` records an older environment and also requires analysis-specific additions.

With the input files available, these commands regenerate Figure 2 and the isomer-envelope analysis using the default D1 matrix:

```bash
python scripts/figure2_descriptor_panels.py --mw_cutoff 250 --span 0.5 --mw_span 0.5
python scripts/isomer_envelope.py
```

A comparison against another matrix can be run explicitly:

```bash
PDAA_ACTIVITY_MATRIX=/path/to/comparison_matrix.parquet \
  python scripts/mw_controlled_descriptor_plots.py --method partial_residuals
```

The scripts write to fixed output directories, so subsequent runs can replace earlier plots. Most `resources/` files and all `cache/` outputs are excluded from Git; obtain the matching archived inputs before running the commands. Gateway submission scripts make remote prediction requests and belong to the model-comparison workflow, outside the D1 reproduction commands above.
