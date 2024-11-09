# Phthalates Data Aggregation and Analysis (PDAA)

## Resources
- [google drive folder](https://drive.google.com/drive/folders/1bj4s3G3d6bLRhCHrCUK-s9ka8AvT2wxq?usp=drive_link)

## Overview

This repository contains the code, data structures, and documentation for the **Phthalates Data Aggregation and Analysis (PDAA)** project, developed by Insilica LLC in collaboration with EMBSI. The project's primary focus is to aggregate, organize, and analyze public data related to phthalates, with a specific emphasis on developmental and reproductive toxicity.

- **Project Duration**: July 1, 2024 - December 31, 2024

## Objectives

1. **Data Aggregation**: Compile and maintain a comprehensive chemical-property-value dataset for phthalates.
2. **Data Analysis**: Generate reports with statistical analyses, providing insights into source prioritization and important chemical properties.
3. **Document Extraction**: Collect phthalate-related open-access publications and automatically extract compound mentions.
4. **Predictive Modeling**: Run predictive models on phthalate compounds, updating based on EMBSI's feedback.
5. **Stakeholder Communication**: Ensure consistent collaboration through regular biweekly meetings.

## Deliverables

### D1 - Data Compilation
- **Task 1.1**: Aggregate phthalate data from existing sources, generating a structured data store.
- **Task 1.2**: Analyze aggregated data and generate reports on chemical-property distributions and statistics.
- **Milestone 1**: Final report and data export, usable by EMBSI without ongoing support.

### D2 - Data Extraction From Documents
- **Task 2.1**: Systematic collection of open-access documents related to phthalates.
- **Task 2.2**: Use Named Entity Recognition (NER) models to identify phthalate-related compounds in documents.
- **Milestone 2**: Corpus export with structured compound mentions.

### D3 - Predictive Modeling
- **Task 3.1**: Run ToxIndex models on phthalate data for property prediction.
- **Task 3.2**: Generate reports to aid in model updates and application of results.
- **Milestone 3**: Export of predictive modeling results.

### D4 - Stakeholder Communication
Regular biweekly meetings to align on project progress and updates.

## Project Schedule

This is a six-month project. The project schedule can be compressed by up to three months based on feedback from EMBSI.

| Month       | D1 | D2 | D3 | D4 |
|-------------|----|----|----|----|
| July 2024   | X  |    |    | X  |
| August 2024 | X  | X  |    | X  |
| September 2024 | X  | X  |    | X  |
| October 2024 |    | X  | X  | X  |
| November 2024 |    |    | X  | X  |
| December 2024 |    |    | X  | X  |


Personnel involved:
- **Dr. Thomas Hartung** - Project Advisor
- **Dr. Thomas Luechtefeld** - Senior Developer & Administrator
- **Zakarriya Mughal** - Lead Developer

For more information on the PDAA project, contact Insilica LLC.

## Developed Supporting Resources
1. [**github.com/biobricks-ai/pubtator**](https://github.com/biobricks-ai/pubtator)   
The pubtator brick provides mentions of chemicals in pubmed documents.
2. [**github.com/biobricks-ai/zinc**](https://github.com/biobricks-ai/zinc)   
The zinc brick contains 230 million purchaseable compounds and was used as a source for real phthalates.
3. [**github.com/biobricks-ai/zinc**](https://github.com/biobricks-ai/pubchem-annotations)   
Millions of chemical annotations organized and downloaded from pubchem.
4. [**github.com/biobricks-ai/chemharmony**](https://github.com/biobricks-ai/chemharmony)   
A harmonized resource creating a simple substance-property-value database from 15 source databases.
5. [**github.com/biobricks-ai/zinc**](https://github.com/biobricks-ai/OpenStemDocs)   
A collection of all open access peer reviewed publications.
