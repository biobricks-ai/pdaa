import biobricks as bb
import pandas as pd

from fastparquet import ParquetFile

# pubtatorAssets = bb.assets('pubtator')

# bioconcepts2pubtator3_parquet = ParquetFile(pubtatorAssets.bioconcepts2pubtator3_parquet).to_pandas()
# print(list(set(bioconcepts2pubtator3_parquet['concept_id'])))

# cellline2pubtator3_parquet = ParquetFile(pubtatorAssets.cellline2pubtator3_parquet).to_pandas()
# print(list(set(cellline2pubtator3_parquet['concept_id'])))

# chemical2pubtator3_parquet = ParquetFile(pubtatorAssets.chemical2pubtator3_parquet).to_pandas()
# print(list(set(chemical2pubtator3_parquet['concept_id'])))

# disease2pubtator3_parquet = ParquetFile(pubtatorAssets.disease2pubtator3_parquet).to_pandas()
# print(list(set(disease2pubtator3_parquet['concept_id'])))

# gene2pubtator3_parquet = ParquetFile(pubtatorAssets.gene2pubtator3_parquet).to_pandas()
# print(list(set(gene2pubtator3_parquet['concept_id'])))

# mutation2pubtator3_parquet = ParquetFile(pubtatorAssets.mutation2pubtator3_parquet).to_pandas()
# print(list(set(mutation2pubtator3_parquet['concept_id'])))

# relation2pubtator3_parquet = ParquetFile(pubtatorAssets.relation2pubtator3_parquet).to_pandas()
# print(list(set(relation2pubtator3_parquet['concept_id'])))

# species2pubtator3_parquet = ParquetFile(pubtatorAssets.species2pubtator3_parquet).to_pandas()
# print(list(set(species2pubtator3_parquet['concept_id'])))






chemharmonyAssets = bb.assets('chemharmony')
print(dir(chemharmonyAssets))

# activities_parquet = ParquetFile(chemharmonyAssets.activities_parquet).to_pandas()
# print(activities_parquet.columns)
properties_parquet = ParquetFile(chemharmonyAssets.substances_parquet).to_pandas()
print(properties_parquet)
# property_categories_parquet = ParquetFile(chemharmonyAssets.property_categories_parquet).to_pandas()
# print(property_categories_parquet.columns)
# property_titles_parquet = ParquetFile(chemharmonyAssets.property_titles_parquet).to_pandas()
# print(property_titles_parquet.columns)
# substances_parquet = ParquetFile(chemharmonyAssets.substances_parquet).to_pandas()
# print(substances_parquet.columns)