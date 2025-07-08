import pathlib
import gc
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import biobricks as bb

# prep
tqdm.pandas()
outdir = pathlib.Path('cache/chemarmony_properties')
outdir.mkdir(exist_ok=True)

# 1) load phthalates and build InChI set
print('Loading Zinc Phthalates data...')
ph_path = 'cache/zinc_phthalates/zinc_phthalates.parquet'
phthalates = pd.read_parquet(ph_path)
phthalates['inchi'] = phthalates['smiles'].progress_apply(
    lambda s: Chem.MolToInchi(Chem.MolFromSmiles(s)) if pd.notnull(s) else None
)
inchi_set = set(phthalates['inchi'].dropna())
# drop smiles to free RAM
del phthalates['smiles']
gc.collect()

# 2) iterate over the ChemHarmony directory, piece by piece
# print('Filtering ChemHarmony by InChI...')
chem_dir = bb.assets('chemharmony').activities_parquet
dataset = ds.dataset(chem_dir, format="parquet")
total_fragments = len(dataset.files)

filtered_chunks = []
for fragment in tqdm(
    dataset.get_fragments(),
    total=total_fragments,
    desc='Filtering ChemHarmony by InChI',
):
    # adjust columns list to only the props you actually need
    table = fragment.to_table()
    df = table.to_pandas()
    mask = df['inchi'].isin(inchi_set)
    if mask.any():
        filtered_chunks.append(df.loc[mask])
    # clean up before next piece
    del table, df, mask
    gc.collect()

chemharmony_filtered = pd.concat(filtered_chunks, ignore_index=True)
del filtered_chunks
gc.collect()

# 3) reload full phthalates (to preserve all columns) and join
print('Reloading Zinc Phthalates for join...')
phthalates = pd.read_parquet(ph_path)
phthalates['inchi'] = phthalates['smiles'].progress_apply(
    lambda s: Chem.MolToInchi(Chem.MolFromSmiles(s)) if pd.notnull(s) else None
)

print('Joining and saving...')
joined = phthalates.merge(chemharmony_filtered, on='inchi', how='inner')
outfile = outdir / 'phthalates_chemharmony.parquet'
joined.to_parquet(outfile, index=False)

print(f'Done - saved to {outfile}')



# # TODO: job getting killed, seems to be a memory issue. Try to reduce memory usage or split the task into smaller chunks.
# #       Also, try to use a more efficient way to handle the data, like using Dask or Polars.
# #       Alternatively, try to use a smaller subset of the data for testing
# #       or increase the memory limit of the job.
# #       Also, consider using a more efficient file format like Feather or Parquet for intermediate results
# import pandas as pd, biobricks as bb, glob, time, pathlib
# from rdkit import Chem
# from tqdm import tqdm
# from multiprocessing import Pool
# import pyarrow.parquet as pq

# tqdm.pandas()

# outdir = pathlib.Path('cache/chemarmony_properties')
# outdir.mkdir(exist_ok=True)

# print('Loading ChemHarmony data...')
# chemharmony = bb.assets('chemharmony')
# chemharmony = pd.read_parquet(chemharmony.activities_parquet)

# print('Loading Zinc Phthalates data...')
# phthalates = pd.read_parquet('cache/zinc_phthalates/zinc_phthalates.parquet')
# phthalates['inchi'] = phthalates['smiles'].progress_apply(lambda x: Chem.MolToInchi(Chem.MolFromSmiles(x)) if pd.notnull(x) else None)

# # join on inchi
# print('Joining Phthalates with ChemHarmony...')
# joined = phthalates.merge(chemharmony, on='inchi', how='inner')

# print('Saving joined data...')
# outfile = outdir / 'phthalates_chemharmony.parquet'
# joined.to_parquet(outfile)