import pandas as pd
import os
import glob

# pubtator_brick_path = '/home/jjaramillo/workspace/PDAA/temporalRepos/pubtator/brick'


# for path in glob.glob(os.path.join('/home/jjaramillo/workspace/PDAA/temporalRepos/pubtator/brick', '*.parquet')):
#     df = pd.read_parquet(path)
#     print(set(df[df['Type']]=='Chemical'))
# pubtator_paths = glob.glob(os.path.join(pubtator_brick_path, '*.parquet'))

# for path in pubtator_paths:
#     pubtator = pd.read_parquet(path)
#     print(pubtator.head())