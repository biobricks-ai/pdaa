import pandas as pd

from pathlib import Path

import sys
sys.path.append('./')
from stages.utils.pdaa import predict_all_properties_with_sqlite_cache

resourcedir = Path('resources')
edkb_parquet = resourcedir / 'edkb_log_rba.parquet'
edkb = pd.read_parquet(edkb_parquet)

predictions = predict_all_properties_with_sqlite_cache(edkb['inchi'])
