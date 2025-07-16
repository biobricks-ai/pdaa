import numpy as np
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    # Define the path to the parquet file
    cachedir = Path('cache/entity_similarity/cache/entity_similarity/activity_matrix_filled.parquet')
    cachedir.mkdir(parents=True, exist_ok=True)
    activity_matrix_path = cachedir / 'activity_matrix.parquet'

    # Load the activity matrix from the parquet file
    activity_matrix = pd.read_parquet(activity_matrix_path)

    # Convert the DataFrame to a NumPy array
    activity_array = activity_matrix.to_numpy()

    # Print the shape of the NumPy array
    print(f"Shape of the activity matrix as NumPy array: {activity_array.shape}")