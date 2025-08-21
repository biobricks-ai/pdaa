"""
Apply the best XGBoost model to predict logRBA for phthalates.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

cachedir = Path("cache")
modeldir = cachedir / "eadb"
datadir = cachedir / "entity_similarity"

# Create a new XGBClassifier instance
model = XGBClassifier()
# Load the model from the JSON file
model.load_model(modeldir / "xgb_classifier_logRBA_model.json")

# load the phthalates data
activity_matrix_filled = pd.read_parquet(
    datadir / "activity_matrix_filled.parquet"
)

# scratch
predictions = []
for index, row in activity_matrix_filled.iterrows():
    # # Get the InChI for the current row
    # inchi = row['inchi']
    
    # Predict the logRBA using the model
    prediction = model.predict_proba(row.values.reshape(1, -1))[0, 1]  # Probability of class 1
    predictions.append(prediction)

predictions = np.array(predictions)