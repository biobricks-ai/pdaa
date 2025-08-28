"""
Apply the best XGBoost model to predict logRBA for phthalates.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append('./')
from scripts.utils.helpers import zscore_columns

cachedir = Path("cache")
modeldir = cachedir / "combined"
datadir = cachedir / "entity_similarity"

# Create a new XGBClassifier instance
model = XGBClassifier()
# Load the model from the JSON file
model.load_model(modeldir / "xgb_classifier_logRBA_model.json")

# load the phthalates data
activity_matrix_filled = pd.read_parquet(
    datadir / "activity_matrix_filled.parquet"
)
Z = zscore_columns(activity_matrix_filled)[0]
MAV = Z.mean(axis=1)

# Comparing predictions with assay clustering
activity_by_cluster = pd.read_parquet("cache/assay_cluster_heatmap/activity_by_cluster.parquet")
ZC = zscore_columns(activity_by_cluster)[0]
CMAV = ZC.mean(axis=1)

# scratch
predictions = []
for index, row in Z.iterrows():
    # # Get the InChI for the current row
    # inchi = row['inchi']
    
    # Predict the logRBA using the model
    prediction = model.predict_proba(row.values.reshape(1, -1))[0, 1]  # Probability of class 1
    predictions.append(prediction)

predictions = np.array(predictions)

# Plotting
plt.figure(figsize=(10, 6))
sns.regplot(x=MAV, y=predictions, scatter_kws={'s': 10, 'color': 'red'}, line_kws={'color': 'red'})
# plt.xlabel('Mean Activity Value (MAV)')
plt.xlabel('MAV and CMAV')
plt.ylabel('Binary log(RBA) Prediction')
# plt.show()




# # Plotting
# plt.figure(figsize=(10, 6))
sns.regplot(x=CMAV, y=predictions, scatter_kws={'s': 10, 'color': 'blue'}, line_kws={'color': 'blue'})
# plt.xlabel('Clustered Mean Activity Value (CMAV)')
# plt.ylabel('Binary log(RBA) Prediction')
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(x=MAV, y=CMAV, scatter_kws={'s': 10, 'color': 'green'}, line_kws={'color': 'green'})
plt.xlabel('MAV')
plt.ylabel('CMAV')
plt.show()