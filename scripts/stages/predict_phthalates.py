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

def mean_column(df: pd.DataFrame, col_title: str):
    Z = zscore_columns(df)[0]
    MZ = Z.mean(axis=1)
    MZ.name = col_title

    return MZ, Z

cachedir = Path("cache")
modeldir = cachedir / "combined"
datadir = cachedir / "entity_similarity"

# Create a new XGBClassifier instance
model = XGBClassifier()
# Load the model from the JSON file
model.load_model(modeldir / "xgb_classifier_logRBA_model.json")

fnames = {
    "MAV": datadir / "activity_matrix_filled.parquet",
    "CMAV": "cache/assay_cluster_heatmap/activity_by_cluster.parquet",
}

column_means = {}
for col_title, fname in fnames.items():
    df = pd.read_parquet(fname)
    column_means[col_title], Z_temp = mean_column(df, col_title)
    if col_title == "MAV":
        Z = Z_temp

# # load the phthalates data
# activity_matrix_filled = pd.read_parquet(
#     datadir / "activity_matrix_filled.parquet"
# )
# activity_by_cluster = pd.read_parquet("cache/assay_cluster_heatmap/activity_by_cluster.parquet")

# MAV, Z = mean_column(activity_matrix_filled, "MAV")
# CMAV = mean_column(activity_by_cluster, "CMAV")[0]

# scratch
predictions = []
for index, row in Z.iterrows():
    # # Get the InChI for the current row
    # inchi = row['inchi']
    
    # Predict the logRBA using the model
    prediction = model.predict_proba(row.values.reshape(1, -1))[0, 1]  # Probability of class 1
    predictions.append(prediction)

predictions = np.array(predictions)

show_scatterplots = False

if show_scatterplots:
    show_double_plot = False

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.regplot(x=column_means["MAV"], y=predictions, scatter_kws={'s': 10, 'color': 'red'}, line_kws={'color': 'red'})
    # plt.xlabel('Mean Activity Value (MAV)')
    plt.xlabel('MAV and CMAV')
    plt.ylabel('Binary log(RBA) Prediction')
    if not show_double_plot:
        plt.show()

        # Plotting
        plt.figure(figsize=(10, 6))

    sns.regplot(x=column_means["CMAV"], y=predictions, scatter_kws={'s': 10, 'color': 'blue'}, line_kws={'color': 'blue'})
    # plt.xlabel('Clustered Mean Activity Value (CMAV)')
    # plt.ylabel('Binary log(RBA) Prediction')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.regplot(x=column_means["MAV"], y=column_means["CMAV"], scatter_kws={'s': 10, 'color': 'green'}, line_kws={'color': 'green'})
    plt.xlabel('MAV')
    plt.ylabel('CMAV')
    plt.show()

# label the example phthalates
