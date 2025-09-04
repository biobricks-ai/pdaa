"""
Apply the best XGBoost model to predict logRBA for phthalates.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import transforms as mtransforms
from matplotlib.text import Text

import sys
sys.path.append('./')
from scripts.utils.helpers import zscore_columns, get_example_phthalates_df

def mean_column(df: pd.DataFrame, col_title: str):
    # Z = zscore_columns(df)[0]
    Z = df.copy()
    MZ = Z.mean(axis=1)
    MZ.name = col_title

    return MZ, Z

def scale_figure_fonts(fig, factor=1.5):
    # Multiply the fontsize of every Text object in this figure
    for t in fig.findobj(match=Text):
        t.set_fontsize(t.get_fontsize() * factor)


cachedir = Path("cache")
modeldir = cachedir / "combined"
datadir = cachedir / "entity_similarity"

logRBA_label = "Predicted Probability that log(RBA) > 0"

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

# region PREDICTIONS
predictions = pd.Series(dtype=float, index=Z.index)
for index, row in Z.iterrows():
    # Predict the logRBA using the model
    activity_values = row.values.reshape(1, -1)
    predictions.loc[index] = model.predict_proba(activity_values)[0, 1]  # Probability of class 1

# region SCATTERPLOTS
show_scatterplots = False
if show_scatterplots:
    show_simultaneous = False

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.regplot(x=column_means["MAV"], y=predictions, scatter_kws={'s': 10, 'color': 'red'}, line_kws={'color': 'red'})
    # plt.xlabel('Mean Activity Value (MAV)')
    plt.xlabel('MAV and CMAV')
    plt.ylabel(logRBA_label)
    if not show_simultaneous:
        plt.show()

        # Plotting
        plt.figure(figsize=(10, 6))

    sns.regplot(x=column_means["CMAV"], y=predictions, scatter_kws={'s': 10, 'color': 'blue'}, line_kws={'color': 'blue'})
    # plt.xlabel('Clustered Mean Activity Value (CMAV)')
    # plt.ylabel(logRBA_label)
    plt.show()

show_double_plot = False
if show_double_plot:
    plt.figure(figsize=(10, 6))
    sns.regplot(x=column_means["MAV"], y=column_means["CMAV"], scatter_kws={'s': 10, 'color': 'green'}, line_kws={'color': 'green'})
    plt.xlabel('MAV')
    plt.ylabel('CMAV')
    plt.show()


# region HISTOGRAM WITH EXAMPLE PHTHALATES
# histogram + numbered markers and legend (numbers sorted by prediction)
# display_stat = 'density'
display_stat = 'cumulative'
if display_stat in ['count', 'frequency', 'probability']:
    ylabel = display_stat.capitalize()
elif display_stat == 'density':
    ylabel = 'PDF'
elif display_stat == 'cumulative':
    ylabel = 'CDF'
else:
    raise ValueError(f"Invalid display_stat: {display_stat}")

# load examples and sort by predicted value (ascending)
example_phthalates = get_example_phthalates_df()
ex_df = example_phthalates.assign(
    pred=example_phthalates['inchi'].map(predictions)
).sort_values('pred', ascending=True).reset_index(drop=True)

max_example_pred = ex_df['pred'].max()
prediction_threshold = 0.1
resrict_range = max_example_pred < prediction_threshold

if resrict_range:
    predictions_to_plot = predictions[predictions <= prediction_threshold]
    print(f"Limiting histogram to predictions <= {prediction_threshold} to show examples better.")
    print(f"Max example prediction: {max_example_pred:.4f}")
    print(f"Number of predictions shown: {len(predictions_to_plot)} / {len(predictions)}")
else:
    predictions_to_plot = predictions

fig, ax = plt.subplots(figsize=(11, 6))
sns.histplot(
    predictions_to_plot,
    bins=30,
    # kde=False,
    # color='grey',
    color='darkseagreen',
    # alpha=0.4,
    stat='density' if display_stat == 'cumulative' else display_stat,
    cumulative=(display_stat == 'cumulative'),  # cumulative density function
    ax=ax
)
ax.set_xlabel(logRBA_label)  # RBA = relative binding affinity
ax.set_ylabel(ylabel)


# if max_example_pred < 0.1:
#     # limit x-axis to show examples better
#     ax.set_xlim(0, 0.1)

# --- compute stagger tiers based on pixel spacing ---
fig.canvas.draw()  # ensure transforms are up to date
xs = ex_df['pred'].to_numpy(dtype=float)
# transform x-data to display pixels
x_pixels = ax.transData.transform(np.column_stack([xs, np.zeros_like(xs)]))[:, 0]

px_threshold = 16  # min horizontal separation in pixels to keep labels on same row
tiers_last_px = []  # last x-pixel used in each tier
tiers = []          # assigned tier index per label

for xp in x_pixels:
    placed = False
    for t, last in enumerate(tiers_last_px):
        if xp - last >= px_threshold:
            tiers_last_px[t] = xp
            tiers.append(t)
            placed = True
            break
    if not placed:
        tiers_last_px.append(xp)
        tiers.append(len(tiers_last_px) - 1)

# vertical placement for tiers (0=top row)
y_top_frac = 0.98
row_step = 0.045  # vertical gap between staggered rows
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

# draw dashed lines; put a large number near the top; build legend entries
handles, labels = [], []
label_color = 'darkslategrey'
for i, row in ex_df.iterrows():
    idx = i + 1
    x = float(row['pred'])
    ax.axvline(x, color=label_color, linestyle='--', linewidth=1)
    y_frac = y_top_frac - row_step * tiers[i]
    ax.text(x, y_frac, str(idx), transform=trans, ha='center', va='top',
            fontsize=12, color=label_color, fontweight='bold')
    handles.append(Line2D([0], [0], color=label_color, linestyle='--', linewidth=1))
    labels.append(f"{idx}: {row['name']}")

# legend on the right; no frame to keep it clean
ax.legend(handles, labels, title="Example Phthalates:",
          loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.0,
          frameon=False, fontsize=9, title_fontsize=10)

scale_figure_fonts(fig, factor=1.3)

fig.tight_layout()
plt.show()





