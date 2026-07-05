"""
Two-panel line plot of Current MAV vs three adjusted MAV variants.

Phthalates are ordered by descending Current MAV (z-scored).

Usage:
    python scripts/plot_adjusted_mav.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("cache/adjusted_mav")

raw = pd.read_csv(OUTPUT_DIR / "adjusted_mav_raw.csv")
zsc = pd.read_csv(OUTPUT_DIR / "adjusted_mav_zscore.csv")

order = zsc.sort_values("Current MAV", ascending=False)["Phthalate"].tolist()
raw = raw.set_index("Phthalate").loc[order]
zsc = zsc.set_index("Phthalate").loc[order]

colors = {
    "Current MAV": "#2c5f7c",
    "Adjusted MAV 1": "#e07b54",
    "Adjusted MAV 2": "#5ea86e",
    "Adjusted MAV 3": "#9b6fb0",
}
labels = {
    "Current MAV": "Current MAV (all 975)",
    "Adjusted MAV 1": "Adj 1: ToxTransformer DART/ED > 8 (236)",
    "Adjusted MAV 2": "Adj 2: Manual DART/ED keywords (329)",
    "Adjusted MAV 3": "Adj 3: CMRE intersection (931)",
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)

for col in colors:
    ax1.plot(range(len(raw)), raw[col], marker='o', ms=5, color=colors[col],
             label=labels[col], linewidth=1.5, alpha=0.85)
    ax2.plot(range(len(zsc)), zsc[col], marker='o', ms=5, color=colors[col],
             label=labels[col], linewidth=1.5, alpha=0.85)

for ax, ylabel, title in [
    (ax1, "MAV (raw)", "Raw MAV"),
    (ax2, "MAV (z-scored)", "Z-scored MAV"),
]:
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="medium")
    ax.grid(True, ls="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax2.set_xticks(range(len(order)))
ax2.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)

plt.tight_layout()
outpath = OUTPUT_DIR / "adjusted_mav_lineplot.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {outpath}")
