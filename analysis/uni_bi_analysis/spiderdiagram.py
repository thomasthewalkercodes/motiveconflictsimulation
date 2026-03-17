import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
RUN_DIR = Path(
    r"C:\Users\thoma\OneDrive\Dokumente\Interpersonal stuffies\github_motiveconflict\motiveconflictsimulation\runs\uni_bi_series_flat_00000_2026-03-17_18-09-20"
)
SIM_FILE = RUN_DIR / "simulation_0.csv"
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(SIM_FILE)

# Count active steps per motive (active_motive is 1-indexed, NaN when inactive)
motive_cols = [c for c in df.columns if c.startswith("motive_")]
n_motives = len(motive_cols)
labels = [f"Motive {i+1}" for i in range(n_motives)]

counts = (
    df["active_motive"].value_counts().reindex(range(1, n_motives + 1), fill_value=0)
)
total = counts.sum()
ratios = (counts / total).values  # shape: (n_motives,)

# Close the polygon by repeating the first value
angles = np.linspace(0, 2 * np.pi, n_motives, endpoint=False).tolist()
angles += angles[:1]
ratios_plot = np.concatenate([ratios, ratios[:1]])

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

# Fill + line
ax.fill(angles, ratios_plot, alpha=0.25, color="steelblue")
ax.plot(angles, ratios_plot, color="steelblue", linewidth=2)

# Axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11)

# Radial ticks — evenly spaced up to the max ratio, rounded nicely
max_ratio = max(ratios)
ax.set_ylim(0, max_ratio * 1.15)
r_ticks = np.linspace(0, max_ratio, 4)[1:]  # skip 0
ax.set_yticks(r_ticks)
ax.set_yticklabels([f"{v:.0%}" for v in r_ticks], fontsize=8, color="grey")

# Annotate each data point with its exact ratio
for angle, ratio, label in zip(angles[:-1], ratios, labels):
    if ratio > 0:
        ax.annotate(
            f"{ratio:.1%}",
            xy=(angle, ratio),
            xytext=(angle, ratio + max_ratio * 0.07),
            ha="center",
            va="center",
            fontsize=9,
            color="steelblue",
            fontweight="bold",
        )

ax.set_title(
    f"Motive activation ratios\n{RUN_DIR.name}",
    fontsize=12,
    pad=20,
)

plt.tight_layout()
plt.show()
