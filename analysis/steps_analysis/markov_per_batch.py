# One Markov-chain diagram per run for a batch of runs that share a tag
# prefix. Each diagram pools all simulations inside one run directory.
#
# For the current default (TAG_PREFIX = "master_proof_inf_compare") there are
# 4 matching directories, so 4 PNGs will be saved in the "plots" subfolder.
#
# Style is APA 7 friendly: serif font, plain white background, no titles or
# extra borders, 300 dpi.

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

# ── Configure ────────────────────────────────────────────────────────────
# Every directory in runs/ whose name *starts with* this prefix gets its
# own Markov diagram. Change this to point at a different batch later.
TAG_PREFIX = "master_proof_inf_compare"

# Only draw arrows for transitions with P(j|i) above this. Lower = busier,
# higher = only the strongest links survive.
MARKOV_MIN_PROB = 0.05

N_MOTIVES = 8
OUTPUT_SUBFOLDER = "plots"

# Motive labels in agency (A) / communion (C) notation. Order matches
# motive index 1..8.
MOTIVE_LABELS = [
    "C+",     # 1
    "A+ C+",  # 2
    "A+",     # 3
    "A+ C-",  # 4
    "C-",     # 5
    "A- C-",  # 6
    "A-",     # 7
    "A- C+",  # 8
]

# ── Paths ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
RUNS_DIR = os.path.join(BASE, "..", "..", "runs")
OUTPUT_DIR = os.path.join(BASE, OUTPUT_SUBFOLDER)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── APA 7 styling ────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


# ── Helper: build the transition matrix from this run's simulations ──────
def transition_matrix_from_sequences(sequences):
    """Pool transition counts across simulations and normalise per row, so
    T[i, j] = P(next = j+1 | current = i+1)."""
    T = np.zeros((N_MOTIVES, N_MOTIVES))
    for seq in sequences:
        # zip(seq[:-1], seq[1:]) gives every consecutive pair in the sequence
        for a, b in zip(seq[:-1], seq[1:]):
            T[a - 1, b - 1] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    # Avoid divide-by-zero for motives that never fired.
    row_sums[row_sums == 0] = 1
    return T / row_sums


# ── Helper: draw the Markov diagram ──────────────────────────────────────
def make_markov_plot(transition_matrix, output_path):
    # Place the 8 motives on a unit circle, motive 1 on the right and
    # going counterclockwise (so motive 2 sits just above motive 1).
    angles = np.array([i * 2 * np.pi / N_MOTIVES for i in range(N_MOTIVES)])
    xs = np.cos(angles)
    ys = np.sin(angles)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw arrows first so the node circles sit on top of them.
    for i in range(N_MOTIVES):
        for j in range(N_MOTIVES):
            if i == j:
                continue  # skip self-transitions to keep the picture clean
            p = transition_matrix[i, j]
            if p < MARKOV_MIN_PROB:
                continue
            # Width and opacity both scale with the transition probability,
            # so strong links stand out and weak ones fade away.
            arrow = FancyArrowPatch(
                (xs[i], ys[i]),
                (xs[j], ys[j]),
                connectionstyle="arc3,rad=0.15",  # gentle curve
                arrowstyle="-|>",
                mutation_scale=10 + 25 * p,
                lw=0.5 + 4 * p,
                color="black",
                alpha=0.25 + 0.65 * p,
                shrinkA=14,
                shrinkB=14,
            )
            ax.add_patch(arrow)

    # Draw the nodes (white circle with a black border) and their labels.
    ax.scatter(xs, ys, s=1500, facecolor="white", edgecolor="black",
               lw=1.5, zorder=3)
    for i in range(N_MOTIVES):
        ax.text(
            xs[i], ys[i], MOTIVE_LABELS[i],
            ha="center", va="center", fontsize=9, zorder=4,
        )

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(output_path)
    plt.close(fig)


# ── Main loop: one Markov plot per matching run directory ────────────────
run_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, f"{TAG_PREFIX}_*")))
if not run_dirs:
    print(f"No directories found under runs/ matching '{TAG_PREFIX}_*'")

for run_dir in run_dirs:
    run_name = os.path.basename(run_dir)
    # The run tag is everything up to and including the 5-digit index, i.e.
    # we drop the trailing "_YYYY-MM-DD_HH-MM-SS" timestamp suffix.
    run_tag = "_".join(run_name.split("_")[:-2])
    print(f"Processing {run_name} ...")

    # Load active_motive sequences (one per simulation, NaNs dropped).
    sequences = []
    for sim_file in sorted(glob.glob(os.path.join(run_dir, "simulation_*.csv"))):
        df = pd.read_csv(sim_file, usecols=["active_motive"])
        seq = df["active_motive"].dropna().astype(int).tolist()
        sequences.append(seq)
    if not sequences:
        print(f"  [skip] no simulation_*.csv in {run_name}")
        continue
    print(f"  pooled {len(sequences)} simulations")

    # Build the transition matrix and plot it.
    trans = transition_matrix_from_sequences(sequences)
    output_path = os.path.join(OUTPUT_DIR, f"{run_tag}_markov.png")
    make_markov_plot(trans, output_path)
    print(f"  saved {os.path.basename(output_path)}")

print(f"\nDone. Plots are in {OUTPUT_DIR}")
