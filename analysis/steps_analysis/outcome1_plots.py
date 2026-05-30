# Per-run plots for the master_reliability experiment.
#
# For each run listed in RUNS_TO_ANALYZE this script saves two PNGs into
# a subfolder next to this file:
#   1) a spider/radar plot of the proportional occurrence of each motive,
#      with motive 1 on the right and the rest going around counterclockwise
#      (so motive 2 sits above motive 1, etc.).
#   2) a Markov-chain diagram showing how likely each motive is to follow
#      another motive.
#
# The style is APA 7 friendly: serif font, plain white background, no
# chartjunk, 300 dpi.

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import FancyArrowPatch

# ── Configure which runs to plot ─────────────────────────────────────────
# Just list the tag prefix; the date/time suffix is matched automatically.
# Add or remove entries as you like.
RUNS_TO_ANALYZE = [
    "outcome1_00007",  # small
    "outcome1_00003",  # medium
    "outcome1_00002",  # high
    "outcome2_00000",  # 10 steps
    "outcome2_00001",  # 20 steps
    "outcome2_00002",  # 30 steps
    "outcome2_00003",  # 50 steps
    "outcome2_00004",  # 100 steps
    "outcome2_00005",  # 500 steps
    "outcome2_00006",  # 1000 steps
    "outcome2_000012",  # 500 steps, 5 sims
]

# Markov diagram: only draw arrows for transitions with P(j|i) above this.
# Lower = busier diagram, higher = only the strongest links survive.
MARKOV_MIN_PROB = 0.05

N_MOTIVES = 8
OUTPUT_SUBFOLDER = "plots"

# Motive labels in agency (A) / communion (C) notation. Order matches
# motive index 1..8.
MOTIVE_LABELS = [
    "C+",  # 1
    "A+ C+",  # 2
    "A+",  # 3
    "A+ C-",  # 4
    "C-",  # 5
    "A- C-",  # 6
    "A-",  # 7
    "A- C+",  # 8
]

# ── Paths ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
RUNS_DIR = os.path.join(BASE, "..", "..", "runs")
OUTPUT_DIR = os.path.join(BASE, OUTPUT_SUBFOLDER)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── APA 7 styling ────────────────────────────────────────────────────────
# Serif font, white background, no extra borders, high-resolution save.
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


# ── Step 1: helpers to compute the two things we need per run ────────────
def proportions_from_sequences(sequences):
    """Return an array of length N_MOTIVES with each motive's share of all
    active steps, pooled across the run's simulations."""
    all_motives = [m for seq in sequences for m in seq]
    counts = (
        pd.Series(all_motives)
        .value_counts()
        .reindex(range(1, N_MOTIVES + 1), fill_value=0)
    )
    return (counts / counts.sum()).to_numpy()


def transition_matrix_from_sequences(sequences):
    """Return an 8x8 matrix where T[i, j] = P(next = j+1 | current = i+1),
    pooling transition *counts* across simulations and normalising per row."""
    T = np.zeros((N_MOTIVES, N_MOTIVES))
    for seq in sequences:
        # zip(seq[:-1], seq[1:]) gives every consecutive pair in the sequence
        for a, b in zip(seq[:-1], seq[1:]):
            T[a - 1, b - 1] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    # Avoid divide-by-zero for motives that never fired in this run.
    row_sums[row_sums == 0] = 1
    return T / row_sums


# ── Step 2: the spider plot ──────────────────────────────────────────────
def make_spider_plot(proportions, output_path):
    # One angle per motive. matplotlib's polar default has 0° at the right
    # and counterclockwise as positive, which is exactly what we want:
    # motive 1 on the right, then motive 2 above-right, etc.
    angles = np.array([i * 2 * np.pi / N_MOTIVES for i in range(N_MOTIVES)])

    # Close the polygon by repeating the first value at the end.
    values_closed = np.append(proportions, proportions[0])
    angles_closed = np.append(angles, angles[0])

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))

    ax.plot(angles_closed, values_closed, color="black", lw=1.5)
    ax.fill(angles_closed, values_closed, color="gray", alpha=0.25)

    # Motive labels around the edge (agency / communion notation).
    ax.set_xticks(angles)
    ax.set_xticklabels(MOTIVE_LABELS)

    # A small amount of headroom so the line doesn't kiss the outer ring.
    ymax = max(proportions) * 1.2
    ax.set_ylim(0, ymax)
    # Show only every second 0.025-step tick: 0.025, 0.075, 0.125, ...
    ticks = np.arange(0.025, ymax, 0.05)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.3f}" for t in ticks])
    ax.set_rlabel_position(90)  # move the radial tick labels off the data
    ax.grid(color="lightgray", lw=0.5)

    fig.savefig(output_path)
    plt.close(fig)


# ── Step 3: the Markov-chain diagram ─────────────────────────────────────
def make_markov_plot(transition_matrix, output_path):
    # Place the 8 motives on a unit circle, same orientation as the spider:
    # motive 1 on the right, counterclockwise going up.
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
                shrinkA=14,  # leave room around the node circles
                shrinkB=14,
            )
            ax.add_patch(arrow)

    # Draw the nodes (white circle with a black border) and their labels
    # in agency / communion notation. Bigger circles + smaller font so the
    # multi-character labels fit.
    ax.scatter(xs, ys, s=1500, facecolor="white", edgecolor="black", lw=1.5, zorder=3)
    for i in range(N_MOTIVES):
        ax.text(
            xs[i],
            ys[i],
            MOTIVE_LABELS[i],
            ha="center",
            va="center",
            fontsize=9,
            zorder=4,
        )

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(output_path)
    plt.close(fig)


# ── Step 4: loop over the configured runs and produce the plots ──────────
for run_tag in RUNS_TO_ANALYZE:
    # Resolve the run directory (the timestamp suffix is matched by the glob).
    matches = glob.glob(os.path.join(RUNS_DIR, f"{run_tag}_*"))
    if not matches:
        print(f"[skip] no directory found for {run_tag}")
        continue
    run_dir = matches[0]
    print(f"Processing {os.path.basename(run_dir)} ...")

    # Read the run's own yaml so the plot title can show the real params.
    yaml_path = glob.glob(os.path.join(run_dir, f"{run_tag}.yaml"))
    cfg = {}
    if yaml_path:
        with open(yaml_path[0], "r") as f:
            cfg = yaml.safe_load(f)
    active_steps = cfg.get("active_motive_steps", "?")
    n_sims = cfg.get("n_simulations", "?")

    # Load active_motive sequences (one per simulation, NaNs dropped).
    sequences = []
    for sim_file in sorted(glob.glob(os.path.join(run_dir, "simulation_*.csv"))):
        df = pd.read_csv(sim_file, usecols=["active_motive"])
        seq = df["active_motive"].dropna().astype(int).tolist()
        sequences.append(seq)
    if not sequences:
        print(f"  [skip] no simulation_*.csv in {run_tag}")
        continue

    # Compute the two things we plot.
    props = proportions_from_sequences(sequences)
    trans = transition_matrix_from_sequences(sequences)

    # Titles are intentionally omitted from the PNGs; the filename and the
    # run's yaml carry the active_motive_steps / n_simulations info.
    _ = (active_steps, n_sims)  # kept for the console log below

    spider_path = os.path.join(OUTPUT_DIR, f"{run_tag}_spider.png")
    markov_path = os.path.join(OUTPUT_DIR, f"{run_tag}_markov.png")

    make_spider_plot(props, spider_path)
    make_markov_plot(trans, markov_path)
    print(
        f"  saved {os.path.basename(spider_path)} and "
        f"{os.path.basename(markov_path)}"
    )

print(f"\nDone. Plots are in {OUTPUT_DIR}")
