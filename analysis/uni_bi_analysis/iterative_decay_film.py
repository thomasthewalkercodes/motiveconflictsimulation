"""
Iterative decay film
====================
For each influence pair (0, x) [unilateral & bilateral]:
  1. Run the algorithm with the current per-motive decay rates
  2. Compute activation ratios from the result
  3. Rescale ratios -> new per-motive decay rates
  4. Repeat 100 times
  5. Save a GIF (flip-motion film) of the spider diagrams across iterations
"""
import sys
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# ── make sure project root is on path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algorithm.algorithm import algorithm
from algorithm.activation_check import linear_check
from algorithm.growth import flat_growth
from algorithm.influence import uni_bi_influence
from algorithm.starting_values import random_starting_values

# ── Config ────────────────────────────────────────────────────────────────────
N_MOTIVES       = 8
STEPS           = 1_000_000
ACTIVE_STEPS    = 1_000
SEED            = 42
GROWTH_RATE     = 2
CONFLICT_STR    = -0.15
N_ITERATIONS    = 100
SOURCE_MOTIVE   = 0          # influence pairs (SOURCE_MOTIVE, x)

# Decay scaling: activation ratios sum to 1.0
# We scale them so the mean decay rate matches the original elevation (0.05)
# and the spread matches amplitude (0.025).
# Simple approach: decay_rate_i = ratio_i * DECAY_SCALE
# so that sum(decay_rates) = DECAY_SCALE, mean = DECAY_SCALE / N_MOTIVES
DECAY_SCALE = 0.6            # total decay budget per step across all motives

OUT_DIR = Path(__file__).parent / "output_films"
OUT_DIR.mkdir(exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def ratio_decay(satisfaction_levels, active_motive, decay_rates):
    """Per-motive flat decay. decay_rates is a list/array of length N_MOTIVES."""
    for i in range(len(satisfaction_levels)):
        if i != active_motive:
            satisfaction_levels[i] -= decay_rates[i]
    return satisfaction_levels


def run_simulation(decay_rates, motive_focus, unilateral):
    np.random.seed(SEED)

    activation_check_fn = functools.partial(linear_check)
    growth_fn           = functools.partial(flat_growth, growth_rate=GROWTH_RATE)
    influence_fn        = functools.partial(
        uni_bi_influence,
        motive_focus=motive_focus,
        conflict_strength=CONFLICT_STR,
        unilateral=unilateral,
    )
    decay_fn            = functools.partial(ratio_decay, decay_rates=decay_rates)
    starting_values_fn  = functools.partial(
        random_starting_values, num_motives=N_MOTIVES, low=0, high=1
    )

    history = algorithm(
        steps=STEPS,
        active_motive_steps=ACTIVE_STEPS,
        activation_check=activation_check_fn,
        growth=growth_fn,
        influence=influence_fn,
        decay=decay_fn,
        starting_values=starting_values_fn,
    )
    return history


def get_ratios(history):
    active = [a for a in history["active_motive"] if a is not None]
    counts = np.zeros(N_MOTIVES)
    for a in active:
        counts[a] += 1
    total = counts.sum()
    if total == 0:
        return np.ones(N_MOTIVES) / N_MOTIVES
    return counts / total


def ratios_to_decay_rates(ratios):
    """Scale ratios (sum=1) to decay rates (sum=DECAY_SCALE)."""
    return ratios * DECAY_SCALE


def draw_spider(ax, ratios, title, max_ratio, iteration):
    ax.clear()
    angles = np.linspace(0, 2 * np.pi, N_MOTIVES, endpoint=False).tolist()
    angles += angles[:1]
    ratios_plot = np.concatenate([ratios, ratios[:1]])

    ax.fill(angles, ratios_plot, alpha=0.25, color="steelblue")
    ax.plot(angles, ratios_plot, color="steelblue", linewidth=1.5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"M{i+1}" for i in range(N_MOTIVES)], fontsize=8)
    ax.set_ylim(0, max_ratio * 1.2)

    r_ticks = np.linspace(0, max_ratio, 3)[1:]
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{v:.0%}" for v in r_ticks], fontsize=6, color="grey")

    for angle, ratio in zip(angles[:-1], ratios):
        if ratio > 0.001:
            ax.annotate(
                f"{ratio:.1%}",
                xy=(angle, ratio),
                xytext=(angle, ratio + max_ratio * 0.12),
                ha="center", va="center",
                fontsize=6, color="steelblue", fontweight="bold",
            )
    ax.set_title(f"{title}\nIteration {iteration}", fontsize=8, pad=6)


def make_film(pair, unilateral):
    uni_label = "unilateral" if unilateral else "bilateral"
    pair_str  = f"M{pair[0]+1}-M{pair[1]+1}"
    print(f"  Running {pair_str} {uni_label}...")

    # Initial decay rates: uniform
    decay_rates = np.ones(N_MOTIVES) / N_MOTIVES * DECAY_SCALE

    all_ratios = []
    for i in range(N_ITERATIONS):
        history    = run_simulation(decay_rates, list(pair), unilateral)
        ratios     = get_ratios(history)
        all_ratios.append(ratios.copy())
        decay_rates = ratios_to_decay_rates(ratios)
        if (i + 1) % 10 == 0:
            print(f"    iter {i+1}/{N_ITERATIONS}")

    # Shared max ratio for consistent scale
    max_ratio = max(r.max() for r in all_ratios)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    title = f"Influence {pair_str} ({uni_label})"

    frames = []
    for i, ratios in enumerate(all_ratios):
        draw_spider(ax, ratios, title, max_ratio, i + 1)
        fig.canvas.draw()
        # capture frame as image array
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(buf[:, :, 1:])  # drop alpha, keep RGB

    plt.close(fig)

    # Save as GIF using matplotlib's animation
    fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    title2 = f"Influence {pair_str} ({uni_label})"

    def animate(i):
        draw_spider(ax2, all_ratios[i], title2, max_ratio, i + 1)

    ani = animation.FuncAnimation(
        fig2, animate, frames=N_ITERATIONS, interval=200, repeat=True
    )

    fname = OUT_DIR / f"film_{pair_str}_{uni_label}.gif"
    ani.save(fname, writer="pillow", fps=5)
    plt.close(fig2)
    print(f"  Saved: {fname.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
target_pairs = [
    (SOURCE_MOTIVE, x) for x in range(N_MOTIVES) if x != SOURCE_MOTIVE
]

for unilateral in [True, False]:
    label = "unilateral" if unilateral else "bilateral"
    print(f"\n=== {label} ===")
    for pair in target_pairs:
        make_film(pair, unilateral)

print("\nAll done.")
