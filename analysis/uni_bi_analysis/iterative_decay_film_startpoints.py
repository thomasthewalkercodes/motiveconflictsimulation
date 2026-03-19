"""
Iterative decay film — varying starting decay focus
====================================================
Fixed influence pair (0,1) [M1-M2], both unilateral & bilateral.
For each of the 8 decay starting points (cos_decay centered on motive 0-7):
  1. Compute initial decay rates from that cos_decay profile
  2. Run 100 iterations: run -> ratios -> rescale -> new decay rates
  3. Save a GIF of the spider diagram across iterations

Output: 16 GIFs in output_films/startpoints/
  film_M1-M2_unilateral_start-M1.gif  ...  film_M1-M2_unilateral_start-M8.gif
  film_M1-M2_bilateral_start-M1.gif   ...  film_M1-M2_bilateral_start-M8.gif
"""

import sys
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

from algorithm.algorithm import algorithm
from algorithm.activation_check import linear_check
from algorithm.growth import flat_growth
from algorithm.influence import uni_bi_influence
from algorithm.starting_values import random_starting_values

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ── Config ────────────────────────────────────────────────────────────────────
N_MOTIVES = 8
STEPS = 1_000_000
ACTIVE_STEPS = 1_000
SEED = 42
GROWTH_RATE = 2
CONFLICT_STR = -0.15
N_ITERATIONS = 100
PAIR = (0, 2)  # M1 -> M2

# cos_decay params (same as the series)
AMPLITUDE = 0.025
ELEVATION = 0.05
DECAY_SCALE = 0.6  # total decay budget (ratios scaled to sum to this)

OUT_DIR = Path(__file__).parent / "output_films" / "startpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def cos_decay_profile(
    motive_focus, n=N_MOTIVES, amplitude=AMPLITUDE, elevation=ELEVATION
):
    """Return the decay rate vector that cos_decay would apply per motive
    (ignoring the active-motive exclusion, which varies per step)."""
    rates = np.zeros(n)
    for i in range(n):
        distance = min(abs(i - motive_focus), n - abs(i - motive_focus))
        angle = distance * (2 * np.pi / n)
        rates[i] = amplitude * np.cos(angle) + elevation
    # normalise to sum=DECAY_SCALE so it's comparable to ratio-derived rates
    rates = rates / rates.sum() * DECAY_SCALE
    return rates


def ratio_decay(satisfaction_levels, active_motive, decay_rates):
    for i in range(len(satisfaction_levels)):
        if i != active_motive:
            satisfaction_levels[i] -= decay_rates[i]
    return satisfaction_levels


def run_simulation(decay_rates, motive_focus, unilateral):
    np.random.seed(SEED)
    history = algorithm(
        steps=STEPS,
        active_motive_steps=ACTIVE_STEPS,
        activation_check=functools.partial(linear_check),
        growth=functools.partial(flat_growth, growth_rate=GROWTH_RATE),
        influence=functools.partial(
            uni_bi_influence,
            motive_focus=list(motive_focus),
            conflict_strength=CONFLICT_STR,
            unilateral=unilateral,
        ),
        decay=functools.partial(ratio_decay, decay_rates=decay_rates),
        starting_values=functools.partial(
            random_starting_values, num_motives=N_MOTIVES, low=0, high=1
        ),
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
                ha="center",
                va="center",
                fontsize=6,
                color="steelblue",
                fontweight="bold",
            )
    ax.set_title(f"{title}\nIteration {iteration}", fontsize=8, pad=6)


def make_film(pair, unilateral, start_focus):
    uni_label = "unilateral" if unilateral else "bilateral"
    pair_str = f"M{pair[0]+1}-M{pair[1]+1}"
    start_str = f"start-M{start_focus+1}"
    print(f"  {pair_str} {uni_label} {start_str}...")

    decay_rates = cos_decay_profile(start_focus)

    all_ratios = []
    for i in range(N_ITERATIONS):
        history = run_simulation(decay_rates, pair, unilateral)
        ratios = get_ratios(history)
        all_ratios.append(ratios.copy())
        decay_rates = ratios / ratios.sum() * DECAY_SCALE
        if (i + 1) % 10 == 0:
            print(f"    iter {i+1}/{N_ITERATIONS}")

    max_ratio = max(r.max() for r in all_ratios)
    title = f"Influence {pair_str} ({uni_label}) | decay {start_str}"

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    def animate(i):
        draw_spider(ax, all_ratios[i], title, max_ratio, i + 1)

    ani = animation.FuncAnimation(
        fig, animate, frames=N_ITERATIONS, interval=200, repeat=True
    )
    fname = OUT_DIR / f"film_{pair_str}_{uni_label}_{start_str}.gif"
    ani.save(fname, writer="pillow", fps=5)
    plt.close(fig)
    print(f"  Saved: {fname.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
for unilateral in [True, False]:
    label = "unilateral" if unilateral else "bilateral"
    print(f"\n=== {label} ===")
    for start_focus in range(N_MOTIVES):
        make_film(PAIR, unilateral, start_focus)

print("\nAll done.")
