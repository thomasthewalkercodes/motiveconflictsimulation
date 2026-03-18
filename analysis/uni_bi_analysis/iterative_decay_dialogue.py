"""
Iterative decay dialogue
========================
Two influence pairs (A and B) alternate 1 iteration each for N_ROUNDS rounds.
After each iteration, the resulting activation ratios are MIRRORED and passed
as the starting decay for the other pair.

Mirror rule (0-indexed):  mirror(i) = (N_MOTIVES - i) % N_MOTIVES
  0->0, 1->7, 2->6, 3->5, 4->4, 5->3, 6->2, 7->1

Each GIF frame shows both spiders side by side (A left, B right).
8 GIFs are produced, one per cos_decay starting focus (0-7).

Output: output_films/dialogue/
  film_dialogue_<pairA>_<pairB>_<uni_A>_<uni_B>_start-M<n>.gif
"""

import sys
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algorithm.algorithm import algorithm
from algorithm.activation_check import linear_check
from algorithm.growth import flat_growth
from algorithm.influence import uni_bi_influence
from algorithm.starting_values import random_starting_values

# ── Config ────────────────────────────────────────────────────────────────────
N_MOTIVES = 8
STEPS = 1_000_000
ACTIVE_STEPS = 1_000
SEED = 42
GROWTH_RATE = 2
CONFLICT_STR = -0.15
N_ROUNDS = 50  # each round = 1 iter for A + 1 iter for B

# Pair A
PAIR_A = (0, 2)
UNILATERAL_A = True
# Pair B
PAIR_B = (6, 0)
UNILATERAL_B = True

# cos_decay params for initial starting profiles
AMPLITUDE = 0.025
ELEVATION = 0.05
DECAY_SCALE = 0.6

OUT_DIR = Path(__file__).parent / "output_films" / "dialogue"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

# Mirror lookup: mirror(i) = (N_MOTIVES - i) % N_MOTIVES
MIRROR = [(N_MOTIVES - i) % N_MOTIVES for i in range(N_MOTIVES)]

# Quick sanity check printed at startup
print("Mirror map (0-indexed):")
for i, m in enumerate(MIRROR):
    print(f"  {i} -> {m}")


def mirror_ratios(ratios):
    """Reorder ratios according to the mirror map."""
    mirrored = np.zeros_like(ratios)
    for i, m in enumerate(MIRROR):
        mirrored[m] = ratios[i]
    return mirrored


def cos_decay_profile(
    motive_focus, n=N_MOTIVES, amplitude=AMPLITUDE, elevation=ELEVATION
):
    rates = np.zeros(n)
    for i in range(n):
        distance = min(abs(i - motive_focus), n - abs(i - motive_focus))
        angle = distance * (2 * np.pi / n)
        rates[i] = amplitude * np.cos(angle) + elevation
    return rates / rates.sum() * DECAY_SCALE


def ratio_decay(satisfaction_levels, active_motive, decay_rates):
    for i in range(len(satisfaction_levels)):
        if i != active_motive:
            satisfaction_levels[i] -= decay_rates[i]
    return satisfaction_levels


def run_simulation(decay_rates, pair, unilateral):
    np.random.seed(SEED)
    history = algorithm(
        steps=STEPS,
        active_motive_steps=ACTIVE_STEPS,
        activation_check=functools.partial(linear_check),
        growth=functools.partial(flat_growth, growth_rate=GROWTH_RATE),
        influence=functools.partial(
            uni_bi_influence,
            motive_focus=list(pair),
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


def draw_spider(ax, ratios, title, max_ratio, iteration, color="steelblue"):
    ax.clear()
    angles = np.linspace(0, 2 * np.pi, N_MOTIVES, endpoint=False).tolist()
    angles += angles[:1]
    ratios_plot = np.concatenate([ratios, ratios[:1]])

    ax.fill(angles, ratios_plot, alpha=0.25, color=color)
    ax.plot(angles, ratios_plot, color=color, linewidth=1.5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"M{i+1}" for i in range(N_MOTIVES)], fontsize=7)
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
                color=color,
                fontweight="bold",
            )
    ax.set_title(f"{title}\nRound {iteration}", fontsize=8, pad=6)


def make_film(start_focus):
    start_str = f"start-M{start_focus+1}"
    pair_a_str = f"M{PAIR_A[0]+1}-M{PAIR_A[1]+1}"
    pair_b_str = f"M{PAIR_B[0]+1}-M{PAIR_B[1]+1}"
    uni_a_str = "uni" if UNILATERAL_A else "bi"
    uni_b_str = "uni" if UNILATERAL_B else "bi"
    print(f"\n  {start_str}: A={pair_a_str}({uni_a_str})  B={pair_b_str}({uni_b_str})")

    # Initial decay for A comes from cos_decay profile of start_focus
    decay_a = cos_decay_profile(start_focus)
    # B starts with the mirror of A's initial decay
    decay_b = mirror_ratios(decay_a) / mirror_ratios(decay_a).sum() * DECAY_SCALE

    # Each entry: (ratios_a, ratios_b, whose_turn: 'A' or 'B')
    frames_data = []

    for round_idx in range(N_ROUNDS):
        # A runs 1 iteration with its current decay
        hist_a = run_simulation(decay_a, PAIR_A, UNILATERAL_A)
        ratios_a = get_ratios(hist_a)
        frames_data.append(("A", ratios_a.copy(), decay_b.copy()))

        # Mirror A's output -> new decay for B
        mirrored_a = mirror_ratios(ratios_a)
        decay_b = mirrored_a / mirrored_a.sum() * DECAY_SCALE

        # B runs 1 iteration with its new decay
        hist_b = run_simulation(decay_b, PAIR_B, UNILATERAL_B)
        ratios_b = get_ratios(hist_b)
        frames_data.append(("B", ratios_b.copy(), decay_a.copy()))

        # Mirror B's output -> new decay for A
        mirrored_b = mirror_ratios(ratios_b)
        decay_a = mirrored_b / mirrored_b.sum() * DECAY_SCALE

        if (round_idx + 1) % 10 == 0:
            print(f"    round {round_idx+1}/{N_ROUNDS}")

    # For each frame we need both the latest A and B ratios
    # Build parallel lists
    latest_a = np.ones(N_MOTIVES) / N_MOTIVES
    latest_b = np.ones(N_MOTIVES) / N_MOTIVES
    synced_frames = []
    for turn, ratios, _ in frames_data:
        if turn == "A":
            latest_a = ratios
        else:
            latest_b = ratios
        synced_frames.append((latest_a.copy(), latest_b.copy(), turn))

    max_ratio = max(max(f[0].max(), f[1].max()) for f in synced_frames)

    title_a = f"Pair A: {pair_a_str} ({uni_a_str})"
    title_b = f"Pair B: {pair_b_str} ({uni_b_str})"

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))
    fig.suptitle(f"Decay dialogue | {start_str} | mirror: {MIRROR}", fontsize=9)

    def animate(i):
        ra, rb, turn = synced_frames[i]
        # Highlight whose turn it is with a slightly different alpha
        ca = "steelblue" if turn == "A" else "royalblue"
        cb = "tomato" if turn == "B" else "salmon"
        draw_spider(ax_a, ra, title_a, max_ratio, i + 1, color=ca)
        draw_spider(ax_b, rb, title_b, max_ratio, i + 1, color=cb)

    ani = animation.FuncAnimation(
        fig, animate, frames=len(synced_frames), interval=200, repeat=True
    )

    fname = OUT_DIR / (
        f"film_dialogue_{pair_a_str}_{uni_a_str}_{pair_b_str}_{uni_b_str}_{start_str}.gif"
    )
    ani.save(fname, writer="pillow", fps=5)
    plt.close(fig)
    print(f"  Saved: {fname.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
for start_focus in range(N_MOTIVES):
    make_film(start_focus)

print("\nAll done.")
