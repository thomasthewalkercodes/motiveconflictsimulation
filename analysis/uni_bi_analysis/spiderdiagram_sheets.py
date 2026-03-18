import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(r"C:\Users\thoma\OneDrive\Dokumente\Interpersonal stuffies\github_motiveconflict\motiveconflictsimulation")
RUNS_DIR = ROOT / "runs"
LOG_PATH = ROOT / "master_log.csv"
OUT_DIR = Path(__file__).parent / "output_spiders"
OUT_DIR.mkdir(exist_ok=True)

N_MOTIVES = 8
SOURCE_MOTIVE = 0   # change this to generate sheets for other source motives
# ─────────────────────────────────────────────────────────────────────────────

# ── Load master log ───────────────────────────────────────────────────────────
log = pd.read_csv(
    LOG_PATH,
    names=[
        "tag", "git_commit", "steps", "active_motive_steps", "n_simulations",
        "seed", "activation_check", "activation_check_params",
        "decay", "decay_params", "growth", "growth_params",
        "influence", "influence_params", "starting_values", "starting_values_params",
    ],
    skiprows=1,
    on_bad_lines="skip",
    engine="python",
)
log.columns = log.columns.str.strip()

# Keep only cos series
log = log[log["tag"].str.startswith("uni_bi_series_cos")].copy()

# Parse influence_params and decay_params from string dicts
log["influence_params_parsed"] = log["influence_params"].apply(ast.literal_eval)
log["decay_params_parsed"]     = log["decay_params"].apply(ast.literal_eval)

log["infl_motive_focus"] = log["influence_params_parsed"].apply(lambda d: d["motive_focus"])
log["infl_unilateral"]   = log["influence_params_parsed"].apply(lambda d: d["unilateral"])
log["decay_motive_focus"]= log["decay_params_parsed"].apply(lambda d: d["motive_focus"])

# ── Build tag → folder mapping ────────────────────────────────────────────────
folder_map = {}
for folder in RUNS_DIR.iterdir():
    if not folder.name.startswith("uni_bi_series_cos"):
        continue
    # tag is everything up to the date part: name without last two _YYYY-MM-DD_HH-MM-SS
    parts = folder.name.rsplit("_", 2)  # split off date and time
    tag = parts[0]
    folder_map[tag] = folder

# ── Helper: compute activation ratios from simulation_0.csv ──────────────────
def get_ratios(tag):
    folder = folder_map.get(tag)
    if folder is None:
        return None
    sim_file = folder / "simulation_0.csv"
    df = pd.read_csv(sim_file)
    counts = df["active_motive"].value_counts().reindex(range(1, N_MOTIVES + 1), fill_value=0)
    total = counts.sum()
    return (counts / total).values if total > 0 else np.zeros(N_MOTIVES)

# ── Helper: draw one spider on a given polar axis ────────────────────────────
def draw_spider(ax, ratios, title, max_ratio):
    angles = np.linspace(0, 2 * np.pi, N_MOTIVES, endpoint=False).tolist()
    angles += angles[:1]
    ratios_plot = np.concatenate([ratios, ratios[:1]])

    ax.fill(angles, ratios_plot, alpha=0.25, color="steelblue")
    ax.plot(angles, ratios_plot, color="steelblue", linewidth=1.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"M{i+1}" for i in range(N_MOTIVES)], fontsize=7)
    ax.set_ylim(0, max_ratio * 1.2)

    r_ticks = np.linspace(0, max_ratio, 3)[1:]
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{v:.0%}" for v in r_ticks], fontsize=6, color="grey")

    for angle, ratio in zip(angles[:-1], ratios):
        if ratio > 0:
            ax.annotate(
                f"{ratio:.1%}",
                xy=(angle, ratio),
                xytext=(angle, ratio + max_ratio * 0.12),
                ha="center", va="center",
                fontsize=6, color="steelblue", fontweight="bold",
            )

    ax.set_title(title, fontsize=8, pad=8)

# ── Generate sheets ───────────────────────────────────────────────────────────
# Target pairs: (SOURCE_MOTIVE, x) for x != SOURCE_MOTIVE
target_pairs = [(SOURCE_MOTIVE, x) for x in range(N_MOTIVES) if x != SOURCE_MOTIVE]

for unilateral in [True, False]:
    uni_label = "unilateral" if unilateral else "bilateral"

    for pair in target_pairs:
        # Get the 8 runs for this pair (varying decay_motive_focus 0-7)
        mask = (
            log["infl_motive_focus"].apply(lambda v: tuple(v) == pair) &
            (log["infl_unilateral"] == unilateral)
        )
        subset = log[mask].sort_values("decay_motive_focus")

        if len(subset) == 0:
            print(f"No runs found for pair={pair}, unilateral={unilateral}")
            continue

        # Compute ratios for each of the 8 runs
        all_ratios = []
        for _, row in subset.iterrows():
            r = get_ratios(row["tag"])
            if r is not None:
                all_ratios.append((int(row["decay_motive_focus"]), r))

        if not all_ratios:
            continue

        # Shared max ratio for consistent scale across subplots
        max_ratio = max(r.max() for _, r in all_ratios)

        # Layout: 2 rows × 4 cols for 8 spiders
        fig, axes = plt.subplots(2, 4, figsize=(16, 8),
                                  subplot_kw=dict(polar=True))
        axes_flat = axes.flatten()

        for ax, (decay_focus, ratios) in zip(axes_flat, all_ratios):
            draw_spider(ax, ratios, f"Decay focus: M{decay_focus+1}", max_ratio)

        pair_str = f"({pair[0]+1},{pair[1]+1})"
        fig.suptitle(
            f"Activation ratios — influence pair {pair_str}, {uni_label}\n"
            f"(source motive {SOURCE_MOTIVE+1}, cos decay varying focus)",
            fontsize=11, y=1.01
        )
        plt.tight_layout()

        fname = OUT_DIR / f"spider_M{SOURCE_MOTIVE+1}_pair{pair[0]+1}-{pair[1]+1}_{uni_label}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname.name}")

print("Done.")
