import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.transforms import ScaledTranslation
from pathlib import Path
import yaml

#########################
#########################
RUN_PREFIX = (
    "outcome8"  # <- INPUT YOUR RUN PREFIX HERE (to filter which runs to make films for)
)
#########################
#########################

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
OUT_DIR = Path(__file__).resolve().parent / "films"
OUT_DIR.mkdir(exist_ok=True)

# APA 7 styling, kept in sync with outcome_plots.py so the GIF frames and
# the saved PNGs both share the same look (serif font, white background,
# clean axes, 300 dpi for PNG saves).
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


def get_behavior_ratios(run_dir, person, sim, n_motives):
    round_files = sorted(
        run_dir.glob(f"simulation_{person}_sim{sim}_round*.csv"),
        key=lambda p: int(p.stem.split("round")[-1]),
    )
    ratios = []
    for f in round_files:
        df = pd.read_csv(f)
        active = df["active_motive"].dropna().astype(int)
        counts = np.zeros(n_motives)
        for a in active:
            counts[a - 1] += 1  # active_motive is 1-indexed in csv
        total = counts.sum()
        ratios.append(counts / total if total > 0 else counts)
    return np.array(ratios)  # shape: (n_rounds, n_motives)


def draw_spider(ax, ratios, angles, max_ratio, n_motives, round_idx):
    # Same look as outcome_plots.py — black polygon with grey fill,
    # 8 spokes with only cardinal labels named, 5 concentric rings with
    # the 0.125 reference ring overlaid in a darker style, percentage
    # labels at each octant vertex on a white background patch.
    ax.clear()
    fig = ax.figure

    # Polygon (closed by repeating first value, which `angles` already has).
    r_plot = np.concatenate([ratios, ratios[:1]])
    ax.plot(angles, r_plot, color="black", lw=1.5)
    ax.fill(angles, r_plot, color="gray", alpha=0.25)

    # 8 spokes, cardinal labels only (when we actually have 8 motives).
    ax.set_xticks(angles[:-1])
    if n_motives == 8:
        ax.set_xticklabels(["warm", "", "dominant", "", "cold", "", "submissive", ""])
        ax.tick_params(axis="x", labelsize=11)
        # Pull "warm" inward a few points so it sits closer to the outer ring.
        warm_shift = ScaledTranslation(-6 / 72, 0, fig.dpi_scale_trans)
        warm_label = ax.get_xticklabels()[0]
        warm_label.set_transform(warm_label.get_transform() + warm_shift)
    else:
        ax.set_xticklabels([f"M{j+1}" for j in range(n_motives)], fontsize=9)

    # Ensure all rings up to 0.225 fit comfortably regardless of data size.
    ymax = max(max_ratio * 1.3, 0.25)
    ax.set_ylim(0, ymax)

    # Five concentric rings (no numeric labels) with 0.125 in a darker overlay.
    ax.set_yticks([0.025, 0.075, 0.125, 0.175, 0.225])
    ax.set_yticklabels(["", "", "", "", ""])
    ax.grid(color="lightgray", lw=0.5)
    theta_ring = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        theta_ring,
        [0.125] * len(theta_ring),
        color="dimgray",
        lw=1.4,
        zorder=1.5,
    )

    # Percentage labels at each polygon vertex on a small white bbox so
    # the underlying axis / polygon line doesn't muddle the digits.
    label_offset = ymax * 0.06
    for i in range(n_motives):
        ax.text(
            angles[i],
            ratios[i] + label_offset,
            f"{ratios[i] * 100:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2),
        )

    # Only the round counter is kept on top so the animation still tells you
    # which frame you're on; no person/run labels.
    ax.set_title(f"round {round_idx}", fontsize=10, pad=10)


def group_key(cfg):
    """Hashable key of all params except decay.cos_decay.motive_focus."""
    decay = cfg.get("decay", {})
    cos = decay.get("cos_decay", {})
    pa = cfg.get("person_a", {}).get("influence", {})
    pb = cfg.get("person_b", {}).get("influence", {})
    chosen_a = pa.get("chosen_influence", "")
    params_a = pa.get(chosen_a, {})
    chosen_b = pb.get("chosen_influence", "")
    params_b = pb.get(chosen_b, {})
    return (
        decay.get("chosen_decay"),
        cos.get("amplitude"),
        cos.get("elevation"),
        chosen_a,
        str(params_a.get("motive_focus")),
        params_a.get("conflict_strength"),
        params_a.get("unilateral"),
        chosen_b,
        str(params_b.get("motive_focus")),
        params_b.get("conflict_strength"),
        params_b.get("unilateral"),
    )


def make_grouped_film(group_runs, sim, n_motives):
    """group_runs: list of (run_dir, motive_focus_label) sorted by motive_focus.
    The label is no longer rendered on the plots, but the ordering is still
    what the caller uses to lay rows out top-to-bottom."""
    n = len(group_runs)
    ncols = 2  # person A | person B per run
    nrows = n
    all_ratios_a, all_ratios_b = [], []
    for run_dir, _ in group_runs:
        ra = get_behavior_ratios(run_dir, "a", sim, n_motives)
        rb = get_behavior_ratios(run_dir, "b", sim, n_motives)
        all_ratios_a.append(ra)
        all_ratios_b.append(rb)

    n_rounds = min(
        min(len(r) for r in all_ratios_a),
        min(len(r) for r in all_ratios_b),
    )
    if n_rounds == 0:
        print(f"  No data for grouped sim{sim}, skipping.")
        return

    max_ratio = max(
        max(r.max() for r in all_ratios_a),
        max(r.max() for r in all_ratios_b),
    )
    angles = np.linspace(0, 2 * np.pi, n_motives, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(
        nrows,
        ncols,
        subplot_kw={"polar": True},
        figsize=(10, 3.5 * nrows),
    )
    if nrows == 1:
        axes = [axes]

    def draw(i):
        for row, (ra, rb) in enumerate(zip(all_ratios_a, all_ratios_b)):
            draw_spider(axes[row][0], ra[i], angles, max_ratio, n_motives, i)
            draw_spider(axes[row][1], rb[i], angles, max_ratio, n_motives, i)

    ani = animation.FuncAnimation(fig, draw, frames=n_rounds, interval=200, repeat=True)
    # name the file after the first run dir (strip timestamp) + sim
    base = group_runs[0][0].name.rsplit("_", 2)[0]
    out_file = OUT_DIR / f"{base}_grouped_sim{sim}.gif"
    ani.save(out_file, writer="pillow", fps=5)

    # Render the last round and save it as a static PNG alongside the GIF.
    # Same as the single-run case: drop the per-subplot "round N" title so
    # the PNG carries no top labels.
    draw(n_rounds - 1)
    for ax in fig.axes:
        ax.set_title("")
    png_file = OUT_DIR / f"{base}_grouped_sim{sim}_final.png"
    fig.savefig(png_file)

    plt.close(fig)
    print(f"  Saved grouped: {out_file.name} and {png_file.name}")


def make_film(run_dir, sim, n_motives):
    ratios_a = get_behavior_ratios(run_dir, "a", sim, n_motives)
    ratios_b = get_behavior_ratios(run_dir, "b", sim, n_motives)
    n_rounds = min(len(ratios_a), len(ratios_b))
    if n_rounds == 0:
        print(f"  No data for sim{sim} in {run_dir.name}, skipping.")
        return

    max_ratio = max(ratios_a.max(), ratios_b.max())
    angles = np.linspace(0, 2 * np.pi, n_motives, endpoint=False).tolist()
    angles += angles[:1]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, subplot_kw={"polar": True}, figsize=(12, 5))

    def draw(i):
        draw_spider(ax_a, ratios_a[i], angles, max_ratio, n_motives, i)
        draw_spider(ax_b, ratios_b[i], angles, max_ratio, n_motives, i)

    ani = animation.FuncAnimation(fig, draw, frames=n_rounds, interval=200, repeat=True)
    out_file = OUT_DIR / f"{run_dir.name}_sim{sim}.gif"
    ani.save(out_file, writer="pillow", fps=5)

    # Render the last round explicitly and also save it as a static PNG so
    # there's a still snapshot of both agents' final state alongside the GIF.
    # The "round N" title is fine inside the animated GIF but the PNG should
    # have no top labels at all, so we wipe the subplot titles first.
    draw(n_rounds - 1)
    for ax in fig.axes:
        ax.set_title("")
    png_file = OUT_DIR / f"{run_dir.name}_sim{sim}_final.png"
    fig.savefig(png_file)

    plt.close(fig)
    print(f"  Saved: {out_file.name} and {png_file.name}")


if __name__ == "__main__":
    run_dirs = sorted(RUNS_DIR.glob(f"{RUN_PREFIX}*"))
    if not run_dirs:
        print(f"No runs found matching prefix '{RUN_PREFIX}' in {RUNS_DIR}")
    else:
        print(f"Found {len(run_dirs)} run(s) matching '{RUN_PREFIX}'")

    for run_dir in run_dirs:
        if not any(run_dir.glob("simulation_a_sim0_round0.csv")):
            print(f"  Skipping {run_dir.name} (no simulation files found)")
            continue
        yaml_files = list(run_dir.glob("*.yaml"))
        n_motives = yaml.full_load(yaml_files[0].read_text())["n_motives"]
        sims = sorted(
            {
                int(p.stem.split("_sim")[1].split("_")[0])
                for p in run_dir.glob("simulation_a_sim*_round0.csv")
            }
        )
        print(f"  {run_dir.name} — motives: {n_motives}, sims: {sims}")
        for sim in sims:
            make_film(run_dir, sim, n_motives)

    # --- grouped GIFs: group by all params except cos_decay.motive_focus ---
    from collections import defaultdict

    groups = defaultdict(
        list
    )  # key -> list of (run_dir, motive_focus_label, n_motives)
    for run_dir in sorted(RUNS_DIR.glob(f"{RUN_PREFIX}*")):
        yaml_files = list(run_dir.glob("*.yaml"))
        if not yaml_files:
            continue
        cfg = yaml.full_load(yaml_files[0].read_text())
        focus = cfg.get("decay", {}).get("cos_decay", {}).get("motive_focus", "?")
        key = group_key(cfg)
        nm = cfg.get("n_motives", 8)
        groups[key].append((run_dir, str(focus), nm))

    for key, members in groups.items():
        if len(members) < 2:
            continue  # nothing to group
        nm = members[0][2]
        members_sorted = sorted(members, key=lambda x: x[1])
        # find all sims from first run
        first_dir = members_sorted[0][0]
        sims = sorted(
            {
                int(p.stem.split("_sim")[1].split("_")[0])
                for p in first_dir.glob("simulation_a_sim*_round0.csv")
            }
        )
        print(f"  Grouped film: {len(members_sorted)} runs, sims: {sims}")
        for sim in sims:
            make_grouped_film([(d, f) for d, f, _ in members_sorted], sim, nm)

    print("Done.")
