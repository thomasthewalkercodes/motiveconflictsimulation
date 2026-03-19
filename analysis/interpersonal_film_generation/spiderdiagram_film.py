import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

#########################
#########################
RUN_PREFIX = "ip_smallbatch"
#########################
#########################

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
OUT_DIR = Path(__file__).resolve().parent / "films"
OUT_DIR.mkdir(exist_ok=True)


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


def draw_spider(ax, ratios, angles, max_ratio, n_motives, title, round_idx):
    ax.clear()
    r_plot = np.concatenate([ratios, ratios[:1]])
    ax.fill(angles, r_plot, alpha=0.25, color="steelblue")
    ax.plot(angles, r_plot, color="steelblue", linewidth=1.5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"M{j+1}" for j in range(n_motives)], fontsize=8)
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
    ax.set_title(f"{title}\nround {round_idx}", fontsize=8, pad=6)


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

    fig, (ax_a, ax_b) = plt.subplots(1, 2, subplot_kw={"polar": True}, figsize=(10, 5))
    fig.suptitle(run_dir.name, fontsize=9)

    def draw(i):
        draw_spider(ax_a, ratios_a[i], angles, max_ratio, n_motives, "Person A", i)
        draw_spider(ax_b, ratios_b[i], angles, max_ratio, n_motives, "Person B", i)

    ani = animation.FuncAnimation(fig, draw, frames=n_rounds, interval=200, repeat=True)
    out_file = OUT_DIR / f"{run_dir.name}_sim{sim}.gif"
    ani.save(out_file, writer="pillow", fps=5)
    plt.close(fig)
    print(f"  Saved: {out_file.name}")


if __name__ == "__main__":
    run_dirs = sorted(RUNS_DIR.glob(f"{RUN_PREFIX}*"))
    if not run_dirs:
        print(f"No runs found matching prefix '{RUN_PREFIX}' in {RUNS_DIR}")
    else:
        print(f"Found {len(run_dirs)} run(s) matching '{RUN_PREFIX}'")

    for run_dir in run_dirs:
        sample = next(run_dir.glob("simulation_a_sim0_round0.csv"), None)
        if sample is None:
            print(f"  Skipping {run_dir.name} (no simulation files found)")
            continue
        df = pd.read_csv(sample)
        n_motives = len([c for c in df.columns if c.startswith("motive_")])
        sims = sorted({
            int(p.stem.split("_sim")[1].split("_")[0])
            for p in run_dir.glob("simulation_a_sim*_round0.csv")
        })
        print(f"  {run_dir.name} — motives: {n_motives}, sims: {sims}")
        for sim in sims:
            make_film(run_dir, sim, n_motives)

    print("Done.")
