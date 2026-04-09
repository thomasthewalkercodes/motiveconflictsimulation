import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import yaml

#########################
#########################
RUN_PREFIX = "ip_further_test"  # <- INPUT YOUR RUN PREFIX HERE (to filter which runs to make films for)
#########################
#########################

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
OUT_DIR = Path(__file__).resolve().parent / "films"
OUT_DIR.mkdir(exist_ok=True)


def load_influence_info(run_dir, person):
    yaml_files = list(run_dir.glob("*.yaml"))
    if not yaml_files:
        return {}
    cfg = yaml.full_load(yaml_files[0].read_text())
    person_key = f"person_{person}"
    inf = cfg.get(person_key, {}).get("influence", {})
    chosen = inf.get("chosen_influence", "")
    params = inf.get(chosen, {})
    unilateral = params.get("unilateral", "?")
    uni_label = "unilateral" if unilateral else "bilateral"
    motive_focus = params.get("motive_focus", "?")
    conflict_strength = params.get("conflict_strength", "?")
    return {
        "chosen": chosen,
        "uni_label": uni_label,
        "motive_focus": motive_focus,
        "conflict_strength": conflict_strength,
    }


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
                ha="center",
                va="center",
                fontsize=6,
                color="steelblue",
                fontweight="bold",
            )
    ax.set_title(f"{title}\nround {round_idx}", fontsize=8, pad=6)


def load_run_config(run_dir):
    yaml_files = list(run_dir.glob("*.yaml"))
    if not yaml_files:
        return {}
    return yaml.full_load(yaml_files[0].read_text())


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
    """group_runs: list of (run_dir, motive_focus_label) sorted by motive_focus."""
    n = len(group_runs)
    ncols = 2  # person A | person B per run
    nrows = n
    all_ratios_a, all_ratios_b, all_labels = [], [], []
    for run_dir, focus_label in group_runs:
        ra = get_behavior_ratios(run_dir, "a", sim, n_motives)
        rb = get_behavior_ratios(run_dir, "b", sim, n_motives)
        all_ratios_a.append(ra)
        all_ratios_b.append(rb)
        all_labels.append(focus_label)

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

    cfg = load_run_config(group_runs[0][0])
    pa = cfg.get("person_a", {}).get("influence", {})
    chosen_a = pa.get("chosen_influence", "")
    params_a = pa.get(chosen_a, {})
    uni_a = "unilateral" if params_a.get("unilateral") else "bilateral"
    cs_a = params_a.get("conflict_strength", "?")
    pb = cfg.get("person_b", {}).get("influence", {})
    chosen_b = pb.get("chosen_influence", "")
    params_b = pb.get(chosen_b, {})
    uni_b = "unilateral" if params_b.get("unilateral") else "bilateral"
    cs_b = params_b.get("conflict_strength", "?")

    fig, axes = plt.subplots(
        nrows,
        ncols,
        subplot_kw={"polar": True},
        figsize=(10, 3.5 * nrows),
    )
    if nrows == 1:
        axes = [axes]
    fig.suptitle(
        f"A: {uni_a} | strength {cs_a}    B: {uni_b} | strength {cs_b}",
        fontsize=9,
    )

    def draw(i):
        for row, (ra, rb, label) in enumerate(
            zip(all_ratios_a, all_ratios_b, all_labels)
        ):
            draw_spider(
                axes[row][0],
                ra[i],
                angles,
                max_ratio,
                n_motives,
                f"A  focus={label}",
                i,
            )
            draw_spider(
                axes[row][1],
                rb[i],
                angles,
                max_ratio,
                n_motives,
                f"B  focus={label}",
                i,
            )

    ani = animation.FuncAnimation(fig, draw, frames=n_rounds, interval=200, repeat=True)
    # name the file after the first run dir (strip timestamp) + sim
    base = group_runs[0][0].name.rsplit("_", 2)[0]
    out_file = OUT_DIR / f"{base}_grouped_sim{sim}.gif"
    ani.save(out_file, writer="pillow", fps=5)
    plt.close(fig)
    print(f"  Saved grouped: {out_file.name}")


def make_film(run_dir, sim, n_motives):
    ratios_a = get_behavior_ratios(run_dir, "a", sim, n_motives)
    ratios_b = get_behavior_ratios(run_dir, "b", sim, n_motives)
    n_rounds = min(len(ratios_a), len(ratios_b))
    if n_rounds == 0:
        print(f"  No data for sim{sim} in {run_dir.name}, skipping.")
        return

    info_a = load_influence_info(run_dir, "a")
    info_b = load_influence_info(run_dir, "b")

    def subtitle(info):
        return (
            f"{info.get('uni_label','')}  |  "
            f"focus: {info.get('motive_focus','')}  |  "
            f"strength: {info.get('conflict_strength','')}"
        )

    max_ratio = max(ratios_a.max(), ratios_b.max())
    angles = np.linspace(0, 2 * np.pi, n_motives, endpoint=False).tolist()
    angles += angles[:1]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, subplot_kw={"polar": True}, figsize=(12, 5))
    fig.suptitle(run_dir.name, fontsize=9)

    def draw(i):
        draw_spider(
            ax_a,
            ratios_a[i],
            angles,
            max_ratio,
            n_motives,
            f"Person A — {subtitle(info_a)}",
            i,
        )
        draw_spider(
            ax_b,
            ratios_b[i],
            angles,
            max_ratio,
            n_motives,
            f"Person B — {subtitle(info_b)}",
            i,
        )

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
