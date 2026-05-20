# Question:
#   Does the proportion of the 8 motives change, depending on how many
#   n_simulations were run for a given active_motive_steps amount?
#
# Approach:
#   - Find every run whose tag starts with "master_reliability".
#   - Read each run's own yaml to get its active_motive_steps and n_simulations
#     (the master_log.csv column headers are mislabeled, so the yaml is the
#     reliable source of truth).
#   - Pool the "active_motive" column from every simulation_*.csv in that run
#     and compute how often each motive (1..8) appears as a share of all
#     active steps.
#   - Write one row per (active_motive_steps, n_simulations) combination so we
#     can read down the table and see whether the proportions stabilize as
#     n_simulations grows.

import glob
import os

import pandas as pd
import yaml

# ── Paths ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
RUNS_DIR = os.path.join(BASE, "..", "..", "runs")
OUTPUT_PATH = os.path.join(BASE, "motive_proportions_reliability.csv")

N_MOTIVES = 8

# ── Collect every "master_reliability_*" run directory ───────────────────
run_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, "master_reliability_*")))
print(f"Found {len(run_dirs)} run directories.")

rows = []

for i, run_dir in enumerate(run_dirs, start=1):
    # The run's yaml has the same stem as the directory's tag prefix,
    # e.g. master_reliability_00000.yaml inside master_reliability_00000_<date>/
    yaml_files = glob.glob(os.path.join(run_dir, "master_reliability_*.yaml"))
    if not yaml_files:
        print(f"  [skip] no yaml in {os.path.basename(run_dir)}")
        continue

    with open(yaml_files[0], "r") as f:
        cfg = yaml.safe_load(f)

    active_motive_steps = cfg["active_motive_steps"]
    n_simulations = cfg["n_simulations"]

    print(
        f"[{i}/{len(run_dirs)}] {os.path.basename(run_dir)}  "
        f"active_motive_steps={active_motive_steps}, n_simulations={n_simulations}"
    )

    # Pool the active_motive column across every simulation in this run.
    # Pooling (instead of per-sim averaging) keeps the math straightforward:
    # the proportion is just "how many active steps picked motive m" divided
    # by "how many active steps there were overall, across all sims".
    pooled = []
    for sim_file in glob.glob(os.path.join(run_dir, "simulation_*.csv")):
        df = pd.read_csv(sim_file, usecols=["active_motive"])
        # Drop the steps where no motive was active (NaN rows).
        pooled.append(df["active_motive"].dropna())

    if not pooled:
        print("  [skip] no simulation_*.csv files found")
        continue

    all_active = pd.concat(pooled, ignore_index=True).astype(int)
    total_active_steps = len(all_active)

    # Count each motive 1..N_MOTIVES and turn it into a proportion.
    # reindex(...) guarantees every motive appears even if it never fired.
    counts = all_active.value_counts().reindex(range(1, N_MOTIVES + 1), fill_value=0)
    proportions = counts / total_active_steps

    row = {
        "active_motive_steps": active_motive_steps,
        "n_simulations": n_simulations,
        "total_active_steps": total_active_steps,
    }
    for m in range(1, N_MOTIVES + 1):
        row[f"motive_{m}_prop"] = round(float(proportions[m]), 6)
    rows.append(row)

# ── Build the result table ───────────────────────────────────────────────
# Sorting by (active_motive_steps, n_simulations) makes the table easy to
# scan: rows with the same active_motive_steps sit next to each other, so
# you can compare proportions across n_simulations = 1, 5, 10 at a glance.
results_df = pd.DataFrame(rows).sort_values(
    ["active_motive_steps", "n_simulations"]
).reset_index(drop=True)

results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nDone. Results saved to {OUTPUT_PATH}")
print(results_df.to_string(index=False))
