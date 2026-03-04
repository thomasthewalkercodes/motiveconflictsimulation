# analysis on how many steps stabilize the result.
# Comparing 50, 200, 500 & 1000
# Behavior occurance divided by amount of steps.
# over all conditions

import pandas as pd
import os
import glob

BASE = os.path.dirname(__file__)
RUNS_DIR = os.path.join(BASE, "../runs")
MASTER_LOG = os.path.join(BASE, "../master_log.csv")

master = pd.read_csv(MASTER_LOG)
results = []
total = len(master)

for i, (_, run) in enumerate(master.iterrows()):
    tag = run["tag"]
    steps = run["steps"]
    print(f"[{i+1}/{total}] Processing {tag}...")
    run_path = glob.glob(os.path.join(RUNS_DIR, f"{tag}*"))
    if not run_path:
        continue

    for sim_file in glob.glob(os.path.join(run_path[0], "simulation_*.csv")):
        df = pd.read_csv(sim_file)
        results.append(
            {
                "tag": tag,
                "simulation": os.path.basename(sim_file),
                "active proportion": df["active_motive"].notna().sum() / len(df),
            }
        )
results_df = pd.DataFrame(results)
output_path = os.path.join(BASE, "active_proportion.csv")
results_df.to_csv(output_path, index=False)
print(f"\nDone. Results saved to {output_path}")
