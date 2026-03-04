import pandas as pd
import os
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)


def plot_sd_by_steps(BASE):
    master = pd.read_csv(os.path.join(BASE, "../../master_log.csv"))
    df = pd.read_csv(os.path.join(BASE, "active_proportion.csv"))

    condition_cols = [
        "decay_params",
        "growth_params",
        "influence_params",
        "starting_values_params",
    ]

    df = df.merge(master[condition_cols + ["tag", "steps"]], on="tag")

    grouped = (
        df.groupby(condition_cols + ["steps"])["active proportion"]
        .std()
        .round(3)
        .reset_index()
        .rename(columns={"active proportion": "sd"})
    )

    sd_by_steps = grouped.groupby("steps")["sd"].mean().round(3).reset_index()
    print(sd_by_steps)

    plt.figure()
    plt.plot(sd_by_steps["steps"], sd_by_steps["sd"], marker="o")
    plt.xlabel("Steps")
    plt.ylabel("Mean SD of active proportion")
    plt.title("SD stabilization across step counts")
    plt.xticks([50, 200, 500, 1000])
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "sd_by_steps.png"))
    plt.show()


plot_sd_by_steps(BASE)
