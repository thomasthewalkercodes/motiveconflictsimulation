import pandas as pd
import os
import matplotlib.pyplot as plt


BASE = os.path.dirname(__file__)


def meanSD(BASE):
    df = pd.read_csv(os.path.join(BASE, "active_proportion.csv"))
    df = df[["tag", "simulation", "active proportion"]]  # careful, it only selects 3!

    stats = (
        df.groupby("tag")["active proportion"]
        .agg(mean="mean", std="std")
        .round(3)
        .reset_index()
    )

    df = df.merge(stats, on="tag")

    df.to_csv(os.path.join(BASE, "active_proportion.csv"), index=False)
    print("Mean SD:", stats["std"].mean().round(3))
    print("Max SD", stats["std"].max().round(3))

    return stats


# Mean SD: 0.004
# Max SD 0.145 <- This one is because there are many unsatisfied motives
# which then turn out to be random, which can help or not help
# the amount of behaviors shown


def plot_sd(BASE, stats):
    stats["std"].plot.hist(bins=20, edgecolor="black")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Count")
    plt.title("Distribution of SD in active proportion across conditions")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "sd_distribution.png"))
    plt.show()


def compare_steps(BASE):
    master = pd.read_csv(os.path.join(BASE, "../../master_log.csv"))
    df = pd.read_csv(os.path.join(BASE, "active_proportion.csv"))

    condition_cols = [
        "decay_params",
        "growth_params",
        "influence_params",
        "starting_values_params",
    ]

    # Attach condition columns to each row via tag
    df = df.merge(master[condition_cols + ["tag"]], on="tag")

    # Compute mean/std of active proportion across all step variants
    # of the same condition
    steps_stats = (
        df.groupby(condition_cols)["active proportion"]
        .agg(STEPSmean="mean", STEPSsd="std")
        .round(3)
        .reset_index()
    )

    # Merge back so every row gets the pooled stats
    df = df.merge(steps_stats, on=condition_cols)

    # Drop the condition columns (they were just for grouping)
    df = df.drop(columns=condition_cols)

    df.to_csv(os.path.join(BASE, "active_proportion.csv"), index=False)
    print("Mean StepsSD:", steps_stats["STEPSsd"].mean().round(3))
    print("Max StepsSD", steps_stats["STEPSsd"].max().round(3))
    return df


def plot_stepssd(BASE, stats):
    stats["STEPSsd"].plot.hist(bins=20, edgecolor="black")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Count")
    plt.title("Distribution of STEPsd in active proportion across conditions & steps")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "STEPsd_distribution.png"))
    plt.show()


stats = meanSD(BASE)
plot_sd(BASE, stats)
df = compare_steps(BASE)
plot_stepssd(BASE, df)
