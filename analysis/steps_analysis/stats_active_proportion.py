import pandas as pd
import os

BASE = os.path.dirname(__file__)
import matplotlib.pyplot as plt


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


stats = meanSD(BASE)
plot_sd(BASE, stats)
