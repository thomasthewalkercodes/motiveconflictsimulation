# pip install seabor
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import ast


BASE = os.path.dirname(__file__)


def heatmap_share_sd(BASE):

    master = pd.read_csv(os.path.join(BASE, "../../master_log.csv"))
    df = pd.read_csv(os.path.join(BASE, "share_percentage.csv"))

    master["decay_rate"] = master["decay_params"].apply(
        lambda x: ast.literal_eval(x)["decay_rate"]
    )
    master["amplitude"] = master["influence_params"].apply(
        lambda x: ast.literal_eval(x)["amplitude"]
    )

    df = df.merge(master[["tag", "decay_rate", "amplitude"]], on="tag")

    grouped = (
        df.groupby(["decay_rate", "amplitude"])["share"]
        .std()
        .round(3)
        .reset_index()
        .rename(columns={"share": "sd"})
    )

    # Pivot for heatmap: rows=decay_rate, cols=amplitude
    pivot = grouped.pivot(index="decay_rate", columns="amplitude", values="sd")
    pivot = pivot.sort_index(ascending=False)  # high decay_rate on top

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd")
    plt.xlabel("Amplitude (influence)")
    plt.ylabel("Decay rate")
    plt.title("SD of motive share across conditions")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "heatmap_share_sd.png"))
    plt.show()


heatmap_share_sd(BASE)
