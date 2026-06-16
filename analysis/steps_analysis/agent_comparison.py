# Compare agent a vs agent b on their angular-change behaviour.
#
# This reads the per-CSV table produced by angular_change_analysis.py
# (angular_change_per_csv.csv) and asks: do the two agents differ in how far
# their behaviour jumps around the circumplex, and how reliably can we tell
# them apart?
#
# The unit of observation is one CSV (one simulation round). Each CSV already
# carries a single mean jump, so each agent contributes a sample of per-CSV
# means. We compare those two samples with:
#   * descriptive stats (n, mean, sd) per agent,
#   * a Welch two-sample t-test (does not assume equal variance),
#   * Cohen's d as a scale-free effect size (the practical "how separable"),
# and we draw overlaid histograms + KDE and a boxplot so the distributions can
# be eyeballed side by side.
#
# Both the plain (mean_jump_deg) and cosine-scaled (mean_jump_cos) metrics are
# compared. Run angular_change_analysis.py first so the input CSV is current.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────────
INPUT_PER_CSV = "angular_change_per_csv.csv"

# Which agents to compare and which per-CSV metrics to test.
AGENT_A = "a"
AGENT_B = "b"
METRICS = [
    ("mean_jump_deg", "Mean angular jump (deg)"),
    ("mean_jump_cos", "Mean cosine-scaled jump"),
    ("opposite_switch_rate", "Opposite-octant switch rate"),
]

OUTPUT_PLOT = "agent_comparison.png"
OUTPUT_STATS = "agent_comparison_stats.csv"
OUTPUT_PER_RUN_STATS = "agent_comparison_per_run.csv"

# Significance threshold used only to count how many runs separate the agents.
ALPHA = 0.05

# ── Paths ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
INPUT_PATH = os.path.join(BASE, INPUT_PER_CSV)

# ── APA-ish styling (matches outcome_plots.py) ───────────────────────────
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


def cohens_d(x, y):
    """Standardised mean difference using the pooled standard deviation."""
    nx, ny = len(x), len(y)
    sp2 = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (
        nx + ny - 2
    )
    sp = np.sqrt(sp2)
    return (np.mean(x) - np.mean(y)) / sp if sp > 0 else 0.0


def interpret_d(d):
    """Plain-language label for an effect size (Cohen's rough conventions)."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


# ── Load the per-CSV table ───────────────────────────────────────────────
if not os.path.exists(INPUT_PATH):
    raise SystemExit(
        f"{INPUT_PER_CSV} not found. Run angular_change_analysis.py first."
    )

df = pd.read_csv(INPUT_PATH)

# ── Compare each metric and collect the stats ────────────────────────────
stats_rows = []
fig, axes = plt.subplots(len(METRICS), 2, figsize=(11, 4.2 * len(METRICS)))
axes = np.atleast_2d(axes)

for row, (metric, label) in enumerate(METRICS):
    a_vals = df.loc[df["agent"] == AGENT_A, metric].dropna().to_numpy()
    b_vals = df.loc[df["agent"] == AGENT_B, metric].dropna().to_numpy()

    # Welch's t-test (unequal variances) + effect size.
    t_res = stats.ttest_ind(a_vals, b_vals, equal_var=False)
    d = cohens_d(a_vals, b_vals)

    stats_rows.append(
        {
            "metric": metric,
            "n_a": len(a_vals),
            "n_b": len(b_vals),
            "mean_a": round(float(np.mean(a_vals)), 4),
            "mean_b": round(float(np.mean(b_vals)), 4),
            "sd_a": round(float(np.std(a_vals, ddof=1)), 4),
            "sd_b": round(float(np.std(b_vals, ddof=1)), 4),
            "mean_diff": round(float(np.mean(a_vals) - np.mean(b_vals)), 4),
            "t": round(float(t_res.statistic), 4),
            "df": round(float(t_res.df), 2),
            "p_value": float(t_res.pvalue),
            "cohens_d": round(float(d), 4),
            "effect": interpret_d(d),
        }
    )

    # ── Left panel: overlaid histograms + KDE ────────────────────────────
    ax_h = axes[row, 0]
    lo = min(a_vals.min(), b_vals.min())
    hi = max(a_vals.max(), b_vals.max())
    bins = np.linspace(lo, hi, 40)
    ax_h.hist(a_vals, bins=bins, density=True, alpha=0.45, color="#4477aa",
              label=f"agent {AGENT_A}")
    ax_h.hist(b_vals, bins=bins, density=True, alpha=0.45, color="#cc6677",
              label=f"agent {AGENT_B}")
    grid = np.linspace(lo, hi, 200)
    for vals, color in ((a_vals, "#4477aa"), (b_vals, "#cc6677")):
        if len(np.unique(vals)) > 1:
            ax_h.plot(grid, stats.gaussian_kde(vals)(grid), color=color, lw=1.8)
    ax_h.axvline(np.mean(a_vals), color="#4477aa", ls="--", lw=1)
    ax_h.axvline(np.mean(b_vals), color="#cc6677", ls="--", lw=1)
    ax_h.set_xlabel(label)
    ax_h.set_ylabel("density")
    ax_h.legend(frameon=False)

    # ── Right panel: boxplot ─────────────────────────────────────────────
    ax_b = axes[row, 1]
    bp = ax_b.boxplot(
        [a_vals, b_vals],
        tick_labels=[f"agent {AGENT_A}", f"agent {AGENT_B}"],
        patch_artist=True,
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], ("#4477aa", "#cc6677")):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for median in bp["medians"]:
        median.set_color("black")
    ax_b.set_ylabel(label)

    # Annotate the panel with the test result.
    p = t_res.pvalue
    p_txt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax_b.set_title(f"d = {d:.2f} ({interpret_d(d)}), {p_txt}")

fig.suptitle(f"Agent {AGENT_A} vs agent {AGENT_B}: angular-change distributions")
fig.tight_layout()
plot_path = os.path.join(BASE, OUTPUT_PLOT)
fig.savefig(plot_path)
plt.close(fig)

# ── Save + print the pooled stats table ──────────────────────────────────
stats_df = pd.DataFrame(stats_rows)
stats_path = os.path.join(BASE, OUTPUT_STATS)
stats_df.to_csv(stats_path, index=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print("=== Pooled across all runs ===")
print(stats_df.to_string(index=False))


# ── Per-run comparison: test agent a vs b inside each run separately ──────
def compare(a_vals, b_vals):
    """Welch t-test + Cohen's d for two samples; returns a dict of results,
    or None if either sample is too small to test."""
    if len(a_vals) < 2 or len(b_vals) < 2:
        return None
    t_res = stats.ttest_ind(a_vals, b_vals, equal_var=False)
    d = cohens_d(a_vals, b_vals)
    return {
        "n_a": len(a_vals),
        "n_b": len(b_vals),
        "mean_a": round(float(np.mean(a_vals)), 4),
        "mean_b": round(float(np.mean(b_vals)), 4),
        "mean_diff": round(float(np.mean(a_vals) - np.mean(b_vals)), 4),
        "t": round(float(t_res.statistic), 4),
        "df": round(float(t_res.df), 2),
        "p_value": float(t_res.pvalue),
        "cohens_d": round(float(d), 4),
        "effect": interpret_d(d),
        "significant": bool(t_res.pvalue < ALPHA),
    }


per_run_rows = []
for run_tag in sorted(df["run_tag"].unique()):
    run_df = df[df["run_tag"] == run_tag]
    for metric, _label in METRICS:
        a_vals = run_df.loc[run_df["agent"] == AGENT_A, metric].dropna().to_numpy()
        b_vals = run_df.loc[run_df["agent"] == AGENT_B, metric].dropna().to_numpy()
        res = compare(a_vals, b_vals)
        if res is None:
            continue
        per_run_rows.append({"run_tag": run_tag, "metric": metric, **res})

per_run_df = pd.DataFrame(per_run_rows)
per_run_path = os.path.join(BASE, OUTPUT_PER_RUN_STATS)
per_run_df.to_csv(per_run_path, index=False)

# Summarise how often a run actually separates the two agents.
print("\n=== Per-run comparison ===")
for metric, _label in METRICS:
    m = per_run_df[per_run_df["metric"] == metric]
    n_runs = len(m)
    n_sig = int(m["significant"].sum())
    n_nonneg = int((m["effect"] != "negligible").sum())
    print(
        f"{metric}: {n_sig}/{n_runs} runs significant (p < {ALPHA}); "
        f"{n_nonneg}/{n_runs} with a non-negligible effect "
        f"(median |d| = {m['cohens_d'].abs().median():.3f})"
    )

print(f"\nWrote plot to         {plot_path}")
print(f"Wrote pooled stats to {stats_path}")
print(f"Wrote per-run stats to {per_run_path}")
