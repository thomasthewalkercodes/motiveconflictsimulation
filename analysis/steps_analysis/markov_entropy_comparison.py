# Markov / conditional-entropy comparison of agent a vs agent b.
#
# The earlier scripts collapse a whole behaviour sequence to one scalar (mean
# jump), which washes most of the structure out and leaves only a tiny effect.
# Here we keep the structure: we look at the transition matrix
#       T[i, j] = P(X_{t+1} = j | X_t = i)
# and at the per-state conditional entropy
#       H(X_{t+1} | X_t = i) = - sum_j T[i, j] * log2 T[i, j].
# H_i answers "once the agent is in motive i, how predictable is its next
# move?" Low H_i = motive i is 'locked in' (it almost always leads somewhere
# specific); high H_i = it scatters.
#
# Why this gives a clearer effect: the two agents are built from different
# influence matrices. Agent a has a single warm-cold conflict (motive 1 <-> 5,
# -0.15), so being in 1 makes 5 the next pressing need and vice versa -- a 1<->5
# oscillation that should LOCK IN states 1 and 5. Agent b has a smooth
# circumplex, so its states scatter to neighbours. The difference is therefore
# concentrated in a couple of rows, so a per-state measure isolates a large
# effect where the global mean saw almost nothing.
#
# For each motive we report:
#   * H_i for each agent (pooled), and a per-CSV t-test + Cohen's d on H_i,
#   * Jensen-Shannon divergence between the agents' rows T_a[i] vs T_b[i]
#     (a bounded 0..1 'how different are the dynamics from i' number),
#   * a G-test (likelihood-ratio chi-square) of homogeneity for row i, with
#     Cramer's V as effect size,
# and we flag which motives the THEORY (the influence matrices) says should
# differ, so the empirical hits can be checked against the model.

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────────
RUNS_TO_ANALYZE = ["newoutcome1"]
N_MOTIVES = 8
AGENT_A = "a"
AGENT_B = "b"

# A CSV's row i is only used for the per-CSV entropy distribution if motive i
# was left at least this many times in that CSV (keeps the entropy estimate
# stable). With ~1000 steps / 8 states each row sees ~125 transitions.
MIN_ROW_COUNT = 20

OUTPUT_STATS = "markov_entropy_stats.csv"
OUTPUT_TARGETED = "markov_targeted_cells.csv"
OUTPUT_HEATMAP = "markov_transition_heatmaps.png"
OUTPUT_PERSTATE = "markov_perstate_comparison.png"

# How many theory-predicted transition cells to test directly. The influence
# matrices of a and b differ most at the conflict cells; testing just those
# (instead of all 64) is a confirmatory test with far more power.
N_TARGET_CELLS = 4

# ── Paths ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
RUNS_DIR = os.path.join(BASE, "..", "..", "runs")

# Motive labels in agency/communion notation (index 1..8), as in outcome_plots.
MOTIVE_LABELS = ["C+", "A+C+", "A+", "A+C-", "C-", "A-C-", "A-", "A-C+"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


# ── Helpers ──────────────────────────────────────────────────────────────
def resolve_run_dirs(entries):
    seen, out = set(), []
    for entry in entries:
        for d in sorted(glob.glob(os.path.join(RUNS_DIR, f"{entry}_*"))):
            if os.path.isdir(d) and d not in seen:
                seen.add(d)
                out.append(d)
    return out


def agent_from_filename(csv_name):
    m = re.match(r"simulation_([A-Za-z]+)_sim", csv_name)
    return m.group(1) if m else "all"


def count_matrix(seq):
    """8x8 transition-count matrix for one sequence (motives 1..8)."""
    C = np.zeros((N_MOTIVES, N_MOTIVES))
    for a, b in zip(seq[:-1], seq[1:]):
        C[a - 1, b - 1] += 1
    return C


def row_entropy(counts_row):
    """Conditional entropy (bits) of the next state given this row's counts."""
    total = counts_row.sum()
    if total == 0:
        return np.nan
    p = counts_row / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def js_divergence(p, q):
    """Jensen-Shannon divergence (bits, 0..1) between two prob vectors."""
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cohens_d(x, y):
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sp2 = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    sp = np.sqrt(sp2)
    return (np.mean(x) - np.mean(y)) / sp if sp > 0 else 0.0


def interpret_d(d):
    ad = abs(d)
    return ("negligible", "small", "medium", "large")[
        sum(ad >= t for t in (0.2, 0.5, 0.8))
    ]


# ── Load theoretical influence matrices (for the overlay) ────────────────
def load_influence(run_dir, agent):
    path = os.path.join(run_dir, "influence_matrices", f"influence_matrix_{agent}.csv")
    return pd.read_csv(path, index_col=0).to_numpy()


# ── Gather data ──────────────────────────────────────────────────────────
run_dirs = resolve_run_dirs(RUNS_TO_ANALYZE)
if not run_dirs:
    raise SystemExit("No runs matched.")

# Pooled transition counts per agent, plus per-CSV per-state entropy samples
# and the full per-CSV count matrices (kept so targeted cells can be tested).
pooled = {AGENT_A: np.zeros((N_MOTIVES, N_MOTIVES)),
          AGENT_B: np.zeros((N_MOTIVES, N_MOTIVES))}
per_csv_entropy = {AGENT_A: [[] for _ in range(N_MOTIVES)],
                   AGENT_B: [[] for _ in range(N_MOTIVES)]}
per_csv_counts = {AGENT_A: [], AGENT_B: []}

for run_dir in run_dirs:
    for sim_file in glob.glob(os.path.join(run_dir, "simulation_*.csv")):
        agent = agent_from_filename(os.path.basename(sim_file))
        if agent not in pooled:
            continue
        seq = pd.read_csv(sim_file, usecols=["active_motive"])["active_motive"]
        seq = seq.dropna().astype(int).tolist()
        if len(seq) < 2:
            continue
        C = count_matrix(seq)
        pooled[agent] += C
        per_csv_counts[agent].append(C)
        for i in range(N_MOTIVES):
            if C[i].sum() >= MIN_ROW_COUNT:
                per_csv_entropy[agent][i].append(row_entropy(C[i]))

# Pooled transition probability matrices.


def to_prob(C):
    rs = C.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return C / rs


P_a, P_b = to_prob(pooled[AGENT_A]), to_prob(pooled[AGENT_B])

# Theoretical structure: which states have a non-zero influence row.
infl_a = load_influence(run_dirs[0], AGENT_A)
infl_b = load_influence(run_dirs[0], AGENT_B)
theory_states_a = {i for i in range(N_MOTIVES) if np.any(infl_a[i] != 0)}
theory_states_b = {i for i in range(N_MOTIVES) if np.any(infl_b[i] != 0)}

# ── Per-state statistics ─────────────────────────────────────────────────
rows = []
for i in range(N_MOTIVES):
    ea = np.array(per_csv_entropy[AGENT_A][i])
    eb = np.array(per_csv_entropy[AGENT_B][i])

    # t-test + effect size on per-CSV entropy of state i.
    if len(ea) >= 2 and len(eb) >= 2:
        t_res = stats.ttest_ind(ea, eb, equal_var=False)
        t_p = float(t_res.pvalue)
        d = cohens_d(ea, eb)
    else:
        t_p, d = np.nan, np.nan

    # Jensen-Shannon divergence between the pooled rows.
    jsd = js_divergence(P_a[i], P_b[i])

    # G-test of homogeneity for row i (drop columns zero in both agents).
    ca, cb = pooled[AGENT_A][i], pooled[AGENT_B][i]
    keep = (ca + cb) > 0
    table = np.vstack([ca[keep], cb[keep]])
    if table.shape[1] > 1 and table.sum() > 0:
        g, g_p, _, _ = stats.chi2_contingency(table, lambda_="log-likelihood")
        cramers_v = float(np.sqrt(g / (table.sum() * (min(table.shape) - 1))))
    else:
        g_p, cramers_v = np.nan, np.nan

    rows.append(
        {
            "motive": i + 1,
            "label": MOTIVE_LABELS[i],
            "H_a_bits": round(row_entropy(pooled[AGENT_A][i]), 4),
            "H_b_bits": round(row_entropy(pooled[AGENT_B][i]), 4),
            "H_diff": round(row_entropy(pooled[AGENT_A][i])
                            - row_entropy(pooled[AGENT_B][i]), 4),
            "entropy_cohens_d": round(d, 4) if not np.isnan(d) else np.nan,
            "entropy_effect": interpret_d(d) if not np.isnan(d) else "n/a",
            "entropy_p": t_p,
            "jsd_bits": round(jsd, 4),
            "gtest_p": g_p,
            "cramers_v": round(cramers_v, 4) if not np.isnan(cramers_v) else np.nan,
            "theory_a": (i in theory_states_a),
            "theory_b": (i in theory_states_b),
        }
    )

stats_df = pd.DataFrame(rows)
stats_df.to_csv(os.path.join(BASE, OUTPUT_STATS), index=False)

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 30)
print("=== Per-state Markov / entropy comparison (agent a vs b) ===\n")
print(stats_df.to_string(index=False))

# Headline: which motive separates the agents best, by each measure.
best_d = stats_df.loc[stats_df["entropy_cohens_d"].abs().idxmax()]
best_jsd = stats_df.loc[stats_df["jsd_bits"].idxmax()]
print(
    f"\nLargest entropy effect: motive {int(best_d['motive'])} "
    f"({best_d['label']}), d = {best_d['entropy_cohens_d']:.2f} "
    f"({best_d['entropy_effect']})"
)
print(
    f"Largest row divergence: motive {int(best_jsd['motive'])} "
    f"({best_jsd['label']}), JSD = {best_jsd['jsd_bits']:.3f} bits"
)
print(f"Theory says agent a is structured at motives: "
      f"{sorted(m + 1 for m in theory_states_a)}")


# ── Theory-targeted cell tests ───────────────────────────────────────────
# The influence matrices of a and b differ cell-by-cell; the transition matrix
# should differ most exactly where the *influence* differs. So instead of
# testing all 64 cells (diffuse, multiple-comparison-heavy), we rank cells by
# |infl_a - infl_b| and test only the top few directed transitions P(j|i),
# comparing the per-CSV cell probability between agents. This is the
# confirmatory, theory-informed version of "where do the matrices differ?".
def cell_probs(agent, i, j):
    """Per-CSV P(next=j+1 | cur=i+1) for every CSV where row i had enough mass."""
    out = []
    for C in per_csv_counts[agent]:
        ri = C[i].sum()
        if ri >= MIN_ROW_COUNT:
            out.append(C[i, j] / ri)
    return np.array(out)


infl_gap = np.abs(infl_a - infl_b)
# Rank off-diagonal cells by how much the theory separates the two agents.
cell_order = sorted(
    ((i, j) for i in range(N_MOTIVES) for j in range(N_MOTIVES) if i != j),
    key=lambda ij: infl_gap[ij],
    reverse=True,
)

target_rows = []
for i, j in cell_order[:N_TARGET_CELLS]:
    a_p, b_p = cell_probs(AGENT_A, i, j), cell_probs(AGENT_B, i, j)
    t_res = stats.ttest_ind(a_p, b_p, equal_var=False)
    target_rows.append(
        {
            "transition": f"{i + 1}->{j + 1}",
            "from_label": MOTIVE_LABELS[i],
            "to_label": MOTIVE_LABELS[j],
            "infl_gap": round(float(infl_gap[i, j]), 4),
            "P_a": round(float(a_p.mean()), 4),
            "P_b": round(float(b_p.mean()), 4),
            "diff": round(float(a_p.mean() - b_p.mean()), 4),
            "cohens_d": round(float(cohens_d(a_p, b_p)), 4),
            "effect": interpret_d(cohens_d(a_p, b_p)),
            "p_value": float(t_res.pvalue),
        }
    )

# Combined "conflict oscillation" metric for the single conflict pair of agent
# a: how often, when in either conflict motive, the next move is the other one.
conflict_pair = None
for i, j in cell_order[:N_TARGET_CELLS]:
    if (j, i) in [(c[1], c[0]) for c in cell_order[:N_TARGET_CELLS]] and i < j:
        conflict_pair = (i, j)
        break


def oscillation(agent, i, j):
    out = []
    for C in per_csv_counts[agent]:
        denom = C[i].sum() + C[j].sum()
        if denom > 0:
            out.append((C[i, j] + C[j, i]) / denom)
    return np.array(out)


if conflict_pair is not None:
    i, j = conflict_pair
    a_o, b_o = oscillation(AGENT_A, i, j), oscillation(AGENT_B, i, j)
    t_res = stats.ttest_ind(a_o, b_o, equal_var=False)
    target_rows.append(
        {
            "transition": f"{i + 1}<->{j + 1} osc",
            "from_label": MOTIVE_LABELS[i],
            "to_label": MOTIVE_LABELS[j],
            "infl_gap": round(float(infl_gap[i, j]), 4),
            "P_a": round(float(a_o.mean()), 4),
            "P_b": round(float(b_o.mean()), 4),
            "diff": round(float(a_o.mean() - b_o.mean()), 4),
            "cohens_d": round(float(cohens_d(a_o, b_o)), 4),
            "effect": interpret_d(cohens_d(a_o, b_o)),
            "p_value": float(t_res.pvalue),
        }
    )

target_df = pd.DataFrame(target_rows)
target_df.to_csv(os.path.join(BASE, OUTPUT_TARGETED), index=False)
print("\n=== Theory-targeted transition-cell tests (highest |infl_a - infl_b|) ===\n")
print(target_df.to_string(index=False))

# ── Plot 1: transition-matrix heatmaps (a, b, difference) ────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ticks = [f"{i+1}\n{MOTIVE_LABELS[i]}" for i in range(N_MOTIVES)]
for ax, M, title, cmap, vlim in (
    (axes[0], P_a, f"Agent {AGENT_A}: P(next | current)", "viridis", (0, None)),
    (axes[1], P_b, f"Agent {AGENT_B}: P(next | current)", "viridis", (0, None)),
    (axes[2], P_a - P_b, "Difference (a - b)", "coolwarm", (None, None)),
):
    if title.startswith("Difference"):
        vmax = np.abs(P_a - P_b).max()
        im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(M, cmap=cmap, vmin=0)
    ax.set_xticks(range(N_MOTIVES))
    ax.set_xticklabels([str(i + 1) for i in range(N_MOTIVES)])
    ax.set_yticks(range(N_MOTIVES))
    ax.set_yticklabels(ticks, fontsize=7)
    ax.set_xlabel("next motive")
    ax.set_ylabel("current motive")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Transition matrices: empirical P(next | current)")
fig.tight_layout()
fig.savefig(os.path.join(BASE, OUTPUT_HEATMAP))
plt.close(fig)

# ── Plot 2: per-state entropy + JSD bars ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.arange(N_MOTIVES)
w = 0.38
axes[0].bar(x - w / 2, stats_df["H_a_bits"], w, label=f"agent {AGENT_A}",
            color="#4477aa", alpha=0.8)
axes[0].bar(x + w / 2, stats_df["H_b_bits"], w, label=f"agent {AGENT_B}",
            color="#cc6677", alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"{i+1}\n{MOTIVE_LABELS[i]}" for i in range(N_MOTIVES)],
                        fontsize=7)
axes[0].set_ylabel("H(next | current)  [bits]")
axes[0].set_title("Per-state conditional entropy (lower = more locked in)")
axes[0].legend(frameon=False)

bars = axes[1].bar(x, stats_df["jsd_bits"], color="#228833", alpha=0.8)
# Mark the theoretically-structured states for agent a.
for i in theory_states_a:
    bars[i].set_color("#ee6677")
    bars[i].set_alpha(0.95)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"{i+1}\n{MOTIVE_LABELS[i]}" for i in range(N_MOTIVES)],
                        fontsize=7)
axes[1].set_ylabel("Jensen-Shannon divergence  [bits]")
axes[1].set_title("Per-row a-vs-b divergence (red = theory-predicted conflict)")
fig.tight_layout()
fig.savefig(os.path.join(BASE, OUTPUT_PERSTATE))
plt.close(fig)

print(f"\nWrote per-state stats -> {os.path.join(BASE, OUTPUT_STATS)}")
print(f"Wrote targeted cells  -> {os.path.join(BASE, OUTPUT_TARGETED)}")
print(f"Wrote heatmap         -> {os.path.join(BASE, OUTPUT_HEATMAP)}")
print(f"Wrote bars            -> {os.path.join(BASE, OUTPUT_PERSTATE)}")
