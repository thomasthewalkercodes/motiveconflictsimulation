# Per-transition hypothesis tests on the net flows shown in the heatmaps.
#
# The transition-asymmetry heatmap cell (i, j) is the POOLED net flow
# A[i,j] = count(i->j) - count(j->i). Here we stop pooling and instead treat
# every round file as one replicate observation of that net flow, so each
# transition gets a distribution over rounds we can do statistics on.
#
# Usage:  python transition_ttests.py newoutcome6     (default: newoutcome6)
#
# For each run group <tag>, each agent (a = Agent 1, b = Agent 2), and each of
# the 28 unordered motive pairs (i < j):
#
#   PRIMARY — is the transition directional?  Per round r take
#       fwd_r = C_r[i,j],  rev_r = C_r[j,i],  net_r = fwd_r - rev_r.
#     Across rounds report mean & SD of net, then a PAIRED t-test of fwd vs rev
#     (equivalently a one-sample t-test of net vs 0) with Cohen's d = mean/SD,
#     a 95% CI, and Benjamini-Hochberg FDR + Holm corrected p over the 28 pairs.
#     d (effect size), not p, is the meaningful number: with ~1000 rounds even a
#     micro asymmetry is "significant", so look at |d| and the CI.
#
#   SECONDARY — do the two agents differ on this transition?  A Welch two-sample
#     t-test of net_r(Agent 1) vs net_r(Agent 2) across rounds, with Hedges' g.
#
# Outputs (folder transition_ttests_<tag>/):
#   directional_ttests.csv     28 rows x 2 agents, the primary test
#   agent_comparison_ttests.csv 28 rows, Agent 1 vs Agent 2 per transition
#   effectsize_heatmap_<tag>.png signed Cohen's d per cell, * = FDR-significant
#
# CAVEAT: rounds in a dialogue run are coupled (each round's decay feeds back
# from the previous), so they are not perfectly independent; treat the p-values
# as slightly anti-conservative. Effect sizes are unaffected by this.

import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

TAG = sys.argv[1] if len(sys.argv) > 1 else "newoutcome6"
N = 8
AGENTS = {"a": "Agent 1", "b": "Agent 2"}
ALPHA = 0.05

BASE = os.path.dirname(__file__)
RUNS_DIR = os.path.join(BASE, "..", "..", "runs")
OUT = os.path.join(BASE, f"transition_ttests_{TAG}")
os.makedirs(OUT, exist_ok=True)

PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]  # 28 unordered pairs


# ── load per-round transition-count cubes ────────────────────────────────
def round_count_cube(run_dir, agent):
    """Array (n_rounds, N, N): one transition-count matrix per round file."""
    cubes = []
    for f in sorted(glob.glob(os.path.join(run_dir, f"simulation_{agent}_*.csv"))):
        s = pd.read_csv(f, usecols=["active_motive"])["active_motive"]
        s = s.dropna().astype(int).to_numpy() - 1  # 1..8 -> 0..7
        if s.size < 2:
            continue
        flat = np.bincount(s[:-1] * N + s[1:], minlength=N * N)
        cubes.append(flat.reshape(N, N))
    return np.array(cubes, dtype=float)  # (R, N, N)


def multipletests(pvals):
    """Return (holm, bh_fdr) adjusted p-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    # Holm
    holm = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        holm[idx] = min(running, 1.0)
    # Benjamini-Hochberg
    ranked = p[order] * n / (np.arange(1, n + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    fdr = np.empty(n)
    fdr[order] = np.clip(ranked, 0, 1)
    return holm, fdr


# ── run ──────────────────────────────────────────────────────────────────
def main():
    cdirs = [d for d in sorted(glob.glob(os.path.join(RUNS_DIR, f"{TAG}_*")))
             if os.path.isdir(d)]
    if not cdirs:
        raise SystemExit(f"No run dirs match {TAG}_*")
    if len(cdirs) > 1:
        print(f"NOTE: {len(cdirs)} conditions found; pooling their rounds together.")

    # gather per-round count + net cubes per agent across all conditions
    cubes_by_agent = {}  # agent -> (R, N, N) raw transition counts per round
    nets = {}            # agent -> (R, N, N) of (C_r - C_r.T)
    for ag in AGENTS:
        cubes = [round_count_cube(d, ag) for d in cdirs]
        C = np.concatenate(cubes, axis=0)
        cubes_by_agent[ag] = C
        nets[ag] = C - C.transpose(0, 2, 1)  # net_r[i,j] = fwd - rev
        print(f"{AGENTS[ag]} ({ag}): {C.shape[0]} rounds loaded")

    # PRIMARY: directional test per agent
    prim_rows = []
    dmat = {ag: np.zeros((N, N)) for ag in AGENTS}      # signed Cohen's d
    sigmat = {ag: np.zeros((N, N), bool) for ag in AGENTS}
    for ag in AGENTS:
        net = nets[ag]                      # (R, N, N)
        n = net.shape[0]
        tcrit = stats.t.ppf(0.975, n - 1)
        Ccube = cubes_by_agent[ag]
        pvals = []
        recs = []
        for (i, j) in PAIRS:
            fwd, rev = Ccube[:, i, j], Ccube[:, j, i]  # per-round fwd / rev counts
            x = net[:, i, j]                # per-round net i->j = fwd - rev
            mean, sd = x.mean(), x.std(ddof=1)
            d = mean / sd if sd > 0 else 0.0
            if sd > 0:
                t, p = stats.ttest_1samp(x, 0.0)
            else:
                t, p = (np.inf if mean != 0 else 0.0), (0.0 if mean != 0 else 1.0)
            ci = tcrit * sd / np.sqrt(n)
            recs.append(dict(agent=ag, i=i + 1, j=j + 1,
                             mean_fwd=float(fwd.mean()), mean_rev=float(rev.mean()),
                             mean_net=mean, sd_net=sd, n_rounds=n,
                             t=float(t), p_value=float(p), cohens_d=float(d),
                             ci95_lo=mean - ci, ci95_hi=mean + ci))
            pvals.append(p)
            dmat[ag][i, j], dmat[ag][j, i] = d, -d
        holm, fdr = multipletests(pvals)
        for k, (i, j) in enumerate(PAIRS):
            recs[k]["p_holm"] = float(holm[k])
            recs[k]["p_fdr"] = float(fdr[k])
            recs[k]["sig_fdr"] = bool(fdr[k] < ALPHA)
            sig = fdr[k] < ALPHA
            sigmat[ag][i, j] = sigmat[ag][j, i] = sig
        prim_rows += recs

    prim = pd.DataFrame(prim_rows)
    prim.to_csv(os.path.join(OUT, "directional_ttests.csv"), index=False)

    # SECONDARY: Agent 1 vs Agent 2 per transition (Welch + Hedges' g)
    comp_rows, comp_p = [], []
    na, nb = nets["a"].shape[0], nets["b"].shape[0]
    for (i, j) in PAIRS:
        xa, xb = nets["a"][:, i, j], nets["b"][:, i, j]
        t, p = stats.ttest_ind(xa, xb, equal_var=False)
        sa, sb = xa.std(ddof=1), xb.std(ddof=1)
        sp = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
        d = (xa.mean() - xb.mean()) / sp if sp > 0 else 0.0
        g = d * (1 - 3 / (4 * (na + nb) - 9))   # Hedges' bias correction
        comp_rows.append(dict(i=i + 1, j=j + 1,
                              mean_net_a=xa.mean(), mean_net_b=xb.mean(),
                              diff=xa.mean() - xb.mean(), t=float(t),
                              p_value=float(p), hedges_g=float(g)))
        comp_p.append(p)
    _, comp_fdr = multipletests(comp_p)
    comp = pd.DataFrame(comp_rows)
    comp["p_fdr"] = comp_fdr
    comp["sig_fdr"] = comp_fdr < ALPHA
    comp.to_csv(os.path.join(OUT, "agent_comparison_ttests.csv"), index=False)

    # FIGURE: signed Cohen's d heatmap per agent, * marks FDR-significant
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, ag in zip(axes, AGENTS):
        D = dmat[ag]
        vmax = np.abs(D).max() or 1
        im = ax.imshow(D, cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{AGENTS[ag]} ({ag}) — signed Cohen's d (i->j)\n"
                     f"* = FDR-significant (q<{ALPHA})", fontsize=10)
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(range(1, N + 1)); ax.set_yticklabels(range(1, N + 1))
        ax.set_xlabel("to motive j"); ax.set_ylabel("from motive i")
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                star = "*" if sigmat[ag][i, j] else ""
                ax.text(j, i, f"{D[i, j]:+.2f}{star}", ha="center", va="center",
                        fontsize=6.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Per-transition directional effect size — {TAG} "
                 f"(green = net i->j, red = net j->i)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, f"effectsize_heatmap_{TAG}.png"), dpi=130)
    plt.close(fig)

    # console summary: strongest directional transitions per agent
    print(f"\n=== strongest directional transitions (|Cohen's d|), {TAG} ===")
    for ag in AGENTS:
        sub = prim[prim.agent == ag].reindex(
            prim[prim.agent == ag].cohens_d.abs().sort_values(ascending=False).index)
        nsig = int(sub.sig_fdr.sum())
        print(f"\n{AGENTS[ag]} ({ag}) - {nsig}/{len(PAIRS)} FDR-significant; top 6:")
        print(sub.head(6)[["i", "j", "mean_net", "sd_net", "t",
                           "cohens_d", "p_fdr", "sig_fdr"]]
              .round({"mean_net": 1, "sd_net": 2, "t": 1,
                      "cohens_d": 3, "p_fdr": 4}).to_string(index=False))
    print("\n=== Agent 1 vs Agent 2 differences (top 6 by |Hedges g|) ===")
    cs = comp.reindex(comp.hedges_g.abs().sort_values(ascending=False).index)
    print(cs.head(6)[["i", "j", "mean_net_a", "mean_net_b", "diff",
                      "hedges_g", "p_fdr", "sig_fdr"]]
          .round({"mean_net_a": 1, "mean_net_b": 1, "diff": 1,
                  "hedges_g": 3, "p_fdr": 4}).to_string(index=False))
    print(f"\nWrote CSVs + heatmap to {OUT}")


if __name__ == "__main__":
    main()
