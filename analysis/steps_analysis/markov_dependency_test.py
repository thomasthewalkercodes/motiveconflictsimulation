# Independence vs. first-order dependency in the behaviour sequences.
#
# This generalises the minimal "occurrence rates -> independent matrix ->
# empirical matrix -> MAD/KL/MI/rank" script to the full newoutcome2 design:
# 8 conditions x 2 agents (a = Agent 1, b = Agent 2), 150 rounds each, with
# transitions counted *within* each round only (never across file boundaries).
#
# It answers one question with statistics, not just descriptive scalars:
#   "Does each agent's motive sequence carry first-order (Markov) memory, or is
#    it indistinguishable from independent draws from its own occurrence rates?"
#
# NULL MODEL  H0 (independence): P(next = j | current = i) = p_j  for all i,
# i.e. every row of the transition matrix equals the marginal p. Then the
# expected transition counts are E[i, j] = R_i * p_j (R_i = times i was a
# 'from' state).
#
# WHAT IS REPORTED, per (condition, agent) and pooled ALL:
#   descriptive (from the prompt): MAD, mean row-KL, mutual information (MI),
#       numeric matrix rank, 2nd-singular-value ratio.
#   inferential (the additions):
#     (1) G-test (likelihood ratio) of independence: G = 2 * sum C ln(C/E),
#         which equals 2 * N * MI exactly. Under H0, G ~ chi2 with df=(n-1)^2.
#         Reported with p-value and Cramer's V effect size. NOTE: with ~150k
#         transitions any real structure is "significant", so the EFFECT SIZE
#         (MI / V), not p, is the meaningful comparison across cells.
#     (2) Permutation test: shuffle each round's order (destroying sequence
#         memory, preserving its marginal), recompute pooled MI many times to
#         build the null band, and locate the observed MI in it (empirical p,
#         z-score). This needs no asymptotic assumptions and is what the
#         permutation-null figure visualises.
#   plus round-level bootstrap 95% CIs on MI, so Agent 1 vs Agent 2 can be
#   compared with uncertainty.
#
# VISUALS: per-condition heatmap triptychs (empirical / independent /
# difference) for both agents; an MI-by-condition bar chart with bootstrap CIs;
# and a grid of permutation-null histograms with the observed MI marked.

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import chi2

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

TAG = sys.argv[1] if len(sys.argv) > 1 else "newoutcome6"
N = 8
AGENTS = ["a", "b"]
B_PERM = 300  # permutation-null replicates
B_BOOT = 300  # bootstrap replicates for MI CIs
EPS = 1e-12
RNG = np.random.default_rng(0)

BASE = os.path.dirname(__file__)
RUNS_DIR = os.path.join(BASE, "..", "..", "runs")
_SUFFIX = f"_{TAG}"  # each run group -> its own tag-named folder
OUT = os.path.join(BASE, "markov_dependency" + _SUFFIX)
os.makedirs(OUT, exist_ok=True)


# ── data loading ─────────────────────────────────────────────────────────
def short_cond(d):
    m = re.match(r"(.+?_\d+)_\d{4}-\d{2}-\d{2}", os.path.basename(d))
    return m.group(1) if m else os.path.basename(d)


def load_rounds(run_dir, agent):
    seqs = []
    for f in sorted(glob.glob(os.path.join(run_dir, f"simulation_{agent}_*.csv"))):
        s = pd.read_csv(f, usecols=["active_motive"])["active_motive"]
        s = s.dropna().astype(int).to_numpy() - 1  # 1..8 -> 0..7
        if s.size >= 2:
            seqs.append(s)
    return seqs


def counts_from_rounds(seqs):
    """Pooled 8x8 transition counts, transitions taken within each round."""
    flat = np.zeros(N * N, dtype=np.int64)
    for s in seqs:
        flat += np.bincount(s[:-1] * N + s[1:], minlength=N * N)
    return flat.reshape(N, N).astype(float)


# ── the measures ─────────────────────────────────────────────────────────
def mutual_information(C):
    """MI in nats between current and next state from a count matrix."""
    tot = C.sum()
    if tot == 0:
        return 0.0
    Pxy = C / tot
    Px = Pxy.sum(1, keepdims=True)
    Py = Pxy.sum(0, keepdims=True)
    mask = Pxy > 0
    return float(np.sum(Pxy[mask] * np.log(Pxy[mask] / (Px @ Py)[mask] + EPS)))


def describe(C):
    """Descriptive + inferential scalars for one count matrix."""
    tot = C.sum()
    p = C.sum(0) / tot  # marginal next-state distribution
    R = C.sum(1, keepdims=True)  # 'from' totals
    T_emp = np.nan_to_num(C / np.where(R == 0, 1, R))
    T_ind = np.tile(p, (N, 1))

    mad = float(np.abs(T_emp - T_ind).mean())
    mean_row_kl = float(np.sum(T_emp * np.log((T_emp + EPS) / (T_ind + EPS))) / N)
    mi = mutual_information(C)  # nats
    G = 2.0 * tot * mi  # = 2 sum C ln(C/E), E=R*p
    df = (N - 1) ** 2
    p_val = float(chi2.sf(G, df)) if HAVE_SCIPY else np.nan
    cramers_v = float(np.sqrt(2 * mi / (N - 1)))  # sqrt(G / (N (k-1)))
    rank = int(np.linalg.matrix_rank(T_emp))
    sv = np.linalg.svd(T_emp, compute_uv=False)
    sv2_ratio = float(sv[1] / sv[0]) if sv[0] > 0 else 0.0
    return dict(
        n_transitions=int(tot),
        MAD=mad,
        mean_row_KL=mean_row_kl,
        MI_nats=mi,
        MI_bits=mi / np.log(2),
        G_stat=G,
        df=df,
        p_value=p_val,
        cramers_v=cramers_v,
        matrix_rank=rank,
        sv2_ratio=sv2_ratio,
        T_emp=T_emp,
        T_ind=T_ind,
    )


def permutation_null(seqs, observed_mi):
    """Shuffle each round in place, recompute pooled MI; B_PERM times."""
    arrs = [s.copy() for s in seqs]
    null = np.empty(B_PERM)
    for b in range(B_PERM):
        flat = np.zeros(N * N, dtype=np.int64)
        for s in arrs:
            RNG.shuffle(s)
            flat += np.bincount(s[:-1] * N + s[1:], minlength=N * N)
        null[b] = mutual_information(flat.reshape(N, N).astype(float))
    p_emp = float((np.sum(null >= observed_mi) + 1) / (B_PERM + 1))
    z = float((observed_mi - null.mean()) / (null.std() + EPS))
    return null, p_emp, z


def bootstrap_mi_ci(seqs):
    """Round-level bootstrap 95% CI on pooled MI."""
    n = len(seqs)
    boots = np.empty(B_BOOT)
    for b in range(B_BOOT):
        pick = RNG.integers(0, n, n)
        boots[b] = mutual_information(counts_from_rounds([seqs[i] for i in pick]))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ── figures ──────────────────────────────────────────────────────────────
def triptych(label, res_a, res_b):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, (ag, res) in enumerate([("Agent 1 (a)", res_a), ("Agent 2 (b)", res_b)]):
        diff = res["T_emp"] - res["T_ind"]
        dmax = np.abs(diff).max() or 1
        for col, (M, ttl, cmap, vlim) in enumerate(
            [
                (
                    res["T_emp"],
                    "empirical  P(j|i)",
                    "viridis",
                    (0, max(0.001, res["T_emp"].max())),
                ),
                (
                    res["T_ind"],
                    "independent  rows=p",
                    "viridis",
                    (0, max(0.001, res["T_emp"].max())),
                ),
                (diff, "empirical - independent", "RdBu_r", (-dmax, dmax)),
            ]
        ):
            ax = axes[row, col]
            im = ax.imshow(M, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
            ax.set_title(f"{ag}: {ttl}", fontsize=9)
            ax.set_xticks(range(N))
            ax.set_yticks(range(N))
            ax.set_xticklabels(range(1, N + 1), fontsize=7)
            ax.set_yticklabels(range(1, N + 1), fontsize=7)
            ax.set_xlabel("next j", fontsize=8)
            ax.set_ylabel("current i", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Transition structure — {label}   "
        f"(MI_a={res_a['MI_bits']:.4f} bit, MI_b={res_b['MI_bits']:.4f} bit)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, f"heatmaps_{label}.png"), dpi=120)
    plt.close(fig)


def mi_bar(rows, ci):
    conds = [r["condition"] for r in rows if r["agent"] == "a"]
    xa = np.arange(len(conds))
    ma = [r["MI_bits"] for r in rows if r["agent"] == "a"]
    mb = [r["MI_bits"] for r in rows if r["agent"] == "b"]
    ln2 = np.log(2)

    def err(vals, ag):  # CIs are stored in nats -> convert to bits, clip >=0
        lo = np.array([ci[(conds[i], ag)][0] / ln2 for i in range(len(conds))])
        hi = np.array([ci[(conds[i], ag)][1] / ln2 for i in range(len(conds))])
        v = np.array(vals)
        return np.clip(np.vstack([v - lo, hi - v]), 0, None)

    ea, eb = err(ma, "a"), err(mb, "b")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(xa - 0.2, ma, 0.4, yerr=ea, label="Agent 1 (a)", capsize=3)
    ax.bar(xa + 0.2, mb, 0.4, yerr=eb, label="Agent 2 (b)", capsize=3)
    ax.set_xticks(xa)
    ax.set_xticklabels([c.replace(TAG + "_", "") for c in conds], rotation=0)
    ax.set_ylabel("Mutual information (bits)")
    ax.set_title("Sequential dependency per condition (bootstrap 95% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "MI_by_condition.png"), dpi=130)
    plt.close(fig)


def perm_grid(nulls, observed, conds):
    fig, axes = plt.subplots(
        len(conds), 2, figsize=(11, 2.1 * len(conds)), squeeze=False
    )
    for r, cond in enumerate(conds):
        for c, ag in enumerate(AGENTS):
            ax = axes[r, c]
            nl = np.array(nulls[(cond, ag)]) / np.log(2)
            obs = observed[(cond, ag)] / np.log(2)
            ax.hist(nl, bins=30, color="lightgray", edgecolor="gray")
            ax.axvline(obs, color="crimson", lw=2)
            ax.set_title(
                f"{cond.replace(TAG+'_','')}  {ag}: obs={obs:.4f} bit "
                f"(null max {nl.max():.4f})",
                fontsize=8,
            )
            ax.set_yticks([])
    fig.suptitle(
        "Permutation null of MI (gray) vs observed (red) — "
        "observed far right = real sequential structure"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(os.path.join(OUT, "permutation_nulls.png"), dpi=120)
    plt.close(fig)


# ── run ──────────────────────────────────────────────────────────────────
def main():
    cdirs = [
        d
        for d in sorted(glob.glob(os.path.join(RUNS_DIR, f"{TAG}_*")))
        if os.path.isdir(d)
    ]
    rounds = {}
    rows, ci, nulls, observed = [], {}, {}, {}
    pooled = {ag: [] for ag in AGENTS}

    for d in cdirs:
        cond = short_cond(d)
        res_cell = {}
        for ag in AGENTS:
            seqs = load_rounds(d, ag)
            rounds[(cond, ag)] = seqs
            pooled[ag] += seqs
            C = counts_from_rounds(seqs)
            res = describe(C)
            res_cell[ag] = res
            observed[(cond, ag)] = res["MI_nats"]
            nl, p_emp, z = permutation_null(seqs, res["MI_nats"])
            nulls[(cond, ag)] = nl
            ci[(cond, ag)] = bootstrap_mi_ci(seqs)
            rows.append(
                {
                    "condition": cond,
                    "agent": ag,
                    **{
                        k: res[k]
                        for k in [
                            "n_transitions",
                            "MAD",
                            "mean_row_KL",
                            "MI_nats",
                            "MI_bits",
                            "G_stat",
                            "df",
                            "p_value",
                            "cramers_v",
                            "matrix_rank",
                            "sv2_ratio",
                        ]
                    },
                    "perm_p": p_emp,
                    "perm_z": round(z, 2),
                    "MI_ci_lo_bits": ci[(cond, ag)][0] / np.log(2),
                    "MI_ci_hi_bits": ci[(cond, ag)][1] / np.log(2),
                }
            )
        triptych(cond, res_cell["a"], res_cell["b"])
        print(
            f"[done] {cond}: "
            f"MI_a={res_cell['a']['MI_bits']:.4f}b (G={res_cell['a']['G_stat']:.0f}) | "
            f"MI_b={res_cell['b']['MI_bits']:.4f}b (G={res_cell['b']['G_stat']:.0f})"
        )

    # pooled ALL
    res_all = {}
    for ag in AGENTS:
        C = counts_from_rounds(pooled[ag])
        res = describe(C)
        res_all[ag] = res
        ci[("ALL", ag)] = bootstrap_mi_ci(pooled[ag])
        rows.append(
            {
                "condition": "ALL",
                "agent": ag,
                **{
                    k: res[k]
                    for k in [
                        "n_transitions",
                        "MAD",
                        "mean_row_KL",
                        "MI_nats",
                        "MI_bits",
                        "G_stat",
                        "df",
                        "p_value",
                        "cramers_v",
                        "matrix_rank",
                        "sv2_ratio",
                    ]
                },
                "perm_p": np.nan,
                "perm_z": np.nan,
                "MI_ci_lo_bits": ci[("ALL", ag)][0] / np.log(2),
                "MI_ci_hi_bits": ci[("ALL", ag)][1] / np.log(2),
            }
        )
    triptych("ALL", res_all["a"], res_all["b"])

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "dependency_stats.csv"), index=False)

    conds = [short_cond(d) for d in cdirs]
    mi_bar(
        [r for r in rows if r["condition"] != "ALL"],
        {k: v for k, v in ci.items() if k[0] != "ALL"},
    )
    perm_grid(nulls, observed, conds)

    # console summary
    show = df[
        [
            "condition",
            "agent",
            "MI_bits",
            "cramers_v",
            "G_stat",
            "p_value",
            "perm_p",
            "perm_z",
            "matrix_rank",
        ]
    ].copy()
    for c in ["MI_bits", "cramers_v"]:
        show[c] = show[c].round(4)
    show["G_stat"] = show["G_stat"].round(0)
    show["p_value"] = show["p_value"].apply(
        lambda v: (
            "<1e-300"
            if (pd.notna(v) and v == 0)
            else (f"{v:.1e}" if pd.notna(v) else "n/a")
        )
    )
    print("\n=== independence test per condition/agent ===")
    print(show.to_string(index=False))
    print(f"\nWrote CSV + figures to {OUT}")
    if not HAVE_SCIPY:
        print(
            "(scipy not found: G-test p-values shown as n/a; permutation p still valid)"
        )


if __name__ == "__main__":
    main()
