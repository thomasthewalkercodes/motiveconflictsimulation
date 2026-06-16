# Transition-asymmetry analysis comparing Agent 1 (a) vs Agent 2 (b).
#
# For the run group TAG (default "newoutcome2") this walks every condition
# folder (newoutcome2_00000 .. _00007), and for each agent builds an 8x8
# transition-count matrix C where C[i, j] = number of times motive i is
# immediately followed by motive j (counted *within* each round file, motives
# relabelled 1..8 -> 0..7). Transitions are never counted across file
# boundaries.
#
# Three things are produced, per condition and pooled over all conditions
# ("ALL"), for agent a and agent b:
#
#   1. Net transition matrix  A = C - C.T   (i->j minus j->i).
#      Saved as CSV and drawn as a red/green heatmap (green > 0, red < 0).
#
#   2. Directional irreversibility  sigma_i = sum_j F_ij * ln(F_ij / F_ji)
#      with F = C / C.sum() (joint transition probabilities). Reported per
#      motive and summed to a total. This is the KL divergence between the
#      forward flow and its time-reverse; it is >= 0 and 0 only if the flow is
#      perfectly time-reversible. Pairs where one direction never occurs are
#      skipped (logged) for the primary value; a Laplace-smoothed value is also
#      reported for robustness.
#
#   3. NEW IDEA (see module docstring at bottom / README note): a single signed
#      "circumplex circulation" scalar from a discrete Helmholtz-Hodge split of
#      the net flow, capturing net clockwise vs counter-clockwise drift around
#      the 8-motive circle. This is the most compact learnable feature that
#      separates the two complementary agents.

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────
TAG = sys.argv[1] if len(sys.argv) > 1 else "newoutcome6"  # run-group prefix
N = 8
AGENTS = ["a", "b"]  # a = Agent 1, b = Agent 2
LAPLACE_ALPHA = 0.5  # pseudocount for the smoothed sigma only

BASE = os.path.dirname(__file__)
RUNS_DIR = os.path.join(BASE, "..", "..", "runs")
# each run group writes to its own tag-named folder (isolated, no overwrites)
_SUFFIX = f"_{TAG}"
OUT_DIR = os.path.join(BASE, "transition_asymmetry" + _SUFFIX)
os.makedirs(OUT_DIR, exist_ok=True)


# ── Step 1: count transitions ────────────────────────────────────────────
def transition_counts(sim_files):
    """8x8 count matrix summed over a list of round CSVs (motives 1..8)."""
    C = np.zeros((N, N), dtype=np.int64)
    for f in sim_files:
        seq = pd.read_csv(f, usecols=["active_motive"])["active_motive"]
        seq = seq.dropna().astype(int).to_numpy() - 1  # -> 0..7
        if seq.size < 2:
            continue
        src, dst = seq[:-1], seq[1:]
        np.add.at(C, (src, dst), 1)  # vectorised pair-count
    return C


def condition_dirs(tag):
    dirs = sorted(glob.glob(os.path.join(RUNS_DIR, f"{tag}_*")))
    return [d for d in dirs if os.path.isdir(d)]


def short_cond(run_dir):
    """'newoutcome2_00003_2026-..' -> 'newoutcome2_00003'."""
    m = re.match(r"(.+?_\d+)_\d{4}-\d{2}-\d{2}", os.path.basename(run_dir))
    return m.group(1) if m else os.path.basename(run_dir)


# ── Step 2: the two requested metrics ────────────────────────────────────
def net_matrix(C):
    """A[i, j] = C[i, j] - C[j, i]  (antisymmetric net flow i->j)."""
    return C - C.T


def sigma_per_motive(C, alpha=0.0):
    """sigma_i = sum_j F_ij ln(F_ij / F_ji), F = (C+alpha)/sum.

    Returns (sigma_vec[N], total, n_oneway_pairs). With alpha=0 the undefined
    one-way pairs (F_ij>0, F_ji==0) are skipped and counted; with alpha>0 every
    pair is defined (Laplace smoothing)."""
    M = C.astype(float) + alpha
    total = M.sum()
    if total == 0:
        return np.zeros(N), 0.0, 0
    F = M / total
    sigma = np.zeros(N)
    oneway = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            fij, fji = F[i, j], F[j, i]
            if fij <= 0:
                continue
            if fji <= 0:  # only forward seen -> undefined log, skip & flag
                oneway += 1
                continue
            sigma[i] += fij * np.log(fij / fji)
    return sigma, float(sigma.sum()), oneway


def circumplex_circulation(C):
    """NEW IDEA — signed net circulation around the 8-cycle of adjacent octants.

    The net flow A = C - C.T lives on the directed graph of motives. Restricted
    to neighbouring octants (the circumplex ring 0-1-2-..-7-0), the summed
    forward flow is a single signed scalar: positive = net counter-clockwise
    drift around the circle, negative = net clockwise. Because the two agents
    are coupled by the warmth-axis-mirrored decay, their rings tend to circulate
    oppositely, so this one number is a strong, interpretable discriminator.
    Normalised by total adjacent traffic to a [-1, 1] 'net circulation rate'."""
    A = net_matrix(C)
    fwd = sum(A[i, (i + 1) % N] for i in range(N))  # i -> i+1 (counter-cw)
    traffic = sum(C[i, (i + 1) % N] + C[(i + 1) % N, i] for i in range(N))
    rate = fwd / traffic if traffic else 0.0
    return int(fwd), float(rate)


# ── Step 3: red/green heatmap of the net matrix ──────────────────────────
def draw_heatmap(ax, A, title):
    vmax = np.abs(A).max() or 1
    im = ax.imshow(A, cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels([f"{k+1}" for k in range(N)], fontsize=8)
    ax.set_yticklabels([f"{k+1}" for k in range(N)], fontsize=8)
    ax.set_xlabel("to motive j", fontsize=8)
    ax.set_ylabel("from motive i", fontsize=8)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            ax.text(
                j,
                i,
                f"{A[i, j]:+d}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="black",
            )
    return im


# ── Step 4: run everything ───────────────────────────────────────────────
def main():
    cdirs = condition_dirs(TAG)
    if not cdirs:
        raise SystemExit(f"No run dirs match {TAG}_* in {RUNS_DIR}")

    counts = {}  # (cond, agent) -> C
    pooled = {ag: np.zeros((N, N), dtype=np.int64) for ag in AGENTS}

    for d in cdirs:
        cond = short_cond(d)
        for ag in AGENTS:
            files = sorted(glob.glob(os.path.join(d, f"simulation_{ag}_*.csv")))
            C = transition_counts(files)
            counts[(cond, ag)] = C
            pooled[ag] += C
        print(
            f"[counted] {cond}: a={counts[(cond,'a')].sum()} "
            f"b={counts[(cond,'b')].sum()} transitions"
        )

    conds = [short_cond(d) for d in cdirs]

    # --- sigma + circulation tables ---
    sig_rows, circ_rows = [], []
    for key_label, getter in (
        *[(c, lambda c=c: {ag: counts[(c, ag)] for ag in AGENTS}) for c in conds],
        ("ALL", lambda: pooled),
    ):
        mats = getter()
        for ag in AGENTS:
            C = mats[ag]
            sig, tot, oneway = sigma_per_motive(C, alpha=0.0)
            _, tot_s, _ = sigma_per_motive(C, alpha=LAPLACE_ALPHA)
            fwd, rate = circumplex_circulation(C)
            sig_rows.append(
                {
                    "condition": key_label,
                    "agent": ag,
                    **{f"sigma_m{k+1}": round(sig[k], 5) for k in range(N)},
                    "sigma_total": round(tot, 5),
                    "sigma_total_smoothed": round(tot_s, 5),
                    "n_oneway_pairs": oneway,
                    "n_transitions": int(C.sum()),
                }
            )
            circ_rows.append(
                {
                    "condition": key_label,
                    "agent": ag,
                    "circulation_net": fwd,
                    "circulation_rate": round(rate, 5),
                }
            )

    pd.DataFrame(sig_rows).to_csv(
        os.path.join(OUT_DIR, "sigma_per_motive.csv"), index=False
    )
    pd.DataFrame(circ_rows).to_csv(
        os.path.join(OUT_DIR, "circulation.csv"), index=False
    )

    # --- net matrices: CSVs + heatmaps ---
    labels = [f"m{k+1}" for k in range(N)]
    panels = [(c, {ag: counts[(c, ag)] for ag in AGENTS}) for c in conds]
    panels.append(("ALL", pooled))

    for label, mats in panels:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        Aa, Ab = net_matrix(mats["a"]), net_matrix(mats["b"])
        pd.DataFrame(Aa, index=labels, columns=labels).to_csv(
            os.path.join(OUT_DIR, f"asym_{label}_a.csv")
        )
        pd.DataFrame(Ab, index=labels, columns=labels).to_csv(
            os.path.join(OUT_DIR, f"asym_{label}_b.csv")
        )
        im0 = draw_heatmap(axes[0], Aa, f"{label}  Agent 1 (a)\nnet i->j")
        im1 = draw_heatmap(axes[1], Ab, f"{label}  Agent 2 (b)\nnet i->j")
        draw_heatmap(axes[2], Aa - Ab, f"{label}  Agent1 - Agent2\n(difference)")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        fig.suptitle(
            f"Transition asymmetry (green = net i->j > 0, red < 0)  —  {label}"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(OUT_DIR, f"heatmap_{label}.png"), dpi=130)
        plt.close(fig)

    # --- console summary ---
    print("\n=== sigma_total (directional irreversibility) ===")
    sdf = pd.DataFrame(sig_rows)
    pv = sdf.pivot(index="condition", columns="agent", values="sigma_total")
    print(pv.round(4).to_string())
    print("\n=== circumplex circulation rate (NEW IDEA; + = counter-cw) ===")
    cdf = pd.DataFrame(circ_rows)
    print(
        cdf.pivot(index="condition", columns="agent", values="circulation_rate")
        .round(4)
        .to_string()
    )
    print(f"\nWrote CSVs + heatmaps to {OUT_DIR}")


if __name__ == "__main__":
    main()
