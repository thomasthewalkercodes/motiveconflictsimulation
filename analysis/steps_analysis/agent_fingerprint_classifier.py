# Fingerprint classifier: given a NEW behaviour sequence, how likely is it to be
# Agent 1 (a) vs Agent 2 (b)? -- using ONLY the "significant connections": the
# motive-pairs where the two agents reliably differ in travel direction.
#
# ───────────────────────────── plain-language idea ──────────────────────────
# Picture the 8 motives as 8 spots on a circle. Moving from one motive to the
# next is like walking between spots. On MOST connections the two agents walk
# the same way, so those edges tell us nothing about who is who. On a FEW edges
# they reliably differ: e.g. on edge 7-8 Agent 1 usually walks 7->8 while Agent 2
# usually walks 8->7. Those few edges are the agents' "tells" / fingerprints.
#
# To label a new sequence we IGNORE every same-behaviour edge and look only at
# the tell edges. On each tell edge we ask: which way did this sequence tend to
# walk, and does that match Agent 1's habit or Agent 2's? Each tell edge casts a
# vote; the vote is stronger when the agents' habits are more lopsided and when
# the sequence used that edge more often. We add the votes up (as log-odds) and
# squash the total into a probability: "this sequence is 78% Agent 1, 22% Agent
# 2." Few crossings of the tell edges -> the answer stays near 50/50 (honest
# uncertainty). This is just Naive Bayes / a log-likelihood-ratio test built on
# the discriminative edges.
# ─────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   python agent_fingerprint_classifier.py <tag> [new_sequences_path]
#     <tag>                 run group to learn the fingerprints from (def newoutcome6)
#     new_sequences_path    optional file or folder of CSVs (each with an
#                           'active_motive' column) to label with the model
#
# Outputs (folder agent_fingerprint_<tag>/):
#   model_edges.csv     the significant tell-edges and each agent's direction prob
#   model.json          the full trained model (reloadable / inspectable)
#   cv_report.csv       cross-validated accuracy + calibration of P(agent 1)
#   prob_separation_<tag>.png  out-of-fold P(agent 1) for true-A vs true-B rounds
#   predictions.csv     (only if new_sequences_path given) P(agent 1) per sequence
#
# How "significant" is decided: exactly as in transition_ttests.py — a Welch
# t-test of per-round net flow (Agent 1 vs Agent 2) for each of the 28 motive
# pairs, Benjamini-Hochberg FDR < 0.05. To keep the cross-validated accuracy
# honest, the edge set AND the probabilities are re-fit inside each training fold
# (no peeking at the held-out rounds).

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import expit

TAG = sys.argv[1] if len(sys.argv) > 1 else "newoutcome6"
NEW_PATH = sys.argv[2] if len(sys.argv) > 2 else None
N = 8
K_FOLDS = 5
ALPHA_FDR = 0.05     # significance threshold for a tell-edge
SMOOTH = 0.5         # Laplace pseudocount for direction probabilities
RNG = np.random.default_rng(0)

BASE = os.path.dirname(__file__)
RUNS_DIR = os.path.join(BASE, "..", "..", "runs")
OUT = os.path.join(BASE, f"agent_fingerprint_{TAG}")
os.makedirs(OUT, exist_ok=True)

PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]  # 28 edges, i<j


# ── data loading ─────────────────────────────────────────────────────────
def load_round_seqs(run_dir, agent):
    seqs = []
    for f in sorted(glob.glob(os.path.join(run_dir, f"simulation_{agent}_*.csv"))):
        s = pd.read_csv(f, usecols=["active_motive"])["active_motive"]
        s = s.dropna().astype(int).to_numpy() - 1
        if s.size >= 2:
            seqs.append(s)
    return seqs


def load_named_sequences(path):
    """List of (name, seq) from a CSV file or a folder of CSVs."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
    else:
        files = [path]
    out = []
    for f in files:
        try:
            s = pd.read_csv(f, usecols=["active_motive"])["active_motive"]
        except (ValueError, KeyError):
            continue
        s = s.dropna().astype(int).to_numpy() - 1
        if s.size >= 2:
            out.append((os.path.basename(f), s))
    return out


# ── per-round net flow on every edge (for significance testing) ──────────
def round_nets(seqs):
    """Array (n_rounds, 28): net flow (i->j minus j->i) per edge per round."""
    out = np.zeros((len(seqs), len(PAIRS)))
    for r, s in enumerate(seqs):
        C = np.zeros((N, N))
        np.add.at(C, (s[:-1], s[1:]), 1)
        for k, (i, j) in enumerate(PAIRS):
            out[r, k] = C[i, j] - C[j, i]
    return out


def agg_counts(seqs):
    C = np.zeros((N, N))
    for s in seqs:
        np.add.at(C, (s[:-1], s[1:]), 1)
    return C


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


# ── train: pick the tell-edges and learn each agent's direction habit ────
def fit_model(seqs_a, seqs_b):
    na, nb = round_nets(seqs_a), round_nets(seqs_b)
    pvals = []
    for k in range(len(PAIRS)):
        xa, xb = na[:, k], nb[:, k]
        if xa.std() == 0 and xb.std() == 0:
            pvals.append(1.0)
        else:
            pvals.append(stats.ttest_ind(xa, xb, equal_var=False).pvalue)
    fdr = bh_fdr(pvals)
    sig = [k for k in range(len(PAIRS)) if fdr[k] < ALPHA_FDR]

    Ca, Cb = agg_counts(seqs_a), agg_counts(seqs_b)
    edges = []
    for k in sig:
        i, j = PAIRS[k]
        pa = (Ca[i, j] + SMOOTH) / (Ca[i, j] + Ca[j, i] + 2 * SMOOTH)  # P(i->j|edge), A
        pb = (Cb[i, j] + SMOOTH) / (Cb[i, j] + Cb[j, i] + 2 * SMOOTH)  # P(i->j|edge), B
        edges.append(dict(i=i, j=j, p_a=pa, p_b=pb, fdr=float(fdr[k])))
    return {"edges": edges}


# ── score: turn a sequence into P(agent 1) ───────────────────────────────
def score_sequence(seq, model):
    """Return (P_agent1, log_likelihood_ratio, n_tell_crossings)."""
    src, dst = seq[:-1], seq[1:]
    llr = 0.0
    crossings = 0
    for e in model["edges"]:
        i, j, pa, pb = e["i"], e["j"], e["p_a"], e["p_b"]
        nij = int(np.sum((src == i) & (dst == j)))   # walked i->j
        nji = int(np.sum((src == j) & (dst == i)))   # walked j->i
        crossings += nij + nji
        # log P(data|A) - log P(data|B); binomial coefficient cancels in the ratio
        llr += nij * (np.log(pa) - np.log(pb))
        llr += nji * (np.log(1 - pa) - np.log(1 - pb))
    return float(expit(llr)), float(llr), crossings  # equal prior -> sigmoid(llr)


# ── honest cross-validation (edges + probs re-fit per fold) ──────────────
def cross_validate(seqs_a, seqs_b):
    data = [(s, 0) for s in seqs_a] + [(s, 1) for s in seqs_b]
    idx = RNG.permutation(len(data))
    folds = np.array_split(idx, K_FOLDS)
    oof_p, oof_y, nsig = [], [], []
    for f in range(K_FOLDS):
        test_ix = set(folds[f].tolist())
        tr_a = [data[i][0] for i in range(len(data)) if i not in test_ix and data[i][1] == 0]
        tr_b = [data[i][0] for i in range(len(data)) if i not in test_ix and data[i][1] == 1]
        model = fit_model(tr_a, tr_b)
        nsig.append(len(model["edges"]))
        for i in folds[f]:
            s, y = data[i]
            pA, _, _ = score_sequence(s, model)
            oof_p.append(pA)
            oof_y.append(y)
    oof_p, oof_y = np.array(oof_p), np.array(oof_y)
    pred = (oof_p < 0.5).astype(int)  # P(agent1)<0.5 -> predict agent 2 (label 1)
    acc = float(np.mean(pred == oof_y))
    return acc, oof_p, oof_y, float(np.mean(nsig))


# ── run ──────────────────────────────────────────────────────────────────
def main():
    cdirs = [d for d in sorted(glob.glob(os.path.join(RUNS_DIR, f"{TAG}_*")))
             if os.path.isdir(d)]
    if not cdirs:
        raise SystemExit(f"No run dirs match {TAG}_*")
    seqs_a, seqs_b = [], []
    for d in cdirs:
        seqs_a += load_round_seqs(d, "a")
        seqs_b += load_round_seqs(d, "b")
    print(f"Loaded {len(seqs_a)} Agent 1 rounds, {len(seqs_b)} Agent 2 rounds "
          f"from {TAG}")

    # 1) cross-validated performance
    acc, oof_p, oof_y, avg_nsig = cross_validate(seqs_a, seqs_b)
    meanA = oof_p[oof_y == 0].mean()
    meanB = oof_p[oof_y == 1].mean()
    print(f"\n5-fold CV accuracy = {acc:.3f}  (0.5 = can't tell, 1.0 = perfect)")
    print(f"  avg # tell-edges used per fold: {avg_nsig:.1f}")
    print(f"  mean P(agent1) on true Agent-1 rounds: {meanA:.3f}")
    print(f"  mean P(agent1) on true Agent-2 rounds: {meanB:.3f}")
    pd.DataFrame([{"tag": TAG, "cv_accuracy": round(acc, 4),
                   "avg_tell_edges": round(avg_nsig, 2),
                   "mean_Pa_on_trueA": round(float(meanA), 4),
                   "mean_Pa_on_trueB": round(float(meanB), 4),
                   "n_rounds_each": len(seqs_a)}]
                 ).to_csv(os.path.join(OUT, "cv_report.csv"), index=False)

    # separation histogram (out-of-fold, so honest)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 31)
    ax.hist(oof_p[oof_y == 0], bins=bins, alpha=0.6, label="true Agent 1",
            color="#1b7837")
    ax.hist(oof_p[oof_y == 1], bins=bins, alpha=0.6, label="true Agent 2",
            color="#762a83")
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("model's P(sequence is Agent 1)")
    ax.set_ylabel("number of held-out rounds")
    ax.set_title(f"Fingerprint separation — {TAG}\n"
                 f"CV acc={acc:.3f}, {avg_nsig:.0f} tell-edges; "
                 f"overlap = agents hard to tell apart")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"prob_separation_{TAG}.png"), dpi=130)
    plt.close(fig)

    # 2) final model on ALL rounds (for labelling genuinely new sequences)
    model = fit_model(seqs_a, seqs_b)
    rows = []
    print(f"\n=== tell-edges (significant connections), final model, {TAG} ===")
    if not model["edges"]:
        print("  none — the two agents are statistically indistinguishable here.")
    for e in model["edges"]:
        i1, j1 = e["i"] + 1, e["j"] + 1
        favored = "Agent 1" if e["p_a"] > e["p_b"] else "Agent 2"
        print(f"  edge {i1}-{j1}: P({i1}->{j1}) = {e['p_a']:.2f} for Agent 1 vs "
              f"{e['p_b']:.2f} for Agent 2  ->  '{i1}->{j1}' favours {favored}")
        rows.append(dict(i=i1, j=j1, p_a_fwd=round(e["p_a"], 4),
                         p_b_fwd=round(e["p_b"], 4),
                         separation=round(e["p_a"] - e["p_b"], 4),
                         fwd_favours=favored, fdr=round(e["fdr"], 6)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "model_edges.csv"), index=False)
    with open(os.path.join(OUT, "model.json"), "w") as fh:
        json.dump({"tag": TAG, "n_rounds_each": len(seqs_a),
                   "edges": [{"i": e["i"] + 1, "j": e["j"] + 1,
                              "p_a_fwd": e["p_a"], "p_b_fwd": e["p_b"]}
                             for e in model["edges"]]}, fh, indent=2)

    # 3) classify any new sequences the user points us at
    if NEW_PATH:
        new = load_named_sequences(NEW_PATH)
        if not new:
            print(f"\nNo usable CSVs (need an 'active_motive' column) in {NEW_PATH}")
        else:
            preds = []
            for name, s in new:
                pA, llr, cr = score_sequence(s, model)
                preds.append(dict(sequence=name, P_agent1=round(pA, 4),
                                  P_agent2=round(1 - pA, 4), log_LR=round(llr, 3),
                                  tell_crossings=cr,
                                  label="Agent 1" if pA >= 0.5 else "Agent 2"))
            pdf = pd.DataFrame(preds)
            pdf.to_csv(os.path.join(OUT, "predictions.csv"), index=False)
            print(f"\n=== predictions for {len(new)} new sequence(s) ===")
            print(pdf.to_string(index=False))
            share = (pdf.P_agent1 >= 0.5).mean()
            print(f"\noverall: {share*100:.0f}% lean Agent 1, "
                  f"{(1-share)*100:.0f}% lean Agent 2")

    print(f"\nWrote model + report to {OUT}")


if __name__ == "__main__":
    main()
