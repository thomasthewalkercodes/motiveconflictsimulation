# NEW IDEA, made concrete: *learn* to tell Agent 1 (a) from Agent 2 (b) from a
# single behaviour sequence, using the transition structure the previous script
# measured.
#
# Method (per condition, and pooled "ALL"):
#   - Each round file is one labelled sequence (label = which agent emitted it).
#   - 5-fold cross-validation. On the training rounds we fit one first-order
#     Markov transition matrix per agent, P_a and P_b (Laplace-smoothed, rows
#     normalised). For each held-out round we score its log-likelihood under
#     both models and predict the agent whose model fits better (a log-
#     likelihood-ratio test). Accuracy on held-out rounds = how separable the
#     two agents' *dynamics* are. 0.5 = indistinguishable, 1.0 = perfectly told
#     apart from a single round.
#   - We also report a one-number baseline: classify each round by the sign of
#     its circumplex-circulation scalar (the Helmholtz-Hodge ring-flow feature),
#     thresholded at the training-set midpoint. This shows how much of the
#     separability is carried by that single interpretable feature alone.

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

TAG = sys.argv[1] if len(sys.argv) > 1 else "newoutcome6"
N = 8
K_FOLDS = 5
ALPHA = 0.5  # Laplace smoothing for the Markov models
RNG = np.random.default_rng(0)

BASE = os.path.dirname(__file__)
RUNS_DIR = os.path.join(BASE, "..", "..", "runs")
_SUFFIX = f"_{TAG}"  # each run group -> its own tag-named folder
OUT = os.path.join(BASE, "transition_asymmetry" + _SUFFIX)
os.makedirs(OUT, exist_ok=True)


def short_cond(d):
    m = re.match(r"(.+?_\d+)_\d{4}-\d{2}-\d{2}", os.path.basename(d))
    return m.group(1) if m else os.path.basename(d)


def load_round_seqs(run_dir, agent):
    """List of 0-indexed motive arrays, one per round file for this agent."""
    seqs = []
    for f in sorted(glob.glob(os.path.join(run_dir, f"simulation_{agent}_*.csv"))):
        s = pd.read_csv(f, usecols=["active_motive"])["active_motive"]
        s = s.dropna().astype(int).to_numpy() - 1
        if s.size >= 2:
            seqs.append(s)
    return seqs


def counts(seqs):
    C = np.zeros((N, N))
    for s in seqs:
        np.add.at(C, (s[:-1], s[1:]), 1)
    return C


def markov_logP(C):
    """Row-normalised log transition matrix with Laplace smoothing."""
    M = C + ALPHA
    return np.log(M / M.sum(axis=1, keepdims=True))


def seq_loglik(s, logP):
    return logP[s[:-1], s[1:]].sum()


def circulation_scalar(s):
    """Signed net adjacent-ring flow for one sequence (the Hodge curl feature)."""
    C = counts([s])
    A = C - C.T
    return sum(A[i, (i + 1) % N] for i in range(N))


def cv_eval(seqs_a, seqs_b):
    """5-fold CV. Returns (markov_acc, circulation_acc)."""
    data = [(s, 0) for s in seqs_a] + [(s, 1) for s in seqs_b]
    idx = RNG.permutation(len(data))
    folds = np.array_split(idx, K_FOLDS)
    mk_correct = circ_correct = total = 0
    for f in range(K_FOLDS):
        test_ix = set(folds[f].tolist())
        train = [data[i] for i in range(len(data)) if i not in test_ix]
        test = [data[i] for i in folds[f]]

        tr_a = [s for s, y in train if y == 0]
        tr_b = [s for s, y in train if y == 1]
        logP_a, logP_b = markov_logP(counts(tr_a)), markov_logP(counts(tr_b))

        # circulation threshold = midpoint of class means on the training set
        ca = np.mean([circulation_scalar(s) for s in tr_a])
        cb = np.mean([circulation_scalar(s) for s in tr_b])
        thr, a_side = (ca + cb) / 2, np.sign(ca - cb)

        for s, y in test:
            pred_mk = 0 if seq_loglik(s, logP_a) >= seq_loglik(s, logP_b) else 1
            mk_correct += pred_mk == y
            c = circulation_scalar(s)
            pred_c = 0 if np.sign(c - thr) == a_side else 1
            circ_correct += pred_c == y
            total += 1
    return mk_correct / total, circ_correct / total


def main():
    cdirs = [
        d
        for d in sorted(glob.glob(os.path.join(RUNS_DIR, f"{TAG}_*")))
        if os.path.isdir(d)
    ]
    all_a, all_b, rows = [], [], []
    for d in cdirs:
        cond = short_cond(d)
        sa, sb = load_round_seqs(d, "a"), load_round_seqs(d, "b")
        all_a += sa
        all_b += sb
        mk, cc = cv_eval(sa, sb)
        rows.append(
            {
                "condition": cond,
                "n_rounds_each": len(sa),
                "markov_LR_accuracy": round(mk, 4),
                "circulation_only_accuracy": round(cc, 4),
            }
        )
        print(f"{cond}: Markov LR acc={mk:.3f}  |  circulation-only acc={cc:.3f}")

    mk, cc = cv_eval(all_a, all_b)
    rows.append(
        {
            "condition": "ALL",
            "n_rounds_each": len(all_a),
            "markov_LR_accuracy": round(mk, 4),
            "circulation_only_accuracy": round(cc, 4),
        }
    )
    print(f"ALL: Markov LR acc={mk:.3f}  |  circulation-only acc={cc:.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "classifier_accuracy.csv"), index=False)
    print(f"\nWrote {os.path.join(OUT, 'classifier_accuracy.csv')}")
    print("(0.5 = indistinguishable from one round; 1.0 = perfectly separable)")


if __name__ == "__main__":
    main()
