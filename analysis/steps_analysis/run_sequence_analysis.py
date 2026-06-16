# One-command launcher for the full sequence-analysis suite on a run group.
#
# Usage:
#   python run_sequence_analysis.py newoutcome7
#   python run_sequence_analysis.py            # defaults to newoutcome6
#
# It runs all analyses for the given tag, in order, and stops if any fails (so
# you never end up with a half-analysed run like before). Each tag's outputs
# land in its own isolated folders:
#   transition_asymmetry_<tag>/   net i->j matrices + heatmaps, sigma, the
#                                 circulation scalar, and classifier_accuracy.csv
#   markov_dependency_<tag>/      independence G-test + MI + Cramer's V,
#                                 triptych heatmaps, MI bar chart, perm nulls
#   transition_ttests_<tag>/      per-transition t-tests (mean/SD over rounds),
#                                 Cohen's d effect sizes + significance heatmap
#   agent_fingerprint_<tag>/      trained A-vs-B classifier (significant edges
#                                 only), CV accuracy + reusable model.json
#
# So for every new run you generate, this is the only command you need.
#
# To LABEL new sequences with a run's fingerprint model, call that script
# directly with a second argument, e.g.:
#   python agent_fingerprint_classifier.py newoutcome6 path/to/new_sequences/

import os
import subprocess
import sys

BASE = os.path.dirname(__file__)
TAG = sys.argv[1] if len(sys.argv) > 1 else "newoutcome6"

# order matters only for readability; they are independent
SCRIPTS = [
    "transition_asymmetry.py",   # net matrices, sigma, circulation
    "transition_classifier.py",  # learned Agent1-vs-Agent2 separability
    "markov_dependency_test.py",  # independence hypothesis test + MI
    "transition_ttests.py",      # per-transition t-tests + Cohen's d
    "agent_fingerprint_classifier.py",  # A-vs-B classifier on significant edges
]


def main():
    print(f"Running sequence analysis suite for tag = '{TAG}'\n")
    for i, script in enumerate(SCRIPTS, 1):
        print(f"########## [{i}/{len(SCRIPTS)}] {script} ##########")
        result = subprocess.run(
            [sys.executable, os.path.join(BASE, script), TAG]
        )
        if result.returncode != 0:
            sys.exit(f"\nFAILED on {script} (exit {result.returncode}) — stopping.")
        print()
    print(f"All {len(SCRIPTS)} analyses complete for '{TAG}'. Outputs in:")
    print(f"  {os.path.join(BASE, 'transition_asymmetry_' + TAG)}")
    print(f"  {os.path.join(BASE, 'markov_dependency_' + TAG)}")
    print(f"  {os.path.join(BASE, 'transition_ttests_' + TAG)}")
    print(f"  {os.path.join(BASE, 'agent_fingerprint_' + TAG)}")


if __name__ == "__main__":
    main()
