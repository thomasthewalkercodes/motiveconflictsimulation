# Angular-change analysis for the behaviour sequences in a run.
#
# For each run listed in RUNS_TO_ANALYZE this script walks every
# simulation_*.csv inside the run folder, reads the ordered "active_motive"
# column (the behaviours that were shown, motives 1..8 = octants 0..7), and
# measures how far each behaviour jumped from the previous one on the
# circumplex.
#
# Angular distance is the shortest way around the 8-motive circle, exactly
# like the circular distance used in algorithm/influence.py
# (distance = min(|i-j|, n-|i-j|)), but expressed in degrees: one octant step
# = 360/8 = 45 degrees. So octant 0 is 45 deg from octant 1 AND from octant 7.
#
# Every consecutive pair counts (1st->2nd, 2nd->3rd, ...). For each CSV we
# describe the distribution of jumps with a single mean and a single variance:
# the mean is the typical jump, and the variance (deg^2) is how spread out the
# jumps are around that mean within that CSV.
#
# An entry in RUNS_TO_ANALYZE can be a specific run tag ("outcome8_00000") or a
# whole outcome group ("outcome8"), which expands to every matching run folder.

import glob
import os
import re

import numpy as np
import pandas as pd

# ── Configure which runs to analyse ──────────────────────────────────────
# Each entry is matched as "<entry>_*" against the run folders, so:
#   "outcome8"        -> every outcome8_* run
#   "outcome8_00000"  -> just that one run
RUNS_TO_ANALYZE = [
    "newoutcome1",
]

N_MOTIVES = 8
DEGREES_PER_STEP = 360.0 / N_MOTIVES  # 45 degrees per octant step

# Sample variance (ddof=1). Set to 0 for the population variance.
VAR_DDOF = 1

# Cosine-scaled distance: instead of the linear angular distance we bend it
# through a (raised, inverted) cosine so near octants barely count and
# opposite octants count fully. A jump of 180 deg maps to COSINE_MAX, so the
# scaled values keep a degree-like range while the middle gets stretched.
COSINE_MAX = 180.0

# Extra punishment for big switches: the normalised cosine weight (0 = same
# octant, 1 = opposite octant) is raised to this power. 1.0 = plain cosine;
# >1 squashes near-octant switches toward zero so opposing switches dominate
# the mean even more (e.g. 2.0 makes an adjacent jump count ~4 instead of ~26).
COSINE_EXPONENT = 3.0

# A full U-turn = jumping to the opposite octant (180 deg). OPPOSITE_DEG is that
# distance for the current N_MOTIVES; we also report each CSV's rate of them.
OPPOSITE_DEG = (N_MOTIVES // 2) * DEGREES_PER_STEP  # 180 deg for 8 motives

# Write a one-row-per-CSV table and a one-row-per-run summary next to this file.
OUTPUT_PER_CSV = "angular_change_per_csv.csv"
OUTPUT_SUMMARY = "angular_change_summary.csv"

# ── Paths ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
RUNS_DIR = os.path.join(BASE, "..", "..", "runs")


# ── Step 1: the angular-distance helper ──────────────────────────────────
def angular_distance(m_a, m_b, n=N_MOTIVES):
    """Shortest angular distance (in degrees) between two motives on the
    circumplex. Mirrors the circular distance in influence.py:
    distance = min(|a-b|, n-|a-b|) steps, each step worth 360/n degrees.
    Works for motives whether they are labelled 0..n-1 or 1..n, as long as
    the labelling is consistent."""
    step_distance = min(abs(m_a - m_b), n - abs(m_a - m_b))
    return step_distance * (360.0 / n)


def cosine_scaled_distance(m_a, m_b, n=N_MOTIVES):
    """Same shortest octant distance, but bent through a cosine so the result
    grows non-linearly with separation: 0 deg -> 0, 180 deg -> COSINE_MAX,
    with adjacent octants compressed and opposing octants stretched.
    Uses the raised inverted cosine weight (1 - cos(angle)) / 2 (0..1), raised
    to COSINE_EXPONENT to punish big switches harder, where the angle is the
    same step-based angle used in influence.py."""
    step_distance = min(abs(m_a - m_b), n - abs(m_a - m_b))
    angle = step_distance * (2 * np.pi / n)  # 0..pi over the half-circle
    weight = (1 - np.cos(angle)) / 2  # 0 (same octant) .. 1 (opposite octant)
    return COSINE_MAX * weight ** COSINE_EXPONENT


# ── Step 2: consecutive distances for one behaviour sequence ─────────────
def consecutive_distances(sequence):
    """Angular distance for every consecutive pair in the sequence
    (1st->2nd, 2nd->3rd, ...)."""
    return [angular_distance(a, b) for a, b in zip(sequence[:-1], sequence[1:])]


def consecutive_cosine_distances(sequence):
    """Cosine-scaled distance for every consecutive pair in the sequence."""
    return [cosine_scaled_distance(a, b) for a, b in zip(sequence[:-1], sequence[1:])]


def short_tag(run_dir_name):
    """Turn 'outcome8_00000_2026-05-31_10-25-47' into 'outcome8_00000'."""
    m = re.match(r"(.+?_\d+)_\d{4}-\d{2}-\d{2}", run_dir_name)
    return m.group(1) if m else run_dir_name


def agent_from_filename(csv_name):
    """Pull the agent label out of 'simulation_a_sim0_round0.csv' -> 'a'.
    Files without an agent token (e.g. old 'simulation_3.csv') return 'all'."""
    m = re.match(r"simulation_([A-Za-z]+)_sim", csv_name)
    return m.group(1) if m else "all"


# ── Step 3: resolve the configured entries into concrete run folders ─────
def resolve_run_dirs(entries):
    """Expand each entry ('outcome8' or 'outcome8_00000') into the matching
    run directories, de-duplicated and sorted."""
    seen = set()
    run_dirs = []
    for entry in entries:
        matches = sorted(glob.glob(os.path.join(RUNS_DIR, f"{entry}_*")))
        dirs = [d for d in matches if os.path.isdir(d)]
        if not dirs:
            print(f"[skip] no run directory matches '{entry}'")
        for d in dirs:
            if d not in seen:
                seen.add(d)
                run_dirs.append(d)
    return run_dirs


# ── Step 4: loop over the resolved runs ──────────────────────────────────
summary_rows = []
per_csv_rows = []

for run_dir in resolve_run_dirs(RUNS_TO_ANALYZE):
    tag = short_tag(os.path.basename(run_dir))

    sim_files = sorted(glob.glob(os.path.join(run_dir, "simulation_*.csv")))
    if not sim_files:
        print(f"[skip] no simulation_*.csv in {tag}")
        continue

    # Per CSV we compute two parallel jump distributions: the plain angular
    # distance and the cosine-scaled one. Both are kept grouped by agent so the
    # summary can split them out.
    by_agent = {}  # agent -> {"means", "variances", "cos_means", "cos_variances"}
    for sim_file in sim_files:
        df = pd.read_csv(sim_file, usecols=["active_motive"])
        seq = df["active_motive"].dropna().astype(int).tolist()
        dists = consecutive_distances(seq)
        if not dists:
            continue
        cos_dists = consecutive_cosine_distances(seq)

        csv_mean = float(np.mean(dists))
        csv_var = float(np.var(dists, ddof=VAR_DDOF)) if len(dists) > 1 else 0.0
        csv_std = csv_var**0.5  # spread in degrees: most jumps within mean +/- std

        cos_mean = float(np.mean(cos_dists))
        cos_var = float(np.var(cos_dists, ddof=VAR_DDOF)) if len(cos_dists) > 1 else 0.0
        cos_std = cos_var**0.5

        # Rate of full U-turns: jumps straight to the opposite octant (180 deg).
        opp_rate = float(np.mean([d == OPPOSITE_DEG for d in dists]))

        agent = agent_from_filename(os.path.basename(sim_file))
        bucket = by_agent.setdefault(
            agent,
            {
                "means": [],
                "variances": [],
                "cos_means": [],
                "cos_variances": [],
                "opp_rates": [],
            },
        )
        bucket["means"].append(csv_mean)
        bucket["variances"].append(csv_var)
        bucket["cos_means"].append(cos_mean)
        bucket["cos_variances"].append(cos_var)
        bucket["opp_rates"].append(opp_rate)
        per_csv_rows.append(
            {
                "run_tag": tag,
                "agent": agent,
                "csv": os.path.basename(sim_file),
                "n_jumps": len(dists),
                "mean_jump_deg": round(csv_mean, 4),
                "variance_jump_deg2": round(csv_var, 4),
                "std_jump_deg": round(csv_std, 4),
                "mean_jump_cos": round(cos_mean, 4),
                "variance_jump_cos": round(cos_var, 4),
                "std_jump_cos": round(cos_std, 4),
                "opposite_switch_rate": round(opp_rate, 4),
            }
        )

    if not by_agent:
        print(f"[skip] no usable transitions in {tag}")
        continue

    print(f"{os.path.basename(run_dir)}")
    # One summary line + one summary row per agent in this run.
    for agent in sorted(by_agent):
        means = np.array(by_agent[agent]["means"])
        variances = np.array(by_agent[agent]["variances"])
        cos_means = np.array(by_agent[agent]["cos_means"])
        cos_variances = np.array(by_agent[agent]["cos_variances"])
        opp_rates = np.array(by_agent[agent]["opp_rates"])

        mean_jump = means.mean()
        std_deg = variances.mean() ** 0.5
        cos_mean_jump = cos_means.mean()
        cos_std = cos_variances.mean() ** 0.5
        mean_opp_rate = opp_rates.mean()

        print(
            f"  agent {agent}: {len(means)} csv | "
            f"mean {mean_jump:.3f} deg (std {std_deg:.3f}) | "
            f"cos mean {cos_mean_jump:.3f} (std {cos_std:.3f}) | "
            f"opp {mean_opp_rate * 100:.1f}%"
        )

        summary_rows.append(
            {
                "run_tag": tag,
                "run_dir": os.path.basename(run_dir),
                "agent": agent,
                "n_csvs": len(means),
                "mean_jump_deg": round(mean_jump, 4),
                "mean_variance_deg2": round(variances.mean(), 4),
                "std_deg": round(std_deg, 4),
                "mean_jump_cos": round(cos_mean_jump, 4),
                "mean_variance_cos": round(cos_variances.mean(), 4),
                "std_cos": round(cos_std, 4),
                "mean_opposite_rate": round(mean_opp_rate, 4),
            }
        )
    print()

# ── Step 5: save the tables ──────────────────────────────────────────────
if per_csv_rows:
    per_csv_path = os.path.join(BASE, OUTPUT_PER_CSV)
    pd.DataFrame(per_csv_rows).to_csv(per_csv_path, index=False)
    summary_path = os.path.join(BASE, OUTPUT_SUMMARY)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote per-csv table to {per_csv_path}")
    print(f"Wrote run summary to {summary_path}")
