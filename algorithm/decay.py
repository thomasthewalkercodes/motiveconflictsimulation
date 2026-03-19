import numpy as np


def flat_decay(satisfaction_levels, active_motive, decay_rate=0.3):
    for i in range(len(satisfaction_levels)):
        if i != active_motive:
            satisfaction_levels[i] -= decay_rate
    return satisfaction_levels


def cos_decay(satisfaction_levels, active_motive, motive_focus, amplitude, elevation):
    for i in range(len(satisfaction_levels)):
        if i != active_motive:
            distance = min(
                abs(i - motive_focus), len(satisfaction_levels) - abs(i - motive_focus)
            )
            angle = distance * (2 * np.pi / len(satisfaction_levels))
            satisfaction_levels[i] -= amplitude * np.cos(angle) + elevation
    return satisfaction_levels


def generate_decay(behavior_ratio, total_budget):
    decay_rates = (
        behavior_ratio / behavior_ratio.sum() * total_budget
    )  # redistributes the right amount of decay, so all stays in bounds
    n = len(decay_rates)
    mirrored = np.zeros_like(
        decay_rates
    )  # mirrors it because its what gets "invoked" or "invited" from the other
    for i in range(n):
        mirrored[(n - i) % n] = decay_rates[i]  # flips on the "warmth" axis
    decay_rates = mirrored
    return decay_rates


def ratio_decay(satisfaction_levels, active_motive, decay_rates):
    for i in range(len(satisfaction_levels)):
        if i != active_motive:
            satisfaction_levels[i] -= decay_rates[i]
    return satisfaction_levels
