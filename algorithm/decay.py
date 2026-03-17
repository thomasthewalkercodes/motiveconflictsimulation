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
                abs(
                    (i - motive_focus), len(satisfaction_levels) - abs(i - motive_focus)
                )
            )
            angle = distance * (2 * np.pi / len(satisfaction_levels))
            satisfaction_levels[i] -= amplitude * np.cos(angle) + elevation
    return satisfaction_levels
