def flat_decay(satisfaction_levels, active_motive, decay_rate=0.3):
    for i in range(len(satisfaction_levels)):
        if i != active_motive:
            satisfaction_levels[i] -= decay_rate
    return satisfaction_levels
