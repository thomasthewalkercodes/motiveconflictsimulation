def flat_growth(satisfaction_levels, active_motive, growth_rate=1):
    satisfaction_levels[active_motive] += growth_rate
    return satisfaction_levels
