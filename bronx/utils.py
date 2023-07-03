def anneal_schedule(dt, t):
    return max(min(dt/t, 1.0), 1e-8)
