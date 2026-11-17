import numpy as np


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int = 500,
    q: float = 0.0,
    kind: str = "call",
    american: bool = False,
) -> float:
    """Cox-Ross-Rubinstein binomial tree pricer for European and American options."""
    if T <= 0.0:
        return float(max(S - K, 0.0) if kind == "call" else max(K - S, 0.0))

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if not (0.0 < p < 1.0):
        raise ValueError(f"Risk-neutral probability p={p:.4f} outside (0,1). Increase N.")

    payoff_fn = (
        (lambda s: np.maximum(s - K, 0.0))
        if kind == "call"
        else (lambda s: np.maximum(K - s, 0.0))
    )

    # Terminal node prices at step N (j = 0, 1, ..., N up moves)
    j = np.arange(N + 1)
    V = payoff_fn(S * (u**j) * (d ** (N - j)))

    # Backward induction from step N-1 down to 0
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[1 : i + 2] + (1.0 - p) * V[0 : i + 1])
        if american:
            jj = np.arange(i + 1)
            spot_nodes = S * (u**jj) * (d ** (i - jj))
            V = np.maximum(V, payoff_fn(spot_nodes))

    return float(V[0])