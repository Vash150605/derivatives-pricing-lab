import numpy as np


def mc_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n: int = 200_000,
    kind: str = "call",
    antithetic: bool = False,
    seed: int = 0,
) -> tuple[float, float]:
    """Monte Carlo European option pricer with exact GBM step. Returns (price, std_error)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)

    def disc_payoff(z_draws):
        ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z_draws)
        pay = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
        return np.exp(-r * T) * pay

    if antithetic:
        # Pairwise average of +Z and -Z preserves independence across n pairs
        x = 0.5 * (disc_payoff(z) + disc_payoff(-z))
    else:
        x = disc_payoff(z)

    price = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    return price, se


def mc_price_cv(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n: int = 200_000,
    kind: str = "call",
    seed: int = 0,
) -> tuple[float, float]:
    """Monte Carlo pricer using discounted terminal stock price as a Control Variate."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)

    ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    pay = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)

    X = np.exp(-r * T) * pay
    Y = np.exp(-r * T) * ST
    expected_Y = S0 * np.exp(-q * T)

    b_opt = np.cov(X, Y)[0, 1] / Y.var(ddof=1)
    X_cv = X - b_opt * (Y - expected_Y)

    return float(X_cv.mean()), float(X_cv.std(ddof=1) / np.sqrt(n))


def simulate_gbm_paths(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_paths: int = 1000,
    n_steps: int = 252,
    seed: int = 0,
) -> np.ndarray:
    """Simulate exact GBM paths of shape (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_inc = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1)
    return S0 * np.exp(log_paths)