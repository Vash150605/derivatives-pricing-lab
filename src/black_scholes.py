import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """Compute d1 and d2 terms for Black-Scholes."""
    sq_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sq_t)
    d2 = d1 - sigma * sq_t
    return d1, d2


def bs_price(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, kind: str = "call"
) -> float:
    """Closed-form European option price ('call' or 'put')."""
    if T <= 0.0:
        return float(max(S - K, 0.0) if kind == "call" else max(K - S, 0.0))
    if sigma <= 0.0:
        fwd_diff = S * np.exp(-q * T) - K * np.exp(-r * T)
        return float(max(fwd_diff, 0.0) if kind == "call" else max(-fwd_diff, 0.0))

    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    eq = np.exp(-q * T)
    er = np.exp(-r * T)

    if kind == "call":
        return float(S * eq * norm.cdf(d1) - K * er * norm.cdf(d2))
    elif kind == "put":
        return float(K * er * norm.cdf(-d2) - S * eq * norm.cdf(-d1))
    raise ValueError("kind must be 'call' or 'put'")


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, kind: str = "call"
) -> dict:
    """Analytical Black-Scholes Greeks (delta, gamma, vega, theta, rho)."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    pdf = norm.pdf(d1)
    sq_t = np.sqrt(T)
    eq = np.exp(-q * T)
    er = np.exp(-r * T)

    gamma = eq * pdf / (S * sigma * sq_t)
    vega = S * eq * pdf * sq_t
    common_theta = -S * eq * pdf * sigma / (2.0 * sq_t)

    if kind == "call":
        delta = eq * norm.cdf(d1)
        theta = common_theta - r * K * er * norm.cdf(d2) + q * S * eq * norm.cdf(d1)
        rho = K * T * er * norm.cdf(d2)
    elif kind == "put":
        delta = -eq * norm.cdf(-d1)
        theta = common_theta + r * K * er * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1)
        rho = -K * T * er * norm.cdf(-d2)
    else:
        raise ValueError("kind must be 'call' or 'put'")

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def implied_vol(
    price: float, S: float, K: float, T: float, r: float, q: float = 0.0, kind: str = "call"
) -> float:
    """Invert Black-Scholes using Brent's method. Returns NaN if arbitrage bounds fail."""
    if T <= 0.0:
        return np.nan

    eq = np.exp(-q * T)
    er = np.exp(-r * T)

    if kind == "call":
        low_bound = max(S * eq - K * er, 0.0)
        high_bound = S * eq
    else:
        low_bound = max(K * er - S * eq, 0.0)
        high_bound = K * er

    if not (low_bound < price < high_bound):
        return np.nan

    objective = lambda sig: bs_price(S, K, T, r, sig, q, kind) - price
    try:
        return float(brentq(objective, 1e-5, 5.0, xtol=1e-10))
    except ValueError:
        return np.nan