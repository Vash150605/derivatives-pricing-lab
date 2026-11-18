import numpy as np
from src.black_scholes import bs_price
from src.monte_carlo import mc_price, mc_price_cv, simulate_gbm_paths

BASE = dict(S=100.0, K=105.0, T=0.5, r=0.03, sigma=0.20, q=0.0)


def test_mc_within_3_se_of_closed_form():
    bs_val = bs_price(**BASE, kind="call")
    price, se = mc_price(100.0, 105.0, 0.5, 0.03, 0.20, n=200_000, kind="call", seed=42)
    assert abs(price - bs_val) < 3.0 * se


def test_variance_reduction_lowers_standard_error():
    _, se_plain = mc_price(100.0, 105.0, 0.5, 0.03, 0.20, n=100_000, antithetic=False, seed=7)
    _, se_anti = mc_price(100.0, 105.0, 0.5, 0.03, 0.20, n=100_000, antithetic=True, seed=7)
    _, se_cv = mc_price_cv(100.0, 105.0, 0.5, 0.03, 0.20, n=100_000, seed=7)

    assert se_anti < se_plain
    assert se_cv < se_plain


def test_confidence_interval_coverage():
    # Over 100 independent seeds, ~95% of 95% CIs should contain the true Black-Scholes price
    true_price = bs_price(**BASE, kind="call")
    hits = 0
    for s in range(100):
        est, se = mc_price(100.0, 105.0, 0.5, 0.03, 0.20, n=10_000, seed=s)
        if abs(est - true_price) <= 1.96 * se:
            hits += 1
    assert 88 <= hits <= 99


def test_simulate_gbm_paths_shape_and_initial_spot():
    paths = simulate_gbm_paths(100.0, 0.5, 0.03, 0.20, n_paths=50, n_steps=20, seed=1)
    assert paths.shape == (50, 21)
    assert np.allclose(paths[:, 0], 100.0)