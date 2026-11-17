import numpy as np
import pytest
from src.black_scholes import bs_price
from src.binomial import crr_price

BASE = dict(S=100.0, K=105.0, T=0.5, r=0.03, sigma=0.20, q=0.0)


def test_two_step_hand_calculation():
    S, K, T, r, sigma, N = 100.0, 100.0, 1.0, 0.05, 0.20, 2
    dt = 0.5
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Terminal call payoffs at N=2: down-down (0), up-down (0), up-up (S*u^2 - K)
    v_uu = max(S * u**2 - K, 0.0)
    v_ud = max(S * u * d - K, 0.0)
    v_dd = max(S * d**2 - K, 0.0)

    # Step 1
    v_u = disc * (p * v_uu + (1 - p) * v_ud)
    v_d = disc * (p * v_ud + (1 - p) * v_dd)
    # Step 0
    expected = disc * (p * v_u + (1 - p) * v_d)

    assert abs(crr_price(S, K, T, r, sigma, N=2, kind="call") - expected) < 1e-12


def test_european_tree_converges_to_bs():
    bs_call = bs_price(**BASE, kind="call")
    err_50 = abs(crr_price(**BASE, N=50, kind="call") - bs_call)
    err_1000 = abs(crr_price(**BASE, N=1000, kind="call") - bs_call)
    assert err_1000 < err_50
    assert err_1000 < 0.01


def test_american_call_no_dividend_equals_european_call():
    # Without dividends (q=0), it is never optimal to exercise an American call early
    euro_call = crr_price(**BASE, N=400, kind="call", american=False)
    amer_call = crr_price(**BASE, N=400, kind="call", american=True)
    assert abs(amer_call - euro_call) < 1e-12


def test_american_put_exceeds_european_put_and_intrinsic():
    euro_put = crr_price(**BASE, N=400, kind="put", american=False)
    amer_put = crr_price(**BASE, N=400, kind="put", american=True)
    intrinsic = max(BASE["K"] - BASE["S"], 0.0)
    assert amer_put > euro_put
    assert amer_put >= intrinsic


def test_invalid_probability_raises_error():
    # Huge rate r with tiny volatility forces p > 1 for small N
    with pytest.raises(ValueError):
        crr_price(100.0, 100.0, 1.0, r=0.80, sigma=0.01, N=2)