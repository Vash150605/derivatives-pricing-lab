import numpy as np
import pytest
from src.black_scholes import bs_price, bs_greeks, implied_vol

BASE = dict(S=100.0, K=105.0, T=0.5, r=0.03, sigma=0.20, q=0.01)


def test_put_call_parity():
    call = bs_price(**BASE, kind="call")
    put = bs_price(**BASE, kind="put")
    lhs = call - put
    rhs = BASE["S"] * np.exp(-BASE["q"] * BASE["T"]) - BASE["K"] * np.exp(-BASE["r"] * BASE["T"])
    assert abs(lhs - rhs) < 1e-12


def test_boundary_cases():
    # T = 0 gives intrinsic payoff
    assert bs_price(110.0, 100.0, 0.0, 0.03, 0.2, kind="call") == 10.0
    assert bs_price(90.0, 100.0, 0.0, 0.03, 0.2, kind="put") == 10.0

    # sigma = 0 gives discounted forward value
    expected_zero_vol = 110.0 * np.exp(-0.01 * 0.5) - 100.0 * np.exp(-0.03 * 0.5)
    assert abs(bs_price(110.0, 100.0, 0.5, 0.03, 0.0, q=0.01, kind="call") - expected_zero_vol) < 1e-12

    # Deep OTM call is near 0; Deep ITM call is near discounted forward
    assert bs_price(100.0, 500.0, 0.5, 0.03, 0.2, kind="call") < 1e-10
    deep_itm = bs_price(500.0, 50.0, 0.5, 0.03, 0.2, q=0.01, kind="call")
    fwd = 500.0 * np.exp(-0.01 * 0.5) - 50.0 * np.exp(-0.03 * 0.5)
    assert abs(deep_itm - fwd) < 1e-6


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    g = bs_greeks(**BASE, kind=kind)
    P = lambda **kw: bs_price(**{**BASE, "kind": kind, **kw})

    h_s = 1e-4 * BASE["S"]
    h = 1e-5

    delta_fd = (P(S=BASE["S"] + h_s) - P(S=BASE["S"] - h_s)) / (2.0 * h_s)
    gamma_fd = (P(S=BASE["S"] + h_s) - 2.0 * P() + P(S=BASE["S"] - h_s)) / (h_s**2)
    vega_fd = (P(sigma=BASE["sigma"] + h) - P(sigma=BASE["sigma"] - h)) / (2.0 * h)
    theta_fd = -(P(T=BASE["T"] + h) - P(T=BASE["T"] - h)) / (2.0 * h)
    rho_fd = (P(r=BASE["r"] + h) - P(r=BASE["r"] - h)) / (2.0 * h)

    assert abs(g["delta"] - delta_fd) < 1e-6
    assert abs(g["gamma"] - gamma_fd) < 1e-6
    assert abs(g["vega"] - vega_fd) < 1e-6
    assert abs(g["theta"] - theta_fd) < 1e-6
    assert abs(g["rho"] - rho_fd) < 1e-6


@pytest.mark.parametrize("sigma", [0.10, 0.25, 0.50])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_vol_round_trip(sigma, kind):
    price = bs_price(100.0, 105.0, 0.5, 0.03, sigma, q=0.01, kind=kind)
    iv = implied_vol(price, 100.0, 105.0, 0.5, 0.03, q=0.01, kind=kind)
    assert abs(iv - sigma) < 1e-8

    # Arbitrage-violating price returns NaN
    assert np.isnan(implied_vol(-1.0, 100.0, 105.0, 0.5, 0.03, kind=kind))
    assert np.isnan(implied_vol(150.0, 100.0, 105.0, 0.5, 0.03, kind=kind))