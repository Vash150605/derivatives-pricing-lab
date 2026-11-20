import numpy as np
from src.smile import generate_realistic_spy_snapshot, clean_chain


def test_clean_chain_filters_and_recovers_skew():
    S, T, r, q = 580.0, 45 / 365, 0.045, 0.012
    raw_df = generate_realistic_spy_snapshot(S, T, r, q)
    cleaned, stats = clean_chain(raw_df, S, T, r, q)

    # Ensure illiquid quotes were dropped
    assert stats["final_valid_iv"] < stats["initial_rows"]
    assert (cleaned["bid"] > 0).all()
    assert (cleaned["spread_pct"] <= 0.30).all()

    # Check OTM separation at forward price F
    F = S * np.exp((r - q) * T)
    puts = cleaned[cleaned["kind"] == "put"]
    calls = cleaned[cleaned["kind"] == "call"]
    assert (puts["strike"] < F).all()
    assert (calls["strike"] >= F).all()

    # Downside OTM puts should have higher implied vol than ATM calls (equity skew)
    low_put_iv = puts.iloc[0]["iv"]
    atm_call_iv = calls.iloc[0]["iv"]
    assert low_put_iv > atm_call_iv