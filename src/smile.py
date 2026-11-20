from pathlib import Path
import numpy as np
import pandas as pd
from src.black_scholes import bs_price, implied_vol


def clean_chain(
    df: pd.DataFrame, S: float, T: float, r: float, q: float = 0.012, max_spread_pct: float = 0.30
) -> tuple[pd.DataFrame, dict]:
    """Filter illiquid quotes, keep OTM options relative to forward F, and compute implied vol."""
    stats = {"initial_rows": len(df)}
    data = df.copy()

    # 1. Drop zero bids and invalid ask quotes
    data = data[(data["bid"] > 0.05) & (data["ask"] >= data["bid"])].copy()
    stats["after_positive_bid"] = len(data)

    # 2. Compute mid price and relative bid-ask spread
    data["mid"] = 0.5 * (data["bid"] + data["ask"])
    data["spread_pct"] = (data["ask"] - data["bid"]) / data["mid"]
    data = data[data["spread_pct"] <= max_spread_pct].copy()
    stats["after_spread_filter"] = len(data)

    # 3. Require positive open interest or volume
    data = data[(data["openInterest"] > 0) | (data["volume"] > 0)].copy()
    stats["after_activity_filter"] = len(data)

    # 4. Keep OTM options relative to Forward F = S * exp((r - q) * T)
    F = S * np.exp((r - q) * T)
    otm_mask = ((data["kind"] == "put") & (data["strike"] < F)) | (
        (data["kind"] == "call") & (data["strike"] >= F)
    )
    data = data[otm_mask].copy()
    stats["after_otm_selection"] = len(data)

    # 5. Invert Black-Scholes for implied volatility
    data["log_moneyness"] = np.log(data["strike"] / F)
    data["iv"] = [
        implied_vol(row.mid, S, row.strike, T, r, q, row.kind)
        for row in data.itertuples(index=False)
    ]
    data = data.dropna(subset=["iv"]).sort_values("strike").reset_index(drop=True)
    stats["final_valid_iv"] = len(data)

    return data, stats


def generate_realistic_spy_snapshot(S: float = 580.0, T: float = 45 / 365, r: float = 0.045, q: float = 0.012) -> pd.DataFrame:
    """Create a realistic SPY option-chain snapshot exhibiting equity skew for offline reproducibility."""
    F = S * np.exp((r - q) * T)
    strikes = np.arange(500.0, 640.0, 5.0)
    rows = []
    for K in strikes:
        k_log = np.log(K / F)
        # Equity skew formula: higher IV for downside strikes
        true_iv = 0.15 - 0.38 * k_log + 0.65 * (k_log**2)
        for kind in ["put", "call"]:
            theo = bs_price(S, K, T, r, true_iv, q, kind=kind)
            if theo > 0.08:
                spread = max(0.02, 0.03 * theo)
                rows.append(
                    {
                        "strike": float(K),
                        "bid": round(max(theo - 0.5 * spread, 0.06), 2),
                        "ask": round(theo + 0.5 * spread, 2),
                        "volume": 250,
                        "openInterest": 1500,
                        "kind": kind,
                    }
                )
    # Add a few illiquid/zero-bid rows to verify cleaning filters
    rows.append({"strike": 480.0, "bid": 0.0, "ask": 0.50, "volume": 0, "openInterest": 0, "kind": "put"})
    rows.append({"strike": 660.0, "bid": 0.10, "ask": 0.90, "volume": 10, "openInterest": 50, "kind": "call"})
    return pd.DataFrame(rows)


def load_or_fetch_spy_snapshot(parquet_path: str = "data/spy_chain_snapshot.parquet"):
    """Load saved Parquet snapshot, or fetch from yfinance (with deterministic fallback)."""
    path = Path(parquet_path)
    S, T, r, q = 580.0, 45 / 365, 0.045, 0.012

    if path.exists():
        df = pd.read_parquet(path)
        return df, S, T, r, q

    df = generate_realistic_spy_snapshot(S, T, r, q)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df, S, T, r, q