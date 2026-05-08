"""
Synthetic OHLCV generator. Produces realistic-looking minute or daily bars
with trending periods and chop so the system actually fires signals.

Usage:
  python scripts/generate_synthetic_data.py --out data/synthetic_ohlcv.csv

Each symbol gets its own random regime sequence so different symbols generate
different setups.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_symbol(symbol: str, n_bars: int, start: datetime, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # build a piecewise-trend price series. Each regime has its own drift and vol.
    bars_per_regime = 60
    n_regimes = (n_bars // bars_per_regime) + 1
    drifts = rng.normal(0, 0.0008, n_regimes)         # daily drift per regime
    vols = rng.uniform(0.008, 0.025, n_regimes)       # vol per regime
    regime_idx = np.repeat(np.arange(n_regimes), bars_per_regime)[:n_bars]

    drift_series = drifts[regime_idx]
    vol_series = vols[regime_idx]

    # log-returns -> price
    rets = rng.normal(drift_series, vol_series)
    price = 100.0 * np.exp(np.cumsum(rets))

    # build OHLC around close-to-close path
    close = price
    open_ = np.concatenate([[price[0]], close[:-1]])
    intra_vol = vol_series * 0.6
    high_offset = rng.uniform(0, 1, n_bars) * intra_vol * close
    low_offset = rng.uniform(0, 1, n_bars) * intra_vol * close
    high = np.maximum(open_, close) + high_offset
    low = np.minimum(open_, close) - low_offset

    # volume: log-normal, with bursts during high-vol regimes
    base_vol = rng.lognormal(mean=12.5, sigma=0.4, size=n_bars)
    burst = (vol_series / vols.mean())
    volume = (base_vol * burst).astype(int)

    timestamps = [start + timedelta(minutes=i) for i in range(n_bars)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "symbol": symbol,
        "open": np.round(open_, 4),
        "high": np.round(high, 4),
        "low": np.round(low, 4),
        "close": np.round(close, 4),
        "volume": volume,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_ohlcv.csv")
    parser.add_argument("--symbols", nargs="+", default=["AAA", "BBB", "CCC"])
    parser.add_argument("--bars", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start = datetime(2024, 1, 1, 9, 30)
    frames = []
    for i, sym in enumerate(args.symbols):
        frames.append(generate_symbol(sym, args.bars, start, args.seed + i))

    df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} bars across {len(args.symbols)} symbols to {args.out}")


if __name__ == "__main__":
    main()
