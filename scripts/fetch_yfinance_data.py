"""
Fetch real OHLCV from Yahoo Finance and write it in the schema the
system expects: timestamp, symbol, open, high, low, close, volume.

Usage:
  python scripts/fetch_yfinance_data.py --symbols AAPL MSFT NVDA \
                                        --period 6mo --interval 1d
"""
from __future__ import annotations
import argparse, os, sys
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fetch(symbols: list[str], period: str, interval: str, out: str) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        df = yf.download(sym, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"warning: no data for {sym}")
            continue
        df = df.reset_index()
        # yfinance gives 'Date' for daily and 'Datetime' for intraday
        ts_col = "Datetime" if "Datetime" in df.columns else "Date"
        # handle multi-index columns that yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        out_df = pd.DataFrame({
            "timestamp": pd.to_datetime(df[ts_col]),
            "symbol": sym,
            "open":   df["Open"].astype(float),
            "high":   df["High"].astype(float),
            "low":    df["Low"].astype(float),
            "close":  df["Close"].astype(float),
            "volume": df["Volume"].astype(int),
        })
        frames.append(out_df)
    if not frames:
        raise RuntimeError("no data fetched for any symbol")
    out_df = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Wrote {len(out_df)} bars across {len(frames)} symbols to {out}")
    return out_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--period", default="6mo",
                   help="1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")
    p.add_argument("--interval", default="1d",
                   help="1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk")
    p.add_argument("--out", default="data/yf_ohlcv.csv")
    args = p.parse_args()
    fetch(args.symbols, args.period, args.interval, args.out)


if __name__ == "__main__":
    main()