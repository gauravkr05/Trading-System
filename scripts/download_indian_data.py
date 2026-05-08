import yfinance as yf
import pandas as pd
import yaml
from datetime import datetime, timedelta

# load symbols from config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

symbols  = cfg["backtest"]["symbols"]
end      = datetime.today()
start    = end - timedelta(days=59)

all_dfs = []

for symbol in symbols:
    print(f"Downloading {symbol}...")
    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval="15m",       # 15 minute candles
            progress=False,
            auto_adjust=True,
        )

        if df.empty:
            print(f"  ⚠ No data for {symbol}, skipping")
            continue

        # NEW
        df = df.reset_index()
        # flatten multi-level columns if present
        if isinstance(df.columns[0], tuple):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"datetime": "timestamp", "date": "timestamp"})
        df["symbol"] = symbol
        df = df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
        all_dfs.append(df)
        print(f"  ✓ {len(df)} bars")

    except Exception as e:
        print(f"  ✗ Error: {e}")

# combine and save
final = pd.concat(all_dfs, ignore_index=True)
final = final.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
final.to_csv("data/yf_ohlcv.csv", index=False)
print(f"\n✅ Saved {len(final)} total bars for {len(all_dfs)} symbols")