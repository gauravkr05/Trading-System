import pandas as pd

# Adjust path to your actual data file
df = pd.read_csv("data/yf_ohlcv.csv")  # whatever your real path is

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Pick one symbol to inspect
sym_df = df[df['symbol'] == df['symbol'].iloc[0]].copy().sort_values('timestamp').reset_index(drop=True)

# Compute the time gap between consecutive bars
sym_df['gap_minutes'] = sym_df['timestamp'].diff().dt.total_seconds() / 60

# Show distribution of gap sizes
print("Gap distribution between consecutive bars (in minutes):")
print(sym_df['gap_minutes'].value_counts().sort_index().head(20))

# Show the first few bars of each day to see open/close times
sym_df['date'] = sym_df['timestamp'].dt.date
sym_df['time'] = sym_df['timestamp'].dt.time
print("\nFirst bar of each day (first 5 days):")
print(sym_df.groupby('date').first().head(5)[['time']])
print("\nLast bar of each day (last 5 days):")
print(sym_df.groupby('date').last().tail(5)[['time']])