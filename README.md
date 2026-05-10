# Trading System with Regression Gatekeeper + Claude MCP Reasoning

A reference implementation of a layered intraday trading pipeline:
scanner → portfolio check → filter (ATR/ADX) → bias (VWAP / MA / MACD) →
entry layer (Stage 1 state + Stage 1.5 exhaustion + Stage 2 trigger) →
feature vector → regression model → confidence gate → Claude MCP for
borderline cases → risk sizing → order. Every layer writes to a decision
log spine, and the simulator models real Indian retail intraday trading
costs and constraints (slippage, brokerage, gap fills, EOD discipline,
MIS leverage).

## Project layout

```
trading_system/
├── config.yaml                     # all thresholds and tunables
├── requirements.txt
├── data/                           # OHLCV csvs and the SQLite log
├── models/                         # saved regression model + scaler
├── src/
│   ├── data/
│   │   ├── data_loader.py          # OHLCV CSV loader (UTC -> IST)
│   │   └── indicators.py           # ATR, ADX, VWAP, EMA, MACD, RSI
│   ├── layers/
│   │   ├── scanner.py              # universe -> watchlist
│   │   ├── portfolio.py            # Position + manage path + EOD exit
│   │   ├── filter_layer.py         # ATR / ADX gate
│   │   ├── bias_layer.py           # VWAP primary, MA + MACD confirm
│   │   ├── entry_layer.py          # Stage 1 state + 1.5 exhaustion + Stage 2 trigger
│   │   └── risk_sizing.py          # 2x ATR stop, 3x ATR target + cost model
│   ├── ml/
│   │   ├── feature_builder.py      # log spine -> feature vector
│   │   ├── regression_model.py     # Ridge gatekeeper
│   │   └── confidence_gate.py      # far+/borderline/far- routing
│   ├── reasoning/
│   │   └── claude_mcp.py           # Claude API + heuristic fallback
│   ├── logs/
│   │   └── decision_log.py         # SQLite log spine
│   └── orchestrator.py             # ties every layer together + EOD entry block
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── train_regression.py
│   └── run_backtest.py
└── tests/
    └── test_pipeline.py
```

## Step 1: install

You need Python 3.10 or newer.

```bash
cd trading_system
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies:

- numpy, pandas — data and indicators
- scikit-learn, joblib — Ridge regression + persistence
- PyYAML — config loading
- python-dotenv — environment variables for Claude API key
- anthropic — optional, only if you want real Claude calls instead of the
  heuristic fallback
- matplotlib, plotly — optional, for charting your own analysis

## Step 2: prepare data

The system was designed for Indian equity intraday trading at 15-minute
bars. The data loader expects a CSV with columns `timestamp, symbol,
open, high, low, close, volume`. Yahoo Finance and most NSE data vendors
provide this format.

**Important:** the loader assumes timestamps are in UTC and converts them
to IST internally. All downstream code (EOD logic, time-of-day filters,
SQL queries) operates in IST. If your source data is already in IST,
adjust `data_loader.py` accordingly.

For testing without real data, the system ships with a synthetic
generator:

```bash
python scripts/generate_synthetic_data.py \
    --symbols AAA BBB CCC \
    --bars 2000 \
    --out data/synthetic_ohlcv.csv
```

Point `config.yaml -> backtest.data_path` at your CSV.

## Step 3: run the first backtest (cold start, heuristics only)

For the cleanest starting baseline, disable both the regression gatekeeper
and Claude reasoning so every setup that survives the heuristic layers
becomes a trade. In `config.yaml`:

```yaml
regression:
  gatekeeper_enabled: false

claude_mcp:
  enabled: false
```

Then run:

```bash
python scripts/run_backtest.py --config config.yaml --reset-log
```

You will see a summary like:

```
=== Backtest summary ===
                trades: 173
                 holds: 1230
              no_trade: 33221
                  wait: 1073
       skipped_scanner: 47825
   total_setups_logged: 83693

=== Closed trades: 171 ===
  mean R    : -0.578
  win rate  : 23.4%
  best      : +3.409
  worst     : -2.678
  median    : -0.594
```

These numbers are net of all realistic frictions (see Realism Layers
below). The decision log now contains every setup at every layer.

## Step 4: train the regression gatekeeper

```bash
python scripts/train_regression.py --config config.yaml
```

This pulls every closed setup with both a feature vector and a realized R
from the decision log, fits a Ridge model on standard-scaled features,
and saves the model and scaler to `models/`. Don't expect a high R² —
financial data has low signal — but the coefficients should make
directional sense (positive for ADX and confirmation count, etc).

## Step 5: run with the trained gate

Re-enable the gatekeeper in config:

```yaml
regression:
  gatekeeper_enabled: true
```

Then re-run the backtest. The regression now scores every setup.
Far-positive scores skip Claude and go straight to risk sizing;
far-negative scores get rejected without Claude; only the borderline
middle band escalates to Claude (or the heuristic fallback). Compare
trade count and mean R against the heuristics-only baseline — a
well-trained gatekeeper should improve mean R by filtering out the worst
setups.

## Step 6: enable real Claude reasoning (optional)

By default the Claude MCP layer uses a deterministic heuristic so the
system runs offline. To use the real Claude API:

1. Get an API key from https://console.anthropic.com.
2. Set it in a `.env` file at the project root:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
3. In `config.yaml` set `claude_mcp.enabled: true`.
4. Run a backtest. Borderline setups will hit the API. Far+ and far-
   setups still skip Claude entirely so you don't burn tokens on
   confident decisions.

If the key is missing or the `anthropic` package is not installed, the
reasoner silently falls back to the heuristic — no errors, just a
`mode: heuristic` field in the log.

## Realism layers (how the simulator avoids lying to you)

The simulator models the real frictions of Indian retail intraday
trading. Each layer was added because, without it, backtest results
overstate live performance.

| Layer | What it models |
|-------|----------------|
| **Tiered slippage** | 0.10% on entries, 0.10% on targets, 0.25% on stops. Stops slip worse because price is moving against you fast and liquidity vanishes. |
| **Brokerage and charges** | ₹40 round-trip brokerage (Zerodha/Upstox-style) + 0.08% combined for STT, exchange transaction charges, GST, and stamp duty. |
| **Gap fills** | If a bar opens past your stop, fill at the open (not the stop level). Same for gap-ups past target. Long and short symmetric. |
| **Integer share rounding** | Position size rounded down to whole shares. Effective risk is recomputed using actual integer size. |
| **Cash availability check** | If position notional exceeds account equity, position is capped at affordable size. |
| **MIS leverage modeling** | Account equity in config is buying power (real capital × MIS leverage). Risk percentage is set so absolute risk per trade matches 1% of real capital. |
| **EOD force-exit** | Existing positions force-closed at 15:00 IST to avoid broker auto-square-off slippage and overnight gap risk. |
| **EOD entry block** | New entries blocked at/after 15:00 IST so positions don't open on the EOD bar (which would either immediately force-exit or silently carry overnight). |
| **IST timestamps** | UTC source data converted to IST at load time. All EOD logic, queries, and config values are in IST. |
| **Short-trade direction filter** | Optional config flag to disable shorts (needed if you only trade cash equity delivery; intraday MIS allows shorts). |
| **Session column** | Auto-derived from date for session-VWAP reset. |

The simulator does NOT yet model:

- Order rejection / partial fills (~5% of orders in real markets)
- Circuit limits (stocks frozen at upper/lower bound)
- T2T segment exclusion (some stocks can't be intraday-squared)
- Survivorship bias in the universe
- Time-of-day quality variations within the session (first 30 min and
  last 30 min behave differently from midday)

These are diminishing returns. Adding them might shift mean R by another
0.05–0.10 R but won't change the directional picture.

## Configuration cheat sheet

Key sections of `config.yaml`:

`# =============================================================================
# Trading system configuration
# Every threshold, period, and tunable lives here so layers stay clean.
# =============================================================================

scanner:
  min_avg_volume: 100000      # below this and the symbol drops out of the watchlist
  min_price: 5.0              # avoid penny stocks
  max_price: 5000.0           # avoid ultra-high priced

filter_layer:
  atr_period: 14
  atr_min: 0.5                # below this and the symbol is too quiet to trade
  adx_period: 14
  adx_min: 20.0
  adx_max: 45.0               # below this and there is no trend strength

bias_layer:
  vwap_lookback: 0            # 0 = session VWAP, >0 = rolling VWAP window
  ma_period: 20               # the moving average that confirms or stays neutral
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9

entry_layer:
  # Stage 1 state confirmations
  rsi_period: 14
  rsi_long_min: 50            # in a long bias we want momentum above 50
  rsi_short_max: 50           # in a short bias we want momentum below 50
  volume_ma_period: 20
  volume_ratio_min: 1.0       # current volume must beat the recent average
  expansion_lookback: 5       # current ATR vs ATR n bars ago
  expansion_ratio_min: 1.0    # ATR must be expanding
  level_lookback: 20          # bars to scan for swing levels
  level_proximity_atr: 1.0    # within 1x ATR counts as "at level"
  min_state_confirmations: 3  # of {momentum, volume, expansion, level}, need this many

  # Stage 2 trigger
  pattern_body_ratio: 0.6     # body must be at least this fraction of range for engulfing
  wick_ratio_min: 2.0 
  
  observability:
    bias_continuity_lookback: 30   # max bars to look back for "bars since cross"
    pullback_lookback: 20  
    
  exhaustion:
    enabled: true
    max_distance_from_ma_atr: 1.5   # reject if close > 1.8 ATR from MA in bias direction
    min_pullback_depth_atr: 0.4     # reject if entering within 0.4 ATR of recent swing extreme      # wick must be this many times body for hammer / shooting star
  

risk_sizing:
  stop_atr_mult: 2.0
  target_atr_mult: 3.0
  account_equity: 80000.0
  risk_per_trade_pct: 0.0025
  allow_shorts: true
      # risk 1% of equity per trade
      # risk 1% of equity per trade

regression:
  model_path: "models/regression_model.joblib"
  scaler_path: "models/regression_scaler.joblib"
  features:
    - atr
    - adx
    - bias_strength
    - state_confirmations
    - rsi
    - volume_ratio
    - expansion_ratio
    - level_distance_atr
    - pattern_strength
  gatekeeper_enabled: false

confidence_gate:
  # the regression outputs y_hat (expected R-multiple).
  # we route on |y_hat| against these thresholds.
  far_positive_threshold: 0.6  # y_hat >= this -> take, skip Claude
  far_negative_threshold: -0.2 # y_hat <= this -> NO_TRADE, skip Claude
  # anything in between is borderline and goes to Claude.
  
manage:
  trail_start_r: 1.0
  trail_step_r: 999.0    # huge step = only the first ratchet fires
  trail_offset_r: 1.0
  eod_exit: true
  eod_hour: 15        # 3:00 PM IST (15:00) — was 9 (UTC)
  eod_minute: 0       # was 30 (UTC)
  eod_block_entries: true    # at MFE 1.0R, stop -> entry, never moves again   # stop sits this far below current step level
                         # offset=1.0 means: at MFE 1.0 stop=entry; at MFE 1.5 stop=+0.5R; etc.

ollama:
  enabled: false
  model: llama3.1:latest          # match exactly what `ollama list` shows
  temperature: 0.2
  timeout_seconds: 120

costs:
  slippage_entry_pct: 0.0010      # 0.10% on entries (mid-cap default)
  slippage_target_pct: 0.0010     # 0.10% on target exits
  slippage_stop_pct: 0.0025       # 0.25% on stop-outs (slippage is worse here)
  brokerage_round_trip: 40.0      # rupees, Zerodha/Upstox-style
  other_charges_pct: 0.0008       # STT + exchange + GST + stamp duty      # STT + exchange + GST + stamp duty combined

decision_log:
  db_path: "data/decision_log.sqlite"

backtest:
  data_path: "data/yf_ohlcv.csv"   # was data/synthetic_ohlcv.csv
  symbols: [
    # NIFTY 50
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NESTLEIND.NS", "POWERGRID.NS",
    "NTPC.NS", "ONGC.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "HCLTECH.NS",
    "TECHM.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "GRASIM.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "INDUSINDBK.NS", "JSWSTEEL.NS", "M&M.NS", "TATACONSUM.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "UPL.NS", "VEDL.NS", "BPCL.NS",
    "EICHERMOT.NS", "BRITANNIA.NS", "SHREECEM.NS", "SBILIFE.NS", "HDFCLIFE.NS",
    # NIFTY NEXT 50
    "BANKBARODA.NS", "CANBK.NS", "CHOLAFIN.NS", "DABUR.NS", "DLF.NS",
    "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS", "ICICIlombard.NS", "ICICIGI.NS",
    "INDHOTEL.NS", "IOC.NS", "IRCTC.NS", "JINDALSTEL.NS", "LICI.NS",
    "LUPIN.NS", "MARICO.NS", "MUTHOOTFIN.NS", "NAUKRI.NS", "OBEROIRLTY.NS",
    "OFSS.NS", "PAGEIND.NS", "PIDILITIND.NS", "PIIND.NS", "PNB.NS",
    "RECLTD.NS", "SAIL.NS", "SIEMENS.NS", "SRF.NS", "TORNTPHARM.NS",
    "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", "UNIONBANK.NS", "VBL.NS",
    "VOLTAS.NS", "WHIRLPOOL.NS", "ZOMATO.NS", "NYKAA.NS", "PAYTM.NS",
    "POLICYBZR.NS", "DELHIVERY.NS", "CAMPUS.NS", "BIKAJI.NS", "RAINBOW.NS",
    "MEDANTA.NS", "MANKIND.NS", "TIINDIA.NS", "ELGIEQUIP.NS", "KALYANKJIL.NS"
  ]
  warmup_bars: 50
  interval: "15m"
  period_days: 60            # wait this many bars before the system starts trading




  

## Retraining loop

The architecture is self-improving:

1. Live or backtest trades flow into the decision log.
2. When trades close, `manage_path` writes the realized R (after costs)
   into the outcome column of the original setup's row.
3. Periodically re-run `scripts/train_regression.py`. The model now has
   more data, including borderline cases that Claude resolved.
4. The far-positive and far-negative bands widen as the model sharpens,
   so fewer setups need Claude over time.

Monitor by counting `claude.skipped == True` rows vs `False` rows in the
log. The ratio should rise over time if retraining is working.

**Important:** retrain whenever you materially change the cost model,
EOD logic, or feature set. The model learns the joint distribution of
features and outcomes — if outcomes are systematically different (e.g.
more EOD exits, smaller R per trade), old models become stale.

## Inspecting the log

Open the SQLite log directly:

```bash
sqlite3 data/decision_log.sqlite

# How did each layer reject setups?
sqlite> SELECT status, COUNT(*) FROM setups GROUP BY status ORDER BY 2 DESC;

# Distribution of exits among closed trades
sqlite> SELECT
   ...>   json_extract(outcome, '$.exit_reason') AS reason,
   ...>   COUNT(*) AS n,
   ...>   ROUND(AVG(json_extract(outcome, '$.holding_bars')), 1) AS avg_bars,
   ...>   ROUND(AVG(json_extract(costs, '$.realized_r_net')), 3) AS avg_net_r
   ...> FROM setups WHERE status = 'closed' GROUP BY reason;

# Per-symbol performance (find your best and worst stocks)
sqlite> SELECT symbol, COUNT(*) AS n,
   ...>        ROUND(AVG(json_extract(costs, '$.realized_r_net')), 3) AS avg_r
   ...>   FROM setups WHERE status = 'closed'
   ...>   GROUP BY symbol HAVING n >= 3
   ...>   ORDER BY avg_r DESC;
```

Every column except `setup_id`, `symbol`, `timestamp`, `status` is JSON,
so `json_extract` is your friend. Note: `order` is a SQL reserved
keyword, use `"order"` (with quotes) when querying that column.

## Running the smoke tests

```bash
python tests/test_pipeline.py
```

Exercises every layer plus the orchestrator end-to-end on a synthetic
series. Catches refactor breakage in under a second.

## Customizing for your strategy

- **Change a threshold**: edit `config.yaml` and rerun. No code changes.
- **Add a new feature to the regression**: append to `regression.features`
  in config, update `feature_builder.py` to emit the new key. Retrain.
- **Add a new candlestick pattern**: add a `_is_xxx` helper to
  `entry_layer.py` and reference it in `entry_trigger_check`.
- **Swap Ridge for gradient boosting**: keep the `RegressionGatekeeper`
  class API identical (`train`, `predict`, `save`, `load`) and replace
  the underlying sklearn estimator.
- **Add a new bias indicator**: extend `bias_layer.py`. Secondaries can
  disagree but cannot flip the primary VWAP signal.
- **Tune the cost model for different brokers**: edit the `costs` block
  in `config.yaml`. Slippage assumptions may vary by liquidity tier.

## What each layer expects and returns

| Layer | Input | Output keys |
|-------|-------|-------------|
| Scanner | symbol, history df | in_watchlist, reason |
| EOD entry block | timestamp, portfolio | passes through or blocks |
| Portfolio check | symbol, portfolio | open_position, manage path action |
| Filter | history df | atr, adx, tradable |
| Bias | history df | direction, strength |
| Entry Stage 1 | history df, direction | confirmations count, passes |
| Entry Stage 1.5 (exhaustion) | state output | passes, reason |
| Entry Stage 2 (trigger) | history df, direction | pattern, valid |
| Feature builder | log row up to this point | flat dict of model inputs |
| Regression | features | y_hat, distance, trained |
| Confidence gate | regression output | band |
| Claude MCP | full log row | action, reasoning |
| Risk sizing | history df, direction | side, entry, stop, target, size, tradable |
| Cost adjustment | trade close info | realized_r_gross, realized_r_net, cost_in_r |

Every one of these writes to the decision log spine via
`DecisionLog.write(setup_id, layer, payload)`.

## Strategy work: what the simulator is for

Once the simulator is verified clean, the actual trading edge has to
come from your strategy. The simulator's job is to honestly evaluate
hypotheses, not generate them.

A reasonable strategy research loop:

1. **Export closed trades** to CSV from the decision log.
2. **Look at the actual charts** of your best and worst trades. Patterns
   emerge that no SQL query reveals.
3. **Form testable hypotheses** ("morning trades outperform afternoon",
   "low-ADX setups are noise", "certain symbols don't trade well at
   15-min").
4. **Test each hypothesis with a SQL aggregation first** before changing
   any code. If the hypothesis isn't supported by the data, don't bother
   coding the filter.
5. **Add one filter at a time** to the relevant layer, re-run, compare.
6. **Stop when you have a positive net mean R** sustained across re-runs
   on out-of-sample data.

If after several weeks of work the strategy can't reach positive net
mean R, the right move is usually to pivot timeframe (daily bars) or
instrument (options) rather than keep tuning. Some strategy archetypes
simply don't work for retail capital at intraday frequencies.

## Capital and broker assumptions

The default config assumes:

- **Real capital**: ₹20,000
- **Buying power**: ₹80,000 (4× MIS leverage on Indian equities)
- **Per-trade risk**: ₹200 (1% of real capital)
- **Broker**: discount broker with ~₹20/order brokerage and standard
  STT/exchange/GST/stamp charges

Adjust the `risk_sizing` and `costs` blocks in `config.yaml` if your
real setup differs.

## License and disclaimer

Reference implementation only. Not financial advice. The numbers in
this README are from a specific backtest configuration on specific data
and should not be interpreted as a forecast or recommendation. Run
extensive paper trading before risking real capital. Past backtest
performance does not predict future live performance, especially at
retail capital levels where costs are a meaningful fraction of risk.