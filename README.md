# Trading System with Regression Gatekeeper + Claude MCP Reasoning

A complete reference implementation of the architecture from the flowchart:
scanner → portfolio check → filter (ATR/ADX) → bias (VWAP / MA / MACD) →
entry layer (Stage 1 state + Stage 2 trigger) → feature vector → regression
model → confidence gate → Claude MCP for borderline cases → risk sizing →
order. Every layer writes to a decision log spine.

## Project layout

```
trading_system/
├── config.yaml                     # all thresholds and tunables
├── requirements.txt
├── data/                           # OHLCV csvs and the SQLite log
├── models/                         # saved regression model + scaler
├── src/
│   ├── data/
│   │   ├── data_loader.py          # OHLCV CSV loader
│   │   └── indicators.py           # ATR, ADX, VWAP, EMA, MACD, RSI
│   ├── layers/
│   │   ├── scanner.py              # universe -> watchlist
│   │   ├── portfolio.py            # Position + manage path
│   │   ├── filter_layer.py         # ATR / ADX gate
│   │   ├── bias_layer.py           # VWAP primary, MA + MACD confirm
│   │   ├── entry_layer.py          # Stage 1 state + Stage 2 trigger
│   │   └── risk_sizing.py          # 2x ATR stop, 3x ATR target
│   ├── ml/
│   │   ├── feature_builder.py      # log spine -> feature vector
│   │   ├── regression_model.py     # Ridge gatekeeper
│   │   └── confidence_gate.py      # far+/borderline/far- routing
│   ├── reasoning/
│   │   └── claude_mcp.py           # Claude API + heuristic fallback
│   ├── logs/
│   │   └── decision_log.py         # SQLite log spine
│   └── orchestrator.py             # ties every layer together
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
- anthropic — optional, only if you want real Claude calls instead of the
  heuristic fallback
- matplotlib, plotly — optional, for charting your own analysis

## Step 2: generate synthetic data

The system ships with a generator that produces multi-symbol OHLCV bars with
trending and choppy regimes, so the pipeline actually fires some setups.

```bash
python scripts/generate_synthetic_data.py \
    --symbols AAA BBB CCC \
    --bars 2000 \
    --out data/synthetic_ohlcv.csv
```

To use real data, write a CSV with columns `timestamp, symbol, open, high,
low, close, volume` and point `config.yaml -> backtest.data_path` at it.

## Step 3: run the first backtest (cold start)

The first time you run the system, no regression model exists yet. The
confidence gate detects this and routes EVERY setup to the borderline branch
so the heuristic Claude reasoner (or the real Claude API if enabled) decides
each one. This is intentional — it bootstraps the training set.

```bash
python scripts/run_backtest.py --config config.yaml --reset-log
```

You will see a summary like:

```
=== Backtest summary ===
                trades: 73
                 holds: 1102
              no_trade: 412
                  wait: 1908
       skipped_scanner: 0
   total_setups_logged: 3495

=== Closed trades: 70 ===
  mean R    : +0.214
  win rate  : 38.6%
  best      : +3.000
  worst     : -1.000
  median    : -0.214
```

The decision log now contains every setup, including layer-by-layer state, the
Claude verdict, the regression prediction (an untrained 0.0 at this stage),
and the realized R for closed trades.

## Step 4: train the regression gatekeeper

```bash
python scripts/train_regression.py --config config.yaml
```

This pulls every closed setup with both a feature vector and a realized R from
the decision log, fits a Ridge model on standard-scaled features, and saves
the model and scaler to `models/`.

You will see the per-feature coefficients and a train R^2. Don't expect a
high R^2 — financial data has low signal — but the coefficients should make
directional sense (positive for ADX and confirmation count, etc).

## Step 5: run the second backtest (with the trained gate)

```bash
python scripts/run_backtest.py --config config.yaml --reset-log
```

Now the regression scores every setup. Far-positive scores skip Claude and go
straight to risk sizing; far-negative scores get rejected without Claude;
only the borderline middle band escalates to Claude. Compare the trade count
and per-setup latency against step 3 — the borderline-only escalation is the
whole point of the gatekeeper.

## Step 6: enable real Claude reasoning (optional)

By default the Claude MCP layer uses a deterministic heuristic so the system
runs offline. To use the real Claude API:

1. Get an API key from https://console.anthropic.com.
2. Export it:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   ```
3. In `config.yaml` set `claude_mcp.enabled: true` (this is the default).
4. Run a backtest. Borderline setups will hit the API. Far+ and far- setups
   still skip Claude entirely so you don't burn tokens on confident decisions.

If the key is missing or the `anthropic` package is not installed, the
reasoner silently falls back to the heuristic — no errors, just a `mode:
heuristic` field in the log so you can tell the difference.

## Retraining loop

The architecture is self-improving:

1. Live trades flow into the decision log.
2. When trades close, `manage_path` writes the realized R into the outcome
   column of the original setup's row.
3. Periodically (daily, weekly) re-run `scripts/train_regression.py`. The
   model now has more data, including the borderline cases that Claude
   resolved.
4. The far-positive and far-negative bands widen as the model gets sharper,
   so fewer setups need Claude over time.

You can monitor this by counting `claude.skipped == True` rows vs
`claude.skipped == False` rows in the log. The ratio should rise over time
if retraining is working.

## Inspecting the log

Open the SQLite log directly:

```bash
sqlite3 data/decision_log.sqlite
sqlite> .schema setups
sqlite> SELECT status, COUNT(*) FROM setups GROUP BY status;
sqlite> SELECT setup_id, json_extract(regression, '$.y_hat') AS yhat,
   ...>        json_extract(outcome,    '$.realized_r') AS realized
   ...>   FROM setups WHERE status = 'closed' ORDER BY timestamp DESC LIMIT 20;
```

Every column except `setup_id`, `symbol`, `timestamp`, `status` is JSON, so
`json_extract` is your friend.

## Running the smoke tests

```bash
python tests/test_pipeline.py
```

This exercises every layer plus the orchestrator end-to-end on a 400-bar
synthetic series. If anything breaks during a refactor, this catches it in
under a second.

## Customizing for your strategy

- **Change a threshold**: edit `config.yaml` and rerun. No code changes
  needed.
- **Add a new feature to the regression**: append it to
  `regression.features` in config, then update `feature_builder.py` to emit
  the new key. Retrain.
- **Add a new candlestick pattern**: add a `_is_xxx` helper to
  `entry_layer.py` and reference it in `entry_trigger_check`.
- **Swap Ridge for gradient boosting**: keep the `RegressionGatekeeper`
  class API identical (`train`, `predict`, `save`, `load`) and replace the
  underlying sklearn estimator. The orchestrator will not need any changes.
- **Add a new bias indicator**: extend `bias_layer.py` and remember the
  rule from the diagram — secondaries can disagree but cannot flip the
  primary VWAP signal.

## What each layer expects and returns

| Layer            | Input                                     | Output keys                                           |
|------------------|-------------------------------------------|-------------------------------------------------------|
| Scanner          | symbol, history df                        | in_watchlist, reason                                  |
| Portfolio check  | symbol, portfolio                         | open_position, manage path action                     |
| Filter           | history df                                | atr, adx, tradable                                    |
| Bias             | history df                                | direction (long/short/neutral), strength              |
| Entry Stage 1    | history df, direction                     | confirmations count, passes                           |
| Entry Stage 2    | history df, direction                     | pattern, valid                                        |
| Feature builder  | log row up to this point                  | flat dict of model inputs                             |
| Regression       | features                                  | y_hat, distance, trained                              |
| Confidence gate  | regression output                         | band: far_positive / borderline / far_negative        |
| Claude MCP       | full log row                              | action: take/skip/wait, reasoning                     |
| Risk sizing      | history df, direction                     | side, entry, stop, target, position_size              |

Every one of these writes to the decision log spine via
`DecisionLog.write(setup_id, layer, payload)`.

## License and disclaimer

Reference implementation only. Not financial advice. Run paper-trading
extensively before risking real capital.
