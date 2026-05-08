#!/usr/bin/env bash
# End-to-end demo. Generates synthetic data, runs the cold-start backtest,
# trains the regression gatekeeper, then runs the trained backtest.
# Use this to verify your install in one go.
set -e

cd "$(dirname "$0")/.."

echo "=== 1. Generate synthetic data ==="
python scripts/generate_synthetic_data.py --bars 1500 --symbols AAA BBB CCC

echo ""
echo "=== 2. Cold-start backtest (no model -> every setup goes to Claude/heuristic) ==="
python scripts/run_backtest.py --reset-log

echo ""
echo "=== 3. Train the regression gatekeeper on the cold-start log ==="
python scripts/train_regression.py

echo ""
echo "=== 4. Trained backtest (regression gate active, only borderline reaches Claude) ==="
python scripts/run_backtest.py --reset-log

echo ""
echo "Done. Inspect data/decision_log.sqlite to see every layer's output per setup."
