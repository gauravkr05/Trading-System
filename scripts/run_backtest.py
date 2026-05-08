"""
Backtest runner. Walks every bar after the warmup period, calls the
orchestrator, simulates fills against future bars, and writes everything
to the decision log.

Usage:
  python scripts/run_backtest.py --config config.yaml

This is the top-level entry point. The 'live' path (real broker, real data
feed) would replace `iter_bars` with a streaming feed and remove the explicit
manage-path simulation that the manage_path layer already handles.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import load_ohlcv, split_by_symbol
from src.data.indicators import add_all_indicators
from src.layers.portfolio import Portfolio
from src.logs.decision_log import DecisionLog
from src.ml.regression_model import RegressionGatekeeper
from src.orchestrator import run_pipeline


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_backtest(cfg: dict) -> dict:
    bcfg = cfg["backtest"]
    warmup = bcfg["warmup_bars"]

    # --- load and prepare data ----------------------------------------------
    df = load_ohlcv(bcfg["data_path"])
    if bcfg.get("symbols"):
        df = df[df["symbol"].isin(bcfg["symbols"])].reset_index(drop=True)
    by_symbol = split_by_symbol(df)
    enriched = {sym: add_all_indicators(s, cfg) for sym, s in by_symbol.items()}

    # --- core objects -------------------------------------------------------
    log = DecisionLog(cfg["decision_log"]["db_path"])
    portfolio = Portfolio()
    model = RegressionGatekeeper.load(
        cfg["regression"]["model_path"],
        cfg["regression"]["scaler_path"],
    )
    if not model.feature_names:
        # no trained model on disk yet; use the feature names from config and
        # operate in 'untrained' mode (every setup goes to Claude/heuristic).
        model.feature_names = cfg["regression"]["features"]

    # --- step through bars in chronological order --------------------------
    # build a list of (timestamp, symbol, bar_index) so we can iterate in time order
    timeline = []
    for sym, sdf in enriched.items():
        for i in range(warmup, len(sdf)):
            timeline.append((sdf.iloc[i]["timestamp"], sym, i))
    timeline.sort(key=lambda x: x[0])

    n_trades = 0
    n_holds = 0
    n_no_trade = 0
    n_wait = 0
    n_skipped_scanner = 0

    for ts, sym, idx in timeline:
        history = enriched[sym].iloc[: idx + 1]
        result = run_pipeline(sym, history, portfolio, model, cfg, log)
        action = result.get("action")
        if action == "trade":
            n_trades += 1
        elif action == "hold":
            n_holds += 1
        elif action == "no_trade":
            n_no_trade += 1
        elif action == "wait":
            n_wait += 1
        elif action == "skip":
            n_skipped_scanner += 1

    return {
        "trades": n_trades,
        "holds": n_holds,
        "no_trade": n_no_trade,
        "wait": n_wait,
        "skipped_scanner": n_skipped_scanner,
        "total_setups_logged": len(log.all_setups()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--reset-log", action="store_true",
                        help="delete the decision log database before running")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.reset_log:
        db = cfg["decision_log"]["db_path"]
        if os.path.exists(db):
            os.remove(db)
            print(f"Removed existing log at {db}")

    summary = run_backtest(cfg)
    print("\n=== Backtest summary ===")
    for k, v in summary.items():
        print(f"  {k:>22}: {v}")

    # show the closed-trade R distribution if any
    log = DecisionLog(cfg["decision_log"]["db_path"])
    closed = [s for s in log.all_setups()
              if s.get("outcome") and isinstance(s["outcome"], dict)]
    if closed:
        rs = [s["outcome"]["realized_r"] for s in closed]
        rs_sorted = sorted(rs)
        print(f"\n=== Closed trades: {len(closed)} ===")
        print(f"  mean R    : {sum(rs)/len(rs):+.3f}")
        print(f"  win rate  : {sum(1 for r in rs if r > 0)/len(rs):.1%}")
        print(f"  best      : {max(rs):+.3f}")
        print(f"  worst     : {min(rs):+.3f}")
        print(f"  median    : {rs_sorted[len(rs_sorted)//2]:+.3f}")


if __name__ == "__main__":
    main()
