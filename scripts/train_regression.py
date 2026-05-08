"""
Train the regression gatekeeper.

Workflow:
  1. Read the decision log spine.
  2. Pull every closed setup that has both a feature vector and an outcome.
  3. Build (X, y) where y is realized R-multiple.
  4. Fit a Ridge regression with standard-scaled features.
  5. Save model + scaler to disk.

The first time you run this you typically need to run a backtest first
(scripts/run_backtest.py) so the log has rows. Subsequent retrains use the
log that the live system has been writing to.

Usage:
  python scripts/train_regression.py --config config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logs.decision_log import DecisionLog
from src.ml.regression_model import RegressionGatekeeper


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_training_set(log: DecisionLog, feature_names: list[str]):
    rows = log.closed_setups_with_features()
    X_list, y_list, ids = [], [], []
    for s in rows:
        feats = s["features"]
        outcome = s["outcome"]
        if not isinstance(feats, dict) or not isinstance(outcome, dict):
            continue
        if "realized_r" not in outcome:
            continue
        X_list.append([float(feats.get(n, 0.0)) for n in feature_names])
        y_list.append(float(outcome["realized_r"]))
        ids.append(s["setup_id"])
    if not X_list:
        return None, None, []
    return np.array(X_list), np.array(y_list), ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="ridge regularization strength")
    args = parser.parse_args()

    cfg = load_config(args.config)
    feature_names = cfg["regression"]["features"]

    log = DecisionLog(cfg["decision_log"]["db_path"])
    X, y, ids = build_training_set(log, feature_names)

    if X is None or len(X) < 10:
        print(f"Not enough labeled setups in the log to train (have "
              f"{0 if X is None else len(X)}). Run a backtest first.")
        return

    print(f"Training on {len(X)} labeled setups with {len(feature_names)} features.")
    print(f"Label stats: mean R = {y.mean():+.3f}, std = {y.std():.3f}, "
          f"win rate = {(y > 0).mean():.1%}")

    gk = RegressionGatekeeper(feature_names)
    metrics = gk.train(X, y, alpha=args.alpha)

    print("\n=== Training metrics ===")
    print(f"  n_samples : {metrics['n_samples']}")
    print(f"  n_features: {metrics['n_features']}")
    print(f"  train_rmse: {metrics['train_rmse']:.4f}")
    print(f"  train_r2  : {metrics['train_r2']:.4f}")
    print(f"  intercept : {metrics['intercept']:+.4f}")
    print("  coefs     :")
    for name, c in metrics["coef"].items():
        print(f"    {name:>22} = {c:+.4f}")

    gk.save(cfg["regression"]["model_path"], cfg["regression"]["scaler_path"])
    print(f"\nSaved model to {cfg['regression']['model_path']}")
    print(f"Saved scaler to {cfg['regression']['scaler_path']}")


if __name__ == "__main__":
    main()
