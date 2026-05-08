"""
Risk sizing layer.

Stop = entry +/- (stop_atr_mult * ATR)
Target = entry +/- (target_atr_mult * ATR)
Position size = (account_equity * risk_per_trade_pct) / risk_per_share
"""
from __future__ import annotations

import pandas as pd


def risk_sizing(history: pd.DataFrame, direction: str, cfg: dict) -> dict:
    r = cfg["risk_sizing"]
    last = history.iloc[-1]
    entry = float(last["close"])
    atr_val = float(last["atr"])

    if direction == "long":
        stop = entry - r["stop_atr_mult"] * atr_val
        target = entry + r["target_atr_mult"] * atr_val
        side = "BUY"
    else:
        stop = entry + r["stop_atr_mult"] * atr_val
        target = entry - r["target_atr_mult"] * atr_val
        side = "SELL"

    risk_per_share = abs(entry - stop)
    risk_dollars = r["account_equity"] * r["risk_per_trade_pct"]
    position_size = risk_dollars / risk_per_share if risk_per_share > 0 else 0.0

    return {
        "side": side,
        "entry": entry,
        "stop": float(stop),
        "target": float(target),
        "atr": atr_val,
        "risk_per_share": float(risk_per_share),
        "risk_dollars": float(risk_dollars),
        "position_size": float(position_size),
        "reward_to_risk": float(r["target_atr_mult"] / r["stop_atr_mult"]),
    }
