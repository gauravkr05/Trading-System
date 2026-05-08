"""
Filter layer. ATR and ADX gate.

A symbol is tradable only when:
  - ATR is large enough that there is room for a 2x ATR stop and 3x ATR target
  - ADX is high enough that we are in a trend (not chop)

If either gate fails the orchestrator drops the setup with a NO_TRADE.
"""
from __future__ import annotations

import pandas as pd


def filter_layer(history: pd.DataFrame, cfg: dict) -> dict:
    f = cfg["filter_layer"]
    last = history.iloc[-1]

    atr_val = float(last["atr"]) if pd.notna(last["atr"]) else 0.0
    adx_val = float(last["adx"]) if pd.notna(last["adx"]) else 0.0

    atr_ok = atr_val >= f["atr_min"]
    adx_min = f["adx_min"]
    adx_max = f.get("adx_max", 45.0) 
    adx_ok = adx_min <= adx_val <= adx_max
    # adx_ok = adx_val >= f["adx_min"]
    tradable = atr_ok and adx_ok

    return {
        "atr": atr_val,
        "adx": adx_val,
        "atr_ok": atr_ok,
        "adx_ok": adx_ok,
        "tradable": tradable,
    }
