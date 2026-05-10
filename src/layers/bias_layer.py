"""
Bias layer. Decides long / short / neutral.

Rules from the architecture diagram:
  - VWAP is the PRIMARY signal. Price above VWAP -> long candidate;
    below VWAP -> short candidate.
  - MA and MACD are CONFIRMERS. They can confirm the VWAP bias or stay quiet.
  - Secondaries can NEVER flip the primary. If VWAP says long but MA/MACD
    say short, the result is `neutral`, NOT a flip to short.

Strength is the count of confirmers that agree (0, 1, or 2).
"""

from __future__ import annotations

import pandas as pd


def bias_layer(history: pd.DataFrame, cfg: dict) -> dict:
    last = history.iloc[-1]
    close = float(last["close"])
    vwap_v = float(last["vwap"]) if pd.notna(last["vwap"]) else close
    ma_v = float(last["ma"]) if pd.notna(last["ma"]) else close
    macd_hist = float(last["macd_hist"]) if pd.notna(last["macd_hist"]) else 0.0

    # primary
    primary = "long" if close > vwap_v else "short" if close < vwap_v else "neutral"

    # confirmers
    ma_long = close > ma_v
    macd_long = macd_hist > 0
    ma_short = close < ma_v
    macd_short = macd_hist < 0

    if primary == "long":
        agree = int(ma_long) + int(macd_long)
        disagree = int(ma_short) + int(macd_short)
    elif primary == "short":
        agree = int(ma_short) + int(macd_short)
        disagree = int(ma_long) + int(macd_long)
    else:
        agree = 0
        disagree = 0

    # if both confirmers actively disagree, the bias becomes neutral (cannot be flipped).
    direction = primary if disagree < 2 else "neutral"

    strength = agree  # 0, 1, or 2

    return {
        "direction": direction,
        "primary": primary,
        "strength": strength,
        "agree": agree,
        "disagree": disagree,
        "close": close,
        "vwap": vwap_v,
        "ma": ma_v,
        "macd_hist": macd_hist,
    }
