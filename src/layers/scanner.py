"""
Scanner layer. Universe -> watchlist.

Filters the full symbol universe down to a manageable watchlist using simple
liquidity and price-range gates. Logs the result so we can later see which
symbols were considered.
"""
from __future__ import annotations

import pandas as pd


def in_watchlist(symbol: str, history: pd.DataFrame, cfg: dict) -> dict:
    """
    Decide whether `symbol` belongs in today's watchlist.
    `history` is the symbol's recent OHLCV (with indicators already added).
    """
    s = cfg["scanner"]

    if len(history) < 20:
        return {"symbol": symbol, "in_watchlist": False, "reason": "insufficient_history"}

    last = history.iloc[-1]
    avg_volume = history["volume"].tail(20).mean()
    price = last["close"]

    if avg_volume < s["min_avg_volume"]:
        return {"symbol": symbol, "in_watchlist": False, "reason": "low_volume",
                "avg_volume": float(avg_volume)}
    if price < s["min_price"] or price > s["max_price"]:
        return {"symbol": symbol, "in_watchlist": False, "reason": "price_out_of_range",
                "price": float(price)}

    return {
        "symbol": symbol,
        "in_watchlist": True,
        "avg_volume": float(avg_volume),
        "price": float(price),
    }
