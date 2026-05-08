"""
Entry layer. Stage 1 (state check) and Stage 2 (trigger check).

Stage 1 counts confirmations in the bias direction across:
  - momentum (RSI)
  - volume (vs recent average)
  - expansion (current ATR vs ATR n bars ago)
  - levels (proximity to recent swing high/low)

If the count meets the minimum, Stage 2 looks for a trigger pattern on the
current bar in the bias direction at a level. Patterns supported:
  - bullish / bearish engulfing
  - hammer / shooting star
"""
from __future__ import annotations

import pandas as pd


# ---------- Stage 1: state ---------------------------------------------------

def _check_momentum(rsi_val: float, direction: str, cfg: dict) -> bool:
    e = cfg["entry_layer"]
    if direction == "long":
        return rsi_val >= e["rsi_long_min"]
    if direction == "short":
        return rsi_val <= e["rsi_short_max"]
    return False


def _check_volume(volume_ratio: float, cfg: dict) -> bool:
    return volume_ratio >= cfg["entry_layer"]["volume_ratio_min"]


def _check_expansion(expansion_ratio: float, cfg: dict) -> bool:
    return expansion_ratio >= cfg["entry_layer"]["expansion_ratio_min"]


def _check_level(history: pd.DataFrame, direction: str, cfg: dict) -> tuple[bool, float]:
    """
    Distance to the nearest relevant swing level, measured in ATRs.
    Long bias -> we want price near a recent swing high (breakout setup) or
                 near a recent swing low (pullback to support).
    Short bias -> mirror.
    Returns (within_proximity, distance_in_atr).
    """
    e = cfg["entry_layer"]
    look = e["level_lookback"]
    proximity_atr = e["level_proximity_atr"]

    last = history.iloc[-1]
    atr_val = float(last["atr"])
    if atr_val <= 0:
        return False, float("inf")

    window = history.iloc[-look:]
    swing_high = float(window["high"].max())
    swing_low = float(window["low"].min())
    close = float(last["close"])

    if direction == "long":
        # closer of: pullback to support (swing_low) or breakout from resistance (swing_high)
        d = min(abs(close - swing_low), abs(close - swing_high)) / atr_val
    else:
        d = min(abs(close - swing_high), abs(close - swing_low)) / atr_val

    return d <= proximity_atr, float(d)


def entry_state_check(history: pd.DataFrame, direction: str, cfg: dict) -> dict:
    last = history.iloc[-1]
    rsi_val = float(last["rsi"]) if pd.notna(last["rsi"]) else 50.0
    vol_ratio = float(last["volume_ratio"]) if pd.notna(last["volume_ratio"]) else 0.0
    exp_ratio = float(last["expansion_ratio"]) if pd.notna(last["expansion_ratio"]) else 0.0

    momentum_ok = _check_momentum(rsi_val, direction, cfg)
    volume_ok = _check_volume(vol_ratio, cfg)
    expansion_ok = _check_expansion(exp_ratio, cfg)
    level_ok, level_dist = _check_level(history, direction, cfg)

    confirmations = sum([momentum_ok, volume_ok, expansion_ok, level_ok])
    minimum = cfg["entry_layer"]["min_state_confirmations"]
    passes = confirmations >= minimum

    return {
        "rsi": rsi_val,
        "volume_ratio": vol_ratio,
        "expansion_ratio": exp_ratio,
        "level_distance_atr": level_dist,
        "momentum_ok": momentum_ok,
        "volume_ok": volume_ok,
        "expansion_ok": expansion_ok,
        "level_ok": level_ok,
        "confirmations": int(confirmations),
        "minimum_required": int(minimum),
        "passes": bool(passes),
    }


# ---------- Stage 2: trigger -------------------------------------------------

def _is_bullish_engulfing(prev: pd.Series, curr: pd.Series, body_ratio: float) -> bool:
    prev_red = prev["close"] < prev["open"]
    curr_green = curr["close"] > curr["open"]
    body = abs(curr["close"] - curr["open"])
    rng = curr["high"] - curr["low"]
    if rng <= 0:
        return False
    big_body = body >= body_ratio * rng
    engulf = curr["close"] >= prev["open"] and curr["open"] <= prev["close"]
    return prev_red and curr_green and big_body and engulf


def _is_bearish_engulfing(prev: pd.Series, curr: pd.Series, body_ratio: float) -> bool:
    prev_green = prev["close"] > prev["open"]
    curr_red = curr["close"] < curr["open"]
    body = abs(curr["close"] - curr["open"])
    rng = curr["high"] - curr["low"]
    if rng <= 0:
        return False
    big_body = body >= body_ratio * rng
    engulf = curr["close"] <= prev["open"] and curr["open"] >= prev["close"]
    return prev_green and curr_red and big_body and engulf


def _is_hammer(curr: pd.Series, wick_ratio: float) -> bool:
    body = abs(curr["close"] - curr["open"])
    lower_wick = min(curr["open"], curr["close"]) - curr["low"]
    upper_wick = curr["high"] - max(curr["open"], curr["close"])
    if body <= 0:
        return False
    return lower_wick >= wick_ratio * body and upper_wick <= body


def _is_shooting_star(curr: pd.Series, wick_ratio: float) -> bool:
    body = abs(curr["close"] - curr["open"])
    lower_wick = min(curr["open"], curr["close"]) - curr["low"]
    upper_wick = curr["high"] - max(curr["open"], curr["close"])
    if body <= 0:
        return False
    return upper_wick >= wick_ratio * body and lower_wick <= body


def entry_trigger_check(history: pd.DataFrame, direction: str, cfg: dict) -> dict:
    e = cfg["entry_layer"]
    body_ratio = e["pattern_body_ratio"]
    wick_ratio = e["wick_ratio_min"]

    if len(history) < 2:
        return {"valid": False, "pattern": None, "reason": "insufficient_history"}

    prev = history.iloc[-2]
    curr = history.iloc[-1]

    pattern = None
    strength = 0.0

    if direction == "long":
        if _is_bullish_engulfing(prev, curr, body_ratio):
            pattern = "bullish_engulfing"
            strength = 1.0
        elif _is_hammer(curr, wick_ratio):
            pattern = "hammer"
            strength = 0.7
    elif direction == "short":
        if _is_bearish_engulfing(prev, curr, body_ratio):
            pattern = "bearish_engulfing"
            strength = 1.0
        elif _is_shooting_star(curr, wick_ratio):
            pattern = "shooting_star"
            strength = 0.7

    valid = pattern is not None

    return {
        "valid": bool(valid),
        "pattern": pattern,
        "pattern_strength": float(strength),
        "direction": direction,
    }
