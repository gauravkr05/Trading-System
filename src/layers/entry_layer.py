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

Stage 5 observability fields (NO filtering, just measurement):
  - distance_from_ma_atr        : signed distance of close from MA in ATRs
  - bars_since_close_crossed_ma : trend-maturity proxy via MA side-flip
  - bars_since_macd_cross       : momentum-shift recency
  - entry_bar_range_atr         : width of the current bar in ATRs
  - pullback_depth_atr          : distance from the recent swing extreme
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


# ---------- Stage 5 observability helpers -----------------------------------

def _distance_from_ma_atr(history: pd.DataFrame, direction: str) -> float:
    """
    Signed distance of close from MA, in ATRs.
    Positive = price extended in the bias direction.
    Negative = price on the wrong side of MA (against the bias).
    Returns 0.0 if MA or ATR are missing/invalid.
    """
    last = history.iloc[-1]
    ma_val = last.get("ma")
    atr_val = last.get("atr")
    close = last.get("close")

    if ma_val is None or atr_val is None or close is None:
        return 0.0
    if pd.isna(ma_val) or pd.isna(atr_val) or pd.isna(close):
        return 0.0
    if float(atr_val) <= 0:
        return 0.0

    raw = (float(close) - float(ma_val)) / float(atr_val)
    if direction == "short":
        raw = -raw
    return float(raw)


def _bars_since_close_crossed_ma(history: pd.DataFrame, direction: str, lookback: int) -> int:
    """
    How many consecutive bars (counting back from current) has close stayed on
    the bias side of MA? Capped at `lookback`. Returns lookback if no cross
    found within the window (i.e. trend has been mature for at least that long).
    Returns 0 if current bar is on the wrong side of MA.
    """
    if len(history) == 0:
        return 0
    window = history.iloc[-lookback:] if len(history) >= lookback else history
    closes = window["close"].to_numpy()
    mas = window["ma"].to_numpy()

    # Walk backwards from the most recent bar.
    count = 0
    for i in range(len(closes) - 1, -1, -1):
        c = closes[i]
        m = mas[i]
        if pd.isna(c) or pd.isna(m):
            break
        on_bias_side = (c > m) if direction == "long" else (c < m)
        if not on_bias_side:
            break
        count += 1
    return int(count)


def _bars_since_macd_cross(history: pd.DataFrame, direction: str, lookback: int) -> int:
    """
    How many bars since macd line crossed signal line in the bias direction?
    For long: macd > signal. For short: macd < signal.
    Returns lookback if no flip within window. Returns 0 if currently against bias.
    """
    if len(history) == 0:
        return 0
    window = history.iloc[-lookback:] if len(history) >= lookback else history
    macd_vals = window["macd"].to_numpy()
    sig_vals = window["macd_signal"].to_numpy()

    count = 0
    for i in range(len(macd_vals) - 1, -1, -1):
        m = macd_vals[i]
        s = sig_vals[i]
        if pd.isna(m) or pd.isna(s):
            break
        on_bias_side = (m > s) if direction == "long" else (m < s)
        if not on_bias_side:
            break
        count += 1
    return int(count)


def _entry_bar_range_atr(history: pd.DataFrame) -> float:
    """Width of the current bar in ATRs. Wide bars often mean we're chasing."""
    last = history.iloc[-1]
    high = last.get("high")
    low = last.get("low")
    atr_val = last.get("atr")
    if high is None or low is None or atr_val is None:
        return 0.0
    if pd.isna(high) or pd.isna(low) or pd.isna(atr_val):
        return 0.0
    if float(atr_val) <= 0:
        return 0.0
    return float((float(high) - float(low)) / float(atr_val))


def _pullback_depth_atr(history: pd.DataFrame, direction: str, lookback: int) -> float:
    """
    For long bias: distance from recent swing high in ATRs.
                   0 = entering at the high (chasing).
                   Higher = entering on a pullback.
    For short bias: distance from recent swing low in ATRs (mirrored).
    """
    last = history.iloc[-1]
    atr_val = last.get("atr")
    close = last.get("close")
    if atr_val is None or close is None:
        return 0.0
    if pd.isna(atr_val) or pd.isna(close):
        return 0.0
    if float(atr_val) <= 0:
        return 0.0

    window = history.iloc[-lookback:] if len(history) >= lookback else history
    if direction == "long":
        swing_high = float(window["high"].max())
        return float((swing_high - float(close)) / float(atr_val))
    else:
        swing_low = float(window["low"].min())
        return float((float(close) - swing_low) / float(atr_val))


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

    # ----- Stage 5 observability (no filtering, just measurement) -----
    obs_cfg = cfg["entry_layer"].get("observability", {})
    bias_lookback = int(obs_cfg.get("bias_continuity_lookback", 30))
    pb_lookback = int(obs_cfg.get("pullback_lookback", cfg["entry_layer"]["level_lookback"]))

    distance_from_ma_atr = _distance_from_ma_atr(history, direction)
    bars_since_ma_cross = _bars_since_close_crossed_ma(history, direction, bias_lookback)
    bars_since_macd_cross_v = _bars_since_macd_cross(history, direction, bias_lookback)
    entry_bar_range_atr = _entry_bar_range_atr(history)
    pullback_depth_atr = _pullback_depth_atr(history, direction, pb_lookback)

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
        # Stage 5 observability fields:
        "distance_from_ma_atr": float(distance_from_ma_atr),
        "bars_since_close_crossed_ma": int(bars_since_ma_cross),
        "bars_since_macd_cross": int(bars_since_macd_cross_v),
        "entry_bar_range_atr": float(entry_bar_range_atr),
        "pullback_depth_atr": float(pullback_depth_atr),
    }


# ---------- Stage 1.5: exhaustion check -------------------------------------
# Built from Stage 5 diagnostic data:
#   - Worst-30 trades avg distance_from_ma_atr = 1.88, best-30 = 1.57
#   - Worst-30 pullback_depth_atr = 0.55, best-30 = 0.78
# Late entries (close too far from MA) and chasing entries (close too near
# the recent extreme) underperform. This is a soft filter — calibrated from
# data, not theory — and lives between the state check and trigger check so
# we reject cleanly before pattern matching.

def entry_exhaustion_check(state: dict, cfg: dict) -> dict:
    """
    Reject entries that are too far from MA (chasing the trend) or too close
    to the recent swing extreme (chasing the breakout). Both thresholds come
    from config; both default to permissive values that only cut the tails.

    Inputs:
        state: dict returned by entry_state_check. Must contain
               distance_from_ma_atr and pullback_depth_atr.
        cfg:   full config dict.

    Returns dict with:
        passes:  bool, True if entry survives all exhaustion checks
        reason:  str, name of the failed check or "ok"
        distance_from_ma_atr: echoed for log convenience
        pullback_depth_atr:   echoed for log convenience
        max_distance_from_ma_atr: threshold used
        min_pullback_depth_atr:   threshold used
    """
    ex_cfg = cfg["entry_layer"].get("exhaustion", {})
    enabled = bool(ex_cfg.get("enabled", True))
    max_dist = float(ex_cfg.get("max_distance_from_ma_atr", 1.8))
    min_pullback = float(ex_cfg.get("min_pullback_depth_atr", 0.4))

    dist = float(state.get("distance_from_ma_atr", 0.0))
    pullback = float(state.get("pullback_depth_atr", 0.0))

    if not enabled:
        return {
            "passes": True,
            "reason": "disabled",
            "distance_from_ma_atr": dist,
            "pullback_depth_atr": pullback,
            "max_distance_from_ma_atr": max_dist,
            "min_pullback_depth_atr": min_pullback,
        }

    # Check 1: too far from MA (overextended trend chase).
    # Note: distance_from_ma_atr is signed by direction. Negative means price
    # is on the wrong side of MA; we don't reject those here because the bias
    # layer already filters those out. Only positive overextension matters.
    if dist > max_dist:
        return {
            "passes": False,
            "reason": "overextended_from_ma",
            "distance_from_ma_atr": dist,
            "pullback_depth_atr": pullback,
            "max_distance_from_ma_atr": max_dist,
            "min_pullback_depth_atr": min_pullback,
        }

    # Check 2: too close to swing extreme (chasing breakout, no pullback).
    if pullback < min_pullback:
        return {
            "passes": False,
            "reason": "chasing_extreme_no_pullback",
            "distance_from_ma_atr": dist,
            "pullback_depth_atr": pullback,
            "max_distance_from_ma_atr": max_dist,
            "min_pullback_depth_atr": min_pullback,
        }

    return {
        "passes": True,
        "reason": "ok",
        "distance_from_ma_atr": dist,
        "pullback_depth_atr": pullback,
        "max_distance_from_ma_atr": max_dist,
        "min_pullback_depth_atr": min_pullback,
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