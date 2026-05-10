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
    raw_size = risk_dollars / risk_per_share if risk_per_share > 0 else 0.0

    # Round DOWN to whole shares (you can't buy 4.7 shares)
    position_size = float(int(raw_size))

    # Cash availability check: can the account actually afford this position?
    notional = position_size * entry
    if notional > r["account_equity"]:
        # Cap at affordable size, again whole shares
        position_size = float(int(r["account_equity"] / entry))

    # Effective risk after rounding (this is the REAL risk, not the intended)
    effective_risk = position_size * risk_per_share

    # Short-selling filter: retail equity cannot short overnight.
    # If you intend to hold across the EOD, this trade is invalid for cash equity.
    # Set allow_shorts=false in config if you only trade cash equity.
    allow_shorts = r.get("allow_shorts", True)
    tradable = True
    skip_reason = None
    if side == "SELL" and not allow_shorts:
        tradable = False
        skip_reason = "shorts_disabled_cash_equity"
    if position_size < 1:
        tradable = False
        skip_reason = "position_size_below_one_share"

    return {
        "side": side,
        "entry": entry,
        "stop": float(stop),
        "target": float(target),
        "atr": atr_val,
        "risk_per_share": float(risk_per_share),
        "risk_dollars": float(risk_dollars),
        "raw_size": float(raw_size),
        "position_size": float(position_size),
        "effective_risk": float(effective_risk),
        "notional": float(position_size * entry),
        "reward_to_risk": float(r["target_atr_mult"] / r["stop_atr_mult"]),
        "tradable": tradable,
        "skip_reason": skip_reason,
    }

def apply_costs_to_r(
    realized_r: float,
    entry_price: float,
    exit_price: float,
    position_size: float,
    risk_per_share: float,
    side: str,
    cfg: dict,
    exit_reason: str = "target", 
) -> dict:
    """
    Adjust a backtest realized_r for realistic Indian retail trading costs.

    Costs modeled:
      - Slippage on entry and exit (worsens both fills)
      - Flat round-trip brokerage (Zerodha/Upstox-style)
      - Combined STT + exchange + GST + stamp as a percentage of turnover

    Returns gross R, net R, and a breakdown so both numbers are preserved
    in the log spine.
    """
    c = cfg.get("costs", {})
    slippage_entry_pct = c.get("slippage_entry_pct", 0.0010)  # 0.10% on entries
    slippage_target_pct = c.get("slippage_target_pct", 0.0010)  # 0.10% on targets
    slippage_stop_pct = c.get("slippage_stop_pct", 0.0025)  # 0.25% on stops (worse)
    brokerage_rt = c.get("brokerage_round_trip", 40.0)
    other_pct    = c.get("other_charges_pct", 0.0008)

    # Pick exit slippage based on which exit type happened.
    # exit_reason is passed in via kwargs (see Edit 3 in orchestrator).
    exit_slip = slippage_stop_pct if exit_reason in ("stop", "trail_stop") else slippage_target_pct

    # Apply slippage: entry worsens, exit worsens (sign depends on side)
    if side in ("long", "BUY"):
        real_entry = entry_price * (1 + slippage_entry_pct)
        real_exit  = exit_price  * (1 - exit_slip)
        gross_pnl  = (real_exit - real_entry) * position_size
    else:  # short / SELL
        real_entry = entry_price * (1 - slippage_entry_pct)
        real_exit  = exit_price  * (1 + exit_slip)
        gross_pnl  = (real_entry - real_exit) * position_size

    # Fixed + percentage charges
    turnover = (real_entry + real_exit) * position_size
    charges  = brokerage_rt + (turnover * other_pct)
    net_pnl  = gross_pnl - charges

    # Convert to R using ORIGINAL risk (not slipped risk)
    risk_amount = risk_per_share * position_size
    if risk_amount <= 0:
        return {
            "realized_r_gross": float(realized_r),
            "realized_r_net": float(realized_r),
            "cost_in_r": 0.0,
            "charges_inr": 0.0,
            "slippage_inr": 0.0,
        }

    realized_r_net = net_pnl / risk_amount
    cost_in_r = realized_r - realized_r_net
    slippage_inr = (abs(real_entry - entry_price) +
                    abs(real_exit  - exit_price)) * position_size

    return {
        "realized_r_gross": float(realized_r),
        "realized_r_net": float(realized_r_net),
        "cost_in_r": float(cost_in_r),
        "charges_inr": float(charges),
        "slippage_inr": float(slippage_inr),
    }