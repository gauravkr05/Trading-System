"""
Portfolio tracker + manage path logic.

Portfolio is in-memory for simplicity. In production you would back this with
a broker SDK or a positions table.

Manage path decides what to do with an already-open position on each new bar:
  - exit if stop hit
  - exit if target hit
  - hold otherwise
  - (extension point: add to winners)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# @dataclass
# class Position:
#     symbol: str
#     side: str          # 'long' or 'short'
#     entry_price: float
#     stop: float
#     target: float
#     size: float
#     bars_held: int = 0
#     entry_setup_id: str = ""

@dataclass
class Position:
    symbol: str
    side: str          # 'long' or 'short'
    entry_price: float
    stop: float
    target: float
    size: float
    bars_held: int = 0
    entry_setup_id: str = ""
    mfe_price: float = 0.0
    mae_price: float = 0.0
    initial_stop: float = 0.0       # original stop, never mutated (for R-unit math)
    highest_stop_step_r: float = 0.0  # highest R-level at which stop has been ratcheted

    def __post_init__(self):
        if self.mfe_price == 0.0:
            self.mfe_price = self.entry_price
        if self.mae_price == 0.0:
            self.mae_price = self.entry_price
        if self.initial_stop == 0.0:
            self.initial_stop = self.stop

@dataclass
class Portfolio:
    positions: dict = field(default_factory=dict)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def open(self, p: Position) -> None:
        self.positions[p.symbol] = p

    def close(self, symbol: str) -> Optional[Position]:
        return self.positions.pop(symbol, None)

    def get(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)


# def manage_path(position: Position, current_bar: dict) -> dict:
#     """
#     Decide hold / exit / add for an existing position.
#     Returns a dict describing the decision and any realized R-multiple if exited.
#     """
#     high = current_bar["high"]
#     low = current_bar["low"]
#     close = current_bar["close"]
#     position.bars_held += 1

#     risk_per_share = abs(position.entry_price - position.stop)
#     if risk_per_share <= 0:
#         return {"action": "hold", "bars_held": position.bars_held}

#     if position.side == "long":
#         if low <= position.stop:
#             r = (position.stop - position.entry_price) / risk_per_share
#             return {"action": "exit", "exit_reason": "stop", "exit_price": position.stop,
#                     "realized_r": float(r), "bars_held": position.bars_held}
#         if high >= position.target:
#             r = (position.target - position.entry_price) / risk_per_share
#             return {"action": "exit", "exit_reason": "target", "exit_price": position.target,
#                     "realized_r": float(r), "bars_held": position.bars_held}
#     else:  # short
#         if high >= position.stop:
#             r = (position.entry_price - position.stop) / risk_per_share
#             return {"action": "exit", "exit_reason": "stop", "exit_price": position.stop,
#                     "realized_r": float(r), "bars_held": position.bars_held}
#         if low <= position.target:
#             r = (position.entry_price - position.target) / risk_per_share
#             return {"action": "exit", "exit_reason": "target", "exit_price": position.target,
#                     "realized_r": float(r), "bars_held": position.bars_held}

#     unrealized = ((close - position.entry_price) / risk_per_share
#                   if position.side == "long"
#                   else (position.entry_price - close) / risk_per_share)
#     return {"action": "hold", "bars_held": position.bars_held,
#             "unrealized_r": float(unrealized)}

def manage_path(position: Position, current_bar: dict, cfg: dict | None = None) -> dict:
    """
    Decide hold / exit / add for an existing position.
    Tracks MFE/MAE and applies a stepped trailing stop.

    Trail rule (default, configurable):
      - At MFE >= 1.0R, stop moves to entry (breakeven)
      - At MFE >= 1.5R, stop moves to entry + 0.5R
      - At MFE >= 2.0R, stop moves to entry + 1.0R
      - At MFE >= 2.5R, stop moves to entry + 1.5R
      - In general: stop = entry + (MFE_R_floor - 1.0) * R, where R is initial risk
    """
    high = current_bar["high"]
    low = current_bar["low"]
    close = current_bar["close"]
    position.bars_held += 1

    # always compute R units against the ORIGINAL stop, not the current one
    risk_per_share = abs(position.entry_price - position.initial_stop)
    if risk_per_share <= 0:
        return {"action": "hold", "bars_held": position.bars_held}

    # --- update MFE / MAE (capped at target/stop bounds) ---
    if position.side == "long":
        capped_high = min(high, position.target)
        position.mfe_price = max(position.mfe_price, capped_high)
        capped_low = max(low, position.stop)
        position.mae_price = min(position.mae_price, capped_low)
    else:  # short
        capped_low = max(low, position.target)
        position.mfe_price = min(position.mfe_price, capped_low)
        capped_high = min(high, position.stop)
        position.mae_price = max(position.mae_price, capped_high)

    # MFE/MAE in R units
    if position.side == "long":
        mfe_r = (position.mfe_price - position.entry_price) / risk_per_share
        mae_r = (position.mae_price - position.entry_price) / risk_per_share
    else:
        mfe_r = (position.entry_price - position.mfe_price) / risk_per_share
        mae_r = (position.entry_price - position.mae_price) / risk_per_share

    # --- stepped trailing stop ---
    manage_cfg = (cfg or {}).get("manage", {})
    trail_start_r = manage_cfg.get("trail_start_r", 1.0)   # first ratchet at this MFE
    trail_step_r = manage_cfg.get("trail_step_r", 0.5)     # ratchet every this many R
    trail_offset_r = manage_cfg.get("trail_offset_r", 1.0) # stop = entry + (step_r - offset)

    # find the highest step crossed: largest k such that mfe_r >= trail_start_r + k*trail_step_r
    if mfe_r >= trail_start_r:
        steps = int((mfe_r - trail_start_r) / trail_step_r)
        current_step_r = trail_start_r + steps * trail_step_r  # e.g. 1.0, 1.5, 2.0...
        if current_step_r > position.highest_stop_step_r:
            # ratchet stop. New stop in R units = current_step_r - trail_offset_r
            new_stop_r = current_step_r - trail_offset_r
            if position.side == "long":
                new_stop_price = position.entry_price + new_stop_r * risk_per_share
                # only ratchet UP (never lower the stop)
                if new_stop_price > position.stop:
                    position.stop = new_stop_price
                    position.highest_stop_step_r = current_step_r
            else:  # short
                new_stop_price = position.entry_price - new_stop_r * risk_per_share
                if new_stop_price < position.stop:
                    position.stop = new_stop_price
                    position.highest_stop_step_r = current_step_r

    # --- exit checks (use position.stop, which may have been ratcheted) ---
    if position.side == "long":
        if low <= position.stop:
            r = (position.stop - position.entry_price) / risk_per_share
            return {
                "action": "exit",
                "exit_reason": "trail_stop" if position.highest_stop_step_r > 0 else "stop",
                "exit_price": position.stop, "realized_r": float(r),
                "bars_held": position.bars_held,
                "mfe_r": float(mfe_r), "mae_r": float(mae_r),
            }
        if high >= position.target:
            r = (position.target - position.entry_price) / risk_per_share
            return {
                "action": "exit", "exit_reason": "target",
                "exit_price": position.target, "realized_r": float(r),
                "bars_held": position.bars_held,
                "mfe_r": float(mfe_r), "mae_r": float(mae_r),
            }
    else:  # short
        if high >= position.stop:
            r = (position.entry_price - position.stop) / risk_per_share
            return {
                "action": "exit",
                "exit_reason": "trail_stop" if position.highest_stop_step_r > 0 else "stop",
                "exit_price": position.stop, "realized_r": float(r),
                "bars_held": position.bars_held,
                "mfe_r": float(mfe_r), "mae_r": float(mae_r),
            }
        if low <= position.target:
            r = (position.entry_price - position.target) / risk_per_share
            return {
                "action": "exit", "exit_reason": "target",
                "exit_price": position.target, "realized_r": float(r),
                "bars_held": position.bars_held,
                "mfe_r": float(mfe_r), "mae_r": float(mae_r),
            }

    unrealized = ((close - position.entry_price) / risk_per_share
                  if position.side == "long"
                  else (position.entry_price - close) / risk_per_share)
    return {
        "action": "hold",
        "bars_held": position.bars_held,
        "unrealized_r": float(unrealized),
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
        "trail_step_r": position.highest_stop_step_r,
    }