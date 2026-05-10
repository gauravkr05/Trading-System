"""
Orchestrator. Walks the architecture diagram top-to-bottom for one bar of
one symbol. Every layer writes its output to the decision log spine (the
right-side rail in the diagram).

Returns the final action so a backtester can record fills.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()


from typing import Any

import pandas as pd

from src.layers.scanner import in_watchlist
from src.layers.portfolio import Portfolio, Position, manage_path
from src.layers.filter_layer import filter_layer
from src.layers.bias_layer import bias_layer
from src.layers.entry_layer import entry_state_check, entry_exhaustion_check, entry_trigger_check
from src.layers.risk_sizing import risk_sizing
from src.ml.feature_builder import build_feature_vector
from src.ml.regression_model import RegressionGatekeeper
from src.ml.confidence_gate import confidence_band
from src.reasoning.claude_mcp import claude_reason
from src.logs.decision_log import DecisionLog


def run_pipeline(
    symbol: str,
    history: pd.DataFrame,
    portfolio: Portfolio,
    model: RegressionGatekeeper,
    cfg: dict,
    log: DecisionLog,
) -> dict[str, Any]:
    """
    Run the full architecture on one symbol on the latest bar of `history`.
    """
    last_ts = str(history.iloc[-1]["timestamp"])
    setup_id = log.new_setup(symbol, last_ts)
    last_bar = history.iloc[-1].to_dict()

    # --- Scanner ------------------------------------------------------------
    scan = in_watchlist(symbol, history, cfg)
    log.write(setup_id, "scanner", scan)
    if not scan["in_watchlist"]:
        log.set_status(setup_id, "skipped_scanner")
        return {"setup_id": setup_id, "action": "skip", "stage": "scanner"}

    # --- EOD entry block ---------------------------------------------------
    # Block new entries on/after the EOD cutoff (default 15:00 IST).
    # Existing positions are still managed normally below — only NEW entries
    # are blocked. This prevents positions from opening on the EOD bar (where
    # they couldn't be cleanly EOD-exited on the same bar) or after it
    # (where they would silently carry overnight).
    eod_block_enabled = cfg.get("manage", {}).get("eod_block_entries", True)
    if eod_block_enabled and not portfolio.has_position(symbol):
        ts = history.iloc[-1]["timestamp"]
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        eod_hour = cfg.get("manage", {}).get("eod_hour", 15)
        eod_minute = cfg.get("manage", {}).get("eod_minute", 0)
        ts_minutes = ts.hour * 60 + ts.minute
        eod_minutes = eod_hour * 60 + eod_minute
        if ts_minutes >= eod_minutes:
            log.set_status(setup_id, "no_trade_eod_block")
            return {"setup_id": setup_id, "action": "no_trade",
                    "stage": "eod_block", "reason": "after_eod_cutoff"}

    # --- Open position? -> Manage path --------------------------------------
    if portfolio.has_position(symbol):
        pos = portfolio.get(symbol)
        decision = manage_path(pos, last_bar,cfg)
        log.write(setup_id, "portfolio", {"open_position": True, **decision})
        if decision["action"] == "exit":
            closed = portfolio.close(symbol)

            # Apply realistic cost model to realized R.
            # Use closed.initial_stop so cost-in-R is computed against ORIGINAL
            # risk, not the trailed stop. This keeps R units consistent with
            # how realized_r was calculated inside manage_path.
            from src.layers.risk_sizing import apply_costs_to_r
            cost_adj = apply_costs_to_r(
                realized_r=decision["realized_r"],
                entry_price=closed.entry_price,
                exit_price=decision["exit_price"],
                position_size=closed.size,
                risk_per_share=abs(closed.entry_price - closed.initial_stop),
                side=closed.side,
                cfg=cfg,
                exit_reason=decision["exit_reason"], 
            )
            log.write(closed.entry_setup_id, "costs", cost_adj)

            log.write_outcome(
                closed.entry_setup_id,
                cost_adj["realized_r_net"],   # <-- net R, not gross
                decision["exit_reason"],
                decision["bars_held"],
                mfe_r=decision.get("mfe_r", 0.0),
                mae_r=decision.get("mae_r", 0.0),
            )

            log.set_status(setup_id, "manage_exit")
            # Surface net R in the returned action so backtest summary uses it
            decision_out = {**decision, "realized_r_net": cost_adj["realized_r_net"]}
            return {"setup_id": setup_id, "action": "exit",
                    "stage": "manage", **decision_out}

        log.set_status(setup_id, "manage_hold")
        return {"setup_id": setup_id, "action": "hold", "stage": "manage", **decision}

    log.write(setup_id, "portfolio", {"open_position": False})

    # --- Filter layer (ATR/ADX gate) ---------------------------------------
    flt = filter_layer(history, cfg)
    log.write(setup_id, "filter", flt)
    if not flt["tradable"]:
        log.set_status(setup_id, "no_trade_filter")
        return {"setup_id": setup_id, "action": "no_trade", "stage": "filter"}

    # --- Bias layer ---------------------------------------------------------
    bias = bias_layer(history, cfg)
    log.write(setup_id, "bias", bias)
    if bias["direction"] == "neutral":
        log.set_status(setup_id, "no_trade_bias")
        return {"setup_id": setup_id, "action": "no_trade", "stage": "bias"}

    # --- Entry layer Stage 1 ------------------------------------------------
    state = entry_state_check(history, bias["direction"], cfg)
    log.write(setup_id, "entry_state", state)
    if not state["passes"]:
        log.set_status(setup_id, "no_trade_state")
        return {"setup_id": setup_id, "action": "no_trade", "stage": "state"}

    # --- Entry layer Stage 1.5: exhaustion check ----------------------------
    # Calibrated from Stage 5 diagnostics. Rejects entries that are
    # overextended from MA or chasing the swing extreme. Lives between
    # Stage 1 (state) and Stage 2 (trigger) so we cut cleanly before
    # pattern matching consumes resources.
    exhaustion = entry_exhaustion_check(state, cfg)
    log.write(setup_id, "entry_exhaustion", exhaustion)
    if not exhaustion["passes"]:
        log.set_status(setup_id, f"no_trade_exhaustion_{exhaustion['reason']}")
        return {"setup_id": setup_id, "action": "no_trade",
                "stage": "exhaustion", "reason": exhaustion["reason"]}

    # --- Entry layer Stage 2 ------------------------------------------------
    trigger = entry_trigger_check(history, bias["direction"], cfg)
    log.write(setup_id, "entry_trigger", trigger)
    if not trigger["valid"]:
        log.set_status(setup_id, "wait_trigger")
        return {"setup_id": setup_id, "action": "wait", "stage": "trigger"}

    # --- Build feature vector (reads from log spine) ------------------------
    setup_so_far = log.get(setup_id)
    features = build_feature_vector(setup_so_far)
    log.write(setup_id, "features", features)

    # --- Regression model ---------------------------------------------------
    pred = model.predict(features)
    log.write(setup_id, "regression", pred)

    # --- Confidence band ---------------------------------------------------
    gatekeeper_enabled = cfg.get("regression", {}).get("gatekeeper_enabled", True)
    claude_enabled = cfg.get("claude_mcp", {}).get("enabled", True)

    if gatekeeper_enabled:
        band_info = confidence_band(pred, cfg)

        if band_info["band"] == "far_negative":
            log.set_status(setup_id, "no_trade_model")
            return {"setup_id": setup_id, "action": "no_trade",
                    "stage": "regression_far_neg", "y_hat": pred["y_hat"]}

        if band_info["band"] == "borderline" and claude_enabled:
            verdict = claude_reason(log.get(setup_id), cfg)
            log.write(setup_id, "claude", {**band_info, **verdict})
            if verdict["action"] != "take":
                log.set_status(setup_id, f"no_trade_claude_{verdict['action']}")
                return {"setup_id": setup_id, "action": verdict["action"],
                        "stage": "claude", "y_hat": pred["y_hat"],
                        "reasoning": verdict["reasoning"]}
        else:
            log.write(setup_id, "claude", {**band_info, "skipped": True,
                                           "reason": "far_positive_or_claude_off"})
    else:
        log.write(setup_id, "claude", {"skipped": True,
                                        "reason": "gatekeeper_disabled"})
    # --- Risk sizing --------------------------------------------------------
    sizing = risk_sizing(history, bias["direction"], cfg)
    log.write(setup_id, "sizing", sizing)

    # Reject trades that aren't tradable (size < 1 share, shorts disabled, etc.)
    if not sizing.get("tradable", True):
        log.set_status(setup_id, f"no_trade_sizing_{sizing.get('skip_reason', 'untradable')}")
        return {"setup_id": setup_id, "action": "no_trade",
                "stage": "sizing", "reason": sizing.get("skip_reason")}

    # --- BUY / SELL signal --------------------------------------------------
    order = {
        "symbol": symbol,
        "side": sizing["side"],
        "entry": sizing["entry"],
        "stop": sizing["stop"],
        "target": sizing["target"],
        "size": sizing["position_size"],
        "timestamp": last_ts,
    }
    log.write(setup_id, "order", order)
    log.set_status(setup_id, "order_placed")

    # --- Open the position in the portfolio ---------------------------------
    portfolio.open(Position(
        symbol=symbol,
        side="long" if sizing["side"] == "BUY" else "short",
        entry_price=sizing["entry"],
        stop=sizing["stop"],
        target=sizing["target"],
        size=sizing["position_size"],
        entry_setup_id=setup_id,
    ))

    return {"setup_id": setup_id, "action": "trade", "stage": "order",
            "y_hat": pred["y_hat"], **order}