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
from src.layers.entry_layer import entry_state_check, entry_trigger_check
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

    # --- Open position? -> Manage path --------------------------------------
    if portfolio.has_position(symbol):
        pos = portfolio.get(symbol)
        decision = manage_path(pos, last_bar,cfg)
        log.write(setup_id, "portfolio", {"open_position": True, **decision})
        if decision["action"] == "exit":
            closed = portfolio.close(symbol)
            # log.write_outcome(
            #     closed.entry_setup_id,
            #     decision["realized_r"],
            #     decision["exit_reason"],
            #     decision["bars_held"],
            # )
            log.write_outcome(
                closed.entry_setup_id,
                decision["realized_r"],
                decision["exit_reason"],
                decision["bars_held"],
                mfe_r=decision.get("mfe_r", 0.0),
                mae_r=decision.get("mae_r", 0.0),
            )
            
            log.set_status(setup_id, "manage_exit")
            return {"setup_id": setup_id, "action": "exit",
                    "stage": "manage", **decision}
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
    band_info = confidence_band(pred, cfg)

    if band_info["band"] == "far_negative":
        log.set_status(setup_id, "no_trade_model")
        return {"setup_id": setup_id, "action": "no_trade",
                "stage": "regression_far_neg", "y_hat": pred["y_hat"]}

    if band_info["band"] == "borderline":
        # escalate to Claude
        verdict = claude_reason(log.get(setup_id), cfg)
        log.write(setup_id, "claude", {**band_info, **verdict})
        if verdict["action"] != "take":
            log.set_status(setup_id, f"no_trade_claude_{verdict['action']}")
            return {"setup_id": setup_id, "action": verdict["action"],
                    "stage": "claude", "y_hat": pred["y_hat"],
                    "reasoning": verdict["reasoning"]}
    else:
        # far_positive -> log the band only, skip Claude
        log.write(setup_id, "claude", {**band_info, "skipped": True,
                                       "reason": "far_positive"})

    # --- Risk sizing --------------------------------------------------------
    sizing = risk_sizing(history, bias["direction"], cfg)
    log.write(setup_id, "sizing", sizing)

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
