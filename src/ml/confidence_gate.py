"""
Confidence gate. The diamond after the regression model.

Routes the prediction into one of three bands:
  - 'far_positive'  : take the trade, skip Claude.
  - 'far_negative'  : NO_TRADE, skip Claude.
  - 'borderline'    : escalate to Claude MCP reasoning.

Thresholds live in config so they can be tuned per model.
"""
from __future__ import annotations


def confidence_band(prediction: dict, cfg: dict) -> dict:
    g = cfg["confidence_gate"]
    y_hat = prediction.get("y_hat", 0.0)
    trained = prediction.get("trained", False)

    if not trained:
        # if no model yet, every setup is borderline -> Claude decides everything
        # until we've collected enough labeled outcomes to train.
        return {"band": "borderline", "y_hat": y_hat, "reason": "untrained_model"}

    if y_hat >= g["far_positive_threshold"]:
        band = "far_positive"
    elif y_hat <= g["far_negative_threshold"]:
        band = "far_negative"
    else:
        band = "borderline"

    return {
        "band": band,
        "y_hat": y_hat,
        "far_positive_threshold": g["far_positive_threshold"],
        "far_negative_threshold": g["far_negative_threshold"],
    }
