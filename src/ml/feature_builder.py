"""
Feature builder. Pulls logged layer outputs from the decision log and
assembles a feature vector for the regression model.

This is the 'Build feature vector' node in the architecture diagram. Its job is
to produce the SAME representation at inference time and at training time, so
the regression learns and predicts on identical inputs.

The list of features is driven by config so a different model can extend it
without code changes.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _bias_strength_score(bias: dict) -> float:
    """Encode bias direction + strength as a signed scalar in [-2, 2]."""
    if not bias:
        return 0.0
    sign = 1 if bias.get("direction") == "long" else -1 if bias.get("direction") == "short" else 0
    return float(sign * bias.get("strength", 0))


def build_feature_vector(setup: dict[str, Any]) -> dict[str, float]:
    """
    Build the feature vector for one setup. `setup` is a row from DecisionLog.get
    after the entry layers have run.
    """
    filt = setup.get("filter") or {}
    bias = setup.get("bias") or {}
    state = setup.get("entry_state") or {}
    trig = setup.get("entry_trigger") or {}

    return {
        "atr": float(filt.get("atr", 0.0) or 0.0),
        "adx": float(filt.get("adx", 0.0) or 0.0),
        "bias_strength": _bias_strength_score(bias),
        "state_confirmations": float(state.get("confirmations", 0) or 0),
        "rsi": float(state.get("rsi", 50.0) or 50.0),
        "volume_ratio": float(state.get("volume_ratio", 0.0) or 0.0),
        "expansion_ratio": float(state.get("expansion_ratio", 0.0) or 0.0),
        "level_distance_atr": float(state.get("level_distance_atr", 0.0) or 0.0),
        "pattern_strength": float(trig.get("pattern_strength", 0.0) or 0.0),
    }


def vector_to_array(features: dict[str, float], feature_names: list[str]) -> np.ndarray:
    return np.array([[features.get(n, 0.0) for n in feature_names]], dtype=float)
