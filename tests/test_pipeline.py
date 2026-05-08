"""
Smoke tests for the trading system. Run with:
  python -m pytest tests/
or just:
  python tests/test_pipeline.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.data.indicators import add_all_indicators
from src.layers.scanner import in_watchlist
from src.layers.filter_layer import filter_layer
from src.layers.bias_layer import bias_layer
from src.layers.entry_layer import entry_state_check, entry_trigger_check
from src.layers.risk_sizing import risk_sizing
from src.layers.portfolio import Portfolio
from src.logs.decision_log import DecisionLog
from src.ml.regression_model import RegressionGatekeeper
from src.ml.feature_builder import build_feature_vector
from src.ml.confidence_gate import confidence_band
from src.orchestrator import run_pipeline


def _load_cfg():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(here, "..", "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _make_history():
    """Use the synthetic generator inline so the test is self-contained."""
    from scripts.generate_synthetic_data import generate_symbol
    from datetime import datetime
    df = generate_symbol("TEST", n_bars=400, start=datetime(2024, 1, 1), seed=1)
    return df


def test_indicators_compute():
    df = _make_history()
    cfg = _load_cfg()
    enriched = add_all_indicators(df, cfg)
    expected = {"atr", "adx", "vwap", "ma", "macd", "rsi", "volume_ratio"}
    assert expected.issubset(set(enriched.columns))
    assert enriched["atr"].iloc[-1] > 0


def test_layers_run_individually():
    cfg = _load_cfg()
    df = add_all_indicators(_make_history(), cfg)

    scan = in_watchlist("TEST", df, cfg)
    assert "in_watchlist" in scan

    flt = filter_layer(df, cfg)
    assert {"atr", "adx", "tradable"}.issubset(flt.keys())

    bias = bias_layer(df, cfg)
    assert bias["direction"] in {"long", "short", "neutral"}

    if bias["direction"] != "neutral":
        state = entry_state_check(df, bias["direction"], cfg)
        assert "passes" in state
        trig = entry_trigger_check(df, bias["direction"], cfg)
        assert "valid" in trig

        sizing = risk_sizing(df, bias["direction"], cfg)
        assert sizing["entry"] > 0
        assert sizing["risk_per_share"] > 0


def test_decision_log_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        log = DecisionLog(os.path.join(d, "log.sqlite"))
        try:
            sid = log.new_setup("XYZ", "2024-01-01")
            log.write(sid, "filter", {"atr": 1.0, "adx": 25.0, "tradable": True})
            log.write(sid, "bias", {"direction": "long", "strength": 2})
            rec = log.get(sid)
            assert rec["filter"]["tradable"] is True
            assert rec["bias"]["direction"] == "long"
        finally:
            log.close()  # ✅ releases the file lock before tempdir cleanup

def test_regression_untrained_returns_borderline():
    cfg = _load_cfg()
    gk = RegressionGatekeeper(cfg["regression"]["features"])
    pred = gk.predict({"atr": 1.0, "adx": 25.0})
    assert pred["trained"] is False
    band = confidence_band(pred, cfg)
    assert band["band"] == "borderline"


def test_orchestrator_end_to_end():
    cfg = _load_cfg()
    df = add_all_indicators(_make_history(), cfg)

    with tempfile.TemporaryDirectory() as d:
        cfg["decision_log"]["db_path"] = os.path.join(d, "log.sqlite")
        log = DecisionLog(cfg["decision_log"]["db_path"])
        try :
            portfolio = Portfolio()
            model = RegressionGatekeeper(cfg["regression"]["features"])
            # disable real API in tests
            cfg["claude_mcp"]["enabled"] = False

            results = []
            for i in range(60, len(df)):
                history = df.iloc[: i + 1]
                r = run_pipeline("TEST", history, portfolio, model, cfg, log)
                results.append(r)

            actions = {r.get("action") for r in results}
            # the pipeline must at minimum run without error and emit some action
            assert len(actions) > 0
            assert len(log.all_setups()) > 0
        finally:
            log.close() 

if __name__ == "__main__":
    test_indicators_compute()
    print("ok: test_indicators_compute")
    test_layers_run_individually()
    print("ok: test_layers_run_individually")
    test_decision_log_roundtrip()
    print("ok: test_decision_log_roundtrip")
    test_regression_untrained_returns_borderline()
    print("ok: test_regression_untrained_returns_borderline")
    test_orchestrator_end_to_end()
    print("ok: test_orchestrator_end_to_end")
    print("\nAll smoke tests passed.")