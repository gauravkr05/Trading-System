# """
# Claude MCP reasoning. The escalation path for borderline regression scores.

# Two modes:
#   1. API mode: if ANTHROPIC_API_KEY is set in the env and `anthropic` is
#      importable, send a structured prompt to Claude and parse its verdict.
#   2. Heuristic mode: a deterministic fallback so the pipeline runs end-to-end
#      even with no API access. Useful for offline backtesting and CI.

# The verdict shape is identical in both modes:
#   {"action": "take" | "skip" | "wait", "reasoning": str, "mode": str}
# """
# from __future__ import annotations

# import json
# import os
# from typing import Any


# SYSTEM_PROMPT = """You are a trading reasoning assistant. You receive a setup
# that scored borderline on a regression gatekeeper. Your job is to decide
# whether to TAKE, SKIP, or WAIT on this setup.

# Use the layer outputs to reason about whether the setup is genuinely strong
# despite the borderline score. Reply with a JSON object only:
#   {"action": "take" | "skip" | "wait", "reasoning": "<one short sentence>"}
# Do not include any text outside the JSON.
# """


# def _build_prompt(setup: dict[str, Any]) -> str:
#     """Compact JSON of the relevant log fields."""
#     parts = {
#         "symbol": setup.get("symbol"),
#         "filter": setup.get("filter"),
#         "bias": setup.get("bias"),
#         "entry_state": setup.get("entry_state"),
#         "entry_trigger": setup.get("entry_trigger"),
#         "regression": setup.get("regression"),
#     }
#     return f"Setup data:\n{json.dumps(parts, indent=2, default=str)}"


# def _heuristic_verdict(setup: dict[str, Any]) -> dict:
#     """
#     Deterministic borderline tiebreaker used when the API is unavailable.

#     Logic: take when both bias is strong AND state confirmations are above
#     the minimum AND pattern_strength is full. Otherwise skip. WAIT is reserved
#     for cases where the trigger is weak but the rest looks healthy.
#     """
#     bias = setup.get("bias") or {}
#     state = setup.get("entry_state") or {}
#     trig = setup.get("entry_trigger") or {}

#     bias_strength = bias.get("strength", 0)
#     confirmations = state.get("confirmations", 0)
#     minimum = state.get("minimum_required", 3)
#     pattern_strength = trig.get("pattern_strength", 0.0)

#     score = (
#         (bias_strength >= 2)
#         + (confirmations > minimum)
#         + (pattern_strength >= 1.0)
#     )

#     if score >= 2:
#         action = "take"
#         reasoning = "Borderline regression but bias, state, and trigger all look healthy."
#     elif score == 1 and pattern_strength < 1.0:
#         action = "wait"
#         reasoning = "Setup looks ok but trigger is weak; wait for a stronger pattern."
#     else:
#         action = "skip"
#         reasoning = "Borderline regression and at least two of three quality checks weak."

#     return {"action": action, "reasoning": reasoning, "mode": "heuristic"}


# def _api_verdict(setup: dict[str, Any], cfg: dict) -> dict | None:
#     """Call the real Claude API. Returns None if anything fails."""
#     api_key = os.environ.get(cfg["claude_mcp"]["api_key_env"])
#     if not api_key:
#         return None

#     try:
#         import anthropic  # type: ignore
#     except ImportError:
#         return None

#     try:
#         client = anthropic.Anthropic(api_key=api_key)
#         msg = client.messages.create(
#             model=cfg["claude_mcp"]["model"],
#             max_tokens=200,
#             system=SYSTEM_PROMPT,
#             messages=[{"role": "user", "content": _build_prompt(setup)}],
#         )
#         text = msg.content[0].text if msg.content else ""
#         # strip markdown fences if Claude added any
#         text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
#         parsed = json.loads(text)
#         action = parsed.get("action", "skip").lower()
#         if action not in {"take", "skip", "wait"}:
#             action = "skip"
#         return {
#             "action": action,
#             "reasoning": parsed.get("reasoning", "(no reasoning)"),
#             "mode": "api",
#         }
#     except Exception as e:
#         return {"action": "skip", "reasoning": f"API error: {e}", "mode": "api_error"}


# def claude_reason(setup: dict[str, Any], cfg: dict) -> dict:
#     """Main entry point. Tries API mode first, falls back to heuristic."""
#     if cfg["claude_mcp"]["enabled"]:
#         api_result = _api_verdict(setup, cfg)
#         if api_result is not None:
#             return api_result
#     return _heuristic_verdict(setup)

# """
# Groq reasoning. The escalation path for borderline regression scores.

# Two modes:
#   1. API mode: if GROQ_API_KEY is set in the env and `groq` is
#      importable, send a structured prompt to Groq and parse its verdict.
#   2. Heuristic mode: a deterministic fallback so the pipeline runs end-to-end
#      even with no API access. Useful for offline backtesting and CI.

# The verdict shape is identical in both modes:
#   {"action": "take" | "skip" | "wait", "reasoning": str, "mode": str}
# """
# from __future__ import annotations

# import json
# import os
# from typing import Any


# SYSTEM_PROMPT = """You are a trading reasoning assistant. You receive a setup
# that scored borderline on a regression gatekeeper. Your job is to decide
# whether to TAKE, SKIP, or WAIT on this setup.

# Use the layer outputs to reason about whether the setup is genuinely strong
# despite the borderline score. Reply with a JSON object only:
#   {"action": "take" | "skip" | "wait", "reasoning": "<one short sentence>"}
# Do not include any text outside the JSON.
# """


# def _build_prompt(setup: dict[str, Any]) -> str:
#     """Compact JSON of the relevant log fields."""
#     parts = {
#         "symbol": setup.get("symbol"),
#         "filter": setup.get("filter"),
#         "bias": setup.get("bias"),
#         "entry_state": setup.get("entry_state"),
#         "entry_trigger": setup.get("entry_trigger"),
#         "regression": setup.get("regression"),
#     }
#     return f"Setup data:\n{json.dumps(parts, indent=2, default=str)}"


# def _heuristic_verdict(setup: dict[str, Any]) -> dict:
#     """
#     Deterministic borderline tiebreaker used when the API is unavailable.

#     Logic: take when both bias is strong AND state confirmations are above
#     the minimum AND pattern_strength is full. Otherwise skip. WAIT is reserved
#     for cases where the trigger is weak but the rest looks healthy.
#     """
#     bias = setup.get("bias") or {}
#     state = setup.get("entry_state") or {}
#     trig = setup.get("entry_trigger") or {}

#     bias_strength = bias.get("strength", 0)
#     confirmations = state.get("confirmations", 0)
#     minimum = state.get("minimum_required", 3)
#     pattern_strength = trig.get("pattern_strength", 0.0)

#     score = (
#         (bias_strength >= 2)
#         + (confirmations > minimum)
#         + (pattern_strength >= 1.0)
#     )

#     if score >= 2:
#         action = "take"
#         reasoning = "Borderline regression but bias, state, and trigger all look healthy."
#     elif score == 1 and pattern_strength < 1.0:
#         action = "wait"
#         reasoning = "Setup looks ok but trigger is weak; wait for a stronger pattern."
#     else:
#         action = "skip"
#         reasoning = "Borderline regression and at least two of three quality checks weak."

#     return {"action": action, "reasoning": reasoning, "mode": "heuristic"}


# def _api_verdict(setup: dict[str, Any], cfg: dict) -> dict | None:
#     """Call the Groq API. Returns None if anything fails."""
#     api_key = os.environ.get(cfg["groq"]["api_key_env"])
#     if not api_key:
#         return None

#     try:
#         from groq import Groq  # type: ignore
#     except ImportError:
#         return None

#     try:
#         client = Groq(api_key=api_key)
#         response = client.chat.completions.create(
#             model=cfg["groq"]["model"],
#             max_tokens=200,
#             temperature=cfg["groq"].get("temperature", 0.2),
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": _build_prompt(setup)},
#             ],
#         )
#         text = response.choices[0].message.content or ""
#         # strip markdown fences if model added any
#         text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
#         parsed = json.loads(text)
#         action = parsed.get("action", "skip").lower()
#         if action not in {"take", "skip", "wait"}:
#             action = "skip"
#         return {
#             "action": action,
#             "reasoning": parsed.get("reasoning", "(no reasoning)"),
#             "mode": "groq_api",
#         }
#     except Exception as e:
#         return {"action": "skip", "reasoning": f"API error: {e}", "mode": "api_error"}


# def claude_reason(setup: dict[str, Any], cfg: dict) -> dict:
#     """Main entry point. Tries Groq API mode first, falls back to heuristic."""
#     if cfg["groq"]["enabled"]:
#         api_result = _api_verdict(setup, cfg)
#         if api_result is not None:
#             return api_result
#     return _heuristic_verdict(setup)


"""
Ollama (local Llama 3.1) reasoning — drop-in replacement for the Groq version.

Two modes:
  1. Ollama mode: if Ollama is running at localhost:11434, sends a structured
     prompt and parses the verdict.
  2. Heuristic mode: same deterministic fallback as before — pipeline runs
     end-to-end even if Ollama isn't running.

The verdict shape is identical in both modes:
  {"action": "take" | "skip" | "wait", "reasoning": str, "mode": str}
"""
from __future__ import annotations

import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """You are a trading reasoning assistant. You receive a setup
that scored borderline on a regression gatekeeper. Your job is to decide
whether to TAKE, SKIP, or WAIT on this setup.

Use the layer outputs to reason about whether the setup is genuinely strong
despite the borderline score. Reply with a JSON object only:
  {"action": "take" | "skip" | "wait", "reasoning": "<one short sentence>"}
Do not include any text outside the JSON.
"""


def _build_prompt(setup: dict[str, Any]) -> str:
    """Compact JSON of the relevant log fields — identical to Groq version."""
    parts = {
        "symbol":        setup.get("symbol"),
        "filter":        setup.get("filter"),
        "bias":          setup.get("bias"),
        "entry_state":   setup.get("entry_state"),
        "entry_trigger": setup.get("entry_trigger"),
        "regression":    setup.get("regression"),
    }
    return f"{SYSTEM_PROMPT}\n\nSetup data:\n{json.dumps(parts, indent=2, default=str)}"


def _heuristic_verdict(setup: dict[str, Any]) -> dict:
    """
    Deterministic borderline tiebreaker — unchanged from Groq version.
    Used when Ollama is unreachable or returns unparseable output.
    """
    bias   = setup.get("bias")          or {}
    state  = setup.get("entry_state")   or {}
    trig   = setup.get("entry_trigger") or {}

    bias_strength    = bias.get("strength", 0)
    confirmations    = state.get("confirmations", 0)
    minimum          = state.get("minimum_required", 3)
    pattern_strength = trig.get("pattern_strength", 0.0)

    score = (
        (bias_strength >= 2)
        + (confirmations > minimum)
        + (pattern_strength >= 1.0)
    )

    if score >= 2:
        action    = "take"
        reasoning = "Borderline regression but bias, state, and trigger all look healthy."
    elif score == 1 and pattern_strength < 1.0:
        action    = "wait"
        reasoning = "Setup looks ok but trigger is weak; wait for a stronger pattern."
    else:
        action    = "skip"
        reasoning = "Borderline regression and at least two of three quality checks weak."

    return {"action": action, "reasoning": reasoning, "mode": "heuristic"}


def _ollama_verdict(setup: dict[str, Any], cfg: dict) -> dict | None:
    """
    Call local Ollama. Returns None on any failure so caller falls back
    to heuristic — same pattern as the Groq _api_verdict.
    """
    model   = cfg["ollama"].get("model", "llama3.1")
    timeout = cfg["ollama"].get("timeout_seconds", 30)

    payload = {
        "model":  model,
        "prompt": _build_prompt(setup),
        "stream": False,
        "options": {
            "temperature": cfg["ollama"].get("temperature", 0.2),
            "num_predict": 200,
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json().get("response", "")

        # strip markdown fences if model adds them
        clean = (
            raw.strip()
               .removeprefix("```json")
               .removeprefix("```")
               .removesuffix("```")
               .strip()
        )

        parsed = json.loads(clean)
        action = parsed.get("action", "skip").lower()
        if action not in {"take", "skip", "wait"}:
            action = "skip"

        return {
            "action":    action,
            "reasoning": parsed.get("reasoning", "(no reasoning)"),
            "mode":      "ollama",
        }

    except requests.exceptions.ConnectionError:
        logger.warning("Ollama not reachable at %s — falling back to heuristic", OLLAMA_URL)
        return None
    except requests.exceptions.Timeout:
        logger.warning("Ollama timed out after %ds — falling back to heuristic", timeout)
        return None
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Ollama returned unparseable output (%s) — falling back to heuristic", exc)
        return None
    except Exception as exc:
        logger.warning("Unexpected Ollama error (%s) — falling back to heuristic", exc)
        return None


def claude_reason(setup: dict[str, Any], cfg: dict) -> dict:
    """
    Main entry point — identical signature to the Groq version.
    Orchestrator calls this exactly as before, no changes needed there.
    """
    if cfg["ollama"]["enabled"]:
        result = _ollama_verdict(setup, cfg)
        if result is not None:
            return result
    return _heuristic_verdict(setup)
