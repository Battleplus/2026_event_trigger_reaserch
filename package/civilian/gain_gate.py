"""Decision rule for a public net-gain prediction."""

from __future__ import annotations

import math
from typing import Any


def decide_gain(
    gain_hat: Any = None,
    threshold: Any = 0.0,
    lower_bound: Any = None,
    *,
    gate_mode: str = "mean",
    forced: bool = False,
    safe_to_suppress: bool = True,
) -> dict:
    """Return a gate decision; safety-forced interventions always pass."""
    if type(forced) is not bool or type(safe_to_suppress) is not bool:
        raise TypeError("forced and safe_to_suppress must be bools")
    if gate_mode not in {"mean", "calibrated_lower_bound"}:
        raise ValueError("gate_mode must be 'mean' or 'calibrated_lower_bound'")
    if forced or not safe_to_suppress:
        return {
            "schema": "gain-gate-decision/1", "allow_replan": True,
            "forced": True, "reason": "safety_override", "score": None,
            "gate_mode": gate_mode, "threshold": None, "decision_score": None, "margin": None,
        }
    score = float(gain_hat)
    epsilon = float(threshold)
    if not all(math.isfinite(value) for value in (score, epsilon)):
        raise ValueError("gain values and threshold must be finite")
    if gate_mode == "calibrated_lower_bound":
        if lower_bound is None:
            return {
                "schema": "gain-gate-decision/1", "allow_replan": False,
                "forced": False, "reason": "calibration_unavailable", "score": score,
                "gate_mode": gate_mode, "threshold": epsilon,
                "decision_score": None, "margin": None,
            }
        conservative = float(lower_bound)
    else:
        conservative = score
    if not math.isfinite(conservative):
        raise ValueError("gain values and threshold must be finite")
    margin = conservative - epsilon
    allow = margin > 0.0
    return {
        "schema": "gain-gate-decision/1", "allow_replan": allow,
        "forced": False, "reason": "positive_lower_bound" if allow else "gain_below_threshold",
        "gate_mode": gate_mode,
        "score": score, "threshold": epsilon, "decision_score": conservative,
        "margin": margin,
    }
