"""Scenario-grouped epsilon selection and independent residual calibration."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence


EPSILON_GRID = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
LOWER_BOUND_ALPHA = 0.1


def _groups(rows: Sequence[dict[str, Any]], predictions: Sequence[float], epsilon: float,
            residual_shift: float = 0.0) -> dict[str, Any]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row, prediction in zip(rows, predictions):
        scenario = str(row["scenario_id"])
        grouped.setdefault(scenario, []).append((float(row["target"]), float(prediction) + residual_shift))
    per_scenario = {}
    for scenario, pairs in sorted(grouped.items()):
        errors = [abs(target - pred) for target, pred in pairs]
        regrets = [abs(target) if (pred > epsilon) != (target > 0.0) else 0.0
                   for target, pred in pairs]
        per_scenario[scenario] = {"n": len(pairs), "mae": sum(errors) / len(errors),
                                  "decision_regret": sum(regrets) / len(regrets)}
    macro = (sum(item["decision_regret"] for item in per_scenario.values()) / len(per_scenario)
             if per_scenario else None)
    return {"scenario_count": len(per_scenario), "scenario_macro_decision_regret": macro,
            "per_scenario": per_scenario}


def select_epsilon(rows: Sequence[dict[str, Any]], predictions: Sequence[float],
                   grid: Sequence[float] = EPSILON_GRID) -> dict[str, Any]:
    """Choose epsilon by scenario-macro decision regret with stable grid-order ties."""
    if len(rows) != len(predictions) or not rows:
        raise ValueError("selection rows and predictions must be non-empty and aligned")
    if not all(isinstance(row.get("scenario_id"), str) and row["scenario_id"] for row in rows):
        raise ValueError("every selection row needs a scenario_id")
    if any(not math.isfinite(float(value)) for value in predictions):
        raise ValueError("selection predictions must be finite")
    if not grid or any(not math.isfinite(float(value)) or float(value) < 0 for value in grid):
        raise ValueError("epsilon grid must contain finite non-negative values")
    evaluated = []
    for index, epsilon in enumerate(grid):
        details = _groups(rows, predictions, float(epsilon))
        evaluated.append({"epsilon": float(epsilon), "scenario_macro_decision_regret":
                          details["scenario_macro_decision_regret"], "grid_index": index})
    best = min(evaluated, key=lambda item: (item["scenario_macro_decision_regret"], item["grid_index"]))
    return {"epsilon": best["epsilon"], "selection_metric": "scenario_macro_decision_regret",
            "selection_scenarios": sorted({row["scenario_id"] for row in rows}),
            "selection_row_count": len(rows), "grid": list(map(float, grid)),
            "grid_results": evaluated, "scenario_diagnostics": _groups(rows, predictions, best["epsilon"])}


def lower_residual_quantile(rows: Sequence[dict[str, Any]], predictions: Sequence[float],
                            alpha: float = LOWER_BOUND_ALPHA) -> dict[str, Any]:
    """Fit a one-sided finite-sample residual quantile on calibration rows only."""
    if len(rows) != len(predictions) or not rows:
        raise ValueError("calibration rows and predictions must be non-empty and aligned")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    residuals = [float(row["target"]) - float(pred) for row, pred in zip(rows, predictions)]
    if any(not math.isfinite(value) for value in residuals):
        raise ValueError("calibration residuals must be finite")
    scenario_residuals: dict[str, list[float]] = {}
    for row, residual in zip(rows, residuals):
        scenario_residuals.setdefault(str(row["scenario_id"]), []).append(residual)
    # A scenario contributes one conservative score, so correlated states do not
    # inflate the effective calibration sample count.
    grouped_residuals = {scenario: min(values) for scenario, values in scenario_residuals.items()}
    ordered = sorted(grouped_residuals.values())
    rank = math.floor((len(ordered) + 1) * alpha)
    if rank < 1:
        return {"status": "NOT_RUN_INSUFFICIENT_CALIBRATION_SCENARIOS", "q": None,
                "alpha": alpha, "sample_count": len(ordered), "calibration_row_count": len(rows),
                "score_unit": "scenario_minimum_target_minus_prediction",
                "order_statistic_rank_1_based": None,
                "calibration_scenarios": sorted(scenario_residuals),
                "calibration_scenario_count": len(scenario_residuals),
                "scenario_diagnostics": {scenario: {"n": len(values),
                    "mean_residual": sum(values) / len(values),
                    "minimum_residual": grouped_residuals[scenario]}
                    for scenario, values in sorted(scenario_residuals.items())}}
    return {"status": "PASS", "q": ordered[rank - 1], "alpha": alpha, "sample_count": len(ordered),
            "calibration_row_count": len(rows), "score_unit": "scenario_minimum_target_minus_prediction",
            "order_statistic_rank_1_based": rank,
            "calibration_scenarios": sorted(scenario_residuals),
            "calibration_scenario_count": len(scenario_residuals),
            "residuals_sha256": hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest(),
            "scenario_diagnostics": {scenario: {"n": len(values), "mean_residual": sum(values) / len(values),
                                                   "minimum_residual": grouped_residuals[scenario]}
                                     for scenario, values in sorted(scenario_residuals.items())}}


def build_deployment_metadata(selection_rows: Sequence[dict[str, Any]],
                              selection_predictions: Sequence[float],
                              calibration_rows: Sequence[dict[str, Any]],
                              calibration_predictions: Sequence[float],
                              *, data_identity: str, feature_schema: Sequence[str],
                              alpha: float = LOWER_BOUND_ALPHA) -> dict[str, Any]:
    """Combine selection and calibration outputs without using calibration to tune epsilon."""
    selection_scenarios = {row.get("scenario_id") for row in selection_rows}
    calibration_scenarios = {row.get("scenario_id") for row in calibration_rows}
    if None in selection_scenarios or None in calibration_scenarios:
        raise ValueError("selection and calibration rows require scenario_id")
    overlap = selection_scenarios & calibration_scenarios
    if overlap:
        raise ValueError(f"selection and calibration scenarios overlap: {sorted(overlap)}")
    selected = select_epsilon(selection_rows, selection_predictions)
    calibrated = lower_residual_quantile(calibration_rows, calibration_predictions, alpha)
    schema_hash = hashlib.sha256(json.dumps(list(feature_schema), separators=(",", ":")).encode()).hexdigest()
    modes = {"mean": {"score": "prediction", "epsilon": selected["epsilon"], "q": 0.0}}
    if calibrated["status"] == "PASS":
        modes["lower_bound"] = {"score": "prediction_plus_calibrated_q",
                                 "epsilon": selected["epsilon"], "q": calibrated["q"]}
    return {"schema": "isaac_roi_gain_deployment_v1", "epsilon": selected["epsilon"],
            "threshold_source": "scenario_grouped_selection_validation_only",
            "selection": selected, "lower_bound_calibration": calibrated,
            "modes": modes,
            "calibration_does_not_tune_threshold": True,
            "data_identity": data_identity, "feature_schema": list(feature_schema),
            "feature_schema_sha256": schema_hash}


def deployment_mode_metadata(deployment: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode not in {"mean", "lower_bound"}:
        raise ValueError("deployment mode must be 'mean' or 'lower_bound'")
    selected = deployment.get("modes", {}).get(mode)
    if not isinstance(selected, dict):
        raise ValueError(f"deployment mode {mode!r} is NOT_RUN or unavailable")
    return {**selected, "mode": mode, "calibration": deployment.get("lower_bound_calibration"),
            "data_identity": deployment.get("data_identity"),
            "feature_schema": deployment.get("feature_schema"),
            "feature_schema_sha256": deployment.get("feature_schema_sha256")}
