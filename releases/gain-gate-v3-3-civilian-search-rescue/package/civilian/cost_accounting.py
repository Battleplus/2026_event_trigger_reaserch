"""Cost accounting for paired keep/replan counterfactual traces.

The objective is expressed in synthetic simulation units.  Physical execution
terms belong to ``J_exec``; costs caused by requesting a reconstruction belong
to ``C_replan`` and are charged exactly once.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class GainCostConfig:
    completion: float = 1.0
    delay: float = 0.2
    energy: float = 0.01
    risk: float = 0.25
    communication: float = 0.01
    switch: float = 0.05
    reconstruction: float = 0.05
    reconstruction_size: float = 0.01
    cpu_second: float = 0.01

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"cost weight {name} must be finite and nonnegative")


DEFAULT_CONFIG = GainCostConfig()
CPU_WEIGHT_SENSITIVITY = (0.0, 0.001, 0.1)


def task_priorities_from_truth(truth: Mapping[str, Any]) -> dict[str, float]:
    """Extract frozen logical task priorities from a private truth snapshot."""
    rows = truth.get("tasks")
    if not isinstance(rows, list):
        raise ValueError("gain cost truth must contain logical task rows")
    result: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
            raise ValueError("gain cost truth has malformed task rows")
        priority = row.get("priority")
        if isinstance(priority, bool) or not isinstance(priority, (int, float)) or not math.isfinite(float(priority)) or float(priority) <= 0:
            raise ValueError("gain cost task priority must be positive")
        result[str(row["id"])] = float(priority)
    if not result:
        raise ValueError("gain cost truth has no logical tasks")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _normalize_priorities(priorities: Mapping[str, float]) -> dict[str, float]:
    checked = {str(key): _finite_nonnegative(value, "task priority") for key, value in priorities.items()}
    if not checked or any(value <= 0 for value in checked.values()):
        raise ValueError("task priorities must be nonempty and strictly positive")
    total = sum(checked.values())
    return {key: value / total for key, value in checked.items()}


def _event_rows(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    steps = trace.get("steps")
    if not isinstance(steps, list):
        raise ValueError("counterfactual trace must contain steps")
    events: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, Mapping) or not isinstance(step.get("raw_events"), list):
            raise ValueError("counterfactual step lacks raw events")
        events.extend(row for row in step["raw_events"] if isinstance(row, dict))
    return events


def _weighted_completion_and_residual(
    end_truth: Mapping[str, Any], priorities: Mapping[str, float]
) -> tuple[float, float]:
    rows = end_truth.get("tasks")
    if not isinstance(rows, list):
        raise ValueError("endpoint truth must contain logical task rows")
    by_id = {str(row.get("id")): row for row in rows if isinstance(row, Mapping)}
    weights = _normalize_priorities(priorities)
    if not set(weights) <= set(by_id):
        raise ValueError("endpoint is missing weighted logical tasks")
    weighted_completion = 0.0
    residual_fraction = 0.0
    for task_id, weight in weights.items():
        row = by_id[task_id]
        if row.get("status") == "completed":
            weighted_completion += weight
            continue
        remaining = row.get("progress")
        if isinstance(remaining, bool) or not isinstance(remaining, (int, float)):
            raise ValueError("endpoint logical task progress is missing")
        remaining_fraction = max(0.0, min(1.0, 1.0 - float(remaining)))
        residual_fraction += weight * remaining_fraction
    return weighted_completion, residual_fraction


def raw_execution_primitives(trace: Mapping[str, Any]) -> dict[str, float]:
    """Sum physical execution primitives from persisted per-step raw events."""
    events = _event_rows(trace)
    delay = sum(
        _finite_nonnegative(row.get("minutes", 0.0), "delay minutes")
        for row in events if row.get("type") == "delay_accrued"
    )
    energy = sum(
        _finite_nonnegative(row.get("energy", 0.0), "energy")
        for row in events if row.get("type") in {"move", "service"}
    )
    risk = sum(
        _finite_nonnegative(row.get("environmental_risk", 0.0), "environmental risk")
        for row in events if row.get("type") == "move"
    )
    risk += sum(row.get("type") in {"task_failed", "platform_failure"} for row in events)
    return {"delay_minutes": float(delay), "energy_units": float(energy), "risk_units": float(risk)}


def replan_primitives(trace: Mapping[str, Any]) -> dict[str, float]:
    """Count replan-only communication, switching, calls, and changed actions."""
    events = _event_rows(trace)
    messages = sum(
        int(_finite_nonnegative(row.get("message_count", 0), "message count"))
        for row in events if row.get("type") == "communication_tx"
    )
    switches = len({
        (int(row.get("time", -1)), str(row.get("task_id", "")))
        for row in events if row.get("type") == "task_switch"
    })
    size = 0
    calls = int(trace.get("initial_planner_call_count", 0))
    continuation_planner_cpu = 0.0
    for step in trace.get("steps", []):
        step_events = step.get("raw_events", [])
        owners = {str(row.get("task_id")) for row in step_events if row.get("type") == "task_owner_change"}
        reorder_count = sum(row.get("type") == "queue_reorder" for row in step_events)
        step_size = len(owners) + reorder_count
        size += step_size
        if step.get("stage") != "replan_apply":
            calls += int(step.get("planner_invoked", False))
    # The continuation receipt is the authoritative CPU split. The separate
    # planner-call list is a derived audit view and may omit a legacy callback
    # receipt or carry measured callback time rather than planner CPU.
    receipts = trace.get("continuation_policy_receipts", [])
    if not isinstance(receipts, list):
        receipts = trace.get("planner_call_receipts", [])
    for receipt in receipts:
        if receipt.get("call_type") == "continuation_safety_replan" or receipt.get("planner_invoked") is True:
            continuation_planner_cpu += _finite_nonnegative(
                receipt.get("planner_cpu_seconds", receipt.get("cpu_seconds", 0.0)),
                "continuation planner CPU seconds",
            )
    return {
        "communication_messages": float(messages),
        "switch_count": float(switches),
        "call_count": float(calls),
        "reconstruction_size": float(size),
        "continuation_planner_cpu_seconds": float(continuation_planner_cpu),
    }


def calculate_branch_cost(
    trace: Mapping[str, Any],
    *,
    task_priorities: Mapping[str, float],
    planner_cpu_seconds: float = 0.0,
    config: GainCostConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Calculate the frozen V2 score decomposition from one raw branch trace."""
    if not isinstance(trace.get("start_truth"), Mapping) or not isinstance(trace.get("raw_truth"), Mapping):
        raise ValueError("counterfactual trace must bind start and endpoint truth")
    start_ids = {str(row.get("id")) for row in trace["start_truth"].get("tasks", []) if isinstance(row, Mapping)}
    if not set(map(str, task_priorities)) <= start_ids:
        raise ValueError("frozen priority weights contain tasks absent at the label snapshot")
    cpu = _finite_nonnegative(planner_cpu_seconds, "planner CPU seconds")
    safety_cpu = _finite_nonnegative(trace.get("continuation_policy_cpu_seconds", 0.0), "continuation policy CPU seconds")
    online_monitor_cpu = _finite_nonnegative(trace.get("evaluation_monitor_cpu_seconds", 0.0), "evaluation monitor CPU seconds")
    weighted_completion, residual_fraction = _weighted_completion_and_residual(trace["raw_truth"], task_priorities)
    execution = raw_execution_primitives(trace)
    replanning = replan_primitives(trace)
    execution_components = {
        "weighted_completion": weighted_completion,
        "weighted_remaining_fraction": residual_fraction,
        "completion_cost": config.completion * residual_fraction,
        "delay_cost": config.delay * execution["delay_minutes"],
        "energy_cost": config.energy * execution["energy_units"],
        "risk_cost": config.risk * execution["risk_units"],
        **execution,
    }
    replan_components = {
        **replanning,
        "planner_cpu_seconds": cpu,
        "continuation_planner_cpu_seconds": replanning["continuation_planner_cpu_seconds"],
        "communication_cost": config.communication * replanning["communication_messages"],
        "switch_cost": config.switch * replanning["switch_count"],
        "reconstruction_cost": config.reconstruction * replanning["call_count"],
        "reconstruction_size_cost": config.reconstruction_size * replanning["reconstruction_size"],
        "planner_cpu_cost": config.cpu_second * (cpu + replanning["continuation_planner_cpu_seconds"]),
    }
    j_exec = sum(execution_components[key] for key in ("completion_cost", "delay_cost", "energy_cost", "risk_cost"))
    c_replan = sum(replan_components[key] for key in (
        "communication_cost", "switch_cost", "reconstruction_cost", "reconstruction_size_cost", "planner_cpu_cost"
    ))
    c_safety = config.cpu_second * (safety_cpu + online_monitor_cpu)
    safety_components = {
        "continuation_policy_cpu_seconds": safety_cpu,
        "continuation_policy_cpu_cost": config.cpu_second * safety_cpu,
    }
    if "evaluation_monitor_cpu_seconds" in trace:
        safety_components.update({
            "evaluation_monitor_cpu_seconds": online_monitor_cpu,
            "evaluation_monitor_cpu_cost": config.cpu_second * online_monitor_cpu,
        })
    return {
        "J_exec": float(j_exec),
        "C_replan": float(c_replan),
        "C_safety": float(c_safety),
        "J_total": float(j_exec + c_replan + c_safety),
        "execution_components": execution_components,
        "replan_components": replan_components,
        "safety_components": safety_components,
        "cost_config": asdict(config),
        "cost_accounting_schema": "gain-cost-accounting/2",
    }


def paired_gain(keep_total: float, replan_total: float) -> float:
    """Return keep-minus-replan gain (positive values favor reconstruction)."""
    keep = _finite_nonnegative(keep_total, "keep total cost")
    replan = _finite_nonnegative(replan_total, "replan total cost")
    return float(keep - replan)
