"""Closed-loop evaluation of frozen gain-gate policies on test/OOD scenarios."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

try:
    from .cost_accounting import DEFAULT_CONFIG, GainCostConfig, calculate_branch_cost
    from . import gain_models
except ImportError:
    from cost_accounting import DEFAULT_CONFIG, GainCostConfig, calculate_branch_cost
    import gain_models


EVALUATION_SCHEMA = "gain-closed-loop-evaluation/1"
REPLANNER_VERSION = "Public-C1-Catalog-Replanner-v2"
METHODS = (
    "Periodic4", "ExistingET", "ExistingETAttemptCooldown", "RiskET", "HeuristicGain", "GainTree",
    "GainMLP8201", "GainMLP8202", "GainMLP8203",
)
BASELINES = ("ExistingET", "ExistingETAttemptCooldown")
MODEL_PATH_KEYS = {
    "GainTree": "gain_tree",
    "GainMLP8201": "gain_mlp_8201",
    "GainMLP8202": "gain_mlp_8202",
    "GainMLP8203": "gain_mlp_8203",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"GAIN_EVALUATION_FINITE_VALUE_REQUIRED:{name}")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"GAIN_EVALUATION_VALUE_OUT_OF_RANGE:{name}")
    return number


def load_evaluation_modules() -> dict[str, Any]:
    """Load the runner's public environment and safety adapters."""
    try:
        from . import gain_runner
    except ImportError:
        import gain_runner
    return gain_runner.load_evaluation_modules()


def _runner_api() -> Any:
    try:
        from . import gain_runner
    except ImportError:
        import gain_runner
    return gain_runner


def _cost_config(config: Mapping[str, Any]) -> GainCostConfig:
    values = dict(DEFAULT_CONFIG.__dict__)
    configured = config.get("score_weights", {})
    if not isinstance(configured, Mapping):
        raise ValueError("GAIN_EVALUATION_SCORE_WEIGHTS_INVALID")
    aliases = {"reconstruction_call": "reconstruction", "reconstruction_size": "reconstruction_size"}
    for source, target in aliases.items():
        if source in configured:
            values[target] = configured[source]
    for key in ("completion", "delay", "energy", "risk", "communication", "switch"):
        if key in configured:
            values[key] = configured[key]
    label_config = config.get("labels", {})
    if isinstance(label_config, Mapping) and "planning_cpu_cost_weight_per_second" in label_config:
        values["cpu_second"] = label_config["planning_cpu_cost_weight_per_second"]
    return GainCostConfig(**values)


def _normalize_weights(priorities: Mapping[str, float]) -> dict[str, float]:
    clean = {str(key): _finite(value, "task_priority", minimum=1e-300) for key, value in priorities.items()}
    if not clean:
        raise ValueError("GAIN_EVALUATION_TASK_PRIORITIES_EMPTY")
    total = sum(clean.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("GAIN_EVALUATION_TASK_PRIORITIES_INVALID")
    return {key: value / total for key, value in clean.items()}


def _scenario_id(scenario: Mapping[str, Any]) -> str:
    value = scenario.get("scenario_id", scenario.get("scenario_instance_id", scenario.get("parent_id")))
    if value is None or not str(value):
        raise ValueError("GAIN_EVALUATION_SCENARIO_ID_REQUIRED")
    return str(value)


def recompute_episode_score(
    trajectory: Mapping[str, Any],
    *,
    task_priorities: Mapping[str, float],
    planner_cpu_seconds: float,
    cost_config: GainCostConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Recompute one episode score from raw trajectory evidence.

    Task priorities are frozen at first public observation. The private start
    truth binds their logical identities, including tasks hidden until arrival.
    """
    if not isinstance(trajectory.get("start_truth"), Mapping) or not isinstance(trajectory.get("raw_truth"), Mapping):
        raise ValueError("GAIN_EVALUATION_EPISODE_TRUTH_REQUIRED")
    weights = _normalize_weights(task_priorities)
    start_tasks = trajectory["start_truth"].get("tasks")
    end_tasks = trajectory["raw_truth"].get("tasks")
    if not isinstance(start_tasks, list) or not isinstance(end_tasks, list):
        raise ValueError("GAIN_EVALUATION_EPISODE_TASK_TRUTH_INVALID")
    start_ids = {str(row.get("id")) for row in start_tasks if isinstance(row, Mapping)}
    end_by_id = {str(row.get("id")): row for row in end_tasks if isinstance(row, Mapping)}
    if not set(weights) <= start_ids or not set(weights) <= set(end_by_id):
        raise ValueError("GAIN_EVALUATION_TASK_UNIVERSE_BINDING_MISMATCH")
    weighted_completion = 0.0
    weighted_remaining = 0.0
    for task_id, weight in weights.items():
        task = end_by_id[task_id]
        progress = min(1.0, max(0.0, _finite(task.get("progress"), "endpoint_task_progress", minimum=-1e-300)))
        weighted_completion += weight * float(task.get("status") == "completed")
        weighted_remaining += weight * (1.0 - progress)

    events = _trajectory_events(trajectory)
    delay = sum(_finite(row.get("minutes", 0.0), "delay_minutes") for row in events if row.get("type") == "delay_accrued")
    energy = sum(_finite(row.get("energy", 0.0), "energy_units") for row in events if row.get("type") in {"move", "service"})
    risk = sum(_finite(row.get("environmental_risk", 0.0), "environmental_risk") for row in events if row.get("type") == "move")
    risk += sum(row.get("type") in {"task_failed", "platform_failure"} for row in events)
    messages = sum(int(_finite(row.get("message_count", 0), "message_count")) for row in events if row.get("type") == "communication_tx")
    switches = len({(int(row.get("time", -1)), str(row.get("task_id", ""))) for row in events if row.get("type") == "task_switch"})
    reconstruction_size = 0
    calls = int(trajectory.get("initial_planner_call_count", 0))
    for step in trajectory["steps"]:
        step_events = step["raw_events"]
        owners = {str(row.get("task_id")) for row in step_events if row.get("type") == "task_owner_change"}
        reconstruction_size += len(owners) + sum(row.get("type") == "queue_reorder" for row in step_events)
        if step.get("stage") != "replan_apply":
            calls += int(bool(step.get("planner_invoked", False)))

    planner_cpu = _finite(planner_cpu_seconds, "planner_cpu_seconds")
    continuation_cpu = _finite(trajectory.get("continuation_policy_cpu_seconds", 0.0), "continuation_policy_cpu_seconds")
    monitor_cpu = _finite(trajectory.get("evaluation_monitor_cpu_seconds", 0.0), "evaluation_monitor_cpu_seconds")
    continuation_planner_cpu = sum(
        _finite(row.get("cpu_seconds", 0.0), "continuation_planner_cpu_seconds")
        for row in trajectory.get("planner_call_receipts", [])
        if isinstance(row, Mapping) and row.get("call_type") == "continuation_safety_replan"
    )
    execution = {
        "weighted_completion": float(weighted_completion),
        "weighted_remaining_fraction": float(weighted_remaining),
        "completion_cost": float(cost_config.completion * weighted_remaining),
        "delay_cost": float(cost_config.delay * delay),
        "energy_cost": float(cost_config.energy * energy),
        "risk_cost": float(cost_config.risk * risk),
        "delay_minutes": float(delay), "energy_units": float(energy), "risk_units": float(risk),
    }
    replan = {
        "communication_messages": float(messages), "switch_count": float(switches),
        "call_count": float(calls), "reconstruction_size": float(reconstruction_size),
        "continuation_planner_cpu_seconds": float(continuation_planner_cpu), "planner_cpu_seconds": planner_cpu,
        "communication_cost": float(cost_config.communication * messages),
        "switch_cost": float(cost_config.switch * switches),
        "reconstruction_cost": float(cost_config.reconstruction * calls),
        "reconstruction_size_cost": float(cost_config.reconstruction_size * reconstruction_size),
        "planner_cpu_cost": float(cost_config.cpu_second * (planner_cpu + continuation_planner_cpu)),
    }
    execution_total = sum(execution[key] for key in ("completion_cost", "delay_cost", "energy_cost", "risk_cost"))
    replan_total = sum(replan[key] for key in (
        "communication_cost", "switch_cost", "reconstruction_cost", "reconstruction_size_cost", "planner_cpu_cost",
    ))
    safety_cost = float(cost_config.cpu_second * (monitor_cpu + continuation_cpu))
    safety_components = {
        "continuation_policy_cpu_seconds": continuation_cpu,
        "continuation_policy_cpu_cost": float(cost_config.cpu_second * continuation_cpu),
    }
    if "evaluation_monitor_cpu_seconds" in trajectory:
        safety_components.update({
            "evaluation_monitor_cpu_seconds": monitor_cpu,
            "evaluation_monitor_cpu_cost": float(cost_config.cpu_second * monitor_cpu),
        })
    return {
        "J_exec": float(execution_total), "C_replan": float(replan_total), "C_safety": safety_cost,
        "J_total": float(execution_total + replan_total + safety_cost),
        "execution_components": execution, "replan_components": replan,
        "safety_components": safety_components,
        "cost_config": cost_config.__dict__.copy(), "cost_accounting_schema": "gain-cost-accounting/2",
        "weighted_completion_rate": float(weighted_completion), "task_priority_weights": weights,
        "online_cpu_seconds": float(planner_cpu + continuation_planner_cpu + continuation_cpu + monitor_cpu),
        "online_cpu_cost": float(cost_config.cpu_second * (planner_cpu + continuation_planner_cpu + continuation_cpu + monitor_cpu)),
    }


def _trajectory_events(trajectory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        raise ValueError("GAIN_EVALUATION_TRACE_STEPS_REQUIRED")
    events: list[Mapping[str, Any]] = []
    for step in steps:
        if not isinstance(step, Mapping) or not isinstance(step.get("raw_events"), list):
            raise ValueError("GAIN_EVALUATION_TRACE_STEP_EVENTS_INVALID")
        if not all(isinstance(row, Mapping) for row in step["raw_events"]):
            raise ValueError("GAIN_EVALUATION_TRACE_RAW_EVENT_INVALID")
        events.extend(step["raw_events"])
    return events


def _assert_score_parity(production: Mapping[str, Any], independent: Mapping[str, Any]) -> None:
    def visit(expected: Any, observed: Any, path: str) -> None:
        if isinstance(expected, Mapping):
            if not isinstance(observed, Mapping) or not set(expected) <= set(observed):
                raise ValueError(f"GAIN_EVALUATION_SCORE_PARITY_MISMATCH:{path}")
            for key, value in expected.items():
                visit(value, observed[key], f"{path}.{key}")
            return
        if isinstance(expected, bool) or isinstance(observed, bool):
            if expected is not observed:
                raise ValueError(f"GAIN_EVALUATION_SCORE_PARITY_MISMATCH:{path}")
            return
        if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
            if not math.isclose(float(expected), float(observed), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"GAIN_EVALUATION_SCORE_PARITY_MISMATCH:{path}")
            return
        if expected != observed:
            raise ValueError(f"GAIN_EVALUATION_SCORE_PARITY_MISMATCH:{path}")

    visit(production, independent, "score")


def _weighted_completion_rate(trajectory: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    end_tasks = trajectory["raw_truth"].get("tasks")
    if not isinstance(end_tasks, list):
        raise ValueError("GAIN_EVALUATION_ENDPOINT_TASKS_REQUIRED")
    by_id = {str(row.get("id")): row for row in end_tasks if isinstance(row, Mapping)}
    if not set(weights) <= set(by_id):
        raise ValueError("GAIN_EVALUATION_ENDPOINT_TASK_UNIVERSE_MISMATCH")
    return float(sum(weight for task_id, weight in weights.items() if by_id[task_id].get("status") == "completed"))


def _observe_task_priorities(
    scenario: Mapping[str, Any], modules: Mapping[str, Any], helpers: Any,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Freeze one public task universe per scenario using a passive NOOP run."""
    pairs = modules["pairs"]
    env = modules["env_class"](helpers.legacy_environment_scenario(scenario))
    env.reset()
    start_truth = copy.deepcopy(env._raw_truth())
    priorities: dict[str, float] = {}
    _task_priority_update(pairs.model_public_observation(env), priorities)
    horizon = int(scenario["base_scenario"]["horizon"])
    while int(env.time) < horizon:
        env.step(copy.deepcopy(pairs.NOOP_ACTION))
        _task_priority_update(pairs.model_public_observation(env), priorities)
    trajectory = copy.deepcopy(env.export_trajectory())
    trajectory["start_truth"] = start_truth
    trajectory["evaluation"] = {
        "method": "NoopPriorityObserver", "scenario_id": _scenario_id(scenario),
        "split": str(scenario["split"]), "task_priority_at_first_public_observation": copy.deepcopy(priorities),
    }
    return priorities, trajectory


def _task_priority_update(public_observation: Mapping[str, Any], priorities: dict[str, float]) -> None:
    rows = public_observation.get("tasks", [])
    if not isinstance(rows, list):
        raise ValueError("GAIN_EVALUATION_PUBLIC_TASKS_INVALID")
    for task in rows:
        if not isinstance(task, Mapping) or not bool(task.get("known", True)):
            continue
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("GAIN_EVALUATION_PUBLIC_TASK_ID_INVALID")
        if task_id not in priorities:
            priority = _finite(task.get("priority"), "public_task_priority", minimum=1e-300)
            priorities[task_id] = priority


def _risk_score(detection: Mapping[str, Any]) -> float:
    margins = detection.get("constraint_margin_losses", {})
    values = [float(value) for value in margins.values() if isinstance(value, (int, float)) and not isinstance(value, bool)] if isinstance(margins, Mapping) else []
    return max(values, default=float(detection.get("d_margin", 0.0)))


def _heuristic_gain(features: Mapping[str, float], detection: Mapping[str, Any]) -> float:
    """A small, public-only heuristic score used as a non-learned gain gate."""
    impact = min(1.0, max(0.0, float(features.get("affected_task_fraction", 0.0))))
    platform_impact = min(1.0, max(0.0, float(features.get("affected_platform_fraction", 0.0))))
    margin_loss = min(1.0, max(0.0, _risk_score(detection)))
    return float(0.5 * impact + 0.25 * platform_impact + 0.25 * margin_loss)


def _attempt_cooldown_candidate(
    event_candidate: bool, *, forced: bool, current_time: int, next_allowed_time: int,
) -> bool:
    if forced:
        return True
    return bool(event_candidate) and int(current_time) >= int(next_allowed_time)


def _episode_horizon(scenario: Mapping[str, Any], config: Mapping[str, Any]) -> int:
    declared = int(scenario["base_scenario"]["horizon"])
    expected = config.get("scenario", {}).get("horizon_minutes") if isinstance(config.get("scenario"), Mapping) else None
    if expected is not None and declared != int(expected):
        raise ValueError("GAIN_EVALUATION_SCENARIO_HORIZON_MISMATCH")
    return declared


def _validate_scenarios(scenarios: Sequence[Mapping[str, Any]], modules: Mapping[str, Any]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_content: dict[str, str] = {}
    for source in scenarios:
        if not isinstance(source, Mapping):
            raise ValueError("GAIN_EVALUATION_SCENARIO_INVALID")
        scenario = copy.deepcopy(dict(source))
        split = scenario.get("split")
        if split not in {"test", "ood"}:
            raise ValueError(f"GAIN_EVALUATION_ONLY_TEST_OOD_SCENARIOS:{split}")
        for event in scenario.get("event_tape", []):
            if isinstance(event, Mapping) and event.get("kind") == "task_cancel":
                raise ValueError("GAIN_EVALUATION_TASK_CANCEL_UNSUPPORTED")
        modules["scenarios"].validate_gain_scenario(scenario)
        try:
            scenario_id = _scenario_id(scenario)
        except ValueError:
            raise
        if scenario_id in seen_ids:
            raise ValueError("GAIN_EVALUATION_SCENARIO_ID_DUPLICATE_OR_MISSING")
        seen_ids.add(scenario_id)
        content_hash = str(scenario.get("scenario_content_sha256", ""))
        if not content_hash:
            raise ValueError("GAIN_EVALUATION_SCENARIO_CONTENT_HASH_REQUIRED")
        prior_split = seen_content.get(content_hash)
        if prior_split is not None:
            raise ValueError(f"GAIN_EVALUATION_SCENARIO_CONTENT_LEAKAGE:{prior_split}:{split}")
        seen_content[content_hash] = str(split)
        validated.append(scenario)
    if not validated:
        raise ValueError("GAIN_EVALUATION_SCENARIOS_EMPTY")
    return validated


def _new_timing() -> dict[str, list[float]]:
    names = ("detection", "features", "inference", "safety", "planning", "repair", "environment_step")
    return {key: [] for name in names for key in (name, f"{name}_wall_seconds")}


def _timed(timing: dict[str, list[float]], name: str, operation: Callable[[], Any]) -> tuple[Any, float, float]:
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    value = operation()
    cpu_elapsed = max(0.0, time.process_time() - cpu_start)
    wall_elapsed = max(0.0, time.perf_counter() - wall_start)
    timing[name].append(cpu_elapsed)
    timing[f"{name}_wall_seconds"].append(wall_elapsed)
    return value, cpu_elapsed, wall_elapsed


def _stage_timing(timing: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, samples in timing.items():
        if name.endswith("_wall_seconds"):
            continue
        ordered = sorted(float(value) for value in samples)
        ordered_wall = sorted(float(value) for value in timing.get(f"{name}_wall_seconds", []))
        result[name] = {
            "calls": float(len(ordered)),
            "cpu_seconds_total": float(sum(ordered)),
            "cpu_seconds_mean": float(statistics.mean(ordered)) if ordered else 0.0,
            "cpu_seconds_p95": float(ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]) if ordered else 0.0,
            "wall_seconds_total": float(sum(ordered_wall)),
            "wall_seconds_mean": float(statistics.mean(ordered_wall)) if ordered_wall else 0.0,
            "wall_seconds_p95": float(ordered_wall[min(len(ordered_wall) - 1, math.ceil(0.95 * len(ordered_wall)) - 1)]) if ordered_wall else 0.0,
        }
    return result


def _mlp_seed_average_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seed_methods = ("GainMLP8201", "GainMLP8202", "GainMLP8203")
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("method") in seed_methods:
            grouped.setdefault(str(row["scenario_id"]), {})[str(row["method"])] = row
    averaged: list[dict[str, Any]] = []
    for scenario_id, by_seed in sorted(grouped.items()):
        if set(by_seed) != set(seed_methods):
            continue
        values = [by_seed[method] for method in seed_methods]
        averaged.append({
            "scenario_id": scenario_id, "method": "GainMLPSeedMean", "split": "test",
            "J_total": float(statistics.mean(float(row["J_total"]) for row in values)),
            "weighted_completion_rate": float(statistics.mean(float(row["weighted_completion_rate"]) for row in values)),
        })
    return averaged


def _load_runner_helpers(runner_api: Any | None, modules: Mapping[str, Any]) -> Any:
    if runner_api is not None:
        return SimpleNamespace(**dict(runner_api)) if isinstance(runner_api, Mapping) else runner_api
    helper_names = (
        "legacy_environment_scenario", "safety_receipt", "safety_scope",
        "verify_safety_disposition", "apply_live_safety_disposition", "invoke_replanner",
    )
    if all(callable(modules.get(name)) for name in helper_names):
        return SimpleNamespace(**dict(modules))
    return _runner_api()


def _run_one_episode(
    scenario: Mapping[str, Any], method: str, runtime: Any | None,
    modules: Mapping[str, Any], helpers: Any, config: Mapping[str, Any], cost_config: GainCostConfig,
    task_priorities: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pairs = modules["pairs"]
    env = modules["env_class"](helpers.legacy_environment_scenario(scenario))
    initial_public = env.reset()
    start_truth = copy.deepcopy(env._raw_truth())
    priorities = copy.deepcopy(dict(task_priorities))
    horizon = _episode_horizon(scenario, config)
    evaluator_config = config.get("evaluation", {})
    if not isinstance(evaluator_config, Mapping):
        raise ValueError("GAIN_EVALUATION_SETTINGS_INVALID")
    interval = int(evaluator_config.get("periodic_interval_minutes", 4))
    attempt_cooldown = int(evaluator_config.get("existing_et_attempt_cooldown_minutes", 4))
    risk_threshold = float(evaluator_config.get("risk_threshold", 0.5))
    heuristic_threshold = float(evaluator_config.get("heuristic_gain_threshold", 0.25))
    gate_mode = str(evaluator_config.get("gate_mode", "calibrated_lower_bound"))
    if interval < 1 or attempt_cooldown < 1 or not math.isfinite(risk_threshold) or not math.isfinite(heuristic_threshold):
        raise ValueError("GAIN_EVALUATION_METHOD_THRESHOLD_INVALID")
    if gate_mode not in {"mean", "calibrated_lower_bound"}:
        raise ValueError("GAIN_EVALUATION_GATE_MODE_INVALID")

    trigger = modules["trigger_class"]({}, {"threshold": 0.5, "delta_min": 4})
    pending_action = copy.deepcopy(pairs.NOOP_ACTION)
    next_ordinary_attempt_time = 0
    previous_action_receipt: dict[str, Any] | None = None
    decisions: list[dict[str, Any]] = []
    timing = _new_timing()
    counts = {
        "event_candidates": 0, "method_candidates": 0, "suppressed_candidates": 0,
        "cooldown_suppressed_candidates": 0,
        "forced_events": 0, "forced_handled": 0, "safety_resolved": 0,
        "safety_unresolved": 0, "planner_calls": 0, "candidate_catalog_items": 0,
        "physical_duplicate_actions": 0, "infeasible_candidates": 0,
        "strong_event_attempts": 0, "queue_safety_forced_attempts": 0,
    }
    failures: list[dict[str, Any]] = []
    safety_records: list[dict[str, Any]] = []

    try:
        while int(env.time) < horizon:
            step_started_cpu, step_started_wall = time.process_time(), time.perf_counter()
            _, _, terminated, truncated, _ = env.step(copy.deepcopy(pending_action))
            timing["environment_step"].append(max(0.0, time.process_time() - step_started_cpu))
            timing["environment_step_wall_seconds"].append(max(0.0, time.perf_counter() - step_started_wall))
            if previous_action_receipt is not None:
                reward_primitives = env._steps[-1].get("reward_primitives", {}) if getattr(env, "_steps", None) else {}
                effective = float(reward_primitives.get("reconstruction_count", 0.0)) > 0.0
                if effective:
                    trigger.record_reconstruction(int(env.time), True)
            detection_started_cpu, detection_started_wall = time.process_time(), time.perf_counter()
            current_public = pairs.model_public_observation(env)
            history = pairs._public_history(env)
            previous_public = history[-2] if len(history) >= 2 else initial_public
            if int(env.time) >= horizon or terminated or truncated:
                break
            detection = trigger.detect(previous_public, current_public, history=history[-4:])
            event_decision = trigger.decide(detection, int(env.time), mode="event")
            scope = modules["identify_scope"](current_public, detection)
            timing["detection"].append(max(0.0, time.process_time() - detection_started_cpu))
            timing["detection_wall_seconds"].append(max(0.0, time.perf_counter() - detection_started_wall))

            safety, safety_cpu, safety_wall = _timed(
                timing, "safety", lambda: helpers.safety_receipt(modules, current_public),
            )
            forced = bool(detection.get("strong_event")) or bool(safety.get("forced", safety.get("needs_forced_handling", False))) or not bool(safety.get("safe_to_suppress", False))
            if type(safety.get("safe_to_suppress")) is not bool:
                raise ValueError("GAIN_EVALUATION_SAFETY_RECEIPT_MISSING_SAFE_TO_SUPPRESS")
            counts["forced_events"] += int(forced)
            counts["strong_event_attempts"] += int(bool(detection.get("strong_event")))
            counts["queue_safety_forced_attempts"] += int(
                bool(safety.get("forced", safety.get("needs_forced_handling", False)))
                or not bool(safety.get("safe_to_suppress", False))
            )
            live_started_cpu, live_started_wall = time.process_time(), time.perf_counter()
            current_public, safety_disposition, safety_apply_receipt = helpers.apply_live_safety_disposition(
                modules, env, current_public, safety,
            )
            timing["safety"].append(max(0.0, time.process_time() - live_started_cpu))
            timing["safety_wall_seconds"].append(max(0.0, time.perf_counter() - live_started_wall))
            history = pairs._public_history(env)
            scope = modules["identify_scope"](current_public, detection)
            event_candidate = bool(event_decision.get("triggered"))
            counts["event_candidates"] += int(event_candidate)
            periodic_due = int(env.time) > 0 and int(env.time) % interval == 0
            method_candidate = forced
            score_info: dict[str, Any] | None = None
            feature_values: Mapping[str, float] | None = None
            if not forced:
                if method == "Periodic4":
                    method_candidate = periodic_due
                elif method == "ExistingET":
                    method_candidate = event_candidate
                elif method == "ExistingETAttemptCooldown":
                    method_candidate = _attempt_cooldown_candidate(
                        event_candidate, forced=False, current_time=int(env.time),
                        next_allowed_time=next_ordinary_attempt_time,
                    )
                    counts["cooldown_suppressed_candidates"] += int(event_candidate and not method_candidate)
                elif method == "RiskET":
                    method_candidate = event_candidate and _risk_score(detection) >= risk_threshold
                elif method in {"HeuristicGain", *MODEL_PATH_KEYS}:
                    method_candidate = event_candidate
                else:
                    raise ValueError(f"GAIN_EVALUATION_UNKNOWN_METHOD:{method}")
            counts["method_candidates"] += int(method_candidate and not forced)
            pending_action = copy.deepcopy(pairs.NOOP_ACTION)
            planner_receipt: dict[str, Any] | None = None
            if method_candidate and int(env.time) < horizon:
                allow_replan = forced
                if not forced and method in {"HeuristicGain", *MODEL_PATH_KEYS}:
                    feature_values, feature_cpu, feature_wall = _timed(
                        timing, "features", lambda: modules["features"].extract_public_features(current_public, detection, scope),
                    )
                    if method == "HeuristicGain":
                        gain_hat = _heuristic_gain(feature_values, detection)
                        epsilon = heuristic_threshold
                        lower_bound = gain_hat
                        model_forward_seconds = 0.0
                    else:
                        inference_started_cpu, inference_started_wall = time.process_time(), time.perf_counter()
                        prediction = runtime.predict(feature_values)
                        inference_cpu = max(0.0, time.process_time() - inference_started_cpu)
                        timing["inference"].append(inference_cpu)
                        timing["inference_wall_seconds"].append(max(0.0, time.perf_counter() - inference_started_wall))
                        gain_hat = float(prediction["gain_hat"])
                        epsilon = float(prediction["epsilon"])
                        lower_bound = prediction.get("lower_bound")
                        model_forward_seconds = float(prediction.get("model_forward_seconds", 0.0))
                    decision_started_cpu, decision_started_wall = time.process_time(), time.perf_counter()
                    gate_decision = modules["gate"].decide_gain(
                        gain_hat, epsilon, lower_bound, gate_mode=gate_mode,
                        forced=False, safe_to_suppress=bool(safety["safe_to_suppress"]),
                    )
                    gate_cpu = max(0.0, time.process_time() - decision_started_cpu)
                    timing["features"].append(gate_cpu)
                    timing["features_wall_seconds"].append(max(0.0, time.perf_counter() - decision_started_wall))
                    score_info = {
                        "gain_hat": gain_hat, "lower_bound": lower_bound,
                        "epsilon": epsilon, "model_forward_seconds": model_forward_seconds,
                        "gate_decision": copy.deepcopy(gate_decision),
                    }
                    allow_replan = bool(gate_decision["allow_replan"])
                    if not allow_replan:
                        counts["suppressed_candidates"] += 1
                elif not forced:
                    allow_replan = True

                if allow_replan:
                    selected_scope = helpers.safety_scope(safety, event_decision.get("affected_scope")) if forced else scope
                    action, planner_receipt, planner_cpu, planner_wall = helpers.invoke_replanner(
                        modules, env, selected_scope,
                        max_candidates=int(config.get("replanner", {}).get("max_candidates", 16)),
                    )
                    timing["planning"].append(float(planner_cpu))
                    timing["planning_wall_seconds"].append(float(planner_wall))
                    # The replanner adapter has already checked the selected action;
                    # a second timed check here would double-count validation CPU.
                    if not isinstance(action, Mapping):
                        raise ValueError("GAIN_EVALUATION_SELECTED_ACTION_INVALID")
                    action = copy.deepcopy(dict(action))
                    pending_action = action
                    counts["planner_calls"] += 1
                    if method == "ExistingETAttemptCooldown" and not forced:
                        next_ordinary_attempt_time = int(env.time) + attempt_cooldown
                    counts["candidate_catalog_items"] += int(planner_receipt.get("actual_candidate_count", 0))
                    counts["physical_duplicate_actions"] += int(planner_receipt.get("physical_duplicate_count", 0))
                    counts["infeasible_candidates"] += int(planner_receipt.get("infeasible_count", 0))
                    if forced:
                        counts["forced_handled"] += 1
                        if safety.get("forced") or not safety.get("safe_to_suppress", True):
                            counts["safety_resolved"] += int(bool(safety_disposition.get("resolved")))
                            counts["safety_unresolved"] += int(not bool(safety_disposition.get("resolved")))
                            if not safety_disposition.get("resolved"):
                                safety_records.append({
                                    "time": int(env.time),
                                    "reason": str(safety_disposition.get("unknown_reason") or "safety_disposition_unresolved"),
                                })
                    decision_record = {
                        "time": int(env.time), "detection": copy.deepcopy(detection),
                        "event_decision": copy.deepcopy(event_decision), "safety_report": copy.deepcopy(safety),
                        "forced": forced, "method_candidate": method_candidate,
                        "feature_values": copy.deepcopy(dict(feature_values)) if feature_values is not None else None,
                        "score": score_info, "planner_invoked": True,
                        "planner_receipt": copy.deepcopy(planner_receipt), "action": copy.deepcopy(action),
                        "safety_disposition": copy.deepcopy(safety_disposition),
                        "safety_apply_receipt": copy.deepcopy(safety_apply_receipt),
                        "public_history": copy.deepcopy(history),
                        "command_issued_at": int(env.time), "command_effective_at": int(env.time),
                    }
                    previous_action_receipt = decision_record
                else:
                    previous_action_receipt = None
                    decision_record = {
                        "time": int(env.time), "detection": copy.deepcopy(detection),
                        "event_decision": copy.deepcopy(event_decision), "safety_report": copy.deepcopy(safety),
                        "forced": forced, "method_candidate": method_candidate,
                        "feature_values": copy.deepcopy(dict(feature_values)) if feature_values is not None else None,
                        "score": score_info, "planner_invoked": False,
                        "planner_receipt": None, "action": copy.deepcopy(pairs.NOOP_ACTION),
                        "safety_disposition": copy.deepcopy(safety_disposition),
                        "safety_apply_receipt": copy.deepcopy(safety_apply_receipt),
                        "public_history": copy.deepcopy(history),
                        "command_issued_at": None, "command_effective_at": None,
                    }
                decisions.append(decision_record)
            else:
                previous_action_receipt = None
                decisions.append({
                    "time": int(env.time), "detection": copy.deepcopy(detection),
                    "event_decision": copy.deepcopy(event_decision), "safety_report": copy.deepcopy(safety),
                    "forced": forced, "method_candidate": method_candidate,
                    "feature_values": None, "score": None, "planner_invoked": False,
                    "planner_receipt": None, "action": copy.deepcopy(pairs.NOOP_ACTION),
                    "safety_disposition": copy.deepcopy(safety_disposition),
                    "safety_apply_receipt": copy.deepcopy(safety_apply_receipt),
                    "public_history": copy.deepcopy(history),
                    "command_issued_at": None, "command_effective_at": None,
                })
            if terminated or truncated:
                break
    except Exception as exc:
        failures.append({"error_type": type(exc).__name__, "reason": str(exc), "time": int(getattr(env, "time", 0))})

    trajectory = copy.deepcopy(env.export_trajectory())
    trajectory["start_truth"] = start_truth
    trajectory["initial_planner_call_count"] = counts["planner_calls"]
    planning_cpu = sum(timing["planning"])
    monitor_cpu = sum(sum(timing[name]) for name in ("detection", "features", "inference", "safety"))
    trajectory["continuation_policy_cpu_seconds"] = 0.0
    trajectory["evaluation_monitor_cpu_seconds"] = float(monitor_cpu)
    trajectory["evaluation"] = {
        "method": method, "scenario_id": _scenario_id(scenario),
        "split": str(scenario["split"]), "scenario_content_sha256": scenario["scenario_content_sha256"],
        "event_tape_sha256": _sha256(_canonical(scenario["event_tape"])),
        "method_config": {
            "periodic_interval_minutes": interval,
            "risk_threshold": risk_threshold,
            "heuristic_gain_threshold": heuristic_threshold,
            "gate_mode": gate_mode,
            "existing_et_attempt_cooldown_minutes": attempt_cooldown,
        },
        "task_priority_at_first_public_observation": copy.deepcopy(priorities),
        "decisions": decisions, "counts": counts,
        "stage_timings": _stage_timing(timing), "failure_records": failures,
        "safety_records": safety_records,
        "timing_stage_semantics": {
            "planning": "includes fixed-replanner candidate generation, selection, and nested repair/validation",
            "repair": "not timed separately because it is nested within planning; duplicate external validation is omitted",
        },
    }
    summary: dict[str, Any] = {
        "schema": "gain-closed-loop-episode/1", "method": method,
        "scenario_id": _scenario_id(scenario),
        "parent_family": str(scenario.get("parent_family", "")),
        "split": str(scenario["split"]), "condition": str(scenario.get("condition", "")),
        "scenario_content_sha256": str(scenario["scenario_content_sha256"]),
        "event_tape_sha256": trajectory["evaluation"]["event_tape_sha256"],
        "episode_status": "failed" if failures else "complete_unsafe" if safety_records else "complete",
        "horizon_minutes": int(trajectory.get("end_time", 0)),
        "forced_handled": counts["forced_handled"], "forced_events": counts["forced_events"],
        "safety_resolved": counts["safety_resolved"], "safety_unresolved": counts["safety_unresolved"],
        "failure_records": failures, "safety_records": safety_records, "counts": counts,
        "stage_timings": trajectory["evaluation"]["stage_timings"],
        "timing_stage_semantics": trajectory["evaluation"]["timing_stage_semantics"],
        "decision_regret": None,
    }
    if not failures:
        independent_score = recompute_episode_score(
            trajectory, task_priorities=priorities, planner_cpu_seconds=planning_cpu,
            cost_config=cost_config,
        )
        production_score = calculate_branch_cost(
            trajectory, task_priorities=priorities, planner_cpu_seconds=planning_cpu,
            config=cost_config,
        )
        _assert_score_parity(production_score, independent_score)
        trace_sha256 = _sha256(_canonical(trajectory))
        score_binding = {
            "schema": "gain-evaluation-score-binding/1",
            "raw_trace_sha256": trace_sha256,
            "production_score_sha256": _sha256(_canonical(production_score)),
            "independent_score_sha256": _sha256(_canonical(independent_score)),
            "scores_match": True,
            "production_score": production_score,
            "independent_score": independent_score,
        }
        trajectory["evaluation"]["score_binding"] = score_binding
        score = production_score
        summary.update({
            "J_total": float(score["J_total"]),
            "J_exec": float(score["J_exec"]), "C_replan": float(score["C_replan"]),
            "C_safety": float(score["C_safety"]),
            "weighted_completion_rate": float(score["execution_components"]["weighted_completion"]),
            "weighted_remaining_fraction": float(score["execution_components"]["weighted_remaining_fraction"]),
            "delay_minutes": float(score["execution_components"]["delay_minutes"]),
            "energy_units": float(score["execution_components"]["energy_units"]),
            "risk_units": float(score["execution_components"]["risk_units"]),
            "cost_config": score["cost_config"],
            "cost_accounting_schema": score["cost_accounting_schema"],
            "total_online_cpu_seconds": float(independent_score["online_cpu_seconds"]),
            "score_binding_sha256": _sha256(_canonical(score_binding)),
            "episode_metrics": copy.deepcopy(trajectory.get("summary", {})),
        })
    return summary, trajectory


def run_closed_loop_evaluation(
    scenarios: Sequence[Mapping[str, Any]],
    model_bundle_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    config: Mapping[str, Any],
    runtime_loader: Callable[[str | Path], Any] | None = None,
    *,
    evaluation_modules: Mapping[str, Any] | None = None,
    runner_api: Any | None = None,
) -> dict[str, Any]:
    """Evaluate frozen methods; raw environment trajectories are gzip streamed."""
    if not isinstance(config, Mapping):
        raise TypeError("config must be a JSON-like mapping")
    modules = dict(evaluation_modules or load_evaluation_modules())
    required = {"pairs", "scenarios", "features", "gate", "env_class", "trigger_class", "identify_scope"}
    missing = sorted(required - set(modules))
    if missing:
        raise ValueError(f"GAIN_EVALUATION_MODULES_MISSING:{','.join(missing)}")
    checked_scenarios = _validate_scenarios(scenarios, modules)
    helpers = _load_runner_helpers(runner_api, modules)
    loader = runtime_loader or gain_models.load_runtime
    model_bundle_paths = dict(model_bundle_paths)
    required_model_keys = set(MODEL_PATH_KEYS.values())
    missing_paths = sorted(required_model_keys - set(model_bundle_paths))
    if missing_paths:
        raise ValueError(f"GAIN_EVALUATION_FROZEN_MODEL_PATHS_MISSING:{','.join(missing_paths)}")

    runtimes: dict[str, Any] = {}
    model_receipts: dict[str, dict[str, Any]] = {}
    for method, key in MODEL_PATH_KEYS.items():
        path = model_bundle_paths[key]
        bundle_path = Path(path)
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        runtime = loader(path)
        cpu_seconds = max(0.0, time.process_time() - cpu_start)
        wall_seconds = max(0.0, time.perf_counter() - wall_start)
        if not callable(getattr(runtime, "predict", None)):
            raise ValueError(f"GAIN_EVALUATION_RUNTIME_PREDICT_REQUIRED:{key}")
        runtimes[method] = runtime
        model_receipts[method] = {
            "bundle_path": str(path),
            "bundle_sha256": _file_sha256(bundle_path) if bundle_path.is_file() else None,
            "bundle_size_bytes": bundle_path.stat().st_size if bundle_path.is_file() else None,
            "cold_load_cpu_seconds": cpu_seconds,
            "cold_load_wall_seconds": wall_seconds,
            "runtime_reported_cold_load_seconds": float(getattr(runtime, "cold_load_seconds", wall_seconds)),
        }

    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    episodes_path = target / "EPISODES.jsonl"
    trajectories_path = target / "TRAJECTORIES.jsonl.gz"
    if episodes_path.exists() or trajectories_path.exists():
        raise FileExistsError("GAIN_EVALUATION_OUTPUT_FILES_ALREADY_EXIST")
    cost_config = _cost_config(config)
    rows: list[dict[str, Any]] = []
    observer_receipts: dict[str, dict[str, Any]] = {}
    resource_check = getattr(helpers, "resource_check", None)
    resource_check_count = 0

    def check_resources() -> None:
        nonlocal resource_check_count
        if callable(resource_check):
            resource_check()
            resource_check_count += 1

    with episodes_path.open("x", encoding="utf-8", newline="\n") as episode_file, gzip.open(
        trajectories_path, "xt", encoding="utf-8", newline="\n",
    ) as trajectory_file:
        for scenario in checked_scenarios:
            check_resources()
            priorities, observer_trace = _observe_task_priorities(scenario, modules, helpers)
            scenario_id = _scenario_id(scenario)
            priority_sha = _sha256(_canonical(priorities))
            observer_receipts[scenario_id] = {
                "task_priority_sha256": priority_sha,
                "task_count": len(priorities),
                "observer_trajectory_sha256": _sha256(_canonical(observer_trace)),
            }
            trajectory_file.write(json.dumps(observer_trace, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
            trajectory_file.flush()
            for method in METHODS:
                runtime = runtimes.get(method)
                summary, trajectory = _run_one_episode(
                    scenario, method, runtime, modules, helpers, config, cost_config, priorities,
                )
                episode_file.write(json.dumps(summary, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
                trajectory_file.write(json.dumps(trajectory, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
                episode_file.flush()
                trajectory_file.flush()
                rows.append(summary)
                check_resources()
    result = {
        "schema": EVALUATION_SCHEMA,
        "status": "complete" if all("J_total" in row for row in rows) else "complete_with_episode_failures",
        "method_names": list(METHODS), "scenario_count": len(checked_scenarios),
        "episode_count": len(rows), "scenario_ids": [_scenario_id(row) for row in checked_scenarios],
        "scenario_input_sha256": _sha256(_canonical(checked_scenarios)),
        "configuration_sha256": _sha256(_canonical(config)),
        "resource_check_count": resource_check_count,
        "baseline_contract": {
            "ExistingET": "preserves event-trigger cooldown based on effective allocation change",
            "ExistingETAttemptCooldown": {
                "ordinary_attempt_cooldown_minutes": int(config.get("evaluation", {}).get(
                    "existing_et_attempt_cooldown_minutes", 4,
                )),
                "starts_after_every_formal_replanner_attempt": True,
                "forced_safety_bypasses_cooldown": True,
            },
        },
        "priority_observer_receipts": observer_receipts,
        "model_load_receipts": model_receipts,
        "scenario_content_sha256": {_scenario_id(s): str(s["scenario_content_sha256"]) for s in checked_scenarios},
        "event_tape_sha256": {_scenario_id(s): _sha256(_canonical(s["event_tape"])) for s in checked_scenarios},
        "episode_results": rows,
        "files": {
            episodes_path.name: {"size_bytes": episodes_path.stat().st_size, "sha256": _file_sha256(episodes_path)},
            trajectories_path.name: {"size_bytes": trajectories_path.stat().st_size, "sha256": _file_sha256(trajectories_path)},
        },
    }
    score_rows = [row for row in rows if "J_total" in row]
    result["panel_analysis_status"] = "not_run_incomplete_panel"
    result["method_support"] = {
        method: {"status": "INCONCLUSIVE", "reason": "test_panel_not_analyzed"}
        for method in METHODS if method not in BASELINES
    }
    if len(score_rows) != len(rows):
        unscored = [
            {"scenario_id": row.get("scenario_id"), "method": row.get("method"),
             "episode_status": row.get("episode_status")}
            for row in rows if "J_total" not in row
        ]
        result["panel_analysis_status"] = "rejected_incomplete_panel"
        result["panel_analysis_error"] = json.dumps(unscored, sort_keys=True, ensure_ascii=True)
    else:
        try:
            try:
                from . import gain_analysis
            except ImportError:
                import gain_analysis
            candidate_methods = [method for method in METHODS if method not in BASELINES]
            comparisons = [
                {"baseline": baseline, "method": method}
                for baseline in BASELINES for method in candidate_methods
            ]
            comparisons.append({"baseline": "ExistingET", "method": "ExistingETAttemptCooldown"})
            panel_reports = {}
            for split in ("test", "ood"):
                split_rows = [row for row in score_rows if row["split"] == split]
                expected_ids = sorted(_scenario_id(s) for s in checked_scenarios if s["split"] == split)
                if not expected_ids:
                    continue
                panel_reports[split] = gain_analysis.paired_analysis(
                    split_rows, comparisons, ("J_total", "weighted_completion_rate"),
                    bootstrap_replicates=int(config.get("statistics", {}).get("bootstrap_replicates", 10_000)),
                    permutation_replicates=int(config.get("statistics", {}).get("permutation_replicates", 10_000)),
                    seed=8201, expected_scenario_ids=expected_ids,
                )
            result["panel_analysis"] = panel_reports
            result["panel_analysis_status"] = "complete"
            result["analysis_contract"] = {
                "primary_split": "test",
                "primary_endpoints": ["J_total", "weighted_completion_rate"],
                "primary_comparison_family": (
                    "each non-baseline method versus ExistingET and ExistingETAttemptCooldown "
                    "across J_total and weighted_completion_rate, plus the attempt-cooldown "
                    "versus original ExistingET baseline contrast"
                ),
                "confidence_interval_semantics": (
                    "paired scenario-instance bootstrap marginal 95% CIs; completion noninferiority "
                    "uses the unadjusted lower CI bound, and support additionally requires J_total "
                    "Holm-adjusted p<0.05 and upper CI<0 against both ET baselines"
                ),
                "ood_claim_limit": "separately reported; not used for primary support status",
            }
            test_report = panel_reports.get("test")
            if test_report is not None:
                comparison_lookup = {item["comparison_id"]: item for item in test_report["comparisons"]}
                margin_pp = float(config.get("statistics", {}).get(
                    "completion_noninferiority_margin_percentage_points", -2.0,
                ))
                completion_floor = margin_pp / 100.0
                result["analysis_contract"]["primary_holm_family_size"] = test_report["holm_family_size"]
                for method in result["method_support"]:
                    baseline_comparisons = [
                        (
                            comparison_lookup.get(f"{method}__vs__{baseline}__J_total"),
                            comparison_lookup.get(f"{method}__vs__{baseline}__weighted_completion_rate"),
                            baseline,
                        )
                        for baseline in BASELINES
                    ]
                    if any(cost is None or completion is None for cost, completion, _ in baseline_comparisons):
                        continue
                    method_rows = [row for row in score_rows if row["split"] == "test" and row["method"] == method]
                    safety_pass = all(
                        row.get("safety_unresolved", 0) == 0
                        and row.get("forced_handled", 0) == row.get("forced_events", 0)
                        for row in method_rows
                    )
                    criteria = []
                    for cost, completion, baseline in baseline_comparisons:
                        cost_ci = cost["paired_difference_ci95"]
                        completion_ci = completion["paired_difference_ci95"]
                        criteria.append({
                            "baseline": baseline,
                            "J_total_ci95": cost_ci,
                            "J_total_holm_adjusted_p": cost["holm_adjusted_p"],
                            "completion_ci95": completion_ci,
                            "completion_noninferiority_floor": completion_floor,
                            "supported_vs_baseline": (
                                cost_ci[1] < 0.0
                                and float(cost["holm_adjusted_p"]) < 0.05
                                and completion_ci[0] > completion_floor
                            ),
                            "evidence_against_vs_baseline": (
                                cost_ci[0] > 0.0 or completion_ci[1] < completion_floor
                            ),
                        })
                    if not safety_pass:
                        status, reason = "INCONCLUSIVE", "unresolved_or_unhandled_safety"
                    elif all(row["supported_vs_baseline"] for row in criteria):
                        status, reason = "SUPPORTED", "completion_noninferiority_and_total_cost_superiority"
                    elif any(row["evidence_against_vs_baseline"] for row in criteria):
                        status, reason = "NOT_SUPPORTED", "evidence_against_prespecified_success_criteria"
                    else:
                        status, reason = "INCONCLUSIVE", "CIs_or_Holm_adjusted_p_do_not_resolve_success_criteria"
                    result["method_support"][method] = {
                        "status": status, "reason": reason, "safety_pass": safety_pass,
                        "required_baselines": list(BASELINES), "baseline_criteria": criteria,
                    }
                test_score_rows = [row for row in score_rows if row["split"] == "test"]
                mlp_average_rows = _mlp_seed_average_rows(test_score_rows)
                result["mlp_seed_average_analysis"] = {
                    "claim_limit": "derived mean across all three prespecified MLP training seeds; not a deployed ensemble or a best-seed selection",
                    "analysis": gain_analysis.paired_analysis(
                        [*test_score_rows, *mlp_average_rows],
                        [{"baseline": baseline, "method": "GainMLPSeedMean"} for baseline in BASELINES],
                        ("J_total", "weighted_completion_rate"),
                        bootstrap_replicates=int(config.get("statistics", {}).get("bootstrap_replicates", 10_000)),
                        permutation_replicates=int(config.get("statistics", {}).get("permutation_replicates", 10_000)),
                        seed=8202, expected_scenario_ids=sorted(
                            _scenario_id(s) for s in checked_scenarios if s["split"] == "test"
                        ),
                    ),
                }
        except ValueError as exc:
            result["panel_analysis_status"] = "rejected_incomplete_panel"
            result["panel_analysis_error"] = str(exc)
    manifest_path = target / "EVALUATION.json"
    if manifest_path.exists():
        raise FileExistsError("GAIN_EVALUATION_MANIFEST_ALREADY_EXISTS")
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    result["output_dir"] = str(target)
    return result
