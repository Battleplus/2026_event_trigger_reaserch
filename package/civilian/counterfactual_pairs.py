"""Persisted, paired keep/replan branches for gain-gate labels.

The serializer records the complete Python simulator state using an explicit
JSON codec.  It is private label-generation data; policies receive only the
current public observation after identity metadata is removed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .cost_accounting import DEFAULT_CONFIG, GainCostConfig, calculate_branch_cost, task_priorities_from_truth
except ImportError:
    from cost_accounting import DEFAULT_CONFIG, GainCostConfig, calculate_branch_cost, task_priorities_from_truth


SNAPSHOT_SCHEMA = "gain-counterfactual-snapshot/1"
SUFFIX_SCHEMA = "gain-counterfactual-suffix-snapshot/1"
TRACE_SCHEMA = "gain-counterfactual-trace/1"
PAIR_SCHEMA = "gain-counterfactual-pair/1"
LABEL_SCHEMA = "gain-counterfactual-label/2"
CONTINUATION_RECEIPT_SCHEMA = "gain-continuation-receipt/1"
CONTINUATION_SAFETY_FIELDS = ("safety_report", "safety_disposition", "unknown_reason", "detection")
_SUPPORTED_ENV_CLASSES = {"EventReliefEnv", "ReliefEnv"}
NOOP_ACTION = {
    "action_type": "reconstruct", "scope": "NOOP", "noop": True,
    "queue_operations": [], "dispatches": [],
}


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _encode(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("snapshot refuses non-finite simulator state")
        return value
    if isinstance(value, defaultdict):
        return {"__gain_type__": "defaultdict", "items": _encode(dict(value))}
    if isinstance(value, dict):
        # Simulator ``__dict__`` objects may contain bound helper methods or
        # callbacks. They are executable behavior, not mutable simulator
        # state, and cannot be restored from JSON. Exclude them at the state
        # boundary while retaining every serializable state field.
        pairs = [[_encode(key), _encode(item)] for key, item in value.items()
                 if not callable(item)]
        pairs.sort(key=lambda row: _canonical(row[0]))
        return {"__gain_type__": "dict", "items": pairs}
    if isinstance(value, tuple):
        return {"__gain_type__": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, set):
        rows = [_encode(item) for item in value]
        rows.sort(key=_canonical)
        return {"__gain_type__": "set", "items": rows}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if value.__class__.__name__ in _SUPPORTED_ENV_CLASSES and hasattr(value, "__dict__"):
        return {
            "__gain_type__": "simulator_object",
            "class_name": value.__class__.__name__,
            "module_name": value.__class__.__module__.rsplit(".", 1)[-1],
            "state": _encode(copy.deepcopy(value.__dict__)),
        }
    raise TypeError(f"unsupported simulator snapshot type: {type(value).__name__}")


def _resolve_env_class(class_name: str, module_name: str):
    if class_name not in _SUPPORTED_ENV_CLASSES or module_name not in {"event_environment", "relief_env"}:
        raise ValueError("snapshot contains an unsupported simulator class")
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ImportError:
        module = __import__(f".{module_name}", globals(), locals(), [class_name], 1)
    cls = getattr(module, class_name, None)
    if cls is None or cls.__name__ != class_name:
        raise ValueError("simulator class is unavailable for snapshot restore")
    return cls


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    marker = value.get("__gain_type__")
    if marker == "dict":
        return {_decode(key): _decode(item) for key, item in value["items"]}
    if marker == "defaultdict":
        return defaultdict(float, _decode(value["items"]))
    if marker == "tuple":
        return tuple(_decode(item) for item in value["items"])
    if marker == "set":
        return set(_decode(item) for item in value["items"])
    if marker == "simulator_object":
        cls = _resolve_env_class(str(value.get("class_name")), str(value.get("module_name")))
        instance = cls.__new__(cls)
        state = _decode(value.get("state"))
        if not isinstance(state, dict):
            raise ValueError("simulator object state is malformed")
        instance.__dict__.update(state)
        return instance
    raise ValueError("snapshot contains an unknown JSON type marker")


def _model_public_view(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return observations without private run identities or content hashes."""
    result = copy.deepcopy(dict(observation))
    result.pop("provenance", None)
    return result


def model_public_observation(env: Any) -> dict[str, Any]:
    """Public projection suitable for features; excludes provenance hashes."""
    observation = env.public_observation()
    if not isinstance(observation, dict):
        raise ValueError("simulator public observation must be an object")
    return _model_public_view(observation)


def _public_history(env: Any) -> list[dict[str, Any]]:
    history = getattr(env, "_public_history", [])
    if not isinstance(history, list):
        raise ValueError("simulator public history is malformed")
    return [_model_public_view(row) for row in history]


def _environment_scenario(env: Any) -> dict[str, Any]:
    scenario = getattr(env, "scenario", None)
    if not isinstance(scenario, dict):
        raise ValueError("simulator must expose its scenario for snapshot binding")
    return scenario


def _snapshot_payload(snapshot_json: str, expected_schema: str | None = None) -> dict[str, Any]:
    if not isinstance(snapshot_json, str):
        raise ValueError("snapshot must be JSON text")
    try:
        payload = json.loads(snapshot_json)
    except json.JSONDecodeError as exc:
        raise ValueError("snapshot JSON is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema") not in {SNAPSHOT_SCHEMA, SUFFIX_SCHEMA}:
        raise ValueError("snapshot schema is unsupported")
    if expected_schema and payload.get("schema") != expected_schema:
        raise ValueError("snapshot schema does not match the requested operation")
    expected = payload.get("snapshot_sha256")
    if expected != _digest({key: value for key, value in payload.items() if key != "snapshot_sha256"}):
        raise ValueError("snapshot envelope hash mismatch")
    state = payload.get("state")
    if not isinstance(state, dict) or payload.get("state_sha256") != _digest(state):
        raise ValueError("snapshot state hash mismatch")
    return payload


def _scenario_digest(scenario: Mapping[str, Any]) -> str:
    return _digest(scenario)


def _make_snapshot_json(payload: dict[str, Any]) -> str:
    payload.pop("snapshot_sha256", None)
    payload["state_sha256"] = _digest(payload["state"])
    payload["snapshot_sha256"] = _digest(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"


def capture_snapshot(
    env: Any,
    *,
    state_id: str,
    scenario_id: str,
    parent_family: str,
    split: str,
    event_type: str,
    forced: bool,
    replanner_version: str,
) -> str:
    """Capture complete mutable simulator state before planner invocation."""
    strings = (state_id, scenario_id, parent_family, split, event_type, replanner_version)
    if any(not isinstance(value, str) or not value for value in strings):
        raise ValueError("snapshot identity strings must be nonempty")
    if split not in {"train", "validation", "test", "ood", "smoke", "pilot"}:
        raise ValueError("snapshot split is not recognized")
    if type(forced) is not bool:
        raise ValueError("forced marker must be boolean")
    scenario = copy.deepcopy(_environment_scenario(env))
    if scenario.get("parent_id") != scenario_id:
        raise ValueError("snapshot scenario identity mismatch")
    if getattr(env, "_has_reset", True) is not True:
        raise ValueError("snapshot requires a reset, active simulator")
    if not hasattr(env, "_raw_truth") or not hasattr(env, "_steps"):
        raise ValueError("simulator does not expose required complete-state evidence")
    truth = copy.deepcopy(env._raw_truth())
    priorities = task_priorities_from_truth(truth)
    known_at_start = {
        str(row["id"]): bool(row.get("known"))
        for row in truth["tasks"]
    }
    status_at_start = {
        str(row["id"]): str(row.get("status", "unknown"))
        for row in truth["tasks"]
    }
    public = model_public_observation(env)
    history = _public_history(env)
    state = _encode(copy.deepcopy(env.__dict__))
    tape = copy.deepcopy(scenario.get("event_tape", []))
    payload = {
        "schema": SNAPSHOT_SCHEMA,
        "state_id": state_id,
        "scenario_id": scenario_id,
        "parent_family": parent_family,
        "split": split,
        "event_type": event_type,
        "forced": forced,
        "replanner_version": replanner_version,
        "time": int(getattr(env, "time")),
        "public_observation": public,
        "public_observation_sha256": _digest(public),
        "public_history_sha256": _digest(history),
        "private_snapshot_sha256": _digest(state),
        "scenario_sha256": _scenario_digest(scenario),
        "event_tape_id": str(scenario.get("event_tape_id", "")),
        "event_tape_sha256": _digest(tape),
        "task_priorities": priorities,
        "task_known_at_start": known_at_start,
        "task_status_at_start": status_at_start,
        "state": state,
    }
    return _make_snapshot_json(payload)


def _instantiate_root(env_factory: Callable[..., Any] | None, state: dict[str, Any]) -> Any:
    scenario = copy.deepcopy(state.get("scenario"))
    if not isinstance(scenario, dict):
        raise ValueError("snapshot simulator scenario is missing")
    factory = env_factory
    if factory is None:
        try:
            from .event_environment import EventReliefEnv
        except ImportError:
            from event_environment import EventReliefEnv
        factory = EventReliefEnv
    config = copy.deepcopy(state.get("reward_config"))
    try:
        fresh = factory(copy.deepcopy(scenario), copy.deepcopy(config))
    except TypeError:
        fresh = factory(copy.deepcopy(scenario))
    if fresh.__class__.__name__ not in _SUPPORTED_ENV_CLASSES:
        raise ValueError("snapshot factory returned an unsupported simulator")
    return fresh


def restore_snapshot(snapshot_json: str, env_factory: Callable[..., Any] | None = None) -> Any:
    """Restore and verify a fresh simulator from a complete JSON envelope."""
    payload = _snapshot_payload(snapshot_json)
    state = _decode(payload["state"])
    if not isinstance(state, dict):
        raise ValueError("snapshot root state must be an object")
    env = _instantiate_root(env_factory, state)
    env.__dict__.clear()
    env.__dict__.update(state)
    scenario = _environment_scenario(env)
    if _scenario_digest(scenario) != payload.get("scenario_sha256"):
        raise ValueError("snapshot scenario digest mismatch")
    if _digest(_encode(copy.deepcopy(env.__dict__))) != payload.get("state_sha256"):
        raise ValueError("snapshot restored state digest mismatch")
    if int(getattr(env, "time", -1)) != int(payload.get("time", -2)):
        raise ValueError("snapshot time mismatch")
    public = model_public_observation(env)
    history = _public_history(env)
    if public != payload.get("public_observation") or _digest(public) != payload.get("public_observation_sha256"):
        raise ValueError("snapshot public observation mismatch")
    if _digest(history) != payload.get("public_history_sha256"):
        raise ValueError("snapshot public history mismatch")
    if _digest(scenario.get("event_tape", [])) != payload.get("event_tape_sha256"):
        raise ValueError("snapshot event tape mismatch")
    return env


def _derived_base_disturbances(event_tape: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in event_tape:
        if row.get("kind") == "platform_failure":
            rows.append({
                "id": row["event_id"], "time": row["time"], "kind": "failure",
                "report_delay": 0, "platform_id": row["platform_id"],
            })
        elif row.get("kind") == "energy_drop":
            rows.append({
                "id": row["event_id"], "time": row["time"], "kind": "energy_change",
                "report_delay": 0, "platform_id": row["platform_id"], "value": row["new_energy"],
            })
    return rows


def prepare_suffix_snapshot(
    snapshot_json: str,
    *,
    future_events: Sequence[Mapping[str, Any]],
    suffix_seed: int,
    env_factory: Callable[..., Any] | None = None,
) -> str:
    """Bind a sampled future suffix to an immutable already-observed prefix.

    Sampling is deliberately delegated to the frozen scenario generator.  This
    function validates the sampled suffix, replaces future tape rows only, and
    makes two freshly restored siblings share the resulting tape byte-for-byte.
    """
    if type(suffix_seed) is not int or suffix_seed < 0:
        raise ValueError("suffix seed must be a nonnegative integer")
    source = _snapshot_payload(snapshot_json, SNAPSHOT_SCHEMA)
    env = restore_snapshot(snapshot_json, env_factory)
    now = int(source["time"])
    if not isinstance(future_events, Sequence) or isinstance(future_events, (str, bytes)):
        raise ValueError("future event suffix must be a sequence")
    suffix = [copy.deepcopy(dict(row)) for row in future_events]
    if any(type(row.get("time")) is not int or int(row["time"]) <= now for row in suffix):
        raise ValueError("sampled suffix may contain only events after the observed prefix")
    prior_tape = list(env.scenario.get("event_tape", []))
    prefix = [copy.deepcopy(row) for row in prior_tape if int(row["time"]) <= now]
    prefix_ids = {str(row["event_id"]) for row in prefix}
    suffix_ids = [str(row.get("event_id", "")) for row in suffix]
    if any(not value for value in suffix_ids) or prefix_ids.intersection(suffix_ids) or len(suffix_ids) != len(set(suffix_ids)):
        raise ValueError("sampled suffix event identities collide or are missing")
    tape = prefix + sorted(suffix, key=lambda row: (int(row["time"]), str(row["event_id"])))
    try:
        try:
            from .event_scenarios import _digest as event_digest, validate_event_scenario
            from .event_environment import validate_supported_event_scenario
        except ImportError:
            from event_scenarios import _digest as event_digest, validate_event_scenario
            from event_environment import validate_supported_event_scenario
        env.scenario["event_tape"] = tape
        env.scenario["event_tape_id"] = event_digest(tape)
        validate_supported_event_scenario(env.scenario)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"sampled suffix violates the frozen event scenario contract: {exc}") from exc
    env._base.scenario["disturbances"] = (
        copy.deepcopy(env.scenario["base_scenario"].get("disturbances", []))
        + _derived_base_disturbances(tape)
    )
    if _digest(model_public_observation(env)) != source["public_observation_sha256"]:
        raise ValueError("future suffix changed the current public observation")
    if _digest(_public_history(env)) != source["public_history_sha256"]:
        raise ValueError("future suffix changed the observed public history")
    new_state = _encode(copy.deepcopy(env.__dict__))
    payload = {
        key: copy.deepcopy(value) for key, value in source.items()
        if key not in {"snapshot_sha256", "state_sha256", "schema", "state", "private_snapshot_sha256",
                       "scenario_sha256", "event_tape_id", "event_tape_sha256"}
    }
    payload.update({
        "schema": SUFFIX_SCHEMA,
        "source_snapshot_sha256": source["snapshot_sha256"],
        "suffix_seed": suffix_seed,
        "suffix_rows_sha256": _digest(suffix),
        "state": new_state,
        "private_snapshot_sha256": _digest(new_state),
        "scenario_sha256": _scenario_digest(env.scenario),
        "event_tape_id": str(env.scenario["event_tape_id"]),
        "event_tape_sha256": _digest(tape),
    })
    return _make_snapshot_json(payload)


def noop_continuation(_env: Any) -> dict[str, Any]:
    """Default continuation: committed queues keep running automatically."""
    return copy.deepcopy(NOOP_ACTION)


def _continuation_safety_evidence(result: Mapping[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for field in ("safety_report", "safety_disposition", "detection"):
        if field not in result:
            continue
        value = result[field]
        if not isinstance(value, Mapping):
            raise ValueError(f"continuation receipt {field} must be an object")
        evidence[field] = copy.deepcopy(dict(value))

    if "unknown_reason" in result:
        reason = result["unknown_reason"]
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError("continuation receipt unknown_reason must be null or nonempty text")
        evidence["unknown_reason"] = reason

    disposition = evidence.get("safety_disposition")
    if disposition is not None:
        resolved = disposition.get("resolved")
        if type(resolved) is not bool:
            raise ValueError("continuation receipt safety_disposition resolved must be boolean")
        disposition_reason = disposition.get("unknown_reason")
        if disposition_reason is not None and (not isinstance(disposition_reason, str) or not disposition_reason.strip()):
            raise ValueError("continuation receipt safety_disposition unknown_reason must be nonempty text")
        if resolved is False and not evidence.get("unknown_reason"):
            evidence["unknown_reason"] = disposition_reason or "safety_disposition_unresolved"
        elif disposition_reason and evidence.get("unknown_reason") != disposition_reason:
            raise ValueError("continuation receipt safety unknown reasons disagree")
    return evidence


def _continuation_action_receipt(
    env: Any,
    continuation_policy: Callable[[Any], Mapping[str, Any]],
    *,
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any], float, float]:
    """Run continuation logic and separate actual planner CPU from safety CPU.

    Legacy callbacks return an action only; their measured CPU is conservatively
    charged to safety monitoring. New callbacks may return a structured receipt.
    """
    before = time.process_time()
    result = continuation_policy(env)
    measured_cpu = max(0.0, time.process_time() - before)
    if not isinstance(result, Mapping):
        raise ValueError("continuation policy must return an action or structured receipt")
    explicit = result.get("receipt_schema") == CONTINUATION_RECEIPT_SCHEMA
    if explicit:
        action = result.get("action")
        invoked = result.get("planner_invoked")
        planner_cpu = result.get("planner_cpu_seconds", 0.0)
        declared_safety_cpu = result.get("safety_monitor_cpu_seconds", 0.0)
        if type(invoked) is not bool:
            raise ValueError("continuation receipt planner_invoked must be boolean")
        for name, value in (("planner_cpu_seconds", planner_cpu), ("safety_monitor_cpu_seconds", declared_safety_cpu)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"continuation receipt {name} must be finite and nonnegative")
        planner_cpu = float(planner_cpu)
        declared_safety_cpu = float(declared_safety_cpu)
        if not invoked and planner_cpu > 0:
            raise ValueError("continuation receipt cannot charge planner CPU without a planner call")
        if planner_cpu + declared_safety_cpu > measured_cpu + 1e-6:
            raise ValueError("continuation receipt CPU breakdown exceeds measured callback CPU")
        safety_cpu = declared_safety_cpu + max(0.0, measured_cpu - planner_cpu - declared_safety_cpu)
        call_type = "continuation_safety_replan" if invoked else "safety_monitor"
        policy_version = result.get("policy_version")
        if policy_version is not None and (not isinstance(policy_version, str) or not policy_version):
            raise ValueError("continuation receipt policy_version must be a nonempty string")
    elif "receipt_schema" in result:
        raise ValueError("continuation receipt schema is unsupported")
    else:
        action = dict(result)
        invoked = not _is_noop(action)
        planner_cpu = 0.0
        safety_cpu = measured_cpu
        call_type = "continuation_safety_replan" if invoked else "safety_monitor"
        policy_version = None
    if not isinstance(action, Mapping):
        raise ValueError("continuation receipt must include a full environment action")
    action = copy.deepcopy(dict(action))
    safety_evidence = _continuation_safety_evidence(result) if explicit else {}
    receipt = {
        "receipt_schema": CONTINUATION_RECEIPT_SCHEMA,
        "stage": stage,
        "call_type": call_type,
        "planner_invoked": invoked,
        "policy_version": policy_version,
        "action_sha256": _digest(action),
        "action": copy.deepcopy(action),
        "planner_cpu_seconds": float(planner_cpu),
        "safety_monitor_cpu_seconds": float(safety_cpu),
        "measured_callback_cpu_seconds": float(measured_cpu),
    }
    if safety_evidence:
        receipt.update(safety_evidence)
        receipt["safety_evidence_sha256"] = _digest(safety_evidence)
    return action, receipt, float(planner_cpu), float(safety_cpu)


def _validated_action(env: Any, action: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(action, Mapping):
        raise ValueError("environment action must be an object")
    normalized, issues = env._check_action(copy.deepcopy(dict(action)))
    if not isinstance(normalized, dict) or not isinstance(issues, list):
        raise ValueError("environment action validator returned malformed data")
    return copy.deepcopy(normalized), list(issues)


def _is_noop(action: Mapping[str, Any]) -> bool:
    return bool(action.get("noop")) and not action.get("queue_operations") and not action.get("dispatches")


def _step_one(env: Any, action: Mapping[str, Any], *, planner_invoked: bool, stage: str) -> tuple[dict[str, Any], float, bool]:
    normalized, issues = _validated_action(env, action)
    if issues:
        raise ValueError(f"illegal {stage} action: {sorted(set(issues))}")
    before = len(env._steps)
    _, _, terminated, truncated, info = env.step(normalized)
    if len(env._steps) != before + 1:
        raise ValueError("environment did not persist exactly one step")
    step = copy.deepcopy(env._steps[-1])
    step["planner_invoked"] = bool(planner_invoked)
    step["stage"] = stage
    return step, float(info.get("reward", 0.0)) if isinstance(info, Mapping) else 0.0, bool(terminated or truncated)


def _trace_for_branch(
    env: Any,
    *,
    snapshot: Mapping[str, Any],
    suffix_snapshot: Mapping[str, Any],
    arm: str,
    planned_action: Mapping[str, Any],
    steps: list[dict[str, Any]],
    start_truth: dict[str, Any],
    planner_cpu_seconds: float,
    action_delay_ticks: int,
    action_applied_step: int | None,
    action_expired_reason: str | None,
    continuation_cpu_seconds: float,
    continuation_planner_cpu_seconds: float,
    continuation_receipts: Sequence[Mapping[str, Any]],
    horizon_minutes: int,
    collection_error: Exception | None = None,
) -> dict[str, Any]:
    progress_contract = _task_progress_contract(env)
    unresolved_receipts = [
        row for row in continuation_receipts
        if isinstance(row.get("safety_disposition"), Mapping)
        and row["safety_disposition"].get("resolved") is False
    ]
    trace = {
        "schema": TRACE_SCHEMA,
        "arm": arm,
        "state_id": snapshot["state_id"],
        "scenario_id": snapshot["scenario_id"],
        "parent_family": snapshot["parent_family"],
        "split": snapshot["split"],
        "event_type": snapshot["event_type"],
        "forced": snapshot["forced"],
        "replanner_version": snapshot["replanner_version"],
        "source_snapshot_sha256": snapshot["snapshot_sha256"],
        "public_history_sha256": snapshot["public_history_sha256"],
        "event_tape_id": suffix_snapshot["event_tape_id"],
        "event_tape_sha256": suffix_snapshot["event_tape_sha256"],
        "suffix_seed": suffix_snapshot["suffix_seed"],
        "start_time": int(snapshot["time"]),
        "end_time": int(env.time),
        "horizon_requested_minutes": int(horizon_minutes),
        "horizon_actual_minutes": int(env.time) - int(snapshot["time"]),
        "planned_action": copy.deepcopy(dict(planned_action)),
        "planned_action_sha256": _digest(dict(planned_action)),
        "action_delay_ticks": int(action_delay_ticks),
        "action_applied_step": action_applied_step,
        "action_expired_reason": action_expired_reason,
        "initial_planner_call_count": int(arm == "replan"),
        "initial_planner_cpu_seconds": float(planner_cpu_seconds if arm == "replan" else 0.0),
        "continuation_policy_cpu_seconds": float(continuation_cpu_seconds),
        "continuation_planner_cpu_seconds": float(continuation_planner_cpu_seconds),
        "continuation_policy_receipts": copy.deepcopy(list(continuation_receipts)),
        "planner_call_receipts": [
            copy.deepcopy(dict(row)) for row in continuation_receipts if row.get("planner_invoked") is True
        ],
        "unresolved_safety_disposition_count": len(unresolved_receipts),
        "unresolved_safety_unknown_reasons": [str(row.get("unknown_reason")) for row in unresolved_receipts],
        "start_truth": copy.deepcopy(start_truth),
        "raw_truth": copy.deepcopy(env._raw_truth()),
        "task_progress_contract": progress_contract,
        "task_progress_contract_sha256": _digest(progress_contract),
        "steps": steps,
        "valid_mask": collection_error is None,
        "unknown_reason": None if collection_error is None else "branch_execution_failed",
    }
    if collection_error is not None:
        trace["collection_error"] = {
            "error_type": type(collection_error).__name__,
            "message": str(collection_error),
        }
    trace["trace_sha256"] = _digest(trace)
    return trace


def _priority_at_arrival(
    task_id: str,
    arrival_time: int,
    *,
    snapshot: Mapping[str, Any],
    suffix_snapshot: Mapping[str, Any],
    traces: Sequence[Mapping[str, Any]],
) -> float:
    """Use the task's public priority when the arrival first enters observation."""
    for trace in traces:
        for step in trace.get("steps", []):
            next_observation = step.get("next_public_observation", {})
            if int(next_observation.get("time", -1)) != arrival_time:
                continue
            task = next((row for row in next_observation.get("tasks", []) if row.get("id") == task_id), None)
            if task is not None and isinstance(task.get("priority"), (int, float)) and not isinstance(task.get("priority"), bool):
                return float(task["priority"])
    state = _decode(suffix_snapshot["state"])
    tape = state["scenario"].get("event_tape", [])
    prior = float(snapshot["task_priorities"][task_id])
    for event in tape:
        if event.get("task_id") == task_id and event.get("kind") == "priority_change" and int(event["time"]) <= arrival_time:
            prior = float(event["new_priority"])
    return prior


def _task_progress_contract(env: Any) -> dict[str, dict[str, Any]]:
    specs = getattr(env, "_task_specs", None)
    if not isinstance(specs, Mapping):
        raise ValueError("simulator does not expose task progress definitions")
    result: dict[str, dict[str, Any]] = {}
    for task_id, spec in sorted(specs.items()):
        if not isinstance(spec, Mapping):
            raise ValueError("simulator task progress definition is malformed")
        service_total = float(spec.get("service_minutes", spec.get("load_minutes", 1) + spec.get("unload_minutes", 1)))
        share = float(spec.get("workload_share", 1.0))
        if not math.isfinite(service_total) or service_total <= 0 or not math.isfinite(share) or share <= 0:
            raise ValueError("simulator task progress definition must be finite and positive")
        result[str(task_id)] = {
            "service_total": service_total,
            "parent_task_id": spec.get("parent_task_id"),
            "workload_share": share,
        }
    return result


def _window_task_universe(
    snapshot: Mapping[str, Any],
    suffix_snapshot: Mapping[str, Any],
    traces: Sequence[Mapping[str, Any]],
    *,
    horizon_minutes: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Build one logical obligation set shared by both arms of this suffix."""
    start = int(snapshot["time"])
    state = _decode(suffix_snapshot["state"])
    scenario = state["scenario"]
    end = min(start + horizon_minutes, int(scenario["base_scenario"]["horizon"]))
    known = snapshot.get("task_known_at_start", {})
    start_status = snapshot.get("task_status_at_start", {})
    priorities_at_snapshot = snapshot.get("task_priorities", {})
    task_specs = {str(row["id"]): row for row in scenario["base_scenario"].get("tasks", [])}
    included: set[str] = {
        task_id for task_id, is_known in known.items()
        if is_known and start_status.get(task_id) not in {"completed", "cancelled"}
    }
    tape = scenario.get("event_tape", [])
    arrivals: dict[str, int] = {}
    for event in tape:
        if event.get("kind") != "task_arrival":
            continue
        task_id = str(event.get("task_id"))
        arrival_time = int(event["time"])
        if start < arrival_time <= end:
            arrivals[task_id] = min(arrival_time, arrivals.get(task_id, arrival_time))
    included.update(arrivals)
    cancel_time: dict[str, int] = {}
    for event in tape:
        if event.get("kind") == "task_cancel":
            task_id = str(event.get("task_id"))
            event_time = int(event["time"])
            if start < event_time <= end:
                cancel_time[task_id] = min(event_time, cancel_time.get(task_id, event_time))
    completed_before_cancel: set[str] = set()
    for task_id, when in cancel_time.items():
        if any(
            isinstance(trace.get("raw_truth"), Mapping)
            and any(
                row.get("id") == task_id
                and row.get("completed_at") is not None
                and int(row["completed_at"]) <= when
                for row in trace["raw_truth"].get("tasks", [])
            )
            for trace in traces
        ):
            completed_before_cancel.add(task_id)
    excluded_cancelled = sorted(set(cancel_time) & included - completed_before_cancel)
    included.difference_update(excluded_cancelled)
    priority_map: dict[str, float] = {}
    for task_id in sorted(included):
        if task_id in arrivals:
            priority = _priority_at_arrival(
                task_id, arrivals[task_id], snapshot=snapshot,
                suffix_snapshot=suffix_snapshot, traces=traces,
            )
        else:
            priority = float(priorities_at_snapshot[task_id])
        if not math.isfinite(priority) or priority <= 0:
            raise ValueError("task universe contains an invalid priority")
        priority_map[task_id] = priority
    if not priority_map:
        raise ValueError("counterfactual horizon contains no active logical obligations")
    total = sum(priority_map.values())
    normalized = {task_id: value / total for task_id, value in priority_map.items()}
    return normalized, {
        "task_ids": sorted(normalized),
        "priority_weights": normalized,
        "excluded_cancelled_before_completion": excluded_cancelled,
        "arrival_times": arrivals,
        "weighting_rule": "snapshot-known outstanding tasks plus arrivals by window end; arrival priority at first public observation; cancelled obligations removed only when neither branch completed before cancellation",
    }


def _run_arm(
    suffix_json: str,
    *,
    snapshot: Mapping[str, Any],
    env_factory: Callable[..., Any] | None,
    arm: str,
    replan_action: Mapping[str, Any],
    continuation_policy: Callable[[Any], Mapping[str, Any]],
    replanner_cpu_seconds: float,
    action_delay_ticks: int,
    horizon_minutes: int,
) -> dict[str, Any]:
    suffix = _snapshot_payload(suffix_json, SUFFIX_SCHEMA)
    env = restore_snapshot(suffix_json, env_factory)
    if int(env.time) != int(snapshot["time"]):
        raise ValueError("paired suffix snapshot has a different start time")
    start_truth = copy.deepcopy(env._raw_truth())
    remaining = min(int(horizon_minutes), max(0, int(env._base.scenario["horizon"]) - int(env.time)))
    steps: list[dict[str, Any]] = []
    continuation_safety_cpu = 0.0
    continuation_planner_cpu = 0.0
    continuation_receipts: list[dict[str, Any]] = []
    action_applied_step: int | None = None
    expired_reason: str | None = None
    collection_error: Exception | None = None

    def continue_once(stage: str) -> bool:
        nonlocal continuation_safety_cpu, continuation_planner_cpu
        action, receipt, planner_cpu, safety_cpu = _continuation_action_receipt(
            env, continuation_policy, stage=stage,
        )
        continuation_receipts.append(receipt)
        continuation_planner_cpu += planner_cpu
        continuation_safety_cpu += safety_cpu
        step, _, terminal = _step_one(
            env, action, planner_invoked=bool(receipt["planner_invoked"]), stage=stage,
        )
        step["continuation_receipt"] = copy.deepcopy(receipt)
        steps.append(step)
        return terminal

    try:
        if arm == "keep":
            if remaining:
                step, _, terminal = _step_one(env, NOOP_ACTION, planner_invoked=False, stage="keep")
                steps.append(step)
                for _ in range(1, remaining):
                    if terminal:
                        break
                    terminal = continue_once("continuation")
        elif arm == "replan":
            terminated = False
            for _ in range(min(action_delay_ticks, remaining)):
                terminated = continue_once("compute_wait")
                if terminated:
                    break
            if len(steps) < remaining and not terminated:
                # Validate only after any modeled compute wait.  A stale action is
                # rejected and recorded; the existing queues continue normally.
                normalized, issues = _validated_action(env, replan_action)
                if issues:
                    expired_reason = ";".join(sorted(set(map(str, issues))))
                    normalized = copy.deepcopy(NOOP_ACTION)
                else:
                    action_applied_step = len(steps)
                step, _, terminal = _step_one(
                    env, normalized, planner_invoked=True, stage="replan_apply"
                )
                steps.append(step)
                for _ in range(len(steps), remaining):
                    if terminal:
                        break
                    terminal = continue_once("continuation")
            elif steps and not terminated and not env._base._terminal:
                expired_reason = "command_expires_after_requested_window"
        else:
            raise ValueError("counterfactual arm must be keep or replan")
    except Exception as exc:
        collection_error = exc

    trace = _trace_for_branch(
        env, snapshot=snapshot, suffix_snapshot=suffix, arm=arm,
        planned_action=NOOP_ACTION if arm == "keep" else replan_action,
        steps=steps, start_truth=start_truth,
        planner_cpu_seconds=replanner_cpu_seconds if arm == "replan" else 0.0,
        action_delay_ticks=action_delay_ticks if arm == "replan" else 0,
        action_applied_step=action_applied_step,
        action_expired_reason=expired_reason,
        continuation_cpu_seconds=continuation_safety_cpu,
        continuation_planner_cpu_seconds=continuation_planner_cpu,
        continuation_receipts=continuation_receipts,
        horizon_minutes=horizon_minutes,
        collection_error=collection_error,
    )
    return trace


def run_paired_counterfactual(
    snapshot_json: str,
    *,
    replan_action: Mapping[str, Any],
    env_factory: Callable[..., Any] | None = None,
    suffixes: Sequence[str] | None = None,
    continuation_policy: Callable[[Any], Mapping[str, Any]] = noop_continuation,
    replanner_version: str,
    replanner_cpu_seconds: float,
    cost_config: GainCostConfig = DEFAULT_CONFIG,
    horizon_minutes: int = 30,
    action_delay_ticks: int = 0,
) -> dict[str, Any]:
    """Execute keep/replan siblings on identical suffixes and score each pair."""
    source = _snapshot_payload(snapshot_json, SNAPSHOT_SCHEMA)
    if not isinstance(replan_action, Mapping):
        raise ValueError("fixed replanner must return a full environment action")
    if not isinstance(replanner_version, str) or not replanner_version:
        raise ValueError("replanner version must be nonempty")
    if source.get("replanner_version") != replanner_version:
        raise ValueError("snapshot and fixed replanner versions differ")
    if type(horizon_minutes) is not int or horizon_minutes <= 0:
        raise ValueError("counterfactual horizon must be a positive integer minute count")
    if type(action_delay_ticks) is not int or action_delay_ticks < 0:
        raise ValueError("action delay must use explicit nonnegative simulator ticks")
    if isinstance(replanner_cpu_seconds, bool) or not isinstance(replanner_cpu_seconds, (int, float)) or not math.isfinite(float(replanner_cpu_seconds)) or replanner_cpu_seconds < 0:
        raise ValueError("planner CPU time must be finite and nonnegative")
    if source["forced"]:
        return _invalid_label(source, "forced_safety_state", pairs=[])
    suffix_jsons = list(suffixes or [])
    if not suffix_jsons:
        try:
            suffix_jsons = [prepare_suffix_snapshot(snapshot_json, future_events=[
                row for row in _decode(source["state"])["scenario"].get("event_tape", [])
                if int(row["time"]) > int(source["time"])
            ], suffix_seed=0, env_factory=env_factory)]
        except Exception as exc:
            return _invalid_label(source, "suffix_preparation_failed", pairs=[], details={
                "error_type": type(exc).__name__, "message": str(exc),
            })
    if len(suffix_jsons) > 8:
        raise ValueError("paired counterfactual K is capped at eight")
    pair_rows = []
    for suffix_json in suffix_jsons:
        pair: dict[str, Any] = {
            "schema": PAIR_SCHEMA,
            "state_id": source["state_id"],
            "scenario_id": source["scenario_id"],
            "parent_family": source["parent_family"],
            "split": source["split"],
            "time": int(source["time"]),
            "event_type": source["event_type"],
            "forced": bool(source["forced"]),
            "horizon_requested_minutes": horizon_minutes,
            "source_snapshot_sha256": source["snapshot_sha256"],
            "private_snapshot_sha256": source["private_snapshot_sha256"],
            "public_history_sha256": source["public_history_sha256"],
            "replanner_version": replanner_version,
            "suffix_json_sha256": _digest(suffix_json) if isinstance(suffix_json, str) else None,
            "action_delay_ticks": action_delay_ticks,
            "simulator_time_resolution_minutes": 1,
            "subminute_delay_effect_simulated": False,
        }
        suffix: dict[str, Any] | None = None
        keep_trace: dict[str, Any] | None = None
        replan_trace: dict[str, Any] | None = None
        error: Exception | None = None
        try:
            if not isinstance(suffix_json, str):
                raise ValueError("suffix snapshot must be JSON text")
            suffix = _snapshot_payload(suffix_json, SUFFIX_SCHEMA)
            if suffix.get("source_snapshot_sha256") != source.get("snapshot_sha256"):
                raise ValueError("suffix snapshot is not bound to the source state")
            pair.update({
                "event_tape_id": suffix["event_tape_id"],
                "event_tape_sha256": suffix["event_tape_sha256"],
                "suffix_seed": suffix["suffix_seed"],
            })
            keep_trace = _run_arm(
                suffix_json, snapshot=source, env_factory=env_factory, arm="keep",
                replan_action=replan_action, continuation_policy=continuation_policy,
                replanner_cpu_seconds=float(replanner_cpu_seconds), action_delay_ticks=action_delay_ticks,
                horizon_minutes=horizon_minutes,
            )
            replan_trace = _run_arm(
                suffix_json, snapshot=source, env_factory=env_factory, arm="replan",
                replan_action=replan_action, continuation_policy=continuation_policy,
                replanner_cpu_seconds=float(replanner_cpu_seconds), action_delay_ticks=action_delay_ticks,
                horizon_minutes=horizon_minutes,
            )
            if not keep_trace["valid_mask"] or not replan_trace["valid_mask"]:
                raise ValueError("one or both paired branches failed during execution")
            task_priorities, task_universe = _window_task_universe(
                source, suffix, (keep_trace, replan_trace), horizon_minutes=horizon_minutes,
            )
            keep_cost = calculate_branch_cost(keep_trace, task_priorities=task_priorities, config=cost_config)
            replan_cost = calculate_branch_cost(
                replan_trace, task_priorities=task_priorities,
                planner_cpu_seconds=float(replanner_cpu_seconds), config=cost_config,
            )
            pair.update({
                "horizon_actual_minutes": min(horizon_minutes, int(keep_trace["horizon_actual_minutes"]), int(replan_trace["horizon_actual_minutes"])),
                "task_universe": task_universe,
                "keep": {"action": copy.deepcopy(NOOP_ACTION), "trace": keep_trace, "cost": keep_cost},
                "replan": {"action": copy.deepcopy(dict(replan_action)), "trace": replan_trace, "cost": replan_cost},
                "gain": float(keep_cost["J_total"] - replan_cost["J_total"]),
                "valid_mask": True,
                "unknown_reason": None,
            })
        except Exception as exc:
            error = exc
            pair.update({
                "horizon_actual_minutes": min(
                    [int(trace["horizon_actual_minutes"]) for trace in (keep_trace, replan_trace) if trace is not None]
                    or [0]
                ),
                "task_universe": None,
                "keep": None if keep_trace is None else {"action": copy.deepcopy(NOOP_ACTION), "trace": keep_trace, "cost": None},
                "replan": None if replan_trace is None else {"action": copy.deepcopy(dict(replan_action)), "trace": replan_trace, "cost": None},
                "gain": None,
                "valid_mask": False,
                "unknown_reason": "paired_collection_failed",
                "collection_error": {"error_type": type(exc).__name__, "message": str(exc)},
            })
        unresolved_reasons = [
            reason
            for trace in (keep_trace, replan_trace)
            if trace is not None
            for reason in trace.get("unresolved_safety_unknown_reasons", [])
        ]
        pair["unresolved_safety_disposition_count"] = len(unresolved_reasons)
        pair["unresolved_safety_unknown_reasons"] = unresolved_reasons
        pair["pair_sha256"] = _digest(pair)
        pair_rows.append(pair)
    valid_pairs = all(bool(row.get("valid_mask")) for row in pair_rows)
    gains = [float(row["gain"]) for row in pair_rows] if valid_pairs else []
    variance = statistics.variance(gains) if len(gains) > 1 else None
    label = {
        "schema": LABEL_SCHEMA,
        "state_id": source["state_id"],
        "scenario_id": source["scenario_id"],
        "parent_family": source["parent_family"],
        "split": source["split"],
        "time": int(source["time"]),
        "public_history_sha256": source["public_history_sha256"],
        "private_snapshot_sha256": source["private_snapshot_sha256"],
        "source_snapshot_sha256": source["snapshot_sha256"],
        "replanner_version": replanner_version,
        "forced": bool(source["forced"]),
        "event_type": source["event_type"],
        "horizon_requested_minutes": horizon_minutes,
        "horizon_actual_minutes": min((int(row["horizon_actual_minutes"]) for row in pair_rows), default=0),
        "suffix_count": len(pair_rows),
        "gain_mean": statistics.mean(gains) if gains else None,
        "paired_variance": variance,
        "valid_mask": valid_pairs,
        "variance_valid_mask": variance is not None,
        "variance_unknown_reason": None if variance is not None else "K<2_or_invalid_pair",
        "unknown_reason": None if valid_pairs else next((row.get("unknown_reason") for row in pair_rows if not row.get("valid_mask")), "no_pairs"),
        "pairs": pair_rows,
    }
    label["label_sha256"] = _digest(label)
    return label


def _invalid_label(
    source: Mapping[str, Any],
    reason: str,
    *,
    pairs: Sequence[Mapping[str, Any]],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "schema": LABEL_SCHEMA,
        "state_id": source["state_id"],
        "scenario_id": source["scenario_id"],
        "parent_family": source["parent_family"],
        "split": source["split"],
        "time": int(source["time"]),
        "public_history_sha256": source["public_history_sha256"],
        "private_snapshot_sha256": source["private_snapshot_sha256"],
        "source_snapshot_sha256": source["snapshot_sha256"],
        "replanner_version": source["replanner_version"],
        "forced": bool(source["forced"]),
        "event_type": source["event_type"],
        "horizon_requested_minutes": None,
        "horizon_actual_minutes": None,
        "suffix_count": len(pairs),
        "gain_mean": None,
        "paired_variance": None,
        "valid_mask": False,
        "variance_valid_mask": False,
        "variance_unknown_reason": reason,
        "unknown_reason": reason,
        "pairs": [copy.deepcopy(dict(row)) for row in pairs],
    }
    if details is not None:
        result["collection_details"] = copy.deepcopy(dict(details))
    result["label_sha256"] = _digest(result)
    return result


def collect_paired_label(
    env: Any,
    *,
    state_id: str,
    scenario_id: str,
    parent_family: str,
    split: str,
    event_type: str,
    forced: bool,
    replanner_version: str,
    replan_action: Mapping[str, Any],
    suffix_rows: Sequence[Mapping[str, Any]],
    env_factory: Callable[..., Any] | None = None,
    continuation_policy: Callable[[Any], Mapping[str, Any]] = noop_continuation,
    replanner_cpu_seconds: float = 0.0,
    cost_config: GainCostConfig = DEFAULT_CONFIG,
    horizon_minutes: int = 30,
    action_delay_ticks: int = 0,
) -> dict[str, Any]:
    """Capture first, then collect a label over caller-sampled future suffixes.

    Each suffix row must contain a nonnegative integer ``suffix_seed`` and a
    sequence ``future_events`` sampled by the frozen scenario mechanism.
    """
    snapshot_json = capture_snapshot(
        env, state_id=state_id, scenario_id=scenario_id, parent_family=parent_family,
        split=split, event_type=event_type, forced=forced, replanner_version=replanner_version,
    )
    source = _snapshot_payload(snapshot_json, SNAPSHOT_SCHEMA)
    if forced:
        return _invalid_label(source, "forced_safety_state", pairs=[])
    suffix_jsons = []
    suffix_receipts = []
    for row in suffix_rows:
        if not isinstance(row, Mapping) or "suffix_seed" not in row or "future_events" not in row:
            raise ValueError("suffix rows require suffix_seed and future_events")
        try:
            suffix_json = prepare_suffix_snapshot(
                snapshot_json, future_events=row["future_events"], suffix_seed=row["suffix_seed"],
                env_factory=env_factory,
            )
        except Exception as exc:
            return _invalid_label(source, "suffix_preparation_failed", pairs=[], details={
                "suffix_seed": row.get("suffix_seed"),
                "prepared_suffixes": suffix_receipts,
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
        suffix_jsons.append(suffix_json)
        suffix_receipts.append({
            "suffix_seed": int(row["suffix_seed"]),
            "suffix_snapshot_sha256": _snapshot_payload(suffix_json, SUFFIX_SCHEMA)["snapshot_sha256"],
        })
    return run_paired_counterfactual(
        snapshot_json, replan_action=replan_action, env_factory=env_factory,
        suffixes=suffix_jsons, continuation_policy=continuation_policy,
        replanner_version=replanner_version, replanner_cpu_seconds=replanner_cpu_seconds,
        cost_config=cost_config, horizon_minutes=horizon_minutes,
        action_delay_ticks=action_delay_ticks,
    )


def persist_label(path: str | Path, label: Mapping[str, Any]) -> None:
    """Persist one immutable label without silently replacing prior evidence."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to replace counterfactual evidence: {target}")
    payload = json.dumps(dict(label), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
