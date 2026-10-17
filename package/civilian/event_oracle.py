"""Independent offline labels recomputed from executed event-relief traces.

This module deliberately does not import the environment, candidate producer,
or any label-generation helper.  Its only inputs are persisted raw branch
traces, including the explicit NOOP branch and all task outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA = "event-consequence-oracle/1"
MODEL_HEADS = (
    "next_state",
    "progress",
    "delay",
    "energy",
    "failure",
    "communication_risk",
    "constraint_risk",
    "utility",
)
_FAILURE_STATUSES = {"failed", "expired"}
_COMPLETE_STATUSES = {"completed", "delivered"}
_EPSILON = 1e-9
REWARD_FORMULA = "n02-configured-reward/1"
REWARD_WEIGHTS = {
    "completion": 1.0,
    "delay": 0.2,
    "energy": 0.01,
    "risk": 0.25,
    "communication": 0.01,
    "switch": 0.05,
    "reconstruction": 0.05,
    "reconstruction_scale": 0.01,
}
_DIVERSITY_HEADS = tuple(head for head in MODEL_HEADS if head != "utility")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _truth_rows(value: Any, code: str) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        rows = [dict(row, id=str(key)) for key, row in value.items() if isinstance(row, Mapping)]
    elif isinstance(value, list):
        rows = [dict(row) for row in value if isinstance(row, Mapping)]
    else:
        rows = []
    _require(bool(rows), code)
    ids = [str(row.get("id", "")) for row in rows]
    _require(all(ids) and len(ids) == len(set(ids)), code + "_IDS")
    return sorted(rows, key=lambda row: str(row["id"]))


def _task_progress(task: Mapping[str, Any]) -> float | None:
    for key in ("progress", "service_progress", "completion_fraction"):
        number = _finite_number(task.get(key))
        if number is not None:
            return min(1.0, max(0.0, number))
    if bool(task.get("delivered")) or str(task.get("status", "")) in _COMPLETE_STATUSES:
        return 1.0
    return None


def _task_outcome(task: Mapping[str, Any], end_time: int) -> dict[str, Any]:
    status = str(task.get("status", "")).lower()
    deadline = _finite_number(task.get("deadline"))
    completed_at = _finite_number(task.get("completed_at"))
    complete = bool(task.get("delivered")) or status in _COMPLETE_STATUSES
    failed = (
        status in _FAILURE_STATUSES
        or bool(task.get("failed"))
        or bool(task.get("failure_reason"))
    )
    progress = _task_progress(task)

    on_time: bool | None
    if isinstance(task.get("on_time"), bool):
        on_time = task["on_time"]
    elif complete and completed_at is not None and deadline is not None:
        on_time = completed_at <= deadline
    elif bool(task.get("deadline_missed")) or (deadline is not None and end_time >= deadline and not complete):
        on_time = False
    elif failed:
        on_time = False
    else:
        on_time = None

    if completed_at is not None and deadline is not None:
        delay: float | None = max(0.0, completed_at - deadline)
    elif on_time is False and deadline is not None and end_time >= deadline:
        delay = float(max(0, end_time - int(deadline)))
    else:
        delay = None

    return {
        "id": str(task.get("id", "")),
        "kind": task.get("kind"),
        "status": status or None,
        "progress": progress,
        "progress_valid": progress is not None,
        "complete": complete,
        "complete_valid": bool(status) or "delivered" in task,
        "on_time": on_time,
        "on_time_valid": on_time is not None,
        "delay_minutes": delay,
        "delay_valid": delay is not None,
        "failed": failed,
        "failure_valid": bool(status) or "failed" in task or "failure_reason" in task,
        "deadline": deadline,
    }


def _event_energy(events: list[dict[str, Any]]) -> tuple[float, bool]:
    total = 0.0
    for event in events:
        kind = str(event.get("type", "")).lower()
        if kind not in {"move", "service"}:
            continue
        amount = _finite_number(event.get("energy"))
        if amount is None:
            amount = _finite_number(event.get("energy_used"))
        if amount is None:
            return 0.0, False
        total += max(0.0, amount)
    return total, True


def _reward_weights(value: Any) -> dict[str, float]:
    if value is None:
        return dict(REWARD_WEIGHTS)
    _require(isinstance(value, Mapping), "EVENT_ORACLE_REWARD_CONFIG")
    _require(set(value) == set(REWARD_WEIGHTS), "EVENT_ORACLE_REWARD_CONFIG_KEYS")
    weights = {key: _finite_number(value.get(key)) for key in REWARD_WEIGHTS}
    _require(all(weight is not None for weight in weights.values()), "EVENT_ORACLE_REWARD_CONFIG_VALUES")
    resolved = {key: float(weight) for key, weight in weights.items()}
    _require(all(weight >= 0.0 for weight in resolved.values()), "EVENT_ORACLE_REWARD_CONFIG_NEGATIVE")
    return resolved


def _reward_primitives(action: Mapping[str, Any], events: list[dict[str, Any]]) -> tuple[dict[str, float], bool]:
    components = {
        "completion": 0.0,
        "delay": 0.0,
        "energy": 0.0,
        "risk": 0.0,
        "communication": 0.0,
        "switch": 0.0,
        "reconstruction": 0.0,
        "reconstruction_size": 0.0,
    }
    valid = True
    switched_tasks: set[str] = set()
    changed_owner_tasks: set[str] = set()
    queue_reorders = 0
    for event in events:
        kind = str(event.get("type", "")).lower()
        if kind == "logical_task_completed":
            components["completion"] += 1.0
        elif kind == "delay_accrued":
            minutes = _finite_number(event.get("minutes"))
            if minutes is None or minutes < 0:
                valid = False
            else:
                components["delay"] += minutes
        elif kind in {"move", "service"}:
            energy = _finite_number(event.get("energy"))
            risk = _finite_number(event.get("environmental_risk")) if kind == "move" else 0.0
            if energy is None or energy < 0 or risk is None or risk < 0:
                valid = False
            else:
                components["energy"] += energy
                components["risk"] += risk
        elif kind in {"task_failed", "platform_failure"}:
            components["risk"] += 1.0
        elif kind == "communication_tx":
            count = _finite_number(event.get("message_count"))
            if count is None or count < 0:
                valid = False
            else:
                components["communication"] += count
        elif kind == "task_switch":
            task_id = event.get("task_id")
            if task_id is None:
                valid = False
            else:
                switched_tasks.add(str(task_id))
        elif kind == "task_owner_change":
            task_id = event.get("task_id")
            if task_id is None:
                valid = False
            else:
                changed_owner_tasks.add(str(task_id))
        elif kind in {"queue_reorder", "queue_reordered"}:
            queue_reorders += 1

    action_noop = action.get("noop")
    queue_operations = action.get("queue_operations")
    dispatches = action.get("dispatches")
    if not isinstance(action_noop, bool) or not isinstance(queue_operations, list) or not isinstance(dispatches, list):
        valid = False
    else:
        effective_size = len(changed_owner_tasks) + queue_reorders
        components["switch"] = float(len(switched_tasks))
        components["reconstruction"] = float(effective_size > 0)
        components["reconstruction_size"] = float(effective_size)
    return components, valid


def _configured_step_reward(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return (
        components["completion"] * weights["completion"]
        - components["delay"] * weights["delay"]
        - components["energy"] * weights["energy"]
        - components["risk"] * weights["risk"]
        - components["communication"] * weights["communication"]
        - components["switch"] * weights["switch"]
        - components["reconstruction"] * weights["reconstruction"]
        - components["reconstruction_size"] * weights["reconstruction_scale"]
    )


def _configured_reward_terms(components: Mapping[str, float], weights: Mapping[str, float]) -> dict[str, float]:
    return {
        "completion": components["completion"] * weights["completion"],
        "delay": -components["delay"] * weights["delay"],
        "energy": -components["energy"] * weights["energy"],
        "risk": -components["risk"] * weights["risk"],
        "communication": -components["communication"] * weights["communication"],
        "switch": -components["switch"] * weights["switch"],
        "reconstruction": -components["reconstruction"] * weights["reconstruction"],
        "reconstruction_scale": -components["reconstruction_size"] * weights["reconstruction_scale"],
    }


def _communication_risk(steps: list[dict[str, Any]], events: list[dict[str, Any]]) -> tuple[float, bool, dict[str, int]]:
    scores: list[float] = []
    comm_event_count = 0
    for event in events:
        kind = str(event.get("type", "")).lower()
        if not any(token in kind for token in ("comm", "message", "link", "packet")):
            continue
        comm_event_count += 1
        explicit = _finite_number(event.get("risk"))
        quality = _finite_number(event.get("quality"))
        if explicit is not None:
            scores.append(min(1.0, max(0.0, explicit)))
        elif quality is not None:
            scores.append(1.0 - min(1.0, max(0.0, quality)))
        elif event.get("connected") is False or event.get("dropped") is True or "drop" in kind or "down" in kind or "disconnect" in kind:
            scores.append(1.0)
        else:
            score = _finite_number(event.get("packet_loss"))
            if score is None:
                score = _finite_number(event.get("loss_rate"))
            if score is not None:
                scores.append(min(1.0, max(0.0, score)))

    link_samples = 0
    for step in steps:
        for observation_key in ("public_observation", "next_public_observation"):
            observation = step.get(observation_key)
            if not isinstance(observation, Mapping):
                continue
            links = observation.get("communication", observation.get("links", []))
            if isinstance(links, Mapping):
                links = list(links.values())
            if not isinstance(links, list):
                continue
            for link in links:
                if not isinstance(link, Mapping):
                    continue
                quality = _finite_number(link.get("quality", link.get("link_quality")))
                loss = _finite_number(link.get("packet_loss", link.get("loss_rate")))
                if quality is None and loss is None:
                    continue
                risk = 0.0
                if quality is not None:
                    risk = 1.0 - min(1.0, max(0.0, quality))
                if loss is not None:
                    risk = max(risk, min(1.0, max(0.0, loss)))
                scores.append(risk)
                link_samples += 1

    if scores:
        return sum(scores) / len(scores), True, {"communication_events": comm_event_count, "link_samples": link_samples}
    if comm_event_count == 0 and all("raw_events" in step for step in steps):
        # A complete event trace with no communication events records zero observed risk.
        return 0.0, True, {"communication_events": 0, "link_samples": link_samples}
    return 0.0, False, {"communication_events": comm_event_count, "link_samples": link_samples}


def _constraint_risk(steps: list[dict[str, Any]], events: list[dict[str, Any]]) -> tuple[float, bool, dict[str, int]]:
    violations = 0
    receipts_seen = 0
    for step in steps:
        receipts = step.get("operation_receipts")
        if isinstance(receipts, Mapping):
            receipts = [receipts]
        if isinstance(receipts, list):
            receipts_seen += 1
            for receipt in receipts:
                if not isinstance(receipt, Mapping):
                    continue
                legal = receipt.get("postrepair_legal", receipt.get("legal"))
                failures = receipt.get("constraint_violations", receipt.get("violations", []))
                if legal is False:
                    violations += 1
                elif isinstance(failures, list):
                    violations += len(failures)
    event_violations = sum(
        1 for event in events
        if "constraint" in str(event.get("type", "")).lower()
        and any(token in str(event.get("type", "")).lower() for token in ("violation", "rejected", "infeasible"))
    )
    violations += event_violations
    return float(violations), receipts_seen == len(steps), {
        "receipt_steps": receipts_seen,
        "constraint_events": event_violations,
    }


def _next_state(tasks: list[dict[str, Any]], platforms: list[dict[str, Any]]) -> tuple[list[float], list[bool]]:
    values: list[float] = []
    valid: list[bool] = []
    for task in tasks:
        progress = _task_progress(task)
        status = str(task.get("status", "")).lower()
        complete = bool(task.get("delivered")) or status in _COMPLETE_STATUSES
        failed = status in _FAILURE_STATUSES or bool(task.get("failed")) or bool(task.get("failure_reason"))
        on_time = task.get("on_time")
        components = (
            progress,
            1.0 if complete else (0.0 if status else None),
            1.0 if failed else (0.0 if status or "failure_reason" in task else None),
            1.0 if on_time is True else (0.0 if isinstance(on_time, bool) else None),
        )
        for component in components:
            values.append(float(component) if component is not None else 0.0)
            valid.append(component is not None)
    for platform in platforms:
        position = platform.get("position")
        x = _finite_number(position[0]) if isinstance(position, list) and len(position) >= 2 else None
        y = _finite_number(position[1]) if isinstance(position, list) and len(position) >= 2 else None
        energy = _finite_number(platform.get("energy"))
        failed_value = platform.get("failed")
        components = (
            x,
            y,
            energy,
            1.0 if failed_value is True else (0.0 if isinstance(failed_value, bool) else None),
        )
        for component in components:
            values.append(float(component) if component is not None else 0.0)
            valid.append(component is not None)
    return values, valid


def _branch_labels(branch: Mapping[str, Any], window: Mapping[str, Any]) -> dict[str, Any]:
    branch_candidate_key = branch.get("candidate_key")
    _require(isinstance(branch_candidate_key, str) and branch_candidate_key, "EVENT_ORACLE_CANDIDATE_KEY")
    candidate = branch.get("candidate")
    _require(isinstance(candidate, Mapping), "EVENT_ORACLE_CANDIDATE_RECORD")
    candidate_key = candidate.get("candidate_key")
    _require(isinstance(candidate_key, str) and candidate_key, "EVENT_ORACLE_CANDIDATE_KEY")
    _require(branch_candidate_key == candidate_key, "EVENT_ORACLE_BRANCH_CANDIDATE_KEY_BINDING")
    legal = candidate.get("legal") is True
    action = candidate.get("full_environment_action")
    _require(isinstance(action, Mapping), "EVENT_ORACLE_FULL_ACTION")
    expected_candidate_key = f"event-action-{_digest(dict(action))[:24]}"
    _require(candidate_key == expected_candidate_key, "EVENT_ORACLE_CANDIDATE_KEY_DERIVATION")
    action_scope = action.get("scope")
    action_noop = action.get("noop") is True
    _require(action_noop == (action_scope == "NOOP"), "EVENT_ORACLE_ACTION_SCOPE")
    if action_noop or action_scope == "NOOP":
        expected_candidate_scope = "noop"
    elif action_scope == "LOCAL_REPAIR":
        expected_candidate_scope = "local"
    elif action_scope == "GLOBAL_REPLAN":
        expected_candidate_scope = "global"
    else:
        expected_candidate_scope = None
    _require(expected_candidate_scope is not None, "EVENT_ORACLE_ACTION_SCOPE")
    _require(candidate.get("candidate_scope") == expected_candidate_scope, "EVENT_ORACLE_CANDIDATE_SCOPE_BINDING")
    public_observation = window.get("public_observation")
    provenance = public_observation.get("provenance") if isinstance(public_observation, Mapping) else None
    _require(isinstance(provenance, Mapping), "EVENT_ORACLE_PUBLIC_PROVENANCE")
    for name in ("snapshot_hash", "public_prefix_hash"):
        expected = provenance.get(name)
        _require(isinstance(expected, str) and expected, "EVENT_ORACLE_PUBLIC_PROVENANCE")
        _require(candidate.get(name) == expected, "EVENT_ORACLE_CANDIDATE_PROVENANCE_BINDING")
    _require(candidate.get("event_tape_id") == window.get("event_tape_id"), "EVENT_ORACLE_CANDIDATE_TAPE_BINDING")
    if "candidates" in window:
        catalog = window.get("candidates")
        _require(isinstance(catalog, list), "EVENT_ORACLE_CANDIDATE_CATALOG")
        catalog_matches = [row for row in catalog if isinstance(row, Mapping) and row.get("candidate_key") == candidate_key]
        _require(len(catalog_matches) == 1, "EVENT_ORACLE_CANDIDATE_CATALOG_BINDING")
        _require(_canonical(dict(catalog_matches[0])) == _canonical(dict(candidate)), "EVENT_ORACLE_CANDIDATE_CATALOG_BINDING")
    trajectory = branch.get("trajectory")
    _require(isinstance(trajectory, Mapping), "EVENT_ORACLE_TRAJECTORY")
    _require(trajectory.get("parent_id") == window.get("parent_id"), "EVENT_ORACLE_PARENT_BINDING")
    _require(trajectory.get("event_tape_id") == window.get("event_tape_id"), "EVENT_ORACLE_TAPE_BINDING")
    window_binding = window.get("private_binding")
    branch_binding = branch.get("private_binding")
    _require(isinstance(window_binding, Mapping) and isinstance(branch_binding, Mapping), "EVENT_ORACLE_PRIVATE_BINDING")
    for name in ("simulator_snapshot_sha256", "event_tape_sha256"):
        _require(_is_sha256(window_binding.get(name)), "EVENT_ORACLE_PRIVATE_WINDOW_HASH")
        _require(branch_binding.get(name) == window_binding.get(name), "EVENT_ORACLE_PRIVATE_BRANCH_BINDING")
    steps = trajectory.get("steps")
    _require(isinstance(steps, list) and steps, "EVENT_ORACLE_STEPS")
    start = int(window["window_start"])
    planned_end = int(window.get("window_end", start + len(steps)))
    expected_end = int(trajectory.get("end_time", planned_end))
    _require(int(trajectory.get("start_time", -1)) == start, "EVENT_ORACLE_START_TIME")
    _require(start < expected_end <= planned_end, "EVENT_ORACLE_END_TIME")
    _require(len(steps) == expected_end - start, "EVENT_ORACLE_STEP_COUNT")
    raw_events: list[dict[str, Any]] = []
    rewards: list[float] = []
    recomputed_rewards: list[float] = []
    reward_components: list[dict[str, float]] = []
    reward_weights = _reward_weights(window.get("reward_weights"))
    for index, step in enumerate(steps):
        _require(isinstance(step, Mapping), "EVENT_ORACLE_STEP")
        time_value = step.get("time")
        _require(int(time_value) == start + index, "EVENT_ORACLE_STEP_TIME")
        submitted = step.get("full_environment_action")
        _require(isinstance(submitted, Mapping), "EVENT_ORACLE_SUBMITTED_ACTION")
        if index == 0:
            _require(_canonical(dict(submitted)) == _canonical(dict(action)), "EVENT_ORACLE_ACTION_BINDING")
        else:
            _require(submitted.get("noop") is True, "EVENT_ORACLE_CONTINUATION_NOT_NOOP")
        event_rows = step.get("raw_events")
        _require(isinstance(event_rows, list), "EVENT_ORACLE_RAW_EVENTS")
        step_events = [dict(event) for event in event_rows if isinstance(event, Mapping)]
        raw_events.extend(step_events)
        step_action = submitted
        components, components_valid = _reward_primitives(step_action, step_events)
        _require(components_valid, "EVENT_ORACLE_REWARD_PRIMITIVES_INCOMPLETE")
        reward_components.append(components)
        recomputed_rewards.append(_configured_step_reward(components, reward_weights))
        reward = _finite_number(step.get("reward"))
        _require(reward is not None, "EVENT_ORACLE_RECORDED_REWARD_MISSING")
        rewards.append(float(reward))
        _require(math.isclose(float(reward), recomputed_rewards[-1], rel_tol=1e-9, abs_tol=1e-7),
                 "EVENT_ORACLE_RECORDED_REWARD_MISMATCH")
        reported_primitives = step.get("reward_primitives")
        _require(isinstance(reported_primitives, Mapping), "EVENT_ORACLE_REPORTED_REWARD_PRIMITIVES_MISSING")
        primitive_names = {
            "completion": "completion_count",
            "delay": "delay_minutes",
            "energy": "energy_units",
            "risk": "risk_units",
            "communication": "communication_messages",
            "switch": "switch_count",
            "reconstruction": "reconstruction_count",
            "reconstruction_size": "reconstruction_size",
        }
        for name, primitive_name in primitive_names.items():
            reported = _finite_number(reported_primitives.get(primitive_name))
            _require(reported is not None and math.isclose(reported, components[name], rel_tol=1e-9, abs_tol=1e-7),
                     "EVENT_ORACLE_REPORTED_REWARD_PRIMITIVE_MISMATCH")
        reported_terms = step.get("reward_terms")
        _require(isinstance(reported_terms, Mapping), "EVENT_ORACLE_REPORTED_REWARD_TERMS_MISSING")
        for name, expected_term in _configured_reward_terms(components, reward_weights).items():
            reported = _finite_number(reported_terms.get(name))
            _require(reported is not None and math.isclose(reported, expected_term, rel_tol=1e-9, abs_tol=1e-7),
                     "EVENT_ORACLE_REPORTED_REWARD_TERM_MISMATCH")
        reported_components = step.get("reward_components")
        if isinstance(reported_components, Mapping):
            for name, value in components.items():
                reported = _finite_number(reported_components.get(name))
                _require(reported is not None and math.isclose(reported, value, rel_tol=1e-9, abs_tol=1e-7),
                         "EVENT_ORACLE_REPORTED_REWARD_COMPONENT_MISMATCH")

    raw_truth = trajectory.get("raw_truth")
    _require(isinstance(raw_truth, Mapping), "EVENT_ORACLE_RAW_TRUTH")
    tasks = _truth_rows(raw_truth.get("tasks"), "EVENT_ORACLE_TRUTH_TASKS")
    platforms = _truth_rows(raw_truth.get("platforms"), "EVENT_ORACLE_TRUTH_PLATFORMS")
    end_time = int(raw_truth.get("time", expected_end))
    _require(end_time == expected_end, "EVENT_ORACLE_TRUTH_TIME")
    task_rows = [_task_outcome(task, end_time) for task in tasks]

    progress_values = [row["progress"] for row in task_rows if row["progress_valid"]]
    progress_valid = len(progress_values) == len(task_rows) and bool(task_rows)
    progress = sum(progress_values) / len(progress_values) if progress_values else 0.0
    on_time_rows = [row for row in task_rows if row["on_time_valid"]]
    on_time_valid = len(on_time_rows) == len(task_rows) and bool(task_rows)
    on_time_rate = (
        sum(bool(row["on_time"]) for row in on_time_rows) / len(on_time_rows)
        if on_time_rows else 0.0
    )
    delay_rows = [row for row in task_rows if row["delay_valid"]]
    delay_valid = bool(delay_rows) and len(delay_rows) == sum(row["deadline"] is not None for row in task_rows)
    delay = sum(float(row["delay_minutes"]) for row in delay_rows) / len(delay_rows) if delay_rows else 0.0
    failure_valid = all(row["failure_valid"] for row in task_rows) and bool(task_rows)
    failure = sum(bool(row["failed"]) for row in task_rows) / len(task_rows) if task_rows else 0.0
    energy, event_energy_valid = _event_energy(raw_events)
    public_observation = window.get("public_observation")
    start_platforms = public_observation.get("platforms") if isinstance(public_observation, Mapping) else None
    start_energy_rows = [
        _finite_number(platform.get("energy_used"))
        for platform in start_platforms
        if isinstance(platform, Mapping)
    ] if isinstance(start_platforms, list) else []
    end_energy_rows = [_finite_number(platform.get("energy_used")) for platform in platforms]
    endpoint_energy_valid = (
        len(start_energy_rows) == len(platforms)
        and all(value is not None for value in start_energy_rows)
        and all(value is not None for value in end_energy_rows)
    )
    endpoint_energy_delta = (
        sum(float(value) for value in end_energy_rows) - sum(float(value) for value in start_energy_rows)
        if endpoint_energy_valid else 0.0
    )
    energy_valid = event_energy_valid and endpoint_energy_valid and math.isclose(
        energy, endpoint_energy_delta, rel_tol=1e-8, abs_tol=1e-6
    )
    _require(not event_energy_valid or not endpoint_energy_valid or energy_valid, "EVENT_ORACLE_ENERGY_CROSSCHECK_MISMATCH")
    communication_risk, communication_valid, communication_counts = _communication_risk(steps, raw_events)
    constraint_risk, constraint_valid, constraint_counts = _constraint_risk(steps, raw_events)
    utility = sum(recomputed_rewards)
    utility_valid = len(recomputed_rewards) == len(steps)
    next_state_values, next_state_valid = _next_state(tasks, platforms)

    model_targets = {
        "next_state": {"values": next_state_values, "valid": next_state_valid},
        "progress": {"values": progress, "valid": progress_valid},
        "delay": {"values": delay, "valid": delay_valid},
        "energy": {"values": energy, "valid": energy_valid},
        "failure": {"values": failure, "valid": failure_valid},
        "communication_risk": {"values": communication_risk, "valid": communication_valid},
        "constraint_risk": {"values": constraint_risk, "valid": constraint_valid},
        "utility": {"values": utility, "valid": utility_valid},
    }
    return {
        "candidate_key": candidate_key,
        "legal": legal,
        "task_outcomes": task_rows,
        "global_outcomes": {
            "progress_rate": {"value": progress, "valid": progress_valid},
            "on_time_rate": {"value": on_time_rate, "valid": on_time_valid},
            "mean_delay_minutes": {"value": delay, "valid": delay_valid},
            "energy_used": {"value": energy, "valid": energy_valid},
            "failure_rate": {"value": failure, "valid": failure_valid},
            "communication_risk": {"value": communication_risk, "valid": communication_valid},
            "constraint_violation_count": {"value": constraint_risk, "valid": constraint_valid},
            "utility": {"value": utility, "valid": utility_valid},
        },
        "model_targets": model_targets,
        "reward_recomputation": {
            "formula": REWARD_FORMULA,
            "weights": reward_weights,
            "component_totals": {
                name: sum(row[name] for row in reward_components)
                for name in reward_components[0]
            },
            "recomputed_step_rewards": recomputed_rewards,
            "recorded_step_rewards": rewards,
            "recomputed_utility": utility,
            "recorded_utility": sum(rewards),
            "max_abs_step_error": max(abs(left - right) for left, right in zip(recomputed_rewards, rewards)),
        },
        "energy_cross_check": {
            "event_energy": energy,
            "endpoint_energy_used_delta": endpoint_energy_delta,
            "valid": energy_valid,
        },
        "unknown": {
            "unresolved_task_ids": [row["id"] for row in task_rows if not row["on_time_valid"]],
            "invalid_heads": [name for name, target in model_targets.items() if not _target_any_valid(target)],
            "communication_counts": communication_counts,
            "constraint_counts": constraint_counts,
        },
        "trace_sha256": _digest(trajectory),
    }


def _target_any_valid(target: Mapping[str, Any]) -> bool:
    valid = target.get("valid")
    return any(valid) if isinstance(valid, list) else bool(valid)


def _signature(label: Mapping[str, Any]) -> bytes | None:
    signature: dict[str, Any] = {}
    for name in _DIVERSITY_HEADS:
        target = label["model_targets"][name]
        values = target["values"]
        valid = target["valid"]
        if isinstance(values, list) and isinstance(valid, list):
            kept = [float(value) for value, is_valid in zip(values, valid) if is_valid]
            if kept:
                signature[name] = kept
        elif valid:
            signature[name] = float(values)
    return _canonical(signature) if signature else None


def _differing_physical_heads(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    differing = []
    for name in _DIVERSITY_HEADS:
        left_target = left["model_targets"][name]
        right_target = right["model_targets"][name]
        lv, rv = left_target["values"], right_target["values"]
        lm, rm = left_target["valid"], right_target["valid"]
        if isinstance(lv, list) and isinstance(rv, list) and isinstance(lm, list) and isinstance(rm, list):
            for left_value, right_value, left_valid, right_valid in zip(lv, rv, lm, rm):
                if left_valid and right_valid and not math.isclose(
                    float(left_value), float(right_value), rel_tol=1e-9, abs_tol=1e-9
                ):
                    differing.append(name)
                    break
        elif not isinstance(lv, list) and not isinstance(rv, list) and lm is True and rm is True:
            if not math.isclose(float(lv), float(rv), rel_tol=1e-9, abs_tol=1e-9):
                differing.append(name)
    return differing


def _outcomes_differ(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(_differing_physical_heads(left, right))


def recompute_consequence_labels(window: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute consequence labels and outcome distinctness from raw traces.

    `window` contains `branches`, each binding one full candidate action to an
    independently executed trajectory. The trajectory must include explicit
    actions for every minute, raw event slices, and endpoint truth. Producer
    label fields are ignored even if present.
    """

    _require(isinstance(window, Mapping), "EVENT_ORACLE_WINDOW")
    _require(isinstance(window.get("parent_id"), str), "EVENT_ORACLE_WINDOW_PARENT")
    _require(window.get("split") in {"train", "development"}, "EVENT_ORACLE_SPLIT")
    _require(isinstance(window.get("event_tape_id"), str), "EVENT_ORACLE_WINDOW_TAPE")
    _require(type(window.get("window_start")) is int, "EVENT_ORACLE_WINDOW_START")
    branches = window.get("branches")
    _require(isinstance(branches, list) and bool(branches), "EVENT_ORACLE_BRANCHES")
    binding = window.get("private_binding")
    _require(isinstance(binding, Mapping), "EVENT_ORACLE_PRIVATE_BINDING")
    _require(_is_sha256(binding.get("simulator_snapshot_sha256")), "EVENT_ORACLE_PRIVATE_WINDOW_HASH")
    _require(_is_sha256(binding.get("event_tape_sha256")), "EVENT_ORACLE_PRIVATE_WINDOW_HASH")
    labels = [_branch_labels(branch, window) for branch in branches]
    keys = [row["candidate_key"] for row in labels]
    _require(len(keys) == len(set(keys)), "EVENT_ORACLE_DUPLICATE_CANDIDATE")
    noop_ids = [
        row["candidate_key"]
        for row, branch in zip(labels, branches)
        if isinstance(branch.get("candidate"), Mapping)
        and isinstance(branch["candidate"].get("full_environment_action"), Mapping)
        and branch["candidate"]["full_environment_action"].get("noop") is True
        and not branch["candidate"]["full_environment_action"].get("queue_operations")
        and not branch["candidate"]["full_environment_action"].get("dispatches")
    ]
    _require(len(noop_ids) == 1, "EVENT_ORACLE_EXPLICIT_NOOP_BRANCH")
    noop = next(row for row in labels if row["candidate_key"] == noop_ids[0])
    noop_branch = next(branch for branch in branches if branch.get("candidate_key") == noop["candidate_key"])
    first_action = noop_branch["trajectory"]["steps"][0]["full_environment_action"]
    _require(first_action.get("noop") is True, "EVENT_ORACLE_NOOP_NOT_EXECUTED")

    legal_labels = [row for row in labels if row["legal"]]
    labels_by_key = {row["candidate_key"]: row for row in labels}
    distinct_representatives: list[dict[str, Any]] = []
    for label in legal_labels:
        if _signature(label) is None:
            continue
        if not any(not _outcomes_differ(label, current) for current in distinct_representatives):
            distinct_representatives.append(label)
    valid_candidate_count = sum(_signature(row) is not None for row in legal_labels)
    valid_window = valid_candidate_count >= 2
    distinct_outcomes = len(distinct_representatives)

    for label in labels:
        reference = noop
        deltas: dict[str, Any] = {}
        for name in MODEL_HEADS:
            target, base = label["model_targets"][name], reference["model_targets"][name]
            values, base_values = target["values"], base["values"]
            valid, base_valid = target["valid"], base["valid"]
            if isinstance(values, list) and isinstance(valid, list):
                base_mask = base_valid if isinstance(base_valid, list) else []
                base_list = base_values if isinstance(base_values, list) else []
                common = [
                    float(value) - float(base_list[index])
                    if index < len(base_list) and index < len(base_mask) and is_valid and base_mask[index]
                    else 0.0
                    for index, (value, is_valid) in enumerate(zip(values, valid))
                ]
                delta_valid = [
                    bool(is_valid and index < len(base_mask) and base_mask[index])
                    for index, is_valid in enumerate(valid)
                ]
                deltas[name] = {"values": common, "valid": delta_valid}
            else:
                deltas[name] = {
                    "value": float(values) - float(base_values) if valid and base_valid else 0.0,
                    "valid": bool(valid and base_valid),
                }
        label["delta_vs_noop"] = deltas
        label["reference_candidate_key"] = noop["candidate_key"]

    public_observation = window.get("public_observation", {})
    public_task_ids = sorted({
        str(task.get("id")) for task in public_observation.get("tasks", [])
        if isinstance(task, Mapping) and task.get("id") is not None
    }) if isinstance(public_observation, Mapping) else []
    all_truth_task_ids = sorted({row["id"] for label in labels for row in label["task_outcomes"]})
    return {
        "schema": SCHEMA,
        "window_id": window.get("window_id"),
        "parent_id": window["parent_id"],
        "split": window["split"],
        "window_start": int(window["window_start"]),
        "window_end": int(window.get("window_end", 0)),
        "event_tape_id": window["event_tape_id"],
        "label_source": "independent_recomputation_from_raw_executed_trace",
        "producer_label_fields_read": False,
        "explicit_noop_candidate_key": noop["candidate_key"],
        "public_task_ids_at_start": public_task_ids,
        "all_truth_task_ids_including_non_targets": all_truth_task_ids,
        "valid_candidate_count": valid_candidate_count,
        "legal_candidate_count": len(legal_labels),
        "distinct_legal_outcome_count": distinct_outcomes,
        "valid_window": valid_window,
        "valid_window_has_two_distinct_legal_outcomes": valid_window and distinct_outcomes >= 2,
        "candidate_labels": labels,
        "outcome_pair_differences": [
            {"left_candidate_key": left["candidate_key"], "right_candidate_key": right["candidate_key"],
             "differing_physical_heads": _differing_physical_heads(left, right)}
            for index, left in enumerate(legal_labels)
            for right in legal_labels[index + 1:]
            if _outcomes_differ(left, right)
        ],
        "source_trace_sha256": _digest([branch["trajectory"] for branch in branches]),
    }


def make_model_sample(window: Mapping[str, Any], label_report: Mapping[str, Any]) -> dict[str, Any]:
    """Pack public inputs and independently computed per-candidate targets."""

    candidate_records = [
        dict(branch["candidate"])
        for branch in window["branches"]
        if branch["candidate"].get("legal") is True
    ]
    label_by_key = {row["candidate_key"]: row for row in label_report["candidate_labels"]}
    candidate_records.sort(key=lambda row: next(
        index for index, branch in enumerate(window["branches"])
        if branch["candidate"].get("candidate_key") == row.get("candidate_key")
    ))
    candidate_keys = [str(row["candidate_key"]) for row in candidate_records]
    targets: dict[str, dict[str, Any]] = {}
    for head in MODEL_HEADS:
        values = [label_by_key[key]["model_targets"][head]["values"] for key in candidate_keys]
        valid = [label_by_key[key]["model_targets"][head]["valid"] for key in candidate_keys]
        targets[head] = {"values": values, "valid": valid}
    history = window.get("history")
    if not isinstance(history, list) or not history:
        observation = window.get("public_observation")
        history = [observation] if isinstance(observation, Mapping) else []
    _require(bool(history), "EVENT_ORACLE_PUBLIC_HISTORY")

    # Provenance binds persisted data but is excluded from model candidate tensors.
    model_candidates = []
    for candidate in candidate_records:
        model_candidates.append({
            key: candidate[key]
            for key in ("candidate_key", "full_environment_action", "candidate_scope", "legal")
            if key in candidate
        })
    return {
        "sample_id": str(window.get("window_id")),
        "split": str(window["split"]),
        "parent_id": str(window["parent_id"]),
        "history": history[-4:],
        "candidates": model_candidates,
        "targets": targets,
        "provenance": {
            "event_tape_id": window["event_tape_id"],
            "window_start": int(window["window_start"]),
            "window_end": int(window.get("window_end", 0)),
            "oracle_schema": SCHEMA,
            "label_report_sha256": _digest(label_report),
        },
    }
