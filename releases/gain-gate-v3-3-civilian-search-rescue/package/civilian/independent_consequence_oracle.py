"""Independent replay and consequence-label audit for persisted N2 windows.

This module intentionally does not import the branch label producers. It
replays the recorded action IDs from each immutable snapshot and derives task
and global outcomes directly from the resulting environment state and events.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable

from branching import restore_snapshot


GLOBAL_NAMES = (
    "other_order_lateness",
    "unmet_demand",
    "energy",
    "reassignment",
    "global_utility_delta",
)
GLOBAL_UNITS = ("count", "count", "energy_units", "count", "reward")
TASK_NAMES = ("delivery_on_time", "inspection_on_time", "report_on_time")
TASK_UNITS = ("indicator", "indicator", "indicator")
GLOBAL_SEMANTICS = {
    "other_order_lateness": "candidate minus reference count of newly late non-target delivery orders after window_start at the common endpoint",
    "unmet_demand": "candidate minus reference count of undelivered delivery tasks",
    "energy": "candidate minus reference move/service energy units",
    "reassignment": "candidate minus reference count of reassigned tasks",
    "global_utility_delta": "candidate minus reference undiscounted environment reward over the common physical window",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _key_digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _contract_digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value) + b"\n").hexdigest().upper()


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _mapping(value: Any, code: str) -> dict[str, Any]:
    _require(isinstance(value, dict), code)
    return value


_SAFETY_STATE_FIELDS = (
    "status", "phase", "progress", "service_progress", "cargo_onboard",
    "assignee", "active_platform_id", "queue_owner", "stock_commitment", "safety_paused",
)

_SAFETY_PAUSE_TYPES = {"safety_task_paused", "safety_task_capability_paused"}
_SAFETY_RESUME_TYPES = {"safety_task_resumed", "safety_task_capability_resumed"}
_CAPABILITY_PAUSE_REASONS = {
    "capability_lost", "capacity_lost", "active_task_capability_lost", "active_task_capacity_lost",
}
_CAPABILITY_RESTORE_REASONS = {"active_task_capability_restored", "active_task_capacity_restored"}


def _validate_safety_transition_event(event: dict[str, Any]) -> str:
    _require(event.get("schema") == "gain-safety-task-transition/1", "N2_ORACLE_SAFETY_TRANSITION_SCHEMA")
    event_type = event.get("type")
    _require(event_type in _SAFETY_PAUSE_TYPES | _SAFETY_RESUME_TYPES | {"safety_task_released"}, "N2_ORACLE_SAFETY_TRANSITION_TYPE")
    _require(isinstance(event.get("task_id"), str) and bool(event.get("task_id")), "N2_ORACLE_SAFETY_TRANSITION_TASK")
    _require(type(event.get("time")) is int and event.get("time") >= 0, "N2_ORACLE_SAFETY_TRANSITION_TIME")
    states: list[dict[str, Any]] = []
    for side in ("before", "after"):
        state = _mapping(event.get(side), "N2_ORACLE_SAFETY_TRANSITION_STATE")
        _require(all(field in state for field in _SAFETY_STATE_FIELDS), "N2_ORACLE_SAFETY_TRANSITION_STATE_FIELDS")
        _require(isinstance(state.get("status"), str) and isinstance(state.get("phase"), str), "N2_ORACLE_SAFETY_TRANSITION_STATE_TEXT")
        _require(type(state.get("safety_paused")) is bool, "N2_ORACLE_SAFETY_TRANSITION_PAUSE_FLAG")
        pause_reason = state.get("safety_pause_reason")
        _require(pause_reason is None or (isinstance(pause_reason, str) and bool(pause_reason.strip())), "N2_ORACLE_SAFETY_TRANSITION_PAUSE_REASON")
        for field in ("progress", "service_progress", "cargo_onboard"):
            number = state.get(field)
            _require(isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(float(number)) and float(number) >= 0.0, "N2_ORACLE_SAFETY_TRANSITION_STATE_NUMBER")
        commitment = state.get("stock_commitment")
        if commitment is not None:
            commitment = _mapping(commitment, "N2_ORACLE_SAFETY_TRANSITION_STOCK")
            _require(isinstance(commitment.get("depot_id"), str) and isinstance(commitment.get("commodity"), str), "N2_ORACLE_SAFETY_TRANSITION_STOCK_ID")
            quantity = commitment.get("quantity")
            _require(isinstance(quantity, (int, float)) and not isinstance(quantity, bool) and math.isfinite(float(quantity)) and float(quantity) >= 0.0, "N2_ORACLE_SAFETY_TRANSITION_STOCK_QUANTITY")
        states.append(dict(state))
    _require(event.get("event_sha256") == _key_digest({key: value for key, value in event.items() if key != "event_sha256"}), "N2_ORACLE_SAFETY_TRANSITION_HASH")
    before, after = states
    preserved = ("phase", "progress", "service_progress", "cargo_onboard", "assignee", "active_platform_id", "queue_owner", "stock_commitment")
    before_reason = before.get("safety_pause_reason")
    after_reason = after.get("safety_pause_reason")
    event_reason = event.get("reason")
    _require(event_reason is None or (isinstance(event_reason, str) and bool(event_reason.strip())), "N2_ORACLE_SAFETY_TRANSITION_REASON")
    if event_type == "safety_task_capability_paused":
        _require((after_reason is None or after_reason in _CAPABILITY_PAUSE_REASONS) and before_reason is None, "N2_ORACLE_SAFETY_CAPABILITY_PAUSE_REASON")
        _require(event_reason is None or event_reason in _CAPABILITY_PAUSE_REASONS, "N2_ORACLE_SAFETY_CAPABILITY_EVENT_REASON")
    elif event_type == "safety_task_capability_resumed":
        _require((before_reason is None or before_reason in _CAPABILITY_PAUSE_REASONS) and after_reason is None, "N2_ORACLE_SAFETY_CAPABILITY_RESUME_REASON")
        _require(event_reason is None or event_reason in _CAPABILITY_RESTORE_REASONS, "N2_ORACLE_SAFETY_CAPABILITY_RESTORE_REASON")
    elif event_type == "safety_task_paused":
        _require(after_reason not in _CAPABILITY_PAUSE_REASONS, "N2_ORACLE_SAFETY_PAUSE_TYPE_REASON")
    elif event_type == "safety_task_resumed":
        _require(before_reason not in _CAPABILITY_PAUSE_REASONS, "N2_ORACLE_SAFETY_RESUME_TYPE_REASON")
    if event_type in _SAFETY_PAUSE_TYPES:
        _require(not before["safety_paused"] and after["safety_paused"] and after["status"] == "paused", "N2_ORACLE_SAFETY_PAUSE_STATE")
        _require(all(before[key] == after[key] for key in preserved), "N2_ORACLE_SAFETY_PAUSE_MUTATION")
    elif event_type in _SAFETY_RESUME_TYPES:
        _require(before["safety_paused"] and not after["safety_paused"], "N2_ORACLE_SAFETY_RESUME_STATE")
        _require(all(before[key] == after[key] for key in preserved), "N2_ORACLE_SAFETY_RESUME_MUTATION")
    else:
        _require(not before["safety_paused"] and not after["safety_paused"], "N2_ORACLE_SAFETY_RELEASE_STATE")
        _require(after["status"] == "pending" and after["phase"] == "released", "N2_ORACLE_SAFETY_RELEASE_TARGET")
        _require(all(after[key] is None for key in ("assignee", "active_platform_id", "queue_owner", "stock_commitment")), "N2_ORACLE_SAFETY_RELEASE_OWNER")
        _require(after["cargo_onboard"] == 0.0 and after["progress"] == 0.0 and after["service_progress"] == 0.0, "N2_ORACLE_SAFETY_RELEASE_PHYSICAL")
    return str(event["event_sha256"])


def validate_safety_transition_events(events: Any) -> dict[str, Any]:
    """Check persisted same-tick safety transitions independently of labels."""
    _require(isinstance(events, list), "N2_ORACLE_SAFETY_EVENTS")
    hashes: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("schema") == "gain-safety-task-transition/1":
            hashes.append(_validate_safety_transition_event(event))
        elif "safety_task_" in str(event.get("type", "")):
            raise ValueError("N2_ORACLE_SAFETY_TRANSITION_SCHEMA_MISSING")
        if event.get("type") == "cargo_lost":
            quantity = event.get("quantity", event.get("amount", event.get("cargo_lost")))
            _require(isinstance(quantity, (int, float)) and not isinstance(quantity, bool) and math.isfinite(float(quantity)) and float(quantity) >= 0.0, "N2_ORACLE_CARGO_LOST_QUANTITY")
            before = event.get("cargo_before"); after = event.get("cargo_after")
            if before is not None or after is not None:
                _require(isinstance(before, (int, float)) and isinstance(after, (int, float)) and not isinstance(before, bool) and not isinstance(after, bool), "N2_ORACLE_CARGO_LOST_STATE")
                _require(float(after) <= float(before) + 1e-9 and math.isclose(float(before) - float(after), float(quantity), rel_tol=1e-9, abs_tol=1e-9), "N2_ORACLE_CARGO_LOST_PHYSICAL")
    return {"transition_event_count": len(hashes), "transition_event_hashes": sorted(hashes)}


def _task_indicator(task: dict[str, Any], name: str, now: int, horizon: int) -> tuple[float, bool, str | None]:
    deadline = task.get("deadline")
    terminal = now >= horizon
    deadline_passed = deadline is not None and now >= int(deadline)
    task_terminal = task.get("status") in {"completed", "failed", "expired"}

    if name == "delivery_on_time":
        applicable = task.get("kind") == "delivery"
        resolved = task.get("on_time") is not None or task_terminal or deadline_passed or terminal
        value = 1.0 if task.get("delivered") and task.get("on_time") is True else 0.0
    elif name == "inspection_on_time":
        applicable = task.get("kind") == "inspection"
        resolved = task.get("on_time") is not None or task_terminal or deadline_passed or terminal
        value = 1.0 if task.get("status") == "completed" and task.get("on_time") is True else 0.0
    elif name == "report_on_time":
        applicable = task.get("kind") == "inspection"
        report_at = task.get("report_received_at")
        resolved = report_at is not None or task_terminal or deadline_passed or terminal
        value = 1.0 if report_at is not None and (deadline is None or int(report_at) <= int(deadline)) else 0.0
    else:
        raise ValueError("N2_ORACLE_UNKNOWN_TASK_TARGET")

    if not applicable:
        return 0.0, False, "not_applicable"
    if not resolved:
        return 0.0, False, "unresolved_before_horizon"
    return value, True, None


def _task_targets(candidate: Any, reference: Any, task_ids: list[str]) -> dict[str, Any]:
    candidate_tasks = {str(task.get("id")): task for task in candidate._tasks.values()}
    reference_tasks = {str(task.get("id")): task for task in reference._tasks.values()}
    values: list[list[float]] = []
    candidate_values: list[list[float]] = []
    reference_values: list[list[float]] = []
    valid: list[list[bool]] = []
    unknown_reasons: list[list[str | None]] = []

    for task_id in task_ids:
        candidate_task = candidate_tasks.get(task_id)
        reference_task = reference_tasks.get(task_id)
        if candidate_task is None or reference_task is None:
            values.append([0.0] * len(TASK_NAMES))
            candidate_values.append([0.0] * len(TASK_NAMES))
            reference_values.append([0.0] * len(TASK_NAMES))
            valid.append([False] * len(TASK_NAMES))
            unknown_reasons.append(["task_missing"] * len(TASK_NAMES))
            continue

        row_values: list[float] = []
        row_candidate: list[float] = []
        row_reference: list[float] = []
        row_valid: list[bool] = []
        row_reasons: list[str | None] = []
        for name in TASK_NAMES:
            candidate_value, candidate_valid, candidate_reason = _task_indicator(
                candidate_task, name, int(candidate.time), int(candidate.scenario["horizon"])
            )
            reference_value, reference_valid, reference_reason = _task_indicator(
                reference_task, name, int(reference.time), int(reference.scenario["horizon"])
            )
            is_valid = bool(candidate_valid and reference_valid)
            row_values.append(float(candidate_value - reference_value))
            row_candidate.append(float(candidate_value))
            row_reference.append(float(reference_value))
            row_valid.append(is_valid)
            row_reasons.append(None if is_valid else ";".join(
                reason for reason in (candidate_reason, reference_reason) if reason
            ))
        values.append(row_values)
        candidate_values.append(row_candidate)
        reference_values.append(row_reference)
        valid.append(row_valid)
        unknown_reasons.append(row_reasons)

    return {
        "task_ids": task_ids,
        "names": list(TASK_NAMES),
        "units": list(TASK_UNITS),
        "values": values,
        "candidate_values": candidate_values,
        "reference_values": reference_values,
        "valid": valid,
        "unknown_reasons": unknown_reasons,
    }


def _target_task_ids(actions: list[str], catalog_by_id: dict[str, dict[str, Any]]) -> set[str]:
    return {
        str(catalog_by_id[action]["task_id"])
        for action in actions
        if action != "NOOP" and catalog_by_id[action].get("task_id") is not None
    }


def _global_targets(
    candidate: Any,
    reference: Any,
    candidate_return: float,
    reference_return: float,
    window_start: int,
    excluded_task_ids: set[str],
) -> dict[str, Any]:
    def outcomes(env: Any) -> tuple[float, float, float, bool, list[str]]:
        now = int(env.time)
        horizon = int(env.scenario["horizon"])
        terminal = bool(env._terminal) or now >= horizon
        overdue = 0
        unmet = 0
        reassigned = 0
        unresolved: list[str] = []
        for raw_task in env._tasks.values():
            task = raw_task
            task_id = str(task.get("id", ""))
            is_delivery = task.get("kind") == "delivery"
            deadline = task.get("deadline")
            deadline_passed = deadline is not None and now >= int(deadline)
            resolved = (
                task.get("on_time") is not None
                or task.get("status") in {"completed", "failed", "expired"}
                or bool(task.get("deadline_missed"))
            )
            is_new_non_target = (
                is_delivery
                and task_id not in excluded_task_ids
                and deadline is not None
                and int(deadline) > window_start
            )
            if is_new_non_target and not (resolved or deadline_passed or terminal):
                unresolved.append(task_id)
            if is_new_non_target and (task.get("on_time") is False or bool(task.get("deadline_missed"))):
                overdue += 1
            if is_delivery and not bool(task.get("delivered")):
                unmet += 1
            if bool(task.get("_reassigned")):
                reassigned += 1
        return float(overdue), float(unmet), float(reassigned), not unresolved, unresolved

    def energy(env: Any) -> float:
        return float(sum(
            float(event.get("energy", 0.0))
            for event in env.events
            if event.get("type") in {"move", "service"}
            and int(event.get("time", -1)) > window_start
        ))

    candidate_overdue, candidate_unmet, candidate_reassigned, candidate_valid, candidate_unknown = outcomes(candidate)
    reference_overdue, reference_unmet, reference_reassigned, reference_valid, reference_unknown = outcomes(reference)
    candidate_values = [
        candidate_overdue,
        candidate_unmet,
        energy(candidate),
        candidate_reassigned,
        float(candidate_return),
    ]
    reference_values = [
        reference_overdue,
        reference_unmet,
        energy(reference),
        reference_reassigned,
        float(reference_return),
    ]
    values = [float(left - right) for left, right in zip(candidate_values, reference_values)]
    valid = [candidate_valid and reference_valid, True, True, True, True]
    unknown = [
        None if valid[0] else ";".join(
            ([f"candidate_unresolved_other_delivery:{','.join(candidate_unknown)}"] if candidate_unknown else [])
            + ([f"reference_unresolved_other_delivery:{','.join(reference_unknown)}"] if reference_unknown else [])
        ),
        None,
        None,
        None,
        None,
    ]
    return {
        "names": list(GLOBAL_NAMES),
        "units": list(GLOBAL_UNITS),
        "values": values,
        "candidate_values": candidate_values,
        "reference_values": reference_values,
        "valid": valid,
        "unknown_reasons": unknown,
        "hidden_state_label_only": True,
        "semantics": dict(GLOBAL_SEMANTICS),
    }


def _recompute_targets(
    candidate: Any,
    reference: Any,
    candidate_return: float,
    reference_return: float,
    window_start: int,
    public_task_ids: list[str],
    excluded_task_ids: set[str],
) -> dict[str, Any]:
    return {
        "global": _global_targets(
            candidate,
            reference,
            candidate_return,
            reference_return,
            window_start,
            excluded_task_ids,
        ),
        "task": _task_targets(candidate, reference, public_task_ids),
    }


def _outcome_signature(targets: dict[str, Any]) -> bytes | None:
    """Normalize valid observed deltas without treating unknown as an outcome."""

    global_targets = targets["global"]
    global_values = {
        name: (0.0 if abs(float(value)) < 1e-12 else float(value))
        for name, value, valid in zip(
            global_targets["names"], global_targets["values"], global_targets["valid"]
        )
        if valid
    }
    task_targets = targets["task"]
    by_name: dict[str, list[float]] = {name: [] for name in task_targets["names"]}
    for value_row, valid_row in zip(task_targets["values"], task_targets["valid"]):
        for name, value, valid in zip(task_targets["names"], value_row, valid_row):
            if valid:
                by_name[name].append(0.0 if abs(float(value)) < 1e-12 else float(value))
    task_values = {
        name: sorted(values)
        for name, values in by_name.items()
        if values
    }
    if not global_values and not task_values:
        return None
    return _canonical({"global": global_values, "task": task_values})


def _valid_global_values(targets: dict[str, Any]) -> dict[str, float]:
    global_targets = targets["global"]
    return {
        name: (0.0 if abs(float(value)) < 1e-12 else float(value))
        for name, value, valid in zip(
            global_targets["names"], global_targets["values"], global_targets["valid"]
        )
        if valid
    }


def _global_consequence_differences(
    candidates: list[tuple[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for left_index, (left_id, left_values) in enumerate(candidates):
        for right_id, right_values in candidates[left_index + 1:]:
            common_names = sorted(left_values.keys() & right_values.keys())
            common_values = {
                name: {"left": left_values[name], "right": right_values[name]}
                for name in common_names
            }
            changed = {
                name: {
                    **common_values[name],
                    "delta": float(left_values[name] - right_values[name]),
                }
                for name in common_names
                if not math.isclose(left_values[name], right_values[name], rel_tol=1e-12, abs_tol=1e-12)
            }
            if changed:
                differences.append({
                    "left_candidate_id": left_id,
                    "right_candidate_id": right_id,
                    "common_valid_global_values": common_values,
                    "global_differences": changed,
                })
    return differences


def _validate_window(
    window: Any,
    step_callback: Callable[[Any, list[str], dict[str, Any]], tuple[Any, float, bool, bool, dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    item = _mapping(window, "N2_ORACLE_WINDOW_TYPE")
    snapshot_json = item.get("snapshot_json")
    _require(isinstance(snapshot_json, str), "N2_ORACLE_SNAPSHOT_TYPE")
    try:
        payload = json.loads(snapshot_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("N2_ORACLE_SNAPSHOT_JSON") from exc
    _mapping(payload, "N2_ORACLE_SNAPSHOT_PAYLOAD")
    env = restore_snapshot(snapshot_json)

    parent_id = payload.get("parent_id")
    repeat = payload.get("repeat")
    window_id = payload.get("window_id")
    event_tape_id = payload.get("event_tape_id")
    window_ids = item.get("window_ids")
    _require(
        isinstance(parent_id, str)
        and type(repeat) is int
        and isinstance(window_id, str)
        and isinstance(event_tape_id, str)
        and window_ids == [parent_id, repeat, window_id],
        "N2_ORACLE_WINDOW_SNAPSHOT_BINDING",
    )
    _require(str(env.scenario.get("scenario_id")) == parent_id, "N2_ORACLE_PARENT_SCENARIO")
    _require(int(payload.get("time", -1)) == int(env.time), "N2_ORACLE_WINDOW_TIME")

    catalog = env.candidates()
    _require(isinstance(catalog, list) and catalog, "N2_ORACLE_CATALOG")
    catalog_by_id = {str(entry.get("id")): entry for entry in catalog}
    _require(len(catalog_by_id) == len(catalog) and "NOOP" in catalog_by_id, "N2_ORACLE_CATALOG_IDENTITY")
    policy_input = item.get("policy_input")
    expected_policy_input = {"observation": payload.get("public_prefix"), "candidates": payload.get("candidate_catalog")}
    _require(policy_input == expected_policy_input, "N2_ORACLE_PUBLIC_PREFIX_BINDING")

    records = item.get("branch_records")
    _require(isinstance(records, list) and records, "N2_ORACLE_BRANCH_RECORDS")
    evidence = _mapping(item.get("trajectory_evidence"), "N2_ORACLE_TRAJECTORY_EVIDENCE")
    transitions = _mapping(item.get("trajectory_transitions"), "N2_ORACLE_TRAJECTORY_TRANSITIONS")
    normalized_records = [_mapping(record, "N2_ORACLE_BRANCH_RECORD") for record in records]
    record_ids = [record.get("candidate_id") for record in normalized_records]
    _require(all(isinstance(value, str) and value for value in record_ids), "N2_ORACLE_CANDIDATE_ID")
    _require(len(record_ids) == len(set(record_ids)), "N2_ORACLE_DUPLICATE_CANDIDATE")
    expected_trajectory_keys = set(record_ids) | {"NOOP"}
    _require(set(evidence) == expected_trajectory_keys, "N2_ORACLE_ACTION_TRAJECTORY_KEYS")
    _require(set(transitions) == expected_trajectory_keys, "N2_ORACLE_TRANSITION_TRAJECTORY_KEYS")

    noop_records = [record for record in normalized_records if record.get("candidate_action_batch") == ["NOOP"]]
    _require(len(noop_records) == 1, "N2_ORACLE_NOOP_BASELINE_COUNT")
    noop_record = noop_records[0]
    noop_id = str(noop_record["candidate_id"])
    _require(evidence["NOOP"] == evidence[noop_id], "N2_ORACLE_NOOP_ACTION_ALIAS")
    _require(transitions["NOOP"] == transitions[noop_id], "N2_ORACLE_NOOP_TRANSITION_ALIAS")
    _require(noop_record.get("candidate_is_noop") is True, "N2_ORACLE_NOOP_MARKER")

    state_digest = str(payload.get("state_sha256", ""))
    prefix_digest = str(payload.get("source_prefix_sha256", ""))
    scenario_digest = str(payload.get("scenario_sha256", ""))
    expected_event_tape_id = _key_digest({
        "scenario_id": parent_id,
        "disturbances": env.scenario.get("disturbances", []),
    })
    _require(event_tape_id == expected_event_tape_id, "N2_ORACLE_EVENT_TAPE_IDENTITY")
    expected_policy_digest = _key_digest(expected_policy_input)
    window_start = int(payload["time"])
    public_task_ids = [
        str(task["id"])
        for task in payload.get("public_prefix", {}).get("tasks", [])
        if isinstance(task, dict) and task.get("id") is not None
    ]
    _require(len(public_task_ids) == len(set(public_task_ids)), "N2_ORACLE_PUBLIC_TASK_IDS")

    def check_action_batch(batch: Any, current_catalog: list[dict[str, Any]]) -> list[str]:
        _require(isinstance(batch, list) and all(type(action) is str for action in batch), "N2_ORACLE_ACTION_BATCH_TYPE")
        _require(len(batch) == len(set(batch)), "N2_ORACLE_ACTION_BATCH_DUPLICATE")
        _require("NOOP" not in batch, "N2_ORACLE_NOOP_IN_TRANSITION")
        current_ids = {str(entry.get("id")) for entry in current_catalog}
        _require(all(action in current_ids for action in batch), "N2_ORACLE_ILLEGAL_ACTION_ID")
        return batch

    replayed_environment_step_count = 0

    def replay(
        action_rows: Any,
        transition_rows: Any,
        first_action_batch: list[str],
        *,
        replay_candidate_id: str,
        replay_role: str,
    ) -> tuple[Any, float]:
        nonlocal replayed_environment_step_count
        _require(isinstance(action_rows, list) and isinstance(transition_rows, list), "N2_ORACLE_TRAJECTORY_TYPE")
        _require(bool(action_rows) and len(action_rows) == len(transition_rows), "N2_ORACLE_TRAJECTORY_LENGTH")
        branch = restore_snapshot(snapshot_json)
        total_reward = 0.0
        for step_index, (raw_actions, saved_transition) in enumerate(zip(action_rows, transition_rows)):
            _require(isinstance(raw_actions, list), "N2_ORACLE_STEP_ACTIONS")
            _mapping(saved_transition, "N2_ORACLE_SAVED_TRANSITION")
            step_actions = check_action_batch(raw_actions, branch.candidates())
            if step_index == 0:
                _require(step_actions == first_action_batch, "N2_ORACLE_INITIAL_ACTION_BINDING")
            try:
                metadata = {
                    "component": "independent_consequence_oracle",
                    "parent_id": parent_id,
                    "repeat": repeat,
                    "window_id": window_id,
                    "candidate_id": replay_candidate_id,
                    "replay_role": replay_role,
                    "step_index": step_index + 1,
                }
                if step_callback is None:
                    step_result = branch.step(step_actions)
                else:
                    step_result = step_callback(branch, step_actions, metadata)
                _, reward, terminated, truncated, info = step_result
            except Exception as exc:
                raise ValueError("N2_ORACLE_REPLAY_ACTION_REJECTED") from exc
            replayed_environment_step_count += 1
            _require(math.isfinite(float(reward)), "N2_ORACLE_NONFINITE_REWARD")
            total_reward += float(reward)
            realized = info.get("transition")
            _require(realized == saved_transition, "N2_ORACLE_TRANSITION_REPLAY_MISMATCH")
            realized_actions = [str(action.get("id")) for action in realized.get("selected_actions", [])]
            _require(realized_actions == step_actions, "N2_ORACLE_REALIZED_ACTION_MISMATCH")
            _require(not (terminated or truncated) or step_index == len(action_rows) - 1, "N2_ORACLE_ACTIONS_AFTER_TERMINAL")
        return branch, total_reward

    noop_actions = evidence[noop_id]
    _require(isinstance(noop_actions, list) and bool(noop_actions), "N2_ORACLE_NOOP_TRAJECTORY")
    _require(noop_record.get("reference_action_batch") == ["NOOP"], "N2_ORACLE_NOOP_REFERENCE_ACTION")
    noop_branch, noop_return = replay(
        noop_actions,
        transitions[noop_id],
        [],
        replay_candidate_id=noop_id,
        replay_role="reference_baseline",
    )
    safety_validation_reports: dict[str, dict[str, Any]] = {
        noop_id: validate_safety_transition_events(noop_branch.events),
    }
    _require(noop_record.get("candidate_return") is not None, "N2_ORACLE_NOOP_RETURN")
    _require(math.isclose(float(noop_record["candidate_return"]), noop_return, rel_tol=1e-9, abs_tol=1e-9), "N2_ORACLE_NOOP_RETURN_MISMATCH")
    _require(noop_record.get("candidate_events") == noop_branch.events, "N2_ORACLE_NOOP_EVENT_MISMATCH")

    verified: list[dict[str, Any]] = []
    per_window_signatures: set[bytes] = set()
    global_candidates: list[tuple[str, dict[str, float]]] = []
    valid_non_noop_count = 0
    non_noop_count = 0
    for record in normalized_records:
        candidate_id = str(record["candidate_id"])
        batch = record.get("candidate_action_batch")
        _require(
            isinstance(batch, list)
            and bool(batch)
            and all(type(action) is str for action in batch),
            "N2_ORACLE_RECORDED_ACTION_BATCH",
        )
        _require(len(batch) == len(set(batch)), "N2_ORACLE_RECORDED_ACTION_DUPLICATE")
        _require("NOOP" not in batch or batch[-1] == "NOOP", "N2_ORACLE_RECORDED_NOOP_ORDER")
        _require(all(action == "NOOP" or action in catalog_by_id for action in batch), "N2_ORACLE_RECORDED_ILLEGAL_ACTION")
        _require(_key_digest(batch) == candidate_id, "N2_ORACLE_CANDIDATE_ACTION_IDENTITY")
        _require(record.get("candidate_action_batch_sha256") == _contract_digest(batch), "N2_ORACLE_ACTION_BATCH_DIGEST")
        _require(record.get("candidate_is_noop") is (batch == ["NOOP"]), "N2_ORACLE_NOOP_CLASSIFICATION")
        _require(record.get("parent_id") == parent_id, "N2_ORACLE_RECORD_PARENT")
        _require(record.get("repeat") == repeat, "N2_ORACLE_RECORD_REPEAT")
        _require(record.get("window_id") == window_id, "N2_ORACLE_RECORD_WINDOW")
        _require(record.get("event_tape_id") == event_tape_id, "N2_ORACLE_RECORD_EVENT_TAPE")
        _require(int(record.get("window_start", -1)) == window_start, "N2_ORACLE_RECORD_WINDOW_START")
        _require(record.get("source_state_sha256") == state_digest, "N2_ORACLE_RECORD_SOURCE_STATE")
        _require(record.get("source_prefix_sha256") == prefix_digest, "N2_ORACLE_RECORD_SOURCE_PREFIX")
        _require(record.get("scenario_sha256") == scenario_digest, "N2_ORACLE_RECORD_SCENARIO")
        _require(record.get("policy_input_sha256") == expected_policy_digest, "N2_ORACLE_RECORD_POLICY_INPUT")
        _require(record.get("reference_action_batch") == ["NOOP"], "N2_ORACLE_REFERENCE_ACTION")
        _require(record.get("reference_action_batch_sha256") == _contract_digest(["NOOP"]), "N2_ORACLE_REFERENCE_ACTION_DIGEST")
        first_actions = [action for action in batch if action != "NOOP"]
        if candidate_id != noop_id:
            non_noop_count += 1
        if candidate_id == noop_id:
            branch, branch_return = noop_branch, noop_return
        else:
            branch, branch_return = replay(
                evidence[candidate_id],
                transitions[candidate_id],
                first_actions,
                replay_candidate_id=candidate_id,
                replay_role="candidate",
            )
        safety_validation_reports[candidate_id] = validate_safety_transition_events(branch.events)
        _require(math.isclose(float(record.get("candidate_return")), branch_return, rel_tol=1e-9, abs_tol=1e-9), "N2_ORACLE_CANDIDATE_RETURN_MISMATCH")
        _require(record.get("candidate_events") == branch.events, "N2_ORACLE_CANDIDATE_EVENT_MISMATCH")
        _require(math.isclose(float(record.get("reference_return")), noop_return, rel_tol=1e-9, abs_tol=1e-9), "N2_ORACLE_REFERENCE_RETURN_MISMATCH")
        _require(record.get("reference_events") == noop_branch.events, "N2_ORACLE_REFERENCE_EVENT_MISMATCH")

        target_task_ids = _target_task_ids(batch, catalog_by_id)
        target_task_ids.update(_target_task_ids(["NOOP"], catalog_by_id))
        recomputed = _recompute_targets(
            branch,
            noop_branch,
            branch_return,
            noop_return,
            window_start,
            public_task_ids,
            target_task_ids,
        )
        _require(record.get("paired_targets") == recomputed, "N2_ORACLE_PAIRED_TARGETS_MISMATCH")
        global_candidates.append((candidate_id, _valid_global_values(recomputed)))
        signature = _outcome_signature(recomputed)
        if candidate_id != noop_id and signature is not None:
            per_window_signatures.add(signature)
            valid_non_noop_count += 1
        verified.append({
            "candidate_id": candidate_id,
            "is_noop": candidate_id == noop_id,
            "candidate_action_batch": batch,
            "replayed_return": branch_return,
            "paired_targets": recomputed,
        })

    global_differences = _global_consequence_differences(global_candidates)
    return verified, {
        "parent_id": parent_id,
        "repeat": repeat,
        "window_id": window_id,
        "event_tape_id": event_tape_id,
        "window_start": window_start,
        "branch_count": len(normalized_records),
        "non_noop_candidate_count": non_noop_count,
        "valid_non_noop_candidate_count": valid_non_noop_count,
        "distinct_valid_non_noop_consequences": len(per_window_signatures),
        "valid_global_candidate_count": len(global_candidates),
        "distinct_valid_global_consequence_count": len({
            _canonical(values) for _, values in global_candidates
        }),
        "qualifies_global_consequence_diversity": bool(global_differences),
        "global_consequence_differences": global_differences,
        "environment_step_count": replayed_environment_step_count,
        "safety_transition_validation": safety_validation_reports,
    }


def validate_collection_windows(
    windows: Any,
    step_callback: Callable[[Any, list[str], dict[str, Any]], tuple[Any, float, bool, bool, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Replay persisted windows and recompute every NOOP-relative target.

    Consequence diversity is qualified within one immutable window by a pair
    of replay-valid candidates, including NOOP, that differ on a shared-valid
    global target. Aggregate task and consequence counts remain diagnostics.
    """

    _require(isinstance(windows, list) and windows, "N2_ORACLE_WINDOWS_TYPE")
    verified_windows: list[dict[str, Any]] = []
    window_stats: list[dict[str, Any]] = []
    signatures: set[bytes] = set()
    identities: set[tuple[str, int, str]] = set()
    branch_count = 0
    valid_non_noop_count = 0
    environment_step_count = 0

    for window in windows:
        verified, stats = _validate_window(window, step_callback)
        identity = (stats["parent_id"], stats["repeat"], stats["window_id"])
        _require(identity not in identities, "N2_ORACLE_DUPLICATE_WINDOW")
        identities.add(identity)
        environment_step_count += stats["environment_step_count"]
        for result in verified:
            branch_count += 1
            if result["is_noop"]:
                continue
            signature = _outcome_signature(result["paired_targets"])
            if signature is not None:
                valid_non_noop_count += 1
                signatures.add(signature)
        verified_windows.append(verified)
        window_stats.append(stats)

    input_sha256 = hashlib.sha256(_canonical(windows)).hexdigest()
    recomputed_sha256 = hashlib.sha256(_canonical(verified_windows)).hexdigest()
    distinct_valid_consequence_count = len(signatures)
    qualifying_global_differences = [
        {
            "parent_id": stats["parent_id"],
            "repeat": stats["repeat"],
            "window_id": stats["window_id"],
            "candidate_pairs": stats["global_consequence_differences"],
        }
        for stats in window_stats
        if stats["qualifies_global_consequence_diversity"]
    ]
    qualifying_window_count = len(qualifying_global_differences)
    audit_body = {
        "schema": "n2-independent-consequence-oracle/1",
        "status": "passed" if qualifying_window_count >= 1 else "insufficient_consequence_diversity",
        "input_windows_sha256": input_sha256,
        "recomputed_targets_sha256": recomputed_sha256,
        "window_count": len(windows),
        "branch_count": branch_count,
        "valid_non_noop_candidate_count": valid_non_noop_count,
        "distinct_valid_consequence_count": distinct_valid_consequence_count,
        "qualifying_window_count": qualifying_window_count,
        "qualifying_global_differences": qualifying_global_differences,
        "environment_step_count": environment_step_count,
        "windows": window_stats,
    }
    evidence_sha256 = hashlib.sha256(_canonical(audit_body)).hexdigest()
    return {**audit_body, "evidence_sha256": evidence_sha256}
