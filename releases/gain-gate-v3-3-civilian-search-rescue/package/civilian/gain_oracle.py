"""Independent recomputation and binding checks for gain-label evidence.

This module intentionally does not import the producer or production scorer.
It recomputes costs from raw branch events and endpoint truth using the frozen
protocol weights carried in each pair.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


LABEL_SCHEMA = "gain-counterfactual-label/2"
PAIR_SCHEMA = "gain-counterfactual-pair/1"
TRACE_SCHEMA = "gain-counterfactual-trace/1"
RECEIPT_SCHEMA = "gain-continuation-receipt/1"
SAFETY_EVIDENCE_FIELDS = (
    "safety_report", "safety_disposition", "safety_apply_receipt", "unknown_reason", "detection",
)
NOOP_ACTION = {
    "action_type": "reconstruct", "scope": "NOOP", "noop": True,
    "queue_operations": [], "dispatches": [],
}
_COST_KEYS = {
    "completion", "delay", "energy", "risk", "communication", "switch",
    "reconstruction", "reconstruction_size", "cpu_second",
}


def _canonical(value: Any) -> bytes:
    try:
        rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("GAIN_ORACLE_NON_JSON_EVIDENCE") from exc
    return (rendered + "\n").encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _number(value: Any, code: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(code)
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(code)
    return number


def _verify_digest(envelope: Mapping[str, Any], field: str, *, omitted: str) -> None:
    expected = envelope.get(field)
    body = {key: value for key, value in envelope.items() if key != omitted}
    if not isinstance(expected, str) or expected != _digest(body):
        raise ValueError(f"GAIN_ORACLE_{omitted.upper()}_HASH_MISMATCH")


def _live_digest(value: Any) -> str:
    """Digest used by live event-environment safety receipts."""
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(rendered.encode("ascii")).hexdigest()


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


def _validate_safety_state(state: Any, code: str) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError(code)
    for field in _SAFETY_STATE_FIELDS:
        if field not in state:
            raise ValueError(code)
    if not isinstance(state.get("status"), str) or not isinstance(state.get("phase"), str):
        raise ValueError(code)
    if type(state.get("safety_paused")) is not bool:
        raise ValueError(code)
    pause_reason = state.get("safety_pause_reason")
    if pause_reason is not None and (not isinstance(pause_reason, str) or not pause_reason.strip()):
        raise ValueError(code)
    for field in ("progress", "service_progress", "cargo_onboard"):
        _number(state.get(field), code)
    commitment = state.get("stock_commitment")
    if commitment is not None:
        if not isinstance(commitment, Mapping):
            raise ValueError(code)
        for field in ("depot_id", "commodity", "quantity"):
            if field not in commitment:
                raise ValueError(code)
        if not isinstance(commitment.get("depot_id"), str) or not isinstance(commitment.get("commodity"), str):
            raise ValueError(code)
        _number(commitment.get("quantity"), code)
    return dict(state)


def _validate_safety_transition(event: Mapping[str, Any]) -> str:
    if event.get("schema") != "gain-safety-task-transition/1":
        raise ValueError("GAIN_ORACLE_SAFETY_TRANSITION_SCHEMA_INVALID")
    event_type = event.get("type")
    if event_type not in _SAFETY_PAUSE_TYPES | _SAFETY_RESUME_TYPES | {"safety_task_released"}:
        raise ValueError("GAIN_ORACLE_SAFETY_TRANSITION_TYPE_INVALID")
    if not isinstance(event.get("task_id"), str) or not event.get("task_id"):
        raise ValueError("GAIN_ORACLE_SAFETY_TRANSITION_TASK_INVALID")
    if isinstance(event.get("time"), bool) or not isinstance(event.get("time"), int) or event.get("time") < 0:
        raise ValueError("GAIN_ORACLE_SAFETY_TRANSITION_TIME_INVALID")
    before = _validate_safety_state(event.get("before"), "GAIN_ORACLE_SAFETY_TRANSITION_BEFORE_INVALID")
    after = _validate_safety_state(event.get("after"), "GAIN_ORACLE_SAFETY_TRANSITION_AFTER_INVALID")
    event_hash = event.get("event_sha256")
    body = {key: value for key, value in event.items() if key != "event_sha256"}
    if not isinstance(event_hash, str) or event_hash != _live_digest(body):
        raise ValueError("GAIN_ORACLE_SAFETY_TRANSITION_HASH_INVALID")
    preserved = ("phase", "progress", "service_progress", "cargo_onboard", "assignee",
                 "active_platform_id", "queue_owner", "stock_commitment")
    before_reason = before.get("safety_pause_reason")
    after_reason = after.get("safety_pause_reason")
    event_reason = event.get("reason")
    if event_reason is not None and (not isinstance(event_reason, str) or not event_reason.strip()):
        raise ValueError("GAIN_ORACLE_SAFETY_TRANSITION_REASON_INVALID")
    if event_type == "safety_task_capability_paused":
        if (after_reason is not None and after_reason not in _CAPABILITY_PAUSE_REASONS) or before_reason is not None:
            raise ValueError("GAIN_ORACLE_SAFETY_CAPABILITY_PAUSE_REASON_INVALID")
        if event_reason is not None and event_reason not in _CAPABILITY_PAUSE_REASONS:
            raise ValueError("GAIN_ORACLE_SAFETY_CAPABILITY_EVENT_REASON_INVALID")
    elif event_type == "safety_task_capability_resumed":
        if (before_reason is not None and before_reason not in _CAPABILITY_PAUSE_REASONS) or after_reason is not None:
            raise ValueError("GAIN_ORACLE_SAFETY_CAPABILITY_RESUME_REASON_INVALID")
        if event_reason is not None and event_reason not in _CAPABILITY_RESTORE_REASONS:
            raise ValueError("GAIN_ORACLE_SAFETY_CAPABILITY_RESTORE_REASON_INVALID")
    elif event_type in {"safety_task_paused", "safety_task_resumed"}:
        if event_type == "safety_task_paused" and after_reason in _CAPABILITY_PAUSE_REASONS:
            raise ValueError("GAIN_ORACLE_SAFETY_PAUSE_TYPE_REASON_MISMATCH")
        if event_type == "safety_task_resumed" and before_reason in _CAPABILITY_PAUSE_REASONS:
            raise ValueError("GAIN_ORACLE_SAFETY_RESUME_TYPE_REASON_MISMATCH")
    if event_type in _SAFETY_PAUSE_TYPES:
        if before["safety_paused"] or not after["safety_paused"] or after["status"] != "paused":
            raise ValueError("GAIN_ORACLE_SAFETY_PAUSE_STATE_INVALID")
        if any(before[key] != after[key] for key in preserved):
            raise ValueError("GAIN_ORACLE_SAFETY_PAUSE_MUTATED_PHYSICAL_STATE")
    elif event_type in _SAFETY_RESUME_TYPES:
        if not before["safety_paused"] or after["safety_paused"]:
            raise ValueError("GAIN_ORACLE_SAFETY_RESUME_STATE_INVALID")
        if any(before[key] != after[key] for key in preserved):
            raise ValueError("GAIN_ORACLE_SAFETY_RESUME_MUTATED_PHYSICAL_STATE")
    else:
        if before["safety_paused"] or after["safety_paused"]:
            raise ValueError("GAIN_ORACLE_SAFETY_RELEASE_STATE_INVALID")
        if after["status"] != "pending" or after["phase"] != "released":
            raise ValueError("GAIN_ORACLE_SAFETY_RELEASE_TARGET_STATE_INVALID")
        if any(after[key] is not None for key in ("assignee", "active_platform_id", "queue_owner", "stock_commitment")):
            raise ValueError("GAIN_ORACLE_SAFETY_RELEASE_OWNERSHIP_INVALID")
        if after["cargo_onboard"] != 0.0 or after["progress"] != 0.0 or after["service_progress"] != 0.0:
            raise ValueError("GAIN_ORACLE_SAFETY_RELEASE_PHYSICAL_STATE_INVALID")
    return str(event_hash)


def _validate_live_safety_receipt(receipt: Mapping[str, Any]) -> list[str]:
    if receipt.get("schema") != "gain-live-safety-disposition/1":
        raise ValueError("GAIN_ORACLE_LIVE_SAFETY_RECEIPT_SCHEMA_INVALID")
    if receipt.get("status") not in {"not_required", "resolved", "unresolved", "unknown"}:
        raise ValueError("GAIN_ORACLE_LIVE_SAFETY_RECEIPT_STATUS_INVALID")
    if type(receipt.get("resolved")) is not bool:
        raise ValueError("GAIN_ORACLE_LIVE_SAFETY_RECEIPT_RESOLUTION_INVALID")
    for field in ("pre_public_sha256", "post_public_sha256"):
        value = receipt.get(field)
        if value is not None and (not isinstance(value, str) or len(value) != 64):
            raise ValueError("GAIN_ORACLE_LIVE_SAFETY_RECEIPT_PUBLIC_HASH_INVALID")
    actions = receipt.get("actions")
    if not isinstance(actions, list):
        raise ValueError("GAIN_ORACLE_LIVE_SAFETY_RECEIPT_ACTIONS_INVALID")
    hashes = [_validate_safety_transition(action) for action in actions]
    actions_hash = receipt.get("actions_sha256")
    if not isinstance(actions_hash, str) or actions_hash != _live_digest(actions):
        raise ValueError("GAIN_ORACLE_LIVE_SAFETY_RECEIPT_ACTION_HASH_INVALID")
    return hashes


def _validate_continuation_safety_evidence(receipt: Mapping[str, Any]) -> str | None:
    present = [field for field in SAFETY_EVIDENCE_FIELDS if field in receipt]
    digest = receipt.get("safety_evidence_sha256")
    if not present:
        if digest is not None:
            raise ValueError("GAIN_ORACLE_CONTINUATION_SAFETY_EVIDENCE_ORPHAN_HASH")
        return None
    for field in ("safety_report", "safety_disposition", "detection"):
        if field in receipt and receipt[field] is not None and not isinstance(receipt[field], Mapping):
            raise ValueError(f"GAIN_ORACLE_CONTINUATION_SAFETY_OBJECT_INVALID:{field}")
    reason = receipt.get("unknown_reason")
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("GAIN_ORACLE_CONTINUATION_SAFETY_UNKNOWN_REASON_INVALID")
    disposition = receipt.get("safety_disposition")
    if disposition is not None:
        resolved = disposition.get("resolved")
        if type(resolved) is not bool:
            raise ValueError("GAIN_ORACLE_CONTINUATION_SAFETY_RESOLUTION_INVALID")
        disposition_reason = disposition.get("unknown_reason")
        if disposition_reason is not None and (not isinstance(disposition_reason, str) or not disposition_reason.strip()):
            raise ValueError("GAIN_ORACLE_CONTINUATION_SAFETY_UNKNOWN_REASON_INVALID")
        if resolved is False and not reason:
            raise ValueError("GAIN_ORACLE_CONTINUATION_UNRESOLVED_SAFETY_REASON_MISSING")
        if disposition_reason and reason != disposition_reason:
            raise ValueError("GAIN_ORACLE_CONTINUATION_SAFETY_REASON_BINDING_MISMATCH")
    live_receipt = receipt.get("safety_apply_receipt")
    if live_receipt is not None:
        if not isinstance(live_receipt, Mapping):
            raise ValueError("GAIN_ORACLE_LIVE_SAFETY_RECEIPT_INVALID")
        _validate_live_safety_receipt(live_receipt)
        if live_receipt.get("resolved") is False and not reason:
            raise ValueError("GAIN_ORACLE_CONTINUATION_UNRESOLVED_SAFETY_REASON_MISSING")
        if live_receipt.get("resolved") is False and disposition is not None and disposition.get("resolved") is True:
            raise ValueError("GAIN_ORACLE_CONTINUATION_SAFETY_RESOLUTION_MISMATCH")
    evidence = {field: receipt[field] for field in SAFETY_EVIDENCE_FIELDS if field in receipt}
    if not isinstance(digest, str) or digest != _digest(evidence):
        raise ValueError("GAIN_ORACLE_CONTINUATION_SAFETY_EVIDENCE_HASH_MISMATCH")
    unresolved = (
        disposition is not None and disposition.get("resolved") is False
    ) or (
        isinstance(live_receipt, Mapping) and live_receipt.get("resolved") is False
    )
    return str(reason) if unresolved else None


def _validate_masked_pair_safety(pair: Mapping[str, Any]) -> str | None:
    """Recompute safety masking from persisted branch receipts, without producer code."""
    unresolved_reasons: list[str] = []
    for arm in ("keep", "replan"):
        branch = pair.get(arm)
        if not isinstance(branch, Mapping):
            continue
        trace = branch.get("trace")
        if not isinstance(trace, Mapping) or trace.get("schema") != TRACE_SCHEMA:
            continue
        _verify_digest(trace, "trace_sha256", omitted="trace_sha256")
        if trace.get("arm") != arm:
            raise ValueError("GAIN_ORACLE_MASKED_TRACE_ARM_INVALID")
        receipts = trace.get("continuation_policy_receipts", [])
        if not isinstance(receipts, list):
            raise ValueError("GAIN_ORACLE_MASKED_TRACE_RECEIPTS_INVALID")
        branch_reasons = []
        for receipt in receipts:
            if not isinstance(receipt, Mapping) or receipt.get("receipt_schema") != RECEIPT_SCHEMA:
                raise ValueError("GAIN_ORACLE_MASKED_TRACE_RECEIPT_INVALID")
            reason = _validate_continuation_safety_evidence(receipt)
            if reason is not None:
                branch_reasons.append(reason)
        recorded_reasons = trace.get("unresolved_safety_unknown_reasons")
        if recorded_reasons != branch_reasons or trace.get("unresolved_safety_disposition_count") != len(branch_reasons):
            raise ValueError("GAIN_ORACLE_MASKED_TRACE_SAFETY_SUMMARY_MISMATCH")
        if branch_reasons:
            if trace.get("valid_mask") is not False or trace.get("unknown_reason") != branch_reasons[0]:
                raise ValueError("GAIN_ORACLE_UNRESOLVED_SAFETY_TRACE_NOT_MASKED")
            unresolved_reasons.extend(branch_reasons)
    if not unresolved_reasons:
        return None
    if pair.get("valid_mask") is not False or pair.get("gain") is not None:
        raise ValueError("GAIN_ORACLE_UNRESOLVED_SAFETY_PAIR_NOT_MASKED")
    if pair.get("unknown_reason") != unresolved_reasons[0]:
        raise ValueError("GAIN_ORACLE_UNRESOLVED_SAFETY_PAIR_REASON_MISMATCH")
    if pair.get("unresolved_safety_disposition_count") != len(unresolved_reasons):
        raise ValueError("GAIN_ORACLE_PAIR_UNRESOLVED_SAFETY_SUMMARY_MISMATCH")
    if pair.get("unresolved_safety_unknown_reasons") != unresolved_reasons:
        raise ValueError("GAIN_ORACLE_PAIR_UNRESOLVED_SAFETY_SUMMARY_MISMATCH")
    return unresolved_reasons[0]


def _close_tree(expected: Any, actual: Any, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise ValueError(f"GAIN_ORACLE_COST_COMPONENT_MISMATCH:{path}:expected_keys={sorted(expected) if isinstance(expected, Mapping) else None!r}:actual_keys={sorted(actual) if isinstance(actual, Mapping) else None!r}")
        for key in expected:
            _close_tree(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        observed = _number(actual, f"GAIN_ORACLE_NONFINITE_COMPONENT:{path}", nonnegative=False)
        if not math.isclose(float(expected), observed, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"GAIN_ORACLE_COST_COMPONENT_MISMATCH:{path}:expected={float(expected)!r}:actual={observed!r}")
        return
    if expected != actual:
        raise ValueError(f"GAIN_ORACLE_COST_COMPONENT_MISMATCH:{path}")


def _config(raw: Any) -> dict[str, float]:
    if not isinstance(raw, Mapping) or set(raw) != _COST_KEYS:
        raise ValueError("GAIN_ORACLE_COST_CONFIG_INVALID")
    return {key: _number(raw[key], f"GAIN_ORACLE_COST_WEIGHT_INVALID:{key}") for key in _COST_KEYS}


def _trace_event_rows(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps = trace.get("steps")
    if not isinstance(steps, list):
        raise ValueError("GAIN_ORACLE_TRACE_STEPS_MISSING")
    events: list[Mapping[str, Any]] = []
    for step in steps:
        if not isinstance(step, Mapping) or not isinstance(step.get("raw_events"), list):
            raise ValueError("GAIN_ORACLE_STEP_RAW_EVENTS_MISSING")
        if not all(isinstance(row, Mapping) for row in step["raw_events"]):
            raise ValueError("GAIN_ORACLE_RAW_EVENT_INVALID")
        events.extend(step["raw_events"])
    return events


def _validate_trace_safety_events(trace: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> None:
    """Validate live safety transitions and their receipt/event hash binding."""
    initial = trace.get("initial_safety_transition_events", [])
    if initial is None:
        initial = []
    if not isinstance(initial, list):
        raise ValueError("GAIN_ORACLE_INITIAL_SAFETY_TRANSITIONS_INVALID")
    all_events = list(initial) + list(events)
    transition_hashes: set[str] = set()
    for event in all_events:
        if not isinstance(event, Mapping):
            continue
        if event.get("schema") == "gain-safety-task-transition/1":
            transition_hashes.add(_validate_safety_transition(event))
        elif "safety_task_" in str(event.get("type", "")):
            raise ValueError("GAIN_ORACLE_SAFETY_TRANSITION_SCHEMA_MISSING")
        if event.get("type") == "cargo_lost":
            quantity = event.get("quantity", event.get("amount", event.get("cargo_lost")))
            _number(quantity, "GAIN_ORACLE_CARGO_LOST_QUANTITY_INVALID")
            before = event.get("cargo_before")
            after = event.get("cargo_after")
            if before is not None or after is not None:
                _number(before, "GAIN_ORACLE_CARGO_LOST_BEFORE_INVALID")
                _number(after, "GAIN_ORACLE_CARGO_LOST_AFTER_INVALID")
                if float(after) > float(before) + 1e-9 or not math.isclose(
                    float(before) - float(after), float(quantity), rel_tol=1e-9, abs_tol=1e-9,
                ):
                    raise ValueError("GAIN_ORACLE_CARGO_LOST_PHYSICAL_MISMATCH")
    for receipt in trace.get("continuation_policy_receipts", []):
        if not isinstance(receipt, Mapping):
            continue
        live = receipt.get("safety_apply_receipt")
        if not isinstance(live, Mapping):
            continue
        for action in live.get("actions", []):
            event_hash = action.get("event_sha256") if isinstance(action, Mapping) else None
            if event_hash not in transition_hashes:
                raise ValueError("GAIN_ORACLE_SAFETY_RECEIPT_EVENT_HASH_UNBOUND")


def _verify_endpoint_against_events(trace: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> None:
    """Check completion, discovery, cancellation, and recoverable service state."""
    start_truth, end_truth = trace["start_truth"], trace["raw_truth"]
    start_tasks = {str(row["id"]): row for row in start_truth["tasks"] if isinstance(row, Mapping) and isinstance(row.get("id"), str)}
    end_tasks = {str(row["id"]): row for row in end_truth["tasks"] if isinstance(row, Mapping) and isinstance(row.get("id"), str)}
    start_children = {str(row["id"]): row for row in start_truth.get("subtasks", []) if isinstance(row, Mapping) and isinstance(row.get("id"), str)}
    end_children = {str(row["id"]): row for row in end_truth.get("subtasks", []) if isinstance(row, Mapping) and isinstance(row.get("id"), str)}
    completed_at: dict[str, int] = {}
    arrived_or_discovered: set[str] = set()
    cancelled: set[str] = set()
    failed: set[str] = set()
    for event in events:
        task_id = event.get("task_id")
        if not isinstance(task_id, str):
            continue
        event_type = event.get("type")
        if event_type == "logical_task_completed":
            completed_at[task_id] = int(event.get("time", -1))
        elif event_type in {"task_arrived", "task_discovered"}:
            arrived_or_discovered.add(task_id)
        elif event_type == "report_received" and event.get("kind") == "task_discovered":
            arrived_or_discovered.add(task_id)
        elif event_type == "task_cancel":
            cancelled.add(task_id)
        elif event_type == "task_failed":
            failed.add(task_id)

    for task_id, endpoint in end_tasks.items():
        start = start_tasks.get(task_id)
        if start is None:
            raise ValueError("GAIN_ORACLE_ENDPOINT_TASK_UNKNOWN_AT_START")
        was_known = bool(start.get("known"))
        is_known = bool(endpoint.get("known"))
        if was_known and not is_known:
            raise ValueError("GAIN_ORACLE_ENDPOINT_KNOWN_STATE_REGRESSED")
        if is_known and not was_known and task_id not in arrived_or_discovered:
            raise ValueError("GAIN_ORACLE_ENDPOINT_KNOWN_WITHOUT_ARRIVAL")
        if is_known and not was_known and task_id in arrived_or_discovered:
            discovery = next((event for event in events if
                              event.get("type") == "report_received"
                              and event.get("kind") == "task_discovered"
                              and event.get("task_id") == task_id), None)
            if discovery is not None:
                inspection_id = discovery.get("inspection_id")
                discovery_time = int(discovery.get("time", -1))
                parent_completion = completed_at.get(str(inspection_id))
                if (inspection_id != endpoint.get("discovered_by") or parent_completion is None
                        or parent_completion > discovery_time):
                    raise ValueError("GAIN_ORACLE_DISCOVERY_CAUSALITY_INVALID")
        was_complete = start.get("status") == "completed"
        is_complete = endpoint.get("status") == "completed"
        if is_complete and not was_complete:
            if task_id not in completed_at:
                raise ValueError("GAIN_ORACLE_ENDPOINT_COMPLETION_WITHOUT_EVENT")
            if endpoint.get("completed_at") is None or int(endpoint["completed_at"]) != completed_at[task_id]:
                raise ValueError("GAIN_ORACLE_ENDPOINT_COMPLETION_TIME_MISMATCH")
        if endpoint.get("status") == "cancelled" and not (start.get("status") == "cancelled") and task_id not in cancelled:
            raise ValueError("GAIN_ORACLE_ENDPOINT_CANCELLATION_WITHOUT_EVENT")
        if endpoint.get("status") == "failed" and start.get("status") != "failed" and task_id not in failed:
            raise ValueError("GAIN_ORACLE_ENDPOINT_FAILURE_WITHOUT_EVENT")

    contract = trace.get("task_progress_contract")
    if not isinstance(contract, Mapping) or trace.get("task_progress_contract_sha256") != _digest(contract):
        raise ValueError("GAIN_ORACLE_TASK_PROGRESS_CONTRACT_HASH_MISMATCH")
    expected_service: dict[str, float] = {}
    for task_id, row in {**start_tasks, **start_children}.items():
        expected_service[task_id] = _number(row.get("service_progress", 0.0), "GAIN_ORACLE_START_SERVICE_PROGRESS_INVALID")
    for task_id in set(end_tasks) | set(end_children):
        expected_service.setdefault(task_id, 0.0)
    for event in events:
        task_id = event.get("task_id")
        if not isinstance(task_id, str) or task_id not in expected_service:
            continue
        if event.get("type") == "loaded":
            expected_service[task_id] = 0.0
        elif event.get("type") == "service":
            expected_service[task_id] += _number(event.get("duration", 1.0), "GAIN_ORACLE_SERVICE_DURATION_INVALID")
        elif event.get("type") in {"task_failed", "task_cancel"}:
            # The simulator releases a failed/cancelled task and resets its
            # recoverable service state.  A prior service event therefore does
            # not remain reflected in the endpoint after this terminal event.
            expected_service[task_id] = 0.0

    for task_id, child in end_children.items():
        expected = expected_service.get(task_id)
        if expected is None or not math.isclose(
            _number(child.get("service_progress"), "GAIN_ORACLE_ENDPOINT_SERVICE_PROGRESS_INVALID"),
            expected, rel_tol=1e-10, abs_tol=1e-10,
        ):
            raise ValueError("GAIN_ORACLE_ENDPOINT_SERVICE_PROGRESS_MISMATCH")
        definition = contract.get(task_id)
        if not isinstance(definition, Mapping):
            raise ValueError("GAIN_ORACLE_TASK_PROGRESS_DEFINITION_MISSING")
        service_total = _number(definition.get("service_total"), "GAIN_ORACLE_TASK_SERVICE_TOTAL_INVALID")
        expected_progress = 1.0 if child.get("status") == "completed" else min(1.0, expected / service_total)
        if not math.isclose(_number(child.get("progress"), "GAIN_ORACLE_ENDPOINT_PROGRESS_INVALID"), expected_progress, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("GAIN_ORACLE_ENDPOINT_PROGRESS_MISMATCH")

    for task_id, endpoint in end_tasks.items():
        child_ids = endpoint.get("subtask_ids")
        if isinstance(child_ids, list) and child_ids:
            if not all(isinstance(child_id, str) and child_id in end_children for child_id in child_ids):
                raise ValueError("GAIN_ORACLE_SPLIT_CHILD_BINDING_MISMATCH")
            child_rows = [end_children[child_id] for child_id in child_ids]
            service_sum = sum(_number(row.get("service_progress"), "GAIN_ORACLE_ENDPOINT_SERVICE_PROGRESS_INVALID") for row in child_rows)
            if not math.isclose(_number(endpoint.get("service_progress"), "GAIN_ORACLE_PARENT_SERVICE_PROGRESS_INVALID"), service_sum, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("GAIN_ORACLE_PARENT_SERVICE_PROGRESS_MISMATCH")
            derived_progress = sum(
                _number(contract[child_id].get("workload_share"), "GAIN_ORACLE_WORKLOAD_SHARE_INVALID")
                * (1.0 if end_children[child_id].get("status") == "completed" else min(
                    1.0, _number(end_children[child_id].get("service_progress"), "GAIN_ORACLE_CHILD_SERVICE_PROGRESS_INVALID")
                    / _number(contract[child_id].get("service_total"), "GAIN_ORACLE_CHILD_SERVICE_TOTAL_INVALID"),
                ))
                for child_id in child_ids
            )
            if not math.isclose(_number(endpoint.get("progress"), "GAIN_ORACLE_PARENT_PROGRESS_INVALID"), min(1.0, derived_progress), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("GAIN_ORACLE_PARENT_PROGRESS_MISMATCH")
            all_complete = all(row.get("status") == "completed" for row in child_rows)
            if (endpoint.get("status") == "completed") != all_complete:
                raise ValueError("GAIN_ORACLE_PARENT_COMPLETION_ROLLUP_MISMATCH")
            continue
        expected = expected_service.get(task_id)
        if expected is None:
            raise ValueError("GAIN_ORACLE_ENDPOINT_SERVICE_PROGRESS_UNBOUND")
        if not math.isclose(_number(endpoint.get("service_progress"), "GAIN_ORACLE_ENDPOINT_SERVICE_PROGRESS_INVALID"), expected, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("GAIN_ORACLE_ENDPOINT_SERVICE_PROGRESS_MISMATCH")
        definition = contract.get(task_id)
        if not isinstance(definition, Mapping):
            raise ValueError("GAIN_ORACLE_TASK_PROGRESS_DEFINITION_MISSING")
        expected_progress = 1.0 if endpoint.get("status") == "completed" else min(1.0, expected / _number(definition.get("service_total"), "GAIN_ORACLE_TASK_SERVICE_TOTAL_INVALID"))
        if not math.isclose(_number(endpoint.get("progress"), "GAIN_ORACLE_ENDPOINT_PROGRESS_INVALID"), expected_progress, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("GAIN_ORACLE_ENDPOINT_PROGRESS_MISMATCH")


def _trace_cost(
    trace: Mapping[str, Any],
    *,
    task_weights: Mapping[str, Any],
    config: Mapping[str, float],
) -> dict[str, Any]:
    start_truth, end_truth = trace.get("start_truth"), trace.get("raw_truth")
    if not isinstance(start_truth, Mapping) or not isinstance(end_truth, Mapping):
        raise ValueError("GAIN_ORACLE_TRUTH_BINDING_MISSING")
    if not isinstance(start_truth.get("tasks"), list) or not isinstance(end_truth.get("tasks"), list):
        raise ValueError("GAIN_ORACLE_TASK_TRUTH_INVALID")
    weights = {str(key): _number(value, "GAIN_ORACLE_TASK_WEIGHT_INVALID") for key, value in task_weights.items()}
    if not weights or any(value <= 0 for value in weights.values()):
        raise ValueError("GAIN_ORACLE_TASK_WEIGHTS_EMPTY")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("GAIN_ORACLE_TASK_WEIGHTS_NOT_NORMALIZED")
    start_ids = {str(row.get("id")) for row in start_truth["tasks"] if isinstance(row, Mapping)}
    end_by_id = {str(row.get("id")): row for row in end_truth["tasks"] if isinstance(row, Mapping)}
    if not set(weights) <= start_ids or not set(weights) <= set(end_by_id):
        raise ValueError("GAIN_ORACLE_TASK_UNIVERSE_BINDING_MISMATCH")

    complete = 0.0
    residual = 0.0
    for task_id, weight in weights.items():
        task = end_by_id[task_id]
        if task.get("status") == "completed":
            complete += weight
            continue
        progress = _number(task.get("progress"), "GAIN_ORACLE_TASK_PROGRESS_INVALID", nonnegative=False)
        residual += weight * max(0.0, min(1.0, 1.0 - progress))

    events = _trace_event_rows(trace)
    _validate_trace_safety_events(trace, events)
    _verify_endpoint_against_events(trace, events)
    delay = sum(
        _number(row.get("minutes", 0.0), "GAIN_ORACLE_DELAY_EVENT_INVALID")
        for row in events if row.get("type") == "delay_accrued"
    )
    energy = sum(
        _number(row.get("energy", 0.0), "GAIN_ORACLE_ENERGY_EVENT_INVALID")
        for row in events if row.get("type") in {"move", "service"}
    )
    risk = sum(
        _number(row.get("environmental_risk", 0.0), "GAIN_ORACLE_RISK_EVENT_INVALID")
        for row in events if row.get("type") == "move"
    ) + sum(row.get("type") in {"task_failed", "platform_failure"} for row in events)
    messages = sum(
        int(_number(row.get("message_count", 0), "GAIN_ORACLE_MESSAGE_EVENT_INVALID"))
        for row in events if row.get("type") == "communication_tx"
    )
    switches = len({
        (int(row.get("time", -1)), str(row.get("task_id", "")))
        for row in events if row.get("type") == "task_switch"
    })
    size = 0
    calls = int(trace.get("initial_planner_call_count", 0))
    for step in trace["steps"]:
        rows = step["raw_events"]
        owners = {str(row.get("task_id")) for row in rows if row.get("type") == "task_owner_change"}
        size += len(owners) + sum(row.get("type") == "queue_reorder" for row in rows)
        if step.get("stage") != "replan_apply":
            calls += int(bool(step.get("planner_invoked", False)))

    initial_cpu = _number(trace.get("initial_planner_cpu_seconds", 0.0), "GAIN_ORACLE_INITIAL_CPU_INVALID")
    continuation_receipts = trace.get("continuation_policy_receipts", [])
    if not isinstance(continuation_receipts, list):
        raise ValueError("GAIN_ORACLE_CONTINUATION_RECEIPTS_INVALID")
    receipt_planner = 0.0
    receipt_safety = 0.0
    planner_receipts: list[Mapping[str, Any]] = []
    unresolved_safety_reasons: list[str] = []
    for receipt in continuation_receipts:
        if not isinstance(receipt, Mapping) or receipt.get("receipt_schema") != RECEIPT_SCHEMA:
            raise ValueError("GAIN_ORACLE_CONTINUATION_RECEIPT_INVALID")
        action = receipt.get("action")
        if not isinstance(action, Mapping) or receipt.get("action_sha256") != _digest(action):
            raise ValueError("GAIN_ORACLE_CONTINUATION_ACTION_HASH_MISMATCH")
        invoked = receipt.get("planner_invoked")
        if type(invoked) is not bool:
            raise ValueError("GAIN_ORACLE_CONTINUATION_CALL_FLAG_INVALID")
        planner_cpu = _number(receipt.get("planner_cpu_seconds", 0.0), "GAIN_ORACLE_CONTINUATION_PLANNER_CPU_INVALID")
        safety_cpu = _number(receipt.get("safety_monitor_cpu_seconds", 0.0), "GAIN_ORACLE_CONTINUATION_SAFETY_CPU_INVALID")
        measured_cpu = _number(receipt.get("measured_callback_cpu_seconds", 0.0), "GAIN_ORACLE_CONTINUATION_MEASURED_CPU_INVALID")
        if planner_cpu + safety_cpu > measured_cpu + 1e-6:
            raise ValueError("GAIN_ORACLE_CONTINUATION_CPU_SPLIT_INVALID")
        if not invoked and planner_cpu > 0:
            raise ValueError("GAIN_ORACLE_CONTINUATION_CPU_WITHOUT_CALL")
        call_type = "continuation_safety_replan" if invoked else "safety_monitor"
        if receipt.get("call_type") != call_type:
            raise ValueError("GAIN_ORACLE_CONTINUATION_CALL_TYPE_INVALID")
        unresolved_reason = _validate_continuation_safety_evidence(receipt)
        if unresolved_reason is not None:
            unresolved_safety_reasons.append(unresolved_reason)
        receipt_planner += planner_cpu
        receipt_safety += safety_cpu
        if invoked:
            planner_receipts.append(receipt)
    expected_trace_planner = _number(trace.get("continuation_planner_cpu_seconds", 0.0), "GAIN_ORACLE_TRACE_PLANNER_CPU_INVALID")
    expected_trace_safety = _number(trace.get("continuation_policy_cpu_seconds", 0.0), "GAIN_ORACLE_TRACE_SAFETY_CPU_INVALID")
    if not math.isclose(receipt_planner, expected_trace_planner, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("GAIN_ORACLE_CONTINUATION_PLANNER_CPU_BINDING_MISMATCH")
    if not math.isclose(receipt_safety, expected_trace_safety, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("GAIN_ORACLE_CONTINUATION_SAFETY_CPU_BINDING_MISMATCH")
    recorded_planner_calls = trace.get("planner_call_receipts", [])
    if not isinstance(recorded_planner_calls, list) or list(recorded_planner_calls) != list(planner_receipts):
        raise ValueError("GAIN_ORACLE_PLANNER_CALL_RECEIPT_BINDING_MISMATCH")
    expected_unresolved_count = len(unresolved_safety_reasons)
    expected_unresolved_reasons = list(unresolved_safety_reasons)
    trace_unresolved_count = trace.get("unresolved_safety_disposition_count")
    trace_unresolved_reasons = trace.get("unresolved_safety_unknown_reasons")
    if trace_unresolved_count is None:
        if expected_unresolved_count:
            raise ValueError("GAIN_ORACLE_TRACE_UNRESOLVED_SAFETY_COUNT_MISSING")
    elif type(trace_unresolved_count) is not int or trace_unresolved_count != expected_unresolved_count:
        raise ValueError("GAIN_ORACLE_TRACE_UNRESOLVED_SAFETY_COUNT_MISMATCH")
    if trace_unresolved_reasons is None:
        if expected_unresolved_reasons:
            raise ValueError("GAIN_ORACLE_TRACE_UNRESOLVED_SAFETY_REASONS_MISSING")
    elif trace_unresolved_reasons != expected_unresolved_reasons:
        raise ValueError("GAIN_ORACLE_TRACE_UNRESOLVED_SAFETY_REASONS_MISMATCH")
    for step in trace["steps"]:
        receipt = step.get("continuation_receipt")
        if receipt is not None and receipt not in continuation_receipts:
            raise ValueError("GAIN_ORACLE_STEP_RECEIPT_BINDING_MISMATCH")
        if receipt is not None and step.get("planner_invoked") is not receipt.get("planner_invoked"):
            raise ValueError("GAIN_ORACLE_STEP_CALL_FLAG_MISMATCH")

    completion_cost = config["completion"] * residual
    delay_cost = config["delay"] * delay
    energy_cost = config["energy"] * energy
    risk_cost = config["risk"] * risk
    execution_cost = completion_cost + delay_cost + energy_cost + risk_cost
    communication_cost = config["communication"] * messages
    switch_cost = config["switch"] * switches
    reconstruction_cost = config["reconstruction"] * calls
    reconstruction_size_cost = config["reconstruction_size"] * size
    planner_cpu_cost = config["cpu_second"] * (initial_cpu + receipt_planner)
    safety_cost = config["cpu_second"] * receipt_safety
    return {
        "J_exec": execution_cost,
        "C_replan": communication_cost + switch_cost + reconstruction_cost + reconstruction_size_cost + planner_cpu_cost,
        "C_safety": safety_cost,
        "J_total": execution_cost + communication_cost + switch_cost + reconstruction_cost + reconstruction_size_cost + planner_cpu_cost + safety_cost,
        "execution_components": {
            "weighted_completion": complete,
            "weighted_remaining_fraction": residual,
            "completion_cost": completion_cost,
            "delay_cost": delay_cost,
            "energy_cost": energy_cost,
            "risk_cost": risk_cost,
            "delay_minutes": delay,
            "energy_units": energy,
            "risk_units": risk,
        },
        "replan_components": {
            "communication_messages": float(messages),
            "switch_count": float(switches),
            "call_count": float(calls),
            "reconstruction_size": float(size),
            "continuation_planner_cpu_seconds": receipt_planner,
            "planner_cpu_seconds": initial_cpu,
            "communication_cost": communication_cost,
            "switch_cost": switch_cost,
            "reconstruction_cost": reconstruction_cost,
            "reconstruction_size_cost": reconstruction_size_cost,
            "planner_cpu_cost": planner_cpu_cost,
        },
        "safety_components": {
            "continuation_policy_cpu_seconds": receipt_safety,
            "continuation_policy_cpu_cost": safety_cost,
        },
        "cost_config": dict(config),
        "cost_accounting_schema": "gain-cost-accounting/2",
    }


def _validate_trace(trace: Any, *, pair: Mapping[str, Any], arm: str, cost_config: Mapping[str, float]) -> dict[str, Any]:
    if not isinstance(trace, Mapping) or trace.get("schema") != TRACE_SCHEMA:
        raise ValueError("GAIN_ORACLE_TRACE_SCHEMA_INVALID")
    _verify_digest(trace, "trace_sha256", omitted="trace_sha256")
    if trace.get("arm") != arm or trace.get("valid_mask") is not True:
        raise ValueError(f"GAIN_ORACLE_{arm.upper()}_TRACE_INVALID")
    bindings = {
        "state_id": "state_id", "scenario_id": "scenario_id", "parent_family": "parent_family",
        "split": "split", "event_type": "event_type", "forced": "forced",
        "replanner_version": "replanner_version", "source_snapshot_sha256": "source_snapshot_sha256",
        "public_history_sha256": "public_history_sha256", "event_tape_id": "event_tape_id",
        "event_tape_sha256": "event_tape_sha256", "suffix_seed": "suffix_seed",
    }
    for pair_key, trace_key in bindings.items():
        if pair.get(pair_key) != trace.get(trace_key):
            raise ValueError(f"GAIN_ORACLE_TRACE_BINDING_MISMATCH:{pair_key}")
    if trace.get("initial_planner_call_count") != int(arm == "replan"):
        raise ValueError("GAIN_ORACLE_INITIAL_CALL_COUNT_MISMATCH")
    if arm == "keep":
        if trace.get("planned_action") != NOOP_ACTION:
            raise ValueError("GAIN_ORACLE_KEEP_ACTION_INVALID")
    elif not isinstance(trace.get("planned_action"), Mapping):
        raise ValueError("GAIN_ORACLE_REPLAN_ACTION_INVALID")
    if trace.get("planned_action_sha256") != _digest(trace["planned_action"]):
        raise ValueError("GAIN_ORACLE_PLANNED_ACTION_HASH_MISMATCH")
    if not isinstance(trace.get("start_truth"), Mapping) or not isinstance(trace.get("raw_truth"), Mapping):
        raise ValueError("GAIN_ORACLE_TRACE_TRUTH_MISSING")
    return _trace_cost(trace, task_weights=pair["task_universe"]["priority_weights"], config=cost_config)


def _recompute_obligation_weights(pair: Mapping[str, Any], keep: Mapping[str, Any], replan: Mapping[str, Any]) -> tuple[list[str], dict[str, float], list[str]]:
    """Rebuild the shared obligation set directly from paired raw truth/events."""
    start_truth = keep.get("start_truth")
    if not isinstance(start_truth, Mapping) or start_truth != replan.get("start_truth"):
        raise ValueError("GAIN_ORACLE_OBLIGATION_START_TRUTH_MISMATCH")
    rows = start_truth.get("tasks")
    if not isinstance(rows, list):
        raise ValueError("GAIN_ORACLE_OBLIGATION_TASKS_MISSING")
    definitions = {str(row.get("id")): row for row in rows if isinstance(row, Mapping) and isinstance(row.get("id"), str)}
    start_time = _number(keep.get("start_time"), "GAIN_ORACLE_OBLIGATION_START_TIME_INVALID")
    end_time = start_time + _number(pair.get("horizon_actual_minutes"), "GAIN_ORACLE_OBLIGATION_HORIZON_INVALID")
    included = {
        task_id for task_id, row in definitions.items()
        if row.get("known") is True and row.get("status") not in {"completed", "cancelled"}
    }
    traces = (keep, replan)
    event_rows = [_trace_event_rows(trace) for trace in traces]
    arrivals: dict[str, int] = {}
    cancellations: dict[str, int] = {}
    for events in event_rows:
        for event in events:
            task_id = event.get("task_id")
            event_time = event.get("time")
            if not isinstance(task_id, str) or isinstance(event_time, bool) or not isinstance(event_time, (int, float)):
                continue
            time_value = int(event_time)
            if not start_time < time_value <= end_time:
                continue
            if event.get("type") == "task_arrived":
                arrivals[task_id] = min(time_value, arrivals.get(task_id, time_value))
            elif event.get("type") == "task_cancel":
                cancellations[task_id] = min(time_value, cancellations.get(task_id, time_value))
    included.update(arrivals)
    completed_before_cancel = set()
    for task_id, when in cancellations.items():
        if any(
            any(row.get("id") == task_id and row.get("completed_at") is not None and int(row["completed_at"]) <= when
                for row in trace.get("raw_truth", {}).get("tasks", []) if isinstance(row, Mapping))
            for trace in traces
        ):
            completed_before_cancel.add(task_id)
    cancelled_obligations = set(cancellations) - completed_before_cancel
    included.difference_update(cancelled_obligations)
    children: dict[str, list[str]] = {}
    for task_id, row in definitions.items():
        parent_id = row.get("discovered_by")
        if isinstance(parent_id, str):
            children.setdefault(parent_id, []).append(task_id)
    dependency_added: set[str] = set()
    frontier = list(included)
    while frontier:
        parent_id = frontier.pop()
        for child_id in children.get(parent_id, []):
            if child_id in included or definitions[child_id].get("status") in {"completed", "cancelled"}:
                continue
            included.add(child_id)
            dependency_added.add(child_id)
            frontier.append(child_id)
    included.difference_update(cancelled_obligations)
    included = {task_id for task_id in included if task_id in definitions}
    if not included:
        raise ValueError("GAIN_ORACLE_EMPTY_OBLIGATION_UNIVERSE")
    priorities: dict[str, float] = {}
    for task_id in sorted(included):
        priority = definitions[task_id].get("priority")
        if task_id in arrivals:
            found = None
            for trace in traces:
                for step in trace.get("steps", []):
                    observation = step.get("next_public_observation", {}) if isinstance(step, Mapping) else {}
                    if not isinstance(observation, Mapping) or int(observation.get("time", -1)) != arrivals[task_id]:
                        continue
                    public_task = next((row for row in observation.get("tasks", []) if isinstance(row, Mapping) and row.get("id") == task_id), None)
                    if public_task is not None:
                        found = public_task.get("priority")
                        break
                if found is not None:
                    break
            if found is not None:
                priority = found
        value = _number(priority, "GAIN_ORACLE_OBLIGATION_PRIORITY_INVALID")
        if value <= 0:
            raise ValueError("GAIN_ORACLE_OBLIGATION_PRIORITY_NOT_POSITIVE")
        priorities[task_id] = value
    total = sum(priorities.values())
    normalized = {task_id: value / total for task_id, value in priorities.items()}
    return sorted(included), normalized, sorted(dependency_added & included)


def recompute_gain_label(label: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute paired gains from persisted raw trace evidence."""
    if not isinstance(label, Mapping) or label.get("schema") != LABEL_SCHEMA:
        raise ValueError("GAIN_ORACLE_LABEL_SCHEMA_INVALID")
    pairs = label.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("GAIN_ORACLE_PAIRS_MISSING")
    recomputed: list[dict[str, Any]] = []
    all_valid = bool(pairs)
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping) or pair.get("schema") != PAIR_SCHEMA:
            raise ValueError("GAIN_ORACLE_PAIR_SCHEMA_INVALID")
        _verify_digest(pair, "pair_sha256", omitted="pair_sha256")
        if pair.get("valid_mask") is not True:
            all_valid = False
            safety_reason = _validate_masked_pair_safety(pair)
            if safety_reason is not None:
                recomputed.append({"index": index, "valid_mask": False, "unknown_reason": safety_reason})
                continue
            recomputed.append({"index": index, "valid_mask": False, "unknown_reason": pair.get("unknown_reason")})
            continue
        task_universe = pair.get("task_universe")
        if not isinstance(task_universe, Mapping) or not isinstance(task_universe.get("priority_weights"), Mapping):
            raise ValueError("GAIN_ORACLE_TASK_UNIVERSE_MISSING")
        keep = pair.get("keep")
        replan = pair.get("replan")
        if not isinstance(keep, Mapping) or not isinstance(replan, Mapping):
            raise ValueError("GAIN_ORACLE_BRANCH_EVIDENCE_MISSING")
        keep_config = _config(keep.get("cost", {}).get("cost_config") if isinstance(keep.get("cost"), Mapping) else None)
        replan_config = _config(replan.get("cost", {}).get("cost_config") if isinstance(replan.get("cost"), Mapping) else None)
        if keep_config != replan_config:
            raise ValueError("GAIN_ORACLE_BRANCH_COST_CONFIG_MISMATCH")
        keep_trace, replan_trace = keep.get("trace"), replan.get("trace")
        expected_ids, expected_weights, expected_dependencies = _recompute_obligation_weights(pair, keep_trace, replan_trace)
        recorded_weights = task_universe["priority_weights"]
        if (
            task_universe.get("task_ids") != expected_ids
            or task_universe.get("dependency_task_ids") != expected_dependencies
            or set(recorded_weights) != set(expected_weights)
            or any(not math.isclose(
                _number(recorded_weights[task_id], "GAIN_ORACLE_OBLIGATION_WEIGHT_INVALID"),
                expected_weights[task_id], rel_tol=1e-10, abs_tol=1e-10,
            ) for task_id in expected_weights)
        ):
            raise ValueError("GAIN_ORACLE_TASK_UNIVERSE_RECOMPUTATION_MISMATCH")
        keep_cost = _validate_trace(keep_trace, pair=pair, arm="keep", cost_config=keep_config)
        replan_cost = _validate_trace(replan_trace, pair=pair, arm="replan", cost_config=replan_config)
        unresolved_reasons = [
            reason
            for trace in (keep_trace, replan_trace)
            for reason in trace.get("unresolved_safety_unknown_reasons", [])
        ]
        recorded_unresolved_count = pair.get("unresolved_safety_disposition_count")
        recorded_unresolved_reasons = pair.get("unresolved_safety_unknown_reasons")
        if recorded_unresolved_count is None and recorded_unresolved_reasons is None:
            if unresolved_reasons:
                raise ValueError("GAIN_ORACLE_PAIR_UNRESOLVED_SAFETY_SUMMARY_MISSING")
        elif (
            type(recorded_unresolved_count) is not int
            or recorded_unresolved_count != len(unresolved_reasons)
            or recorded_unresolved_reasons != unresolved_reasons
        ):
            raise ValueError("GAIN_ORACLE_PAIR_UNRESOLVED_SAFETY_SUMMARY_MISMATCH")
        if keep.get("action") != keep_trace.get("planned_action") or replan.get("action") != replan_trace.get("planned_action"):
            raise ValueError("GAIN_ORACLE_BRANCH_ACTION_BINDING_MISMATCH")
        if keep_trace.get("start_truth") != replan_trace.get("start_truth"):
            raise ValueError("GAIN_ORACLE_PAIRED_START_TRUTH_MISMATCH")
        _close_tree(keep_cost, keep.get("cost"), "keep.cost")
        _close_tree(replan_cost, replan.get("cost"), "replan.cost")
        if not math.isclose(
            _number(keep_trace.get("initial_planner_cpu_seconds"), "GAIN_ORACLE_KEEP_CPU_INVALID"), 0.0,
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("GAIN_ORACLE_KEEP_HAS_PLANNER_CPU")
        if not math.isclose(
            _number(replan_trace.get("initial_planner_cpu_seconds"), "GAIN_ORACLE_REPLAN_CPU_INVALID"),
            _number(replan_cost["replan_components"]["planner_cpu_seconds"], "GAIN_ORACLE_REPLAN_CPU_INVALID"),
            rel_tol=1e-10, abs_tol=1e-10,
        ):
            raise ValueError("GAIN_ORACLE_INITIAL_CPU_BINDING_MISMATCH")
        gain = keep_cost["J_total"] - replan_cost["J_total"]
        if not math.isclose(_number(pair.get("gain"), "GAIN_ORACLE_RECORDED_GAIN_INVALID", nonnegative=False), gain, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("GAIN_ORACLE_RECORDED_GAIN_MISMATCH")
        recomputed.append({
            "index": index, "valid_mask": True, "gain": gain,
            "keep_J_total": keep_cost["J_total"], "replan_J_total": replan_cost["J_total"],
        })
    gains = [row["gain"] for row in recomputed if row.get("valid_mask") is True]
    if all_valid:
        mean = statistics.mean(gains)
        variance = statistics.variance(gains) if len(gains) > 1 else None
        return {
            "valid_mask": True, "pair_count": len(gains), "gain_mean": mean,
            "paired_variance": variance, "pairs": recomputed,
        }
    return {
        "valid_mask": False, "pair_count": len(pairs), "gain_mean": None,
        "paired_variance": None, "pairs": recomputed,
        "unknown_reason": next(
            (row.get("unknown_reason") for row in recomputed if not row.get("valid_mask")),
            label.get("unknown_reason") if not pairs else "no_valid_pairs",
        ),
    }


def validate_gain_label(label: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the label hash and every nested trace binding and reported cost."""
    if not isinstance(label, Mapping) or label.get("schema") != LABEL_SCHEMA:
        raise ValueError("GAIN_ORACLE_LABEL_SCHEMA_INVALID")
    _verify_digest(label, "label_sha256", omitted="label_sha256")
    if type(label.get("valid_mask")) is not bool or type(label.get("forced")) is not bool:
        raise ValueError("GAIN_ORACLE_LABEL_FLAGS_INVALID")
    pairs = label.get("pairs")
    if not isinstance(pairs, list) or label.get("suffix_count") != len(pairs):
        raise ValueError("GAIN_ORACLE_LABEL_PAIR_COUNT_MISMATCH")
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("GAIN_ORACLE_PAIR_INVALID")
        for key in ("state_id", "scenario_id", "parent_family", "split", "time", "event_type", "forced", "replanner_version", "source_snapshot_sha256", "private_snapshot_sha256", "public_history_sha256"):
            if pair.get(key) != label.get(key):
                raise ValueError(f"GAIN_ORACLE_LABEL_PAIR_BINDING_MISMATCH:{key}")
    result = recompute_gain_label(label)
    if result["valid_mask"] != label["valid_mask"]:
        raise ValueError("GAIN_ORACLE_LABEL_VALIDITY_MISMATCH")
    if result["valid_mask"]:
        if label.get("forced") is not False:
            raise ValueError("GAIN_ORACLE_FORCED_LABEL_MUST_BE_INVALID")
        if not math.isclose(_number(label.get("gain_mean"), "GAIN_ORACLE_RECORDED_MEAN_INVALID", nonnegative=False), result["gain_mean"], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("GAIN_ORACLE_RECORDED_MEAN_MISMATCH")
        expected_variance = result["paired_variance"]
        actual_variance = label.get("paired_variance")
        if expected_variance is None:
            if actual_variance is not None or label.get("variance_valid_mask") is not False:
                raise ValueError("GAIN_ORACLE_VARIANCE_MASK_MISMATCH")
        elif not math.isclose(_number(actual_variance, "GAIN_ORACLE_RECORDED_VARIANCE_INVALID"), expected_variance, rel_tol=1e-10, abs_tol=1e-10) or label.get("variance_valid_mask") is not True:
            raise ValueError("GAIN_ORACLE_RECORDED_VARIANCE_MISMATCH")
    else:
        if label.get("gain_mean") is not None or label.get("paired_variance") is not None:
            raise ValueError("GAIN_ORACLE_INVALID_LABEL_MUST_NOT_HAVE_GAIN")
        if label.get("unknown_reason") != result.get("unknown_reason"):
            raise ValueError("GAIN_ORACLE_INVALID_LABEL_REASON_MISMATCH")
    return {
        "schema": "gain-oracle-validation/1",
        "valid": True,
        "label_valid_mask": result["valid_mask"],
        "pair_count": result["pair_count"],
        "recomputed_gain_mean": result["gain_mean"],
        "recomputed_paired_variance": result["paired_variance"],
        "label_sha256": label["label_sha256"],
    }
