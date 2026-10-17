"""Public-prefix snapshots and paired consequence records for N2 collection.

The collector owns hidden simulator state only for restoring siblings and
producing labels.  Policy inputs and candidate action batches are copied from
the public observation and are separately identity-bound.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from baselines import choose_step_c1_actions
from relief_env import ReliefEnv
from snapshot_contract import SnapshotEnvelope, _canonical, _digest, capture as capture_public_snapshot, restore as restore_public_snapshot
from stable_identity import key_digest


SCHEMA = "n2-civilian-paired-branch/1"
ENGINEERING_WINDOW_STARTS = (8, 16, 24, 32)
MAX_CONTINUATION_MINUTES = 30
GLOBAL_TARGET_NAMES = (
    "other_order_lateness",
    "unmet_demand",
    "energy",
    "reassignment",
    "global_utility_delta",
)
GLOBAL_TARGET_UNITS = ("count", "count", "energy_units", "count", "reward")
TASK_TARGET_NAMES = ("delivery_on_time", "inspection_on_time", "report_on_time")
TASK_TARGET_UNITS = ("indicator", "indicator", "indicator")


def _implementation_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = ("branching.py", "snapshot_contract.py", "relief_env.py", "baselines.py", "stable_identity.py")
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in names
        if (root / name).is_file()
    }


def _implementation_sha256() -> str:
    return hashlib.sha256(_canonical(_implementation_identity())).hexdigest()


def _require_identity(parent_id: str, repeat: int, window_id: str, event_tape_id: str) -> None:
    if not isinstance(parent_id, str) or not parent_id:
        raise ValueError("BRANCH_PARENT_ID")
    if type(repeat) is not int or repeat < 0:
        raise ValueError("BRANCH_REPEAT")
    if not isinstance(window_id, str) or not window_id:
        raise ValueError("BRANCH_WINDOW_ID")
    if not isinstance(event_tape_id, str) or not event_tape_id:
        raise ValueError("BRANCH_EVENT_TAPE_ID")


def _validate_action_batch(candidate_catalog: Sequence[dict[str, Any]], action_batch: Sequence[str]) -> list[str]:
    if not isinstance(action_batch, (list, tuple)) or any(type(item) is not str for item in action_batch):
        raise ValueError("BRANCH_ACTION_BATCH_TYPE")
    actions = list(action_batch)
    catalog_ids = {str(item["id"]) for item in candidate_catalog}
    if not actions or any(item not in catalog_ids for item in actions):
        raise ValueError("BRANCH_ACTION_BATCH_CATALOG")
    if len(actions) != len(set(actions)):
        raise ValueError("BRANCH_ACTION_BATCH_DUPLICATE")
    if "NOOP" in actions and actions[-1] != "NOOP":
        raise ValueError("BRANCH_ACTION_BATCH_NOOP_ORDER")
    return actions


def capture_snapshot(
    env: ReliefEnv,
    *,
    parent_id: str,
    repeat: int,
    window_id: str,
    event_tape_id: str,
) -> str:
    """Capture only C1-produced pre-action state at a registered window."""
    _require_identity(parent_id, repeat, window_id, event_tape_id)
    if int(getattr(env, "time", -1)) not in ENGINEERING_WINDOW_STARTS:
        raise ValueError("BRANCH_WINDOW_TIME")
    if getattr(env, "_policy", None) != "C1":
        raise ValueError("BRANCH_REQUIRES_C1_PREFIX")
    envelope = capture_public_snapshot(
        env,
        parent_id=parent_id,
        repeat=repeat,
        window_id=window_id,
        event_tape_id=event_tape_id,
    )
    payload = dict(envelope.payload)
    # Keep the snapshot-contract digest over the base envelope.  Branch
    # metadata is an outer layer and must have its own digest; replacing the
    # base digest makes restore_public_snapshot reject every branch.
    base_envelope_sha256 = payload["envelope_sha256"]
    payload.update({
        "branch_schema": SCHEMA,
        "implementation_sources": _implementation_identity(),
        "implementation_sha256": _implementation_sha256(),
        "source_state_sha256": payload["state_sha256"],
        "source_prefix_sha256": _digest(getattr(env, "transitions", [])),
        "event_tape_id": event_tape_id,
        "envelope_sha256": base_envelope_sha256,
    })
    payload["branch_envelope_sha256"] = _digest({key: value for key, value in payload.items() if key != "branch_envelope_sha256"})
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def restore_snapshot(snapshot_json: str, env_factory: Callable[[dict[str, Any]], ReliefEnv] | None = None) -> ReliefEnv:
    try:
        payload = json.loads(snapshot_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("BRANCH_SNAPSHOT_JSON") from exc
    if payload.get("branch_schema") != SCHEMA or payload.get("implementation_sources") != _implementation_identity():
        raise ValueError("BRANCH_IMPLEMENTATION_IDENTITY")
    if payload.get("implementation_sha256") != _implementation_sha256():
        raise ValueError("BRANCH_IMPLEMENTATION_DIGEST")
    if payload.get("branch_envelope_sha256") != _digest({key: value for key, value in payload.items() if key != "branch_envelope_sha256"}):
        raise ValueError("BRANCH_ENVELOPE_DIGEST")
    envelope_payload = {key: value for key, value in payload.items() if key not in {
        "branch_schema", "implementation_sources", "implementation_sha256", "source_state_sha256", "source_prefix_sha256",
        "branch_envelope_sha256",
    }}
    envelope = SnapshotEnvelope(envelope_payload)
    factory = env_factory or ReliefEnv
    env = restore_public_snapshot(envelope, factory)
    if payload.get("source_state_sha256") != payload.get("state_sha256"):
        raise ValueError("BRANCH_STATE_IDENTITY")
    if int(getattr(env, "time", -1)) != int(payload.get("time", -2)):
        raise ValueError("BRANCH_TIME_IDENTITY")
    env._branch_source_state_sha256 = str(payload["source_state_sha256"])
    env._branch_source_prefix_sha256 = str(payload["source_prefix_sha256"])
    env._branch_event_tape_id = str(payload["event_tape_id"])
    env._branch_window_start = int(payload["time"])
    env._branch_prefix_length = len(getattr(env, "transitions", []))
    return env


def _task_outcomes(env: ReliefEnv, *, window_start: int, exclude_lateness_ids: set[str] | None = None) -> tuple[int, int, int, bool, list[str]]:
    tasks = list(getattr(env, "_tasks", {}).values())
    excluded = exclude_lateness_ids or set()
    overdue = 0
    unmet = 0
    reassigned = 0
    lateness_valid = True
    unresolved_lateness: list[str] = []
    now = int(getattr(env, "time", 0))
    terminal = bool(getattr(env, "_terminal", False)) or now >= int(env.scenario["horizon"])
    for task in tasks:
        task_id = str(task.get("id", ""))
        is_delivery = task.get("kind") == "delivery"
        deadline = task.get("deadline")
        deadline_passed = deadline is not None and now >= int(deadline)
        resolved = task.get("on_time") is not None or task.get("status") in {"completed", "failed", "expired"}
        other_new_order = is_delivery and task_id not in excluded and deadline is not None and int(deadline) > window_start
        if other_new_order and not (resolved or task.get("deadline_missed") or deadline_passed or terminal):
            # A branch ending before a deadline cannot supply a valid late or
            # lateness label for that task.
            lateness_valid = False
            unresolved_lateness.append(task_id)
        late = other_new_order and (task.get("on_time") is False or bool(task.get("deadline_missed")))
        if late and task_id not in excluded:
            overdue += 1
        if is_delivery and not bool(task.get("delivered")):
            unmet += 1
        if bool(task.get("_reassigned")):
            reassigned += 1
    return overdue, unmet, reassigned, lateness_valid, unresolved_lateness


def _task_ids_from_actions(env: ReliefEnv, actions: Sequence[str], candidate_catalog: Sequence[dict[str, Any]] | None = None) -> set[str]:
    catalog = candidate_catalog if candidate_catalog is not None else env.candidates()
    by_id = {str(item.get("id")): item for item in catalog}
    return {
        str(by_id[action]["task_id"])
        for action in actions
        if action != "NOOP" and action in by_id and by_id[action].get("task_id") is not None
    }


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
        raise ValueError("BRANCH_TASK_TARGET_NAME")
    if not applicable:
        return 0.0, False, "not_applicable"
    if not resolved:
        return 0.0, False, "unresolved_before_horizon"
    return value, True, None


def paired_task_targets(
    candidate_env: ReliefEnv,
    reference_env: ReliefEnv,
    task_ids: Sequence[str],
) -> dict[str, Any]:
    """Return named per-public-task candidate-minus-reference indicators."""
    candidate_by_id = {str(item.get("id")): item for item in getattr(candidate_env, "_tasks", {}).values()}
    reference_by_id = {str(item.get("id")): item for item in getattr(reference_env, "_tasks", {}).values()}
    values: list[list[float]] = []
    candidate_values: list[list[float]] = []
    reference_values: list[list[float]] = []
    valid: list[list[bool]] = []
    unknown_reasons: list[list[str | None]] = []
    for task_id in task_ids:
        candidate = candidate_by_id.get(str(task_id))
        reference = reference_by_id.get(str(task_id))
        if candidate is None or reference is None:
            values.append([0.0] * len(TASK_TARGET_NAMES))
            candidate_values.append([0.0] * len(TASK_TARGET_NAMES))
            reference_values.append([0.0] * len(TASK_TARGET_NAMES))
            valid.append([False] * len(TASK_TARGET_NAMES))
            unknown_reasons.append(["task_missing"] * len(TASK_TARGET_NAMES))
            continue
        row_values: list[float] = []
        row_candidate_values: list[float] = []
        row_reference_values: list[float] = []
        row_valid: list[bool] = []
        row_reasons: list[str | None] = []
        for name in TASK_TARGET_NAMES:
            cv, cok, creason = _task_indicator(candidate, name, int(getattr(candidate_env, "time", 0)), int(candidate_env.scenario["horizon"]))
            rv, rok, rreason = _task_indicator(reference, name, int(getattr(reference_env, "time", 0)), int(reference_env.scenario["horizon"]))
            row_values.append(float(cv - rv))
            row_candidate_values.append(float(cv))
            row_reference_values.append(float(rv))
            row_valid.append(bool(cok and rok))
            row_reasons.append(None if cok and rok else ";".join(item for item in (creason, rreason) if item))
        values.append(row_values)
        candidate_values.append(row_candidate_values)
        reference_values.append(row_reference_values)
        valid.append(row_valid)
        unknown_reasons.append(row_reasons)
    return {
        "task_ids": [str(item) for item in task_ids],
        "names": list(TASK_TARGET_NAMES),
        "units": list(TASK_TARGET_UNITS),
        "values": values,
        "candidate_values": candidate_values,
        "reference_values": reference_values,
        "valid": valid,
        "unknown_reasons": unknown_reasons,
    }


def paired_global_targets(candidate_env: ReliefEnv, reference_env: ReliefEnv, candidate_return: float, reference_return: float, candidate_events: list[dict], reference_events: list[dict], *, window_start: int, exclude_task_ids: set[str] | None = None) -> dict[str, Any]:
    candidate_overdue, candidate_unmet, candidate_reassigned, candidate_valid, candidate_unknown = _task_outcomes(candidate_env, window_start=window_start, exclude_lateness_ids=exclude_task_ids)
    reference_overdue, reference_unmet, reference_reassigned, reference_valid, reference_unknown = _task_outcomes(reference_env, window_start=window_start, exclude_lateness_ids=exclude_task_ids)
    def energy(events: list[dict]) -> float:
        return sum(float(item.get("energy", 0.0)) for item in events if item.get("type") in {"move", "service"} and int(item.get("time", -1)) > window_start)
    candidate_values = [
        float(candidate_overdue), float(candidate_unmet), energy(candidate_events),
        float(candidate_reassigned), float(candidate_return),
    ]
    reference_values = [
        float(reference_overdue), float(reference_unmet), energy(reference_events),
        float(reference_reassigned), float(reference_return),
    ]
    values = [candidate - reference for candidate, reference in zip(candidate_values, reference_values)]
    valid = [candidate_valid and reference_valid, True, True, True, True]
    unknown_reasons = [
        None if valid[0] else ";".join((
            ([f"candidate_unresolved_other_delivery:{','.join(candidate_unknown)}"] if candidate_unknown else [])
            + ([f"reference_unresolved_other_delivery:{','.join(reference_unknown)}"] if reference_unknown else [])
        )),
        None, None, None, None,
    ]
    return {
        "names": list(GLOBAL_TARGET_NAMES),
        "units": list(GLOBAL_TARGET_UNITS),
        "values": values,
        "candidate_values": candidate_values,
        "reference_values": reference_values,
        "valid": valid,
        "unknown_reasons": unknown_reasons,
        "hidden_state_label_only": True,
        "semantics": {
            "other_order_lateness": "candidate minus reference count of newly late non-target delivery orders after window_start at the common endpoint",
            "unmet_demand": "candidate minus reference count of undelivered delivery tasks",
            "energy": "candidate minus reference move/service energy units",
            "reassignment": "candidate minus reference count of reassigned tasks",
            "global_utility_delta": "candidate minus reference undiscounted environment reward over the common physical window",
        },
    }


def build_branch_record(
    snapshot_json: str,
    *,
    candidate_action_batch: Sequence[str],
    reference_action_batch: Sequence[str],
    parent_id: str,
    repeat: int,
    window_id: str,
    event_tape_id: str,
    candidate_id: str,
    candidate_return: float,
    reference_return: float,
    candidate_env: ReliefEnv,
    reference_env: ReliefEnv,
) -> dict[str, Any]:
    payload = json.loads(snapshot_json)
    source_env = restore_snapshot(snapshot_json)
    source_catalog = source_env.candidates()
    _require_identity(parent_id, repeat, window_id, event_tape_id)
    if (payload.get("parent_id"), payload.get("repeat"), payload.get("window_id"), payload.get("event_tape_id")) != (parent_id, repeat, window_id, event_tape_id):
        raise ValueError("BRANCH_RECORD_WINDOW_BINDING")
    candidate_actions = _validate_action_batch(source_catalog, candidate_action_batch)
    reference_actions = _validate_action_batch(source_catalog, reference_action_batch)
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("BRANCH_CANDIDATE_ID")
    if candidate_id != key_digest(candidate_actions):
        raise ValueError("BRANCH_CANDIDATE_ACTION_BINDING")
    expected_source = {
        "source_state_sha256": str(payload["state_sha256"]),
        "source_prefix_sha256": str(payload.get("source_prefix_sha256")),
        "scenario_sha256": str(payload["scenario_sha256"]),
        "event_tape_id": event_tape_id,
        "window_start": int(payload["time"]),
    }
    for env in (candidate_env, reference_env):
        if (
            getattr(env, "_branch_source_state_sha256", None) != expected_source["source_state_sha256"]
            or getattr(env, "_branch_source_prefix_sha256", None) != expected_source["source_prefix_sha256"]
            or getattr(env, "_branch_event_tape_id", None) != expected_source["event_tape_id"]
            or int(getattr(env, "_branch_window_start", -1)) != expected_source["window_start"]
            or _digest(getattr(env, "scenario", {})) != expected_source["scenario_sha256"]
            or _digest(getattr(env, "scenario", {}).get("disturbances", [])) != str(payload.get("event_tape_sha256"))
        ):
            raise ValueError("BRANCH_SOURCE_BINDING")
    if int(getattr(candidate_env, "time", -1)) != int(getattr(reference_env, "time", -2)):
        raise ValueError("BRANCH_ENDPOINT_TIME_MISMATCH")
    prefix_length = int(getattr(candidate_env, "_branch_prefix_length", -1))
    if prefix_length < 0 or prefix_length != int(getattr(reference_env, "_branch_prefix_length", -2)):
        raise ValueError("BRANCH_PREFIX_LENGTH_MISMATCH")
    candidate_prefix = getattr(candidate_env, "transitions", [])[:prefix_length]
    reference_prefix = getattr(reference_env, "transitions", [])[:prefix_length]
    if _digest(candidate_prefix) != expected_source["source_prefix_sha256"] or candidate_prefix != reference_prefix:
        raise ValueError("BRANCH_PREFIX_MISMATCH")
    for env, actions in ((candidate_env, candidate_actions), (reference_env, reference_actions)):
        transitions = getattr(env, "transitions", [])
        if len(transitions) <= prefix_length:
            raise ValueError("BRANCH_ACTION_TRANSITION_MISSING")
        realized = [str(item.get("id")) for item in transitions[prefix_length].get("selected_actions", [])]
        expected = [item for item in actions if item != "NOOP"]
        if realized != expected:
            raise ValueError("BRANCH_ACTION_TRANSITION_MISMATCH")
    candidate_events = copy.deepcopy(getattr(candidate_env, "events", []))
    reference_events = copy.deepcopy(getattr(reference_env, "events", []))
    candidate_target_ids = _task_ids_from_actions(candidate_env, candidate_actions, source_catalog)
    reference_target_ids = _task_ids_from_actions(reference_env, reference_actions, source_catalog)
    exclude_task_ids = candidate_target_ids | reference_target_ids
    public_task_ids = [str(item.get("id")) for item in payload.get("public_prefix", {}).get("tasks", []) if item.get("id") is not None]
    return {
        "schema": SCHEMA,
        "parent_id": parent_id,
        "repeat": repeat,
        "window_id": window_id,
        "candidate_id": candidate_id,
        "window_start": int(payload["time"]),
        "event_tape_id": event_tape_id,
        "source_state_sha256": payload["state_sha256"],
        "scenario_sha256": payload["scenario_sha256"],
        "source_prefix_sha256": payload.get("source_prefix_sha256"),
        "policy_input_sha256": key_digest({"observation": payload["public_prefix"], "candidates": payload["candidate_catalog"]}),
        "candidate_action_batch": candidate_actions,
        "candidate_is_noop": candidate_actions == ["NOOP"],
        "reference_action_batch": reference_actions,
        "candidate_action_batch_sha256": _digest(candidate_actions),
        "reference_action_batch_sha256": _digest(reference_actions),
        "candidate_return": float(candidate_return),
        "reference_return": float(reference_return),
        "paired_targets": {
            "global": paired_global_targets(
                candidate_env, reference_env, candidate_return, reference_return,
                candidate_events, reference_events, window_start=int(payload["time"]), exclude_task_ids=exclude_task_ids,
            ),
            "task": paired_task_targets(candidate_env, reference_env, public_task_ids),
        },
        "candidate_events": candidate_events,
        "reference_events": reference_events,
        "model_input_scope": "public observation, public candidate catalog, complete action batches only",
        "hidden_state_scope": "restore and result labels only",
    }


def collect_window_snapshot(env: ReliefEnv, *, parent_id: str, repeat: int, event_tape_id: str) -> str:
    if int(env.time) not in ENGINEERING_WINDOW_STARTS:
        raise ValueError("BRANCH_WINDOW_TIME")
    window_id = f"{parent_id}:{repeat}:{int(env.time)}"
    return capture_snapshot(env, parent_id=parent_id, repeat=repeat, window_id=window_id, event_tape_id=event_tape_id)


def reference_action_batch(env: ReliefEnv) -> list[str]:
    catalog = env.candidates()
    actions = choose_step_c1_actions(env.public_observation(), catalog)
    return [str(item["id"]) for item in actions] or ["NOOP"]
