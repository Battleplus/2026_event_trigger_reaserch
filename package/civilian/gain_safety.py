"""Public queue-feasibility override shared by all gain-study methods."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

try:
    from .event_environment import _distance, _communication_metrics
except ImportError:
    from event_environment import _distance, _communication_metrics


_TERMINAL_TASK_STATES = {"completed", "failed", "cancelled", "expired", "split_parent"}
_STOCK_COMMITMENT_PHASES = {"assigned", "to_depot", "loading"}
_LOADED_TASK_PHASES = {"in_transit", "to_delivery_site", "unloading", "delivery_confirmed"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _public_digest(observation: dict) -> str:
    """Hash the observation body using the event environment's provenance rule."""
    body = copy.deepcopy(observation)
    if isinstance(body.get("provenance"), dict):
        body.pop("provenance", None)
    return hashlib.sha256(_canonical(body)).hexdigest()


def _observation_digest(observation: dict) -> str:
    provenance = observation.get("provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("snapshot_hash"), str):
        return str(provenance["snapshot_hash"])
    return _public_digest(observation)


def _active_mission(platform: dict, task: dict, depot: dict, position: list[float]) -> tuple[float, float] | None:
    """Return public remaining travel distance and service minutes for an active task."""
    phase = str(task.get("phase", ""))
    progress = max(0.0, float(task.get("service_progress", 0.0)))
    target = list(task.get("position", position))
    if task.get("kind") == "inspection":
        if phase == "observe":
            return 0.0, max(0.0, float(task.get("service_minutes", 1.0)) - progress)
        if phase in {"to_inspection", "assigned"}:
            return _distance(position, target), max(0.0, float(task.get("service_minutes", 1.0)))
        return None

    depot_position = list(depot.get("position", position))
    load = max(0.0, float(task.get("load_minutes", 1.0)))
    unload = max(0.0, float(task.get("unload_minutes", 1.0)))
    if phase in {"assigned", "to_depot"}:
        distance = _distance(position, depot_position) + _distance(depot_position, target)
        service = load + unload
    elif phase == "loading":
        distance = _distance(position, depot_position) + _distance(depot_position, target)
        service = max(0.0, load - progress) + unload
    elif phase in {"in_transit", "to_delivery_site"}:
        distance = _distance(position, target)
        service = unload
    elif phase == "unloading":
        distance = _distance(position, target)
        service = max(0.0, unload - max(0.0, progress - load))
    else:
        return None
    return distance, service


def assess_queue_safety(public_observation: dict) -> dict[str, Any]:
    """Flag committed public queues that need immediate simulator repair.

    This is a deterministic queue-feasibility rule. It is not a flight-safety
    assurance and it does not inspect hidden task or future-event data.
    """
    if not isinstance(public_observation, dict):
        raise TypeError("public_observation must be a mapping")
    platforms = {str(row.get("id")): row for row in public_observation.get("platforms", [])
                 if isinstance(row, dict) and isinstance(row.get("id"), str)}
    tasks = {str(row.get("id")): row for row in public_observation.get("tasks", [])
             if isinstance(row, dict) and isinstance(row.get("id"), str) and bool(row.get("known", True))}
    depots = {str(row.get("id")): row for row in public_observation.get("depots", [])
              if isinstance(row, dict) and isinstance(row.get("id"), str)}
    queues = public_observation.get("queues", {})
    if not isinstance(queues, dict):
        raise ValueError("public queue map is malformed")
    now = float(public_observation.get("time", 0.0))
    violations: list[dict[str, Any]] = []
    soft_deadline_risks: list[dict[str, Any]] = []
    checked_tasks: set[str] = set()
    queued_task_owners: dict[str, str] = {}
    stock_commitments: dict[tuple[str, str], list[dict[str, Any]]] = {}
    platform_count = 0
    for platform_id, queue in sorted(queues.items()):
        platform = platforms.get(str(platform_id))
        if not isinstance(queue, list):
            violations.append({"code": "queued_platform_queue_malformed", "platform_id": str(platform_id)})
            continue
        if platform is None:
            if queue:
                violations.append({"code": "queued_platform_identity_unknown", "platform_id": str(platform_id)})
            continue
        if not queue:
            continue
        platform_count += 1
        if platform.get("failed"):
            violations.append({"code": "queued_work_on_failed_platform", "platform_id": platform_id})
            continue
        position = list(platform.get("position", [0.0, 0.0]))
        speed = max(1e-9, float(platform.get("speed_km_min", 1.0)))
        energy_left = max(0.0, float(platform.get("energy", 0.0)))
        elapsed = 0.0
        for task_id in queue:
            task = tasks.get(str(task_id))
            if task is None:
                violations.append({"code": "queued_task_identity_unknown",
                                   "platform_id": str(platform_id), "task_id": str(task_id)})
                continue
            if task.get("status") in _TERMINAL_TASK_STATES:
                continue
            checked_tasks.add(str(task_id))
            previous_owner = queued_task_owners.get(str(task_id))
            if previous_owner is not None:
                violations.append({
                    "code": "queued_task_committed_to_multiple_platforms",
                    "platform_id": platform_id,
                    "affected_platform_ids": sorted({previous_owner, str(platform_id)}),
                    "task_id": str(task_id),
                })
                first_queue_occurrence = False
            else:
                queued_task_owners[str(task_id)] = str(platform_id)
                first_queue_occurrence = True
            required_capability = "inspect" if task.get("kind") == "inspection" else "deliver"
            target = list(task.get("position", position))
            is_active = task.get("status") == "active" or platform.get("active_task_id") == str(task_id)
            # A live safety disposition may hold an active task while its
            # control link is unavailable.  The wrapper marks this state
            # explicitly and preserves the physical assignment, so it is no
            # longer an unresolved queue violation until the link recovers.
            # A paused status alone is not evidence that the live safety hook
            # applied a hold.  Require the explicit marker so stale or
            # malformed observations cannot suppress a capability violation.
            safety_paused = task.get("safety_paused") is True
            # Capability loss on an active task is handled by the simulator's
            # same-tick capability hold.  Keep the task in its queue and
            # preserve progress/cargo/stock; only an unheld task is a forced
            # queue violation here.  Pre-start work remains releasable.
            if required_capability not in platform.get("capabilities", []) and not (is_active and safety_paused):
                violations.append({"code": "queued_task_capability_infeasible", "platform_id": platform_id,
                                   "task_id": task_id, "required_capability": required_capability})
            if task.get("kind") == "delivery":
                depot_id = str(task.get("depot_id"))
                depot = depots.get(depot_id, {})
                if is_active:
                    mission = _active_mission(platform, task, depot, position)
                    if mission is None:
                        violations.append({"code": "active_task_phase_unknown", "platform_id": platform_id,
                                           "task_id": str(task_id), "phase": task.get("phase")})
                        continue
                    distance, service = mission
                else:
                    depot_position = list(depot.get("position", position))
                    distance = _distance(position, depot_position) + _distance(depot_position, target)
                    service = max(0.0, float(task.get("load_minutes", 1))) + max(0.0, float(task.get("unload_minutes", 1)))
                quantity = max(0.0, float(task.get("quantity", 0.0)))
                if quantity > float(platform.get("capacity_kg", 0.0)) + 1e-9:
                    violations.append({"code": "queued_task_capacity_infeasible", "platform_id": platform_id,
                                       "task_id": task_id, "quantity": quantity})
                commodity = str(task.get("commodity"))
                phase = str(task.get("phase", ""))
                not_loaded = phase not in _LOADED_TASK_PHASES
                if first_queue_occurrence and not_loaded:
                    key = (depot_id, commodity)
                    stock_commitments.setdefault(key, []).append({
                        "platform_id": str(platform_id), "task_id": str(task_id), "quantity": quantity,
                    })
            else:
                if is_active:
                    mission = _active_mission(platform, task, {}, position)
                    if mission is None:
                        violations.append({"code": "active_task_phase_unknown", "platform_id": platform_id,
                                           "task_id": str(task_id), "phase": task.get("phase")})
                        continue
                    distance, service = mission
                else:
                    distance = _distance(position, target)
                    service = max(0.0, float(task.get("service_minutes", 1.0)))
            service_left = service
            travel = distance / speed
            required_energy = distance * max(0.0, float(platform.get("energy_per_km", 0.0)))
            required_energy += service_left * max(0.0, float(platform.get("service_energy_per_min", 0.0)))
            elapsed += travel + service_left
            energy_left -= required_energy
            deadline = float(task.get("deadline", public_observation.get("horizon", now)))
            eta = now + elapsed
            if eta > deadline + 1e-9 and task.get("hard_deadline") is True:
                violations.append({"code": "queued_task_deadline_infeasible", "platform_id": platform_id,
                                   "task_id": task_id, "eta": eta, "deadline": deadline})
            elif eta > deadline + 1e-9:
                soft_deadline_risks.append({
                    "platform_id": str(platform_id), "task_id": str(task_id),
                    "eta": eta, "deadline": deadline, "lateness_minutes": eta - deadline,
                })
            if energy_left < -1e-9:
                violations.append({"code": "committed_queue_energy_infeasible", "platform_id": platform_id,
                                   "task_id": task_id, "energy_shortfall": -energy_left})
            energy_left = max(0.0, energy_left)
            position = target
            if task.get("status") == "active" and not safety_paused:
                link = _communication_metrics(public_observation, "n02-uav-1", str(platform_id))
                if not link.get("connected", False):
                    violations.append({"code": "active_task_control_link_lost", "platform_id": platform_id,
                                       "task_id": task_id})
    for (depot_id, commodity), commitments in sorted(stock_commitments.items()):
        stock = float(depots.get(depot_id, {}).get("stock", {}).get(commodity, 0.0))
        quantity = sum(float(row["quantity"]) for row in commitments)
        if quantity > stock + 1e-9:
            violations.append({
                "code": "committed_queue_inventory_infeasible",
                "affected_platform_ids": sorted({str(row["platform_id"]) for row in commitments}),
                "affected_task_ids": sorted({str(row["task_id"]) for row in commitments}),
                "depot_id": depot_id, "commodity": commodity,
                "quantity": quantity, "stock": stock, "shortfall": quantity - stock,
            })
    violations.sort(key=lambda row: (
        str(row.get("platform_id", "")), str(row.get("task_id", "")), str(row["code"]),
    ))
    affected_platforms = sorted({
        platform_id
        for row in violations
        for platform_id in ([str(row["platform_id"])] if row.get("platform_id") is not None else [])
        + list(map(str, row.get("affected_platform_ids", [])))
    })
    affected_tasks = sorted({
        task_id
        for row in violations
        for task_id in ([str(row["task_id"])] if row.get("task_id") is not None else [])
        + list(map(str, row.get("affected_task_ids", [])))
    })
    global_required = any(row["code"] in {"queued_work_on_failed_platform", "active_task_control_link_lost"}
                          for row in violations) or len(affected_platforms) > 1
    return {
        "schema": "gain-public-queue-safety/1",
        "forced": bool(violations),
        "safe_to_suppress": not bool(violations),
        "required_scope": "GLOBAL_REPLAN" if global_required else "LOCAL_REPAIR",
        "violations": violations,
        "affected_platform_ids": affected_platforms,
        "affected_task_ids": affected_tasks,
        "checked_task_count": len(checked_tasks),
        "checked_platform_count": platform_count,
        "soft_deadline_risks": sorted(soft_deadline_risks, key=lambda row: (row["platform_id"], row["task_id"])),
        "deadline_semantics": "soft_unless_public_task_explicitly_sets_hard_deadline_true",
        "scope": "public_queue_feasibility_only",
    }


def apply_safety_disposition(
    env: Any,
    public_observation: dict,
    safety_report: dict,
) -> dict[str, Any]:
    """Apply only a current-time, public-triggered safety disposition.

    The event environment owns physical mutation (pause/release/resume).  This
    adapter deliberately does not call ``step`` or inspect a private future
    event tape.  A missing hook is an explicit unknown result rather than a
    projected or inferred resolution.
    """
    if not isinstance(public_observation, dict) or not isinstance(safety_report, dict):
        raise TypeError("public_observation and safety_report must be mappings")
    if not safety_report.get("forced", False):
        return {
            "schema": "gain-live-safety-disposition/1", "status": "not_required",
            "resolved": True, "actions": [],
            "pre_public_sha256": _observation_digest(public_observation),
            "post_public_sha256": _observation_digest(public_observation),
            "actions_sha256": _digest([]),
            "remaining_report": copy.deepcopy(safety_report), "unknown_reason": None,
        }
    hook = getattr(env, "apply_current_safety_disposition", None)
    if not callable(hook):
        return {
            "schema": "gain-live-safety-disposition/1", "status": "unknown",
            "resolved": False, "actions": [],
            "pre_public_sha256": _observation_digest(public_observation),
            "post_public_sha256": None, "remaining_report": None,
            "actions_sha256": _digest([]),
            "unknown_reason": "current_time_safety_hook_unavailable",
        }
    try:
        result = hook(copy.deepcopy(public_observation), copy.deepcopy(safety_report))
    except Exception as exc:
        return {
            "schema": "gain-live-safety-disposition/1", "status": "unknown",
            "resolved": False, "actions": [],
            "pre_public_sha256": _observation_digest(public_observation),
            "post_public_sha256": None, "remaining_report": None,
            "actions_sha256": _digest([]),
            "unknown_reason": f"current_time_safety_hook_error:{type(exc).__name__}",
        }
    if not isinstance(result, dict):
        return {
            "schema": "gain-live-safety-disposition/1", "status": "unknown",
            "resolved": False, "actions": [],
            "pre_public_sha256": _observation_digest(public_observation),
            "post_public_sha256": None, "remaining_report": None,
            "actions_sha256": _digest([]),
            "unknown_reason": "current_time_safety_hook_malformed",
        }
    normalized = copy.deepcopy(result)
    actions = normalized.get("actions")
    if isinstance(actions, list) and "actions_sha256" not in normalized:
        normalized["actions_sha256"] = _digest(actions)
    return normalized


def _unknown_disposition(reason: str, before: dict, *, receipt: dict | None = None) -> dict[str, Any]:
    result = {
        "schema": "gain-safety-disposition/1", "status": "unknown", "resolved": False,
        "remaining_report": None, "unknown_reason": reason,
    }
    if receipt is not None:
        result["execution_receipt"] = copy.deepcopy(receipt)
    return result


def _legacy_projection_disposition(env: Any, public_observation: dict, action: dict, before: dict) -> dict[str, Any]:
    """Compatibility result for pre-V2 fixtures; it never certifies resolution."""
    projection = getattr(env, "_planned_queue_state", None)
    if not callable(projection):
        return _unknown_disposition("current_time_safety_receipt_missing", before)
    try:
        queues, projected_tasks = projection(
            public_observation,
            action.get("queue_operations", []),
            action.get("dispatches", []),
        )
        projected = copy.deepcopy(public_observation)
        projected["queues"] = {str(key): list(value) for key, value in queues.items()}
        observed_tasks = {
            str(row["id"]): copy.deepcopy(row)
            for row in public_observation.get("tasks", [])
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        }
        if isinstance(projected_tasks, dict):
            for task_id in list(observed_tasks):
                if isinstance(projected_tasks.get(task_id), dict):
                    observed_tasks[task_id].update(copy.deepcopy(projected_tasks[task_id]))
            known_split_parents = {
                str(row.get("task_id"))
                for row in action.get("queue_operations", [])
                if isinstance(row, dict) and row.get("op") in {"split", "cooperate"}
                and str(row.get("task_id")) in observed_tasks
            }
            allowed_child_prefixes = tuple(
                f"{parent_id}::{kind}-" for parent_id in sorted(known_split_parents)
                for kind in ("part", "team")
            )
            queued_ids = {
                str(task_id) for queue in projected.get("queues", {}).values()
                for task_id in queue
            }
            for task_id in sorted(queued_ids - set(observed_tasks)):
                if (
                    allowed_child_prefixes
                    and task_id.startswith(allowed_child_prefixes)
                    and isinstance(projected_tasks.get(task_id), dict)
                ):
                    observed_tasks[task_id] = copy.deepcopy(projected_tasks[task_id])
        projected["tasks"] = [observed_tasks[key] for key in sorted(observed_tasks)]
        remaining = assess_queue_safety(projected)
    except Exception:
        return _unknown_disposition("legacy_queue_projection_failed", before)
    return {
        "schema": "gain-safety-disposition/1", "status": "unresolved", "resolved": False,
        "remaining_report": remaining,
        "unknown_reason": "selected_action_leaves_public_queue_infeasible",
    }


def verify_safety_disposition(
    env: Any,
    public_observation: dict,
    action: dict | None = None,
    initial_report: dict | None = None,
    *,
    apply_receipt: dict | None = None,
    post_public_observation: dict | None = None,
) -> dict[str, Any]:
    """Verify an already-applied same-time disposition without stepping.

    Resolution is accepted only from the live hook receipt and the actual
    post-disposition public state.  The optional projection fallback exists for
    old unit fixtures and is deliberately unable to return ``resolved=True``.
    """
    if not isinstance(public_observation, dict):
        raise TypeError("public_observation must be a mapping")
    if action is None:
        action = {}
    if not isinstance(action, dict):
        raise TypeError("action must be a mapping")
    before = initial_report or assess_queue_safety(public_observation)
    if not before.get("forced", False):
        return {"schema": "gain-safety-disposition/1", "status": "not_required",
                "resolved": True, "remaining_report": before}

    # New callers must pass the hook receipt and refreshed current observation.
    if apply_receipt is None or post_public_observation is None:
        return _legacy_projection_disposition(env, public_observation, action, before)
    if not isinstance(apply_receipt, dict) or apply_receipt.get("schema") != "gain-live-safety-disposition/1":
        return _unknown_disposition("current_time_safety_receipt_invalid", before, receipt=apply_receipt if isinstance(apply_receipt, dict) else None)
    if not isinstance(post_public_observation, dict):
        return _unknown_disposition("post_disposition_public_observation_missing", before, receipt=apply_receipt)
    pre_hash = _observation_digest(public_observation)
    post_hash = _observation_digest(post_public_observation)
    if apply_receipt.get("pre_public_sha256") != pre_hash:
        return _unknown_disposition("safety_receipt_pre_public_hash_mismatch", before, receipt=apply_receipt)
    if apply_receipt.get("post_public_sha256") != post_hash:
        return _unknown_disposition("safety_receipt_post_public_hash_mismatch", before, receipt=apply_receipt)
    if apply_receipt.get("status") == "unknown":
        return _unknown_disposition(str(apply_receipt.get("unknown_reason") or "current_time_safety_unverified"), before, receipt=apply_receipt)

    remaining = assess_queue_safety(post_public_observation)
    receipt_remaining = apply_receipt.get("remaining_report")
    if isinstance(receipt_remaining, dict) and receipt_remaining != remaining:
        return _unknown_disposition("safety_receipt_remaining_report_mismatch", before, receipt=apply_receipt)
    if remaining.get("forced", False):
        return {
            "schema": "gain-safety-disposition/1", "status": "unresolved", "resolved": False,
            "remaining_report": remaining,
            "unknown_reason": "current_time_disposition_leaves_queue_infeasible",
            "execution_receipt": copy.deepcopy(apply_receipt),
        }
    if apply_receipt.get("resolved") is not True or apply_receipt.get("status") not in {"resolved", "not_required"}:
        return _unknown_disposition("safety_receipt_resolution_flag_invalid", before, receipt=apply_receipt)
    return {
        "schema": "gain-safety-disposition/1", "status": "resolved", "resolved": True,
        "remaining_report": remaining, "unknown_reason": None,
        "execution_receipt": copy.deepcopy(apply_receipt),
    }


def verify_live_safety_disposition(
    env: Any,
    public_observation: dict,
    safety_report: dict,
    apply_receipt: dict,
    post_public_observation: dict,
) -> dict[str, Any]:
    """Named V2 entry point for pure verification of the live hook result."""
    return verify_safety_disposition(
        env, public_observation, {}, initial_report=safety_report,
        apply_receipt=apply_receipt, post_public_observation=post_public_observation,
    )
