"""Public queue-feasibility override shared by all gain-study methods."""

from __future__ import annotations

import copy
import math
from typing import Any

try:
    from .event_environment import _distance, _communication_metrics
except ImportError:
    from event_environment import _distance, _communication_metrics


_TERMINAL_TASK_STATES = {"completed", "failed", "cancelled", "expired", "split_parent"}
_STOCK_COMMITMENT_PHASES = {"assigned", "to_depot", "loading"}
_LOADED_TASK_PHASES = {"in_transit", "to_delivery_site", "unloading", "delivery_confirmed"}


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
            if required_capability not in platform.get("capabilities", []):
                violations.append({"code": "queued_task_capability_infeasible", "platform_id": platform_id,
                                   "task_id": task_id, "required_capability": required_capability})
            target = list(task.get("position", position))
            is_active = task.get("status") == "active" or platform.get("active_task_id") == str(task_id)
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
            if task.get("status") == "active":
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


def verify_safety_disposition(
    env: Any,
    public_observation: dict,
    action: dict,
    initial_report: dict | None = None,
) -> dict[str, Any]:
    """Check whether a public action would resolve forced queue infeasibility."""
    if not isinstance(public_observation, dict) or not isinstance(action, dict):
        raise TypeError("public_observation and action must be mappings")
    before = initial_report or assess_queue_safety(public_observation)
    if not before.get("forced", False):
        return {"schema": "gain-safety-disposition/1", "status": "not_required",
                "resolved": True, "remaining_report": before}
    projection = getattr(env, "_planned_queue_state", None)
    if not callable(projection):
        return {"schema": "gain-safety-disposition/1", "status": "unknown",
                "resolved": False, "unknown_reason": "public_queue_projection_unavailable"}
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
    projected_tasks = projected_tasks if isinstance(projected_tasks, dict) else {}
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
        f"{parent_id}::{kind}-" for parent_id in sorted(known_split_parents) for kind in ("part", "team")
    )
    queued_ids = {str(task_id) for queue in projected["queues"].values() for task_id in queue}
    for task_id in sorted(queued_ids - set(observed_tasks)):
        if allowed_child_prefixes and task_id.startswith(allowed_child_prefixes) and isinstance(projected_tasks.get(task_id), dict):
            observed_tasks[task_id] = copy.deepcopy(projected_tasks[task_id])
    projected["tasks"] = [observed_tasks[key] for key in sorted(observed_tasks)]
    remaining = assess_queue_safety(projected)
    resolved = not bool(remaining.get("forced", False))
    return {
        "schema": "gain-safety-disposition/1",
        "status": "resolved" if resolved else "unresolved",
        "resolved": resolved,
        "remaining_report": remaining,
        "unknown_reason": None if resolved else "selected_action_leaves_public_queue_infeasible",
    }
