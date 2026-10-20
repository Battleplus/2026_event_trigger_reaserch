"""Deterministic public-information replanner for gain-gate experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

try:
    from .event_environment import (
        EventReliefEnv,
        _communication_metrics,
        _distance,
        generate_reconstruction_candidates,
        repair_action,
    )
except ImportError:
    from event_environment import (
        EventReliefEnv,
        _communication_metrics,
        _distance,
        generate_reconstruction_candidates,
        repair_action,
    )


REPLANNER_VERSION = "Public-C1-Catalog-Replanner-v33"
RECEIPT_SCHEMA = "gain-fixed-replanner-receipt/1"
MAX_CANDIDATES = 16
_EPS = 1e-9
_COORDINATOR = "n02-uav-1"


def _coordinator_id(observation: dict) -> str:
    if observation.get("map", {}).get("region_boundaries") == [0, 33, 66, 100]:
        platforms = [row for row in observation.get("platforms", []) if row.get("type") == "uav"]
        return min((str(row.get("id")) for row in platforms), default="")
    return _COORDINATOR


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def canonical_action_key(action: dict) -> str:
    return hashlib.sha256(_canonical(action)).hexdigest()


def _route_estimate(platform: dict, task: dict, position: list[float],
                    depots: dict[str, dict]) -> tuple[float, float, float, list[float]]:
    """Estimate remaining route/service from public phase and position."""
    target = list(task.get("position", position))
    phase = str(task.get("phase", ""))
    total_service = (
        max(0.0, float(task.get("load_minutes", 1)))
        + max(0.0, float(task.get("unload_minutes", 1)))
        if task.get("kind") == "delivery"
        else max(0.0, float(task.get("service_minutes", 1)))
    )
    if task.get("kind") == "delivery":
        depot = depots.get(str(task.get("depot_id")), {})
        depot_position = list(depot.get("position", position))
        if phase in {"to_delivery_site", "in_transit"}:
            distance = _distance(position, target)
            service = max(0.0, float(task.get("unload_minutes", 1)))
        elif phase == "unloading":
            distance = 0.0
            completed_service = max(0.0, float(task.get("service_progress", 0.0)))
            service = max(0.0, total_service - completed_service)
        elif phase == "loading":
            distance = _distance(position, target)
            service = max(0.0, total_service - float(task.get("service_progress", 0.0)))
        elif phase == "to_depot":
            distance = _distance(position, depot_position) + _distance(depot_position, target)
            service = total_service
        else:
            distance = _distance(position, depot_position) + _distance(depot_position, target)
            service = total_service
    else:
        if phase == "observe":
            distance = 0.0
            service = max(0.0, float(task.get("service_minutes", 1)) - float(task.get("service_progress", 0.0)))
        else:
            distance = _distance(position, target)
            service = total_service
    speed = max(_EPS, float(platform.get("speed_km_min", 1.0)))
    travel = distance / speed
    energy = distance * max(0.0, float(platform.get("energy_per_km", 0.0)))
    energy += service * max(0.0, float(platform.get("service_energy_per_min", 0.0)))
    return travel, service, energy, target


def estimate_public_remaining_cost(env: EventReliefEnv, action: dict) -> float:
    """Estimate all known logical-task costs using only current public fields."""
    observation = env.public_observation()
    queues, projected_tasks = env._planned_queue_state(
        observation, action.get("queue_operations", []), action.get("dispatches", []),
    )
    tasks = {str(row.get("id")): row for row in projected_tasks.values() if isinstance(row, dict)}
    platforms = {str(row.get("id")): row for row in observation.get("platforms", []) if isinstance(row, dict)}
    depots = {str(row.get("id")): row for row in observation.get("depots", []) if isinstance(row, dict)}
    current_time = float(observation.get("time", 0))
    current_time = float(observation.get("time", 0))
    horizon = float(observation.get("horizon", current_time))
    current_tasks = {
        str(row["id"]): row for row in observation.get("tasks", [])
        if isinstance(row, dict) and bool(row.get("known", True))
        and row.get("status") not in {"completed", "failed", "cancelled", "expired"}
    }
    logical_costs: dict[str, dict[str, Any]] = {}
    stock_commitments: dict[tuple[str, str], float] = {}
    score = 0.0
    for platform_id, queue in sorted(queues.items()):
        platform = platforms.get(str(platform_id))
        if platform is None:
            score += 1e9
            continue
        position = list(platform.get("position", [0.0, 0.0]))
        elapsed = 0.0
        energy_left = max(0.0, float(platform.get("energy", 0.0)))
        for task_id in queue:
            task = tasks.get(str(task_id))
            if task is None or task.get("status") in {"completed", "failed", "cancelled", "expired", "split_parent"}:
                continue
            logical_id = str(task.get("parent_task_id") or task_id)
            parent = current_tasks.get(logical_id, current_tasks.get(str(task_id)))
            if parent is None:
                continue
            needed = "inspect" if task.get("kind") == "inspection" else "deliver"
            priority = max(0.0, float(parent.get("priority", 1.0)))
            if platform.get("failed"):
                score += priority * 1e6
                continue
            if needed not in platform.get("capabilities", []):
                score += priority * 1e5
            if task.get("kind") == "delivery":
                if float(task.get("quantity", 0.0)) > float(platform.get("capacity_kg", 0.0)) + _EPS:
                    score += priority * 1e5
                depot_id = str(task.get("depot_id"))
                commodity = str(task.get("commodity"))
                commitment_phase = str(task.get("phase", ""))
                if commitment_phase not in {"to_delivery_site", "in_transit", "unloading", "delivery_confirmed"}:
                    key = (depot_id, commodity)
                    stock_commitments[key] = stock_commitments.get(key, 0.0) + max(0.0, float(task.get("quantity", 0.0)))
            travel, service, required_energy, target = _route_estimate(platform, task, position, depots)
            elapsed += travel + service
            energy_left -= required_energy
            eta = current_time + elapsed
            deadline = float(parent.get("deadline", horizon))
            communication = _communication_metrics(
                observation, str(parent.get("assignee") or _coordinator_id(observation)), str(platform_id),
            )
            environment_risk = max(
                (float(row.get("risk", row.get("risk_level", 0.0)))
                 for row in observation.get("environment", [])
                 if isinstance(row, dict) and tuple(row.get("grid_cell", [])) == (math.floor(target[0]), math.floor(target[1]))),
                default=0.0,
            )
            row = logical_costs.setdefault(logical_id, {
                "priority": priority, "deadline": deadline, "etas": [], "energy": 0.0,
                "risk": 0.0, "communication": 0.0, "owners": set(),
            })
            row["etas"].append(eta)
            row["energy"] += required_energy
            row["risk"] += environment_risk
            row["communication"] += (
                float(communication.get("latency_minutes", 0.0))
                + 10.0 * float(communication.get("loss_rate", 0.0))
                + (100.0 if not communication.get("connected", False) else 0.0)
            )
            row["owners"].add(str(platform_id))
            if energy_left < -_EPS:
                score += priority * (1.0 + abs(energy_left))
            position = target

    for (depot_id, commodity), quantity in stock_commitments.items():
        stock = float(depots.get(depot_id, {}).get("stock", {}).get(commodity, 0.0))
        if quantity > stock + _EPS:
            score += (quantity - stock) * 1e4

    for logical_id, task in current_tasks.items():
        estimate = logical_costs.get(logical_id)
        priority = max(0.0, float(task.get("priority", 1.0)))
        if estimate is None:
            score += priority
            continue
        eta = max(estimate["etas"], default=horizon + 1.0)
        score += priority * (
            float(eta > horizon)
            + 0.2 * max(0.0, eta - float(task.get("deadline", horizon)))
            + 0.01 * float(estimate["energy"])
            + 0.25 * float(estimate["risk"])
            + 0.01 * float(estimate["communication"])
        )
        current_owners = {
            str(owner) for owner, queue in getattr(env, "_queues", {}).items()
            if any(str(item) == logical_id or str(item).startswith(logical_id + "::") for item in queue)
        }
        if current_owners and current_owners != estimate["owners"]:
            score += 0.05

    old_queues = getattr(env, "_queues", {})
    new_queues = queues
    owner_changes = 0
    reorder_changes = 0
    old_owner = {str(task_id): str(platform_id) for platform_id, queue in old_queues.items() for task_id in queue}
    new_owner = {str(task_id): str(platform_id) for platform_id, queue in new_queues.items() for task_id in queue}
    for task_id in set(old_owner) | set(new_owner):
        if old_owner.get(task_id) != new_owner.get(task_id):
            owner_changes += 1
    for platform_id in set(old_queues) | set(new_queues):
        old = list(old_queues.get(platform_id, [])); new = list(new_queues.get(platform_id, []))
        if old != new and {item for item in old if item in new} == {item for item in new if item in old}:
            reorder_changes += 1
    reconstruction_size = owner_changes + reorder_changes
    if reconstruction_size:
        score += 0.05 + 0.01 * reconstruction_size
    if not math.isfinite(score):
        raise ValueError("public residual-cost estimate is non-finite")
    return score


def _physical_outcome_key(env: EventReliefEnv, action: dict) -> str:
    observation = env.public_observation()
    queues, projected_tasks = env._planned_queue_state(
        observation, action.get("queue_operations", []), action.get("dispatches", []),
    )
    assigned = {str(task_id) for queue in queues.values() for task_id in queue}
    workload = []
    for task_id in sorted(assigned):
        task = projected_tasks.get(task_id, {})
        workload.append({
            "task_id": task_id, "parent_task_id": task.get("parent_task_id"),
            "workload_share": float(task.get("workload_share", 1.0)),
            "quantity": float(task.get("quantity", 0.0)),
            "service_minutes": float(task.get("service_minutes", 0.0)),
            "cooperation_group": task.get("cooperation_group"),
        })
    operational = sorted((str(row.get("platform_id")), str(row.get("task_id")))
                         for row in action.get("dispatches", []))
    payload = {
        "queues": {key: list(value) for key, value in sorted(queues.items())},
        "workload": workload, "operational_dispatches": operational,
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def replan(env: EventReliefEnv, affected_scope: Any = None, *, max_candidates: int = MAX_CANDIDATES) -> dict:
    """Generate, rank, repair, and select one public legal action.

    Candidate generation is intentionally confined to this call. A gain gate
    that suppresses reconstruction must not call this function.
    """
    if not isinstance(env, EventReliefEnv):
        raise TypeError("env must be an EventReliefEnv")
    if type(max_candidates) is not int or not 1 <= max_candidates <= MAX_CANDIDATES:
        raise ValueError("max_candidates must be an integer from 1 to 16")
    catalog = generate_reconstruction_candidates(env, scope=affected_scope, max_candidates=max_candidates)
    ranked: dict[str, dict] = {}
    infeasible = len(getattr(env, "last_candidate_rejections", []))
    physical_duplicates = 0
    for candidate in catalog:
        action = copy.deepcopy(candidate.get("full_environment_action"))
        repaired, repair_receipt = repair_action(env, action)
        normalized, issues = env._check_action(repaired)
        if issues or normalized != repaired or not repair_receipt.get("postrepair_legal"):
            infeasible += 1
            continue
        outcome_key = _physical_outcome_key(env, normalized)
        score = estimate_public_remaining_cost(env, normalized)
        row = {
            "candidate_key": candidate.get("candidate_key"), "physical_outcome_key": outcome_key,
            "full_environment_action": normalized, "candidate_scope": candidate.get("candidate_scope"),
            "public_cost_estimate": score, "repair_receipt": repair_receipt,
        }
        previous = ranked.get(outcome_key)
        tie_key = (score, canonical_action_key(normalized))
        if previous is None:
            ranked[outcome_key] = row
        else:
            physical_duplicates += 1
            previous_key = (float(previous["public_cost_estimate"]), canonical_action_key(previous["full_environment_action"]))
            if tie_key < previous_key:
                ranked[outcome_key] = row
    if not ranked:
        raise RuntimeError("fixed replanner produced no legal physical action")
    selected = min(ranked.values(), key=lambda row: (
        float(row["public_cost_estimate"]), canonical_action_key(row["full_environment_action"]),
    ))
    return {
        "schema": RECEIPT_SCHEMA,
        "replanner_version": REPLANNER_VERSION,
        "selected_action": copy.deepcopy(selected["full_environment_action"]),
        "selected_candidate_key": selected["candidate_key"],
        "selection_key": canonical_action_key(selected["full_environment_action"]),
        "selected_public_cost_estimate": float(selected["public_cost_estimate"]),
        "candidates": [ranked[key] for key in sorted(ranked)],
        "actual_candidate_count": len(catalog),
        "physical_duplicate_count": physical_duplicates,
        "infeasible_count": infeasible,
        "candidate_generation_called": True,
    }
