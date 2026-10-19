"""Small public-only feature projection for expected-gain gating."""

from __future__ import annotations

import math
from typing import Any


FEATURE_SCHEMA = "gain-public-features/1"
_EVENT_CATEGORIES = (
    "task", "platform", "energy", "communication", "environment", "state", "other",
)
_NUMERIC_DETECTION_FIELDS = (
    "gamma_value", "threshold", "d_state", "d_task", "d_platform", "d_environment",
    "d_communication", "d_margin", "d_margin_capability", "d_margin_time",
    "d_margin_energy", "d_margin_communication",
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def _distance(left: Any, right: Any) -> float:
    try:
        return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))
    except (TypeError, ValueError, IndexError):
        return 0.0


def _task_route(platform: dict, task: dict, depots: dict[str, dict]) -> tuple[float, float]:
    start = platform.get("position", [0.0, 0.0])
    target = task.get("position", start)
    if task.get("kind") == "delivery":
        depot = depots.get(str(task.get("depot_id")), {})
        depot_position = depot.get("position", start)
        distance = _distance(start, depot_position) + _distance(depot_position, target)
        service = _number(task.get("load_minutes")) + _number(task.get("unload_minutes"))
    else:
        distance = _distance(start, target)
        service = _number(task.get("service_minutes", task.get("load_minutes", 1) + task.get("unload_minutes", 1)))
    return distance, max(0.0, service)


def _queue_delay(platform: dict, task_id: str, queues: dict, tasks_by_id: dict,
                 depots: dict[str, dict]) -> float:
    queue = queues.get(str(platform.get("id")), [])
    if not isinstance(queue, list) or task_id not in queue:
        return 0.0
    delay = 0.0
    for queued_id in queue[:queue.index(task_id)]:
        queued = tasks_by_id.get(str(queued_id), {})
        distance, service = _task_route(platform, queued, depots)
        speed = max(1e-9, _number(platform.get("speed_km_min"), 1.0))
        delay += distance / speed + service
    return delay


def _task_margins(task: dict, platforms: list[dict], depots: dict[str, dict],
                  queues: dict, tasks_by_id: dict) -> tuple[list[float], list[float]]:
    needed = "inspect" if task.get("kind") == "inspection" else "deliver"
    etas: list[float] = []
    energy_margins: list[float] = []
    for platform in platforms:
        if platform.get("failed") or needed not in platform.get("capabilities", []):
            continue
        quantity = max(0.0, _number(task.get("quantity")))
        if task.get("kind") == "delivery" and quantity > _number(platform.get("capacity_kg")) + 1e-9:
            continue
        if task.get("kind") == "delivery":
            depot = depots.get(str(task.get("depot_id")), {})
            stock = _number(depot.get("stock", {}).get(str(task.get("commodity")), 0.0))
            if stock + 1e-9 < quantity:
                continue
        distance, service = _task_route(platform, task, depots)
        speed = max(1e-9, _number(platform.get("speed_km_min"), 1.0))
        eta = _queue_delay(platform, str(task.get("id", "")), queues, tasks_by_id, depots) + distance / speed + service
        service_energy = service * max(0.0, _number(platform.get("service_energy_per_min")))
        required_energy = distance * max(0.0, _number(platform.get("energy_per_km"))) + service_energy
        remaining_energy = max(0.0, _number(platform.get("energy")))
        etas.append(eta)
        energy_margins.append(remaining_energy - required_energy)
    return etas, energy_margins


def _event_category(detection: dict) -> str:
    details = detection.get("event_details", {})
    text = " ".join(str(value) for value in (
        detection.get("event_type", ""), detection.get("event_severity", ""),
        details.get("primary_category", "") if isinstance(details, dict) else "",
    )).lower()
    if any(token in text for token in ("task", "arrival", "deadline", "priority", "cancel")):
        return "task"
    if any(token in text for token in ("platform", "failure", "capability", "load")):
        return "platform"
    if any(token in text for token in ("energy", "battery")):
        return "energy"
    if any(token in text for token in ("communication", "network", "packet", "link")):
        return "communication"
    if any(token in text for token in ("environment", "wind", "obstacle", "threat", "passability")):
        return "environment"
    if text:
        return "state"
    return "other"


def extract_public_features(
    public_observation: dict,
    detection: dict | None = None,
    affected_scope: dict | None = None,
) -> dict[str, float]:
    """Return stable, numeric features derived only from current public inputs.

    Scenario identifiers, hashes, event tape identities, hidden task templates,
    candidate catalogs, and simulator truth are never traversed or copied.
    """
    if not isinstance(public_observation, dict):
        raise TypeError("public_observation must be a mapping")
    detection = detection if isinstance(detection, dict) else {}
    affected_scope = affected_scope if isinstance(affected_scope, dict) else {}
    platforms = [row for row in public_observation.get("platforms", []) if isinstance(row, dict)]
    tasks = [row for row in public_observation.get("tasks", []) if isinstance(row, dict)]
    known_tasks = [row for row in tasks if bool(row.get("known", True))]
    task_by_id = {str(row.get("id")): row for row in known_tasks}
    depots = {str(row.get("id")): row for row in public_observation.get("depots", [])
              if isinstance(row, dict)}
    queues = public_observation.get("queues", {})
    if not isinstance(queues, dict):
        queues = {}
    horizon = max(0.0, _number(public_observation.get("horizon")))
    time = max(0.0, _number(public_observation.get("time")))
    energies = []
    energy_fractions = []
    for platform in platforms:
        energy = max(0.0, _number(platform.get("energy")))
        initial = max(0.0, _number(platform.get("energy_initial")))
        energies.append(energy)
        if initial > 0:
            energy_fractions.append(min(1.0, energy / initial))
    deadlines = [_number(task.get("deadline"), horizon) for task in known_tasks]
    slack = [deadline - time for deadline in deadlines]
    pending = [task for task in known_tasks if task.get("status") in {"pending", "assigned", "released"}]
    pending_slacks = [_number(task.get("deadline"), horizon) - time for task in pending]
    pending_margins = [_task_margins(task, platforms, depots, queues, task_by_id) for task in pending]
    pending_etas = [min(etas) for etas, _ in pending_margins if etas]
    pending_energy_margins = [max(margins) for etas, margins in pending_margins if etas and margins]
    pending_capability_counts = [len(etas) for etas, _ in pending_margins]
    pending_eta_slacks = [
        _number(task.get("deadline"), horizon) - time - min(etas)
        for task, (etas, _) in zip(pending, pending_margins) if etas
    ]
    task_states = ("pending", "assigned", "active", "completed", "failed", "cancelled", "expired")
    features = {
        "time_fraction": _ratio(time, horizon),
        "remaining_time_fraction": max(0.0, 1.0 - _ratio(time, horizon)),
        "platform_count": float(len(platforms)),
        "task_count": float(len(known_tasks)),
        "platform_failed_fraction": _ratio(sum(bool(row.get("failed")) for row in platforms), len(platforms)),
        "platform_busy_fraction": _ratio(sum(bool(row.get("busy")) for row in platforms), len(platforms)),
        "energy_mean": _ratio(sum(energies), len(energies)),
        "energy_min": min(energies, default=0.0),
        "energy_fraction_mean": _ratio(sum(energy_fractions), len(energy_fractions)),
        "energy_fraction_min": min(energy_fractions, default=0.0),
        "task_priority_mean": _ratio(sum(_number(task.get("priority")) for task in known_tasks), len(known_tasks)),
        "task_priority_max": max((_number(task.get("priority")) for task in known_tasks), default=0.0),
        "deadline_slack_min": min(slack, default=0.0),
        "deadline_slack_mean": _ratio(sum(slack), len(slack)),
        "pending_task_count": float(len(pending)),
        "pending_deadline_slack_min": min(pending_slacks, default=0.0),
        "pending_deadline_slack_mean": _ratio(sum(pending_slacks), len(pending_slacks)),
        "pending_eta_min": min(pending_etas, default=0.0),
        "pending_eta_mean": _ratio(sum(pending_etas), len(pending_etas)),
        "pending_eta_max": max(pending_etas, default=0.0),
        "pending_deadline_after_eta_min": min(pending_eta_slacks, default=0.0),
        "pending_deadline_after_eta_mean": _ratio(sum(pending_eta_slacks), len(pending_eta_slacks)),
        "pending_capability_options_min": min(pending_capability_counts, default=0),
        "pending_capability_options_mean": _ratio(sum(pending_capability_counts), len(pending_capability_counts)),
        "pending_energy_margin_min": min(pending_energy_margins, default=0.0),
        "pending_energy_margin_mean": _ratio(sum(pending_energy_margins), len(pending_energy_margins)),
        "pending_energy_feasible_fraction": _ratio(sum(margin >= 0.0 for margin in pending_energy_margins), len(pending_energy_margins)),
        "task_progress_mean": _ratio(sum(min(1.0, max(0.0, _number(row.get("progress")))) for row in tasks), len(tasks)),
        "queue_task_count": float(sum(len(queue) for queue in public_observation.get("queues", {}).values()
                                       if isinstance(queue, list))),
        "communication_quality_mean": _ratio(sum(_number(row.get("quality")) for row in public_observation.get("communication", [])
                                                  if isinstance(row, dict)),
                                               sum(isinstance(row, dict) for row in public_observation.get("communication", []))),
        "communication_disconnected_fraction": _ratio(sum(not bool(row.get("connected", True)) for row in public_observation.get("communication", [])
                                                            if isinstance(row, dict)),
                                                       sum(isinstance(row, dict) for row in public_observation.get("communication", []))),
        "affected_task_fraction": _ratio(len(set(affected_scope.get("affected_task_ids", [])) & set(task_by_id)), len(known_tasks)),
        "affected_platform_fraction": _ratio(len(set(affected_scope.get("affected_platform_ids", []))), len(platforms)),
        "local_scope_eligible": float(bool(affected_scope.get("local_eligible", False))),
        "global_scope_required": float(bool(affected_scope.get("global_eligible", False))),
        "safety_event": float(bool(detection.get("strong_event", False))),
    }
    for state in task_states:
        features[f"task_status_{state}_fraction"] = _ratio(
            sum(row.get("status") == state for row in known_tasks), len(known_tasks),
        )
    for field in _NUMERIC_DETECTION_FIELDS:
        features[f"event_{field}"] = _number(detection.get(field))
    category = _event_category(detection)
    for name in _EVENT_CATEGORIES:
        features[f"event_category_{name}"] = float(name == category)
    return {key: value if math.isfinite(value) else 0.0 for key, value in features.items()}
