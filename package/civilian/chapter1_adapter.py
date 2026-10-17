"""Public-only C1 initial assignment and the Chapter 1 handoff contract."""

from __future__ import annotations

import copy
import math

try:
    from .baselines import choose_step_c1_actions
    from .scenario_schema import C1_ALGORITHM_ID, C1_SNAPSHOT_SOURCE, validate_scenario
    from .stable_identity import key_digest
except ImportError:
    from baselines import choose_step_c1_actions
    from scenario_schema import C1_ALGORITHM_ID, C1_SNAPSHOT_SOURCE, validate_scenario
    from stable_identity import key_digest


def _distance(left: list[float], right: list[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _travel_ticks(distance: float, speed: float) -> int:
    if distance <= 1e-9:
        return 0
    return math.ceil(distance / speed - 1e-9)


def _mission_metrics(platform: dict, task: dict, depots: dict[str, dict]) -> tuple[float, float]:
    speed = float(platform["speed_km_min"])
    energy_per_km = float(platform["energy_per_km"])
    service_energy = float(platform["service_energy_per_min"])
    if task["kind"] == "inspection":
        distance = _distance(platform["position"], task["position"])
        service = float(task["service_minutes"])
        return (
            _travel_ticks(distance, speed) + service,
            distance * energy_per_km + service * service_energy,
        )
    depot = depots[task["depot_id"]]
    to_depot = _distance(platform["position"], depot["position"])
    to_task = _distance(depot["position"], task["position"])
    load = float(task["load_minutes"])
    unload = float(task["unload_minutes"])
    completion = _travel_ticks(to_depot, speed) + load + _travel_ticks(to_task, speed) + unload
    required_energy = (to_depot + to_task) * energy_per_km + (load + unload) * service_energy
    return completion, required_energy


def generate_public_c1_initial_assignment(scenario: dict, *, seed: int) -> dict:
    """Run the existing deterministic civilian C1 rule over public t=0 inputs."""

    if type(seed) is not int or seed < 0:
        raise ValueError("C1_INITIAL_SEED")
    known_tasks = [copy.deepcopy(task) for task in scenario["tasks"] if task["known_initial"]]
    platforms = [copy.deepcopy(platform) for platform in scenario["platforms"]]
    depots = [copy.deepcopy(depot) for depot in scenario["depots"]]
    task_by_id = {task["id"]: task for task in known_tasks}
    platform_by_id = {platform["id"]: platform for platform in platforms}
    depot_by_id = {depot["id"]: depot for depot in depots}

    public_input = {
        "schema": "civilian-public-c1-input/1",
        "seed": seed,
        "map": copy.deepcopy(scenario["map"]),
        "platforms": platforms,
        "depots": depots,
        "known_tasks": known_tasks,
    }
    observation = {
        "schema": "relief-public-observation/1",
        "time": 0,
        "horizon": int(scenario["horizon"]),
        "map": copy.deepcopy(scenario["map"]),
        "platforms": [
            {
                **copy.deepcopy(platform),
                "energy": float(platform["energy_initial"]),
                "failed": False,
                "busy": False,
            }
            for platform in platforms
        ],
        "depots": depots,
        "tasks": [
            {
                **copy.deepcopy(task),
                "status": "pending",
                "phase": "pending",
                "assignee": None,
                "original_assignee": None,
                "discovered": True,
            }
            for task in known_tasks
        ],
        "chapter1_snapshot": {
            "source": C1_SNAPSHOT_SOURCE,
            "time": 0,
            "queues": {platform["id"]: [] for platform in platforms},
        },
    }
    candidates = []
    rejected_by_task: dict[str, set[str]] = {task["id"]: set() for task in known_tasks}
    for platform_id in sorted(platform_by_id):
        platform = platform_by_id[platform_id]
        for task_id in sorted(task_by_id):
            task = task_by_id[task_id]
            required_capability = "inspect" if task["kind"] == "inspection" else "deliver"
            if required_capability not in platform["capabilities"] or bool(platform["local"]) != bool(task["local"]):
                rejected_by_task[task_id].add("no_compatible_platform")
                continue
            if task["kind"] == "delivery" and float(task["quantity"]) > float(platform["capacity_kg"]) + 1e-9:
                rejected_by_task[task_id].add("capacity_exceeded")
                continue
            if task["kind"] == "delivery":
                depot = depot_by_id[task["depot_id"]]
                stock = float(depot["stock"].get(task["commodity"], 0.0))
                if stock + 1e-9 < float(task["quantity"]):
                    rejected_by_task[task_id].add("insufficient_stock")
                    continue
            completion, required_energy = _mission_metrics(platform, task, depot_by_id)
            if float(platform["energy_initial"]) + 1e-9 < required_energy:
                rejected_by_task[task_id].add("insufficient_energy")
                continue
            candidates.append({
                "id": f"dispatch:{platform_id}:{task_id}",
                "type": "dispatch",
                "platform_id": platform_id,
                "task_id": task_id,
                "kind": task["kind"],
                "estimated_completion": float(completion),
                "required_energy": float(required_energy),
                "locked": False,
            })

    selected = choose_step_c1_actions(observation, candidates)
    queues = {platform_id: [] for platform_id in sorted(platform_by_id)}
    for action in selected:
        queues[action["platform_id"]].append(action["task_id"])
    assigned = {task_id for values in queues.values() for task_id in values}
    unassigned_reasons = {}
    for task in known_tasks:
        task_id = task["id"]
        if task_id in assigned:
            continue
        reasons = rejected_by_task[task_id]
        unassigned_reasons[task_id] = (
            sorted(reasons) if reasons else ["c1_priority_or_one_task_per_platform"]
        )

    receipt = {
        "schema": "chapter1-public-c1-assignment/1",
        "algorithm_id": C1_ALGORITHM_ID,
        "algorithm_source": "package/civilian/baselines.py:C1/choose_step_c1_actions",
        "seed": seed,
        "seed_usage": "recorded_for_reproducibility; deterministic C1 uses no random draws",
        "public_input_sha256": key_digest(public_input),
        "known_task_ids": sorted(task_by_id),
        "candidate_pair_count": len(candidates),
        "selected_action_ids": [action["id"] for action in selected],
        "assignments": copy.deepcopy(queues),
        "assignment_output_sha256": key_digest(queues),
        "feasibility": {
            "all_selected_actions_legal": True,
            "known_task_count": len(known_tasks),
            "feasible_candidate_pair_count": len(candidates),
            "assigned_task_count": len(assigned),
            "unassigned_known_task_reasons": unassigned_reasons,
        },
    }
    return {"queues": queues, "receipt": receipt}


def normalize_initial_assignment(scenario):
    validate_scenario(scenario)
    snapshot = scenario["chapter1_snapshot"]
    handoff = {
        "schema": "relief-chapter-handoff/1",
        "source": snapshot["source"],
        "synthetic": True,
        "time": snapshot["time"],
        "units": copy.deepcopy(scenario["units"]),
        "map": copy.deepcopy(scenario["map"]),
        "platforms": copy.deepcopy(scenario["platforms"]),
        "depots": copy.deepcopy(scenario["depots"]),
        "known_tasks": [copy.deepcopy(task) for task in scenario["tasks"] if task["known_initial"]],
        "queues": copy.deepcopy(snapshot["queues"]),
    }
    if "assignment_receipt" in snapshot:
        handoff["assignment_receipt"] = copy.deepcopy(snapshot["assignment_receipt"])
    handoff["public_sha256"] = key_digest(handoff)
    return handoff
