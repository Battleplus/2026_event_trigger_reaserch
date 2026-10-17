"""Deterministic six-parent scenarios for the executable event-trigger study."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

try:
    from .chapter1_adapter import generate_public_c1_initial_assignment
    from .scenario_schema import C1_SNAPSHOT_SOURCE, GRID_CELLS, MAP_COLS, MAP_CONTRACT, MAP_ROWS, validate_scenario
except ImportError:
    from chapter1_adapter import generate_public_c1_initial_assignment
    from scenario_schema import C1_SNAPSHOT_SOURCE, GRID_CELLS, MAP_COLS, MAP_CONTRACT, MAP_ROWS, validate_scenario


BASE_SEED = 8201
HORIZON = 60
TRAIN_CONDITIONS = ("new_arrival", "platform_failure", "energy_drop", "communication_drop")
DEVELOPMENT_CONDITIONS = ("environment_perturbation", "mixed_events")


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _platforms() -> list[dict]:
    local = [(40.2, 40.2), (41.2, 40.2), (40.2, 41.2), (41.2, 41.2)]
    result = []
    for index, position in enumerate(local, 1):
        result.append({
            "id": f"n02-uav-{index}", "type": "uav", "position": list(position), "local": True,
            "capabilities": ["inspect", "deliver"], "capacity_kg": 9.0, "speed_km_min": 1.6,
            "energy_initial": 100.0, "energy_per_km": 1.0, "service_energy_per_min": 0.2,
        })
    for index, position in enumerate(((95.0, 96.0), (97.0, 95.0)), 1):
        result.append({
            "id": f"n02-heli-{index}", "type": "helicopter", "position": list(position), "local": False,
            "capabilities": ["inspect", "deliver"], "capacity_kg": 30.0, "speed_km_min": 2.2,
            "energy_initial": 180.0, "energy_per_km": 1.1, "service_energy_per_min": 0.3,
        })
    return result


def _tasks(parent_index: int) -> list[dict]:
    local_positions = [(40.3, 40.3), (41.4, 40.3), (40.4, 41.4), (41.5, 41.5), (40.8, 40.6), (41.1, 41.1), (40.6, 41.2), (41.6, 40.7)]
    tasks = []
    for index in range(1, 9):
        delivery = index not in {3, 8}
        local = index <= 6
        position = list(local_positions[index - 1]) if local else ([96.0, 95.0] if index == 7 else [95.5, 96.5])
        task = {
            "id": f"n02-task-{index:02d}", "kind": "delivery" if delivery else "inspection",
            "position": position, "local": local, "known_initial": True, "deadline": 30 + index * 3,
            "priority": float(6 - index * 0.35), "discovered_by": None, "report_delay": 0,
        }
        if delivery:
            task.update({"commodity": "water" if local else "medical", "quantity": 2.0 + (index % 3),
                         "depot_id": "n02-local-depot" if local else "n02-remote-depot",
                         "load_minutes": 1, "unload_minutes": 1})
        else:
            task.update({"service_minutes": 2 + index % 2})
        tasks.append(task)
    late_specs = [
        (9, "delivery", [40.7, 40.8], True, "water", 2.0, "n02-local-depot"),
        (10, "inspection", [41.3, 40.8], True, None, None, None),
        (11, "delivery", [96.5, 96.0], False, "medical", 5.0, "n02-remote-depot"),
        (12, "delivery", [41.0, 40.5], True, "water", 3.0, "n02-local-depot"),
    ]
    for index, kind, position, local, commodity, quantity, depot in late_specs:
        task = {
            "id": f"n02-task-{index:02d}", "kind": kind, "position": position, "local": local,
            "known_initial": False, "deadline": 44 + index, "priority": 4.0 - 0.1 * index,
            "discovered_by": None, "report_delay": 0,
        }
        if kind == "delivery":
            task.update({"commodity": commodity, "quantity": quantity, "depot_id": depot,
                         "load_minutes": 1, "unload_minutes": 1})
        else:
            task["service_minutes"] = 3
        tasks.append(task)
    return tasks


def _event_tape(condition: str, parent_index: int) -> list[dict]:
    prefix = f"n02-{condition}-{parent_index}"
    arrivals = [
        {"time": 8, "kind": "task_arrival", "task_id": "n02-task-09"},
        {"time": 16, "kind": "task_arrival", "task_id": "n02-task-10"},
        {"time": 24, "kind": "task_arrival", "task_id": "n02-task-11"},
        {"time": 32, "kind": "task_arrival", "task_id": "n02-task-12"},
    ]
    if condition == "new_arrival":
        rows = arrivals
    elif condition == "platform_failure":
        rows = arrivals + [{"time": 10, "kind": "platform_failure", "platform_id": "n02-uav-2"}]
    elif condition == "energy_drop":
        rows = arrivals + [{"time": 12, "kind": "energy_drop", "platform_id": "n02-uav-3", "new_energy": 6.0}]
    elif condition == "communication_drop":
        rows = arrivals + [{"time": 14, "kind": "communication_drop", "links": [["n02-uav-4", "n02-heli-1"]], "quality": 0.05}]
    elif condition == "environment_perturbation":
        rows = arrivals + [{"time": 16, "kind": "environment_perturbation", "patch": [40, 40, 2, 2],
                            "wind_vector": [1.2, -0.8], "risk_delta": 0.55, "blocked_cell": [41, 41]}]
    elif condition == "mixed_events":
        rows = arrivals + [
            {"time": 11, "kind": "platform_failure", "platform_id": "n02-uav-2"},
            {"time": 11, "kind": "communication_drop", "links": [["n02-uav-4", "n02-heli-1"]], "quality": 0.0},
            {"time": 11, "kind": "environment_perturbation", "patch": [40, 40, 2, 2],
             "wind_vector": [0.9, 1.0], "risk_delta": 0.4, "blocked_cell": [40, 41]},
        ]
    else:
        raise ValueError(f"unknown condition {condition!r}")
    for ordinal, row in enumerate(rows):
        row["event_id"] = f"{prefix}-{ordinal:02d}"
    return rows


def _base_scenario(parent_id: str, seed: int) -> dict:
    platforms = _platforms()
    tasks = _tasks(seed)
    scenario = {
        "schema_version": "relief-engineering/1", "scenario_id": parent_id, "synthetic": True,
        "units": {"distance": "km", "time": "min", "quantity": "kg", "energy": "synthetic_unit"},
        "map": {"grid_contract": MAP_CONTRACT, "rows": MAP_ROWS, "cols": MAP_COLS,
                "grid_cells": GRID_CELLS, "cell_km": 1.0, "local_block": [40, 40, 2, 2]},
        "horizon": HORIZON, "dt": 1, "platforms": platforms,
        "depots": [
            {"id": "n02-local-depot", "position": [40.5, 40.5], "stock": {"water": 45.0}},
            {"id": "n02-remote-depot", "position": [96.0, 96.0], "stock": {"medical": 20.0}},
        ],
        "tasks": tasks,
        "chapter1_snapshot": {"source": C1_SNAPSHOT_SOURCE, "time": 0,
                              "queues": {p["id"]: [] for p in platforms}},
        "disturbances": [],
    }
    assignment = generate_public_c1_initial_assignment(scenario, seed=seed)
    scenario["chapter1_snapshot"]["queues"] = assignment["queues"]
    scenario["chapter1_snapshot"]["assignment_receipt"] = assignment["receipt"]
    validate_scenario(scenario)
    return scenario


def validate_event_scenario(scenario: dict) -> dict:
    """Validate event wrapper identity, baseline map, workload, and tape."""
    if not isinstance(scenario, dict):
        raise ValueError("event scenario must be an object")
    required = {"parent_id", "split", "condition", "seed_variant", "map", "base_scenario", "queue_plan", "event_tape", "event_tape_id"}
    if set(scenario) != required:
        raise ValueError("event scenario fields do not match the event contract")
    if scenario["split"] not in {"train", "development"}:
        raise ValueError("event scenario split must be train or development")
    if scenario["map"] != {"grid_contract": MAP_CONTRACT, "rows": 100, "cols": 100, "grid_cells": 10000, "cell_km": 1.0, "local_block": [40, 40, 2, 2]}:
        raise ValueError("event scenario map contract mismatch")
    base = scenario["base_scenario"]
    validate_scenario(base)
    if base["map"] != scenario["map"] or len(base["platforms"]) != 6 or len(base["tasks"]) != 12:
        raise ValueError("event base scenario must have the 100x100 map, six platforms, and twelve tasks")
    if sum(platform["type"] == "uav" for platform in base["platforms"]) != 4 or sum(platform["type"] == "helicopter" for platform in base["platforms"]) != 2:
        raise ValueError("event base platform mix must be 4 UAV and 2 helicopter")
    queue_plan = scenario["queue_plan"]
    if set(queue_plan) != {platform["id"] for platform in base["platforms"]}:
        raise ValueError("queue plan must name all six platforms")
    known_ids = {task["id"] for task in base["tasks"] if task["known_initial"]}
    queued = [task_id for queue in queue_plan.values() for task_id in queue]
    if set(queued) != known_ids or len(queued) != len(set(queued)):
        raise ValueError("queue plan must assign every initial public task exactly once")
    ids = [row.get("event_id") for row in scenario["event_tape"]]
    if any(not isinstance(row, dict) for row in scenario["event_tape"]) or len(ids) != len(set(ids)):
        raise ValueError("event tape must contain uniquely identified events")
    task_ids = {task["id"] for task in base["tasks"]}
    platform_ids = {platform["id"] for platform in base["platforms"]}
    for row in scenario["event_tape"]:
        if type(row.get("time")) is not int or not 0 < row["time"] < base["horizon"]:
            raise ValueError("event tape times must be within the episode")
        kind = row.get("kind")
        if kind == "task_arrival" and (row.get("task_id") not in task_ids or next(t for t in base["tasks"] if t["id"] == row["task_id"])["known_initial"]):
            raise ValueError("task arrival must refer to a hidden task")
        if kind in {"platform_failure", "energy_drop", "energy_fraction_drop", "capability_loss", "load_change"} and row.get("platform_id") not in platform_ids:
            raise ValueError("platform event references unknown platform")
        if kind == "energy_fraction_drop":
            fraction = row.get("loss_fraction")
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
                raise ValueError("energy fraction event loss_fraction must be numeric")
            if not 0.0 < float(fraction) <= 1.0:
                raise ValueError("energy fraction event loss_fraction must be in (0, 1]")
        if kind in {"task_cancel", "priority_change", "deadline_change"} and row.get("task_id") not in task_ids:
            raise ValueError("task event references unknown task")
        if kind in {"environment_perturbation", "wind_change", "obstacle_change", "threat_change", "passability_change"} and row.get("patch", [0, 0, 0, 0])[2:] != [2, 2]:
            raise ValueError("environment perturbation must be a 2x2 patch")
        if kind in {"communication_drop", "packet_loss", "communication_delay", "communication_split"}:
            if not isinstance(row.get("links"), list) or not row["links"]:
                raise ValueError("communication event must identify affected links")
        if kind not in {
            "task_arrival", "platform_failure", "energy_drop", "energy_fraction_drop", "communication_drop", "environment_perturbation",
            "task_cancel", "priority_change", "deadline_change", "capability_loss", "load_change",
            "wind_change", "obstacle_change", "threat_change", "passability_change",
            "packet_loss", "communication_delay", "communication_split",
        }:
            raise ValueError(f"unsupported event kind {kind!r}")
    expected_tape_id = _digest(scenario["event_tape"])
    if scenario["event_tape_id"] != expected_tape_id:
        raise ValueError("event tape identity mismatch")
    return {"parent_id": scenario["parent_id"], "split": scenario["split"], "platform_count": 6,
            "task_count": 12, "event_count": len(ids), "event_tape_id": expected_tape_id}


def build_event_scenarios() -> list[dict]:
    """Build four training and two development parent scenarios."""
    result = []
    rows = [("train", condition) for condition in TRAIN_CONDITIONS]
    rows.extend(("development", condition) for condition in DEVELOPMENT_CONDITIONS)
    for index, (split, condition) in enumerate(rows):
        seed_variant = BASE_SEED + index
        parent_id = f"n02-event-{split}-{index + 1:02d}-{condition}"
        tape = _event_tape(condition, index + 1)
        scenario = {
            "parent_id": parent_id, "split": split, "condition": condition, "seed_variant": seed_variant,
            "map": {"grid_contract": MAP_CONTRACT, "rows": 100, "cols": 100, "grid_cells": 10000,
                    "cell_km": 1.0, "local_block": [40, 40, 2, 2]},
            "base_scenario": _base_scenario(parent_id, seed_variant), "event_tape": tape,
            "event_tape_id": _digest(tape),
        }
        scenario["queue_plan"] = copy.deepcopy(scenario["base_scenario"]["chapter1_snapshot"]["queues"])
        scenario["queue_plan"]["n02-uav-3"].append("n02-task-05")
        scenario["queue_plan"]["n02-uav-4"].append("n02-task-06")
        validate_event_scenario(scenario)
        result.append(scenario)
    return result
