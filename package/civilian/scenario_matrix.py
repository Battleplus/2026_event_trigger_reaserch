"""Deterministic, feasible N2 parent scenarios.

The generator is deliberately independent of prior N1 registries. It creates
new IDs and seeds, keeps all declared delivery quantities below available
carrier capacity and stock, and exposes a separate causal-case contract for
server-side replay. This module only builds/validates JSON-like inputs; it does
not run the environment.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

try:
    from .scenario_schema import C1_SNAPSHOT_SOURCE, MAP_CONTRACT, MAP_COLS, MAP_ROWS, GRID_CELLS, validate_scenario
    from .protocol_contract import PROTOCOL_ID
    from .chapter1_adapter import generate_public_c1_initial_assignment
except ImportError:
    from scenario_schema import C1_SNAPSHOT_SOURCE, MAP_CONTRACT, MAP_COLS, MAP_ROWS, GRID_CELLS, validate_scenario
    from protocol_contract import PROTOCOL_ID
    from chapter1_adapter import generate_public_c1_initial_assignment


HORIZON_MINUTES = 60
UAV_COUNT = 3
HELICOPTER_COUNT = 1
TRAIN_PARENTS = 2
DEVELOPMENT_PARENTS = 1
BASE_SEED = 8201
WINDOW_STARTS = (8, 16, 24, 32)
MAX_PUBLIC_CANDIDATES = 4
MAX_BRANCHES = 128


def _seed(split: str, index: int, condition: str) -> int:
    value = f"{PROTOCOL_ID}|{BASE_SEED}|{split}|{index}|{condition}".encode("ascii")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _pick(seed: int, name: str, low: int, high: int) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("ascii")).digest()
    return low + int.from_bytes(digest[:8], "big") % (high - low + 1)


def _platforms(seed: int) -> list[dict[str, Any]]:
    # The default local disturbance region is the central 2x2 block [40, 42).
    local_positions = [(40.25, 40.25), (40.75, 40.25), (40.25, 40.75)]
    platforms = []
    for index, (x, y) in enumerate(local_positions, start=1):
        capabilities = ["inspect"] if index <= 2 else ["inspect", "deliver"]
        platforms.append({
            "id": f"n02-uav-{index}",
            "type": "uav",
            "position": [float(x), float(y)],
            "local": True,
            "capabilities": capabilities,
            "capacity_kg": 8.0,
            "speed_km_min": 1.5,
            "energy_initial": 100.0,
            "energy_per_km": 1.0,
            "service_energy_per_min": 0.2,
        })
    platforms.append({
        "id": "n02-heli-1",
        "type": "helicopter",
        "position": [97.0 + _pick(seed, "heli-x", 0, 1), 97.0],
        "local": False,
        "capabilities": ["inspect", "deliver"],
        "capacity_kg": 30.0,
        "speed_km_min": 1.0,
        "energy_initial": 160.0,
        "energy_per_km": 1.2,
        "service_energy_per_min": 0.3,
    })
    return platforms


def _tasks(seed: int, condition: str) -> list[dict[str, Any]]:
    # The public delivery backlog is assigned through one local carrier;
    # inspection UAVs handle the delayed task-discovery chain.
    local_x = 40.5 + 0.25 * _pick(seed, "local-x", 0, 1)
    local_y = 40.5 + 0.25 * _pick(seed, "local-y", 0, 1)
    tasks = [
        {
            "id": "n02-local-delivery-a",
            "kind": "delivery",
            "position": [float(local_x), float(local_y)],
            "local": True,
            "known_initial": True,
            "deadline": 34,
            "priority": 5.0,
            "discovered_by": None,
            "report_delay": 0,
            "commodity": "water",
            "quantity": 4.0,
            "depot_id": "n02-local-depot",
            "load_minutes": 1,
            "unload_minutes": 1,
        },
        {
            "id": "n02-local-delivery-c",
            "kind": "delivery",
            "position": [float(local_x + 0.5), float(local_y + 0.25)],
            "local": True,
            "known_initial": True,
            "deadline": 42,
            "priority": 4.0,
            "discovered_by": None,
            "report_delay": 0,
            "commodity": "water",
            "quantity": 2.0,
            "depot_id": "n02-local-depot",
            "load_minutes": 1,
            "unload_minutes": 1,
        },
        {
            "id": "n02-local-delivery-b",
            "kind": "delivery",
            "position": [float(local_x + 0.25), float(local_y)],
            "local": True,
            "known_initial": True,
            "deadline": 38,
            "priority": 4.0,
            "discovered_by": None,
            "report_delay": 0,
            "commodity": "water",
            "quantity": 4.0,
            "depot_id": "n02-local-depot",
            "load_minutes": 1,
            "unload_minutes": 1,
        },
        {
            "id": "n02-local-inspection",
            "kind": "inspection",
            "position": [float(local_x), float(local_y + 0.25)],
            "local": True,
            "known_initial": True,
            "deadline": 30,
            "priority": 3.0,
            "discovered_by": None,
            "report_delay": 1,
            "service_minutes": 2,
        },
        {
            "id": "n02-remote-delivery",
            "kind": "delivery",
            "position": [99.0, 99.0],
            "local": False,
            "known_initial": True,
            "deadline": 46,
            "priority": 5.0,
            "discovered_by": None,
            "report_delay": 0,
            "commodity": "medical",
            "quantity": 6.0,
            "depot_id": "n02-remote-depot",
            "load_minutes": 1,
            "unload_minutes": 1,
        },
        {
            "id": "n02-hidden-inspection",
            "kind": "inspection",
            "position": [float(local_x + 0.5), float(local_y + 0.25)],
            "local": True,
            "known_initial": False,
            "deadline": 44,
            "priority": 2.0,
            "discovered_by": "n02-local-inspection",
            "report_delay": 5,
            "service_minutes": 6,
        },
        {
            "id": "n02-hidden-delivery",
            "kind": "delivery",
            "position": [float(local_x + 0.5), float(local_y + 0.5)],
            "local": True,
            "known_initial": False,
            "deadline": 54,
            "priority": 2.0,
            "discovered_by": "n02-hidden-inspection",
            "report_delay": 1,
            "commodity": "food",
            "quantity": 3.0,
            "depot_id": "n02-local-depot",
            "load_minutes": 1,
            "unload_minutes": 1,
        },
    ]
    previous_inspection = "n02-hidden-inspection"
    for stage in range(2, 5):
        inspection_id = f"n02-stage-inspection-{stage}"
        tasks.extend([
            {
                "id": inspection_id,
                "kind": "inspection",
                "position": [float(local_x + 0.1 * stage), float(local_y + 0.1 * stage)],
                "local": True,
                "known_initial": False,
                "deadline": 55,
                "priority": 3.0,
                "discovered_by": previous_inspection,
                "report_delay": 1,
                "service_minutes": 6,
            },
            {
                "id": f"n02-wave-delivery-{stage - 1}",
                "kind": "delivery",
                "position": [float(local_x + 0.1 * stage), float(local_y + 0.25)],
                "local": True,
                "known_initial": False,
                "deadline": min(58, 24 + 8 * (stage - 1)),
                "priority": 2.5,
                "discovered_by": previous_inspection,
                "report_delay": 1,
                "commodity": "food",
                "quantity": 3.0,
                "depot_id": "n02-local-depot",
                "load_minutes": 1,
                "unload_minutes": 1,
            },
        ])
        previous_inspection = inspection_id
    # A single delivery UAV must work through this public backlog while the
    # two inspection UAVs remain available for the delayed inspection chain.
    for index in range(3, 12):
        x = local_x + 0.25 * (index % 4)
        y = local_y + 0.25 * ((index // 4) % 4)
        tasks.append({
            "id": f"n02-local-delivery-{index:02d}",
            "kind": "delivery",
            "position": [float(x), float(y)],
            "local": True,
            "known_initial": True,
            "deadline": min(58, 18 + index * 3),
            "priority": max(1.5, 4.5 - 0.25 * (index - 3)),
            "discovered_by": None,
            "report_delay": 0,
            "commodity": "water",
            "quantity": 2.0,
            "depot_id": "n02-local-depot",
            "load_minutes": 1,
            "unload_minutes": 1,
        })
    if condition == "base":
        # A base parent leaves the shared carriers unconstrained at t=0.
        next(task for task in tasks if task["id"] == "n02-local-inspection")["priority"] = 2.5
    elif condition == "carrier_competition":
        # The queue gives one local UAV a first commitment while both delivery
        # orders remain compatible with the same local carrier pool.
        tasks[0]["priority"] = 5.0
        tasks[1]["priority"] = 5.0
    else:
        raise ValueError(f"unknown condition: {condition}")
    return tasks


def make_parent(split: str, index: int, condition: str) -> dict[str, Any]:
    if split not in {"train", "development"}:
        raise ValueError("N2_PARENT_SPLIT")
    if condition not in {"base", "carrier_competition"}:
        raise ValueError("N2_PARENT_CONDITION")
    seed = _seed(split, index, condition)
    platforms = _platforms(seed)
    tasks = _tasks(seed, condition)
    scenario = {
        "schema_version": "relief-engineering/1",
        "scenario_id": f"n02-{split}-{condition}-{index:02d}-{seed:016x}",
        "synthetic": True,
        "units": {"distance": "km", "time": "min", "quantity": "kg", "energy": "synthetic_unit"},
        "map": {
            "grid_contract": MAP_CONTRACT,
            "rows": MAP_ROWS,
            "cols": MAP_COLS,
            "grid_cells": GRID_CELLS,
            "cell_km": 1.0,
            "local_block": [40, 40, 2, 2],
        },
        "horizon": HORIZON_MINUTES,
        "dt": 1,
        "platforms": platforms,
        "depots": [
            {"id": "n02-local-depot", "position": [40.5, 40.5], "stock": {"water": 30.0, "food": 20.0}},
            {"id": "n02-remote-depot", "position": [97.0, 97.0], "stock": {"medical": 20.0}},
        ],
        "tasks": tasks,
        "chapter1_snapshot": {
            "source": C1_SNAPSHOT_SOURCE,
            "time": 0,
            "queues": {platform["id"]: [] for platform in platforms},
        },
        "disturbances": [
            {
                "id": f"n02-spatial-{seed:016x}",
                "time": 20 + _pick(seed, "disturbance", 0, 3),
                "kind": "spatial_perturbation",
                "report_delay": 1,
                "target_type": "task",
        "target_id": "n02-local-delivery-10",
                "position": [41.75, 41.75],
            },
            {
                "id": f"n02-energy-{seed:016x}",
                "time": 20 + _pick(seed, "disturbance", 0, 3),
                "kind": "energy_change",
                "report_delay": 1,
                "platform_id": "n02-uav-2",
                "value": 95.0,
            }
        ],
    }
    initial_assignment = generate_public_c1_initial_assignment(scenario, seed=seed)
    scenario["chapter1_snapshot"]["queues"] = initial_assignment["queues"]
    scenario["chapter1_snapshot"]["assignment_receipt"] = initial_assignment["receipt"]
    validate_feasibility(scenario, expected_condition=condition)
    return scenario


def build_parent_matrix() -> dict[str, list[dict[str, Any]]]:
    matrix = {
        "train": [make_parent("train", index, "base" if index == 0 else "carrier_competition") for index in range(TRAIN_PARENTS)],
        "development": [make_parent("development", index, "carrier_competition") for index in range(DEVELOPMENT_PARENTS)],
    }
    ids = [scenario["scenario_id"] for rows in matrix.values() for scenario in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("N2_PARENT_ID_COLLISION")
    return matrix


def _compatible(platform: dict[str, Any], task: dict[str, Any]) -> bool:
    needed = "inspect" if task["kind"] == "inspection" else "deliver"
    return (
        needed in platform["capabilities"]
        and bool(platform["local"]) == bool(task["local"])
        and (task["kind"] != "delivery" or float(task["quantity"]) <= float(platform["capacity_kg"]))
    )


def validate_feasibility(scenario: dict[str, Any], *, expected_condition: str | None = None) -> dict[str, Any]:
    validate_scenario(scenario)
    platforms = scenario.get("platforms", [])
    tasks = scenario.get("tasks", [])
    depots = {depot["id"]: depot for depot in scenario.get("depots", [])}
    if len(platforms) != 4 or sum(item.get("type") == "uav" for item in platforms) != UAV_COUNT or sum(item.get("type") == "helicopter" for item in platforms) != 1:
        raise ValueError("N2_PLATFORM_MATRIX")
    task_by_id = {task["id"]: task for task in tasks}
    platform_by_id = {platform["id"]: platform for platform in platforms}
    queues = scenario.get("chapter1_snapshot", {}).get("queues", {})
    queued = {task_id for values in queues.values() for task_id in values}
    if len(queued) != len([task_id for values in queues.values() for task_id in values]):
        raise ValueError("N2_DUPLICATE_INITIAL_ASSIGNMENT")
    for task in tasks:
        if task["kind"] == "delivery":
            if float(task["quantity"]) >= 16.0:
                raise ValueError("N2_OLD_16KG_VARIATION_FORBIDDEN")
            stock = float(depots[task["depot_id"]].get("stock", {}).get(task["commodity"], 0.0))
            if stock < float(task["quantity"]):
                raise ValueError(f"N2_INSUFFICIENT_STOCK:{task['id']}")
        if task["known_initial"] and not any(_compatible(platform, task) for platform in platforms):
            raise ValueError(f"N2_NO_COMPATIBLE_CARRIER:{task['id']}")
    for platform_id, task_ids in queues.items():
        platform = platform_by_id[platform_id]
        for task_id in task_ids:
            if task_id not in task_by_id or not task_by_id[task_id]["known_initial"] or not _compatible(platform, task_by_id[task_id]):
                raise ValueError(f"N2_ILLEGAL_INITIAL_QUEUE:{platform_id}:{task_id}")
    shared_eligible = 0
    known_tasks = [task for task in tasks if task["known_initial"]]
    for platform in platforms:
        shared_eligible += sum(_compatible(platform, task) for task in known_tasks) >= 2
    if shared_eligible < 1:
        raise ValueError("N2_CARRIER_COMPETITION_MISSING")
    if expected_condition == "carrier_competition":
        local_deliveries = [task for task in known_tasks if task["kind"] == "delivery" and task["local"]]
        if len(local_deliveries) < 2:
            raise ValueError("N2_LOCAL_COMPETITION_MISSING")
    return {
        "scenario_id": scenario.get("scenario_id"),
        "condition": expected_condition,
        "platform_count": len(platforms),
        "known_task_count": len(known_tasks),
        "compatible_known_task_count": sum(any(_compatible(platform, task) for platform in platforms) for task in known_tasks),
        "shared_carrier_platforms": int(shared_eligible),
        "all_delivery_stock_feasible": True,
        "max_delivery_quantity_kg": max((float(task["quantity"]) for task in tasks if task["kind"] == "delivery"), default=0.0),
    }


def manual_causal_case() -> dict[str, Any]:
    """Return a hand-authored allocation-effect contract for server replay."""
    scenario = make_parent("development", 99, "carrier_competition")
    scenario["platforms"][0]["capabilities"] = ["inspect"]
    scenario["platforms"][2]["capabilities"] = ["inspect"]
    scenario["tasks"] = [
        task for task in scenario["tasks"]
        if not (task["kind"] == "delivery" and task["local"])
    ]
    for disturbance in scenario["disturbances"]:
        if disturbance["kind"] == "spatial_perturbation":
            disturbance["target_id"] = "n02-local-inspection"
    scenario["tasks"].extend([
        {
            "id": "n02-causal-delivery-a", "kind": "delivery", "position": [98.0, 97.5],
            "local": False, "known_initial": True, "deadline": 15, "priority": 5.0,
            "discovered_by": None, "report_delay": 0, "commodity": "medical", "quantity": 6.0,
            "depot_id": "n02-remote-depot", "load_minutes": 1, "unload_minutes": 1,
        },
        {
            "id": "n02-causal-delivery-b", "kind": "delivery", "position": [97.5, 98.0],
            "local": False, "known_initial": True, "deadline": 19, "priority": 5.0,
            "discovered_by": None, "report_delay": 0, "commodity": "medical", "quantity": 6.0,
            "depot_id": "n02-remote-depot", "load_minutes": 1, "unload_minutes": 1,
        },
    ])
    scenario["chapter1_snapshot"]["queues"] = {platform["id"]: [] for platform in scenario["platforms"]}
    scenario["chapter1_snapshot"]["queues"]["n02-heli-1"] = ["n02-causal-delivery-a", "n02-causal-delivery-b"]
    scenario["chapter1_snapshot"]["source"] = "synthetic_test_fixture"
    scenario["chapter1_snapshot"].pop("assignment_receipt", None)
    validate_feasibility(scenario)
    return {
        "schema": "n2-causal-allocation-case/1",
        "scenario_id": scenario["scenario_id"],
        "decision_time": 8,
        "shared_platform": "n02-heli-1",
        "candidate_a": ("dispatch:n02-heli-1:n02-causal-delivery-a", "NOOP"),
        "candidate_b": ("dispatch:n02-heli-1:n02-causal-delivery-b", "NOOP"),
        "effect_requirement": {
            "same_event_tape": True,
            "same_public_prefix": True,
            "later_other_order_outcome_may_differ": True,
            "must_record_actual_label_delta": True,
        },
        "scenario": copy.deepcopy(scenario),
    }
