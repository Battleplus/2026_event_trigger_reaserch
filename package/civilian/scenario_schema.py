"""Validation and loading for the synthetic disaster-relief schema.

The validator deliberately does not normalize or mutate its input.  The
engine makes its own copy after validation and supplies defaults for optional
kind-specific fields.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    from .stable_identity import key_digest
except ImportError:
    from stable_identity import key_digest


SCHEMA_VERSION = "relief-engineering/1"
MAP_CONTRACT = "relief-grid-100x100-v1"
MAP_ROWS = 100
MAP_COLS = 100
GRID_CELLS = MAP_ROWS * MAP_COLS
LOCAL_REGION_SIZES = frozenset({2, 4, 8})
C1_SNAPSHOT_SOURCE = "civilian-public-C1/1"
C1_ALGORITHM_ID = "civilian-public-C1-priority-deadline-completion-v1"
TEST_SNAPSHOT_SOURCE = "synthetic_test_fixture"
C1_MULTIROUND_ADAPTER_SCHEMA = "gain-c1-multiround-adapter/1"


def _fail(message: str) -> None:
    raise ValueError(message)


def _dict(value: Any, path: str) -> dict:
    if type(value) is not dict:
        _fail(f"{path} must be an object")
    return value


def _known_keys(value: dict, path: str, allowed: set[str]) -> None:
    if any(type(key) is not str for key in value):
        _fail(f"{path} field names must be strings")
    unknown = set(value) - allowed
    if unknown:
        _fail(f"{path} contains unknown fields: {sorted(unknown)}")


def _list(value: Any, path: str) -> list:
    if type(value) is not list:
        _fail(f"{path} must be an array")
    return value


def _string(value: Any, path: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value.strip()):
        _fail(f"{path} must be a non-empty string")
    return value


def _bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        _fail(f"{path} must be boolean")
    return value


def _number(value: Any, path: str, *, minimum: float | None = None, positive: bool = False) -> float:
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)):
        _fail(f"{path} must be finite numeric")
    result = float(value)
    if positive and result <= 0:
        _fail(f"{path} must be positive")
    if minimum is not None and result < minimum:
        _fail(f"{path} must be >= {minimum}")
    return result


def _integer(value: Any, path: str, *, minimum: int | None = None, positive: bool = False) -> int:
    if type(value) is not int or isinstance(value, bool):
        _fail(f"{path} must be an integer")
    if positive and value <= 0:
        _fail(f"{path} must be positive")
    if minimum is not None and value < minimum:
        _fail(f"{path} must be >= {minimum}")
    return value


def _position(value: Any, path: str) -> None:
    values = _list(value, path)
    if len(values) != 2:
        _fail(f"{path} must have two coordinates")
    _number(values[0], path + "[0]")
    _number(values[1], path + "[1]")


def validate_map_contract(map_data: Any) -> None:
    data = _dict(map_data, "map")
    if data.get("grid_contract") != MAP_CONTRACT:
        _fail(f"map.grid_contract must be {MAP_CONTRACT!r}")
    if data.get("rows") != MAP_ROWS or data.get("cols") != MAP_COLS:
        _fail("map dimensions must be exactly 100x100")
    if type(data.get("grid_cells")) is not int or data["grid_cells"] != GRID_CELLS:
        _fail("map.grid_cells must be exactly 10000")
    if data.get("cell_km") != 1.0:
        _fail("map.cell_km must be 1.0 for global cell coordinates")


def _position_in_map(value: Any, path: str) -> None:
    _position(value, path)
    x, y = map(float, value)
    if not 0.0 <= x <= MAP_COLS - 1 or not 0.0 <= y <= MAP_ROWS - 1:
        _fail(f"{path} must lie within global x/y coordinates [0, 99]")


def _inside_local_block(position: list[float], map_data: dict) -> bool:
    x, y = map(float, position)
    local_x, local_y, width, height = map_data["local_block"]
    return local_x <= x < local_x + width and local_y <= y < local_y + height


def _unique_ids(items: list[dict], path: str) -> set[str]:
    ids: set[str] = set()
    for index, item in enumerate(items):
        current = _dict(item, f"{path}[{index}]")
        ident = _string(current.get("id"), f"{path}[{index}].id")
        if ident in ids:
            _fail(f"duplicate id {ident!r} in {path}")
        ids.add(ident)
    return ids


def _mission_projection(platform: dict, task: dict, depots: dict[str, dict]) -> tuple[float, float]:
    speed = max(1e-9, float(platform["speed_km_min"]))
    energy_rate = max(0.0, float(platform["energy_per_km"]))
    service_rate = max(0.0, float(platform["service_energy_per_min"]))
    position = list(platform["position"])
    target = list(task["position"])
    if task["kind"] == "inspection":
        distance = math.hypot(position[0] - target[0], position[1] - target[1])
        service = float(task["service_minutes"])
        duration = math.ceil(distance / speed - 1e-9) + service
    else:
        depot = depots[str(task["depot_id"])]
        depot_position = list(depot["position"])
        to_depot = math.hypot(position[0] - depot_position[0], position[1] - depot_position[1])
        to_task = math.hypot(depot_position[0] - target[0], depot_position[1] - target[1])
        distance = to_depot + to_task
        service = float(task["load_minutes"]) + float(task["unload_minutes"])
        duration = math.ceil(to_depot / speed - 1e-9) + float(task["load_minutes"])
        duration += math.ceil(to_task / speed - 1e-9) + float(task["unload_minutes"])
    energy = distance * energy_rate + service * service_rate
    return float(duration), float(energy)


def _validate_c1_multiround_adapter(
    scenario: dict,
    queues: dict,
    receipt: dict,
    platforms: list[dict],
    depots: list[dict],
    tasks: list[dict],
    map_data: dict,
) -> None:
    path = "chapter1_snapshot.assignment_receipt"
    _known_keys(receipt, path, {
        "schema", "adapter_source", "algorithm_id", "seed", "seed_usage", "known_task_ids",
        "round_count", "rounds", "assignments", "selected_action_ids", "assignment_output_sha256",
        "feasibility",
    })
    if (
        receipt.get("schema") != C1_MULTIROUND_ADAPTER_SCHEMA
        or receipt.get("adapter_source") != "package/civilian/gain_scenarios.py:_multi_round_c1_assignment"
        or receipt.get("algorithm_id") != C1_ALGORITHM_ID
        or type(receipt.get("seed")) is not int
        or receipt["seed"] < 0
        or receipt.get("seed_usage") != "recorded_for_reproducibility; deterministic C1 uses no random draws"
    ):
        _fail("chapter1_snapshot multi-round C1 adapter identity invalid")

    known_task_ids = sorted(task["id"] for task in tasks if task["known_initial"])
    if receipt.get("known_task_ids") != known_task_ids:
        _fail("chapter1_snapshot multi-round C1 known task boundary mismatch")
    if receipt.get("assignments") != queues or receipt.get("assignment_output_sha256") != key_digest(queues):
        _fail("chapter1_snapshot multi-round C1 output does not match the queues")

    selected_action_ids = receipt.get("selected_action_ids")
    expected_action_ids = {
        f"dispatch:{platform_id}:{task_id}"
        for platform_id, queue in queues.items()
        for task_id in queue
    }
    if (
        type(selected_action_ids) is not list
        or len(selected_action_ids) != len(expected_action_ids)
        or set(selected_action_ids) != expected_action_ids
    ):
        _fail("chapter1_snapshot multi-round C1 selected action identity mismatch")

    rounds = _list(receipt.get("rounds"), path + ".rounds")
    _integer(receipt.get("round_count"), path + ".round_count", minimum=0)
    if receipt["round_count"] != len(rounds):
        _fail("chapter1_snapshot multi-round C1 round count mismatch")

    platform_state = {row["id"]: {**row, "position": list(row["position"])} for row in platforms}
    platform_time = {platform_id: 0.0 for platform_id in platform_state}
    depot_state = {row["id"]: {**row, "stock": dict(row["stock"])} for row in depots}
    tasks_by_id = {row["id"]: row for row in tasks}
    remaining = set(known_task_ids)
    cumulative_energy = {platform_id: 0.0 for platform_id in platform_state}
    cumulative_stock: dict[tuple[str, str], float] = {}
    task_projection: dict[str, dict] = {}
    final_unassigned_reasons: dict[str, list[str]] = {}

    try:
        from .chapter1_adapter import generate_public_c1_initial_assignment
    except ImportError:
        from chapter1_adapter import generate_public_c1_initial_assignment

    for index, row in enumerate(rounds):
        round_path = f"{path}.rounds[{index}]"
        round_row = _dict(row, round_path)
        _known_keys(round_row, round_path, {"round_index", "known_task_ids", "public_input", "c1_receipt"})
        if round_row.get("round_index") != index or round_row.get("known_task_ids") != sorted(remaining):
            _fail(f"{round_path} round index or known task boundary mismatch")

        current_platforms = [platform_state[row["id"]] for row in platforms]
        current_depots = [depot_state[row["id"]] for row in depots]
        expected_input = {
            "schema": "civilian-public-c1-input/1",
            "seed": receipt["seed"],
            "map": map_data,
            "platforms": current_platforms,
            "depots": current_depots,
            "known_tasks": [
                {**task, "known_initial": task["id"] in remaining}
                for task in tasks if task["id"] in remaining
            ],
        }
        public_input = _dict(round_row.get("public_input"), round_path + ".public_input")
        if public_input != expected_input:
            _fail(f"{round_path} public input does not match the cumulative public state")

        replay_scenario = {**scenario, "map": map_data, "platforms": current_platforms,
                           "depots": current_depots}
        replay_scenario["tasks"] = [
            {**task, "known_initial": task["id"] in remaining}
            for task in tasks
        ]
        replay_scenario["chapter1_snapshot"] = {
            "source": "synthetic_initial_assignment", "time": 0,
            "queues": {row["id"]: [] for row in platforms},
        }
        expected_assignment = generate_public_c1_initial_assignment(replay_scenario, seed=receipt["seed"])
        raw_receipt = _dict(round_row.get("c1_receipt"), round_path + ".c1_receipt")
        if raw_receipt != expected_assignment["receipt"]:
            _fail(f"{round_path} raw C1 receipt does not match its saved public input")

        selected = {task_id for queue in expected_assignment["queues"].values() for task_id in queue}
        if not selected:
            final_unassigned_reasons = raw_receipt["feasibility"]["unassigned_known_task_reasons"]
            if index != len(rounds) - 1:
                _fail("chapter1_snapshot multi-round C1 contains rounds after an empty assignment")
            break
        if not selected <= remaining:
            _fail(f"{round_path} selected a task outside its public known-task set")

        round_platforms = {row["id"]: row for row in current_platforms}
        round_depots = {row["id"]: row for row in current_depots}
        for platform_id in sorted(expected_assignment["queues"]):
            for task_id in expected_assignment["queues"][platform_id]:
                task = tasks_by_id[task_id]
                platform = round_platforms[platform_id]
                duration, energy = _mission_projection(platform, task, round_depots)
                energy_before = float(platform_state[platform_id]["energy_initial"])
                energy_after = energy_before - energy
                if energy_after < -1e-8:
                    _fail(f"{round_path} exceeds cumulative platform energy")
                completion_time = platform_time[platform_id] + duration
                task_projection[task_id] = {
                    "platform_id": platform_id, "duration_minutes": duration,
                    "completion_time_minutes": completion_time,
                    "deadline": float(task["deadline"]),
                    "deadline_lateness_minutes": max(0.0, completion_time - float(task["deadline"])),
                    "energy_before": energy_before, "energy_used": energy,
                    "energy_after": max(0.0, energy_after),
                }
                platform_time[platform_id] = completion_time
                platform_state[platform_id]["position"] = list(task["position"])
                platform_state[platform_id]["energy_initial"] = max(0.0, energy_after)
                cumulative_energy[platform_id] += energy
                if task["kind"] == "delivery":
                    depot_id, commodity = str(task["depot_id"]), str(task["commodity"])
                    quantity = float(task["quantity"])
                    available = float(depot_state[depot_id]["stock"].get(commodity, 0.0))
                    if quantity > available + 1e-8:
                        _fail(f"{round_path} exceeds cumulative depot stock")
                    depot_state[depot_id]["stock"][commodity] = max(0.0, available - quantity)
                    cumulative_stock[(depot_id, commodity)] = cumulative_stock.get((depot_id, commodity), 0.0) + quantity
        remaining -= selected
        if not remaining and index != len(rounds) - 1:
            _fail("chapter1_snapshot multi-round C1 contains trailing rounds after all tasks were assigned")

    assigned = set(known_task_ids) - remaining
    if remaining and not final_unassigned_reasons and rounds:
        _fail("chapter1_snapshot multi-round C1 did not record unassigned task reasons")
    feasibility = _dict(receipt.get("feasibility"), path + ".feasibility")
    _known_keys(feasibility, path + ".feasibility", {
        "all_selected_actions_legal", "known_task_count", "assigned_task_count",
        "unassigned_known_task_reasons", "cumulative_energy_feasible", "cumulative_inventory_feasible",
        "platform_energy_remaining", "platform_completion_time_minutes", "total_platform_energy_used",
        "total_depot_stock_used", "task_projection", "soft_deadline_semantics",
    })
    reasons = feasibility.get("unassigned_known_task_reasons")
    expected_unassigned = set(known_task_ids) - assigned
    expected_energy_remaining = {
        platform_id: float(platform_state[platform_id]["energy_initial"])
        for platform_id in sorted(platform_state)
    }
    expected_stock_used = {
        f"{depot_id}|{commodity}": float(quantity)
        for (depot_id, commodity), quantity in sorted(cumulative_stock.items())
    }
    if (
        feasibility.get("all_selected_actions_legal") is not True
        or feasibility.get("known_task_count") != len(known_task_ids)
        or feasibility.get("assigned_task_count") != len(assigned)
        or type(reasons) is not dict
        or set(reasons) != expected_unassigned
        or any(type(values) is not list or not values or any(type(value) is not str for value in values) for values in reasons.values())
        or feasibility.get("cumulative_energy_feasible") is not True
        or feasibility.get("cumulative_inventory_feasible") is not True
        or feasibility.get("platform_energy_remaining") != expected_energy_remaining
        or feasibility.get("platform_completion_time_minutes") != {
            platform_id: float(platform_time[platform_id]) for platform_id in sorted(platform_time)
        }
        or feasibility.get("total_platform_energy_used") != {
            platform_id: float(cumulative_energy[platform_id]) for platform_id in sorted(cumulative_energy)
        }
        or feasibility.get("total_depot_stock_used") != expected_stock_used
        or feasibility.get("task_projection") != task_projection
        or feasibility.get("soft_deadline_semantics") != "lateness_is_reported_and_does_not_make_a_C1_action_illegal"
    ):
        _fail("chapter1_snapshot multi-round C1 feasibility receipt invalid")


def validate_scenario(scenario: dict) -> None:
    """Validate a schema-v1 scenario and raise ``ValueError`` on failure."""

    root = _dict(scenario, "scenario")
    _known_keys(root, "scenario", {
        "schema_version", "scenario_id", "synthetic", "units", "map", "horizon", "dt",
        "platforms", "depots", "tasks", "chapter1_snapshot", "disturbances",
    })
    if root.get("schema_version") != SCHEMA_VERSION:
        _fail(f"schema_version must be {SCHEMA_VERSION!r}")
    _string(root.get("scenario_id"), "scenario_id")
    if root.get("synthetic") is not True:
        _fail("synthetic must be true")

    units = _dict(root.get("units"), "units")
    _known_keys(units, "units", {"distance", "time", "quantity", "energy"})
    for key, expected in (("distance", "km"), ("time", "min"), ("quantity", "kg"), ("energy", "synthetic_unit")):
        if units.get(key) != expected:
            _fail(f"units.{key} must be {expected!r}")

    map_data = _dict(root.get("map"), "map")
    _known_keys(map_data, "map", {"grid_contract", "rows", "cols", "grid_cells", "cell_km", "local_block"})
    _integer(map_data.get("rows"), "map.rows", positive=True)
    _integer(map_data.get("cols"), "map.cols", positive=True)
    _number(map_data.get("cell_km"), "map.cell_km", positive=True)
    validate_map_contract(map_data)
    block = _list(map_data.get("local_block"), "map.local_block")
    if len(block) != 4:
        _fail("map.local_block must be [x, y, width, height]")
    for i, value in enumerate(block):
        _integer(value, f"map.local_block[{i}]")
    if block[2] != block[3] or block[2] not in LOCAL_REGION_SIZES:
        _fail("map.local_block must be a 2x2, 4x4, or 8x8 region")
    if block[0] < 0 or block[1] < 0 or block[0] + block[2] > map_data["cols"] or block[1] + block[3] > map_data["rows"]:
        _fail("map.local_block outside map")

    horizon = _integer(root.get("horizon"), "horizon", positive=True)
    if root.get("dt") != 1:
        _fail("dt must be 1")

    platforms = _list(root.get("platforms"), "platforms")
    platform_ids = _unique_ids(platforms, "platforms")
    allowed_types = {"uav", "helicopter"}
    allowed_caps = {"inspect", "deliver"}
    for i, platform in enumerate(platforms):
        path = f"platforms[{i}]"
        _known_keys(platform, path, {
            "id", "type", "position", "local", "capabilities", "capacity_kg", "speed_km_min",
            "energy_initial", "energy_per_km", "service_energy_per_min",
        })
        _position_in_map(platform.get("position"), path + ".position")
        if platform.get("type") not in allowed_types:
            _fail(f"{path}.type must be uav or helicopter")
        _bool(platform.get("local"), path + ".local")
        if platform["local"] != _inside_local_block(platform["position"], map_data):
            _fail(f"{path}.local does not match its position in map.local_block")
        caps = _list(platform.get("capabilities"), path + ".capabilities")
        if not caps or any(cap not in allowed_caps for cap in caps) or len(set(caps)) != len(caps):
            _fail(f"{path}.capabilities contains an unknown or duplicate capability")
        _number(platform.get("capacity_kg"), path + ".capacity_kg", minimum=0)
        _number(platform.get("speed_km_min"), path + ".speed_km_min", positive=True)
        _number(platform.get("energy_initial"), path + ".energy_initial", minimum=0)
        _number(platform.get("energy_per_km"), path + ".energy_per_km", minimum=0)
        _number(platform.get("service_energy_per_min"), path + ".service_energy_per_min", minimum=0)

    depots = _list(root.get("depots"), "depots")
    depot_ids = _unique_ids(depots, "depots")
    for i, depot in enumerate(depots):
        path = f"depots[{i}]"
        _known_keys(depot, path, {"id", "position", "stock"})
        _position_in_map(depot.get("position"), path + ".position")
        stock = _dict(depot.get("stock"), path + ".stock")
        for commodity, quantity in stock.items():
            _string(commodity, path + ".stock key")
            _number(quantity, path + ".stock." + commodity, minimum=0)

    tasks = _list(root.get("tasks"), "tasks")
    task_ids = _unique_ids(tasks, "tasks")
    task_by_id = {task["id"]: task for task in tasks}
    for i, task in enumerate(tasks):
        path = f"tasks[{i}]"
        _known_keys(task, path, {
            "id", "kind", "position", "local", "known_initial", "deadline", "priority",
            "discovered_by", "report_delay", "service_minutes", "commodity", "quantity",
            "depot_id", "load_minutes", "unload_minutes",
        })
        kind = task.get("kind")
        if kind not in {"inspection", "delivery"}:
            _fail(f"{path}.kind must be inspection or delivery")
        _position_in_map(task.get("position"), path + ".position")
        _bool(task.get("local"), path + ".local")
        task_x, task_y = task["position"]
        cell_km = float(map_data["cell_km"])
        local_x, local_y, local_width, local_height = block
        in_local_block = (
            local_x * cell_km <= task_x < (local_x + local_width) * cell_km
            and local_y * cell_km <= task_y < (local_y + local_height) * cell_km
        )
        if task["local"] != in_local_block:
            _fail(f"{path}.local does not match its position in map.local_block")
        _bool(task.get("known_initial"), path + ".known_initial")
        deadline = _integer(task.get("deadline"), path + ".deadline", positive=True)
        if deadline > horizon:
            _fail(f"{path}.deadline exceeds horizon")
        _number(task.get("priority"), path + ".priority", positive=True)
        discovered = task.get("discovered_by")
        if discovered is not None and discovered not in task_ids:
            _fail(f"{path}.discovered_by references an unknown task")
        if discovered is not None and task_by_id[discovered].get("kind") != "inspection":
            _fail(f"{path}.discovered_by must reference an inspection")
        if task.get("known_initial") and discovered is not None:
            _fail(f"{path}.known_initial cannot be true with discovered_by")
        _integer(task.get("report_delay", 0), path + ".report_delay", minimum=0)
        if kind == "inspection":
            _integer(task.get("service_minutes"), path + ".service_minutes", positive=True)
            if "commodity" in task and task["commodity"] is not None:
                _string(task["commodity"], path + ".commodity")
            if "quantity" in task and task["quantity"] is not None:
                _number(task["quantity"], path + ".quantity", positive=True)
        else:
            _string(task.get("commodity"), path + ".commodity")
            _number(task.get("quantity"), path + ".quantity", positive=True)
            depot_id = _string(task.get("depot_id"), path + ".depot_id")
            if depot_id not in depot_ids:
                _fail(f"{path}.depot_id references an unknown depot")
            _integer(task.get("load_minutes"), path + ".load_minutes", positive=True)
            _integer(task.get("unload_minutes"), path + ".unload_minutes", positive=True)

    snapshot = _dict(root.get("chapter1_snapshot"), "chapter1_snapshot")
    _known_keys(snapshot, "chapter1_snapshot", {"source", "time", "queues", "assignment_receipt"})
    snapshot_source = snapshot.get("source")
    if snapshot_source not in {"synthetic_initial_assignment", C1_SNAPSHOT_SOURCE, TEST_SNAPSHOT_SOURCE} or type(snapshot.get("time")) is not int or snapshot.get("time") != 0:
        _fail("chapter1_snapshot source/time invalid")
    queues = _dict(snapshot.get("queues"), "chapter1_snapshot.queues")
    queued_ids: set[str] = set()
    for platform_id, queue in queues.items():
        if platform_id not in platform_ids:
            _fail(f"chapter1_snapshot references unknown platform {platform_id!r}")
        queue_values = _list(queue, f"chapter1_snapshot.queues.{platform_id}")
        for task_id in queue_values:
            if task_id not in task_ids:
                _fail(f"chapter1_snapshot references unknown task {task_id!r}")
            if task_id in queued_ids:
                _fail(f"task {task_id!r} appears in more than one initial queue")
            queued_ids.add(task_id)
            task = task_by_id[task_id]
            if not task["known_initial"]:
                _fail(f"hidden task {task_id!r} cannot be in chapter1_snapshot")
    # An initial queue can contain a task only when the platform can perform it.
    for platform_id, queue in queues.items():
        platform = next(item for item in platforms if item["id"] == platform_id)
        for task_id in queue:
            task = task_by_id[task_id]
            needed = "inspect" if task["kind"] == "inspection" else "deliver"
            if needed not in platform["capabilities"]:
                _fail(f"initial platform {platform_id!r} lacks {needed} capability")
            if task["local"] != platform["local"]:
                _fail(f"task {task_id!r} and initial platform {platform_id!r} have incompatible locality")

    if snapshot_source == C1_SNAPSHOT_SOURCE:
        adapter_or_c1 = _dict(snapshot.get("assignment_receipt"), "chapter1_snapshot.assignment_receipt")
        if adapter_or_c1.get("schema") == C1_MULTIROUND_ADAPTER_SCHEMA:
            _validate_c1_multiround_adapter(root, queues, adapter_or_c1, platforms, depots, tasks, map_data)
        elif adapter_or_c1.get("schema") != "chapter1-public-c1-assignment/1":
            _fail("chapter1_snapshot assignment receipt schema invalid")

    if snapshot_source == C1_SNAPSHOT_SOURCE and snapshot["assignment_receipt"].get("schema") == "chapter1-public-c1-assignment/1":
        receipt = _dict(snapshot.get("assignment_receipt"), "chapter1_snapshot.assignment_receipt")
        _known_keys(receipt, "chapter1_snapshot.assignment_receipt", {
            "schema", "algorithm_id", "algorithm_source", "seed", "seed_usage",
            "public_input_sha256", "known_task_ids", "candidate_pair_count",
            "selected_action_ids", "assignments", "assignment_output_sha256", "feasibility",
        })
        if (
            receipt.get("schema") != "chapter1-public-c1-assignment/1"
            or receipt.get("algorithm_id") != C1_ALGORITHM_ID
            or receipt.get("algorithm_source") != "package/civilian/baselines.py:C1/choose_step_c1_actions"
            or type(receipt.get("seed")) is not int
            or receipt["seed"] < 0
            or receipt.get("seed_usage") != "recorded_for_reproducibility; deterministic C1 uses no random draws"
        ):
            _fail("chapter1_snapshot C1 receipt identity invalid")
        input_digest = receipt.get("public_input_sha256")
        if type(input_digest) is not str or len(input_digest) != 64 or any(char not in "0123456789abcdef" for char in input_digest):
            _fail("chapter1_snapshot C1 public input digest invalid")
        known_task_ids = sorted(task["id"] for task in tasks if task["known_initial"])
        if receipt.get("known_task_ids") != known_task_ids:
            _fail("chapter1_snapshot C1 known task boundary mismatch")
        if receipt.get("assignments") != queues or receipt.get("assignment_output_sha256") != key_digest(queues):
            _fail("chapter1_snapshot C1 output does not match the queues")
        selected_action_ids = receipt.get("selected_action_ids")
        expected_action_ids = {
            f"dispatch:{platform_id}:{task_id}"
            for platform_id, queue in queues.items()
            for task_id in queue
        }
        if (
            type(selected_action_ids) is not list
            or len(selected_action_ids) != len(expected_action_ids)
            or set(selected_action_ids) != expected_action_ids
        ):
            _fail("chapter1_snapshot C1 selected action identity mismatch")
        feasibility = _dict(receipt.get("feasibility"), "chapter1_snapshot.assignment_receipt.feasibility")
        _known_keys(feasibility, "chapter1_snapshot.assignment_receipt.feasibility", {
            "all_selected_actions_legal", "known_task_count", "feasible_candidate_pair_count",
            "assigned_task_count", "unassigned_known_task_reasons",
        })
        queued_task_count = len(queued_ids)
        if (
            feasibility.get("all_selected_actions_legal") is not True
            or feasibility.get("known_task_count") != len(known_task_ids)
            or feasibility.get("feasible_candidate_pair_count") != receipt.get("candidate_pair_count")
            or type(receipt.get("candidate_pair_count")) is not int
            or receipt["candidate_pair_count"] < queued_task_count
            or feasibility.get("assigned_task_count") != queued_task_count
        ):
            _fail("chapter1_snapshot C1 feasibility receipt invalid")
        unassigned = set(known_task_ids) - queued_ids
        reasons = feasibility.get("unassigned_known_task_reasons")
        if (
            type(reasons) is not dict
            or set(reasons) != unassigned
            or any(type(values) is not list or not values or any(type(value) is not str for value in values) for values in reasons.values())
        ):
            _fail("chapter1_snapshot C1 unassigned reason ledger invalid")
    elif snapshot_source != C1_SNAPSHOT_SOURCE and "assignment_receipt" in snapshot:
        _fail("only a public C1 snapshot may contain an assignment receipt")

    disturbances = _list(root.get("disturbances", []), "disturbances")
    _unique_ids(disturbances, "disturbances")
    for i, disturbance in enumerate(disturbances):
        path = f"disturbances[{i}]"
        _known_keys(disturbance, path, {
            "id", "time", "kind", "report_delay", "platform_id", "value", "depot_id",
            "commodity", "quantity", "target_type", "target_id", "position",
        })
        time = _integer(disturbance.get("time"), path + ".time", minimum=0)
        if time > horizon:
            _fail(f"{path}.time exceeds horizon")
        kind = disturbance.get("kind")
        if kind not in {"failure", "energy_change", "stock_loss", "spatial_perturbation"}:
            _fail(f"{path}.kind is invalid")
        _integer(disturbance.get("report_delay", 0), path + ".report_delay", minimum=0)
        if kind in {"failure", "energy_change"}:
            platform_id = _string(disturbance.get("platform_id"), path + ".platform_id")
            if platform_id not in platform_ids:
                _fail(f"{path}.platform_id references unknown platform")
            if kind == "energy_change":
                _number(disturbance.get("value"), path + ".value", minimum=0)
        elif kind == "stock_loss":
            depot_id = _string(disturbance.get("depot_id"), path + ".depot_id")
            if depot_id not in depot_ids:
                _fail(f"{path}.depot_id references unknown depot")
            _string(disturbance.get("commodity"), path + ".commodity")
            _number(disturbance.get("quantity"), path + ".quantity", positive=True)
        else:
            target_type = _string(disturbance.get("target_type"), path + ".target_type")
            if target_type not in {"platform", "task"}:
                _fail(f"{path}.target_type must be platform or task")
            target_id = _string(disturbance.get("target_id"), path + ".target_id")
            target = None
            if target_type == "platform":
                if target_id not in platform_ids:
                    _fail(f"{path}.target_id references an unknown platform")
                target = next(item for item in platforms if item["id"] == target_id)
            else:
                if target_id not in task_ids:
                    _fail(f"{path}.target_id references an unknown task")
                target = task_by_id[target_id]
            if target.get("local") is not True or not _inside_local_block(target["position"], map_data):
                _fail(f"{path}.target_id must identify a local entity in map.local_block")
            position = disturbance.get("position")
            _position_in_map(position, path + ".position")
            if not _inside_local_block(position, map_data):
                _fail(f"{path}.position must remain inside map.local_block")
            if map_data["local_block"][2:] != [2, 2]:
                _fail("spatial_perturbation requires a 2x2 map.local_block")
def load_scenario(path: str | Path) -> dict:
    """Load a UTF-8 JSON scenario and validate it."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            scenario = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load scenario {source}: {exc}") from exc
    validate_scenario(scenario)
    return scenario
