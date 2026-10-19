"""Public-observation graph world models for the N02 event-trigger study."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

try:
    from .scenario_schema import GRID_CELLS, MAP_COLS, MAP_CONTRACT, MAP_ROWS
except ImportError:
    from scenario_schema import GRID_CELLS, MAP_COLS, MAP_CONTRACT, MAP_ROWS


EVENT_MODEL_SCHEMA = "n02-event-world-model-ensemble/1"
WORLD_MODEL_HEADS = (
    "next_state",
    "progress",
    "delay",
    "energy",
    "failure",
    "communication_risk",
    "constraint_risk",
    "utility",
)
PROBABILITY_HEADS = frozenset({"failure", "communication_risk", "constraint_risk"})
ENSEMBLE_SEEDS = (8201, 8202, 8203)
HISTORY_LENGTH = 4
NODE_FEATURE_DIM = 36
CANDIDATE_BASE_FEATURE_DIM = 31
CANDIDATE_DISPATCH_SLOT_DIM = 18
CANDIDATE_DISPATCH_SLOTS = 6
CANDIDATE_OPERATION_SLOT_DIM = 41
CANDIDATE_OPERATION_SLOTS = 12
CANDIDATE_FEATURE_DIM = (
    CANDIDATE_BASE_FEATURE_DIM
    + CANDIDATE_DISPATCH_SLOT_DIM * CANDIDATE_DISPATCH_SLOTS
    + CANDIDATE_OPERATION_SLOT_DIM * CANDIDATE_OPERATION_SLOTS
)
CANDIDATE_FEATURE_LAYOUT = "relational-action-slots-v1"
RELATION_COUNT = 7
NODE_TYPES = {"padding": 0, "platform": 1, "task": 2, "environment": 3, "communication": 4}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> list:
    return value if isinstance(value, list) else []


def _xy(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        return _finite(value.get("x")), _finite(value.get("y"))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _finite(value[0]), _finite(value[1])
    return 0.0, 0.0


def _public_observation(history_item: dict) -> dict:
    if not isinstance(history_item, dict):
        raise ValueError("EVENT_WM_HISTORY_ITEM")
    observation = history_item.get("observation", history_item)
    if not isinstance(observation, dict):
        raise ValueError("EVENT_WM_PUBLIC_OBSERVATION")
    return observation


def _node_features(observation: dict) -> tuple[list[list[float]], list[int], list[tuple[int, int, int]]]:
    """Return public node rows, types, and typed directed edges."""
    map_data = _mapping(observation.get("map"))
    if (
        _finite(map_data.get("rows"), float("nan")) != MAP_ROWS
        or _finite(map_data.get("cols"), float("nan")) != MAP_COLS
        or ("grid_cells" in map_data and _finite(map_data.get("grid_cells"), float("nan")) != GRID_CELLS)
        or ("grid_contract" in map_data and map_data.get("grid_contract") != MAP_CONTRACT)
    ):
        raise ValueError("EVENT_WM_MAP_CONTRACT")
    platforms = sorted(_sequence(observation.get("platforms")), key=lambda row: str(_mapping(row).get("id", "")))
    tasks = sorted(_sequence(observation.get("tasks")), key=lambda row: str(_mapping(row).get("id", "")))
    environments = sorted(_sequence(observation.get("environment")), key=lambda row: str(_mapping(row).get("id", "")))
    links = sorted(_sequence(observation.get("communication")), key=lambda row: (str(_mapping(row).get("source", "")), str(_mapping(row).get("target", ""))))
    queues = _mapping(observation.get("queues"))
    predecessors = _mapping(observation.get("predecessors"))
    cooperation = _mapping(observation.get("cooperation"))
    progress_map = _mapping(observation.get("task_progress"))
    horizon = max(1.0, _finite(observation.get("horizon"), 1.0))
    now = _finite(observation.get("time"))
    width = float(MAP_COLS)
    height = float(MAP_ROWS)
    extent = max(width, height)
    platform_ids = {str(_mapping(item).get("id", "")): index for index, item in enumerate(platforms)}
    task_offset = len(platforms)
    task_ids = {str(_mapping(item).get("id", "")): task_offset + index for index, item in enumerate(tasks)}
    environment_offset = task_offset + len(tasks)
    environment_ids = {
        str(_mapping(item).get("id", "")): environment_offset + index
        for index, item in enumerate(environments)
    }
    communication_offset = environment_offset + len(environments)

    rows: list[list[float]] = []
    types: list[int] = []
    edges: list[tuple[int, int, int]] = []

    def add_node(features: list[float], node_type: int) -> int:
        if len(features) != NODE_FEATURE_DIM or not all(math.isfinite(value) for value in features):
            raise ValueError("EVENT_WM_NODE_FEATURE_SCHEMA")
        index = len(rows)
        rows.append(features)
        types.append(node_type)
        edges.append((index, index, 0))
        return index

    for index, platform in enumerate(platforms):
        item = _mapping(platform)
        x, y = _xy(item.get("position"))
        capabilities = set(str(value) for value in _sequence(item.get("capabilities")))
        health_raw = item.get("health", 1.0)
        if isinstance(health_raw, dict):
            health = _finite(health_raw.get("fraction", health_raw.get("value", 1.0)), 1.0)
        elif isinstance(health_raw, str):
            health = 0.0 if health_raw.lower() in {"failed", "critical", "offline"} else 1.0
        else:
            health = _finite(health_raw, 1.0)
        queue = _sequence(queues.get(str(item.get("id", ""))))
        load_raw = item.get("load")
        if isinstance(load_raw, dict):
            load = (_finite(load_raw.get("active_tasks")) + _finite(load_raw.get("queued_tasks"))) / 6.0
        else:
            load = _finite(item.get("load_kg", load_raw)) / max(1.0, _finite(item.get("capacity_kg"), 1.0))
        features = [0.0] * NODE_FEATURE_DIM
        features[:18] = [
            x / extent, y / extent, now / horizon,
            _finite(item.get("energy")) / 100.0, health,
            load,
            _finite(item.get("capacity_kg")) / 100.0,
            _finite(item.get("speed_km_min")) / 2.0,
            _finite(item.get("energy_per_km")) / 10.0,
            _finite(item.get("service_energy_per_min")) / 10.0,
            float(str(item.get("type", "")).lower() == "uav"),
            float(str(item.get("type", "")).lower() in {"helicopter", "helo"}),
            float(bool(item.get("failed"))), float(bool(item.get("busy"))),
            float(bool(item.get("local"))), float(len(queue)) / 12.0,
            float("inspect" in capabilities), float("deliver" in capabilities),
        ]
        add_node(features, NODE_TYPES["platform"])
        for task_id in queue:
            task_node = task_ids.get(str(task_id))
            if task_node is not None:
                edges.extend(((index, task_node, 1), (task_node, index, 1)))

    for offset, task in enumerate(tasks):
        item = _mapping(task)
        task_id = str(item.get("id", ""))
        x, y = _xy(item.get("position"))
        task_progress = _finite(progress_map.get(task_id, item.get("progress", item.get("completion_fraction", 0.0))))
        status = str(item.get("status", "")).lower()
        features = [0.0] * NODE_FEATURE_DIM
        features[:26] = [
            x / extent, y / extent, now / horizon,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, float(status in {"failed", "expired"}), float(status == "active"),
            float(bool(item.get("local"))), 0.0,
            float("inspect" in set(str(value) for value in _sequence(item.get("capabilities")))),
            float("deliver" in set(str(value) for value in _sequence(item.get("capabilities")))),
            task_progress,
            _finite(item.get("remaining_work", item.get("quantity", 0.0))) / 20.0,
            max(0.0, _finite(item.get("deadline"), horizon) - now) / horizon,
            _finite(item.get("priority")) / 5.0,
            _finite(item.get("quantity")) / 20.0,
            float(item.get("kind") == "inspection"),
            float(item.get("kind") == "delivery"),
            float(bool(_sequence(predecessors.get(task_id)))),
        ]
        add_node(features, NODE_TYPES["task"])
        for predecessor in _sequence(predecessors.get(task_id)):
            predecessor_node = task_ids.get(str(predecessor))
            if predecessor_node is not None:
                node = task_offset + offset
                edges.extend(((predecessor_node, node, 3), (node, predecessor_node, 3)))
        for peer in _sequence(item.get("cooperates_with", item.get("cooperative_with", item.get("cooperation_ids")))) + _sequence(cooperation.get(task_id)):
            peer_node = task_ids.get(str(peer))
            if peer_node is not None:
                node = task_offset + offset
                edges.extend(((node, peer_node, 4), (peer_node, node, 4)))

    for offset, environment in enumerate(environments):
        item = _mapping(environment)
        x, y = _xy(item.get("center", item.get("grid_cell", item.get("cell"))))
        wind = _xy(item.get("wind_vector"))
        features = [0.0] * NODE_FEATURE_DIM
        features[:36] = [
            x / extent, y / extent, now / horizon,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, _finite(item.get("radius_cells")) / extent,
            0.0, 0.0, 0.0, 0.0, 0.0,
            wind[0] / 10.0, wind[1] / 10.0, _finite(item.get("risk", item.get("risk_level"))),
            float(bool(item.get("blocked"))), 0.0, 0.0, 0.0, 0.0, 0.0,
            float(str(item.get("kind", "")).lower() in {"wind", "weather"}),
        ]
        node = add_node(features, NODE_TYPES["environment"])
        for entity in range(len(platforms) + len(tasks)):
            edges.extend(((node, entity, 5), (entity, node, 5)))

    for offset, link in enumerate(links):
        item = _mapping(link)
        source, target = str(item.get("source", "")), str(item.get("target", ""))
        source_node, target_node = platform_ids.get(source), platform_ids.get(target)
        mid_x = mid_y = 0.0
        if source_node is not None and target_node is not None:
            p_source, p_target = _mapping(platforms[source_node]), _mapping(platforms[target_node])
            sx, sy = _xy(p_source.get("position")); tx, ty = _xy(p_target.get("position"))
            mid_x, mid_y = (sx + tx) / (2 * extent), (sy + ty) / (2 * extent)
        features = [0.0] * NODE_FEATURE_DIM
        features[0:3] = [mid_x, mid_y, now / horizon]
        features[31:36] = [
            _finite(item.get("quality")),
            _finite(item.get("latency_minutes")) / horizon,
            _finite(item.get("loss_rate")),
            float(bool(item.get("connected"))),
            1.0,
        ]
        node = add_node(features, NODE_TYPES["communication"])
        for endpoint in (source_node, target_node):
            if endpoint is not None:
                edges.extend(((node, endpoint, 2), (endpoint, node, 2)))

    # Preserve external predecessor/cooperation declarations even if the task
    # record does not duplicate them.
    for task_id, related in predecessors.items():
        task_node = task_ids.get(str(task_id))
        if task_node is None:
            continue
        for predecessor in _sequence(related):
            other = task_ids.get(str(predecessor))
            if other is not None:
                edges.extend(((other, task_node, 3), (task_node, other, 3)))
    unique_edges = sorted(set(edges))
    return rows, types, unique_edges


def _candidate_features(candidate: dict, observation: dict) -> list[float]:
    if not isinstance(candidate, dict):
        raise ValueError("EVENT_WM_CANDIDATE")
    action = candidate.get("full_environment_action", {})
    if not isinstance(action, dict):
        raise ValueError("EVENT_WM_FULL_ACTION_REQUIRED")
    # Provenance fields are intentionally never traversed or converted to data.
    scope = str(candidate.get("candidate_scope", action.get("scope", ""))).upper()
    scope = str(action.get("scope", scope)).upper()
    scope = {"LOCAL": "LOCAL_REPAIR", "GLOBAL": "GLOBAL_REPLAN", "NONE": "NOOP"}.get(scope, scope)
    ops = _sequence(action.get("queue_operations"))
    dispatches = _sequence(action.get("dispatches"))
    if len(dispatches) > CANDIDATE_DISPATCH_SLOTS or len(ops) > CANDIDATE_OPERATION_SLOTS:
        raise ValueError("EVENT_WM_ACTION_FEATURE_LIMIT")
    platforms = {str(_mapping(item).get("id")): _mapping(item) for item in _sequence(observation.get("platforms"))}
    tasks = {str(_mapping(item).get("id")): _mapping(item) for item in _sequence(observation.get("tasks"))}
    depots = {str(_mapping(item).get("id")): _mapping(item) for item in _sequence(observation.get("depots"))}
    extent = max(1.0, _finite(_mapping(observation.get("map")).get("rows"), MAP_ROWS), _finite(_mapping(observation.get("map")).get("cols"), MAP_COLS))
    horizon = max(1.0, _finite(observation.get("horizon"), 1.0))
    now = _finite(observation.get("time"))
    priorities = []
    slacks = []
    energies = []
    selected_tasks: set[str] = set()
    selected_platforms: set[str] = set()

    def _xy_for(item: dict | None) -> tuple[float, float]:
        x, y = _xy(_mapping(item).get("position"))
        return x / extent, y / extent

    def _required_capability(task: dict) -> str:
        return "inspect" if task.get("kind") == "inspection" else "deliver"

    def _depot_and_route(task: dict, platform: dict) -> tuple[dict, float]:
        task_x, task_y = _xy(_mapping(task).get("position"))
        platform_x, platform_y = _xy(_mapping(platform).get("position"))
        depot = depots.get(str(task.get("depot_id")), {})
        depot_x, depot_y = _xy(_mapping(depot).get("position"))
        phase = str(task.get("phase", ""))
        direct = math.hypot(platform_x - task_x, platform_y - task_y)
        if task.get("kind") != "delivery" or phase in {"in_transit", "to_delivery_site", "unloading", "delivery_confirmed"}:
            return depot, direct
        return depot, math.hypot(platform_x - depot_x, platform_y - depot_y) + math.hypot(depot_x - task_x, depot_y - task_y)

    def _service_remaining(task: dict) -> float:
        if task.get("kind") == "delivery":
            total = _finite(task.get("load_minutes"), 1.0) + _finite(task.get("unload_minutes"), 1.0)
        else:
            total = _finite(task.get("service_minutes"), 1.0)
        progress = min(1.0, max(0.0, _finite(task.get("progress", _mapping(observation.get("task_progress")).get(str(task.get("id")))))))
        return max(0.0, total * (1.0 - progress))

    def _platform_has_task_capability(platform: dict, task: dict) -> bool:
        return _required_capability(task) in set(str(value) for value in _sequence(platform.get("capabilities")))

    def _quantity(task: dict) -> float:
        return max(0.0, _finite(task.get("quantity", task.get("remaining_work", 0.0))))

    def _task_features(task: dict, platform: dict) -> list[float]:
        px, py = _xy(_mapping(platform).get("position"))
        tx, ty = _xy(_mapping(task).get("position"))
        depot, route_distance = _depot_and_route(task, platform)
        dx, dy = _xy(_mapping(depot).get("position"))
        direct_distance = math.hypot(px - tx, py - ty)
        initial_energy = max(1.0, _finite(platform.get("energy_initial", platform.get("energy", 1.0))))
        capacity = max(1.0, _finite(platform.get("capacity_kg"), 1.0))
        required_energy = route_distance * max(0.0, _finite(platform.get("energy_per_km")))
        required_energy += _service_remaining(task) * max(0.0, _finite(platform.get("service_energy_per_min")))
        quantity = _quantity(task)
        required = _required_capability(task)
        return [
            px / extent, py / extent, tx / extent, ty / extent,
            (tx - px) / extent, (ty - py) / extent,
            direct_distance / extent, route_distance / extent,
            dx / extent, dy / extent,
            required_energy / initial_energy,
            _finite(platform.get("energy")) / initial_energy,
            quantity / capacity,
            float(required == "inspect"), float(required == "deliver"),
            float(_platform_has_task_capability(platform, task)),
            min(1.0, max(0.0, _finite(task.get("progress", 0.0)))),
            quantity / 20.0,
        ]

    def _operation_features(operation: dict) -> list[float]:
        op = str(operation.get("op", "")).lower()
        operation_type = [float(op == name) for name in ("transfer", "swap", "insert", "split", "cooperate")]
        operation_type.append(float(op not in {"transfer", "swap", "insert", "split", "cooperate"}))
        task = tasks.get(str(operation.get("task_id", "")), {})
        before = tasks.get(str(operation.get("before_task_id", "")), {})
        other = tasks.get(str(operation.get("other_task_id", "")), {})
        source = platforms.get(str(operation.get("from_platform", "")), {})
        target = platforms.get(str(operation.get("to_platform", "")), {})
        task_x, task_y = _xy(_mapping(task).get("position"))
        depot, route_distance = _depot_and_route(task, source or target)
        depot_x, depot_y = _xy(_mapping(depot).get("position"))
        source_x, source_y = _xy(_mapping(source).get("position"))
        target_x, target_y = _xy(_mapping(target).get("position"))
        before_x, before_y = _xy(_mapping(before).get("position"))
        other_x, other_y = _xy(_mapping(other).get("position"))
        source_energy_initial = max(1.0, _finite(source.get("energy_initial", source.get("energy", 1.0))))
        target_energy_initial = max(1.0, _finite(target.get("energy_initial", target.get("energy", 1.0))))
        required = _required_capability(task) if task else ""
        deadline_slack = max(0.0, _finite(task.get("deadline"), now) - now) / horizon if task else 0.0
        return operation_type + [
            task_x / extent, task_y / extent,
            depot_x / extent, depot_y / extent,
            source_x / extent, source_y / extent,
            target_x / extent, target_y / extent,
            (target_x - source_x) / extent, (target_y - source_y) / extent,
            math.hypot(target_x - source_x, target_y - source_y) / extent,
            math.hypot(task_x - source_x, task_y - source_y) / extent,
            route_distance / extent,
            _finite(operation.get("share", operation.get("workload_share", 0.0))),
            _quantity(task) / 20.0,
            min(1.0, max(0.0, _finite(task.get("progress", 0.0)))),
            _finite(task.get("priority")) / 5.0,
            deadline_slack,
            float(required == "inspect"), float(required == "deliver"),
            float(bool(before)), before_x / extent, before_y / extent,
            _finite(before.get("priority")) / 5.0,
            float(bool(other)), other_x / extent, other_y / extent,
            _finite(other.get("priority")) / 5.0,
            max(0.0, _finite(other.get("deadline"), now) - now) / horizon if other else 0.0,
            _finite(source.get("energy")) / source_energy_initial,
            _finite(target.get("energy")) / target_energy_initial,
            _finite(source.get("capacity_kg")) / 100.0,
            _finite(target.get("capacity_kg")) / 100.0,
            float(bool(source) and _platform_has_task_capability(source, task)),
            float(bool(target) and _platform_has_task_capability(target, task)),
        ]

    for dispatch in dispatches:
        row = _mapping(dispatch)
        task_id, platform_id = str(row.get("task_id", "")), str(row.get("platform_id", ""))
        task, platform = tasks.get(task_id, {}), platforms.get(platform_id, {})
        if task:
            selected_tasks.add(task_id)
            priorities.append(_finite(task.get("priority")) / 5.0)
            slacks.append(max(0.0, _finite(task.get("deadline"), _finite(observation.get("time"))) - _finite(observation.get("time"))) / max(1.0, _finite(observation.get("horizon"), 1.0)))
        if platform:
            selected_platforms.add(platform_id)
            energies.append(_finite(platform.get("energy")) / 100.0)
    for op in ops:
        row = _mapping(op)
        task_id = str(row.get("task_id", ""))
        if task_id in tasks:
            selected_tasks.add(task_id)
            priorities.append(_finite(tasks[task_id].get("priority")) / 5.0)
            slacks.append(max(0.0, _finite(tasks[task_id].get("deadline"), _finite(observation.get("time"))) - _finite(observation.get("time"))) / max(1.0, _finite(observation.get("horizon"), 1.0)))
        for key in ("from_platform", "to_platform"):
            platform_id = str(row.get(key, ""))
            if platform_id in platforms:
                selected_platforms.add(platform_id)
                energies.append(_finite(platforms[platform_id].get("energy")) / 100.0)
    operation_names = [str(_mapping(item).get("op", "")).lower() for item in ops]
    shared_operations = [
        _mapping(item) for item in ops
        if str(_mapping(item).get("op", "")).lower() in {"split", "cooperate"}
    ]
    group_sizes: dict[str, int] = {}
    for row in ops:
        operation = _mapping(row)
        group = operation.get("cooperation_group")
        if group is not None and str(group):
            key = str(group)
            group_sizes[key] = group_sizes.get(key, 0) + 1
    shared_participants = {
        str(_mapping(row).get(key))
        for row in shared_operations
        for key in ("from_platform", "to_platform")
        if _mapping(row).get(key) is not None
    }
    shares = [_finite(row.get("share"), 0.0) for row in shared_operations]
    base = [
        float(bool(candidate.get("legal", True))),
        float(bool(action.get("noop")) or str(action.get("action_type", "")).lower() == "noop"),
        float(scope == "NOOP"), float(scope == "LOCAL_REPAIR"), float(scope == "GLOBAL_REPLAN"),
        float(str(action.get("action_type", "")).lower() == "reconstruct"),
        min(1.0, len(dispatches) / 6.0), min(1.0, len(ops) / 12.0),
        float(operation_names.count("transfer")) / 12.0,
        float(operation_names.count("swap")) / 12.0,
        float(operation_names.count("insert")) / 12.0,
        float(operation_names.count("split")) / 12.0,
        float(operation_names.count("cooperate")) / 12.0,
        min(1.0, max(0.0, sum(shares) / max(1, len(shares)))),
        min(1.0, len(shared_participants) / 6.0),
        min(1.0, len(group_sizes) / 12.0),
        min(1.0, sum(group_sizes.values()) / 12.0),
        min(1.0, max(group_sizes.values(), default=0) / 6.0),
        min(1.0, len(selected_platforms) / 6.0), min(1.0, len(selected_tasks) / 12.0),
        sum(priorities) / max(1, len(priorities)),
        min(slacks, default=0.0), sum(energies) / max(1, len(energies)),
        float(str(candidate.get("candidate_scope", "")).upper() in {"LOCAL", "LOCAL_REPAIR"}),
        float(str(candidate.get("candidate_scope", "")).upper() in {"GLOBAL", "GLOBAL_REPLAN"}),
        float(bool(candidate.get("infeasibility_reason"))),
        _finite(observation.get("time")) / max(1.0, _finite(observation.get("horizon"), 1.0)),
        len(_sequence(observation.get("platforms"))) / 6.0,
        len(_sequence(observation.get("tasks"))) / 12.0,
        sum(_finite(_mapping(item).get("risk")) for item in _sequence(observation.get("environment"))) / max(1, len(_sequence(observation.get("environment")))),
        sum(_finite(_mapping(item).get("quality")) for item in _sequence(observation.get("communication"))) / max(1, len(_sequence(observation.get("communication")))),
    ]
    if len(base) != CANDIDATE_BASE_FEATURE_DIM:
        raise ValueError("EVENT_WM_CANDIDATE_BASE_FEATURE_SCHEMA")
    relational: list[float] = []
    for dispatch in dispatches:
        row = _mapping(dispatch)
        task = tasks.get(str(row.get("task_id", "")), {})
        platform = platforms.get(str(row.get("platform_id", "")), {})
        dispatch_features = _task_features(task, platform)
        if len(dispatch_features) != CANDIDATE_DISPATCH_SLOT_DIM:
            raise ValueError("EVENT_WM_DISPATCH_FEATURE_SCHEMA")
        relational.extend(dispatch_features)
    relational.extend([0.0] * ((CANDIDATE_DISPATCH_SLOTS - len(dispatches)) * CANDIDATE_DISPATCH_SLOT_DIM))
    for operation in ops:
        operation_features = _operation_features(_mapping(operation))
        if len(operation_features) != CANDIDATE_OPERATION_SLOT_DIM:
            raise ValueError("EVENT_WM_OPERATION_FEATURE_SCHEMA")
        relational.extend(operation_features)
    relational.extend([0.0] * ((CANDIDATE_OPERATION_SLOTS - len(ops)) * CANDIDATE_OPERATION_SLOT_DIM))
    result = base + relational
    if len(result) != CANDIDATE_FEATURE_DIM or not all(math.isfinite(value) for value in result):
        raise ValueError("EVENT_WM_CANDIDATE_FEATURE_SCHEMA")
    return result


def _encode_sample_graph(sample: dict) -> tuple[list[list[list[float]]], list[list[int]], list[list[list[list[float]]]], list[int]]:
    history = sample.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("EVENT_WM_HISTORY_REQUIRED")
    selected = history[-HISTORY_LENGTH:]
    rows_by_time: list[list[list[float]]] = []
    types_by_time: list[list[int]] = []
    edges_by_time: list[list[list[float]]] = []
    for entry in selected:
        observation = _public_observation(entry)
        nodes, types, edges = _node_features(observation)
        if not nodes:
            # A public map-only state still has a valid global environment node.
            nodes, types, edges = [[0.0] * NODE_FEATURE_DIM], [NODE_TYPES["environment"]], [(0, 0, 0)]
        count = len(nodes)
        adjacency = [[[0.0 for _ in range(count)] for _ in range(count)] for _ in range(RELATION_COUNT)]
        for source, target, relation in edges:
            adjacency[relation][target][source] = 1.0
        for relation in range(RELATION_COUNT):
            for target in range(count):
                degree = sum(adjacency[relation][target])
                if degree > 0:
                    adjacency[relation][target] = [value / degree for value in adjacency[relation][target]]
        rows_by_time.append(nodes)
        types_by_time.append(types)
        edges_by_time.append(adjacency)
    if len(rows_by_time) < HISTORY_LENGTH:
        pad = HISTORY_LENGTH - len(rows_by_time)
        rows_by_time.extend(copy.deepcopy(rows_by_time[-1]) for _ in range(pad))
        types_by_time.extend(copy.deepcopy(types_by_time[-1]) for _ in range(pad))
        edges_by_time.extend(copy.deepcopy(edges_by_time[-1]) for _ in range(pad))
    return rows_by_time, types_by_time, edges_by_time, [len(history[-HISTORY_LENGTH:])]


def _target_dimensions(samples: Sequence[dict]) -> dict[str, int]:
    dimensions: dict[str, int] = {}
    for head in WORLD_MODEL_HEADS:
        for sample in samples:
            entry = _mapping(_mapping(sample.get("targets")).get(head))
            values = _sequence(entry.get("values"))
            found = next((value for value in values if value is not None), None)
            if found is not None:
                dimensions[head] = len(found) if isinstance(found, (list, tuple)) else 1
                break
        if head not in dimensions:
            raise ValueError(f"EVENT_WM_NO_LABELS:{head}")
        if dimensions[head] <= 0:
            raise ValueError(f"EVENT_WM_LABEL_WIDTH:{head}")
    return dimensions


def _label_row(value: Any, valid: Any, width: int, head: str) -> tuple[list[float], list[bool]]:
    if value is None:
        return [0.0] * width, [False] * width
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if len(values) != width:
        raise ValueError(f"EVENT_WM_LABEL_WIDTH_MISMATCH:{head}")
    masks = list(valid) if isinstance(valid, (list, tuple)) else [bool(valid)] * width
    if len(masks) != width or any(type(flag) is not bool for flag in masks):
        raise ValueError(f"EVENT_WM_LABEL_MASK:{head}")
    clean = [_finite(item, float("nan")) for item in values]
    if any(not math.isfinite(item) for item in clean):
        raise ValueError(f"EVENT_WM_LABEL_NONFINITE:{head}")
    if head in PROBABILITY_HEADS and any(flag and not 0.0 <= item <= 1.0 for item, flag in zip(clean, masks)):
        raise ValueError(f"EVENT_WM_PROBABILITY_LABEL:{head}")
    return clean, masks


def _normalize_statistics(samples: Sequence[dict], dimensions: dict[str, int]) -> dict[str, dict[str, list[float]]]:
    node_values: list[list[float]] = []
    candidate_values: list[list[float]] = []
    target_values = {head: [[] for _ in range(width)] for head, width in dimensions.items()}
    for sample in samples:
        history_nodes, _, _, _ = _encode_sample_graph(sample)
        for observation_nodes in history_nodes:
            node_values.extend(observation_nodes)
        history = sample["history"][-1]
        observation = _public_observation(history)
        candidate_values.extend(_candidate_features(candidate, observation) for candidate in sample["candidates"])
        for head, width in dimensions.items():
            entry = sample["targets"][head]
            values, masks = entry["values"], entry["valid"]
            if len(values) != len(sample["candidates"]) or len(masks) != len(values):
                raise ValueError(f"EVENT_WM_CANDIDATE_TARGET_ALIGNMENT:{head}")
            for value, valid in zip(values, masks):
                row, mask_row = _label_row(value, valid, width, head)
                for column, (number, flag) in enumerate(zip(row, mask_row)):
                    if flag:
                        target_values[head][column].append(number)
    if not node_values or not candidate_values:
        raise ValueError("EVENT_WM_EMPTY_TRAIN_FEATURES")

    def statistics(rows: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
        tensor = torch.tensor(rows, dtype=torch.float64)
        mean = tensor.mean(dim=0)
        scale = tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
        return mean.float().tolist(), scale.float().tolist()

    node_mean, node_scale = statistics(node_values)
    candidate_mean, candidate_scale = statistics(candidate_values)
    target_stats: dict[str, dict[str, list[float]]] = {}
    for head, width in dimensions.items():
        means, scales = [], []
        for column in range(width):
            values = target_values[head][column]
            if not values:
                means.append(0.0); scales.append(1.0)
            elif head in PROBABILITY_HEADS:
                means.append(0.0); scales.append(1.0)
            else:
                tensor = torch.tensor(values, dtype=torch.float64)
                means.append(float(tensor.mean().item()))
                scales.append(max(1e-6, float(tensor.std(unbiased=False).item())))
        target_stats[head] = {"mean": means, "scale": scales}
    return {
        "source": "training_samples_only",
        "node_features": {"mean": node_mean, "scale": node_scale, "observations": len(node_values)},
        "candidate_features": {"mean": candidate_mean, "scale": candidate_scale, "observations": len(candidate_values)},
        "targets": target_stats,
    }


class TemporalGraphWorldModel(nn.Module):
    """Heterogeneous public graph encoder, latest-four GRU, and masked heads."""

    format_version = EVENT_MODEL_SCHEMA

    def __init__(self, head_dims: dict[str, int], hidden_dim: int = 64):
        super().__init__()
        if set(head_dims) != set(WORLD_MODEL_HEADS) or any(int(value) <= 0 for value in head_dims.values()):
            raise ValueError("EVENT_WM_HEAD_LAYOUT")
        if hidden_dim < 16:
            raise ValueError("EVENT_WM_HIDDEN_DIM")
        self.config = {
            "schema": EVENT_MODEL_SCHEMA,
            "head_dims": {key: int(head_dims[key]) for key in WORLD_MODEL_HEADS},
            "hidden_dim": int(hidden_dim),
            "node_feature_dim": NODE_FEATURE_DIM,
            "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
            "candidate_feature_layout": CANDIDATE_FEATURE_LAYOUT,
            "history_length": HISTORY_LENGTH,
            "node_types": NODE_TYPES,
            "relation_count": RELATION_COUNT,
            "map_contract": MAP_CONTRACT,
            "map_rows": MAP_ROWS,
            "map_cols": MAP_COLS,
            "grid_cells": GRID_CELLS,
        }
        self.register_buffer("node_mean", torch.zeros(NODE_FEATURE_DIM))
        self.register_buffer("node_scale", torch.ones(NODE_FEATURE_DIM))
        self.register_buffer("candidate_mean", torch.zeros(CANDIDATE_FEATURE_DIM))
        self.register_buffer("candidate_scale", torch.ones(CANDIDATE_FEATURE_DIM))
        self.node_encoder = nn.Sequential(nn.Linear(NODE_FEATURE_DIM, hidden_dim), nn.Tanh())
        self.node_type_embedding = nn.Embedding(len(NODE_TYPES), hidden_dim)
        self.relation_transforms = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(RELATION_COUNT))
        self.message_update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.LayerNorm(hidden_dim))
        self.graph_projection = nn.Sequential(nn.Linear(hidden_dim * len(NODE_TYPES), hidden_dim), nn.Tanh())
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.candidate_encoder = nn.Sequential(nn.Linear(CANDIDATE_FEATURE_DIM, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.candidate_context = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh())
        self.heads = nn.ModuleDict({head: nn.Linear(hidden_dim, int(head_dims[head])) for head in WORLD_MODEL_HEADS})
        self.head_means = nn.ParameterDict()
        for head in WORLD_MODEL_HEADS:
            self.register_buffer(f"target_mean_{head}", torch.zeros(int(head_dims[head])))
            self.register_buffer(f"target_scale_{head}", torch.ones(int(head_dims[head])))

    def set_normalization(self, statistics: dict[str, Any]) -> None:
        if statistics.get("source") != "training_samples_only":
            raise ValueError("EVENT_WM_NORMALIZATION_SCOPE")
        for name, width in (("node_features", NODE_FEATURE_DIM), ("candidate_features", CANDIDATE_FEATURE_DIM)):
            entry = _mapping(statistics.get(name)); mean, scale = entry.get("mean"), entry.get("scale")
            if not isinstance(mean, list) or not isinstance(scale, list) or len(mean) != width or len(scale) != width or any(_finite(item, float("nan")) != _finite(item, float("nan")) for item in mean + scale) or any(_finite(item) <= 0 for item in scale):
                raise ValueError(f"EVENT_WM_NORMALIZATION:{name}")
            getattr(self, name.replace("_features", "") + "_mean").copy_(torch.tensor(mean, dtype=torch.float32))
            getattr(self, name.replace("_features", "") + "_scale").copy_(torch.tensor(scale, dtype=torch.float32))
        target_stats = _mapping(statistics.get("targets"))
        if set(target_stats) != set(WORLD_MODEL_HEADS):
            raise ValueError("EVENT_WM_TARGET_NORMALIZATION")
        for head in WORLD_MODEL_HEADS:
            row = target_stats[head]
            mean, scale = row.get("mean"), row.get("scale")
            width = self.config["head_dims"][head]
            if not isinstance(mean, list) or not isinstance(scale, list) or len(mean) != width or len(scale) != width or any(not math.isfinite(_finite(item, float("nan"))) for item in mean + scale) or any(_finite(item) <= 0.0 for item in scale):
                raise ValueError(f"EVENT_WM_TARGET_NORMALIZATION:{head}")
            getattr(self, f"target_mean_{head}").copy_(torch.tensor(mean, dtype=torch.float32))
            getattr(self, f"target_scale_{head}").copy_(torch.tensor(scale, dtype=torch.float32))

    def _graph_sequence(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        node_features = (batch["node_features"] - self.node_mean) / self.node_scale
        node_types = batch["node_types"].clamp(0, len(NODE_TYPES) - 1)
        node_mask = batch["node_mask"].unsqueeze(-1)
        batch_size, steps, node_count, _ = node_features.shape
        pooled_steps = []
        for time_index in range(steps):
            hidden = self.node_encoder(node_features[:, time_index]) + self.node_type_embedding(node_types[:, time_index])
            hidden = hidden * node_mask[:, time_index].to(hidden.dtype)
            aggregate = torch.zeros_like(hidden)
            for relation, transform in enumerate(self.relation_transforms):
                adjacency = batch["adjacency"][:, time_index, relation]
                messages = torch.matmul(adjacency, hidden)
                aggregate = aggregate + transform(messages)
            hidden = self.message_update(torch.cat((hidden, aggregate), dim=-1)) * node_mask[:, time_index].to(hidden.dtype)
            pooled = []
            for node_type in range(len(NODE_TYPES)):
                mask = (batch["node_types"][:, time_index] == node_type) & batch["node_mask"][:, time_index]
                weights = mask.to(hidden.dtype).unsqueeze(-1)
                pooled.append((hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0))
            pooled_steps.append(self.graph_projection(torch.cat(pooled, dim=-1)))
        sequence = torch.stack(pooled_steps, dim=1)
        lengths = batch["history_lengths"].to(device="cpu", dtype=torch.long).clamp(1, steps)
        packed = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.temporal(packed)
        return hidden[-1].reshape(batch_size, -1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = {"node_features", "node_types", "node_mask", "adjacency", "history_lengths", "candidate_features", "candidate_mask"}
        if set(batch) != required:
            raise ValueError("EVENT_WM_BATCH_FIELDS")
        graph = self._graph_sequence(batch)
        candidates = (batch["candidate_features"] - self.candidate_mean) / self.candidate_scale
        candidate_hidden = self.candidate_encoder(candidates)
        repeated_graph = graph[:, None, :].expand(-1, candidates.shape[1], -1)
        hidden = self.candidate_context(torch.cat((repeated_graph, candidate_hidden), dim=-1))
        mask = batch["candidate_mask"].unsqueeze(-1)
        return {head: self.heads[head](hidden).masked_fill(~mask, 0.0) for head in WORLD_MODEL_HEADS}


@dataclass
class WorldModelEnsemble:
    models: list[TemporalGraphWorldModel]
    metadata: list[dict[str, Any]]
    ensemble_hash: str
    forward_calls: int = 0
    prediction_calls: int = 0

    def eval(self) -> "WorldModelEnsemble":
        for model in self.models:
            model.eval()
            model.requires_grad_(False)
        return self


def _collate_samples(samples: Sequence[dict], head_dims: dict[str, int] | None = None) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    encoded = []
    candidate_rows = []
    max_nodes = max_candidates = 1
    dimensions = head_dims or _target_dimensions(samples)
    for sample in samples:
        if not isinstance(sample.get("candidates"), list) or not sample["candidates"]:
            raise ValueError("EVENT_WM_CANDIDATES_REQUIRED")
        histories, node_types, adjacencies, lengths = _encode_sample_graph(sample)
        observation = _public_observation(sample["history"][-1])
        features = [_candidate_features(candidate, observation) for candidate in sample["candidates"]]
        encoded.append((histories, node_types, adjacencies, lengths[0]))
        candidate_rows.append(features)
        max_nodes = max(max_nodes, *(len(nodes) for nodes in histories))
        max_candidates = max(max_candidates, len(features))
    batch_size = len(samples)
    node_tensor = torch.zeros((batch_size, HISTORY_LENGTH, max_nodes, NODE_FEATURE_DIM), dtype=torch.float32)
    type_tensor = torch.zeros((batch_size, HISTORY_LENGTH, max_nodes), dtype=torch.long)
    node_mask = torch.zeros((batch_size, HISTORY_LENGTH, max_nodes), dtype=torch.bool)
    adjacency = torch.zeros((batch_size, HISTORY_LENGTH, RELATION_COUNT, max_nodes, max_nodes), dtype=torch.float32)
    history_lengths = torch.zeros(batch_size, dtype=torch.long)
    candidate_tensor = torch.zeros((batch_size, max_candidates, CANDIDATE_FEATURE_DIM), dtype=torch.float32)
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    targets = {head: torch.zeros((batch_size, max_candidates, dimensions[head]), dtype=torch.float32) for head in WORLD_MODEL_HEADS}
    target_masks = {head: torch.zeros((batch_size, max_candidates, dimensions[head]), dtype=torch.bool) for head in WORLD_MODEL_HEADS}
    for batch_index, (sample, graph, features) in enumerate(zip(samples, encoded, candidate_rows)):
        histories, node_types, adjacencies, length = graph
        history_lengths[batch_index] = length
        for time_index, (nodes, types, edges) in enumerate(zip(histories, node_types, adjacencies)):
            count = len(nodes)
            node_tensor[batch_index, time_index, :count] = torch.tensor(nodes, dtype=torch.float32)
            type_tensor[batch_index, time_index, :count] = torch.tensor(types, dtype=torch.long)
            node_mask[batch_index, time_index, :count] = True
            adjacency[batch_index, time_index, :, :count, :count] = torch.tensor(edges, dtype=torch.float32)
        candidate_count = len(features)
        candidate_tensor[batch_index, :candidate_count] = torch.tensor(features, dtype=torch.float32)
        candidate_mask[batch_index, :candidate_count] = True
        sample_targets = _mapping(sample.get("targets"))
        for head, width in dimensions.items():
            entry = _mapping(sample_targets.get(head))
            values, masks = entry.get("values"), entry.get("valid")
            if not isinstance(values, list) or not isinstance(masks, list) or len(values) != candidate_count or len(masks) != candidate_count:
                raise ValueError(f"EVENT_WM_CANDIDATE_TARGET_ALIGNMENT:{head}")
            for candidate_index, (value, valid) in enumerate(zip(values, masks)):
                row, row_mask = _label_row(value, valid, width, head)
                targets[head][batch_index, candidate_index] = torch.tensor(row)
                target_masks[head][batch_index, candidate_index] = torch.tensor(row_mask)
    return {
        "node_features": node_tensor,
        "node_types": type_tensor,
        "node_mask": node_mask,
        "adjacency": adjacency,
        "history_lengths": history_lengths,
        "candidate_features": candidate_tensor,
        "candidate_mask": candidate_mask,
    }, {head: torch.cat((targets[head], target_masks[head].to(torch.float32)), dim=-1) for head in WORLD_MODEL_HEADS}


def _identity(samples: Sequence[dict]) -> str:
    records = []
    for index, sample in enumerate(samples):
        public_history = copy.deepcopy(sample.get("history", []))
        for history_item in public_history:
            observation = history_item.get("observation") if isinstance(history_item, dict) and isinstance(history_item.get("observation"), dict) else history_item
            if isinstance(observation, dict):
                observation.pop("provenance", None)
        public_candidates = copy.deepcopy(sample.get("candidates", []))
        for candidate in public_candidates:
            if isinstance(candidate, dict):
                for name in ("public_prefix_hash", "snapshot_hash", "event_tape_id"):
                    candidate.pop(name, None)
        records.append({
            "sample_id": str(sample.get("sample_id", f"sample-{index:06d}")),
            "split": sample.get("split"),
            "parent_id": sample.get("parent_id"),
            "public_input_sha256": hashlib.sha256(json.dumps({"history": public_history, "candidates": public_candidates}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
            "candidate_keys": [str(candidate.get("candidate_key", candidate.get("id", index))) for candidate in sample.get("candidates", [])],
            "targets": sample.get("targets"),
        })
    raw = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(str(tensor.dtype).encode("ascii")); digest.update(str(tuple(tensor.shape)).encode("ascii")); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, head: str) -> torch.Tensor | None:
    if not bool(valid.any()):
        return None
    if head in PROBABILITY_HEADS:
        return F.binary_cross_entropy_with_logits(prediction[valid], target[valid])
    return F.mse_loss(prediction[valid], target[valid])


def _train_single_model(
    samples: Sequence[dict],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
    data_identity: str,
) -> tuple[TemporalGraphWorldModel, dict[str, Any]]:
    dimensions = _target_dimensions(samples)
    statistics = _normalize_statistics(samples, dimensions)
    model = TemporalGraphWorldModel(dimensions, hidden_dim=hidden_dim)
    model.set_normalization(statistics)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    initial_state_digest = _state_digest(model.state_dict())
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    logs: list[dict[str, float]] = []
    update_log: list[dict[str, Any]] = []
    forward_calls = candidate_evaluations = 0
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(len(samples), generator=generator).tolist()
        epoch_losses: list[float] = []
        for start in range(0, len(samples), batch_size):
            selected = [samples[index] for index in permutation[start:start + batch_size]]
            batch, packed_targets = _collate_samples(selected, dimensions)
            output = model(batch)
            forward_calls += 1
            candidate_evaluations += sum(len(sample["candidates"]) for sample in selected)
            losses = []
            for head in WORLD_MODEL_HEADS:
                width = dimensions[head]
                target = packed_targets[head][..., :width]
                valid = packed_targets[head][..., width:].bool()
                if head not in PROBABILITY_HEADS:
                    mean = getattr(model, f"target_mean_{head}")
                    scale = getattr(model, f"target_scale_{head}")
                    target = (target - mean) / scale
                loss = _loss(output[head], target, valid, head)
                if loss is not None:
                    losses.append(loss)
            if not losses:
                raise ValueError("EVENT_WM_BATCH_WITHOUT_VALID_LABELS")
            total = torch.stack(losses).mean()
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("EVENT_WM_NONFINITE_LOSS")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                raise FloatingPointError("EVENT_WM_EMPTY_OR_NONFINITE_GRADIENT")
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("EVENT_WM_NONFINITE_GRADIENT_NORM")
            state_before = _state_digest(model.state_dict())
            optimizer.step()
            state_after = _state_digest(model.state_dict())
            batch_loss = float(total.detach().item())
            epoch_losses.append(batch_loss)
            update_log.append({
                "update": len(update_log) + 1,
                "epoch": epoch + 1,
                "batch_index": start // batch_size,
                "sample_ids": [str(row.get("sample_id", f"sample-{index:06d}")) for index, row in zip(permutation[start:start + batch_size], selected)],
                "parent_ids": sorted({str(row["parent_id"]) for row in selected}),
                "candidate_keys": [str(candidate.get("candidate_key", candidate.get("id", ""))) for row in selected for candidate in row["candidates"]],
                "loss": batch_loss,
                "gradient_norm_preclip": float(gradient_norm.detach().item()),
                "parameter_digest_before": state_before,
                "parameter_digest_after": state_after,
                "optimizer_step_applied": True,
            })
        logs.append({"epoch": epoch + 1, "loss": sum(epoch_losses) / len(epoch_losses), "optimizer_updates": len(epoch_losses)})
    model.eval()
    model.requires_grad_(False)
    final_state_digest = _state_digest(model.state_dict())
    parameter_delta = math.sqrt(sum(float(torch.sum((after.detach() - before) ** 2).item()) for before, after in zip(initial_parameters, model.parameters())))
    if initial_state_digest == final_state_digest or not math.isfinite(parameter_delta) or parameter_delta <= 0.0:
        raise AssertionError("EVENT_WM_OPTIMIZER_DID_NOT_CHANGE_PARAMETERS")
    metadata = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer_updates": sum(row["optimizer_updates"] for row in logs),
        "history": logs,
        "update_log": update_log,
        "initial_model_state_sha256": initial_state_digest,
        "final_model_state_sha256": final_state_digest,
        "parameter_delta_l2": parameter_delta,
        "forward_calls": forward_calls,
        "candidate_evaluations": candidate_evaluations,
        "normalization": statistics,
        "model_state_sha256": _state_digest(model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "data_identity": data_identity,
    }
    return model, metadata


def train_ensemble(samples: Sequence[dict], output_dir: str | Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fit three independently seeded models using training samples only."""
    if not isinstance(samples, Sequence) or not samples:
        raise ValueError("EVENT_WM_SAMPLES_REQUIRED")
    training_samples = list(samples)
    if any(sample.get("split") != "train" for sample in training_samples):
        raise ValueError("EVENT_WM_TRAIN_SPLIT_ONLY")
    parent_ids = {str(sample.get("parent_id", "")) for sample in training_samples}
    if "" in parent_ids or len(parent_ids) < 1:
        raise ValueError("EVENT_WM_PARENT_ID_REQUIRED")
    config = dict(config or {})
    seeds = tuple(config.get("seeds", ENSEMBLE_SEEDS))
    if seeds != ENSEMBLE_SEEDS:
        raise ValueError("EVENT_WM_SEEDS_FROZEN")
    epochs = int(config.get("epochs", 40))
    batch_size = int(config.get("batch_size", 8))
    hidden_dim = int(config.get("hidden_dim", 64))
    learning_rate = float(config.get("learning_rate", 1e-3))
    if epochs != 40 or batch_size != 8 or hidden_dim < 16 or learning_rate <= 0.0:
        raise ValueError("EVENT_WM_TRAINING_CONFIG_FROZEN")
    dimensions = _target_dimensions(training_samples)
    identity = _identity(training_samples)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: list[str] = []
    records = []
    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        model, training = _train_single_model(training_samples, seed=seed, epochs=epochs, batch_size=batch_size, hidden_dim=hidden_dim, learning_rate=learning_rate, data_identity=identity)
        payload = {
            "schema": EVENT_MODEL_SCHEMA,
            "seed": seed,
            "data_identity": identity,
            "config": copy.deepcopy(model.config),
            "training_config": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate},
            "normalization": training["normalization"],
            "model_state": model.state_dict(),
            "optimizer_state": training["optimizer_state"],
            "torch_rng_state": training["torch_rng_state"],
            "python_rng_state": training["python_rng_state"],
            "model_state_sha256": training["model_state_sha256"],
            "training_history": training["history"],
            "update_log": training["update_log"],
            "initial_model_state_sha256": training["initial_model_state_sha256"],
            "parameter_delta_l2": training["parameter_delta_l2"],
            "training_forward_calls": training["forward_calls"],
            "candidate_evaluations": training["candidate_evaluations"],
            "optimizer_updates": training["optimizer_updates"],
        }
        path = directory / f"world_model_seed_{seed}.pt"
        _atomic_save(payload, path)
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint_paths.append(str(path.resolve()))
        records.append({"seed": seed, "checkpoint_path": str(path.resolve()), "checkpoint_sha256": file_hash, "model_state_sha256": training["model_state_sha256"], "initial_model_state_sha256": training["initial_model_state_sha256"], "parameter_delta_l2": training["parameter_delta_l2"], "optimizer_updates": training["optimizer_updates"], "training_forward_calls": training["forward_calls"], "candidate_evaluations": training["candidate_evaluations"], "final_loss": training["history"][-1]["loss"], "training_history": training["history"], "update_log": training["update_log"]})
    return {
        "status": "complete",
        "schema": EVENT_MODEL_SCHEMA,
        "checkpoint_paths": checkpoint_paths,
        "checkpoints": records,
        "data_identity": identity,
        "parent_ids": sorted(parent_ids),
        "sample_count": len(training_samples),
        "head_dims": dimensions,
        "config": {"seeds": list(seeds), "epochs": epochs, "batch_size": batch_size, "hidden_dim": hidden_dim, "learning_rate": learning_rate},
        "training_forward_calls": sum(record["training_forward_calls"] for record in records),
        "candidate_evaluations": sum(record["candidate_evaluations"] for record in records),
    }


def load_ensemble(checkpoint_paths: Sequence[str | Path]) -> WorldModelEnsemble:
    if not isinstance(checkpoint_paths, Sequence) or len(checkpoint_paths) != 3:
        raise ValueError("EVENT_WM_THREE_CHECKPOINTS_REQUIRED")
    models, metadata = [], []
    expected_seeds = list(ENSEMBLE_SEEDS)
    file_digests = []
    identities = set()
    for index, path_value in enumerate(checkpoint_paths):
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != EVENT_MODEL_SCHEMA or int(payload.get("seed", -1)) != expected_seeds[index]:
            raise ValueError("EVENT_WM_CHECKPOINT_SCHEMA_OR_SEED")
        config = payload.get("config")
        if (
            not isinstance(config, dict)
            or config.get("schema") != EVENT_MODEL_SCHEMA
            or config.get("node_feature_dim") != NODE_FEATURE_DIM
            or config.get("candidate_feature_dim") != CANDIDATE_FEATURE_DIM
            or config.get("candidate_feature_layout") != CANDIDATE_FEATURE_LAYOUT
            or config.get("history_length") != HISTORY_LENGTH
            or config.get("map_contract") != MAP_CONTRACT
            or config.get("map_rows") != MAP_ROWS
            or config.get("map_cols") != MAP_COLS
            or config.get("grid_cells") != GRID_CELLS
        ):
            raise ValueError("EVENT_WM_CHECKPOINT_LAYOUT")
        if not isinstance(payload.get("data_identity"), str) or not payload.get("data_identity"):
            raise ValueError("EVENT_WM_CHECKPOINT_IDENTITY")
        identities.add(payload["data_identity"])
        normalization = payload.get("normalization")
        if not isinstance(normalization, dict) or normalization.get("source") != "training_samples_only":
            raise ValueError("EVENT_WM_CHECKPOINT_NORMALIZATION_SCOPE")
        model = TemporalGraphWorldModel(config["head_dims"], hidden_dim=int(config["hidden_dim"]))
        model.set_normalization(normalization)
        state = payload.get("model_state")
        if not isinstance(state, dict):
            raise ValueError("EVENT_WM_CHECKPOINT_STATE")
        model.load_state_dict(state, strict=True)
        if _state_digest(model.state_dict()) != payload.get("model_state_sha256"):
            raise ValueError("EVENT_WM_CHECKPOINT_STATE_DIGEST")
        model.eval().requires_grad_(False)
        models.append(model)
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        file_digests.append(file_digest)
        metadata.append({"seed": expected_seeds[index], "path": str(path.resolve()), "checkpoint_sha256": file_digest, "model_state_sha256": payload["model_state_sha256"], "data_identity": payload["data_identity"], "config": config, "training_config": payload.get("training_config"), "optimizer_state": payload.get("optimizer_state"), "optimizer_updates": payload.get("optimizer_updates")})
    if len(identities) != 1 or any(model.config != models[0].config for model in models[1:]):
        raise ValueError("EVENT_WM_ENSEMBLE_MEMBER_MISMATCH")
    ensemble_hash = hashlib.sha256("".join(file_digests).encode("ascii")).hexdigest()
    return WorldModelEnsemble(models=models, metadata=metadata, ensemble_hash=ensemble_hash).eval()


def _single_sample_batch(history: Sequence[dict], candidates: Sequence[dict]) -> tuple[dict[str, torch.Tensor], list[str]]:
    if not history or not candidates:
        raise ValueError("EVENT_WM_PREDICTION_INPUT")
    observation = _public_observation(history[-1])
    sample = {"history": list(history), "candidates": list(candidates)}
    batch, _ = _collate_samples_with_dummy_targets(sample)
    candidate_ids = [str(candidate.get("candidate_key", candidate.get("id", index))) for index, candidate in enumerate(candidates)]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("EVENT_WM_DUPLICATE_CANDIDATE_KEYS")
    return batch, candidate_ids


def _collate_samples_with_dummy_targets(sample: dict) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    prepared = copy.deepcopy(sample)
    count = len(prepared.get("candidates", []))
    prepared["targets"] = {head: {"values": [0.0] * count, "valid": [False] * count} for head in WORLD_MODEL_HEADS}
    dimensions = {head: 1 for head in WORLD_MODEL_HEADS}
    # Prediction batches do not need label widths; use a small, internal layout.
    encoded = _encode_sample_graph(prepared)
    nodes, node_types, adjacency, lengths = encoded
    max_nodes = max(1, *(len(row) for row in nodes)); candidate_count = count
    node_tensor = torch.zeros((1, HISTORY_LENGTH, max_nodes, NODE_FEATURE_DIM), dtype=torch.float32)
    type_tensor = torch.zeros((1, HISTORY_LENGTH, max_nodes), dtype=torch.long)
    node_mask = torch.zeros((1, HISTORY_LENGTH, max_nodes), dtype=torch.bool)
    edge_tensor = torch.zeros((1, HISTORY_LENGTH, RELATION_COUNT, max_nodes, max_nodes), dtype=torch.float32)
    for time_index, (node_rows, type_rows, edge_rows) in enumerate(zip(nodes, node_types, adjacency)):
        size = len(node_rows)
        node_tensor[0, time_index, :size] = torch.tensor(node_rows)
        type_tensor[0, time_index, :size] = torch.tensor(type_rows)
        node_mask[0, time_index, :size] = True
        edge_tensor[0, time_index, :, :size, :size] = torch.tensor(edge_rows)
    current = _public_observation(prepared["history"][-1])
    candidate_tensor = torch.tensor([[_candidate_features(candidate, current) for candidate in prepared["candidates"]]], dtype=torch.float32)
    candidate_mask = torch.tensor([[bool(candidate.get("legal", True)) for candidate in prepared["candidates"]]], dtype=torch.bool)
    if not bool(candidate_mask.any()):
        raise ValueError("EVENT_WM_NO_LEGAL_CANDIDATES")
    batch = {"node_features": node_tensor, "node_types": type_tensor, "node_mask": node_mask, "adjacency": edge_tensor, "history_lengths": torch.tensor(lengths, dtype=torch.long), "candidate_features": candidate_tensor, "candidate_mask": candidate_mask}
    return batch, {}


@torch.no_grad()
def predict_ensemble(ensemble: WorldModelEnsemble | Sequence[TemporalGraphWorldModel], history: Sequence[dict], candidates: Sequence[dict]) -> dict[str, Any]:
    models = ensemble.models if isinstance(ensemble, WorldModelEnsemble) else list(ensemble)
    if len(models) != 3:
        raise ValueError("EVENT_WM_THREE_MODELS_REQUIRED")
    for model in models:
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("EVENT_WM_MUST_BE_FROZEN")
    batch, candidate_ids = _single_sample_batch(history, candidates)
    predictions: dict[str, list[torch.Tensor]] = {head: [] for head in WORLD_MODEL_HEADS}
    for model in models:
        model.eval()
        output = model(batch)
        for head in WORLD_MODEL_HEADS:
            value = output[head][0]
            if head in PROBABILITY_HEADS:
                value = torch.sigmoid(value)
            else:
                value = value * getattr(model, f"target_scale_{head}") + getattr(model, f"target_mean_{head}")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("EVENT_WM_NONFINITE_PREDICTION")
            predictions[head].append(value.detach().cpu())
    if isinstance(ensemble, WorldModelEnsemble):
        ensemble.forward_calls += len(models)
        ensemble.prediction_calls += 1
    means, variances = {}, {}
    for head, values in predictions.items():
        stack = torch.stack(values, dim=0)
        mean = stack.mean(dim=0)
        variance = stack.var(dim=0, unbiased=False)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(variance).all()):
            raise FloatingPointError("EVENT_WM_NONFINITE_ENSEMBLE_STATISTICS")
        means[head] = mean.tolist()
        variances[head] = variance.tolist()
    per_candidate = []
    for index, candidate_id in enumerate(candidate_ids):
        per_candidate.append({"candidate_key": candidate_id, "valid": bool(_mapping(candidates[index]).get("legal", True)), "mean": {head: means[head][index] for head in WORLD_MODEL_HEADS}, "variance": {head: variances[head][index] for head in WORLD_MODEL_HEADS}})
    if isinstance(ensemble, WorldModelEnsemble):
        ensemble_hash = ensemble.ensemble_hash
    else:
        ensemble_hash = hashlib.sha256("".join(_state_digest(model.state_dict()) for model in models).encode("ascii")).hexdigest()
    return {"schema": EVENT_MODEL_SCHEMA, "candidate_keys": candidate_ids, "means": means, "variances": variances, "per_candidate": per_candidate, "ensemble_hash": ensemble_hash, "head_names": list(WORLD_MODEL_HEADS)}
