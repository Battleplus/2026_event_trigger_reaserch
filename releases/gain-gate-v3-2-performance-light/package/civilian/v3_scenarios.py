"""Seeded V3 scenarios for nine responsibility regions and a 30-platform fleet."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any

try:
    from .event_environment import EventReliefEnv
    from .event_scenarios import _digest, validate_event_scenario
    from .gain_scenarios import _multi_round_c1_assignment, scenario_content_sha256
    from .region_contract import REGION_CONTRACT, REGION_BOUNDARIES, region_id
    from .scenario_schema import C1_SNAPSHOT_SOURCE, GRID_CELLS, MAP_COLS, MAP_CONTRACT, MAP_ROWS, validate_scenario
except ImportError:
    from event_environment import EventReliefEnv
    from event_scenarios import _digest, validate_event_scenario
    from gain_scenarios import _multi_round_c1_assignment, scenario_content_sha256
    from region_contract import REGION_CONTRACT, REGION_BOUNDARIES, region_id
    from scenario_schema import C1_SNAPSHOT_SOURCE, GRID_CELLS, MAP_COLS, MAP_CONTRACT, MAP_ROWS, validate_scenario


SCENARIO_SCHEMA = "gain-gate-v3-regional-scenario/1"
HORIZON_MINUTES = 90
REGION_COUNT = 9
UAV_COUNT = 24
HELICOPTER_COUNT = 6
TASK_COUNT = 54
BASELINE_ARRIVALS = ((24, 4), (56, 5))
_BASE_UAV_DISTRIBUTION = (3, 3, 3, 2, 3, 3, 2, 3, 2)


def _stable_int(seed: int, *parts: Any) -> int:
    raw = json.dumps([seed, *parts], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _region_bounds(region: int) -> tuple[float, float, float, float]:
    row, col = divmod(region, 3)
    return float(REGION_BOUNDARIES[col]), float(REGION_BOUNDARIES[row]), float(REGION_BOUNDARIES[col + 1]), float(REGION_BOUNDARIES[row + 1])


def _position(rng: random.Random, region: int) -> list[float]:
    x0, y0, x1, y1 = _region_bounds(region)
    return [round(rng.uniform(x0 + 4.0, x1 - 4.0), 3), round(rng.uniform(y0 + 4.0, y1 - 4.0), 3)]


def _inside_local(position: list[float], block: list[int]) -> bool:
    x, y, width, height = block
    return x <= position[0] < x + width and y <= position[1] < y + height


def _platforms(rng: random.Random, seed: int) -> list[dict[str, Any]]:
    rotation = seed % REGION_COUNT
    counts = _BASE_UAV_DISTRIBUTION[-rotation:] + _BASE_UAV_DISTRIBUTION[:-rotation] if rotation else _BASE_UAV_DISTRIBUTION
    block = [40, 40, 2, 2]
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for region, count in enumerate(counts):
        for _ in range(count):
            position = _position(rng, region)
            rows.append({
                "id": f"v3-uav-{ordinal:02d}", "type": "uav", "position": position,
                "local": _inside_local(position, block), "home_region": f"r{region // 3}{region % 3}",
                "capabilities": ["deliver", "inspect"], "capacity_kg": round(rng.uniform(7.0, 13.0), 3),
                "speed_km_min": round(rng.uniform(1.35, 1.95), 4),
                "energy_initial": round(rng.uniform(175.0, 220.0), 3),
                "energy_per_km": round(rng.uniform(0.72, 0.95), 4),
                "service_energy_per_min": round(rng.uniform(0.12, 0.22), 4),
            })
            ordinal += 1
    for ordinal in range(HELICOPTER_COUNT):
        region = (seed + ordinal * 4) % REGION_COUNT
        position = _position(rng, region)
        rows.append({
            "id": f"v3-heli-{ordinal:02d}", "type": "helicopter", "position": position,
            "local": _inside_local(position, block), "home_region": f"r{region // 3}{region % 3}",
            "capabilities": ["deliver", "inspect"], "capacity_kg": round(rng.uniform(24.0, 38.0), 3),
            "speed_km_min": round(rng.uniform(1.8, 2.5), 4),
            "energy_initial": round(rng.uniform(280.0, 360.0), 3),
            "energy_per_km": round(rng.uniform(0.88, 1.08), 4),
            "service_energy_per_min": round(rng.uniform(0.22, 0.34), 4),
        })
    return rows


def _tasks(rng: random.Random, block: list[int]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for region in range(REGION_COUNT):
        rid = f"r{region // 3}{region % 3}"
        depot_id = f"v3-depot-{region:02d}"
        for ordinal in range(6):
            is_delivery = ordinal in {0, 1, 4}
            initially_known = ordinal < 4
            position = _position(rng, region)
            index = region * 6 + ordinal
            task: dict[str, Any] = {
                "id": f"v3-task-{index:02d}", "kind": "delivery" if is_delivery else "inspection",
                "position": position, "region_id": rid, "local": _inside_local(position, block),
                "known_initial": initially_known, "deadline": 76 + (ordinal % 3) * 4,
                "priority": round(rng.uniform(2.0, 6.0), 3), "discovered_by": None, "report_delay": 0,
            }
            if is_delivery:
                task.update({
                    "commodity": "water" if ordinal == 0 else "medical",
                    "quantity": round(rng.uniform(2.0, 4.5), 2), "depot_id": depot_id,
                    "load_minutes": 1 + ordinal % 2, "unload_minutes": 1,
                })
            else:
                task["service_minutes"] = 2 + ordinal % 3
            tasks.append(task)
    return tasks


def _make_base(seed: int, parent_id: str) -> dict[str, Any]:
    rng = random.Random(seed)
    block = [40, 40, 2, 2]
    platforms = _platforms(rng, seed)
    tasks = _tasks(rng, block)
    depots = []
    for region in range(REGION_COUNT):
        position = _position(rng, region)
        depots.append({
            "id": f"v3-depot-{region:02d}", "position": position,
            "stock": {"water": 45.0, "medical": 45.0},
        })
    base = {
        "schema_version": "relief-engineering/1", "scenario_id": parent_id, "synthetic": True,
        "region_contract": REGION_CONTRACT,
        "units": {"distance": "km", "time": "min", "quantity": "kg", "energy": "synthetic_unit"},
        "map": {
            "grid_contract": MAP_CONTRACT, "rows": MAP_ROWS, "cols": MAP_COLS,
            "grid_cells": GRID_CELLS, "cell_km": 1.0, "local_block": block,
            "region_boundaries": list(REGION_BOUNDARIES),
        },
        "horizon": HORIZON_MINUTES, "dt": 1, "platforms": platforms, "depots": depots,
        "tasks": tasks,
        "chapter1_snapshot": {"source": C1_SNAPSHOT_SOURCE, "time": 0,
                              "queues": {platform["id"]: [] for platform in platforms}},
        "disturbances": [],
    }
    known_ids = {task["id"] for task in tasks if task["known_initial"]}
    queues, receipt = _multi_round_c1_assignment(base, known_ids, seed)
    base["chapter1_snapshot"]["queues"] = queues
    base["chapter1_snapshot"]["assignment_receipt"] = receipt
    validate_scenario(base)
    return base


def _event(event_seed: int, index: int, time: int, kind: str, **fields: Any) -> dict[str, Any]:
    return {"time": time, "kind": kind,
            "event_id": f"v3-event-{event_seed}-{index:03d}-{hashlib.sha256(kind.encode()).hexdigest()[:6]}", **fields}


def _communication_links(platforms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from .event_environment import EventReliefEnv
    except ImportError:
        from event_environment import EventReliefEnv
    return EventReliefEnv._regional_communication(platforms)


def _event_slots(condition: str) -> list[dict[str, Any]]:
    slots = []
    for region in range(REGION_COUNT):
        for nominal, ordinal in BASELINE_ARRIVALS:
            slots.append({"slot_id": f"arrival:r{region // 3}{region % 3}:{ordinal}",
                          "category": "arrival", "kind": "task_arrival", "region": region,
                          "nominal": nominal, "ordinal": ordinal})
    condition_slots = {
        "none_or_light": [],
        "task_update_or_arrival": [
            ("task_update", "priority_change", "task_update:priority"),
            ("task_update", "deadline_change", "task_update:deadline"),
        ],
        "platform_failure_or_capability_loss": [
            ("platform_failure", "platform_failure", f"platform_failure:{i}") for i in range(2)
        ],
        "energy_anomaly": [("energy_drop", "energy_fraction_drop", f"energy_drop:{i}") for i in range(2)],
        "communication_degradation": [("communication_drop", "communication_drop", f"communication_drop:{i}") for i in range(2)],
        "environment_or_obstacle_change": [
            ("environment_perturbation", "environment_perturbation", f"environment_perturbation:{i}") for i in range(2)
        ],
        "compound": [
            ("task_update", "priority_change", "task_update:priority"),
            ("platform_failure", "platform_failure", "platform_failure:0"),
            ("energy_drop", "energy_fraction_drop", "energy_drop:0"),
            ("communication_drop", "communication_drop", "communication_drop:0"),
            ("environment_perturbation", "environment_perturbation", "environment_perturbation:0"),
        ],
    }
    for category, kind, slot_id in condition_slots[condition]:
        slots.append({"slot_id": slot_id, "category": category, "kind": kind})
    return slots


def _event_tape(base: dict[str, Any], *, condition: str, seed: int) -> list[dict[str, Any]]:
    platforms = list(base["platforms"])
    public_tasks = [task for task in base["tasks"] if task["known_initial"]]
    links = _communication_links(platforms)
    slots = _event_slots(condition)
    target_pools = {
        "task_update": [row["id"] for row in public_tasks],
        "platform_failure": [row["id"] for row in platforms],
        "energy_drop": [row["id"] for row in platforms],
        "communication_drop": [tuple(sorted((row["source"], row["target"]))) for row in links],
        "environment_perturbation": list(range(REGION_COUNT)),
    }
    targets: dict[str, Any] = {}
    for category, pool in target_pools.items():
        category_slots = [row for row in slots if row["category"] == category]
        rng = random.Random(_stable_int(seed, "target", category))
        selected = rng.sample(pool, len(category_slots)) if category_slots else []
        targets.update({row["slot_id"]: target for row, target in zip(category_slots, selected)})

    rows = []
    for index, slot in enumerate(slots):
        slot_id, kind = slot["slot_id"], slot["kind"]
        time_rng = random.Random(_stable_int(seed, "time", slot_id))
        if slot["category"] == "arrival":
            time = int(slot["nominal"]) + time_rng.randint(-2, 2)
            task_id = f"v3-task-{int(slot['region']) * 6 + int(slot['ordinal']):02d}"
            event = _event(seed, index, time, kind, task_id=task_id)
        else:
            time = time_rng.randint(4, 84)
            category = slot["category"]
            target = targets[slot_id]
            if kind in {"priority_change", "deadline_change"}:
                task = next(row for row in public_tasks if row["id"] == target)
                if kind == "priority_change":
                    event = _event(seed, index, time, kind, task_id=target,
                                   new_priority=round(float(task["priority"]) * 1.5, 3))
                else:
                    event = _event(seed, index, time, kind, task_id=target,
                                   new_deadline=max(time + 5, int(task["deadline"]) - 12))
            elif category in {"platform_failure", "energy_drop"}:
                if category == "platform_failure":
                    event = _event(seed, index, time, kind, platform_id=target)
                else:
                    event = _event(seed, index, time, kind, platform_id=target, loss_fraction=0.45)
            elif category == "communication_drop":
                event = _event(seed, index, time, kind, links=[list(target)], quality=0.05,
                               latency_minutes=1.0, loss_rate=0.8)
            else:
                region = int(target)
                x0, y0, x1, y1 = _region_bounds(region)
                rng = random.Random(_stable_int(seed, "patch", slot_id))
                x, y = int(rng.uniform(x0 + 4, x1 - 6)), int(rng.uniform(y0 + 4, y1 - 6))
                event = _event(seed, index, time, kind, patch=[x, y, 2, 2],
                               wind_vector=[0.6, -0.4], risk_delta=0.2, blocked_cell=[x, y])
        event["slot_id"] = slot_id
        rows.append(event)
    rows.sort(key=lambda row: (int(row["time"]), str(row["event_id"])))
    return rows


def build_v3_gain_scenario(
    *, split: str, condition: str, seed_variant: int, instance_index: int = 0,
) -> dict[str, Any]:
    if split not in {"pilot", "train", "validation", "test", "ood"}:
        raise ValueError("V3_SCENARIO_SPLIT_INVALID")
    if condition not in {
        "none_or_light", "task_update_or_arrival", "platform_failure_or_capability_loss",
        "energy_anomaly", "communication_degradation", "environment_or_obstacle_change", "compound",
    }:
        raise ValueError("V3_SCENARIO_CONDITION_INVALID")
    if type(seed_variant) is not int or seed_variant < 0 or type(instance_index) is not int or instance_index < 0:
        raise ValueError("V3_SCENARIO_IDENTITY_INVALID")
    parent_id = f"v3-{split}-{condition}-{instance_index:03d}-{seed_variant:08d}"
    base = _make_base(seed_variant, parent_id)
    tape = _event_tape(base, condition=condition, seed=seed_variant)
    scenario = {
        "schema": SCENARIO_SCHEMA,
        "parent_id": parent_id, "parent_family": parent_id,
        "split": split, "condition": condition, "seed_variant": seed_variant,
        "map": copy.deepcopy(base["map"]), "regions": [f"r{row}{col}" for row in range(3) for col in range(3)],
        "base_scenario": base,
        "queue_plan": copy.deepcopy(base["chapter1_snapshot"]["queues"]),
        "event_tape": tape, "event_tape_id": _digest(tape),
    }
    scenario["scenario_content_sha256"] = scenario_content_sha256(scenario)
    validate_v3_scenario(scenario)
    return scenario


def validate_v3_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    event_view = {key: value for key, value in scenario.items()
                  if key in {"parent_id", "split", "condition", "seed_variant", "map", "base_scenario", "queue_plan", "event_tape", "event_tape_id"}}
    report = validate_event_scenario(event_view)
    base = scenario["base_scenario"]
    if base.get("region_contract") != REGION_CONTRACT:
        raise ValueError("V3_SCENARIO_REGION_CONTRACT")
    if len(base["platforms"]) != 30 or len(base["tasks"]) != TASK_COUNT:
        raise ValueError("V3_SCENARIO_FLEET_OR_TASK_COUNT")
    if sum(row["type"] == "uav" for row in base["platforms"]) != UAV_COUNT:
        raise ValueError("V3_SCENARIO_UAV_COUNT")
    if sum(row["type"] == "helicopter" for row in base["platforms"]) != HELICOPTER_COUNT:
        raise ValueError("V3_SCENARIO_HELICOPTER_COUNT")
    for region in range(REGION_COUNT):
        region_tasks = [task for task in base["tasks"] if task["region_id"] == f"r{region // 3}{region % 3}"]
        if len(region_tasks) != 6 or sum(task["kind"] == "delivery" for task in region_tasks) != 3:
            raise ValueError("V3_SCENARIO_REGIONAL_TASK_ALLOCATION")
        if sum(task["kind"] == "inspection" for task in region_tasks) != 3:
            raise ValueError("V3_SCENARIO_REGIONAL_TASK_MIX")
        if sum(bool(task["known_initial"]) for task in region_tasks) != 4:
            raise ValueError("V3_SCENARIO_REGIONAL_PUBLIC_BOUNDARY")
    if scenario.get("scenario_content_sha256") != scenario_content_sha256(scenario):
        raise ValueError("V3_SCENARIO_CONTENT_HASH")
    report["scenario_content_sha256"] = scenario["scenario_content_sha256"]
    return report


def sample_v3_future_suffix(
    scenario: dict[str, Any], *, after_time: int, horizon_minutes: int, suffix_seed: int,
) -> list[dict[str, Any]]:
    """Sample future events from the same fixed episode quota, conditional on prefix."""
    validate_v3_scenario(scenario)
    if type(after_time) is not int or after_time < 0 or type(horizon_minutes) is not int or horizon_minutes < 1:
        raise ValueError("V3_SUFFIX_TIME_INVALID")
    if type(suffix_seed) is not int or suffix_seed < 0:
        raise ValueError("V3_SUFFIX_SEED_INVALID")
    end = min(HORIZON_MINUTES - 1, after_time + horizon_minutes)
    if end <= after_time:
        return []
    # Only events already visible at the decision time may condition suffix sampling.
    prefix = [copy.deepcopy(row) for row in scenario["event_tape"] if int(row["time"]) <= after_time]
    # The same suffix seed and decision time define one episode suffix; horizon only crops it.
    seed = _stable_int(suffix_seed, scenario["condition"], after_time)
    slots = _event_slots(scenario["condition"])
    consumed = {str(row.get("slot_id")) for row in prefix if row.get("slot_id")}
    prefix_by_category: dict[str, list[dict[str, Any]]] = {}
    for row in prefix:
        slot = next((item for item in slots if item["slot_id"] == row.get("slot_id")), None)
        if slot and slot["category"] != "arrival":
            prefix_by_category.setdefault(slot["category"], []).append(row)

    base = scenario["base_scenario"]
    public_tasks = [row for row in base["tasks"] if row.get("known_initial")]
    platforms = list(base["platforms"])
    links = _communication_links(platforms)
    target_pools = {
        "task_update": [row["id"] for row in public_tasks],
        "platform_failure": [row["id"] for row in platforms],
        "energy_drop": [row["id"] for row in platforms],
        "communication_drop": [tuple(sorted((row["source"], row["target"]))) for row in links],
        "environment_perturbation": list(range(REGION_COUNT)),
    }
    sampled_targets: dict[str, Any] = {}
    for category, pool in target_pools.items():
        category_slots = [row for row in slots if row["category"] == category]
        prefix_rows = prefix_by_category.get(category, [])
        used: set[Any] = set()
        for row in prefix_rows:
            if category == "task_update":
                used.add(row.get("task_id"))
            elif category in {"platform_failure", "energy_drop"}:
                used.add(row.get("platform_id"))
            elif category == "communication_drop":
                used.update(tuple(sorted(pair)) for pair in row.get("links", []))
            elif row.get("patch"):
                used.add(region_id((float(row["patch"][0]) + 1, float(row["patch"][1]) + 1)))
        rng = random.Random(_stable_int(seed, "target", category))
        for slot in category_slots:
            if slot["slot_id"] in consumed:
                continue
            eligible = [target for target in pool if target not in used]
            if not eligible:
                continue
            target = rng.choice(eligible)
            used.add(target)
            sampled_targets[slot["slot_id"]] = target

    sampled_rows = []
    for index, slot in enumerate(slots):
        if slot["slot_id"] in consumed:
            continue
        time_rng = random.Random(_stable_int(seed, "time", slot["slot_id"]))
        if slot["category"] == "arrival":
            low = max(int(slot["nominal"]) - 2, after_time + 1)
            high = int(slot["nominal"]) + 2
        else:
            low, high = max(4, after_time + 1), 84
        if low > high:
            continue
        time = time_rng.randint(low, high)
        category, kind = slot["category"], slot["kind"]
        target = sampled_targets.get(slot["slot_id"])
        if category == "arrival":
            task_id = f"v3-task-{int(slot['region']) * 6 + int(slot['ordinal']):02d}"
            event = _event(seed, index, time, kind, task_id=task_id)
        elif category == "task_update":
            task = next(row for row in public_tasks if row["id"] == target)
            if kind == "priority_change":
                event = _event(seed, index, time, kind, task_id=target,
                               new_priority=round(float(task["priority"]) * 1.5, 3))
            else:
                event = _event(seed, index, time, kind, task_id=target,
                               new_deadline=max(time + 5, int(task["deadline"]) - 12))
        elif category in {"platform_failure", "energy_drop"}:
            if category == "platform_failure":
                event = _event(seed, index, time, kind, platform_id=target)
            else:
                event = _event(seed, index, time, kind, platform_id=target, loss_fraction=0.45)
        elif category == "communication_drop":
            event = _event(seed, index, time, kind, links=[list(target)], quality=0.05,
                           latency_minutes=1.0, loss_rate=0.8)
        else:
            region = int(target)
            x0, y0, x1, y1 = _region_bounds(region)
            patch_rng = random.Random(_stable_int(seed, "patch", slot["slot_id"]))
            x, y = int(patch_rng.uniform(x0 + 4, x1 - 6)), int(patch_rng.uniform(y0 + 4, y1 - 6))
            event = _event(seed, index, time, kind, patch=[x, y, 2, 2],
                           wind_vector=[0.6, -0.4], risk_delta=0.2, blocked_cell=[x, y])
        event["slot_id"] = slot["slot_id"]
        if time <= end:
            sampled_rows.append(event)
    rows = sampled_rows
    rows.sort(key=lambda row: (int(row["time"]), str(row["event_id"])))
    return rows


def run_noop_episode(scenario: dict[str, Any]) -> dict[str, Any]:
    """Run a full public NOOP episode and return the simulator trajectory."""
    env = EventReliefEnv(scenario)
    observation = env.reset()
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action = {"action_type": "reconstruct", "scope": "NOOP", "noop": True,
                  "queue_operations": [], "dispatches": []}
        observation, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if steps > HORIZON_MINUTES + 1:
            raise RuntimeError("V3_SMOKE_EPISODE_DID_NOT_TERMINATE")
    return {"env": env, "final_observation": observation, "trajectory": env.export_trajectory(), "steps": steps}
