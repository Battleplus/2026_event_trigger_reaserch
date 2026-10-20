"""Civilian search and rescue scenarios for the V3.3 multi-region study."""

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
    from .region_contract import REGION_BOUNDARIES, REGION_CONTRACT, region_id
    from .scenario_schema import C1_SNAPSHOT_SOURCE, GRID_CELLS, MAP_COLS, MAP_CONTRACT, MAP_ROWS, validate_scenario
except ImportError:
    from event_environment import EventReliefEnv
    from event_scenarios import _digest, validate_event_scenario
    from gain_scenarios import _multi_round_c1_assignment, scenario_content_sha256
    from region_contract import REGION_BOUNDARIES, REGION_CONTRACT, region_id
    from scenario_schema import C1_SNAPSHOT_SOURCE, GRID_CELLS, MAP_COLS, MAP_CONTRACT, MAP_ROWS, validate_scenario


SCENARIO_SCHEMA = "gain-gate-v33-civilian-scenario/1"
HORIZON_MINUTES = 90
REGION_COUNT = 9
UAV_COUNT = 18
HELICOPTER_COUNT = 18
PLATFORM_COUNT = UAV_COUNT + HELICOPTER_COUNT
DEFAULT_TARGETS_PER_REGION = 3
DEFAULT_TASK_MODE = "search_rescue"
TASK_MODES = ("search_only", "search_rescue")
TARGET_DENSITIES = (1, 3, 6)
CONDITIONS = (
    "none_or_light", "task_update_or_arrival", "platform_failure_or_capability_loss",
    "energy_anomaly", "communication_degradation", "environment_or_obstacle_change", "compound",
)


def _stable_int(seed: int, *parts: Any) -> int:
    raw = json.dumps([seed, *parts], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _region_bounds(region: int) -> tuple[float, float, float, float]:
    row, col = divmod(region, 3)
    return (
        float(REGION_BOUNDARIES[col]), float(REGION_BOUNDARIES[row]),
        float(REGION_BOUNDARIES[col + 1]), float(REGION_BOUNDARIES[row + 1]),
    )


def _position(rng: random.Random, region: int) -> list[float]:
    x0, y0, x1, y1 = _region_bounds(region)
    return [round(rng.uniform(x0 + 4.0, x1 - 4.0), 3), round(rng.uniform(y0 + 4.0, y1 - 4.0), 3)]


def _local(position: list[float], block: list[int]) -> bool:
    x, y, width, height = block
    return x <= position[0] < x + width and y <= position[1] < y + height


def _platforms(rng: random.Random) -> list[dict[str, Any]]:
    block = [40, 40, 2, 2]
    rows = []
    for region in range(REGION_COUNT):
        home = f"r{region // 3}{region % 3}"
        for kind, prefix, capacity, speed, energy, energy_rate, service_rate in (
            ("uav", "uav", (7.0, 13.0), (1.35, 1.95), (175.0, 220.0), (0.72, 0.95), (0.12, 0.22)),
            ("helicopter", "heli", (24.0, 38.0), (1.8, 2.5), (280.0, 360.0), (0.88, 1.08), (0.22, 0.34)),
        ):
            for local_index in range(2):
                position = _position(rng, region)
                rows.append({
                    "id": f"v33-{prefix}-r{region:02d}-{local_index}", "type": kind,
                    "position": position, "local": _local(position, block), "home_region": home,
                    "capabilities": ["inspect", "deliver"],
                    "capacity_kg": round(rng.uniform(*capacity), 3),
                    "speed_km_min": round(rng.uniform(*speed), 4),
                    "energy_initial": round(rng.uniform(*energy), 3),
                    "energy_per_km": round(rng.uniform(*energy_rate), 4),
                    "service_energy_per_min": round(rng.uniform(*service_rate), 4),
                })
    return rows


def _tasks(seed: int, *, task_mode: str, targets_per_region: int, block: list[int]) -> list[dict[str, Any]]:
    searches: list[dict[str, Any]] = []
    rescues: list[dict[str, Any]] = []
    for region in range(REGION_COUNT):
        rid = f"r{region // 3}{region % 3}"
        depot_id = f"v33-depot-{region:02d}"
        for site in range(targets_per_region):
            search_rng = random.Random(_stable_int(seed, "search-site", region, site))
            position = _position(search_rng, region)
            search_id = f"v33-search-r{region:02d}-s{site:02d}"
            arrives_later = site == 0
            searches.append({
                "id": search_id, "kind": "inspection", "position": position,
                "region_id": rid, "local": _local(position, block), "known_initial": not arrives_later,
                "deadline": 82, "priority": round(search_rng.uniform(3.0, 6.0), 3),
                "discovered_by": None, "report_delay": 0, "service_minutes": search_rng.randint(2, 4),
            })
            if task_mode == "search_rescue":
                rescue_rng = random.Random(_stable_int(seed, "rescue-demand", region, site))
                rescue_position = [round(min(99.9, position[0] + 0.5), 3), round(min(99.9, position[1] + 0.5), 3)]
                rescues.append({
                    "id": f"v33-rescue-r{region:02d}-s{site:02d}", "kind": "delivery",
                    "position": rescue_position, "region_id": rid,
                    "local": _local(rescue_position, block), "known_initial": False,
                    "deadline": 88, "priority": round(rescue_rng.uniform(4.0, 7.0), 3),
                    "discovered_by": search_id, "report_delay": 0,
                    "commodity": "water", "quantity": round(rescue_rng.uniform(4.0, 9.0), 2),
                    "depot_id": depot_id, "load_minutes": 1, "unload_minutes": 1,
                })
    return searches + rescues


def _event_slots(condition: str, *, public_task_available: bool = True) -> list[dict[str, Any]]:
    arrivals = [
        ("task_arrival", "task_arrival", f"task_arrival:{region}", f"r{region // 3}{region % 3}")
        for region in range(REGION_COUNT)
    ]
    condition_slots = {
        "none_or_light": [],
        "task_update_or_arrival": [("task_update", "deadline_change", "task_update:deadline")],
        "platform_failure_or_capability_loss": [
            ("platform_failure", "platform_failure", f"platform_failure:{i}") for i in range(2)
        ],
        "energy_anomaly": [("energy_drop", "energy_fraction_drop", f"energy_drop:{i}") for i in range(2)],
        "communication_degradation": [
            ("communication_drop", "communication_drop", f"communication_drop:{i}") for i in range(2)
        ],
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
    if condition not in condition_slots:
        raise ValueError("V33_SCENARIO_CONDITION_INVALID")
    slots: list[dict[str, Any]] = []
    for slot_definition in arrivals + condition_slots[condition]:
        category, kind, slot_id = slot_definition[:3]
        if category == "task_update" and not public_task_available:
            continue
        slot = {"slot_id": slot_id, "category": category, "kind": kind}
        if len(slot_definition) == 4:
            slot["target_region"] = slot_definition[3]
        slots.append(slot)
    return slots


def _communication_links(platforms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return EventReliefEnv._regional_communication(platforms)


def _draw_event(
    *, seed: int, ordinal: int, slot: dict[str, Any], time: int,
    target: Any, public_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    kind = slot["kind"]
    if slot["category"] == "task_update":
        task = next(row for row in public_tasks if row["id"] == target)
        if kind == "priority_change":
            fields = {"task_id": target, "new_priority": round(float(task["priority"]) * 1.5, 3)}
        else:
            fields = {"task_id": target, "new_deadline": max(time + 5, int(task["deadline"]) - 12)}
    elif slot["category"] == "task_arrival":
        fields = {"task_id": target}
    elif slot["category"] == "platform_failure":
        fields = {"platform_id": target}
    elif slot["category"] == "energy_drop":
        fields = {"platform_id": target, "loss_fraction": 0.45}
    elif slot["category"] == "communication_drop":
        fields = {"links": [list(target)], "quality": 0.05, "latency_minutes": 1.0, "loss_rate": 0.8}
    else:
        x0, y0, x1, y1 = _region_bounds(int(target))
        patch_rng = random.Random(_stable_int(seed, "patch", slot["slot_id"]))
        x, y = int(patch_rng.uniform(x0 + 4, x1 - 6)), int(patch_rng.uniform(y0 + 4, y1 - 6))
        fields = {"patch": [x, y, 2, 2], "wind_vector": [0.6, -0.4], "risk_delta": 0.2, "blocked_cell": [x, y]}
    digest = hashlib.sha256(str(kind).encode("ascii")).hexdigest()[:6]
    return {
        "time": time, "kind": kind,
        "event_id": f"v33-event-{seed}-{ordinal:03d}-{digest}",
        "slot_id": slot["slot_id"], **fields,
    }


def _sample_targets(
    seed: int, slots: list[dict[str, Any]], base: dict[str, Any],
    used_by_category: dict[str, set[Any]] | None = None,
) -> dict[str, Any]:
    platforms = list(base["platforms"])
    public_tasks = [row for row in base["tasks"] if row["known_initial"]]
    links = _communication_links(platforms)
    pools: dict[str, list[Any]] = {
        "task_update": [row["id"] for row in public_tasks],
        "task_arrival": [
            row["id"] for row in base["tasks"]
            if row["kind"] == "inspection" and not row["known_initial"] and row.get("discovered_by") is None
        ],
        "platform_failure": [row["id"] for row in platforms],
        "energy_drop": [row["id"] for row in platforms],
        "communication_drop": [tuple(sorted((row["source"], row["target"]))) for row in links],
        "environment_perturbation": list(range(REGION_COUNT)),
    }
    result: dict[str, Any] = {}
    used_by_category = used_by_category or {}
    for category in sorted({row["category"] for row in slots}):
        category_slots = [row for row in slots if row["category"] == category]
        if category == "task_arrival":
            for slot in category_slots:
                target = next(
                    row["id"] for row in base["tasks"]
                    if row["kind"] == "inspection" and not row["known_initial"]
                    and row.get("discovered_by") is None and row["region_id"] == slot["target_region"]
                )
                result[slot["slot_id"]] = target
            continue
        pool = [item for item in pools[category] if item not in used_by_category.get(category, set())]
        rng = random.Random(_stable_int(seed, "target", category))
        selected = rng.sample(pool, min(len(pool), len(category_slots)))
        result.update({row["slot_id"]: target for row, target in zip(category_slots, selected)})
    return result


def _event_tape(base: dict[str, Any], *, condition: str, seed: int) -> list[dict[str, Any]]:
    slots = _event_slots(condition, public_task_available=any(row["known_initial"] for row in base["tasks"]))
    targets = _sample_targets(seed, slots, base)
    public_tasks = [row for row in base["tasks"] if row["known_initial"]]
    rows = []
    for ordinal, slot in enumerate(slots):
        if slot["category"] == "task_arrival":
            time = random.Random(_stable_int(seed, "time", slot["slot_id"])).randint(22, 26)
        else:
            time = random.Random(_stable_int(seed, "time", slot["slot_id"])).randint(4, 84)
        event = _draw_event(seed=seed, ordinal=ordinal, slot=slot, time=time,
                            target=targets[slot["slot_id"]], public_tasks=public_tasks)
        rows.append(event)
    rows.sort(key=lambda row: (int(row["time"]), str(row["event_id"])))
    return rows


def _make_base(seed: int, parent_id: str, *, task_mode: str, targets_per_region: int) -> dict[str, Any]:
    block = [40, 40, 2, 2]
    platforms = _platforms(random.Random(_stable_int(seed, "platforms")))
    tasks = _tasks(seed, task_mode=task_mode, targets_per_region=targets_per_region, block=block)
    depot_rng = random.Random(_stable_int(seed, "depots"))
    depots = []
    for region in range(REGION_COUNT):
        position = _position(depot_rng, region)
        depots.append({
            "id": f"v33-depot-{region:02d}", "position": position,
            "stock": {"water": 80.0, "medical": 40.0},
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


def make_v33_scenario(
    *, split: str, condition: str, seed_variant: int, instance_index: int,
    task_mode: str = DEFAULT_TASK_MODE, targets_per_region: int = DEFAULT_TARGETS_PER_REGION,
) -> dict[str, Any]:
    if split not in {"pilot", "train", "validation", "test", "ood"}:
        raise ValueError("V33_SCENARIO_SPLIT_INVALID")
    if condition not in CONDITIONS:
        raise ValueError("V33_SCENARIO_CONDITION_INVALID")
    if task_mode not in TASK_MODES or targets_per_region not in TARGET_DENSITIES:
        raise ValueError("V33_SCENARIO_DESIGN_INVALID")
    if type(seed_variant) is not int or seed_variant < 0 or type(instance_index) is not int or instance_index < 0:
        raise ValueError("V33_SCENARIO_IDENTITY_INVALID")
    parent_id = f"v33-{split}-{condition}-{task_mode}-n{targets_per_region}-{instance_index:03d}-{seed_variant:08d}"
    base = _make_base(seed_variant, parent_id, task_mode=task_mode, targets_per_region=targets_per_region)
    tape = _event_tape(base, condition=condition, seed=seed_variant)
    scenario = {
        "schema": SCENARIO_SCHEMA, "parent_id": parent_id, "parent_family": parent_id,
        "split": split, "condition": condition, "seed_variant": seed_variant,
        "task_mode": task_mode, "targets_per_region": targets_per_region,
        "map": copy.deepcopy(base["map"]),
        "regions": [f"r{row}{col}" for row in range(3) for col in range(3)],
        "base_scenario": base,
        "queue_plan": copy.deepcopy(base["chapter1_snapshot"]["queues"]),
        "event_tape": tape, "event_tape_id": _digest(tape),
    }
    scenario["scenario_content_sha256"] = scenario_content_sha256(scenario)
    validate_v33_scenario(scenario)
    return scenario


def validate_v33_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    if scenario.get("schema") != SCENARIO_SCHEMA:
        raise ValueError("V33_SCENARIO_SCHEMA_INVALID")
    event_view = {key: scenario[key] for key in (
        "parent_id", "split", "condition", "seed_variant", "map", "base_scenario",
        "queue_plan", "event_tape", "event_tape_id",
    )}
    report = validate_event_scenario(event_view)
    base = scenario["base_scenario"]
    mode, targets = scenario.get("task_mode"), scenario.get("targets_per_region")
    if mode not in TASK_MODES or targets not in TARGET_DENSITIES:
        raise ValueError("V33_SCENARIO_DESIGN_INVALID")
    if base.get("region_contract") != REGION_CONTRACT or len(base["platforms"]) != PLATFORM_COUNT:
        raise ValueError("V33_SCENARIO_FLEET_INVALID")
    if sum(row["type"] == "uav" for row in base["platforms"]) != UAV_COUNT:
        raise ValueError("V33_SCENARIO_UAV_COUNT_INVALID")
    if sum(row["type"] == "helicopter" for row in base["platforms"]) != HELICOPTER_COUNT:
        raise ValueError("V33_SCENARIO_HELICOPTER_COUNT_INVALID")
    expected_tasks = REGION_COUNT * targets * (2 if mode == "search_rescue" else 1)
    if len(base["tasks"]) != expected_tasks or int(base.get("horizon", -1)) != HORIZON_MINUTES:
        raise ValueError("V33_SCENARIO_TASK_OR_HORIZON_INVALID")
    for region in range(REGION_COUNT):
        rid = f"r{region // 3}{region % 3}"
        fleet = [row for row in base["platforms"] if row["home_region"] == rid]
        if sum(row["type"] == "uav" for row in fleet) != 2 or sum(row["type"] == "helicopter" for row in fleet) != 2:
            raise ValueError("V33_SCENARIO_REGIONAL_FLEET_INVALID")
        local_tasks = [row for row in base["tasks"] if row["region_id"] == rid]
        searches = [row for row in local_tasks if row["kind"] == "inspection"]
        initial_searches = [row for row in searches if row["known_initial"]]
        arrival_searches = [row for row in searches if not row["known_initial"]]
        rescues = [row for row in local_tasks if row["kind"] == "delivery"]
        if (len(initial_searches) != targets - 1 or len(arrival_searches) != 1
                or any(row.get("discovered_by") is not None for row in initial_searches)
                or any(row.get("discovered_by") is not None for row in arrival_searches)):
            raise ValueError("V33_SCENARIO_SEARCH_TASKS_INVALID")
        if mode == "search_rescue":
            if len(rescues) != targets or any(row["known_initial"] for row in rescues):
                raise ValueError("V33_SCENARIO_RESCUE_TASKS_INVALID")
            if {row.get("discovered_by") for row in rescues} != {row["id"] for row in searches}:
                raise ValueError("V33_SCENARIO_RESCUE_DISCOVERY_LINK_INVALID")
        elif rescues:
            raise ValueError("V33_SCENARIO_SEARCH_ONLY_HAS_RESCUE")
    known_ids = {row["id"] for row in base["tasks"] if row["known_initial"]}
    arrivals = [row for row in scenario["event_tape"] if row["kind"] == "task_arrival"]
    deadline_updates = [row for row in scenario["event_tape"] if row["kind"] == "deadline_change"]
    hidden_searches = {
        row["region_id"]: row["id"] for row in base["tasks"]
        if row["kind"] == "inspection" and not row["known_initial"]
    }
    arrived = {row["task_id"]: row for row in arrivals}
    if set(arrived) != set(hidden_searches.values()) or len(arrivals) != REGION_COUNT:
        raise ValueError("V33_SCENARIO_REGIONAL_ARRIVAL_COVERAGE_INVALID")
    for task_id, event in arrived.items():
        task = next(row for row in base["tasks"] if row["id"] == task_id)
        if (task["region_id"] not in hidden_searches
                or hidden_searches[task["region_id"]] != task_id
                or not 22 <= int(event["time"]) <= 26):
            raise ValueError("V33_SCENARIO_REGIONAL_ARRIVAL_INVALID")
    if scenario["condition"] == "task_update_or_arrival":
        if len(deadline_updates) != int(bool(known_ids)) or any(
                row.get("task_id") not in known_ids for row in deadline_updates):
            raise ValueError("V33_SCENARIO_DEADLINE_UPDATE_INVALID")
    elif deadline_updates:
        raise ValueError("V33_SCENARIO_UNEXPECTED_DEADLINE_UPDATE")
    if any(row.get("task_id") not in known_ids for row in deadline_updates):
        raise ValueError("V33_SCENARIO_HIDDEN_TASK_UPDATE_LEAK")
    if scenario.get("queue_plan") != base["chapter1_snapshot"]["queues"]:
        raise ValueError("V33_SCENARIO_C1_QUEUE_BINDING_INVALID")
    expected_public = {row["id"] for row in base["tasks"] if row["known_initial"]}
    queued = [task_id for queue in scenario["queue_plan"].values() for task_id in queue]
    if not set(queued) <= expected_public or len(queued) != len(set(queued)):
        raise ValueError("V33_SCENARIO_C1_PUBLIC_TASK_LEAK")
    if scenario.get("scenario_content_sha256") != scenario_content_sha256(scenario):
        raise ValueError("V33_SCENARIO_CONTENT_HASH_INVALID")
    report.update({
        "schema": SCENARIO_SCHEMA, "platform_count": PLATFORM_COUNT,
        "uav_count": UAV_COUNT, "helicopter_count": HELICOPTER_COUNT,
        "task_count": expected_tasks, "task_mode": mode, "targets_per_region": targets,
    })
    return report


def build_v33_pilot_scenarios(*, seed_variants: tuple[int, ...] = (930101, 930102, 930103)) -> list[dict[str, Any]]:
    """Build 18 paired task-update/arrival scenarios across mode, density, and seed."""
    return [
        make_v33_scenario(
            split="pilot", condition="task_update_or_arrival", seed_variant=seed,
            instance_index=seed_index, task_mode=mode, targets_per_region=targets,
        )
        for seed_index, seed in enumerate(seed_variants)
        for mode in TASK_MODES
        for targets in TARGET_DENSITIES
    ]


def sample_v33_future_suffix(
    scenario: dict[str, Any], *, after_time: int, horizon_minutes: int, suffix_seed: int,
) -> list[dict[str, Any]]:
    """Sample only future exogenous events, conditioned on the observed event prefix."""
    validate_v33_scenario(scenario)
    if type(after_time) is not int or after_time < 0 or type(horizon_minutes) is not int or horizon_minutes < 1:
        raise ValueError("V33_SUFFIX_TIME_INVALID")
    if type(suffix_seed) is not int or suffix_seed < 0:
        raise ValueError("V33_SUFFIX_SEED_INVALID")
    end = min(HORIZON_MINUTES - 1, after_time + horizon_minutes)
    if end <= after_time:
        return []
    prefix = [row for row in scenario["event_tape"] if int(row["time"]) <= after_time]
    slots = _event_slots(
        str(scenario["condition"]),
        public_task_available=any(row["known_initial"] for row in scenario["base_scenario"]["tasks"]),
    )
    consumed = {str(row.get("slot_id")) for row in prefix}
    used: dict[str, set[Any]] = {}
    for row in prefix:
        slot = next((item for item in slots if item["slot_id"] == row.get("slot_id")), None)
        if slot is None:
            continue
        category = slot["category"]
        if category == "task_update":
            used.setdefault(category, set()).add(row.get("task_id"))
        elif category in {"platform_failure", "energy_drop"}:
            used.setdefault(category, set()).add(row.get("platform_id"))
        elif category == "communication_drop":
            used.setdefault(category, set()).update(tuple(sorted(pair)) for pair in row.get("links", []))
        elif row.get("patch"):
            rid = region_id((float(row["patch"][0]) + 1, float(row["patch"][1]) + 1))
            used.setdefault(category, set()).add(int(rid[1]) * 3 + int(rid[2]))
    remaining = [slot for slot in slots if slot["slot_id"] not in consumed]
    targets = _sample_targets(_stable_int(suffix_seed, scenario["condition"], after_time), remaining,
                              scenario["base_scenario"], used)
    public_tasks = [row for row in scenario["base_scenario"]["tasks"] if row["known_initial"]]
    rows = []
    for ordinal, slot in enumerate(remaining):
        if slot["category"] == "task_arrival":
            low, high = max(22, after_time + 1), 26
        else:
            low, high = max(4, after_time + 1), 84
        if low > high:
            continue
        time = random.Random(_stable_int(suffix_seed, "time", after_time, slot["slot_id"])).randint(low, high)
        if time > end or slot["slot_id"] not in targets:
            continue
        row = _draw_event(
            seed=suffix_seed, ordinal=ordinal, slot=slot, time=time,
            target=targets[slot["slot_id"]], public_tasks=public_tasks,
        )
        rows.append(row)
    return sorted(rows, key=lambda row: (int(row["time"]), str(row["event_id"])))


def run_v33_noop_episode(scenario: dict[str, Any]) -> dict[str, Any]:
    """Run a full 90-minute NOOP episode for the focused engineering check."""
    env = EventReliefEnv(scenario)
    observation = env.reset()
    terminated = truncated = False
    steps = 0
    noop = {"action_type": "reconstruct", "scope": "NOOP", "noop": True, "queue_operations": [], "dispatches": []}
    while not (terminated or truncated):
        observation, _, terminated, truncated, _ = env.step(copy.deepcopy(noop))
        steps += 1
    return {"env": env, "final_observation": observation, "trajectory": env.export_trajectory(), "steps": steps}
