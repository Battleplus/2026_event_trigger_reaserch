"""Gain-study scenarios and a counter-addressed future suffix sampler."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from random import Random
from typing import Any

try:
    from .chapter1_adapter import generate_public_c1_initial_assignment
    from .event_scenarios import (
        _digest as _event_digest,
        build_event_scenarios,
    )
    from .stable_identity import key_digest
    from .scenario_schema import validate_scenario
except ImportError:
    from chapter1_adapter import generate_public_c1_initial_assignment
    from event_scenarios import _digest as _event_digest, build_event_scenarios
    from stable_identity import key_digest
    from scenario_schema import validate_scenario


GAIN_SPLITS = frozenset({"pilot", "train", "validation", "test", "ood"})
FORMAL_CONDITIONS = (
    "none_or_light",
    "task_update_or_arrival",
    "platform_failure_or_capability_loss",
    "energy_anomaly",
    "communication_degradation",
    "environment_or_obstacle_change",
    "compound",
)
DEFAULT_SPLIT_COUNTS = {"train": 40, "validation": 12, "test": 20}
SAMPLER_VERSION = "gain-future-suffix-keyed-v1"
DEFAULT_SUFFIX_RATES = {
    "task_arrival_per_minute": 0.05,
    "task_update_per_minute_per_task": 0.005,
    "platform_event_per_minute_per_platform": 0.003,
    "energy_anomaly_per_minute_per_platform": 0.002,
    "communication_event_per_minute_per_link": 0.003,
    "environment_event_per_minute_per_region": 0.002,
}
DGP_PROFILE_MULTIPLIERS = {
    "none_or_light": {
        "task_arrival_per_minute": 0.05, "task_update_per_minute_per_task": 0.05,
        "platform_event_per_minute_per_platform": 0.05, "energy_anomaly_per_minute_per_platform": 0.05,
        "communication_event_per_minute_per_link": 0.05, "environment_event_per_minute_per_region": 0.05,
    },
    "task_update_or_arrival": {
        "task_arrival_per_minute": 3.0, "task_update_per_minute_per_task": 3.0,
        "platform_event_per_minute_per_platform": 0.05, "energy_anomaly_per_minute_per_platform": 0.05,
        "communication_event_per_minute_per_link": 0.05, "environment_event_per_minute_per_region": 0.05,
    },
    "platform_failure_or_capability_loss": {
        "task_arrival_per_minute": 0.05, "task_update_per_minute_per_task": 0.05,
        "platform_event_per_minute_per_platform": 3.0, "energy_anomaly_per_minute_per_platform": 0.05,
        "communication_event_per_minute_per_link": 0.05, "environment_event_per_minute_per_region": 0.05,
    },
    "energy_anomaly": {
        "task_arrival_per_minute": 0.05, "task_update_per_minute_per_task": 0.05,
        "platform_event_per_minute_per_platform": 0.05, "energy_anomaly_per_minute_per_platform": 3.0,
        "communication_event_per_minute_per_link": 0.05, "environment_event_per_minute_per_region": 0.05,
    },
    "communication_degradation": {
        "task_arrival_per_minute": 0.05, "task_update_per_minute_per_task": 0.05,
        "platform_event_per_minute_per_platform": 0.05, "energy_anomaly_per_minute_per_platform": 0.05,
        "communication_event_per_minute_per_link": 3.0, "environment_event_per_minute_per_region": 0.05,
    },
    "environment_or_obstacle_change": {
        "task_arrival_per_minute": 0.05, "task_update_per_minute_per_task": 0.05,
        "platform_event_per_minute_per_platform": 0.05, "energy_anomaly_per_minute_per_platform": 0.05,
        "communication_event_per_minute_per_link": 0.05, "environment_event_per_minute_per_region": 3.0,
    },
    "compound": {key: 2.0 for key in DEFAULT_SUFFIX_RATES},
}

_TEMPLATE_CONDITION = {
    "none_or_light": "new_arrival",
    "task_update_or_arrival": "new_arrival",
    "platform_failure_or_capability_loss": "platform_failure",
    "energy_anomaly": "energy_drop",
    "communication_degradation": "communication_drop",
    "environment_or_obstacle_change": "environment_perturbation",
    "compound": "mixed_events",
}
_LEGACY_TO_GAIN = {
    "new_arrival": "task_update_or_arrival",
    "platform_failure": "platform_failure_or_capability_loss",
    "energy_drop": "energy_anomaly",
    "communication_drop": "communication_degradation",
    "environment_perturbation": "environment_or_obstacle_change",
    "mixed_events": "compound",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def scenario_content_projection(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Return identity-free physical scenario content for split-leakage checks."""
    base = copy.deepcopy(dict(scenario["base_scenario"]))
    base.pop("scenario_id", None)
    snapshot = base.get("chapter1_snapshot")
    if isinstance(snapshot, dict):
        snapshot.pop("assignment_receipt", None)
    tape = [
        {key: copy.deepcopy(value) for key, value in row.items() if key != "event_id"}
        for row in scenario["event_tape"]
    ]
    return {
        "map": copy.deepcopy(scenario["map"]),
        "base_scenario": base,
        "queue_plan": copy.deepcopy(scenario["queue_plan"]),
        "event_tape_effects": tape,
    }


def scenario_content_sha256(scenario: Mapping[str, Any]) -> str:
    return _digest(scenario_content_projection(scenario))


def _key(seed: int, event_class: str, entity_id: str, minute: int, draw_kind: str) -> bytes:
    return hashlib.sha256(f"{seed}|{event_class}|{entity_id}|{minute}|{draw_kind}".encode("utf-8")).digest()


def _uniform(seed: int, event_class: str, entity_id: str, minute: int, draw_kind: str) -> float:
    return int.from_bytes(_key(seed, event_class, entity_id, minute, draw_kind)[:8], "big") / 2**64


def _new_event_id(seed: int, event_class: str, entity_id: str, minute: int) -> str:
    return f"gain-suffix-{seed}-{minute:03d}-{event_class}-{_digest(entity_id)[:8]}"


def _sample_position(rng: Random, local: bool, local_block: list[int]) -> list[float]:
    x, y, width, height = local_block
    if local:
        return [rng.uniform(x + 0.05, x + width - 0.05), rng.uniform(y + 0.05, y + height - 0.05)]
    # Keep remote missions cross-region yet feasible for the declared energy
    # envelope; local and remote blocks remain spatially distinct.
    return [rng.uniform(70.05, 98.95), rng.uniform(70.05, 98.95)]


def dgp_profile_for_condition(condition: str) -> dict[str, Any]:
    if condition not in DGP_PROFILE_MULTIPLIERS:
        raise ValueError(f"unknown DGP condition {condition!r}")
    multipliers = dict(DGP_PROFILE_MULTIPLIERS[condition])
    rates = {
        key: min(1.0, float(DEFAULT_SUFFIX_RATES[key]) * float(multipliers[key]))
        for key in DEFAULT_SUFFIX_RATES
    }
    return {
        "schema": "gain-event-dgp-profile/1",
        "condition": condition,
        "sampler_version": SAMPLER_VERSION,
        "base_rates": dict(DEFAULT_SUFFIX_RATES),
        "multipliers": multipliers,
        "rates": rates,
    }


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


def _multi_round_c1_assignment(base: dict, known_task_ids: set[str], seed: int) -> tuple[dict, dict]:
    """Preserve source C1 receipts while projecting cumulative fleet resources."""
    platform_state = {row["id"]: copy.deepcopy(row) for row in base["platforms"]}
    platform_time = {platform_id: 0.0 for platform_id in platform_state}
    depot_state = {row["id"]: copy.deepcopy(row) for row in base["depots"]}
    tasks_by_id = {row["id"]: copy.deepcopy(row) for row in base["tasks"]}
    remaining = set(known_task_ids)
    queues = {platform_id: [] for platform_id in sorted(platform_state)}
    rounds: list[dict[str, Any]] = []
    task_projection: dict[str, dict[str, Any]] = {}
    final_unassigned_reasons: dict[str, list[str]] = {}
    total_energy_used = {platform_id: 0.0 for platform_id in platform_state}
    total_stock_used: dict[tuple[str, str], float] = {}

    while remaining:
        round_scenario = copy.deepcopy(base)
        round_scenario["platforms"] = [copy.deepcopy(platform_state[row["id"]]) for row in base["platforms"]]
        round_scenario["depots"] = [copy.deepcopy(depot_state[row["id"]]) for row in base["depots"]]
        for task in round_scenario["tasks"]:
            task["known_initial"] = task["id"] in remaining
        public_input = {
            "schema": "civilian-public-c1-input/1", "seed": int(seed),
            "map": copy.deepcopy(round_scenario["map"]),
            "platforms": copy.deepcopy(round_scenario["platforms"]),
            "depots": copy.deepcopy(round_scenario["depots"]),
            "known_tasks": copy.deepcopy([task for task in round_scenario["tasks"] if task["known_initial"]]),
        }
        assignment = generate_public_c1_initial_assignment(round_scenario, seed=int(seed))
        raw_receipt = copy.deepcopy(assignment["receipt"])
        if key_digest(public_input) != raw_receipt["public_input_sha256"]:
            raise AssertionError("C1 round input digest does not match preserved source receipt")
        rounds.append({
            "round_index": len(rounds),
            "known_task_ids": sorted(remaining),
            "public_input": public_input,
            "c1_receipt": raw_receipt,
        })
        selected = {task_id for queue in assignment["queues"].values() for task_id in queue}
        if not selected:
            final_unassigned_reasons = copy.deepcopy(raw_receipt["feasibility"]["unassigned_known_task_reasons"])
            break
        if not selected <= remaining:
            raise ValueError("public C1 round returned a task outside its known input")

        round_platform_by_id = {row["id"]: row for row in round_scenario["platforms"]}
        round_depot_by_id = {row["id"]: row for row in round_scenario["depots"]}
        for platform_id in sorted(assignment["queues"]):
            for task_id in assignment["queues"][platform_id]:
                task = tasks_by_id[task_id]
                platform = round_platform_by_id[platform_id]
                duration, energy = _mission_projection(platform, task, round_depot_by_id)
                energy_before = float(platform_state[platform_id]["energy_initial"])
                energy_after = energy_before - energy
                if energy_after < -1e-8:
                    raise AssertionError("C1 round selection exceeded cumulative platform energy")
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
                platform_state[platform_id]["position"] = copy.deepcopy(task["position"])
                platform_state[platform_id]["energy_initial"] = max(0.0, energy_after)
                total_energy_used[platform_id] += energy
                if task["kind"] == "delivery":
                    depot_id, commodity = str(task["depot_id"]), str(task["commodity"])
                    quantity = float(task["quantity"])
                    available = float(depot_state[depot_id]["stock"].get(commodity, 0.0))
                    if quantity > available + 1e-8:
                        raise AssertionError("C1 round selection exceeded cumulative depot stock")
                    depot_state[depot_id]["stock"][commodity] = max(0.0, available - quantity)
                    total_stock_used[(depot_id, commodity)] = total_stock_used.get((depot_id, commodity), 0.0) + quantity
        for platform_id, assigned in assignment["queues"].items():
            queues[platform_id].extend(assigned)
        remaining -= selected

    assigned = {task_id for queue in queues.values() for task_id in queue}
    reasons = copy.deepcopy(final_unassigned_reasons)
    if remaining and set(reasons) != remaining:
        raise AssertionError("C1 final round did not record every remaining task")
    adapter_receipt = {
        "schema": "gain-c1-multiround-adapter/1",
        "adapter_source": "package/civilian/gain_scenarios.py:_multi_round_c1_assignment",
        "algorithm_id": "civilian-public-C1-priority-deadline-completion-v1",
        "seed": int(seed),
        "seed_usage": "recorded_for_reproducibility; deterministic C1 uses no random draws",
        "known_task_ids": sorted(known_task_ids),
        "round_count": len(rounds),
        "rounds": rounds,
        "assignments": copy.deepcopy(queues),
        "selected_action_ids": [
            f"dispatch:{platform_id}:{task_id}"
            for platform_id in sorted(queues) for task_id in queues[platform_id]
        ],
        "assignment_output_sha256": key_digest(queues),
        "feasibility": {
            "all_selected_actions_legal": True,
            "known_task_count": len(known_task_ids),
            "assigned_task_count": len(assigned),
            "unassigned_known_task_reasons": reasons,
            "cumulative_energy_feasible": True,
            "cumulative_inventory_feasible": True,
            "platform_energy_remaining": {
                platform_id: float(platform_state[platform_id]["energy_initial"])
                for platform_id in sorted(platform_state)
            },
            "platform_completion_time_minutes": {
                platform_id: float(platform_time[platform_id]) for platform_id in sorted(platform_time)
            },
            "total_platform_energy_used": {
                platform_id: float(total_energy_used[platform_id]) for platform_id in sorted(total_energy_used)
            },
            "total_depot_stock_used": {
                f"{depot_id}|{commodity}": float(quantity)
                for (depot_id, commodity), quantity in sorted(total_stock_used.items())
            },
            "task_projection": task_projection,
            "soft_deadline_semantics": "lateness_is_reported_and_does_not_make_a_C1_action_illegal",
        },
    }
    return queues, adapter_receipt


def _randomized_gain_scenario(
    template: dict,
    *,
    split: str,
    condition: str,
    seed_variant: int,
    parent_id: str,
    parent_family: str,
) -> dict:
    rng = Random(int(seed_variant))
    scenario = copy.deepcopy(template)
    scenario["split"] = split
    scenario["condition"] = condition
    scenario["seed_variant"] = int(seed_variant)
    scenario["parent_id"] = parent_id
    scenario["parent_family"] = parent_family
    dgp_profile = dgp_profile_for_condition(condition)
    scenario["dgp_profile"] = dgp_profile
    scenario["dgp_profile_sha256"] = _digest(dgp_profile)
    base = scenario["base_scenario"]
    base["scenario_id"] = parent_id
    local_block = list(base["map"]["local_block"])
    for platform in base["platforms"]:
        platform["position"] = _sample_position(rng, bool(platform["local"]), local_block)
        if rng.random() < 0.3:
            dropped = "inspect" if rng.random() < 0.5 else "deliver"
            platform["capabilities"] = [value for value in platform["capabilities"] if value != dropped]
        capacity_range = (6.5, 12.0) if platform["type"] == "uav" else (20.0, 40.0)
        platform["capacity_kg"] = round(rng.uniform(*capacity_range), 3)
        platform["speed_km_min"] = round(float(platform["speed_km_min"]) * rng.uniform(0.85, 1.15), 4)
        platform["energy_initial"] = round(float(platform["energy_initial"]) * rng.uniform(0.9, 1.1), 3)
        platform["energy_per_km"] = round(float(platform["energy_per_km"]) * rng.uniform(0.9, 1.1), 4)
        platform["service_energy_per_min"] = round(float(platform["service_energy_per_min"]) * rng.uniform(0.9, 1.1), 4)
    for depot in base["depots"]:
        depot["position"] = _sample_position(rng, depot["id"] == "n02-local-depot", local_block)
        for commodity, stock in list(depot["stock"].items()):
            depot["stock"][commodity] = round(float(stock) * rng.uniform(1.0, 1.35), 2)
    for task in base["tasks"]:
        task["position"] = _sample_position(rng, bool(task["local"]), local_block)
        task["priority"] = round(rng.uniform(2.0, 6.0), 3)
        task["deadline"] = rng.randint(max(5, int(base["horizon"] * 0.45)), int(base["horizon"]))
        if task["kind"] == "delivery":
            task["quantity"] = round(rng.uniform(1.5, 5.5), 2)
            task["load_minutes"] = rng.randint(1, 3)
            task["unload_minutes"] = rng.randint(1, 3)
        else:
            task["service_minutes"] = rng.randint(1, 5)
    for is_local in (True, False):
        eligible = [row for row in base["platforms"] if bool(row["local"]) == is_local]
        for capability in ("inspect", "deliver"):
            if not any(capability in row["capabilities"] for row in eligible):
                eligible[0]["capabilities"].append(capability)
        for platform in eligible:
            platform["capabilities"] = sorted(set(platform["capabilities"]))

    all_known = {task["id"] for task in base["tasks"] if task["known_initial"]}
    queues, assignment_receipt = _multi_round_c1_assignment(base, all_known, int(seed_variant))
    base["chapter1_snapshot"]["queues"] = copy.deepcopy(queues)
    base["chapter1_snapshot"]["assignment_receipt"] = assignment_receipt
    scenario["queue_plan"] = copy.deepcopy(queues)
    scenario["map"] = copy.deepcopy(base["map"])

    if split == "pilot":
        tape = copy.deepcopy(scenario["event_tape"])
        if condition == "none_or_light":
            tape = []
        else:
            arrival_times = sorted(rng.sample(range(1, base["horizon"]), min(4, len(tape))))
            arrival_index = 0
            for row in tape:
                if row["kind"] == "task_arrival":
                    row["time"] = arrival_times[arrival_index]
                    arrival_index += 1
                else:
                    row["time"] = rng.randint(1, base["horizon"] - 1)
            if condition == "task_update_or_arrival":
                update_time = rng.randint(1, base["horizon"] - 1)
                tape.extend([
                    {"time": update_time, "kind": "priority_change", "task_id": "n02-task-01",
                     "new_priority": round(rng.uniform(4.0, 8.0), 3)},
                    {"time": min(base["horizon"] - 1, update_time + 1), "kind": "deadline_change",
                     "task_id": "n02-task-02", "new_deadline": rng.randint(10, base["horizon"] + 5)},
                ])
        tape.sort(key=lambda row: (int(row["time"]), str(row["kind"]), str(row.get("task_id", row.get("platform_id", "")))))
        for index, row in enumerate(tape):
            row["event_id"] = f"{parent_id}-event-{index:03d}"
        scenario["event_tape"] = tape
    else:
        scenario["event_tape"] = []
        scenario["event_tape_id"] = _event_digest([])
        scenario["scenario_content_sha256"] = scenario_content_sha256(scenario)
        scenario["event_tape"] = sample_future_event_suffix(
            scenario, after_time=0, horizon_minutes=int(base["horizon"]) - 1,
            suffix_seed=int(seed_variant), rates=dgp_profile["rates"],
        )
    tape = scenario["event_tape"]
    scenario["event_tape"] = tape
    scenario["event_tape_id"] = _event_digest(tape)
    scenario["scenario_content_sha256"] = scenario_content_sha256(scenario)
    validate_gain_scenario(scenario)
    return scenario


def validate_gain_scenario(scenario: dict) -> dict:
    """Validate a gain scenario without the legacy train/development restriction."""
    if not isinstance(scenario, dict):
        raise ValueError("gain scenario must be an object")
    required = {
        "parent_id", "parent_family", "split", "condition", "seed_variant", "dgp_profile",
        "dgp_profile_sha256", "map",
        "base_scenario", "queue_plan", "event_tape", "event_tape_id", "scenario_content_sha256",
    }
    if set(scenario) != required:
        raise ValueError("gain scenario fields do not match gain contract")
    if scenario["split"] not in GAIN_SPLITS:
        raise ValueError("gain scenario split must be pilot, train, validation, test, or ood")
    for field in ("parent_id", "parent_family", "condition"):
        if not isinstance(scenario[field], str) or not scenario[field].strip():
            raise ValueError(f"gain scenario {field} must be nonempty")
    if type(scenario["seed_variant"]) is not int:
        raise ValueError("gain scenario seed_variant must be integer")
    expected_profile = dgp_profile_for_condition(scenario["condition"])
    if scenario["dgp_profile"] != expected_profile or scenario["dgp_profile_sha256"] != _digest(expected_profile):
        raise ValueError("gain event DGP profile identity mismatch")
    if scenario["scenario_content_sha256"] != scenario_content_sha256(scenario):
        raise ValueError("gain scenario physical content hash mismatch")
    base = scenario["base_scenario"]
    validate_scenario(base)
    if base["map"] != scenario["map"]:
        raise ValueError("gain scenario map does not match base map")
    queue_plan = scenario["queue_plan"]
    platform_ids = {platform["id"] for platform in base["platforms"]}
    if set(queue_plan) != platform_ids:
        raise ValueError("gain queue plan must name every platform")
    known_ids = {task["id"] for task in base["tasks"] if task["known_initial"]}
    queued = [task_id for queue in queue_plan.values() for task_id in queue]
    if not set(queued) <= known_ids or len(queued) != len(set(queued)):
        raise ValueError("gain queue plan may assign only initially known tasks, at most once each")
    event_ids = []
    task_by_id = {task["id"]: task for task in base["tasks"]}
    for row in scenario["event_tape"]:
        if not isinstance(row, dict):
            raise ValueError("gain event tape entries must be objects")
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("gain event IDs must be nonempty strings")
        event_ids.append(event_id)
        if type(row.get("time")) is not int or not 0 < row["time"] < int(base["horizon"]):
            raise ValueError("gain event times must be within the episode")
        kind = row.get("kind")
        if kind == "task_arrival":
            task = task_by_id.get(row.get("task_id"))
            if task is None or task["known_initial"]:
                raise ValueError("gain task arrival must target an initially hidden task")
        elif kind in {"platform_failure", "energy_drop", "capability_loss", "load_change"}:
            if row.get("platform_id") not in platform_ids:
                raise ValueError("gain platform event references unknown platform")
            if kind == "energy_fraction_drop":
                fraction = row.get("loss_fraction")
                if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0.0 < float(fraction) <= 1.0:
                    raise ValueError("energy loss fraction must be in (0, 1]")
        elif kind == "energy_fraction_drop":
            if row.get("platform_id") not in platform_ids:
                raise ValueError("gain energy event references unknown platform")
            fraction = row.get("loss_fraction")
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(float(fraction)) or not 0.0 < float(fraction) <= 1.0:
                raise ValueError("energy loss fraction must be in (0, 1]")
        elif kind in {"task_cancel", "priority_change", "deadline_change"}:
            if row.get("task_id") not in task_by_id:
                raise ValueError("gain task event references unknown task")
        elif kind in {"communication_drop", "packet_loss", "communication_delay", "communication_split"}:
            links = row.get("links")
            if not isinstance(links, list) or not links:
                raise ValueError("gain communication event must identify links")
        elif kind in {"environment_perturbation", "wind_change", "obstacle_change", "threat_change", "passability_change"}:
            patch = row.get("patch")
            if not isinstance(patch, list) or len(patch) != 4 or patch[2:] != [2, 2]:
                raise ValueError("gain environment event must use a 2x2 patch")
        else:
            raise ValueError(f"unsupported gain event kind {kind!r}")
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("gain event IDs must be unique")
    if scenario["event_tape_id"] != _event_digest(scenario["event_tape"]):
        raise ValueError("gain event tape identity mismatch")
    return {
        "schema": "gain-scenario-validation/1", "parent_id": scenario["parent_id"],
        "parent_family": scenario["parent_family"], "split": scenario["split"],
        "condition": scenario["condition"], "platform_count": len(platform_ids),
        "task_count": len(base["tasks"]), "event_count": len(event_ids),
        "event_tape_id": scenario["event_tape_id"],
    }


def build_pilot_scenarios(instances_per_parent: int = 3, seed: int = 8201) -> list[dict]:
    """Return deterministic smoke variants of each of the six legacy parents."""
    if type(instances_per_parent) is not int or instances_per_parent < 1:
        raise ValueError("instances_per_parent must be a positive integer")
    result = []
    for parent_index, template in enumerate(build_event_scenarios()):
        legacy_id = str(template["parent_id"])
        condition = _LEGACY_TO_GAIN.get(template["condition"], str(template["condition"]))
        family = condition
        for replicate in range(instances_per_parent):
            scenario_seed = int(seed + parent_index * 1000 + replicate)
            parent_id = f"gain-pilot-{parent_index + 1:02d}-{replicate + 1:02d}"
            row = _randomized_gain_scenario(
                template, split="pilot", condition=condition, seed_variant=scenario_seed,
                parent_id=parent_id, parent_family=family,
            )
            result.append(row)
    return result


def iter_gain_scenarios(
    counts_by_split: Mapping[str, int] | None = None,
    *,
    seed: int = 8201,
    conditions: tuple[str, ...] = FORMAL_CONDITIONS,
) -> Iterator[dict]:
    """Yield independently seeded 100x100 formal train/validation/test cases."""
    counts = dict(DEFAULT_SPLIT_COUNTS if counts_by_split is None else counts_by_split)
    if set(counts) - {"train", "validation", "test"}:
        raise ValueError("formal gain splits are train, validation, and test")
    if any(type(count) is not int or count < 0 for count in counts.values()):
        raise ValueError("split counts must be nonnegative integers")
    if not conditions or any(condition not in FORMAL_CONDITIONS for condition in conditions):
        raise ValueError("conditions must be selected from FORMAL_CONDITIONS")
    templates = {row["condition"]: row for row in build_event_scenarios()}
    templates_by_condition = {value: templates[key] for value, key in _TEMPLATE_CONDITION.items()}
    split_offset = {"train": 0, "validation": 1_000_000, "test": 2_000_000}
    for split in ("train", "validation", "test"):
        count = counts.get(split, 0)
        for condition in conditions:
            condition_index = FORMAL_CONDITIONS.index(condition)
            for index in range(count):
                scenario_seed = int(seed + split_offset[split] + condition_index * 100_000 + index)
                parent_id = f"gain-{split}-{condition}-{index + 1:04d}"
                family = condition
                yield _randomized_gain_scenario(
                    templates_by_condition[condition], split=split, condition=condition,
                    seed_variant=scenario_seed, parent_id=parent_id, parent_family=family,
                )


def build_gain_scenarios(
    counts_by_split: Mapping[str, int] | None = None,
    *,
    seed: int = 8201,
    conditions: tuple[str, ...] = FORMAL_CONDITIONS,
) -> list[dict]:
    """Materialize formal scenarios when a caller has explicit memory budget."""
    return list(iter_gain_scenarios(counts_by_split, seed=seed, conditions=conditions))


def sample_future_event_suffix(
    scenario: dict,
    *,
    after_time: int,
    horizon_minutes: int,
    suffix_seed: int,
    rates: Mapping[str, float] | None = None,
) -> list[dict]:
    """Sample a fresh future suffix conditional on the realized tape prefix.

    Event draws are addressed by category/entity/minute/draw-kind. Future rows
    from the scenario tape are intentionally ignored; only prefix rows are used
    to condition availability and avoid replaying realized events.
    """
    validate_gain_scenario(scenario)
    if type(after_time) is not int or after_time < 0:
        raise ValueError("after_time must be a nonnegative integer")
    if type(horizon_minutes) is not int or horizon_minutes < 1:
        raise ValueError("horizon_minutes must be a positive integer")
    if type(suffix_seed) is not int:
        raise ValueError("suffix_seed must be an integer")
    frozen_rates = dict(scenario["dgp_profile"]["rates"])
    if rates is not None:
        unknown = set(rates) - set(DEFAULT_SUFFIX_RATES)
        if unknown:
            raise ValueError(f"unknown suffix rates: {sorted(unknown)}")
        supplied = {**frozen_rates, **{key: float(value) for key, value in rates.items()}}
        if scenario["split"] != "pilot" and supplied != frozen_rates:
            raise ValueError("formal suffix rates must match the frozen scenario DGP profile")
        probabilities = supplied
    else:
        probabilities = frozen_rates
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities.values()):
        raise ValueError("suffix rates must be finite probabilities")

    base = scenario["base_scenario"]
    final_time = min(int(base["horizon"]) - 1, after_time + horizon_minutes)
    if final_time <= after_time:
        return []
    prefix = [row for row in scenario["event_tape"] if int(row["time"]) <= after_time]
    arrived = {row["task_id"] for row in prefix if row["kind"] == "task_arrival"}
    available_hidden = [task for task in base["tasks"] if not task["known_initial"] and task["id"] not in arrived]
    existing_platform_events = {
        row["platform_id"] for row in prefix
        if row["kind"] in {"platform_failure", "capability_loss", "load_change"}
    }
    existing_energy_events = {
        row["platform_id"] for row in prefix
        if row["kind"] in {"energy_drop", "energy_fraction_drop"}
    }
    existing_task_updates = {
        row["task_id"] for row in prefix
        if row["kind"] in {"priority_change", "deadline_change", "task_cancel"}
    }
    existing_links = {
        tuple(sorted((str(link[0]), str(link[1]))))
        for row in prefix if row["kind"] in {"communication_drop", "packet_loss", "communication_delay", "communication_split"}
        for link in row.get("links", [])
    }
    platforms = base["platforms"]
    platform_ids = [str(row["id"]) for row in platforms]
    links = [(platform_ids[index], platform_ids[index + 1]) for index in range(len(platform_ids) - 1)]
    region = list(base["map"]["local_block"])
    region_identity = ":".join(str(value) for value in region)
    current_tasks = {task["id"] for task in base["tasks"] if task["known_initial"]} | arrived
    generated: list[dict] = []

    def append(minute: int, kind: str, entity: str, **fields: Any) -> None:
        generated.append({
            "time": minute, "kind": kind,
            "event_id": _new_event_id(suffix_seed, kind, entity, minute), **fields,
        })

    for minute in range(after_time + 1, final_time + 1):
        if available_hidden and _uniform(suffix_seed, "task_arrival", "all", minute, "hazard") < probabilities["task_arrival_per_minute"]:
            index = int.from_bytes(_key(suffix_seed, "task_arrival", "all", minute, "task_choice")[:8], "big") % len(available_hidden)
            task = available_hidden.pop(index)
            current_tasks.add(str(task["id"]))
            append(minute, "task_arrival", str(task["id"]), task_id=str(task["id"]))

        for platform in platforms:
            platform_id = str(platform["id"])
            if platform_id not in existing_platform_events and _uniform(
                suffix_seed, "platform_event", platform_id, minute, "hazard"
            ) < probabilities["platform_event_per_minute_per_platform"]:
                capabilities = list(platform.get("capabilities", []))
                if len(capabilities) > 1 and _uniform(suffix_seed, "platform_event", platform_id, minute, "type") < 0.5:
                    capability_index = int.from_bytes(_key(suffix_seed, "platform_event", platform_id, minute, "capability")[:8], "big") % len(capabilities)
                    append(minute, "capability_loss", platform_id, platform_id=platform_id, capability=capabilities[capability_index])
                else:
                    append(minute, "platform_failure", platform_id, platform_id=platform_id)
                existing_platform_events.add(platform_id)
            if platform_id not in existing_energy_events and _uniform(
                suffix_seed, "energy_anomaly", platform_id, minute, "hazard"
            ) < probabilities["energy_anomaly_per_minute_per_platform"]:
                append(minute, "energy_fraction_drop", platform_id, platform_id=platform_id,
                       loss_fraction=0.5)
                existing_energy_events.add(platform_id)

        for link in links:
            link_key = tuple(sorted(link))
            link_identity = "|".join(link_key)
            if link_key not in existing_links and _uniform(
                suffix_seed, "communication_degradation", link_identity, minute, "hazard"
            ) < probabilities["communication_event_per_minute_per_link"]:
                append(minute, "communication_drop", link_identity, links=[list(link)], quality=0.1)
                existing_links.add(link_key)

        if _uniform(suffix_seed, "environment_event", region_identity, minute, "hazard") < probabilities["environment_event_per_minute_per_region"]:
            x, y, _, _ = region
            append(minute, "environment_perturbation", region_identity,
                   patch=[int(x), int(y), 2, 2], wind_vector=[0.5, -0.4],
                   risk_delta=0.2, blocked_cell=[int(x), int(y)])

        for task_id in sorted(current_tasks):
            if task_id in existing_task_updates:
                continue
            if _uniform(suffix_seed, "task_update", task_id, minute, "hazard") < probabilities["task_update_per_minute_per_task"]:
                if _uniform(suffix_seed, "task_update", task_id, minute, "type") < 0.5:
                    new_priority = round(2.0 + 6.0 * _uniform(suffix_seed, "task_update", task_id, minute, "priority"), 3)
                    append(minute, "priority_change", task_id, task_id=task_id, new_priority=new_priority)
                else:
                    new_deadline = minute + 1 + int(30 * _uniform(suffix_seed, "task_update", task_id, minute, "deadline"))
                    append(minute, "deadline_change", task_id, task_id=task_id, new_deadline=new_deadline)
                existing_task_updates.add(task_id)
    generated.sort(key=lambda row: (int(row["time"]), str(row["kind"]), str(row.get("task_id", row.get("platform_id", "")))))
    ids = [row["event_id"] for row in generated]
    if len(ids) != len(set(ids)):
        raise AssertionError("suffix event ID collision")
    return generated
