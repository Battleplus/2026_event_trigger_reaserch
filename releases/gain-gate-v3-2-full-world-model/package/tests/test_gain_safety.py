from __future__ import annotations

import copy
import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import event_scenarios
import gain_safety
from event_environment import EventReliefEnv
import gain_scenarios


def _delivery_observation(*, stock: float = 5.0, deadline: float = 100.0) -> dict:
    platforms = []
    tasks = []
    queues = {}
    for index in range(2):
        platform_id = f"n02-uav-{index + 1}"
        task_id = f"n02-task-{index + 1}"
        platforms.append({
            "id": platform_id, "position": [0.0, 0.0], "energy": 100.0,
            "speed_km_min": 1.0, "energy_per_km": 1.0,
            "service_energy_per_min": 0.0, "capacity_kg": 10.0,
            "capabilities": ["deliver"], "failed": False, "active_task_id": None,
        })
        tasks.append({
            "id": task_id, "known": True, "kind": "delivery", "status": "pending",
            "phase": "pending", "position": [0.0, 0.0], "depot_id": "depot-1",
            "commodity": "water", "quantity": 3.0, "load_minutes": 1,
            "unload_minutes": 1, "deadline": deadline,
        })
        queues[platform_id] = [task_id]
    return {
        "time": 0.0, "horizon": 60.0, "platforms": platforms, "tasks": tasks,
        "depots": [{"id": "depot-1", "position": [0.0, 0.0], "stock": {"water": stock}}],
        "queues": queues, "communication": [],
    }


def test_depot_inventory_commitments_are_aggregated_across_platforms() -> None:
    report = gain_safety.assess_queue_safety(_delivery_observation(stock=5.0))

    assert report["forced"] is True
    violation = next(row for row in report["violations"]
                     if row["code"] == "committed_queue_inventory_infeasible")
    assert violation["quantity"] == 6.0
    assert violation["stock"] == 5.0
    assert violation["affected_platform_ids"] == ["n02-uav-1", "n02-uav-2"]
    assert violation["affected_task_ids"] == ["n02-task-1", "n02-task-2"]
    assert report["required_scope"] == "GLOBAL_REPLAN"


def test_soft_deadline_lateness_is_reported_without_forcing_reconstruction() -> None:
    report = gain_safety.assess_queue_safety(_delivery_observation(stock=10.0, deadline=0.5))

    assert report["forced"] is False
    assert report["safe_to_suppress"] is True
    assert report["deadline_semantics"] == "soft_unless_public_task_explicitly_sets_hard_deadline_true"
    assert report["soft_deadline_risks"]
    assert report["soft_deadline_risks"][0]["lateness_minutes"] == 1.5


def test_loaded_active_delivery_uses_remaining_delivery_leg_and_stock_is_not_recommitted() -> None:
    observation = {
        "time": 4.0, "horizon": 60.0,
        "platforms": [{
            "id": "n02-uav-1", "position": [10.0, 10.0], "energy": 100.0,
            "speed_km_min": 1.0, "energy_per_km": 1.0,
            "service_energy_per_min": 0.0, "capacity_kg": 10.0,
            "capabilities": ["deliver"], "failed": False, "active_task_id": "task-1",
        }],
        "tasks": [{
            "id": "task-1", "known": True, "kind": "delivery", "status": "active",
            "phase": "to_delivery_site", "service_progress": 2.0,
            "position": [10.0, 10.0], "depot_id": "depot-1",
            "commodity": "water", "quantity": 3.0, "load_minutes": 2,
            "unload_minutes": 1, "deadline": 6.0,
        }],
        "depots": [{"id": "depot-1", "position": [90.0, 90.0], "stock": {"water": 0.0}}],
        "queues": {"n02-uav-1": ["task-1"]}, "communication": [],
    }

    report = gain_safety.assess_queue_safety(observation)

    assert report["forced"] is False
    assert report["soft_deadline_risks"] == []
    assert not any(row["code"] == "committed_queue_inventory_infeasible" for row in report["violations"])


class _QueueProjection:
    @staticmethod
    def _planned_queue_state(observation: dict, operations: list, dispatches: list) -> tuple[dict, dict]:
        assert operations == []
        assert dispatches == []
        return copy.deepcopy(observation["queues"]), {}


def test_noop_cannot_be_reported_as_resolving_a_forced_safety_state() -> None:
    observation = _delivery_observation(stock=5.0)
    initial = gain_safety.assess_queue_safety(observation)
    noop = {
        "queue_operations": [], "dispatches": [],
    }

    receipt = gain_safety.verify_safety_disposition(_QueueProjection(), observation, noop, initial)

    assert receipt["status"] == "unresolved"
    assert receipt["resolved"] is False
    assert receipt["unknown_reason"] == "selected_action_leaves_public_queue_infeasible"
    assert receipt["remaining_report"]["forced"] is True


def test_unknown_queued_task_identity_is_never_treated_as_safe() -> None:
    observation = _delivery_observation(stock=10.0)
    observation["queues"]["n02-uav-1"].append("hidden-or-unregistered-task")

    report = gain_safety.assess_queue_safety(observation)

    assert report["forced"] is True
    assert any(row["code"] == "queued_task_identity_unknown" for row in report["violations"])


class _SplitProjection:
    @staticmethod
    def _planned_queue_state(observation: dict, operations: list, dispatches: list) -> tuple[dict, dict]:
        assert dispatches == []
        queues = copy.deepcopy(observation["queues"])
        tasks = {row["id"]: copy.deepcopy(row) for row in observation["tasks"]}
        parent = tasks["parent"]
        parent.update({"status": "split_parent", "phase": "split_parent"})
        queues["n02-uav-1"].remove("parent")
        children = {}
        for child_id, platform_id in (("parent::part-01-1", "n02-uav-1"), ("parent::part-01-2", "n02-uav-2")):
            child = copy.deepcopy(tasks["parent"])
            child.update({
                "id": child_id, "status": "pending", "phase": "pending", "assignee": platform_id,
                "quantity": 4.0, "progress": 0.0, "service_progress": 0.0,
            })
            children[child_id] = child
            queues[platform_id].append(child_id)
        return queues, {**tasks, **children}


def test_split_disposition_retains_target_capacity_failure() -> None:
    observation = {
        "time": 0.0, "horizon": 60.0,
        "platforms": [
            {"id": "n02-uav-1", "position": [0.0, 0.0], "energy": 100.0, "speed_km_min": 1.0,
             "energy_per_km": 1.0, "service_energy_per_min": 0.0, "capacity_kg": 6.0,
             "capabilities": ["deliver"], "failed": False, "active_task_id": None},
            {"id": "n02-uav-2", "position": [0.0, 0.0], "energy": 100.0, "speed_km_min": 1.0,
             "energy_per_km": 1.0, "service_energy_per_min": 0.0, "capacity_kg": 3.0,
             "capabilities": ["deliver"], "failed": False, "active_task_id": None},
        ],
        "tasks": [{
            "id": "parent", "known": True, "kind": "delivery", "status": "pending", "phase": "pending",
            "position": [0.0, 0.0], "depot_id": "depot-1", "commodity": "water",
            "quantity": 8.0, "load_minutes": 1, "unload_minutes": 1, "deadline": 60.0,
        }],
        "depots": [{"id": "depot-1", "position": [0.0, 0.0], "stock": {"water": 10.0}}],
        "queues": {"n02-uav-1": ["parent"], "n02-uav-2": []}, "communication": [],
    }
    initial = gain_safety.assess_queue_safety(observation)
    action = {
        "queue_operations": [{
            "op": "split", "task_id": "parent", "from_platform": "n02-uav-1",
            "to_platform": "n02-uav-2", "share": 0.5,
        }],
        "dispatches": [],
    }

    receipt = gain_safety.verify_safety_disposition(_SplitProjection(), observation, action, initial)

    assert initial["forced"] is True
    assert receipt["status"] == "unresolved"
    assert any(
        row["code"] == "queued_task_capacity_infeasible" and row["task_id"] == "parent::part-01-2"
        for row in receipt["remaining_report"]["violations"]
    )


@lru_cache(maxsize=1)
def _pilot_scenario_json() -> str:
    import json
    scenario = gain_scenarios.build_pilot_scenarios(instances_per_parent=1, seed=8201)[0]
    scenario["event_tape"] = [{
        "time": 1, "kind": "energy_fraction_drop", "platform_id": "n02-uav-1",
        "loss_fraction": 0.5, "event_id": "energy-fraction-drop-at-minute-1",
    }]
    scenario["event_tape_id"] = event_scenarios._digest(scenario["event_tape"])
    scenario["scenario_content_sha256"] = gain_scenarios.scenario_content_sha256(scenario)
    gain_scenarios.validate_gain_scenario(scenario)
    return json.dumps(scenario)


def test_proportional_energy_loss_never_recharges_an_already_depleted_platform() -> None:
    scenario = __import__("json").loads(_pilot_scenario_json())
    env = EventReliefEnv(scenario)
    env.reset()
    platform_id = "n02-uav-1"
    env._base._platforms[platform_id]["energy"] = 0.0
    env._base._public_platforms[platform_id]["energy"] = 0.0
    env._reported_platforms[platform_id]["energy"] = 0.0

    env.step({
        "action_type": "reconstruct", "scope": "NOOP", "noop": True,
        "queue_operations": [], "dispatches": [],
    })

    assert env._base._platforms[platform_id]["energy"] == 0.0
    assert env._base._public_platforms[platform_id]["energy"] == 0.0
    assert env._reported_platforms[platform_id]["energy"] == 0.0
    event = next(row for row in env._extension_events if row["type"] == "energy_fraction_drop")
    assert event["old_energy"] == 0.0
    assert event["new_energy"] == 0.0


def test_legacy_event_wrapper_accepts_energy_fraction_suffix() -> None:
    scenario = copy.deepcopy(event_scenarios.build_event_scenarios()[0])
    scenario["event_tape"] = [{
        "time": 1, "kind": "energy_fraction_drop", "platform_id": "n02-uav-1",
        "loss_fraction": 0.25, "event_id": "legacy-energy-fraction-at-minute-1",
    }]
    scenario["event_tape_id"] = event_scenarios._digest(scenario["event_tape"])

    receipt = event_scenarios.validate_event_scenario(scenario)
    assert receipt["event_count"] == 1
    env = EventReliefEnv(scenario)
    env.reset()
    env.step({
        "action_type": "reconstruct", "scope": "NOOP", "noop": True,
        "queue_operations": [], "dispatches": [],
    })
    event = next(row for row in env._extension_events if row["type"] == "energy_fraction_drop")
    assert event["loss_fraction"] == 0.25


def test_multiround_c1_adapter_preserves_and_recomputes_each_source_receipt() -> None:
    scenario = gain_scenarios.build_pilot_scenarios(instances_per_parent=1, seed=8201)[0]
    receipt = scenario["base_scenario"]["chapter1_snapshot"]["assignment_receipt"]

    assert receipt["schema"] == "gain-c1-multiround-adapter/1"
    assert receipt["round_count"] == 2
    assert all(row["c1_receipt"]["schema"] == "chapter1-public-c1-assignment/1" for row in receipt["rounds"])
    assert gain_scenarios.validate_gain_scenario(scenario)["split"] == "pilot"


def test_multiround_c1_adapter_rejects_a_changed_saved_public_input() -> None:
    scenario = copy.deepcopy(gain_scenarios.build_pilot_scenarios(instances_per_parent=1, seed=8201)[0])
    receipt = scenario["base_scenario"]["chapter1_snapshot"]["assignment_receipt"]
    receipt["rounds"][0]["public_input"]["platforms"][0]["energy_initial"] += 0.1

    with pytest.raises(ValueError, match="public input does not match the cumulative public state"):
        gain_scenarios.validate_gain_scenario(scenario)
