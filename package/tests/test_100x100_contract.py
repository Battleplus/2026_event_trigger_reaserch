from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CIVILIAN_ROOT = PACKAGE_ROOT / "civilian"
sys.path.insert(0, str(CIVILIAN_ROOT))

from branching import (  # noqa: E402
    build_branch_record,
    capture_snapshot,
    paired_global_targets,
    paired_task_targets,
    reference_action_batch,
    restore_snapshot,
)
from chapter1_adapter import generate_public_c1_initial_assignment  # noqa: E402
from dataset_protocol import validate_registry  # noqa: E402
from graph_policy import encode_public_graph  # noqa: E402
from relief_env import ReliefEnv  # noqa: E402
from scenario_matrix import make_parent  # noqa: E402
from scenario_schema import C1_SNAPSHOT_SOURCE, MAP_CONTRACT, validate_scenario  # noqa: E402
from stable_identity import key_digest  # noqa: E402
from world_model import DynamicConsequenceModel, public_policy_tensors  # noqa: E402


def _causal_choice_scenario() -> dict:
    scenario = make_parent("train", 0, "base")
    scenario["scenario_id"] = "n02-test-causal-choice-100x100"
    scenario["disturbances"] = []
    scenario["platforms"][0]["capabilities"] = ["inspect", "deliver"]
    scenario["platforms"][1]["capabilities"] = ["inspect"]
    scenario["platforms"][2]["capabilities"] = ["inspect"]
    scenario["tasks"] = [
        {
            "id": "causal-inspection",
            "kind": "inspection",
            "position": [40.25, 40.25],
            "local": True,
            "known_initial": True,
            "deadline": 30,
            "priority": 3.0,
            "discovered_by": None,
            "report_delay": 6,
            "service_minutes": 2,
        },
        {
            "id": "causal-delivery-a",
            "kind": "delivery",
            "position": [41.0, 40.5],
            "local": True,
            "known_initial": False,
            "deadline": 14,
            "priority": 5.0,
            "discovered_by": "causal-inspection",
            "report_delay": 0,
            "commodity": "water",
            "quantity": 4.0,
            "depot_id": "n02-local-depot",
            "load_minutes": 1,
            "unload_minutes": 1,
        },
        {
            "id": "causal-delivery-b",
            "kind": "delivery",
            "position": [40.5, 41.0],
            "local": True,
            "known_initial": False,
            "deadline": 14,
            "priority": 5.0,
            "discovered_by": "causal-inspection",
            "report_delay": 0,
            "commodity": "water",
            "quantity": 4.0,
            "depot_id": "n02-local-depot",
            "load_minutes": 1,
            "unload_minutes": 1,
        },
        {
            "id": "causal-delivery-c",
            "kind": "delivery",
            "position": [41.25, 41.25],
            "local": True,
            "known_initial": False,
            "deadline": 15,
            "priority": 10.0,
            "discovered_by": "causal-inspection",
            "report_delay": 0,
            "commodity": "water",
            "quantity": 4.0,
            "depot_id": "n02-local-depot",
            "load_minutes": 1,
            "unload_minutes": 1,
        },
    ]
    for task in scenario["tasks"]:
        if task["id"] in {"causal-delivery-a", "causal-delivery-b"}:
            task["priority"] = 1.0
    scenario["chapter1_snapshot"]["queues"] = {
        "n02-uav-1": ["causal-inspection"],
        "n02-uav-2": [],
        "n02-uav-3": [],
        "n02-heli-1": [],
    }
    scenario["chapter1_snapshot"]["source"] = "synthetic_test_fixture"
    scenario["chapter1_snapshot"].pop("assignment_receipt", None)
    validate_scenario(scenario)
    return scenario


def _run_choice(snapshot_json: str, action_batch: list[str]) -> tuple[ReliefEnv, float]:
    env = restore_snapshot(snapshot_json)
    total_reward = 0.0
    _, reward, terminated, _, _ = env.step(action_batch)
    total_reward += reward
    stop_at = int(env.time) + 29
    while not terminated and int(env.time) < stop_at:
        _, reward, terminated, _, _ = env.step(reference_action_batch(env))
        total_reward += reward
    return env, total_reward


def test_registered_inputs_and_fixtures_use_the_100x100_contract() -> None:
    report = validate_registry(CIVILIAN_ROOT / "registered_data")
    assert report["allocation"] == {"train": 2, "development": 1}
    assert report["scenarios_verified"] is True

    for path in sorted((CIVILIAN_ROOT / "fixtures").glob("*.json")):
        scenario = json.loads(path.read_text(encoding="utf-8"))
        validate_scenario(scenario)
        assert scenario["map"]["grid_contract"] == MAP_CONTRACT
        assert scenario["map"]["grid_cells"] == 10000


def test_schema_rejects_non_100x100_maps_and_out_of_range_coordinates() -> None:
    valid = make_parent("train", 0, "base")

    for rows, cols, cells in ((12, 12, 144), (100, 99, 9900), (101, 100, 10100)):
        scenario = copy.deepcopy(valid)
        scenario["map"].update({"rows": rows, "cols": cols, "grid_cells": cells})
        with pytest.raises(ValueError, match="100x100|10000"):
            validate_scenario(scenario)

    scenario = copy.deepcopy(valid)
    scenario["platforms"][0]["position"] = [100.0, 40.0]
    with pytest.raises(ValueError):
        validate_scenario(scenario)


def test_two_legal_global_choices_change_other_order_and_recompute_labels() -> None:
    scenario = _causal_choice_scenario()
    source = ReliefEnv(scenario)
    source.reset("C1")
    inspection_id = next(
        item["id"] for item in source.candidates()
        if item.get("type") == "dispatch" and item.get("task_id") == "causal-inspection"
        and item.get("platform_id") == "n02-uav-1"
    )
    source.step([inspection_id])
    while source.time < 8:
        source.step(["NOOP"])

    observation = source.public_observation()
    catalog = source.candidates()
    dispatches = {
        item["task_id"]: item["id"]
        for item in catalog
        if item.get("type") == "dispatch" and item.get("platform_id") == "n02-uav-1"
    }
    assert set(dispatches) == {"causal-delivery-a", "causal-delivery-b", "causal-delivery-c"}

    snapshot_json = capture_snapshot(
        source,
        parent_id=scenario["scenario_id"],
        repeat=0,
        window_id="causal-window-8",
        event_tape_id="causal-event-tape-8201",
    )
    payload = json.loads(snapshot_json)
    policy_input = {"observation": payload["public_prefix"], "candidates": payload["candidate_catalog"]}
    assert policy_input["observation"]["map"]["grid_cells"] == 10000

    graph = encode_public_graph(policy_input)
    platforms = sorted(observation["platforms"], key=lambda item: item["id"])
    platform_index = next(index for index, item in enumerate(platforms) if item["id"] == "n02-uav-1")
    platform = platforms[platform_index]
    assert graph.node_features[platform_index, 6].item() == pytest.approx(platform["position"][0] / 100.0)
    assert graph.node_features[platform_index, 7].item() == pytest.approx(platform["position"][1] / 100.0)

    wm_tensors, binding = public_policy_tensors(policy_input)
    assert wm_tensors["context"][0, 2:4].tolist() == [1.0, 1.0]
    assert binding.candidate_ids[0]
    model_layout = DynamicConsequenceModel().config
    assert (model_layout["map_contract"], model_layout["map_rows"], model_layout["map_cols"], model_layout["grid_cells"]) == (MAP_CONTRACT, 100, 100, 10000)

    action_a = [dispatches["causal-delivery-a"], "NOOP"]
    action_b = [dispatches["causal-delivery-b"], "NOOP"]
    candidate_env, candidate_return = _run_choice(snapshot_json, action_b)
    reference_env, reference_return = _run_choice(snapshot_json, action_a)
    tampered_payload = copy.deepcopy(payload)
    tampered_payload["candidate_catalog"] = [
        item for item in tampered_payload["candidate_catalog"] if item["id"] != action_b[0]
    ]
    tampered_payload["branch_envelope_sha256"] = hashlib.sha256(
        (json.dumps(
            {key: value for key, value in tampered_payload.items() if key != "branch_envelope_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )).encode("ascii")
    ).hexdigest().upper()
    tampered_snapshot = json.dumps(tampered_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    with pytest.raises(ValueError):
        build_branch_record(
            tampered_snapshot,
            candidate_action_batch=action_b,
            reference_action_batch=action_a,
            parent_id=scenario["scenario_id"],
            repeat=0,
            window_id="causal-window-8",
            event_tape_id="causal-event-tape-8201",
            candidate_id=key_digest(action_b),
            candidate_return=candidate_return,
            reference_return=reference_return,
            candidate_env=candidate_env,
            reference_env=reference_env,
        )
    candidate_record = build_branch_record(
        snapshot_json,
        candidate_action_batch=action_b,
        reference_action_batch=action_a,
        parent_id=scenario["scenario_id"],
        repeat=0,
        window_id="causal-window-8",
        event_tape_id="causal-event-tape-8201",
        candidate_id=key_digest(action_b),
        candidate_return=candidate_return,
        reference_return=reference_return,
        candidate_env=candidate_env,
        reference_env=reference_env,
    )

    assert candidate_env._tasks["causal-delivery-b"]["on_time"] is True
    assert candidate_env._tasks["causal-delivery-a"]["on_time"] is False
    assert reference_env._tasks["causal-delivery-a"]["on_time"] is True
    assert reference_env._tasks["causal-delivery-b"]["on_time"] is False

    labels = candidate_record["paired_targets"]["task"]
    a_index = labels["task_ids"].index("causal-delivery-a")
    b_index = labels["task_ids"].index("causal-delivery-b")
    delivery_head = labels["names"].index("delivery_on_time")
    assert labels["valid"][a_index][delivery_head] is True
    assert labels["valid"][b_index][delivery_head] is True
    assert labels["values"][a_index][delivery_head] == -1.0
    assert labels["values"][b_index][delivery_head] == 1.0
    assert labels == paired_task_targets(candidate_env, reference_env, labels["task_ids"])

    recomputed_global = paired_global_targets(
        candidate_env,
        reference_env,
        candidate_return,
        reference_return,
        candidate_env.events,
        reference_env.events,
        window_start=8,
        exclude_task_ids={"causal-delivery-a", "causal-delivery-b"},
    )
    assert candidate_record["paired_targets"]["global"] == recomputed_global
    assert candidate_record["policy_input_sha256"] == key_digest({"observation": payload["public_prefix"], "candidates": payload["candidate_catalog"]})


def test_infeasible_stock_candidate_remains_excluded() -> None:
    scenario = make_parent("train", 0, "base")
    scenario["tasks"] = [copy.deepcopy(next(task for task in scenario["tasks"] if task["id"] == "n02-local-delivery-a"))]
    scenario["disturbances"] = []
    scenario["depots"][0]["stock"]["water"] = 0.0
    scenario["chapter1_snapshot"]["queues"] = {platform["id"]: [] for platform in scenario["platforms"]}
    scenario["chapter1_snapshot"]["source"] = "synthetic_test_fixture"
    scenario["chapter1_snapshot"].pop("assignment_receipt", None)
    env = ReliefEnv(scenario)
    env.reset("C1")
    catalog = env.candidates()
    assert [item["id"] for item in catalog] == ["NOOP"]


def test_c1_initial_assignment_is_public_recomputable_and_legal() -> None:
    scenario = make_parent("train", 0, "base")
    receipt = scenario["chapter1_snapshot"]["assignment_receipt"]
    assert scenario["chapter1_snapshot"]["source"] == C1_SNAPSHOT_SOURCE
    recomputed = generate_public_c1_initial_assignment(scenario, seed=receipt["seed"])
    assert recomputed["queues"] == scenario["chapter1_snapshot"]["queues"]
    assert recomputed["receipt"] == receipt
    assert receipt["feasibility"]["all_selected_actions_legal"] is True
    assert receipt["feasibility"]["assigned_task_count"] == len(receipt["selected_action_ids"])
    validate_scenario(scenario)


def test_c1_ignores_hidden_future_fields_but_responds_to_public_priority() -> None:
    scenario = make_parent("train", 0, "base")
    competing_inspection = copy.deepcopy(next(item for item in scenario["tasks"] if item["id"] == "n02-local-inspection"))
    competing_inspection.update({
        "id": "n02-competing-known-inspection",
        "position": [40.25, 40.25],
        "priority": 4.0,
    })
    scenario["tasks"].append(competing_inspection)
    baseline = generate_public_c1_initial_assignment(scenario, seed=8201)
    scenario["chapter1_snapshot"]["queues"] = baseline["queues"]
    scenario["chapter1_snapshot"]["assignment_receipt"] = baseline["receipt"]
    scenario["chapter1_snapshot"]["source"] = C1_SNAPSHOT_SOURCE

    hidden_changed = copy.deepcopy(scenario)
    for task in hidden_changed["tasks"]:
        if not task["known_initial"]:
            task["position"] = [99.0, 99.0]
            task["deadline"] = min(hidden_changed["horizon"], task["deadline"] + 5)
            task["priority"] = max(1.0, task["priority"] - 1.0)
            task["report_delay"] += 2
    hidden_result = generate_public_c1_initial_assignment(hidden_changed, seed=8201)
    assert hidden_result["receipt"]["public_input_sha256"] == baseline["receipt"]["public_input_sha256"]
    assert hidden_result["queues"] == baseline["queues"]

    priority_changed = copy.deepcopy(scenario)
    inspection = next(item for item in priority_changed["tasks"] if item["id"] == "n02-local-inspection")
    inspection["priority"] = 100.0
    priority_result = generate_public_c1_initial_assignment(priority_changed, seed=8201)
    assert priority_result["receipt"]["public_input_sha256"] != baseline["receipt"]["public_input_sha256"]
    assert priority_result["queues"] != baseline["queues"]
    priority_changed["chapter1_snapshot"]["queues"] = priority_result["queues"]
    priority_changed["chapter1_snapshot"]["assignment_receipt"] = priority_result["receipt"]
    priority_changed["chapter1_snapshot"]["source"] = C1_SNAPSHOT_SOURCE
    validate_scenario(priority_changed)


def test_2x2_spatial_perturbation_changes_truth_before_delayed_public_report() -> None:
    scenario = make_parent("train", 0, "base")
    scenario["scenario_id"] = "n02-test-spatial-2x2"
    scenario["disturbances"] = [{
        "id": "n02-spatial-2x2-01",
        "time": 8,
        "kind": "spatial_perturbation",
        "report_delay": 2,
        "target_type": "platform",
        "target_id": "n02-uav-1",
        "position": [41.75, 41.75],
    }]
    validate_scenario(scenario)
    env = ReliefEnv(scenario)
    env.reset("C1")
    public_position = lambda: next(
        item["position"] for item in env.public_observation()["platforms"] if item["id"] == "n02-uav-1"
    )
    original = list(public_position())
    for _ in range(8):
        env.step(["NOOP"])
    assert env.time == 8
    assert env._platforms["n02-uav-1"]["position"] == [41.75, 41.75]
    assert public_position() == original
    assert any(item["type"] == "spatial_perturbation" and item["time"] == 8.0 for item in env.events)
    env.step(["NOOP"])
    assert env.time == 9
    assert public_position() == original
    env.step(["NOOP"])
    assert env.time == 10
    assert public_position() == [41.75, 41.75]
    generated = next(item for item in env.events if item["type"] == "report_generated")
    received = next(item for item in env.events if item["type"] == "report_received")
    assert generated["kind"] == received["kind"] == "spatial_perturbation"
    assert generated["report_delay"] == 2
    assert received["time"] - generated["time"] == 2.0

    outside = copy.deepcopy(scenario)
    outside["disturbances"][0]["position"] = [42.0, 41.75]
    with pytest.raises(ValueError, match="local_block"):
        validate_scenario(outside)

    nonlocal_target = copy.deepcopy(scenario)
    nonlocal_target["disturbances"][0]["target_id"] = "n02-heli-1"
    with pytest.raises(ValueError, match="local entity"):
        validate_scenario(nonlocal_target)


def test_noop_reference_and_independent_oracle_expose_non_target_global_effect() -> None:
    scenario = _causal_choice_scenario()
    source = ReliefEnv(scenario)
    source.reset("C1")
    inspection_id = next(
        item["id"] for item in source.candidates()
        if item.get("type") == "dispatch" and item.get("task_id") == "causal-inspection"
        and item.get("platform_id") == "n02-uav-1"
    )
    source.step([inspection_id])
    while source.time < 8:
        source.step(["NOOP"])
    catalog = source.candidates()
    dispatches = {
        item["task_id"]: item["id"]
        for item in catalog
        if item.get("type") == "dispatch" and item.get("platform_id") == "n02-uav-1"
    }
    assert set(dispatches) == {"causal-delivery-a", "causal-delivery-b", "causal-delivery-c"}
    snapshot_json = capture_snapshot(
        source,
        parent_id=scenario["scenario_id"],
        repeat=0,
        window_id="causal-window-8-independent-oracle",
        event_tape_id="causal-event-tape-8201-independent-oracle",
    )
    candidate_action = [dispatches["causal-delivery-a"], "NOOP"]
    reference_action = ["NOOP"]
    candidate_env, candidate_return = _run_choice(snapshot_json, candidate_action)
    reference_env, reference_return = _run_choice(snapshot_json, reference_action)

    target_ids = {"causal-delivery-a", "causal-delivery-b"}
    non_target_ids = {
        task_id for task_id, task in candidate_env._tasks.items()
        if task["kind"] == "delivery" and task_id not in target_ids
    }

    def oracle_values(env: ReliefEnv, events: list[dict]) -> list[float]:
        non_target_late = sum(
            task["on_time"] is False
            for task_id, task in env._tasks.items()
            if task["kind"] == "delivery" and task_id in non_target_ids
        )
        unmet = sum(task["delivered"] is False for task in env._tasks.values() if task["kind"] == "delivery")
        energy = sum(float(event.get("energy", 0.0)) for event in events if event.get("type") in {"move", "service"})
        reassignment = sum(bool(task.get("_reassigned")) for task in env._tasks.values())
        return [float(non_target_late), float(unmet), float(energy), float(reassignment), float(candidate_return if env is candidate_env else reference_return)]

    candidate_oracle = oracle_values(candidate_env, candidate_env.events)
    reference_oracle = oracle_values(reference_env, reference_env.events)
    oracle_delta = [left - right for left, right in zip(candidate_oracle, reference_oracle)]
    global_names = ["other_order_lateness", "unmet_demand", "energy", "reassignment", "global_utility_delta"]
    global_units = ["count", "count", "energy_units", "count", "reward"]
    expected_nonzero = {
        name for name, value in zip(global_names, oracle_delta)
        if abs(value) > 1e-9
    }
    assert len([reference_action]) == 1
    assert len(non_target_ids) == 1
    assert candidate_env._tasks["causal-delivery-c"]["on_time"] is False
    assert reference_env._tasks["causal-delivery-c"]["on_time"] is True
    assert expected_nonzero == {"other_order_lateness", "global_utility_delta"}
    paired = paired_global_targets(
        candidate_env,
        reference_env,
        candidate_return,
        reference_return,
        candidate_env.events,
        reference_env.events,
        window_start=8,
        exclude_task_ids={"causal-delivery-a"},
    )
    assert paired["names"] == global_names
    assert paired["units"] == global_units
    assert paired["values"] == oracle_delta
    assert paired["valid"] == [True, True, True, True, True]
    assert paired["unknown_reasons"] == [None, None, None, None, None]
