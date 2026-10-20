from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
CIVILIAN_MODULE_ROOT = PACKAGE_ROOT / "civilian"
if str(CIVILIAN_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(CIVILIAN_MODULE_ROOT))

from civilian.event_environment import EventReliefEnv
from civilian import gain_oracle
from civilian.counterfactual_pairs import (
    CONTINUATION_RECEIPT_SCHEMA,
    capture_snapshot,
    prepare_suffix_snapshot,
    run_paired_counterfactual,
)
from civilian.fixed_replanner import REPLANNER_VERSION
from civilian.gain_oracle import validate_gain_label
from civilian.v33_scenarios import (
    CONDITIONS,
    HORIZON_MINUTES,
    build_v33_pilot_scenarios,
    make_v33_scenario,
    sample_v33_future_suffix,
)


NOOP = {
    "action_type": "reconstruct", "scope": "NOOP", "noop": True,
    "queue_operations": [], "dispatches": [],
}


def test_pilot_density_matrix_has_region_arrivals_and_public_c1_inputs() -> None:
    scenarios = build_v33_pilot_scenarios(seed_variants=(930101, 930102, 930103))
    assert len(scenarios) == 18

    for scenario in scenarios:
        base = scenario["base_scenario"]
        public_ids = {row["id"] for row in base["tasks"] if row["known_initial"]}
        queued_ids = [task_id for queue in scenario["queue_plan"].values() for task_id in queue]
        arrivals = [row for row in scenario["event_tape"] if row["kind"] == "task_arrival"]
        updates = [row for row in scenario["event_tape"] if row["kind"] == "deadline_change"]

        assert set(queued_ids) <= public_ids
        assert len(arrivals) == 9
        assert {row["task_id"] for row in arrivals} == {
            row["id"] for row in base["tasks"]
            if row["kind"] == "inspection" and not row["known_initial"]
        }
        assert all(22 <= row["time"] <= 26 for row in arrivals)
        assert len(updates) == int(bool(public_ids))
        assert all(row["task_id"] in public_ids for row in updates)
        if scenario["targets_per_region"] == 1:
            assert not queued_ids


def test_all_conditions_share_nine_arrivals_and_paired_suffix_slots() -> None:
    for condition in CONDITIONS:
        scenario = make_v33_scenario(
            split="pilot", condition=condition, seed_variant=930101, instance_index=0,
        )
        arrivals = [row for row in scenario["event_tape"] if row["kind"] == "task_arrival"]
        assert len(arrivals) == 9
        assert {row["task_id"] for row in arrivals} == {
            row["id"] for row in scenario["base_scenario"]["tasks"]
            if row["kind"] == "inspection" and not row["known_initial"]
        }
        suffix = sample_v33_future_suffix(
            scenario, after_time=0, horizon_minutes=30, suffix_seed=4567,
        )
        suffix_arrivals = [row for row in suffix if row["kind"] == "task_arrival"]
        assert len(suffix_arrivals) == 9
        assert {row["task_id"] for row in suffix_arrivals} == {row["task_id"] for row in arrivals}


def test_hidden_search_sites_become_public_only_when_arrivals_execute() -> None:
    scenario = next(
        row for row in build_v33_pilot_scenarios(seed_variants=(930101,))
        if row["task_mode"] == "search_rescue" and row["targets_per_region"] == 3
    )
    hidden_ids = {
        row["id"] for row in scenario["base_scenario"]["tasks"]
        if row["kind"] == "inspection" and not row["known_initial"]
    }
    env = EventReliefEnv(copy.deepcopy(scenario))
    env.reset()
    assert hidden_ids.isdisjoint({row["id"] for row in env.public_observation()["tasks"]})

    last_arrival = max(row["time"] for row in scenario["event_tape"] if row["kind"] == "task_arrival")
    while env.time < last_arrival:
        env.step(copy.deepcopy(NOOP))
    assert hidden_ids <= {row["id"] for row in env.public_observation()["tasks"]}


def test_full_remaining_suffix_passes_contract_while_label_stays_at_30_minutes() -> None:
    scenario = make_v33_scenario(
        split="pilot", condition="task_update_or_arrival", seed_variant=930101,
        instance_index=0, task_mode="search_rescue", targets_per_region=3,
    )
    env = EventReliefEnv(copy.deepcopy(scenario))
    env.reset()
    initial_public_task_count = len(env.public_observation()["tasks"])
    while env.time < 12 and len(env.public_observation()["tasks"]) == initial_public_task_count:
        env.step(copy.deepcopy(NOOP))
    assert len(env.public_observation()["tasks"]) > initial_public_task_count

    snapshot = capture_snapshot(
        env, state_id="v33-suffix-contract-state-930101",
        scenario_id=scenario["parent_id"], parent_family=scenario["parent_family"],
        split="pilot", event_type="task_discovered_new_rescue", forced=False,
        replanner_version=REPLANNER_VERSION,
    )
    suffix_seed = 930201
    future_events = sample_v33_future_suffix(
        scenario, after_time=int(env.time),
        horizon_minutes=HORIZON_MINUTES - int(env.time), suffix_seed=suffix_seed,
    )
    assert sum(row["kind"] == "task_arrival" for row in future_events) == 9
    assert sum(row["kind"] == "deadline_change" for row in future_events) == 1
    suffix = prepare_suffix_snapshot(
        snapshot, future_events=future_events, suffix_seed=suffix_seed,
        env_factory=EventReliefEnv,
    )
    label = run_paired_counterfactual(
        snapshot, replan_action=copy.deepcopy(NOOP), suffixes=[suffix],
        replanner_version=REPLANNER_VERSION, replanner_cpu_seconds=0.0,
        horizon_minutes=30,
    )
    assert label["valid_mask"] is True
    assert label["pairs"][0]["horizon_actual_minutes"] == 30
    assert validate_gain_label(label)["valid"] is True


def test_unresolved_post_state_safety_masks_label_and_oracle_reads_raw_receipts() -> None:
    scenario = make_v33_scenario(
        split="pilot", condition="none_or_light", seed_variant=930101,
        instance_index=0, task_mode="search_rescue", targets_per_region=3,
    )
    env = EventReliefEnv(copy.deepcopy(scenario))
    env.reset()
    snapshot = capture_snapshot(
        env, state_id="v33-unresolved-safety-state-930101",
        scenario_id=scenario["parent_id"], parent_family=scenario["parent_family"],
        split="pilot", event_type="focused_unresolved_safety", forced=False,
        replanner_version=REPLANNER_VERSION,
    )
    suffix_seed = 930202
    suffix = prepare_suffix_snapshot(
        snapshot,
        future_events=sample_v33_future_suffix(
            scenario, after_time=0, horizon_minutes=HORIZON_MINUTES, suffix_seed=suffix_seed,
        ),
        suffix_seed=suffix_seed,
        env_factory=EventReliefEnv,
    )
    empty_actions_sha = hashlib.sha256(b"[]").hexdigest()

    def unresolved_continuation(_env: EventReliefEnv) -> dict:
        return {
            "receipt_schema": CONTINUATION_RECEIPT_SCHEMA,
            "action": copy.deepcopy(NOOP),
            "planner_invoked": False,
            "planner_cpu_seconds": 0.0,
            "safety_monitor_cpu_seconds": 0.0,
            "policy_version": "focused-unresolved-safety-test",
            "safety_apply_receipt": {
                "schema": "gain-live-safety-disposition/1", "status": "unresolved",
                "resolved": False, "pre_public_sha256": "0" * 64,
                "post_public_sha256": "1" * 64, "actions": [],
                "actions_sha256": empty_actions_sha,
            },
            "unknown_reason": "post_state_safety_unresolved",
        }

    label = run_paired_counterfactual(
        snapshot, replan_action=copy.deepcopy(NOOP), env_factory=EventReliefEnv,
        suffixes=[suffix], continuation_policy=unresolved_continuation,
        replanner_version=REPLANNER_VERSION, replanner_cpu_seconds=0.0,
        horizon_minutes=2,
    )

    assert label["valid_mask"] is False
    assert label["gain_mean"] is None
    assert label["unknown_reason"] == "post_state_safety_unresolved"
    pair = label["pairs"][0]
    assert pair["valid_mask"] is False
    assert pair["gain"] is None
    assert pair["keep"]["cost"] is None and pair["replan"]["cost"] is None
    for arm in ("keep", "replan"):
        trace = pair[arm]["trace"]
        assert trace["valid_mask"] is False
        assert trace["unknown_reason"] == "post_state_safety_unresolved"
        assert trace["steps"]
        assert trace["raw_truth"]["time"] == 2
        assert trace["continuation_policy_receipts"][0]["safety_apply_receipt"]["resolved"] is False
    oracle = validate_gain_label(label)
    assert oracle["valid"] is True
    assert oracle["label_valid_mask"] is False
    assert oracle["recomputed_gain_mean"] is None


def test_oracle_rejects_resigned_valid_label_with_live_unresolved_receipt() -> None:
    scenario = make_v33_scenario(
        split="pilot", condition="task_update_or_arrival", seed_variant=930101,
        instance_index=0, task_mode="search_rescue", targets_per_region=3,
    )
    env = EventReliefEnv(copy.deepcopy(scenario))
    env.reset()
    initial_public_task_count = len(env.public_observation()["tasks"])
    while env.time < 12 and len(env.public_observation()["tasks"]) == initial_public_task_count:
        env.step(copy.deepcopy(NOOP))
    snapshot = capture_snapshot(
        env, state_id="v33-oracle-live-only-tamper-930101",
        scenario_id=scenario["parent_id"], parent_family=scenario["parent_family"],
        split="pilot", event_type="focused_live_receipt_tamper", forced=False,
        replanner_version=REPLANNER_VERSION,
    )
    suffix_seed = 930203
    suffix = prepare_suffix_snapshot(
        snapshot,
        future_events=sample_v33_future_suffix(
            scenario, after_time=int(env.time),
            horizon_minutes=HORIZON_MINUTES - int(env.time), suffix_seed=suffix_seed,
        ),
        suffix_seed=suffix_seed, env_factory=EventReliefEnv,
    )
    label = run_paired_counterfactual(
        snapshot, replan_action=copy.deepcopy(NOOP), suffixes=[suffix],
        replanner_version=REPLANNER_VERSION, replanner_cpu_seconds=0.0,
        horizon_minutes=30,
    )
    assert label["valid_mask"] is True
    pair = label["pairs"][0]
    trace = copy.deepcopy(pair["keep"]["trace"])
    assert trace["continuation_policy_receipts"]
    receipt = copy.deepcopy(trace["continuation_policy_receipts"][0])
    live_receipt = {
        "schema": "gain-live-safety-disposition/1", "status": "unresolved",
        "resolved": False, "pre_public_sha256": "0" * 64,
        "post_public_sha256": "1" * 64, "actions": [],
        "actions_sha256": hashlib.sha256(b"[]").hexdigest(),
    }
    receipt["safety_apply_receipt"] = live_receipt
    receipt["unknown_reason"] = "tampered_live_receipt"
    evidence = {
        field: receipt[field]
        for field in gain_oracle.SAFETY_EVIDENCE_FIELDS
        if field in receipt
    }
    receipt["safety_evidence_sha256"] = gain_oracle._digest(evidence)
    trace["continuation_policy_receipts"][0] = receipt
    trace["unresolved_safety_disposition_count"] = 1
    trace["unresolved_safety_unknown_reasons"] = ["tampered_live_receipt"]
    trace["trace_sha256"] = gain_oracle._digest({
        key: value for key, value in trace.items() if key != "trace_sha256"
    })
    pair["keep"]["trace"] = trace
    pair["unresolved_safety_disposition_count"] = 1
    pair["unresolved_safety_unknown_reasons"] = ["tampered_live_receipt"]
    pair["pair_sha256"] = gain_oracle._digest({
        key: value for key, value in pair.items() if key != "pair_sha256"
    })
    label["pairs"][0] = pair
    label["label_sha256"] = gain_oracle._digest({
        key: value for key, value in label.items() if key != "label_sha256"
    })

    with pytest.raises(ValueError, match="GAIN_ORACLE_VALID_TRACE_HAS_UNRESOLVED_SAFETY"):
        validate_gain_label(label)


def test_cooperative_search_completion_publishes_dependent_rescue() -> None:
    scenario = make_v33_scenario(
        split="pilot", condition="none_or_light", seed_variant=930103,
        instance_index=1, task_mode="search_rescue", targets_per_region=3,
    )
    env = EventReliefEnv(copy.deepcopy(scenario))
    env.reset()
    rescue = next(
        row for row in scenario["base_scenario"]["tasks"]
        if row["kind"] == "delivery" and row.get("discovered_by")
        and next(task for task in scenario["base_scenario"]["tasks"]
                 if task["id"] == row["discovered_by"])["known_initial"]
    )
    search_id = str(rescue["discovered_by"])
    search = env._base._tasks[search_id]
    source = str(search["assignee"])
    source_platform = env._base._platforms[source]
    partner = next(
        row["id"] for row in env._base._platforms.values()
        if row["type"] == "uav" and row["home_region"] == source_platform["home_region"]
        and row["id"] != source
    )
    env._queues = {platform_id: [] for platform_id in env._queues}
    env._queues[source] = [search_id]
    env._base._queues = copy.deepcopy(env._queues)
    rescue_id = str(rescue["id"])
    for mapping in (env._task_specs, env._base._task_spec, env._base._tasks):
        mapping[rescue_id]["report_delay"] = 1
    for mapping in (env._task_specs, env._base._task_spec, env._original_task_specs):
        mapping[search_id]["service_minutes"] = 2.0
    for mapping in (env._base._tasks, env._base._public_tasks, env._reported_tasks):
        mapping[search_id]["service_minutes"] = 2.0
    position = list(search["position"])
    region_row, region_col = map(int, source_platform["home_region"][1:])
    bounds = (0, 33, 66, 100)
    partner_position = [
        float(bounds[region_col] if position[0] > (bounds[region_col] + bounds[region_col + 1]) / 2
              else bounds[region_col + 1] - 0.1),
        float(bounds[region_row] if position[1] > (bounds[region_row] + bounds[region_row + 1]) / 2
              else bounds[region_row + 1] - 0.1),
    ]
    for platform_id, target_position in ((source, position), (partner, partner_position)):
        for platform_map in (env._base._platforms, env._base._public_platforms, env._reported_platforms):
            platform_map[platform_id]["position"] = copy.deepcopy(target_position)
    observation = env.public_observation()
    env._candidate_validation_context = {
        "time": env.time, "observation": observation, "dispatch_catalog": {},
    }
    action = {
        "action_type": "reconstruct", "scope": "GLOBAL_REPLAN", "noop": False,
        "queue_operations": [{
            "op": "cooperate", "task_id": search_id, "from_platform": source,
            "to_platform": partner, "share": 0.5,
        }],
        "dispatches": [],
    }
    _, _, _, _, first_info = env.step(action)
    child_ids = env._split_manifests[search_id]["child_ids"]
    assert {row["task_id"] for row in first_info["operational_dispatches"]} == set(child_ids)
    completed_children = [row for row in env._raw_truth()["subtasks"] if row["id"] in child_ids and row["status"] == "completed"]
    assert len(completed_children) == 1
    assert rescue_id not in {row["id"] for row in env.public_observation()["tasks"]}

    for _ in range(50):
        if search_id in {row["id"] for row in env.public_observation()["tasks"] if row["status"] == "completed"}:
            break
        env.step(copy.deepcopy(NOOP))

    public_tasks = {row["id"]: row for row in env.public_observation()["tasks"]}
    assert public_tasks[search_id]["status"] == "completed"
    assert rescue_id not in public_tasks
    env.step(copy.deepcopy(NOOP))
    public_tasks = {row["id"]: row for row in env.public_observation()["tasks"]}
    assert rescue_id in public_tasks
    assert public_tasks[rescue_id]["discovered"] is True
    events = [event for step in env._steps for event in step["raw_events"]]
    completion = next(event for event in events if
                      event["type"] == "logical_task_completed" and event["task_id"] == search_id)
    discovery = next(event for event in events if
                     event["type"] == "report_received" and event.get("kind") == "task_discovered"
                     and event["task_id"] == rescue_id)
    assert discovery["inspection_id"] == search_id
    assert discovery["time"] >= completion["time"]
    assert public_tasks[rescue_id]["report_received_at"] == discovery["time"]
    generated = next(event for event in events if
                     event["type"] == "report_generated" and event.get("kind") == "task_discovered"
                     and event["task_id"] == rescue_id)
    assert generated["inspection_id"] == search_id
