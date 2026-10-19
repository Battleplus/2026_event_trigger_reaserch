from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from civilian.event_environment import EventReliefEnv
from civilian.v3_scenarios import build_v3_gain_scenario


NOOP = {
    "action_type": "reconstruct", "scope": "NOOP", "noop": True,
    "queue_operations": [], "dispatches": [],
}


def _fresh_constructed_case(seed: int) -> tuple[EventReliefEnv, dict[str, Any]]:
    scenario = build_v3_gain_scenario(
        split="pilot", condition="none_or_light", seed_variant=seed, instance_index=0,
    )
    env = EventReliefEnv(scenario)
    env.reset()
    # Constructed public-state probe: preserve the 30-platform, 54-task, 3x3
    # contract while clearing unrelated queues so each tested operation is isolated.
    env._queues = {platform_id: [] for platform_id in env._queues}
    env._base._queues = copy.deepcopy(env._queues)
    for task_id in env._original_task_specs:
        for mapping in (env._base._tasks, env._base._public_tasks, env._reported_tasks):
            if task_id in mapping:
                mapping[task_id].update({
                    "status": "pending", "phase": "pending", "assignee": None,
                    "active_platform_id": None, "started_at": None, "completed_at": None,
                })
    for platform_id in env._reported_platforms:
        for mapping in (env._base._platforms, env._base._public_platforms, env._reported_platforms):
            if platform_id in mapping:
                mapping[platform_id]["active_task_id"] = None
    env._pending_communications = []
    env._communication = []
    return env, scenario


def _platform(env: EventReliefEnv, platform_id: str) -> dict:
    return next(row for row in env._reported_platforms.values() if row["id"] == platform_id)


def _set_platform_position(env: EventReliefEnv, platform_id: str, position: list[float]) -> None:
    for mapping in (env._base._platforms, env._base._public_platforms, env._reported_platforms):
        mapping[platform_id]["position"] = list(position)


def _set_connected_pair(env: EventReliefEnv, left: str, right: str) -> None:
    env._communication = [{
        "source": left, "target": right, "quality": 0.99,
        "latency_minutes": 0.0, "loss_rate": 0.01,
    }]


def _assign_constructed_task(
    env: EventReliefEnv, task_id: str, owner: str, *, position: list[float] | None = None,
    service_minutes: float = 0.2,
) -> dict:
    task = env._base._tasks[task_id]
    spec = env._task_specs[task_id]
    target_position = list(position or task["position"])
    for mapping in (env._base._tasks, env._base._public_tasks, env._reported_tasks):
        mapping[task_id].update({
            "status": "pending", "phase": "pending", "assignee": owner,
            "active_platform_id": None, "position": target_position,
            "service_minutes": service_minutes,
        })
    for mapping in (env._task_specs, env._base._task_spec, env._original_task_specs):
        mapping[task_id].update({"position": target_position, "service_minutes": service_minutes})
    env._queues[owner] = [task_id]
    env._base._queues = copy.deepcopy(env._queues)
    env._reported_progress_counts[task_id] = 0.0
    return env.public_observation()


def _validation_context(env: EventReliefEnv, observation: dict) -> None:
    # Queue-operation cases have no explicit dispatch candidates. Supplying an
    # empty current-time catalog avoids the known expensive all-pairs scan.
    env._candidate_validation_context = {
        "time": env.time, "observation": observation, "dispatch_catalog": {},
    }


def _operation_action(operation: dict) -> dict:
    return {
        "action_type": "reconstruct", "scope": "GLOBAL_REPLAN", "noop": False,
        "queue_operations": [operation], "dispatches": [],
    }


def _task_truth(env: EventReliefEnv, task_id: str) -> dict:
    return next(row for row in env._raw_truth()["tasks"] if row["id"] == task_id)


def _platform_truth(env: EventReliefEnv, platform_id: str) -> dict:
    return next(row for row in env._raw_truth()["platforms"] if row["id"] == platform_id)


def _completion_reward(env: EventReliefEnv) -> float:
    return sum(float(step.get("reward_terms", {}).get("completion", 0.0)) for step in env._steps)


def run_regional_execution_probe() -> dict[str, Any]:
    cases: dict[str, Any] = {}

    # Neighboring-region UAV takeover of a constructed r00 inspection task.
    env, scenario = _fresh_constructed_case(741911)
    platforms = list(env._reported_platforms.values())
    owner = next(p["id"] for p in platforms if p["type"] == "uav" and p["home_region"] == "r00")
    neighbor = next(p["id"] for p in platforms if p["type"] == "uav" and p["home_region"] == "r01")
    task_id = "v3-task-02"
    _set_platform_position(env, owner, [31.0, 15.0])
    _set_platform_position(env, neighbor, [33.2, 15.0])
    before = _assign_constructed_task(env, task_id, owner, position=[32.8, 15.0])
    _set_connected_pair(env, owner, neighbor)
    before = env.public_observation()
    operation = {"op": "transfer", "task_id": task_id, "from_platform": owner, "to_platform": neighbor}
    _validation_context(env, before)
    _, _, _, _, info = env.step(_operation_action(operation))
    truth = _task_truth(env, task_id)
    cases["neighbor_uav_takeover"] = {
        "constructed": True, "contract": _contract_receipt(env, scenario),
        "task_region": "r00", "source_region": _platform(env, owner)["home_region"],
        "target_region": _platform(env, neighbor)["home_region"], "action": operation,
        "operation_receipt": info["operation_receipts"][0],
        "operational_dispatches": info["operational_dispatches"], "raw_truth_task": truth,
        "raw_truth_target_platform": _platform_truth(env, neighbor),
        "queue_after": copy.deepcopy(env._queues),
    }
    assert cases["neighbor_uav_takeover"]["target_region"] == "r01"
    assert truth["assignee"] == neighbor
    assert neighbor in env._queues and task_id in env._queues[neighbor]
    assert _platform_truth(env, neighbor)["active_task_id"] == task_id

    # Far-region UAV rejection, even with a constructed direct public link.
    env, scenario = _fresh_constructed_case(741912)
    platforms = list(env._reported_platforms.values())
    owner = next(p["id"] for p in platforms if p["type"] == "uav" and p["home_region"] == "r00")
    far = next(p["id"] for p in platforms if p["type"] == "uav" and p["home_region"] == "r22")
    task_id = "v3-task-02"
    _set_platform_position(env, owner, [31.0, 15.0])
    _set_platform_position(env, far, [80.0, 80.0])
    _assign_constructed_task(env, task_id, owner, position=[32.8, 15.0])
    _set_connected_pair(env, owner, far)
    before_truth = copy.deepcopy(env._raw_truth())
    before_time = env.time
    before = env.public_observation()
    operation = {"op": "transfer", "task_id": task_id, "from_platform": owner, "to_platform": far}
    _validation_context(env, before)
    rejection = None
    try:
        env.step(_operation_action(operation))
    except ValueError as exc:
        rejection = str(exc)
    after_truth = env._raw_truth()
    cases["far_region_uav_rejection"] = {
        "constructed": True, "contract": _contract_receipt(env, scenario),
        "task_region": "r00", "source_region": _platform(env, owner)["home_region"],
        "target_region": _platform(env, far)["home_region"], "public_direct_link": True,
        "action": operation, "step_rejection": rejection,
        "raw_truth_unchanged": before_truth == after_truth, "time_before": before_time,
        "time_after": env.time, "raw_truth_task_before": next(t for t in before_truth["tasks"] if t["id"] == task_id),
        "raw_truth_task_after": _task_truth(env, task_id),
    }
    assert rejection is not None and "capability_or_capacity_mismatch" in rejection
    assert before_truth == after_truth and env.time == before_time

    # Helicopters can take over work across responsibility-region boundaries.
    env, scenario = _fresh_constructed_case(741913)
    platforms = list(env._reported_platforms.values())
    owner = next(p["id"] for p in platforms if p["type"] == "uav" and p["home_region"] == "r00")
    heli = next(p["id"] for p in platforms if p["type"] == "helicopter" and p["home_region"] != "r00")
    task_id = "v3-task-02"
    _set_platform_position(env, owner, [31.0, 15.0])
    _set_platform_position(env, heli, [80.0, 80.0])
    _assign_constructed_task(env, task_id, owner, position=[32.8, 15.0])
    _set_connected_pair(env, owner, heli)
    before = env.public_observation()
    operation = {"op": "transfer", "task_id": task_id, "from_platform": owner, "to_platform": heli}
    _validation_context(env, before)
    _, _, _, _, info = env.step(_operation_action(operation))
    truth = _task_truth(env, task_id)
    cases["cross_region_helicopter_takeover"] = {
        "constructed": True, "contract": _contract_receipt(env, scenario),
        "task_region": "r00", "source_region": _platform(env, owner)["home_region"],
        "target_region": _platform(env, heli)["home_region"], "action": operation,
        "operation_receipt": info["operation_receipts"][0],
        "operational_dispatches": info["operational_dispatches"], "raw_truth_task": truth,
        "raw_truth_target_platform": _platform_truth(env, heli),
        "queue_after": copy.deepcopy(env._queues),
    }
    assert truth["assignee"] == heli
    assert _platform_truth(env, heli)["active_task_id"] == task_id
    assert task_id in env._queues[heli]

    # Compare one logical inspection's completion reward as a single task and
    # as an atomic two-platform cooperative workload.
    env_single, scenario_single = _fresh_constructed_case(741914)
    platforms = list(env_single._reported_platforms.values())
    owner = next(p["id"] for p in platforms if p["type"] == "uav" and p["home_region"] == "r00")
    partner = next(p["id"] for p in platforms if p["type"] == "uav" and p["home_region"] == "r00" and p["id"] != owner)
    task_id = "v3-task-02"
    _set_platform_position(env_single, owner, [30.0, 15.0])
    _set_platform_position(env_single, partner, [31.0, 15.0])
    _assign_constructed_task(env_single, task_id, owner, position=[31.0, 15.0], service_minutes=0.2)
    _set_connected_pair(env_single, owner, partner)
    before_single = env_single.public_observation()
    _validation_context(env_single, before_single)
    single_info = None
    env_single.step(copy.deepcopy(NOOP))
    for _ in range(5):
        if _task_truth(env_single, task_id)["status"] == "completed":
            break
        obs = env_single.public_observation()
        _validation_context(env_single, obs)
        _, _, _, _, single_info = env_single.step(copy.deepcopy(NOOP))
    single_truth = _task_truth(env_single, task_id)
    assert single_truth["status"] == "completed"
    single_completion_reward = _completion_reward(env_single)

    env, scenario = _fresh_constructed_case(741914)
    platforms = list(env._reported_platforms.values())
    owner = next(p["id"] for p in platforms if p["type"] == "uav" and p["home_region"] == "r00")
    partner = next(p["id"] for p in platforms if p["type"] == "uav" and p["home_region"] == "r00" and p["id"] != owner)
    task_id = "v3-task-02"
    _set_platform_position(env, owner, [30.0, 15.0])
    _set_platform_position(env, partner, [31.0, 15.0])
    _assign_constructed_task(env, task_id, owner, position=[31.0, 15.0], service_minutes=0.2)
    _set_connected_pair(env, owner, partner)
    before = env.public_observation()
    operation = {"op": "cooperate", "task_id": task_id, "from_platform": owner,
                 "to_platform": partner, "share": 0.5}
    _validation_context(env, before)
    _, _, _, _, info = env.step(_operation_action(operation))
    parent_after_dispatch = next(row for row in env.public_observation()["tasks"] if row["id"] == task_id)
    children = list(parent_after_dispatch["subtask_ids"])
    child_dispatches = [row for row in info["operational_dispatches"] if row["task_id"] in children]
    for _ in range(5):
        if _task_truth(env, task_id)["status"] == "completed":
            break
        obs = env.public_observation()
        _validation_context(env, obs)
        env.step(copy.deepcopy(NOOP))
    cooperative_truth = env._raw_truth()
    logical_task_truth = next(row for row in cooperative_truth["tasks"] if row["id"] == task_id)
    child_truth = [next(row for row in cooperative_truth["subtasks"] if row["id"] == child_id) for child_id in children]
    summary = env.episode_metrics()
    logical_outcomes = [row for row in summary["task_outcomes"] if row["task_id"] == task_id]
    cooperative_completion_reward = _completion_reward(env)
    cases["cooperative_logical_task"] = {
        "constructed": True, "contract": _contract_receipt(env, scenario),
        "operation": operation, "operation_receipt": info["operation_receipts"][0],
        "child_ids": children, "child_dispatches": child_dispatches,
        "child_truth": child_truth, "logical_task_truth": logical_task_truth,
        "logical_task_outcomes": logical_outcomes,
        "episode_task_count": summary["task_count"], "completion_reward_terms": [
            float(step.get("reward_terms", {}).get("completion", 0.0)) for step in env._steps
        ],
        "cooperative_completion_reward": cooperative_completion_reward,
        "single_owner_completion_reward": single_completion_reward,
        "no_duplicate_logical_completion_reward": math.isclose(
            cooperative_completion_reward, single_completion_reward, rel_tol=0.0, abs_tol=1e-9,
        ),
        "raw_events": [event for step in env._steps for event in step["raw_events"]],
    }
    assert len(children) == 2
    assert {row["task_id"] for row in child_dispatches} == set(children)
    assert {row["platform_id"] for row in child_dispatches} == {owner, partner}
    assert logical_task_truth["status"] == "completed"
    assert all(row["status"] == "completed" for row in child_truth)
    assert len(logical_outcomes) == 1 and logical_outcomes[0]["completed"] is True
    assert summary["task_count"] == 54
    assert math.isclose(cooperative_completion_reward, single_completion_reward, rel_tol=0.0, abs_tol=1e-9)

    return {"schema": "v3-regional-execution-probe/1", "constructed_cases": True, "cases": cases}


def _contract_receipt(env: EventReliefEnv, scenario: dict[str, Any]) -> dict[str, Any]:
    observation = env.public_observation()
    return {
        "region_contract": scenario["base_scenario"]["region_contract"],
        "region_boundaries": observation["map"]["region_boundaries"],
        "region_count": len(scenario["regions"]),
        "platform_count": len(observation["platforms"]),
        "task_count": len(env._original_task_specs),
        "runtime_physical_task_count": len(env._base._tasks),
        "public_known_task_count": len(observation["tasks"]),
        "scenario_task_count": len(scenario["base_scenario"]["tasks"]),
        "fixed_runtime_contract": len(env._base._tasks) == 54,
        "uav_count": sum(row["type"] == "uav" for row in observation["platforms"]),
        "helicopter_count": sum(row["type"] == "helicopter" for row in observation["platforms"]),
    }


def test_v3_regional_takeover_and_cooperation_use_real_step_and_truth() -> None:
    result = run_regional_execution_probe()
    assert result["constructed_cases"] is True
    assert set(result["cases"]) == {
        "neighbor_uav_takeover", "far_region_uav_rejection",
        "cross_region_helicopter_takeover", "cooperative_logical_task",
    }
    for case in result["cases"].values():
        contract = case["contract"]
        assert contract["region_contract"] == "relief-regions-3x3-v1"
        assert contract["region_count"] == 9
        assert contract["platform_count"] == 30
        assert contract["scenario_task_count"] == 54
        if case is result["cases"]["cooperative_logical_task"]:
            assert contract["task_count"] == 54
            assert contract["runtime_physical_task_count"] == 56
            assert contract["fixed_runtime_contract"] is False
        else:
            assert contract["task_count"] == 54
            assert contract["runtime_physical_task_count"] == 54
            assert contract["fixed_runtime_contract"] is True
