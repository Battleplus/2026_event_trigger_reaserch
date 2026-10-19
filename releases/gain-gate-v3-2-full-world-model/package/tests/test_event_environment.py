from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import event_scenarios
from event_environment import EventReliefEnv


_SCENARIOS = event_scenarios.build_event_scenarios()


def _noop() -> dict:
    return {
        "action_type": "reconstruct", "scope": "NOOP", "noop": True,
        "queue_operations": [], "dispatches": [],
    }


def _recompute_reward_primitives(events: list[dict]) -> dict:
    owner_changes = len({event.get("task_id") for event in events if event.get("type") == "task_owner_change"})
    queue_reorders = sum(event.get("type") == "queue_reorder" for event in events)
    reconstruction_size = owner_changes + queue_reorders
    return {
        "completion_count": sum(event.get("type") == "logical_task_completed" for event in events),
        "delay_minutes": sum(float(event.get("minutes", 0.0)) for event in events if event.get("type") == "delay_accrued"),
        "energy_units": sum(float(event.get("energy", 0.0)) for event in events
                             if event.get("type") in {"move", "service"}),
        "risk_units": sum(float(event.get("environmental_risk", 0.0)) for event in events
                          if event.get("type") == "move")
                      + sum(event.get("type") in {"task_failed", "platform_failure"} for event in events),
        "communication_messages": sum(int(event.get("message_count", 0)) for event in events
                                       if event.get("type") == "communication_tx"),
        "switch_count": len({event.get("task_id") for event in events if event.get("type") == "task_switch"}),
        "reconstruction_count": int(reconstruction_size > 0),
        "reconstruction_size": reconstruction_size,
    }


def test_canonical_noop_validation_does_not_build_dispatch_catalog() -> None:
    env = EventReliefEnv(copy.deepcopy(_SCENARIOS[0]))
    env.reset()

    def unexpected_catalog() -> list[dict]:
        raise AssertionError("canonical NOOP should not enumerate dispatch candidates")

    env.candidate_dispatches = unexpected_catalog
    normalized, issues = env._check_action(_noop())
    assert normalized == _noop()
    assert issues == []


def test_episode_metrics_report_original_task_denominators_and_validity() -> None:
    env = EventReliefEnv(copy.deepcopy(_SCENARIOS[0]))
    env.reset()

    summary = env.episode_metrics()
    assert summary["task_count"] == 12
    assert len(summary["task_outcomes"]) == 12
    assert summary["metrics"]["completion_rate"] == {
        "value": 0.0, "valid": True, "numerator": 0, "denominator": 12, "unit": "fraction",
    }
    assert summary["metrics"]["mean_delay_minutes"]["value"] == 0.0
    assert summary["metrics"]["mean_delay_minutes"]["denominator"] == 12
    assert summary["metrics"]["on_time_rate"]["valid"] is False
    assert summary["metrics"]["on_time_rate"]["denominator"] == 0
    assert summary["metrics"]["global_utility"]["value"] == 0.0
    assert summary["metrics"]["global_utility"]["valid"] is True
    assert summary["validity_mask"]["on_time_rate"] is False
    assert summary["denominators"]["completion_rate"] == 12
    assert summary["effective_scope_ratios"]["LOCAL_REPAIR"]["valid"] is False
    assert summary["reconstruction_size"] == 0
    assert summary["metrics"]["time_violation_rate"]["denominator"] == 12
    assert summary["metrics"]["time_violation_rate"]["valid"] is True


def test_unfinished_task_delay_accrues_and_unresolved_critical_rate_stays_unknown() -> None:
    scenario = copy.deepcopy(_SCENARIOS[0])
    scenario["event_tape"].append({
        "time": 1, "kind": "capability_loss", "platform_id": "n02-uav-1",
        "capability": "deliver", "event_id": "test-capability-loss",
    })
    scenario["event_tape_id"] = event_scenarios._digest(scenario["event_tape"])
    env = EventReliefEnv(scenario)
    env.reset()
    task_id = "n02-task-01"
    env._base._tasks[task_id]["deadline"] = 3
    env._base._public_tasks[task_id]["deadline"] = 3
    env._reported_tasks[task_id]["deadline"] = 3
    env._base._tasks["n02-task-09"]["priority"] = 5.2

    for _ in range(4):
        env.step(_noop())

    summary = env.episode_metrics()
    task_outcome = next(row for row in summary["task_outcomes"] if row["task_id"] == task_id)
    assert task_outcome["completed"] is False
    assert task_outcome["delay_minutes"] == 1.0
    assert summary["metrics"]["mean_delay_minutes"]["value"] == 1.0 / 12.0
    assert summary["metrics"]["critical_failure_rate"]["valid"] is False
    assert summary["validity_mask"]["critical_failure_rate"] is False


def test_split_delivery_conserves_original_demand_and_rolls_up_completion() -> None:
    env = EventReliefEnv(copy.deepcopy(_SCENARIOS[0]))
    observation = env.reset()
    task_id = "n02-task-01"
    task = next(row for row in observation["tasks"] if row["id"] == task_id)
    owner = task["assignee"]
    target = "n02-uav-2"
    target_head = env._queues[target][0]
    action = {
        "action_type": "reconstruct", "scope": "GLOBAL_REPLAN", "noop": False,
        "queue_operations": [{
            "op": "split", "task_id": task_id, "from_platform": owner,
            "to_platform": target, "share": 0.4, "before_task_id": target_head,
        }],
        "dispatches": [],
    }

    env.step(action)
    observation = env.public_observation()
    parent = next(row for row in observation["tasks"] if row["id"] == task_id)
    children = [next(row for row in observation["tasks"] if row["id"] == child_id)
                for child_id in parent["subtask_ids"]]
    original_quantity = float(task["quantity"])
    assert math.isclose(sum(float(child["quantity"]) for child in children), original_quantity)
    assert env.episode_metrics()["task_count"] == 12

    for _ in range(60):
        parent_truth = next(row for row in env._raw_truth()["tasks"] if row["id"] == task_id)
        if parent_truth["status"] == "completed":
            break
        env.step(_noop())

    summary = env.episode_metrics()
    parent_truth = next(row for row in env._raw_truth()["tasks"] if row["id"] == task_id)
    outcome = next(row for row in summary["task_outcomes"] if row["task_id"] == task_id)
    assert parent_truth["status"] == "completed"
    assert math.isclose(parent_truth["delivered_quantity"], original_quantity)
    assert parent_truth["unmet_demand_quantity"] == 0.0
    assert outcome["completed"] is True
    assert len(summary["task_outcomes"]) == 12


def test_cooperative_workloads_dispatch_as_a_pair() -> None:
    env = EventReliefEnv(copy.deepcopy(_SCENARIOS[0]))
    observation = env.reset()
    task_id = "n02-task-01"
    owner = next(row for row in observation["tasks"] if row["id"] == task_id)["assignee"]
    target = "n02-uav-2"
    action = {
        "action_type": "reconstruct", "scope": "GLOBAL_REPLAN", "noop": False,
        "queue_operations": [{
            "op": "cooperate", "task_id": task_id, "from_platform": owner,
            "to_platform": target, "share": 0.5, "before_task_id": env._queues[target][0],
        }],
        "dispatches": [],
    }

    _, _, _, _, info = env.step(action)
    parent = next(row for row in env.public_observation()["tasks"] if row["id"] == task_id)
    children = set(parent["subtask_ids"])
    child_dispatches = [row for row in info["operational_dispatches"] if row["task_id"] in children]
    dispatched = {row["task_id"] for row in child_dispatches}
    assert len(children) == 2
    assert dispatched == children
    assert len({row["platform_id"] for row in child_dispatches}) == 2


def test_pending_dispatch_is_retained_when_platform_dedup_skips_it() -> None:
    env = EventReliefEnv(copy.deepcopy(_SCENARIOS[0]))
    env.reset()
    platform_id = "n02-uav-3"
    first_task, second_task = env._queues[platform_id][:2]
    env._pending_communications = [
        {"platform_id": platform_id, "task_id": first_task, "delivery_time": env.time,
         "retry_count": 0, "reason": "test"},
        {"platform_id": platform_id, "task_id": second_task, "delivery_time": env.time,
         "retry_count": 0, "reason": "test"},
    ]
    env.step(_noop())
    retained = [row for row in env._pending_communications
                if row["platform_id"] == platform_id and row["task_id"] == second_task]
    assert retained
    assert retained[0]["delivery_time"] == env.time
    assert retained[0]["reason"] == "dispatch_dedup_retry"


def test_queue_projection_catches_cumulative_deadline_overrun() -> None:
    env = EventReliefEnv(copy.deepcopy(_SCENARIOS[0]))
    observation = env.reset()
    platform_id = "n02-uav-3"
    first_id, second_id = env._queues[platform_id]
    platform = next(row for row in observation["platforms"] if row["id"] == platform_id)
    second = next(row for row in observation["tasks"] if row["id"] == second_id)
    individual = env._estimate_public_mission(platform, second, observation)
    second["deadline"] = int(individual["estimated_completion"])

    single_queues = {key: [] for key in env._queues}
    single_queues[platform_id] = [second_id]
    single_issues = env._queue_schedule_issues(observation, single_queues)
    combined_queues = copy.deepcopy(single_queues)
    combined_queues[platform_id] = [first_id, second_id]
    combined_issues = env._queue_schedule_issues(observation, combined_queues)

    assert not any(issue[0] == "cumulative_deadline_exceeded" and issue[1] == second_id for issue in single_issues)
    assert any(issue[0] == "cumulative_deadline_exceeded" and issue[1] == second_id for issue in combined_issues)


def test_disconnected_remote_telemetry_stays_stale() -> None:
    scenario = copy.deepcopy(_SCENARIOS[3])
    scenario["base_scenario"]["platforms"][-2]["speed_km_min"] = 0.4
    next(task for task in scenario["base_scenario"]["tasks"] if task["id"] == "n02-task-07")["position"] = [99.0, 99.0]
    communication_event = next(row for row in scenario["event_tape"] if row["kind"] == "communication_drop")
    communication_event["time"] = 2
    scenario["event_tape_id"] = event_scenarios._digest(scenario["event_tape"])
    env = EventReliefEnv(scenario)
    env.reset()

    env.step(_noop())
    observation, _, _, _, _ = env.step(_noop())
    heli_id = "n02-heli-1"
    link = next(row for row in observation["communication"]
                if {row["source"], row["target"]} == {"n02-uav-4", heli_id})
    assert link["connected"] is False
    reported_position = copy.deepcopy(env._reported_platforms[heli_id]["position"])

    observation, _, _, _, _ = env.step(_noop())
    truth_position = env._base._platforms[heli_id]["position"]
    public_position = next(row["position"] for row in observation["platforms"] if row["id"] == heli_id)
    assert truth_position != reported_position
    assert public_position == reported_position


def test_remote_delivery_has_feasible_helicopter_dispatches() -> None:
    env = EventReliefEnv(copy.deepcopy(_SCENARIOS[0]))
    env.reset()
    for _ in range(24):
        env.step(_noop())

    task_id = "n02-task-11"
    platforms = {row["id"]: row for row in env.public_observation()["platforms"]}
    candidates = [row for row in env.candidate_dispatches()
                  if row.get("task_id") == task_id and platforms[row["platform_id"]]["type"] == "helicopter"]
    assert candidates
    assert all(row["type"] == "dispatch" for row in candidates)


def test_environment_perturbation_changes_local_but_not_remote_movement_primitives() -> None:
    scenario = copy.deepcopy(_SCENARIOS[4])
    local_depot = next(row for row in scenario["base_scenario"]["depots"] if row["id"] == "n02-local-depot")
    local_depot["position"] = [40.95, 40.95]
    local_task = next(row for row in scenario["base_scenario"]["tasks"] if row["id"] == "n02-task-01")
    local_task["position"] = [40.05, 40.05]

    treatment = EventReliefEnv(scenario)
    treatment.reset()
    control = copy.deepcopy(treatment)
    perturbation = next(row for row in control.scenario["event_tape"]
                        if row["kind"] == "environment_perturbation")
    control.scenario["event_tape"] = [row for row in control.scenario["event_tape"]
                                      if row["event_id"] != perturbation["event_id"]]
    control.scenario["event_tape_id"] = event_scenarios._digest(control.scenario["event_tape"])

    # Keep all tasks idle until both treatment arms reach the perturbation time.
    for env in (treatment, control):
        env._queues = {platform_id: [] for platform_id in env._queues}
        env._base._queues = copy.deepcopy(env._queues)
        for task_id in env._base._tasks:
            for mapping in (env._base._tasks, env._base._public_tasks, env._reported_tasks):
                if task_id in mapping:
                    mapping[task_id]["assignee"] = None

    for _ in range(16):
        treatment.step(_noop())
        control.step(_noop())
    assert treatment.time == control.time == 16
    assert perturbation["event_id"] in treatment._event_ids_applied
    assert perturbation["event_id"] not in control._event_ids_applied

    action = {
        "action_type": "reconstruct", "scope": "GLOBAL_REPLAN", "noop": False,
        "queue_operations": [
            {"op": "insert", "task_id": "n02-task-01", "from_platform": None, "to_platform": "n02-uav-1"},
            {"op": "insert", "task_id": "n02-task-07", "from_platform": None, "to_platform": "n02-heli-1"},
        ],
        "dispatches": [
            {"platform_id": "n02-uav-1", "task_id": "n02-task-01"},
            {"platform_id": "n02-heli-1", "task_id": "n02-task-07"},
        ],
    }
    for env in (treatment, control):
        _, issues = env._check_action(action)
        assert issues == []
        env.step(action)
        for _ in range(8):
            env.step(_noop())
        for row in env._steps:
            assert row["reward_primitives"] == _recompute_reward_primitives(row["raw_events"])

    def movement_outcome(env: EventReliefEnv, task_id: str) -> dict:
        moves = [event for step in env._steps for event in step["raw_events"]
                 if event.get("type") == "move" and event.get("task_id") == task_id]
        task = next(row for row in env._raw_truth()["tasks"] if row["id"] == task_id)
        return {
            "energy": sum(float(event["energy"]) for event in moves),
            "duration": sum(float(event["duration"]) for event in moves),
            "risk": sum(float(event["environmental_risk"]) for event in moves),
            "completed_at": task["completed_at"],
            "moves": [{key: event[key] for key in ("from", "to", "distance", "duration", "energy", "environmental_risk")}
                      for event in moves],
        }

    treatment_local = movement_outcome(treatment, "n02-task-01")
    control_local = movement_outcome(control, "n02-task-01")
    assert treatment_local["energy"] > control_local["energy"]
    assert treatment_local["duration"] > control_local["duration"]
    assert treatment_local["risk"] > control_local["risk"]
    assert treatment_local["completed_at"] == control_local["completed_at"] + 1

    treatment_remote = movement_outcome(treatment, "n02-task-07")
    control_remote = movement_outcome(control, "n02-task-07")
    assert treatment_remote == control_remote
