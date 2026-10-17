from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import event_scenarios
from event_environment import EventReliefEnv, generate_reconstruction_candidates


SCENARIO = event_scenarios.build_event_scenarios()[0]
NOOP = {
    "action_type": "reconstruct", "scope": "NOOP", "noop": True,
    "queue_operations": [], "dispatches": [],
}


def _env_at(time: int = 0) -> EventReliefEnv:
    env = EventReliefEnv(copy.deepcopy(SCENARIO))
    env.reset()
    for _ in range(time):
        env.step(NOOP)
    return env


def _queue_after(before: dict[str, list[str]], operation: dict) -> dict[str, list[str]]:
    queues = copy.deepcopy(before)
    op = operation["op"]
    task_id = operation["task_id"]
    if op == "swap":
        other_id = operation["other_task_id"]
        for platform_id, queue in queues.items():
            queues[platform_id] = [
                other_id if item == task_id else task_id if item == other_id else item
                for item in queue
            ]
        return queues
    for queue in queues.values():
        while task_id in queue:
            queue.remove(task_id)
    target = operation["to_platform"]
    destination = queues.setdefault(target, [])
    anchor = operation.get("before_task_id")
    if anchor in destination:
        destination.insert(destination.index(anchor), task_id)
    else:
        destination.append(task_id)
    return queues


def _run_queue_action(env: EventReliefEnv, action: dict) -> tuple[dict, dict]:
    normalized, issues = env._check_action(action)
    assert issues == [], issues
    assert normalized == action
    before_queues = copy.deepcopy(env._queues)
    before_active = {
        platform_id: row.get("active_task_id")
        for platform_id, row in env._base._platforms.items()
    }
    env.step(action)
    operation = action["queue_operations"][0]
    expected = _queue_after(before_queues, operation)
    assert env._queues == expected
    after_active = {
        platform_id: row.get("active_task_id")
        for platform_id, row in env._base._platforms.items()
    }
    # Queue commits may let an idle platform start its new head during the
    # minute step, but they may never replace a task that was already active.
    for platform_id, task_id in before_active.items():
        if task_id is not None:
            assert after_active[platform_id] == task_id
    return expected, env._raw_truth()


def test_transfer_catalog_has_two_pure_queue_actions_on_busy_targets() -> None:
    env = _env_at(1)
    candidates = generate_reconstruction_candidates(env, max_candidates=16)
    actions = [
        row["full_environment_action"] for row in candidates
        if row["full_environment_action"]["queue_operations"]
        and row["full_environment_action"]["queue_operations"][0]["op"] == "transfer"
    ]
    assert len(actions) >= 2
    outcomes = []
    for action in actions[:2]:
        assert action["dispatches"] == []
        clone = copy.deepcopy(env)
        outcomes.append(_run_queue_action(clone, action)[0])
    assert outcomes[0] != outcomes[1]


def test_exchange_cases_are_two_distinct_legal_queue_mutations() -> None:
    env = _env_at()
    actions = []
    for left, right, source, target in (
        ("n02-task-01", "n02-task-02", "n02-uav-1", "n02-uav-2"),
        ("n02-task-07", "n02-task-08", "n02-heli-1", "n02-heli-2"),
    ):
        actions.append({
            "action_type": "reconstruct", "scope": "GLOBAL_REPLAN", "noop": False,
            "queue_operations": [{
                "op": "swap", "task_id": left, "other_task_id": right,
                "from_platform": source, "to_platform": target,
            }], "dispatches": [],
        })
    outcomes = [_run_queue_action(copy.deepcopy(env), action)[0] for action in actions]
    assert outcomes[0] != outcomes[1]


def test_new_task_insertion_has_two_independent_destinations() -> None:
    env = _env_at()
    task_id = "n02-task-01"
    for queue in env._queues.values():
        while task_id in queue:
            queue.remove(task_id)
    for mapping in (env._base._tasks, env._base._public_tasks, env._reported_tasks):
        mapping[task_id]["assignee"] = None
        mapping[task_id]["status"] = "pending"
        mapping[task_id]["phase"] = "pending"
    env._base._queues = copy.deepcopy(env._queues)
    candidates = generate_reconstruction_candidates(env, max_candidates=16)
    actions = [
        row["full_environment_action"] for row in candidates
        if row["full_environment_action"]["queue_operations"]
        and row["full_environment_action"]["queue_operations"][0]["op"] == "insert"
        and row["full_environment_action"]["queue_operations"][0]["task_id"] == task_id
    ]
    assert len(actions) >= 2
    outcomes = [_run_queue_action(copy.deepcopy(env), action)[0] for action in actions[:2]]
    assert outcomes[0] != outcomes[1]


def test_cooperative_task_has_two_physically_distinct_legal_outcomes() -> None:
    env = _env_at()
    owner = "n02-uav-1"
    task_id = "n02-task-01"
    actions = []
    for target in ("n02-uav-2", "n02-uav-3"):
        actions.append({
            "action_type": "reconstruct", "scope": "GLOBAL_REPLAN", "noop": False,
            "queue_operations": [{
                "op": "cooperate", "task_id": task_id, "from_platform": owner,
                "to_platform": target, "share": 0.5,
                "before_task_id": env._queues[target][0],
            }], "dispatches": [],
        })
    outcomes = []
    for action in actions:
        clone = copy.deepcopy(env)
        normalized, issues = clone._check_action(action)
        assert issues == [], issues
        assert normalized == action
        clone.step(action)
        parent = next(row for row in clone._raw_truth()["tasks"] if row["id"] == task_id)
        children = [row for row in clone._raw_truth()["subtasks"] if row["parent_task_id"] == task_id]
        assert parent["status"] == "pending"
        assert len(children) == 2
        assert {row["assignee"] for row in children} == {owner, action["queue_operations"][0]["to_platform"]}
        assert sum(float(row["workload_share"]) for row in children) == 1.0
        outcomes.append(tuple(sorted(row["assignee"] for row in children)))
    assert outcomes[0] != outcomes[1]


def test_cross_region_transfer_can_be_recomputed_in_both_directions() -> None:
    env = _env_at()
    forward = {
        "action_type": "reconstruct", "scope": "GLOBAL_REPLAN", "noop": False,
        "queue_operations": [{
            "op": "transfer", "task_id": "n02-task-08",
            "from_platform": "n02-heli-2", "to_platform": "n02-heli-1",
        }], "dispatches": [],
    }
    reverse = {
        "action_type": "reconstruct", "scope": "GLOBAL_REPLAN", "noop": False,
        "queue_operations": [{
            "op": "transfer", "task_id": "n02-task-08",
            "from_platform": "n02-heli-1", "to_platform": "n02-heli-2",
        }], "dispatches": [],
    }
    clone = copy.deepcopy(env)
    _run_queue_action(clone, forward)
    _run_queue_action(clone, reverse)
    assert clone._queues == env._queues
