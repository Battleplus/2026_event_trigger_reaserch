from __future__ import annotations

import copy
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from civilian.event_environment import EventReliefEnv, _coordinator_id
from civilian.gain_safety import assess_queue_safety
from civilian.v3_scenarios import build_v3_gain_scenario


def test_v3_active_task_link_uses_existing_regional_coordinator() -> None:
    scenario = build_v3_gain_scenario(
        split="pilot", condition="none_or_light", seed_variant=1701,
    )
    env = EventReliefEnv(scenario)
    observation = env.reset()
    coordinator_id = _coordinator_id(observation)
    assert coordinator_id.startswith("v3-uav-")
    assert all(row["id"] != "n02-uav-1" for row in observation["platforms"])

    target = next(row for row in observation["platforms"] if row["id"] != coordinator_id and row["type"] == "uav")
    task = next(row for row in observation["tasks"] if row["kind"] == "inspection")
    task_id = task["id"]
    target_id = target["id"]
    target_position = list(target["position"])
    # Build an active task in simulator truth, then obtain the public
    # observation from that live environment. This avoids testing a detached
    # dictionary that the safety hook could never act on.
    for mapping in (env._base._tasks, env._base._public_tasks, env._reported_tasks):
        mapping[task_id].update({
            "known": True, "status": "active", "phase": "observe", "assignee": target_id,
            "active_platform_id": target_id, "position": target_position,
            "service_minutes": 0.2, "service_progress": 0.0,
            "deadline": observation["horizon"], "started_at": 0,
            "cargo_onboard": 0.0,
        })
    env._task_specs[task_id].update({"position": target_position, "service_minutes": 0.2})
    env._base._task_spec[task_id].update({"position": target_position, "service_minutes": 0.2})
    for mapping in (env._base._platforms, env._base._public_platforms, env._reported_platforms):
        mapping[target_id].update({"active_task_id": task_id, "busy": True})
    env._queues = {row["id"]: [] for row in observation["platforms"]}
    env._queues[target_id] = [task_id]
    env._base._queues = copy.deepcopy(env._queues)
    edge = {
        "source": coordinator_id, "target": target_id, "quality": 0.95,
        "latency_minutes": 0.05, "loss_rate": 0.02, "connected": True,
    }
    env._communication = [edge]
    observation = env.public_observation()

    connected_report = assess_queue_safety(copy.deepcopy(observation))
    assert not any(row["code"] == "active_task_control_link_lost" for row in connected_report["violations"])

    env._communication = []
    disconnected_observation = env.public_observation()
    disconnected_report = assess_queue_safety(disconnected_observation)
    assert any(row["code"] == "active_task_control_link_lost" for row in disconnected_report["violations"])
