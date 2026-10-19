from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import gain_safety
import event_scenarios
from event_environment import EventReliefEnv


def _noop_action() -> dict:
    return {
        "action_type": "reconstruct", "scope": "NOOP", "noop": True,
        "queue_operations": [], "dispatches": [],
    }


def _link_loss_observation(*, paused: bool = False) -> dict:
    return {
        "time": 4.0,
        "horizon": 60.0,
        "platforms": [{
            "id": "n02-uav-2", "position": [0.0, 0.0], "energy": 100.0,
            "speed_km_min": 1.0, "energy_per_km": 1.0,
            "service_energy_per_min": 0.0, "capacity_kg": 10.0,
            "capabilities": ["deliver"], "failed": False,
            "busy": True, "active_task_id": "task-1",
        }],
        "tasks": [{
            "id": "task-1", "known": True, "kind": "delivery", "status": "paused" if paused else "active",
            "safety_paused": paused, "phase": "to_delivery_site", "service_progress": 1.0,
            "progress": 0.25, "position": [1.0, 0.0], "depot_id": "depot-1",
            "commodity": "water", "quantity": 3.0, "load_minutes": 1,
            "unload_minutes": 1, "deadline": 60.0, "assignee": "n02-uav-2",
        }],
        "depots": [{"id": "depot-1", "position": [0.0, 0.0], "stock": {"water": 0.0}}],
        "queues": {"n02-uav-2": ["task-1"]},
        "communication": [],
    }


class _LiveHook:
    def __init__(self, post: dict):
        self.post = copy.deepcopy(post)
        self.projection_called = False

    def _planned_queue_state(self, *_args):
        self.projection_called = True
        raise AssertionError("V2 live verification must not use queue projection")

    def apply_current_safety_disposition(self, public: dict, report: dict) -> dict:
        assert report["forced"] is True
        return {
            "schema": "gain-live-safety-disposition/1",
            "status": "resolved", "resolved": True, "actions": [],
            "pre_public_sha256": gain_safety._observation_digest(public),
            "post_public_sha256": gain_safety._observation_digest(self.post),
            "remaining_report": gain_safety.assess_queue_safety(self.post),
            "unknown_reason": None,
        }


def test_paused_active_task_is_safe_only_when_explicitly_marked() -> None:
    active = gain_safety.assess_queue_safety(_link_loss_observation(paused=False))
    paused = gain_safety.assess_queue_safety(_link_loss_observation(paused=True))

    assert any(row["code"] == "active_task_control_link_lost" for row in active["violations"])
    assert not any(row["code"] == "active_task_control_link_lost" for row in paused["violations"])


def test_unmarked_paused_task_does_not_suppress_capability_violation() -> None:
    observation = _link_loss_observation(paused=True)
    observation["tasks"][0]["safety_paused"] = False
    observation["platforms"][0]["capabilities"] = []

    report = gain_safety.assess_queue_safety(observation)

    assert any(
        row["code"] == "queued_task_capability_infeasible"
        and row["task_id"] == "task-1"
        for row in report["violations"]
    )


def test_live_verification_uses_actual_post_observation_and_never_projection() -> None:
    before = _link_loss_observation(paused=False)
    after = _link_loss_observation(paused=True)
    safety = gain_safety.assess_queue_safety(before)
    env = _LiveHook(after)

    receipt = gain_safety.apply_safety_disposition(env, before, safety)
    result = gain_safety.verify_live_safety_disposition(env, before, safety, receipt, after)

    assert result["status"] == "resolved"
    assert result["resolved"] is True
    assert env.projection_called is False


def test_live_verification_rejects_hash_mismatch_as_unknown() -> None:
    before = _link_loss_observation(paused=False)
    after = _link_loss_observation(paused=True)
    safety = gain_safety.assess_queue_safety(before)
    env = _LiveHook(after)
    receipt = gain_safety.apply_safety_disposition(env, before, safety)
    receipt["post_public_sha256"] = "0" * 64

    result = gain_safety.verify_live_safety_disposition(env, before, safety, receipt, after)

    assert result["status"] == "unknown"
    assert result["resolved"] is False
    assert result["unknown_reason"] == "safety_receipt_post_public_hash_mismatch"


def test_active_capability_loss_pauses_without_releasing_queue_work() -> None:
    env = EventReliefEnv(copy.deepcopy(event_scenarios.build_event_scenarios()[0]))
    env.reset()
    observation, *_ = env.step(_noop_action())
    task_id = "n02-task-01"
    platform_id = "n02-uav-1"
    for mapping_name in ("_platforms", "_public_platforms"):
        mapping = getattr(env._base, mapping_name, None)
        if isinstance(mapping, dict) and platform_id in mapping:
            mapping[platform_id]["capabilities"] = ["inspect"]
    env._reported_platforms[platform_id]["capabilities"] = ["inspect"]

    report = gain_safety.assess_queue_safety(env.public_observation())
    assert any(
        row["code"] == "queued_task_capability_infeasible" and row["task_id"] == task_id
        for row in report["violations"]
    )
    observation = env.public_observation()
    before = observation["tasks"]
    before_task = next(row for row in before if row["id"] == task_id)
    receipt = gain_safety.apply_safety_disposition(env, observation, report)
    assert receipt["resolved"] is True
    assert receipt["actions"][0]["type"] == "safety_task_capability_paused"
    transition = receipt["actions"][0]
    for field in ("phase", "progress", "service_progress", "cargo_onboard", "assignee", "active_platform_id", "queue_owner", "stock_commitment"):
        assert transition["before"][field] == transition["after"][field]
    post = env.public_observation()
    post_task = next(row for row in post["tasks"] if row["id"] == task_id)
    assert post_task["status"] == "paused"
    assert post_task["safety_paused"] is True
    assert task_id in post["queues"][platform_id]
    assert not gain_safety.assess_queue_safety(post)["forced"]
    assert post_task["progress"] == before_task["progress"]

    for mapping_name in ("_platforms", "_public_platforms"):
        mapping = getattr(env._base, mapping_name, None)
        if isinstance(mapping, dict) and platform_id in mapping:
            mapping[platform_id]["capabilities"] = ["deliver", "inspect"]
    env._reported_platforms[platform_id]["capabilities"] = ["deliver", "inspect"]
    env._refresh_reported_state([])
    assert any(
        row["type"] == "safety_task_capability_resumed"
        for row in env._extension_events
    )
    resumed = env.public_observation()
    resumed_task = next(row for row in resumed["tasks"] if row["id"] == task_id)
    assert resumed_task["safety_paused"] is False
