from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import event_runner


def test_m1_falls_back_to_global_catalog_when_local_has_no_legal_action(monkeypatch) -> None:
    noop_action = {
        "action_type": "reconstruct", "scope": "NOOP", "noop": True,
        "queue_operations": [], "dispatches": [],
    }
    global_action = {
        "action_type": "reconstruct", "scope": "GLOBAL_REPLAN", "noop": False,
        "queue_operations": [],
        "dispatches": [{"platform_id": "u2", "task_id": "t1"}],
    }

    def candidate(key: str, action: dict) -> dict:
        return {
            "candidate_key": key, "full_environment_action": copy.deepcopy(action),
            "legal": True, "snapshot_hash": "snapshot", "event_tape_id": "tape",
        }

    class FakeEnv:
        time = 0

        def __init__(self, _scenario):
            pass

        def reset(self):
            return {"time": 0, "provenance": {"snapshot_hash": "snapshot"}}

        def step(self, action):
            assert action == global_action
            self.time = 1
            info = {
                "requested_reconstruction_count": 1,
                "effective_reconstruction_count": 1,
                "task_owner_change_count": 1, "task_switch_count": 1,
                "queue_reorder_count": 0, "raw_events": [],
            }
            return {"time": 1}, 0.0, True, False, info

        def episode_metrics(self):
            return {"effective_reconstruction_count": 1}

    class FakeTriggerEngine:
        instances = []

        def __init__(self, _config, _threshold):
            self.records = []
            self.instances.append(self)

        def detect(self, *_args, **_kwargs):
            return {"triggered": True, "event_severity": {"damage": 1.0}}

        def decide(self, _detection, _time, mode="event"):
            return {"triggered": True, "scope": "LOCAL_REPAIR", "mode": mode}

        def record_reconstruction(self, time, effective):
            self.records.append((time, effective))

    def generate(_env, *, scope, max_candidates):
        assert max_candidates == 4
        if isinstance(scope, dict):
            return [candidate("local-noop", noop_action)]
        assert scope == "GLOBAL_REPLAN"
        return [candidate("global-noop", noop_action), candidate("global-plan", global_action)]

    def repair(_env, action):
        return action, {"postrepair_legal": True}

    def identify(_observation, _detection):
        return {
            "scope": "LOCAL_REPAIR", "local_eligible": True, "global_eligible": False,
            "affected_task_ids": ["t1"], "affected_platform_ids": ["u1"], "reasons": [],
        }

    monkeypatch.setattr(
        event_runner, "_modules",
        lambda: (FakeEnv, generate, repair, identify, None, None, FakeTriggerEngine, None),
    )
    class FakeLedger:
        counts = []

        def count(self, name, amount):
            self.counts.append((name, amount))

    logs, scopes, summary, used = event_runner._run_m1_mode(
        {"parent_id": "parent-1", "split": "train", "condition": "test"},
        "event", {"threshold": 0.5},
        {"trigger": {}, "collection": {"max_candidates": 4}}, FakeLedger(),
    )

    assert logs[0]["selected_full_environment_action"] == global_action
    assert logs[0]["decision"]["high_level_action"] == "GLOBAL_REPLAN"
    assert logs[0]["scope_catalog_receipt"] == {
        "status": "fallback", "code": "local_repair_no_feasible_solution",
        "local_non_noop_candidate_count": 0, "global_non_noop_candidate_count": 1,
    }
    assert scopes[0]["scope"]["scope"] == "GLOBAL_REPLAN"
    assert summary["local_catalog_fallback_count"] == 1
    assert used[0]["scope_catalog_receipt"]["status"] == "fallback"
    assert FakeTriggerEngine.instances[-1].records == [(0, True)]


def test_m1_event_fields_preserve_severity_and_trigger_scope() -> None:
    row = event_runner._event_fields(
        {"event_type": "platform_disturbance", "event_severity": "strong", "gamma_value": 0.9},
        {"triggered": True, "affected_scope": {"scope": "GLOBAL_REPLAN"}},
    )
    assert row["event_severity"] == "strong"
    assert row["event_severity_value"] == 0.0
    assert row["affected_scope"] == "GLOBAL_REPLAN"
