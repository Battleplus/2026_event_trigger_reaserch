from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import event_oracle


def _window(*, recorded_reward: float = -0.03, branch_tape_hash: str | None = None) -> dict:
    action = {"action_type": "reconstruct", "scope": "NOOP", "noop": True, "queue_operations": [], "dispatches": []}
    action_key = f"event-action-{event_oracle._digest(action)[:24]}"
    private_binding = {"simulator_snapshot_sha256": "a" * 64, "event_tape_sha256": "b" * 64}
    branch_binding = dict(private_binding)
    if branch_tape_hash is not None:
        branch_binding["event_tape_sha256"] = branch_tape_hash
    candidate = {
        "candidate_key": action_key, "candidate_scope": "noop",
        "snapshot_hash": "snapshot-1", "public_prefix_hash": "prefix-1", "event_tape_id": "tape-1",
        "legal": True, "full_environment_action": action,
    }
    return {
        "window_id": "p1-t00", "parent_id": "p1", "split": "train",
        "window_start": 0, "window_end": 1, "event_tape_id": "tape-1",
        "private_binding": private_binding,
        "public_observation": {
            "provenance": {"snapshot_hash": "snapshot-1", "public_prefix_hash": "prefix-1"},
            "platforms": [{"id": "p1", "energy_used": 0.0}],
        },
        "reward_weights": dict(event_oracle.REWARD_WEIGHTS),
        "candidates": [candidate],
        "branches": [{
            "candidate_key": action_key, "candidate": candidate,
            "private_binding": branch_binding,
            "trajectory": {
                "parent_id": "p1", "event_tape_id": "tape-1", "start_time": 0, "end_time": 1,
                "steps": [{
                    "time": 0, "full_environment_action": action,
                    "raw_events": [{"time": 0, "type": "service", "energy": 3.0}],
                    "reward": recorded_reward,
                    "reward_primitives": {
                        "completion_count": 0, "delay_minutes": 0.0, "energy_units": 3.0,
                        "risk_units": 0.0, "communication_messages": 0, "switch_count": 0,
                        "reconstruction_count": 0, "reconstruction_size": 0,
                    },
                    "reward_terms": {
                        "completion": 0.0, "delay": 0.0, "energy": -0.03, "risk": 0.0,
                        "communication": 0.0, "switch": 0.0, "reconstruction": 0.0,
                        "reconstruction_scale": 0.0,
                    },
                    "operation_receipts": [{"postrepair_legal": True}],
                }],
                "raw_truth": {
                    "time": 1,
                    "tasks": [{"id": "t1", "status": "pending", "progress": 0.0, "deadline": 60}],
                    "platforms": [{"id": "p1", "position": [1.0, 0.0], "energy": 90.0,
                                   "energy_used": 3.0, "failed": False}],
                },
            },
        }],
    }


def test_reward_reconstruction_counts_owner_changes_and_queue_mutations() -> None:
    components, valid = event_oracle._reward_primitives(
        {"noop": False, "queue_operations": [], "dispatches": []},
        [
            {"type": "task_owner_change", "task_id": "t1"},
            {"type": "task_owner_change", "task_id": "t2"},
            {"type": "task_owner_change", "task_id": "t2"},
            {"type": "task_switch", "task_id": "t2"},
            {"type": "queue_reorder", "task_id": "t3"},
            {"type": "queue_reorder", "task_id": "t4"},
        ],
    )

    assert valid is True
    assert components["switch"] == 1.0
    assert components["reconstruction"] == 1.0
    assert components["reconstruction_size"] == 4.0


def test_service_energy_is_recomputed_without_inventing_risk() -> None:
    labels = event_oracle.recompute_consequence_labels(_window())
    label = labels["candidate_labels"][0]

    assert label["reward_recomputation"]["recomputed_utility"] == pytest.approx(-0.03)
    assert label["global_outcomes"]["energy_used"] == {"value": 3.0, "valid": True}
    assert label["global_outcomes"]["utility"]["value"] == pytest.approx(-0.03)
    assert label["global_outcomes"]["utility"]["valid"] is True


def test_oracle_rejects_reward_or_private_branch_binding_mismatch() -> None:
    with pytest.raises(ValueError, match="EVENT_ORACLE_RECORDED_REWARD_MISMATCH"):
        event_oracle.recompute_consequence_labels(_window(recorded_reward=0.0))

    with pytest.raises(ValueError, match="EVENT_ORACLE_PRIVATE_BRANCH_BINDING"):
        event_oracle.recompute_consequence_labels(_window(branch_tape_hash="c" * 64))


@pytest.mark.parametrize("field", ["candidate_key", "snapshot_hash", "public_prefix_hash", "event_tape_id", "candidate_scope"])
def test_oracle_rejects_candidate_metadata_mutations(field: str) -> None:
    window = _window()
    window["branches"][0]["candidate"][field] = "CORRUPTED"
    with pytest.raises(ValueError, match="EVENT_ORACLE_"):
        event_oracle.recompute_consequence_labels(window)
