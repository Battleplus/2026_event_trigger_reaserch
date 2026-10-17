from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import event_oracle
import gain_oracle
import independent_consequence_oracle


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _state(*, paused: bool, status: str, reason: str | None = None, cargo: float = 2.5) -> dict:
    return {
        "status": status,
        "phase": "to_delivery_site",
        "progress": 0.4,
        "service_progress": 2.0,
        "cargo_onboard": cargo,
        "assignee": "uav-1",
        "active_platform_id": "uav-1",
        "queue_owner": "uav-1",
        "stock_commitment": None,
        "safety_paused": paused,
        **({"safety_pause_reason": reason} if reason is not None else {}),
    }


def _capability_pause_event() -> dict:
    event = {
        "schema": "gain-safety-task-transition/1",
        "type": "safety_task_capability_paused",
        "time": 7,
        "task_id": "task-1",
        "platform_id": "uav-1",
        "reason": "active_task_capability_lost",
        "before": _state(paused=False, status="active"),
        "after": _state(paused=True, status="paused"),
    }
    event["event_sha256"] = _digest(event)
    return event


def test_all_oracles_accept_capability_pause_and_preserve_physical_state() -> None:
    event = _capability_pause_event()
    assert gain_oracle._validate_safety_transition(event) == event["event_sha256"]
    trajectory = {
        "steps": [{"raw_events": [event]}],
        "continuation_policy_receipts": [],
    }
    report = event_oracle.validate_safety_transition_receipt(trajectory)
    assert report["transition_event_count"] == 1
    independent = independent_consequence_oracle.validate_safety_transition_events([event])
    assert independent["transition_event_count"] == 1


def test_capability_resume_requires_capability_transition_and_preserves_state() -> None:
    event = _capability_pause_event()
    resume = {
        **event,
        "type": "safety_task_capability_resumed",
        "reason": "active_task_capability_restored",
        "before": _state(paused=True, status="paused", reason="active_task_capability_lost"),
        "after": _state(paused=False, status="active"),
    }
    resume["event_sha256"] = _digest({key: value for key, value in resume.items() if key != "event_sha256"})
    assert gain_oracle._validate_safety_transition(resume) == resume["event_sha256"]
    assert event_oracle.validate_safety_transition_receipt({
        "steps": [{"raw_events": [resume]}], "continuation_policy_receipts": [],
    })["transition_event_count"] == 1
    assert independent_consequence_oracle.validate_safety_transition_events([resume])["transition_event_count"] == 1


@pytest.mark.parametrize("oracle_name", ["gain", "event", "independent"])
def test_capability_pause_rejects_progress_or_cargo_mutation(oracle_name: str) -> None:
    event = _capability_pause_event()
    event["after"]["cargo_onboard"] = 0.0
    event["event_sha256"] = _digest({key: value for key, value in event.items() if key != "event_sha256"})
    if oracle_name == "gain":
        with pytest.raises(ValueError, match="MUTATED_PHYSICAL_STATE|PAUSE_MUTATION"):
            gain_oracle._validate_safety_transition(event)
    elif oracle_name == "event":
        with pytest.raises(ValueError, match="PAUSE_MUTATION"):
            event_oracle.validate_safety_transition_receipt({
                "steps": [{"raw_events": [event]}], "continuation_policy_receipts": [],
            })
    else:
        with pytest.raises(ValueError, match="PAUSE_MUTATION"):
            independent_consequence_oracle.validate_safety_transition_events([event])
