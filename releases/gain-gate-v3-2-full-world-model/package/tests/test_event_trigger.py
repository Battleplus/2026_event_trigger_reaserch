from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import event_trigger


def _platform(platform_id: str = "n02-uav-1", *, local: bool = True, capabilities: list[str] | None = None,
              position: list[float] | None = None, energy: float = 100.0, **extra) -> dict:
    return {
        "id": platform_id, "type": "uav", "local": local,
        "position": list(position or [0.0, 0.0]), "energy": energy,
        "energy_initial": 100.0, "capacity_kg": 20.0,
        "speed_km_min": 1.0, "energy_per_km": 1.0,
        "service_energy_per_min": 0.1, "failed": False,
        "active_task_id": None, "capabilities": list(capabilities or ["inspect", "deliver"]),
        **extra,
    }


def _task(task_id: str = "t1", *, kind: str = "inspection", status: str = "pending", **extra) -> dict:
    return {
        "id": task_id, "kind": kind, "position": [2.0, 0.0], "local": True,
        "deadline": 50, "priority": 3, "status": status, "phase": "pending",
        "progress": 0.0, "service_minutes": 5,
        **extra,
    }


def _observation(time: int = 0) -> dict:
    return {
        "time": time, "horizon": 60,
        "platforms": [_platform()],
        "tasks": [_task()],
        "depots": [],
        "environment": [],
        "communication": [],
    }


def test_state_residuals_compare_public_position_energy_and_progress_to_expectation() -> None:
    previous = _observation()
    previous["platforms"][0].update({"active_task_id": "t1"})
    previous["tasks"][0].update({"status": "active", "phase": "to_inspection", "assignee": "n02-uav-1"})
    current = copy.deepcopy(previous)
    current["time"] = 1
    current["platforms"][0].update({"position": [1.0, 0.0], "energy": 99.0})

    expected = event_trigger._public_execution_residuals(previous, current)
    assert expected["d_state"] == 0.0
    assert expected["residuals"] == {"position": 0.0, "energy": 0.0, "progress": 0.0}

    current["platforms"][0].update({"position": [2.0, 0.0], "energy": 90.0})
    current["tasks"][0]["progress"] = 0.4
    residual = event_trigger._public_execution_residuals(previous, current)
    assert residual["residuals"]["position"] == 1.0
    assert residual["residuals"]["energy"] == 0.09
    assert residual["residuals"]["progress"] == 0.4
    assert residual["d_state"] > 0.49


def test_public_degradation_and_local_alternatives_ignore_local_flag_mismatch() -> None:
    observation = _observation()
    observation["platforms"][0].update({"local": True, "energy": 100.0, "speed_km_min": 10.0, "energy_per_km": 0.0})
    observation["tasks"][0].update({
        "kind": "delivery", "status": "pending", "local": False, "depot_id": "d1",
        "commodity": "water", "quantity": 2.0, "deadline": 20,
        "load_minutes": 1, "unload_minutes": 1,
    })
    observation["depots"] = [{"id": "d1", "position": [0.0, 0.0], "stock": {"water": 4.0}}]
    assert event_trigger._public_degradation(observation) == 0.0

    observation["platforms"].append(_platform("n02-uav-2", local=True))
    observation["communication"] = [{
        "source": "n02-uav-1", "target": "n02-uav-2", "quality": 0.9,
        "latency_minutes": 0.1, "loss_rate": 0.1, "connected": True,
    }]
    detection = {
        "affected_task_ids": ["t1"], "affected_platform_ids": ["n02-uav-1"],
        "event_details": {"category_count": 1, "failed_platform_ids": [], "energy_drop_ids": [], "environment_node_ids": []},
    }
    scope = event_trigger.identify_affected_scope(observation, detection)
    assert "n02-uav-2" in scope["affected_platform_ids"]
    assert scope["local_eligible"] is True


def test_task_and_communication_changes_include_cancellation_priority_deadline_latency_and_loss() -> None:
    previous = _observation()
    previous["platforms"].append(_platform("n02-uav-2"))
    previous["tasks"][0].update({"status": "active", "assignee": "n02-uav-1", "phase": "observe"})
    previous["platforms"][0]["active_task_id"] = "t1"
    previous["communication"] = [{
        "source": "n02-uav-1", "target": "n02-uav-2", "quality": 0.95,
        "latency_minutes": 0.05, "loss_rate": 0.02, "connected": True,
    }]
    current = copy.deepcopy(previous)
    current["time"] = 1
    current["tasks"][0].update({"status": "cancelled", "priority": 6, "deadline": 20})
    current["communication"][0].update({"quality": 0.8, "latency_minutes": 0.4, "loss_rate": 0.2})

    detection = event_trigger.TriggerEngine({}, {"threshold": 0.5}).detect(previous, current)
    details = detection["event_details"]
    assert details["cancelled_task_ids"] == ["t1"]
    assert details["priority_change_ids"] == ["t1"]
    assert details["deadline_change_ids"] == ["t1"]
    assert {"quality", "latency_minutes", "loss_rate"} <= set(details["communication_changes"][0]["changed_fields"])
    assert detection["d_communication"] > 0.0
    assert detection["strong_event"] is True


def test_four_constraint_margin_losses_are_reported_independently() -> None:
    cases = {
        "capability": (
            [_platform(capabilities=["deliver"])],
            [_task(kind="inspection")], [], [],
        ),
        "time": (
            [_platform(capabilities=["inspect"], speed_km_min=1.0)],
            [_task(kind="inspection", position=[20.0, 0.0], deadline=1)], [], [],
        ),
        "energy": (
            [_platform(capabilities=["deliver"], energy=1.0, energy_per_km=1.0)],
            [_task(kind="delivery", position=[10.0, 0.0], status="pending", depot_id="d1", commodity="water",
                   quantity=2.0, deadline=50, load_minutes=1, unload_minutes=1)],
            [{"id": "d1", "position": [0.0, 0.0], "stock": {"water": 5.0}}], [],
        ),
        "communication": (
            [_platform(), _platform("n02-uav-2")],
            [_task(kind="inspection", status="active", assignee="n02-uav-2")], [], [],
        ),
    }
    for expected_loss, (platforms, tasks, depots, communication) in cases.items():
        observation = _observation()
        observation.update({"platforms": platforms, "tasks": tasks, "depots": depots, "communication": communication})
        _, _, losses = event_trigger._margins(observation)
        assert losses[expected_loss] > 0.0, (expected_loss, losses)
        assert all(0.0 <= value <= 1.0 for value in losses.values())


def test_detection_exposes_individual_margin_losses_and_uses_public_state_residual() -> None:
    previous = _observation()
    current = copy.deepcopy(previous)
    current["time"] = 1
    current["platforms"][0]["position"] = [1.0, 0.0]
    current["platforms"][0]["energy"] = 80.0
    current["tasks"][0]["progress"] = 0.5
    result = event_trigger.TriggerEngine({}, {"threshold": 0.5}).detect(previous, current)
    assert result["d_state"] > 0.0
    assert set(result["constraint_margin_losses"]) == {"capability", "time", "energy", "communication"}
    assert result["d_margin"] == sum(result["constraint_margin_losses"].values()) / 4


def test_capability_and_capacity_disturbances_raise_platform_severity_and_scope() -> None:
    previous = _observation()
    current = copy.deepcopy(previous)
    current["time"] = 1
    current["platforms"][0]["capabilities"] = []
    current["platforms"][0]["capacity_kg"] = 0.0

    result = event_trigger.TriggerEngine({}, {"threshold": 0.5}).detect(previous, current)
    details = result["event_details"]

    assert details["capability_change_ids"] == ["n02-uav-1"]
    assert details["load_change_ids"] == ["n02-uav-1"]
    assert result["d_platform"] > 0.0
    assert result["affected_platform_ids"] == ["n02-uav-1"]

    scope = event_trigger.identify_affected_scope(current, result)
    assert "n02-uav-1" in scope["affected_platform_ids"]
    assert "capability_disturbance" in scope["reasons"]
    assert "load_disturbance" in scope["reasons"]


def test_light_events_cannot_enable_global_replan_when_local_fallback_is_unavailable() -> None:
    mask = event_trigger.high_level_action_mask(
        {"triggered": True, "strong_event": False, "event_severity": "light"},
        {"local_eligible": False, "global_eligible": True},
    )

    assert mask == {"NOOP": True, "LOCAL_REPAIR": False, "GLOBAL_REPLAN": False}


def test_reconstruction_cooldown_advances_only_after_an_effective_mutation() -> None:
    engine = event_trigger.TriggerEngine({}, {"threshold": 0.5, "delta_min": 4})
    detection = {"gamma_value": 0.6, "threshold": 0.5, "strong_event": False}

    first = engine.decide(detection, 1)
    assert first["triggered"] is True
    assert first["minutes_since_reconstruction"] is None
    engine.record_reconstruction(1, False)

    second = engine.decide(detection, 2)
    assert second["triggered"] is True
    assert second["cooldown_active"] is False
    engine.record_reconstruction(2, True)

    suppressed = engine.decide(detection, 3)
    assert suppressed["triggered"] is False
    assert suppressed["cooldown_suppressed"] is True
    assert suppressed["minutes_since_reconstruction"] == 1

    strong = {**detection, "strong_event": True}
    assert engine.decide(strong, 3)["triggered"] is True
    assert engine.decide(detection, 6)["triggered"] is True
