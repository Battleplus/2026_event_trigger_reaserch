from __future__ import annotations

import copy
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import event_models


@contextmanager
def _test_output_dir():
    parent = ROOT.parent / "evidence" / "local-test-temp"
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / uuid.uuid4().hex
    output.mkdir()
    try:
        yield output
    finally:
        for child in output.iterdir():
            if not child.is_file():
                raise AssertionError(f"unexpected nested test output: {child}")
            child.unlink()
        output.rmdir()


def _observation(time: int = 0) -> dict:
    return {
        "time": time,
        "horizon": 60,
        "map": {"rows": 100, "cols": 100},
        "platforms": [{
            "id": "u1", "type": "uav", "position": [10, 12], "energy": 80 - time / 10,
            "capacity_kg": 20, "speed_km_min": 1.2, "energy_per_km": 0.5,
            "service_energy_per_min": 0.1, "failed": False, "busy": False,
            "health": "healthy", "load": {"active_tasks": 1, "queued_tasks": 0},
            "capabilities": ["inspect", "deliver"],
        }],
        "tasks": [{
            "id": "t1", "kind": "delivery", "position": [20, 22], "local": True,
            "deadline": 50, "priority": 4, "quantity": 5, "status": "active", "progress": time / 100,
        }],
        "queues": {"u1": ["t1"]},
        "predecessors": {"t1": []},
        "cooperation": {"t1": []},
        "task_progress": {"t1": time / 100},
        "environment": [{"id": "env1", "kind": "threat", "grid_cell": [40, 41], "radius_cells": 2, "wind_vector": [1, -1], "risk_level": 0.2, "blocked": False}],
        "communication": [{"source": "u1", "target": "u1", "quality": 0.9, "latency_minutes": 0.1, "loss_rate": 0.01, "connected": True}],
        "provenance": {"parent_id": "p1", "event_tape_id": "excluded", "snapshot_hash": "excluded"},
    }


def _candidates() -> list[dict]:
    noop = {"action_type": "reconstruct", "scope": "NOOP", "noop": True, "queue_operations": [], "dispatches": []}
    local = {"action_type": "reconstruct", "scope": "LOCAL_REPAIR", "noop": False, "queue_operations": [], "dispatches": [{"platform_id": "u1", "task_id": "t1"}]}
    global_action = {"action_type": "reconstruct", "scope": "GLOBAL_REPLAN", "noop": False, "queue_operations": [{"op": "transfer", "task_id": "t1", "from_platform": "u1", "to_platform": "u1"}], "dispatches": [{"platform_id": "u1", "task_id": "t1"}]}
    return [
        {"candidate_key": "k-noop", "candidate_scope": "noop", "full_environment_action": noop, "legal": True},
        {"candidate_key": "k-local", "candidate_scope": "local", "full_environment_action": local, "legal": True},
        {"candidate_key": "k-global", "candidate_scope": "global", "full_environment_action": global_action, "legal": True},
    ]


def _sample(index: int = 0) -> dict:
    candidates = _candidates()
    targets = {}
    for head in event_models.WORLD_MODEL_HEADS:
        values = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]] if head == "next_state" else [0.1, 0.2, 0.3]
        targets[head] = {"values": values, "valid": [True, True, True]}
    return {
        "sample_id": f"sample-{index}", "split": "train", "parent_id": f"parent-{index % 4}",
        "history": [_observation(0), _observation(1), _observation(2), _observation(3)],
        "candidates": candidates, "targets": targets,
    }


def test_temporal_graph_world_model_uses_masks_and_has_finite_gradients() -> None:
    sample = _sample()
    dimensions = event_models._target_dimensions([sample])
    statistics = event_models._normalize_statistics([sample], dimensions)
    model = event_models.TemporalGraphWorldModel(dimensions, hidden_dim=16)
    model.set_normalization(statistics)
    batch, packed = event_models._collate_samples([sample], dimensions)
    batch["candidate_mask"][0, 2] = False
    before = event_models._state_digest(model.state_dict())
    predictions = model(batch)
    assert predictions["next_state"].shape == (1, 3, 2)
    assert torch.equal(predictions["utility"][0, 2], torch.zeros(1))
    losses = []
    for head, width in dimensions.items():
        target = packed[head][..., :width]
        valid = packed[head][..., width:].bool()
        if head not in event_models.PROBABILITY_HEADS:
            target = (target - getattr(model, f"target_mean_{head}")) / getattr(model, f"target_scale_{head}")
        loss = event_models._loss(predictions[head], target, valid, head)
        if loss is not None:
            losses.append(loss)
    torch.stack(losses).mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() for gradient in gradients)
    assert event_models._state_digest(model.state_dict()) == before


def test_ensemble_trains_restores_and_predicts_finitely() -> None:
    with _test_output_dir() as output:
        samples = [_sample(index) for index in range(8)]
        trained = event_models.train_ensemble(samples, output, {"hidden_dim": 16})
        assert trained["config"]["epochs"] == 40
        assert trained["config"]["batch_size"] == 8
        assert trained["training_forward_calls"] == 120
        assert all(row["optimizer_updates"] == 40 for row in trained["checkpoints"])
        assert all(row["parameter_delta_l2"] > 0 for row in trained["checkpoints"])
        ensemble = event_models.load_ensemble(trained["checkpoint_paths"])
        prediction = event_models.predict_ensemble(ensemble, samples[0]["history"], samples[0]["candidates"])
        assert prediction["candidate_keys"] == ["k-noop", "k-local", "k-global"]
        assert set(prediction["means"]) == set(event_models.WORLD_MODEL_HEADS)
        assert all(torch.isfinite(torch.tensor(prediction["means"][head])).all() for head in event_models.WORLD_MODEL_HEADS)
        assert all(torch.isfinite(torch.tensor(prediction["variances"][head])).all() for head in event_models.WORLD_MODEL_HEADS)
        assert ensemble.forward_calls == 3
        assert all(not parameter.requires_grad for model in ensemble.models for parameter in model.parameters())


def test_world_model_identity_ignores_provenance_only_metadata() -> None:
    first, second = _sample(0), _sample(0)
    second["history"][0]["provenance"]["event_tape_id"] = "different-private-tape-id"
    assert event_models._identity([first]) == event_models._identity([second])


def test_invalid_target_labels_do_not_affect_normalization_or_loss() -> None:
    sample = _sample()
    for head, entry in sample["targets"].items():
        entry["valid"][-1] = False
        if head == "next_state":
            entry["values"][-1] = [1.0e6, -1.0e6]
        else:
            entry["values"][-1] = 1.0e6
    changed_unknown = copy.deepcopy(sample)
    for head, entry in changed_unknown["targets"].items():
        if head == "next_state":
            entry["values"][-1] = [-9.0e8, 9.0e8]
        else:
            entry["values"][-1] = -9.0e8
    dims = event_models._target_dimensions([sample])
    assert event_models._normalize_statistics([sample], dims) == event_models._normalize_statistics([changed_unknown], dims)

    prediction = torch.tensor([0.0, 30.0], requires_grad=True)
    target = torch.tensor([1.0, -2.0e6])
    valid = torch.tensor([True, False])
    loss = event_models._loss(prediction, target, valid, "utility")
    assert loss is not None and torch.allclose(loss, torch.tensor(1.0))
    loss.backward()
    assert torch.equal(prediction.grad, torch.tensor([-2.0, 0.0]))


def test_candidate_encoding_distinguishes_split_and_cooperation_structure() -> None:
    observation = _observation()
    base = _candidates()[0]
    split = copy.deepcopy(base)
    split["candidate_key"] = "split"
    split["full_environment_action"].update({
        "noop": False,
        "scope": "LOCAL_REPAIR",
        "queue_operations": [{"op": "split", "task_id": "t1", "from_platform": "u1", "to_platform": "u1", "share": 0.4}],
        "dispatches": [{"platform_id": "u1", "task_id": "t1"}],
    })
    cooperate = copy.deepcopy(split)
    cooperate["candidate_key"] = "cooperate"
    cooperate["full_environment_action"]["queue_operations"] = [
        {"op": "cooperate", "task_id": "t1", "from_platform": "u1", "to_platform": "u2", "share": 0.6, "cooperation_group": "group-1"},
        {"op": "cooperate", "task_id": "t1", "from_platform": "u2", "to_platform": "u1", "share": 0.4, "cooperation_group": "group-1"},
    ]
    split_features = event_models._candidate_features(split, observation)
    cooperate_features = event_models._candidate_features(cooperate, observation)
    assert len(split_features) == event_models.CANDIDATE_FEATURE_DIM
    assert len(cooperate_features) == event_models.CANDIDATE_FEATURE_DIM
    assert split_features != cooperate_features
    assert split_features[11] > 0.0 and split_features[12] == 0.0
    assert cooperate_features[11] == 0.0 and cooperate_features[12] > 0.0
    assert cooperate_features[15] > 0.0


def test_world_model_rejects_non_100x100_public_history() -> None:
    sample = _sample()
    sample["history"][2]["map"]["cols"] = 99
    with pytest.raises(ValueError, match="EVENT_WM_MAP_CONTRACT"):
        event_models._encode_sample_graph(sample)
