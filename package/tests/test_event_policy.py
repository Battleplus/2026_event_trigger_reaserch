from __future__ import annotations

import copy
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import event_models
import event_policy


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
        "environment": [],
        "communication": [{"source": "u1", "target": "u1", "quality": 0.9, "latency_minutes": 0.1, "loss_rate": 0.01, "connected": True}],
        "provenance": {"parent_id": "p1", "event_tape_id": "excluded", "snapshot_hash": "excluded"},
    }


def _candidates() -> list[dict]:
    actions = [
        ("noop", {"action_type": "reconstruct", "scope": "NOOP", "noop": True, "queue_operations": [], "dispatches": []}),
        ("local", {"action_type": "reconstruct", "scope": "LOCAL_REPAIR", "noop": False, "queue_operations": [], "dispatches": [{"platform_id": "u1", "task_id": "t1"}]}),
        ("global", {"action_type": "reconstruct", "scope": "GLOBAL_REPLAN", "noop": False, "queue_operations": [{"op": "transfer", "task_id": "t1", "from_platform": "u1", "to_platform": "u1"}], "dispatches": [{"platform_id": "u1", "task_id": "t1"}]}),
    ]
    return [
        {"candidate_key": f"candidate-{scope}", "candidate_scope": scope, "full_environment_action": action, "legal": True}
        for scope, action in actions
    ]


def _wm_predictions(value: float) -> dict:
    means = {}
    variances = {}
    for head in event_models.WORLD_MODEL_HEADS:
        width = 2 if head == "next_state" else 1
        means[head] = [[value] * width for _ in range(3)]
        variances[head] = [[0.25] * width for _ in range(3)]
    return {"means": means, "variances": variances, "candidate_keys": [row["candidate_key"] for row in _candidates()]}


def test_scope_masks_exclude_invalid_candidates_and_unavailable_scopes() -> None:
    candidates = _candidates()
    candidates[1]["legal"] = False
    high_mask, candidate_masks = event_policy._scope_masks(
        candidates, {"NOOP": True, "LOCAL_REPAIR": True, "GLOBAL_REPLAN": True}, None,
    )
    assert high_mask == [True, False, True]
    assert candidate_masks["LOCAL_REPAIR"] == [False, False, False]
    policy = event_policy.build_hierarchical_policy(8201)
    for _ in range(10):
        action = policy.act(_observation(), candidates, high_mask, deterministic=False)
        assert action["scope"] in {"NOOP", "GLOBAL_REPLAN"}
        assert candidates[action["candidate_index"]]["legal"]
    with pytest.raises(ValueError, match="EVENT_POLICY_CANDIDATE_MASK_LENGTH"):
        event_policy._scope_masks(candidates, None, [True])


def test_policy_rejects_non_100x100_public_map() -> None:
    observation = _observation()
    observation["map"]["rows"] = 101
    policy = event_policy.build_hierarchical_policy(8201)
    with pytest.raises(ValueError, match="EVENT_WM_MAP_CONTRACT"):
        policy.act(observation, _candidates(), [True, True, True], deterministic=True)


def test_world_model_features_feed_only_auxiliary_critic() -> None:
    policy = event_policy.build_hierarchical_policy(8201, use_wm_aux=True)
    observation, candidates = _observation(), _candidates()
    first = policy.act(observation, candidates, [True, True, True], wm_predictions=_wm_predictions(0.0), deterministic=True)
    second = policy.act(observation, candidates, [True, True, True], wm_predictions=_wm_predictions(4.0), deterministic=True)
    assert first["high_logits"] == second["high_logits"]
    assert first["low_logits"] == second["low_logits"]

    policy.zero_grad(set_to_none=True)
    total_value, _, auxiliary_value = policy.critic_values(observation, _wm_predictions(2.0))
    total_value.backward()
    auxiliary_gradients = [parameter.grad for parameter in policy.auxiliary_critic.parameters() if parameter.grad is not None]
    actor_parameters = list(policy.actor_encoder.parameters()) + list(policy.candidate_encoder.parameters()) + list(policy.high_actor.parameters()) + list(policy.low_actor.parameters())
    assert auxiliary_value.requires_grad
    assert auxiliary_gradients and all(torch.isfinite(gradient).all() for gradient in auxiliary_gradients)
    assert any(torch.count_nonzero(gradient).item() for gradient in auxiliary_gradients)
    assert all(parameter.grad is None for parameter in actor_parameters)


def test_relational_action_slots_reach_actor_and_world_model_inputs() -> None:
    observation = _observation()
    observation["platforms"].extend([
        {**copy.deepcopy(observation["platforms"][0]), "id": "u2", "position": [40, 45], "energy": 65},
        {**copy.deepcopy(observation["platforms"][0]), "id": "u3", "position": [75, 70], "energy": 45},
    ])
    observation["tasks"].append({
        **copy.deepcopy(observation["tasks"][0]), "id": "t2", "position": [60, 55], "priority": 2,
    })

    def candidate(key: str, operations: list[dict]) -> dict:
        return {
            "candidate_key": key,
            "candidate_scope": "global",
            "legal": True,
            "full_environment_action": {
                "action_type": "reconstruct", "scope": "GLOBAL_REPLAN", "noop": False,
                "queue_operations": copy.deepcopy(operations), "dispatches": [],
            },
        }

    transfer_u2 = {"op": "transfer", "task_id": "t1", "from_platform": "u1", "to_platform": "u2"}
    transfer_u3 = {"op": "transfer", "task_id": "t1", "from_platform": "u1", "to_platform": "u3"}
    reverse_transfer = {"op": "transfer", "task_id": "t1", "from_platform": "u2", "to_platform": "u1"}
    transfer_t2 = {"op": "insert", "task_id": "t2", "from_platform": "u2", "to_platform": "u3", "before_task_id": "t1"}
    candidates = [
        candidate("u2", [transfer_u2]),
        candidate("u3", [transfer_u3]),
        candidate("reverse", [reverse_transfer]),
        candidate("order-ab", [transfer_u2, transfer_t2]),
        candidate("order-ba", [transfer_t2, transfer_u2]),
    ]
    features = [event_models._candidate_features(row, observation) for row in candidates]
    assert all(len(row) == event_models.CANDIDATE_FEATURE_DIM == 631 for row in features)
    assert features[0] != features[1]
    assert features[0] != features[2]
    assert features[3] != features[4]

    policy = event_policy.build_hierarchical_policy(8201)
    actor_inputs: list[torch.Tensor] = []
    handle = policy.candidate_encoder.register_forward_pre_hook(
        lambda _module, inputs: actor_inputs.append(inputs[0].detach().clone())
    )
    try:
        for row in candidates:
            policy.actor_logits(observation, [row])
    finally:
        handle.remove()
    assert len(actor_inputs) == len(candidates)
    assert all(not torch.equal(actor_inputs[0], actor_inputs[index]) for index in range(1, len(actor_inputs)))

    sample = {"history": [observation], "candidates": candidates}
    world_model_batch, _ = event_models._collate_samples_with_dummy_targets(sample)
    wm_inputs = world_model_batch["candidate_features"][0]
    assert all(not torch.equal(wm_inputs[0], wm_inputs[index]) for index in range(1, len(candidates)))


def test_catalog_deduplicates_physical_actions_and_masks_use_actual_scope() -> None:
    noop = {
        "candidate_key": "noop", "candidate_scope": "noop", "legal": True,
        "full_environment_action": {"action_type": "reconstruct", "scope": "NOOP", "noop": True, "queue_operations": [], "dispatches": []},
    }
    local_action = {
        "action_type": "reconstruct", "scope": "LOCAL_REPAIR", "noop": False,
        "queue_operations": [{"op": "transfer", "task_id": "t1", "from_platform": "u1", "to_platform": "u2"}],
        "dispatches": [{"platform_id": "u2", "task_id": "t1"}],
    }
    global_same_action = {**copy.deepcopy(local_action), "scope": "GLOBAL_REPLAN"}
    global_wide_action = {
        "action_type": "reconstruct", "scope": "GLOBAL_REPLAN", "noop": False,
        "queue_operations": [{"op": "transfer", "task_id": "t2", "from_platform": "u2", "to_platform": "u3"}],
        "dispatches": [{"platform_id": "u3", "task_id": "t2"}],
    }
    affected = {"scope": "LOCAL_REPAIR", "local_eligible": True, "affected_task_ids": ["t1"], "affected_platform_ids": ["u1", "u2"]}
    calls: list[object] = []

    def generate(_env, *, scope, max_candidates):
        calls.append(scope)
        if isinstance(scope, dict):
            return [noop, {"candidate_key": "local", "candidate_scope": "local", "legal": True, "full_environment_action": copy.deepcopy(local_action)}]
        return [
            {**noop, "candidate_key": "global-noop"},
            {"candidate_key": "global-duplicate", "candidate_scope": "global", "legal": True, "full_environment_action": global_same_action},
            {"candidate_key": "global-wide", "candidate_scope": "global", "legal": True, "full_environment_action": global_wide_action},
        ]

    catalog = event_policy._generate_policy_catalog(None, generate, 16, affected)
    physical_keys = [event_policy._physical_action_key(row) for row in catalog]
    assert len(physical_keys) == len(set(physical_keys))
    assert [event_policy._candidate_scope(row) for row in catalog] == ["NOOP", "LOCAL_REPAIR", "GLOBAL_REPLAN"]
    assert calls[0] == affected and calls[1] == "GLOBAL_REPLAN"

    fallback_scope, fallback_receipt = event_policy._catalog_scope_fallback(
        affected, [catalog[0], catalog[2]],
    )
    assert fallback_receipt["code"] == "local_repair_no_feasible_solution"
    assert fallback_scope["local_eligible"] is False and fallback_scope["global_eligible"] is True
    actual_mask = event_policy._mask_from_trigger(
        {"triggered": True, "high_level_mask": [True, True, False]},
        {"triggered": True},
        fallback_scope,
        lambda _detection, scope: {"NOOP": True, "LOCAL_REPAIR": scope["local_eligible"], "GLOBAL_REPLAN": scope["global_eligible"]},
    )
    assert actual_mask == [True, False, True]


def test_baseline_episode_skips_world_model_and_records_local_catalog_fallback(monkeypatch) -> None:
    observation = _observation()
    noop = {
        "candidate_key": "noop", "candidate_scope": "noop", "legal": True,
        "full_environment_action": {"action_type": "reconstruct", "scope": "NOOP", "noop": True, "queue_operations": [], "dispatches": []},
    }
    global_action = {
        "candidate_key": "global", "candidate_scope": "global", "legal": True,
        "full_environment_action": {"action_type": "reconstruct", "scope": "GLOBAL_REPLAN", "noop": False,
                                  "queue_operations": [], "dispatches": [{"platform_id": "u1", "task_id": "t1"}]},
    }

    class FakeEnv:
        def __init__(self, _scenario):
            self.metrics = {
                "requested_reconstruction_count": 0, "effective_reconstruction_count": 0,
                "reconstruction_size": 0, "operational_dispatch_count": 0,
            }

        def reset(self):
            return copy.deepcopy(observation)

        def step(self, _action):
            next_state = copy.deepcopy(observation)
            next_state["time"] = 1
            return next_state, 0.0, True, False, {
                "effective_reconstruction_count": 0, "operational_dispatch_count": 0,
            }

        def episode_metrics(self):
            return copy.deepcopy(self.metrics)

        def export_trajectory(self):
            return {"summary": copy.deepcopy(self.metrics)}

    class FakeTriggerEngine:
        instances = []

        def __init__(self, _config, _manifest):
            self.detect_summaries = []
            self.reconstruction_records = []
            self.manifest = copy.deepcopy(_manifest)
            self.instances.append(self)

        def detect(self, _previous, _current, history=None, wm_summary=None):
            self.detect_summaries.append(wm_summary)
            return {"triggered": False, "strong_event": False, "gamma_value": 0.0, "threshold": 0.5}

        def decide(self, _detection, _time, mode="event"):
            return {
                "triggered": mode == "periodic", "high_level_mask": [True, True, False],
                "affected_scope": {"scope": "LOCAL_REPAIR"},
            }

        def record_reconstruction(self, time, effective):
            self.reconstruction_records.append((time, effective))

    def generate(_env, *, scope, max_candidates):
        if isinstance(scope, dict):
            assert scope["affected_task_ids"] == ["t1"]
            return [noop]
        return [noop, global_action]

    def repair(_env, action):
        return action, {"postrepair_legal": True}

    def identify(_observation, _detection):
        return {
            "scope": "LOCAL_REPAIR", "local_eligible": True, "global_eligible": False,
            "affected_task_ids": ["t1"], "affected_platform_ids": ["u1"], "reasons": [],
        }

    def mask(_detection, affected):
        return {
            "NOOP": True, "LOCAL_REPAIR": affected["local_eligible"],
            "GLOBAL_REPLAN": affected["global_eligible"],
        }

    monkeypatch.setattr(event_policy, "_import_event_apis", lambda: (FakeEnv, generate, repair, FakeTriggerEngine, identify, mask))
    monkeypatch.setattr(event_policy, "predict_ensemble", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("baseline queried ensemble")))
    monkeypatch.setattr(event_policy, "world_model_trigger_summary", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("baseline built WM trigger summary")))

    policy = event_policy.build_hierarchical_policy(8201, use_wm_aux=False)
    records, summary = event_policy._event_episode(
        {"parent_id": "fake", "split": "train", "horizon": 1}, policy,
        variant="Periodic-G", seed=8201, threshold_manifest={"threshold": 0.5},
        ensemble=object(), config={"collection": {"max_candidates": 16}},
        training=True, deterministic=True,
    )
    record = records[0]
    assert record["wm_predictions"] is None and record["wm_trigger_summary"] is None
    assert record["world_model_use"] == {"trigger": False, "auxiliary_critic": False}
    assert record["scope_catalog_receipt"]["code"] == "local_repair_no_feasible_solution"
    assert record["affected_scope"]["local_eligible"] is False
    assert record["affected_scope"]["global_eligible"] is True
    assert record["decision"]["high_level_mask"] == [True, False, True]
    assert record["high_level_mask"] == [True, False, True]
    assert summary["requested_non_noop_decision_count"] in (0, 1)
    assert summary["effective_reconstruction_count"] == 0
    assert FakeTriggerEngine.instances[-1].reconstruction_records == [(0, False)]

    wm_predictions = _wm_predictions(0.25)
    wm_summary = {"d_degradation": 0.1, "d_uncertainty": 0.2}
    calls = []
    monkeypatch.setattr(event_policy, "predict_ensemble", lambda *_args, **_kwargs: (calls.append(True) or wm_predictions))
    monkeypatch.setattr(event_policy, "world_model_trigger_summary", lambda *_args, **_kwargs: wm_summary)
    wm_policy = event_policy.build_hierarchical_policy(8201, use_wm_aux=True)
    wm_records, wm_episode_summary = event_policy._event_episode(
        {"parent_id": "fake-wm", "split": "train", "horizon": 1}, wm_policy,
        variant="Periodic-G-WM", seed=8201,
        threshold_manifest={"threshold": 0.5, "pre_model_threshold": 0.8, "source_split": "train"},
        ensemble=object(), config={"collection": {"max_candidates": 16}},
        training=True, deterministic=True,
    )
    assert calls == [True]
    assert FakeTriggerEngine.instances[-1].detect_summaries == [None, wm_summary]
    assert FakeTriggerEngine.instances[-1].manifest["threshold"] == 0.5
    assert wm_records[0]["wm_predictions"] == wm_predictions
    assert wm_records[0]["world_model_use"] == {"trigger": True, "auxiliary_critic": True}
    assert wm_episode_summary["trigger_threshold_semantics"]["source"] == "final_world_model_training_calibration"


def _rollout(policy: event_policy.HierarchicalGraphPolicy) -> list[dict]:
    observation, candidates = _observation(), _candidates()
    records = []
    for index in range(32):
        observation_i = copy.deepcopy(observation)
        observation_i["time"] = index
        observation_i["platforms"][0]["energy"] -= index * 0.15
        output = policy.act(observation_i, candidates, [True, True, True], deterministic=True)
        records.append({
            "observation": observation_i,
            "candidates": copy.deepcopy(candidates),
            "scope_index": output["scope_index"],
            "candidate_index": output["candidate_index"],
            "high_level_mask": output["high_level_mask"],
            "candidate_mask": output["candidate_mask"],
            "wm_predictions": None,
            "old_log_probability": output["log_probability"],
            "old_value": output["value"],
            "reward": float((index % 5) - 2),
            "terminated": False,
            "truncated": index == 31,
        })
    event_policy._attach_episode_gae(records)
    return records


def test_ppo_update_changes_parameters_and_checkpoint_restores_strictly() -> None:
    with _test_output_dir() as output:
        policy = event_policy.build_hierarchical_policy(8201)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        records = _rollout(policy)
        before = event_policy._tensor_digest(policy.state_dict())
        update_log = event_policy._update_policy(
            policy, optimizer, records, seed=8201, epochs=1, minibatch_size=32,
            updates_per_epoch=1, learning_rate=3e-4, clip_ratio=0.2,
            entropy_coefficient=0.01,
        )
        assert len(update_log) == 1
        assert torch.isfinite(torch.tensor(update_log[0]["total_loss"]))
        assert update_log[0]["optimizer_step_applied"] is True
        assert update_log[0]["parameter_digest_before"] != update_log[0]["parameter_digest_after"]
        assert event_policy._tensor_digest(policy.state_dict()) != before
        assert update_log[0]["actor_gradient_norm_preclip"] > 0.0

        probe = records[0]
        checkpoint = output / "policy.pt"
        saved = event_policy._save_policy_checkpoint(
            checkpoint, policy, optimizer, variant="Periodic-G", seed=8201,
            config={"test": True}, data_identity="test-data", wm_hash="test-wm",
            update_log=update_log,
            restore_probe={
                "observation": probe["observation"], "candidates": probe["candidates"],
                "high_level_mask": probe["high_level_mask"], "wm_predictions": None,
            },
        )
        assert saved["optimizer_updates"] == 1
        restored = event_policy.verify_policy_checkpoint(checkpoint)
        assert restored["status"] == "complete"
        assert restored["model_state_sha256"] == saved["state_sha256"]
        assert restored["actor_forward_calls"] == restored["critic_forward_calls"] == 1

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        malformed_map_payload = copy.deepcopy(payload)
        malformed_map_payload["map_cols"] = 99
        malformed_map = output / "policy-wrong-map.pt"
        torch.save(malformed_map_payload, malformed_map)
        with pytest.raises(ValueError, match="EVENT_POLICY_CHECKPOINT_MAP_CONTRACT"):
            event_policy.load_policy_checkpoint(malformed_map)

        payload["model_state"]["unexpected.weight"] = torch.zeros(1)
        payload["model_state_sha256"] = event_policy._tensor_digest(payload["model_state"])
        malformed = output / "policy-unexpected-key.pt"
        torch.save(payload, malformed)
        with pytest.raises(RuntimeError, match="Unexpected key"):
            event_policy.load_policy_checkpoint(malformed)


def test_world_model_live_member_digests_detect_mutation_beyond_ensemble_identity() -> None:
    dimensions = {head: 1 for head in event_models.WORLD_MODEL_HEADS}
    model = event_models.TemporalGraphWorldModel(dimensions, hidden_dim=16)
    ensemble = event_models.WorldModelEnsemble(
        models=[model], metadata=[{"seed": 8201}], ensemble_hash="immutable-file-identity",
    )

    before = event_policy._world_model_member_state_digests(ensemble)
    with torch.no_grad():
        model.node_encoder[0].weight[0, 0].add_(1.0)
    after = event_policy._world_model_member_state_digests(ensemble)

    assert before != after
    assert ensemble.ensemble_hash == "immutable-file-identity"
