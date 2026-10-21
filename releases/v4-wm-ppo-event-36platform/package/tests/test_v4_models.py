import io
import hashlib
import os
import json
import base64
import subprocess
import sys
from pathlib import Path

import torch

from civilian import event_models, v4_models, v4_policy
from civilian.v4_checkpoint_witness import (
    _wm_probe_identity, check_policies, check_world_models, policy_probe_prediction,
)


def _observation(platform_count=2, task_count=2):
    return {
        "map": {"rows": 100, "cols": 100, "grid_cells": 10000,
                "grid_contract": "relief-grid-100x100-v1"},
        "time": 0, "horizon": 100,
        "platforms": [{"id": f"p{i}", "position": [i, 0], "energy": 90,
                       "capacity_kg": 20, "speed_km_min": 1,
                       "energy_per_km": 1, "service_energy_per_min": 1}
                      for i in range(platform_count)],
        "tasks": [{"id": f"t{i}", "position": [i, 1], "quantity": 1,
                   "deadline": 80, "kind": "delivery"}
                  for i in range(task_count)],
        "environment": [], "communication": [], "queues": {},
        "predecessors": {}, "cooperation": {}, "task_progress": {},
    }


def _sample(action, observation=None):
    observation = observation or _observation()
    return {"history": [{"observation": observation}], "action": action}


def test_dynamic_graph_ids_match_event_encoder_and_action_operations_are_unbounded():
    observation = _observation(platform_count=3, task_count=4)
    action = {"scope": "LOCAL_REPAIR", "dispatches": [
        {"platform_id": "p0", "task_id": "t0"} for _ in range(40)],
        "queue_operations": [
            {"op": "split", "task_id": "t0", "from_platform": "p0",
             "to_platform": "p1", "share": 0.25}
            for _ in range(14)]}
    batch, _ = v4_models.collate_v4_samples([_sample(action, observation)])
    encoded = event_models._encode_sample_graph(_sample(action, observation))
    expected_count = len(encoded[1][-1])
    assert batch["node_mask"][0, -1].sum().item() == expected_count
    assert batch["action_features"].shape[1] == expected_count
    ids = v4_models._graph(_sample(action, observation))[6]
    assert ids == ["p0", "p1", "p2", "t0", "t1", "t2", "t3"]
    assert batch["action_features"][0, 0, 4].item() > 1.0
    assert batch["action_features"][0, 0, 5].item() > 0.1
    assert batch["action_features"][0, 0, 10].item() == 14.0


def test_reversing_operation_roles_changes_action_encoding_and_relation_graph():
    observation = _observation()
    forward = {"scope": "LOCAL_REPAIR", "queue_operations": [
        {"op": "transfer", "task_id": "t0", "from_platform": "p0",
         "to_platform": "p1"}]}
    reverse = {"scope": "LOCAL_REPAIR", "queue_operations": [
        {"op": "transfer", "task_id": "t0", "from_platform": "p1",
         "to_platform": "p0"}]}
    first, _ = v4_models.collate_v4_samples([_sample(forward, observation)])
    second, _ = v4_models.collate_v4_samples([_sample(reverse, observation)])
    assert not torch.equal(first["action_features"], second["action_features"])
    assert not torch.equal(first["action_adjacency"], second["action_adjacency"])


def test_normalization_streams_samples_and_identity_matches_canonical_list_digest():
    samples = [_sample({"scope": "NOOP", "noop": True}),
               _sample({"scope": "LOCAL_REPAIR", "dispatches": [
                   {"platform_id": "p0", "task_id": "t1"}]})]
    model = v4_models.V4DynamicWorldModel()
    norm = v4_models._fit_norm(model, iter(samples))
    batches = [v4_models.collate_v4_samples([sample])[0] for sample in samples]
    node = torch.cat([batch["node_features"][batch["node_mask"]] for batch in batches])
    action = torch.cat([batch["action_features"][batch["node_mask"][:, -1]] for batch in batches])
    assert torch.allclose(model.node_mean, node.mean(0), atol=1e-6)
    assert torch.allclose(model.action_scale, action.std(0, unbiased=False).clamp_min(1e-6), atol=1e-6)
    expected = hashlib.sha256(json.dumps(
        [{key: sample.get(key) for key in ("sample_id", "split", "parent_id", "history", "action", "targets")}
         for sample in samples], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert v4_models._identity(iter(samples)) == expected


def test_world_model_action_conditioning_changes_predictions_and_optimizer_updates():
    observation = _observation()
    noop = _sample({"scope": "NOOP", "noop": True}, observation)
    dispatch = _sample({"scope": "LOCAL_REPAIR", "dispatches": [
        {"platform_id": "p0", "task_id": "t0"}]}, observation)
    model = v4_models.V4DynamicWorldModel()
    b_noop, _ = v4_models.collate_v4_samples([noop])
    b_dispatch, _ = v4_models.collate_v4_samples([dispatch])
    assert not torch.equal(model(b_noop)["progress"], model(b_dispatch)["progress"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.heads["progress"].weight.detach().clone()
    prediction = model(b_dispatch)
    loss = prediction["progress"].square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.heads["progress"].weight.detach())


def test_global_heads_condition_on_recipient_platform_state():
    torch.manual_seed(5521)
    observation = _observation()
    observation["platforms"][0]["energy"] = 12
    observation["platforms"][1]["energy"] = 97
    observation["platforms"][0]["position"] = [3, 8]
    observation["platforms"][1]["position"] = [81, 2]
    first = _sample({"scope": "LOCAL_REPAIR", "dispatches": [
        {"platform_id": "p0", "task_id": "t0"}]}, observation)
    second = _sample({"scope": "LOCAL_REPAIR", "dispatches": [
        {"platform_id": "p1", "task_id": "t0"}]}, observation)
    batches = [v4_models.collate_v4_samples([sample])[0] for sample in (first, second)]
    model = v4_models.V4DynamicWorldModel().eval()
    with torch.no_grad():
        first_score = model(batches[0])["progress"]
        second_score = model(batches[1])["progress"]
    assert not torch.equal(first_score, second_score)


def test_validation_evaluator_reports_masked_counts_and_probability_brier():
    sample = _sample({"scope": "NOOP", "noop": True})
    sample["split"] = "validation"
    sample["targets"] = {
        "next_minute": {"entity_ids": ["p0", "t0"],
                        "values": [[1, 0, 0, 0], [0, 2, 0, 0]],
                        "valid": [[True, False, False, False],
                                  [False, True, False, False]]},
        "progress": {"value": 2.0, "valid": True},
        "failure": {"value": 1.0, "valid": True},
    }
    models = [v4_models.V4DynamicWorldModel() for _ in v4_models.V4_SEEDS]
    result = v4_models.evaluate_v4_predictions(models, [sample])
    x_metrics = result["heads"]["next_minute"]["dimensions"]["x_delta_km"]
    assert x_metrics["model"]["valid_count"] == 1
    assert x_metrics["model"]["valid_sample_count"] == 1
    assert result["heads"]["progress"]["model"]["valid_count"] == 1
    assert result["heads"]["failure"]["model"]["brier"] is not None


def test_probability_ensemble_averages_member_probabilities_and_returns_disagreement():
    models = [v4_models.V4DynamicWorldModel() for _ in range(3)]
    for index, model in enumerate(models):
        model.heads["failure"].weight.data.zero_()
        model.heads["failure"].bias.data.fill_((-1.0, 0.0, 1.0)[index])
    prediction = v4_models.predict_v4_ensemble(models, [_sample({"scope": "NOOP"})])
    expected = torch.tensor([torch.sigmoid(torch.tensor(-1.0)), 0.5,
                             torch.sigmoid(torch.tensor(1.0))]).mean()
    assert torch.allclose(prediction["failure"], expected.reshape(1))
    assert prediction["failure_std"].item() > 0
    assert prediction["failure_variance"].item() > 0
    assert prediction["next_minute_std"].shape == prediction["next_minute"].shape


def test_training_requires_explicit_cuda_and_validation_split():
    sample = _sample({"scope": "NOOP"}); sample["split"] = "train"
    validation = _sample({"scope": "NOOP"}); validation["split"] = "validation"
    try:
        v4_models.train_v4_ensemble([sample], "unused", {"device": "cpu"},
                                    validation_samples=[])
    except ValueError as error:
        assert "VALIDATION" in str(error)
    else:
        raise AssertionError("training accepted missing validation samples")
    try:
        v4_models.train_v4_ensemble([sample], "unused", {"device": "cpu"},
                                    validation_samples=[validation])
    except RuntimeError as error:
        assert "REQUIRES_CUDA" in str(error)
    else:
        raise AssertionError("training accepted a CPU device")


def test_checkpoint_predictions_restore_in_fresh_process():
    sample = _sample({"scope": "NOOP"})
    models = [v4_models.V4DynamicWorldModel() for _ in v4_models.V4_SEEDS]
    expected = v4_models.predict_v4_ensemble(models, [sample])["progress"].tolist()
    package = Path(__file__).resolve().parents[1]
    payload = io.BytesIO()
    torch.save({"states": [model.state_dict() for model in models], "sample": sample}, payload)
    script = (
        "import base64,io,json,sys,torch; from civilian.v4_models import V4DynamicWorldModel,predict_v4_ensemble; "
        "payload=torch.load(io.BytesIO(base64.b64decode(sys.stdin.read())),weights_only=False); models=[V4DynamicWorldModel() for _ in range(3)]; "
        "[model.load_state_dict(state) for model,state in zip(models,payload['states'])]; "
        "print(json.dumps(predict_v4_ensemble(models,[payload['sample']])['progress'].tolist()))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package)
    completed = subprocess.run([sys.executable, "-c", script], input=base64.b64encode(payload.getvalue()).decode(), text=True,
                               capture_output=True, check=True, env=env)
    actual = json.loads(completed.stdout)
    assert torch.allclose(torch.tensor(actual), torch.tensor(expected))


def test_formal_restore_witnesses_compare_presave_prediction_and_optimizer_state():
    tmp_path = Path(__file__).resolve().parent / "_restore_witness_fixture"
    tmp_path.mkdir(exist_ok=True)
    sample = _sample({"scope": "NOOP"})
    sample["split"] = "train"
    model = v4_models.V4DynamicWorldModel().eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch, _ = v4_models.collate_v4_samples([sample])
    loss = sum(value.square().mean() for value in model(batch).values())
    loss.backward(); optimizer.step()
    with torch.no_grad():
        expected_wm = {key: value[0].cpu().tolist() for key, value in model(batch).items()}
    data_identity = v4_models._identity([sample])
    norm = {name: getattr(model, name).cpu().tolist() for name in
            ("node_mean", "node_scale", "action_mean", "action_scale")}
    wm_path = tmp_path / "wm.pt"
    torch.save({"schema": v4_models.V4_MODEL_SCHEMA, "seed": 8201, "config": model.config,
                "data_identity": data_identity, "normalization": norm, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "python_rng_state": __import__("random").getstate(),
                "torch_rng_state": torch.get_rng_state(), "cuda_rng_state_all": [torch.zeros(1)],
                "restore_witness": {"presave_prediction": expected_wm,
                                    "reference_device": "cpu",
                                    "probe_sample_identity": _wm_probe_identity(sample)}}, wm_path)
    wm_check = check_world_models([{"checkpoint_path": str(wm_path), "probe": sample,
                                    "presave_prediction": expected_wm, "data_identity": data_identity,
                                    "probe_sample_identity": _wm_probe_identity(sample)}])
    assert wm_check["checks"][0]["optimizer_state_entries"] > 0
    assert wm_check["checks"][0]["optimizer_state_matches"] is True
    assert wm_check["checks"][0]["normalization_restored"]
    assert torch.load(wm_path, map_location="cpu", weights_only=False)["restore_witness"]["reference_device"] == "cpu"

    observation = _observation()
    candidates = [{"candidate_key": "NOOP", "candidate_scope": "NOOP", "legal": True,
                   "full_environment_action": {"scope": "NOOP", "noop": True}},
                  {"candidate_key": "LOCAL", "candidate_scope": "LOCAL_REPAIR", "legal": False,
                   "full_environment_action": {"scope": "LOCAL_REPAIR"}}]
    policy = v4_policy.build_v4_policy(13)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
    policy.value(observation).backward(); policy_optimizer.step()
    policy_path = tmp_path / "policy.pt"
    probe = {"observation": observation, "candidates": candidates,
             "high_level_mask": [True, False, False], "candidate_mask": [True, False], "wm_predictions": None}
    expected_policy = policy_probe_prediction(policy, probe)
    v4_policy.save_v4_checkpoint(policy_path, policy, policy_optimizer, variant="Event-G", seed=13,
                                 config={"learning_rate": 0.0003}, source_identity="fixture",
                                 restore_witness={"data_identity": "fixture", "presave_prediction": expected_policy,
                                                  "reference_device": "cpu",
                                                  "probe": probe})
    policy_check = check_policies([{"checkpoint_path": str(policy_path), "data_identity": "fixture",
                                    "presave_prediction": expected_policy, "probe": probe}])
    assert policy_check["checks"][0]["optimizer_state_entries"] > 0
    assert policy_check["checks"][0]["optimizer_state_matches"] is True
    assert torch.load(policy_path, map_location="cpu", weights_only=False)["restore_witness"]["reference_device"] == "cpu"
    wm_path.unlink(); policy_path.unlink()
