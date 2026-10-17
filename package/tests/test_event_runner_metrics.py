from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import event_models
import event_runner


class _Ledger:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def count(self, name: str, amount: int) -> None:
        self.counts[name] = self.counts.get(name, 0) + amount


def test_world_model_evaluation_uses_target_masks_and_reports_metrics_by_parent(monkeypatch) -> None:
    predictions = {
        "candidate_keys": ["c1", "c2", "c3"],
        "means": {
            "next_state": [[0.0, 0.0], [0.5, 1.5], [50.0, 50.0]],
            "progress": [0.1, 0.2, 50.0], "delay": [1.0, 2.0, 50.0],
            "energy": [3.0, 4.0, 50.0], "failure": [0.0, 1.0, 50.0],
            "communication_risk": [0.1, 0.2, 50.0], "constraint_risk": [0.0, 1.0, 50.0],
            "utility": [0.9, 0.2, 99.0],
        },
        "variances": {
            "next_state": [[0.1, 0.1], [0.1, 0.1], [10.0, 10.0]],
            "progress": [0.1, 0.1, 10.0], "delay": [0.1, 0.1, 10.0],
            "energy": [0.1, 0.1, 10.0], "failure": [0.1, 0.1, 10.0],
            "communication_risk": [0.1, 0.1, 10.0], "constraint_risk": [0.1, 0.1, 10.0],
            "utility": [0.1, 0.1, 10.0],
        },
    }
    monkeypatch.setattr(event_models, "predict_ensemble", lambda *_: predictions)
    targets = {
        "next_state": {"values": [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], "valid": [[True, True], [True, False], [False, False]]},
        "progress": {"values": [0.0, 0.0, 0.0], "valid": [True, True, False]},
        "delay": {"values": [1.0, 1.0, 0.0], "valid": [True, True, False]},
        "energy": {"values": [3.0, 3.0, 0.0], "valid": [True, True, False]},
        "failure": {"values": [0.0, 0.0, 0.0], "valid": [True, True, False]},
        "communication_risk": {"values": [0.0, 0.0, 0.0], "valid": [True, True, False]},
        "constraint_risk": {"values": [0.0, 0.0, 0.0], "valid": [True, True, False]},
        "utility": {"values": [1.0, 0.0, 0.0], "valid": [True, True, False]},
    }
    sample = {
        "sample_id": "dev-1-t08", "parent_id": "dev-1", "split": "development",
        "history": [{}], "candidates": [{"candidate_key": key} for key in predictions["candidate_keys"]],
        "targets": targets,
    }
    ledger = _Ledger()

    metrics = event_runner._evaluate_world_model([sample], object(), ledger)

    assert metrics["sample_count"] == 1
    assert metrics["parent_ids"] == ["dev-1"]
    assert metrics["heads"]["utility"]["target_slots"] == 3
    assert metrics["heads"]["utility"]["valid_values"] == 2
    assert metrics["heads"]["utility"]["unknown_values"] == 1
    assert metrics["heads"]["failure"]["masked_bce"] is not None
    assert metrics["heads"]["failure"]["unknown_fraction"] == 1 / 3
    assert metrics["utility_ranking"]["mean_pairwise_concordance"] == 1.0
    assert metrics["utility_regret"]["mean_regret_from_predicted_best"] == 0.0
    assert metrics["oracle_best_candidate_hit_rate"] == {
        "hit_count": 1, "denominator": 1, "hit_rate": 1.0,
        "tie_policy": "any_shared_candidate_among_predicted_and_oracle_ties",
    }
    assert metrics["per_parent"]["dev-1"]["sample_count"] == 1
    assert metrics["per_parent"]["dev-1"]["utility_ranking"]["mean_pairwise_concordance"] == 1.0
    assert ledger.counts["world_model_forwards"] == 3


def test_event_fields_expose_reconstruction_cooldown_basis() -> None:
    fields = event_runner._event_fields(
        {"gamma_value": 0.7, "threshold": 0.5},
        {"triggered": False, "cooldown_active": True, "minutes_since_reconstruction": 2, "minutes_since_trigger": 1},
    )

    assert fields["cooldown_active"] is True
    assert fields["minutes_since_reconstruction"] == 2
    assert fields["minutes_since_trigger"] == 1


def test_final_report_marks_unrun_m2_as_unavailable_not_zero() -> None:
    parent = ROOT.parent / "evidence" / "local-test-temp"
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / uuid.uuid4().hex
    output.mkdir()
    settlement = {
        "stages": {"m0": {"status": "complete"}, "m1": {"status": "failed"}, "m2": {"status": "not_started"}},
        "totals": {}, "first_error": {"stage": "m1", "type": "RuntimeError", "message": "probe failure"},
    }
    try:
        (output / "RESOURCE_SETTLEMENT.json").write_text(json.dumps(settlement), encoding="utf-8")
        event_runner._write_final_report(output, {})
        report = (output / "FINAL_EVENT_TRIGGER_REPORT.md").read_text(encoding="utf-8")

        assert "M2: not_started (summary unavailable; counts unavailable)." in report
        assert "0/16" not in report
        assert "First recorded failure: stage=m1" in report
    finally:
        for child in output.iterdir():
            child.unlink()
        output.rmdir()


def test_m3_persists_final_threshold_before_development_evaluation(monkeypatch) -> None:
    from types import SimpleNamespace

    import event_policy
    import event_trigger

    parent = ROOT.parent / "evidence" / "local-test-temp"
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / uuid.uuid4().hex
    output.mkdir()
    train_parent_ids = [f"train-{index}" for index in range(4)]
    development_parent_ids = [f"development-{index}" for index in range(2)]
    scenarios = [
        *({"parent_id": parent_id, "split": "train"} for parent_id in train_parent_ids),
        *({"parent_id": parent_id, "split": "development"} for parent_id in development_parent_ids),
    ]
    training_samples = [
        {"sample_id": f"{parent_id}-{index}", "parent_id": parent_id, "split": "train", "history": [{}], "candidates": []}
        for parent_id in train_parent_ids for index in range(4)
    ]
    development_samples = [
        {"sample_id": f"{parent_id}-{index}", "parent_id": parent_id, "split": "development", "history": [{}], "candidates": []}
        for parent_id in development_parent_ids for index in range(4)
    ]
    sequences = [
        {"parent_id": parent_id, "observations": [{}, {}], "candidate_catalogs": [[{}], [{}]]}
        for parent_id in train_parent_ids
    ]
    event_runner._write_json(output / "CANDIDATE_RECONSTRUCTION_REPORT.json", {"gate_passed": True})
    event_runner._write_jsonl(output / "M2_MODEL_SAMPLES.jsonl", training_samples)
    event_runner._write_jsonl(output / "M2_DEVELOPMENT_MODEL_SAMPLES.jsonl", development_samples)
    event_runner._write_jsonl(output / "TRAINING_OBSERVATION_SEQUENCES.jsonl", sequences)
    event_runner._write_json(output / "THRESHOLD_MANIFEST_INITIAL.json", {"threshold": 0.4})

    ledger = _Ledger()
    ledger.output = output
    ledger.run = {"ledger": {"stages": {"m2": {"status": "complete"}}}}
    ensemble = SimpleNamespace(ensemble_hash="fixture-ensemble")
    predictions = {"fixture": "prediction"}
    training = {
        "status": "complete", "training_forward_calls": 0,
        "checkpoint_paths": ["seed-1", "seed-2", "seed-3"],
        "checkpoints": [{"optimizer_updates": 1, "parameter_delta_l2": 1.0} for _ in range(3)],
    }
    monkeypatch.setattr(event_models, "train_ensemble", lambda *_: training)
    monkeypatch.setattr(event_models, "load_ensemble", lambda *_: ensemble)
    monkeypatch.setattr(event_models, "predict_ensemble", lambda *_: predictions)
    monkeypatch.setattr(event_policy, "world_model_trigger_summary", lambda *_: {"trigger_score": 0.2})

    def calibrate(_sequences, _config):
        return {
            "wm_summaries_used": True, "source_parents": train_parent_ids,
            "threshold": 0.72, "training_input_sha256": "training-only-fixture",
        }

    monkeypatch.setattr(event_trigger, "calibrate_threshold", calibrate)
    monkeypatch.setattr(
        event_runner, "_fresh_process_json",
        lambda *_: {
            "status": "complete", "ensemble_hash": ensemble.ensemble_hash,
            "prediction_sha256": event_runner._prediction_digest(predictions),
        },
    )
    development_evaluation_sizes = []

    def evaluate(samples, _ensemble, _ledger):
        if samples[0]["split"] == "development":
            final_path = output / "THRESHOLD_MANIFEST_FINAL.json"
            active_path = output / "THRESHOLD_MANIFEST.json"
            assert final_path.is_file()
            assert active_path.is_file()
            final_manifest = event_runner._read_json(final_path)
            assert event_runner._read_json(active_path) == final_manifest
            assert final_manifest["frozen_before_development_evaluation"] is True
            assert final_manifest["source_split"] == "train"
            assert final_manifest["source_parent_ids"] == train_parent_ids
            development_evaluation_sizes.append(len(samples))
        return {"sample_count": len(samples)}

    monkeypatch.setattr(event_runner, "_evaluate_world_model", evaluate)
    try:
        metrics = event_runner._run_m3(
            scenarios, {"world_model": {}, "runtime": {"wall_timeout_seconds": 1}, "trigger": {}}, ledger,
        )

        assert development_evaluation_sizes == [8]
        assert metrics["evaluation_on_development_samples"]["sample_count"] == 8
        assert metrics["threshold_calibration"]["threshold"] == 0.72
        assert ledger.counts["world_model_forwards"] == 18
    finally:
        for child in output.iterdir():
            child.unlink()
        output.rmdir()
