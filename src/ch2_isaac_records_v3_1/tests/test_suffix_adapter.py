import hashlib
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_suffix_light
from evaluate_suffix_light import _legacy_evaluator_inputs


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_legacy_evaluator_adapter_reconstructs_inputs_in_temp_tree(tmp_path, monkeypatch):
    scene = tmp_path / "scene"
    labels = scene / "labels"
    snapshots = labels / "snapshots"
    snapshots.mkdir(parents=True)
    private = snapshots / "state_00_000010.pickle"
    public = snapshots / "state_00_000010.public.json"
    private.write_bytes(b"private snapshot")
    public.write_text("{}\n", encoding="utf-8")
    index = {"states": [{"state_id": "state_00_000010", "sim_time_s": 10,
                         "decision_time_s": 10, "backend_time_s": 9.0,
                         "continuation_phase": "post_public_updates_pre_candidate_generation_pre_step",
                         "private_file": "snapshots/state_00_000010.pickle",
                         "private_sha256": _sha(private),
                         "public_file": "snapshots/state_00_000010.public.json",
                         "public_sha256": _sha(public)}]}
    source_index = labels / "state_index.json"
    source_index.write_text(json.dumps(index), encoding="utf-8")
    (labels / "evaluator_truth.json").write_text(
        json.dumps({"targets": [{"target_id": "target-1", "zone": 0,
                                 "target_xy_km": [1.0, 2.0], "kind": "uav"}]}),
        encoding="utf-8")
    source_manifest = {"run_id": "scene-a", "scenario_id": "scenario-a",
                       "config_sha256": "internal-hash", "seed": 7}
    config = {"run_id": "scene-a", "seed": 7}
    suffix = tmp_path / "suffix"
    suffix.mkdir()
    (suffix / "manifest.json").write_text(json.dumps({
        "run_id": "scene-a", "config": config,
        "config_sha256": "internal-hash"}), encoding="utf-8")
    receipt = {"run_id": "scene-a", "suffix": {
        "state_id": "state_00_000010", "source_config_sha256": "internal-hash",
        "source_manifest_config_sha256": "internal-hash",
        "source_manifest_metadata": source_manifest}}
    (suffix / "run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with gzip.open(suffix / "trajectory.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write('{"format":"lightweight_actor_delta_v1"}\n')
        stream.write(json.dumps({"branch": "keep", "aircraft_id": "uav-0-0",
                                 "set": {"sim_time_s": 10}, "delete": []}) + "\n")
    (suffix / "events.jsonl").write_text("{}\n", encoding="utf-8")

    adapted_suffix, adapted_index = _legacy_evaluator_inputs(
        suffix, source_index, tmp_path / "adapted")
    adapted_manifest = json.loads((adapted_index.parent.parent / "manifest.json").read_text())
    adapted_receipt = json.loads((adapted_index.parent.parent / "run_receipt.json").read_text())
    adapted_suffix_receipt = json.loads((adapted_suffix / "run_receipt.json").read_text())
    assert adapted_manifest["config"] == config
    assert adapted_receipt["state_sampling"] == index
    assert adapted_suffix_receipt["suffix"]["source_manifest_metadata"] == adapted_manifest
    assert (adapted_suffix / "events.jsonl").is_file()
    assert not (scene / "manifest.json").exists()

    class FakeEvaluator:
        _jsonl = staticmethod(lambda _path: iter(()))
        _read_snapshot = staticmethod(lambda *_args, **_kwargs: ({}, {}, "snapshot"))

        def evaluate_suffix(self, suffix_path, _index_path, _source_root, _collector_root):
            rows = list(self._jsonl(suffix_path / "trajectory.jsonl.gz"))
            return {"status": "PASS", "decoded_rows": len(rows)}

    monkeypatch.setattr(evaluate_suffix_light, "load_evaluator", lambda _path: FakeEvaluator())
    report = evaluate_suffix_light.evaluate_suffix(suffix, source_index, tmp_path, tmp_path,
                                                   tmp_path / "evaluator.py")
    assert report["decoded_rows"] == 1
    assert report["trajectory_storage"]["decoded_rows"] == 1
