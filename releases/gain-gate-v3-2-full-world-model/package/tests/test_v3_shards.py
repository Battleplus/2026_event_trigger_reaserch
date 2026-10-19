from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from civilian import v3_shards
from civilian import v3_runner


def _scenario(split: str = "train") -> dict:
    return {
        "parent_id": "scenario-001", "split": split, "condition": "none_or_light",
        "scenario_content_sha256": "a" * 64,
    }


def _commit(tmp_path: Path, *, rows=None, binding=None):
    return v3_shards.commit_scenario_shard(
        output_dir=tmp_path, run_id="run-001", scenario=_scenario(),
        rows=rows or [{"state_id": "s1", "gain": 0.25}],
        binding=binding or {"source_manifest_sha256": "b" * 64, "config_sha256": "c" * 64, "evidence_sha256": "d" * 64},
        runtime_receipt={"cpu_threads": 1, "cuda_visible_devices": ""},
    )


def test_commit_is_readback_validated_and_bound(tmp_path):
    marker = _commit(tmp_path)
    path = tmp_path / "run-001" / "shards" / "train" / "none_or_light" / "scenario-001.commit.json"
    verified = v3_shards.verify_scenario_shard(path, expected_binding=marker["binding"])
    assert verified["row_count"] == 1
    assert verified["marker_sha256"] == marker["marker_sha256"]


def test_corrupt_compressed_data_is_rejected_without_repair(tmp_path):
    _commit(tmp_path)
    shard = tmp_path / "run-001" / "shards" / "train" / "none_or_light" / "scenario-001.jsonl.gz"
    original = shard.read_bytes()
    shard.write_bytes(original[:-4])
    with pytest.raises(ValueError, match="COMPRESSED_HASH_MISMATCH"):
        v3_shards.verify_scenario_shard(shard.with_name("scenario-001.commit.json"))
    assert shard.read_bytes() == original[:-4]


def test_partial_data_without_marker_is_not_a_completed_shard(tmp_path):
    directory = tmp_path / "run-001" / "shards" / "train" / "none_or_light"
    directory.mkdir(parents=True)
    with gzip.open(directory / "orphan.jsonl.gz", "wb") as stream:
        stream.write(b'{"state_id":"orphan"}\n')
    result = v3_shards.verify_run(tmp_path, "run-001")
    assert result["committed_shards"] == 0
    assert result["incomplete_data_files"] == [str(directory / "orphan.jsonl.gz")]


def test_marker_hash_and_source_config_binding_are_checked(tmp_path):
    marker = _commit(tmp_path)
    marker_path = tmp_path / "run-001" / "shards" / "train" / "none_or_light" / "scenario-001.commit.json"
    value = json.loads(marker_path.read_text())
    value["row_count"] = 17
    marker_path.write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(ValueError, match="MARKER_HASH_MISMATCH"):
        v3_shards.verify_scenario_shard(marker_path)

    canonical_marker = dict(marker)
    canonical_marker.pop("marker_path")
    marker_path.write_text(json.dumps(canonical_marker), encoding="ascii")
    with pytest.raises(ValueError, match="BINDING_MISMATCH"):
        v3_shards.verify_scenario_shard(marker_path, expected_binding={"source_manifest_sha256": "x"})


def test_registry_is_exactly_168_train_validation_and_seals_test_ood():
    registry = v3_shards.expected_registry()
    assert len(registry) == 168
    assert sum(row["split"] == "train" for row in registry) == 84
    assert sum(row["split"] == "validation" for row in registry) == 84
    assert all(row["split"] in {"train", "validation"} for row in registry)
    with pytest.raises(ValueError, match="REGISTRY_NOT_FROZEN"):
        v3_shards.validate_registry([*registry, {"split": "test"}])


def test_atomic_marker_failure_leaves_uncommitted_data_and_no_receipt(tmp_path, monkeypatch):
    original = v3_shards._write_exclusive

    def fail_marker(path: Path, payload: bytes):
        if ".commit.json." in path.name:
            raise OSError("injected marker write failure")
        return original(path, payload)

    monkeypatch.setattr(v3_shards, "_write_exclusive", fail_marker)
    with pytest.raises(OSError, match="injected marker"):
        _commit(tmp_path)
    result = v3_shards.verify_run(tmp_path, "run-001")
    assert result["committed_shards"] == 0
    assert len(result["incomplete_data_files"]) == 1
    assert not list((tmp_path / "run-001").glob("**/RUN_RECEIPT.json"))


def test_review_approval_must_bind_live_source_config_and_baseline_evidence(tmp_path):
    binding = v3_shards.current_binding()
    instances = []
    for index in range(18):
        marker = v3_shards.commit_scenario_shard(
            output_dir=tmp_path, run_id="smoke-fixture",
            scenario={
                "parent_id": f"smoke-{index:02d}", "split": "train",
                "condition": v3_shards.CONDITIONS[index // 3],
                "scenario_content_sha256": f"{index:064x}",
            },
            rows=[{"state_id": f"state-{index}"}], binding=binding,
            runtime_receipt={"within_limits": True, "exit_state": "complete"},
        )
        instances.append({
            "scenario_id": marker["scenario_id"],
            "marker_path": marker["marker_path"],
            "marker_sha256": marker["marker_sha256"],
        })
    manifest_path = tmp_path / "smoke-manifest.json"
    summary_path = tmp_path / "smoke-summary.json"
    manifest_path.write_text(json.dumps({
        "schema": "gain-gate-v3-smoke-manifest/1", "status": "complete",
        "completed_instances": 18, "binding": binding, "instances": instances,
        "checks": {
            "scenario_scale_and_regions": True,
            "complete_public_trajectory": True,
            "paired_label_independent_recomputation": True,
            "committed_shard_integrity": True,
        },
    }), encoding="utf-8")
    summary_path.write_text(json.dumps({
        "schema": "gain-gate-v3-smoke-summary/1", "status": "complete",
        "completed_instances": 18, "binding": binding,
        "resource_receipt": {"within_limits": True},
        "checks": {
            "scenario_scale_and_regions": True,
            "complete_public_trajectory": True,
            "paired_label_independent_recomputation": True,
            "committed_shard_integrity": True,
        },
        "manifest_sha256": v3_shards.sha256_file(manifest_path),
    }), encoding="utf-8")
    review_path = tmp_path / "readonly-review.json"
    review_path.write_text(json.dumps({
        "schema": v3_runner.REVIEW_SCHEMA,
        "decision": "approved",
        "binding": binding,
        "smoke_evidence": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": v3_shards.sha256_file(manifest_path),
            "summary_path": str(summary_path),
            "summary_sha256": v3_shards.sha256_file(summary_path),
        },
    }), encoding="utf-8")
    assert v3_runner._load_review(review_path, binding)["decision"] == "approved"
    mismatched = {**binding, "source_manifest_sha256": "0" * 64}
    with pytest.raises(ValueError, match="BINDING_MISMATCH"):
        v3_runner._load_review(review_path, mismatched)


def test_resource_receipt_has_no_cpu_or_wall_stop_and_keeps_memory_limit():
    receipt = v3_shards.resource_receipt(stage="smoke", started_wall=0.0, started_cpu=0.0, peak_rss_bytes=1234)
    assert receipt["limits"] == {"wall_seconds": None, "cpu_seconds": None, "rss_bytes": 8 * 1024**3, "cpu_threads": 1, "cuda": "disabled"}
    assert receipt["peak_rss_bytes"] == 1234
    assert "working sets" in receipt["rss_measurement"]
    assert "unknown" in receipt["historical_v2_cpu_total"]


def test_random_suppress_decision_does_not_call_online_replanner():
    calls = []
    allowed, receipt, cpu = v3_runner._online_plan(
        "Random-Allow-Suppress", False, __import__("random").Random(0),
        lambda: calls.append("called") or {"selected_action": {}},
    )
    assert allowed is False
    assert receipt is None
    assert cpu == 0.0
    assert calls == []


def test_forced_state_bypasses_random_suppression():
    calls = []
    allowed, receipt, _ = v3_runner._online_plan(
        "Random-Allow-Suppress", True, __import__("random").Random(0),
        lambda: calls.append("called") or {"selected_action": {"noop": True}},
    )
    assert allowed is True
    assert receipt == {"selected_action": {"noop": True}}
    assert calls == ["called"]


def test_guardian_terminates_cpu_bound_child_at_budget():
    process = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    try:
        reason, cpu_seconds, _rss, _wall = v3_runner._monitor_guarded_process(
            process, cpu_limit=0.20, wall_limit=10.0, poll_interval=0.02,
        )
        assert reason == "cpu_limit_exceeded"
        assert cpu_seconds >= 0.20
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_guardian_records_normal_worker_exit_without_time_limits():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.15)"])
    reason, cpu_seconds, peak_rss, wall_seconds = v3_runner._monitor_guarded_process(
        process, cpu_limit=None, wall_limit=None, poll_interval=0.02,
    )
    assert process.wait() == 0
    assert reason is None
    assert cpu_seconds >= 0.0
    assert peak_rss > 0
    assert wall_seconds >= 0.15


def test_process_tree_working_set_includes_descendants():
    class FakeProcess:
        def __init__(self, rss, children=()):
            self.rss = rss
            self._children = list(children)

        def children(self, recursive=True):
            return self._children

        def memory_info(self):
            return type("MemoryInfo", (), {"rss": self.rss})()

    child = FakeProcess(300)
    root = FakeProcess(200, [child])
    assert v3_runner._process_tree_working_set(root) == 500
