from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pytest

from diagnose_batch import _suffix_evidence, analyze_batch


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _registry(tmp_path: Path) -> tuple[Path, dict]:
    row = {
        "run_id": "run-1",
        "scenario_id": "condition-seed1",
        "split": "train",
        "condition": "condition",
        "source": "Existing-ET",
        "seed": 1,
        "config_sha256": "config-sha",
        "output_path": "scenes/run-1",
    }
    value = {"run_count": 1, "suffix_seeds": [730201, 730202], "runs": [row]}
    path = tmp_path / "registry.json"
    _write_json(path, value)
    return path, row


@pytest.fixture
def local_tmp_path():
    path = Path(__file__).parent / f".diagnose-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_missing_raw_evidence_cannot_count_toward_30_30(local_tmp_path: Path):
    tmp_path = local_tmp_path
    registry_path, row = _registry(tmp_path)
    batch = {
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "execution": "COMPLETE",
        "scenes": [{**row, "execution": "COMPLETE", "representative": {"return_code": 0},
                    "states": []}],
    }
    batch_path = tmp_path / "batch_receipt.json"
    _write_json(batch_path, batch)
    aggregate_path = tmp_path / "aggregate.json"
    _write_json(aggregate_path, {
        "status": "PASS", "met_30_30": True,
        "states": [{
            "scenario_id": row["scenario_id"], "local_state_id": "state_00_000100",
            "valid_mask": True, "gain_mean": 1.0,
            "source_reports": [
                "run-1/state_00_000100/suffix_730201/reduction.json",
                "run-1/state_00_000100/suffix_730202/reduction.json",
            ],
        }],
    })

    report = analyze_batch(registry_path, batch_path, aggregate_path)
    labels = report["state_coverage"]["state_label_coverage"]
    assert labels["verified_unique_ordinary_state_count"] == 0
    assert labels["at_least_30_per_class"] is False
    assert report["data_status"] == "INCOMPLETE_OR_INSUFFICIENT"


def test_failed_suffix_worker_cannot_count_as_raw_or_reduction(local_tmp_path: Path):
    tmp_path = local_tmp_path
    row = {
        "run_id": "run-1", "scenario_id": "condition-seed1", "split": "train",
        "condition": "condition", "source": "Existing-ET", "config_sha256": "config-sha",
        "output_path": "scenes/run-1",
    }
    suffix_dir = tmp_path / "scenes" / "run-1" / "suffixes" / "state_00_000100" / "suffix_730201"
    _write_json(suffix_dir / "run_receipt.json", {
        "run_id": row["run_id"], "phase": "suffix_pair", "status": "complete",
        "actual_isaac_run": True, "config_sha256": row["config_sha256"],
        "formal_identity": {key: row[key] for key in ("scenario_id", "split", "condition", "source")},
        "suffix": {"seed": 730201, "decision_time_s": 100,
                   "audit_interface": {"state_id": "state_00_000100",
                                        "state_kind": "ordinary_sampled_decision_state",
                                        "forced": False}},
        "trajectory": "trajectory.jsonl.gz", "events": "events.jsonl",
    })
    (suffix_dir / "trajectory.jsonl.gz").write_bytes(b"raw")
    (suffix_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")
    evidence = _suffix_evidence(
        tmp_path, row, "state_00_000100", 730201, 100,
        {"suffix_seed": 730201, "return_code": 1},
    )
    assert evidence["raw_complete"] is False
    assert evidence["valid_reduction"] is False
    assert "suffix_worker_receipt_missing_or_failed" in evidence["unknown_reasons"]
