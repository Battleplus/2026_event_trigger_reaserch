from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import gain_runner


def _temporary_directory() -> Path:
    path = ROOT / "tests" / f".gain-runner-test-{uuid4().hex}"
    path.mkdir()
    return path


def _remove_temporary_directory(path: Path) -> None:
    for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


class _ActionEnv:
    def _check_action(self, action):
        return action, []


def test_replanner_receipt_allows_deduplicated_candidates() -> None:
    action = {"noop": True}
    receipt = {
        "schema": "gain-fixed-replanner-receipt/1",
        "replanner_version": gain_runner.REPLANNER_VERSION,
        "selected_action": action,
        "candidates": [{"full_environment_action": action}],
        "actual_candidate_count": 2,
    }
    modules = {"replanner": SimpleNamespace(replan=lambda *args, **kwargs: receipt)}

    selected, validated, cpu_seconds, wall_seconds = gain_runner._invoke_replanner(
        modules, _ActionEnv(), {}, max_candidates=16,
    )

    assert selected == action
    assert validated["actual_candidate_count"] == 2
    assert cpu_seconds >= 0.0
    assert wall_seconds >= 0.0


def test_p1_review_verification_does_not_write_into_smoke_run(monkeypatch) -> None:
    temp_dir = _temporary_directory()
    try:
        project = temp_dir / "project"
        civilian = project / "package" / "civilian"
        civilian.mkdir(parents=True)
        (civilian / "source.py").write_text("SOURCE = True\n", encoding="utf-8")
        monkeypatch.setattr(gain_runner, "PROJECT_ROOT", project)
        monkeypatch.setattr(gain_runner, "CIVILIAN_DIR", civilian)

        run_dir = project / "outputs" / "smoke-1"
        run_dir.mkdir(parents=True)
        source_manifest = gain_runner._source_manifest()
        source_path = run_dir / "SOURCE_MANIFEST.json"
        gain_runner._write_json(source_path, source_manifest)
        source_sha = gain_runner._sha256_file(source_path)
        config_sha = "c" * 64
        run = {
            "schema": gain_runner.RUN_SCHEMA,
            "run_id": "smoke-1",
            "stage": "smoke",
            "config_sha256": config_sha,
            "source_manifest_path": "SOURCE_MANIFEST.json",
            "source_manifest_sha256": source_sha,
        }
        gain_runner._write_json(run_dir / "RUN.json", run)
        artifact = run_dir / "P1_SMOKE_REPORT.json"
        gain_runner._write_json(artifact, {"status": "complete"})
        files = []
        for path in sorted(run_dir.iterdir()):
            if path.is_file():
                files.append({
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": gain_runner._sha256_file(path),
                })
        evidence_path = run_dir / "EVIDENCE_MANIFEST.json"
        gain_runner._write_json(evidence_path, {
            "schema": gain_runner.EVIDENCE_SCHEMA,
            "run_id": "smoke-1",
            "stage": "smoke",
            "status": "complete",
            "config_sha256": config_sha,
            "source_manifest_sha256": source_sha,
            "files": files,
        })
        review_path = temp_dir / "P1_REVIEW.json"
        gain_runner._write_json(review_path, {
            "schema": gain_runner.REVIEW_SCHEMA,
            "decision": "PASS",
            "smoke_run_id": "smoke-1",
            "config_sha256": config_sha,
            "source_manifest_sha256": source_sha,
            "evidence_manifest_path": str(evidence_path),
            "evidence_manifest_sha256": gain_runner._sha256_file(evidence_path),
            "reviewer_model": "Codex GPT-6",
            "scope": ["source", "smoke", "pairing", "oracle", "leakage", "timing"],
        })
        before = sorted(path.name for path in run_dir.iterdir())

        result = gain_runner._verify_p1_review(review_path, config_sha)

        assert result["decision"] == "PASS"
        assert sorted(path.name for path in run_dir.iterdir()) == before
        assert ".CURRENT_SOURCE_MANIFEST.json" not in before
    finally:
        _remove_temporary_directory(temp_dir)


def test_replanner_receipt_rejects_candidate_count_above_cap() -> None:
    action = {"noop": True}
    receipt = {
        "schema": "gain-fixed-replanner-receipt/1",
        "replanner_version": gain_runner.REPLANNER_VERSION,
        "selected_action": action,
        "candidates": [{"full_environment_action": action}],
        "actual_candidate_count": 17,
    }
    modules = {"replanner": SimpleNamespace(replan=lambda *args, **kwargs: receipt)}

    with pytest.raises(ValueError, match="CANDIDATE_LIMIT_OR_RECEIPT"):
        gain_runner._invoke_replanner(modules, _ActionEnv(), {}, max_candidates=16)


def test_counterfactual_suffix_seed_is_namespaced_and_state_specific() -> None:
    first = gain_runner._suffix_seed("scenario-a", "state-a", 8, 0)
    assert first == gain_runner._suffix_seed("scenario-a", "state-a", 8, 0)
    assert 0 <= first < 2**63
    assert first != gain_runner._suffix_seed("scenario-b", "state-a", 8, 0)
    assert first != gain_runner._suffix_seed("scenario-a", "state-a", 8, 1)
    assert first != gain_runner._suffix_seed("scenario-a", "state-a", 9, 0)


def test_collection_label_suffix_binding_allows_forced_zero_pair_audit() -> None:
    forced = {
        "suffix_count": 0,
        "pairs": [],
        "forced": True,
        "valid_mask": False,
        "unknown_reason": "forced_safety_state",
    }
    assert gain_runner._validate_collection_label_suffix_binding(
        forced, [11, 22], forced=True,
    ) == 0

    paired = {
        "suffix_count": 2,
        "pairs": [{"suffix_seed": 11}, {"suffix_seed": 22}],
    }
    assert gain_runner._validate_collection_label_suffix_binding(
        paired, [11, 22], forced=False,
    ) == 2

    with pytest.raises(ValueError, match="ZERO_PAIR_REASON_INVALID"):
        gain_runner._validate_collection_label_suffix_binding(
            {**forced, "unknown_reason": "paired_collection_failed"}, [11, 22], forced=True,
        )


def test_variance_record_loader_rechecks_artifact_hashes_and_bindings() -> None:
    temp_dir = _temporary_directory()
    try:
        scenario = {"parent_id": "scenario-a", "split": "train"}
        snapshot_json = '{"snapshot_sha256":"snapshot-a"}'
        label = {"label_sha256": "label-a"}
        suffix_jsons = ['{"suffix_seed":1}', '{"suffix_seed":2}']
        artifacts = {
            "scenario.json.gz": scenario,
            "snapshot.json.gz": {"snapshot_json": snapshot_json},
            "suffix-1.json.gz": {"suffix_json": suffix_jsons[0]},
            "suffix-2.json.gz": {"suffix_json": suffix_jsons[1]},
            "label.json.gz": {"label": label},
        }
        for name, payload in artifacts.items():
            gain_runner._write_gzip_json(temp_dir / name, payload)

        def source(path: str) -> dict[str, str]:
            return {"path": path, "sha256": gain_runner._sha256_file(temp_dir / path)}

        sources = {
            "state-a": {
                "collection_directory": temp_dir,
                "scenario_path": "scenario.json.gz",
                "scenario_file_sha256": source("scenario.json.gz")["sha256"],
                "snapshot_path": "snapshot.json.gz",
                "snapshot_file_sha256": source("snapshot.json.gz")["sha256"],
                "suffix_paths": ["suffix-1.json.gz", "suffix-2.json.gz"],
                "suffix_file_hashes": [
                    source("suffix-1.json.gz")["sha256"],
                    source("suffix-2.json.gz")["sha256"],
                ],
                "label_path": "label.json.gz",
                "label_file_sha256": source("label.json.gz")["sha256"],
            },
        }
        row = {"state_id": "state-a", "snapshot_sha256": "snapshot-a", "label_sha256": "label-a"}

        loaded = gain_runner._load_variance_record(row, sources)

        assert loaded == {
            "row": row, "scenario": scenario, "snapshot_json": snapshot_json,
            "label": label, "suffix_jsons": suffix_jsons,
        }
        (temp_dir / "suffix-2.json.gz").write_bytes(b"changed")
        with pytest.raises(ValueError, match="VARIANCE_SUFFIX_HASH_MISMATCH"):
            gain_runner._load_variance_record(row, sources)
    finally:
        _remove_temporary_directory(temp_dir)


def test_validated_suffix_gain_extraction_uses_label_pairs_and_checks_aggregates() -> None:
    label = {
        "valid_mask": True,
        "pairs": [
            {"valid_mask": True, "gain": 1.0},
            {"valid_mask": True, "gain": 3.0},
        ],
    }
    receipt = {
        "pair_count": 2, "label_valid_mask": True,
        "recomputed_gain_mean": 2.0, "recomputed_paired_variance": 2.0,
    }

    assert gain_runner._validated_suffix_gains(label, receipt) == [1.0, 3.0]

    receipt["recomputed_paired_variance"] = 1.0
    with pytest.raises(ValueError, match="ORACLE_SUFFIX_SUMMARY_MISMATCH"):
        gain_runner._validated_suffix_gains(label, receipt)
