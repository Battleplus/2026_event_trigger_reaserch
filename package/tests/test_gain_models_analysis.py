from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "civilian"))

import gain_analysis
import gain_gate
import gain_models


def _temporary_directory() -> Path:
    path = ROOT / "tests" / f".gain-test-{uuid4().hex}"
    path.mkdir()
    return path


def _remove_temporary_directory(path: Path) -> None:
    for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


def _rows() -> list[dict]:
    rows = []
    for split, prefix, count in (("train", "tr", 4), ("validation", "va", 8), ("test", "te", 2)):
        for group in range(count):
            scenario_id = f"{prefix}-{group}"
            for window in range(2):
                rows.append({
                    "state_id": f"{scenario_id}-s{window}", "scenario_id": scenario_id,
                    "parent_family": "arrival", "split": split, "forced": False,
                    "valid_mask": True,
                    "features": {"event_type_arrival": float(window), "affected_task_count": float(group + window), "time_fraction": group / 10},
                    "gain_mean": float(group - window) / 5,
                })
    return rows


def test_public_feature_guard_rejects_hidden_identity_and_future_fields() -> None:
    with pytest.raises(ValueError, match="PRIVATE_FEATURE"):
        gain_models._feature_map({"future_event_count": 2.0})
    with pytest.raises(ValueError, match="PRIVATE_FEATURE"):
        gain_models._feature_map({"scenario_id_numeric": 3.0})
    with pytest.raises(ValueError, match="NUMERIC_FEATURE"):
        gain_models._feature_map({"communication_quality": "good"})


def test_group_leakage_rejects_a_scenario_split_across_splits() -> None:
    rows = _rows()
    rows.append({**rows[0], "split": "test"})
    with pytest.raises(ValueError, match="GROUP_SPLIT_LEAKAGE"):
        gain_models.check_group_leakage(rows)


@pytest.mark.parametrize("lineage_field", [
    "scenario_content_sha256", "scenario_seed", "private_snapshot_sha256", "trajectory_id",
])
def test_group_leakage_rejects_shared_lineage_after_scenario_rename(lineage_field: str) -> None:
    rows = [
        {"scenario_id": "train-name", "parent_family": "arrival", "split": "train", lineage_field: "shared"},
        {"scenario_id": "test-name", "parent_family": "arrival", "split": "test", lineage_field: "shared"},
    ]
    with pytest.raises(ValueError, match="LINEAGE_SPLIT_LEAKAGE"):
        gain_models.check_group_leakage(rows)


def test_group_leakage_hashes_scenario_content_without_identity_fields() -> None:
    shared = {"tasks": [{"id": "task-1", "priority": 3}], "map": {"rows": 100, "cols": 100}}
    rows = [
        {"scenario_id": "train-name", "split": "train", "scenario": {**shared, "parent_id": "train-name", "seed": 1}},
        {"scenario_id": "test-name", "split": "test", "scenario": {**shared, "parent_id": "test-name", "seed": 2}},
    ]
    with pytest.raises(ValueError, match="scenario_content_sha256"):
        gain_models.check_group_leakage(rows)


def test_validation_partition_keeps_scenario_instances_whole() -> None:
    rows = _rows()
    model_rows, calibration_rows = gain_models._validation_parts(rows)
    model_groups = {row["scenario_id"] for row in model_rows}
    calibration_groups = {row["scenario_id"] for row in calibration_rows}
    assert model_groups
    assert calibration_groups
    assert not (model_groups & calibration_groups)
    assert len(model_groups | calibration_groups) == 8


def test_threshold_and_calibration_use_explicit_validation_samples() -> None:
    selection = gain_models.select_epsilon([1.0, -1.0, 0.1], [0.8, 0.5, 0.05], [0.0, 0.5, 0.75])
    assert selection["epsilon"] == 0.75
    calibration = gain_models.calibrate_lower_bound([1.0] * 20, [1.2] * 20, coverage=0.9)
    assert calibration["q"] == pytest.approx(0.2)
    assert calibration["calibration_count"] == 20
    assert "not a safety guarantee" in calibration["claim_limit"]


def test_lcb_gate_suppresses_when_calibration_is_unavailable() -> None:
    decision = gain_gate.decide_gain(
        1.0, 0.1, None, gate_mode="calibrated_lower_bound",
    )
    assert decision["allow_replan"] is False
    assert decision["reason"] == "calibration_unavailable"
    assert gain_gate.decide_gain(1.0, 0.1, None, gate_mode="mean")["allow_replan"] is True
    assert gain_gate.decide_gain(
        None, forced=True, gate_mode="calibrated_lower_bound",
    )["reason"] == "safety_override"


def test_constant_bundle_loads_and_predicts_after_fresh_process() -> None:
    features = {"affected_task_count": 2.0, "time_fraction": 0.4}
    normalizer = {"feature_names": sorted(features), "mean": [0.0, 0.0], "scale": [1.0, 1.0], "fit_row_count": 4, "fit_split": "train"}
    bundle = {
        "model_name": "constant-fixture", "model": {"kind": "constant", "value": 0.25},
        "normalizer": normalizer, "threshold": {"epsilon": 0.05},
        "calibration": {"q": 0.1}, "training": {"fit_info": {}},
    }
    temp_dir = _temporary_directory()
    try:
        path = temp_dir / "fixture.json"
        gain_models._save_bundle(path, bundle)
        loaded = gain_models.load_runtime(path)
        before_rng = gain_models.torch.get_rng_state().clone()
        prediction = gain_models.predict_gain(loaded, features)
        assert prediction["gain_hat"] == pytest.approx(0.25)
        assert prediction["lower_bound"] == pytest.approx(0.15)
        assert loaded.forward_calls == 1
        assert gain_models.torch.equal(before_rng, gain_models.torch.get_rng_state())
        restored = gain_models.fresh_process_restore_check(path, features)
        assert restored["fresh_process"] is True
        assert restored["prediction_matches"] is True
    finally:
        _remove_temporary_directory(temp_dir)


def test_analysis_counts_scenario_instances_and_rejects_incomplete_panels() -> None:
    rows = []
    for scenario in ("s1", "s2", "s3"):
        rows.extend([
            {"scenario_id": scenario, "method": "base", "j_total": 10.0},
            {"scenario_id": scenario, "method": "gain", "j_total": 8.0},
            {"scenario_id": scenario, "method": "gain", "j_total": 10.0},
        ])
    report = gain_analysis.paired_analysis(
        rows, [{"baseline": "base", "method": "gain"}], ["j_total"],
        bootstrap_replicates=200, permutation_replicates=200, seed=31,
    )
    comparison = report["comparisons"][0]
    assert comparison["scenario_count"] == 3
    assert comparison["method_mean"] == pytest.approx(9.0)
    assert comparison["paired_difference_mean"] == pytest.approx(-1.0)

    incomplete = [row for row in rows if not (row["scenario_id"] == "s3" and row["method"] == "gain")]
    with pytest.raises(ValueError, match="INCOMPLETE_METHOD_SCENARIO_PANEL"):
        gain_analysis.paired_analysis(
            incomplete, [{"baseline": "base", "method": "gain"}], ["j_total"],
            bootstrap_replicates=20, permutation_replicates=20,
        )


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    result = gain_analysis.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert result == {"a": 0.03, "c": 0.06, "b": 0.06}
